import os
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
import os
import re
import sys
import time
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from gguf import GGUFReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("HOLO_CACHE_DIR", os.path.join(BASE_DIR, "cache_holocell_v8"))
GGUF_PATH = os.environ.get("HOLO_GGUF_PATH", os.path.join(BASE_DIR, "qwen38_27b_full.gguf"))

# ============================================================================
# 1. TRITON KERNELS AT THE EMPIRICAL SWEET SPOT (V6 STRUCTURE @ 14 TOK/S)
# ============================================================================
@triton.jit
def _rms_norm_5120_kernel(
    x_ptr,            # [5120] fp16
    w_ptr,            # [5120] fp16
    y_ptr,            # [5120] fp16
):
    offs = tl.arange(0, 1024)
    sq_sum = tl.zeros((1024,), dtype=tl.float32)
    for k in range(0, 5120, 1024):
        v = tl.load(x_ptr + k + offs).to(tl.float32)
        sq_sum += v * v
    var = tl.sum(sq_sum, axis=0) * (1.0 / 5120.0)
    rstd = tl.rsqrt(var + 1e-6)
    for k in range(0, 5120, 1024):
        v = tl.load(x_ptr + k + offs).to(tl.float32)
        w = tl.load(w_ptr + k + offs).to(tl.float32)
        out = (v * rstd).to(tl.float16).to(tl.float32) * w
        tl.store(y_ptr + k + offs, out.to(tl.float16))


@triton.jit
def _swiglu_17408_kernel(
    gate_ptr,         # [17408] fp16
    up_ptr,           # [17408] fp16
    out_ptr,          # [17408] fp16
):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    g = tl.load(gate_ptr + offs).to(tl.float32)
    u = tl.load(up_ptr + offs).to(tl.float32)
    silu_g = g / (1.0 + tl.exp(-g))
    tl.store(out_ptr + offs, (silu_g * u).to(tl.float16))


@triton.jit
def _fused_scale_hadamard_kernel(
    x_ptr,            # [K] fp16
    col_scale_ptr,    # [K] fp16
    h64_ptr,          # [64, 64] fp16
    x_scaled_ptr,     # [K] fp16
    x_had_ptr,        # [K] fp16 (in FP16 for maximum L1 cache speed!)
):
    b = tl.program_id(0)
    offs = tl.arange(0, 64)
    idx = b * 64 + offs

    x_val = tl.load(x_ptr + idx).to(tl.float32)
    c_val = tl.load(col_scale_ptr + idx).to(tl.float32)
    xs = x_val * c_val
    tl.store(x_scaled_ptr + idx, xs.to(tl.float16))

    h_mat = tl.load(h64_ptr + offs[:, None] * 64 + offs[None, :]).to(tl.float32)
    xh = tl.sum(xs[:, None] * h_mat, axis=0)
    tl.store(x_had_ptr + idx, xh.to(tl.float16))


@triton.jit
def _holo_v7_turbo_gemv_kernel(
    x_had_ptr,        # [K] fp16
    x_scaled_ptr,     # [K] fp16
    packed_ptr,       # [N, K_PACKED] int32
    scale_u8_ptr,     # [N, K_BLOCKS] uint8
    cb_i32_ptr,       # [256] int32 (Single 32-bit L1 read per byte!)
    cb_scale_ptr,     # [1] fp32
    out_cols_ptr,     # [N, TOPK] int16
    out_vals_u8_ptr,  # [N, TOPK] uint8
    out_rscale_ptr,   # [N] fp16
    fp8_lut_ptr,      # [256] fp16
    res_ptr,          # [N] fp16
    y_ptr,            # [N] fp16
    N,
    K_PACKED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,      # 128 threads
    ADD_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * 4
    if row0 >= N:
        return
    row1 = row0 + 1
    row2 = row0 + 2
    row3 = row0 + 3

    offs_k = tl.arange(0, BLOCK_K)
    K_BLOCKS: tl.constexpr = K_PACKED // 4
    cb_scale = tl.load(cb_scale_ptr)

    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)

    p_base0 = packed_ptr + row0 * K_PACKED
    p_base1 = packed_ptr + row1 * K_PACKED
    p_base2 = packed_ptr + row2 * K_PACKED
    p_base3 = packed_ptr + row3 * K_PACKED

    s_base0 = scale_u8_ptr + row0 * K_BLOCKS
    s_base1 = scale_u8_ptr + row1 * K_BLOCKS
    s_base2 = scale_u8_ptr + row2 * K_BLOCKS
    s_base3 = scale_u8_ptr + row3 * K_BLOCKS

    for k_start in range(0, K_PACKED, BLOCK_K):
        k_idx = k_start + offs_k
        base_x = k_idx << 4
        b_idx = k_idx >> 2
        k_mask = k_idx < K_PACKED

        xh0  = tl.load(x_had_ptr + base_x + 0,  mask=k_mask, other=0.0).to(tl.float32)
        xh1  = tl.load(x_had_ptr + base_x + 1,  mask=k_mask, other=0.0).to(tl.float32)
        xh2  = tl.load(x_had_ptr + base_x + 2,  mask=k_mask, other=0.0).to(tl.float32)
        xh3  = tl.load(x_had_ptr + base_x + 3,  mask=k_mask, other=0.0).to(tl.float32)
        xh4  = tl.load(x_had_ptr + base_x + 4,  mask=k_mask, other=0.0).to(tl.float32)
        xh5  = tl.load(x_had_ptr + base_x + 5,  mask=k_mask, other=0.0).to(tl.float32)
        xh6  = tl.load(x_had_ptr + base_x + 6,  mask=k_mask, other=0.0).to(tl.float32)
        xh7  = tl.load(x_had_ptr + base_x + 7,  mask=k_mask, other=0.0).to(tl.float32)
        xh8  = tl.load(x_had_ptr + base_x + 8,  mask=k_mask, other=0.0).to(tl.float32)
        xh9  = tl.load(x_had_ptr + base_x + 9,  mask=k_mask, other=0.0).to(tl.float32)
        xh10 = tl.load(x_had_ptr + base_x + 10, mask=k_mask, other=0.0).to(tl.float32)
        xh11 = tl.load(x_had_ptr + base_x + 11, mask=k_mask, other=0.0).to(tl.float32)
        xh12 = tl.load(x_had_ptr + base_x + 12, mask=k_mask, other=0.0).to(tl.float32)
        xh13 = tl.load(x_had_ptr + base_x + 13, mask=k_mask, other=0.0).to(tl.float32)
        xh14 = tl.load(x_had_ptr + base_x + 14, mask=k_mask, other=0.0).to(tl.float32)
        xh15 = tl.load(x_had_ptr + base_x + 15, mask=k_mask, other=0.0).to(tl.float32)

        x_sum_all = (xh0 + xh1 + xh2 + xh3 + xh4 + xh5 + xh6 + xh7 + xh8 + xh9 + xh10 + xh11 + xh12 + xh13 + xh14 + xh15) * 128.0

        # --- ROW 0 ---
        p0 = tl.load(p_base0 + k_idx, mask=k_mask, other=0)
        bs0 = tl.load(fp8_lut_ptr + tl.load(s_base0 + b_idx, mask=k_mask, other=0).to(tl.int32)).to(tl.float32)
        c0_0 = tl.load(cb_i32_ptr + (p0 & 0xFF))
        c0_1 = tl.load(cb_i32_ptr + ((p0 >> 8) & 0xFF))
        c0_2 = tl.load(cb_i32_ptr + ((p0 >> 16) & 0xFF))
        c0_3 = tl.load(cb_i32_ptr + ((p0 >> 24) & 0xFF))
        dot0 = (
            (c0_0 & 0xFF).to(tl.float32) * xh0 + ((c0_0 >> 8) & 0xFF).to(tl.float32) * xh1 + ((c0_0 >> 16) & 0xFF).to(tl.float32) * xh2 + ((c0_0 >> 24) & 0xFF).to(tl.float32) * xh3 +
            (c0_1 & 0xFF).to(tl.float32) * xh4 + ((c0_1 >> 8) & 0xFF).to(tl.float32) * xh5 + ((c0_1 >> 16) & 0xFF).to(tl.float32) * xh6 + ((c0_1 >> 24) & 0xFF).to(tl.float32) * xh7 +
            (c0_2 & 0xFF).to(tl.float32) * xh8 + ((c0_2 >> 8) & 0xFF).to(tl.float32) * xh9 + ((c0_2 >> 16) & 0xFF).to(tl.float32) * xh10 + ((c0_2 >> 24) & 0xFF).to(tl.float32) * xh11 +
            (c0_3 & 0xFF).to(tl.float32) * xh12 + ((c0_3 >> 8) & 0xFF).to(tl.float32) * xh13 + ((c0_3 >> 16) & 0xFF).to(tl.float32) * xh14 + ((c0_3 >> 24) & 0xFF).to(tl.float32) * xh15
            - x_sum_all
        )
        acc0 += dot0 * bs0

        # --- ROW 1 ---
        p1 = tl.load(p_base1 + k_idx, mask=k_mask, other=0)
        bs1 = tl.load(fp8_lut_ptr + tl.load(s_base1 + b_idx, mask=k_mask, other=0).to(tl.int32)).to(tl.float32)
        c1_0 = tl.load(cb_i32_ptr + (p1 & 0xFF))
        c1_1 = tl.load(cb_i32_ptr + ((p1 >> 8) & 0xFF))
        c1_2 = tl.load(cb_i32_ptr + ((p1 >> 16) & 0xFF))
        c1_3 = tl.load(cb_i32_ptr + ((p1 >> 24) & 0xFF))
        dot1 = (
            (c1_0 & 0xFF).to(tl.float32) * xh0 + ((c1_0 >> 8) & 0xFF).to(tl.float32) * xh1 + ((c1_0 >> 16) & 0xFF).to(tl.float32) * xh2 + ((c1_0 >> 24) & 0xFF).to(tl.float32) * xh3 +
            (c1_1 & 0xFF).to(tl.float32) * xh4 + ((c1_1 >> 8) & 0xFF).to(tl.float32) * xh5 + ((c1_1 >> 16) & 0xFF).to(tl.float32) * xh6 + ((c1_1 >> 24) & 0xFF).to(tl.float32) * xh7 +
            (c1_2 & 0xFF).to(tl.float32) * xh8 + ((c1_2 >> 8) & 0xFF).to(tl.float32) * xh9 + ((c1_2 >> 16) & 0xFF).to(tl.float32) * xh10 + ((c1_2 >> 24) & 0xFF).to(tl.float32) * xh11 +
            (c1_3 & 0xFF).to(tl.float32) * xh12 + ((c1_3 >> 8) & 0xFF).to(tl.float32) * xh13 + ((c1_3 >> 16) & 0xFF).to(tl.float32) * xh14 + ((c1_3 >> 24) & 0xFF).to(tl.float32) * xh15
            - x_sum_all
        )
        acc1 += dot1 * bs1

        # --- ROW 2 ---
        p2 = tl.load(p_base2 + k_idx, mask=k_mask, other=0)
        bs2 = tl.load(fp8_lut_ptr + tl.load(s_base2 + b_idx, mask=k_mask, other=0).to(tl.int32)).to(tl.float32)
        c2_0 = tl.load(cb_i32_ptr + (p2 & 0xFF))
        c2_1 = tl.load(cb_i32_ptr + ((p2 >> 8) & 0xFF))
        c2_2 = tl.load(cb_i32_ptr + ((p2 >> 16) & 0xFF))
        c2_3 = tl.load(cb_i32_ptr + ((p2 >> 24) & 0xFF))
        dot2 = (
            (c2_0 & 0xFF).to(tl.float32) * xh0 + ((c2_0 >> 8) & 0xFF).to(tl.float32) * xh1 + ((c2_0 >> 16) & 0xFF).to(tl.float32) * xh2 + ((c2_0 >> 24) & 0xFF).to(tl.float32) * xh3 +
            (c2_1 & 0xFF).to(tl.float32) * xh4 + ((c2_1 >> 8) & 0xFF).to(tl.float32) * xh5 + ((c2_1 >> 16) & 0xFF).to(tl.float32) * xh6 + ((c2_1 >> 24) & 0xFF).to(tl.float32) * xh7 +
            (c2_2 & 0xFF).to(tl.float32) * xh8 + ((c2_2 >> 8) & 0xFF).to(tl.float32) * xh9 + ((c2_2 >> 16) & 0xFF).to(tl.float32) * xh10 + ((c2_2 >> 24) & 0xFF).to(tl.float32) * xh11 +
            (c2_3 & 0xFF).to(tl.float32) * xh12 + ((c2_3 >> 8) & 0xFF).to(tl.float32) * xh13 + ((c2_3 >> 16) & 0xFF).to(tl.float32) * xh14 + ((c2_3 >> 24) & 0xFF).to(tl.float32) * xh15
            - x_sum_all
        )
        acc2 += dot2 * bs2

        # --- ROW 3 ---
        p3 = tl.load(p_base3 + k_idx, mask=k_mask, other=0)
        bs3 = tl.load(fp8_lut_ptr + tl.load(s_base3 + b_idx, mask=k_mask, other=0).to(tl.int32)).to(tl.float32)
        c3_0 = tl.load(cb_i32_ptr + (p3 & 0xFF))
        c3_1 = tl.load(cb_i32_ptr + ((p3 >> 8) & 0xFF))
        c3_2 = tl.load(cb_i32_ptr + ((p3 >> 16) & 0xFF))
        c3_3 = tl.load(cb_i32_ptr + ((p3 >> 24) & 0xFF))
        dot3 = (
            (c3_0 & 0xFF).to(tl.float32) * xh0 + ((c3_0 >> 8) & 0xFF).to(tl.float32) * xh1 + ((c3_0 >> 16) & 0xFF).to(tl.float32) * xh2 + ((c3_0 >> 24) & 0xFF).to(tl.float32) * xh3 +
            (c3_1 & 0xFF).to(tl.float32) * xh4 + ((c3_1 >> 8) & 0xFF).to(tl.float32) * xh5 + ((c3_1 >> 16) & 0xFF).to(tl.float32) * xh6 + ((c3_1 >> 24) & 0xFF).to(tl.float32) * xh7 +
            (c3_2 & 0xFF).to(tl.float32) * xh8 + ((c3_2 >> 8) & 0xFF).to(tl.float32) * xh9 + ((c3_2 >> 16) & 0xFF).to(tl.float32) * xh10 + ((c3_2 >> 24) & 0xFF).to(tl.float32) * xh11 +
            (c3_3 & 0xFF).to(tl.float32) * xh12 + ((c3_3 >> 8) & 0xFF).to(tl.float32) * xh13 + ((c3_3 >> 16) & 0xFF).to(tl.float32) * xh14 + ((c3_3 >> 24) & 0xFF).to(tl.float32) * xh15
            - x_sum_all
        )
        acc3 += dot3 * bs3

    v0_out = tl.zeros((BLOCK_K,), dtype=tl.float32)
    v1_out = tl.zeros((BLOCK_K,), dtype=tl.float32)
    v2_out = tl.zeros((BLOCK_K,), dtype=tl.float32)
    v3_out = tl.zeros((BLOCK_K,), dtype=tl.float32)

    ob0 = row0 * TOPK
    ob1 = row1 * TOPK
    ob2 = row2 * TOPK
    ob3 = row3 * TOPK

    for out_start in tl.static_range(0, TOPK, BLOCK_K):
        o_offs = out_start + offs_k
        o_mask = o_offs < TOPK

        c0 = tl.load(out_cols_ptr + ob0 + o_offs, mask=o_mask, other=0).to(tl.int32) & 0xFFFF
        u0 = tl.load(out_vals_u8_ptr + ob0 + o_offs, mask=o_mask, other=0).to(tl.int32)
        v0_out += tl.load(fp8_lut_ptr + u0, mask=o_mask, other=0.0).to(tl.float32) * tl.load(x_scaled_ptr + c0, mask=o_mask, other=0.0).to(tl.float32)

        c1 = tl.load(out_cols_ptr + ob1 + o_offs, mask=o_mask, other=0).to(tl.int32) & 0xFFFF
        u1 = tl.load(out_vals_u8_ptr + ob1 + o_offs, mask=o_mask, other=0).to(tl.int32)
        v1_out += tl.load(fp8_lut_ptr + u1, mask=o_mask, other=0.0).to(tl.float32) * tl.load(x_scaled_ptr + c1, mask=o_mask, other=0.0).to(tl.float32)

        c2 = tl.load(out_cols_ptr + ob2 + o_offs, mask=o_mask, other=0).to(tl.int32) & 0xFFFF
        u2 = tl.load(out_vals_u8_ptr + ob2 + o_offs, mask=o_mask, other=0).to(tl.int32)
        v2_out += tl.load(fp8_lut_ptr + u2, mask=o_mask, other=0.0).to(tl.float32) * tl.load(x_scaled_ptr + c2, mask=o_mask, other=0.0).to(tl.float32)

        c3 = tl.load(out_cols_ptr + ob3 + o_offs, mask=o_mask, other=0).to(tl.int32) & 0xFFFF
        u3 = tl.load(out_vals_u8_ptr + ob3 + o_offs, mask=o_mask, other=0).to(tl.int32)
        v3_out += tl.load(fp8_lut_ptr + u3, mask=o_mask, other=0.0).to(tl.float32) * tl.load(x_scaled_ptr + c3, mask=o_mask, other=0.0).to(tl.float32)

    tot0 = tl.sum(acc0 * cb_scale + v0_out * tl.load(out_rscale_ptr + row0).to(tl.float32), axis=0)
    if ADD_RESIDUAL:
        tot0 += tl.load(res_ptr + row0).to(tl.float32)
    tl.store(y_ptr + row0, tot0.to(tl.float16))

    tot1 = tl.sum(acc1 * cb_scale + v1_out * tl.load(out_rscale_ptr + row1).to(tl.float32), axis=0)
    if ADD_RESIDUAL:
        tot1 += tl.load(res_ptr + row1).to(tl.float32)
    tl.store(y_ptr + row1, tot1.to(tl.float16))

    tot2 = tl.sum(acc2 * cb_scale + v2_out * tl.load(out_rscale_ptr + row2).to(tl.float32), axis=0)
    if ADD_RESIDUAL:
        tot2 += tl.load(res_ptr + row2).to(tl.float32)
    tl.store(y_ptr + row2, tot2.to(tl.float16))

    tot3 = tl.sum(acc3 * cb_scale + v3_out * tl.load(out_rscale_ptr + row3).to(tl.float32), axis=0)
    if ADD_RESIDUAL:
        tot3 += tl.load(res_ptr + row3).to(tl.float32)
    tl.store(y_ptr + row3, tot3.to(tl.float16))


@triton.jit
def _int4_head_gemv_kernel(
    x_ptr,            # [5120] fp16
    packed_u8_ptr,    # [N, 2560] uint8
    scales_ptr,       # [N, 80] fp16
    y_ptr,            # [N] fp16
    N,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    base_r = pid * 8
    if base_r >= N:
        return

    offs_k = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k_start in range(0, 2560, BLOCK_K):
        k_idx = k_start + offs_k
        x_even = tl.load(x_ptr + k_idx * 2 + 0).to(tl.float32)
        x_odd  = tl.load(x_ptr + k_idx * 2 + 1).to(tl.float32)
        scale_idx = k_idx >> 5

        b0 = tl.load(packed_u8_ptr + (base_r + 0) * 2560 + k_idx).to(tl.int32)
        s0 = tl.load(scales_ptr + (base_r + 0) * 80 + scale_idx).to(tl.float32)
        acc0 += (((b0 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b0 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s0

        b1 = tl.load(packed_u8_ptr + (base_r + 1) * 2560 + k_idx).to(tl.int32)
        s1 = tl.load(scales_ptr + (base_r + 1) * 80 + scale_idx).to(tl.float32)
        acc1 += (((b1 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b1 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s1

        b2 = tl.load(packed_u8_ptr + (base_r + 2) * 2560 + k_idx).to(tl.int32)
        s2 = tl.load(scales_ptr + (base_r + 2) * 80 + scale_idx).to(tl.float32)
        acc2 += (((b2 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b2 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s2

        b3 = tl.load(packed_u8_ptr + (base_r + 3) * 2560 + k_idx).to(tl.int32)
        s3 = tl.load(scales_ptr + (base_r + 3) * 80 + scale_idx).to(tl.float32)
        acc3 += (((b3 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b3 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s3

        b4 = tl.load(packed_u8_ptr + (base_r + 4) * 2560 + k_idx).to(tl.int32)
        s4 = tl.load(scales_ptr + (base_r + 4) * 80 + scale_idx).to(tl.float32)
        acc4 += (((b4 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b4 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s4

        b5 = tl.load(packed_u8_ptr + (base_r + 5) * 2560 + k_idx).to(tl.int32)
        s5 = tl.load(scales_ptr + (base_r + 5) * 80 + scale_idx).to(tl.float32)
        acc5 += (((b5 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b5 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s5

        b6 = tl.load(packed_u8_ptr + (base_r + 6) * 2560 + k_idx).to(tl.int32)
        s6 = tl.load(scales_ptr + (base_r + 6) * 80 + scale_idx).to(tl.float32)
        acc6 += (((b6 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b6 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s6

        b7 = tl.load(packed_u8_ptr + (base_r + 7) * 2560 + k_idx).to(tl.int32)
        s7 = tl.load(scales_ptr + (base_r + 7) * 80 + scale_idx).to(tl.float32)
        acc7 += (((b7 & 0x0F).to(tl.float32) - 8.0) * x_even + (((b7 >> 4) & 0x0F).to(tl.float32) - 8.0) * x_odd) * s7

    tl.store(y_ptr + base_r + 0, tl.sum(acc0, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 1, tl.sum(acc1, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 2, tl.sum(acc2, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 3, tl.sum(acc3, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 4, tl.sum(acc4, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 5, tl.sum(acc5, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 6, tl.sum(acc6, axis=0).to(tl.float16))
    tl.store(y_ptr + base_r + 7, tl.sum(acc7, axis=0).to(tl.float16))


@triton.jit
def _fused_conv1d_silu_kernel(
    qkv_ptr,
    c_state_ptr,
    conv_w_ptr,
    qkv_conv_ptr,
):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    mask = offs < 10240

    s0 = tl.load(c_state_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(c_state_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(c_state_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    qkv = tl.load(qkv_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    w0 = tl.load(conv_w_ptr + offs * 4 + 0, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(conv_w_ptr + offs * 4 + 1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(conv_w_ptr + offs * 4 + 2, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(conv_w_ptr + offs * 4 + 3, mask=mask, other=0.0).to(tl.float32)

    val = s0 * w0 + s1 * w1 + s2 * w2 + qkv * w3
    silu_val = val / (1.0 + tl.exp(-val))
    tl.store(qkv_conv_ptr + offs, silu_val.to(tl.float16), mask=mask)

    tl.store(c_state_ptr + offs * 3 + 0, s1.to(tl.float16), mask=mask)
    tl.store(c_state_ptr + offs * 3 + 1, s2.to(tl.float16), mask=mask)
    tl.store(c_state_ptr + offs * 3 + 2, qkv.to(tl.float16), mask=mask)


@triton.jit
def _fused_deltanet_step_kernel(
    qkv_conv_ptr,
    gate_ptr,
    alpha_ptr,
    beta_ptr,
    dt_bias_ptr,
    ssm_a_ptr,
    ssm_norm_w_ptr,
    state_ptr,
    y_out_ptr,
):
    h = tl.program_id(0)
    g = h % 16
    offs = tl.arange(0, 128)

    q_raw = tl.load(qkv_conv_ptr + g * 128 + offs).to(tl.float32)
    k_raw = tl.load(qkv_conv_ptr + 2048 + g * 128 + offs).to(tl.float32)
    v_head = tl.load(qkv_conv_ptr + 4096 + h * 128 + offs).to(tl.float32)

    q_norm_inv = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1e-12) * 0.08838834764831845
    k_norm_inv = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1e-12)

    alpha_val = tl.load(alpha_ptr + h).to(tl.float32) + tl.load(dt_bias_ptr + h).to(tl.float32)
    dt = tl.where(alpha_val > 20.0, alpha_val, tl.log(1.0 + tl.exp(alpha_val)))
    ssm_a = tl.load(ssm_a_ptr + h).to(tl.float32)
    decay = tl.exp(ssm_a * dt)

    beta_in = tl.load(beta_ptr + h).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-beta_in))

    base_s = state_ptr + h * 16384
    r32 = tl.arange(0, 32)

    row0 = r32 + 0
    S0 = tl.load(base_s + row0[:, None] * 128 + offs[None, :]) * decay
    k0 = tl.load(qkv_conv_ptr + 2048 + g * 128 + row0).to(tl.float32) * k_norm_inv
    q0 = tl.load(qkv_conv_ptr + g * 128 + row0).to(tl.float32) * q_norm_inv

    row1 = r32 + 32
    S1 = tl.load(base_s + row1[:, None] * 128 + offs[None, :]) * decay
    k1 = tl.load(qkv_conv_ptr + 2048 + g * 128 + row1).to(tl.float32) * k_norm_inv
    q1 = tl.load(qkv_conv_ptr + g * 128 + row1).to(tl.float32) * q_norm_inv

    row2 = r32 + 64
    S2 = tl.load(base_s + row2[:, None] * 128 + offs[None, :]) * decay
    k2 = tl.load(qkv_conv_ptr + 2048 + g * 128 + row2).to(tl.float32) * k_norm_inv
    q2 = tl.load(qkv_conv_ptr + g * 128 + row2).to(tl.float32) * q_norm_inv

    row3 = r32 + 96
    S3 = tl.load(base_s + row3[:, None] * 128 + offs[None, :]) * decay
    k3 = tl.load(qkv_conv_ptr + 2048 + g * 128 + row3).to(tl.float32) * k_norm_inv
    q3 = tl.load(qkv_conv_ptr + g * 128 + row3).to(tl.float32) * q_norm_inv

    sk = (
        tl.sum(S0 * k0[:, None], axis=0) +
        tl.sum(S1 * k1[:, None], axis=0) +
        tl.sum(S2 * k2[:, None], axis=0) +
        tl.sum(S3 * k3[:, None], axis=0)
    )

    delta = (v_head - sk) * beta
    S0 += k0[:, None] * delta[None, :]
    S1 += k1[:, None] * delta[None, :]
    S2 += k2[:, None] * delta[None, :]
    S3 += k3[:, None] * delta[None, :]

    tl.store(base_s + row0[:, None] * 128 + offs[None, :], S0)
    tl.store(base_s + row1[:, None] * 128 + offs[None, :], S1)
    tl.store(base_s + row2[:, None] * 128 + offs[None, :], S2)
    tl.store(base_s + row3[:, None] * 128 + offs[None, :], S3)

    y = (
        tl.sum(S0 * q0[:, None], axis=0) +
        tl.sum(S1 * q1[:, None], axis=0) +
        tl.sum(S2 * q2[:, None], axis=0) +
        tl.sum(S3 * q3[:, None], axis=0)
    )

    y_fp16 = y.to(tl.float16).to(tl.float32)
    var = tl.sum(y_fp16 * y_fp16, axis=0) * (1.0 / 128.0)
    norm_w = tl.load(ssm_norm_w_ptr + offs).to(tl.float32)
    y_norm = (y_fp16 * tl.rsqrt(var + 1e-6)).to(tl.float16).to(tl.float32) * norm_w

    g_val = tl.load(gate_ptr + h * 128 + offs).to(tl.float32)
    silu_g = g_val / (1.0 + tl.exp(-g_val))
    out_val = (y_norm * silu_g).to(tl.float16)

    tl.store(y_out_ptr + h * 128 + offs, out_val)


@triton.jit
def _fused_gqa_decode_step_kernel(
    q_full_ptr,
    k_cur_ptr,
    v_cur_ptr,
    q_norm_w_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    pos_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ctx_out_ptr,
    MAX_SEQ: tl.constexpr,
):
    h = tl.program_id(0)
    kv_h = h // 6
    offs = tl.arange(0, 256)
    pos = tl.load(pos_ptr).to(tl.int32)

    base_qg = h * 512
    q_val = tl.load(q_full_ptr + base_qg + offs).to(tl.float32)
    gate_val = tl.load(q_full_ptr + base_qg + 256 + offs).to(tl.float32)

    q_var = tl.sum(q_val * q_val, axis=0) * (1.0 / 256.0)
    qw = tl.load(q_norm_w_ptr + offs).to(tl.float32)
    q_normed = (q_val * tl.rsqrt(q_var + 1e-6)).to(tl.float16).to(tl.float32) * qw

    is_first32 = offs < 32
    is_second32 = (offs >= 32) & (offs < 64)
    pair_offs = tl.where(is_first32, offs + 32, tl.where(is_second32, offs - 32, offs))
    qp_raw = tl.load(q_full_ptr + base_qg + pair_offs).to(tl.float32)
    qp_w = tl.load(q_norm_w_ptr + pair_offs).to(tl.float32)
    qp_normed = (qp_raw * tl.rsqrt(q_var + 1e-6)).to(tl.float16).to(tl.float32) * qp_w

    rot_idx = offs & 31
    c_full = tl.load(rope_cos_ptr + pos * 32 + rot_idx)
    s_full = tl.load(rope_sin_ptr + pos * 32 + rot_idx)

    q_rot = tl.where(
        is_first32,
        q_normed * c_full - qp_normed * s_full,
        tl.where(is_second32, q_normed * c_full + qp_normed * s_full, q_normed)
    )

    base_kv = kv_h * 256
    k_val = tl.load(k_cur_ptr + base_kv + offs).to(tl.float32)
    v_val = tl.load(v_cur_ptr + base_kv + offs).to(tl.float16)

    k_var = tl.sum(k_val * k_val, axis=0) * (1.0 / 256.0)
    kw = tl.load(k_norm_w_ptr + offs).to(tl.float32)
    k_normed = (k_val * tl.rsqrt(k_var + 1e-6)).to(tl.float16).to(tl.float32) * kw

    kp_raw = tl.load(k_cur_ptr + base_kv + pair_offs).to(tl.float32)
    kp_w = tl.load(k_norm_w_ptr + pair_offs).to(tl.float32)
    kp_normed = (kp_raw * tl.rsqrt(k_var + 1e-6)).to(tl.float16).to(tl.float32) * kp_w

    k_rot = tl.where(
        is_first32,
        k_normed * c_full - kp_normed * s_full,
        tl.where(is_second32, k_normed * c_full + kp_normed * s_full, k_normed)
    ).to(tl.float16)

    cache_offset = (kv_h * MAX_SEQ + pos) * 256 + offs
    tl.store(k_cache_ptr + cache_offset, k_rot)
    tl.store(v_cache_ptr + cache_offset, v_val)

    q_scaled = q_rot * 0.0625
    m_i = -1e20
    l_i = 0.0
    acc_o = tl.zeros((256,), dtype=tl.float32)

    for t in range(0, pos + 1):
        t_offset = (kv_h * MAX_SEQ + t) * 256 + offs
        kt = tl.where(t == pos, k_rot.to(tl.float32), tl.load(k_cache_ptr + t_offset).to(tl.float32))
        vt = tl.where(t == pos, v_val.to(tl.float32), tl.load(v_cache_ptr + t_offset).to(tl.float32))

        score = tl.sum(q_scaled * kt, axis=0)
        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)

        acc_o = acc_o * alpha + vt * p
        l_i = l_i * alpha + p
        m_i = m_new

    ctx_h = (acc_o / l_i).to(tl.float16).to(tl.float32)
    sig_gate = 1.0 / (1.0 + tl.exp(-gate_val))
    out_h = (ctx_h * sig_gate).to(tl.float16)

    tl.store(ctx_out_ptr + h * 256 + offs, out_h)


# ============================================================================
# 2. NATIVE BPE TOKENIZER WITH INCREMENTAL UTF-8 DECODER
# ============================================================================
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class GGUFNativeTokenizer:
    def __init__(self, reader=None, vocab_pt=None):
        if vocab_pt is None:
            vocab_pt = os.path.join(CACHE_DIR, "tokenizer_vocab.pt")
        if os.path.exists(vocab_pt):
            print(f"[*] Loading instant native vocabulary from {vocab_pt}...")
            vd = torch.load(vocab_pt, map_location="cpu", weights_only=False)
            self.__dict__.update(vd)
            print("[OK] Native tokenizer ready (Loaded in 0.1s without requiring GGUF file)")
            return
        if reader is None:
            reader = GGUFReader(GGUF_PATH, mode="r")
        print("[*] Extracting native vocabulary of 248,320 tokens from GGUF...")
        tok_field = reader.fields["tokenizer.ggml.tokens"]
        self.id_to_token = [bytes(tok_field.parts[i]).decode("utf-8", errors="replace") for i in tok_field.data]
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        merges_field = reader.fields.get("tokenizer.ggml.merges")
        self.bpe_ranks = {}
        if merges_field is not None:
            for rank, idx in enumerate(merges_field.data):
                m_str = bytes(merges_field.parts[idx]).decode("utf-8", errors="replace")
                parts = m_str.split(" ")
                if len(parts) == 2:
                    self.bpe_ranks[(parts[0], parts[1])] = rank

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.eos_token_id = 248044
        self.im_start_id = self.token_to_id.get("<|im_start|>", 248045)
        self.im_end_id = self.token_to_id.get("<|im_end|>", 248046)
        self.think_start_id = self.token_to_id.get("<think>", 248068)
        self.think_end_id = self.token_to_id.get("</think>", 248069)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        print(f"[OK] Native tokenizer ready ({len(self.id_to_token):,} tokens)")

    def _bpe(self, token):
        if token in self.token_to_id:
            return [token]
        word = tuple(token)
        if len(word) <= 1:
            return list(word)
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return list(word)

    def encode_text(self, text):
        ids = []
        for chunk in re.findall(self.pat, text):
            b_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for bpe_tok in self._bpe(b_str):
                if bpe_tok in self.token_to_id:
                    ids.append(self.token_to_id[bpe_tok])
                else:
                    for ch in bpe_tok:
                        if ch in self.token_to_id:
                            ids.append(self.token_to_id[ch])
        return ids

    def encode_chat(self, user_msg, enable_thinking=False):
        base = (
            [self.im_start_id] + self.encode_text("user\n" + user_msg) + [self.im_end_id] +
            self.encode_text("\n") + [self.im_start_id] + self.encode_text("assistant\n")
        )
        if enable_thinking:
            return base + [self.think_start_id] + self.encode_text("\n")
        else:
            return base + [self.think_start_id] + self.encode_text("\n\n") + [self.think_end_id] + self.encode_text("\n\n")

    def decode(self, token_ids):
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.id_to_token):
                tok = self.id_to_token[tid]
                if (tok.startswith("<|") and tok.endswith("|>")) or tok in ("<think>", "</think>"):
                    continue
                pieces.append(tok)
        text = "".join(pieces)
        byte_arr = bytearray([self.byte_decoder[c] for c in text if c in self.byte_decoder])
        return byte_arr.decode("utf-8", errors="ignore")


# ============================================================================
# 3. HOLOCELL-V8 ENGINE (FP16 BUFFERS IN L1 FOR 13.5-14.2 REAL TOK/S)
# ============================================================================
class HoloCellV7Engine:
    def __init__(self, device='cuda'):
        self.device = device
        h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
        h = h2
        for _ in range(5):
            h = torch.kron(h, h2)
        self.H64 = (h / 8.0).to(torch.float16).contiguous()

        lut_u8 = torch.arange(256, dtype=torch.uint8, device=device)
        lut_fp16 = lut_u8.view(torch.float8_e4m3fn).to(torch.float16)
        self.fp8_lut = torch.nan_to_num(lut_fp16, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

        # En FP16 puro (2 bytes) para no saturar la caché L1 como pasó en V10!
        self.buf_x_scaled = torch.zeros((17408,), dtype=torch.float16, device=device)
        self.buf_x_had = torch.zeros((17408,), dtype=torch.float16, device=device)

    @torch.no_grad()
    def prepare_input_hadamard(self, x_1d: torch.Tensor, col_scale: torch.Tensor, K: int):
        _fused_scale_hadamard_kernel[(K // 64,)](
            x_1d, col_scale, self.H64, self.buf_x_scaled, self.buf_x_had, num_warps=2
        )

    @torch.no_grad()
    def run_gemv_prepped(self, layer_dict, out_1d: torch.Tensor, res_1d: torch.Tensor = None):
        cells = layer_dict["packed"]
        N, K_packed = cells.shape
        add_res = res_1d is not None
        _holo_v7_turbo_gemv_kernel[(N // 4,)](
            self.buf_x_had,
            self.buf_x_scaled,
            cells,
            layer_dict["scale_u8"],
            layer_dict["cb_i32"],
            layer_dict["cb_scale"],
            layer_dict["outlier_cols"],
            layer_dict["out_vals_u8"],
            layer_dict["out_rscale"],
            self.fp8_lut,
            res_1d if add_res else out_1d,
            out_1d,
            N,
            K_PACKED=K_packed,
            TOPK=layer_dict["outlier_cols"].shape[1],
            BLOCK_K=128,
            ADD_RESIDUAL=add_res,
            num_warps=4,
        )


# ============================================================================
# 4. QWEN 3.8 RUNTIME
# ============================================================================
class Qwen38HoloRuntime:
    def __init__(self, engine: HoloCellV7Engine, max_seq_len: int = 16384):
        self.engine = engine
        self.device = engine.device
        self.max_seq_len = max_seq_len
        self.layers = {}
        self.aux = {}
        self.head_vram = None

        half = 32
        freqs = 1.0 / (1e7 ** (torch.arange(0, half, device=self.device, dtype=torch.float32) / half))
        positions = torch.arange(0, max_seq_len, device=self.device, dtype=torch.float32).unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        self.rope_cos = torch.cos(angles).contiguous()
        self.rope_sin = torch.sin(angles).contiguous()

        self.static_pos = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.static_x = torch.zeros((1, 5120), dtype=torch.float16, device=self.device)
        self.static_h = torch.zeros((1, 5120), dtype=torch.float16, device=self.device)
        self.static_qkv = torch.zeros((10240,), dtype=torch.float16, device=self.device)
        self.static_qkv_conv = torch.zeros((10240,), dtype=torch.float16, device=self.device)
        self.static_gate = torch.zeros((6144,), dtype=torch.float16, device=self.device)
        self.static_y_gated = torch.zeros((6144,), dtype=torch.float16, device=self.device)
        self.static_q_full = torch.zeros((12288,), dtype=torch.float16, device=self.device)
        self.static_k = torch.zeros((1024,), dtype=torch.float16, device=self.device)
        self.static_v = torch.zeros((1024,), dtype=torch.float16, device=self.device)
        self.static_ctx = torch.zeros((6144,), dtype=torch.float16, device=self.device)
        self.static_ffn_gate = torch.zeros((17408,), dtype=torch.float16, device=self.device)
        self.static_ffn_up = torch.zeros((17408,), dtype=torch.float16, device=self.device)
        self.static_logits = torch.zeros((1, 248320), dtype=torch.float16, device=self.device)

        self.cuda_graph = None

    def load_to_vram(self):
        print("[*] Loading auxiliary tensors and norms...")
        raw_aux = torch.load(os.path.join(CACHE_DIR, "aux_tensors.pt"), map_location="cpu")
        for k, v in raw_aux.items():
            if k.startswith("token_embd"):
                self.aux[k] = v
            else:
                self.aux[k] = v.to(
                    self.device,
                    dtype=(torch.float32 if k.endswith(".ssm_a") or k.endswith(".ssm_dt.bias") else torch.float16)
                ).contiguous()

        print("[*] Loading 400 Holo-V8 matrices to RTX 3060...")
        for f in sorted(os.listdir(CACHE_DIR)):
            if f.startswith("blk.") and f.endswith(".pt"):
                t_name = f[:-3]
                d = torch.load(os.path.join(CACHE_DIR, f), map_location="cpu")
                cb = d["dynamic_codebook"].to(self.device).contiguous()
                cb_max = torch.amax(torch.abs(cb.float())).clamp(min=1e-6)
                cb_scale = (cb_max / 127.0).view(1).to(torch.float32).contiguous()
                cb_u8 = torch.round(cb.float() / cb_scale).clamp(-128, 127).to(torch.int32) + 128
                cb_i32 = (cb_u8[:, 0] | (cb_u8[:, 1] << 8) | (cb_u8[:, 2] << 16) | (cb_u8[:, 3] << 24)).to(torch.int32).contiguous()
                self.layers[t_name] = {
                    "packed": d["packed"].to(self.device).contiguous(),
                    "scale_u8": d["scale_fp8"].view(torch.uint8).to(self.device).contiguous(),
                    "col_scale": d["col_scale"].squeeze(0).to(self.device).contiguous(),
                    "outlier_cols": d["outlier_cols"].to(self.device).contiguous(),
                    "out_vals_u8": d["outlier_vals"].view(torch.uint8).to(self.device).contiguous(),
                    "out_rscale": d["out_rscale"].squeeze(-1).to(self.device).contiguous(),
                    "cb_i32": cb_i32,
                    "cb_scale": cb_scale
                }

        print("[*] Loading output.weight head (INT4) directly to VRAM...")
        head_path = os.path.join(CACHE_DIR, "output_head_int4_vram.pt")
        hd = torch.load(head_path, map_location="cpu")
        self.head_vram = {
            "packed_u8": hd["packed_u8"].to(self.device).contiguous(),
            "scales": hd["scales"].to(self.device).contiguous()
        }

        self.conv_states = {}
        self.ssm_states = {}
        self.k_cache = {}
        self.v_cache = {}
        for i in range(64):
            if i % 4 != 3:
                self.conv_states[i] = torch.zeros((10240, 3), device=self.device, dtype=torch.float16)
                self.ssm_states[i] = torch.zeros((48, 128, 128), device=self.device, dtype=torch.float32)
            else:
                self.k_cache[i] = torch.zeros((4, self.max_seq_len, 256), device=self.device, dtype=torch.float16)
                self.v_cache[i] = torch.zeros((4, self.max_seq_len, 256), device=self.device, dtype=torch.float16)

        torch.cuda.empty_cache()
        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[OK] ¡Qwen3.8 27B + KV Cache ({self.max_seq_len} ctx) ready! Allocated: {vram_gb:.2f} GB | Reserved: {res_gb:.2f} GB / 12.00 GB")

    def init_states(self):
        for i in range(64):
            if i % 4 != 3:
                self.conv_states[i].zero_()
                self.ssm_states[i].zero_()

    @torch.no_grad()
    def embed_tokens(self, token_ids: list) -> torch.Tensor:
        rows_int8 = self.aux["token_embd_int8"][token_ids].to(self.device, dtype=torch.float16)
        scales = self.aux["token_embd_scale"][token_ids].to(self.device, dtype=torch.float16)
        return rows_int8 * scales

    @torch.no_grad()
    def _graph_decode_body(self):
        for i in range(64):
            prefix = f"blk.{i}."
            _rms_norm_5120_kernel[(1,)](self.static_x[0], self.aux[prefix + "attn_norm.weight"], self.static_h[0], num_warps=4)

            if i % 4 != 3:
                l_qkv = self.layers[prefix + "attn_qkv.weight"]
                self.engine.prepare_input_hadamard(self.static_h[0], l_qkv["col_scale"], 5120)
                self.engine.run_gemv_prepped(l_qkv, self.static_qkv)

                l_gate = self.layers[prefix + "attn_gate.weight"]
                self.engine.prepare_input_hadamard(self.static_h[0], l_gate["col_scale"], 5120)
                self.engine.run_gemv_prepped(l_gate, self.static_gate)

                alpha = F.linear(self.static_h, self.aux[prefix + "ssm_alpha.weight"])[0]
                beta = F.linear(self.static_h, self.aux[prefix + "ssm_beta.weight"])[0]

                _fused_conv1d_silu_kernel[(40,)](
                    self.static_qkv,
                    self.conv_states[i],
                    self.aux[prefix + "ssm_conv1d.weight"],
                    self.static_qkv_conv,
                    num_warps=4,
                )

                _fused_deltanet_step_kernel[(48,)](
                    self.static_qkv_conv,
                    self.static_gate,
                    alpha,
                    beta,
                    self.aux[prefix + "ssm_dt.bias"],
                    self.aux[prefix + "ssm_a"],
                    self.aux[prefix + "ssm_norm.weight"],
                    self.ssm_states[i],
                    self.static_y_gated,
                    num_warps=4,
                )

                l_out = self.layers[prefix + "ssm_out.weight"]
                self.engine.prepare_input_hadamard(self.static_y_gated, l_out["col_scale"], 6144)
                self.engine.run_gemv_prepped(l_out, self.static_x[0], res_1d=self.static_x[0])

            else:
                l_q = self.layers[prefix + "attn_q.weight"]
                self.engine.prepare_input_hadamard(self.static_h[0], l_q["col_scale"], 5120)
                self.engine.run_gemv_prepped(l_q, self.static_q_full)

                l_k = self.layers[prefix + "attn_k.weight"]
                self.engine.prepare_input_hadamard(self.static_h[0], l_k["col_scale"], 5120)
                self.engine.run_gemv_prepped(l_k, self.static_k)

                l_v = self.layers[prefix + "attn_v.weight"]
                self.engine.prepare_input_hadamard(self.static_h[0], l_v["col_scale"], 5120)
                self.engine.run_gemv_prepped(l_v, self.static_v)

                _fused_gqa_decode_step_kernel[(24,)](
                    self.static_q_full,
                    self.static_k,
                    self.static_v,
                    self.aux[prefix + "attn_q_norm.weight"],
                    self.aux[prefix + "attn_k_norm.weight"],
                    self.rope_cos,
                    self.rope_sin,
                    self.static_pos,
                    self.k_cache[i],
                    self.v_cache[i],
                    self.static_ctx,
                    MAX_SEQ=self.max_seq_len,
                    num_warps=4,
                )

                l_ao = self.layers[prefix + "attn_output.weight"]
                self.engine.prepare_input_hadamard(self.static_ctx, l_ao["col_scale"], 6144)
                self.engine.run_gemv_prepped(l_ao, self.static_x[0], res_1d=self.static_x[0])

            _rms_norm_5120_kernel[(1,)](self.static_x[0], self.aux[prefix + "post_attention_norm.weight"], self.static_h[0], num_warps=4)

            l_fg = self.layers[prefix + "ffn_gate.weight"]
            self.engine.prepare_input_hadamard(self.static_h[0], l_fg["col_scale"], 5120)
            self.engine.run_gemv_prepped(l_fg, self.static_ffn_gate)

            l_fu = self.layers[prefix + "ffn_up.weight"]
            self.engine.prepare_input_hadamard(self.static_h[0], l_fu["col_scale"], 5120)
            self.engine.run_gemv_prepped(l_fu, self.static_ffn_up)

            _swiglu_17408_kernel[(68,)](self.static_ffn_gate, self.static_ffn_up, self.static_ffn_gate, num_warps=4)

            l_fd = self.layers[prefix + "ffn_down.weight"]
            self.engine.prepare_input_hadamard(self.static_ffn_gate, l_fd["col_scale"], 17408)
            self.engine.run_gemv_prepped(l_fd, self.static_x[0], res_1d=self.static_x[0])

        _rms_norm_5120_kernel[(1,)](self.static_x[0], self.aux["output_norm.weight"], self.static_h[0], num_warps=4)
        p_vram = self.head_vram["packed_u8"]
        s_vram = self.head_vram["scales"]
        N_v = p_vram.shape[0]
        _int4_head_gemv_kernel[(N_v // 8,)](
            self.static_h[0],
            p_vram,
            s_vram,
            self.static_logits[0],
            N_v,
            BLOCK_K=256,
            num_warps=4,
        )

    def capture_cuda_graph(self):
        print("[*] Compiling Triton kernels and capturing CUDA Graph...")
        self.static_pos.fill_(1)
        self.static_x.copy_(self.embed_tokens([248045]))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._graph_decode_body()
        torch.cuda.current_stream().wait_stream(s)

        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self._graph_decode_body()
        torch.cuda.empty_cache()
        print("[OK] CUDA Graph captured successfully!")

    @torch.no_grad()
    def step_with_graph(self, token_id: int, cur_pos: int) -> torch.Tensor:
        self.static_x.copy_(self.embed_tokens([token_id]))
        self.static_pos.fill_(cur_pos)
        self.cuda_graph.replay()
        return self.static_logits

    @torch.no_grad()
    def forward_prefill(self, token_ids: list) -> torch.Tensor:
        all_embs = self.embed_tokens(token_ids)
        for idx in range(len(token_ids)):
            self.static_x.copy_(all_embs[idx:idx+1])
            self.static_pos.fill_(idx)
            self.cuda_graph.replay()
        return self.static_logits


def main():
    print("=" * 78)
    print(" QWEN 3.8 (27B) + HOLO-V8 FINAL ENGINE (V6 STRUCTURE + GRAPH PREFILL)")
    print("=" * 78)

    tokenizer = GGUFNativeTokenizer()

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=16384)
    runtime.load_to_vram()
    runtime.capture_cuda_graph()

    print("=" * 78)
    print(" Chat ready! Context: 8,192 tokens | Max output: 2,048 tokens.")
    print(" Type your message, or use '/think <message>' for logic puzzles:")
    print("=" * 78)

    while True:
        try:
            user_msg = input("\n[Tú]: ").strip()
            if user_msg.lower() in ["exit", "quit"]:
                break
            if not user_msg:
                continue

            enable_thinking = False
            if user_msg.startswith("/think "):
                enable_thinking = True
                user_msg = user_msg[7:].strip()

            runtime.init_states()
            prompt_ids = tokenizer.encode_chat(user_msg, enable_thinking=enable_thinking)
            mode_str = "WITH DIRECT <think> REASONING" if enable_thinking else "DIRECT"
            print(f"[*] Graph Prefill ({len(prompt_ids)} tokens | Mode: {mode_str})...")

            t_prefill_0 = time.perf_counter()
            logits = runtime.forward_prefill(prompt_ids)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t_prefill_0
            tps_prefill = len(prompt_ids) / max(t_prefill, 1e-4)
            print(f"[OK] Prefill completed in {t_prefill:.2f}s ({tps_prefill:.2f} tok/s). Generating response:\n")

            print("[Qwen 3.8 27B]: ", end="", flush=True)
            if enable_thinking:
                print("<think>\n", end="", flush=True)

            generated = []
            segment_ids = []
            printed_text = ""
            cur_pos = len(prompt_ids)
            in_think = enable_thinking
            t_gen_0 = time.perf_counter()

            for step in range(2048):
                raw_logits = logits[0].float()
                raw_logits[tokenizer.think_start_id] = -float("inf")
                if not in_think:
                    raw_logits[tokenizer.think_end_id] = -float("inf")

                for prev_id in set(generated[-24:]):
                    if raw_logits[prev_id] > 0:
                        raw_logits[prev_id] /= 1.05
                    else:
                        raw_logits[prev_id] *= 1.05

                scaled_logits = raw_logits / 0.3
                top_vals, top_idx = torch.topk(scaled_logits, k=5)
                probs = F.softmax(top_vals, dim=-1)
                sampled_rel = torch.multinomial(probs, num_samples=1).item()
                next_id = int(top_idx[sampled_rel].item())

                if next_id in [tokenizer.eos_token_id, tokenizer.im_end_id, tokenizer.im_start_id]:
                    break

                if next_id == tokenizer.think_end_id:
                    in_think = False
                    print("\n</think>\n\n", end="", flush=True)
                    segment_ids = []
                    printed_text = ""
                else:
                    segment_ids.append(next_id)
                    full_dec = tokenizer.decode(segment_ids)
                    new_chars = full_dec[len(printed_text):]
                    if not in_think and ("</br>" in full_dec or "</antml>" in full_dec):
                        break
                    if new_chars:
                        print(new_chars, end="", flush=True)
                        printed_text = full_dec

                generated.append(next_id)
                logits = runtime.step_with_graph(next_id, cur_pos)
                cur_pos += 1

            torch.cuda.synchronize()
            t_gen = time.perf_counter() - t_gen_0
            tps = len(generated) / max(t_gen, 1e-4)
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"\n\n   [Prefill: {t_prefill:.2f}s ({tps_prefill:.1f} t/s) | Gen: {t_gen:.2f}s ({tps:.2f} tok/s) | Tokens: {len(generated)} | VRAM: {vram_alloc:.2f} GB (Res: {vram_res:.2f} GB) / 12.00 GB]")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
