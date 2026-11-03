import os
import time
import math
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader, dequantize

from holocell_v8_qwen38_engine import (
    GGUF_PATH,
    CACHE_DIR,
    GGUFNativeTokenizer,
    HoloCellV7Engine,
    Qwen38HoloRuntime,
    _rms_norm_5120_kernel,
    _swiglu_17408_kernel,
    _fused_conv1d_silu_kernel,
    _fused_deltanet_step_kernel,
    _fused_gqa_decode_step_kernel,
)

RESULTS_DIR = os.path.expanduser("~/holovram_lab/results_overnight")
ZIP_PATH = os.path.join(RESULTS_DIR, "wikitext-2-raw-v1.zip")
ACT_CACHE_PATH = os.path.expanduser("~/holovram_lab/holo_v8_act_scales.pt")


def extract_split_from_zip(split_filename):
    target_path = os.path.join(RESULTS_DIR, split_filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 50000:
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if name.endswith(split_filename):
                data = z.read(name).decode("utf-8")
                with open(target_path, "w", encoding="utf-8") as out_f:
                    out_f.write(data)
                return data
    raise RuntimeError(f"No se encontró {split_filename}")


@torch.no_grad()
def collect_activation_rms(train_text: str):
    if os.path.exists(ACT_CACHE_PATH):
        print(f"[*] Loading precomputed activation profile from {ACT_CACHE_PATH}...")
        return torch.load(ACT_CACHE_PATH, map_location="cuda")

    print("[*] Step 3.1: Capturing real activation profile E[x^2] from wiki.train.raw (1024 tokens)...")
    reader = GGUFReader(GGUF_PATH, mode="r")
    tokenizer = GGUFNativeTokenizer(reader)
    del reader

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=2048)
    runtime.load_to_vram()

    tokens = tokenizer.encode_text(train_text)[:1024]
    act_sq = {
        "h_attn":    {i: torch.zeros((5120,), dtype=torch.float64, device="cuda") for i in range(64)},
        "post_attn": {i: torch.zeros((6144,), dtype=torch.float64, device="cuda") for i in range(64)},
        "h_ffn":     {i: torch.zeros((5120,), dtype=torch.float64, device="cuda") for i in range(64)},
        "swiglu":    {i: torch.zeros((17408,), dtype=torch.float64, device="cuda") for i in range(64)},
    }

    runtime.init_states()
    all_embs = runtime.embed_tokens(tokens)

    for pos in range(len(tokens)):
        if pos % 512 == 0:
            runtime.init_states()
        cur_pos = pos % 512
        runtime.static_x.copy_(all_embs[pos : pos + 1])
        runtime.static_pos.fill_(cur_pos)

        for i in range(64):
            prefix = f"blk.{i}."
            _rms_norm_5120_kernel[(1,)](
                runtime.static_x[0], runtime.aux[prefix + "attn_norm.weight"], runtime.static_h[0], num_warps=4
            )
            act_sq["h_attn"][i] += runtime.static_h[0].double().pow(2)

            if i % 4 != 3:
                l_qkv = runtime.layers[prefix + "attn_qkv.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_qkv["col_scale"], 5120)
                runtime.engine.run_gemv_prepped(l_qkv, runtime.static_qkv)

                l_gate = runtime.layers[prefix + "attn_gate.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_gate["col_scale"], 5120)
                runtime.engine.run_gemv_prepped(l_gate, runtime.static_gate)

                alpha = F.linear(runtime.static_h, runtime.aux[prefix + "ssm_alpha.weight"])[0]
                beta = F.linear(runtime.static_h, runtime.aux[prefix + "ssm_beta.weight"])[0]

                _fused_conv1d_silu_kernel[(40,)](
                    runtime.static_qkv, runtime.conv_states[i], runtime.aux[prefix + "ssm_conv1d.weight"],
                    runtime.static_qkv_conv, num_warps=4
                )
                _fused_deltanet_step_kernel[(48,)](
                    runtime.static_qkv_conv, runtime.static_gate, alpha, beta,
                    runtime.aux[prefix + "ssm_dt.bias"], runtime.aux[prefix + "ssm_a"],
                    runtime.aux[prefix + "ssm_norm.weight"], runtime.ssm_states[i],
                    runtime.static_y_gated, num_warps=4
                )
                act_sq["post_attn"][i] += runtime.static_y_gated.double().pow(2)

                l_out = runtime.layers[prefix + "ssm_out.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_y_gated, l_out["col_scale"], 6144)
                runtime.engine.run_gemv_prepped(l_out, runtime.static_x[0], res_1d=runtime.static_x[0])
            else:
                l_q = runtime.layers[prefix + "attn_q.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_q["col_scale"], 5120)
                runtime.engine.run_gemv_prepped(l_q, runtime.static_q_full)

                l_k = runtime.layers[prefix + "attn_k.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_k["col_scale"], 5120)
                runtime.engine.run_gemv_prepped(l_k, runtime.static_k)

                l_v = runtime.layers[prefix + "attn_v.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_v["col_scale"], 5120)
                runtime.engine.run_gemv_prepped(l_v, runtime.static_v)

                _fused_gqa_decode_step_kernel[(24,)](
                    runtime.static_q_full, runtime.static_k, runtime.static_v,
                    runtime.aux[prefix + "attn_q_norm.weight"], runtime.aux[prefix + "attn_k_norm.weight"],
                    runtime.rope_cos, runtime.rope_sin, runtime.static_pos,
                    runtime.k_cache[i], runtime.v_cache[i], runtime.static_ctx,
                    MAX_SEQ=runtime.max_seq_len, num_warps=4
                )
                act_sq["post_attn"][i] += runtime.static_ctx.double().pow(2)

                l_ao = runtime.layers[prefix + "attn_output.weight"]
                runtime.engine.prepare_input_hadamard(runtime.static_ctx, l_ao["col_scale"], 6144)
                runtime.engine.run_gemv_prepped(l_ao, runtime.static_x[0], res_1d=runtime.static_x[0])

            _rms_norm_5120_kernel[(1,)](
                runtime.static_x[0], runtime.aux[prefix + "post_attention_norm.weight"], runtime.static_h[0], num_warps=4
            )
            act_sq["h_ffn"][i] += runtime.static_h[0].double().pow(2)

            l_fg = runtime.layers[prefix + "ffn_gate.weight"]
            runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_fg["col_scale"], 5120)
            runtime.engine.run_gemv_prepped(l_fg, runtime.static_ffn_gate)

            l_fu = runtime.layers[prefix + "ffn_up.weight"]
            runtime.engine.prepare_input_hadamard(runtime.static_h[0], l_fu["col_scale"], 5120)
            runtime.engine.run_gemv_prepped(l_fu, runtime.static_ffn_up)

            _swiglu_17408_kernel[(68,)](
                runtime.static_ffn_gate, runtime.static_ffn_up, runtime.static_ffn_gate, num_warps=4
            )
            act_sq["swiglu"][i] += runtime.static_ffn_gate.double().pow(2)

            l_fd = runtime.layers[prefix + "ffn_down.weight"]
            runtime.engine.prepare_input_hadamard(runtime.static_ffn_gate, l_fd["col_scale"], 17408)
            runtime.engine.run_gemv_prepped(l_fd, runtime.static_x[0], res_1d=runtime.static_x[0])

    act_rms = {}
    for k_group in act_sq:
        act_rms[k_group] = {}
        for i in range(64):
            rms = torch.sqrt(act_sq[k_group][i] / len(tokens)).float().clamp(min=1e-5)
            # Normalizar a media 1.0 para modular importancia relativa entre canales
            act_rms[k_group][i] = (rms / torch.mean(rms)).clamp(min=0.25, max=4.0)

    del runtime, engine
    torch.cuda.empty_cache()
    torch.save(act_rms, ACT_CACHE_PATH)
    print("[OK] Activation profile saved!")
    return act_rms


class HoloV8AWQCompressor:
    def __init__(self, device="cuda"):
        self.device = device
        h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
        h = h2
        for _ in range(5):
            h = torch.kron(h, h2)
        self.H64 = (h / 8.0).to(torch.float16).contiguous()

        lut_u8 = torch.arange(256, dtype=torch.uint8, device=device)
        lut_fp16 = lut_u8.view(torch.float8_e4m3fn).to(torch.float16)
        self.fp8_lut = torch.nan_to_num(lut_fp16, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    @torch.no_grad()
    def adapt_codebook_kmeans_v8(self, data_samples: torch.Tensor, k=256, iters=12):
        n_total = data_samples.shape[0]
        sample_size = min(32768, n_total)
        perm = torch.randperm(n_total, device=self.device)
        samples = data_samples[perm[:sample_size]].float()

        norms = torch.sum(samples * samples, dim=-1) + 1e-6
        probs = norms / torch.sum(norms)
        init_idx = torch.multinomial(probs, num_samples=k, replacement=False)
        centroids = samples[init_idx].clone()

        chunk_size = 4096
        for _ in range(iters):
            assignments_list = []
            for i in range(0, samples.shape[0], chunk_size):
                chunk = samples[i : i + chunk_size]
                dists = torch.cdist(chunk, centroids, p=2.0)
                assignments_list.append(torch.argmin(dists, dim=1))
            assignments = torch.cat(assignments_list)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros((k, 1), device=self.device, dtype=torch.float32)
            ones = torch.ones((samples.shape[0], 1), device=self.device, dtype=torch.float32)
            new_centroids.index_add_(0, assignments, samples)
            counts.index_add_(0, assignments, ones)

            mask = counts.squeeze(-1) > 0
            centroids[mask] = new_centroids[mask] / counts[mask]

        return centroids.to(torch.float16)

    @torch.no_grad()
    def compress_layer_awq(self, weights_fp16: torch.Tensor, act_importance: torch.Tensor, top_k_per_row: int, residual_boost: float):
        N, K = weights_fp16.shape
        act_imp = act_importance.view(1, K).float()

        # 1. AWQ + Column Norm Hybrid Pre-Scaling:
        # If an input channel has high activation, we slightly reduce col_scale in that channel
        # so that w_balanced = W / col_scale becomes LARGER in that channel and receives more
        # K-Means resolution and outlier priority!
        w_col_rms = torch.sqrt(torch.mean(weights_fp16.float() ** 2, dim=0, keepdim=True)).clamp(min=1e-6)
        col_scale = (w_col_rms / (act_imp ** 0.35)).clamp(min=1e-6).to(torch.float16)
        w_balanced = weights_fp16 / col_scale

        # 2. Hessian-Guided Outlier Selection: |W_balanced| * (act_imp ** 0.25)
        salience = torch.abs(w_balanced.float()) * (act_imp ** 0.25)
        _, topk_cols = torch.topk(salience, k=top_k_per_row, dim=-1, largest=True, sorted=False)

        outlier_vals = torch.gather(w_balanced, -1, topk_cols)
        out_row_scale = (torch.amax(torch.abs(outlier_vals), dim=-1, keepdim=True) / 240.0).clamp(min=1e-6)
        outlier_vals_fp8 = (outlier_vals / out_row_scale).to(torch.float8_e4m3fn)
        outlier_cols_u16 = topk_cols.to(torch.int16)

        clean_w = w_balanced.clone()
        clean_w.scatter_(-1, topk_cols, 0.0)

        # 3. Hadamard-64 Rotation + FP8 Scale
        blocks_64 = clean_w.view(N, K // 64, 64)
        h_blocks = torch.matmul(blocks_64, self.H64)

        block_std = torch.sqrt(torch.mean(h_blocks.float() ** 2, dim=-1, keepdim=True)).clamp(min=1e-6, max=400.0).to(torch.float16)
        scale_fp8 = block_std.squeeze(-1).to(torch.float8_e4m3fn)
        std_rec = self.fp8_lut[scale_fp8.view(torch.uint8).long()].unsqueeze(-1).clamp(min=1e-6)

        norm_h = (h_blocks / std_rec).view(-1, 4)

        # 4. 12-iteration K-Means
        dynamic_codebook = self.adapt_codebook_kmeans_v8(norm_h, k=256, iters=12)
        cb_norm_sq = torch.sum(dynamic_codebook.float() ** 2, dim=-1)

        # 5. Vector Quantization
        norm_h_reshaped = norm_h.view(N, K // 4, 4).float()
        idx_uint8 = torch.empty((N, K // 4), dtype=torch.uint8, device=self.device)
        cb_f32_t = dynamic_codebook.float().t()
        chunk_size = 512
        for start_r in range(0, N, chunk_size):
            end_r = min(start_r + chunk_size, N)
            sub_x = norm_h_reshaped[start_r:end_r]
            dots = torch.matmul(sub_x, cb_f32_t)
            scores = cb_norm_sq - 2.0 * dots
            idx_uint8[start_r:end_r] = torch.argmin(scores, dim=-1).to(torch.uint8)

        # 6. Analytical L2 Contraction Self-Compensation
        cb_max = torch.amax(torch.abs(dynamic_codebook.float())).clamp(min=1e-6)
        cb_s = cb_max / 127.0
        cb_q = torch.round(dynamic_codebook.float() / cb_s).clamp(-128, 127) * cb_s
        h_rec_norm = cb_q[idx_uint8.long()].view(N, K // 64, 64)
        h_rec = (h_rec_norm * std_rec.float()).double()
        h_true = h_blocks.double()

        norm_true = torch.sqrt(torch.sum(h_true * h_true)).item()
        norm_rec = max(torch.sqrt(torch.sum(h_rec * h_rec)).item(), 1e-8)
        l2_ratio = min(max(norm_true / norm_rec, 1.0), 1.12)
        dynamic_codebook = (dynamic_codebook.float() * l2_ratio * residual_boost).to(torch.float16)

        idx_grouped = idx_uint8.view(N, K // 16, 4).to(torch.int32)
        packed_vram_cell = (
            idx_grouped[:, :, 0] |
            (idx_grouped[:, :, 1] << 8) |
            (idx_grouped[:, :, 2] << 16) |
            (idx_grouped[:, :, 3] << 24)
        )

        return (
            packed_vram_cell,
            scale_fp8,
            col_scale,
            outlier_cols_u16,
            outlier_vals_fp8,
            out_row_scale,
            dynamic_codebook,
            l2_ratio * residual_boost,
        )


def get_layer_config(t_name: str, act_rms: dict):
    parts = t_name.split(".")
    blk_idx = int(parts[1])
    suffix = ".".join(parts[2:])

    if suffix == "ffn_down.weight":
        return 768, 1.035, act_rms["swiglu"][blk_idx]
    elif suffix in ("ssm_out.weight", "attn_output.weight"):
        return 512, 1.025, act_rms["post_attn"][blk_idx]
    elif suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
        return 512, 1.020, act_rms["h_attn"][blk_idx]
    elif suffix in ("attn_qkv.weight", "attn_gate.weight"):
        # Subimos de 220 a 384 (3 bloques exactos de 128) para proteger la recurrencia DeltaNet
        return 384, 1.015, act_rms["h_attn"][blk_idx]
    elif suffix in ("ffn_gate.weight", "ffn_up.weight"):
        # Subimos de 220 a 256 (rellena el warp de 256 hilos con coste 0 de instrucciones)
        return 256, 1.025, act_rms["h_ffn"][blk_idx]
    else:
        raise ValueError(f"Capa desconocida: {t_name}")


def upgrade_all_400_matrices_awq(act_rms: dict):
    reader = GGUFReader(GGUF_PATH, mode="r")
    gguf_map = {t.name: t for t in reader.tensors}
    compressor = HoloV8AWQCompressor(device="cuda")

    files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("blk.") and f.endswith(".pt")])
    t0 = time.perf_counter()

    print(f"\n[*] Step 3.2: Re-packing 400 matrices with AWQ Hessian-Guided Outliers + 12-iter K-Means...")
    for idx, f in enumerate(files):
        t_name = f[:-3]
        dst_path = os.path.join(CACHE_DIR, f)

        topk, res_boost, act_imp = get_layer_config(t_name, act_rms)
        t_info = gguf_map[t_name]
        raw_np = dequantize(t_info.data, t_info.tensor_type)
        K_orig, N_orig = int(t_info.shape[0]), int(t_info.shape[1])
        w_fp16 = torch.from_numpy(raw_np).view(N_orig, K_orig).to(device="cuda", dtype=torch.float16)
        if w_fp16.shape[1] % 64 != 0:
            w_fp16 = w_fp16.t().contiguous()

        packed, s8, c_scale, o_cols, o_vals, o_rscale, d_cb, eff_gain = compressor.compress_layer_awq(
            w_fp16, act_imp, top_k_per_row=topk, residual_boost=res_boost
        )

        torch.save({
            "name": t_name,
            "packed": packed.cpu(),
            "scale_fp8": s8.cpu(),
            "col_scale": c_scale.cpu(),
            "outlier_cols": o_cols.cpu(),
            "outlier_vals": o_vals.cpu(),
            "out_rscale": o_rscale.cpu(),
            "dynamic_codebook": d_cb.cpu(),
        }, dst_path)

        del w_fp16, packed, s8, c_scale, o_cols, o_vals, o_rscale, d_cb
        if (idx + 1) % 25 == 0 or (idx + 1) == len(files):
            el = time.perf_counter() - t0
            print(f"    [AWQ-V8 {idx+1:3d}/400] {t_name:28s} | TOPK={topk} | Gain={eff_gain:.4f} ({el:.1f}s)")

    torch.cuda.empty_cache()
    print("[OK] All 400 matrices upgraded to Holo-V8 AWQ Full HD!")


@torch.no_grad()
def eval_and_fine_tune(train_text: str, test_text: str):
    print("\n" + "=" * 80)
    print(" STEP 3.3: FINE-TUNING CALIBRATION AND OFFICIAL WIKITEXT-2-RAW-V1 EVALUATION")
    print("=" * 80)
    print("=" * 80)

    reader = GGUFReader(GGUF_PATH, mode="r")
    tokenizer = GGUFNativeTokenizer(reader)
    del reader

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=4096)
    runtime.load_to_vram()

    runtime._orig_cb = {name: l["cb_scale"].clone() for name, l in runtime.layers.items()}
    runtime.capture_cuda_graph()

    train_ids = tokenizer.encode_text(train_text)
    test_ids = tokenizer.encode_text(test_text)

    def eval_blocks(token_ids, n_blocks):
        nll, cnt = 0.0, 0
        for b in range(n_blocks):
            chunk = token_ids[b * 512 : (b + 1) * 512]
            runtime.init_states()
            embs = runtime.embed_tokens(chunk)
            for i in range(511):
                runtime.static_x.copy_(embs[i : i + 1])
                runtime.static_pos.fill_(i)
                runtime.cuda_graph.replay()
                if i >= 256:
                    logits = runtime.static_logits[0].float()
                    log_probs = F.log_softmax(logits, dim=-1)
                    nll += -log_probs[chunk[i + 1]].item()
                    cnt += 1
        torch.cuda.synchronize()
        return math.exp(nll / cnt)

    # Ajuste fino rápido de escala global post-AWQ en 2 bloques de Train
    best_ppl = 999.0
    best_trim = (1.0, 1.0)
    for t_ffn in [0.985, 1.000, 1.015]:
        for t_att in [0.985, 1.000, 1.015]:
            for name, orig_s in runtime._orig_cb.items():
                g = t_ffn if "ffn_" in name else t_att
                runtime.layers[name]["cb_scale"].copy_(orig_s * g)
            p = eval_blocks(train_ids, 2)
            print(f"    [Fine-Trim] ffn={t_ffn:.3f}, att={t_att:.3f} -> Train PPL (2b): {p:.4f}")
            if p < best_ppl:
                best_ppl = p
                best_trim = (t_ffn, t_att)

    t_ffn, t_att = best_trim
    for name, orig_s in runtime._orig_cb.items():
        g = t_ffn if "ffn_" in name else t_att
        runtime.layers[name]["cb_scale"].copy_(orig_s * g)

    tr4 = eval_blocks(train_ids, 4)
    print(f"[*] PPL on wiki.train.raw (4 blocks): {tr4:.4f} (V7 Base: 14.0537 | Phase 2B: 12.1449)")

    print("\n[*] Evaluating 20 official blocks from WIKI.TEST.RAW...")
    nll_test = 0.0
    cnt_test = 0
    t0 = time.perf_counter()

    for b in range(20):
        chunk = test_ids[b * 512 : (b + 1) * 512]
        runtime.init_states()
        embs = runtime.embed_tokens(chunk)
        blk_nll = 0.0
        blk_cnt = 0
        for i in range(511):
            runtime.static_x.copy_(embs[i : i + 1])
            runtime.static_pos.fill_(i)
            runtime.cuda_graph.replay()
            if i >= 256:
                logits = runtime.static_logits[0].float()
                log_probs = F.log_softmax(logits, dim=-1)
                loss_i = -log_probs[chunk[i + 1]].item()
                nll_test += loss_i
                blk_nll += loss_i
                cnt_test += 1
                blk_cnt += 1
        torch.cuda.synchronize()
        bppl = math.exp(blk_nll / blk_cnt)
        cppl = math.exp(nll_test / cnt_test)
        el = time.perf_counter() - t0
        print(f"  [Test Block {b+1:02d}/20] Block PPL: {bppl:6.2f} | Cumulative PPL: {cppl:6.4f} ({el:.1f}s)")

    final_test_ppl = math.exp(nll_test / cnt_test)
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    print("\n" + "=" * 80)
    print(f" OFFICIAL HOLO-V8 AWQ FULL-HD RESULT ON WIKITEXT-2-RAW-V1:")
    print(f"   - Final Subword Token PPL  : {final_test_ppl:.4f} (V7: 10.3420 -> 2A: 9.4676 -> 2B: 9.0993)")
    print(f"   - VRAM Allocated / Reserved: {vram_gb:.2f} GB / {res_gb:.2f} GB (de 12.00 GB)")
    print("=" * 80)


if __name__ == "__main__":
    train_text = extract_split_from_zip("wiki.train.raw")
    test_text = extract_split_from_zip("wiki.test.raw")
    act_rms = collect_activation_rms(train_text)
    upgrade_all_400_matrices_awq(act_rms)
    eval_and_fine_tune(train_text, test_text)
