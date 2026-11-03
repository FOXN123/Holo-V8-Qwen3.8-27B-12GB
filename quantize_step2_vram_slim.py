import os
import time
import math
import zipfile
import torch
import torch.nn.functional as F
from gguf import GGUFReader, dequantize

from holocell_v8_qwen38_engine import (
    GGUF_PATH,
    CACHE_DIR,
    GGUFNativeTokenizer,
    HoloCellV7Engine,
    Qwen38HoloRuntime,
)
from run_holo_v8_awq_hessian_full import (
    ACT_CACHE_PATH,
    HoloV8AWQCompressor,
    extract_split_from_zip,
)

def slim_non_critical_224_matrices():
    print("=" * 80)
    print(" PHASE 5.1: SLIMMING 224 INTERNAL MATRICES TO FREE ~418 MiB (8K CTX MODE)")
    print("=" * 80)

    act_rms = torch.load(ACT_CACHE_PATH, map_location="cuda")
    reader = GGUFReader(GGUF_PATH, mode="r")
    gguf_map = {t.name: t for t in reader.tensors}
    compressor = HoloV8AWQCompressor(device="cuda")

    files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("blk.") and f.endswith(".pt")])
    t0 = time.perf_counter()
    count = 0

    for f in files:
        t_name = f[:-3]
        parts = t_name.split(".")
        blk_idx = int(parts[1])
        suffix = ".".join(parts[2:])

        # We only touch the 224 non-residual matrices (the 176 critical ones at 768/512 remain intact)
        if suffix in ("ffn_gate.weight", "ffn_up.weight"):
            topk = 128
            res_boost = 1.025 * 0.9550  # Already incorporates the winning scale 0.9550 from Phase 4
            act_imp = act_rms["h_ffn"][blk_idx]
        elif suffix in ("attn_qkv.weight", "attn_gate.weight"):
            topk = 256
            res_boost = 1.015 * 0.9550  # Already incorporates the winning scale 0.9550 from Phase 4
            act_imp = act_rms["h_attn"][blk_idx]
        else:
            continue

        t_info = gguf_map[t_name]
        raw_np = dequantize(t_info.data, t_info.tensor_type)
        K_orig, N_orig = int(t_info.shape[0]), int(t_info.shape[1])
        w_fp16 = torch.from_numpy(raw_np).view(N_orig, K_orig).to(device="cuda", dtype=torch.float16)
        if w_fp16.shape[1] % 64 != 0:
            w_fp16 = w_fp16.t().contiguous()

        packed, s8, c_scale, o_cols, o_vals, o_rscale, d_cb, eff_gain = compressor.compress_layer_awq(
            w_fp16, act_imp, top_k_per_row=topk, residual_boost=res_boost
        )
        o_rscale = (o_rscale.float() * 1.0050).to(o_rscale.dtype)

        torch.save({
            "name": t_name,
            "packed": packed.cpu(),
            "scale_fp8": s8.cpu(),
            "col_scale": c_scale.cpu(),
            "outlier_cols": o_cols.cpu(),
            "outlier_vals": o_vals.cpu(),
            "out_rscale": o_rscale.cpu(),
            "dynamic_codebook": d_cb.cpu(),
        }, os.path.join(CACHE_DIR, f))

        del w_fp16, packed, s8, c_scale, o_cols, o_vals, o_rscale, d_cb
        count += 1
        if count % 28 == 0 or count == 224:
            el = time.perf_counter() - t0
            print(f"    [Slim-8K {count:3d}/224] {t_name:28s} | TOPK={topk} | Gain={eff_gain:.4f} ({el:.1f}s)")

    torch.cuda.empty_cache()
    print("[OK] ¡224 internal matrices optimized and ~418 MiB of VRAM freed!")


@torch.no_grad()
def micro_tune_and_eval_8k():
    print("\n" + "=" * 80)
    print(" PHASE 5.2: LOAD WITH 8192 CONTEXT AND FINE-GRAINED SCALE MICRO-TUNING")
    print("=" * 80)

    train_text = extract_split_from_zip("wiki.train.raw")
    test_text = extract_split_from_zip("wiki.test.raw")

    reader = GGUFReader(GGUF_PATH, mode="r")
    tokenizer = GGUFNativeTokenizer(reader)
    del reader

    # Temporarily revert the 0.96 of output_norm to calibrate logit_temp cleanly from 1.0
    aux_path = os.path.join(CACHE_DIR, "aux_tensors.pt")
    aux_backup = os.path.expanduser("~/holovram_lab/cache_holocell_v7/aux_tensors.pt")
    orig_aux = torch.load(aux_backup, map_location="cpu")
    cur_aux = torch.load(aux_path, map_location="cpu")
    cur_aux["output_norm.weight"] = orig_aux["output_norm.weight"].clone()
    torch.save(cur_aux, aux_path)

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=8192)
    runtime.load_to_vram()

    runtime._orig_scales = {
        name: {
            "cb_scale": l["cb_scale"].clone(),
            "out_rscale": l["out_rscale"].clone(),
        }
        for name, l in runtime.layers.items()
    }
    runtime.capture_cuda_graph()

    train_ids = tokenizer.encode_text(train_text)
    test_ids = tokenizer.encode_text(test_text)

    def eval_blocks(token_ids, n_blocks, logit_temp=1.0):
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
                    logits = runtime.static_logits[0].float() * logit_temp
                    log_probs = F.log_softmax(logits, dim=-1)
                    nll += -log_probs[chunk[i + 1]].item()
                    cnt += 1
        torch.cuda.synchronize()
        return math.exp(nll / cnt)

    def apply_deltas(d_down, d_inner, d_out):
        for name, orig in runtime._orig_scales.items():
            l = runtime.layers[name]
            if name.endswith(".ffn_down.weight") or name.endswith(".ssm_out.weight") or name.endswith(".attn_output.weight"):
                l["cb_scale"].copy_(orig["cb_scale"] * d_down)
            else:
                l["cb_scale"].copy_(orig["cb_scale"] * d_inner)
            l["out_rscale"].copy_(orig["out_rscale"] * d_out)

    best_ppl = 999.0
    best_cfg = (1.0, 1.0, 1.0)
    for d_down in [0.985, 0.995, 1.000, 1.005]:
        for d_inner in [0.990, 1.000, 1.010]:
            for d_out in [1.000, 1.008, 1.015]:
                apply_deltas(d_down, d_inner, d_out)
                p = eval_blocks(train_ids, 2, logit_temp=0.96)
                if p < best_ppl:
                    best_ppl = p
                    best_cfg = (d_down, d_inner, d_out)
                    print(f"    [Micro-Opt] res={d_down:.3f}, in={d_inner:.3f}, out={d_out:.3f} -> Train PPL (2b): {p:.4f}")

    dd, di, do = best_cfg
    apply_deltas(dd, di, do)

    best_lt = 0.96
    best_tr4 = 999.0
    for lt in [0.92, 0.94, 0.96, 0.98]:
        p4 = eval_blocks(train_ids, 4, logit_temp=lt)
        print(f"    [Logit-Temp 4b] lt={lt:.2f} -> Train PPL (4b): {p4:.4f}")
        if p4 < best_tr4:
            best_tr4 = p4
            best_lt = lt

    print(f"\n[OK] 8K Winner on Train (4 blks): {best_tr4:.4f} (res={dd:.3f}, in={di:.3f}, out={do:.3f}, lt={best_lt:.2f})")

    # Persist to disk
    files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("blk.") and f.endswith(".pt")])
    for f in files:
        t_name = f[:-3]
        pt_path = os.path.join(CACHE_DIR, f)
        d = torch.load(pt_path, map_location="cpu")
        cb_m = dd if (t_name.endswith(".ffn_down.weight") or t_name.endswith(".ssm_out.weight") or t_name.endswith(".attn_output.weight")) else di
        d["dynamic_codebook"] = (d["dynamic_codebook"].float() * cb_m).to(d["dynamic_codebook"].dtype)
        d["out_rscale"] = (d["out_rscale"].float() * do).to(d["out_rscale"].dtype)
        torch.save(d, pt_path)

    cur_aux["output_norm.weight"] = (orig_aux["output_norm.weight"].float() * best_lt).to(orig_aux["output_norm.weight"].dtype)
    torch.save(cur_aux, aux_path)
    runtime.aux["output_norm.weight"].copy_(cur_aux["output_norm.weight"].to("cuda"))

    print("\n[*] Evaluating the 20 official blocks of WIKI.TEST.RAW (with 8192 KV-Cache active)...")
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
    tps = (20 * 511) / (time.perf_counter() - t0)
    print("\n" + "=" * 80)
    print(f" HOLO-V8 RESULT (8192 CTX + SLIM-VRAM) ON WIKITEXT-2-RAW-V1:")
    print(f"   - Final Subword Token PPL : {final_test_ppl:.4f} (Backup 4K: 8.2265 | V7 Base: 10.3420)")
    print(f"   - VRAM (with 8192 KV-Cache): {vram_gb:.2f} GB allocated / {res_gb:.2f} GB reserved (de 12.00 GB)")
    print(f"   - Sustained Speed     : {tps:.2f} tok/s")
    print("=" * 80)

if __name__ == "__main__":
    slim_non_critical_224_matrices()
    micro_tune_and_eval_8k()
