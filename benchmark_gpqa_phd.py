import os
import re
import json
import time
import random
import argparse
import torch
from gguf import GGUFReader

from holocell_v8_qwen38_engine import (
    GGUF_PATH,
    GGUFNativeTokenizer,
    HoloCellV7Engine,
    Qwen38HoloRuntime,
)

RESULTS_DIR = os.path.expanduser("~/holovram_lab/results_overnight")
STATE_PATH = os.path.join(RESULTS_DIR, "gpqa_thinking_v8_state.json")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Evaluate all 198 questions (default: stratified sample of 40)")
    parser.add_argument("--max-think-tokens", type=int, default=1280)
    args = parser.parse_args()

    gpqa_file = os.path.join(RESULTS_DIR, "gpqa_diamond_authentic_198.json")
    with open(gpqa_file, "r", encoding="utf-8") as f:
        all_rows = json.load(f)

    # Deterministic stratified selection unless --full
    if args.full:
        target_indices = list(range(len(all_rows)))
    else:
        rng_sel = random.Random(42)
        target_indices = sorted(rng_sel.sample(range(len(all_rows)), min(40, len(all_rows))))

    state = {"done_ids": {}, "correct": 0, "total": 0}
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass

    tokenizer = GGUFNativeTokenizer()

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=16384)
    runtime.load_to_vram()
    runtime.capture_cuda_graph()

    choice_tokens = {k: tokenizer.encode_text(k)[0] for k in ["A", "B", "C", "D"]}
    labels = ["A", "B", "C", "D"]

    print("=" * 78)
    print(f" EVALUATING GPQA DIAMOND WITH <THINK> ENABLED ({len(target_indices)} PROBLEMS | {args.max_think_tokens} TOKS)")
    print("=" * 78)

    for count, idx in enumerate(target_indices, 1):
        qid = f"gpqa_think_{idx}"
        if qid in state["done_ids"]:
            continue

        item = all_rows[idx]
        q_text = item["Question"]
        c_ans = item["Correct Answer"]
        texts = [c_ans, item["Incorrect Answer 1"], item["Incorrect Answer 2"], item["Incorrect Answer 3"]]

        rng = random.Random(2026 + idx)
        rng.shuffle(texts)
        ans_key = labels[texts.index(c_ans)]
        domain = item.get("Subdomain", "PhD_Science")

        opts_str = "\n".join([f"{l}) {t}" for l, t in zip(labels, texts)])
        prompt = (
            f"Question: {q_text}\n\nOptions:\n{opts_str}\n\n"
            f"Think step-by-step concisely and end your response with 'Answer: X' where X is A, B, C, or D."
        )

        runtime.init_states()
        prompt_ids = tokenizer.encode_chat(prompt, enable_thinking=True)
        t0 = time.perf_counter()
        logits = runtime.forward_prefill(prompt_ids)
        cur_pos = len(prompt_ids)
        gen = []

        for step in range(args.max_think_tokens):
            raw_logits = logits[0].float()
            raw_logits[tokenizer.think_start_id] = -float("inf")
            next_id = int(torch.argmax(raw_logits).item())
            if next_id in [tokenizer.eos_token_id, tokenizer.im_end_id, tokenizer.im_start_id]:
                break
            gen.append(next_id)
            logits = runtime.step_with_graph(next_id, cur_pos)
            cur_pos += 1
            if len(gen) >= 64 and step % 16 == 0 and gen[-16:] == gen[-32:-16]:
                break

        out_text = tokenizer.decode(gen)
        after_think = out_text.split("</think>")[-1] if "</think>" in out_text else out_text
        m = re.findall(r"(?:Answer|Option|Choice)\s*(?:is|:)?\s*\**([ABCD])\b", after_think, re.IGNORECASE)
        if not m:
            m = re.findall(r"\\boxed\{\s*([ABCD])\s*\}", out_text)

        if m:
            pred_let = m[-1].upper()
        else:
            # If <think> tokens are exhausted without closing, we force </think> and read the direct logit of A, B, C, D
            force_ids = tokenizer.encode_text("\n</think>\nAnswer with the single letter (A, B, C, or D): ")
            for fid in force_ids:
                logits = runtime.step_with_graph(fid, cur_pos)
                cur_pos += 1
            l_vec = logits[0].float()
            pred_let = max(labels, key=lambda k: l_vec[choice_tokens[k]].item())

        torch.cuda.synchronize()
        tps = len(gen) / max(time.perf_counter() - t0, 1e-4)
        ok = (pred_let == ans_key)

        state["done_ids"][qid] = {"pred": pred_let, "gold": ans_key, "ok": ok, "domain": domain, "tokens": len(gen)}
        state["total"] = len(state["done_ids"])
        state["correct"] = sum(1 for v in state["done_ids"].values() if v["ok"])
        with open(STATE_PATH + ".tmp", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(STATE_PATH + ".tmp", STATE_PATH)

        acc = state["correct"] / state["total"] * 100.0
        print(f"  [GPQA-Think {count:03d}/{len(target_indices)}] #{idx:03d} ({domain[:16]:16s}) Pred: {pred_let} (Real: {ans_key}) -> {'[OK  ]' if ok else '[FAIL]'} | Acum: {state['correct']}/{state['total']} ({acc:.2f}%) [{len(gen)} toks, {tps:.1f} t/s]")

if __name__ == "__main__":
    main()
