import os
import sys
import json
import time
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import torch.nn.functional as F
from gguf import GGUFReader

from holocell_v8_qwen38_engine import (
    GGUF_PATH,
    GGUFNativeTokenizer,
    HoloCellV7Engine,
    Qwen38HoloRuntime,
)

MODEL_ID = "holocell-v8-qwen3.8-27b-2bit-16k"
GLOBAL_RUNTIME = None
GLOBAL_TOKENIZER = None
CACHED_SEQ_IDS = []
CACHED_LAST_LOGITS = None


def encode_messages_native(tokenizer, messages, enable_thinking=False):
    """
    Builds the token-ID sequence using exactly the native special tokens
    (im_start_id, im_end_id, think_start_id, think_end_id) from encode_chat.
    """
    nl_ids = tokenizer.encode_text("\n")
    ids = []
    for msg in messages:
        role = msg.get("role", "user").strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        ids.append(tokenizer.im_start_id)
        ids.extend(tokenizer.encode_text(f"{role}\n{content}"))
        ids.append(tokenizer.im_end_id)
        ids.extend(nl_ids)

    # Assistant header identical to GGUFNativeTokenizer.encode_chat()
    ids.append(tokenizer.im_start_id)
    ids.extend(tokenizer.encode_text("assistant\n"))
    if enable_thinking:
        ids.append(tokenizer.think_start_id)
        ids.extend(nl_ids)
    else:
        ids.append(tokenizer.think_start_id)
        ids.extend(nl_ids)
        ids.append(tokenizer.think_end_id)
        ids.extend(nl_ids)
    return ids


@torch.no_grad()
def prefill_with_prefix_cache(prompt_ids):
    global CACHED_SEQ_IDS, CACHED_LAST_LOGITS
    common_len = 0
    max_cmp = min(len(CACHED_SEQ_IDS), len(prompt_ids) - 1)
    while common_len < max_cmp and CACHED_SEQ_IDS[common_len] == prompt_ids[common_len]:
        common_len += 1

    if common_len > 8:
        delta_ids = prompt_ids[common_len:]
        embs = GLOBAL_RUNTIME.embed_tokens(delta_ids)
        for offset, pos_i in enumerate(range(common_len, len(prompt_ids))):
            GLOBAL_RUNTIME.static_x.copy_(embs[offset : offset + 1])
            GLOBAL_RUNTIME.static_pos.fill_(pos_i)
            GLOBAL_RUNTIME.cuda_graph.replay()
        torch.cuda.synchronize()
        CACHED_SEQ_IDS = list(prompt_ids)
        CACHED_LAST_LOGITS = GLOBAL_RUNTIME.static_logits.clone()
        return CACHED_LAST_LOGITS, len(delta_ids), common_len
    else:
        GLOBAL_RUNTIME.init_states()
        logits = GLOBAL_RUNTIME.forward_prefill(prompt_ids)
        torch.cuda.synchronize()
        CACHED_SEQ_IDS = list(prompt_ids)
        CACHED_LAST_LOGITS = logits.clone()
        return logits, len(prompt_ids), 0


@torch.no_grad()
def sample_next_token(logits_1d, generated_ids, temperature=0.3, top_p=0.9, top_k=40, rep_penalty=1.03):
    logits = logits_1d.float().clone()
    # Bloquear siempre im_start_id y think_start_id para que nunca emita etiquetas de sistema
    logits[GLOBAL_TOKENIZER.think_start_id] = -float("inf")
    logits[GLOBAL_TOKENIZER.im_start_id] = -float("inf")

    if rep_penalty != 1.0 and generated_ids:
        for tid in set(generated_ids[-128:]):
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty

    if temperature <= 1e-4:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0:
        v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
        probs = F.softmax(v, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            choice = torch.multinomial(sorted_probs, 1).item()
            return int(ix[sorted_idx[choice]].item())
        choice = torch.multinomial(probs, 1).item()
        return int(ix[choice].item())
    return int(torch.argmax(logits).item())


class OpenAIHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.end_headers()
        payload = {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "holocell-lab"}],
        }
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        global CACHED_SEQ_IDS
        if not (self.path.startswith("/v1/chat/completions") or self.path.startswith("/chat/completions")):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.3))
        max_tokens = min(int(body.get("max_tokens", 1536)), 8192)
        enable_thinking = bool(body.get("enable_thinking", False))

        prompt_ids = encode_messages_native(GLOBAL_TOKENIZER, messages, enable_thinking=enable_thinking)
        max_ctx = GLOBAL_RUNTIME.max_seq_len - max_tokens - 16
        if len(prompt_ids) > max_ctx:
            prompt_ids = prompt_ids[-max_ctx:]

        t0 = time.perf_counter()
        logits, new_toks, cached_toks = prefill_with_prefix_cache(prompt_ids)
        cur_pos = len(prompt_ids)
        generated = []
        created_ts = int(time.time())
        req_id = f"chatcmpl-holov8-{created_ts}"

        stop_ids = {GLOBAL_TOKENIZER.eos_token_id, GLOBAL_TOKENIZER.im_end_id, GLOBAL_TOKENIZER.im_start_id}

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors()
            self.end_headers()

            for _ in range(max_tokens):
                if cur_pos >= GLOBAL_RUNTIME.max_seq_len - 4:
                    break
                raw_l = logits[0].float().clone()
                if not enable_thinking:
                    raw_l[GLOBAL_TOKENIZER.think_end_id] = -float("inf")
                next_id = sample_next_token(raw_l, generated, temperature=temperature)
                if next_id in stop_ids:
                    break

                generated.append(next_id)
                CACHED_SEQ_IDS.append(next_id)
                piece = GLOBAL_TOKENIZER.decode([next_id])
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

                logits = GLOBAL_RUNTIME.step_with_graph(next_id, cur_pos)
                cur_pos += 1

            end_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(end_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
            self.wfile.flush()
            dt = max(time.perf_counter() - t0, 1e-4)
            print(f"[API Stream] Prefill: {new_toks} nuevos ({cached_toks} en caché) | Gen: {len(generated)} toks ({len(generated)/dt:.2f} tok/s)")
        else:
            for _ in range(max_tokens):
                if cur_pos >= GLOBAL_RUNTIME.max_seq_len - 4:
                    break
                raw_l = logits[0].float().clone()
                if not enable_thinking:
                    raw_l[GLOBAL_TOKENIZER.think_end_id] = -float("inf")
                next_id = sample_next_token(raw_l, generated, temperature=temperature)
                if next_id in stop_ids:
                    break
                generated.append(next_id)
                CACHED_SEQ_IDS.append(next_id)
                logits = GLOBAL_RUNTIME.step_with_graph(next_id, cur_pos)
                cur_pos += 1

            full_text = GLOBAL_TOKENIZER.decode(generated)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.end_headers()
            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated), "total_tokens": cur_pos},
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))


def run_interactive_cli():
    global CACHED_SEQ_IDS
    print("\n" + "=" * 78)
    print(f" HOLO-V8 INTERACTIVE CHAT ({MODEL_ID} | 16.384 CTX FP8 | PREFIX-CACHE ENABLED)")
    print(" Commands: '/think' (toggle <think>), '/temp 0.0' (greedy/sampling), '/clear', '/exit'")
    print("=" * 78)

    history = []
    enable_thinking = False
    temperature = 0.2
    stop_ids = {GLOBAL_TOKENIZER.eos_token_id, GLOBAL_TOKENIZER.im_end_id, GLOBAL_TOKENIZER.im_start_id}

    while True:
        try:
            user_in = input("\n[Tú] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExisting...")
            break
        if not user_in:
            continue
        if user_in.lower() in ("/exit", "exit", "quit"):
            break
        if user_in.lower() == "/clear":
            history = []
            CACHED_SEQ_IDS = []
            print("[*] Memory and history cleared.")
            continue
        one_shot_think = False
        if user_in.lower() == "/think":
            enable_thinking = not enable_thinking
            print(f"[*] Modo <think> permanente: {'ACTIVADO' if enable_thinking else 'DESACTIVADO'}")
            continue
        elif user_in.lower().startswith("/think "):
            one_shot_think = True
            user_in = user_in[7:].strip()
        if user_in.lower().startswith("/temp"):
            try:
                temperature = float(user_in.split()[1])
                print(f"[*] Temperature set to: {temperature}")
            except Exception:
                print("[!] Usage: /temp 0.0 (deterministic) o /temp 0.6")
            continue

        history.append({"role": "user", "content": user_in})
        use_think = enable_thinking or one_shot_think
        prompt_ids = encode_messages_native(GLOBAL_TOKENIZER, history, enable_thinking=use_think)

        t_pre = time.perf_counter()
        logits, new_toks, cached_toks = prefill_with_prefix_cache(prompt_ids)
        dt_pre = time.perf_counter() - t_pre
        cur_pos = len(prompt_ids)
        generated = []

        in_think = use_think
        print(f"\n[Holo-V8]{'\n<think>\n' if use_think else ' > '}", end="", flush=True)
        t_gen = time.perf_counter()
        for _ in range(1536):
            if cur_pos >= GLOBAL_RUNTIME.max_seq_len - 4:
                break
            raw_l = logits[0].float().clone()
            if not in_think:
                raw_l[GLOBAL_TOKENIZER.think_end_id] = -float("inf")
            next_id = sample_next_token(raw_l, generated, temperature=temperature)
            if next_id in stop_ids:
                break
            generated.append(next_id)
            CACHED_SEQ_IDS.append(next_id)
            if next_id == GLOBAL_TOKENIZER.think_end_id:
                in_think = False
                sys.stdout.write("\n</think>\n\n")
            else:
                sys.stdout.write(GLOBAL_TOKENIZER.decode([next_id]))
            sys.stdout.flush()
            logits = GLOBAL_RUNTIME.step_with_graph(next_id, cur_pos)
            cur_pos += 1

        torch.cuda.synchronize()
        dt = max(time.perf_counter() - t_gen, 1e-4)
        print(f"\n\n   [⚡ Prefill: {dt_pre:.2f}s ({new_toks} new, {cached_toks} cached) | Gen: {len(generated)} toks a {len(generated)/dt:.2f} tok/s | Ctx: {cur_pos}/16384]")
        history.append({"role": "assistant", "content": GLOBAL_TOKENIZER.decode(generated)})


def main():
    global GLOBAL_RUNTIME, GLOBAL_TOKENIZER
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Iniciar chat interactivo en consola")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ctx", type=int, default=16384, help="Tamaño de ventana (8192, 16384 o 32768)")
    args = parser.parse_args()

    GLOBAL_TOKENIZER = GGUFNativeTokenizer()

    engine = HoloCellV7Engine(device="cuda")
    GLOBAL_RUNTIME = Qwen38HoloRuntime(engine, max_seq_len=args.ctx)
    GLOBAL_RUNTIME.load_to_vram()
    GLOBAL_RUNTIME.capture_cuda_graph()

    if args.cli:
        run_interactive_cli()
    else:
        server = HTTPServer((args.host, args.port), OpenAIHandler)
        print("\n" + "=" * 78)
        print(f" [OK] HOLO-V8 OPENAI-COMPATIBLE SERVER LIVE AT: http://{args.host}:{args.port}/v1")
        print(f"      - Model ID : {MODEL_ID} ({args.ctx} tokens KV-Cache FP8 + Prefix-Cache)")
        print("=" * 78)
        server.serve_forever()


if __name__ == "__main__":
    main()
