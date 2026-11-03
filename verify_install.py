import time
import torch
from gguf import GGUFReader
from holocell_v8_qwen38_engine import (
    GGUF_PATH,
    GGUFNativeTokenizer,
    HoloCellV7Engine,
    Qwen38HoloRuntime,
)

@torch.no_grad()
def main():
    print("=" * 72)
    print(" HOLO-V8 INSTALLATION & HARDWARE SMOKE TEST")
    print("=" * 72)
    assert torch.cuda.is_available(), "CUDA GPU not detected!"
    dev_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[OK] GPU Detected : {dev_name} ({total_vram:.2f} GB VRAM)")

    tokenizer = GGUFNativeTokenizer()

    engine = HoloCellV7Engine(device="cuda")
    runtime = Qwen38HoloRuntime(engine, max_seq_len=8192)
    runtime.load_to_vram()
    runtime.capture_cuda_graph()

    prompt = "Give a one-sentence definition of quantum entanglement."
    p_ids = tokenizer.encode_chat(prompt, enable_thinking=False)
    runtime.init_states()
    logits = runtime.forward_prefill(p_ids)
    cur_pos = len(p_ids)
    gen = []

    # Warmup de 2 pasos para estabilizar relojes de la GPU tras compilar
    for _ in range(2):
        _ = runtime.step_with_graph(248045, cur_pos)
    torch.cuda.synchronize()
    logits = runtime.forward_prefill(p_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(48):
        raw_l = logits[0].float()
        raw_l[tokenizer.think_start_id] = -float("inf")
        raw_l[tokenizer.think_end_id] = -float("inf")
        nxt = int(torch.argmax(raw_l).item())
        if nxt in (tokenizer.eos_token_id, tokenizer.im_end_id):
            break
        gen.append(nxt)
        logits = runtime.step_with_graph(nxt, cur_pos)
        cur_pos += 1
    torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-4)
    tps = len(gen) / dt
    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)

    print(f"\n[Prompt]   : {prompt}")
    print(f"[Response] : {tokenizer.decode(gen).strip()}")
    print("-" * 72)
    print(f"[PASS] Speed: {tps:.2f} tok/s | VRAM Allocated: {vram_used:.2f} GB / {total_vram:.2f} GB")
    print("=" * 72)

if __name__ == "__main__":
    main()
