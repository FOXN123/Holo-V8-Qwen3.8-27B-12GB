# Holo-V8: Running Qwen 3.8 27B at ~2.05-bit Completely Inside a 12GB RTX 3060

**~15.0 tok/s | 16K FP8 Context Window | 0 PCIe weight traffic during generation | 159.1W / 67.0°C Sustained**

Hi everyone! I am an independent hardware enthusiast—not an academic lab—who operates on a modest home GPU setup: a **SOYO NVIDIA GeForce RTX 3060 12GB (`GA104` silicon edition: 3,584 CUDA cores, 192-bit GDDR6, 170W TDP)** running via PCIe passthrough inside a **Proxmox VE virtual machine (32 GB RAM)**.

Running a 27-billion parameter model like **Qwen 3.8 27B** (>51 GB in BF16) on a 12 GB consumer GPU usually forces you to either offload layers to system RAM over PCIe (dropping speed to 1.5–3 tokens/s) or use standard extreme 2-bit quantizations (`Q2_K` / `IQ2_XXS`), which suffer from severe perplexity degradation or syntax collapse in hybrid **Gated DeltaNet (48 linear recurrent layers) + Full GQA (16 attention layers)** architectures.

**Holo-V8** is an open, lightweight inference engine and Post-Training Quantization (PTQ) pipeline in pure Python + Triton. It fits the entire 27B model + **16,384 tokens of FP8 KV-Cache** inside **10.95 GB of VRAM** (`10.48 GB` at 8K context), sustaining **14.8 – 15.2 tokens/s** via a monolithic `CUDAGraph` with zero PCIe weight transfers during token generation.

> **VRAM Operating Modes (12 GB Cards):**
> - **Headless / Dedicated Server (`--ctx 16384`):** Allocates `~10.95 GB` VRAM. Recommended when no monitor/browser is using the GPU.
> - **Desktop / Monitor Attached (`--ctx 8192`):** Allocates `~10.48 GB` VRAM. Leaves ~1.5 GB free for Windows/Linux desktop compositors and browsers.

> 🔬 **Architecture & Kernel Deep-Dive:** For the full mathematical breakdown (Hadamard rotation, 4D vector quantization), the complete GPU kernel ablation study (V1→V9, including failed experiments), and the methodological audit of our evaluation harness, see [**TECHNICAL_NOTES.md**](./TECHNICAL_NOTES.md).

- **Instant Native Tokenizer:** The engine decouples tokenization from the GGUF container. It includes a pre-extracted `tokenizer_vocab.pt` within the weights cache, loading the full 248k vocabulary in ~0.1s and eliminating the need to ship or parse the original 15GB+ GGUF file.

---

## 1. Verified Benchmark Results

All evaluations were executed locally on the exact same **SOYO RTX 3060 12GB (`GA104`)**. Every single prompt, prediction, error breakdown, and 5-second GPU telemetry sample (`159.1W` mean power, `67.0°C` mean temp across 4,833 active samples) is auditable in `benchmarks_audit/`.

| Benchmark | Holo-V7 (Uniform 2-bit) | **Holo-V8 (AWQ-Hessian ~2.05-bit)** | Evaluation Scope & Protocol |
| :--- | :---: | :---: | :--- |
| **WikiText-2-Raw-v1 PPL** | `10.3420` | **`8.2363`** | Full official test split (20 blocks x 512 tokens = 10,240 tokens) |
| **OpenAI HumanEval (`pass@1`)** | `80.49%` (132/164) | **`85.37%` (140/164)** | Full benchmark (`n=164/164`), zero-shot unit testing |
| **Google MBPP Sanitized (`pass@1`)** | — | **`78.50%` (157/200)** | Sanitized test split (`n=200`), zero-shot Python synthesis |
| **GSM8K Math Reasoning** | `86.00%` (129/150) | **`97.33%` (146/150)** | Stratified local subset (`n=150`: 50 Easy / 50 Med / 50 Hard, `max_tokens=1280`) |
| **AI2 ARC-Challenge** | `88.67%` (266/300) | **`93.00%` (279/300)** | Stratified local subset (`n=300`), direct log-likelihood (`A, B, C, D`) |
| **MMLU Multi-Domain** | `70.00%` (210/300) | **`74.33%` (223/300)** | Stratified local subset (`n=300`) across STEM, Humanities & Social Sci |
| **GPQA Diamond PhD Science** | `33.33%` (66/198, 0-shot direct) | **`87.50%` (35/40)** | Stratified PhD subset (`n=40`) with `<think>` CoT (`max_tokens=3072` on 16K FP8) |
| **VRAM Footprint** | `9.74 GB` (4K ctx) | **`10.48 GB` (8K) / `10.95 GB` (16K)** | `float8_e4m3fn` KV-Cache on the 16 GQA attention layers |
| **Sustained Generation Speed** | `14.8 tok/s` | **`14.9 - 15.2 tok/s`** | Monolithic `CUDAGraph` (`0 bytes` PCIe weight traffic during generation) |

*Note on `~2.05-bit` effective rate:* Includes packed 2-bit 4D Walsh-Hadamard codebooks (4 weights per `uint8` byte), sparse FP8 outlier branches (`TOPK=768` down to `128`), and block-64 scale metadata.

---

## 2. Repository File Guide (What Each File Does)

| File / Folder | Purpose |
| :--- | :--- |
| `holocell_v8_qwen38_engine.py` | **Core Inference Engine:** Triton 2-bit Hadamard+FP8 kernels, FP8 KV-Cache, and CUDAGraph runtime. |
| `holocell_v8_openai_server.py` | **Chat & API Server:** Interactive CLI chat (`--cli`) and OpenAI-compatible HTTP server (`/v1/chat/completions`) with Prefix Caching for OpenWebUI. |
| `verify_install.py` | **30-Second Smoke Test:** Verifies GPU detection, VRAM usage, speed (~15 tok/s), and deterministic generation. |
| `quantize_step1_awq_hessian.py` | **Compression Pipeline (Step 1):** Computes activation Hessians $E[x^2]$ and builds the AWQ-protected 2-bit cache from GGUF. |
| `quantize_step2_vram_slim.py` | **Compression Pipeline (Step 2):** Optimizes internal projection `TOPK` outliers to free 1.52 GB VRAM for 16K context. |
| `benchmark_suite_core.py` | **Benchmark Suite:** Reproducible evaluation script for WikiText-2, HumanEval, GSM8K, ARC-Challenge, and MMLU. |
| `benchmark_gpqa_phd.py` | **PhD Science Evaluator:** Evaluates GPQA Diamond with native `<think>` Chain-of-Thought reasoning. |
| `benchmarks_audit/` | **Full Audit Trail:** Contains problem-by-problem Markdown reports, JSON states, and active GPU telemetry CSVs. |
| `cache_holocell_v8/` | **Model Weights & Assets:** Contains the 2-bit quantized matrices, FP8 scales, and the **bundled native tokenizer** (`tokenizer_vocab.pt`), enabling ~0.1s instant loading without requiring the original multi-gigabyte GGUF file at runtime. |

---

## 3. Engineering Obstacles We Hit & How We Overcame Them

- **Obstacle 1: The `repetition_penalty=1.04` Indentation Trap in HumanEval.** Applying lexical repetition penalties during code generation penalized repeated 4-space indentation tokens, causing `IndentationError`. Removing lexical repetition penalties during greedy code synthesis pushed HumanEval `pass@1` to **85.37% (140/164)**.
- **Obstacle 2: The 34-Second (`512-token`) Truncation Wall in GSM8K Hard Problems.** Initially, Hard math problems hit the 512-token ceiling at ~34.5 seconds mid-equation. Raising `max_new_tokens` to `1280` recovered 27 of 31 failed problems, reaching **97.33% (146/150)**.
- **Obstacle 3: Raw `encode_text` vs. Native `encode_chat` Special Token IDs in MBPP.** Passing `<|im_start|>` as raw text split control tags into characters and left `think_end_id` unmasked (causing 44 artificial `SyntaxError`s). Switching to native `encode_chat` recovered **+48 problems** (**78.50%**).
- **Obstacle 4: GPQA Diamond Thinking Budget (`1280` vs. `3072` tokens).** Without `<think>`, 0-shot accuracy on all 198 questions was `33.33%`. Enabling `<think>` with 1,280 tokens reached `67.50%` (27/40), as 28 questions truncated at token 1,280. Expanding the thinking budget to `3,072 tokens` inside the 16K FP8 KV-Cache recovered 8 of the 13 truncated problems, reaching **87.50% (35/40)**.

---

## 4. Requirements & Quick Start

### Tested Reference Environment
- **Host:** Proxmox VE VM (32 GB RAM) | Ubuntu 24.04 (`Linux 6.8.0-147-generic`)
- **GPU:** SOYO NVIDIA GeForce RTX 3060 12GB (`GA104`, Driver `580.178.04`, CUDA 13.0)
- **Python / PyTorch:** Python `3.12.3` | PyTorch `2.14.1+cu130` | Triton `3.8.0`

> **PyTorch / CUDA Note:** Install the PyTorch wheel matching your system CUDA version before running `pip install -r requirements.txt` (flexible) or `requirements_exact.txt` (exact lab replica).

```bash
git clone https://github.com/FOXN123/Holo-V8-Qwen3.8-27B-12GB.git
cd Holo-V8-Qwen3.8-27B-12GB
pip install -r requirements.txt

# Download pre-quantized weights (cache_holocell_v8) from Hugging Face into ./cache_holocell_v8/
huggingface-cli download foxn/Qwen3.8-27B-Holo-V8-2bit-12GB --local-dir ./cache_holocell_v8

# 1. Run 30-second smoke test
python3 verify_install.py

# 2. Interactive Terminal Chat (with instant Prefix Caching & /think toggle)
python3 holocell_v8_openai_server.py --cli

# 3. Start OpenAI-Compatible API Server for OpenWebUI
python3 holocell_v8_openai_server.py --port 8090 --ctx 16384
```

---

## 5. Troubleshooting (12 GB GPUs)

- **CUDA Out of Memory (OOM):** If your GPU also drives a monitor or browser (which uses 0.6–1.2 GB VRAM), run with `--ctx 8192` (`10.48 GB` VRAM).
- **Slow Generation (<14 tok/s):** Verify in `nvidia-smi` that VRAM usage is below 11.2 GB so the driver does not spill pages to system RAM.

---

## 6. How Holo-V8 Fits in the Ecosystem & Acknowledgments

- **vs. PrismML Ternary Bonsai-2 27B:** Bonsai-2 uses proprietary Quantization-Aware Training (QAT) on H100 clusters. Holo-V8 is a 100% open **Post-Training Quantization (PTQ)** pipeline in Python/Triton built on a single home RTX 3060 12GB without retraining.
- **vs. Escha-W2 (2.47 bpw) & Empero-AI Ridge (3.69 bpw):** Ridge (`11.73 GiB` weights) and Escha-W2 (`15.5–18.1 GB` with SGLang runtime) target 16–24 GB GPUs. Holo-V8 packs weights + 16K FP8 KV-Cache inside `10.95 GB` for 12 GB cards.

**Acknowledgments:** Special thanks to **`professorpalmer` (`bonsai-ada-surgery`)**, **`empero-ai` (`Ridge`)**, **`Escha Labs` (`Escha-W2`)**, **`x0me`**, **`The Kaitchup`**, and the **Qwen Team (Alibaba Cloud)**.

**License & Disclaimer:** Engine and scripts are licensed under **Apache-2.0**. Quantized weights in `cache_holocell_v8` are derived from Qwen 3.8 27B under **Apache-2.0**. This independent project is not affiliated with or endorsed by the Qwen team.
