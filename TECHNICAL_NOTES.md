# Holo-V8 Technical Notes: Kernel Ablation & Methodological Audit

> **Note:** This is a technical deep-dive document for researchers and kernel developers.
> For quick-start instructions and benchmark results, see the main [README.md](./README.md).
> The final production engine is **Holo-V8**. This document explains how we got here,
> including the kernel experiments that failed (V1–V6 early iterations, and the V7/V8-early/V9
> regressions that were reverted in the final V8 design).

## 1. Architecture Deep-Dive

Running a 27-billion parameter model (50.9 GB in BF16) within a 12 GB consumer GPU VRAM without resorting to PCIe paging (*CPU offloading*) requires compressing linear weights below 2.1 effective bits per parameter while maintaining decoding speed above the human reading threshold (>10 tok/s).

Existing `llama.cpp` (GGUF) solutions present an insurmountable dichotomy on 12 GB GPUs:
1. **Mixed quantizations like *Empero Ridge* (~3.69 bpw, `11.73 GiB`):** Maintain high quality but occupy almost 100% of the 12 GB in static weights alone. Upon initializing the KV-Cache and recurrent states, they exceed 12 GB (~14+ GB total), forcing PCIe bus transfers and dropping performance.
2. **Extreme uniform quantizations (`IQ2_XXS` / `IQ2_M`, ~8.4–9.8 GB):** Fit in VRAM but lack explicit Hessian-guided row-wise *outlier* extraction combined with dynamic orthogonal rotation in the kernel, suffering severe degradation in code generation and mathematical reasoning.

**Holo-V8** resolves this bottleneck via a three-component hybrid representation executed entirely on the fly in the GPU's L1 cache using **Triton + CUDA Graphs**:
- **Hessian-Guided AWQ Sparse Outlier Branch (`FP8 E4M3`):** Using calibration activations (`wiki.train.raw`), we compute the input activation second moments $\mathbb{E}[x_j^2]$ and isolate the most salient channels per row (`TOPK = 768` in `ffn_down`, `512` in `ssm_out` and `GQA`, and `128–256` in internal projections) in `FP8 (E4M3)` with `int16` column indices and `FP16` row scales (`out_rscale`).
- **Block-wise Walsh-Hadamard Incoherence Projection ($\mathcal{H}_{64}$, QuIP#-style):** Applied to the residual weight body after outlier extraction. Leveraging symmetry and orthogonality ($\mathcal{H}_{64}^T = \mathcal{H}_{64}$), the rotation is not applied to the 400 weight matrices during inference ($\mathcal{O}(NK)$), but rather exactly once on the incoming activation vector $\tilde{x} = (I_{K/64} \otimes \mathcal{H}_{64})(x \odot s_{\text{col}})$ in $\mathcal{O}(K)$.
- **4D Vector Quantization with 32-bit Register Codebook (2.0-bit Core):** Grouping 4 rotated weights into a single 8-bit index (`2.0 bits/weight`, packing 4 weights per `uint8` byte) referencing a matrix-specific 4D vector codebook $C^{(l)} \in \mathbb{R}^{256 \times 4}$, preloaded in VRAM as packed `int32` words (`cb_i32`) to decode 4 weights with a single 32-bit `LDG.E` instruction in L1 cache.

## 2. GPU Kernel Ablation Study (V1 → V9)

During engine development on the SOYO RTX 3060 12GB (Ampere `GA104`), we documented 9 architectural kernel iterations. Three of them resulted in **severe performance regressions** that provide valuable empirical findings on Ampere microarchitecture:

### 2.1 The Fusion and CUDA Graph Capture Milestone (V1 $\rightarrow$ V6: From 1.90 tok/s to 14.27 tok/s)
- **V1–V2 Failure (1.90 – 3.16 tok/s):** Keeping the `output.weight` head ($248,320 \times 5120$) in system RAM forced transferring the hidden vector over PCIe and CPU-GPU synchronization per token.
- **V3 Solution (4.77 tok/s):** Block-wise `INT4` quantization (group 64) of `output.weight` resident in VRAM (+480 MB). PCIe weight traffic dropped to 0 bytes, but the engine remained limited by Python overhead launching ~1,000 CUDA kernels per token.
- **V4 Solution (10.67 tok/s):** Rewriting the **Gated DeltaNet** (48 layers) and **GQA** (16 layers) recurrence in pure Triton kernels with static position pointers (`static_pos`), capturing all 64 layers in a single `torch.cuda.CUDAGraph`.
- **V5–V6 Solution (14.27 tok/s):** Design of the **4-Row Block** GEMV kernel (`BLOCK_N = 4`) with codebook packed in `int32` (`cb_i32`). By processing 4 rows per thread block, the rotated vector $\tilde{x}$ is reused in registers, reducing activation read traffic by 75%.

### 2.2 Failed Kernel Experiments (V7-Transpose, V8-int64, and V9-SwiGLU-Fusion)
1. **FP16 Codebook Transposition Failure (Drop from 14.27 to 10.02 tok/s, -29.8%):**
   * *Hypothesis:* Transposing the codebook to `[4, 256]` in `FP16` would avoid bit-shifting operations (`>> 8`, `& 0xFF`).
   * *Why it failed:* Emitting 4 scattered `LDG.E.U16` load instructions per 4D tuple saturated the Load/Store Unit (LSU) queues of the RTX 3060. On Ampere, it is significantly faster to emit **a single 32-bit load (`cb_i32`)** and unpack in ALU registers.
2. **`int64` Packing Failure (Drop to 12.47 tok/s + VRAM spike):**
   * *Hypothesis:* Packing the 4 FP16 centroid values into a single `int64` integer (one 64-bit read per 4 weights).
   * *Why it failed:* The Ampere architecture executes 64-bit operations at a lower rate on integer units, and the PTX compiler generated costly `PRMT` instruction sequences to extract the 4 `float16` values.
3. **Gate+Up SwiGLU Kernel Fusion Failure (Collapse to 8.84 tok/s, -38.0%):**
   * *Hypothesis:* Fusing `ffn_gate` and `ffn_up` ($17,408 \times 5120$ each) into a single kernel to read the input vector $\tilde{x}$ only once.
   * *Why it failed (*Register Spilling*):* Simultaneously holding accumulators for 4 rows of `gate`, 4 rows of `up`, two distinct codebooks, and double the outlier pointers in registers exceeded the per-thread register limit of the RTX 3060, causing *register spilling* to local memory and crashing SM *Occupancy*.
4. **Definitive Production Architecture in Holo-V8 (14.9–15.2 tok/s):**
   * Reversion to the clean 32-bit register structure (`cb_i32`) combined with `tl.static_range` loop unrolling over the AWQ-Hessian outlier tables (`10.48 GB` VRAM at 8K context; `10.95 GB` at 16K context), executing *prefill* via iterative calls to `cuda_graph.replay()` with Prefix Caching.

## 3. Methodological Audit

Following a strict protocol of scientific transparency, we document the methodological biases detected during preliminary testing and how they were corrected for the definitive evaluation:

### 3.1 Correction of Perplexity Bias in WikiText-2 (From a false `5.30` to the 20-block / 10,240-token protocol)
- **Methodological Error #1 (Preliminary Iteration - PPL `5.3016`):** In the first test, only the first 1,024 tokens of `test.txt` were evaluated after removing empty lines (`if line.strip()`). Since the first 1,024 tokens belong entirely to a single repetitive encyclopedic article (*Valkyria Chronicles III*) and deleting line breaks suppressed paragraph-change uncertainty, the measured perplexity (`5.30`) was artificially lower than even the BF16 baseline (`~7.15`).
- **Methodological Error #2 (Intermediate Iteration - PPL `9.7136`):** When expanding to 10,240 tokens (20 blocks of 512), a string replacement (`.replace(" @-@ ", "-")`) was applied to a pre-tokenized file containing literal `<unk>` tokens, inflating the exponential mean.
- **Definitive Protocol:** Evaluation over 10,240 tokens of raw `wiki.test.raw` (20 independent blocks of 512 tokens, measuring NLL on the second half `[256:512]` of each window after 256 tokens of context), fully respecting blank lines and original tokenization. Holo-V7 scored **`10.3420`**, and **Holo-V8** lowered it to **`8.2363`** (**-2.1057 PPL**).

### 3.2 Correction of Harness Pitfalls in OpenAI HumanEval (164 Problems) and MBPP (200 Problems)
- **What failed in the preliminary pass:**
  1. **Truncation at `220 tokens`:** Medium-to-high complexity functions require 230–400 tokens. Cutting generation at 220 tokens caused artificial `SyntaxError: unexpected EOF`.
  2. **Repetition penalty (`1.04`) in code:** Applying `repetition_penalty` during code synthesis penalized legitimate repeated 4-space indentation tokens, degrading logic in nested loops.
  3. **Raw `encode_text` vs. Native `encode_chat`:** Passing `<|im_start|>` as raw text split control tags into literal characters and left `think_end_id` unmasked in MBPP.
- **Clean and Uniform Correction:** Purely Greedy decoding (`Temperature = 0.0`, **no `repetition_penalty`**), native `encode_chat` special token IDs, a window of **`512 tokens`**, uniform indentation normalization via `textwrap.dedent` + `textwrap.indent`, and execution in a 3-second POSIX sandbox (`signal.SIGALRM`). Result: **85.37% (140/164)** on HumanEval and **78.50% (157/200)** on MBPP Sanitized.

### 3.3 Logit-based Standardization in ARC-Challenge (300) and MMLU (300), and Untruncated Budgets in GSM8K (150) & GPQA (40)
- **ARC-Challenge and MMLU without parsing bias:** Following the standard `lm-evaluation-harness` protocol, after prefilling the prompt and the 4 options via `encode_chat`, the exact logit of the first response token is compared among $\arg\max_{c \in \{A, B, C, D\}} \text{logit}(c)$, achieving **93.00% (279/300)** on ARC-Challenge and **74.33% (223/300)** on MMLU.
- **Stratified GSM8K (150 problems):** 50 Easy, 50 Medium, and 50 Hard problems extracted with a fixed seed (`seed=42`). Raising the generation budget from `512` to **`1,280 tokens`** eliminated the 34-second truncation wall on Hard problems, recovering 27 problems and reaching **97.33% (146/150)**.
- **Stratified GPQA Diamond PhD (40 problems):** Raising the native `<think>` Chain-of-Thought budget from `1,280` to **`3,072 tokens`** inside the 16K Context window recovered 8 of 13 previously truncated PhD derivations, reaching **87.50% (35/40)**.
