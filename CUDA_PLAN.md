\## 0) Ground rules from the existing Voxtral pipelines



Use these as “non-negotiable” design constraints for CUDA:



\* \*\*Keep intermediates on-GPU across whole passes\*\* (Metal keeps residual stream and activations resident and runs encoder/decoder as a single monolithic submission). On CUDA, the analogue is \*\*CUDA Graphs + persistent memory arenas\*\*. 

\* \*\*Specialize attention for decode vs prefill\*\*: Voxtral already has a \*\*single-token decoder attention\*\* kernel (online softmax) and a \*\*tiled prefill/encoder attention\*\* kernel. Your CUDA design should preserve this split. 

\* \*\*Exploit BF16 end-to-end\*\*: CPU AVX-512 path is BF16 weights + specialized dot product; on NVIDIA you want BF16 tensor cores and BF16-friendly layouts. 

\* \*\*Model specifics that must be supported\*\*: conv stem + 32 encoder layers, adapter downsample/projection, decoder with AdaNorm-style per-layer modulation and audio-driven decode loop. 



---



\## 1) Target architecture choices (so kernels match the hardware)



\*\*Minimum target\*\*: Ampere (SM80) for robust BF16 tensor core throughput.

\*\*Optimized paths\*\*:



\* \*\*Hopper/Blackwell\*\*: use warp-group GEMM (WGMMA/GMMA via CUTLASS/CuTe) where it matters (QKV/FFN/output projection). CUTLASS explicitly targets Ampere/Hopper/Blackwell. (\[GitHub]\[1])

\* \*\*General matmul\*\*: use \*\*cuBLASLt\*\* first for correctness + speed; migrate hot spots to CUTLASS when you need tighter fusion/control. cuBLAS has broad tensor core support without many of the old alignment restrictions. (\[NVIDIA Docs]\[2])



Deliverable: a backend that selects kernels by \*\*SM version\*\* at runtime.



---



\## 2) Memory \& layout plan (this is where performance is won)



\### 2.1 Device memory arenas (no per-token allocations)



\* Allocate:



&nbsp; 1. \*\*Weights arena\*\* (BF16, ideally fully resident if VRAM allows).

&nbsp; 2. \*\*KV cache arena\*\* (BF16 or FP16; BF16 preferred if bandwidth is dominant and accuracy holds).

&nbsp; 3. \*\*Activation/workspace arena\*\* (reused buffers per layer, double-buffered when beneficial).

\* Use `cudaMallocAsync` + a dedicated memory pool (dramatically reduces allocator overhead in long decode loops).



\### 2.2 Weight format



\* Unlike Metal (which caches BF16→F16 because MPS wants F16/F32), CUDA can \*\*consume BF16 directly on tensor cores\*\*, so keep weights BF16 end-to-end. 

\* Store weights on disk in a layout that matches your chosen GEMM kernels (often \*\*col-majoprojection weights, or the layout CUTLASS expects).



\### 2.3 KV cache layout



\* Use a layout optimized for your attention kernels:



&nbsp; \* A common performant choice: `\[layer]\[kv\_head]\[seq]\[head\_dim]` contiguous in `head\_dim`, with `seq` stride aligned for vector loads.

\* Because Voxtral uses \*\*sliding window attention\*\* (e.g., 750 frames for encoder), store KV in a \*\*ring buffer\*\* per layer so that “windowed” reads stay contiguous. 



---



\## 3) Direct I/O plan (GPUDirect Storage) for fast startup and/or weight streaming



\##- \*\*Best fit\*\*:



\* very fast startup (load 8–12GB weights without saturating CPU copies),

\* or \*\*layer streaming\*\* if weights don’t fit in VRAM (smaller GPUs).

\* Implement an optional \*\*GPUDirect Storage (GDS)\*\* path using \*\*cuFile\*\* so NVMe DMA can move bytes \*\*directly into GPU memory\*\* (no CPU bounce buffer). (\[NVIDIA Docs]\[3])



\### 3.2 Streaming design (only if needed)



\* Partition weights per layer (or per block of layers).

\* Double-buffer 2–3 “weight slabs” in GPU memory:



&nbsp; \* while layer \*i\* runs, issue `cuFileRead` for layer \*i+1\* into the next slab on a separate stream,

&nbsp; \* synchronize with CUDA events before the GEMMs consume that slab.



---



\## 4) Kernel inventory (what you actually implement)



Start with library GEMMs + custom elementwise kernels, then progressively fuse.



\### 4.1 Convolutional stem (encoder front-end)



\* Use cuDNN for baseline.

\* If stem becomes a hotspot: write a \*\*fused causal conv + activation\*\* kernel (stride 1 then stride 2), keeping it BF16/FP16 internal where possible. 



\### 4.2 RMSNorm + AdaNorm fusion



\* One fused kernel that does:



&nbsp; 1. RMS reduction (FP32 accumulate),

&nbsp; 2. scale by norm weights,

&nbsp; 3. apply AdaNorm modulation `\*(1 + adaoder feature).

\* Use warp-level reductions + vectorized BF16 loads (`nv\_bfloat162`) and write BF16/FP16 outputs as needed.



\### 4.3 RoPE



\* A simple, bandwidth-bound kernel:



&nbsp; \* vectorize over head\_dim (use `bfloat2`), prefetch sin/cos from constant/texture memory,

&nbsp; \* apply in-place on Q and K.



\### 4.4 Attention (two distinct kernels, like Metal)



\*\*A) Prefill/encoder attention (seq\_q > 1)\*\*



\* Implement a \*\*FlashAttention-style fused kernel\*\* with:



&nbsp; \* tiled Q blocks, streaming K/V blocks,

&nbsp; \* online softmax (numerically stable),

&nbsp; \* causal + sliding window mask integrated.

\* FlashAttention-style kernel fusion is a proven approach for Hopper-class GPUs (and CUTLASS is commonly used to build it). (\[arXiv]\[4])

\* Match Metal’s approach: it tiles multiple queries per threadgroup (CUDA: per CTA). 



\*\*B) Decode attention (seq\_q = 1)\*\*



\* Dedicated kernel for single-token:



&nbsp; \* load one Q per head,

&nbsp; \* dot against windowed K cache,

&nbsp; \* online softmax,

&nbsp; \* accumulate V.

\* This mirrors the Metal “decoder\_atne softmax, reduction efficiency). 

\* Keep Q in registers, stage K/V in shared memory (cp.async on Ampere; TMA on Hopper if you go that far).



\### 4.5 FFN (SwiGLU) fusion



\* Keep the “merged W1/W3 matmul then fused SiLU\*gate” pattern:



&nbsp; \* GEM]`,

&nbsp; \* fused epilogue applies `silu(a) \* b`.

\* Implement first with a standalone kernel; then move to \*\*CUTLASS epilogue fusion\*\* once stable.



\### 4.6 Logits + argmax on GPU



\* Do output projection with GEMM/GEMV (M=1 in decode).

\* Run a warp/CTA reduction argmax kernel and copy back \*\*just one int token id\*\* (Metal does the same idea). 

\* If you want to go further: keep the decode loop entirely on GPU and only copy tokens back in chunks.



---



\## 5) “Monolithic” execution on CUDA (equivalent to Metal’s single command buffer)



Metal reducesy collapsing many ops into one command buffer. 

CUDA equivalent plan:



1\. \*\*Capture encoder full pass in a CUDA Graph\*\* (or per-layer subgraphs).

2\. \*\*Capture decoder layer stack as a CUDA Graph\*\* for a single token step.

3\. For iterative decoding:

&nbsp;  -(still much lower overhead than many kernel launches),



&nbsp;  \* update pointers/scalars via `cudaGraphExecUpdate` where possible,

&nbsp;  \* optionally use device-side graph launch capabilities where it fits your control flow. (\[NVIDIA Developer]\[5])



This is usually the biggest “easy win” after switching GEMMs to tensor cores.



---



\## 6) Precision strategy (BF16 without surprises)



\* \*\*Weights\*\*: BF16.

\* \*\*Activations\*\*: BF16 where safe; keep \*\*FP32 accumulation\*\* for:



&nbsp; \* RMSNorm stats,

&nbsp; \* attention softmax sums/max.

\* Add a “numerical safety” mode (FP16 activations + FP32 critical paths) to debug regressions.



This matches the spirit of the AVX design: BF16 for bandwidth + speed, FP32 where needed. 



---



\## 7) Step-by-step build plan (implementation order)



1\. \*\*Baseline correctness\*\*



&nbsp;  \* cuBLASLt BF16 GEMMs for all linears

&nbsp;  \* simple CUDA kernels for RMSNorm, RoPE, residual adds, SwiGLU, argmax

&nbsp;  \* KV cache in global memory

&nbsp;  \* validate vs Python reference outputs layer-by-layer. 



2\. \*\*Perfocks)\*\*



&nbsp;  \* Replace prefill attention with FlashAttention-style fused kernel

&nbsp;  \* Add dedicated seq\_q=1 decode attention kernel

&nbsp;  \* Introduce CUDA Graph capture for encoder and per-token decoder step (\[NVIDIA Developer]\[5])



3\. \*\*Performance pass 2 (fusion + memory)\*\*



&nbsp;  \* Fuse Ring

&nbsp;  \* Fuse FFN epilogue

&nbsp;  \* Ring-buffer KV cache for sliding window

&nbsp;  \* Add memory pool + eliminate runtime allocations



4\. \*\*I/O optimization (optional but “modern CUDA”)\*\*



&nbsp;  \* Add GDS/cuFile weight load path (and optional streaming) (\[NVIDIA Docs]\[3])



5\. \*\*Hardware-specialized fast paths\*\*



&nbsp;  \* CUTLASS kernels for the hottest GEMMs (QKV, FFN, output)

&nbsp;  \* Hopper+ path using advanced pipelines (via CUTLASS/CuTe abstractions) (\[GitHub]\[1])





\[1]: https://github.com/NVIDIA/cutlass?utm\_source=chatgpt.com "GitHub - NVIDIA/cutlass: CUDA Templates and Python DSLs for High ..."

\[2]: https://docs.nvidia.com/cuda/cublas/?utm\_source=chatgpt.com "1. Introduction — cuBLAS 13.1 documentation"

\[3]: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html?utm\_source=chatgpt.com "1. Overview Guide — GPUDirect Storage Overview Guide"

\[4]: https://arxiv.org/pdf/2312.11918?utm\_source=chatgpt.com "A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on ..."

\[5]: https://developer.nvidia.com/blog/cuda-toolkit-12-0-released-for-general-availability/?utm\_source=chatgpt.com "CUDA Toolkit 12.0 Released for General Availability"



Below is a concrete implementation plan for a **CUDA backend tuned for an RTX 5080 Laptop (Blackwell, CC 12.0)**, mirroring the design intent of the existing Voxtral pipelines while taking advantage of **modern CUDA (SM120), BF16 tensor cores, CUDA Graphs, async copies/TMA, and (where possible) direct I/O**.

---

## 0) Target reality check (so we optimize the right things)

* **GPU target:** RTX 5080 (laptop) is **CUDA Compute Capability 12.0**. ([NVIDIA Developer][1])
* **Laptop constraints:** GB203, **16 GB GDDR7**, **256-bit**, **~896 GB/s** bandwidth, **up to 150 W TGP** (often thermally constrained). ([Notebookcheck][2])
* **Toolchain:** CUDA 12.9 already lists **compute_120 / compute_121** as supported NVCC targets (so you can compile native SM120 cubins instead of relying on PTX JIT). ([NVIDIA Developer][3])

---

## 1) Match Voxtral’s “monolithic” execution style on CUDA

The Metal backend’s biggest win is reducing CPU↔GPU overhead by running whole passes “monolithically” (e.g., decoder command buffers from ~53/token → 1) and keeping intermediates on GPU. 

**CUDA analogue (core design choice):**

* Build two **CUDA Graphs**:

  1. **Encoder full-step graph** (all **32 encoder layers**) 
  2. **Decoder full-step graph** (all **26 decoder layers + logits
* Replay graphs per chunk/token to amortize launch overhead and reduce CPU sync—same motivation as Metal’s 

---

## 2) Compile specifically for SM120 “a/f” feature sets ()

CUDA 13.1 programming guide spells out that to use the **full architecture-specific feature set**, you should compile with the **`a` s. ([NVIDIA Docs][4])
Also, `compute_120f` is the family target compatible with **12.0 and 12.1**. ([NVIDIA Docs][4])

**Plan:**

* Ship **SM120A** kernels for maximum performance on the 5080 laptop.
* Optionally also ship **SM120F** kernels if you want portability across 12.0/12.1 devices without losing all family-specific features. ([NVIDIA Docs][4])

Why this matters: CC 12.x is where CUDA indicates support for **BF16 ops, hardware `memcpy_async`, async barriers, L2 cache control, thread-block clusters, and TMA**. ([NVIDIA Docs][4])

---

## 3) Dataflow & layouts: design around Voxtral’s actual shapes and attention windows

From the Python reference:

* Encoder: **32 layers**, dim **1280**, **sliding window 750** frames, RMSNorm + SwiGLU. 
* Adapter: concat every 4 frames then 2-layer MLP to **3072**. 
* Decoder: AdaNorm conditioning applies `h = h * (1 + ada_scale)` per layer. 
* Decoder generation is “audio-driven”: token embedding is added to the “next available” audio embedding as you iterate. 

From the Metal backend:

* Specialized attention kernels: **single-token decoder attention** uses **online softmax**; encoder/prefill attention uses **query-tiling (8 queries per threadgroup)**; window masking handled inside kernels. he GPU memory layouts accordingly:**
* **Weights:** store on GPU i16→FP16 cache needed on NVIDIA; Metal did it because MPS requires FP16/FP32). 
* **KV cache:** use a **ring-buffer (windowed)** layout for encoder window=750 to avoid ever reading “dead” history. *Decoder KV:** support long window sizes (Metal mentions e.g. 8192) and keep a paged/ring design so decode stays bandwidth-predictable. 
* **Tied embeddings:** decoder output projection uses transpose of inix in a layout optimized for your chosen GEMM path. 

---

## 4) Kernel strategy: “library first”, then replace the true hotspots with SM120-specialized k (dominant cost)

**Baseline:**

* Use **cuBLASLt** for all GEMMs in BF16 (Tensor Cores), both encoder and decoder.

**Dec=1 matvecs are often bandwidth-bound and cuBLASLt isn’t always optimal.

* Implement **custom SM120 matvec/GEMM-skinny kernels** with:

  * BF16torized `__nv_bfloat162`)
  * pipelined global→shared staging using **hardware `memcpy_async` / async barriers** (available in 12.x). ([NVIDIA Docs][4])  staging for large weight tiles. ([NVIDIA Docs][4])

**FFN “W1+W3 merged” path:**
Metal explicitly merges W1/W3 and runs `silu_mul_merged` to do **SwiGLU in one pass**. 
On CUDA:

* Prefer **CUTLASS epilogue fusion**: GEMM produces two outputs (or a packed output), epilogue computes `silu(x1) * x2` in-register/shared, writing only the post-SwiGLU tensor.
* Then run W2 GEMM.

### 4.2 Attention kernels (2 distinct kernels, mirroring Metal)

Metal uses:

* `decoder_attention` (seq_q=1) with **online softmax**. 
* `encoder_attention` with **query tiling (8 queries/threadgroup)**. 

**CUDA plan:**

1. **Prefill/encoder FlashAttention-style kernel**

   * Tile Q in blocks (8–16 queries/block to mirror Metal’s amortization idea). 
   * Use tensor core MMA for QKᵀ where it wins; otherwise use vectorized dot products.
   * Apply causal + window masks inside the kernel (
   * Use **async global→shared staging** + multi-stage pipeline (`memcpy_async`/barriers or TMA) to hide memory latency. ([NVIDIA Docs][4])

2. **Decode (seq_q=1) kernel**

   * Online softmax in a single pass (track running max + sum), then weighted sum over V—exactly the rationale in Metal. 
     r warp-group) per head” depending on head dim; use warp-level reductionsion
     Voxtral’s AdaNorm is literally scaling RMSNorm output by `(1 + ada_scale)` each decoder layer. 
     Metal already has aada_scale_mul`). 

**CUDA plan:** fuse into one kernel per layer:

* compute RMS (FP32 accumulation), normalize, multiply weiada_scale)` and (optionally) fuse residual add.

---

## 5) “Best possible” on RTX 5080 laptop: consider narrow precision for bandwidth + speed

If you can tolerate quantization:

* CUTLASS explicitly notes RTX 5000 series (SM120) adds new narrow precision tensor cores (4/6-bit block-scaled and non-block-scaled), with extended PTX MMA and “1× to 4× faster than Ada FP8 tensor cores.” ([NVIDIA Docs][5]) an optional tier:**
* Tier A (safe): BF16 weights + BF16 activations everywhere.
* Tier B: FP8 or block-scaled formats for the **largest linear layers** (QKV, FFN) with BF16 accumulation/output.
* Tier C: FP4/narrow formats for thef WER impact is acceptable.

---

## 6) Direct I/O (what’s realisticect Storage (GDS) can provide a direct DMA path between storage and GPU memory and can be enqueued asynchronously via CUDA streams/graphs. ([NVIDIA Docs][6])

However, NVIDIA’s design guide also states GDS requires **“GPUDirect capable NVIDIA GPUs (Quadro or Data Center GPUs only)”**. ([NVIDIA Docs][6])

**Pragmatic plan for RTX 5080 Laptop:**

* Implement an abstraction:

  1. **If GDS available**: use cuFile stream/graph APIs to stream weights or large audio blocks directly into device buffers. ([NVIDIA Docs][6])
  2. **Else (likely on GeForce)**: use `O_DIRECT` / async reads → **pinned host buffers** → `cudaMemcpyAsync` double-buffering.
* In practice, for Voxtral you’ll get most benefit by:

  * loading weights once at startup into VRAM (16 GB budget ([Notebookcheck][2])),
  * then keeping runtime I/O minimal.

---

## 7) Execution & scheduling details that usually decide “top 5%” performance

1. **CUDA Graphs everywhere**

   * Graph per encoder chunk and per decoder token step (monolithic equivalent). 

2. **Stream-ordered memory allocator**

   * Use `cudaMallocAsync` pools; never allocate/free on the critical path.

3. **Aggressive fusion to reduce bandwidth**

   * RMSNorm(+AdaScale)+residual in one kernel.
   * QKV projection fused (single GEMM producing Q,K,V).
   * FFN W1/W3 + SwiGLU fused like Metal’s `silu_mul_merged`. 

4. **SM120-specific async staging**

   * Use CC 12.x features: `memcpy_async`, async barriers, TMA (where it helps), and L2 cache control for hot small tables (RoPE sin/cos, small MLP weights). ([NVIDIA Docs][4])

5. **Auto-tune on the actual laptop**

   * Laptop thermals change the optimal kernel (cooperative vs pingpong schedule). CUTLASS notes SM120 supports both schedules. ([NVIDIA Docs][5])

---

## 8) Deliverables / roadmap (in order)

1. **Week 1–2: Baseline backend**

   * cuBLASLt BF16 GEMMs + custom kernels for RMSNorm/RoPE + naive attention
   * correctness vs Python reference.

2. **Week 3–4: Monolithic graphs**

   * Encoder full-step graph (32 layers) + decoder full-step graph (26 layers). 
   * persistent buffers + KV ring/paging.

3. **Week 5–6: Replace hotspots**

   * Decode seq_q=1 attention kernel (online softmax). 
     ntion-style kernel (query tiled). 
   * Fused FFN epilogue (SwiGLU).

4. **Week 7+: “Best possible” modes**

   * SM120A specialization + kernel auto-tuning
   * optional narrow precision (FP8/FP4) using SM120 tensor cores. ([NVIDIA Docs][5])



[1]: https://developer.nvidia.com/cuda/gpus "CUDA GPU Compute Capability | NVIDIA Developer"
[2]: https://www.notebookcheck.net/Nvidia-GeForce-RTX-5080-Laptop-Benchmarks-and-Specs.934946.0.html "Nvidia GeForce RTX 5080 Laptop - Benchmarks and Specs - NotebookCheck.net Tech"
[3]: https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/ "Navigating GPU Architecture Support: A Guide for NVIDIA CUDA Developers | NVIDIA Technical Blog"
[4]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html "5.1. Compute Capabilities — CUDA Programming Guide"
[5]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html "Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation"
[6]: https://docs.nvidia.com/gpudirect-storage/design-guide/index.html "1. Design Guide — GPUDirect Storage Design Guide"




