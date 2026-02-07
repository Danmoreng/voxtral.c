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



