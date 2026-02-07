Below is (1) an in-depth review of what you have \*right now\* (with the big correctness + performance blockers called out), and (2) a concrete implementation plan to get to a “Metal-style monolithic pipeline” on an RTX 5080 Laptop (Blackwell, compute capability 12.0). (\[Notebookcheck]\[1])



---



\## 1) In-depth code review of the current implementation



\### A. CUDA backend is currently not functionally correct (several hard blockers)



\#### 1) BF16 GEMM is type-incompatible with how you call it



Your higher-level code clearly expects \*\*FP32 activations × BF16 weights → FP32 output\*\*:



\* `vox\_linear\_bf16()` passes `x` as `float\*` and `W\_bf16` as BF16 (`uint16\_t\*`) into `vox\_cuda\_matmul\_bf16()` (`voxtral\_kernels.c`, around `vox\_linear\_bf16`).

\* Encoder/decoder use `vox\_cuda\_rms\_norm()` (float in/out) and then feed those floats into `vox\_cuda\_matmul\_bf16()`.



But `vox\_cuda\_matmul\_bf16()` sets \*\*both\*\* input layouts as `CUDA\_R\_16BF` (`voxtral\_cuda.cu` ~214–216). That means cuBLASLt will interpret your `float\*` activation buffers as BF16 bit-patterns → wrong results / memory misreads.



\*\*Fix direction:\*\* either



\* (preferred for perf) \*\*store activations as BF16\*\* on GPU, or

\* update `vox\_cuda\_matmul\_bf16()` to use \*\*mixed types\*\*: A = `CUDA\_R\_32F`, B = `CUDA\_R\_16BF` (or convert x to BF16 into a scratch buffer before GEMM).



\#### 2) Large BF16 weights are mmap’d host pointers, but CUDA code treats them like device pointers



`load\_bf16\_direct()` returns a pointer into the mmap’d safetensors buffer (`voxtral.c` ~216+), i.e. \*\*host virtual memory\*\*. Those pointers get stored in layer structs and are later passed straight into CUDA ops (GEMM, etc.).



cuBLASLt expects \*\*device\*\* (or at least properly registered/pinned UVA-accessible) memory. You don’t `cudaHostRegister()` the mmap region, and you don’t upload weights.



\*\*Net:\*\* on a machine with a GPU, `vox\_cuda\_available()` returns true and you’ll route compute through CUDA while weights are still host-backed → best case: crash; worst case: catastrophic perf.



\*\*Fix direction:\*\* introduce a CUDA weight-upload path (device memory) or stop using `\*\_direct` in CUDA mode.



\#### 3) CUDA graph capture path is broken by unconditional synchronizations



You attempt decoder CUDA-graph capture in `vox\_decoder\_forward()` (`voxtral\_decoder.c` ~430+). But many CUDA helpers call `cudaDeviceSynchronize()` unconditionally:



\* `vox\_cuda\_add\_inplace()` (`voxtral\_cuda.cu` ~447)

\* `vox\_cuda\_rope()` (`voxtral\_cuda.cu` ~518)

\* `vox\_cuda\_bias\_add()` (`~499)

\* several others



`cudaDeviceSynchronize()` is not capture-friendly; it will make capture fail.



You \*partially\* guarded sync with `g\_capturing` in some functions (e.g., RMSNorm, matmul), but not consistently.



\*\*Fix direction:\*\* \*\*remove syncs entirely\*\* from hotpath functions (or compile-time `#if VOX\_CUDA\_DEBUG\_SYNC`), and for capture use a dedicated stream + consistent “no-sync” behavior everywhere.



\#### 4) Null residual pointer dereference in CUDA RMSNorm residual kernels



In the decoder capture path you call:



\* `vox\_cuda\_rms\_norm\_ada\_residual(..., NULL, ...)` and

\* `vox\_cuda\_rms\_norm\_residual(..., NULL, ...)`



(see `voxtral\_decoder.c` ~494–496)



But the CUDA kernels unconditionally do `residual\[idx]` (`voxtral\_cuda.cu` ~314+ and ~360+). That’s a guaranteed crash if you ever reach it.



\*\*Fix direction:\*\* either don’t call the residual-fusing variant when residual is absent, or make kernels treat `residual == nullptr` as “no residual”.



---



\### B. Big performance problems even once correctness is fixed



\#### 1) Unified Memory is used as the default allocator for \*everything\*



`vox\_mem\_malloc()` uses `vox\_cuda\_malloc\_managed()` whenever CUDA is available (`voxtral.c` ~27–34; `voxtral\_cuda.cu` ~92–96). That means activations, KV cache, logits, etc. are \*\*Managed Memory\*\* by default.



That creates three major perf hazards:



\* \*\*Page migration stalls\*\* on first touch and on CPU/GPU ping-pong.

\* Your decoder does \*\*CPU argmax\*\* on `logits` after GPU writes. With managed memory, that forces synchronization + migration of (up to) the logits pages every token.

\* KV-cache growth/compaction uses `memcpy` on CPU; if those pages are GPU-resident, you force huge migrations.



\*\*Fix direction:\*\* GPU fast path should use \*\*device memory for model state\*\*, and only transfer minimal outputs (e.g., top token id) back to CPU.



\#### 2) Excessive device-wide synchronization in many primitives



A large fraction of `vox\_cuda\_\*` helpers end with `cudaDeviceSynchronize()`. Even outside graphs, this kills overlap and turns your pipeline into “launch → hard stall → launch → hard stall”.



\*\*Fix direction:\*\* switch to:



\* a dedicated non-blocking stream,

\* `cudaMemcpyAsync`,

\* and only synchronize at \*well-defined boundaries\* (or not at all if you keep everything on-GPU).



\#### 3) cuBLASLt setup overhead per call (encoder path)



`vox\_cuda\_matmul\_bf16()` creates/destroys descriptors and runs heuristic selection every call. In the decoder graph capture, that overhead is “only once”, but encoder likely calls it many times (no monolithic capture yet).



\*\*Fix direction:\*\* cache matmul “plans” per (M,N,K,types,layout) — descriptors + chosen algorithm + workspace requirement.



\#### 4) Attention kernels are extremely naive (will dominate decode latency)



Your decode attention kernel is a straightforward loop over `seq\_k` keys with expf and per-key block reductions. For `seq\_k` up to 8192 and 26 layers, this is a major hotspot even if GEMMs are fast.



\*\*Fix direction:\*\* use a FlashAttention-style kernel (or a dedicated single-query decode attention kernel) and store KV cache in a layout that supports coalesced loads.



\#### 5) Output projection (vocab) is bandwidth-heavy and likely your \*#1 per-token bottleneck\*



`VOX\_VOCAB\_SIZE = 131072`, `VOX\_DEC\_DIM = 3072`: the output matrix is ~805 MB in BF16. You stream that every token unless you quantize/compress.



On Blackwell you have great low-precision options (FP8/FP4 support), and this is exactly the matrix you’d most want to quantize. (\[NVIDIA Developer]\[2])



---



\### C. API/maintenance issues that will bite you as you iterate



\* `voxtral\_cuda.h` declares functions that don’t exist in `voxtral\_cuda.cu` (API drift).

\* `transpose\_b` param in `vox\_cuda\_matmul\_bf16` is effectively unused (contract mismatch).

\* Global CUDA state (`g\_cublas\_handle`, `g\_capturing`, etc.) makes multi-context / multi-thread use difficult later.



---



\## 2) A concrete plan to reach “best possible” RTX 5080 Laptop performance



RTX 5080 Laptop is Blackwell GB203 and compute capability 12.0. (\[Notebookcheck]\[1])

Blackwell supports FP16/BF16 and also FP8/FP4 formats via 5th-gen Tensor Cores, which matters a lot for your giant vocab projection. (\[NVIDIA Developer]\[2])



\### Phase 0 — Make CUDA path correct + graph-capture-safe (no perf heroics yet)



1\. \*\*Split memory domains\*\*



&nbsp;  \* Replace the current “managed-by-default” behavior with explicit allocators:



&nbsp;    \* `vox\_gpu\_malloc()` → device (cudaMallocAsync/pool)

&nbsp;    \* `vox\_cpu\_malloc()` → pageable or pinned

&nbsp;    \* only use Managed as an \*optional fallback\*.

&nbsp;  \* Keep: weights, activations, KV cache, logits on \*\*device\*\*.



2\. \*\*Fix BF16 matmul contract\*\*

&nbsp;  Decide one of these two:



&nbsp;  \* \*\*Option A (recommended): BF16 activations end-to-end\*\*



&nbsp;    \* RMSNorm outputs BF16 (compute stats in FP32).

&nbsp;    \* All elementwise ops use BF16/FP32 accumulation.

&nbsp;    \* GEMMs are BF16×BF16→FP32(or BF16) on Tensor Cores.

&nbsp;  \* \*\*Option B: FP32 activations\*\*



&nbsp;    \* Modify `vox\_cuda\_matmul\_bf16` to A=`CUDA\_R\_32F`, B=`CUDA\_R\_16BF`, compute=`CUBLAS\_COMPUTE\_32F\_FAST\_16BF` (or equivalent).

&nbsp;    \* This is easier but usually slower vs BF16 activations.



3\. \*\*Upload weights to device\*\*



&nbsp;  \* When CUDA is enabled, do \*\*not\*\* use `load\_bf16\_direct()`.

&nbsp;  \* Add: `load\_bf16\_to\_device(name, expected\_elems)`:



&nbsp;    \* parse safetensors tensor offsets,

&nbsp;    \* allocate device BF16 buffer,

&nbsp;    \* copy from mmap → device (pinned staging + `cudaMemcpyAsync`) or directly with GDS (later).

&nbsp;  \* Store device pointers in the model structs.



4\. \*\*Make every CUDA primitive capture-safe\*\*



&nbsp;  \* Remove all `cudaDeviceSynchronize()` from math helpers.

&nbsp;  \* Replace “sync + last error” with:



&nbsp;    \* `cudaGetLastError()` in debug builds only,

&nbsp;    \* optionally `cudaPeekAtLastError()` (also debug).

&nbsp;  \* Ensure \*no forbidden API calls\* occur during capture.



5\. \*\*Fix RMSNorm residual null handling\*\*



&nbsp;  \* Either add separate kernels (`rms\_norm`, `rms\_norm\_residual`, `rms\_norm\_ada`, `rms\_norm\_ada\_residual`) with correct usage,

&nbsp;  \* or accept `residual==nullptr` and branch safely.



6\. \*\*Move argmax/top-k to GPU\*\*



&nbsp;  \* Add a tiny kernel to compute argmax over logits (maybe blockwise reduction).

&nbsp;  \* Copy back only `int next\_token`.

&nbsp;  \* This removes the biggest managed-memory sync trap.



At the end of Phase 0, CUDA should run correctly and graph capture should work reliably.



---



\### Phase 1 — “Metal-style monolithic decode” with CUDA Graphs



Goal: replicate what your Metal doc describes: \*\*one captured decode graph\*\*, replayed per token with minimal per-step CPU work.



1\. \*\*Dedicated stream + graph capture\*\*



&nbsp;  \* Create `cudaStream\_t stream` (non-blocking).

&nbsp;  \* Capture decode graph on that stream (relaxed mode), not the legacy default stream.

&nbsp;  \* Use persistent device buffers for:



&nbsp;    \* `x`, `x\_norm`, `q`, `k`, `v`, `att`, FFN temps, `logits`, etc.

&nbsp;  \* Keep `pos` and `total\_seq` in device memory and update via tiny device kernels or memcpy nodes.



2\. \*\*Prebuild/cached GEMM plans\*\*



&nbsp;  \* For each GEMM shape in the decoder (Q, K, V, out proj, FFN W1/W3/W2, vocab),

&nbsp;    cache:



&nbsp;    \* layouts, matmul desc, chosen algorithm, workspace size.

&nbsp;  \* This matches your Metal “prewarm/caching” concept.



3\. \*\*Fuse what’s cheap to fuse\*\*



&nbsp;  \* Fuse residual adds and simple elementwise ops:



&nbsp;    \* (a) add\_inplace

&nbsp;    \* (b) silu/swiglu

&nbsp;    \* (c) bias add

&nbsp;  \* Keep GEMMs separate initially; cublasLt epilogues can fuse bias, but your pipeline already has bias kernels.



---



\### Phase 2 — Replace attention with a high-performance kernel (this is the real decode win)



Your current attention loop is the wrong shape for modern GPUs.



1\. \*\*Store KV cache in a decode-friendly layout\*\*



&nbsp;  \* Use a \*paged\* layout (like vLLM): blocks of e.g. 128 tokens.

&nbsp;  \* Store K and V as \*\*BF16\*\* (or FP16), not FP32.

&nbsp;  \* Keep per-layer pointers to blocks.



2\. \*\*Use a specialized single-query attention kernel\*\*



&nbsp;  \* Q is (heads × head\_dim), K/V are (seq\_k × kv\_heads × head\_dim).

&nbsp;  \* Compute scores in tiles of keys; do numerically stable softmax with warp-level reductions.

&nbsp;  \* Write one output vector per head.

&nbsp;  \* (If you want peak) consider tensor-core MMA for score tiles (Q·Kᵀ) — Blackwell is great at this, but even a good “SIMD BF16 dot” kernel is a huge improvement.



3\. \*\*Prefill (encoder + first decode step)\*\*



&nbsp;  \* For seq\_q > 1, use FlashAttention-like kernels (or CUTLASS attention examples).

&nbsp;  \* Don’t maintain two completely separate attention implementations forever; share tiling primitives.



---



\### Phase 3 — Attack the real bandwidth hog: vocab projection



Even with perfect attention, output projection can dominate because the vocab matrix is enormous.



1\. \*\*Quantize the vocab matrix\*\*



&nbsp;  \* Blackwell supports FP8 and FP4 formats; NVIDIA explicitly positions these for faster inference and lower memory bandwidth. (\[NVIDIA Developer]\[2])

&nbsp;  \* Practical approach:



&nbsp;    \* Keep most model weights BF16.

&nbsp;    \* Quantize \*\*only\*\* `tok\_embeddings` / output projection to FP8 first (less risky than FP4).

&nbsp;    \* Use a per-row scale scheme compatible with your current safetensors loading.



2\. \*\*Use an FP8 GEMM path\*\*



&nbsp;  \* cublasLt / CUTLASS can do FP8 GEMMs (details depend on your CUDA version + library support).

&nbsp;  \* Output logits can remain FP16/FP32, but do argmax on GPU.



This single change often yields outsized real-world tok/s improvements.



---



\### Phase 4 — Optional: “Direct I/O” weight loading (GPUDirect Storage)



This is mostly about \*\*startup time\*\* and avoiding CPU copies, not per-token speed.



1\. \*\*GPUDirect Storage (cuFile) path (optional)\*\*



&nbsp;  \* Use `cuFileRead` to read weight segments directly into device buffers (nice match for safetensors offsets).

&nbsp;  \* NVIDIA’s GDS stack is designed for DMA between storage and GPU memory. (\[NVIDIA Docs]\[3])



2\. \*\*Reality check on laptops\*\*



&nbsp;  \* GDS has platform + filesystem constraints (e.g., ext4/XFS specifics for local filesystems). (\[NVIDIA Docs]\[4])

&nbsp;  \* On many consumer laptop setups, you may end up falling back to:



&nbsp;    \* `pread`/`io\_uring` into pinned host buffers

&nbsp;    \* `cudaMemcpyAsync` to device

&nbsp;  \* Still worth architecting as an optional fast path.



---



\## What I would do first (highest ROI order)



1\. Fix matmul type mismatch + weight residency (device upload) → make CUDA correct.

2\. Remove syncs everywhere + dedicated stream + decoder CUDA graph → huge latency win.

3\. GPU argmax (stop touching logits on CPU) → removes a major hidden sync.

4\. Replace attention kernel (decode) + BF16 KV cache → big tok/s jump.

5\. Quantize vocab projection to FP8 → likely the biggest real-world throughput boost on Blackwell. (\[NVIDIA Developer]\[2])



---



If you want, I can follow up with a \*surgical patch plan\* (function-by-function) for:



\* fixing `vox\_cuda\_matmul\_bf16` (mixed types + row-major layouts),

\* making every `vox\_cuda\_\*` capture-safe (no sync),

\* adding GPU argmax,

\* and a minimal “upload BF16 weights to device” loader that plugs into your current safetensors offsets.



\[1]: https://www.notebookcheck.com/Nvidia-GeForce-RTX-5080-Laptop-Grafikkarte-Benchmarks-und-Spezifikationen.941440.0.html "https://www.notebookcheck.com/Nvidia-GeForce-RTX-5080-Laptop-Grafikkarte-Benchmarks-und-Spezifikationen.941440.0.html"

\[2]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | NVIDIA Technical Blog"

\[3]: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html "1. Overview Guide — GPUDirect Storage Overview Guide"

\[4]: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html "1. Installation and Troubleshooting Guide — GPUDirect Storage Installation and Troubleshooting Guide"



