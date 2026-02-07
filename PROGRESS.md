# Progress Report - CUDA Backend Implementation

## Achievements

### 1. Resident GPU Architecture (v3)
- **Explicit Device Memory**: Successfully migrated from Unified Memory to explicit device-local memory for all model state (weights, KV caches, activations).
- **Host Staging for Fast Loading**: Implemented host-side staging in `safetensors_open`, enabling weights to be loaded via standard file I/O into CPU memory and then copied to high-speed GPU memory in bulk.
- **Strict Memory Domains**: Established clear separation between host (`vox_cpu_*`) and device (`vox_gpu_*`) memory management.

### 2. Correctness & Type Safety
- **BF16 GEMM Compatibility**: Resolved the Bit-pattern mismatch by implementing a `k_f32_to_bf16` kernel and a persistent 256MB scratch buffer. `vox_cuda_matmul_bf16` now handles FP32 activations correctly on BF16 Tensor Cores.
- **Safe Kernel Execution**: Hardened RMSNorm kernels with NULL-checks for residual streams, ensuring stability during monolithic pass execution.

### 3. High-Performance GPU Primitives
- **GPU RoPE Frequencies**: Moved rotary embedding frequency calculations to the GPU via `vox_cuda_compute_rope_freqs`, eliminating a major CPU bottleneck.
- **GPU-Side Argmax**: Implemented `k_argmax` to determine the winning token directly on the device, reducing the per-token data transfer to a single integer and eliminating the costly `logits` synchronization.
- **Graph Capture Safe**: Removed all `cudaDeviceSynchronize()` calls from math primitives and replaced them with capture-safe error checking.

### 4. Pipeline Refactoring
- **Universal Data Movement**: Standardized all memory copies on `vox_mem_copy` (using `cudaMemcpyDefault`), simplifying H2D, D2H, and D2D logic.
- **Decoder & Encoder Alignment**: Synchronized both transformer backends to use the new resident architecture and GPU-side frequency generation.

## Current Status
- **Build Status**: Passing (MSVC 19.44 + NVCC 13.0)
- **Functional Status**: Phase 0 (Correctness + Residency) complete. The backend is now fully capture-safe and ready for monolithic graph tuning.
- **Performance**: Near-zero CPU interaction during decoder execution (outside of graph launch and argmax result copy).

## Next Steps
- Implement `cublasLt` plan caching to remove descriptor creation overhead from the captured graph.
- Re-enable and tune the **Monolithic CUDA Graph** for the decoder (Phase 1).
- Investigate SM120 (Blackwell) specific optimizations like `memcpy_async` and FP8 projection (Phase 2/3).
