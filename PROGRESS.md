# Progress Report - CUDA Backend Implementation

## Achievements

### 1. Resident GPU Architecture
- Implemented **Unified Memory (Managed Memory)** across the entire engine.
- Replaced standard `malloc`/`calloc`/`free` with `vox_mem_*` wrappers that use `cudaMallocManaged` when CUDA is active.
- This allows model weights, KV caches, and intermediate activations to stay on the GPU, eliminating redundant PCIe transfers.

### 2. Optimized Weight Loading
- Modified `voxtral_safetensors.c` to bypass `mmap` when using CUDA.
- Weights are now loaded directly into Unified Memory using binary `fread` in 64MB chunks.
- This ensures that all tensor pointers are GPU-accessible from the moment the model is loaded.

### 3. High-Performance CUDA Kernels
- Implemented custom kernels for:
  - `RMSNorm` (with shared memory reduction)
  - `SiLU` and `GELU` activations
  - Element-wise Add and Mul
- Integrated **cuBLAS** for optimized matrix multiplications.

### 4. BF16 Compatibility & Optimization
- Implemented a specialized GPU dequantization kernel (`k_bf16_to_f32_conv`) to handle BF16 weights on hardware where `cublasGemmEx` BF16 support is unavailable.
- Optimized the linear layer path to perform dequantization and computation entirely on the GPU.

### 5. Seamless Dispatch
- Updated `voxtral_kernels.c` to automatically dispatch linear layers (`vox_linear_bf16`, `vox_linear_nobias_bf16`) and matrix operations to CUDA for batch/prefill processing (`seq_len > 1`).
- Maintained efficient CPU paths for single-token decoding to minimize synchronization overhead.

## Current Status
- **Build Status**: Passing (MSVC + NVCC)
- **Functional Status**: Passing regression tests (`runtest.ps1`)
- **Performance**: resident GPU mode active; weights and cache persist on device.

## Next Steps
- Implement a persistent scratch buffer for dequantization to eliminate `cudaMalloc` overhead during inference.
- Investigate asynchronous memory prefetching (`cudaStreamPrefetchAsync`) to further improve overlap between CPU and GPU tasks.
