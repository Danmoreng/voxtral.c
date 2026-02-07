# Voxtral Metal Inference Pipeline Documentation

This document details the GPU-accelerated inference pipeline using Apple's Metal framework, optimized for Apple Silicon (M1/M2/M3).

## Overview

The Metal inference pipeline is designed to offload almost all tensor computations to the GPU. It uses **Metal Performance Shaders (MPS)** for heavy matrix multiplications and custom **Compute Shaders** (written in Metal Shading Language) for specialized operations like Rotary Positional Embeddings (RoPE), Layer Normalization, and Attention.

## 1. Memory Management (Zero-Copy)

- **Shared Memory**: Voxtral uses `vox_metal_shared_alloc`, which employs `MTLStorageModeShared`. This allows both the CPU and GPU to access the same physical memory, eliminating the need for expensive `memcpy` operations between host and device.
- **Weight Caching**: Although the model weights are stored as **BF16** on disk, MPS requires **F16** or **F32**. Voxtral implements a transparent BF16-to-F16 conversion cache. The first time a weight tensor is used, it is converted to F16 and cached in GPU memory; subsequent uses are near-instant.

## 2. Monolithic Decoder/Encoder Steps

A key optimization in the Metal backend is the "Monolithic" execution model. Instead of dispatching each operation (Normalization -> Matmul -> Activation -> ...) as a separate command to the GPU, Voxtral groups entire transformer layers or even the entire model forward pass into a single `MTLCommandBuffer`.

- **Benefit**: Reduces the overhead of CPU-GPU synchronization and command submission. In the decoder, this reduces the number of command buffers from ~53 per token to just 1.
- **Data Persistence**: Intermediate activations (like the residual stream `x`) stay in GPU memory throughout the entire monolithic pass.

## 3. Custom Compute Kernels (`voxtral_shaders.metal`)

While MPS handles matmuls, custom kernels are used for:

### A. RMSNorm & AdaNorm
- **Kernel**: `rms_norm`
- Uses threadgroup-local memory and `threadgroup_barrier` for parallel reductions to compute the sum of squares.
- **Adaptive Conditioning**: A specialized `ada_scale_mul` kernel applies the time-conditioning scale `(1 + ada_scale)` directly to normalized activations.

### B. Attention Mechanism
- **Decoder Attention**: A specialized `decoder_attention` kernel for single-token decoding ($seq_q=1$). It uses "Online Softmax" (single-pass) and SIMD-group reductions to achieve high efficiency.
- **Encoder/Prefill Attention**: `encoder_attention` uses a query-tiled approach ($8$ queries per threadgroup) to amortize the cost of loading Key and Value tensors from memory.
- **Causal & Window Masking**: Masking is applied directly within the softmax/attention kernels based on position offsets and a fixed window size (e.g., 750 or 8192).

### C. Rotary Positional Embeddings (RoPE)
- **Kernel**: `rope_apply` / `batched_rope_apply`
- Efficiently applies trigonometric rotations in-place on the Q and K tensors before attention.

### D. Fused FFN (SwiGLU)
- **Kernel**: `silu_mul_merged`
- When weights $W_1$ and $W_3$ are merged into a single matmul output, this kernel applies SiLU to the first half and multiplies it by the second half in one go, keeping the large FFN hidden state entirely on the GPU.

## 4. Pipeline Summary (Metal vs CPU)

| Feature | Metal Pipeline | CPU Pipeline |
| :--- | :--- | :--- |
| **Matmul** | MPS (Highly optimized) | AVX-512 VDPBF16PS / OpenBLAS |
| **Weights** | F16 (Cached from BF16) | BF16 (Direct) |
| **Sync** | Async Command Buffers | Synchronous / OpenMP |
| **Normalization** | GPU Compute Shader | AVX-512 / AVX2 |
| **Activations** | GPU Compute Shader | Vectorized (Approximated) |

## 5. Summary of GPU Data Flow

1.  **Input**: CPU writes audio features/token embeddings to Shared Memory.
2.  **Conversion**: If it's the first run, BF16 weights are converted to F16 on GPU.
3.  **Execution**: 
    - `vox_metal_encoder_full_step`: One command buffer processes all 32 encoder layers.
    - `vox_metal_decoder_full_step`: One command buffer processes all 26 decoder layers and computes logits.
4.  **Argmax**: A final `argmax_f32` kernel runs on the GPU to find the best token ID.
5.  **Output**: CPU reads the single integer result (token ID) back from Shared Memory.
