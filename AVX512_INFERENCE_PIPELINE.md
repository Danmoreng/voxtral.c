# Voxtral AVX-512 Inference Pipeline Documentation

This document provides a detailed overview of the high-performance CPU inference pipeline in Voxtral, specifically targeting CPUs with **AVX-512 BF16** support (AMD Zen 4+, Intel Sapphire Rapids+).

## Overview

The AVX-512 inference pipeline is designed to maximize throughput by using 512-bit wide SIMD registers (ZMM) and specialized Brain Floating Point (BF16) hardware instructions. It handles the most computationally intensive parts of the model: matrix multiplications, normalizations, and activations.

## 1. Hardware Requirements & Detection

The pipeline requires a CPU supporting the following AVX-512 instruction sets:
- **AVX-512F** (Foundation)
- **AVX-512_BF16** (specifically the `VDPBF16PS` instruction)

### Detection Logic
Voxtral performs a runtime check using `cpuid`:
- Checks `CPUID.7.0:EBX` bit 16 for AVX-512F.
- Checks `CPUID.7.1:EAX` bit 5 for AVX-512_BF16.

If the binary is compiled with `USE_AVX512BF16` but the hardware lacks support, the application will terminate with a fatal error to prevent illegal instruction crashes.

## 2. Data Types & Storage

- **Weights**: Stored in **BF16** (16-bit) to reduce memory bandwidth and storage by 2x compared to FP32, while maintaining high accuracy.
- **Activations**: Primarily processed in **FP32** (32-bit) within the pipeline to maintain precision, converted to BF16 on-the-fly for specific dot-product operations.

## 3. Core Compute Kernels

### A. Matvec (Single-token Decoding)
Optimized for the case where the input sequence length $M=1$.

*   **Instruction**: `_mm512_dpbf16_ps` (VDPBF16PS)
*   **Pipeline Flow**:
    1.  **Input**: FP32 vector $A$ (length $K$) and BF16 weight matrix $B^T$ (size $N 	imes K$).
    2.  **Preparation**: The input vector $A$ is converted to BF16 on the stack once per call using `_mm512_cvtneps_pbh`.
    3.  **Accumulation**: Uses 4 independent ZMM accumulators (`sum0` through `sum3`) to process 128 BF16 values per iteration (4 x 32-element blocks). This unrolling improves Instruction Level Parallelism (ILP).
    4.  **Dot Product**: `_mm512_dpbf16_ps` performs a dot product of two pairs of BF16 values and adds the result to an FP32 accumulator.
    5.  **Output**: FP32 vector $C$ (length $N$).

### B. Matmul (Batch/Prefill)
Used for $M > 1$.

*   **Instruction**: `_mm512_dpbf16_ps`
*   **Pipeline Flow**:
    1.  **Tiling**: Processes the $N$ dimension in tiles of 4 rows.
    2.  **Row Conversion**: For each row in the batch $M$, the corresponding row of input $A$ is converted to BF16.
    3.  **Kernel**: For each $N$-tile, it loads the converted input row into a single ZMM register and broadcasts it to dot-product against 4 different weight matrix rows simultaneously.
    4.  **Parallelism**: Uses OpenMP (`#pragma omp parallel for`) to distribute batch rows across available CPU cores.

### C. RMSNorm (Root Mean Square Layer Normalization)
*   **Optimization**: Uses 512-bit wide loads and `_mm512_fmadd_ps` for square-sum calculation.
*   **Reduction**: Employs `_mm512_reduce_add_ps` for efficient cross-lane summation.
*   **Application**: Multiplies normalized values by weights using ZMM registers.

### D. Activation Functions (SiLU & GELU)
Both activations rely on a fast vectorized exponential approximation (`exp512_ps`).

*   **SiLU**: $f(x) = \frac{x}{1 + e^{-x}}$
    *   Implemented using `_mm512_div_ps` and the fast exp approximation.
*   **GELU**: $f(x) \approx 0.5x(1 + 	anh(\sqrt{2/\pi}(x + 0.044715x^3)))$
    *   The $	anh$ is implemented via the exponential identity: $	anh(x) = \frac{e^{2x}-1}{e^{2x}+1}$.
    *   The entire approximation is vectorized into 512-bit operations.

## 4. Summary of Pipeline Inputs & Outputs

| Operation | Inputs | Outputs | SIMD Width |
| :--- | :--- | :--- | :--- |
| **Linear Layer** | FP32 Tensor, BF16 Weights | FP32 Tensor | 512-bit |
| **RMSNorm** | FP32 Tensor, FP32 Weights | FP32 Tensor | 512-bit |
| **SiLU / GELU** | FP32 Tensor | FP32 Tensor (In-place) | 512-bit |
| **Element-wise** | FP32 Tensors | FP32 Tensor | 512-bit |

## 5. Performance Considerations

- **Memory Alignment**: While `_mm512_loadu_ps` is used (unaligned), the memory management in `voxtral.c` generally aligns large allocations to 64-byte boundaries to prevent cache-line splits.
- **Cache Locality**: Tiling in `matmul_avx512bf16_tiled` is tuned to keep working sets within L1/L2 caches where possible.
- **ILP**: Explicit register unrolling in the matvec kernel masks the latency of the `VDPBF16PS` instruction.
