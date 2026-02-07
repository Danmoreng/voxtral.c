#ifndef VOXTRAL_AVX512_H
#define VOXTRAL_AVX512_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <malloc.h>
#define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * CPU Feature Detection
 * ======================================================================== */

static inline int avx512bf16_available(void) {
#ifdef _MSC_VER
    int cpuInfo[4];
    /* Check CPUID.7.0:EBX bit 16 (AVX-512 Foundation) */
    __cpuidex(cpuInfo, 7, 0);
    if (!(cpuInfo[1] & (1 << 16))) return 0;

    /* Check CPUID.7.1:EAX bit 5 (AVX-512 BF16) */
    __cpuidex(cpuInfo, 7, 1);
    return (cpuInfo[0] & (1 << 5)) ? 1 : 0;
#else
    uint32_t eax, ebx, ecx, edx;

    /* Check AVX-512 Foundation (CPUID.7.0:EBX bit 16) */
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    if (!(ebx & (1u << 16))) return 0;  /* No AVX-512F */

    /* Check AVX-512 BF16 (CPUID.7.1:EAX bit 5) */
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(1)
    );
    return (eax & (1u << 5)) ? 1 : 0;  /* AVX512_BF16 */
#endif
}

/* ========================================================================
 * BF16 ↔ FP32 Conversion Utilities
 * ======================================================================== */

/* Scalar BF16 → FP32: just a left shift by 16 bits. */
static inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Bit-cast helpers (safe for C/C++ and MSVC’s strict SIMD types) */
static inline __m512bh voxtral_bitcast_m512i_to_m512bh(__m512i v) {
    __m512bh out;
    memcpy(&out, &v, sizeof(out));
    return out;
}

static inline __m256i voxtral_bitcast_m256bh_to_m256i(__m256bh v) {
    __m256i out;
    memcpy(&out, &v, sizeof(out));
    return out;
}

/* Load 32 bf16 (64 bytes) and reinterpret as __m512bh */
static inline __m512bh voxtral_loadu_pbh(const uint16_t *p) {
    __m512i v = _mm512_loadu_si512((const void*)p);
    return voxtral_bitcast_m512i_to_m512bh(v);
}

/* Convert 16 fp32 -> 16 bf16 (packed in 256b), return as raw bits (__m256i) */
static inline __m256i fp32x16_to_bf16(__m512 v) {
    __m256bh bh = _mm512_cvtneps_pbh(v);   /* MSVC: returns __m256bh */
    return voxtral_bitcast_m256bh_to_m256i(bh);
}

/* ========================================================================
 * Core Matmul Kernel: C[M×N] += A[M×K](fp32) × B^T[N×K](bf16)
 * ======================================================================== */

static void matmul_avx512bf16_tiled(
    float *restrict C,
    const float *restrict A,
    const uint16_t *restrict B,
    int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    int K_padded = (K + 31) & ~31;

    /* Tile size for N dimension — process 4 B-rows at once */
    #define N_TILE 4

    int i;
#ifdef _MSC_VER
    #pragma omp parallel for schedule(static) if(M > 1) private(i)
#else
    #pragma omp parallel for schedule(static) if(M > 1)
#endif
    for (i = 0; i < M; i++) {
        const float *a_row = A + (size_t)i * K;

        /* Convert A row to BF16 once */
        uint16_t a_bf16_stack[((16384 + 31) & ~31)];  /* stack buffer */
        uint16_t *a_bf16 = a_bf16_stack;
#ifdef _MSC_VER
        if (K_padded > (int)sizeof(a_bf16_stack) / (int)sizeof(uint16_t)) {
            a_bf16 = (uint16_t *)_malloca(K_padded * sizeof(uint16_t));
        }
#else
        if (K_padded > (int)sizeof(a_bf16_stack) / (int)sizeof(uint16_t)) {
            a_bf16 = (uint16_t *)__builtin_alloca(K_padded * sizeof(uint16_t));
        }
#endif

        int k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 av = _mm512_loadu_ps(a_row + k);
            __m256i abf = fp32x16_to_bf16(av);
            _mm256_storeu_si256((__m256i *)(a_bf16 + k), abf);
        }
        for (; k < K; k++) {
            uint32_t bits;
            memcpy(&bits, &a_row[k], sizeof(bits));
            a_bf16[k] = (uint16_t)(bits >> 16);
        }
        for (k = K; k < K_padded; k++) {
            a_bf16[k] = 0;
        }

        /* Process N in tiles of N_TILE */
        int j = 0;
        for (; j + N_TILE - 1 < N; j += N_TILE) {
            const uint16_t *b0 = B + (size_t)(j + 0) * K;
            const uint16_t *b1 = B + (size_t)(j + 1) * K;
            const uint16_t *b2 = B + (size_t)(j + 2) * K;
            const uint16_t *b3 = B + (size_t)(j + 3) * K;

            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            for (int kk = 0; kk < K_padded; kk += 32) {
                __m512bh abh = voxtral_loadu_pbh(a_bf16 + kk);
                sum0 = _mm512_dpbf16_ps(sum0, abh, voxtral_loadu_pbh(b0 + kk));
                sum1 = _mm512_dpbf16_ps(sum1, abh, voxtral_loadu_pbh(b1 + kk));
                sum2 = _mm512_dpbf16_ps(sum2, abh, voxtral_loadu_pbh(b2 + kk));
                sum3 = _mm512_dpbf16_ps(sum3, abh, voxtral_loadu_pbh(b3 + kk));
            }

            C[(size_t)i * N + j + 0] = _mm512_reduce_add_ps(sum0);
            C[(size_t)i * N + j + 1] = _mm512_reduce_add_ps(sum1);
            C[(size_t)i * N + j + 2] = _mm512_reduce_add_ps(sum2);
            C[(size_t)i * N + j + 3] = _mm512_reduce_add_ps(sum3);
        }
        /* Remainder columns */
        for (; j < N; j++) {
            const uint16_t *b_row = B + (size_t)j * K;
            __m512 acc = _mm512_setzero_ps();
            for (int kk = 0; kk < K_padded; kk += 32) {
                acc = _mm512_dpbf16_ps(acc, voxtral_loadu_pbh(a_bf16 + kk), voxtral_loadu_pbh(b_row + kk));
            }
            C[(size_t)i * N + j] = _mm512_reduce_add_ps(acc);
        }
#ifdef _MSC_VER
        if (a_bf16 != a_bf16_stack) _freea(a_bf16);
#endif
    }
    #undef N_TILE
}

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_AVX512_H */
