/*
 * voxtral_kernels.c - Math kernels for Voxtral inference
 * Adapted from flux-2-4b project.
 */

#include "voxtral_kernels.h"
#include "voxtral.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_METAL
#include "voxtral_metal.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#endif

#ifdef USE_AVX512BF16
#include "voxtral_avx512.h"
/* Fast vectorized exp approximation for SiLU/GELU (AVX-512) */
static inline __m512 exp512_ps(__m512 x) {
    static const __m512 log2e = {1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f,
                                 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f,
                                 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f,
                                 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f};
    static const __m512 c1 = {0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f,
                              0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f};
    static const __m512 c2 = {-2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f,
                              -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f};
    static const __m512 p0 = {1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f,
                              1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f};
    static const __m512 p1 = {1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f,
                              1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f};
    static const __m512 p2 = {8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f,
                              8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f};
    static const __m512 p3 = {4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f,
                              4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f};
    static const __m512 p4 = {1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f,
                              1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f};
    static const __m512 p5 = {5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f,
                              5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f};

    __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    __m512 t = _mm512_fnmadd_ps(fx, c1, x);
    t = _mm512_fnmadd_ps(fx, c2, t);
    __m512 z = _mm512_mul_ps(t, t);
    __m512 y = _mm512_fmadd_ps(p0, t, p1);
    y = _mm512_fmadd_ps(y, t, p2);
    y = _mm512_fmadd_ps(y, t, p3);
    y = _mm512_fmadd_ps(y, t, p4);
    y = _mm512_fmadd_ps(y, t, p5);
    y = _mm512_add_ps(_mm512_fmadd_ps(y, z, t), _mm512_set1_ps(1.0f));

    /* Build 2^n */
    __m512i imm0 = _mm512_cvtps_epi32(fx);
    imm0 = _mm512_add_epi32(imm0, _mm512_set1_epi32(127));
    imm0 = _mm512_slli_epi32(imm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(imm0);

    return _mm512_mul_ps(y, pow2n);
}

/* Cached runtime check: -1 = unchecked, 0 = unavailable, 1 = available */
static int avx512bf16_detected = -1;
static int avx512bf16_check(void) {
    if (avx512bf16_detected == -1)
        avx512bf16_detected = avx512bf16_available();
    if (!avx512bf16_detected) {
        fprintf(stderr, "FATAL: This binary was compiled with AVX-512 BF16 support,\n"
                        "but this CPU does not support it.\n"
                        "Required: AMD Zen 4+ or Intel Sapphire Rapids+.\n");
        exit(1);
    }
    return 1;
}
#endif

#ifdef USE_CUDA
#include "voxtral_cuda.h"
#endif

/* Minimum matrix size to use GPU */
#define MIN_GPU_ELEMENTS (512 * 512)

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void vox_add_inplace(float *a, const float *b, int n) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        vox_cuda_add_inplace(a, b, n);
        return;
    }
#endif
    int i = 0;
#if defined(USE_AVX512BF16)
    for (; i <= n - 16; i += 16) {
        _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; i++) a[i] += b[i];
}

void vox_mul_inplace(float *a, const float *b, int n) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        vox_cuda_mul_inplace(a, b, n);
        return;
    }
#endif
    int i = 0;
#if defined(USE_AVX512BF16)
    for (; i <= n - 16; i += 16) {
        _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; i++) a[i] *= b[i];
}

void vox_axpy(float *a, float scale, const float *b, int n) {
    int i = 0;
    /* TODO: CUDA axpy kernel */
#if defined(USE_AVX512BF16)
    __m512 s512 = _mm512_set1_ps(scale);
    for (; i <= n - 16; i += 16) {
        _mm512_storeu_ps(a + i, _mm512_fmadd_ps(s512, _mm512_loadu_ps(b + i), _mm512_loadu_ps(a + i)));
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    __m256 s = _mm256_set1_ps(scale);
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(a + i, _mm256_fmadd_ps(s, _mm256_loadu_ps(b + i), _mm256_loadu_ps(a + i)));
    }
#endif
    for (; i < n; i++) a[i] += scale * b[i];
}

void vox_scale(float *x, float s, int n) {
    /* TODO: CUDA scale kernel */
    int i = 0;
#if defined(USE_AVX512BF16)
    __m512 s_vec512 = _mm512_set1_ps(s);
    for (; i <= n - 16; i += 16) {
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), s_vec512));
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    __m256 s_vec = _mm256_set1_ps(s);
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), s_vec));
    }
#endif
    for (; i < n; i++) x[i] *= s;
}

void vox_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* Block size for tiling - tuned for L1/L2 cache */
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

void vox_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        /* cuBLAS handles large matrices efficiently */
        vox_cuda_sgemm(M, N, K, A, B, C);
        return;
    }
#endif
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    int m, n;
    /* Initialize C to zero */
    #pragma omp parallel for private(n)
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            C[m * N + n] = 0.0f;
        }
    }

    /* Tiled matrix multiplication */
    int m0, n0;
    #pragma omp parallel for private(n0) schedule(dynamic)
    for (m0 = 0; m0 < M; m0 += BLOCK_M) {
        for (n0 = 0; n0 < N; n0 += BLOCK_N) {
            int m_end = (m0 + BLOCK_M < M) ? m0 + BLOCK_M : M;
            int n_end = (n0 + BLOCK_N < N) ? n0 + BLOCK_N : N;

            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;

                for (int m = m0; m < m_end; m++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_val = A[m * K + k];
                        for (int n = n0; n < n_end; n++) {
                            C[m * N + n] += a_val * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
#endif
}

void vox_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        vox_cuda_sgemm_t(M, N, K, A, B, C);
        return;
    }
#endif
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    int m, n;
    /* Initialize C to zero */
    #pragma omp parallel for private(n)
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            C[m * N + n] = 0.0f;
        }
    }

    int m0, n0;
    /* Tiled matrix multiplication (B is transposed: B[n][k]) */
    #pragma omp parallel for private(n0) schedule(dynamic)
    for (m0 = 0; m0 < M; m0 += BLOCK_M) {
        for (n0 = 0; n0 < N; n0 += BLOCK_N) {
            int m_end = (m0 + BLOCK_M < M) ? m0 + BLOCK_M : M;
            int n_end = (n0 + BLOCK_N < N) ? n0 + BLOCK_N : N;
            
            /* Process all k blocks for this m,n block */
            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
                
                for (int m = m0; m < m_end; m++) {
                    for (int n = n0; n < n_end; n++) {
                        float sum = 0.0f;
                        for (int k = k0; k < k_end; k++) {
                            sum += A[m * K + k] * B[n * K + k];
                        }
                        C[m * N + n] += sum;
                    }
                }
            }
        }
    }
#endif
}

void vox_linear(float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    int s, o, i;
    #pragma omp parallel for private(o, i)
    for (s = 0; s < seq_len; s++) {
        for (o = 0; o < out_dim; o++) {
            const float *x_row = x + s * in_dim;
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y[s * out_dim + o] = sum;
        }
    }
#endif
}

void vox_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim) {
    vox_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Reusable scratch buffer for bf16->f32 conversion (avoids malloc/free per call) */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        vox_mem_free(bf16_scratch);
        bf16_scratch = (float *)vox_mem_malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 *
 * Reads BF16 weights directly and converts in-register, avoiding the
 * double-streaming penalty of "convert full matrix then BLAS".
 * This is the critical fast path for single-token decoder generation.
 */
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    int o;
    #pragma omp parallel for private(o)
    for (o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        __m256 acc = _mm256_setzero_ps();
        for (; k + 8 <= in_dim; k += 8) {
            /* Load 8 bf16 weights (128 bits) */
            __m128i bf = _mm_loadu_si128((const __m128i*)(w_row + k));
            /* Expand to 8x32-bit ints */
            __m256i w_int = _mm256_cvtepu16_epi32(bf);
            /* Shift left by 16 to get f32 bit pattern */
            w_int = _mm256_slli_epi32(w_int, 16);
            /* Cast to float */
            __m256 w_f32 = _mm256_castsi256_ps(w_int);
            
            /* Load 8 input floats */
            __m256 x_vec = _mm256_loadu_ps(x + k);
            
            /* Fused multiply-add */
            acc = _mm256_fmadd_ps(w_f32, x_vec, acc);
        }
        /* Horizontal sum */
        float temp[8];
        _mm256_storeu_ps(temp, acc);
        for(int i=0; i<8; i++) sum += temp[i];
#elif defined(__ARM_NEON)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        for (; k + 8 <= in_dim; k += 8) {
            /* Load 8 bf16 weights and convert to f32 in registers */
            uint16x8_t bf = vld1q_u16(w_row + k);
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
            float32x4_t w0 = vreinterpretq_f32_u32(lo);
            float32x4_t w1 = vreinterpretq_f32_u32(hi);

            /* Load 8 f32 input values */
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);

            /* Fused multiply-accumulate */
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
        }

        sum += vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

        /* Scalar tail */
        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }

        y[o] = sum;
    }
}

void vox_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim) {
#ifdef USE_CUDA
    if (vox_cuda_available() && seq_len > 1) {
        vox_cuda_matmul_t_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        return;
    }
#endif
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        return;
    }
#endif
#ifdef USE_AVX512BF16
    if (seq_len > 1) {
        avx512bf16_check();
        matmul_avx512bf16_tiled(y, x, W_bf16, seq_len, out_dim, in_dim);
        return;
    } else if (seq_len == 1) {
        avx512bf16_check();
        matvec_avx512bf16(y, x, W_bf16, out_dim, in_dim);
        return;
    }
#endif
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    bf16_to_f32_buf(W_f32, W_bf16, n);
    vox_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void vox_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim) {
#ifdef USE_CUDA
    if (vox_cuda_available() && seq_len > 1) {
        vox_cuda_matmul_t_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif
#ifdef USE_AVX512BF16
    if (seq_len > 1) {
        avx512bf16_check();
        matmul_avx512bf16_tiled(y, x, W_bf16, seq_len, out_dim, in_dim);
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    } else if (seq_len == 1) {
        avx512bf16_check();
        matvec_avx512bf16(y, x, W_bf16, out_dim, in_dim);
        if (b != NULL) {
            for (int o = 0; o < out_dim; o++) y[o] += b[o];
        }
        return;
    }
#endif
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    bf16_to_f32_buf(W_f32, W_bf16, n);
    vox_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

void vox_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        vox_cuda_matmul_t_bf16(M, N, K, A, B_bf16, C);
        return;
    }
#endif
    /*
     * C[M,N] = A[M,K] @ B[N,K]^T
     * For M=1: use fused BF16 matvec (no intermediate buffer needed).
     * For M>1: convert full matrix and use BLAS.
     */
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(M, N, K, A, B_bf16, C);
        return;
    }
#endif
#ifdef USE_AVX512BF16
    if (M > 1) {
        avx512bf16_check();
        matmul_avx512bf16_tiled(C, A, B_bf16, M, N, K);
        return;
    } else if (M == 1) {
        avx512bf16_check();
        matvec_avx512bf16(C, A, B_bf16, N, K);
        return;
    }
#endif
    if (M == 1) {
        bf16_matvec_fused(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        float *B_f32 = bf16_get_scratch(n);
        if (!B_f32) return;
        bf16_to_f32_buf(B_f32, B_bf16, n);
        vox_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * 1D Convolution
 * ======================================================================== */

void vox_conv1d(float *out, const float *in, const float *weight, const float *bias,
                int channels_in, int channels_out, int length,
                int kernel_size, int stride, int padding) {
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;

    for (int oc = 0; oc < channels_out; oc++) {
        float b = (bias != NULL) ? bias[oc] : 0.0f;
        for (int ol = 0; ol < out_length; ol++) {
            float sum = b;
            for (int ic = 0; ic < channels_in; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int il = ol * stride - padding + k;
                    if (il >= 0 && il < length) {
                        int w_idx = oc * channels_in * kernel_size + ic * kernel_size + k;
                        sum += in[ic * length + il] * weight[w_idx];
                    }
                }
            }
            out[oc * out_length + ol] = sum;
        }
    }
}

void vox_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                       int channels_in, int channels_out, int length,
                       int kernel_size, int stride) {
    /* Matches vLLM WhisperCausalConv1d padding scheme.
     * Uses im2col + BLAS sgemm for fast computation. */
    int padding_total = kernel_size - stride;
    float n_frames = ((float)length - kernel_size + padding_total) / (float)stride + 1.0f;
    int out_length = (int)ceilf(n_frames);
    if (out_length <= 0) return;

    int left_pad = padding_total;
    int K = channels_in * kernel_size;

    /* Build im2col matrix: [K, out_length] row-major.
     * im2col[ic*kernel_size + k, ol] = in[ic, ol*stride - left_pad + k] (0 if OOB) */
    float *im2col = (float *)vox_mem_calloc((size_t)K * out_length, sizeof(float));
    for (int ol = 0; ol < out_length; ol++) {
        for (int ic = 0; ic < channels_in; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                int il = ol * stride - left_pad + k;
                if (il >= 0 && il < length) {
                    im2col[(size_t)(ic * kernel_size + k) * out_length + ol] =
                        in[(size_t)ic * length + il];
                }
            }
        }
    }

    /* out = weight × im2col: [channels_out, K] × [K, out_length] → [channels_out, out_length] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                channels_out, out_length, K,
                1.0f,
                weight, K,
                im2col, out_length,
                0.0f,
                out, out_length);
#else
    /* Use vox_matmul (which handles AVX2/AVX512/OpenMP) instead of raw loop */
    vox_matmul(out, weight, im2col, channels_out, K, out_length);
#endif
    vox_mem_free(im2col);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < channels_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * out_length;
            for (int ol = 0; ol < out_length; ol++)
                row[ol] += b;
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void vox_rms_norm(float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps) {
#ifdef USE_CUDA
    if (vox_cuda_available()) {
        vox_cuda_rms_norm(out, x, weight, seq_len, hidden, eps);
        return;
    }
#endif
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        float sum_sq = 0.0f;
        int i = 0;

#if defined(USE_AVX512BF16)
        __m512 v_sum_sq512 = _mm512_setzero_ps();
        for (; i <= hidden - 16; i += 16) {
            __m512 v_x = _mm512_loadu_ps(x_row + i);
            v_sum_sq512 = _mm512_fmadd_ps(v_x, v_x, v_sum_sq512);
        }
        sum_sq = _mm512_reduce_add_ps(v_sum_sq512);
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        __m256 v_sum_sq = _mm256_setzero_ps();
        for (; i <= hidden - 8; i += 8) {
            __m256 v_x = _mm256_loadu_ps(x_row + i);
            v_sum_sq = _mm256_fmadd_ps(v_x, v_x, v_sum_sq);
        }
        float temp[8];
        _mm256_storeu_ps(temp, v_sum_sq);
        for (int j = 0; j < 8; j++) sum_sq += temp[j];
#endif
        for (; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }

        float rms = sqrtf(sum_sq / hidden + eps);
        float rms_inv = 1.0f / rms;

        i = 0;
#if defined(USE_AVX512BF16)
        __m512 v_rms_inv512 = _mm512_set1_ps(rms_inv);
        for (; i <= hidden - 16; i += 16) {
            __m512 v_x = _mm512_loadu_ps(x_row + i);
            __m512 v_w = _mm512_loadu_ps(weight + i);
            _mm512_storeu_ps(out_row + i, _mm512_mul_ps(_mm512_mul_ps(v_x, v_rms_inv512), v_w));
        }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        __m256 v_rms_inv = _mm256_set1_ps(rms_inv);
        for (; i <= hidden - 8; i += 8) {
            __m256 v_x = _mm256_loadu_ps(x_row + i);
            __m256 v_w = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out_row + i, _mm256_mul_ps(_mm256_mul_ps(v_x, v_rms_inv), v_w));
        }
#endif
        for (; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
/* Fast vectorized exp approximation for SiLU/GELU */
static inline __m256 exp256_ps(__m256 x) {
    /* exp(x) = 2^(x * log2(e)) */
    static const __m256 log2e = {1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f,
                                 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f, 1.4426950408889634074f};
    static const __m256 c1 = {0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f};
    static const __m256 c2 = {-2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f};
    static const __m256 p0 = {1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f, 1.9875691500e-4f};
    static const __m256 p1 = {1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f, 1.3981999507e-3f};
    static const __m256 p2 = {8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f, 8.3334519073e-3f};
    static const __m256 p3 = {4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f, 4.1665795894e-2f};
    static const __m256 p4 = {1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f, 1.6666665459e-1f};
    static const __m256 p5 = {5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f, 5.0000001201e-1f};

    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    __m256 t = _mm256_fnmadd_ps(fx, c1, x);
    t = _mm256_fnmadd_ps(fx, c2, t);
    __m256 z = _mm256_mul_ps(t, t);
    __m256 y = _mm256_fmadd_ps(p0, t, p1);
    y = _mm256_fmadd_ps(y, t, p2);
    y = _mm256_fmadd_ps(y, t, p3);
    y = _mm256_fmadd_ps(y, t, p4);
    y = _mm256_fmadd_ps(y, t, p5);
    y = _mm256_add_ps(_mm256_fmadd_ps(y, z, t), _mm256_set1_ps(1.0f));

    /* Build 2^n */
    __m256i imm0 = _mm256_cvtps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(127));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);

    return _mm256_mul_ps(y, pow2n);
}
#endif

void vox_silu(float *x, int n) {
    int i = 0;
#if defined(USE_AVX512BF16)
    __m512 one512 = _mm512_set1_ps(1.0f);
    for (; i <= n - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vexp = exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), vx));
        __m512 res = _mm512_div_ps(vx, _mm512_add_ps(one512, vexp));
        _mm512_storeu_ps(x + i, res);
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    __m256 one = _mm256_set1_ps(1.0f);
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vexp = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), vx));
        __m256 res = _mm256_div_ps(vx, _mm256_add_ps(one, vexp));
        _mm256_storeu_ps(x + i, res);
    }
#endif
    for (; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void vox_gelu(float *x, int n) {
    int i = 0;
#if defined(USE_AVX512BF16)
    __m512 half512 = _mm512_set1_ps(0.5f);
    __m512 one512 = _mm512_set1_ps(1.0f);
    __m512 k0_512 = _mm512_set1_ps(0.7978845608f);
    __m512 k1_512 = _mm512_set1_ps(0.044715f);
    __m512 two512 = _mm512_set1_ps(2.0f);

    for (; i <= n - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(vx, vx), vx);
        __m512 inner = _mm512_mul_ps(k0_512, _mm512_fmadd_ps(k1_512, x3, vx));
        __m512 e2x = exp512_ps(_mm512_mul_ps(two512, inner));
        __m512 vtanh = _mm512_div_ps(_mm512_sub_ps(e2x, one512), _mm512_add_ps(e2x, one512));
        __m512 res = _mm512_mul_ps(half512, _mm512_mul_ps(vx, _mm512_add_ps(one512, vtanh)));
        _mm512_storeu_ps(x + i, res);
    }
#elif defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 k0 = _mm256_set1_ps(0.7978845608f); /* sqrt(2/pi) */
    __m256 k1 = _mm256_set1_ps(0.044715f);

    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(vx, vx), vx);
        __m256 inner = _mm256_mul_ps(k0, _mm256_fmadd_ps(k1, x3, vx));
        
        /* tanh(x) approx using exp: (exp(2x) - 1) / (exp(2x) + 1) */
        __m256 e2x = exp256_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), inner));
        __m256 vtanh = _mm256_div_ps(_mm256_sub_ps(e2x, one), _mm256_add_ps(e2x, one));
        
        __m256 res = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, vtanh)));
        _mm256_storeu_ps(x + i, res);
    }
#endif
    for (; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void vox_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

void vox_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int window_size,
                          int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    /* Process each query head in parallel */
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;  /* GQA: map query head to KV head */

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;

            /* Global position of this query */
            int global_pos = q_offset + i;

            /* Causal mask: can attend to K positions 0..global_pos
             * Sliding window: can attend to positions >= global_pos - window_size + 1 */
            int k_start = 0;
            if (window_size > 0 && global_pos - window_size + 1 > 0) {
                k_start = global_pos - window_size + 1;
            }
            int k_end = global_pos + 1;  /* Causal: up to and including current position */
            if (k_end > seq_k) k_end = seq_k;

            /* Online softmax for memory efficiency */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = k_start; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                /* Compute attention score */
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;

                /* Online softmax update */
                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    float weight = expf(score - max_score);
                    sum_exp += weight;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += weight * v_row[d];
                    }
                }
            }

            /* Normalize */
            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

void vox_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int s = 0; s < seq; s++) {
        float p = (float)pos[s];
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = p * freq;
            freqs[s * half_dim * 2 + d * 2] = cosf(angle);
            freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

void vox_apply_rope(float *x, const float *freqs, int seq, int heads, int head_dim) {
    /* x: [seq, heads * head_dim]
     * freqs: [seq, head_dim/2, 2] (cos, sin pairs)
     * Apply rotary embedding to consecutive pairs */

    int half_dim = head_dim / 2;
    int hidden = heads * head_dim;

    int s, h, d;
    #pragma omp parallel for private(h, d)
    for (s = 0; s < seq; s++) {
        for (h = 0; h < heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

            for (d = 0; d < half_dim; d++) {
                float cos_val = freqs[s * half_dim * 2 + d * 2];
                float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];

                float x0 = vec[d * 2];
                float x1 = vec[d * 2 + 1];

                vec[d * 2]     = x0 * cos_val - x1 * sin_val;
                vec[d * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}
