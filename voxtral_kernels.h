/*
 * voxtral_kernels.h - Math kernels for Voxtral inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 */

#ifndef VOXTRAL_KERNELS_H
#define VOXTRAL_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include "voxtral_cuda.h"

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void vox_add_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_mul_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_axpy(vox_cuda_ctx_t *ctx, float *a, float scale, const float *b, int n);
void vox_scale(float *x, float s, int n);
void vox_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void vox_matmul(vox_cuda_ctx_t *ctx, float *C, const float *A, const float *B, int M, int K, int N);
void vox_matmul_t(vox_cuda_ctx_t *ctx, float *C, const float *A, const float *B, int M, int K, int N);

void vox_linear(vox_cuda_ctx_t *ctx, float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim);

void vox_linear_nobias(vox_cuda_ctx_t *ctx, float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim);

void vox_linear_nobias_bf16(vox_cuda_ctx_t *ctx, float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim);

void vox_linear_bf16(vox_cuda_ctx_t *ctx, float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim);

void vox_matmul_t_bf16(vox_cuda_ctx_t *ctx, float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N);

/* ========================================================================
 * 1D Convolution
 * ======================================================================== */

void vox_conv1d(vox_cuda_ctx_t *ctx, float *out, const float *in, const float *weight, const float *bias,
                int channels_in, int channels_out, int length,
                int kernel_size, int stride, int padding);

void vox_causal_conv1d(vox_cuda_ctx_t *ctx, float *out, const float *in, const float *weight, const float *bias,
                       int channels_in, int channels_out, int length,
                       int kernel_size, int stride);

/* ========================================================================
 * Normalization
 * ======================================================================== */

void vox_rms_norm(vox_cuda_ctx_t *ctx, float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void vox_silu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_gelu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_softmax(float *x, int rows, int cols);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

void vox_causal_attention(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int window_size,
                          int q_offset);

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

void vox_compute_rope_freqs(vox_cuda_ctx_t *ctx, float *freqs, const int *pos, int seq, int dim, float theta);
void vox_apply_rope(vox_cuda_ctx_t *ctx, float *x, const float *freqs, int seq, int heads, int head_dim);

/* Global verbose flag */
extern int vox_verbose;

#endif /* VOXTRAL_KERNELS_H */