#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA Context for thread safety */
typedef struct vox_cuda_ctx vox_cuda_ctx_t;

/* Check if CUDA is available and supported on this system */
int vox_cuda_available(void);

/* Initialize CUDA context and resources */
vox_cuda_ctx_t *vox_cuda_init(void);

/* Shutdown CUDA and free resources */
void vox_cuda_shutdown(vox_cuda_ctx_t *ctx);

/* Managed Memory (Unified) */
void *vox_cuda_malloc_managed(size_t size);

/* Allocate from a memory pool (recommended for activations/KV) */
void *vox_cuda_malloc(vox_cuda_ctx_t *ctx, size_t size);
void vox_cuda_free(vox_cuda_ctx_t *ctx, void *ptr);

/* Memory pool management */
void vox_cuda_pool_trim(vox_cuda_ctx_t *ctx);

/* Graph capture */
void vox_cuda_graph_begin(vox_cuda_ctx_t *ctx);
void *vox_cuda_graph_end(vox_cuda_ctx_t *ctx);
void vox_cuda_graph_exec(void *exec);
void vox_cuda_graph_destroy(void *exec);

/* KV Cache update kernel for Graphs */
void vox_cuda_kv_cache_update(vox_cuda_ctx_t *ctx, float *cache_k, float *cache_v, const float *k, const float *v, 
                              int layer, const int *pos_ptr, int max_seq, int kv_dim);

/* Memory copy helpers */
void vox_cuda_copy_to_device(void *dst, const void *src, size_t size);
void vox_cuda_copy_to_host(void *dst, const void *src, size_t size);

/* Argmax */
void vox_cuda_argmax(vox_cuda_ctx_t *ctx, int *out_gpu, const float *logits_gpu, int n);

/* Math kernels */
void vox_cuda_rms_norm(vox_cuda_ctx_t *ctx, float *out, const float *x, const float *weight, int n, int hidden, float eps);
void vox_cuda_rms_norm_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, int n, int hidden, float eps);
void vox_cuda_rms_norm_ada_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, const float *ada_scale, int n, int hidden, float eps);

void vox_cuda_silu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_cuda_ffn_swiglu(vox_cuda_ctx_t *ctx, float *out, const float *gate, const float *up, int n);
void vox_cuda_gelu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_cuda_add_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_cuda_mul_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_cuda_axpy(vox_cuda_ctx_t *ctx, float *a, float scale, const float *b, int n);
void vox_cuda_bias_add(vox_cuda_ctx_t *ctx, float *y, const float *b, int seq_len, int out_dim);
void vox_cuda_rope(vox_cuda_ctx_t *ctx, float *x, const float *freqs, int seq, int heads, int head_dim);
void vox_cuda_compute_rope_freqs(vox_cuda_ctx_t *ctx, float *freqs, const int *pos, int seq, int dim, float theta);
void vox_cuda_causal_conv1d(vox_cuda_ctx_t *ctx, float *out, const float *in, const float *weight, const float *bias,
                            int channels_in, int channels_out, int length, int out_length,
                            int kernel_size, int stride);
void vox_cuda_causal_attention(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int n_heads, int n_kv_heads,
                               int head_dim, float scale, int window_size, int q_offset);
void vox_cuda_causal_attention_ptr(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, const int *seq_k_ptr, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int window_size, const int *q_offset_ptr);
void vox_cuda_ada_scale(vox_cuda_ctx_t *ctx, float *x, const float *scale, int n);
void vox_cuda_transpose_mel(vox_cuda_ctx_t *ctx, float *out, const float *in, int frames, int bins);
void vox_cuda_transpose_conv(vox_cuda_ctx_t *ctx, float *out, const float *in, int seq_len, int dim);

/* Matrix multiplication: C = alpha * A * B + beta * C
   - sgemm: FP32 * FP32 -> FP32
   - matmul_bf16: BF16 (A) * BF16 (B) -> FP32 (C)
*/
void vox_cuda_sgemm(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_sgemm_t(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_matmul_bf16(vox_cuda_ctx_t *ctx, int m, int n, int k, const void *a, const void *b, float *c, int transpose_b);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_CUDA_H */
