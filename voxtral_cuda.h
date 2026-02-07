#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Check if CUDA is available and supported on this system */
int vox_cuda_available(void);

/* Initialize CUDA context and resources */
void vox_cuda_init(void);

/* Shutdown CUDA and free resources */
void vox_cuda_shutdown(void);

/* Managed Memory (Unified) */
void *vox_cuda_malloc_managed(size_t size);

/* Allocate shared memory (accessible by both Host and Device) if unified memory is used, 
   or just device memory. For now, mirroring metal's interface might change. */
void *vox_cuda_malloc(size_t size);
void vox_cuda_free(void *ptr);

/* Memory copy helpers */
void vox_cuda_copy_to_device(void *dst, const void *src, size_t size);
void vox_cuda_copy_to_host(void *dst, const void *src, size_t size);

/* Math kernels */
void vox_cuda_rms_norm(float *out, const float *x, const float *weight, int n, int hidden, float eps);
void vox_cuda_silu(float *x, int n);
void vox_cuda_gelu(float *x, int n);
void vox_cuda_add_inplace(float *a, const float *b, int n);
void vox_cuda_mul_inplace(float *a, const float *b, int n);
void vox_cuda_axpy(float *a, float scale, const float *b, int n);
void vox_cuda_bias_add(float *y, const float *b, int seq_len, int out_dim);
void vox_cuda_rope(float *x, const float *freqs, int seq, int heads, int head_dim);
void vox_cuda_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                            int channels_in, int channels_out, int length, int out_length,
                            int kernel_size, int stride);
void vox_cuda_causal_attention(float *out, const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int n_heads, int n_kv_heads,
                               int head_dim, float scale, int window_size, int q_offset);
void vox_cuda_ada_scale(float *x, const float *scale, int n);
void vox_cuda_transpose_mel(float *out, const float *in, int frames, int bins);
void vox_cuda_transpose_conv(float *out, const float *in, int seq_len, int dim);

/* Matrix multiplication using cuBLAS: C = alpha * A * B + beta * C */
void vox_cuda_sgemm(int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_sgemm_t(int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_matmul_t_bf16(int m, int n, int k, const float *a, const unsigned short *b_bf16, float *c);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_CUDA_H */
