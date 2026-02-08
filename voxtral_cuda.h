#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "voxtral.h"

int vox_cuda_available(void);
vox_cuda_ctx_t *vox_cuda_init(void);
int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N);
int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);
int vox_cuda_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int K, int N);

/* Convenience wrapper for y = x @ W^T (+bias), where W is BF16 [out_dim, in_dim]. */
int vox_cuda_linear_bf16(float *y, const float *x, const uint16_t *W_bf16, const float *b,
                         int seq_len, int in_dim, int out_dim);

/* Specialized helper for decoder hot path: compute two BF16 linear projections
 * with the same input and dimensions while sharing the input upload and stream
 * sync point.
 *
 * y0 = x[1,in_dim] @ W0^T[out_dim,in_dim]
 * y1 = x[1,in_dim] @ W1^T[out_dim,in_dim]
 *
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_linear2_bf16(float *y0, float *y1,
                          const float *x,
                          const uint16_t *W0_bf16,
                          const uint16_t *W1_bf16,
                          int in_dim,
                          int out_dim);

/* Decoder attention acceleration (seq_q=1 path).
 * - Appends this layer's K/V at `pos` into a device-side cache.
 * - Computes attention for Q against cached K/V for this layer and returns
 *   attn_out (float32, shape [VOX_DEC_HEADS*VOX_DEC_HEAD_DIM]).
 *
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_attention_step(float *attn_out,
                            const float *q,
                            const float *k,
                            const float *v,
                            int layer,
                            int pos,
                            int total_seq,
                            int window_size);

/* Keep CUDA-side KV cache in sync with CPU KV cache compactions/resets. */
void vox_cuda_kv_cache_compact(int discard, int keep, int kv_dim, int max_seq);
void vox_cuda_kv_cache_reset(void);
void vox_cuda_kv_cache_append_block(int layer, int start_pos, int seq_len,
                                    int kv_dim, int window_size,
                                    const float *k, const float *v);

/* Generic causal attention on GPU (float32 Q,K,V).
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_causal_attention(float *out,
                              const float *Q,
                              const float *K,
                              const float *V,
                              int seq_q,
                              int seq_k,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              float scale,
                              int window_size,
                              int q_offset);

/* CUDA fast paths that keep encoder/decoder intermediates on-device. */
int vox_cuda_encode_adapter(float **out, int *out_tokens,
                            vox_ctx_t *ctx,
                            const float *mel,
                            int mel_frames,
                            int overlap_mel);

/* Optional: full CUDA streaming pipeline (encoder+adapter outputs kept on
 * device, decoder consumes adapter embeddings directly). Opt-in via
 * VOX_CUDA_PIPELINE_FULL=1. */
void vox_cuda_stream_adapter_reset(void);

/* Copy the first `n_tokens` adapter embeddings from the device-side adapter
 * buffer into `out_host` (float32, shape [n_tokens, VOX_DEC_DIM]). Used to
 * build the initial decoder prompt on CPU without copying the full adapter. */
int vox_cuda_stream_adapter_copy_prompt(float *out_host, int n_tokens);

/* Run CUDA full encoder+adapter and append the resulting adapter embeddings
 * to the internal device-side adapter buffer. Returns 1 on success. */
int vox_cuda_encode_adapter_stream_append(int *out_tokens,
                                          vox_ctx_t *ctx,
                                          const float *mel,
                                          int mel_frames,
                                          int overlap_mel);

/* Decoder single-token step that pulls the current adapter embedding from the
 * device-side adapter buffer (using logical position = kv_pos_offset+kv_cache_len).
 * prev_token is the previous generated token id used for the input embedding.
 * Returns 1 on success, 0 on fallback. */
int vox_cuda_decoder_forward_from_stream_adapter(int *out_token,
                                                 float *logits_or_null,
                                                 vox_ctx_t *ctx,
                                                 int prev_token);

int vox_cuda_decoder_forward_full(int *out_token,
                                  float *logits_or_null,
                                  vox_ctx_t *ctx,
                                  const float *input_embeds);

const char *vox_cuda_device_name(void);
void vox_cuda_shutdown(void);

/* Low-level CUDA API access */
void *vox_cuda_malloc(vox_cuda_ctx_t *ctx, size_t size);
void vox_cuda_free(vox_cuda_ctx_t *ctx, void *ptr);
void vox_cuda_copy_to_device(void *dst, const void *src, size_t size);
void vox_cuda_copy_to_host(void *dst, const void *src, size_t size);

/* Math Kernels (GPU implementations) */
void vox_cuda_add_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_cuda_mul_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n);
void vox_cuda_axpy(vox_cuda_ctx_t *ctx, float *a, float scale, const float *b, int n);
void vox_cuda_sgemm(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_sgemm_t(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c);
void vox_cuda_bias_add(vox_cuda_ctx_t *ctx, float *y, const float *b, int seq_len, int out_dim);
void vox_cuda_matmul_bf16(vox_cuda_ctx_t *ctx, int m, int n, int k, const void *a, const void *b, float *c, int transpose_b);
void vox_cuda_causal_conv1d(vox_cuda_ctx_t *ctx, float *out, const float *in, const float *weight, const float *bias,
                            int channels_in, int channels_out, int length, int out_length,
                            int kernel_size, int stride);
void vox_cuda_rms_norm(vox_cuda_ctx_t *ctx, float *out, const float *x, const float *weight, int n, int hidden, float eps);
void vox_cuda_silu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_cuda_gelu(vox_cuda_ctx_t *ctx, float *x, int n);
void vox_cuda_compute_rope_freqs(vox_cuda_ctx_t *ctx, float *freqs, const int *pos, int seq, int dim, float theta);
void vox_cuda_rope(vox_cuda_ctx_t *ctx, float *x, const float *freqs, int seq, int heads, int head_dim);
void vox_cuda_transpose_mel(vox_cuda_ctx_t *ctx, float *out, const float *in, int frames, int bins);
void vox_cuda_transpose_conv(vox_cuda_ctx_t *ctx, float *out, const float *in, int seq_len, int dim);
void vox_cuda_rms_norm_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, int n, int hidden, float eps);
void vox_cuda_ffn_swiglu(vox_cuda_ctx_t *ctx, float *out, const float *gate, const float *up, int n);
void vox_cuda_rms_norm_ada_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, const float *ada_scale, int n, int hidden, float eps);
void vox_cuda_kv_cache_update(vox_cuda_ctx_t *ctx, float *cache_k, float *cache_v, const float *k, const float *v, 
                              int layer, const int *pos_ptr, int max_seq, int kv_dim);
void vox_cuda_causal_attention_ptr(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, const int *seq_k_ptr, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int window_size, const int *q_offset_ptr);
void vox_cuda_argmax(vox_cuda_ctx_t *ctx, int *out_gpu, const float *logits_gpu, int n);

/* Graph Capture */
void vox_cuda_graph_begin(vox_cuda_ctx_t *ctx);
void *vox_cuda_graph_end(vox_cuda_ctx_t *ctx);
void vox_cuda_graph_exec(void *exec);
void vox_cuda_graph_destroy(void *exec);

#endif
