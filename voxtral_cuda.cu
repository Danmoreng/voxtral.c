#include "voxtral_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_cuda_initialized = 0;
static cublasHandle_t g_cublas_handle = NULL;
static float *g_scratch_f32 = NULL;
static size_t g_scratch_size = 0;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

int vox_cuda_available(void) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        return 0;
    }
    return deviceCount > 0;
}

void vox_cuda_init(void) {
    if (g_cuda_initialized) return;

    CUDA_CHECK(cudaSetDevice(0));
    
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        exit(1);
    }

    /* Set math mode to allow Tensor Cores (TF32) on Ampere+ */
    cublasSetMathMode(g_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    /* Pre-allocate a reasonable scratch buffer for BF16 dequant (e.g. 64MB) */
    g_scratch_size = 64 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&g_scratch_f32, g_scratch_size));

    g_cuda_initialized = 1;
    fprintf(stderr, "[CUDA] Initialized (Device 0)\n");
}

void vox_cuda_shutdown(void) {
    if (!g_cuda_initialized) return;
    if (g_scratch_f32) {
        cudaFree(g_scratch_f32);
        g_scratch_f32 = NULL;
        g_scratch_size = 0;
    }
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    g_cuda_initialized = 0;
}

void *vox_cuda_malloc_managed(size_t size) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return ptr;
}

void *vox_cuda_malloc(size_t size) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void vox_cuda_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

void vox_cuda_copy_to_device(void *dst, const void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void vox_cuda_copy_to_host(void *dst, const void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

/* ========================================================================
 * Kernels
 * ======================================================================== */

__global__ void k_rms_norm(float *out, const float *x, const float *weight, int hidden, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float val = x[row * hidden + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_inv = 0.0f;
    if (tid == 0) {
        rms_inv = rsqrtf(sdata[0] / hidden + eps);
        sdata[0] = rms_inv;
    }
    __syncthreads();
    rms_inv = sdata[0];

    for (int i = tid; i < hidden; i += blockDim.x) {
        out[row * hidden + i] = x[row * hidden + i] * rms_inv * weight[i];
    }
}

void vox_cuda_rms_norm(float *out, const float *x, const float *weight, int n, int hidden, float eps) {
    int threads = 256;
    while (threads > hidden && threads > 32) threads /= 2;
    size_t shared_mem = threads * sizeof(float);
    k_rms_norm<<<n, threads, shared_mem>>>(out, x, weight, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void vox_cuda_silu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_silu<<<blocks, threads>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void vox_cuda_gelu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_gelu<<<blocks, threads>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

void vox_cuda_add_inplace(float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_add_inplace<<<blocks, threads>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_mul_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= b[i];
    }
}

void vox_cuda_mul_inplace(float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_mul_inplace<<<blocks, threads>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

void vox_cuda_axpy(float *a, float scale, const float *b, int n) {
    float alpha = scale;
    cublasStatus_t status = cublasSaxpy(g_cublas_handle, n, &alpha, b, 1, a, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSaxpy failed: status=%d\n", status);
        exit(1);
    }
    cudaDeviceSynchronize();
}

__global__ void k_bias_add(float *y, const float *b, int seq_len, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * out_dim;
    if (i < total) {
        y[i] += b[i % out_dim];
    }
}

void vox_cuda_bias_add(float *y, const float *b, int seq_len, int out_dim) {
    int total = seq_len * out_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_bias_add<<<blocks, threads>>>(y, b, seq_len, out_dim);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_rope(float *x, const float *freqs, int seq, int heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = seq * heads * half_dim;
    if (i < total) {
        int s = i / (heads * half_dim);
        int h = (i / half_dim) % heads;
        int d = i % half_dim;

        float cos_val = freqs[s * half_dim * 2 + d * 2];
        float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];

        int base = s * heads * head_dim + h * head_dim + d * 2;
        float x0 = x[base];
        float x1 = x[base + 1];

        x[base] = x0 * cos_val - x1 * sin_val;
        x[base + 1] = x0 * sin_val + x1 * cos_val;
    }
}

void vox_cuda_rope(float *x, const float *freqs, int seq, int heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total = seq * heads * half_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_rope<<<blocks, threads>>>(x, freqs, seq, heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                                int channels_in, int channels_out, int length, int out_length,
                                int kernel_size, int stride) {
    int ol = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y;
    
    if (ol < out_length && co < channels_out) {
        int left_pad = kernel_size - stride;
        float sum = (bias) ? bias[co] : 0.0f;
        int K = channels_in * kernel_size;
        
        for (int ci = 0; ci < channels_in; ci++) {
            for (int k = 0; k < kernel_size; k++) {
                int il = ol * stride - left_pad + k;
                if (il >= 0 && il < length) {
                    sum += in[ci * length + il] * weight[co * K + ci * kernel_size + k];
                }
            }
        }
        out[co * out_length + ol] = sum;
    }
}

void vox_cuda_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                            int channels_in, int channels_out, int length, int out_length,
                            int kernel_size, int stride) {
    dim3 threads(256);
    dim3 blocks((out_length + 255) / 256, channels_out);
    k_causal_conv1d<<<blocks, threads>>>(out, in, weight, bias, channels_in, channels_out, length, out_length, kernel_size, stride);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_causal_attention_prefill(float *out, const float *Q, const float *K, const float *V,
                                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                           int head_dim, float scale, int window_size, int q_offset) {
    int h = blockIdx.x;
    int qi = blockIdx.y;
    int tid = threadIdx.x;
    
    if (h < n_heads && qi < seq_q && tid < head_dim) {
        int heads_per_kv = n_heads / n_kv_heads;
        int kv_h = h / heads_per_kv;
        int q_hidden = n_heads * head_dim;
        int kv_hidden = n_kv_heads * head_dim;
        
        const float *q_row = Q + qi * q_hidden + h * head_dim;
        float *o_row = out + qi * q_hidden + h * head_dim;
        
        int global_pos = q_offset + qi;
        int k_start = (window_size > 0 && global_pos - window_size + 1 > 0) ? global_pos - window_size + 1 : 0;
        int k_end = (global_pos + 1 < seq_k) ? global_pos + 1 : seq_k;
        
        float max_score = -1e30f;
        float sum_exp = 0.0f;
        float acc = 0.0f;
        
        for (int j = k_start; j < k_end; j++) {
            const float *k_row = K + (size_t)j * kv_hidden + kv_h * head_dim;
            
            /* Each thread computes its part of dot product, but we need full dot product for softmax.
               This is still slow because of serial dot product per thread. 
               Better: use shared memory for dot product reduction.
            */
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            
            const float *v_row = V + (size_t)j * kv_hidden + kv_h * head_dim;
            float old_max = max_score;
            if (score > max_score) {
                max_score = score;
                float correction = expf(old_max - max_score);
                sum_exp = sum_exp * correction + 1.0f;
                acc = acc * correction + v_row[tid];
            } else {
                float weight = expf(score - max_score);
                sum_exp += weight;
                acc += weight * v_row[tid];
            }
        }
        o_row[tid] = acc / (sum_exp + 1e-10f);
    }
}

__global__ void k_causal_attention_decode(float *out, const float *Q, const float *K, const float *V,
                                          int seq_k, int n_heads, int n_kv_heads,
                                          int head_dim, float scale, int window_size, int q_pos) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    
    if (h < n_heads && tid < head_dim) {
        int heads_per_kv = n_heads / n_kv_heads;
        int kv_h = h / heads_per_kv;
        int kv_hidden = n_kv_heads * head_dim;
        
        const float *q_row = Q + h * head_dim;
        float *o_row = out + h * head_dim;
        
        int k_start = (window_size > 0 && q_pos - window_size + 1 > 0) ? q_pos - window_size + 1 : 0;
        int k_end = q_pos + 1;
        if (k_end > seq_k) k_end = seq_k;
        
        float max_score = -1e30f;
        float sum_exp = 0.0f;
        float acc = 0.0f;
        
        for (int j = k_start; j < k_end; j++) {
            const float *k_row = K + (size_t)j * kv_hidden + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            
            const float *v_row = V + (size_t)j * kv_hidden + kv_h * head_dim;
            float old_max = max_score;
            if (score > max_score) {
                max_score = score;
                float correction = expf(old_max - max_score);
                sum_exp = sum_exp * correction + 1.0f;
                acc = acc * correction + v_row[tid];
            } else {
                float weight = expf(score - max_score);
                sum_exp += weight;
                acc += weight * v_row[tid];
            }
        }
        o_row[tid] = acc / (sum_exp + 1e-10f);
    }
}

void vox_cuda_causal_attention(float *out, const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int n_heads, int n_kv_heads,
                               int head_dim, float scale, int window_size, int q_offset) {
    if (seq_q == 1) {
        dim3 blocks(n_heads);
        dim3 threads(head_dim);
        k_causal_attention_decode<<<blocks, threads>>>(out, Q, K, V, seq_k, n_heads, n_kv_heads, head_dim, scale, window_size, q_offset);
    } else {
        dim3 blocks(n_heads, seq_q);
        dim3 threads(head_dim);
        k_causal_attention_prefill<<<blocks, threads>>>(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, window_size, q_offset);
    }
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_ada_scale(float *x, const float *scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= (1.0f + scale[i]);
    }
}

void vox_cuda_ada_scale(float *x, const float *scale, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_ada_scale<<<blocks, threads>>>(x, scale, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_transpose_mel(float *out, const float *in, int frames, int bins) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (f < frames && m < bins) {
        out[m * frames + f] = in[f * bins + m];
    }
}

void vox_cuda_transpose_mel(float *out, const float *in, int frames, int bins) {
    dim3 threads(256);
    dim3 blocks((frames + 255) / 256, bins);
    k_transpose_mel<<<blocks, threads>>>(out, in, frames, bins);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void k_transpose_conv(float *out, const float *in, int seq_len, int dim) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y;
    if (s < seq_len && d < dim) {
        out[s * dim + d] = in[d * seq_len + s];
    }
}

void vox_cuda_transpose_conv(float *out, const float *in, int seq_len, int dim) {
    dim3 threads(256);
    dim3 blocks((seq_len + 255) / 256, dim);
    k_transpose_conv<<<blocks, threads>>>(out, in, seq_len, dim);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
}

void vox_cuda_sgemm(int m, int n, int k, const float *a, const float *b, float *c) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    /* C = A * B -> C^T = B^T * A^T
       A, B, C are standard FP32 pointers (Managed)
    */
    
    cublasStatus_t status = cublasSgemm(g_cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        b, n,
                                        a, k,
                                        &beta,
                                        c, n);
                                        
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: status=%d\n", status);
        exit(1);
    }
    cudaDeviceSynchronize();
}

void vox_cuda_sgemm_t(int m, int n, int k, const float *a, const float *b, float *c) {
    /* C = A * B^T -> C^T = (A * B^T)^T = B * A^T
       A (MxK), B (NxK) [transposed logic], C (MxN)
    */
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasStatus_t status = cublasSgemm(g_cublas_handle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        b, k,
                                        a, k,
                                        &beta,
                                        c, n);
                                        
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm (T) failed: status=%d\n", status);
        exit(1);
    }
    cudaDeviceSynchronize();
}

__global__ void k_bf16_to_f32_conv(float *out, const unsigned short *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short val = in[i];
        unsigned int res = (unsigned int)val << 16;
        out[i] = *(float*)&res;
    }
}

void vox_cuda_matmul_t_bf16(int m, int n, int k, const float *a, const unsigned short *b_bf16, float *c) {
    /* Fallback implementation: Dequantize to temporary FP32 buffer on GPU, then SGEMM.
       Uses a persistent scratch buffer to avoid cudaMalloc overhead.
    */
    size_t required = (size_t)n * k * sizeof(float);
    if (required > g_scratch_size) {
        if (g_scratch_f32) cudaFree(g_scratch_f32);
        g_scratch_size = required * 2; /* Grow with buffer room */
        CUDA_CHECK(cudaMalloc(&g_scratch_f32, g_scratch_size));
    }
    
    float *d_b_f32 = g_scratch_f32;
    
    /* Dequantize on GPU */
    int total_elems = n * k;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    k_bf16_to_f32_conv<<<blocks, threads>>>(d_b_f32, b_bf16, total_elems);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasStatus_t status = cublasSgemm(g_cublas_handle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        d_b_f32, k,
                                        a, k,
                                        &beta,
                                        c, n);
                                        
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm (BF16 fallback) failed: status=%d\n", status);
        exit(1);
    }
    /* Sync before returning to Host so following CPU logic sees results */
    cudaDeviceSynchronize();
}

