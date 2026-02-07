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

