#include "voxtral_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_cuda_initialized = 0;
static cublasHandle_t g_cublas_handle = NULL;

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

    g_cuda_initialized = 1;
    fprintf(stderr, "[CUDA] Initialized (Device 0)\n");
}

void vox_cuda_shutdown(void) {
    if (!g_cuda_initialized) return;
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    g_cuda_initialized = 0;
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
    
    /* Simple block reduction for RMS */
    extern __shared__ float sdata[];
    
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float val = x[row * hidden + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    /* Reduction in shared memory */
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
}

/* C[m,n] = A[m,k] * B[k,n] */
/* Note: cuBLAS is column-major by default.
   To compute C (row-major) = A (row-major) * B (row-major):
   We want C^T = B^T * A^T.
   cuBLAS sees A as k*m (col-major) and B as n*k (col-major).
   So we compute C^T = B * A using cublasSgemm.
   
   cublasSgemm(handle, OP_N, OP_N, n, m, k, alpha, B, n, A, k, beta, C, n)
*/
void vox_cuda_sgemm(int m, int n, int k, const float *a, const float *b, float *c) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    /* A[m][k] row-major -> viewed as A^T[k][m] col-major
       B[k][n] row-major -> viewed as B^T[n][k] col-major
       C[m][n] row-major -> viewed as C^T[n][m] col-major
       
       We want C = A * B.
       In column-major land: C^T = B^T * A^T.
       
       So we effectively pass:
       LDA = k (leading dimension of A^T is k)
       LDB = n (leading dimension of B^T is n)
       LDC = n (leading dimension of C^T is n)
    */
    
    cublasStatus_t status = cublasSgemm(g_cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        b, n,  /* B^T is n x k, leading dim n */
                                        a, k,  /* A^T is k x m, leading dim k */
                                        &beta,
                                        c, n); /* C^T is n x m, leading dim n */
                                        
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed\n");
        exit(1);
    }
}
