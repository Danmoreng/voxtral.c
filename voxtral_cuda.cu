#include "voxtral_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct vox_cuda_ctx {
    int device;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaMemPool_t mem_pool;
    void *lt_workspace;
    size_t lt_workspace_size;
    
    // Scratch buffer for activations conversion
    void *act_scratch;
    size_t act_scratch_size;

    // Graph state
    cudaGraph_t current_graph;
    cudaGraphExec_t current_exec;
    bool capturing;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)status); \
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

vox_cuda_ctx_t *vox_cuda_init(void) {
    vox_cuda_ctx_t *ctx = (vox_cuda_ctx_t *)calloc(1, sizeof(vox_cuda_ctx_t));
    if (!ctx) return NULL;

    ctx->device = 0;
    CUDA_CHECK(cudaSetDevice(ctx->device));
    
    CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
    CUBLAS_CHECK(cublasLtCreate(&ctx->cublaslt_handle));

    /* Set math mode to allow Tensor Cores (TF32) on Ampere+ */
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    /* Workspace for cublasLt */
    ctx->lt_workspace_size = 32 * 1024 * 1024; // 32MB workspace
    CUDA_CHECK(cudaMalloc(&ctx->lt_workspace, ctx->lt_workspace_size));

    /* Scratch for BF16 conversion (max 32k * 4096 * 2 bytes = 256MB) */
    ctx->act_scratch_size = 256 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&ctx->act_scratch, ctx->act_scratch_size));

    /* Setup Memory Pool if supported (CUDA 11.2+) */
    int memPoolSupported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memPoolSupported, cudaDevAttrMemoryPoolsSupported, ctx->device));
    if (memPoolSupported) {
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&ctx->mem_pool, ctx->device));
        uint64_t setVal = UINT64_MAX; // No limit
        CUDA_CHECK(cudaMemPoolSetAttribute(ctx->mem_pool, cudaMemPoolAttrReleaseThreshold, &setVal));
    }

    fprintf(stderr, "[CUDA] Context Initialized (Device %d, Pool: %s)\n", ctx->device, memPoolSupported ? "Yes" : "No");
    return ctx;
}

void vox_cuda_shutdown(vox_cuda_ctx_t *ctx) {
    if (!ctx) return;
    
    if (ctx->current_exec) cudaGraphExecDestroy(ctx->current_exec);
    if (ctx->current_graph) cudaGraphDestroy(ctx->current_graph);

    if (ctx->act_scratch) cudaFree(ctx->act_scratch);
    if (ctx->lt_workspace) cudaFree(ctx->lt_workspace);
    if (ctx->cublaslt_handle) cublasLtDestroy(ctx->cublaslt_handle);
    if (ctx->cublas_handle) cublasDestroy(ctx->cublas_handle);
    
    free(ctx);
}

void *vox_cuda_malloc_managed(size_t size) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return ptr;
}

void *vox_cuda_malloc(vox_cuda_ctx_t *ctx, size_t size) {
    void *ptr = NULL;
    if (ctx && ctx->mem_pool) {
        CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, size, ctx->mem_pool, NULL));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }
    return ptr;
}

void vox_cuda_free(vox_cuda_ctx_t *ctx, void *ptr) {
    if (!ptr) return;
    if (ctx && ctx->mem_pool) {
        CUDA_CHECK(cudaFreeAsync(ptr, NULL));
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void vox_cuda_pool_trim(vox_cuda_ctx_t *ctx) {
    if (ctx && ctx->mem_pool) {
        CUDA_CHECK(cudaMemPoolTrimTo(ctx->mem_pool, 0));
    }
}

void vox_cuda_graph_begin(vox_cuda_ctx_t *ctx) {
    if (!ctx || ctx->capturing) return;
    CUDA_CHECK(cudaStreamBeginCapture(NULL, cudaStreamCaptureModeGlobal));
    ctx->capturing = true;
}

void *vox_cuda_graph_end(vox_cuda_ctx_t *ctx) {
    if (!ctx || !ctx->capturing) return NULL;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t exec = NULL;
    CUDA_CHECK(cudaStreamEndCapture(NULL, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    ctx->capturing = false;
    return (void *)exec;
}

void vox_cuda_graph_exec(void *exec) {
    if (exec) {
        CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)exec, NULL));
    }
}

void vox_cuda_graph_destroy(void *exec) {
    if (exec) {
        CUDA_CHECK(cudaGraphExecDestroy((cudaGraphExec_t)exec));
    }
}

__device__ __forceinline__ float warp_reduce_max(float val, int &idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) { val = other_val; idx = other_idx; }
    }
    return val;
}

__global__ void k_argmax(int *out, const float *logits, int n) {
    int tid = threadIdx.x;
    float max_val = -1e30f;
    int max_idx = -1;
    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
    }
    max_val = warp_reduce_max(max_val, max_idx);
    static __shared__ float s_max_val[32];
    static __shared__ int s_max_idx[32];
    int lane = tid % 32; int wid = tid / 32;
    if (lane == 0) { s_max_val[wid] = max_val; s_max_idx[wid] = max_idx; }
    __syncthreads();
    if (wid == 0) {
        max_val = (tid < (blockDim.x / 32)) ? s_max_val[lane] : -1e30f;
        max_idx = (tid < (blockDim.x / 32)) ? s_max_idx[lane] : -1;
        max_val = warp_reduce_max(max_val, max_idx);
        if (tid == 0) *out = max_idx;
    }
}

void vox_cuda_argmax(vox_cuda_ctx_t *ctx, int *out_gpu, const float *logits_gpu, int n) {
    k_argmax<<<1, 256>>>(out_gpu, logits_gpu, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_kv_cache_update(float *cache_k, float *cache_v, const float *k, const float *v, 
                                  int layer, const int *pos_ptr, int max_seq, int kv_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < kv_dim) {
        int pos = *pos_ptr;
        size_t offset = ((size_t)layer * max_seq + pos) * kv_dim + tid;
        cache_k[offset] = k[tid];
        cache_v[offset] = v[tid];
    }
}

void vox_cuda_kv_cache_update(vox_cuda_ctx_t *ctx, float *cache_k, float *cache_v, const float *k, const float *v, 
                              int layer, const int *pos_ptr, int max_seq, int kv_dim) {
    int threads = 256;
    int blocks = (kv_dim + threads - 1) / threads;
    k_kv_cache_update<<<blocks, threads>>>(cache_k, cache_v, k, v, layer, pos_ptr, max_seq, kv_dim);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

void vox_cuda_copy_to_device(void *dst, const void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void vox_cuda_copy_to_host(void *dst, const void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

__global__ void k_f32_to_bf16(nv_bfloat16 *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

void vox_cuda_matmul_bf16(vox_cuda_ctx_t *ctx, int m, int n, int k, const void *a, const void *b, float *c, int transpose_b) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t adesc = NULL, bdesc = NULL, cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    /* A is activations [m, k], B is weights [n, k] (row-major)
       We want C = A * B^T -> [m, n]
       In column-major: B is [k, n], A is [k, m]
       C = op(B) * op(A) -> [n, m]
       With op(B)=T (-> [n, k]), op(A)=N (-> [k, m]) -> [n, m] col-major, which is [m, n] row-major.
    */

    /* Convert activations 'a' from f32 to bf16 if needed */
    const void *a_bf16 = a;
    int total_a = m * k;
    if (total_a > 0) {
        int threads = 256;
        int blocks = (total_a + threads - 1) / threads;
        k_f32_to_bf16<<<blocks, threads>>>( (nv_bfloat16*)ctx->act_scratch, (const float*)a, total_a);
        a_bf16 = ctx->act_scratch;
    }

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    cublasOperation_t opA = CUBLAS_OP_T; 
    cublasOperation_t opB = CUBLAS_OP_N; 
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    /* Layouts for column-major sgemm */
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, n, m, n));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ctx->lt_workspace_size, sizeof(ctx->lt_workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ctx->cublaslt_handle, operationDesc, adesc, bdesc, cdesc, cdesc, preference, 1, &heuristicResult, &returnedResults));

    CUBLAS_CHECK(cublasLtMatmul(ctx->cublaslt_handle, operationDesc, &alpha, b, adesc, a_bf16, bdesc, &beta, c, cdesc, c, cdesc, &heuristicResult.algo, ctx->lt_workspace, ctx->lt_workspace_size, NULL));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    
    if (!ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

/* Kernels implementation using ctx */

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
        if (tid < s) sdata[tid] += sdata[tid + s];
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

void vox_cuda_rms_norm(vox_cuda_ctx_t *ctx, float *out, const float *x, const float *weight, int n, int hidden, float eps) {
    int threads = 256;
    while (threads > hidden && threads > 32) threads /= 2;
    size_t shared_mem = threads * sizeof(float);
    k_rms_norm<<<n, threads, shared_mem>>>(out, x, weight, hidden, eps);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float val = x[i]; x[i] = val / (1.0f + expf(-val)); }
}

void vox_cuda_silu(vox_cuda_ctx_t *ctx, float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_silu<<<blocks, threads>>>(x, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rms_norm_residual(float *out, float *x, const float *residual, const float *weight, int hidden, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        int idx = row * hidden + i;
        float val = x[idx] + (residual ? residual[idx] : 0.0f);
        x[idx] = val;
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms_inv = 0.0f;
    if (tid == 0) { rms_inv = rsqrtf(sdata[0] / hidden + eps); sdata[0] = rms_inv; }
    __syncthreads();
    rms_inv = sdata[0];
    for (int i = tid; i < hidden; i += blockDim.x) {
        int idx = row * hidden + i;
        out[idx] = x[idx] * rms_inv * weight[i];
    }
}

void vox_cuda_rms_norm_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, int n, int hidden, float eps) {
    int threads = 256;
    while (threads > hidden && threads > 32) threads /= 2;
    size_t shared_mem = threads * sizeof(float);
    k_rms_norm_residual<<<n, threads, shared_mem>>>(out, x, residual, weight, hidden, eps);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rms_norm_ada_residual(float *out, float *x, const float *residual, const float *weight, const float *ada_scale, int hidden, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        int idx = row * hidden + i;
        float val = x[idx] + (residual ? residual[idx] : 0.0f);
        x[idx] = val;
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms_inv = 0.0f;
    if (tid == 0) { rms_inv = rsqrtf(sdata[0] / hidden + eps); sdata[0] = rms_inv; }
    __syncthreads();
    rms_inv = sdata[0];
    for (int i = tid; i < hidden; i += blockDim.x) {
        int idx = row * hidden + i;
        out[idx] = x[idx] * rms_inv * weight[i] * (1.0f + ada_scale[i]);
    }
}

void vox_cuda_rms_norm_ada_residual(vox_cuda_ctx_t *ctx, float *out, float *x, const float *residual, const float *weight, const float *ada_scale, int n, int hidden, float eps) {
    int threads = 256;
    while (threads > hidden && threads > 32) threads /= 2;
    size_t shared_mem = threads * sizeof(float);
    k_rms_norm_ada_residual<<<n, threads, shared_mem>>>(out, x, residual, weight, ada_scale, hidden, eps);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_ffn_swiglu(float *out, const float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; float u = up[i]; out[i] = (g / (1.0f + expf(-g))) * u; }
}

void vox_cuda_ffn_swiglu(vox_cuda_ctx_t *ctx, float *out, const float *gate, const float *up, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_ffn_swiglu<<<blocks, threads>>>(out, gate, up, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float val = x[i]; float x3 = val * val * val; x[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * x3))); }
}

void vox_cuda_gelu(vox_cuda_ctx_t *ctx, float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_gelu<<<blocks, threads>>>(x, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

void vox_cuda_add_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_add_inplace<<<blocks, threads>>>(a, b, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_mul_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

void vox_cuda_mul_inplace(vox_cuda_ctx_t *ctx, float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_mul_inplace<<<blocks, threads>>>(a, b, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

void vox_cuda_axpy(vox_cuda_ctx_t *ctx, float *a, float scale, const float *b, int n) {
    CUBLAS_CHECK(cublasSaxpy(ctx->cublas_handle, n, &scale, b, 1, a, 1));
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_bias_add(float *y, const float *b, int seq_len, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len * out_dim) y[i] += b[i % out_dim];
}

void vox_cuda_bias_add(vox_cuda_ctx_t *ctx, float *y, const float *b, int seq_len, int out_dim) {
    int total = seq_len * out_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_bias_add<<<blocks, threads>>>(y, b, seq_len, out_dim);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rope(float *x, const float *freqs, int seq, int heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    if (i < seq * heads * half_dim) {
        int s = i / (heads * half_dim); int h = (i / half_dim) % heads; int d = i % half_dim;
        float cos_val = freqs[s * half_dim * 2 + d * 2]; float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];
        int base = s * heads * head_dim + h * head_dim + d * 2;
        float x0 = x[base]; float x1 = x[base + 1];
        x[base] = x0 * cos_val - x1 * sin_val; x[base + 1] = x0 * sin_val + x1 * cos_val;
    }
}

void vox_cuda_rope(vox_cuda_ctx_t *ctx, float *x, const float *freqs, int seq, int heads, int head_dim) {
    int total = seq * heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_rope<<<blocks, threads>>>(x, freqs, seq, heads, head_dim);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (i < seq * half_dim) {
        int s = i / half_dim;
        int d = i % half_dim;
        float p = (float)pos[s];
        float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
        float angle = p * freq;
        freqs[s * half_dim * 2 + d * 2] = cosf(angle);
        freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
    }
}

void vox_cuda_compute_rope_freqs(vox_cuda_ctx_t *ctx, float *freqs, const int *pos, int seq, int dim, float theta) {
    int total = seq * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_compute_rope_freqs<<<blocks, threads>>>(freqs, pos, seq, dim, theta);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
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
                if (il >= 0 && il < length) sum += in[ci * length + il] * weight[co * K + ci * kernel_size + k];
            }
        }
        out[co * out_length + ol] = sum;
    }
}

void vox_cuda_causal_conv1d(vox_cuda_ctx_t *ctx, float *out, const float *in, const float *weight, const float *bias,
                            int channels_in, int channels_out, int length, int out_length,
                            int kernel_size, int stride) {
    dim3 threads(256);
    dim3 blocks((out_length + 255) / 256, channels_out);
    k_causal_conv1d<<<blocks, threads>>>(out, in, weight, bias, channels_in, channels_out, length, out_length, kernel_size, stride);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; int lane = threadIdx.x % 32; int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val); if (lane == 0) shared[wid] = val; __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0; if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void k_causal_attention_decode_opt(float *out, const float *Q, const float *K, const float *V,
                                              int seq_k, int n_heads, int n_kv_heads,
                                              int head_dim, float scale, int window_size, int q_pos) {
    int h = blockIdx.x; int tid = threadIdx.x;
    int heads_per_kv = n_heads / n_kv_heads; int kv_h = h / heads_per_kv; int kv_hidden = n_kv_heads * head_dim;
    const float *q_row = Q + h * head_dim; float *o_row = out + h * head_dim;
    int k_start = (window_size > 0 && q_pos - window_size + 1 > 0) ? q_pos - window_size + 1 : 0;
    int k_end = q_pos + 1; if (k_end > seq_k) k_end = seq_k;
    __shared__ float shared_score;
    float max_score = -1e30f; float sum_exp = 0.0f; float acc_v = 0.0f;
    for (int j = k_start; j < k_end; j++) {
        const float *k_ptr = K + (size_t)j * kv_hidden + kv_h * head_dim;
        float local_dot = 0.0f; for (int d = tid; d < head_dim; d += blockDim.x) local_dot += q_row[d] * k_ptr[d];
        float score = block_reduce_sum(local_dot);
        if (tid == 0) shared_score = score * scale; __syncthreads();
        score = shared_score;
        const float *v_ptr = V + (size_t)j * kv_hidden + kv_h * head_dim;
        float old_max = max_score;
        if (score > max_score) {
            max_score = score; float correction = expf(old_max - max_score);
            sum_exp = sum_exp * correction + 1.0f; acc_v = acc_v * correction + (tid < head_dim ? v_ptr[tid] : 0.0f);
        } else {
            float weight = expf(score - max_score); sum_exp += weight; acc_v += weight * (tid < head_dim ? v_ptr[tid] : 0.0f);
        }
    }
    if (tid < head_dim) o_row[tid] = acc_v / (sum_exp + 1e-10f);
}

__global__ void k_causal_attention_prefill_opt(float *out, const float *Q, const float *K, const float *V,
                                               int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                               int head_dim, float scale, int window_size, int q_offset) {
    int h = blockIdx.x; int qi = blockIdx.y; int tid = threadIdx.x;
    int heads_per_kv = n_heads / n_kv_heads; int kv_h = h / heads_per_kv; int q_hidden = n_heads * head_dim; int kv_hidden = n_kv_heads * head_dim;
    const float *q_row = Q + qi * q_hidden + h * head_dim; float *o_row = out + qi * q_hidden + h * head_dim;
    int global_pos = q_offset + qi;
    int k_start = (window_size > 0 && global_pos - window_size + 1 > 0) ? global_pos - window_size + 1 : 0;
    int k_end = (global_pos + 1 < seq_k) ? global_pos + 1 : seq_k;
    __shared__ float shared_score;
    float max_score = -1e30f; float sum_exp = 0.0f; float acc_v = 0.0f;
    for (int j = k_start; j < k_end; j++) {
        const float *k_ptr = K + (size_t)j * kv_hidden + kv_h * head_dim;
        float local_dot = 0.0f; for (int d = tid; d < head_dim; d += blockDim.x) local_dot += q_row[d] * k_ptr[d];
        float score = block_reduce_sum(local_dot);
        if (tid == 0) shared_score = score * scale; __syncthreads();
        score = shared_score;
        const float *v_ptr = V + (size_t)j * kv_hidden + kv_h * head_dim;
        float old_max = max_score;
        if (score > max_score) {
            max_score = score; float correction = expf(old_max - max_score);
            sum_exp = sum_exp * correction + 1.0f; acc_v = acc_v * correction + (tid < head_dim ? v_ptr[tid] : 0.0f);
        } else {
            float weight = expf(score - max_score); sum_exp += weight; acc_v += weight * (tid < head_dim ? v_ptr[tid] : 0.0f);
        }
    }
    if (tid < head_dim) o_row[tid] = acc_v / (sum_exp + 1e-10f);
}

__global__ void k_causal_attention_decode_opt_ptr(float *out, const float *Q, const float *K, const float *V,
                                                  const int *seq_k_ptr, int n_heads, int n_kv_heads,
                                                  int head_dim, float scale, int window_size, const int *q_pos_ptr) {
    int h = blockIdx.x; int tid = threadIdx.x;
    int heads_per_kv = n_heads / n_kv_heads; int kv_h = h / heads_per_kv; int kv_hidden = n_kv_heads * head_dim;
    const float *q_row = Q + h * head_dim; float *o_row = out + h * head_dim;
    int seq_k = *seq_k_ptr; int q_pos = *q_pos_ptr;
    int k_start = (window_size > 0 && q_pos - window_size + 1 > 0) ? q_pos - window_size + 1 : 0;
    int k_end = q_pos + 1; if (k_end > seq_k) k_end = seq_k;
    __shared__ float shared_score;
    float max_score = -1e30f; float sum_exp = 0.0f; float acc_v = 0.0f;
    for (int j = k_start; j < k_end; j++) {
        const float *k_ptr = K + (size_t)j * kv_hidden + kv_h * head_dim;
        float local_dot = 0.0f; for (int d = tid; d < head_dim; d += blockDim.x) local_dot += q_row[d] * k_ptr[d];
        float score = block_reduce_sum(local_dot);
        if (tid == 0) shared_score = score * scale; __syncthreads();
        score = shared_score;
        const float *v_ptr = V + (size_t)j * kv_hidden + kv_h * head_dim;
        float old_max = max_score;
        if (score > max_score) {
            max_score = score; float correction = expf(old_max - max_score);
            sum_exp = sum_exp * correction + 1.0f; acc_v = acc_v * correction + (tid < head_dim ? v_ptr[tid] : 0.0f);
        } else {
            float weight = expf(score - max_score); sum_exp += weight; acc_v += weight * (tid < head_dim ? v_ptr[tid] : 0.0f);
        }
    }
    if (tid < head_dim) o_row[tid] = acc_v / (sum_exp + 1e-10f);
}

void vox_cuda_causal_attention(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int n_heads, int n_kv_heads,
                               int head_dim, float scale, int window_size, int q_offset) {
    if (seq_q == 1) {
        dim3 blocks(n_heads); dim3 threads(256);
        k_causal_attention_decode_opt<<<blocks, threads>>>(out, Q, K, V, seq_k, n_heads, n_kv_heads, head_dim, scale, window_size, q_offset);
    } else {
        dim3 blocks(n_heads, seq_q); dim3 threads(256);
        k_causal_attention_prefill_opt<<<blocks, threads>>>(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, window_size, q_offset);
    }
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

void vox_cuda_causal_attention_ptr(vox_cuda_ctx_t *ctx, float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, const int *seq_k_ptr, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int window_size, const int *q_offset_ptr) {
    if (seq_q == 1) {
        dim3 blocks(n_heads); dim3 threads(256);
        k_causal_attention_decode_opt_ptr<<<blocks, threads>>>(out, Q, K, V, seq_k_ptr, n_heads, n_kv_heads, head_dim, scale, window_size, q_offset_ptr);
    } else {
        fprintf(stderr, "vox_cuda_causal_attention_ptr: seq_q > 1 not implemented\n"); exit(1);
    }
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_ada_scale(float *x, const float *scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= (1.0f + scale[i]);
}

void vox_cuda_ada_scale(vox_cuda_ctx_t *ctx, float *x, const float *scale, int n) {
    int threads = 256; int blocks = (n + threads - 1) / threads;
    k_ada_scale<<<blocks, threads>>>(x, scale, n);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_transpose_mel(float *out, const float *in, int frames, int bins) {
    int f = blockIdx.x * blockDim.x + threadIdx.x; int m = blockIdx.y;
    if (f < frames && m < bins) out[m * frames + f] = in[f * bins + m];
}

void vox_cuda_transpose_mel(vox_cuda_ctx_t *ctx, float *out, const float *in, int frames, int bins) {
    dim3 threads(256); dim3 blocks((frames + 255) / 256, bins);
    k_transpose_mel<<<blocks, threads>>>(out, in, frames, bins);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

__global__ void k_transpose_conv(float *out, const float *in, int seq_len, int dim) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; int d = blockIdx.y;
    if (s < seq_len && d < dim) out[s * dim + d] = in[d * seq_len + s];
}

void vox_cuda_transpose_conv(vox_cuda_ctx_t *ctx, float *out, const float *in, int seq_len, int dim) {
    dim3 threads(256); dim3 blocks((seq_len + 255) / 256, dim);
    k_transpose_conv<<<blocks, threads>>>(out, in, seq_len, dim);
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

void vox_cuda_sgemm(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c) {
    float alpha = 1.0f; float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n));
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}

void vox_cuda_sgemm_t(vox_cuda_ctx_t *ctx, int m, int n, int k, const float *a, const float *b, float *c) {
    float alpha = 1.0f; float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, a, k, &beta, c, n));
    if (!ctx || !ctx->capturing) CUDA_CHECK(cudaGetLastError());
}