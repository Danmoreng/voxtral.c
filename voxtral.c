/*
 * voxtral.c - Main API for Voxtral Realtime 4B inference
 *
 * Orchestrates the full pipeline:
 *   Load weights -> WAV -> Mel -> Encoder -> Adapter -> Decoder -> Tokenizer -> Text
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#include "voxtral_audio.h"
#include "voxtral_tokenizer.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "voxtral_cuda.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Memory Management Implementation */

/* vox_*_cpu functions: standard CPU-only memory */
void *vox_cpu_malloc(size_t size) {
    return malloc(size);
}

void *vox_cpu_calloc(size_t count, size_t size) {
    return calloc(count, size);
}

void *vox_cpu_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

void vox_cpu_free(void *ptr) {
    free(ptr);
}

/* vox_mem_* functions: potentially GPU-accessible device memory */
vox_backend_t g_selected_backend = VOX_BACKEND_CPU;

void *vox_gpu_malloc(size_t size) {
#ifdef USE_CUDA
    if (g_selected_backend == VOX_BACKEND_CUDA) {
        return vox_cuda_malloc(NULL, size);
    }
#endif
    return malloc(size);
}

void vox_gpu_free(void *ptr) {
#ifdef USE_CUDA
    if (g_selected_backend == VOX_BACKEND_CUDA) {
        vox_cuda_free(NULL, ptr);
        return;
    }
#endif
    free(ptr);
}

void *vox_mem_malloc(size_t size) {
    return vox_gpu_malloc(size);
}

void *vox_mem_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *ptr = vox_mem_malloc(total);
    if (ptr) {
#ifdef USE_CUDA
        if (g_selected_backend == VOX_BACKEND_CUDA) {
            cudaMemset(ptr, 0, total);
        } else
#endif
        memset(ptr, 0, total);
    }
    return ptr;
}

void *vox_mem_realloc(void *ptr, size_t size) {
#ifdef USE_CUDA
    if (g_selected_backend == VOX_BACKEND_CUDA) {
        /* CRITICAL: We cannot safely realloc device memory without tracking old size
           for the copy. Standard realloc on device memory pointer will crash or corrupt. */
        if (ptr) {
            fprintf(stderr, "FATAL: vox_mem_realloc called on CUDA device memory.\n");
            exit(1);
        }
        return vox_gpu_malloc(size);
    }
#endif
    return realloc(ptr, size);
}

void vox_mem_free(void *ptr) {
    if (!ptr) return;
    vox_gpu_free(ptr);
}

void vox_mem_copy(void *dst, const void *src, size_t size) {
#ifdef USE_CUDA
    if (g_selected_backend == VOX_BACKEND_CUDA) {
        cudaMemcpy(dst, src, size, cudaMemcpyDefault);
        return;
    }
#endif
    memcpy(dst, src, size);
}

char *vox_strdup(const char *s) {
    size_t len = strlen(s);
    char *out = (char *)vox_cpu_malloc(len + 1);
    if (out) memcpy(out, s, len + 1);
    return out;
}

/* ========================================================================
 * Context Loading
 * ======================================================================== */

int vox_verbose = 1;

/* ========================================================================
 * Decoder timing conditioning (t_cond + per-layer ada_scale)
 * ======================================================================== */

static void vox_compute_time_embedding(float *out, float t_value) {
    /* Matches vLLM MistralDecoderLayer time embedding:
     * inv_freq = exp(-log(10000) * arange(dim/2) / (dim/2))
     * emb = t * inv_freq
     * out = [cos(emb), sin(emb)] */
    int dim = VOX_DEC_DIM;
    int half = dim / 2;
    float log_theta = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float inv_freq = expf(-log_theta * (float)i / (float)half);
        float emb = t_value * inv_freq;
        out[i] = cosf(emb);
        out[i + half] = sinf(emb);
    }
}

static void vox_update_time_conditioning(vox_ctx_t *ctx) {
    if (!ctx) return;
    vox_compute_time_embedding(ctx->t_cond, (float)ctx->delay_tokens);

    size_t n = (size_t)VOX_DEC_LAYERS * VOX_DEC_DIM;
    if (!ctx->ada_scale) {
        ctx->ada_scale = (float *)vox_mem_malloc(n * sizeof(float));
    }
    if (!ctx->ada_scale) return;

    /* Precompute per-layer ada_scale = ada_up(GELU(ada_down(t_cond))). */
    float hidden[VOX_ADA_NORM_DIM];
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        const vox_dec_layer_t *l = &ctx->decoder.layers[layer];

        /* hidden = ada_down @ t_cond  (32 x 3072) */
        for (int i = 0; i < VOX_ADA_NORM_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_down + (size_t)i * VOX_DEC_DIM;
            for (int j = 0; j < VOX_DEC_DIM; j++) sum += row[j] * ctx->t_cond[j];
            hidden[i] = sum;
        }
        vox_gelu(NULL, hidden, VOX_ADA_NORM_DIM);

        float *scale_host = (float *)vox_cpu_malloc(VOX_DEC_DIM * sizeof(float));
        /* scale = ada_up @ hidden  (3072 x 32) */
        for (int i = 0; i < VOX_DEC_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_up + (size_t)i * VOX_ADA_NORM_DIM;
            for (int j = 0; j < VOX_ADA_NORM_DIM; j++) sum += row[j] * hidden[j];
            scale_host[i] = sum;
        }
        vox_mem_copy(ctx->ada_scale + (size_t)layer * VOX_DEC_DIM, scale_host, VOX_DEC_DIM * sizeof(float));
        vox_cpu_free(scale_host);
    }
}

vox_ctx_t *vox_load(const char *model_dir, vox_backend_t backend) {
    g_selected_backend = backend;

#ifdef USE_CUDA
    if (backend == VOX_BACKEND_CUDA) {
        if (!vox_cuda_available()) {
            fprintf(stderr, "Error: CUDA backend requested but no CUDA device found.\n");
            return NULL;
        }
    }
#endif

    vox_ctx_t *ctx = (vox_ctx_t *)vox_mem_calloc(1, sizeof(vox_ctx_t));
    if (!ctx) return NULL;

    ctx->backend = backend;
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    char path[1024];
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);
    
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        vox_free(ctx);
        return NULL;
    }
    ctx->safetensors = sf;

    if (vox_verbose >= 1) printf("Loading model from %s (Backend: %s)...\n", 
                                path, backend == VOX_BACKEND_CUDA ? "CUDA" : 
                                      (backend == VOX_BACKEND_METAL ? "Metal" : "CPU"));

    /* Load Encoder Weights */
    if (vox_encoder_load(&ctx->encoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load encoder weights\n");
        vox_free(ctx);
        return NULL;
    }

    /* Load Adapter Weights */
    if (vox_adapter_load(&ctx->adapter, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load adapter weights\n");
        vox_free(ctx);
        return NULL;
    }

    /* Load Decoder Weights */
    if (vox_decoder_load(&ctx->decoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load decoder weights\n");
        vox_free(ctx);
        return NULL;
    }

    /* Load Tokenizer */
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    ctx->tokenizer = vox_tokenizer_load(path);
    if (!ctx->tokenizer) {
        fprintf(stderr, "vox_load: failed to load tokenizer from %s\n", path);
        vox_free(ctx);
        return NULL;
    }

    /* Default delay: 480ms (6 tokens) */
    ctx->delay_tokens = 6;
    vox_update_time_conditioning(ctx);

#ifdef USE_CUDA
    if (backend == VOX_BACKEND_CUDA) {
        ctx->cuda_ctx = vox_cuda_init();
    }
#endif

    if (vox_verbose >= 1) printf("Model loaded successfully.\n");
    return ctx;
}

void vox_free(vox_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->safetensors) safetensors_close((safetensors_file_t *)ctx->safetensors);
    
    if (ctx->tokenizer) vox_tokenizer_free(ctx->tokenizer);

    /* Weights are managed by safetensors (mmap or managed memory) */
    
    /* Free KV caches */
    vox_mem_free(ctx->kv_cache_k);
    vox_mem_free(ctx->kv_cache_v);
    vox_mem_free(ctx->enc_kv_cache_k);
    vox_mem_free(ctx->enc_kv_cache_v);

    /* Free scratch buffers */
    vox_mem_free(ctx->enc_inc_x_norm);
    vox_mem_free(ctx->enc_inc_q);
    vox_mem_free(ctx->enc_inc_k);
    vox_mem_free(ctx->enc_inc_v);
    vox_mem_free(ctx->enc_inc_attn_out);
    vox_mem_free(ctx->enc_inc_proj_out);
    vox_mem_free(ctx->enc_inc_gate);
    vox_mem_free(ctx->enc_inc_up);
    vox_mem_free(ctx->enc_inc_ffn_out);
    vox_mem_free(ctx->enc_inc_positions);
    vox_mem_free(ctx->enc_inc_rope_freqs);

    vox_mem_free(ctx->ada_scale);

    vox_mem_free(ctx->dec_x);
    vox_mem_free(ctx->dec_x_norm);
    vox_mem_free(ctx->dec_q);
    vox_mem_free(ctx->dec_k);
    vox_mem_free(ctx->dec_v);
    vox_mem_free(ctx->dec_attn_out);
    vox_mem_free(ctx->dec_proj_out);
    vox_mem_free(ctx->dec_gate);
    vox_mem_free(ctx->dec_up);
    vox_mem_free(ctx->dec_ffn_out);
    vox_mem_free(ctx->dec_rope_freqs);

#ifdef USE_CUDA
    if (ctx->cuda_dec_graph_exec) vox_cuda_graph_destroy(ctx->cuda_dec_graph_exec);
    if (ctx->cuda_enc_graph_exec) vox_cuda_graph_destroy(ctx->cuda_enc_graph_exec);
    vox_mem_free(ctx->cuda_d_pos);
    vox_mem_free(ctx->cuda_d_total_seq);
    vox_mem_free(ctx->cuda_d_rope);
    vox_mem_free(ctx->cuda_d_argmax);
    if (ctx->cuda_ctx) vox_cuda_shutdown(ctx->cuda_ctx);
#endif

    vox_mem_free(ctx);
}

/* ========================================================================
 * Transcription Constants
 * ======================================================================== */

/*
 * Special token IDs (Tekken):
 *   BOS = 1, EOS = 2, [STREAMING_PAD] = 32
 *
 * Decoder input: inputs_embeds[pos] = adapter_out[pos] + tok_embed(input_id)
 * Prompt: [BOS] + [STREAMING_PAD] * (32 + delay_tokens)
 */
#define TOKEN_BOS          1
#define TOKEN_EOS          2
#define TOKEN_STREAMING_PAD 32

#define RAW_AUDIO_LENGTH_PER_TOK 1280
#define OFFLINE_STREAMING_BUFFER_TOKENS 10

/* First chunk minimum mel frames (enough for 39 prompt adapter tokens) */
#define STREAM_FIRST_CHUNK_MIN_MEL  312

/* Default processing interval in seconds (mel rate = 100 fps) */
#define STREAM_DEFAULT_INTERVAL  2.0f

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
static double get_time_ms(void) {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_usec / 1000.0;
}
#endif

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(vox_ctx_t *ctx, float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src_ptr = tok_emb_bf16 + (size_t)token_id * dim;
    uint16_t *host_src = (uint16_t *)vox_cpu_malloc(dim * sizeof(uint16_t));
    
#ifdef USE_CUDA
    if (ctx->backend == VOX_BACKEND_CUDA) {
        vox_cuda_copy_to_host(host_src, src_ptr, dim * sizeof(uint16_t));
    } else
#endif
    {
        memcpy(host_src, src_ptr, dim * sizeof(uint16_t));
    }

    float *host_dst = (float *)vox_cpu_malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)host_src[i]) << 16;
        memcpy(&host_dst[i], &f32_bits, sizeof(float));
    }
    vox_mem_copy(dst, host_dst, dim * sizeof(float));
    
    vox_cpu_free(host_src);
    vox_cpu_free(host_dst);
}

/* ========================================================================
 * Streaming API Implementation
 * ======================================================================== */

struct vox_stream {
    vox_ctx_t *ctx;
    vox_tokenizer_t *tokenizer;

    /* Incremental mel */
    vox_mel_ctx_t *mel_ctx;
    int real_samples_fed;

    /* Encoder chunk tracking */
    int mel_cursor;

    /* Incremental conv stem state */
    float *mel_tail;           /* [128 * 2] last 2 mel frames (column-major: [128, 2]) */
    float *conv0_tail;         /* [1280 * 2] last 2 conv0 outputs consumed by conv1 */
    float *conv0_residual;     /* [1280] 0 or 1 conv0 output pending stride alignment */
    int conv0_residual_count;  /* 0 or 1 */
    int conv_stem_initialized; /* 0 for first chunk */

    /* Residual encoder positions for 4x downsample alignment */
    float *enc_residual;       /* [1280 * 3] leftover positions */
    int enc_residual_count;    /* 0-3 */

    /* Adapter output buffer (growing) */
    float *adapter_buf;
    int total_adapter;
    int adapter_cap;

    /* Decoder state */
    int decoder_started;
    int gen_pos;        /* next adapter position for generation */
    int prev_token;
    int eos_seen;
    int finished;       /* vox_stream_finish() called */

    /* Pending token queue (circular buffer, VOX_MAX_ALT strings per position) */
    const char **token_queue;   /* [queue_cap * VOX_MAX_ALT] */
    int queue_head;     /* next position to read */
    int queue_tail;     /* next position to write */
    int queue_cap;      /* capacity in token positions */

    /* Alternative token settings */
    int n_alt;           /* max alternatives to track (default 1 = no alternatives) */
    float alt_cutoff;    /* max distance from top token (0.0-1.0) */

    /* Decoder working buffers */
    float *logits;
    float *step_embed;
    float *tok_tmp;

    /* Processing interval: minimum new mel frames before encoder triggers */
    int min_new_mel;            /* derived from interval in seconds (mel rate = 100 fps) */

    /* Timing */
    double encoder_ms;
    double decoder_ms;
    double prefill_ms;
    int n_generated;
    int n_text_tokens;          /* tokens with ID >= 1000 (visible text) */
};

/* Enqueue one token position. alts[0]=best, alts[1..VOX_MAX_ALT-1]=alternatives or NULL. */
static void stream_enqueue_token(vox_stream_t *s, const char *alts[VOX_MAX_ALT]) {
    /* Grow queue if full */
    int next_tail = (s->queue_tail + 1) % s->queue_cap;
    if (next_tail == s->queue_head) {
        int old_cap = s->queue_cap;
        int new_cap = old_cap * 2;
        const char **new_q = (const char **)vox_cpu_calloc((size_t)new_cap * VOX_MAX_ALT, sizeof(const char *));
        if (!new_q) return;
        /* Copy old entries in order */
        int n = 0;
        for (int i = s->queue_head; i != s->queue_tail; i = (i + 1) % old_cap) {
            memcpy(&new_q[n * VOX_MAX_ALT], &s->token_queue[i * VOX_MAX_ALT],
                   VOX_MAX_ALT * sizeof(const char *));
            n++;
        }
        vox_cpu_free((void *)s->token_queue);
        s->token_queue = new_q;
        s->queue_head = 0;
        s->queue_tail = n;
        s->queue_cap = new_cap;
        next_tail = (s->queue_tail + 1) % s->queue_cap;
    }
    memcpy((void *)&s->token_queue[s->queue_tail * VOX_MAX_ALT], alts,
           VOX_MAX_ALT * sizeof(const char *));
    s->queue_tail = next_tail;
}

static float *stream_conv_stem(vox_stream_t *s, const float *mel_new,
                                int n_new_mel, int *out_len) {
    vox_encoder_t *enc = &s->ctx->encoder;
    int dim = VOX_ENC_DIM; /* 1280 */
    *out_len = 0;

    if (n_new_mel <= 0) return NULL;

    int is_first = 0;

    /* === Phase 1: Conv0 — produce new conv0 outputs [dim, conv0_new_len] === */
    int conv0_new_len;
    float *conv0_new; /* [dim, conv0_new_len] column-major, caller frees */

    if (!s->conv_stem_initialized) {
        is_first = 1;

        /* Transpose mel [n_new_mel, 128] -> [128, n_new_mel] */
        float *conv_in = (float *)vox_mem_malloc((size_t)VOX_MEL_BINS * n_new_mel * sizeof(float));
        float *host_conv_in = (float *)vox_cpu_malloc((size_t)VOX_MEL_BINS * n_new_mel * sizeof(float));
        for (int f = 0; f < n_new_mel; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                host_conv_in[m * n_new_mel + f] = mel_new[f * VOX_MEL_BINS + m];
        vox_mem_copy(conv_in, host_conv_in, (size_t)VOX_MEL_BINS * n_new_mel * sizeof(float));
        vox_cpu_free(host_conv_in);

        conv0_new_len = n_new_mel;
        conv0_new = (float *)vox_mem_malloc((size_t)dim * conv0_new_len * sizeof(float));
        vox_causal_conv1d(s->ctx->cuda_ctx, conv0_new, conv_in, enc->conv0_weight, enc->conv0_bias,
                          VOX_MEL_BINS, dim, n_new_mel, 3, 1);
        vox_gelu(s->ctx->cuda_ctx, conv0_new, dim * conv0_new_len);
        vox_mem_free(conv_in);

        /* Save last 2 mel frames (column-major [128, 2]) */
        if (!s->mel_tail) s->mel_tail = (float *)vox_cpu_calloc((size_t)VOX_MEL_BINS * 2, sizeof(float));
        int ts = n_new_mel >= 2 ? n_new_mel - 2 : 0;
        int tc = n_new_mel >= 2 ? 2 : n_new_mel;
        memset(s->mel_tail, 0, (size_t)VOX_MEL_BINS * 2 * sizeof(float));
        for (int f = 0; f < tc; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                s->mel_tail[m * 2 + (2 - tc + f)] = mel_new[(ts + f) * VOX_MEL_BINS + m];

        s->conv_stem_initialized = 1;
    } else {
        /* Subsequent chunks: prepend mel_tail for conv0 boundary */
        int padded_mel_len = 2 + n_new_mel;
        float *conv_in = (float *)vox_mem_malloc((size_t)VOX_MEL_BINS * padded_mel_len * sizeof(float));
        float *host_conv_in = (float *)vox_cpu_malloc((size_t)VOX_MEL_BINS * padded_mel_len * sizeof(float));
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            host_conv_in[m * padded_mel_len + 0] = s->mel_tail[m * 2 + 0];
            host_conv_in[m * padded_mel_len + 1] = s->mel_tail[m * 2 + 1];
            for (int f = 0; f < n_new_mel; f++)
                host_conv_in[m * padded_mel_len + 2 + f] = mel_new[f * VOX_MEL_BINS + m];
        }
        vox_mem_copy(conv_in, host_conv_in, (size_t)VOX_MEL_BINS * padded_mel_len * sizeof(float));
        vox_cpu_free(host_conv_in);

        float *conv0_full = (float *)vox_mem_malloc((size_t)dim * padded_mel_len * sizeof(float));
        vox_causal_conv1d(s->ctx->cuda_ctx, conv0_full, conv_in, enc->conv0_weight, enc->conv0_bias,
                          VOX_MEL_BINS, dim, padded_mel_len, 3, 1);
        vox_gelu(s->ctx->cuda_ctx, conv0_full, dim * padded_mel_len);
        vox_mem_free(conv_in);

        /* Discard first 2 (from overlap, contaminated by zero-pad) */
        conv0_new_len = n_new_mel;
        conv0_new = (float *)vox_mem_malloc((size_t)dim * conv0_new_len * sizeof(float));
        vox_mem_copy(conv0_new, (char*)conv0_full + (size_t)2 * sizeof(float), (size_t)dim * conv0_new_len * sizeof(float)); // FIXME: incorrect offset for column-major
        // Actually for column-major [dim, padded_mel_len], discarding first 2 frames means skipping first 2 values of EACH row.
        float *host_full = (float *)vox_cpu_malloc((size_t)dim * padded_mel_len * sizeof(float));
        float *host_new = (float *)vox_cpu_malloc((size_t)dim * conv0_new_len * sizeof(float));
        vox_mem_copy(host_full, conv0_full, (size_t)dim * padded_mel_len * sizeof(float));
        for (int d = 0; d < dim; d++)
            memcpy(host_new + (size_t)d * conv0_new_len,
                   host_full + (size_t)d * padded_mel_len + 2,
                   (size_t)conv0_new_len * sizeof(float));
        vox_mem_copy(conv0_new, host_new, (size_t)dim * conv0_new_len * sizeof(float));
        vox_cpu_free(host_full);
        vox_cpu_free(host_new);
        vox_mem_free(conv0_full);

        /* Update mel_tail */
        int ts = n_new_mel >= 2 ? n_new_mel - 2 : 0;
        int tc = n_new_mel >= 2 ? 2 : n_new_mel;
        memset(s->mel_tail, 0, (size_t)VOX_MEL_BINS * 2 * sizeof(float));
        for (int f = 0; f < tc; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                s->mel_tail[m * 2 + (2 - tc + f)] = mel_new[(ts + f) * VOX_MEL_BINS + m];
    }

    /* === Phase 2: Stride alignment — ensure even count for conv1 === */
    int prev_res = s->conv0_residual_count;
    int total_avail = prev_res + conv0_new_len;
    int new_res = total_avail & 1; /* 1 if odd, 0 if even */
    int feed_from_new = conv0_new_len - new_res;
    int feed_total = prev_res + feed_from_new; /* always even */

    if (feed_total <= 0) {
        /* Not enough to feed conv1 — just save residual */
        if (new_res && conv0_new_len > 0) {
            if (!s->conv0_residual)
                s->conv0_residual = (float *)vox_cpu_malloc((size_t)dim * sizeof(float));
            float *host_conv0_new = (float *)vox_cpu_malloc((size_t)dim * conv0_new_len * sizeof(float));
            vox_mem_copy(host_conv0_new, conv0_new, (size_t)dim * conv0_new_len * sizeof(float));
            for (int d = 0; d < dim; d++)
                s->conv0_residual[d] = host_conv0_new[(size_t)d * conv0_new_len + conv0_new_len - 1];
            vox_cpu_free(host_conv0_new);
        }
        s->conv0_residual_count = new_res;
        vox_mem_free(conv0_new);
        return NULL;
    }

    /* Build feed buffer [dim, feed_total] column-major */
    float *feed = (float *)vox_mem_malloc((size_t)dim * feed_total * sizeof(float));
    float *host_feed = (float *)vox_cpu_malloc((size_t)dim * feed_total * sizeof(float));
    float *host_conv0_new = (float *)vox_cpu_malloc((size_t)dim * conv0_new_len * sizeof(float));
    vox_mem_copy(host_conv0_new, conv0_new, (size_t)dim * conv0_new_len * sizeof(float));
    
    int fpos = 0;
    if (prev_res == 1) {
        for (int d = 0; d < dim; d++)
            host_feed[(size_t)d * feed_total + 0] = s->conv0_residual[d];
        fpos = 1;
    }

    for (int d = 0; d < dim; d++)
        memcpy(host_feed + (size_t)d * feed_total + fpos,
               host_conv0_new + (size_t)d * conv0_new_len,
               (size_t)feed_from_new * sizeof(float));

    if (new_res) {
        if (!s->conv0_residual)
            s->conv0_residual = (float *)vox_cpu_malloc((size_t)dim * sizeof(float));
        for (int d = 0; d < dim; d++)
            s->conv0_residual[d] = host_conv0_new[(size_t)d * conv0_new_len + conv0_new_len - 1];
    }
    s->conv0_residual_count = new_res;
    vox_mem_copy(feed, host_feed, (size_t)dim * feed_total * sizeof(float));
    vox_cpu_free(host_feed);
    vox_cpu_free(host_conv0_new);
    vox_mem_free(conv0_new);

    /* === Phase 3: Conv1 === */
    float *conv1_in;
    int conv1_in_len;
    int conv1_discard; /* outputs to discard at front */

    if (is_first) {
        conv1_in = feed; 
        conv1_in_len = feed_total;
        conv1_discard = 0;
    } else {
        conv1_in_len = 2 + feed_total;
        conv1_in = (float *)vox_mem_malloc((size_t)dim * conv1_in_len * sizeof(float));
        float *host_conv1_in = (float *)vox_cpu_malloc((size_t)dim * conv1_in_len * sizeof(float));
        float *host_feed = (float *)vox_cpu_malloc((size_t)dim * feed_total * sizeof(float));
        vox_mem_copy(host_feed, feed, (size_t)dim * feed_total * sizeof(float));
        for (int d = 0; d < dim; d++) {
            host_conv1_in[(size_t)d * conv1_in_len + 0] = s->conv0_tail[d * 2 + 0];
            host_conv1_in[(size_t)d * conv1_in_len + 1] = s->conv0_tail[d * 2 + 1];
            memcpy(host_conv1_in + (size_t)d * conv1_in_len + 2,
                   host_feed + (size_t)d * feed_total,
                   (size_t)feed_total * sizeof(float));
        }
        vox_mem_copy(conv1_in, host_conv1_in, (size_t)dim * conv1_in_len * sizeof(float));
        vox_cpu_free(host_conv1_in);
        vox_cpu_free(host_feed);
        conv1_discard = 1;
    }

    if (!s->conv0_tail) s->conv0_tail = (float *)vox_cpu_calloc((size_t)dim * 2, sizeof(float));
    float *host_feed_final = (float *)vox_cpu_malloc((size_t)dim * feed_total * sizeof(float));
    vox_mem_copy(host_feed_final, feed, (size_t)dim * feed_total * sizeof(float));
    for (int d = 0; d < dim; d++) {
        s->conv0_tail[d * 2 + 0] = host_feed_final[(size_t)d * feed_total + feed_total - 2];
        s->conv0_tail[d * 2 + 1] = host_feed_final[(size_t)d * feed_total + feed_total - 1];
    }
    vox_cpu_free(host_feed_final);
    if (!is_first) vox_mem_free(feed);

    int conv1_out_len = conv1_in_len / 2;
    float *conv1_out = (float *)vox_mem_malloc((size_t)dim * conv1_out_len * sizeof(float));
    vox_causal_conv1d(s->ctx->cuda_ctx, conv1_out, conv1_in, enc->conv1_weight, enc->conv1_bias,
                      dim, dim, conv1_in_len, 3, 2);
    vox_gelu(s->ctx->cuda_ctx, conv1_out, dim * conv1_out_len);
    if (is_first) vox_mem_free(feed);
    else vox_mem_free(conv1_in);

    int result_len = conv1_out_len - conv1_discard;
    if (result_len <= 0) {
        vox_mem_free(conv1_out);
        return NULL;
    }

    float *result = (float *)vox_mem_malloc((size_t)result_len * dim * sizeof(float));
    float *host_conv1_out = (float *)vox_cpu_malloc((size_t)dim * conv1_out_len * sizeof(float));
    float *host_result = (float *)vox_cpu_malloc((size_t)result_len * dim * sizeof(float));
    vox_mem_copy(host_conv1_out, conv1_out, (size_t)dim * conv1_out_len * sizeof(float));
    for (int si = 0; si < result_len; si++)
        for (int d = 0; d < dim; d++)
            host_result[(size_t)si * dim + d] = host_conv1_out[(size_t)d * conv1_out_len + conv1_discard + si];
    vox_mem_copy(result, host_result, (size_t)result_len * dim * sizeof(float));
    vox_cpu_free(host_conv1_out);
    vox_cpu_free(host_result);
    vox_mem_free(conv1_out);

    *out_len = result_len;
    return result;
}

static void stream_run_encoder(vox_stream_t *s) {
    int total_mel = 0;
    float *mel_data = vox_mel_data(s->mel_ctx, &total_mel);
    int dim = VOX_DEC_DIM;

    int new_mel = total_mel - s->mel_cursor;
    int need_mel = (!s->conv_stem_initialized) ? STREAM_FIRST_CHUNK_MIN_MEL : s->min_new_mel;

    if (new_mel < need_mel && !s->finished) return;
    if (new_mel <= 0) return;

    double t0 = get_time_ms();

    int conv_out_len = 0;
    float *conv_out = stream_conv_stem(s, mel_data + (size_t)s->mel_cursor * VOX_MEL_BINS,
                                        new_mel, &conv_out_len);
    s->mel_cursor = total_mel;

    if (!conv_out || conv_out_len <= 0) {
        vox_mem_free(conv_out);
        return;
    }

    int enc_out_len = 0;
    float *enc_out = vox_encoder_forward_incremental(s->ctx, conv_out, conv_out_len, &enc_out_len);
    vox_mem_free(conv_out);

    if (!enc_out || enc_out_len <= 0) {
        vox_mem_free(enc_out);
        return;
    }

    int total_enc = s->enc_residual_count + enc_out_len;
    int usable = (total_enc / VOX_DOWNSAMPLE) * VOX_DOWNSAMPLE;
    int leftover = total_enc - usable;

    if (usable > 0) {
        float *combined = (float *)vox_mem_malloc((size_t)usable * VOX_ENC_DIM * sizeof(float));
        int pos = 0;

        if (s->enc_residual_count > 0 && s->enc_residual) {
            int from_residual = s->enc_residual_count;
            if (from_residual > usable) from_residual = usable;
            vox_mem_copy(combined, s->enc_residual, (size_t)from_residual * VOX_ENC_DIM * sizeof(float));
            pos = from_residual;
        }

        int from_enc = usable - pos;
        if (from_enc > 0) {
            vox_mem_copy(combined + (size_t)pos * VOX_ENC_DIM,
                         enc_out, (size_t)from_enc * VOX_ENC_DIM * sizeof(float));
        }

        int chunk_tokens = 0;
        float *adapter_chunk = vox_adapter_forward(s->ctx, combined, usable, &chunk_tokens);
        vox_mem_free(combined);

        if (adapter_chunk && chunk_tokens > 0) {
            if (s->total_adapter + chunk_tokens > s->adapter_cap) {
                int new_cap = s->adapter_cap ? s->adapter_cap * 2 : 256;
                while (new_cap < s->total_adapter + chunk_tokens) new_cap *= 2;
                float *tmp = (float *)vox_mem_realloc(s->adapter_buf,
                    (size_t)new_cap * dim * sizeof(float));
                if (!tmp) { vox_mem_free(adapter_chunk); vox_mem_free(enc_out); return; }
                s->adapter_buf = tmp;
                s->adapter_cap = new_cap;
            }
            vox_mem_copy(s->adapter_buf + (size_t)s->total_adapter * dim,
                         adapter_chunk, (size_t)chunk_tokens * dim * sizeof(float));
            s->total_adapter += chunk_tokens;
            vox_mem_free(adapter_chunk);
        } else {
            vox_mem_free(adapter_chunk);
        }
    }

    if (leftover > 0) {
        if (!s->enc_residual)
            s->enc_residual = (float *)vox_mem_malloc(3 * VOX_ENC_DIM * sizeof(float));
        int enc_used = usable - s->enc_residual_count;
        if (enc_used < 0) enc_used = 0;
        vox_mem_copy(s->enc_residual, (char*)enc_out + (size_t)enc_used * VOX_ENC_DIM * sizeof(float),
                     (size_t)leftover * VOX_ENC_DIM * sizeof(float));
    }
    s->enc_residual_count = leftover;

    vox_mem_free(enc_out);
    s->encoder_ms += get_time_ms() - t0;
}

static void stream_fill_alts(vox_stream_t *s, int best_token,
                              const char *alts[VOX_MAX_ALT]) {
    memset(alts, 0, VOX_MAX_ALT * sizeof(const char *));
    alts[0] = vox_tokenizer_decode(s->tokenizer, best_token);

    if (s->n_alt <= 1) return;

    float *logits = (float *)vox_cpu_malloc(VOX_VOCAB_SIZE * sizeof(float));
    vox_mem_copy(logits, s->logits, VOX_VOCAB_SIZE * sizeof(float));
    
    float max_val = logits[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0;
    for (int i = 0; i < VOX_VOCAB_SIZE; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < VOX_VOCAB_SIZE; i++)
        logits[i] *= inv_sum;

    float best_prob = logits[best_token];
    if (best_prob <= 0) { vox_cpu_free(logits); return; }

    int found = 1; 
    int used[VOX_MAX_ALT];
    used[0] = best_token;

    while (found < s->n_alt) {
        int best_idx = -1;
        float best_p = -1;
        for (int i = 1000; i < VOX_VOCAB_SIZE; i++) {
            if (i == best_token) continue;
            int skip = 0;
            for (int j = 1; j < found; j++)
                if (used[j] == i) { skip = 1; break; }
            if (skip) continue;

            if (logits[i] > best_p) {
                best_p = logits[i];
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        float r = 1.0f - best_p / best_prob;
        if (r > s->alt_cutoff) break;

        used[found] = best_idx;
        alts[found] = vox_tokenizer_decode(s->tokenizer, best_idx);
        found++;
    }
    vox_cpu_free(logits);
}

static void stream_run_decoder(vox_stream_t *s) {
    int dim = VOX_DEC_DIM;
    int prompt_len = 1 + 32 + s->ctx->delay_tokens;
    uint16_t *tok_emb_bf16 = s->ctx->decoder.tok_embeddings_bf16;

    if (!s->decoder_started && s->total_adapter >= prompt_len) {
        double t0 = get_time_ms();

        float *prompt_embeds = (float *)vox_mem_malloc((size_t)prompt_len * dim * sizeof(float));
        float *host_prompt_embeds = (float *)vox_cpu_malloc((size_t)prompt_len * dim * sizeof(float));
        float *host_adapter_buf = (float *)vox_cpu_malloc((size_t)prompt_len * dim * sizeof(float));
        vox_mem_copy(host_adapter_buf, s->adapter_buf, (size_t)prompt_len * dim * sizeof(float));

        float *tok_tmp_host = (float *)vox_cpu_malloc(dim * sizeof(float));

        for (int i = 0; i < prompt_len; i++) {
            int tok = (i == 0) ? TOKEN_BOS : TOKEN_STREAMING_PAD;
            tok_embed_bf16_to_f32(s->ctx, s->tok_tmp, tok_emb_bf16, tok, dim);
            vox_mem_copy(tok_tmp_host, s->tok_tmp, dim * sizeof(float));
            float *a = host_adapter_buf + (size_t)i * dim;
            float *dst = host_prompt_embeds + (size_t)i * dim;
            for (int j = 0; j < dim; j++) dst[j] = a[j] + tok_tmp_host[j];
        }
        vox_mem_copy(prompt_embeds, host_prompt_embeds, (size_t)prompt_len * dim * sizeof(float));
        vox_cpu_free(host_prompt_embeds);
        vox_cpu_free(host_adapter_buf);
        vox_cpu_free(tok_tmp_host);

        s->ctx->kv_cache_len = 0;
        s->ctx->kv_pos_offset = 0;

        int prefill_count = prompt_len - 1;
        vox_decoder_prefill(s->ctx, prompt_embeds, prefill_count);

        vox_mem_copy(s->step_embed, prompt_embeds + (size_t)prefill_count * dim,
                     (size_t)dim * sizeof(float));
        vox_mem_free(prompt_embeds);

        s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, s->logits);
        s->n_generated++;

        if (s->prev_token != TOKEN_EOS && s->prev_token >= 1000) {
            const char *alts[VOX_MAX_ALT];
            stream_fill_alts(s, s->prev_token, alts);
            if (alts[0]) { stream_enqueue_token(s, alts); s->n_text_tokens++; }
        }
        if (s->prev_token == TOKEN_EOS) s->eos_seen = 1;

        s->gen_pos = prompt_len;
        s->decoder_started = 1;

        double pf_ms = get_time_ms() - t0;
        s->decoder_ms += pf_ms;
        s->prefill_ms += pf_ms;
    }

    if (s->decoder_started && !s->eos_seen) {
        double t0 = get_time_ms();
        int gen_before = s->n_generated;
        
        float *host_adapter_step = (float *)vox_cpu_malloc(dim * sizeof(float));
        float *tok_tmp_host = (float *)vox_cpu_malloc(dim * sizeof(float));
        float *step_embed_host = (float *)vox_cpu_malloc(dim * sizeof(float));

        while (s->gen_pos < s->total_adapter) {
            tok_embed_bf16_to_f32(s->ctx, s->tok_tmp, tok_emb_bf16, s->prev_token, dim);
            vox_mem_copy(tok_tmp_host, s->tok_tmp, dim * sizeof(float));
            vox_mem_copy(host_adapter_step, s->adapter_buf + (size_t)s->gen_pos * dim, dim * sizeof(float));
            for (int j = 0; j < dim; j++)
                step_embed_host[j] = host_adapter_step[j] + tok_tmp_host[j];
            
            vox_mem_copy(s->step_embed, step_embed_host, dim * sizeof(float));

            s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, s->logits);
            s->n_generated++;

            if (s->prev_token != TOKEN_EOS && s->prev_token >= 1000) {
                const char *alts[VOX_MAX_ALT];
                stream_fill_alts(s, s->prev_token, alts);
                if (alts[0]) { stream_enqueue_token(s, alts); s->n_text_tokens++; }
            }

            s->gen_pos++;
            if (s->prev_token == TOKEN_EOS) { s->eos_seen = 1; break; }
        }
        vox_cpu_free(host_adapter_step);
        vox_cpu_free(tok_tmp_host);
        vox_cpu_free(step_embed_host);

        if (s->n_generated > gen_before) {
            s->decoder_ms += get_time_ms() - t0;
        }
    }
}

vox_stream_t *vox_stream_init(vox_ctx_t *ctx) {
    vox_stream_t *s = (vox_stream_t *)vox_cpu_calloc(1, sizeof(vox_stream_t));
    if (!s) return NULL;

    s->ctx = ctx;
    s->tokenizer = ctx->tokenizer;

    s->mel_ctx = vox_mel_ctx_init(32 * RAW_AUDIO_LENGTH_PER_TOK);
    if (!s->mel_ctx) {
        vox_tokenizer_free(s->tokenizer);
        vox_cpu_free(s);
        return NULL;
    }

    s->queue_cap = 256;
    s->token_queue = (const char **)vox_cpu_calloc((size_t)s->queue_cap * VOX_MAX_ALT, sizeof(const char *));
    s->n_alt = 1;

    int dim = VOX_DEC_DIM;
    s->logits = (float *)vox_mem_malloc(VOX_VOCAB_SIZE * sizeof(float));
    s->step_embed = (float *)vox_mem_malloc(dim * sizeof(float));
    s->tok_tmp = (float *)vox_mem_malloc(dim * sizeof(float));

    if (!s->token_queue || !s->logits || !s->step_embed || !s->tok_tmp) {
        vox_stream_free(s);
        return NULL;
    }

    ctx->enc_kv_cache_len = 0;
    ctx->enc_kv_pos_offset = 0;

    s->min_new_mel = (int)(STREAM_DEFAULT_INTERVAL * 100.0f);

    return s;
}

int vox_stream_feed(vox_stream_t *s, const float *samples, int n_samples) {
    if (!s || s->finished || n_samples <= 0) return -1;

    vox_mel_feed(s->mel_ctx, samples, n_samples);
    s->real_samples_fed += n_samples;

    stream_run_encoder(s);
    stream_run_decoder(s);

    return 0;
}

int vox_stream_finish(vox_stream_t *s) {
    if (!s || s->finished) return -1;
    s->finished = 1;

    int n_delay_tokens = s->ctx->delay_tokens;

    int align_pad = (RAW_AUDIO_LENGTH_PER_TOK -
        (s->real_samples_fed % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK;
    int n_right_pad_tokens = (n_delay_tokens + 1) + OFFLINE_STREAMING_BUFFER_TOKENS;
    int right_pad = align_pad + n_right_pad_tokens * RAW_AUDIO_LENGTH_PER_TOK;

    float zero_buf[4096];
    memset(zero_buf, 0, sizeof(zero_buf));
    int remaining = right_pad;
    while (remaining > 0) {
        int chunk = remaining > 4096 ? 4096 : remaining;
        vox_mel_feed(s->mel_ctx, zero_buf, chunk);
        remaining -= chunk;
    }

    vox_mel_finish(s->mel_ctx, 0);

    stream_run_encoder(s);
    stream_run_decoder(s);

    return 0;
}

int vox_stream_get(vox_stream_t *s, const char **out_tokens, int max) {
    if (!s || max <= 0) return 0;
    int count = 0;
    while (count < max && s->queue_head != s->queue_tail) {
        out_tokens[count++] = s->token_queue[s->queue_head * VOX_MAX_ALT];
        s->queue_head = (s->queue_head + 1) % s->queue_cap;
    }
    return count;
}

void vox_stream_set_alt(vox_stream_t *s, int n_alt, float cutoff) {
    if (!s) return;
    if (n_alt < 1) n_alt = 1;
    if (n_alt > VOX_MAX_ALT) n_alt = VOX_MAX_ALT;
    if (cutoff < 0) cutoff = 0;
    if (cutoff > 1) cutoff = 1;
    s->n_alt = n_alt;
    s->alt_cutoff = cutoff;
}

int vox_stream_get_alt(vox_stream_t *s, const char **out_tokens,
                       int max_tokens, int n_alt) {
    if (!s || max_tokens <= 0 || n_alt <= 0) return 0;
    if (n_alt > VOX_MAX_ALT) n_alt = VOX_MAX_ALT;
    int count = 0;
    while (count < max_tokens && s->queue_head != s->queue_tail) {
        const char **src = &s->token_queue[s->queue_head * VOX_MAX_ALT];
        const char **dst = &out_tokens[count * n_alt];
        for (int a = 0; a < n_alt; a++)
            dst[a] = src[a];
        count++;
        s->queue_head = (s->queue_head + 1) % s->queue_cap;
    }
    return count;
}

void vox_stream_free(vox_stream_t *s) {
    if (!s) return;

    if (vox_verbose >= 1) {
        fprintf(stderr, "Encoder: %d mel -> %d tokens (%.0f ms)\n",
                s->mel_cursor, s->total_adapter, s->encoder_ms);
        if (s->n_text_tokens > 0) {
            double gen_ms = s->decoder_ms - s->prefill_ms;
            fprintf(stderr, "Decoder: %d text tokens (%d steps) in %.0f ms "
                    "(prefill %.0f ms + %.1f ms/step)\n",
                    s->n_text_tokens, s->n_generated, s->decoder_ms,
                    s->prefill_ms,
                    s->n_generated > 1 ? gen_ms / (s->n_generated - 1) : 0);
        }
    }

    vox_mel_free(s->mel_ctx);
    if (s->tokenizer) vox_tokenizer_free(s->tokenizer);
    vox_mem_free(s->adapter_buf);
    vox_cpu_free((void *)s->token_queue);
    vox_mem_free(s->logits);
    vox_mem_free(s->step_embed);
    vox_mem_free(s->tok_tmp);
    vox_cpu_free(s->mel_tail);
    vox_cpu_free(s->conv0_tail);
    vox_cpu_free(s->conv0_residual);
    vox_mem_free(s->enc_residual);
    vox_cpu_free(s);
}

/* ========================================================================
 * Convenience Functions Implementation
 * ======================================================================== */

static void trim_ascii_whitespace(char *s) {
    char *p = s;
    int l = (int)strlen(p);
    while(l > 0 && isspace((unsigned char)p[l-1])) p[--l] = 0;
    while(*p && isspace((unsigned char)*p)) { p++; l--; }
    memmove(s, p, l + 1);
}

char *vox_transcribe_audio(vox_ctx_t *ctx, const float *samples, int n_samples) {
    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    vox_stream_feed(s, samples, n_samples);
    vox_stream_finish(s);

    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)vox_cpu_malloc(text_cap);
    text[0] = '\0';

    const char *tokens[64];
    int n;
    while ((n = vox_stream_get(s, tokens, 64)) > 0) {
        for (int i = 0; i < n; i++) {
            size_t piece_len = strlen(tokens[i]);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)vox_cpu_realloc(text, text_cap);
            }
            memcpy(text + text_len, tokens[i], piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe_stdin(vox_ctx_t *ctx) {
    /* Peak 4 bytes */
    uint8_t header[4];
    size_t nread = fread(header, 1, 4, stdin);
    if (nread < 4) return NULL;

    if (memcmp(header, "RIFF", 4) == 0) {
        /* WAV on stdin */
        size_t capacity = 1024 * 1024;
        size_t size = 4;
        uint8_t *buf = (uint8_t *)vox_cpu_malloc(capacity);
        if (!buf) return NULL;
        memcpy(buf, header, 4);

        while (1) {
            if (size == capacity) {
                capacity *= 2;
                uint8_t *tmp = (uint8_t *)vox_cpu_realloc(buf, capacity);
                if (!tmp) { vox_cpu_free(buf); return NULL; }
                buf = tmp;
            }
            size_t n = fread(buf + size, 1, capacity - size, stdin);
            if (n == 0) break;
            size += n;
        }

        /* Parse WAV header robustly */
        int audio_format = 0, channels = 0, sample_rate = 0, bits_per_sample = 0;
        const uint8_t *pcm_data = NULL;
        uint32_t pcm_size = 0;

        uint8_t *p = buf + 12;
        uint8_t *end = buf + size;
        while (p + 8 <= end) {
            uint32_t chunk_sz = (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
            if (memcmp(p, "fmt ", 4) == 0 && chunk_sz >= 16) {
                audio_format = p[8] | (p[9] << 8);
                channels = p[10] | (p[11] << 8);
                sample_rate = (int)((uint32_t)p[12] | ((uint32_t)p[13] << 8) | ((uint32_t)p[14] << 16) | ((uint32_t)p[15] << 24));
                bits_per_sample = p[22] | (p[23] << 8);
            } else if (memcmp(p, "data", 4) == 0) {
                pcm_data = p + 8;
                pcm_size = chunk_sz;
            }
            p += 8 + chunk_sz;
            if (chunk_sz & 1) p++;
        }

        if (audio_format != 1 || bits_per_sample != 16 || !pcm_data || channels < 1) {
            fprintf(stderr, "Unsupported WAV format on stdin\n");
            vox_cpu_free(buf);
            return NULL;
        }

        int n_frames = pcm_size / (channels * 2);
        float *samples = (float *)vox_cpu_malloc((size_t)n_frames * sizeof(float));
        if (!samples) { vox_cpu_free(buf); return NULL; }
        const int16_t *src = (const int16_t *)pcm_data;
        for (int i = 0; i < n_frames; i++) {
            if (channels == 1) {
                samples[i] = src[i] / 32768.0f;
            } else {
                float sum = 0;
                for (int c = 0; c < channels; c++) {
                    int16_t val;
                    memcpy(&val, &src[i * channels + c], sizeof(int16_t));
                    sum += val;
                }
                samples[i] = (sum / channels) / 32768.0f;
            }
        }
        vox_cpu_free(buf);

        if (sample_rate != VOX_SAMPLE_RATE) {
            int new_n = (int)((long long)n_frames * VOX_SAMPLE_RATE / sample_rate);
            float *resampled = (float *)vox_cpu_malloc((size_t)new_n * sizeof(float));
            if (!resampled) { vox_cpu_free(samples); return NULL; }
            for (int i = 0; i < new_n; i++) {
                float src_pos = (float)i * sample_rate / VOX_SAMPLE_RATE;
                int idx = (int)src_pos;
                float frac = src_pos - idx;
                if (idx + 1 < n_frames)
                    resampled[i] = samples[idx] * (1-frac) + samples[idx+1] * frac;
                else
                    resampled[i] = (idx < n_frames) ? samples[idx] : 0.0f;
            }
            vox_cpu_free(samples);
            samples = resampled;
            n_frames = new_n;
        }

        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_frames, (float)n_frames / VOX_SAMPLE_RATE);

        /* Use stream API so tokens are emitted incrementally */
        vox_stream_t *s = vox_stream_init(ctx);
        if (!s) { vox_cpu_free(samples); return NULL; }
        vox_stream_feed(s, samples, n_frames);
        vox_stream_finish(s);
        vox_cpu_free(samples);

        /* Collect and optionally stream tokens */
        size_t text_cap = 1024;
        size_t text_len = 0;
        char *text = (char *)vox_cpu_malloc(text_cap);
        text[0] = '\0';

        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)vox_cpu_realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
        vox_stream_free(s);
        trim_ascii_whitespace(text);
        return text;
    }

    /* Raw s16le streaming mode */
    if (vox_verbose >= 2)
        fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    /* Feed the 4 peeked header bytes as 2 s16le samples */
    {
        int16_t sv[2];
        memcpy(sv, header, 4);
        float f[2] = { sv[0] / 32768.0f, sv[1] / 32768.0f };
        vox_stream_feed(s, f, 2);
    }

    /* Collect text for non-streaming mode, or emit for streaming */
    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)vox_cpu_malloc(text_cap);
    text[0] = '\0';

    /* Read loop */
    int16_t raw_buf[4096];
    float fbuf[4096];
    const char *tokens[64];
    int eof_reached = 0;

    while (!eof_reached) {
        size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
        if (nread == 0) {
            eof_reached = 1;
            vox_stream_finish(s);
        } else {
            for (size_t i = 0; i < nread; i++)
                fbuf[i] = raw_buf[i] / 32768.0f;
            vox_stream_feed(s, fbuf, (int)nread);
        }

        /* Drain pending tokens */
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)vox_cpu_realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe(vox_ctx_t *ctx, const char *wav_path) {
    int n_samples = 0;
    float *samples = vox_load_wav(wav_path, &n_samples);
    if (!samples) {
        fprintf(stderr, "vox_transcribe: cannot load %s\n", wav_path);
        return NULL;
    }
    if (vox_verbose >= 1)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                n_samples, (float)n_samples / VOX_SAMPLE_RATE);

    char *text = vox_transcribe_audio(ctx, samples, n_samples);
    vox_cpu_free(samples);
    return text;
}

void vox_set_processing_interval(vox_stream_t *s, float seconds) {
    if (!s) return;
    if (seconds <= 0) seconds = 0;
    /* mel rate = sample_rate / hop_length = 16000/160 = 100 fps */
    s->min_new_mel = (int)(seconds * 100.0f);
    if (s->min_new_mel < 1) s->min_new_mel = 1;
}

void vox_set_delay(vox_ctx_t *ctx, int delay_ms) {
    /* Each token represents 80ms (frame_rate=12.5Hz) */
    if (delay_ms < 80) delay_ms = 80;
    if (delay_ms > 2400) delay_ms = 2400;
    ctx->delay_tokens = delay_ms / 80;
    vox_update_time_conditioning(ctx);
}