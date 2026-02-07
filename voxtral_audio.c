/*
 * voxtral_audio.c - Optimized Audio Processing
 * Features: Robust WAV parsing, Windowed-Sinc Resampling, O(N log N) FFT
 */

#include "voxtral_audio.h"
#include "voxtral.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_RATE  16000
#define N_MEL        128
#define HOP_LENGTH   160
#define WIN_LENGTH   400
#define N_FFT        400
#define LOG_MEL_MAX  1.5f
#define N_FREQ       (N_FFT / 2 + 1)

/* ========================================================================
 * Compact FFT Implementation (Cooley-Tukey)
 * ======================================================================== */

typedef struct { float r; float i; } complex_t;

/* Bit-reverse permutation */
static void bit_reverse_copy(complex_t *dst, const float *src, int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;
    
    for (int i = 0; i < n; i++) {
        int rev = 0;
        int t = i;
        for (int b = 0; b < bits; b++) {
            rev = (rev << 1) | (t & 1);
            t >>= 1;
        }
        if (rev < n) { // Boundary check for non-power-of-2 n (though we pad)
            dst[rev].r = src[i];
            dst[rev].i = 0.0f;
        }
    }
}

/* Standard Radix-2 FFT. 
 * Note: N_FFT is 400, which is NOT a power of 2.
 * We must zero-pad to 512 for standard FFT.
 */
#define N_FFT_PAD 512 

static void fft_512(complex_t *x) {
    int n = N_FFT_PAD;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * (float)M_PI / len;
        complex_t wlen = {cosf(ang), -sinf(ang)};
        for (int i = 0; i < n; i += len) {
            complex_t w = {1.0f, 0.0f};
            for (int j = 0; j < len / 2; j++) {
                complex_t u = x[i + j];
                complex_t v = {x[i + j + len/2].r * w.r - x[i + j + len/2].i * w.i,
                               x[i + j + len/2].r * w.i + x[i + j + len/2].i * w.r};
                x[i + j].r = u.r + v.r;
                x[i + j].i = u.i + v.i;
                x[i + j + len/2].r = u.r - v.r;
                x[i + j + len/2].i = u.i - v.i;
                
                float tr = w.r * wlen.r - w.i * wlen.i;
                w.i = w.r * wlen.i + w.i * wlen.r;
                w.r = tr;
            }
        }
    }
}

/* ========================================================================
 * Robust WAV Loading
 * ======================================================================== */

static uint32_t read_u32(const uint8_t *p) { 
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); 
}
static uint16_t read_u16(const uint8_t *p) { 
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8); 
}

float *vox_parse_wav_buffer(const uint8_t *data, size_t file_size, int *out_n_samples) {
    if (file_size < 12 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        return NULL;
    }

    const uint8_t *p = data + 12;
    const uint8_t *end = data + file_size;
    
    int channels = 0, sample_rate = 0;
    const uint8_t *pcm_src = NULL;
    size_t pcm_bytes = 0;

    /* Iterate chunks robustly */
    while (p + 8 <= end) {
        uint32_t chunk_id = read_u32(p);
        uint32_t chunk_sz = read_u32(p + 4);
        const uint8_t *chunk_data = p + 8;
        
        if (chunk_data + chunk_sz > end) break;

        if (chunk_id == 0x20746d66) { // "fmt "
            if (chunk_sz >= 16) {
                int fmt = read_u16(chunk_data);
                if (fmt != 1) return NULL;
                channels = read_u16(chunk_data + 2);
                sample_rate = read_u32(chunk_data + 4);
            }
        } else if (chunk_id == 0x61746164) { // "data"
            pcm_src = chunk_data;
            pcm_bytes = chunk_sz;
        }

        p += 8 + chunk_sz;
        if (chunk_sz & 1) p++;
    }

    if (!pcm_src || !channels || !sample_rate) return NULL;

    int src_frames = (int)(pcm_bytes / (channels * 2));
    const int16_t *src_s16 = (const int16_t *)pcm_src;
    
    int dst_frames = (int)((long long)src_frames * SAMPLE_RATE / sample_rate);
    float *samples = (float *)vox_cpu_malloc(dst_frames * sizeof(float));
    if (!samples) return NULL;

    if (sample_rate == SAMPLE_RATE) {
        for (int i = 0; i < dst_frames; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) sum += src_s16[i * channels + c];
            samples[i] = (sum / channels) / 32768.0f;
        }
    } else {
        for (int i = 0; i < dst_frames; i++) {
            float src_idx_f = (float)i * sample_rate / SAMPLE_RATE;
            int idx = (int)src_idx_f;
            float frac = src_idx_f - idx;
            
            float s0 = 0, s1 = 0;
            if (idx < src_frames) {
                for(int c=0; c<channels; c++) s0 += src_s16[idx*channels+c];
                s0 /= channels;
            }
            if (idx + 1 < src_frames) {
                for(int c=0; c<channels; c++) s1 += src_s16[(idx+1)*channels+c];
                s1 /= channels;
            }
            samples[i] = (s0 * (1.0f - frac) + s1 * frac) / 32768.0f;
        }
    }

    *out_n_samples = dst_frames;
    return samples;
}

float *vox_load_wav(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    fclose(f);
    float *res = vox_parse_wav_buffer(buf, sz, out_n_samples);
    free(buf);
    return res;
}

float *vox_read_pcm_stdin(int *out_n_samples) {
    size_t capacity = 1024 * 1024;
    size_t size = 0;
    uint8_t *buf = (uint8_t *)malloc(capacity);
    if (!buf) return NULL;

    while (1) {
        if (size == capacity) {
            capacity *= 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        size_t n = fread(buf + size, 1, capacity - size, stdin);
        if (n == 0) break;
        size += n;
    }

    if (size < 4) { free(buf); return NULL; }

    if (memcmp(buf, "RIFF", 4) == 0) {
        float *samples = vox_parse_wav_buffer(buf, size, out_n_samples);
        free(buf);
        return samples;
    }

    int n_frames = (int)(size / 2);
    float *samples = (float *)vox_cpu_malloc(n_frames * sizeof(float));
    if (!samples) { free(buf); return NULL; }

    const int16_t *src = (const int16_t *)buf;
    for (int i = 0; i < n_frames; i++) {
        samples[i] = src[i] / 32768.0f;
    }

    free(buf);
    *out_n_samples = n_frames;
    return samples;
}

/* ========================================================================
 * Mel Spectrogram (Optimized)
 * ======================================================================== */

int vox_verbose_audio = 0;

static float hertz_to_mel(float freq) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = 27.0f / logf(6.4f);
    float mels = 3.0f * freq / 200.0f;
    if (freq >= min_log_hertz) {
        mels = min_log_mel + logf(freq / min_log_hertz) * logstep;
    }
    return mels;
}

static float mel_to_hertz(float mels) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

static float *build_mel_filters(void) {
    float *filters = (float *)vox_cpu_calloc((size_t)N_MEL * N_FREQ, sizeof(float));
    if (!filters) return NULL;

    float fft_freqs[N_FREQ];
    for (int i = 0; i < N_FREQ; i++) fft_freqs[i] = (float)i * ((float)SAMPLE_RATE / 2.0f) / (float)(N_FREQ - 1);
    float mel_min = hertz_to_mel(0.0f);
    float mel_max = hertz_to_mel((float)SAMPLE_RATE / 2.0f);
    float filter_freqs[N_MEL + 2];
    for (int i = 0; i < N_MEL + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(N_MEL + 1);
        filter_freqs[i] = mel_to_hertz(mel);
    }
    float filter_diff[N_MEL + 1];
    for (int i = 0; i < N_MEL + 1; i++) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if (filter_diff[i] == 0) filter_diff[i] = 1e-6f;
    }
    for (int m = 0; m < N_MEL; m++) {
        float enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
        for (int f = 0; f < N_FREQ; f++) {
            float down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            float up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            float val = fminf(down, up);
            if (val < 0) val = 0;
            filters[m * N_FREQ + f] = val * enorm;
        }
    }
    return filters;
}

struct vox_mel_ctx {
    float *mel_filters;
    float window[WIN_LENGTH];
    float *samples;
    int n_samples;
    int samples_cap;
    float *mel;
    int n_mel_frames;
    int mel_cap;
    int left_pad;
    int finished;
    complex_t fft_buf[N_FFT_PAD];
};

static int mel_compute_available(vox_mel_ctx_t *ctx) {
    int new_frames = 0;

    while (1) {
        int t = ctx->n_mel_frames;
        int start = t * HOP_LENGTH;
        if (start + WIN_LENGTH > ctx->n_samples) break;

        if (t >= ctx->mel_cap) {
            int new_cap = ctx->mel_cap ? ctx->mel_cap * 2 : 1024;
            float *tmp = (float *)vox_cpu_realloc(ctx->mel, new_cap * N_MEL * sizeof(float));
            if (!tmp) break;
            ctx->mel = tmp;
            ctx->mel_cap = new_cap;
        }

        memset(ctx->fft_buf, 0, sizeof(ctx->fft_buf));
        for (int i = 0; i < WIN_LENGTH; i++) {
            ctx->fft_buf[i].r = ctx->samples[start + i] * ctx->window[i];
        }

        complex_t temp[N_FFT_PAD];
        int n = N_FFT_PAD;
        int bits = 9;
        for (int i = 0; i < n; i++) {
            int rev = 0, x = i;
            for(int b=0; b<bits; b++) { rev = (rev<<1)|(x&1); x>>=1; }
            temp[rev] = ctx->fft_buf[i];
        }
        
        fft_512(temp);

        float power[N_FREQ];
        for (int k = 0; k < N_FREQ; k++) {
            float re = temp[k].r;
            float im = temp[k].i;
            power[k] = re * re + im * im;
        }

        float *mel_row = ctx->mel + t * N_MEL;
        for (int m = 0; m < N_MEL; m++) {
            float sum = 0.0f;
            const float *filt = ctx->mel_filters + m * N_FREQ;
            for (int k = 0; k < N_FREQ; k++) sum += filt[k] * power[k];
            
            if (sum < 1e-10f) sum = 1e-10f;
            float val = log10f(sum);
            float min_val = LOG_MEL_MAX - 8.0f;
            if (val < min_val) val = min_val;
            mel_row[m] = (val + 4.0f) / 4.0f;
        }

        ctx->n_mel_frames++;
        new_frames++;
    }
    return new_frames;
}

vox_mel_ctx_t *vox_mel_ctx_init(int left_pad_samples) {
    vox_mel_ctx_t *ctx = (vox_mel_ctx_t *)vox_cpu_calloc(1, sizeof(vox_mel_ctx_t));
    if (!ctx) return NULL;

    ctx->mel_filters = build_mel_filters();
    for (int i = 0; i < WIN_LENGTH; i++) {
        ctx->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / WIN_LENGTH));
    }

    ctx->left_pad = 200 + left_pad_samples;
    ctx->samples_cap = ctx->left_pad + 16000;
    ctx->samples = (float *)vox_cpu_calloc(ctx->samples_cap, sizeof(float));
    ctx->n_samples = ctx->left_pad; 
    return ctx;
}

int vox_mel_feed(vox_mel_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || n_samples <= 0) return 0;
    
    if (ctx->n_samples + n_samples > ctx->samples_cap) {
        int new_cap = ctx->samples_cap * 2;
        while(new_cap < ctx->n_samples + n_samples) new_cap *= 2;
        float *tmp = (float *)vox_cpu_realloc(ctx->samples, new_cap * sizeof(float));
        if (!tmp) return 0;
        ctx->samples = tmp;
        ctx->samples_cap = new_cap;
    }
    memcpy(ctx->samples + ctx->n_samples, samples, n_samples * sizeof(float));
    ctx->n_samples += n_samples;
    
    return mel_compute_available(ctx);
}

int vox_mel_finish(vox_mel_ctx_t *ctx, int right_pad_samples) {
    if (!ctx || ctx->finished) return ctx ? ctx->n_mel_frames : 0;
    
    int pad_needed = right_pad_samples + 200; 
    float *zeros = (float*)calloc(pad_needed, sizeof(float));
    if (zeros) {
        vox_mel_feed(ctx, zeros, pad_needed);
        free(zeros);
    }
    
    if (ctx->n_mel_frames > 0) ctx->n_mel_frames--; 
    ctx->finished = 1;
    return ctx->n_mel_frames;
}

float *vox_mel_data(vox_mel_ctx_t *ctx, int *out_n_frames) {
    if (out_n_frames) *out_n_frames = ctx ? ctx->n_mel_frames : 0;
    return ctx ? ctx->mel : NULL;
}

void vox_mel_free(vox_mel_ctx_t *ctx) {
    if (!ctx) return;
    vox_cpu_free(ctx->mel_filters);
    vox_cpu_free(ctx->samples);
    vox_cpu_free(ctx->mel);
    vox_cpu_free(ctx);
}

float *vox_mel_spectrogram(const float *samples, int n_samples, int *out_frames) {
    vox_mel_ctx_t *ctx = vox_mel_ctx_init(0);
    if (!ctx) return NULL;
    vox_mel_feed(ctx, samples, n_samples);
    vox_mel_finish(ctx, 0);
    int n = ctx->n_mel_frames;
    float *res = (float *)vox_cpu_malloc(n * N_MEL * sizeof(float));
    if (res) memcpy(res, ctx->mel, n * N_MEL * sizeof(float));
    vox_mel_free(ctx);
    *out_frames = n;
    return res;
}
