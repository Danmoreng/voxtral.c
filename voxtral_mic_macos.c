/*
 * voxtral_mic_macos.c - Microphone capture using AudioQueue Services (macOS)
 *
 * AudioQueue callback runs on its own thread, converts s16le to float,
 * and pushes to a ring buffer protected by a pthread mutex.
 * The main thread polls vox_mic_read() to drain samples.
 */

#ifdef __APPLE__

#include "voxtral_mic.h"
#include <AudioToolbox/AudioToolbox.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>

#define MIC_SAMPLE_RATE   16000
#define MIC_NUM_BUFFERS   3
#define MIC_BUF_SAMPLES   1600   /* 100ms per AudioQueue buffer */
#define RING_CAPACITY     160000 /* 10 seconds at 16kHz */

static AudioQueueRef            queue;
static AudioQueueBufferRef      buffers[MIC_NUM_BUFFERS];
static pthread_mutex_t          ring_mutex = PTHREAD_MUTEX_INITIALIZER;
static float                    ring[RING_CAPACITY];
static int                      ring_head;  /* next write position */
static int                      ring_count; /* samples in ring */
static int                      running;

/* AudioQueue input callback — runs on AudioQueue's own thread */
static void mic_callback(void *userdata,
                          AudioQueueRef inAQ,
                          AudioQueueBufferRef inBuffer,
                          const AudioTimeStamp *inStartTime,
                          UInt32 inNumberPacketDescriptions,
                          const AudioStreamPacketDescription *inPacketDescs) {
    (void)userdata; (void)inStartTime;
    (void)inNumberPacketDescriptions; (void)inPacketDescs;

    int16_t *raw = (int16_t *)inBuffer->mAudioData;
    int n = (int)(inBuffer->mAudioDataByteSize / sizeof(int16_t));

    pthread_mutex_lock(&ring_mutex);
    for (int i = 0; i < n; i++) {
        ring[ring_head] = raw[i] / 32768.0f;
        ring_head = (ring_head + 1) % RING_CAPACITY;
        if (ring_count < RING_CAPACITY)
            ring_count++;
        /* If full, oldest sample is silently overwritten (ring_head advances
         * past ring_tail implicitly — we only track count). */
    }
    pthread_mutex_unlock(&ring_mutex);

    /* Re-enqueue buffer for next capture */
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL);
}

int vox_mic_start(void) {
    AudioStreamBasicDescription fmt = {0};
    fmt.mSampleRate       = MIC_SAMPLE_RATE;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel   = 16;
    fmt.mChannelsPerFrame = 1;
    fmt.mBytesPerFrame    = 2;
    fmt.mFramesPerPacket  = 1;
    fmt.mBytesPerPacket   = 2;

    OSStatus err = AudioQueueNewInput(&fmt, mic_callback, NULL,
                                       NULL, kCFRunLoopCommonModes, 0, &queue);
    if (err != noErr) {
        fprintf(stderr, "AudioQueueNewInput failed: %d\n", (int)err);
        return -1;
    }

    /* Allocate and enqueue capture buffers */
    UInt32 buf_bytes = MIC_BUF_SAMPLES * sizeof(int16_t);
    for (int i = 0; i < MIC_NUM_BUFFERS; i++) {
        AudioQueueAllocateBuffer(queue, buf_bytes, &buffers[i]);
        AudioQueueEnqueueBuffer(queue, buffers[i], 0, NULL);
    }

    ring_head = 0;
    ring_count = 0;
    running = 1;

    err = AudioQueueStart(queue, NULL);
    if (err != noErr) {
        fprintf(stderr, "AudioQueueStart failed: %d\n", (int)err);
        AudioQueueDispose(queue, true);
        return -1;
    }

    return 0;
}

int vox_mic_read(float *out, int max_samples) {
    pthread_mutex_lock(&ring_mutex);
    int n = ring_count < max_samples ? ring_count : max_samples;
    if (n > 0) {
        /* Read position is (ring_head - ring_count + RING_CAPACITY) % RING_CAPACITY */
        int tail = (ring_head - ring_count + RING_CAPACITY) % RING_CAPACITY;
        for (int i = 0; i < n; i++) {
            out[i] = ring[(tail + i) % RING_CAPACITY];
        }
        ring_count -= n;
    }
    pthread_mutex_unlock(&ring_mutex);
    return n;
}

int vox_mic_read_available(void) {
    pthread_mutex_lock(&ring_mutex);
    int n = ring_count;
    pthread_mutex_unlock(&ring_mutex);
    return n;
}

void vox_mic_stop(void) {
    if (!running) return;
    running = 0;
    AudioQueueStop(queue, true);
    AudioQueueDispose(queue, true);
}

#else /* !__APPLE__ */

#include "voxtral_mic.h"
#include <stdio.h>

int vox_mic_start(void) {
    fprintf(stderr, "Microphone capture is not supported on this platform\n");
    return -1;
}

int vox_mic_read(float *out, int max_samples) {
    (void)out; (void)max_samples;
    return 0;
}

int vox_mic_read_available(void) { return 0; }
void vox_mic_stop(void) {}

#endif
