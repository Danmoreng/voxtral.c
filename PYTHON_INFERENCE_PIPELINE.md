# Voxtral Python Reference Inference Pipeline Documentation

This document describes the architecture and data flow of the `python_simple_implementation.py` reference implementation for the Voxtral Realtime 4B model.

## Overview

The Python implementation serves as a self-contained, high-level reference for the Voxtral inference process. It uses PyTorch for tensor operations and `safetensors` for efficient weight loading, avoiding heavy dependencies like `vLLM` while maintaining bit-perfect compatibility with the official model.

## 1. Audio Processing & Feature Extraction

The pipeline begins by converting raw audio into a format suitable for the transformer-based encoder.

### A. Preprocessing
- **Resampling**: Audio is resampled to 16,000 Hz.
- **Normalization**: Mono audio is converted to float32.
- **Padding**: Audio is padded for "offline streaming" mode:
    - **Left Pad**: 32 tokens (40,960 samples) of silence.
    - **Right Pad**: Aligned to 1280 samples + 17 tokens (21,760 samples) extra.

### B. Mel Spectrogram
- **STFT**: Uses a Hann window of size 400 and a hop length of 160.
- **Filter Bank**: A Slaney-style mel filter bank with 128 bins is applied to the STFT magnitudes.
- **Log Scaling**: The magnitudes are log-scaled and normalized: `(log10(mel) + 4.0) / 4.0`.

## 2. Encoder Pipeline

The encoder processes mel frames into a sequence of acoustic embeddings.

### A. Convolutional Stem
- **Causal Conv1d**: Two layers of causal convolutions with GELU activation.
    - Conv0: `stride=1`, `kernel_size=3`.
    - Conv1: `stride=2`, `kernel_size=3`.
- **Downsampling**: The stem effectively reduces the frame rate before entering the transformer.

### B. Transformer Layers (32 Layers)
- **Architecture**: 32 layers of Mistral-style decoder blocks (MHA + SwiGLU) used as an incremental encoder.
- **RoPE**: Interleaved Rotary Positional Embeddings are applied to queries and keys.
- **Attention**: Causal multi-head attention with a sliding window of 750 frames.
- **Normalization**: Pre-attention and pre-FFN `RMSNorm`.

## 3. Multimodal Adapter

The adapter bridges the gap between the encoder's acoustic space (dim 1280) and the decoder's linguistic space (dim 3072).

- **Downsampling**: Every 4 encoder frames are concatenated into a single 5120-dim vector (512 tokens -> 128 tokens).
- **Projection**: A simple 2-layer MLP (Linear -> GELU -> Linear) maps the downsampled features to 3072-dim embeddings.

## 4. Decoder Pipeline

The decoder performs the final transcription by combining acoustic and linguistic information.

### A. Time Conditioning (AdaNorm)
A unique feature of Voxtral is the per-layer time conditioning:
- **Embedding**: A scalar delay value is converted into a sinusoidal embedding (dim 3072).
- **Modulation**: This embedding passes through an MLP to generate a scaling factor applied to the `RMSNorm` outputs in every decoder layer: `h = h * (1 + ada_scale)`.

### B. Prefill & Generation
The decoder operates in a multimodal space where **Audio Embeddings** and **Token Embeddings** are summed:
1.  **Prefix Prefill**: Processes the initial prompt IDs (BOS, Padding) summed with the first $L$ audio embeddings.
2.  **Iterative Sampling**: 
    - At each step, the next token is sampled from the logits.
    - The new token is embedded and added to the *next available audio embedding* from the adapter.
    - This "audio-driven" decoding continues until `[EOS]` is reached or the audio span ends.

### C. Tied Embeddings
- The decoder uses tied weights: the output projection (logits) uses the transpose of the input token embedding matrix.

## 5. Summary of Pipeline Flow

| Component | Operation | Input Shape | Output Shape |
| :--- | :--- | :--- | :--- |
| **Audio** | STFT + Mel | `[samples]` | `[128, frames]` |
| **Encoder** | Conv + Transformer | `[128, frames]` | `[seq, 1280]` |
| **Adapter** | Concatenate + MLP | `[seq, 1280]` | `[seq/4, 3072]` |
| **Decoder** | Prefill + KV Cache | `[1, 3072]` (+ audio) | `[1, vocab]` |

## 6. Implementation Notes

- **BF16**: While the code has a `USE_BF16` flag, it defaults to FP32 for reliability on standard hardware.
- **Tokenizer**: Implements a minimal `Tekken` tokenizer by decoding base64 byte sequences from a JSON vocab file.
- **Compatibility**: The implementation mirrors `vLLM`'s logic for causal convolutions and sliding window attention.
