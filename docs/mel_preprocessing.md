# Mel Spectrogram Preprocessing for Qwen3-ASR

This document describes the mel spectrogram preprocessing pipeline used by Qwen3-ASR, with comparison to whisper.cpp implementation.

## Overview

Qwen3-ASR uses **WhisperFeatureExtractor** for audio preprocessing, which means it follows the same mel spectrogram computation as OpenAI's Whisper model. This is excellent news for our C++ implementation since we can leverage whisper.cpp's battle-tested code.

## Parameters

### Qwen3-ASR Parameters (from `preprocessor_config.json`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `feature_size` | 128 | Number of mel frequency bins |
| `n_fft` | 400 | FFT window size (25ms at 16kHz) |
| `hop_length` | 160 | Hop length between frames (10ms at 16kHz) |
| `chunk_length` | 30 | Audio chunk length in seconds |
| `n_samples` | 480000 | Samples per chunk (30s × 16kHz) |
| `nb_max_frames` | 3000 | Maximum mel frames (480000 / 160) |
| `padding_side` | "right" | Padding direction |
| `padding_value` | 0.0 | Value used for padding |
| `dither` | 0.0 | No dithering applied |

### whisper.cpp Parameters (from `whisper.h`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WHISPER_SAMPLE_RATE` | 16000 | Sample rate in Hz |
| `WHISPER_N_FFT` | 400 | FFT window size |
| `WHISPER_HOP_LENGTH` | 160 | Hop length |
| `WHISPER_CHUNK_SIZE` | 30 | Chunk size in seconds |

### Comparison

| Parameter | Qwen3-ASR | whisper.cpp | Match? |
|-----------|-----------|-------------|--------|
| Sample rate | 16000 Hz | 16000 Hz | ✅ |
| n_fft | 400 | 400 | ✅ |
| hop_length | 160 | 160 | ✅ |
| n_mels | 128 | 80/128* | ✅ |
| chunk_length | 30s | 30s | ✅ |

*whisper.cpp supports both 80 and 128 mel bins depending on model.

## Algorithm Steps

### 1. Audio Padding

```cpp
// Stage 1: Pad 30 seconds of zeros at the end (for chunk processing)
int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;  // 480,000 samples

// Stage 2: Reflective padding at boundaries
int64_t stage_2_pad = frame_size / 2;  // 200 samples

// Total padded size
samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);

// Copy samples with offset for reflective padding
std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

// Pad zeros at end
std::fill(samples_padded.begin() + n_samples + stage_2_pad, 
          samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

// Reflective pad at beginning (mirror first 200 samples)
std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
```

### 2. Hann Window

```cpp
// Periodic Hann window (matches PyTorch's torch.hann_window with periodic=True)
void fill_hann_window(int length, bool periodic, float* output) {
    int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}
```

Reference: https://pytorch.org/docs/stable/generated/torch.hann_window.html

### 3. STFT (Short-Time Fourier Transform)

```cpp
// For each frame
for (int i = 0; i < mel.n_len; i++) {
    const int offset = i * frame_step;  // frame_step = 160
    
    // Apply Hann window
    for (int j = 0; j < frame_size; j++) {  // frame_size = 400
        fft_in[j] = hann[j] * samples[offset + j];
    }
    
    // Compute FFT
    fft(fft_in.data(), frame_size, fft_out.data());
    
    // Calculate power spectrum (magnitude squared)
    // n_fft = 201 (only positive frequencies: 0 to Nyquist)
    for (int j = 0; j < n_fft; j++) {
        fft_out[j] = fft_out[2*j + 0] * fft_out[2*j + 0] + 
                     fft_out[2*j + 1] * fft_out[2*j + 1];
    }
}
```

**Key insight**: `n_fft = 1 + (WHISPER_N_FFT / 2) = 201` (only positive frequencies due to real-valued input)

### 4. Mel Filterbank Application

```cpp
// Apply mel filterbank (stored in model file)
// filters.data has shape [n_mel, n_fft] = [128, 201]
for (int j = 0; j < mel.n_mel; j++) {
    double sum = 0.0;
    for (int k = 0; k < n_fft; k++) {
        sum += fft_out[k] * filters.data[j * n_fft + k];
    }
    // Log scale with floor
    sum = log10(std::max(sum, 1e-10));
    mel.data[j * mel.n_len + i] = sum;
}
```

### 5. Normalization (Critical!)

```cpp
// Find maximum value
double mmax = -1e20;
for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
    if (mel.data[i] > mmax) {
        mmax = mel.data[i];
    }
}

// Clamp to max - 8.0 (dynamic range limiting)
mmax -= 8.0;

for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
    // Clamp minimum
    if (mel.data[i] < mmax) {
        mel.data[i] = mmax;
    }
    
    // Normalize: (mel + 4.0) / 4.0
    mel.data[i] = (mel.data[i] + 4.0) / 4.0;
}
```

**Normalization formula**: `normalized = (log10_mel + 4.0) / 4.0`

This maps the typical log-mel range to approximately [0, 1]:
- log10(1e-10) = -10 → clamped to max-8, then normalized
- Typical speech values around -2 to 0 → normalized to 0.5 to 1.0

## Frame Count Calculation

```cpp
// Number of mel frames
mel.n_len = (samples_padded.size() - frame_size) / frame_step;

// For 30 seconds of audio:
// samples_padded.size() = 480000 + 480000 + 400 = 960400
// mel.n_len = (960400 - 400) / 160 = 6000 frames

// Original length (without 30s padding)
mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
```

## Mel Filterbank

The mel filterbank is stored in the model file and loaded at initialization:

```cpp
struct whisper_filters {
    int32_t n_mel;   // 128 for Qwen3-ASR
    int32_t n_fft;   // 201 (positive frequencies only)
    std::vector<float> data;  // [n_mel × n_fft] = [128 × 201]
};
```

The filterbank converts linear frequency bins to mel scale using triangular filters:
- **fmin**: 0 Hz (typically)
- **fmax**: 8000 Hz (Nyquist frequency at 16kHz sample rate)
- **n_mels**: 128 triangular filters

## Data Layout

The mel spectrogram is stored in **mel-major order**:

```cpp
// mel.data[j * mel.n_len + i] where:
//   j = mel bin index (0 to 127)
//   i = time frame index (0 to n_len-1)

// Shape: [n_mel, n_len] = [128, 3000] for typical inference
```

## Reference Implementation

For exact parity with HuggingFace, we have reference mel output at:
- `tests/reference/mel.npy` - Shape: [128, 3000]

## Implementation Recommendations

### For C++ Implementation

1. **Reuse whisper.cpp code**: The `log_mel_spectrogram` function in whisper.cpp is well-optimized and matches Qwen3-ASR's preprocessing exactly.

2. **Key functions to port**:
   - `fill_hann_window()` - Periodic Hann window
   - `fft()` - Cooley-Tukey FFT (or use a library like FFTW/pocketfft)
   - `log_mel_spectrogram_worker_thread()` - Main computation
   - Normalization step

3. **Mel filterbank**: Either:
   - Load from model file (like whisper.cpp does)
   - Generate programmatically using mel scale formula

4. **Threading**: whisper.cpp uses multi-threaded processing for mel computation. Consider this for performance.

### Differences to Watch

| Aspect | whisper.cpp | Potential Qwen3-ASR Difference |
|--------|-------------|-------------------------------|
| Padding | 30s zeros + reflective | Verify same behavior |
| Normalization | (mel + 4.0) / 4.0 | Verify same formula |
| Filterbank | From model file | May need to generate |

## Verification Strategy

1. Generate mel spectrogram from test audio using our C++ implementation
2. Compare against `tests/reference/mel.npy` 
3. Acceptable tolerance: < 1e-5 absolute difference

## Code References

- whisper.cpp implementation: `/root/whisper.cpp/src/whisper.cpp` lines 3000-3260
- Qwen3-ASR config: `/root/models/Qwen3-ASR-0.6B/preprocessor_config.json`
- OpenAI Whisper Python: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
