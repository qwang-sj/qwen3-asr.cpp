# Qwen3-ASR GGML

A high-performance C++ implementation of Qwen3-ASR and Qwen3-ForcedAligner using the GGML tensor library. Supports automatic speech recognition (ASR) and forced alignment with word-level timestamps.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio files to text
- **Forced Alignment**: Align reference text to audio with word-level timestamps
- **Quantization Support**: Q8_0 quantization for reduced memory usage (~40% smaller)
- **Multi-threaded**: Configurable thread count for parallel processing
- **Multilingual**: Supports 30+ languages including Chinese, English, Japanese, Korean, German, French, Spanish, and more
- **Pure C++**: No Python runtime required for inference

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b-f16.gguf` | ~1.8 GB | ASR model, F16 precision |
| `qwen3-asr-0.6b-q8_0.gguf` | ~1.3 GB | ASR model, Q8_0 quantized |
| `qwen3-forced-aligner-0.6b-f16.gguf` | ~1.8 GB | Forced alignment model |

## Requirements

- CMake 3.14+
- C++17 compatible compiler (GCC 8+, Clang 7+)
- GGML library (included as dependency)

## Building

```bash
# Clone the repository
git clone https://github.com/your-repo/qwen-3-asr-ggml.git
cd qwen-3-asr-ggml

# Build GGML first (if not already built)
cd /root/ggml
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build qwen3-asr-ggml
cd /root/qwen-3-asr-ggml
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

### Transcription

```bash
# Basic transcription
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav

# With quantized model (faster, less memory)
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav

# Save output to file
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -o transcript.txt

# Multi-threaded processing
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -t 8
```

### Forced Alignment

```bash
# Align text to audio (outputs JSON with word timestamps)
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world, this is a test."

# Save alignment to file
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world" \
    -o alignment.json
```

### Output Formats

**Transcription** outputs plain text:
```
Hello world, this is a test transcription.
```

**Forced Alignment** outputs JSON with word-level timestamps:
```json
{
  "words": [
    {"word": "Hello", "start": 0.000, "end": 0.320},
    {"word": "world", "start": 0.320, "end": 0.640}
  ]
}
```

## Model Conversion

Convert HuggingFace models to GGUF format:

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Convert ASR model (F16)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-f16.gguf \
    --type f16

# Convert ASR model (Q8_0 quantized)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-q8_0.gguf \
    --type q8_0

# Convert ForcedAligner model
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ForcedAligner-0.6B \
    --output models/qwen3-forced-aligner-0.6b-f16.gguf \
    --type f16
```

## Supported Languages

The model supports 30+ languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese (Mandarin) | zh | English | en |
| Cantonese | yue | Japanese | ja |
| Korean | ko | German | de |
| French | fr | Spanish | es |
| Italian | it | Portuguese | pt |
| Russian | ru | Arabic | ar |
| Hindi | hi | Thai | th |
| Vietnamese | vi | Indonesian | id |
| Malay | ms | Turkish | tr |
| Polish | pl | Dutch | nl |
| Swedish | sv | Norwegian | no |
| Danish | da | Finnish | fi |
| Greek | el | Czech | cs |
| Hungarian | hu | Romanian | ro |
| Ukrainian | uk | Hebrew | he |

## Audio Requirements

- Format: WAV (PCM)
- Sample rate: 16 kHz
- Channels: Mono
- Bit depth: 16-bit

Convert audio with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Running Tests

```bash
cd /root/qwen-3-asr-ggml

# Run all tests
./tests/run_all_tests.sh

# Run individual tests
./build/test_mel
./build/test_encoder --model models/qwen3-asr-0.6b-f16.gguf
./build/test_decoder --model models/qwen3-asr-0.6b-f16.gguf
```

## Performance Profiling

Build with timing instrumentation enabled to profile performance:

```bash
mkdir -p build && cd build
cmake -DQWEN3_ASR_TIMING=ON ..
make -j$(nproc)

# Run with --profile flag to see detailed timing breakdown
./qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav --profile
```

### Timing Breakdown (30s audio, F16 model, 4 threads)

| Section | Time (ms) | Calls | Avg (ms) |
|---------|-----------|-------|----------|
| mel_spectrogram | 6,703 | 1 | 6,703 |
| audio_encoding.conv_chunk | 1,726 | 30 | 58 |
| audio_encoding.transformer | 2,353 | 1 | 2,353 |
| decode.initial_forward | 5,847 | 1 | 5,847 |
| decode.token | 1,170 | 19 | 62 |
| **Total** | **~18,000** | - | - |

### Performance Notes

- **Mel spectrogram** (~38%): FFT computation is CPU-intensive. Multi-threading helps.
- **Audio encoding** (~23%): Conv layers process 30 chunks of 100 mel frames each.
- **Initial decode** (~33%): First forward pass processes all input tokens with audio injection.
- **Token generation** (~6%): Each subsequent token takes ~62ms (KV cache enabled).

For production builds without timing overhead:
```bash
cmake ..  # Without -DQWEN3_ASR_TIMING=ON
make -j$(nproc)
```

## Project Structure

```
qwen-3-asr-ggml/
├── src/
│   ├── main.cpp              # CLI entry point
│   ├── qwen3_asr.cpp/h       # High-level ASR API
│   ├── forced_aligner.cpp/h  # Forced alignment implementation
│   ├── audio_encoder.cpp/h   # Audio feature encoder
│   ├── text_decoder.cpp/h    # Text decoder (Qwen2 architecture)
│   ├── mel_spectrogram.cpp/h # Mel spectrogram computation
│   ├── audio_injection.cpp/h # Audio-text embedding injection
│   └── gguf_loader.cpp/h     # GGUF model loading
├── tests/
│   ├── test_mel.cpp          # Mel spectrogram tests
│   ├── test_encoder.cpp      # Audio encoder tests
│   ├── test_decoder.cpp      # Text decoder tests
│   └── reference/            # Reference data for validation
├── scripts/
│   └── convert_hf_to_gguf.py # Model conversion script
├── models/                   # GGUF model files
├── docs/
│   └── usage.md              # Detailed CLI documentation
└── CMakeLists.txt
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) - Tensor library for machine learning
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Original model by Alibaba
