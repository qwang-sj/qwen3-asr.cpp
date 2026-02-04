#include "audio_encoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, 
                         std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path.c_str());
        return false;
    }
    
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fclose(f);
        return false;
    }
    
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fprintf(stderr, "Invalid NPY magic: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    
    uint8_t major, minor;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16;
        if (fread(&len16, 2, 1, f) != 1) {
            fclose(f);
            return false;
        }
        header_len = len16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) {
            fclose(f);
            return false;
        }
    }
    
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f);
        return false;
    }
    header[header_len] = '\0';
    
    std::string header_str(header.data());
    
    bool fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
    
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) {
        fprintf(stderr, "Failed to find shape in header: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    shape_start += 10;
    
    size_t shape_end = header_str.find(")", shape_start);
    if (shape_end == std::string::npos) {
        fclose(f);
        return false;
    }
    
    std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) {
            pos++;
        }
        if (pos >= shape_str.size()) break;
        
        int64_t dim = 0;
        while (pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9') {
            dim = dim * 10 + (shape_str[pos] - '0');
            pos++;
        }
        shape.push_back(dim);
    }
    
    int64_t total_elements = 1;
    for (auto d : shape) {
        total_elements *= d;
    }
    
    data.resize(total_elements);
    
    if (fread(data.data(), sizeof(float), total_elements, f) != (size_t)total_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    
    fclose(f);
    
    if (fortran_order && shape.size() == 2) {
        std::vector<float> transposed(total_elements);
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                transposed[i * cols + j] = data[j * rows + i];
            }
        }
        data = std::move(transposed);
    }
    
    return true;
}

static float compute_max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::max();
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static float compute_mean_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return std::numeric_limits<float>::max();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return static_cast<float>(sum / a.size());
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-asr-0.6b-f16.gguf";
    std::string mel_path = "tests/reference/mel.npy";
    std::string ref_path = "tests/reference/audio_features.npy";
    float tolerance = 2e-2f;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mel") == 0 && i + 1 < argc) {
            mel_path = argv[++i];
        } else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --model <path>     Path to GGUF model (default: models/qwen3-asr-0.6b-f16.gguf)\n");
            printf("  --mel <path>       Path to mel spectrogram NPY (default: tests/reference/mel.npy)\n");
            printf("  --ref <path>       Path to reference output NPY (default: tests/reference/audio_features.npy)\n");
            printf("  --tolerance <val>  Max allowed difference (default: 1e-3)\n");
            return 0;
        }
    }
    
    printf("Loading mel spectrogram from: %s\n", mel_path.c_str());
    std::vector<float> mel_data;
    std::vector<int64_t> mel_shape;
    if (!load_npy_f32(mel_path, mel_data, mel_shape)) {
        fprintf(stderr, "Failed to load mel spectrogram\n");
        return 1;
    }
    
    if (mel_shape.size() != 2) {
        fprintf(stderr, "Expected 2D mel spectrogram, got %zu dimensions\n", mel_shape.size());
        return 1;
    }
    
    int n_mel = static_cast<int>(mel_shape[0]);
    int n_frames = static_cast<int>(mel_shape[1]);
    printf("Mel spectrogram shape: [%d, %d]\n", n_mel, n_frames);
    
    printf("Loading reference output from: %s\n", ref_path.c_str());
    std::vector<float> ref_data;
    std::vector<int64_t> ref_shape;
    if (!load_npy_f32(ref_path, ref_data, ref_shape)) {
        fprintf(stderr, "Failed to load reference output\n");
        return 1;
    }
    
    if (ref_shape.size() != 2) {
        fprintf(stderr, "Expected 2D reference output, got %zu dimensions\n", ref_shape.size());
        return 1;
    }
    
    printf("Reference output shape: [%lld, %lld]\n", 
           (long long)ref_shape[0], (long long)ref_shape[1]);
    
    printf("Loading model from: %s\n", model_path.c_str());
    qwen3_asr::AudioEncoder encoder;
    if (!encoder.load_model(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", encoder.get_error().c_str());
        return 1;
    }
    
    const auto & hp = encoder.get_hparams();
    printf("Model loaded:\n");
    printf("  encoder_layers: %d\n", hp.n_encoder_layers);
    printf("  d_model: %d\n", hp.d_model);
    printf("  attention_heads: %d\n", hp.n_attention_heads);
    printf("  ffn_dim: %d\n", hp.ffn_dim);
    printf("  conv_channels: %d\n", hp.conv_channels);
    printf("  n_mel_bins: %d\n", hp.n_mel_bins);
    
    printf("Running audio encoder...\n");
    std::vector<float> output;
    if (!encoder.encode(mel_data.data(), n_mel, n_frames, output)) {
        fprintf(stderr, "Failed to encode: %s\n", encoder.get_error().c_str());
        return 1;
    }
    
    const auto & thp = encoder.get_text_hparams();
    int64_t out_seq_len = output.size() / thp.hidden_size;
    printf("Output shape: [%lld, %d]\n", (long long)out_seq_len, thp.hidden_size);
    
    if (output.size() != ref_data.size()) {
        fprintf(stderr, "Output size mismatch: got %zu, expected %zu\n", 
                output.size(), ref_data.size());
        
        printf("\nFirst 10 output values:\n");
        for (size_t i = 0; i < std::min(output.size(), size_t(10)); ++i) {
            printf("  [%zu] = %f\n", i, output[i]);
        }
        
        printf("\nFirst 10 reference values:\n");
        for (size_t i = 0; i < std::min(ref_data.size(), size_t(10)); ++i) {
            printf("  [%zu] = %f\n", i, ref_data[i]);
        }
        
        return 1;
    }
    
    float max_diff = compute_max_abs_diff(output, ref_data);
    float mean_diff = compute_mean_abs_diff(output, ref_data);
    
    printf("\nComparison results:\n");
    printf("  Max absolute difference: %e\n", max_diff);
    printf("  Mean absolute difference: %e\n", mean_diff);
    printf("  Tolerance: %e\n", tolerance);
    
    if (max_diff <= tolerance) {
        printf("\nTEST PASSED: Output matches reference within tolerance\n");
        return 0;
    } else {
        printf("\nTEST FAILED: Output differs from reference\n");
        
        printf("\nFirst 10 differences:\n");
        int count = 0;
        for (size_t i = 0; i < output.size() && count < 10; ++i) {
            float diff = std::abs(output[i] - ref_data[i]);
            if (diff > tolerance) {
                printf("  [%zu] output=%f, ref=%f, diff=%e\n", 
                       i, output[i], ref_data[i], diff);
                count++;
            }
        }
        
        return 1;
    }
}
