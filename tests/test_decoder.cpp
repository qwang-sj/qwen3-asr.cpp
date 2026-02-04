#include "text_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

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
    std::string ref_path = "";
    float tolerance = 1e-2f;
    bool run_basic_test = true;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --model <path>     Path to GGUF model (default: models/qwen3-asr-0.6b-f16.gguf)\n");
            printf("  --ref <path>       Path to reference logits NPY (optional)\n");
            printf("  --tolerance <val>  Max allowed difference (default: 1e-2)\n");
            return 0;
        }
    }
    
    printf("Loading model from: %s\n", model_path.c_str());
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto & cfg = decoder.get_config();
    printf("Model loaded:\n");
    printf("  vocab_size: %d\n", cfg.vocab_size);
    printf("  hidden_size: %d\n", cfg.hidden_size);
    printf("  n_decoder_layers: %d\n", cfg.n_decoder_layers);
    printf("  n_attention_heads: %d\n", cfg.n_attention_heads);
    printf("  n_key_value_heads: %d\n", cfg.n_key_value_heads);
    printf("  intermediate_size: %d\n", cfg.intermediate_size);
    printf("  head_dim: %d\n", cfg.head_dim);
    printf("  rms_norm_eps: %e\n", cfg.rms_norm_eps);
    printf("  rope_theta: %e\n", cfg.rope_theta);
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to initialize KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("KV cache initialized (n_ctx=512)\n");
    
    if (run_basic_test) {
        printf("\nRunning basic forward pass test...\n");
        
        std::vector<int32_t> test_tokens = {
            cfg.audio_start_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_end_token_id
        };
        
        printf("Input tokens: [");
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            printf("%d%s", test_tokens[i], i < test_tokens.size() - 1 ? ", " : "");
        }
        printf("]\n");
        
        std::vector<float> logits;
        if (!decoder.forward(test_tokens.data(), test_tokens.size(), 0, logits)) {
            fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
            return 1;
        }
        
        int64_t expected_size = test_tokens.size() * cfg.vocab_size;
        if ((int64_t)logits.size() != expected_size) {
            fprintf(stderr, "Output size mismatch: got %zu, expected %lld\n", 
                    logits.size(), (long long)expected_size);
            return 1;
        }
        
        printf("Output shape: [%zu, %d]\n", test_tokens.size(), cfg.vocab_size);
        
        printf("\nFirst 10 logits (position 0):\n");
        for (int i = 0; i < 10; ++i) {
            printf("  logits[0][%d] = %f\n", i, logits[i]);
        }
        
        printf("\nLast 10 logits (position 0):\n");
        for (int i = cfg.vocab_size - 10; i < cfg.vocab_size; ++i) {
            printf("  logits[0][%d] = %f\n", i, logits[i]);
        }
        
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < cfg.vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        printf("\nArgmax at position 0: token %d with logit %f\n", max_idx, max_val);
        
        float sum = 0.0f;
        for (int i = 0; i < cfg.vocab_size; ++i) {
            sum += logits[i];
        }
        float mean = sum / cfg.vocab_size;
        
        float var = 0.0f;
        for (int i = 0; i < cfg.vocab_size; ++i) {
            float diff = logits[i] - mean;
            var += diff * diff;
        }
        var /= cfg.vocab_size;
        float std_dev = sqrtf(var);
        
        printf("Logits stats (position 0): mean=%f, std=%f\n", mean, std_dev);
        
        bool has_nan = false;
        bool has_inf = false;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isnan(logits[i])) has_nan = true;
            if (std::isinf(logits[i])) has_inf = true;
        }
        
        if (has_nan || has_inf) {
            fprintf(stderr, "ERROR: Output contains NaN=%d, Inf=%d\n", has_nan, has_inf);
            return 1;
        }
        
        printf("\nBasic test PASSED: Forward pass completed successfully\n");
    }
    
    if (!ref_path.empty()) {
        printf("\nLoading reference logits from: %s\n", ref_path.c_str());
        std::vector<float> ref_data;
        std::vector<int64_t> ref_shape;
        if (!load_npy_f32(ref_path, ref_data, ref_shape)) {
            fprintf(stderr, "Failed to load reference logits\n");
            return 1;
        }
        
        printf("Reference shape: [");
        for (size_t i = 0; i < ref_shape.size(); ++i) {
            printf("%lld%s", (long long)ref_shape[i], i < ref_shape.size() - 1 ? ", " : "");
        }
        printf("]\n");
        
        std::vector<int32_t> test_tokens = {
            cfg.audio_start_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_pad_token_id,
            cfg.audio_end_token_id
        };
        
        decoder.clear_kv_cache();
        
        std::vector<float> logits;
        if (!decoder.forward(test_tokens.data(), test_tokens.size(), 0, logits)) {
            fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
            return 1;
        }
        
        if (logits.size() != ref_data.size()) {
            fprintf(stderr, "Size mismatch: got %zu, expected %zu\n", 
                    logits.size(), ref_data.size());
            return 1;
        }
        
        float max_diff = compute_max_abs_diff(logits, ref_data);
        float mean_diff = compute_mean_abs_diff(logits, ref_data);
        
        printf("\nComparison results:\n");
        printf("  Max absolute difference: %e\n", max_diff);
        printf("  Mean absolute difference: %e\n", mean_diff);
        printf("  Tolerance: %e\n", tolerance);
        
        if (max_diff <= tolerance) {
            printf("\nTEST PASSED: Output matches reference within tolerance\n");
        } else {
            printf("\nTEST FAILED: Output differs from reference\n");
            
            printf("\nFirst 10 differences:\n");
            int count = 0;
            for (size_t i = 0; i < logits.size() && count < 10; ++i) {
                float diff = std::abs(logits[i] - ref_data[i]);
                if (diff > tolerance) {
                    printf("  [%zu] output=%f, ref=%f, diff=%e\n", 
                           i, logits[i], ref_data[i], diff);
                    count++;
                }
            }
            
            return 1;
        }
    }
    
    printf("\nAll tests PASSED\n");
    return 0;
}
