#include "forced_aligner.h"
#include "mel_spectrogram.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>

#define QWEN3_FA_MAX_NODES 8192

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void compute_sinusoidal_pe(float * pe, int n_ctx, int d_model) {
    const int half_dim = d_model / 2;
    for (int pos = 0; pos < n_ctx; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float div_term = expf(-logf(10000.0f) * i / (half_dim - 1));
            float angle = pos * div_term;
            pe[pos * d_model + i] = sinf(angle);
            pe[pos * d_model + half_dim + i] = cosf(angle);
        }
    }
}

ForcedAligner::ForcedAligner() = default;

ForcedAligner::~ForcedAligner() {
    free_kv_cache();
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
    free_forced_aligner_model(model_);
}

bool ForcedAligner::load_model(const std::string & model_path) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_hparams(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_forced_aligner_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_vocab(ctx)) {
        free_forced_aligner_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }
    
    std::vector<ggml_backend_t> backends = { state_.backend };
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, 1, QWEN3_FA_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_FA_MAX_NODES + ggml_graph_overhead());
    
    model_loaded_ = true;
    return true;
}

bool ForcedAligner::parse_hparams(struct gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & hp = model_.hparams;
    
    hp.audio_encoder_layers = get_u32("qwen3-asr.audio.encoder.layer_count", 24);
    hp.audio_d_model = get_u32("qwen3-asr.audio.encoder.embedding_length", 1024);
    hp.audio_attention_heads = get_u32("qwen3-asr.audio.encoder.attention.head_count", 16);
    hp.audio_ffn_dim = get_u32("qwen3-asr.audio.encoder.feed_forward_length", 4096);
    hp.audio_num_mel_bins = get_u32("qwen3-asr.audio.num_mel_bins", 128);
    hp.audio_conv_channels = get_u32("qwen3-asr.audio.conv_channels", 480);
    
    hp.text_decoder_layers = get_u32("qwen3-asr.block_count", 28);
    hp.text_hidden_size = get_u32("qwen3-asr.embedding_length", 1024);
    hp.text_attention_heads = get_u32("qwen3-asr.attention.head_count", 16);
    hp.text_kv_heads = get_u32("qwen3-asr.attention.head_count_kv", 8);
    hp.text_intermediate_size = get_u32("qwen3-asr.feed_forward_length", 3072);
    hp.text_head_dim = get_u32("qwen3-asr.attention.key_length", 128);
    hp.text_rms_norm_eps = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    hp.text_rope_theta = get_f32("qwen3-asr.rope.freq_base", 1000000.0f);
    hp.vocab_size = get_u32("qwen3-asr.vocab_size", 152064);
    
    hp.classify_num = get_u32("qwen3-asr.classify_num", 5000);
    hp.timestamp_token_id = get_u32("qwen3-asr.timestamp_token_id", 151705);
    hp.audio_start_token_id = get_u32("qwen3-asr.audio.start_token_id", 151669);
    hp.audio_end_token_id = get_u32("qwen3-asr.audio.end_token_id", 151670);
    hp.audio_pad_token_id = get_u32("qwen3-asr.audio.pad_token_id", 151676);
    
    return true;
}

bool ForcedAligner::create_tensors(struct gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & hp = model_.hparams;
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model_.encoder_layers.resize(hp.audio_encoder_layers);
    model_.decoder_layers.resize(hp.text_decoder_layers);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        if (strstr(name, "encoder.conv1.weight")) {
            ne[0] = 3; ne[1] = 3; ne[2] = 1; ne[3] = hp.audio_conv_channels;
            n_dims = 4;
        } else if (strstr(name, "encoder.conv2.weight") || strstr(name, "encoder.conv3.weight")) {
            ne[0] = 3; ne[1] = 3; ne[2] = hp.audio_conv_channels; ne[3] = hp.audio_conv_channels;
            n_dims = 4;
        } else if (strstr(name, "encoder.conv1.bias") || strstr(name, "encoder.conv2.bias") || 
                   strstr(name, "encoder.conv3.bias")) {
            ne[0] = hp.audio_conv_channels;
            n_dims = 1;
        } else if (strstr(name, "encoder.conv_out.weight")) {
            ne[0] = hp.audio_conv_channels * 16;
            ne[1] = hp.audio_d_model;
            n_dims = 2;
        } else if (strstr(name, "audio.encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "audio.encoder.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < hp.audio_encoder_layers) {
                
                if (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") || 
                    strstr(name, "attn_v.weight") || strstr(name, "attn_out.weight")) {
                    ne[0] = hp.audio_d_model;
                    ne[1] = hp.audio_d_model;
                    n_dims = 2;
                } else if (strstr(name, "attn_q.bias") || strstr(name, "attn_k.bias") || 
                           strstr(name, "attn_v.bias") || strstr(name, "attn_out.bias") ||
                           strstr(name, "attn_norm.weight") || strstr(name, "attn_norm.bias")) {
                    ne[0] = hp.audio_d_model;
                    n_dims = 1;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = hp.audio_d_model;
                    ne[1] = hp.audio_ffn_dim;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = hp.audio_ffn_dim;
                    ne[1] = hp.audio_d_model;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.bias")) {
                    ne[0] = hp.audio_ffn_dim;
                    n_dims = 1;
                } else if (strstr(name, "ffn_down.bias") || strstr(name, "ffn_norm.weight") || 
                           strstr(name, "ffn_norm.bias")) {
                    ne[0] = hp.audio_d_model;
                    n_dims = 1;
                }
            }
        } else if (strstr(name, "encoder.ln_post.weight") || strstr(name, "encoder.ln_post.bias")) {
            ne[0] = hp.audio_d_model;
            n_dims = 1;
        } else if (strstr(name, "encoder.proj1.weight")) {
            ne[0] = hp.audio_d_model;
            ne[1] = hp.audio_d_model;
            n_dims = 2;
        } else if (strstr(name, "encoder.proj1.bias")) {
            ne[0] = hp.audio_d_model;
            n_dims = 1;
        } else if (strstr(name, "encoder.proj2.weight")) {
            ne[0] = hp.audio_d_model;
            ne[1] = hp.text_hidden_size;
            n_dims = 2;
        } else if (strstr(name, "encoder.proj2.bias")) {
            ne[0] = hp.text_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "token_embd.weight")) {
            ne[0] = hp.text_hidden_size;
            ne[1] = hp.vocab_size;
            n_dims = 2;
        } else if (strstr(name, "output_norm.weight")) {
            ne[0] = hp.text_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "output.weight") && !strstr(name, "attn_output.weight")) {
            ne[0] = hp.text_hidden_size;
            ne[1] = hp.classify_num;
            n_dims = 2;
        } else if (strstr(name, "blk.") && !strstr(name, "audio.encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < hp.text_decoder_layers) {
                
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = hp.text_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight") || strstr(name, "attn_k_norm.weight")) {
                    ne[0] = hp.text_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = hp.text_hidden_size;
                    ne[1] = hp.text_attention_heads * hp.text_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = hp.text_hidden_size;
                    ne[1] = hp.text_kv_heads * hp.text_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = hp.text_hidden_size;
                    ne[1] = hp.text_kv_heads * hp.text_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = hp.text_attention_heads * hp.text_head_dim;
                    ne[1] = hp.text_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = hp.text_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight") || strstr(name, "ffn_up.weight")) {
                    ne[0] = hp.text_hidden_size;
                    ne[1] = hp.text_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = hp.text_intermediate_size;
                    ne[1] = hp.text_hidden_size;
                    n_dims = 2;
                }
            }
        } else {
            continue;
        }
        
        if (n_dims == 0) continue;
        
        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;
        
        if (strstr(name, "encoder.conv1.weight")) {
            model_.conv2d1_w = tensor;
        } else if (strstr(name, "encoder.conv1.bias")) {
            model_.conv2d1_b = tensor;
        } else if (strstr(name, "encoder.conv2.weight")) {
            model_.conv2d2_w = tensor;
        } else if (strstr(name, "encoder.conv2.bias")) {
            model_.conv2d2_b = tensor;
        } else if (strstr(name, "encoder.conv3.weight")) {
            model_.conv2d3_w = tensor;
        } else if (strstr(name, "encoder.conv3.bias")) {
            model_.conv2d3_b = tensor;
        } else if (strstr(name, "encoder.conv_out.weight")) {
            model_.conv_out_w = tensor;
        } else if (strstr(name, "encoder.ln_post.weight")) {
            model_.ln_post_w = tensor;
        } else if (strstr(name, "encoder.ln_post.bias")) {
            model_.ln_post_b = tensor;
        } else if (strstr(name, "encoder.proj1.weight")) {
            model_.proj1_w = tensor;
        } else if (strstr(name, "encoder.proj1.bias")) {
            model_.proj1_b = tensor;
        } else if (strstr(name, "encoder.proj2.weight")) {
            model_.proj2_w = tensor;
        } else if (strstr(name, "encoder.proj2.bias")) {
            model_.proj2_b = tensor;
        } else if (strstr(name, "audio.encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "audio.encoder.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < hp.audio_encoder_layers) {
                auto & layer = model_.encoder_layers[layer_idx];
                
                if (strstr(name, "attn_q.weight")) layer.attn_q_w = tensor;
                else if (strstr(name, "attn_q.bias")) layer.attn_q_b = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k_w = tensor;
                else if (strstr(name, "attn_k.bias")) layer.attn_k_b = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v_w = tensor;
                else if (strstr(name, "attn_v.bias")) layer.attn_v_b = tensor;
                else if (strstr(name, "attn_out.weight")) layer.attn_out_w = tensor;
                else if (strstr(name, "attn_out.bias")) layer.attn_out_b = tensor;
                else if (strstr(name, "attn_norm.weight")) layer.attn_norm_w = tensor;
                else if (strstr(name, "attn_norm.bias")) layer.attn_norm_b = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up_w = tensor;
                else if (strstr(name, "ffn_up.bias")) layer.ffn_up_b = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down_w = tensor;
                else if (strstr(name, "ffn_down.bias")) layer.ffn_down_b = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm_w = tensor;
                else if (strstr(name, "ffn_norm.bias")) layer.ffn_norm_b = tensor;
            }
        } else if (strstr(name, "token_embd.weight")) {
            model_.token_embd = tensor;
        } else if (strstr(name, "output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "output.weight") && !strstr(name, "attn_output.weight")) {
            model_.classify_head_w = tensor;
        } else if (strstr(name, "blk.") && !strstr(name, "audio.encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < hp.text_decoder_layers) {
                auto & layer = model_.decoder_layers[layer_idx];
                
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        }
    }
    
    return true;
}

bool ForcedAligner::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        error_msg_ = "Failed to initialize CPU backend for loading";
        return false;
    }
    
    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, backend);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg_ = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = model_.tensors.find(name);
        if (it == model_.tensors.end()) {
            continue;
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg_ = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg_ = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    ggml_backend_free(backend);
    
    return true;
}

bool ForcedAligner::load_vocab(struct gguf_context * ctx) {
    int64_t tokens_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        error_msg_ = "Vocabulary not found in GGUF file";
        return false;
    }
    
    int64_t n_vocab = gguf_get_arr_n(ctx, tokens_idx);
    if (n_vocab <= 0) {
        error_msg_ = "Empty vocabulary in GGUF file";
        return false;
    }
    
    model_.vocab.resize(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        model_.vocab[i] = gguf_get_arr_str(ctx, tokens_idx, i);
    }
    
    return true;
}

bool ForcedAligner::init_kv_cache(int32_t n_ctx) {
    const auto & hp = model_.hparams;
    
    free_kv_cache();
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = hp.text_head_dim;
    state_.cache.n_kv_heads = hp.text_kv_heads;
    state_.cache.n_layers = hp.text_decoder_layers;
    
    const size_t n_tensors = hp.text_decoder_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(hp.text_decoder_layers);
    state_.cache.v_cache.resize(hp.text_decoder_layers);
    
    for (int il = 0; il < hp.text_decoder_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F32,
            hp.text_head_dim, hp.text_kv_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F32,
            hp.text_head_dim, hp.text_kv_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

void ForcedAligner::clear_kv_cache() {
    state_.cache.n_used = 0;
}

void ForcedAligner::free_kv_cache() {
    if (state_.cache.buffer) {
        ggml_backend_buffer_free(state_.cache.buffer);
        state_.cache.buffer = nullptr;
    }
    if (state_.cache.ctx) {
        ggml_free(state_.cache.ctx);
        state_.cache.ctx = nullptr;
    }
    state_.cache.k_cache.clear();
    state_.cache.v_cache.clear();
    state_.cache.n_ctx = 0;
    state_.cache.n_used = 0;
}

bool ForcedAligner::encode_audio(const float * mel_data, int n_mel, int n_frames,
                                  std::vector<float> & output) {
    const auto & hp = model_.hparams;
    const int n_state = hp.audio_d_model;
    const int n_head = hp.audio_attention_heads;
    const int n_layer = hp.audio_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.audio_layer_norm_eps;
    const float KQscale = 1.0f / sqrtf(float(n_state_head));
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_FA_MAX_NODES, false);
    
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    
    struct ggml_tensor * mel_4d = ggml_reshape_4d(ctx0, mel, n_frames, n_mel, 1, 1);
    
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model_.conv2d1_w, mel_4d, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d1_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d1_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d2_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d2_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d3_b) {
        struct ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d3_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu(ctx0, cur);
    
    int64_t out_w = cur->ne[0];
    int64_t out_h = cur->ne[1];
    int64_t out_c = cur->ne[2];
    int64_t seq_len = out_w;
    int64_t feat_dim = out_c * out_h;
    
    cur = ggml_reshape_3d(ctx0, cur, out_w, out_h * out_c, 1);
    cur = ggml_transpose(ctx0, cur);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, feat_dim, seq_len);
    
    if (model_.conv_out_w) {
        cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);
    }
    
    int64_t n_ctx = cur->ne[1];
    
    std::vector<float> pos_emb_data(n_ctx * n_state);
    compute_sinusoidal_pe(pos_emb_data.data(), n_ctx, n_state);
    
    struct ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);
    
    cur = ggml_add(ctx0, cur, pos_emb);
    
    struct ggml_tensor * inpL = cur;
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.encoder_layers[il];
        
        cur = ggml_norm(ctx0, inpL, eps);
        if (layer.attn_norm_w) {
            cur = ggml_mul(ctx0, cur, layer.attn_norm_w);
        }
        if (layer.attn_norm_b) {
            cur = ggml_add(ctx0, cur, layer.attn_norm_b);
        }
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
        if (layer.attn_q_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
        }
        
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
        if (layer.attn_k_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.attn_k_b);
        }
        
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
        if (layer.attn_v_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);
        }
        
        struct ggml_tensor * Q = ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx),
            0, 2, 1, 3);
        
        struct ggml_tensor * K = ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
            0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        
        struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
        
        struct ggml_tensor * V = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, Vcur, n_state_head, n_head, n_ctx),
            1, 2, 0, 3));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
        
        struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        
        cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
        
        cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur);
        if (layer.attn_out_b) {
            cur = ggml_add(ctx0, cur, layer.attn_out_b);
        }
        
        cur = ggml_add(ctx0, cur, inpL);
        
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_norm(ctx0, inpFF, eps);
        if (layer.ffn_norm_w) {
            cur = ggml_mul(ctx0, cur, layer.ffn_norm_w);
        }
        if (layer.ffn_norm_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_norm_b);
        }
        
        cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
        if (layer.ffn_up_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        }
        
        cur = ggml_gelu(ctx0, cur);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
        if (layer.ffn_down_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        }
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    if (model_.ln_post_w) {
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model_.ln_post_w);
        if (model_.ln_post_b) {
            cur = ggml_add(ctx0, cur, model_.ln_post_b);
        }
    }
    
    if (model_.proj1_w) {
        cur = ggml_mul_mat(ctx0, model_.proj1_w, cur);
        if (model_.proj1_b) {
            cur = ggml_add(ctx0, cur, model_.proj1_b);
        }
        cur = ggml_gelu(ctx0, cur);
    }
    
    if (model_.proj2_w) {
        cur = ggml_mul_mat(ctx0, model_.proj2_w, cur);
        if (model_.proj2_b) {
            cur = ggml_add(ctx0, cur, model_.proj2_b);
        }
    }
    
    ggml_set_name(cur, "audio_enc_out");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate audio encoder graph";
        ggml_free(ctx0);
        return false;
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    
    std::vector<float> transposed_mel(n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            transposed_mel[f + m * n_frames] = mel_data[m * n_frames + f];
        }
    }
    
    ggml_backend_tensor_set(mel_tensor, transposed_mel.data(), 0, n_mel * n_frames * sizeof(float));
    
    struct ggml_tensor * pos_emb_tensor = ggml_graph_get_tensor(gf, "pos_emb");
    if (pos_emb_tensor) {
        ggml_backend_tensor_set(pos_emb_tensor, pos_emb_data.data(), 0, 
                                 n_ctx * n_state * sizeof(float));
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute audio encoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    
    struct ggml_tensor * audio_out = ggml_graph_get_tensor(gf, "audio_enc_out");
    if (!audio_out) {
        error_msg_ = "Failed to find audio encoder output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    
    int64_t out_n_ctx = audio_out->ne[1];
    int64_t out_n_state = audio_out->ne[0];
    
    output.resize(out_n_ctx * out_n_state);
    ggml_backend_tensor_get(audio_out, output.data(), 0, out_n_ctx * out_n_state * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    
    return true;
}

struct ggml_cgraph * ForcedAligner::build_decoder_graph(
    const int32_t * tokens, int32_t n_tokens,
    const float * audio_embd, int32_t n_audio,
    int32_t audio_start_pos) {
    
    const auto & hp = model_.hparams;
    const int n_head = hp.text_attention_heads;
    const int n_kv_head = hp.text_kv_heads;
    const int head_dim = hp.text_head_dim;
    const int hidden_size = hp.text_hidden_size;
    const float eps = hp.text_rms_norm_eps;
    const float rope_theta = hp.text_rope_theta;
    const int n_layer = hp.text_decoder_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_FA_MAX_NODES, false);
    
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * inp_audio = nullptr;
    if (audio_embd && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);
    
    if (inp_audio && n_audio > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        struct ggml_tensor * embd_before = nullptr;
        struct ggml_tensor * embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio < n_tokens) {
            int after_start = audio_start_pos + n_audio;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            struct ggml_tensor * tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
            cur = ggml_concat(ctx0, tmp, embd_after, 1);
        } else if (embd_before) {
            cur = ggml_concat(ctx0, embd_before, inp_audio, 1);
        } else if (embd_after) {
            cur = ggml_concat(ctx0, inp_audio, embd_after, 1);
        } else {
            cur = inp_audio;
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.decoder_layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v || 
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate || 
            !layer.ffn_up || !layer.ffn_down) {
            ggml_free(ctx0);
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    
    cur = ggml_mul_mat(ctx0, model_.classify_head_w, cur);
    if (model_.classify_head_b) {
        cur = ggml_add(ctx0, cur, model_.classify_head_b);
    }
    
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool ForcedAligner::forward_decoder(
    const int32_t * tokens, int32_t n_tokens,
    const float * audio_embd, int32_t n_audio,
    int32_t audio_start_pos,
    std::vector<float> & output) {
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    struct ggml_cgraph * gf = build_decoder_graph(tokens, n_tokens,
                                                   audio_embd, n_audio, audio_start_pos);
    if (!gf) {
        error_msg_ = "Failed to build decoder graph";
        return false;
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        error_msg_ = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    if (audio_embd && n_audio > 0) {
        struct ggml_tensor * inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_embd, 0, 
                                    n_audio * model_.hparams.text_hidden_size * sizeof(float));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t n_classes = logits->ne[0];
    output.resize(n_tokens * n_classes);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

std::vector<float> ForcedAligner::classes_to_timestamps(const std::vector<int32_t> & classes) {
    std::vector<float> timestamps;
    timestamps.reserve(classes.size());
    
    float segment_time_sec = model_.hparams.timestamp_segment_time_ms / 1000.0f;
    
    for (int32_t cls : classes) {
        timestamps.push_back(cls * segment_time_sec);
    }
    
    return timestamps;
}

std::vector<int32_t> ForcedAligner::extract_timestamp_classes(
    const std::vector<float> & logits,
    const std::vector<int32_t> & tokens,
    int32_t timestamp_token_id) {
    
    const int32_t n_classes = model_.hparams.classify_num;
    std::vector<int32_t> timestamp_classes;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == timestamp_token_id) {
            const float * logit_ptr = logits.data() + i * n_classes;
            
            int32_t best_class = 0;
            float best_score = logit_ptr[0];
            for (int32_t c = 1; c < n_classes; ++c) {
                if (logit_ptr[c] > best_score) {
                    best_score = logit_ptr[c];
                    best_class = c;
                }
            }
            
            timestamp_classes.push_back(best_class);
        }
    }
    
    return timestamp_classes;
}

std::vector<int32_t> ForcedAligner::build_input_tokens(
    const std::vector<int32_t> & text_tokens,
    int32_t n_audio_frames) {
    
    const auto & hp = model_.hparams;
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + text_tokens.size() + 20);
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    tokens.push_back(hp.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(hp.audio_pad_token_id);
    }
    tokens.push_back(hp.audio_end_token_id);
    
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    for (int32_t tok : text_tokens) {
        tokens.push_back(tok);
    }
    
    return tokens;
}

int32_t ForcedAligner::find_audio_start_pos(const std::vector<int32_t> & tokens) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == model_.hparams.audio_start_token_id) {
            return static_cast<int32_t>(i + 1);
        }
    }
    return -1;
}

std::vector<int32_t> ForcedAligner::tokenize_with_timestamps(
    const std::string & text,
    std::vector<std::string> & words) {
    
    words.clear();
    std::vector<int32_t> tokens;
    
    std::istringstream iss(text);
    std::string word;
    bool first_word = true;
    
    while (iss >> word) {
        if (!first_word) {
            tokens.push_back(model_.hparams.timestamp_token_id);
        }
        first_word = false;
        
        words.push_back(word);
        
        std::string lookup = word;
        for (char & c : lookup) {
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        }
        
        bool found = false;
        for (size_t i = 0; i < model_.vocab.size(); ++i) {
            if (model_.vocab[i] == lookup || model_.vocab[i] == word) {
                tokens.push_back(static_cast<int32_t>(i));
                found = true;
                break;
            }
        }
        
        if (!found) {
            for (char c : word) {
                std::string char_str(1, c);
                for (size_t i = 0; i < model_.vocab.size(); ++i) {
                    if (model_.vocab[i] == char_str) {
                        tokens.push_back(static_cast<int32_t>(i));
                        break;
                    }
                }
            }
        }
    }
    
    if (!words.empty()) {
        tokens.push_back(model_.hparams.timestamp_token_id);
    }
    
    return tokens;
}

alignment_result ForcedAligner::align(const std::string & audio_path, const std::string & text) {
    alignment_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return align(samples.data(), samples.size(), text);
}

alignment_result ForcedAligner::align(const float * samples, int n_samples, const std::string & text) {
    alignment_result result;
    int64_t t_total_start = get_time_ms();
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    float audio_duration = static_cast<float>(n_samples) / QWEN_SAMPLE_RATE;
    
    int64_t t_mel_start = get_time_ms();
    MelFilters mel_filters;
    generate_mel_filters(mel_filters, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
    
    MelSpectrogram mel;
    if (!log_mel_spectrogram(samples, n_samples, mel_filters, mel, 4)) {
        result.error_msg = "Failed to compute mel spectrogram";
        return result;
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    if (!encode_audio(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
        result.error_msg = "Failed to encode audio: " + error_msg_;
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    int32_t n_audio_frames = audio_features.size() / model_.hparams.text_hidden_size;
    
    std::vector<std::string> words;
    std::vector<int32_t> text_tokens = tokenize_with_timestamps(text, words);
    
    std::vector<int32_t> input_tokens = build_input_tokens(text_tokens, n_audio_frames);
    
    int32_t audio_start_pos = find_audio_start_pos(input_tokens);
    
    int64_t t_decode_start = get_time_ms();
    std::vector<float> logits;
    if (!forward_decoder(input_tokens.data(), input_tokens.size(),
                         audio_features.data(), n_audio_frames,
                         audio_start_pos, logits)) {
        result.error_msg = "Decoder forward pass failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    std::vector<int32_t> timestamp_classes = extract_timestamp_classes(
        logits, input_tokens, model_.hparams.timestamp_token_id);
    
    std::vector<float> timestamps = classes_to_timestamps(timestamp_classes);
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (i > 0 && timestamps[i] < timestamps[i-1]) {
            timestamps[i] = timestamps[i-1];
        }
        if (timestamps[i] > audio_duration) {
            timestamps[i] = audio_duration;
        }
    }
    
    for (size_t i = 0; i < words.size(); ++i) {
        aligned_word aw;
        aw.word = words[i];
        
        if (i < timestamps.size()) {
            aw.start = (i > 0 && i - 1 < timestamps.size()) ? timestamps[i-1] : 0.0f;
            aw.end = timestamps[i];
        } else {
            aw.start = (i > 0) ? result.words.back().end : 0.0f;
            aw.end = audio_duration;
        }
        
        result.words.push_back(aw);
    }
    
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    
    return result;
}

void free_forced_aligner_model(forced_aligner_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.encoder_layers.clear();
    model.decoder_layers.clear();
    model.vocab.clear();
}

std::vector<int32_t> simple_tokenize(const std::string & text,
                                      const std::vector<std::string> & vocab,
                                      std::vector<std::string> & words) {
    words.clear();
    std::vector<int32_t> tokens;
    
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        words.push_back(word);
        
        for (size_t i = 0; i < vocab.size(); ++i) {
            if (vocab[i] == word) {
                tokens.push_back(static_cast<int32_t>(i));
                break;
            }
        }
    }
    
    return tokens;
}

}
