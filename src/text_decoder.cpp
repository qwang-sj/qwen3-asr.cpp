#include "text_decoder.h"
#include "timing.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>

#define QWEN3_ASR_MAX_NODES 8192

namespace qwen3_asr {

TextDecoder::TextDecoder() = default;

TextDecoder::~TextDecoder() {
    free_kv_cache(state_.cache);
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
    free_decoder_model(model_);
}

bool TextDecoder::load_model(const std::string & model_path) {
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
    
    if (!parse_config(ctx)) {
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
        free_decoder_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_vocab(ctx)) {
        free_decoder_model(model_);
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
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, 1, QWEN3_ASR_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool TextDecoder::parse_config(struct gguf_context * ctx) {
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
    
    auto & cfg = model_.config;
    cfg.vocab_size = get_u32("qwen3-asr.vocab_size", 151936);
    cfg.hidden_size = get_u32("qwen3-asr.embedding_length", 1024);
    cfg.n_decoder_layers = get_u32("qwen3-asr.block_count", 28);
    cfg.n_attention_heads = get_u32("qwen3-asr.attention.head_count", 16);
    cfg.n_key_value_heads = get_u32("qwen3-asr.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("qwen3-asr.feed_forward_length", 3072);
    cfg.head_dim = get_u32("qwen3-asr.attention.key_length", 128);
    cfg.rms_norm_eps = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("qwen3-asr.rope.freq_base", 1000000.0f);
    
    cfg.pad_token_id = 151643;
    cfg.eos_token_id = 151645;
    cfg.audio_start_token_id = get_u32("qwen3-asr.audio.start_token_id", 151669);
    cfg.audio_end_token_id = get_u32("qwen3-asr.audio.end_token_id", 151670);
    cfg.audio_pad_token_id = get_u32("qwen3-asr.audio.pad_token_id", 151676);
    
    return true;
}

bool TextDecoder::create_tensors(struct gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    
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
    
    model_.layers.resize(cfg.n_decoder_layers);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        if (strstr(name, "audio.encoder")) {
            continue;
        } else if (strstr(name, "token_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.vocab_size;
            n_dims = 2;
        } else if (strstr(name, "output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "attn_output.weight")) {
            ne[0] = cfg.n_attention_heads * cfg.head_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "output.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.vocab_size;
            n_dims = 2;
        } else if (strstr(name, "attn_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "attn_q_norm.weight")) {
            ne[0] = cfg.head_dim;
            n_dims = 1;
        } else if (strstr(name, "attn_k_norm.weight")) {
            ne[0] = cfg.head_dim;
            n_dims = 1;
        } else if (strstr(name, "attn_q.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_attention_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "attn_k.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_key_value_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "attn_v.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_key_value_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "ffn_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "ffn_gate.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.intermediate_size;
            n_dims = 2;
        } else if (strstr(name, "ffn_up.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.intermediate_size;
            n_dims = 2;
        } else if (strstr(name, "ffn_down.weight")) {
            ne[0] = cfg.intermediate_size;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;
        
        if (strstr(name, "token_embd.weight")) {
            model_.token_embd = tensor;
        } else if (strstr(name, "output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_decoder_layers) {
                auto & layer = model_.layers[layer_idx];
                
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
        } else if (strstr(name, "output.weight")) {
            model_.output = tensor;
        }
    }
    
    return true;
}

bool TextDecoder::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
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

bool TextDecoder::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_decoder_layers;
    
    const size_t n_tensors = cfg.n_decoder_layers * 2;
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
    
    state_.cache.k_cache.resize(cfg.n_decoder_layers);
    state_.cache.v_cache.resize(cfg.n_decoder_layers);
    
    for (int il = 0; il < cfg.n_decoder_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F32,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F32,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

void TextDecoder::clear_kv_cache() {
    state_.cache.n_used = 0;
}

struct ggml_cgraph * TextDecoder::build_graph(
    const int32_t * tokens, int32_t n_tokens, int32_t n_past,
    const float * audio_embd, int32_t n_audio, int32_t audio_start_pos) {
    
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_decoder_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
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
        ggml_set_name(cur, "embd_with_audio");
        ggml_set_output(cur);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v || 
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate || 
            !layer.ffn_up || !layer.ffn_down) {
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
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
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
        ggml_format_name(cur, "ffn_out_%d", il);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool TextDecoder::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                          std::vector<float> & output) {
    return forward_with_audio(tokens, n_tokens, nullptr, 0, -1, n_past, output);
}

bool TextDecoder::forward_with_audio(
    const int32_t * tokens, int32_t n_tokens,
    const float * audio_embd, int32_t n_audio,
    int32_t audio_start_pos, int32_t n_past,
    std::vector<float> & output) {
    QWEN3_TIMER("decoder.forward");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(2048)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    struct ggml_cgraph * gf = build_graph(tokens, n_tokens, n_past,
                                          audio_embd, n_audio, audio_start_pos);
    
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
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    if (audio_embd && n_audio > 0) {
        struct ggml_tensor * inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_embd, 0, 
                                    n_audio * model_.config.hidden_size * sizeof(float));
        }
    }
    
    {
        QWEN3_TIMER("decoder.compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    output.resize(n_tokens * vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TextDecoder::forward_debug(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                                std::vector<float> & output,
                                std::map<std::string, std::vector<float>> & debug_tensors) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(2048)) {
            return false;
        }
    }
    
    struct ggml_cgraph * gf = build_graph(tokens, n_tokens, n_past, nullptr, 0, -1);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens) {
        ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (logits) {
        int64_t vocab_size = logits->ne[0];
        output.resize(n_tokens * vocab_size);
        ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    }
    
    const char * debug_names[] = {"debug_norm0", "debug_q0_raw", "debug_q0_normed", "debug_q0_rope", 
                                   "debug_attn0_out", "debug_kq_scaled", "debug_kq_masked", "debug_kq_softmax"};
    for (const char * name : debug_names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) {
            size_t nbytes = ggml_nbytes(t);
            std::vector<float> data(nbytes / sizeof(float));
            ggml_backend_tensor_get(t, data.data(), 0, nbytes);
            debug_tensors[name] = std::move(data);
        }
    }
    
    state_.cache.n_used = n_past + n_tokens;
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_decoder_model(text_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
}

void free_kv_cache(kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

bool TextDecoder::load_vocab(struct gguf_context * ctx) {
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
    
    vocab_.resize(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        vocab_[i] = gguf_get_arr_str(ctx, tokens_idx, i);
    }
    
    return true;
}

std::string TextDecoder::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab_.size()) {
        return "";
    }
    
    std::string token = vocab_[token_id];
    
    if (token.size() >= 3 && token[0] == '<' && token[1] == '|' && 
        token[token.size()-1] == '>' && token[token.size()-2] == '|') {
        return "";
    }
    
    if (token.size() >= 5 && token.substr(0, 4) == "[PAD") {
        return "";
    }
    
    std::string result;
    result.reserve(token.size());
    
    for (size_t i = 0; i < token.size(); ++i) {
        unsigned char c = token[i];
        
        if (c == 0xC4 && i + 1 < token.size() && (unsigned char)token[i+1] == 0xA0) {
            result += ' ';
            ++i;
        } else {
            result += token[i];
        }
    }
    
    return result;
}

std::string TextDecoder::decode_tokens(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

} // namespace qwen3_asr
