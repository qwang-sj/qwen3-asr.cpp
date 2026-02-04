#!/usr/bin/env python3
"""
Generate reference intermediate values from HuggingFace Qwen3-ASR text decoder.

This script runs the text decoder on test tokens and saves intermediate values
for comparison with the GGML implementation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Test tokens: [audio_start, audio_pad, audio_pad, audio_pad, audio_end]
TEST_TOKENS = [151669, 151676, 151676, 151676, 151670]


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to Q and K tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def main():
    print("Loading Qwen3-ASR model...")
    from qwen_asr import Qwen3ASRModel
    
    asr_model = Qwen3ASRModel.from_pretrained(
        '/root/models/Qwen3-ASR-0.6B',
        dtype=torch.float32,
        device_map="cpu"
    )
    
    model = asr_model.model
    thinker = model.thinker
    text_model = thinker.model  # The Qwen3 text decoder
    
    print(f"Text model type: {type(text_model)}")
    print(f"Text model config: {text_model.config}")
    
    # Get model components
    embed_tokens = text_model.embed_tokens
    layers = text_model.layers
    norm = text_model.norm
    
    # Get lm_head from thinker
    lm_head = thinker.lm_head
    
    print(f"Number of layers: {len(layers)}")
    print(f"Embedding shape: {embed_tokens.weight.shape}")
    print(f"LM head shape: {lm_head.weight.shape}")
    
    # Create output directory
    output_dir = "/root/qwen-3-asr-ggml/tests/reference"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare input
    tokens = torch.tensor([TEST_TOKENS], dtype=torch.long)
    seq_len = tokens.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    print(f"\nInput tokens: {TEST_TOKENS}")
    print(f"Sequence length: {seq_len}")
    
    with torch.no_grad():
        # Step 1: Token embeddings
        hidden_states = embed_tokens(tokens)
        print(f"Token embeddings shape: {hidden_states.shape}")
        np.save(f"{output_dir}/decoder_embd.npy", hidden_states.numpy())
        print(f"Saved decoder_embd.npy")
        print(f"  First 5 values: {hidden_states[0, 0, :5].tolist()}")
        
        rotary_emb = text_model.rotary_emb
        
        # Process layer 0 in detail
        layer = layers[0]
        residual = hidden_states
        
        # Step 2: Input LayerNorm (RMSNorm)
        hidden_states_norm = layer.input_layernorm(hidden_states)
        print(f"\nAfter input_layernorm shape: {hidden_states_norm.shape}")
        np.save(f"{output_dir}/decoder_norm0.npy", hidden_states_norm.numpy())
        print(f"Saved decoder_norm0.npy")
        print(f"  First 5 values: {hidden_states_norm[0, 0, :5].tolist()}")
        
        # Step 3: Q, K, V projections
        attn = layer.self_attn
        
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        
        print(f"\nAttention config:")
        print(f"  num_heads: {num_heads}")
        print(f"  num_kv_heads: {num_kv_heads}")
        print(f"  head_dim: {head_dim}")
        
        # Q projection
        q_proj = attn.q_proj(hidden_states_norm)
        print(f"\nQ projection shape: {q_proj.shape}")
        np.save(f"{output_dir}/decoder_q0.npy", q_proj.numpy())
        print(f"Saved decoder_q0.npy")
        print(f"  First 5 values: {q_proj[0, 0, :5].tolist()}")
        
        # K projection
        k_proj = attn.k_proj(hidden_states_norm)
        print(f"K projection shape: {k_proj.shape}")
        np.save(f"{output_dir}/decoder_k0.npy", k_proj.numpy())
        print(f"Saved decoder_k0.npy")
        print(f"  First 5 values: {k_proj[0, 0, :5].tolist()}")
        
        # V projection
        v_proj = attn.v_proj(hidden_states_norm)
        print(f"V projection shape: {v_proj.shape}")
        np.save(f"{output_dir}/decoder_v0.npy", v_proj.numpy())
        print(f"Saved decoder_v0.npy")
        print(f"  First 5 values: {v_proj[0, 0, :5].tolist()}")
        
        # Reshape Q, K, V
        bsz = 1
        q_reshaped = q_proj.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k_reshaped = k_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_reshaped = v_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        print(f"\nQ reshaped: {q_reshaped.shape}")
        print(f"K reshaped: {k_reshaped.shape}")
        print(f"V reshaped: {v_reshaped.shape}")
        
        # Q/K norms (Qwen3 specific)
        if hasattr(attn, 'q_norm') and attn.q_norm is not None:
            q_reshaped = attn.q_norm(q_reshaped)
            print(f"After Q norm: {q_reshaped[0, 0, 0, :5].tolist()}")
        if hasattr(attn, 'k_norm') and attn.k_norm is not None:
            k_reshaped = attn.k_norm(k_reshaped)
            print(f"After K norm: {k_reshaped[0, 0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_q0_normed.npy", q_reshaped.numpy())
        np.save(f"{output_dir}/decoder_k0_normed.npy", k_reshaped.numpy())
        
        # Apply RoPE
        cos, sin = rotary_emb(v_reshaped, position_ids)
        print(f"\nRoPE cos shape: {cos.shape}")
        print(f"RoPE sin shape: {sin.shape}")
        print(f"RoPE cos first 5: {cos[0, 0, :5].tolist()}")
        print(f"RoPE sin first 5: {sin[0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_rope_cos.npy", cos.numpy())
        np.save(f"{output_dir}/decoder_rope_sin.npy", sin.numpy())
        
        q_rope, k_rope = apply_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)
        print(f"\nAfter RoPE Q: {q_rope[0, 0, 0, :5].tolist()}")
        print(f"After RoPE K: {k_rope[0, 0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_q0_rope.npy", q_rope.numpy())
        np.save(f"{output_dir}/decoder_k0_rope.npy", k_rope.numpy())
        
        # GQA: repeat K, V for grouped query attention
        n_rep = num_heads // num_kv_heads
        if n_rep > 1:
            k_expanded = k_rope.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
            v_expanded = v_reshaped.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
        else:
            k_expanded = k_rope
            v_expanded = v_reshaped
        
        print(f"\nK expanded: {k_expanded.shape}")
        print(f"V expanded: {v_expanded.shape}")
        
        # Compute attention scores
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(q_rope, k_expanded.transpose(-2, -1)) * scale
        print(f"\nAttention weights shape: {attn_weights.shape}")
        print(f"Attention weights [0,0,:,:]: {attn_weights[0, 0, :, :].tolist()}")
        
        np.save(f"{output_dir}/decoder_attn_weights0.npy", attn_weights.numpy())
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        print(f"Attention probs [0,0,:,:]: {attn_probs[0, 0, :, :].tolist()}")
        
        np.save(f"{output_dir}/decoder_attn_probs0.npy", attn_probs.numpy())
        
        # Attention output
        attn_output = torch.matmul(attn_probs, v_expanded)
        print(f"\nAttention output shape: {attn_output.shape}")
        print(f"Attention output [0,0,0,:5]: {attn_output[0, 0, 0, :5].tolist()}")
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        print(f"Attention output reshaped: {attn_output.shape}")
        
        # Output projection
        attn_output = attn.o_proj(attn_output)
        print(f"After o_proj: {attn_output.shape}")
        print(f"After o_proj [0,0,:5]: {attn_output[0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_attn0.npy", attn_output.numpy())
        print(f"Saved decoder_attn0.npy")
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # FFN
        residual = hidden_states
        hidden_states_norm = layer.post_attention_layernorm(hidden_states)
        
        # MLP
        mlp = layer.mlp
        gate = mlp.gate_proj(hidden_states_norm)
        up = mlp.up_proj(hidden_states_norm)
        gate_silu = F.silu(gate)
        ffn_out = mlp.down_proj(gate_silu * up)
        
        print(f"\nFFN gate [0,0,:5]: {gate[0, 0, :5].tolist()}")
        print(f"FFN up [0,0,:5]: {up[0, 0, :5].tolist()}")
        print(f"FFN out [0,0,:5]: {ffn_out[0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_ffn0.npy", ffn_out.numpy())
        print(f"Saved decoder_ffn0.npy")
        
        # Now run full forward pass to get logits
        print("\n" + "="*60)
        print("Running full forward pass...")
        print("="*60)
        
        # Reset hidden states
        hidden_states = embed_tokens(tokens)
        
        for i, layer in enumerate(layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Self attention
            attn = layer.self_attn
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)
            
            q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            if hasattr(attn, 'q_norm') and attn.q_norm is not None:
                q = attn.q_norm(q)
            if hasattr(attn, 'k_norm') and attn.k_norm is not None:
                k = attn.k_norm(k)
            
            cos, sin = rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # GQA
            if n_rep > 1:
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
                v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            attn_output = attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            
            # FFN
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            
            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            hidden_states = mlp.down_proj(F.silu(gate) * up)
            
            hidden_states = residual + hidden_states
            
            if i == 0:
                print(f"After layer 0: {hidden_states[0, 0, :5].tolist()}")
        
        # Final norm
        hidden_states = norm(hidden_states)
        print(f"\nAfter final norm: {hidden_states.shape}")
        print(f"After final norm [0,0,:5]: {hidden_states[0, 0, :5].tolist()}")
        
        np.save(f"{output_dir}/decoder_final_norm.npy", hidden_states.numpy())
        
        # LM head
        logits = lm_head(hidden_states)
        print(f"\nLogits shape: {logits.shape}")
        print(f"First 10 logits [0,0,:10]: {logits[0, 0, :10].tolist()}")
        print(f"Last 10 logits [0,0,-10:]: {logits[0, 0, -10:].tolist()}")
        
        # Find argmax
        argmax = torch.argmax(logits[0, 0, :]).item()
        max_val = logits[0, 0, argmax].item()
        print(f"\nArgmax at position 0: token {argmax} with logit {max_val}")
        
        np.save(f"{output_dir}/decoder_logits.npy", logits.numpy())
        print(f"Saved decoder_logits.npy")
        
        # Also save token embeddings for specific tokens
        np.save(f"{output_dir}/token_embeddings.npy", hidden_states.numpy())
        
    print("\n" + "="*60)
    print("Reference generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
