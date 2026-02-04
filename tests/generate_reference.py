#!/usr/bin/env python3
"""
Generate reference outputs from HuggingFace Qwen3-ASR model.

This script runs the HF model on sample.wav and saves intermediate outputs
for comparison with the GGML implementation.

Outputs saved to tests/reference/:
- mel.npy: Mel spectrogram from feature extractor
- audio_features.npy: Audio encoder output
- logits.npy: First few decoder logits
- transcript.txt: Full transcription
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample if needed."""
    audio, sr = sf.read(audio_path)
    if sr != target_sr:
        # Try librosa for resampling, fall back to scipy if not available
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
    return audio.astype(np.float32)


def _get_feat_extract_output_lengths(input_lengths):
    """Compute output length after CNN layers."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def extract_audio_features_manual(audio_tower, mel_tensor):
    """
    Manually extract audio features from the audio tower.
    
    The qwen-asr library has a bug with batched input, so we process
    single samples by manually running the forward pass.
    """
    # For single sample, squeeze the batch dimension
    input_feat = mel_tensor[0]  # (128, time)
    feat_len = torch.tensor([input_feat.shape[1]], dtype=torch.long)
    
    n_window = audio_tower.n_window
    
    aftercnn_lens = _get_feat_extract_output_lengths(feat_len)
    chunk_num = torch.ceil(feat_len / (n_window * 2)).long()
    
    chunk_lengths = torch.tensor(
        [n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feat_len % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2
    
    # Split into chunks
    chunk_list = input_feat.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    
    # Run convolutions
    padded_embeds = []
    for chunk in padded_feature.split(audio_tower.conv_chunksize, dim=0):
        padded_embed = F.gelu(audio_tower.conv2d1(chunk))
        padded_embed = F.gelu(audio_tower.conv2d2(padded_embed))
        padded_embed = F.gelu(audio_tower.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)
    
    b, c, f, t = padded_embed.size()
    padded_embed = audio_tower.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    
    positional_embedding = (
        audio_tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]
    
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (audio_tower.n_window_infer // (n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens).cumsum(-1, dtype=torch.int32)
    
    for encoder_layer in audio_tower.layers:
        layer_outputs = encoder_layer(
            hidden_states,
            cu_seqlens,
        )
        hidden_states = layer_outputs[0]
    
    hidden_states = audio_tower.ln_post(hidden_states)
    hidden_states = audio_tower.proj1(hidden_states)
    hidden_states = audio_tower.act(hidden_states)
    hidden_states = audio_tower.proj2(hidden_states)
    
    return hidden_states


def main():
    parser = argparse.ArgumentParser(description="Generate reference outputs from HF Qwen3-ASR")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/models/Qwen3-ASR-0.6B",
        help="Path to HF model"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/root/qwen-3-asr-ggml/sample.wav",
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/qwen-3-asr-ggml/tests/reference",
        help="Directory to save reference outputs"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--save-logits-count",
        type=int,
        default=10,
        help="Number of logit steps to save"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip full inference, only generate mel spectrogram and audio features"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio from {args.audio_path}...")
    audio = load_audio(args.audio_path)
    print(f"Audio shape: {audio.shape}, duration: {len(audio)/16000:.2f}s")

    # Save raw audio as numpy for reference
    np.save(output_dir / "audio.npy", audio)
    print(f"Saved audio.npy")

    print(f"Loading model from {args.model_path}...")
    print("This may take a while on CPU...")

    # Use qwen-asr package for proper model loading
    from qwen_asr import Qwen3ASRModel

    # Load model with CPU settings
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.float32,  # CPU inference
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )

    print("Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")

    # Access the underlying transformers model and processor
    hf_model = model.model
    processor = model.processor

    # Generate mel spectrogram using the feature extractor
    print("Generating mel spectrogram...")
    feature_extractor = processor.feature_extractor
    mel_inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    mel = mel_inputs.input_features[0].numpy()
    print(f"Mel shape: {mel.shape}")
    np.save(output_dir / "mel.npy", mel)
    print(f"Saved mel.npy")

    # Extract audio features using the audio encoder
    print("Extracting audio features...")
    with torch.no_grad():
        mel_tensor = mel_inputs.input_features.float()
        
        # Get the audio encoder from the model
        audio_tower = None
        if hasattr(hf_model, 'thinker') and hasattr(hf_model.thinker, 'audio_tower'):
            audio_tower = hf_model.thinker.audio_tower
            print("Found audio encoder at: thinker.audio_tower")
        
        if audio_tower is not None:
            try:
                # Use manual extraction to avoid library bug with batched input
                audio_features = extract_audio_features_manual(audio_tower, mel_tensor)
                print(f"Audio features shape: {audio_features.shape}")
                np.save(output_dir / "audio_features.npy", audio_features.numpy())
                print(f"Saved audio_features.npy")
            except Exception as e:
                print(f"Warning: Failed to extract audio features: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: Skipping audio_features extraction - encoder not found")

    if args.skip_inference:
        print("\nSkipping full inference (--skip-inference flag set)")
        metadata = {
            "model_path": args.model_path,
            "audio_path": args.audio_path,
            "audio_duration_s": len(audio) / 16000,
            "mel_shape": list(mel.shape),
            "skip_inference": True,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata.json")
        print("\nReference generation complete (mel and audio features only)!")
        print(f"Outputs saved to: {output_dir}")
        return

    # Run transcription using qwen-asr
    print(f"Generating transcription (max {args.max_new_tokens} tokens)...")
    
    # Use the model's transcribe method
    results = model.transcribe(
        audio=(audio, 16000),  # Pass as tuple (audio_array, sample_rate)
        language=None,  # Auto-detect language
    )
    
    transcript = results[0].text if results else ""
    detected_language = results[0].language if results else "unknown"
    
    print(f"Detected language: {detected_language}")
    print(f"Transcription: {transcript}")
    
    # Save transcript
    with open(output_dir / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Saved transcript.txt")

    # Save generation metadata
    metadata = {
        "model_path": args.model_path,
        "audio_path": args.audio_path,
        "audio_duration_s": len(audio) / 16000,
        "mel_shape": list(mel.shape),
        "max_new_tokens": args.max_new_tokens,
        "transcript_length": len(transcript),
        "detected_language": detected_language,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json")

    print("\nReference generation complete!")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
