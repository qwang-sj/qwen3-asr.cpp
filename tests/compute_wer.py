#!/usr/bin/env python3
"""
Compute Word Error Rate (WER) between reference and hypothesis transcripts.

Usage:
    python compute_wer.py reference.txt hypothesis.txt
    python compute_wer.py --ref "reference text" --hyp "hypothesis text"

Returns exit code 0 if WER is below threshold, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

try:
    from jiwer import wer, cer, mer, wil, wip
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False


def normalize_text(text: str) -> str:
    """Basic text normalization for WER calculation."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def compute_wer_simple(reference: str, hypothesis: str) -> float:
    """
    Simple WER implementation without jiwer dependency.
    Uses Levenshtein distance at word level.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Handle edge cases
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n] / m


def compute_metrics(
    reference: str,
    hypothesis: str,
    normalize: bool = True
):
    """
    Compute various transcription metrics.
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
        normalize: Whether to normalize text before comparison
    
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    metrics = {
        "reference": reference,
        "hypothesis": hypothesis,
        "ref_words": len(reference.split()),
        "hyp_words": len(hypothesis.split()),
        "ref_chars": len(reference),
        "hyp_chars": len(hypothesis),
    }
    
    if JIWER_AVAILABLE:
        metrics["wer"] = wer(reference, hypothesis)
        metrics["cer"] = cer(reference, hypothesis)
        metrics["mer"] = mer(reference, hypothesis)  # Match Error Rate
        metrics["wil"] = wil(reference, hypothesis)  # Word Information Lost
        metrics["wip"] = wip(reference, hypothesis)  # Word Information Preserved
    else:
        # Fallback to simple implementation
        metrics["wer"] = compute_wer_simple(reference, hypothesis)
        metrics["cer"] = None  # Not implemented in simple version
        metrics["mer"] = None
        metrics["wil"] = None
        metrics["wip"] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute Word Error Rate between transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two files
    python compute_wer.py reference.txt hypothesis.txt
    
    # Compare strings directly
    python compute_wer.py --ref "hello world" --hyp "hello word"
    
    # Set WER threshold for pass/fail
    python compute_wer.py ref.txt hyp.txt --threshold 0.1
    
    # Output as JSON
    python compute_wer.py ref.txt hyp.txt --json
"""
    )
    
    parser.add_argument(
        "ref_file",
        type=str,
        nargs="?",
        help="Reference transcript file"
    )
    parser.add_argument(
        "hyp_file",
        type=str,
        nargs="?",
        help="Hypothesis transcript file"
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Reference text (instead of file)"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        help="Hypothesis text (instead of file)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="WER threshold for pass/fail (default: 1.0 = always pass)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize text before comparison"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print WER value"
    )
    
    args = parser.parse_args()
    
    # Get reference text
    if args.ref:
        reference = args.ref
    elif args.ref_file:
        reference = Path(args.ref_file).read_text(encoding="utf-8").strip()
    else:
        parser.error("Must provide either ref_file or --ref")
    
    # Get hypothesis text
    if args.hyp:
        hypothesis = args.hyp
    elif args.hyp_file:
        hypothesis = Path(args.hyp_file).read_text(encoding="utf-8").strip()
    else:
        parser.error("Must provide either hyp_file or --hyp")
    
    # Compute metrics
    metrics = compute_metrics(
        reference, hypothesis,
        normalize=not args.no_normalize
    )
    
    # Output results
    if args.json:
        import json
        print(json.dumps(metrics, indent=2))
    elif args.quiet:
        print(f"{metrics['wer']:.4f}")
    else:
        print(f"Reference: {metrics['reference'][:100]}{'...' if len(metrics['reference']) > 100 else ''}")
        print(f"Hypothesis: {metrics['hypothesis'][:100]}{'...' if len(metrics['hypothesis']) > 100 else ''}")
        print(f"Reference words: {metrics['ref_words']}")
        print(f"Hypothesis words: {metrics['hyp_words']}")
        print(f"WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        
        if metrics.get('cer') is not None:
            print(f"CER: {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
        
        if not JIWER_AVAILABLE:
            print("Note: Install jiwer for additional metrics (CER, MER, WIL, WIP)")
        
        # Pass/fail based on threshold
        if metrics['wer'] <= args.threshold:
            print(f"PASS (WER {metrics['wer']:.4f} <= threshold {args.threshold})")
        else:
            print(f"FAIL (WER {metrics['wer']:.4f} > threshold {args.threshold})")
    
    # Exit code based on threshold
    sys.exit(0 if metrics['wer'] <= args.threshold else 1)


if __name__ == "__main__":
    main()
