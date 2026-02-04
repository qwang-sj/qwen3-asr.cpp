#!/usr/bin/env python3
"""
Compare numpy arrays with configurable tolerance.

Usage:
    python compare.py reference.npy test.npy [--rtol 1e-5] [--atol 1e-8]
    python compare.py --dir reference/ test/ [--rtol 1e-5] [--atol 1e-8]

Returns exit code 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def compare_arrays(
    ref: np.ndarray,
    test: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True
) -> Tuple[bool, dict]:
    """
    Compare two numpy arrays with tolerance.
    
    Args:
        ref: Reference array
        test: Test array
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Print detailed comparison info
    
    Returns:
        Tuple of (passed, stats_dict)
    """
    stats = {
        "ref_shape": ref.shape,
        "test_shape": test.shape,
        "ref_dtype": str(ref.dtype),
        "test_dtype": str(test.dtype),
    }
    
    # Check shapes match
    if ref.shape != test.shape:
        stats["error"] = f"Shape mismatch: {ref.shape} vs {test.shape}"
        if verbose:
            print(f"FAIL: {stats['error']}")
        return False, stats
    
    # Convert to float64 for comparison
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)
    
    # Compute differences
    abs_diff = np.abs(ref_f - test_f)
    rel_diff = abs_diff / (np.abs(ref_f) + 1e-10)  # Avoid division by zero
    
    stats["max_abs_diff"] = float(np.max(abs_diff))
    stats["mean_abs_diff"] = float(np.mean(abs_diff))
    stats["max_rel_diff"] = float(np.max(rel_diff))
    stats["mean_rel_diff"] = float(np.mean(rel_diff))
    
    # Check if arrays are close
    passed = np.allclose(ref_f, test_f, rtol=rtol, atol=atol)
    
    # Count failures
    failures = ~np.isclose(ref_f, test_f, rtol=rtol, atol=atol)
    stats["num_failures"] = int(np.sum(failures))
    stats["total_elements"] = int(ref.size)
    stats["failure_rate"] = stats["num_failures"] / stats["total_elements"]
    
    if verbose:
        print(f"Shape: {ref.shape}")
        print(f"Max absolute diff: {stats['max_abs_diff']:.6e}")
        print(f"Mean absolute diff: {stats['mean_abs_diff']:.6e}")
        print(f"Max relative diff: {stats['max_rel_diff']:.6e}")
        print(f"Mean relative diff: {stats['mean_rel_diff']:.6e}")
        print(f"Failures: {stats['num_failures']}/{stats['total_elements']} ({stats['failure_rate']*100:.4f}%)")
        print(f"Tolerance: rtol={rtol}, atol={atol}")
        
        if passed:
            print("PASS")
        else:
            print("FAIL")
            # Show some failure locations
            if stats["num_failures"] > 0:
                fail_indices = np.argwhere(failures)[:5]  # First 5 failures
                print("First few failure locations:")
                for idx in fail_indices:
                    idx_tuple = tuple(idx)
                    print(f"  {idx_tuple}: ref={ref_f[idx_tuple]:.6e}, test={test_f[idx_tuple]:.6e}, diff={abs_diff[idx_tuple]:.6e}")
    
    return passed, stats


def compare_files(
    ref_path: Path,
    test_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True
) -> Tuple[bool, dict]:
    """Compare two numpy files."""
    if verbose:
        print(f"Comparing: {ref_path} vs {test_path}")
    
    try:
        ref = np.load(ref_path)
        test = np.load(test_path)
    except Exception as e:
        if verbose:
            print(f"FAIL: Error loading files: {e}")
        return False, {"error": str(e)}
    
    return compare_arrays(ref, test, rtol=rtol, atol=atol, verbose=verbose)


def compare_directories(
    ref_dir: Path,
    test_dir: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True
) -> Tuple[bool, dict]:
    """Compare all .npy files in two directories."""
    ref_files = sorted(ref_dir.glob("*.npy"))
    
    if not ref_files:
        if verbose:
            print(f"No .npy files found in {ref_dir}")
        return False, {"error": "No reference files found"}
    
    all_passed = True
    results = {}
    
    for ref_file in ref_files:
        test_file = test_dir / ref_file.name
        
        if not test_file.exists():
            if verbose:
                print(f"\nFAIL: {ref_file.name} - test file not found")
            results[ref_file.name] = {"error": "Test file not found"}
            all_passed = False
            continue
        
        if verbose:
            print(f"\n--- {ref_file.name} ---")
        
        passed, stats = compare_files(ref_file, test_file, rtol=rtol, atol=atol, verbose=verbose)
        results[ref_file.name] = stats
        
        if not passed:
            all_passed = False
    
    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Compare numpy arrays with tolerance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two files
    python compare.py reference/mel.npy test/mel.npy
    
    # Compare with custom tolerance
    python compare.py ref.npy test.npy --rtol 1e-3 --atol 1e-6
    
    # Compare all .npy files in directories
    python compare.py --dir reference/ test/
    
    # Self-comparison (should always pass)
    python compare.py reference/mel.npy reference/mel.npy
"""
    )
    
    parser.add_argument(
        "ref",
        type=str,
        help="Reference file or directory"
    )
    parser.add_argument(
        "test",
        type=str,
        help="Test file or directory"
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="Compare directories instead of files"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance (default: 1e-8)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print PASS/FAIL"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    ref_path = Path(args.ref)
    test_path = Path(args.test)
    verbose = not args.quiet
    
    if args.dir:
        passed, results = compare_directories(
            ref_path, test_path,
            rtol=args.rtol, atol=args.atol,
            verbose=verbose
        )
    else:
        passed, results = compare_files(
            ref_path, test_path,
            rtol=args.rtol, atol=args.atol,
            verbose=verbose
        )
    
    if args.json:
        import json
        print(json.dumps({"passed": passed, "results": results}, indent=2, default=str))
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
