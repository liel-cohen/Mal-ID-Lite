#!/usr/bin/env python
"""Run all production-relevant Mal-ID-Lite tests.

Runs tests in logical order: utilities -> data -> model 1 -> model 2
-> model 3 -> ensemble. Each group is a separate pytest invocation so
that failures in one group don't prevent the others from running.

Usage
-----
    # Full suite (unit + integration):
    python tests/run_all_tests.py

    # Unit tests only (fast, no test_data/ required):
    python tests/run_all_tests.py --skip-integration

    # Pass extra pytest flags:
    python tests/run_all_tests.py -- -x --tb=long

    # Verbose with custom n_jobs:
    python tests/run_all_tests.py -v --n-jobs 4

Excluded dev-only tests
-----------------------
- test_fisher_comparison.py  (scipy comparison, not production-relevant)
- test_manage_cache.py       (cache CLI tool, not production-relevant)
- test_mcc_with_abstention.py  (MCC metric dev test)
- test_cap_cv_splits.py        (CV split capping dev test)

Expected runtime
----------------
- Unit tests only:       ~2-5 minutes
- Full suite:            ~10-30 minutes (depends on hardware and caching)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Test groups in logical dependency order.
# Each tuple: (group_name, list_of_test_files)
TEST_GROUPS = [
    ("Data loading & caching", [
        "test_dataloader_quick.py",
        "test_caching_quick.py",
        "test_splits.py",
    ]),
    ("Model 1 (logistic regression)", [
        "test_model1.py",
    ]),
    ("Model 2 (convergent clusters)", [
        "test_model2_quick.py",
        "test_model2_resume_quick.py",
    ]),
    ("Model 3 (sequence-level)", [
        "test_model3.py",
        "test_model3_embeddings.py",
        "test_embedding_quick.py",
    ]),
    ("Ensemble (metamodel)", [
        "test_ensemble_unit.py",
        "test_ensemble_quick.py",
        "test_ensemble_integration.py",
    ]),
]


def _format_elapsed(seconds: float) -> str:
    """Format seconds as 'Xm Ys' or 'Ys'."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def main():
    parser = argparse.ArgumentParser(
        description="Run all production-relevant Mal-ID-Lite tests.",
        epilog="Any arguments after '--' are passed directly to pytest.",
    )
    parser.add_argument(
        "--skip-integration", action="store_true",
        help="Skip slow integration tests (only run fast unit tests).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable pytest verbose output (-v).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Number of parallel workers for integration tests.",
    )

    # Split on '--' to separate our flags from extra pytest flags
    argv = sys.argv[1:]
    extra_pytest_args = []
    if "--" in argv:
        split_idx = argv.index("--")
        extra_pytest_args = argv[split_idx + 1:]
        argv = argv[:split_idx]

    args = parser.parse_args(argv)

    # --- Banner ---
    mode = "UNIT TESTS ONLY" if args.skip_integration else "FULL SUITE (unit + integration)"
    print("=" * 70)
    print(f"  Mal-ID-Lite Test Runner  —  {mode}")
    print("=" * 70)

    if not args.skip_integration:
        print()
        print("  NOTE: The full suite includes integration tests that require")
        print("  tests/test_data/ and may take 10-30 minutes. Use")
        print("  --skip-integration for a quick unit-only run (~2-5 min).")

    print()

    # --- Build pytest base args ---
    base_pytest_args = [sys.executable, "-m", "pytest"]
    if args.skip_integration:
        base_pytest_args += ["-m", "not integration"]
    if args.verbose:
        base_pytest_args.append("-v")
    if args.n_jobs is not None:
        base_pytest_args += ["--n-jobs", str(args.n_jobs)]
    base_pytest_args += extra_pytest_args

    # --- Run each group ---
    overall_start = time.time()
    results = []  # (group_name, n_files, exit_code, elapsed)

    for group_name, test_files in TEST_GROUPS:
        # Resolve to absolute paths and filter to existing files
        paths = []
        for f in test_files:
            p = TESTS_DIR / f
            if p.exists():
                paths.append(str(p))
            else:
                print(f"  WARNING: {f} not found, skipping")

        if not paths:
            results.append((group_name, 0, 0, 0.0))
            continue

        print(f"--- {group_name} ({len(paths)} file{'s' if len(paths) != 1 else ''}) ---")

        cmd = base_pytest_args + paths
        group_start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - group_start

        # Exit code 5 = no tests collected (all were deselected by marker
        # filter). Treat as success — the group just had no applicable tests.
        rc = result.returncode
        if rc == 5:
            rc = 0
        results.append((group_name, len(paths), rc, elapsed))
        print()

    # --- Summary ---
    overall_elapsed = time.time() - overall_start
    n_passed = sum(1 for _, _, rc, _ in results if rc == 0)
    n_failed = sum(1 for _, _, rc, _ in results if rc != 0)
    n_total = len(results)

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for group_name, n_files, rc, elapsed in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        if n_files == 0:
            status = "SKIP (no files)"
        print(f"  {status:20s}  {_format_elapsed(elapsed):>8s}  {group_name}")

    print("-" * 70)
    print(f"  Total: {n_passed}/{n_total} groups passed  "
          f"({_format_elapsed(overall_elapsed)} elapsed)")

    if n_failed > 0:
        print(f"\n  {n_failed} group(s) had failures. "
              "Re-run with -v or -- -x --tb=long for details.")
        sys.exit(1)
    else:
        print(f"\n  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
