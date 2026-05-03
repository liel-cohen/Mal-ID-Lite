#!/usr/bin/env python
"""Run all production-relevant Mal-ID-Lite tests.

Runs each test file as a separate pytest invocation in logical dependency
order: data loading -> model 1 -> model 2 -> model 3 -> ensemble.
Failures are shown immediately after each file finishes.

Usage
-----
    # Full suite (unit + integration):
    python tests/run_all_tests.py

    # Unit tests only (fast, no test_data/ required):
    python tests/run_all_tests.py --skip-integration

    # Stop on first failure:
    python tests/run_all_tests.py -- -x

    # Custom n_jobs:
    python tests/run_all_tests.py --n-jobs 4

Excluded dev-only tests
-----------------------
- test_fisher_comparison.py  (scipy comparison, not production-relevant)
- test_manage_cache.py       (cache CLI tool, not production-relevant)
- test_mcc_with_abstention.py  (MCC metric dev test)
- test_cap_cv_splits.py        (CV split capping dev test)

Expected runtime
----------------
- Unit tests only:       ~1-2 minutes
- Full suite:            ~5-10 minutes (depends on hardware and caching)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Test files in logical dependency order, grouped for display.
# Each tuple: (group_name, list_of_test_files)
TEST_GROUPS = [
    ("Data loading & caching", [
        "test_dataloader.py",
        "test_caching.py",
        "test_clone_id.py",
        "test_splits.py",
    ]),
    ("Model 1 (logistic regression)", [
        "test_model1.py",
    ]),
    ("Model 2 (convergent clusters)", [
        "test_model2.py",
        "test_model2_resume.py",
    ]),
    ("Model 3 (sequence-level)", [
        "test_model3.py",
        "test_model3_embeddings.py",
        "test_embedding.py",
    ]),
    ("Ensemble (metamodel)", [
        "test_ensemble_unit.py",
        "test_ensemble.py",
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
        "--n-jobs", type=int, default=None,
        help="Number of parallel workers for integration tests "
             "(Models 2, 3, ensemble). Default: 2 if not specified.",
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
    # Defaults: verbose test names + long tracebacks on failure.
    # Override via extra args after '--' (e.g., -- --tb=short -q).
    base_pytest_args = [sys.executable, "-m", "pytest", "-v", "--tb=long"]
    if args.skip_integration:
        base_pytest_args += ["-m", "not integration"]
    if args.n_jobs is not None:
        base_pytest_args += ["--n-jobs", str(args.n_jobs)]
    base_pytest_args += extra_pytest_args

    # --- Run each file separately so failures show immediately ---
    overall_start = time.time()
    results = []  # (file_name, group_name, exit_code, elapsed)

    for group_name, test_files in TEST_GROUPS:
        print(f"{'=' * 70}")
        print(f"  {group_name}")
        print(f"{'=' * 70}")

        for filename in test_files:
            filepath = TESTS_DIR / filename
            if not filepath.exists():
                print(f"  WARNING: {filename} not found, skipping\n")
                continue

            print(f"\n--- {filename} ---")
            cmd = base_pytest_args + [str(filepath)]
            file_start = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - file_start

            # Exit code 5 = no tests collected (all deselected by marker).
            rc = result.returncode
            if rc == 5:
                rc = 0
            results.append((filename, group_name, rc, elapsed))
            print()

    # --- Summary ---
    overall_elapsed = time.time() - overall_start
    n_passed = sum(1 for _, _, rc, _ in results if rc == 0)
    n_failed = sum(1 for _, _, rc, _ in results if rc != 0)
    n_total = len(results)

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    current_group = None
    for filename, group_name, rc, elapsed in results:
        if group_name != current_group:
            current_group = group_name
            print(f"\n  {group_name}:")
        status = "PASS" if rc == 0 else "FAIL"
        print(f"    {status:6s}  {_format_elapsed(elapsed):>8s}  {filename}")

    print(f"\n{'-' * 70}")
    print(f"  Total: {n_passed}/{n_total} files passed  "
          f"({_format_elapsed(overall_elapsed)} elapsed)")

    if n_failed > 0:
        failed_files = [f for f, _, rc, _ in results if rc != 0]
        print(f"\n  {n_failed} file(s) had failures:")
        for f in failed_files:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"\n  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
