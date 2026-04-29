#!/usr/bin/env python
"""Benchmark ESM-2 embedding throughput and extrapolate time for large sequence counts.

Loads CDR3 sequences from test data (tests/test_data/participants/), computes
ESM-2 embeddings (esm2_t30_150M_UR50D, 640-dim), measures wall-clock time,
and extrapolates to larger counts (100K, 500K, 1M, 2M, 5M, 10M, 50M).

Two modes of operation:

  Single benchmark (default):
    Runs increasing-size rounds (10%, 25%, 50%, 75%, 100% of loaded sequences)
    with a single (batch_size, num_threads) configuration. Reports throughput
    at each size to verify linear scaling, then extrapolates from the largest
    round.

  Sweep mode (--sweep-batch-sizes and/or --sweep-num-threads):
    Tests multiple configurations to find the optimal settings. Produces a
    throughput matrix and highlights the best combination. On GPU, thread
    sweeps are skipped (GPU parallelism is controlled by batch size, not CPU
    threads). On GPU, batch sizes that exceed device memory are caught as OOM
    and skipped gracefully.

Hardware detection:
    CPU topology (sockets, cores per socket, physical vs logical cores) is
    auto-detected via lscpu (Linux) or sysctl (macOS). Thread sweep values
    are derived from the detected topology: 1, half-socket, full socket, all
    physical cores, all logical cores, plus intermediate values for large
    machines (>32 physical cores).

Arguments:
    --batch-size N        Sequences per ESM-2 forward pass. Default: auto per
                          device (cpu=64, mps=64, cuda=4000). Larger batches
                          improve GPU utilization but increase padding waste
                          (shorter sequences get padded to the longest in the
                          batch). On CPU, 32-64 is typically optimal.
    --max-sequences N     Maximum number of unique CDR3 sequences to load from
                          test data (default: 10000). Each sweep configuration
                          runs on this many sequences.
    --device DEVICE       Force device: 'cuda', 'mps', or 'cpu'. Default:
                          auto-detect (cuda > mps > cpu).
    --num-threads N       CPU threads for PyTorch intra-op parallelism
                          (torch.set_num_threads). Only meaningful for CPU
                          inference. Default: PyTorch default (all logical
                          cores). On many-core servers, physical core count
                          often gives better throughput than all logical cores.
    --sweep-batch-sizes   Test multiple batch sizes. Values are device-specific:
                          cpu=[8,16,32,64,128,256], mps=[16,32,64,128,256,512],
                          cuda=[64,256,512,1000,2000,4000,8000].
    --sweep-num-threads   Test multiple thread counts, auto-detected from CPU
                          topology. On GPU devices this is skipped (threads
                          don't affect GPU). Can be combined with
                          --sweep-batch-sizes for a full matrix.
    --max-threads N       Cap for --sweep-num-threads. Auto-detected values
                          above this are excluded. Useful on shared servers
                          (e.g., --max-threads 50 to avoid consuming all cores).

Output: scripts/dev/output/test_embedding_timing/
    Single mode:
      - timing_report_YYYYMMDD_HHMMSS.txt   (human-readable report)
      - timing_data_YYYYMMDD_HHMMSS.csv     (per-round measurements)
      - timing_log_YYYYMMDD_HHMMSS.log      (full stdout capture)
      - model_log_YYYYMMDD_HHMMSS.log       (ESM-2 model load log)
    Sweep mode:
      - timing_report_YYYYMMDD_HHMMSS.txt   (human-readable report with best config)
      - sweep_results_YYYYMMDD_HHMMSS.csv   (all combinations with throughput)
      - timing_log_YYYYMMDD_HHMMSS.log      (full stdout capture)
      - model_log_YYYYMMDD_HHMMSS.log       (ESM-2 model load log)

Usage:
    cd Mal-ID-Lite

    # Basic benchmark (single config):
    python scripts/dev/test_embedding_timing.py
    python scripts/dev/test_embedding_timing.py --batch-size 128
    python scripts/dev/test_embedding_timing.py --num-threads 64

    # Sweep batch sizes (thread count fixed):
    python scripts/dev/test_embedding_timing.py --sweep-batch-sizes

    # Sweep thread counts (batch size fixed):
    python scripts/dev/test_embedding_timing.py --sweep-num-threads

    # Full matrix sweep:
    python scripts/dev/test_embedding_timing.py --sweep-batch-sizes --sweep-num-threads

    # Limit thread sweep on shared servers:
    python scripts/dev/test_embedding_timing.py --sweep-num-threads --max-threads 50

    # Fewer sequences for a faster (less precise) sweep:
    python scripts/dev/test_embedding_timing.py --sweep-batch-sizes --max-sequences 3000

Requires: torch, fair-esm
Expected runtime: ~1-3 min (single), ~5-15 min (sweep), depending on device.
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.training.compute_model3_embeddings import (
    EMBEDDING_DIM,
    DEFAULT_BATCH_SIZES,
    detect_device,
    load_esm2_model,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "output" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data" / "participants"

EXTRAPOLATION_TARGETS = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
                         10_000_000, 50_000_000]

ROUND_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

WARMUP_SEQUENCES = 200

# Device-specific batch sizes to sweep
SWEEP_BATCH_SIZES = {
    "cpu":  [8, 16, 32, 64, 128, 256],
    "mps":  [16, 32, 64, 128, 256, 512],
    "cuda": [64, 256, 512, 1000, 2000, 4000, 8000],
}

# Sentinel value for OOM / failed runs in the sweep matrix
OOM_SENTINEL = "OOM"


# ---------------------------------------------------------------------------
# Stdout tee (captures all print output to a log file)
# ---------------------------------------------------------------------------

class TeeOutput:
    """Duplicate stdout to a log file so everything the user sees is also saved."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_cpu_topology() -> Dict[str, Optional[int]]:
    """Detect CPU topology: sockets, cores_per_socket, total physical, total logical.

    Uses lscpu on Linux, sysctl on macOS. Returns a dict with None for
    values that could not be detected.
    """
    logical_cores = os.cpu_count()
    result = {
        "sockets": None,
        "cores_per_socket": None,
        "total_physical": None,
        "total_logical": logical_cores,
    }

    system = platform.system()

    if system == "Linux":
        try:
            lscpu_output = subprocess.check_output(
                ["lscpu"], text=True, stderr=subprocess.DEVNULL
            )
            for line in lscpu_output.splitlines():
                if line.startswith("Socket(s):"):
                    result["sockets"] = int(line.split(":")[1].strip())
                elif line.startswith("Core(s) per socket:"):
                    result["cores_per_socket"] = int(line.split(":")[1].strip())

            if result["sockets"] and result["cores_per_socket"]:
                result["total_physical"] = result["sockets"] * result["cores_per_socket"]
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass

    elif system == "Darwin":
        try:
            # macOS: physical cores
            phys = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            result["total_physical"] = int(phys)
            # macOS: single socket
            result["sockets"] = 1
            result["cores_per_socket"] = result["total_physical"]
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass

    # Fallback: if we have logical but not physical, estimate physical = logical / 2
    # (assumes hyperthreading with 2 threads per core, which is the common case)
    if result["total_physical"] is None and logical_cores is not None:
        result["total_physical"] = logical_cores // 2

    return result


def auto_thread_sweep_values(topology: Dict[str, Optional[int]]) -> List[int]:
    """Generate thread counts to sweep based on detected CPU topology.

    Strategy: test a range from small (single-digit) up to all logical cores,
    with emphasis on the boundaries that matter:
      - 1 (baseline)
      - half a socket
      - one full socket
      - all physical cores (both sockets)
      - all logical cores (with hyperthreading)
    """
    values = set()
    values.add(1)  # baseline

    cores_per_socket = topology.get("cores_per_socket")
    total_physical = topology.get("total_physical")
    total_logical = topology.get("total_logical")

    if cores_per_socket:
        values.add(cores_per_socket // 2)  # half a socket
        values.add(cores_per_socket)        # one full socket

    if total_physical:
        values.add(total_physical)          # all physical cores

    if total_logical:
        values.add(total_logical)           # all logical cores (hyperthreaded)

    # Add some intermediate values for finer-grained picture
    if total_physical and total_physical > 32:
        values.add(16)
        values.add(32)

    # Remove zeros and sort
    values.discard(0)
    return sorted(values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_sequences(max_sequences: int) -> List[str]:
    """Load CDR3 AA sequences from test participant parquets.

    Reads participant parquets from tests/test_data/participants/, extracts
    the cdr3_aa column, and returns up to max_sequences unique, non-empty
    sequences (minimum length 5 AA).
    """
    parquets = sorted(TEST_DATA_DIR.glob("*_clean.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No participant parquets found in {TEST_DATA_DIR}. "
            f"Run the test data setup first."
        )

    all_seqs: List[str] = []
    for pq in parquets:
        df = pd.read_parquet(pq, columns=["cdr3_aa"])
        seqs = df["cdr3_aa"].dropna().astype(str).str.strip().tolist()
        all_seqs.extend(seqs)
        if len(all_seqs) >= max_sequences:
            break

    # Deduplicate and filter empty/short sequences
    seen: set = set()
    unique: List[str] = []
    for s in all_seqs:
        if len(s) >= 5 and s not in seen:
            seen.add(s)
            unique.append(s)
            if len(unique) >= max_sequences:
                break

    if len(unique) == 0:
        raise ValueError("No valid CDR3 sequences found in test data.")

    return unique


def embed_sequences(sequences: List[str], model, batch_converter, repr_layer,
                    device, batch_size: int) -> float:
    """Embed sequences and return wall-clock seconds.

    Mirrors the core loop from compute_embeddings_for_sequences but only
    tracks timing — embeddings are discarded to keep memory stable.
    """
    import torch

    n = len(sequences)
    t0 = time.perf_counter()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_seqs = sequences[start: start + batch_size]
            data = [(f"seq{i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps = results["representations"][repr_layer]

            # Mean-pool to match real pipeline (BOS/EOS excluded)
            for j, seq in enumerate(batch_seqs):
                L = len(seq)
                _ = reps[j, 1: L + 1].mean(0).cpu().float().numpy()

    elapsed = time.perf_counter() - t0
    return elapsed


def format_time(seconds: float) -> str:
    """Format seconds as a short human-readable string."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds:.1f}s"


def format_time_long(seconds: float) -> str:
    """Format seconds with days/hours/minutes breakdown for large values."""
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} days ({seconds / 3600:.0f}h)"
    elif seconds >= 3600:
        return f"{seconds / 3600:.1f}h ({seconds / 60:.0f}min)"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}min ({seconds:.0f}s)"
    else:
        return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_sweep(
    sequences: List[str],
    model,
    batch_converter,
    repr_layer: int,
    device,
    device_str: str,
    batch_sizes: List[int],
    thread_counts: List[int],
) -> Tuple[pd.DataFrame, Dict]:
    """Run the (batch_size x num_threads) sweep matrix.

    For each (thread_count, batch_size) combination, embeds all sequences
    and records throughput. GPU OOM errors are caught gracefully and
    recorded as OOM (the sweep continues with the next combination).

    The caller is responsible for collapsing thread_counts to a single
    value on GPU devices (where threads don't affect inference).

    Returns:
        results_df: DataFrame with columns [num_threads, batch_size,
            n_sequences, elapsed_seconds, throughput_seq_per_sec, status].
        best_config: Dict with the highest-throughput combination, or
            empty dict if all combinations failed (OOM).
    """
    import torch

    is_gpu = device_str in ("cuda", "mps")
    n_seqs = len(sequences)

    total_combos = len(batch_sizes) * len(thread_counts)
    print(f"Sweep: {len(batch_sizes)} batch sizes x {len(thread_counts)} "
          f"thread counts = {total_combos} combinations")
    print(f"Sequences per run: {n_seqs}")
    print()

    results = []
    best_throughput = 0.0
    best_config = {}
    combo_idx = 0
    sweep_start = time.perf_counter()

    for n_threads in thread_counts:
        # Set thread count (only affects CPU, but safe to call always)
        torch.set_num_threads(n_threads)

        for bs in batch_sizes:
            combo_idx += 1
            label = f"[{combo_idx}/{total_combos}]"

            if is_gpu:
                print(f"  {label} batch_size={bs:>5} ... ", end="", flush=True)
            else:
                print(f"  {label} threads={n_threads:>4}, batch_size={bs:>4} ... ",
                      end="", flush=True)

            try:
                elapsed = embed_sequences(
                    sequences, model, batch_converter, repr_layer, device, bs
                )
                throughput = n_seqs / elapsed

                results.append({
                    "num_threads": n_threads,
                    "batch_size": bs,
                    "n_sequences": n_seqs,
                    "elapsed_seconds": round(elapsed, 3),
                    "throughput_seq_per_sec": round(throughput, 1),
                    "status": "ok",
                })

                # Progress estimate: average time per combo so far
                sweep_elapsed = time.perf_counter() - sweep_start
                remaining_combos = total_combos - combo_idx
                avg_per_combo = sweep_elapsed / combo_idx
                est_remaining = avg_per_combo * remaining_combos

                eta_str = ""
                if remaining_combos > 0:
                    eta_str = f"  ETA: ~{format_time(est_remaining)}"

                print(f"{format_time(elapsed):>8}  ({throughput:,.0f} seq/s)  "
                      f"[{format_time(sweep_elapsed)} elapsed{eta_str}]")

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = {
                        "num_threads": n_threads,
                        "batch_size": bs,
                        "throughput_seq_per_sec": throughput,
                        "elapsed_seconds": elapsed,
                    }

            except RuntimeError as e:
                error_str = str(e).lower()
                # Catch GPU OOM errors
                if "out of memory" in error_str or "oom" in error_str:
                    results.append({
                        "num_threads": n_threads,
                        "batch_size": bs,
                        "n_sequences": n_seqs,
                        "elapsed_seconds": None,
                        "throughput_seq_per_sec": None,
                        "status": OOM_SENTINEL,
                    })
                    print(f"  {OOM_SENTINEL} (batch too large for device memory)")

                    # On GPU, clear the failed allocation so subsequent runs work
                    if device_str == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

    sweep_total = time.perf_counter() - sweep_start
    print(f"\nSweep completed in {format_time(sweep_total)}")

    results_df = pd.DataFrame(results)
    return results_df, best_config


def print_sweep_matrix(results_df: pd.DataFrame, device_str: str):
    """Print the sweep results as a readable matrix (threads x batch_sizes)."""
    is_gpu = device_str in ("cuda", "mps")

    if is_gpu or results_df["num_threads"].nunique() == 1:
        # Single-dimension: just batch sizes
        print("Throughput by batch size (seq/s):")
        print("-" * 50)
        for _, row in results_df.iterrows():
            if row["status"] == OOM_SENTINEL:
                val = OOM_SENTINEL
            else:
                val = f"{row['throughput_seq_per_sec']:,.0f}"
            print(f"  batch_size={row['batch_size']:>5}:  {val:>10}")
    elif results_df["batch_size"].nunique() == 1:
        # Single-dimension: just thread counts
        print("Throughput by thread count (seq/s):")
        print("-" * 50)
        for _, row in results_df.iterrows():
            if row["status"] == OOM_SENTINEL:
                val = OOM_SENTINEL
            else:
                val = f"{row['throughput_seq_per_sec']:,.0f}"
            print(f"  threads={row['num_threads']:>4}:  {val:>10}")
    else:
        # Full matrix
        thread_counts = sorted(results_df["num_threads"].unique())
        batch_sizes = sorted(results_df["batch_size"].unique())

        # Build lookup
        lookup = {}
        for _, row in results_df.iterrows():
            key = (row["num_threads"], row["batch_size"])
            if row["status"] == OOM_SENTINEL:
                lookup[key] = OOM_SENTINEL
            else:
                lookup[key] = f"{row['throughput_seq_per_sec']:,.0f}"

        # Header
        col_width = 10
        header = f"  {'threads':>8}"
        for bs in batch_sizes:
            header += f"  {'bs=' + str(bs):>{col_width}}"
        print("Throughput matrix (seq/s):")
        print(header)
        print("  " + "-" * (8 + (col_width + 2) * len(batch_sizes)))

        # Rows
        for nt in thread_counts:
            row_str = f"  {nt:>8}"
            for bs in batch_sizes:
                val = lookup.get((nt, bs), "N/A")
                row_str += f"  {val:>{col_width}}"
            print(row_str)

    print()


# ---------------------------------------------------------------------------
# Single-config benchmark (original behavior)
# ---------------------------------------------------------------------------

def run_single_benchmark(
    sequences: List[str],
    model,
    batch_converter,
    repr_layer: int,
    device,
    batch_size: int,
) -> Tuple[List[Dict], float, float]:
    """Run the standard timed-rounds benchmark with a single config.

    Returns (results_list, steady_state_throughput, weighted_avg_throughput).
    """
    n_available = len(sequences)

    round_sizes = sorted(set(
        max(batch_size, int(n_available * frac))
        for frac in ROUND_FRACTIONS
    ))
    round_sizes = [s for s in round_sizes if s <= n_available]
    if n_available not in round_sizes:
        round_sizes.append(n_available)

    results = []
    for n_seqs in round_sizes:
        subset = sequences[:n_seqs]
        elapsed = embed_sequences(subset, model, batch_converter, repr_layer,
                                  device, batch_size)
        throughput = n_seqs / elapsed
        results.append({
            "n_sequences": n_seqs,
            "elapsed_seconds": elapsed,
            "throughput_seq_per_sec": throughput,
        })
        print(f"  {n_seqs:>8,} sequences -> {format_time(elapsed):>8} "
              f"({throughput:,.0f} seq/s)")

    best = results[-1]
    steady_throughput = best["throughput_seq_per_sec"]

    total_seqs = sum(r["n_sequences"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    avg_throughput = total_seqs / total_time

    return results, steady_throughput, avg_throughput


# ---------------------------------------------------------------------------
# Extrapolation & reporting
# ---------------------------------------------------------------------------

def print_extrapolations(throughput: float) -> List[Dict]:
    """Print and return extrapolated times for standard sequence counts."""
    print("Extrapolated times (based on steady-state throughput):")
    print("=" * 60)
    print(f"  {'Sequences':>15}  {'Estimated time':>20}  {'Storage (float16)':>18}")
    print(f"  {'-' * 15}  {'-' * 20}  {'-' * 18}")

    extrapolations = []
    for target in EXTRAPOLATION_TARGETS:
        est_seconds = target / throughput
        storage_gb = (target * EMBEDDING_DIM * 2) / (1024 ** 3)

        extrapolations.append({
            "n_sequences": target,
            "estimated_seconds": est_seconds,
            "estimated_time_human": format_time_long(est_seconds),
            "storage_gb": storage_gb,
        })
        print(f"  {target:>15,}  {format_time_long(est_seconds):>20}  "
              f"{storage_gb:>14.1f} GB")

    print("=" * 60)
    print()
    return extrapolations


def print_cdr3_stats(sequences: List[str]) -> List[int]:
    """Print CDR3 length distribution statistics."""
    lengths = [len(s) for s in sequences]
    print(f"CDR3 length stats (n={len(sequences)}):")
    print(f"  Mean:   {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Std:    {np.std(lengths):.1f}")
    print()
    return lengths


def save_report(
    report_path: Path,
    device_str: str,
    batch_size: int,
    effective_threads: int,
    topology: Dict,
    load_time: float,
    n_available: int,
    n_warmup: int,
    throughput: float,
    avg_throughput: float,
    round_results: Optional[List[Dict]],
    extrapolations: List[Dict],
    lengths: List[int],
    sweep_config: Optional[Dict],
    total_elapsed: float,
):
    """Save human-readable timing report."""
    lines = []
    lines.append("ESM-2 Embedding Timing Report")
    lines.append("=" * 60)
    lines.append(f"Date:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total runtime:   {format_time(total_elapsed)}")
    lines.append(f"Device:          {device_str}")
    lines.append(f"Batch size:      {batch_size}")
    lines.append(f"CPU threads:     {effective_threads} "
                 f"(logical cores: {topology['total_logical']})")
    if topology["total_physical"]:
        if topology["sockets"] and topology["cores_per_socket"]:
            lines.append(f"Physical cores:  {topology['total_physical']} "
                         f"({topology['sockets']} socket(s) x "
                         f"{topology['cores_per_socket']} cores)")
        else:
            lines.append(f"Physical cores:  {topology['total_physical']} (estimated)")
    lines.append(f"Model:           esm2_t30_150M_UR50D (640-dim)")
    lines.append(f"Model load:      {load_time:.1f}s")
    lines.append(f"Test seqs:       {n_available}")
    lines.append(f"Warmup seqs:     {n_warmup}")
    lines.append("")

    if round_results:
        lines.append("Timed Rounds")
        lines.append("-" * 60)
        for r in round_results:
            lines.append(
                f"  {r['n_sequences']:>8,} seqs  ->  "
                f"{format_time(r['elapsed_seconds']):>8}  "
                f"({r['throughput_seq_per_sec']:,.0f} seq/s)"
            )
        lines.append("")

    lines.append(f"Steady-state throughput: {throughput:,.0f} seq/s")
    lines.append(f"Weighted avg throughput: {avg_throughput:,.0f} seq/s")

    if sweep_config:
        lines.append("")
        lines.append("Best sweep config:")
        lines.append(f"  Threads:    {sweep_config['num_threads']}")
        lines.append(f"  Batch size: {sweep_config['batch_size']}")
        lines.append(f"  Throughput: {sweep_config['throughput_seq_per_sec']:,.0f} seq/s")

    lines.append("")
    lines.append("Extrapolations (steady-state throughput)")
    lines.append("=" * 60)
    lines.append(f"  {'Sequences':>15}  {'Estimated time':>20}  {'Storage (f16)':>14}")
    lines.append(f"  {'-' * 15}  {'-' * 20}  {'-' * 14}")
    for e in extrapolations:
        lines.append(
            f"  {e['n_sequences']:>15,}  {e['estimated_time_human']:>20}  "
            f"{e['storage_gb']:>11.1f} GB"
        )

    lines.append("")
    lines.append("CDR3 Length Stats")
    lines.append("-" * 60)
    lines.append(f"  Mean: {np.mean(lengths):.1f}  Median: {np.median(lengths):.1f}  "
                 f"Min: {min(lengths)}  Max: {max(lengths)}  Std: {np.std(lengths):.1f}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Extrapolations assume linear scaling (throughput constant).")
    lines.append("  - Real runs may vary with sequence length distribution,")
    lines.append("    memory pressure, thermal throttling, and padding overhead.")
    lines.append("  - Storage is for float16 embeddings only (640 dims x 2 bytes).")
    lines.append("    Parquet + stats files add ~10-15% overhead.")
    lines.append("  - Model load time is a one-time cost (not included in estimates).")

    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ESM-2 embedding throughput and extrapolate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: auto per device)."
    )
    parser.add_argument(
        "--max-sequences", type=int, default=10000,
        help="Maximum number of test sequences to use (default: 10000). "
             "For sweep mode, each configuration runs on this many sequences."
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Force device: 'cuda', 'mps', or 'cpu'."
    )
    parser.add_argument(
        "--num-threads", type=int, default=None,
        help="Number of CPU threads for PyTorch intra-op parallelism "
             "(torch.set_num_threads). Only meaningful for CPU inference. "
             "Default: PyTorch default (all available cores). On many-core "
             "servers, setting this to the number of physical cores (not "
             "logical/hyperthreaded) often gives best throughput."
    )
    parser.add_argument(
        "--sweep-batch-sizes", action="store_true",
        help="Sweep multiple batch sizes to find the optimal one. "
             "Values are auto-selected based on device type. "
             "Uses --num-threads (or default) for thread count."
    )
    parser.add_argument(
        "--sweep-num-threads", action="store_true",
        help="Sweep multiple thread counts to find the optimal one. "
             "Values are auto-detected from CPU topology (sockets, cores). "
             "On GPU devices, this is skipped (threads don't affect GPU). "
             "Uses --batch-size (or default) for batch size."
    )
    parser.add_argument(
        "--max-threads", type=int, default=None,
        help="Maximum thread count for --sweep-num-threads. Auto-detected "
             "values above this are excluded. Useful on shared servers where "
             "you don't want to consume all cores (e.g., --max-threads 50)."
    )
    args = parser.parse_args()

    if args.max_sequences < 1:
        raise ValueError(f"--max-sequences must be >= 1, got {args.max_sequences}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.num_threads is not None and args.num_threads < 1:
        raise ValueError(f"--num-threads must be >= 1, got {args.num_threads}")
    if args.max_threads is not None:
        if args.max_threads < 1:
            raise ValueError(f"--max-threads must be >= 1, got {args.max_threads}")
        if not args.sweep_num_threads:
            print("Warning: --max-threads has no effect without --sweep-num-threads.")
    if args.num_threads is not None and args.max_threads is not None:
        if args.num_threads > args.max_threads:
            raise ValueError(
                f"--num-threads ({args.num_threads}) exceeds --max-threads "
                f"({args.max_threads}). Either raise --max-threads or lower "
                f"--num-threads."
            )

    is_sweep = args.sweep_batch_sizes or args.sweep_num_threads

    script_start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"timing_report_{timestamp}.txt"
    csv_path = OUTPUT_DIR / f"timing_data_{timestamp}.csv"
    sweep_csv_path = OUTPUT_DIR / f"sweep_results_{timestamp}.csv"
    script_log_path = OUTPUT_DIR / f"timing_log_{timestamp}.log"
    model_log_path = OUTPUT_DIR / f"model_log_{timestamp}.log"

    # Tee stdout to the log file so everything printed is also saved.
    # Model logger gets a separate file to avoid two writers to the same file.
    tee = TeeOutput(script_log_path)
    sys.stdout = tee

    try:
        # --- Step 1: Setup ---
        log = setup_logging(model_log_path, verbose=1)

        import torch

        device_str = args.device or detect_device()
        batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(device_str, 64)
        topology = get_cpu_topology()

        if args.num_threads is not None:
            torch.set_num_threads(args.num_threads)

        effective_threads = torch.get_num_threads()

        # --- Print hardware info ---
        print("Hardware")
        print("-" * 60)
        print(f"  Device:          {device_str}")
        print(f"  Logical cores:   {topology['total_logical']}")
        if topology["total_physical"]:
            if topology["sockets"] and topology["cores_per_socket"]:
                print(f"  Physical cores:  {topology['total_physical']} "
                      f"({topology['sockets']} socket(s) x "
                      f"{topology['cores_per_socket']} cores)")
            else:
                print(f"  Physical cores:  {topology['total_physical']} (estimated)")
        print(f"  CPU threads:     {effective_threads}"
              f"{' [set via --num-threads]' if args.num_threads else ' [PyTorch default]'}")
        print(f"  Batch size:      {batch_size}")

        if device_str == "cpu" and args.num_threads is None:
            if (topology["total_physical"] and topology["total_logical"]
                    and topology["total_logical"] > topology["total_physical"]):
                print(f"  Tip: Default uses all {topology['total_logical']} logical cores. "
                      f"Try --num-threads {topology['total_physical']} (physical cores) "
                      f"or --sweep-num-threads to find optimal.")
        if device_str != "cpu" and args.num_threads is not None:
            print(f"  Note: --num-threads has minimal effect on {device_str} -- "
                  f"GPU parallelism is controlled by batch size.")
        print()

        # --- Step 2: Load ESM-2 model ---
        print("Loading ESM-2 model...")
        model, _alphabet, batch_converter, repr_layer, torch_device, load_time = \
            load_esm2_model(device_str, log)
        print(f"Model loaded in {load_time:.1f}s")
        print()

        # --- Step 3: Load test sequences ---
        print(f"Loading test sequences (max {args.max_sequences})...")
        sequences = load_test_sequences(args.max_sequences)
        n_available = len(sequences)
        print(f"Loaded {n_available} unique CDR3 sequences")
        print()

        # --- Step 4: Warmup ---
        n_warmup = min(WARMUP_SEQUENCES, n_available)
        print(f"Warmup: embedding {n_warmup} sequences (not timed)...")
        embed_sequences(sequences[:n_warmup], model, batch_converter, repr_layer,
                        torch_device, batch_size)
        print("Warmup complete.")
        print()

        # ===================================================================
        # SWEEP MODE
        # ===================================================================
        if is_sweep:
            # Determine batch sizes to test
            if args.sweep_batch_sizes:
                sweep_bs = list(SWEEP_BATCH_SIZES.get(device_str, SWEEP_BATCH_SIZES["cpu"]))
                # If user also specified --batch-size, include it in the sweep
                if args.batch_size and args.batch_size not in sweep_bs:
                    sweep_bs = sorted(set(sweep_bs + [args.batch_size]))
            else:
                sweep_bs = [batch_size]

            # Determine thread counts to test
            if args.sweep_num_threads:
                sweep_threads = auto_thread_sweep_values(topology)
                # Cap at --max-threads if specified
                if args.max_threads is not None:
                    sweep_threads = [t for t in sweep_threads if t <= args.max_threads]
                    # Ensure the cap value itself is included as the upper boundary
                    if args.max_threads not in sweep_threads:
                        sweep_threads.append(args.max_threads)
                        sweep_threads.sort()
                # If user also specified --num-threads, include it in the sweep
                if args.num_threads and args.num_threads not in sweep_threads:
                    sweep_threads = sorted(set(sweep_threads + [args.num_threads]))
                if not sweep_threads:
                    raise ValueError(
                        f"--max-threads {args.max_threads} is too low — no thread "
                        f"counts to test. Minimum auto-detected value is 1."
                    )
            else:
                sweep_threads = [effective_threads]

            # On GPU, thread sweep is not meaningful — collapse to single value
            is_gpu = device_str in ("cuda", "mps")
            if is_gpu and args.sweep_num_threads and len(sweep_threads) > 1:
                print(f"  Note: Thread sweep not meaningful on {device_str} -- "
                      f"GPU parallelism is controlled by batch size.")
                print(f"  Testing batch sizes only.")
                sweep_threads = [effective_threads]
                print()

            print("=" * 60)
            print("SWEEP MODE")
            print("=" * 60)
            if args.sweep_batch_sizes:
                print(f"  Batch sizes:   {sweep_bs}")
            if len(sweep_threads) > 1:
                print(f"  Thread counts: {sweep_threads}")
            else:
                print(f"  Threads:       {sweep_threads[0]} (fixed)")
            print()

            sweep_df, best_config = run_sweep(
                sequences=sequences,
                model=model,
                batch_converter=batch_converter,
                repr_layer=repr_layer,
                device=torch_device,
                device_str=device_str,
                batch_sizes=sweep_bs,
                thread_counts=sweep_threads,
            )

            print()
            print_sweep_matrix(sweep_df, device_str)

            # Save sweep CSV
            sweep_df.to_csv(sweep_csv_path, index=False)
            print(f"Sweep results saved: {sweep_csv_path}")
            print()

            if best_config:
                print("=" * 60)
                print("BEST CONFIGURATION")
                print("=" * 60)
                print(f"  Threads:    {best_config['num_threads']}")
                print(f"  Batch size: {best_config['batch_size']}")
                print(f"  Throughput: {best_config['throughput_seq_per_sec']:,.0f} seq/s")
                print(f"  Time:       {format_time(best_config['elapsed_seconds'])}"
                      f" for {n_available} sequences")
                print()

                # Use best config for extrapolation
                throughput = best_config["throughput_seq_per_sec"]
                avg_throughput = throughput  # single run, same value
            else:
                print("All configurations failed (OOM). Cannot extrapolate.")
                return

            # Extrapolate using best throughput
            extrapolations = print_extrapolations(throughput)

            # CDR3 stats
            lengths = print_cdr3_stats(sequences)

            # Save report
            total_elapsed = time.perf_counter() - script_start
            save_report(
                report_path, device_str, best_config["batch_size"],
                best_config["num_threads"], topology, load_time,
                n_available, n_warmup, throughput, avg_throughput,
                round_results=None, extrapolations=extrapolations,
                lengths=lengths, sweep_config=best_config,
                total_elapsed=total_elapsed,
            )
            print(f"Report saved: {report_path}")

        # ===================================================================
        # SINGLE BENCHMARK MODE (original behavior)
        # ===================================================================
        else:
            print("Running timed rounds...")
            print("-" * 60)

            round_results, throughput, avg_throughput = run_single_benchmark(
                sequences, model, batch_converter, repr_layer, torch_device, batch_size,
            )

            print("-" * 60)
            print()
            print(f"Steady-state throughput (largest round): {throughput:,.0f} seq/s")
            print(f"Weighted average throughput (all rounds): {avg_throughput:,.0f} seq/s")
            print()

            extrapolations = print_extrapolations(throughput)
            lengths = print_cdr3_stats(sequences)

            # Save measurements CSV
            pd.DataFrame(round_results).to_csv(csv_path, index=False)
            print(f"Raw measurements saved: {csv_path}")

            # Save report
            total_elapsed = time.perf_counter() - script_start
            save_report(
                report_path, device_str, batch_size, effective_threads,
                topology, load_time, n_available, n_warmup,
                throughput, avg_throughput, round_results,
                extrapolations, lengths, sweep_config=None,
                total_elapsed=total_elapsed,
            )
            print(f"Report saved: {report_path}")

        total_elapsed = time.perf_counter() - script_start
        print()
        print(f"Done. Total runtime: {format_time(total_elapsed)} "
              f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"Full output log: {script_log_path}")

    finally:
        # Always restore stdout and close the tee log file
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
