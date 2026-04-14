#!/usr/bin/env python
"""
Compute ESM-2 embeddings for all participants' DOWNSAMPLED CDR3 sequences.

This is a standalone, potentially GPU-intensive preprocessing step for Model 3
(sequence-level classifier). It loads each participant's CLEAN cached data,
applies downsampling, extracts CDR3 amino acid sequences, computes ESM-2
embeddings (mean-pooled last layer), and saves the results as per-participant
float16 .npy files alongside their DOWNSAMPLED parquets.

This script should be run BEFORE training Model 3. The training script
(train_model3.py) loads the pre-computed embeddings from the output directory.
At load time, row alignment between the fold DataFrame and the pre-computed
files is ensured using the downsampling unique key (repertoire_id,
igh_or_tcrb_clone_id, isotype_supergroup, amplification_label if present).
If the order differs, embeddings
are automatically reordered to match (with a warning). A biological sanity
check (cdr3_aa, v_gene, j_gene) runs after alignment. If alignment fails
entirely (e.g., embeddings from a different preprocessing run), the training
script errors with a clear message to re-run this script.

Alternatively, train_model3.py can compute embeddings inline if invoked with
the --compute-embeddings flag, but this is much slower for repeated runs.

Resource estimates (based on MacBook Pro M4 Max, 64GB RAM, MPS):
    - Time:    ~3 hours per 10 million downsampled sequences
    - Storage: ~14 GB per 10 million downsampled sequences
               (float16 embeddings: 640 dims x 2 bytes = 1.28 KB/seq,
                plus parquet and stats overhead)
    Actual values depend on hardware, sequence lengths, and batch size.

Output structure:
    cache/<dataset>/embeddings/
    |-- <participant_label>_downsampled.parquet   (exact sequences embedded)
    |-- <participant_label>_embeddings.npy        (float16, shape (N, 640), row-aligned)
    |-- <participant_label>_stats.json            (per-participant metadata)
    |-- ...
    |-- cache_info.json                           (run-level metadata)
    |-- embedding_report_YYYYMMDD_HHMMSS.md       (human-readable summary)
    |-- embedding_log_YYYYMMDD_HHMMSS.log         (full log)

Usage:
    cd Mal-ID-Lite
    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv

    # With raw data (first run, no cache yet):
    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv \\
        --data-dir /path/to/airr_data/

    # Override batch size for GPU:
    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv --batch-size 4000

    # Verbose per-batch progress:
    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv --verbose 2

    # Verify existing embedding files:
    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv --verify
"""

import os
import sys
import json
import time
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path (for running as script)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite import __version__ as MALID_VERSION
from malid_lite.models.model3_sequence_level import CDR3_COL
from malid_lite.training.training_utils import DEFAULT_DATASET_NAME

# Constants
EMBEDDING_DIM = 640
ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"
EXPECTED_NUM_LAYERS = 30

# Batch size defaults per device (based on benchmarks).
# MPS: sweet spot at 64-128 (padding overhead hurts larger batches).
# CUDA: 4000 matches original Mal-ID; short CDR3 sequences allow large batches.
# CPU: 64 (no GPU parallelism to exploit).
DEFAULT_BATCH_SIZES = {
    "mps": 64,
    "cuda": 4000,
    "cpu": 64,
}

# Resource estimate constants (for user-facing messages)
HOURS_PER_10M_SEQS = 3.0   # ~3 hours per 10M seqs on M4 Max MPS
GB_PER_10M_SEQS = 14.0     # ~14 GB per 10M seqs (embeddings + parquet + stats)

# Try to import psutil for machine specs (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# System info helpers
# ---------------------------------------------------------------------------

def get_machine_specs() -> Dict:
    """Collect machine specs for the report."""
    import torch

    specs = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    # RAM (requires psutil)
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        specs["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        specs["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    else:
        specs["ram_total_gb"] = "unknown (install psutil for details)"

    # GPU info
    if torch.cuda.is_available():
        specs["device_type"] = "cuda"
        specs["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        specs["gpu_vram_gb"] = round(props.total_mem / (1024 ** 3), 1)
        specs["gpu_count"] = torch.cuda.device_count()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        specs["device_type"] = "mps"
        specs["gpu_name"] = "Apple Silicon (MPS)"
        specs["gpu_vram_gb"] = "shared (unified memory)"
    else:
        specs["device_type"] = "cpu"
        specs["gpu_name"] = "none"

    return specs


def detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_time(seconds: float) -> str:
    """Format seconds as a human-readable string (e.g., '1.5h' or '45.2s')."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h ({seconds:.0f}s)"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}min ({seconds:.0f}s)"
    else:
        return f"{seconds:.1f}s"


def _compute_dir_size_bytes(directory: Path) -> int:
    """Compute total size of all files in a directory (non-recursive)."""
    total = 0
    if directory.exists():
        for f in directory.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / 1024:.1f} KB"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path, verbose: int) -> logging.Logger:
    """Set up logging to both file and console."""
    log = logging.getLogger("embedding")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    # File handler — always DEBUG level
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    # Console handler — level depends on verbosity
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    return log


# ---------------------------------------------------------------------------
# ESM-2 model loading
# ---------------------------------------------------------------------------

def load_esm2_model(device: str, log: logging.Logger):
    """
    Load ESM-2 model with sanity checks.

    Returns:
        (model, alphabet, batch_converter, repr_layer, torch_device, load_time)
    """
    import esm
    import torch

    log.info(f"Loading ESM-2 model ({ESM2_MODEL_NAME})...")
    t0 = time.time()

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

    # Sanity checks
    num_layers = model.num_layers
    embed_dim = getattr(model, "embed_dim", None) or model.args.embed_dim
    if num_layers != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {ESM2_MODEL_NAME} with {EXPECTED_NUM_LAYERS} layers, "
            f"got {num_layers} layers. Wrong model loaded?"
        )
    if embed_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {embed_dim}. "
            f"Wrong model loaded?"
        )

    repr_layer = num_layers  # last layer (30)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    load_time = time.time() - t0
    log.info(f"Model loaded in {load_time:.1f}s (device={device}, layers={num_layers}, dim={embed_dim})")

    return model, alphabet, batch_converter, repr_layer, torch_device, load_time


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings_for_sequences(
    sequences: List[str],
    model,
    batch_converter,
    repr_layer: int,
    device,
    batch_size: int,
    log: logging.Logger,
    verbose: int,
) -> np.ndarray:
    """
    Compute ESM-2 mean-pooled embeddings for a list of CDR3 amino acid sequences.

    Uses float32 for accumulation during mean pooling (numerical precision),
    then casts to float16 for storage efficiency. The BOS token (position 0)
    and EOS token (position L+1) are excluded — only the L amino acid positions
    [1:L+1] are averaged.

    Args:
        sequences: List of CDR3 AA strings.
        model: Loaded ESM-2 model.
        batch_converter: ESM alphabet batch converter.
        repr_layer: Which layer to extract representations from.
        device: torch device.
        batch_size: Sequences per batch.
        log: Logger instance.
        verbose: Verbosity level.

    Returns:
        np.ndarray of shape (len(sequences), EMBEDDING_DIM), dtype float16.
    """
    import torch

    n = len(sequences)
    if n == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float16)

    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n, batch_size)):
            batch_seqs = sequences[start: start + batch_size]
            data = [(f"seq{i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps = results["representations"][repr_layer]  # (batch, L+2, 640)

            for j, seq in enumerate(batch_seqs):
                L = len(seq)
                embeddings[start + j] = reps[j, 1: L + 1].mean(0).cpu().float().numpy()

            if verbose >= 2:
                log.debug(
                    f"  Batch {batch_idx + 1}/{n_batches}: "
                    f"embedded {min(start + batch_size, n)}/{n} sequences"
                )

    # Cast to float16 for storage
    return embeddings.astype(np.float16)


# ---------------------------------------------------------------------------
# Per-participant processing
# ---------------------------------------------------------------------------

def process_participant(
    participant_label: str,
    loader: MalIDPublishedDataLoader,
    model,
    batch_converter,
    repr_layer: int,
    device,
    batch_size: int,
    output_dir: Path,
    log: logging.Logger,
    verbose: int,
) -> Dict:
    """
    Load CLEAN data, downsample, embed, and save for one participant.

    Returns:
        Stats dict for this participant.
    """
    stats = {
        "participant_label": participant_label,
        "timestamp": datetime.now().isoformat(),
    }

    # Load CLEAN and downsample via the data loader
    t0 = time.time()
    df_downsampled = loader.load_participant_data(
        participant_label, PreprocessingStage.DOWNSAMPLED
    )
    preprocess_time = time.time() - t0

    stats["preprocess_time_seconds"] = round(preprocess_time, 3)

    if df_downsampled.empty:
        # Participant had no data after downsampling
        stats["kept"] = False
        stats["n_sequences_downsampled"] = 0
        stats["n_specimens"] = 0
        stats["n_specimens_kept"] = 0
        stats["embedding_time_seconds"] = 0.0

        # Save empty files for unambiguous "processed but empty" signal
        empty_emb = np.zeros((0, EMBEDDING_DIM), dtype=np.float16)
        np.save(output_dir / f"{participant_label}_embeddings.npy", empty_emb)
        df_downsampled.to_parquet(
            output_dir / f"{participant_label}_downsampled.parquet", index=False
        )
    else:
        n_seqs = len(df_downsampled)
        n_specimens = df_downsampled["repertoire_id"].nunique() if "repertoire_id" in df_downsampled.columns else 0

        stats["kept"] = True
        stats["n_sequences_downsampled"] = n_seqs
        stats["n_specimens"] = n_specimens
        stats["n_specimens_kept"] = n_specimens

        # Save the DOWNSAMPLED parquet (source of truth for alignment)
        df_downsampled.to_parquet(
            output_dir / f"{participant_label}_downsampled.parquet", index=False
        )

        # Extract CDR3 sequences
        if CDR3_COL not in df_downsampled.columns:
            raise ValueError(
                f"Column '{CDR3_COL}' not found in downsampled data for {participant_label}. "
                f"Available columns: {list(df_downsampled.columns)}"
            )
        cdr3_sequences = df_downsampled[CDR3_COL].tolist()

        # Compute embeddings
        t0 = time.time()
        embeddings = compute_embeddings_for_sequences(
            cdr3_sequences, model, batch_converter, repr_layer,
            device, batch_size, log, verbose,
        )
        embedding_time = time.time() - t0
        stats["embedding_time_seconds"] = round(embedding_time, 3)
        stats["sequences_per_second"] = round(n_seqs / embedding_time, 1) if embedding_time > 0 else 0

        # Save embeddings
        np.save(output_dir / f"{participant_label}_embeddings.npy", embeddings)

    # Save per-participant stats
    with open(output_dir / f"{participant_label}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_embeddings(output_dir: Path, log: logging.Logger) -> bool:
    """
    Verify existing embedding files for consistency.

    Checks:
        - Each _embeddings.npy has a matching _downsampled.parquet and _stats.json
        - Embedding shape matches parquet row count
        - Embedding dtype is float16
        - Embedding dim is 640

    Returns:
        True if all checks pass, False otherwise.
    """
    log.info("Verifying existing embedding files...")
    npy_files = sorted(output_dir.glob("*_embeddings.npy"))

    if not npy_files:
        log.warning("No embedding files found to verify.")
        return True

    issues = []
    n_checked = 0

    for npy_path in npy_files:
        participant_label = npy_path.stem.replace("_embeddings", "")
        parquet_path = output_dir / f"{participant_label}_downsampled.parquet"
        stats_path = output_dir / f"{participant_label}_stats.json"

        # Check companion files exist
        if not parquet_path.exists():
            issues.append(f"{participant_label}: missing _downsampled.parquet")
            continue
        if not stats_path.exists():
            issues.append(f"{participant_label}: missing _stats.json")
            continue

        # Load and check
        emb = np.load(npy_path)
        df = pd.read_parquet(parquet_path)

        if emb.dtype != np.float16:
            issues.append(f"{participant_label}: dtype={emb.dtype}, expected float16")

        if emb.ndim != 2 or (emb.shape[0] > 0 and emb.shape[1] != EMBEDDING_DIM):
            issues.append(f"{participant_label}: shape={emb.shape}, expected (N, {EMBEDDING_DIM})")

        if emb.shape[0] != len(df):
            issues.append(
                f"{participant_label}: row mismatch — "
                f"embeddings={emb.shape[0]}, parquet={len(df)}"
            )

        n_checked += 1

    if issues:
        log.error(f"Verification FAILED — {len(issues)} issue(s):")
        for issue in issues:
            log.error(f"  - {issue}")
        return False

    log.info(f"Verification PASSED — {n_checked} participants checked, all consistent.")
    return True


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    output_dir: Path,
    all_stats: List[Dict],
    machine_specs: Dict,
    run_params: Dict,
    total_time: float,
    model_load_time: float,
    timestamp: str,
    log: logging.Logger,
) -> Path:
    """Generate a human-readable markdown report."""
    report_path = output_dir / f"embedding_report_{timestamp}.md"

    kept_stats = [s for s in all_stats if s.get("kept", False)]
    dropped_stats = [s for s in all_stats if not s.get("kept", False)]
    total_sequences = sum(s.get("n_sequences_downsampled", 0) for s in all_stats)
    total_embed_time = sum(s.get("embedding_time_seconds", 0) for s in all_stats)

    # Compute storage size
    storage_bytes = _compute_dir_size_bytes(output_dir)

    lines = []
    lines.append("# ESM-2 Embedding Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append("")

    # Machine specs
    lines.append("## Machine Specs")
    lines.append("")
    lines.append(f"- Platform: {machine_specs.get('platform', 'unknown')}")
    lines.append(f"- Processor: {machine_specs.get('processor', 'unknown')}")
    lines.append(f"- CPU cores: {machine_specs.get('cpu_count', 'unknown')}")
    ram = machine_specs.get('ram_total_gb', 'unknown')
    lines.append(f"- RAM: {ram} GB" if isinstance(ram, (int, float)) else f"- RAM: {ram}")
    lines.append(f"- Device: {machine_specs.get('device_type', 'unknown')}")
    gpu_name = machine_specs.get('gpu_name', 'none')
    if gpu_name != "none":
        lines.append(f"- GPU: {gpu_name}")
        vram = machine_specs.get('gpu_vram_gb', 'unknown')
        lines.append(f"- GPU VRAM: {vram} GB" if isinstance(vram, (int, float)) else f"- GPU VRAM: {vram}")
    lines.append(f"- Python: {machine_specs.get('python_version', 'unknown')}")
    lines.append(f"- PyTorch: {machine_specs.get('torch_version', 'unknown')}")
    lines.append(f"- NumPy: {machine_specs.get('numpy_version', 'unknown')}")
    if not HAS_PSUTIL:
        lines.append(f"- Note: install psutil for more detailed machine specs (pip install psutil)")
    lines.append("")

    # Run parameters
    lines.append("## Run Parameters")
    lines.append("")
    lines.append(f"- Model: {run_params['model_name']}")
    lines.append(f"- Embedding dim: {run_params['embedding_dim']}")
    lines.append(f"- Representation layer: {run_params['repr_layer']}")
    lines.append(f"- Batch size: {run_params['batch_size']}")
    lines.append(f"- Device: {run_params['device']}")
    lines.append(f"- Storage dtype: float16")
    lines.append(f"- Mal-ID-Lite version: {run_params['malid_version']}")
    lines.append("")

    # Aggregate stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total participants: {len(all_stats)}")
    lines.append(f"- Participants with data: {len(kept_stats)}")
    lines.append(f"- Participants dropped (no data after downsampling): {len(dropped_stats)}")
    lines.append(f"- Total sequences embedded: {total_sequences:,}")
    lines.append(f"- Model load time: {_format_time(model_load_time)}")
    lines.append(f"- Total embedding time: {_format_time(total_embed_time)}")
    lines.append(f"- Total wall time: {_format_time(total_time)}")
    if total_embed_time > 0 and total_sequences > 0:
        throughput = total_sequences / total_embed_time
        lines.append(f"- Throughput: {throughput:,.0f} sequences/second")
        if kept_stats:
            lines.append(f"- Average per participant: {_format_time(total_embed_time / len(kept_stats))}")
    lines.append(f"- Total storage size: {_format_size(storage_bytes)}")
    lines.append("")

    # Per-participant table
    lines.append("## Per-Participant Details")
    lines.append("")
    lines.append("| Participant | Kept | Sequences | Specimens | Embed Time (s) | Seq/s |")
    lines.append("|-------------|------|-----------|-----------|----------------|-------|")
    for s in sorted(all_stats, key=lambda x: x["participant_label"]):
        kept_str = "yes" if s.get("kept", False) else "no"
        n_seq = s.get("n_sequences_downsampled", 0)
        n_spec = s.get("n_specimens", 0)
        emb_time = s.get("embedding_time_seconds", 0)
        seq_per_s = s.get("sequences_per_second", 0)
        lines.append(
            f"| {s['participant_label']} | {kept_str} | {n_seq:,} | {n_spec} | {emb_time:.1f} | {seq_per_s:,.0f} |"
        )
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    log.info(f"Report saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ESM-2 embeddings for Model 3 (sequence-level classifier). "
            "Pre-computes per-participant embeddings from DOWNSAMPLED CDR3 sequences. "
            "Run this BEFORE training Model 3, or use --compute-embeddings in train_model3.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Resource estimates (MacBook Pro M4 Max, 64GB, MPS):\n"
            f"  Time:    ~{HOURS_PER_10M_SEQS:.0f} hours per 10 million downsampled sequences\n"
            f"  Storage: ~{GB_PER_10M_SEQS:.0f} GB per 10 million downsampled sequences\n"
            "\n"
            "These are approximate. Actual values depend on hardware, sequence\n"
            "lengths, and batch size. CUDA GPUs with more VRAM are significantly\n"
            "faster. CPU-only mode is much slower.\n"
        ),
    )
    # --- Data and cache paths ---
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to the metadata TSV file (e.g., data/metadata.tsv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Path to raw data directory (AIRR-format files). "
            "Required if the participant cache does not exist yet. "
            "Not needed when a complete cache is available."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for preprocessed data. "
            "Default: cache/<dataset-name>/ under the project root."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset name, used as subdirectory under cache/ (default: {DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--gene-locus",
        default="TCR",
        choices=["TCR"],
        help="Gene locus (default: TCR). Only TCR is supported at the moment.",
    )

    # --- Embedding parameters ---
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Sequences per batch. Auto-selected per device if omitted "
             f"(mps={DEFAULT_BATCH_SIZES['mps']}, cuda={DEFAULT_BATCH_SIZES['cuda']}, "
             f"cpu={DEFAULT_BATCH_SIZES['cpu']}). "
             "Override to tune for your hardware.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity: 0=silent, 1=per-participant progress (default), 2=also per-batch.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help=(
            "Only verify existing embedding files for consistency (no embedding "
            "computation). Exits with code 0 if all checks pass, 1 otherwise."
        ),
    )
    args = parser.parse_args()

    # --- Resolve paths ---
    cache_base = args.cache_dir or (PROJECT_ROOT / "cache" / args.dataset_name)
    participants_dir = cache_base / "participants"
    output_dir = cache_base / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate metadata path
    if not args.metadata_path.exists():
        print(f"Error: --metadata-path does not exist: {args.metadata_path}", file=sys.stderr)
        sys.exit(1)
    if args.data_dir is not None and not args.data_dir.exists():
        print(f"Error: --data-dir does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logging
    log_file = output_dir / f"embedding_log_{timestamp}.log"
    log = setup_logging(log_file, args.verbose)

    log.info("=" * 70)
    log.info("ESM-2 Embedding Computation for Mal-ID-Lite Model 3")
    log.info("=" * 70)

    # Verify mode
    if args.verify:
        ok = verify_embeddings(output_dir, log)
        sys.exit(0 if ok else 1)

    # Resource estimate warning
    log.info("")
    log.info("Resource estimates (approximate, based on M4 Max MPS benchmarks):")
    log.info(f"  Time:    ~{HOURS_PER_10M_SEQS:.0f} hours per 10M downsampled sequences")
    log.info(f"  Storage: ~{GB_PER_10M_SEQS:.0f} GB per 10M downsampled sequences")
    log.info("")

    # Check if data-dir is needed
    clean_parquets_exist = (
        participants_dir.exists()
        and any(participants_dir.glob("*_clean.parquet"))
    )
    if not clean_parquets_exist and args.data_dir is None:
        log.error(
            f"No existing participant cache found at {participants_dir}. "
            "Provide --data-dir so the cache can be built."
        )
        sys.exit(1)

    # Initialize data loader (needed for participant discovery and downsampling)
    log.info("Initializing data loader...")
    loader = MalIDPublishedDataLoader(
        data_dir=args.data_dir or Path("."),  # placeholder if cache covers all reads
        metadata_path=args.metadata_path,
        gene_locus=args.gene_locus,
        verbose=0,
        cache_dir=cache_base,
    )

    # Check participant cache exists; build it if missing
    clean_parquets = sorted(participants_dir.glob("*_clean.parquet")) if participants_dir.exists() else []
    if not clean_parquets:
        all_participant_labels = sorted(loader.metadata["participant_label"].unique())
        total_participants = len(all_participant_labels)
        log.info(
            f"Participant CLEAN cache not found at {participants_dir}. "
            f"Building it now for {total_participants} participants..."
        )
        participants_dir.mkdir(parents=True, exist_ok=True)
        for idx, label in enumerate(all_participant_labels, 1):
            if idx % 50 == 0 or idx == 1:
                log.info(f"  Caching participant {idx}/{total_participants}: {label}")
            loader.load_participant_data(label, PreprocessingStage.CLEAN)
        clean_parquets = sorted(participants_dir.glob("*_clean.parquet"))
        log.info(f"Participant cache built: {len(clean_parquets)} participants cached.")

    all_participant_labels = [p.stem.removesuffix("_clean") for p in clean_parquets]
    total_participants = len(all_participant_labels)

    # Check which are already done (resumption)
    already_done = set()
    for label in all_participant_labels:
        stats_path = output_dir / f"{label}_stats.json"
        if stats_path.exists():
            already_done.add(label)

    remaining_labels = [l for l in all_participant_labels if l not in already_done]
    n_remaining = len(remaining_labels)

    log.info(f"Participants found: {total_participants}")
    log.info(f"Already processed: {len(already_done)}")
    log.info(f"Remaining: {n_remaining}")

    # Device detection
    device = args.device or detect_device()
    log.info(f"Device: {device}")

    # Batch size: use user override, or auto-select based on device
    if args.batch_size is not None:
        batch_size = args.batch_size
        log.info(f"Batch size: {batch_size} (user-specified)")
    else:
        batch_size = DEFAULT_BATCH_SIZES.get(device, 64)
        log.info(f"Batch size: {batch_size} (auto-selected for {device}, override with --batch-size)")

    if n_remaining == 0:
        log.info("All participants already processed. Use --verify to check consistency.")
        # Still generate report from existing stats
        all_stats = []
        for label in all_participant_labels:
            stats_path = output_dir / f"{label}_stats.json"
            with open(stats_path) as f:
                all_stats.append(json.load(f))
        machine_specs = get_machine_specs()
        run_params = {
            "model_name": ESM2_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "repr_layer": EXPECTED_NUM_LAYERS,
            "batch_size": batch_size,
            "device": device,
            "malid_version": MALID_VERSION,
        }
        generate_report(output_dir, all_stats, machine_specs, run_params, 0, 0, timestamp, log)
        sys.exit(0)

    if not HAS_PSUTIL:
        log.info(
            "Note: psutil is not installed. Install it for detailed machine specs "
            "in the report: pip install psutil"
        )

    # Collect machine specs
    machine_specs = get_machine_specs()
    log.info(f"Machine: {machine_specs.get('platform', 'unknown')}")
    if machine_specs.get("gpu_name", "none") != "none":
        log.info(f"GPU: {machine_specs['gpu_name']}")

    # Load ESM-2 model
    model, alphabet, batch_converter, repr_layer, torch_device, model_load_time = \
        load_esm2_model(device, log)

    # Process participants
    log.info(f"\nProcessing {n_remaining} participants (batch_size={batch_size})...")
    log.info("-" * 70)

    all_stats_new = []
    total_start = time.time()

    for idx, participant_label in enumerate(remaining_labels, 1):
        if args.verbose >= 1:
            log.info(f"Processing {idx}/{n_remaining}: {participant_label}")

        try:
            stats = process_participant(
                participant_label=participant_label,
                loader=loader,
                model=model,
                batch_converter=batch_converter,
                repr_layer=repr_layer,
                device=torch_device,
                batch_size=batch_size,
                output_dir=output_dir,
                log=log,
                verbose=args.verbose,
            )
            all_stats_new.append(stats)

            if args.verbose >= 1:
                n_seq = stats.get("n_sequences_downsampled", 0)
                emb_time = stats.get("embedding_time_seconds", 0)
                kept = "kept" if stats.get("kept", False) else "DROPPED"
                log.info(
                    f"  {kept}: {n_seq:,} sequences, "
                    f"embedding: {emb_time:.1f}s"
                )

        except Exception as e:
            log.error(f"  FAILED: {participant_label}: {e}")
            # Save error stats so we don't retry on resume
            error_stats = {
                "participant_label": participant_label,
                "timestamp": datetime.now().isoformat(),
                "kept": False,
                "error": str(e),
                "n_sequences_downsampled": 0,
                "embedding_time_seconds": 0,
            }
            with open(output_dir / f"{participant_label}_stats.json", "w") as f:
                json.dump(error_stats, f, indent=2)
            all_stats_new.append(error_stats)

    total_time = time.time() - total_start

    log.info("-" * 70)
    log.info(f"Embedding complete: {n_remaining} participants in {_format_time(total_time)}")

    # Collect ALL stats (including previously done)
    all_stats = []
    for label in all_participant_labels:
        stats_path = output_dir / f"{label}_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats.append(json.load(f))

    # Save global cache_info.json
    total_sequences = sum(s.get("n_sequences_downsampled", 0) for s in all_stats)
    n_kept = sum(1 for s in all_stats if s.get("kept", False))
    n_dropped = sum(1 for s in all_stats if not s.get("kept", False))

    run_params = {
        "model_name": ESM2_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "repr_layer": repr_layer,
        "batch_size": batch_size,
        "device": device,
        "malid_version": MALID_VERSION,
    }

    storage_bytes = _compute_dir_size_bytes(output_dir)

    cache_info = {
        "created_at": datetime.now().isoformat(),
        "malid_version": MALID_VERSION,
        "cache_type": "embeddings",
        "model_name": ESM2_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "repr_layer": repr_layer,
        "storage_dtype": "float16",
        "batch_size": batch_size,
        "device": device,
        "source_cache_dir": str(participants_dir),
        "n_participants_total": total_participants,
        "n_participants_kept": n_kept,
        "n_participants_dropped": n_dropped,
        "total_sequences_embedded": total_sequences,
        "model_load_time_seconds": round(model_load_time, 1),
        "total_time_seconds": round(total_time, 1),
        "total_storage_bytes": storage_bytes,
        "machine_specs": machine_specs,
    }
    with open(output_dir / "cache_info.json", "w") as f:
        json.dump(cache_info, f, indent=2)
    log.info(f"Cache info saved: {output_dir / 'cache_info.json'}")

    # Generate report
    generate_report(
        output_dir, all_stats, machine_specs, run_params,
        total_time, model_load_time, timestamp, log,
    )

    # End-of-run verification
    log.info("\nRunning post-completion verification...")
    verification_ok = verify_embeddings(output_dir, log)
    if not verification_ok:
        log.error("Post-completion verification FAILED. Some files may be corrupt.")
    else:
        log.info("Post-completion verification PASSED.")

    # Final summary to console
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Participants: {n_kept} kept, {n_dropped} dropped, {total_participants} total")
    log.info(f"Sequences embedded: {total_sequences:,}")
    log.info(f"Model load time: {_format_time(model_load_time)}")
    log.info(f"Total time: {_format_time(total_time)}")
    if total_sequences > 0:
        total_embed_time = sum(s.get("embedding_time_seconds", 0) for s in all_stats)
        if total_embed_time > 0:
            log.info(f"Throughput: {total_sequences / total_embed_time:,.0f} seq/s")
    log.info(f"Storage: {_format_size(storage_bytes)}")
    log.info(f"Output: {output_dir}")
    log.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
