#!/usr/bin/env python
"""Create a subset dataset cache from an existing (reference) cache.

Copies participant and embedding files for a subset of participants into a new
cache directory. The pipeline can then train on the subset without --data-dir,
since all other cache artifacts (data_folds/, splits/) are auto-generated.

Arguments
---------
--metadata-subset PATH  (required)
    Path to the subset metadata TSV file. Must contain columns:
    participant_label, specimen_label, disease,
    CV_fold (or legacy name malid_cross_validation_fold_id_when_in_test_set).
    All participants listed must exist in the reference cache.

--dataset-name NAME  (required)
    Name for the new subset dataset. Used as the subdirectory name under
    cache/ when --output-cache-dir is not provided, and in the training
    command for output organization (trained_models/<name>/...).

--ref-cache-dir PATH  (mutually exclusive with --ref-dataset-name)
    Explicit path to the reference cache directory containing participants/
    and embeddings/ subdirectories.

--ref-dataset-name NAME  (mutually exclusive with --ref-cache-dir)
    Name of the reference dataset. Resolves to cache/<name>/ under the
    project root. Exactly one of --ref-cache-dir or --ref-dataset-name
    must be provided.

--output-cache-dir PATH  (optional)
    Explicit output path for the new subset cache. If omitted, defaults to
    cache/<dataset-name>/ under the project root.

--ref-embedding-dir PATH  (optional)
    Reference embedding directory to read embeddings from. If omitted,
    defaults to <ref-cache-dir>/embeddings/. Use when reference embeddings
    were written to a custom location (e.g., via
    compute_model3_embeddings.py --output-embedding-dir).

--output-embedding-dir PATH  (optional)
    Output embedding directory to write subset embeddings to. If omitted,
    defaults to <output-cache-dir>/embeddings/. Use to write subset
    embeddings to a custom location separate from the cache directory.

--symlink  (optional, default: off)
    Create symbolic links to reference files instead of copying them.
    Saves disk space and is much faster, but the subset cache becomes
    dependent on the reference cache remaining in place.

--force  (optional, default: off)
    If the output cache directory already exists, delete it and recreate
    from scratch. Without this flag, the script errors if the output
    directory exists. When --output-embedding-dir is external, also
    deletes that directory if it exists.

Usage
-----
    # Using a reference cache directory:
    python scripts/data/create_subset_cache.py \\
        --metadata-subset path/to/subset_metadata.tsv \\
        --dataset-name "my-subset" \\
        --ref-cache-dir path/to/reference/cache

    # Using a reference dataset name (resolves to cache/<name>/ under project root):
    python scripts/data/create_subset_cache.py \\
        --metadata-subset path/to/subset_metadata.tsv \\
        --dataset-name "my-subset" \\
        --ref-dataset-name "mal-id-orig-data"

    # With symlinks instead of copies (saves disk space):
    python scripts/data/create_subset_cache.py \\
        --metadata-subset path/to/subset_metadata.tsv \\
        --dataset-name "my-subset" \\
        --ref-cache-dir path/to/reference/cache \\
        --symlink

    # With custom embedding directories:
    python scripts/data/create_subset_cache.py \\
        --metadata-subset path/to/subset_metadata.tsv \\
        --dataset-name "my-subset" \\
        --ref-cache-dir path/to/reference/cache \\
        --ref-embedding-dir /fast-storage/embeddings \\
        --output-embedding-dir /fast-storage/subset-embeddings

    # Force overwrite existing output cache:
    python scripts/data/create_subset_cache.py \\
        --metadata-subset path/to/subset_metadata.tsv \\
        --dataset-name "my-subset" \\
        --ref-cache-dir path/to/reference/cache \\
        --force
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_BASE = PROJECT_ROOT / "cache"

FOLD_COL = "CV_fold"
_LEGACY_FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"

REQUIRED_METADATA_COLS = [
    "participant_label",
    "specimen_label",
    "disease",
    FOLD_COL,
]

# Per-participant filename patterns (without subdirectory prefix).
# The subdirectory (participants/ or embedding dir) is resolved at runtime
# so that embedding files can live in a custom directory.
# Note: "{label}_stats.json" appears in both lists — they are different files
# in different directories (preprocessing stats vs embedding stats).
PARTICIPANT_FILE_PATTERNS = [
    "{label}_clean.parquet",
    "{label}_stats.json",
]
EMBEDDING_FILE_PATTERNS = [
    "{label}_embeddings.npy",
    "{label}_downsampled.parquet",
    "{label}_stats.json",
]


def validate_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load and validate subset metadata.

    Checks that all required columns are present.

    Parameters
    ----------
    metadata_path : Path
        Path to the subset metadata TSV.

    Returns
    -------
    pd.DataFrame
        Validated metadata.
    """
    if not metadata_path.exists():
        print(f"Error: metadata file not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    metadata = pd.read_csv(metadata_path, sep="\t")

    # Normalize legacy fold column name -> "CV_fold"
    if _LEGACY_FOLD_COL in metadata.columns and FOLD_COL not in metadata.columns:
        metadata = metadata.rename(columns={_LEGACY_FOLD_COL: FOLD_COL})

    missing_cols = [c for c in REQUIRED_METADATA_COLS if c not in metadata.columns]
    if missing_cols:
        print(
            f"Error: subset metadata is missing required columns: {missing_cols}\n"
            f"  Required: {REQUIRED_METADATA_COLS}\n"
            f"  Found: {list(metadata.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if metadata.empty:
        print("Error: subset metadata is empty (0 rows).", file=sys.stderr)
        sys.exit(1)

    # Check for NaN in required columns
    for col in REQUIRED_METADATA_COLS:
        n_nan = metadata[col].isna().sum()
        if n_nan > 0:
            print(
                f"Error: subset metadata has {n_nan} NaN value(s) in column '{col}'.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Check for duplicate (participant_label, specimen_label) pairs
    dup_mask = metadata.duplicated(subset=["participant_label", "specimen_label"])
    if dup_mask.any():
        n_dups = dup_mask.sum()
        examples = metadata.loc[dup_mask, ["participant_label", "specimen_label"]].head(10)
        print(
            f"Error: subset metadata has {n_dups} duplicate "
            f"(participant_label, specimen_label) pair(s):\n"
            f"  {examples.values.tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)

    return metadata


def validate_participants_in_ref(
    subset_participants: list,
    ref_cache_dir: Path,
) -> None:
    """Verify all subset participants exist in the reference metadata.

    Parameters
    ----------
    subset_participants : list
        Unique participant labels from the subset metadata.
    ref_cache_dir : Path
        Path to the reference cache directory.
    """
    # Check reference metadata exists
    ref_metadata_processed = ref_cache_dir / "metadata_processed.tsv"
    ref_metadata = ref_cache_dir / "metadata.tsv"

    if ref_metadata_processed.exists():
        ref_meta_path = ref_metadata_processed
    elif ref_metadata.exists():
        ref_meta_path = ref_metadata
    else:
        print(
            f"Error: no metadata file found in reference cache.\n"
            f"  Checked: {ref_metadata_processed}\n"
            f"           {ref_metadata}",
            file=sys.stderr,
        )
        sys.exit(1)

    ref_df = pd.read_csv(ref_meta_path, sep="\t")
    if "participant_label" not in ref_df.columns:
        print(
            f"Error: reference metadata ({ref_meta_path.name}) is missing "
            f"'participant_label' column.\n"
            f"  Available columns: {list(ref_df.columns)[:15]}",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_participants = set(ref_df["participant_label"].unique())

    missing_in_ref = [p for p in subset_participants if p not in ref_participants]
    if missing_in_ref:
        print(
            f"Error: {len(missing_in_ref)} participant(s) from subset metadata not found "
            f"in reference metadata ({ref_meta_path.name}):\n"
            f"  {missing_in_ref[:20]}"
            + (f"\n  ... and {len(missing_in_ref) - 20} more" if len(missing_in_ref) > 20 else ""),
            file=sys.stderr,
        )
        sys.exit(1)


def validate_ref_files(
    subset_participants: list,
    ref_cache_dir: Path,
    ref_embedding_dir: Path = None,
) -> None:
    """Verify all required cache files exist in the reference cache.

    Checks participants/ (2 files) and embedding dir (3 files) per participant.
    Errors with a summary of all missing files.

    Parameters
    ----------
    subset_participants : list
        Unique participant labels to check.
    ref_cache_dir : Path
        Path to the reference cache directory (contains participants/).
    ref_embedding_dir : Path or None
        Path to the reference embedding directory. If None, defaults to
        ref_cache_dir / "embeddings".
    """
    if ref_embedding_dir is None:
        ref_embedding_dir = ref_cache_dir / "embeddings"

    ref_participants_dir = ref_cache_dir / "participants"
    missing = []
    labels_with_missing = set()

    for label in subset_participants:
        for pattern in PARTICIPANT_FILE_PATTERNS:
            file_path = ref_participants_dir / pattern.format(label=label)
            if not file_path.exists():
                missing.append(f"participants/{pattern.format(label=label)}")
                labels_with_missing.add(label)
        for pattern in EMBEDDING_FILE_PATTERNS:
            file_path = ref_embedding_dir / pattern.format(label=label)
            if not file_path.exists():
                # Show the directory name for context in the error message
                missing.append(f"{ref_embedding_dir.name}/{pattern.format(label=label)}")
                labels_with_missing.add(label)

    if missing:
        print(
            f"Error: {len(missing)} required file(s) missing from reference cache "
            f"(affecting {len(labels_with_missing)} participant(s)):\n",
            file=sys.stderr,
        )
        for m in missing[:30]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more", file=sys.stderr)
        sys.exit(1)


def copy_or_link_files(
    subset_participants: list,
    ref_cache_dir: Path,
    output_cache_dir: Path,
    use_symlinks: bool,
    ref_embedding_dir: Path = None,
    output_embedding_dir: Path = None,
) -> dict:
    """Copy (or symlink) participant and embedding files to the output cache.

    Parameters
    ----------
    subset_participants : list
        Participant labels to copy.
    ref_cache_dir : Path
        Source reference cache directory (contains participants/).
    output_cache_dir : Path
        Destination cache directory (participants/ created inside).
    use_symlinks : bool
        If True, create symbolic links instead of copying files.
    ref_embedding_dir : Path or None
        Source embedding directory. If None, defaults to
        ref_cache_dir / "embeddings".
    output_embedding_dir : Path or None
        Destination embedding directory. If None, defaults to
        output_cache_dir / "embeddings".

    Returns
    -------
    dict
        Summary with keys: n_files, total_bytes.
    """
    if ref_embedding_dir is None:
        ref_embedding_dir = ref_cache_dir / "embeddings"
    if output_embedding_dir is None:
        output_embedding_dir = output_cache_dir / "embeddings"

    ref_participants_dir = ref_cache_dir / "participants"
    output_participants_dir = output_cache_dir / "participants"
    output_participants_dir.mkdir(parents=True, exist_ok=True)
    output_embedding_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    total_bytes = 0
    action = "Linking" if use_symlinks else "Copying"

    n_total = len(subset_participants)
    for idx, label in enumerate(subset_participants, 1):
        if idx % 50 == 0 or idx == 1 or idx == n_total:
            print(f"  {action} {idx}/{n_total}: {label}")

        # Participant files: ref_cache_dir/participants/ -> output_cache_dir/participants/
        for pattern in PARTICIPANT_FILE_PATTERNS:
            src = ref_participants_dir / pattern.format(label=label)
            dst = output_participants_dir / pattern.format(label=label)

            if use_symlinks:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

            n_files += 1
            total_bytes += src.stat().st_size

        # Embedding files: ref_embedding_dir/ -> output_embedding_dir/
        for pattern in EMBEDDING_FILE_PATTERNS:
            src = ref_embedding_dir / pattern.format(label=label)
            dst = output_embedding_dir / pattern.format(label=label)

            if use_symlinks:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

            n_files += 1
            total_bytes += src.stat().st_size

    return {"n_files": n_files, "total_bytes": total_bytes}


def save_metadata(metadata: pd.DataFrame, output_cache_dir: Path) -> None:
    """Save subset metadata as metadata.tsv and metadata_processed.tsv.

    Uses atomic writes (tempfile + rename) to prevent corruption.

    Parameters
    ----------
    metadata : pd.DataFrame
        The subset metadata to save.
    output_cache_dir : Path
        Destination cache directory.
    """
    for filename in ("metadata.tsv", "metadata_processed.tsv"):
        target = output_cache_dir / filename
        fd, tmp_path = tempfile.mkstemp(dir=output_cache_dir, suffix=".tsv")
        try:
            os.close(fd)
            metadata.to_csv(tmp_path, sep="\t", index=False)
            os.rename(tmp_path, target)
        except Exception:
            # Clean up temp file on failure
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
            raise


def validate_output(
    subset_participants: list,
    output_cache_dir: Path,
    use_symlinks: bool,
    output_embedding_dir: Path = None,
) -> bool:
    """Validate that all expected files exist in the output cache.

    Parameters
    ----------
    subset_participants : list
        Expected participant labels.
    output_cache_dir : Path
        Output cache directory (contains participants/).
    use_symlinks : bool
        If True, also verify symlinks are not broken.
    output_embedding_dir : Path or None
        Output embedding directory. If None, defaults to
        output_cache_dir / "embeddings".

    Returns
    -------
    bool
        True if all files are present and valid.
    """
    if output_embedding_dir is None:
        output_embedding_dir = output_cache_dir / "embeddings"

    output_participants_dir = output_cache_dir / "participants"
    errors = []

    # Check metadata files
    for filename in ("metadata.tsv", "metadata_processed.tsv"):
        path = output_cache_dir / filename
        if not path.exists():
            errors.append(f"Missing: {filename}")

    # Check per-participant files
    for label in subset_participants:
        for pattern in PARTICIPANT_FILE_PATTERNS:
            file_path = output_participants_dir / pattern.format(label=label)
            if not file_path.exists():
                errors.append(f"Missing: participants/{pattern.format(label=label)}")
            elif use_symlinks and file_path.is_symlink():
                if not file_path.resolve().exists():
                    errors.append(f"Broken symlink: participants/{pattern.format(label=label)}")

        for pattern in EMBEDDING_FILE_PATTERNS:
            file_path = output_embedding_dir / pattern.format(label=label)
            if not file_path.exists():
                errors.append(f"Missing: {output_embedding_dir.name}/{pattern.format(label=label)}")
            elif use_symlinks and file_path.is_symlink():
                if not file_path.resolve().exists():
                    errors.append(f"Broken symlink: {output_embedding_dir.name}/{pattern.format(label=label)}")

    if errors:
        print(
            f"\nValidation FAILED — {len(errors)} issue(s):",
            file=sys.stderr,
        )
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return False

    return True


def print_fold_summary(metadata: pd.DataFrame) -> None:
    """Print a summary of participants and disease distribution per fold.

    Parameters
    ----------
    metadata : pd.DataFrame
        The subset metadata.
    """
    folds = sorted(int(f) for f in metadata[FOLD_COL].unique())
    diseases = sorted(metadata["disease"].unique())

    # Per-participant summary (deduplicate specimens from same participant)
    participants_df = metadata.drop_duplicates(subset="participant_label")

    print(f"\n{'=' * 60}")
    print("  Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Participants: {participants_df['participant_label'].nunique()}")
    print(f"  Specimens:    {metadata['specimen_label'].nunique()}")
    print(f"  Diseases:     {len(diseases)} — {diseases}")
    print(f"  Folds:        {len(folds)} — {folds}")

    # Per-fold breakdown
    print(f"\n  {'Fold':<6} {'Participants':<14} ", end="")
    for d in diseases:
        # Truncate long disease names for table formatting
        label = d[:16] if len(d) > 16 else d
        print(f"{label:<18} ", end="")
    print()
    print(f"  {'-' * 6} {'-' * 14} ", end="")
    for _ in diseases:
        print(f"{'-' * 18} ", end="")
    print()

    for fold_id in folds:
        fold_participants = participants_df[participants_df[FOLD_COL] == fold_id]
        n_participants = len(fold_participants)
        print(f"  {fold_id:<6} {n_participants:<14} ", end="")
        for d in diseases:
            n_disease = (fold_participants["disease"] == d).sum()
            print(f"{n_disease:<18} ", end="")
        print()

    # Overall
    print(f"  {'-' * 6} {'-' * 14} ", end="")
    for _ in diseases:
        print(f"{'-' * 18} ", end="")
    print()
    print(f"  {'Total':<6} {len(participants_df):<14} ", end="")
    for d in diseases:
        n_disease = (participants_df["disease"] == d).sum()
        print(f"{n_disease:<18} ", end="")
    print()
    print()


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a subset dataset cache from an existing reference cache. "
            "Copies participant and embedding files for the specified participants, "
            "enabling training on the subset without --data-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # From a reference cache directory:\n"
            "  python scripts/data/create_subset_cache.py \\\n"
            "      --metadata-subset data/subset.tsv \\\n"
            "      --dataset-name my-subset \\\n"
            "      --ref-cache-dir cache/mal-id-orig-data\n\n"
            "  # From a reference dataset name:\n"
            "  python scripts/data/create_subset_cache.py \\\n"
            "      --metadata-subset data/subset.tsv \\\n"
            "      --dataset-name my-subset \\\n"
            "      --ref-dataset-name mal-id-orig-data\n\n"
            "  # With symlinks (saves disk space):\n"
            "  python scripts/data/create_subset_cache.py \\\n"
            "      --metadata-subset data/subset.tsv \\\n"
            "      --dataset-name my-subset \\\n"
            "      --ref-cache-dir cache/mal-id-orig-data \\\n"
            "      --symlink\n\n"
            "  # With custom embedding directories:\n"
            "  python scripts/data/create_subset_cache.py \\\n"
            "      --metadata-subset data/subset.tsv \\\n"
            "      --dataset-name my-subset \\\n"
            "      --ref-cache-dir cache/mal-id-orig-data \\\n"
            "      --ref-embedding-dir /fast-storage/embeddings \\\n"
            "      --output-embedding-dir /fast-storage/subset-embeddings\n"
        ),
    )

    # Required arguments
    parser.add_argument(
        "--metadata-subset",
        type=Path,
        required=True,
        help="Path to the subset metadata TSV file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the new subset dataset (used as cache subdirectory name).",
    )

    # Reference cache (mutually exclusive)
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--ref-cache-dir",
        type=Path,
        help="Path to the reference cache directory.",
    )
    ref_group.add_argument(
        "--ref-dataset-name",
        type=str,
        help=(
            "Name of the reference dataset. "
            f"Resolves to cache/<name>/ under the project root ({DEFAULT_CACHE_BASE}/)."
        ),
    )

    # Output location
    parser.add_argument(
        "--output-cache-dir",
        type=Path,
        default=None,
        help=(
            "Output cache directory for the subset. "
            f"Default: cache/<dataset-name>/ under project root ({DEFAULT_CACHE_BASE}/)."
        ),
    )

    # Embedding directory overrides
    parser.add_argument(
        "--ref-embedding-dir",
        type=Path,
        default=None,
        help=(
            "Reference embedding directory to read embeddings from. "
            "Default: <ref-cache-dir>/embeddings/. "
            "Use when reference embeddings were written to a custom location."
        ),
    )
    parser.add_argument(
        "--output-embedding-dir",
        type=Path,
        default=None,
        help=(
            "Output embedding directory to write subset embeddings to. "
            "Default: <output-cache-dir>/embeddings/. "
            "Use to place subset embeddings in a custom location."
        ),
    )

    # Options
    parser.add_argument(
        "--symlink",
        action="store_true",
        help=(
            "Create symbolic links instead of copying files. "
            "Saves disk space but the subset depends on the reference cache."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the output cache directory if it already exists.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Validate dataset name is a safe directory name ---
    if not args.dataset_name or "/" in args.dataset_name or args.dataset_name in (".", ".."):
        print(
            f"Error: --dataset-name must be a simple directory name, "
            f"got: '{args.dataset_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve reference cache directory ---
    if args.ref_cache_dir is not None:
        ref_cache_dir = args.ref_cache_dir.resolve()
    else:
        ref_cache_dir = (DEFAULT_CACHE_BASE / args.ref_dataset_name).resolve()

    if not ref_cache_dir.exists():
        print(
            f"Error: reference cache directory does not exist: {ref_cache_dir}\n"
            f"  Build it first with: python scripts/data/cache_and_report_all_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check participants/ subdirectory exists
    ref_participants_dir = ref_cache_dir / "participants"
    if not ref_participants_dir.exists():
        print(
            f"Error: reference cache has no participants/ directory: {ref_participants_dir}\n"
            f"  Build it first with: python scripts/data/cache_and_report_all_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve output cache directory ---
    if args.output_cache_dir is not None:
        output_cache_dir = args.output_cache_dir.resolve()
    else:
        output_cache_dir = (DEFAULT_CACHE_BASE / args.dataset_name).resolve()

    # --- Resolve embedding directories ---
    ref_embedding_dir = (
        args.ref_embedding_dir.resolve()
        if args.ref_embedding_dir is not None
        else ref_cache_dir / "embeddings"
    )
    output_embedding_dir = (
        args.output_embedding_dir.resolve()
        if args.output_embedding_dir is not None
        else output_cache_dir / "embeddings"
    )

    # --- Validate: ref embedding directory must exist ---
    if not ref_embedding_dir.exists():
        msg = f"Error: reference embedding directory does not exist: {ref_embedding_dir}"
        if args.ref_embedding_dir is not None:
            msg += "\n  (specified via --ref-embedding-dir)"
        else:
            msg += "\n  (default: <ref-cache-dir>/embeddings/)"
        print(msg, file=sys.stderr)
        sys.exit(1)

    # --- Check ref and output don't point to the same directory ---
    if ref_cache_dir == output_cache_dir:
        print(
            f"Error: reference and output cache directories are the same:\n"
            f"  {ref_cache_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Check ref and output embedding dirs aren't the same ---
    if ref_embedding_dir == output_embedding_dir:
        print(
            f"Error: reference and output embedding directories are the same:\n"
            f"  {ref_embedding_dir}\n"
            f"  This would copy files onto themselves. Use different directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine if output embedding dir is external (outside output_cache_dir)
    embedding_is_external = (output_embedding_dir != output_cache_dir / "embeddings")

    # --- Check output dirs aren't nested inside each other ---
    # --force deletes both output_cache_dir and output_embedding_dir via rmtree.
    # If one is a parent of the other, rmtree on the parent destroys both.
    if embedding_is_external:
        try:
            output_cache_dir.relative_to(output_embedding_dir)
            print(
                f"Error: output cache directory is inside the output embedding directory:\n"
                f"  Output cache dir:      {output_cache_dir}\n"
                f"  Output embedding dir:  {output_embedding_dir}\n"
                f"  --force would destroy both. Use non-nested directory trees.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass
        try:
            output_embedding_dir.relative_to(output_cache_dir)
            print(
                f"Error: output embedding directory is inside the output cache directory:\n"
                f"  Output embedding dir:  {output_embedding_dir}\n"
                f"  Output cache dir:      {output_cache_dir}\n"
                f"  --force would destroy both. Use non-nested directory trees.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass

    # --- Check for dangerous nesting (ref inside output) ---
    # --force deletes output_cache_dir recursively; if a reference directory
    # is nested inside it, --force would destroy the source data.
    for dir_label, dir_path in [
        ("Reference cache dir", ref_cache_dir),
        ("Reference embedding dir", ref_embedding_dir),
    ]:
        try:
            dir_path.relative_to(output_cache_dir)
            print(
                f"Error: {dir_label} is inside the output cache directory:\n"
                f"  {dir_label}: {dir_path}\n"
                f"  Output cache dir: {output_cache_dir}\n"
                f"  --force would destroy the reference data. "
                f"Use separate, non-nested directory trees.",
                file=sys.stderr,
            )
            sys.exit(1)
        except ValueError:
            pass  # not nested — safe

    # --force also deletes external output_embedding_dir; check ref dirs aren't inside it
    if embedding_is_external:
        for dir_label, dir_path in [
            ("Reference cache dir", ref_cache_dir),
            ("Reference embedding dir", ref_embedding_dir),
        ]:
            try:
                dir_path.relative_to(output_embedding_dir)
                print(
                    f"Error: {dir_label} is inside the output embedding directory:\n"
                    f"  {dir_label}: {dir_path}\n"
                    f"  Output embedding dir: {output_embedding_dir}\n"
                    f"  --force would destroy the reference data. "
                    f"Use separate, non-nested directory trees.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except ValueError:
                pass  # not nested — safe

    # --- Step 1: Validate subset metadata ---
    print("Step 1: Validating subset metadata...")
    metadata = validate_metadata(args.metadata_subset)
    subset_participants = sorted(metadata["participant_label"].unique())
    n_participants = len(subset_participants)
    n_specimens = metadata["specimen_label"].nunique()
    print(f"  {n_participants} participants, {n_specimens} specimens")

    # --- Step 2: Validate participants exist in reference metadata ---
    print("Step 2: Checking participants against reference metadata...")
    validate_participants_in_ref(subset_participants, ref_cache_dir)
    print(f"  All {n_participants} participants found in reference metadata.")

    # --- Step 3: Validate all required files exist in reference cache ---
    print("Step 3: Checking required files in reference cache...")
    validate_ref_files(
        subset_participants, ref_cache_dir, ref_embedding_dir=ref_embedding_dir
    )
    n_expected_files = n_participants * (len(PARTICIPANT_FILE_PATTERNS) + len(EMBEDDING_FILE_PATTERNS))
    print(
        f"  All {n_expected_files} required files found "
        f"({len(PARTICIPANT_FILE_PATTERNS)} participant + "
        f"{len(EMBEDDING_FILE_PATTERNS)} embedding files per participant)."
    )

    # --- Handle existing output directory ---
    # Placed after validation so if validation fails, existing output is preserved.
    if output_cache_dir.exists():
        if args.force:
            print(f"  --force: deleting existing output directory: {output_cache_dir}")
            shutil.rmtree(output_cache_dir)
        else:
            print(
                f"Error: output cache directory already exists: {output_cache_dir}\n"
                f"  Use --force to delete and recreate.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Handle existing external output embedding directory
    if embedding_is_external and output_embedding_dir.exists():
        if args.force:
            print(f"  --force: deleting existing output embedding directory: {output_embedding_dir}")
            shutil.rmtree(output_embedding_dir)
        else:
            print(
                f"Error: output embedding directory already exists: {output_embedding_dir}\n"
                f"  Use --force to delete and recreate.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Step 4: Create output directory and copy/link files ---
    action = "Symlinking" if args.symlink else "Copying"
    print(f"Step 4: {action} files to {output_cache_dir}...")
    if embedding_is_external:
        print(f"  Embeddings -> {output_embedding_dir}")
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    result = copy_or_link_files(
        subset_participants, ref_cache_dir, output_cache_dir, args.symlink,
        ref_embedding_dir=ref_embedding_dir,
        output_embedding_dir=output_embedding_dir,
    )
    print(
        f"  {result['n_files']} files {'linked' if args.symlink else 'copied'} "
        f"({format_size(result['total_bytes'])})"
    )

    # --- Step 5: Save metadata ---
    print("Step 5: Saving metadata...")
    save_metadata(metadata, output_cache_dir)
    print("  Saved metadata.tsv and metadata_processed.tsv")

    # --- Step 6: Validate output ---
    print("Step 6: Validating output cache...")
    ok = validate_output(
        subset_participants, output_cache_dir, args.symlink,
        output_embedding_dir=output_embedding_dir,
    )
    if not ok:
        sys.exit(1)
    print("  All files validated successfully.")

    # --- Step 7: Print summary ---
    print_fold_summary(metadata)

    print(f"{'=' * 60}")
    print(f"  Subset cache created successfully.")
    print(f"{'=' * 60}")
    print(f"  Output:    {output_cache_dir}")
    if embedding_is_external:
        print(f"  Embeddings: {output_embedding_dir}")
    print(f"  Mode:      {'symlinks' if args.symlink else 'copies'}")
    print()
    print("  To train on this subset:")
    print(f"    python malid_lite/training/train_ensemble.py \\")
    print(f"        --metadata-path {output_cache_dir / 'metadata_processed.tsv'} \\")
    print(f"        --cache-dir {output_cache_dir} \\")
    if embedding_is_external:
        print(f"        --model3-embedding-dir {output_embedding_dir} \\")
    print(f"        --dataset-name {args.dataset_name} \\")
    print(f"        --classification-mode multiclass")
    print()


if __name__ == "__main__":
    main()
