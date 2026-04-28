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
    malid_cross_validation_fold_id_when_in_test_set.
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

--symlink  (optional, default: off)
    Create symbolic links to reference files instead of copying them.
    Saves disk space and is much faster, but the subset cache becomes
    dependent on the reference cache remaining in place.

--force  (optional, default: off)
    If the output cache directory already exists, delete it and recreate
    from scratch. Without this flag, the script errors if the output
    directory exists.

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

REQUIRED_METADATA_COLS = [
    "participant_label",
    "specimen_label",
    "disease",
    "malid_cross_validation_fold_id_when_in_test_set",
]

# Per-participant files that must exist in the reference cache
PARTICIPANT_CACHE_FILES = [
    "participants/{label}_clean.parquet",
    "participants/{label}_stats.json",
]
EMBEDDING_CACHE_FILES = [
    "embeddings/{label}_embeddings.npy",
    "embeddings/{label}_downsampled.parquet",
    "embeddings/{label}_stats.json",
]
ALL_REQUIRED_FILES = PARTICIPANT_CACHE_FILES + EMBEDDING_CACHE_FILES


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
) -> None:
    """Verify all required cache files exist in the reference cache.

    Checks participants/ (2 files) and embeddings/ (3 files) per participant.
    Errors with a summary of all missing files.

    Parameters
    ----------
    subset_participants : list
        Unique participant labels to check.
    ref_cache_dir : Path
        Path to the reference cache directory.
    """
    missing = []

    for label in subset_participants:
        for pattern in ALL_REQUIRED_FILES:
            file_path = ref_cache_dir / pattern.format(label=label)
            if not file_path.exists():
                missing.append(str(file_path.relative_to(ref_cache_dir)))

    if missing:
        n_participants_affected = len(set(
            m.split("/")[1].rsplit("_", 1)[0] for m in missing
        ))
        print(
            f"Error: {len(missing)} required file(s) missing from reference cache "
            f"(affecting {n_participants_affected} participant(s)):\n",
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
) -> dict:
    """Copy (or symlink) participant and embedding files to the output cache.

    Parameters
    ----------
    subset_participants : list
        Participant labels to copy.
    ref_cache_dir : Path
        Source reference cache directory.
    output_cache_dir : Path
        Destination cache directory.
    use_symlinks : bool
        If True, create symbolic links instead of copying files.

    Returns
    -------
    dict
        Summary with keys: n_files_copied, total_bytes.
    """
    participants_dir = output_cache_dir / "participants"
    embeddings_dir = output_cache_dir / "embeddings"
    participants_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    total_bytes = 0
    action = "Linking" if use_symlinks else "Copying"

    n_total = len(subset_participants)
    for idx, label in enumerate(subset_participants, 1):
        if idx % 50 == 0 or idx == 1 or idx == n_total:
            print(f"  {action} {idx}/{n_total}: {label}")

        for pattern in ALL_REQUIRED_FILES:
            src = ref_cache_dir / pattern.format(label=label)
            dst = output_cache_dir / pattern.format(label=label)

            if use_symlinks:
                # Use absolute path for symlink target
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
) -> bool:
    """Validate that all expected files exist in the output cache.

    Parameters
    ----------
    subset_participants : list
        Expected participant labels.
    output_cache_dir : Path
        Output cache directory to validate.
    use_symlinks : bool
        If True, also verify symlinks are not broken.

    Returns
    -------
    bool
        True if all files are present and valid.
    """
    errors = []

    # Check metadata files
    for filename in ("metadata.tsv", "metadata_processed.tsv"):
        path = output_cache_dir / filename
        if not path.exists():
            errors.append(f"Missing: {filename}")

    # Check per-participant files
    for label in subset_participants:
        for pattern in ALL_REQUIRED_FILES:
            file_path = output_cache_dir / pattern.format(label=label)
            if not file_path.exists():
                errors.append(f"Missing: {pattern.format(label=label)}")
            elif use_symlinks and file_path.is_symlink():
                # Verify symlink target exists
                if not file_path.resolve().exists():
                    errors.append(f"Broken symlink: {pattern.format(label=label)}")

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
    fold_col = "malid_cross_validation_fold_id_when_in_test_set"
    folds = sorted(int(f) for f in metadata[fold_col].unique())
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
        fold_participants = participants_df[participants_df[fold_col] == fold_id]
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
            "      --symlink\n"
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

    # --- Resolve reference cache directory ---
    if args.ref_cache_dir is not None:
        ref_cache_dir = args.ref_cache_dir.resolve()
    else:
        ref_cache_dir = (DEFAULT_CACHE_BASE / args.ref_dataset_name).resolve()

    if not ref_cache_dir.exists():
        print(f"Error: reference cache directory does not exist: {ref_cache_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Resolve output cache directory ---
    if args.output_cache_dir is not None:
        output_cache_dir = args.output_cache_dir.resolve()
    else:
        output_cache_dir = (DEFAULT_CACHE_BASE / args.dataset_name).resolve()

    # --- Check ref and output don't point to the same directory ---
    if ref_cache_dir == output_cache_dir:
        print(
            f"Error: reference and output cache directories are the same:\n"
            f"  {ref_cache_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Handle existing output directory ---
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
    validate_ref_files(subset_participants, ref_cache_dir)
    n_expected_files = n_participants * len(ALL_REQUIRED_FILES)
    print(f"  All {n_expected_files} required files found ({len(PARTICIPANT_CACHE_FILES)} participant + {len(EMBEDDING_CACHE_FILES)} embedding files per participant).")

    # --- Step 4: Create output directory and copy/link files ---
    action = "Symlinking" if args.symlink else "Copying"
    print(f"Step 4: {action} files to {output_cache_dir}...")
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    result = copy_or_link_files(
        subset_participants, ref_cache_dir, output_cache_dir, args.symlink
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
    ok = validate_output(subset_participants, output_cache_dir, args.symlink)
    if not ok:
        sys.exit(1)
    print("  All files validated successfully.")

    # --- Step 7: Print summary ---
    print_fold_summary(metadata)

    print(f"{'=' * 60}")
    print(f"  Subset cache created successfully.")
    print(f"{'=' * 60}")
    print(f"  Output:    {output_cache_dir}")
    print(f"  Mode:      {'symlinks' if args.symlink else 'copies'}")
    print()
    print("  To train on this subset:")
    print(f"    python malid_lite/training/train_ensemble.py \\")
    print(f"        --metadata-path {output_cache_dir / 'metadata_processed.tsv'} \\")
    print(f"        --cache-dir {output_cache_dir} \\")
    print(f"        --dataset-name {args.dataset_name} \\")
    print(f"        --classification-mode multiclass")
    print()


if __name__ == "__main__":
    main()
