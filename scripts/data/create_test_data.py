#!/usr/bin/env python3
"""Create a small mock dataset for integration tests.

Produces a self-contained test data folder with subsampled real data that
can be used by all model integration tests instead of loading full folds
(~25M sequences). The mock data is small enough to commit to the repo.

Overview
--------
- Selects 4 disease classes × 12 participants (4 per fold) = 48 participants
- Includes 2 participants with 2 specimens each (50 specimens total)
- Subsamples ~5,000 sequences per specimen (with biased sampling to preserve
  convergent CDR3s needed by Model 2)
- Strips columns not used by any model (121 → 19 columns)
- Writes compressed .tsv.gz files + metadata.tsv to tests/test_data/

Output structure
----------------
tests/test_data/
├── metadata.tsv                               # subset metadata
├── raw/                                       # subsampled AIRR-format files
│   ├── part_table_<participant>.tsv.gz
│   └── ...
└── README.md                                  # documents the test data

The test data folder is self-contained: a MalIDPublishedDataLoader pointed
at it can build fold caches, load data, and run the full training pipeline
without accessing the real data at all.

Columns kept from raw data
--------------------------
Metadata columns (from metadata.tsv):
  - participant_label    : primary participant identifier, used for splitting/groupby
  - specimen_label       : specimen identifier, used for per-specimen features
  - disease              : prediction target (y), used for stratified splits
  - malid_cross_validation_fold_id_when_in_test_set : CV fold assignment
  - available_gene_loci  : filters participants by locus (TCR/BCR)

Sequence columns (from part_table_*.tsv.gz):
  - repertoire_id        : AIRR specimen ID, renamed to specimen_label after Stage 2
  - productive           : boolean filter ("T" = productive), Stage 1
  - v_score              : alignment score, Stage 1 filter (>80 for TCR)
  - v_call               : V gene call with allele, used for V gene features (Model 1)
  - j_call               : J gene call with allele, used for J gene features (Model 1)
  - cdr3_aa              : CDR3 amino acid sequence, used by Models 2 and 3
  - fwr1_aa              : framework region 1 AA, overwritten from gene reference
  - cdr1_aa              : CDR1 AA, overwritten from gene reference
  - fwr2_aa              : framework region 2 AA, overwritten from gene reference
  - cdr2_aa              : CDR2 AA, overwritten from gene reference
  - fwr3_aa              : framework region 3 AA, cleaned in Stage 1
  - fwr4_aa              : framework region 4 AA, cleaned in Stage 1
  - sequence             : full nucleotide sequence, deduplication key in Stage 1
  - replicate_label      : technical replicate ID, deduplication key in Stage 1
  - clone_id             : clone identifier, mapped to igh_or_tcrb_clone_id in Stage 1
  - stop_codon           : AIRR boolean, uppercased on load
  - vj_in_frame          : AIRR boolean, uppercased on load
  - amplification_label  : amplification method, groupby key during downsampling
  - participant_label    : participant identifier (also in sequence files)

Columns NOT included (not used by any model):
  - sequence_id, locus, d_call, d_score, j_score, junction, junction_aa,
    all v/d/j_sequence_alignment, rev_comp, germline_alignment, v/d/j_cigar,
    participant2_label, participant_alt_label, participant_age, participant_sex,
    participant_ethnicity, participant_diagnosis, participant_description,
    participant_species, sample_label, specimen_tissue, specimen_cell_subset,
    specimen_time_point, specimen_collected_on, specimen_description,
    specimen_affinity, amplification_done_on, amplification_description,
    amplification_type, primer_set, forward_primer, reverse_primer,
    part_tcrb_id, run_id, run_label, trimmed_read_id, strand,
    n1_sequence, n2_sequence, n1_overlap, n2_overlap,
    q_start, q_end, v_start, v_end, d_start, d_end, j_start, j_end,
    all pre_seq_nt_*, fr*_seq_nt_*, cdr*_seq_nt_*, post_seq_nt_*,
    all pre_seq_aa_*, insertions_*, deletions_*

Usage
-----
    python scripts/data/create_test_data.py

    # With custom parameters:
    python scripts/data/create_test_data.py \\
        --seqs-per-specimen 2000 \\
        --diseases "HIV" "Covid19" "T1D" "Healthy/Background" \\
        --participants-per-fold 4

Expected runtime: ~2-5 minutes (reads full data for selected participants)
"""

import argparse
import gzip
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Project root (scripts/data/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Source data paths (original Mal-ID data)
SOURCE_DATA_DIR = Path(
    "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/"
    "data_clean/airr_format_clean/TCR"
)
SOURCE_METADATA_PATH = Path(
    "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv"
)

# Output path
OUTPUT_DIR = PROJECT_ROOT / "tests" / "test_data"

# Metadata columns to keep
METADATA_COLS = [
    "participant_label",
    "specimen_label",
    "disease",
    "malid_cross_validation_fold_id_when_in_test_set",
    "available_gene_loci",
]

# Sequence columns to keep (from part_table_*.tsv.gz AIRR files)
SEQUENCE_COLS = [
    "repertoire_id",
    "productive",
    "v_score",
    "v_call",
    "j_call",
    "cdr3_aa",
    "fwr1_aa",
    "cdr1_aa",
    "fwr2_aa",
    "cdr2_aa",
    "fwr3_aa",
    "fwr4_aa",
    "sequence",
    "replicate_label",
    "clone_id",
    "stop_codon",
    "vj_in_frame",
    "amplification_label",
    "participant_label",
]

# Default diseases to include (3 diseases + healthy)
DEFAULT_DISEASES = ["HIV", "Covid19", "T1D", "Healthy/Background"]

# Default participants per disease per fold
DEFAULT_PARTICIPANTS_PER_FOLD = 4

# Default sequences per specimen
DEFAULT_SEQS_PER_SPECIMEN = 5000

# Minimum convergent CDR3 occurrences (across specimens of the same disease)
# to be considered "convergent" for biased sampling
MIN_CONVERGENT_SPECIMENS = 3

# Target number of convergent sequences to include per specimen
TARGET_CONVERGENT_SEQS = 200

# Random seed for reproducibility
RANDOM_SEED = 42

FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Select participants
# ---------------------------------------------------------------------------

def select_participants(
    metadata: pd.DataFrame,
    diseases: list,
    participants_per_fold: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select participants: N per fold per disease, ensuring multi-specimen
    participants are included when available.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata (already filtered to TCR).
    diseases : list
        Disease classes to include.
    participants_per_fold : int
        Number of participants per disease per fold.
    rng : np.random.Generator
        Random number generator for reproducible selection.

    Returns
    -------
    pd.DataFrame
        Subset of metadata rows for selected participants.
    """
    # Validate requested diseases exist
    available_diseases = set(metadata["disease"].unique())
    missing = set(diseases) - available_diseases
    if missing:
        raise ValueError(
            f"Diseases not found in metadata: {missing}. "
            f"Available: {sorted(available_diseases)}"
        )

    folds = sorted(metadata[FOLD_COL].unique())
    logger.info(f"Selecting participants: {len(diseases)} diseases × "
                f"{participants_per_fold}/fold × {len(folds)} folds")

    # Identify multi-specimen participants (for realistic testing)
    specimens_per_participant = metadata.groupby("participant_label")[
        "specimen_label"
    ].nunique()
    multi_specimen = set(
        specimens_per_participant[specimens_per_participant > 1].index
    )
    logger.info(f"  Multi-specimen participants available: {len(multi_specimen)}")

    selected_participants = []

    for disease in diseases:
        disease_meta = metadata[metadata["disease"] == disease]

        for fold_id in folds:
            fold_meta = disease_meta[disease_meta[FOLD_COL] == fold_id]
            available = fold_meta["participant_label"].unique()

            if len(available) < participants_per_fold:
                raise ValueError(
                    f"Not enough participants for {disease} fold {fold_id}: "
                    f"need {participants_per_fold}, have {len(available)}"
                )

            # Prioritize multi-specimen participants (include at most 1 per
            # disease×fold to spread them across the dataset)
            multi_in_fold = [p for p in available if p in multi_specimen]
            non_multi = [p for p in available if p not in multi_specimen]

            chosen = []
            if multi_in_fold:
                # Pick 1 multi-specimen participant
                chosen.append(rng.choice(multi_in_fold))
                remaining_needed = participants_per_fold - 1
            else:
                remaining_needed = participants_per_fold

            # Fill the rest randomly from non-multi (or all available if needed)
            pool = [p for p in non_multi if p not in chosen]
            if len(pool) < remaining_needed:
                # Fall back to including more multi-specimen if needed
                pool = [p for p in available if p not in chosen]
            chosen.extend(rng.choice(pool, size=remaining_needed, replace=False).tolist())

            selected_participants.extend(chosen)
            n_multi = sum(1 for p in chosen if p in multi_specimen)
            logger.info(
                f"  {disease} fold {fold_id}: {len(chosen)} participants "
                f"({n_multi} multi-specimen)"
            )

    # Deduplicate (shouldn't happen, but be safe)
    selected_participants = list(dict.fromkeys(selected_participants))

    # Filter metadata to selected participants
    selected_meta = metadata[
        metadata["participant_label"].isin(selected_participants)
    ].copy()

    n_participants = selected_meta["participant_label"].nunique()
    n_specimens = selected_meta["specimen_label"].nunique()
    n_multi = sum(
        1 for p in selected_participants if p in multi_specimen
    )
    logger.info(
        f"  Selected: {n_participants} participants, {n_specimens} specimens "
        f"({n_multi} multi-specimen participants)"
    )

    return selected_meta


# ---------------------------------------------------------------------------
# Step 2: Find convergent CDR3s
# ---------------------------------------------------------------------------

def find_convergent_cdr3s(
    selected_meta: pd.DataFrame,
    source_data_dir: Path,
    min_specimens: int = MIN_CONVERGENT_SPECIMENS,
) -> dict:
    """Scan full data for selected participants to find CDR3s that appear
    in multiple specimens of the same disease.

    Parameters
    ----------
    selected_meta : pd.DataFrame
        Metadata for selected participants.
    source_data_dir : Path
        Path to raw AIRR-format data directory.
    min_specimens : int
        Minimum number of specimens a CDR3 must appear in (within the same
        disease) to be considered convergent.

    Returns
    -------
    dict
        {disease: set of convergent CDR3 strings}
    """
    logger.info(f"Scanning for convergent CDR3s (min {min_specimens} specimens)...")

    # Build specimen → disease mapping
    specimen_disease = dict(
        zip(selected_meta["specimen_label"], selected_meta["disease"])
    )

    # Collect CDR3 → set of specimens per disease
    # {disease: {cdr3_aa: set of specimen_labels}}
    cdr3_specimens: dict = defaultdict(lambda: defaultdict(set))

    participants = selected_meta["participant_label"].unique()
    for i, participant in enumerate(participants, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"  Scanning participant {i}/{len(participants)}")

        filepath = source_data_dir / f"part_table_{participant}.tsv.gz"
        if not filepath.exists():
            logger.warning(f"  File not found: {filepath.name}, skipping")
            continue

        # Read only the columns we need for convergence detection
        df = pd.read_csv(
            filepath, sep="\t", usecols=["repertoire_id", "cdr3_aa", "productive"],
        )

        # Filter to productive sequences with valid CDR3
        # Note: raw data has lowercase "t"/"f"; the dataloader uppercases
        # them during preprocessing, but we're reading raw files here.
        df = df[
            df["productive"].str.upper().eq("T")
            & df["cdr3_aa"].notna()
            & (df["cdr3_aa"].str.len() > 0)
        ]

        # Map repertoire_id to disease (only keep specimens in our selection)
        for specimen_label, group in df.groupby("repertoire_id"):
            if specimen_label not in specimen_disease:
                continue
            disease = specimen_disease[specimen_label]
            for cdr3 in group["cdr3_aa"].unique():
                cdr3_specimens[disease][cdr3].add(specimen_label)

    # Filter to convergent CDR3s (appearing in >= min_specimens specimens)
    convergent: dict = {}
    for disease in sorted(cdr3_specimens.keys()):
        disease_convergent = {
            cdr3 for cdr3, specimens in cdr3_specimens[disease].items()
            if len(specimens) >= min_specimens
        }
        convergent[disease] = disease_convergent
        logger.info(
            f"  {disease}: {len(disease_convergent)} convergent CDR3s "
            f"(in {min_specimens}+ specimens)"
        )

    return convergent


# ---------------------------------------------------------------------------
# Step 3: Subsample specimens with biased sampling
# ---------------------------------------------------------------------------

def subsample_specimen(
    df: pd.DataFrame,
    specimen_label: str,
    disease: str,
    convergent_cdr3s: dict,
    seqs_per_specimen: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsample sequences from a specimen with biased sampling.

    Guarantees inclusion of sequences with convergent CDR3s (needed by
    Model 2 for cluster discovery), then fills the rest randomly.

    Parameters
    ----------
    df : pd.DataFrame
        All sequences for this specimen (full columns).
    specimen_label : str
        Specimen identifier.
    disease : str
        Disease class of this specimen.
    convergent_cdr3s : dict
        {disease: set of convergent CDR3 strings}.
    seqs_per_specimen : int
        Target number of sequences after subsampling.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Subsampled sequences (at most seqs_per_specimen rows).
    """
    if len(df) <= seqs_per_specimen:
        return df

    disease_convergent = convergent_cdr3s.get(disease, set())

    # Split into convergent and non-convergent sequences
    is_convergent = (
        df["cdr3_aa"].notna() & df["cdr3_aa"].isin(disease_convergent)
    )
    convergent_df = df[is_convergent]
    non_convergent_df = df[~is_convergent]

    # Include up to TARGET_CONVERGENT_SEQS convergent sequences
    n_convergent = min(len(convergent_df), TARGET_CONVERGENT_SEQS, seqs_per_specimen)
    if n_convergent > 0 and len(convergent_df) > n_convergent:
        convergent_idx = rng.choice(
            convergent_df.index, size=n_convergent, replace=False
        )
        convergent_sample = df.loc[convergent_idx]
    else:
        convergent_sample = convergent_df.head(n_convergent)

    # Fill the rest with random non-convergent sequences
    n_remaining = seqs_per_specimen - len(convergent_sample)
    if n_remaining > 0 and len(non_convergent_df) > 0:
        n_random = min(n_remaining, len(non_convergent_df))
        random_idx = rng.choice(
            non_convergent_df.index, size=n_random, replace=False
        )
        random_sample = df.loc[random_idx]
    else:
        random_sample = non_convergent_df.head(0)  # empty

    result = pd.concat([convergent_sample, random_sample], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Step 4: Process all participants and write output
# ---------------------------------------------------------------------------

def process_and_write(
    selected_meta: pd.DataFrame,
    source_data_dir: Path,
    output_dir: Path,
    convergent_cdr3s: dict,
    seqs_per_specimen: int,
    rng: np.random.Generator,
):
    """Read full data for selected participants, subsample, strip columns,
    and write to output directory.

    Parameters
    ----------
    selected_meta : pd.DataFrame
        Metadata for selected participants.
    source_data_dir : Path
        Path to raw AIRR-format data directory.
    output_dir : Path
        Output directory for test data.
    convergent_cdr3s : dict
        {disease: set of convergent CDR3 strings}.
    seqs_per_specimen : int
        Target sequences per specimen.
    rng : np.random.Generator
        Random number generator.
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build specimen → disease mapping
    specimen_disease = dict(
        zip(selected_meta["specimen_label"], selected_meta["disease"])
    )

    participants = selected_meta["participant_label"].unique()
    total_seqs = 0
    total_convergent = 0
    specimens_written = 0

    for i, participant in enumerate(participants, 1):
        logger.info(f"Processing participant {i}/{len(participants)}: {participant}")

        filepath = source_data_dir / f"part_table_{participant}.tsv.gz"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {filepath}. "
                f"Check SOURCE_DATA_DIR path."
            )

        # Read full file but only keep needed columns
        df = pd.read_csv(filepath, sep="\t")

        # Validate that all required columns exist
        missing_cols = set(SEQUENCE_COLS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {filepath.name}: {sorted(missing_cols)}. "
                f"Available: {sorted(df.columns)[:20]}..."
            )

        # Keep only needed columns
        df = df[SEQUENCE_COLS].copy()

        # Process each specimen in this participant's file
        specimen_dfs = []
        for specimen_label, specimen_df in df.groupby("repertoire_id"):
            if specimen_label not in specimen_disease:
                # Specimen not in our selection (shouldn't happen, but be safe)
                continue

            disease = specimen_disease[specimen_label]
            subsampled = subsample_specimen(
                specimen_df, specimen_label, disease,
                convergent_cdr3s, seqs_per_specimen, rng,
            )

            n_conv = subsampled["cdr3_aa"].isin(
                convergent_cdr3s.get(disease, set())
            ).sum()
            total_convergent += n_conv
            total_seqs += len(subsampled)
            specimens_written += 1

            logger.info(
                f"  {specimen_label}: {len(specimen_df):,} → "
                f"{len(subsampled):,} sequences "
                f"({n_conv} convergent)"
            )

            specimen_dfs.append(subsampled)

        if not specimen_dfs:
            logger.warning(f"  No specimens found for {participant}, skipping file")
            continue

        # Combine all specimens for this participant and write
        output_df = pd.concat(specimen_dfs, ignore_index=True)
        output_path = raw_dir / f"part_table_{participant}.tsv.gz"
        output_df.to_csv(
            output_path, sep="\t", index=False, compression="gzip",
        )

        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"  Wrote {output_path.name}: {file_size_kb:.0f} KB")

    logger.info(
        f"\nTotal: {specimens_written} specimens, {total_seqs:,} sequences "
        f"({total_convergent:,} convergent)"
    )


# ---------------------------------------------------------------------------
# Step 5: Write metadata and README
# ---------------------------------------------------------------------------

def write_metadata(selected_meta: pd.DataFrame, output_dir: Path):
    """Write subset metadata.tsv with only the needed columns."""
    meta_path = output_dir / "metadata.tsv"
    selected_meta[METADATA_COLS].to_csv(meta_path, sep="\t", index=False)
    logger.info(f"Wrote metadata: {meta_path} ({len(selected_meta)} rows)")


def write_readme(
    output_dir: Path,
    selected_meta: pd.DataFrame,
    seqs_per_specimen: int,
    diseases: list,
    convergent_cdr3s: dict,
):
    """Write a README documenting the test data."""
    n_participants = selected_meta["participant_label"].nunique()
    n_specimens = selected_meta["specimen_label"].nunique()
    multi_spec = (
        selected_meta.groupby("participant_label")["specimen_label"]
        .nunique()
        .pipe(lambda s: (s > 1).sum())
    )

    # Per-disease summary
    disease_summary = []
    for d in diseases:
        d_meta = selected_meta[selected_meta["disease"] == d]
        n_p = d_meta["participant_label"].nunique()
        n_s = d_meta["specimen_label"].nunique()
        n_conv = len(convergent_cdr3s.get(d, set()))
        disease_summary.append(f"  {d}: {n_p} participants, {n_s} specimens, "
                               f"{n_conv} convergent CDR3s")

    # Total file size
    raw_dir = output_dir / "raw"
    total_bytes = sum(f.stat().st_size for f in raw_dir.glob("*.tsv.gz"))
    total_mb = total_bytes / 1048576

    readme_text = f"""\
# Test Data

Small subsampled dataset for integration tests. Created from real Mal-ID data
by `scripts/data/create_test_data.py`.

## Contents

- `metadata.tsv` — specimen metadata ({n_specimens} specimens, {n_participants} participants)
- `raw/` — subsampled AIRR-format .tsv.gz files ({seqs_per_specimen} sequences/specimen, 19 columns)

## Summary

- Diseases: {', '.join(diseases)}
- Participants: {n_participants} ({multi_spec} with multiple specimens)
- Specimens: {n_specimens}
- Sequences per specimen: ~{seqs_per_specimen}
- Columns: 19 (stripped from 121 in full data)
- Total compressed size: {total_mb:.1f} MB

### Per-disease breakdown
{chr(10).join(disease_summary)}

## Usage

```python
from malid_lite.dataloader import MalIDPublishedDataLoader

loader = MalIDPublishedDataLoader(
    data_dir=test_data_dir / "raw",
    metadata_path=None,           # loads from cache_dir/metadata.tsv
    gene_reference_path=None,     # FR/CDR extraction skipped
    cache_dir=test_data_dir,      # fold cache built here on first run
    verbose=0,
)
```

## Generation

Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Seed: {RANDOM_SEED}
Script: scripts/data/create_test_data.py
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_text)
    logger.info(f"Wrote README: {readme_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create small test dataset from real Mal-ID data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seqs-per-specimen", type=int, default=DEFAULT_SEQS_PER_SPECIMEN,
        help=f"Sequences per specimen (default: {DEFAULT_SEQS_PER_SPECIMEN})",
    )
    parser.add_argument(
        "--diseases", nargs="+", default=DEFAULT_DISEASES,
        help=f"Disease classes to include (default: {DEFAULT_DISEASES})",
    )
    parser.add_argument(
        "--participants-per-fold", type=int,
        default=DEFAULT_PARTICIPANTS_PER_FOLD,
        help=f"Participants per disease per fold (default: {DEFAULT_PARTICIPANTS_PER_FOLD})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source-data-dir", type=Path, default=SOURCE_DATA_DIR,
        help=f"Source AIRR data directory (default: {SOURCE_DATA_DIR})",
    )
    parser.add_argument(
        "--source-metadata", type=Path, default=SOURCE_METADATA_PATH,
        help=f"Source metadata.tsv path (default: {SOURCE_METADATA_PATH})",
    )
    parser.add_argument(
        "--min-convergent-specimens", type=int,
        default=MIN_CONVERGENT_SPECIMENS,
        help=f"Min specimens for a CDR3 to count as convergent (default: {MIN_CONVERGENT_SPECIMENS})",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("=" * 60)
    logger.info("CREATE TEST DATA")
    logger.info("=" * 60)
    logger.info(f"Source data: {args.source_data_dir}")
    logger.info(f"Source metadata: {args.source_metadata}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Diseases: {args.diseases}")
    logger.info(f"Participants per fold: {args.participants_per_fold}")
    logger.info(f"Sequences per specimen: {args.seqs_per_specimen}")
    logger.info(f"Random seed: {args.seed}")

    rng = np.random.default_rng(args.seed)

    # --- Validate source paths ---
    if not args.source_data_dir.exists():
        logger.error(f"Source data directory not found: {args.source_data_dir}")
        sys.exit(1)
    if not args.source_metadata.exists():
        logger.error(f"Source metadata not found: {args.source_metadata}")
        sys.exit(1)

    # --- Load full metadata ---
    logger.info("\n--- Step 1: Load metadata and select participants ---")
    full_metadata = pd.read_csv(args.source_metadata, sep="\t")

    # Filter to TCR participants
    full_metadata = full_metadata[
        full_metadata["available_gene_loci"].str.contains("TCR", na=False)
    ].copy()
    logger.info(f"Full metadata: {full_metadata['participant_label'].nunique()} "
                f"TCR participants")

    # Validate required metadata columns
    missing_meta_cols = set(METADATA_COLS) - set(full_metadata.columns)
    if missing_meta_cols:
        logger.error(f"Missing metadata columns: {sorted(missing_meta_cols)}")
        sys.exit(1)

    # --- Select participants ---
    selected_meta = select_participants(
        full_metadata, args.diseases, args.participants_per_fold, rng,
    )

    # --- Find convergent CDR3s ---
    logger.info("\n--- Step 2: Find convergent CDR3s ---")
    convergent_cdr3s = find_convergent_cdr3s(
        selected_meta, args.source_data_dir, args.min_convergent_specimens,
    )

    # --- Prepare output directory ---
    logger.info("\n--- Step 3: Subsample and write data ---")
    if args.output_dir.exists():
        # Warn but don't delete — let the user decide
        logger.warning(
            f"Output directory already exists: {args.output_dir}. "
            f"Files will be overwritten."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Process and write ---
    process_and_write(
        selected_meta, args.source_data_dir, args.output_dir,
        convergent_cdr3s, args.seqs_per_specimen, rng,
    )

    # --- Write metadata and README ---
    logger.info("\n--- Step 4: Write metadata and README ---")
    write_metadata(selected_meta, args.output_dir)
    write_readme(
        args.output_dir, selected_meta, args.seqs_per_specimen,
        args.diseases, convergent_cdr3s,
    )

    # --- Summary ---
    raw_dir = args.output_dir / "raw"
    total_bytes = sum(f.stat().st_size for f in raw_dir.glob("*.tsv.gz"))
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Total compressed size: {total_bytes / 1048576:.1f} MB")
    logger.info(f"Files: {len(list(raw_dir.glob('*.tsv.gz')))} participant files")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
