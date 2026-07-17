"""Abstract base class for immune repertoire data loaders."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Iterator
from pathlib import Path
from enum import Enum
from datetime import datetime
import filecmp
import os
import tempfile
import pandas as pd
import numpy as np
import logging
import json
import shutil

import sklearn
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Canonical fold column name (used in all new metadata files)
FOLD_COL = "CV_fold"
# Legacy name found in existing caches and metadata files
_LEGACY_FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"

# ---------------------------------------------------------------------------
# Training contexts
# ---------------------------------------------------------------------------
# A "training context" controls (a) which participant split roles are produced
# and (b) the output directory layout.
#
#   CV contexts (cross-validation on a single dataset, driven by CV_fold):
#     cv_single_model : roles = test, train_smaller1, train_smaller2
#     cv_ensemble     : roles = test, validation, train_smaller1, train_smaller2
#
#   Train-all contexts (train on the whole dataset, no held-out test fold; used
#   for training a model that will be evaluated on a separate dataset). These do
#   NOT require a CV_fold column and are not keyed by a fold id:
#     train_all          : roles = train_smaller1, train_smaller2 (= ALL participants)
#     train_all_ensemble : roles = validation, train_smaller1, train_smaller2
#                          (validation = 1/3 held out for the metamodel; ts1+ts2 = 2/3)
CV_TRAINING_CONTEXTS = ("cv_single_model", "cv_ensemble")
TRAIN_ALL_TRAINING_CONTEXTS = ("train_all", "train_all_ensemble")
VALID_TRAINING_CONTEXTS = CV_TRAINING_CONTEXTS + TRAIN_ALL_TRAINING_CONTEXTS


def normalize_fold_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy fold column to the canonical 'CV_fold' if present.

    Handles backward compatibility with metadata/cache files that use the old
    column name 'malid_cross_validation_fold_id_when_in_test_set'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain the legacy fold column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the fold column renamed to 'CV_fold' (if the legacy
        name was present), or unchanged (if 'CV_fold' already exists or
        neither column is present).

    Raises
    ------
    ValueError
        If both the legacy and canonical column names are present.
    """
    if _LEGACY_FOLD_COL in df.columns and FOLD_COL in df.columns:
        raise ValueError(
            f"DataFrame has both '{_LEGACY_FOLD_COL}' and '{FOLD_COL}' columns. "
            f"Only one fold ID column should be present."
        )
    if _LEGACY_FOLD_COL in df.columns:
        df = df.rename(columns={_LEGACY_FOLD_COL: FOLD_COL})
    return df


# Columns that must always be string type to ensure consistent matching
# between metadata (always string) and sequence data (may be int for numeric labels)
_IDENTIFIER_COLS = ("participant_label", "specimen_label", "repertoire_id")


def normalize_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce identifier columns to string type.

    Parquet and CSV readers infer numeric-looking labels (e.g., "310101")
    as int64.  Metadata always stores them as strings. This mismatch
    causes silent join/filter failures. Normalizing to string after every
    load prevents that.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose identifier columns should be coerced.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with identifier columns cast to ``str``.
    """
    for col in _IDENTIFIER_COLS:
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df


class PreprocessingStage(Enum):
    """Preprocessing stages for data loading."""

    RAW = "raw"  # No preprocessing (as stored on disk)
    CLEAN = "clean"  # After cleaning/validation (stage 1)
    DOWNSAMPLED = "downsampled"  # After downsampling (stage 2)
    # Note: V-gene filtering and featurization (stage 3) is handled by Model 1


class BaseDataLoader(ABC):
    """
    Abstract base class for repertoire data loaders.

    Provides a unified interface for loading and preprocessing immune repertoire data.
    Implementations should handle specific file formats (e.g., internal format, AIRR format).

    Design philosophy:
    - Hybrid memory strategy: Metadata in memory (lightweight), sequences on-demand
    - Flexible iteration: Support both iterator and bulk loading
    - Automatic statistics: Accumulate preprocessing stats during loading
    - Caching support: Save/load preprocessed data to avoid re-running expensive steps
    - Verbose logging: Track samples found, dropped, and processing progress
    """

    def __init__(
        self,
        data_dir: Optional[Path],
        metadata_path: Optional[Path] = None,
        gene_locus: str = "TCR",
        verbose: int = 1,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Path to directory containing repertoire files. Can be None
                for metadata-only use (e.g. loading pre-computed feature matrices).
                When None, load_metadata() skips the raw-data file scan — all
                participants in the metadata are retained.
            metadata_path: Path to metadata TSV file. Optional if the cache
                already contains a processed copy (cache_dir/metadata_processed.tsv
                or cache_dir/metadata.tsv).
            gene_locus: Gene locus to load ("TCR" or "BCR")
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
            cache_dir: Optional directory for caching preprocessed data
        """
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.gene_locus = gene_locus
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Resolve metadata_path: prefer user-supplied, fall back to cached copies.
        # metadata_processed.tsv = filtered to participants with raw data files.
        # metadata.tsv = original unfiltered copy (for reference/debugging).
        cached_metadata_processed = (
            self.cache_dir / "metadata_processed.tsv" if self.cache_dir else None
        )
        cached_metadata_raw = self.cache_dir / "metadata.tsv" if self.cache_dir else None

        if metadata_path is not None:
            self.metadata_path = Path(metadata_path)
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"metadata_path does not exist: {self.metadata_path}"
                )
            # Check if the supplied path IS the cached processed copy itself
            # (e.g., ensemble passes loader.metadata_path to base model trainers).
            # In that case, treat it the same as auto-discovery: already filtered.
            if (
                cached_metadata_processed is not None
                and cached_metadata_processed.exists()
                and self.metadata_path.resolve() == cached_metadata_processed.resolve()
            ):
                self._metadata_needs_filtering = False
            else:
                self._metadata_needs_filtering = True
                # If a cached raw copy also exists, verify they match
                if cached_metadata_raw is not None and cached_metadata_raw.exists():
                    if self.metadata_path.resolve() != cached_metadata_raw.resolve():
                        if not filecmp.cmp(
                            self.metadata_path, cached_metadata_raw, shallow=False
                        ):
                            raise ValueError(
                                f"Supplied metadata_path ({self.metadata_path}) differs "
                                f"from cached copy ({cached_metadata_raw}). The cache may "
                                f"be stale. Clear caches with: "
                                f"python scripts/data/manage_cache.py clear-all"
                            )
        elif cached_metadata_processed is not None and cached_metadata_processed.exists():
            # Preferred: already filtered to participants with raw data
            self.metadata_path = cached_metadata_processed
            self._metadata_needs_filtering = False
        elif cached_metadata_raw is not None and cached_metadata_raw.exists():
            # Backward compat: old cache without processed metadata — needs filtering
            self.metadata_path = cached_metadata_raw
            self._metadata_needs_filtering = True
        else:
            raise ValueError(
                "metadata_path is required when no cached metadata exists. "
                "Either provide metadata_path or ensure cache_dir contains metadata.tsv."
            )

        # Validate gene locus
        if gene_locus not in ["TCR", "BCR"]:
            raise ValueError(f"gene_locus must be 'TCR' or 'BCR', got: {gene_locus}")

        # Accumulate preprocessing statistics
        self._preprocessing_stats: List[Dict] = []

        # Metadata in memory (lightweight, lazy loaded)
        self._metadata: Optional[pd.DataFrame] = None

        # Set by load_metadata(): filtering stats (n_original, n_filtered_out,
        # n_retained) or None if loaded from pre-processed cache.
        self.metadata_filter_info: Optional[Dict] = None

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Get metadata (lazy load and cache in memory).

        Returns:
            DataFrame with metadata for all samples
        """
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate metadata file (lightweight, kept in memory).

        Should log:
            - Total samples found
            - Samples with data available
            - Missing data files
            - Fold distribution

        Returns:
            DataFrame with metadata columns including:
                - participant_label
                - specimen_label
                - disease
                - CV_fold
                - available_gene_loci
                - (and other study-specific columns)
        """
        pass

    # ========== Memory-Efficient Iterator Methods ==========

    @abstractmethod
    def iter_fold_specimens(
        self,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        """
        Iterate over specimens in a fold one at a time (RECOMMENDED for memory efficiency).

        Hybrid approach: Metadata in memory, sequences loaded on-demand.

        Args:
            fold_id: Cross-validation fold ID (typically 0-4), or None when
                fold_label == "all".
            fold_label: "train" (all specimens except fold_id), "test" (only
                fold_id), or "all" (every specimen, no CV_fold filtering; used
                for train-all).
            preprocessing_stage: Level of preprocessing to apply

        Yields:
            Tuple of (specimen_label, specimen_sequences, specimen_metadata)
            - specimen_label: str - identifier for this specimen
            - specimen_sequences: DataFrame - sequence-level data (long format, one row per sequence)
            - specimen_metadata: Series - specimen-level info from metadata

        Example:
            >>> for spec_label, sequences, meta in loader.iter_fold_specimens(0, "train"):
            ...     print(f"{spec_label}: {len(sequences)} sequences")
        """
        pass

    # ========== Convenience Method ==========

    def get_fold_data(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all data for a CV fold into memory.

        Tries fold cache first. On cache miss, builds the fold from
        specimen-level data and automatically saves the result to the fold
        cache (when cache_dir is set) so that subsequent calls are fast.

        ⚠️  Warning: May use large amounts of memory for big datasets.
        Prefer iter_fold_specimens() for memory-efficient loading without
        caching.

        Args:
            fold_id: Cross-validation fold ID
            fold_label: "train" or "test"
            preprocessing_stage: Level of preprocessing to apply

        Returns:
            Tuple of (sequences_df, metadata_df)
            - sequences_df: All sequences for the fold (long format)
            - metadata_df: Metadata for all specimens in the fold

        Example:
            >>> sequences, metadata = loader.get_fold_data(0, "train")
            >>> print(f"Loaded {len(sequences)} sequences from {len(metadata)} specimens")
        """
        if fold_label not in ("train", "test"):
            raise ValueError(
                f"get_fold_data fold_label must be 'train' or 'test', got "
                f"{fold_label!r}. To load the entire dataset (no CV fold), use "
                f"get_all_data()."
            )
        return self._load_fold_or_all(fold_id, fold_label, preprocessing_stage)

    def get_all_data(
        self,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the ENTIRE dataset into memory (all participants, no CV fold).

        This is the train-all counterpart of get_fold_data(): it ignores the
        CV_fold column entirely (the column need not even exist) and returns
        every specimen. Results are cached under ``data_folds/all_<stage>_*``.

        Unlike the fold loaders, this path enforces a completeness check: if any
        metadata participant fails to load due to an error (unreadable cache,
        etc.), it raises rather than silently training on a shrunken dataset.
        Participants that legitimately have no data after QC (all specimens
        dropped by downsampling thresholds) are reported and skipped.

        The completeness check runs at BUILD time (first call, cache miss). On a
        subsequent cache HIT, staleness is verified against a build-time manifest
        (``all_<stage>_cache_manifest.json``) that records the metadata
        (specimen, participant) pairs the cache was built from: if the metadata has
        since changed (participants added OR removed), this raises a clear error
        telling you to clear the fold cache and rebuild
        (``manage_cache.py clear-folds``). QC-dropped participants never trigger a
        false alarm (the manifest compares metadata-then vs metadata-now, not cache
        contents). Caches built before manifests existed fall back to the
        removals-only parquet check for backward compatibility.

        Returns:
            Tuple of (sequences_df, metadata_df) for the whole dataset.
        """
        return self._load_fold_or_all(None, "all", preprocessing_stage)

    def _load_fold_or_all(
        self,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Shared implementation for get_fold_data() and get_all_data().

        fold_label is "train"/"test" (CV, with an int fold_id) or "all"
        (train-all, with fold_id=None). Tries the fold cache first, then falls
        back to specimen iteration and auto-caches the result.
        """
        # Try fold cache first — single parquet read, much faster than iterating.
        if self.cache_dir is not None:
            cached = self.load_cached_fold(fold_id, fold_label, preprocessing_stage)
            if cached is not None:
                return cached

        # Fall back to specimen-by-specimen loading (used when fold cache is absent)
        all_sequences = []
        all_metadata = []

        for spec_label, spec_seqs, spec_meta in self.iter_fold_specimens(
            fold_id, fold_label, preprocessing_stage
        ):
            all_sequences.append(spec_seqs)
            all_metadata.append(spec_meta)

        if not all_sequences:
            return pd.DataFrame(), pd.DataFrame()

        sequences_df = pd.concat(all_sequences, ignore_index=True)
        metadata_df = pd.DataFrame(all_metadata)

        # Auto-cache the freshly built fold for next time.
        if self.cache_dir is not None and len(sequences_df) > 0:
            try:
                # For the whole-dataset ("all") cache, write the manifest of
                # build-time metadata (specimen, participant) pairs BEFORE the
                # parquet. Ordering matters for partial-failure safety: a cache
                # is only ever READ when its parquet exists, and load skips the
                # manifest when the parquet is absent — so if the parquet write
                # fails, the orphan manifest is never consumed and gets rebuilt.
                # The reverse order could leave a parquet WITHOUT a manifest,
                # silently downgrading the staleness check. See
                # _write_all_cache_manifest / _validate_all_cache_manifest.
                if fold_label == "all":
                    self._write_all_cache_manifest(
                        cached_metadata_df=metadata_df,
                        preprocessing_stage=preprocessing_stage,
                    )
                self._save_fold_cache(
                    sequences_df, metadata_df, fold_id, fold_label, preprocessing_stage
                )
            except Exception as e:
                logger.warning(
                    f"Failed to auto-cache {self._fold_label_desc(fold_id, fold_label)}: "
                    f"{e}. Continuing without caching."
                )

        return sequences_df, metadata_df

    @staticmethod
    def _fold_label_desc(fold_id: Optional[int], fold_label: str) -> str:
        """Human-readable label for fold-data logs, e.g. 'fold 1/train' or 'all'."""
        if fold_label == "all":
            return "all"
        return f"fold {fold_id}/{fold_label}"

    def iter_all_specimens(
        self,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        """Memory-efficient iterator over ALL specimens (train-all counterpart
        of iter_fold_specimens). Thin wrapper: iterates every specimen with no
        CV_fold filtering."""
        yield from self.iter_fold_specimens(None, "all", preprocessing_stage)

    @abstractmethod
    def load_participant_data(
        self,
        participant_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> pd.DataFrame:
        """
        Load data for a single participant (on-demand).

        Note: One participant file may contain multiple specimens.

        Args:
            participant_label: Participant identifier
            preprocessing_stage: Level of preprocessing to apply

        Returns:
            DataFrame with sequence-level data (may contain multiple specimens)
        """
        pass

    def load_specimen_data(
        self,
        specimen_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> pd.DataFrame:
        """
        Load data for a single specimen (on-demand).

        Convenience wrapper around load_participant_data.

        Args:
            specimen_label: Specimen identifier
            preprocessing_stage: Level of preprocessing to apply

        Returns:
            DataFrame with sequence-level data for single specimen
        """
        # Get participant_label from metadata
        specimen_meta = self.metadata[
            self.metadata["specimen_label"] == specimen_label
        ]
        if specimen_meta.empty:
            raise ValueError(f"Specimen not found in metadata: {specimen_label}")

        participant_label = specimen_meta.iloc[0]["participant_label"]

        # Load participant data and filter to specimen
        # Raw/clean data has repertoire_id (AIRR column); downsampled has specimen_label
        participant_df = self.load_participant_data(
            participant_label, preprocessing_stage
        )
        if "specimen_label" in participant_df.columns:
            specimen_df = participant_df[participant_df["specimen_label"] == specimen_label]
        else:
            specimen_df = participant_df[participant_df["repertoire_id"] == specimen_label]
        return specimen_df

    # ========== Preprocessing Methods ==========

    @abstractmethod
    def preprocess_clean(
        self,
        df: pd.DataFrame,
        participant_label: str,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Stage 1: Cleaning and validation (per participant).

        Steps (TCR-specific, BCR may differ):
            1. Filter productive sequences (productive == True)
            2. Filter v_score > 80 (TCR) or > 200 (BCR)
            3. Clean CDR/FR sequences (remove ".", "-", " ", "*"; uppercase; empty→NaN)
            4. Deduplicate identical sequences, sum num_reads
            5. Fix gene names (e.g., TRBV12-4→TRBV12-3, TRBV6-3→TRBV6-2)
            6. Extract FR/CDR regions from reference table (if available).
               Always overwrites raw IgBLAST FR/CDR values with reference values.
               Rows whose V gene is absent from the reference get NaN FR/CDR columns.
            7. Create gene columns:
               - v_gene, j_gene (without alleles, e.g., "TRBV7-8")
               - v_gene_w_allele, j_gene_w_allele (with alleles, e.g., "TRBV7-8*01")
            8. Drop sequences with missing V/J/CDR
            9. Add isotype_supergroup column ("TCRB" for TCR)

        Args:
            df: Raw participant data
            participant_label: Participant identifier for logging

        Returns:
            Tuple of (cleaned_df, drop_stats)
            - cleaned_df: Cleaned sequences
            - drop_stats: Dict with counts for each filter step:
                * productive_filter: sequences filtered (not productive)
                * v_score_filter: sequences filtered (low v_score)
                * sequences_before_dedup: count before deduplication
                * sequences_after_dedup: count after deduplication
                * gene_name_fixes: count of gene names corrected
                * missing_fields: sequences dropped (empty CDR/V/J)
                * total_dropped: total sequences removed
                * genes_missing_from_reference: dict {gene_name: row_count} for V genes
                  absent from the reference table (FR/CDR will be NaN for those rows)
                * n_genes_missing_from_reference: count of such genes
        """
        pass

    @abstractmethod
    def preprocess_downsample(
        self,
        df: pd.DataFrame,
        specimen_label: str,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Stage 2: Downsampling and filtering (per specimen).

        Steps (TCR-specific, BCR may differ):
            1. Filter CDR3 AA length >= 8
            2. Filter to isotype_supergroup == "TCRB" (or relevant BCR isotypes)
            3. Check >= 500 clones for TCRB (drop specimen if fails)
            4. Check >= 1000 sequences after filters (drop specimen if fails)
            5. Downsample: 1 sequence per (specimen, amplification, clone, isotype)
               - Group by: [repertoire_id, amplification_label, igh_or_tcrb_clone_id, isotype_supergroup]
               (uses AIRR repertoire_id internally; renamed to specimen_label after return)
               - Choose sequence with max num_reads per group
               - Add columns: num_clone_members, total_clone_num_reads

        Args:
            df: Cleaned data for one specimen
            specimen_label: Specimen identifier for logging

        Returns:
            Tuple of (downsampled_df, stats)
            - downsampled_df: Downsampled sequences (empty if specimen dropped)
            - stats: Dict with summary statistics:
                * sequences_before_downsample: sequences after cleaning (= input to this stage)
                * after_cdr3_filter: after CDR3 length filter
                * after_isotype_filter: after isotype filter
                * n_clones: unique clone count
                * meets_clone_threshold: bool (>=500 for TCR)
                * meets_sequence_threshold: bool (>=1000)
                * after_downsample: final sequence count
                * kept: bool (specimen kept or dropped)
                * drop_reason: str if dropped, else None
        """
        pass

    # ========== Reporting Methods ==========

    def get_preprocessing_report(self) -> pd.DataFrame:
        """
        Get preprocessing report (accumulated during loading).

        Returns:
            DataFrame with per-specimen statistics:
                - specimen_label
                - participant_label
                - fold_id
                - original_count (raw sequences)
                - after_clean (after stage 1)
                - n_clones (unique clone count)
                - after_downsample (final count)
                - kept (bool - whether specimen was kept)
                - drop_reason (str - why dropped, if applicable)
        """
        if not self._preprocessing_stats:
            logger.warning("No preprocessing statistics accumulated yet")
            return pd.DataFrame()

        return pd.DataFrame(self._preprocessing_stats)

    def save_preprocessing_report(self, output_path: Path):
        """
        Save preprocessing report to CSV.

        Args:
            output_path: Path to save CSV file
        """
        report = self.get_preprocessing_report()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

        if self.verbose >= 1:
            n_kept = report["kept"].sum() if "kept" in report.columns else 0
            n_dropped = len(report) - n_kept
            logger.info(
                f"Saved preprocessing report to {output_path} "
                f"({n_kept} kept, {n_dropped} dropped)"
            )

    # ========== Split Persistence ==========

    # Valid training contexts and their split roles (module-level constants)
    VALID_TRAINING_CONTEXTS = VALID_TRAINING_CONTEXTS
    CV_TRAINING_CONTEXTS = CV_TRAINING_CONTEXTS
    TRAIN_ALL_TRAINING_CONTEXTS = TRAIN_ALL_TRAINING_CONTEXTS
    FOLD_COL = FOLD_COL  # "CV_fold" — module-level constant
    PARTICIPANT_COL = "participant_label"
    DISEASE_COL = "disease"

    def _get_splits_dir(self) -> Path:
        """Get the directory for split CSV files."""
        if self.cache_dir is None:
            raise ValueError(
                "cache_dir not set — required for split persistence. "
                "Pass cache_dir= when constructing the data loader."
            )
        return self.cache_dir / "splits"

    def _get_split_path(
        self, fold_id: Optional[int], training_context: str
    ) -> Path:
        """Get the path to a specific split CSV file.

        CV contexts are keyed by fold id: ``fold_<id>_<context>.csv``.
        Train-all contexts have no fold and are keyed by context alone:
        ``<context>.csv`` (fold_id is ignored, expected to be None).
        """
        if training_context in self.TRAIN_ALL_TRAINING_CONTEXTS:
            return self._get_splits_dir() / f"{training_context}.csv"
        return self._get_splits_dir() / f"fold_{fold_id}_{training_context}.csv"

    def _get_split_metadata_path(self) -> Path:
        """Get the path to the split metadata JSON file."""
        return self._get_splits_dir() / "split_metadata.json"

    def load_splits(
        self,
        fold_id: Optional[int],
        training_context: str,
    ) -> pd.DataFrame:
        """Load participant split assignments, generating if needed.

        If the split CSV already exists, loads and returns it.
        If it does not exist, generates the splits deterministically, saves
        to disk (plus a human-readable summary), and returns the result.

        Parameters
        ----------
        fold_id : int or None
            Cross-validation fold ID (the fold used as test set) for CV
            contexts. Must be None for train-all contexts (which have no
            fold concept).
        training_context : str
            One of the values in ``VALID_TRAINING_CONTEXTS``:
              cv_single_model, cv_ensemble (require a fold_id), or
              train_all, train_all_ensemble (require fold_id=None).

        Returns
        -------
        pd.DataFrame
            Columns: participant_label, disease, split_role.
            split_role values depend on training_context:
              cv_single_model:    "test", "train_smaller1", "train_smaller2"
              cv_ensemble:        "test", "validation", "train_smaller1", "train_smaller2"
              train_all:          "train_smaller1", "train_smaller2"
              train_all_ensemble: "validation", "train_smaller1", "train_smaller2"
        """
        self._validate_context_fold_id(fold_id, training_context)

        split_path = self._get_split_path(fold_id, training_context)

        if split_path.exists():
            try:
                splits_df = pd.read_csv(split_path)
            except Exception as e:
                logger.warning(
                    f"Corrupt split file {split_path.name}: {e}. "
                    f"Deleting and regenerating."
                )
                split_path.unlink(missing_ok=True)
                # Fall through to generation below
            else:
                # Validate expected columns are present
                required_cols = {"participant_label", "disease", "split_role"}
                if not required_cols.issubset(splits_df.columns):
                    logger.warning(
                        f"Split file {split_path.name} missing columns "
                        f"{required_cols - set(splits_df.columns)}. "
                        f"Deleting and regenerating."
                    )
                    split_path.unlink(missing_ok=True)
                else:
                    # Normalize identifiers (int64 → str) for consistency
                    # with metadata and sequence data
                    splits_df = normalize_identifier_columns(splits_df)
                    # Staleness guard: splits are generated from self.metadata, so the
                    # split's participant set must equal the current metadata's. If they
                    # differ, the metadata changed since the split was written — reachable
                    # when metadata_processed.tsv is passed directly or edited in place,
                    # which bypass the __init__ filecmp guard that otherwise clears splits.
                    # Reusing a stale split would silently EXCLUDE added participants (or
                    # list removed ones), shrinking the training set. Regenerate from the
                    # current metadata (symmetric with the all_* cache manifest, which
                    # catches the same additions the parquet-only check misses).
                    split_participants = set(splits_df["participant_label"])
                    meta_participants = set(self.metadata["participant_label"])
                    if split_participants != meta_participants:
                        n_added = len(meta_participants - split_participants)
                        n_removed = len(split_participants - meta_participants)
                        logger.warning(
                            f"Split file {split_path.name} is stale: its participant set "
                            f"differs from the current metadata ({n_added} added, "
                            f"{n_removed} removed since it was written). Regenerating from "
                            f"the current metadata."
                        )
                        split_path.unlink(missing_ok=True)
                        # Fall through to generation below.
                    else:
                        if self.verbose >= 1:
                            n_per_role = splits_df["split_role"].value_counts().to_dict()
                            logger.info(
                                f"Loaded splits for {self._context_label(fold_id, training_context)} "
                                f"from {split_path.name}: {n_per_role}"
                            )
                        return splits_df

        # --- Generate splits ---
        if self.verbose >= 1:
            logger.info(
                f"Split file not found for {self._context_label(fold_id, training_context)}. "
                f"Generating..."
            )
        splits_df = self._generate_splits(fold_id, training_context)

        # Atomic write
        splits_dir = self._get_splits_dir()
        splits_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=splits_dir, suffix=".csv")
        os.close(tmp_fd)
        try:
            splits_df.to_csv(tmp_path, index=False)
            os.rename(tmp_path, split_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Write/update global provenance metadata on first write
        self._write_split_metadata()
        # Write a human-readable per-context summary alongside the CSV
        self._write_split_summary(splits_df, fold_id, training_context)

        if self.verbose >= 1:
            n_per_role = splits_df["split_role"].value_counts().to_dict()
            logger.info(
                f"Saved splits for {self._context_label(fold_id, training_context)} "
                f"to {split_path.name}: {n_per_role}"
            )

        return splits_df

    def _validate_context_fold_id(
        self, fold_id: Optional[int], training_context: str
    ) -> None:
        """Validate the (fold_id, training_context) combination.

        CV contexts require an integer fold_id; train-all contexts require
        fold_id=None (they have no fold concept). Fails fast with a clear
        message on any mismatch.
        """
        if training_context not in self.VALID_TRAINING_CONTEXTS:
            raise ValueError(
                f"training_context must be one of {self.VALID_TRAINING_CONTEXTS}, "
                f"got: {training_context!r}"
            )
        if training_context in self.TRAIN_ALL_TRAINING_CONTEXTS:
            if fold_id is not None:
                raise ValueError(
                    f"Train-all context {training_context!r} has no fold concept — "
                    f"fold_id must be None, got {fold_id!r}."
                )
        else:  # CV context
            if fold_id is None:
                raise ValueError(
                    f"CV context {training_context!r} requires a fold_id, got None."
                )

    @staticmethod
    def _context_label(fold_id: Optional[int], training_context: str) -> str:
        """Human-readable label for logs, e.g. 'fold 1 (cv_ensemble)' or 'train_all'."""
        if fold_id is None:
            return training_context
        return f"fold {fold_id} ({training_context})"

    def _generate_splits(
        self,
        fold_id: Optional[int],
        training_context: str,
    ) -> pd.DataFrame:
        """Generate participant split assignments for one context.

        Split logic matches the original Mal-ID exactly
        (notebooks_src/make_cv_folds.py:322-345):
        - All train_test_split calls use test_size=1/3, random_state=0,
          shuffle=True, stratify=disease
        - Splits are at the participant level
        - Ensemble contexts: train pool -> validation + train_smaller, then
          train_smaller -> train_smaller1 + train_smaller2

        CV contexts (cv_single_model / cv_ensemble) first hold out the test
        fold (participants with CV_fold == fold_id); the remaining participants
        form the train pool. Train-all contexts (train_all / train_all_ensemble)
        have no test fold — ALL participants form the train pool.

        Parameters
        ----------
        fold_id : int or None
            The fold used as the test set (CV contexts). None for train-all.
        training_context : str
            One of VALID_TRAINING_CONTEXTS.

        Returns
        -------
        pd.DataFrame
            Columns: participant_label, disease, split_role
        """
        self._validate_context_fold_id(fold_id, training_context)
        meta = self.metadata

        has_test = training_context in self.CV_TRAINING_CONTEXTS
        is_ensemble = training_context in ("cv_ensemble", "train_all_ensemble")

        # --- Get unique participants with their disease ---
        # Sort by participant_label for deterministic ordering: ensures
        # train_test_split produces the same result regardless of how
        # metadata was loaded or what order rows appear in.
        # The CV_fold column is only needed (and only read) for CV contexts.
        cols = [self.PARTICIPANT_COL, self.DISEASE_COL]
        if has_test:
            if self.FOLD_COL not in meta.columns:
                raise ValueError(
                    f"Cannot generate splits for CV context {training_context!r}: "
                    f"metadata has no '{self.FOLD_COL}' column. Provide metadata with "
                    f"fold assignments, or use a train-all context "
                    f"({', '.join(self.TRAIN_ALL_TRAINING_CONTEXTS)})."
                )
            cols = cols + [self.FOLD_COL]
        participant_disease = (
            meta
            .drop_duplicates(subset=[self.PARTICIPANT_COL])
            [cols]
            .sort_values(self.PARTICIPANT_COL)
            .reset_index(drop=True)
        )

        # --- Determine test holdout (CV only) vs the train pool to sub-split ---
        result_cols = [self.PARTICIPANT_COL, self.DISEASE_COL, "split_role"]
        if has_test:
            test_mask = participant_disease[self.FOLD_COL] == fold_id
            test_participants = participant_disease.loc[
                test_mask, [self.PARTICIPANT_COL, self.DISEASE_COL]
            ].copy()
            train_participants = participant_disease.loc[
                ~test_mask, [self.PARTICIPANT_COL, self.DISEASE_COL]
            ].copy()
            test_participants["split_role"] = "test"
        else:
            # Train-all: no test holdout — every participant is in the train pool.
            test_participants = None
            train_participants = participant_disease[
                [self.PARTICIPANT_COL, self.DISEASE_COL]
            ].copy()

        # --- Fail fast if any disease is too small to stratify-split ---
        self._validate_stratification_counts(
            train_participants, is_ensemble,
            self._context_label(fold_id, training_context),
        )

        # --- Sub-split the train pool into ts1/ts2 (+ validation for ensemble) ---
        train_participants = self._subsplit_train_pool(train_participants, is_ensemble)

        # Combine and return (sorted by participant for readability).
        # For train-all there is no test frame — use the train pool directly
        # (concatenating an empty frame would corrupt column dtypes).
        if test_participants is not None:
            result = pd.concat(
                [test_participants, train_participants],
                ignore_index=True,
            )[result_cols]
        else:
            result = train_participants[result_cols].copy()
        result = result.sort_values(self.PARTICIPANT_COL).reset_index(drop=True)

        # Sanity checks — raise (not assert) so these split-integrity guards are NOT
        # stripped under `python -O`. They catch a split-generation regression that
        # duplicated a participant or left one without a role, which would corrupt
        # training silently.
        n_total = len(result)
        n_unique = result[self.PARTICIPANT_COL].nunique()
        if n_total != n_unique:
            raise RuntimeError(
                f"Duplicate participants in splits: {n_total} rows but {n_unique} "
                f"unique participants (split-generation bug)."
            )
        if result["split_role"].isna().any():
            raise RuntimeError(
                "Some participants have no split_role assigned (split-generation bug)."
            )

        return result

    def _validate_stratification_counts(
        self,
        train_participants: pd.DataFrame,
        is_ensemble: bool,
        context_label: str,
    ) -> None:
        """Fail fast if any disease has too few participants to stratify-split.

        Non-ensemble contexts perform ONE stratified split (train_smaller1 vs
        train_smaller2) → need >= 2 participants per disease. Ensemble contexts
        perform TWO nested stratified splits (hold out validation, then split
        the remainder) → need >= 3 participants per disease.

        These thresholds match sklearn's implicit stratification requirement,
        so this only converts a cryptic sklearn error into an actionable one —
        it never rejects a split that would otherwise have succeeded.
        """
        min_required = 3 if is_ensemble else 2
        counts = train_participants[self.DISEASE_COL].value_counts()
        too_few = counts[counts < min_required]
        if len(too_few) > 0:
            detail = ", ".join(f"{d}={int(n)}" for d, n in too_few.items())
            role_desc = (
                "validation + train_smaller1 + train_smaller2" if is_ensemble
                else "train_smaller1 + train_smaller2"
            )
            raise ValueError(
                f"Cannot generate splits for {context_label}: stratified splitting "
                f"into {role_desc} requires at least {min_required} participant(s) "
                f"per disease, but these are below that: {detail}. "
                f"Add more participants for these diseases, or remove them from the "
                f"training metadata."
            )

    def _subsplit_train_pool(
        self,
        train_participants: pd.DataFrame,
        is_ensemble: bool,
    ) -> pd.DataFrame:
        """Assign ts1/ts2 (+ validation for ensemble) roles to the train pool.

        Shared by CV and train-all contexts. The split sequence is identical to
        the original Mal-ID design (test_size=1/3, random_state=0, shuffle=True,
        stratify=disease, participant level), so CV splits are unchanged.

        Parameters
        ----------
        train_participants : DataFrame with participant_label + disease, already
            sorted by participant_label for determinism.
        is_ensemble : If True, hold out a validation third first, then split the
            remaining two-thirds into ts1/ts2.

        Returns
        -------
        The same DataFrame with a "split_role" column added.
        """
        train_participants = train_participants.copy()

        # Helper: participant/disease lists for train_test_split. .tolist()
        # avoids arrow-backed array issues with sklearn.
        def _split(df):
            return (df[self.PARTICIPANT_COL].tolist(),
                    df[self.DISEASE_COL].tolist())

        if not is_ensemble:
            # Single split: train pool -> train_smaller1 (2/3) + train_smaller2 (1/3)
            parts, diseases = _split(train_participants)
            ts1_labels, ts2_labels = train_test_split(
                parts,
                test_size=1 / 3,
                stratify=diseases,
                random_state=0,
                shuffle=True,
            )
            ts1_set = set(ts1_labels)
            train_participants["split_role"] = train_participants[self.PARTICIPANT_COL].apply(
                lambda p: "train_smaller1" if p in ts1_set else "train_smaller2"
            )
        else:
            # First split: train pool -> validation (1/3) + train_smaller (2/3)
            parts, diseases = _split(train_participants)
            train_smaller_labels, validation_labels = train_test_split(
                parts,
                test_size=1 / 3,
                stratify=diseases,
                random_state=0,
                shuffle=True,
            )

            validation_set = set(validation_labels)
            train_smaller_df = (
                train_participants[
                    ~train_participants[self.PARTICIPANT_COL].isin(validation_set)
                ]
                .sort_values(self.PARTICIPANT_COL)
                .reset_index(drop=True)
            )

            # Second split: train_smaller -> train_smaller1 (2/3) + train_smaller2 (1/3)
            parts_ts, diseases_ts = _split(train_smaller_df)
            ts1_labels, ts2_labels = train_test_split(
                parts_ts,
                test_size=1 / 3,
                stratify=diseases_ts,
                random_state=0,
                shuffle=True,
            )

            ts1_set = set(ts1_labels)
            # Assign roles by participant name (key-based, not order-based)
            roles = {}
            for p in train_participants[self.PARTICIPANT_COL]:
                if p in validation_set:
                    roles[p] = "validation"
                elif p in ts1_set:
                    roles[p] = "train_smaller1"
                else:
                    roles[p] = "train_smaller2"
            train_participants["split_role"] = train_participants[self.PARTICIPANT_COL].map(roles)

        return train_participants

    def _write_split_metadata(self):
        """Write split metadata JSON with generation parameters (atomic)."""
        self._save_metadata_to_cache(self.metadata)

        from malid_lite.__version__ import __version__

        metadata_path = self._get_split_metadata_path()
        metadata = {
            "created_at": datetime.now().isoformat(),
            "malid_lite_version": __version__,
            "sklearn_version": sklearn.__version__,
            "random_state": 0,
            "test_size": "1/3",
            "split_method": "sklearn.model_selection.train_test_split",
            "stratified_by": self.DISEASE_COL,
            "split_level": "participant",
            "metadata_path": str(self.metadata_path),
        }

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=metadata_path.parent, suffix=".json"
        )
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            os.rename(tmp_path, metadata_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if self.verbose >= 2:
            logger.info(f"Wrote split metadata to {metadata_path}")

    def _write_split_summary(
        self,
        splits_df: pd.DataFrame,
        fold_id: Optional[int],
        training_context: str,
    ) -> None:
        """Write a human-readable per-context split summary (atomic).

        Documents exactly what a generated split contains: per-role participant
        counts and a role x disease cross-tab. Applies to ALL contexts (CV and
        train-all) so every generated split is auditable. Pure documentation —
        no code reads this file back.

        File name mirrors the split CSV: ``fold_<id>_<context>_summary.txt`` for
        CV, ``<context>_summary.txt`` for train-all.
        """
        from malid_lite.__version__ import __version__

        splits_dir = self._get_splits_dir()
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Summary filename mirrors the split CSV stem.
        if training_context in self.TRAIN_ALL_TRAINING_CONTEXTS:
            stem = training_context
        else:
            stem = f"fold_{fold_id}_{training_context}"
        summary_path = splits_dir / f"{stem}_summary.txt"

        n_per_role = splits_df["split_role"].value_counts().sort_index()
        # role x disease cross-tab (counts of participants)
        role_by_disease = (
            splits_df.groupby(["split_role", self.DISEASE_COL]).size().unstack(fill_value=0)
        )

        lines = [
            f"Split summary: {self._context_label(fold_id, training_context)}",
            "=" * 60,
            f"Generated:        {datetime.now().isoformat()}",
            f"malid_lite:       {__version__}",
            f"sklearn:          {sklearn.__version__}",
            f"random_state:     0",
            f"test_size:        1/3 per split",
            f"split_method:     sklearn.model_selection.train_test_split",
            f"stratified_by:    {self.DISEASE_COL}",
            f"split_level:      participant",
            f"metadata_path:    {self.metadata_path}",
            "",
            f"Total participants: {len(splits_df)}",
            "",
            "Participants per role:",
        ]
        for role, n in n_per_role.items():
            lines.append(f"  {role:<16} {int(n)}")
        lines.append("")
        lines.append("Participants per role x disease:")
        lines.append(role_by_disease.to_string())
        lines.append("")

        # Atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(dir=splits_dir, suffix=".txt")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                f.write("\n".join(lines))
            os.rename(tmp_path, summary_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if self.verbose >= 2:
            logger.info(f"Wrote split summary to {summary_path}")

    def get_split_participants(
        self,
        fold_id: Optional[int],
        training_context: str,
        split_roles: List[str],
    ) -> List[str]:
        """Convenience: get participant labels for specific split roles.

        Parameters
        ----------
        fold_id : int or None
            Cross-validation fold ID for CV contexts; None for train-all contexts.
        training_context : str
            One of VALID_TRAINING_CONTEXTS (cv_single_model, cv_ensemble,
            train_all, train_all_ensemble).
        split_roles : list of str
            Roles to include, e.g. ["train_smaller1", "train_smaller2"] for
            a model's training set, or ["validation"] for metamodel training.

        Returns
        -------
        list of str
            Participant labels matching the requested roles.
        """
        splits_df = self.load_splits(fold_id, training_context)
        mask = splits_df["split_role"].isin(split_roles)
        return splits_df.loc[mask, self.PARTICIPANT_COL].tolist()

    # ========== Caching Methods ==========
    #
    # Cache architecture: participants/ and embeddings/ are per-participant and
    # self-contained. All other cache artifacts (data_folds/, splits/,
    # metadata_processed.tsv) are auto-generated from the participant cache and
    # metadata on first access. This means a subset dataset can be created by
    # copying the relevant participants/ and embeddings/ files into a new cache
    # directory and providing a subset metadata TSV — the pipeline rebuilds
    # everything else automatically (no --data-dir needed).

    def _get_cache_metadata_path(self, cache_type: str) -> Path:
        """Get path to cache metadata file."""
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        if cache_type == "participants":
            cache_subdir = self.cache_dir / "participants"
        elif cache_type == "data_folds":
            cache_subdir = self.cache_dir / "data_folds"
        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")

        return cache_subdir / "cache_info.json"

    def _save_metadata_to_cache(self, filtered_metadata: pd.DataFrame):
        """Copy original metadata and save filtered metadata to cache.

        Saves two files atomically:
        - metadata.tsv: copy of the original/unfiltered metadata file (for reference)
        - metadata_processed.tsv: filtered to only participants with raw data files

        The processed file is used on subsequent runs so the raw file scan can be
        skipped. The original is kept for debugging and auditing.

        Parameters
        ----------
        filtered_metadata : pd.DataFrame
            Metadata DataFrame already filtered to participants with raw data.
        """
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Copy original metadata (skip if already exists)
        # Only copy when reading from a user-supplied or raw source, not from cache
        cached_raw = self.cache_dir / "metadata.tsv"
        if (
            not cached_raw.exists()
            and self._metadata_needs_filtering
            and self.metadata_path.resolve() != cached_raw.resolve()
        ):
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.cache_dir, suffix=".tsv"
            )
            os.close(tmp_fd)
            try:
                shutil.copy2(self.metadata_path, tmp_path)
                os.rename(tmp_path, cached_raw)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            self._log(f"Copied original metadata to cache: {cached_raw}", level=1)

        # Step 2: Save filtered/processed metadata (always overwrite — cheap
        # and ensures consistency if the source metadata changed)
        cached_processed = self.cache_dir / "metadata_processed.tsv"
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir, suffix=".tsv"
        )
        os.close(tmp_fd)
        try:
            filtered_metadata.to_csv(tmp_path, sep="\t", index=False)
            os.rename(tmp_path, cached_processed)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        self._log(f"Saved processed metadata to cache: {cached_processed}", level=1)

    def _write_cache_metadata(self, cache_type: str, **extra_info):
        """Write cache metadata (timestamp, version, etc.) atomically."""
        if self.cache_dir is None:
            return

        self._save_metadata_to_cache(self.metadata)

        from malid_lite.__version__ import __version__

        metadata_path = self._get_cache_metadata_path(cache_type)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "malid_version": __version__,
            "cache_type": cache_type,
            "data_dir": str(self.data_dir),
            "metadata_path": str(self.metadata_path),
            "gene_locus": self.gene_locus,
            **extra_info
        }

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=metadata_path.parent, suffix=".json"
        )
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            os.rename(tmp_path, metadata_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if self.verbose >= 2:
            logger.info(f"Wrote cache metadata to {metadata_path}")

    def _read_cache_metadata(self, cache_type: str) -> Optional[Dict]:
        """Read cache metadata if it exists. Returns None on corruption."""
        if self.cache_dir is None:
            return None

        metadata_path = self._get_cache_metadata_path(cache_type)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                f"Corrupt cache metadata {metadata_path}: {e}. "
                f"Deleting — it will be recreated on next cache write."
            )
            metadata_path.unlink(missing_ok=True)
            return None

    # ========== Participant-Level Caching ==========

    def get_participant_cache_path(self, participant_label: str) -> Tuple[Path, Path]:
        """
        Get cache file paths for a participant (CLEAN stage).

        Args:
            participant_label: Participant identifier

        Returns:
            Tuple of (data_file_path, stats_file_path)
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        participants_dir = self.cache_dir / "participants"
        data_file = participants_dir / f"{participant_label}_clean.parquet"
        stats_file = participants_dir / f"{participant_label}_stats.json"
        return data_file, stats_file

    def cache_participant(
        self,
        participant_label: str,
        df: pd.DataFrame,
        preprocessing_stats: Optional[Dict] = None,
        update_metadata: bool = True
    ):
        """Cache preprocessed participant data (CLEAN stage) with stats.

        Uses atomic writes (temp file + rename) so that a concurrent
        reader never sees a half-written file.

        Parameters
        ----------
        participant_label : str
            Participant identifier.
        df : pd.DataFrame
            Preprocessed dataframe to cache.
        preprocessing_stats : dict, optional
            Stats from preprocessing (e.g., etl_stats).
        update_metadata : bool
            Whether to update the cache metadata file on first write.
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        cache_file, stats_file = self.get_participant_cache_path(participant_label)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: parquet data
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=cache_file.parent, suffix=".parquet"
        )
        os.close(tmp_fd)
        try:
            df.to_parquet(tmp_path, index=False)
            os.rename(tmp_path, cache_file)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Atomic write: stats JSON (if provided)
        if preprocessing_stats:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=stats_file.parent, suffix=".json"
            )
            os.close(tmp_fd)
            try:
                with open(tmp_path, "w") as f:
                    json.dump(preprocessing_stats, f, indent=2, default=str)
                os.rename(tmp_path, stats_file)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        if self.verbose >= 2:
            logger.info(f"Cached participant {participant_label}: {len(df)} sequences")

        # Update metadata on first write
        if update_metadata and not self._get_cache_metadata_path("participants").exists():
            self._write_cache_metadata("participants", preprocessing_stage="CLEAN")

    def load_cached_participant(self, participant_label: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """Load cached participant data and stats if available.

        If the cache file is corrupt (e.g. from an interrupted write),
        it is deleted and ``None`` is returned so the caller rebuilds
        from the raw data.

        Parameters
        ----------
        participant_label : str
            Participant identifier.

        Returns
        -------
        tuple or None
            ``(dataframe, preprocessing_stats)`` or ``None`` if not cached.
        """
        if self.cache_dir is None:
            return None

        cache_file, stats_file = self.get_participant_cache_path(participant_label)
        if not cache_file.exists():
            return None

        # Read parquet — delete and return None on corruption
        try:
            df = pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(
                f"Corrupt participant cache {cache_file.name}: {e}. "
                f"Deleting and rebuilding from raw data."
            )
            cache_file.unlink(missing_ok=True)
            stats_file.unlink(missing_ok=True)
            return None

        # Load stats if available
        preprocessing_stats = {}
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    preprocessing_stats = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Corrupt participant stats {stats_file.name}: {e}. "
                    f"Ignoring stats — data is still valid."
                )

        if self.verbose >= 2:
            logger.info(f"Loaded participant {participant_label} from cache: {len(df)} sequences")

        return df, preprocessing_stats

    # ========== Fold-Level Caching ==========

    def get_cache_path(
        self,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage,
    ) -> Tuple[Path, Path]:
        """
        Get cache file paths for sequences and metadata.

        Args:
            fold_id: Fold ID (None when fold_label == "all")
            fold_label: "train", "test", or "all"
            preprocessing_stage: Preprocessing stage

        Returns:
            Tuple of (sequences_file_path, metadata_file_path)

        Naming: ``fold_<id>_<label>_<stage>_*`` for CV folds, and
        ``all_<stage>_*`` for the whole-dataset (train-all) cache — the latter
        omits the meaningless fold id so it never collides with a real fold.
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        if fold_label == "all":
            base = f"all_{preprocessing_stage.value}"
        else:
            base = f"fold_{fold_id}_{fold_label}_{preprocessing_stage.value}"
        data_folds_dir = self.cache_dir / "data_folds"
        sequences_file = data_folds_dir / f"{base}_sequences.parquet"
        metadata_file = data_folds_dir / f"{base}_metadata.csv"
        return sequences_file, metadata_file

    def _save_fold_cache(
        self,
        sequences_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ):
        """Save fold data to the fold cache on disk.

        Single entry point for all fold caching. Uses atomic writes
        (write to temp file, then rename) so that a reader never sees
        a half-written file — even if the process is killed mid-write.

        String/object columns are normalized to plain Python ``str``
        before saving to parquet (avoids pyarrow type-inference errors
        on values like ``"HHC 4"``). NaN values are preserved — they
        are NOT converted to the literal string ``"nan"``.

        Parameters
        ----------
        sequences_df : pd.DataFrame
            Sequence-level data for the fold.
        metadata_df : pd.DataFrame
            Specimen-level metadata for the fold.
        fold_id : int
            Cross-validation fold ID.
        fold_label : str
            "train" or "test".
        preprocessing_stage : PreprocessingStage
            Stage of preprocessing applied to the data.

        Raises
        ------
        ValueError
            If ``cache_dir`` is not set.
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        if len(sequences_df) == 0:
            self._log("Skipping fold cache write — no data to cache", level=1)
            return

        data_folds_dir = self.cache_dir / "data_folds"
        data_folds_dir.mkdir(parents=True, exist_ok=True)
        sequences_file, metadata_file = self.get_cache_path(
            fold_id, fold_label, preprocessing_stage
        )

        self._log(
            f"Caching {self._fold_label_desc(fold_id, fold_label)} "
            f"({preprocessing_stage.value})...",
            level=1,
        )

        # --- Prepare sequences for parquet ---
        # Normalize string/object columns to plain Python str so that
        # pyarrow doesn't misinterpret mixed values (e.g. "HHC 4" as int).
        # NaN values are preserved (not converted to the literal "nan").
        seq_to_save = sequences_df.copy()
        for col in seq_to_save.columns:
            if seq_to_save[col].dtype == "object" or str(
                seq_to_save[col].dtype
            ).startswith("string"):
                notna_mask = seq_to_save[col].notna()
                seq_to_save.loc[notna_mask, col] = (
                    seq_to_save.loc[notna_mask, col].astype(str)
                )

        # --- Atomic write: sequences parquet ---
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=data_folds_dir, suffix=".parquet"
        )
        os.close(tmp_fd)
        try:
            seq_to_save.to_parquet(tmp_path, index=False)
            os.rename(tmp_path, sequences_file)
        except BaseException:
            # Clean up temp file on any failure (including KeyboardInterrupt)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # --- Atomic write: metadata CSV ---
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=data_folds_dir, suffix=".csv"
        )
        os.close(tmp_fd)
        try:
            metadata_df.to_csv(tmp_path, index=False)
            os.rename(tmp_path, metadata_file)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        self._log(
            f"Cached {self._fold_label_desc(fold_id, fold_label)}: "
            f"{len(sequences_df):,} sequences to {sequences_file.name}",
            level=1,
        )

        # Write cache metadata (version, timestamps) on first fold write
        metadata_path = self._get_cache_metadata_path("data_folds")
        if not metadata_path.exists():
            self._write_cache_metadata(
                "data_folds", preprocessing_stage=preprocessing_stage.value
            )

    def cache_fold(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ):
        """Build fold data and save it to the fold cache.

        Convenience method that loads the fold via ``get_fold_data()``
        (which itself auto-caches on miss) and ensures the cache is
        populated. If the fold is already cached, this is a no-op.

        Parameters
        ----------
        fold_id : int
            Cross-validation fold ID.
        fold_label : str
            "train" or "test".
        preprocessing_stage : PreprocessingStage
            Stage of preprocessing to cache.
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        # get_fold_data tries fold cache first; on miss it builds from
        # specimens and auto-caches via _save_fold_cache.
        self.get_fold_data(fold_id, fold_label, preprocessing_stage)

    # ---- Whole-dataset ("all") cache manifest ----------------------------
    #
    # The train-all cache (data_folds/all_<stage>_*) must uphold a completeness
    # promise: it should contain every metadata participant that has data. The
    # parquet alone can't verify this on a cache hit — a participant added to the
    # metadata after the cache was built would simply be absent, indistinguishable
    # from one legitimately dropped by QC. The manifest closes that gap by recording
    # the metadata (specimen, participant) pairs AS OF BUILD TIME (the input, which
    # includes QC-dropped pairs), so a later load compares metadata-then vs
    # metadata-now and detects any add/remove.

    @staticmethod
    def _metadata_pairs(df: pd.DataFrame) -> set:
        """Set of (specimen_label, participant_label) string pairs in a frame."""
        return set(
            zip(
                df["specimen_label"].astype(str),
                df["participant_label"].astype(str),
            )
        )

    def _get_all_cache_manifest_path(
        self, preprocessing_stage: PreprocessingStage
    ) -> Path:
        """Path to the whole-dataset cache manifest for a given stage."""
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")
        return (
            self.cache_dir
            / "data_folds"
            / f"all_{preprocessing_stage.value}_cache_manifest.json"
        )

    def _write_all_cache_manifest(
        self,
        cached_metadata_df: pd.DataFrame,
        preprocessing_stage: PreprocessingStage,
    ) -> None:
        """Write the whole-dataset cache manifest (atomic).

        Records the metadata (specimen, participant) pairs at build time plus,
        for human-readable documentation, the subset that was QC-dropped (present
        in the metadata but absent from the cached data because all their
        sequences failed downsampling thresholds).

        Parameters
        ----------
        cached_metadata_df : Specimen-level metadata actually written to the
            "all" cache (i.e. QC-passing specimens only).
        preprocessing_stage : Stage the cache was built at.
        """
        from malid_lite.__version__ import __version__

        metadata_pairs = self._metadata_pairs(self.metadata)
        cached_pairs = self._metadata_pairs(cached_metadata_df)
        # QC-dropped = in metadata but not in the cache (documentation only).
        qc_dropped_pairs = metadata_pairs - cached_pairs

        # Sort for deterministic, diff-friendly output.
        manifest = {
            "built_at": datetime.now().isoformat(),
            "malid_lite_version": __version__,
            "preprocessing_stage": preprocessing_stage.value,
            "n_metadata_pairs": len(metadata_pairs),
            "n_cached_pairs": len(cached_pairs),
            "n_qc_dropped": len(qc_dropped_pairs),
            "metadata_pairs": sorted([list(p) for p in metadata_pairs]),
            "qc_dropped_pairs": sorted([list(p) for p in qc_dropped_pairs]),
        }

        manifest_path = self._get_all_cache_manifest_path(preprocessing_stage)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_path.parent, suffix=".json")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            os.rename(tmp_path, manifest_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if self.verbose >= 2:
            logger.info(f"Wrote whole-dataset cache manifest to {manifest_path}")

    def _validate_all_cache_manifest(
        self, preprocessing_stage: PreprocessingStage
    ) -> bool:
        """Validate a cached "all" bundle against the current metadata.

        Compares the current metadata (specimen, participant) pairs to those
        recorded when the cache was built. Any difference (participants added or
        removed) means the cached whole-dataset bundle is stale.

        Returns
        -------
        bool
            True  — the cached bundle may be used (manifest valid, or absent for a
                    cache built before manifests existed — the caller then falls
                    back to the one-directional removals-only pair check).
            False — the manifest is corrupt/unreadable, so completeness cannot be
                    verified; the caller must DISCARD and rebuild the cache (which
                    regenerates a fresh manifest and re-runs the completeness check).
                    We rebuild rather than silently degrade so the "no added
                    participant is silently dropped" guarantee is never permanently
                    lost by an unverifiable manifest.

        Raises
        ------
        ValueError
            If the manifest is valid but the metadata has changed since build
            (participants added or removed) — the cache is stale.
        """
        manifest_path = self._get_all_cache_manifest_path(preprocessing_stage)
        if not manifest_path.exists():
            return True  # older cache — backward-compatible removals-only fallback

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            built_pairs = {tuple(p) for p in manifest["metadata_pairs"]}
        except Exception as e:
            logger.warning(
                f"Corrupt whole-dataset cache manifest {manifest_path.name}: {e}. "
                f"Discarding the whole-dataset cache and rebuilding so completeness "
                f"is re-verified (a corrupt manifest cannot confirm no participant "
                f"was silently dropped)."
            )
            manifest_path.unlink(missing_ok=True)
            return False  # signal the caller to rebuild

        current_pairs = self._metadata_pairs(self.metadata)
        added = current_pairs - built_pairs      # in metadata now, not at build
        removed = built_pairs - current_pairs     # at build, gone from metadata

        if added or removed:
            added_ex = sorted(added)[:5]
            removed_ex = sorted(removed)[:5]
            raise ValueError(
                f"Whole-dataset ('all') cache is stale: the metadata has changed "
                f"since it was built ({len(added)} specimen(s) added, "
                f"{len(removed)} removed). "
                f"Added examples: {added_ex}. Removed examples: {removed_ex}. "
                f"Clear the fold cache and rebuild: "
                f"python scripts/data/manage_cache.py clear-folds"
            )
        return True

    def load_cached_fold(
        self,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load cached fold data if available.

        If the cache files exist but are corrupt (e.g. from an interrupted
        write before atomic-write support was added), they are deleted and
        ``None`` is returned so the caller can rebuild.

        Parameters
        ----------
        fold_id : int or None
            Cross-validation fold ID (None when fold_label == "all").
        fold_label : str
            "train", "test", or "all".
        preprocessing_stage : PreprocessingStage
            Preprocessing stage.

        Returns
        -------
        tuple or None
            ``(sequences_df, metadata_df)`` if cache is valid, else ``None``.
        """
        if self.cache_dir is None:
            return None

        sequences_file, metadata_file = self.get_cache_path(
            fold_id, fold_label, preprocessing_stage
        )

        if not sequences_file.exists() or not metadata_file.exists():
            return None

        # Read cached files — delete and return None on corruption
        try:
            sequences_df = pd.read_parquet(sequences_file)
        except Exception as e:
            logger.warning(
                f"Corrupt fold cache parquet {sequences_file.name}: {e}. "
                f"Deleting and rebuilding."
            )
            sequences_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None

        try:
            metadata_df = pd.read_csv(metadata_file)
        except Exception as e:
            logger.warning(
                f"Corrupt fold cache metadata {metadata_file.name}: {e}. "
                f"Deleting and rebuilding."
            )
            sequences_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None

        # Backward compat: old fold caches have repertoire_id, new ones have specimen_label
        if "repertoire_id" in sequences_df.columns and "specimen_label" not in sequences_df.columns:
            sequences_df = sequences_df.rename(columns={"repertoire_id": "specimen_label"})

        # Backward compat: old fold caches use the legacy fold column name
        metadata_df = normalize_fold_column(metadata_df)

        # Normalize identifier columns (int64 → str) so comparisons with
        # metadata (always str) work correctly
        sequences_df = normalize_identifier_columns(sequences_df)
        metadata_df = normalize_identifier_columns(metadata_df)

        # For the whole-dataset ("all") cache, validate against the build-time
        # manifest first. This catches BOTH added and removed metadata participants
        # (the parquet-only check below catches removals only), upholding the
        # train-all completeness promise even on a cache hit. Raises if stale.
        # A corrupt/unverifiable manifest returns False → discard the cached bundle
        # and return None so the caller rebuilds it (regenerating a fresh manifest
        # and re-running completeness checks); we never silently trust an
        # unverifiable "all" cache.
        if fold_label == "all":
            if not self._validate_all_cache_manifest(preprocessing_stage):
                logger.warning(
                    f"Discarding whole-dataset cache files ({sequences_file.name}, "
                    f"{metadata_file.name}) so they are rebuilt with a fresh manifest."
                )
                sequences_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                return None

        # Validate that (specimen_label, participant_label) pairs in cached sequences
        # match metadata. A mismatch means the cache is stale or was built from a
        # different metadata file. (Removals only: cached pairs absent from
        # metadata. For the "all" cache, the manifest check above additionally
        # catches additions.)
        if "specimen_label" in sequences_df.columns and "participant_label" in sequences_df.columns:
            seq_pairs = self._metadata_pairs(sequences_df)
            meta_pairs = self._metadata_pairs(self.metadata)
            mismatched = seq_pairs - meta_pairs
            if mismatched:
                examples = sorted(mismatched)[:5]
                raise ValueError(
                    f"Fold cache {sequences_file.name} has (specimen_label, participant_label) "
                    f"pairs not found in metadata ({len(mismatched)} mismatched). "
                    f"Examples: {examples}. "
                    f"The cache may be stale or built from a different metadata file. "
                    f"Clear fold caches with: python scripts/data/manage_cache.py clear-folds"
                )

        if self.verbose >= 1:
            logger.info(f"Loaded fold {fold_id}/{fold_label}: "
                        f"{len(sequences_df):,} sequences, "
                        f"{metadata_df['specimen_label'].nunique()} specimens from cache")

        return sequences_df, metadata_df

    # ========== Cache Management ==========

    def clear_participant_cache(
        self,
        participant_label: Optional[str] = None,
        confirm: bool = True
    ):
        """
        Clear participant-level cache.

        Args:
            participant_label: If specified, only clear this participant. Otherwise clear all.
            confirm: If True, requires confirmation before deleting
        """
        if self.cache_dir is None:
            logger.warning("No cache_dir set")
            return

        participants_dir = self.cache_dir / "participants"
        if not participants_dir.exists():
            logger.info("No participant cache to clear")
            return

        if participant_label:
            # Clear specific participant (both data and stats files)
            cache_file, stats_file = self.get_participant_cache_path(participant_label)
            deleted = False
            if cache_file.exists():
                if confirm:
                    logger.info(f"Deleting participant cache: {cache_file}")
                cache_file.unlink()
                deleted = True
            if stats_file.exists():
                stats_file.unlink()
                deleted = True
            if not deleted:
                logger.info(f"No cache found for participant: {participant_label}")
        else:
            # Clear all participants
            n_files = len(list(participants_dir.glob("*.parquet")))
            if confirm:
                logger.info(f"Deleting {n_files} participant cache files from {participants_dir}")
            shutil.rmtree(participants_dir)
            logger.info("Participant cache cleared")

    def clear_fold_cache(
        self,
        fold_id: Optional[int] = None,
        fold_label: Optional[str] = None,
        confirm: bool = True
    ):
        """
        Clear fold-level cache.

        Args:
            fold_id: If specified, only clear this fold
            fold_label: If specified, only clear this label (train/test)
            confirm: If True, requires confirmation before deleting
        """
        if self.cache_dir is None:
            logger.warning("No cache_dir set")
            return

        if fold_label == "all":
            # Clear ONLY the whole-dataset (train-all) cache — parquet, metadata CSV,
            # and the manifest — without touching CV fold caches. (Without this branch,
            # fold_id=None + fold_label="all" would fall through to the clear-all path
            # and wipe the CV fold caches too.)
            data_folds_dir = self.cache_dir / "data_folds"
            if not data_folds_dir.exists():
                logger.info("No fold cache to clear")
                return
            files = (
                list(data_folds_dir.glob("all_*.parquet"))
                + list(data_folds_dir.glob("all_*.csv"))
                + list(data_folds_dir.glob("all_*.json"))  # whole-dataset cache manifest
            )
            if confirm:
                logger.info(
                    f"Deleting {len(files)} train-all (all_*) cache file(s) from {data_folds_dir}"
                )
            for f in files:
                f.unlink()
            logger.info("Train-all (all_*) cache cleared")
            return

        if fold_id is not None and fold_label is not None:
            # Clear specific fold
            for stage in PreprocessingStage:
                sequences_file, metadata_file = self.get_cache_path(fold_id, fold_label, stage)
                for f in [sequences_file, metadata_file]:
                    if f.exists():
                        if confirm:
                            logger.info(f"Deleting: {f}")
                        f.unlink()
        else:
            # Clear all folds — delete parquet, CSV, and orphaned temp files.
            # Covers both CV fold caches (fold_*) and the train-all whole-dataset
            # cache (all_*), so a "clear folds" never leaves a stale cache behind.
            data_folds_dir = self.cache_dir / "data_folds"
            if not data_folds_dir.exists():
                logger.info("No fold cache to clear")
                return
            files = (
                list(data_folds_dir.glob("fold_*.parquet"))
                + list(data_folds_dir.glob("fold_*.csv"))
                + list(data_folds_dir.glob("all_*.parquet"))
                + list(data_folds_dir.glob("all_*.csv"))
                + list(data_folds_dir.glob("all_*.json"))  # whole-dataset cache manifest
                + list(data_folds_dir.glob("tmp*"))  # orphaned atomic-write temps
            )
            if confirm:
                logger.info(f"Deleting {len(files)} fold cache files from {data_folds_dir}")
            for f in files:
                f.unlink()

            # Remove metadata
            metadata_path = self._get_cache_metadata_path("data_folds")
            if metadata_path.exists():
                metadata_path.unlink()

            logger.info("Fold cache cleared")

    def clear_all_caches(self, confirm: bool = True):
        """
        Clear all caches (participants and folds).

        Args:
            confirm: If True, requires confirmation before deleting
        """
        if self.cache_dir is None:
            logger.warning("No cache_dir set")
            return

        if not self.cache_dir.exists():
            logger.info("No cache directory to clear")
            return

        if confirm:
            logger.info(f"Deleting entire cache directory: {self.cache_dir}")

        shutil.rmtree(self.cache_dir)
        logger.info("All caches cleared")

    def get_cache_info(self) -> Dict[str, any]:
        """
        Get information about current caches.

        Returns:
            Dict with cache statistics and metadata
        """
        info = {
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "participants": {},
            "folds": {},
            "splits": {},
        }

        if self.cache_dir is None or not self.cache_dir.exists():
            return info

        # Participant cache info
        participants_dir = self.cache_dir / "participants"
        if participants_dir.exists():
            participant_files = list(participants_dir.glob("*.parquet"))
            info["participants"] = {
                "count": len(participant_files),
                "metadata": self._read_cache_metadata("participants")
            }

        # Fold cache info — report CV fold caches (fold_*) and the train-all
        # whole-dataset cache (all_*) separately so both are visible.
        data_folds_dir = self.cache_dir / "data_folds"
        if data_folds_dir.exists():
            cv_fold_files = list(data_folds_dir.glob("fold_*.parquet"))
            train_all_files = list(data_folds_dir.glob("all_*.parquet"))
            manifest_files = list(data_folds_dir.glob("all_*_cache_manifest.json"))
        else:
            cv_fold_files, train_all_files, manifest_files = [], [], []
        info["folds"] = {
            "count": len(cv_fold_files) + len(train_all_files),
            "cv_fold_count": len(cv_fold_files),
            "train_all_count": len(train_all_files),
            "train_all_manifests": sorted(p.name for p in manifest_files),
            "metadata": self._read_cache_metadata("data_folds")
        }

        # Split-assignment files (per-context participant→role CSVs). Surfaced so
        # `manage_cache info` shows them alongside the caches they drive.
        splits_dir = self._get_splits_dir()
        if splits_dir.exists():
            split_files = sorted(p.name for p in splits_dir.glob("*.csv"))
        else:
            split_files = []
        info["splits"] = {"count": len(split_files), "files": split_files}

        return info

    def _log(self, message: str, level: int = 1):
        """
        Log message if verbose >= level.

        Args:
            message: Message to log
            level: Minimum verbosity level required (0=always, 1=normal, 2=debug)
        """
        if self.verbose >= level:
            logger.info(message)
