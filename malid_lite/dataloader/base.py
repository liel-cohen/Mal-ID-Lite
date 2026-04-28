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
            self._metadata_needs_filtering = True
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"metadata_path does not exist: {self.metadata_path}"
                )
            # If a cached copy also exists, verify they match
            if cached_metadata_raw is not None and cached_metadata_raw.exists():
                if self.metadata_path.resolve() != cached_metadata_raw.resolve():
                    if not filecmp.cmp(
                        self.metadata_path, cached_metadata_raw, shallow=False
                    ):
                        raise ValueError(
                            f"Supplied metadata_path ({self.metadata_path}) differs from "
                            f"cached copy ({cached_metadata_raw}). The cache may be stale. "
                            f"Clear caches with: python scripts/data/manage_cache.py clear-all"
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
                - malid_cross_validation_fold_id_when_in_test_set
                - available_gene_loci
                - (and other study-specific columns)
        """
        pass

    # ========== Memory-Efficient Iterator Methods ==========

    @abstractmethod
    def iter_fold_specimens(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        """
        Iterate over specimens in a fold one at a time (RECOMMENDED for memory efficiency).

        Hybrid approach: Metadata in memory, sequences loaded on-demand.

        Args:
            fold_id: Cross-validation fold ID (typically 0-4)
            fold_label: "train" (all specimens except fold_id) or "test" (only fold_id)
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
        Load all data for a fold into memory.

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
        # Try fold cache first — single parquet read, much faster than iterating specimens
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

        # Auto-cache the freshly built fold for next time
        if self.cache_dir is not None and len(sequences_df) > 0:
            try:
                self._save_fold_cache(
                    sequences_df, metadata_df, fold_id, fold_label, preprocessing_stage
                )
            except Exception as e:
                logger.warning(
                    f"Failed to auto-cache fold {fold_id}/{fold_label}: {e}. "
                    f"Continuing without caching."
                )

        return sequences_df, metadata_df

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

    # Valid training contexts and their split roles
    VALID_TRAINING_CONTEXTS = ("cv_single_model", "cv_ensemble")
    FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"
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

    def _get_split_path(self, fold_id: int, training_context: str) -> Path:
        """Get the path to a specific split CSV file."""
        return self._get_splits_dir() / f"fold_{fold_id}_{training_context}.csv"

    def _get_split_metadata_path(self) -> Path:
        """Get the path to the split metadata JSON file."""
        return self._get_splits_dir() / "split_metadata.json"

    def load_splits(
        self,
        fold_id: int,
        training_context: str,
    ) -> pd.DataFrame:
        """Load participant split assignments for a fold, generating if needed.

        If the split CSV already exists, loads and returns it.
        If it does not exist, generates the splits deterministically, saves
        to disk, and returns the result.

        Parameters
        ----------
        fold_id : int
            Cross-validation fold ID (the fold used as test set).
        training_context : str
            One of "cv_single_model" or "cv_ensemble".

        Returns
        -------
        pd.DataFrame
            Columns: participant_label, disease, split_role.
            split_role values depend on training_context:
              cv_single_model: "test", "train_smaller1", "train_smaller2"
              cv_ensemble:     "test", "validation", "train_smaller1", "train_smaller2"
        """
        if training_context not in self.VALID_TRAINING_CONTEXTS:
            raise ValueError(
                f"training_context must be one of {self.VALID_TRAINING_CONTEXTS}, "
                f"got: {training_context!r}"
            )

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
                    if self.verbose >= 1:
                        n_per_role = splits_df["split_role"].value_counts().to_dict()
                        logger.info(
                            f"Loaded splits for fold {fold_id} ({training_context}) "
                            f"from {split_path.name}: {n_per_role}"
                        )
                    return splits_df

        # --- Generate splits ---
        if self.verbose >= 1:
            logger.info(
                f"Split file not found for fold {fold_id} ({training_context}). "
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

        # Write/update metadata on first write
        self._write_split_metadata()

        if self.verbose >= 1:
            n_per_role = splits_df["split_role"].value_counts().to_dict()
            logger.info(
                f"Saved splits for fold {fold_id} ({training_context}) "
                f"to {split_path.name}: {n_per_role}"
            )

        return splits_df

    def _generate_splits(
        self,
        fold_id: int,
        training_context: str,
    ) -> pd.DataFrame:
        """Generate participant split assignments for one fold.

        Split logic matches the original Mal-ID exactly
        (notebooks_src/make_cv_folds.py:322-345):
        - All train_test_split calls use test_size=1/3, random_state=0,
          shuffle=True, stratify=disease
        - Splits are at the participant level
        - Sequential calls: train -> validation + train_smaller (cv_ensemble only),
          then train_smaller -> train_smaller1 + train_smaller2

        Parameters
        ----------
        fold_id : int
            The fold used as the test set.
        training_context : str
            "cv_single_model" or "cv_ensemble".

        Returns
        -------
        pd.DataFrame
            Columns: participant_label, disease, split_role
        """
        meta = self.metadata

        # --- Get unique participants with their disease ---
        # Sort by participant_label for deterministic ordering: ensures
        # train_test_split produces the same result regardless of how
        # metadata was loaded or what order rows appear in.
        participant_disease = (
            meta
            .drop_duplicates(subset=[self.PARTICIPANT_COL])
            [[self.PARTICIPANT_COL, self.DISEASE_COL, self.FOLD_COL]]
            .sort_values(self.PARTICIPANT_COL)
            .reset_index(drop=True)
        )

        # --- Test vs train by fold column ---
        test_mask = participant_disease[self.FOLD_COL] == fold_id
        test_participants = participant_disease.loc[test_mask, [self.PARTICIPANT_COL, self.DISEASE_COL]].copy()
        train_participants = participant_disease.loc[~test_mask, [self.PARTICIPANT_COL, self.DISEASE_COL]].copy()

        test_participants["split_role"] = "test"

        # Helper: sorted participant/disease lists for train_test_split.
        # Sorting is already done above, but we call .tolist() to avoid
        # arrow-backed array issues with sklearn.
        def _split(df):
            return (df[self.PARTICIPANT_COL].tolist(),
                    df[self.DISEASE_COL].tolist())

        if training_context == "cv_single_model":
            # Single split: train -> train_smaller1 (2/3) + train_smaller2 (1/3)
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

        elif training_context == "cv_ensemble":
            # First split: train -> validation (1/3) + train_smaller (2/3)
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

        # Combine and return (sorted by participant for readability)
        result = pd.concat(
            [test_participants, train_participants],
            ignore_index=True,
        )[[self.PARTICIPANT_COL, self.DISEASE_COL, "split_role"]]
        result = result.sort_values(self.PARTICIPANT_COL).reset_index(drop=True)

        # Sanity checks
        n_total = len(result)
        n_unique = result[self.PARTICIPANT_COL].nunique()
        assert n_total == n_unique, (
            f"Duplicate participants in splits: {n_total} rows but {n_unique} unique participants"
        )
        assert not result["split_role"].isna().any(), "Some participants have no split_role assigned"

        return result

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

    def get_split_participants(
        self,
        fold_id: int,
        training_context: str,
        split_roles: List[str],
    ) -> List[str]:
        """Convenience: get participant labels for specific split roles.

        Parameters
        ----------
        fold_id : int
            Cross-validation fold ID.
        training_context : str
            "cv_single_model" or "cv_ensemble".
        split_roles : list of str
            Roles to include, e.g. ["train_smaller1", "train_smaller2"] for
            Model 1's training set, or ["validation"] for metamodel training.

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
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage,
    ) -> Tuple[Path, Path]:
        """
        Get cache file paths for sequences and metadata.

        Args:
            fold_id: Fold ID
            fold_label: Fold label
            preprocessing_stage: Preprocessing stage

        Returns:
            Tuple of (sequences_file_path, metadata_file_path)
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        base = f"fold_{fold_id}_{fold_label}_{preprocessing_stage.value}"
        data_folds_dir = self.cache_dir / "data_folds"
        sequences_file = data_folds_dir / f"{base}_sequences.parquet"
        metadata_file = data_folds_dir / f"{base}_metadata.csv"
        return sequences_file, metadata_file

    def _save_fold_cache(
        self,
        sequences_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fold_id: int,
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
            f"Caching fold {fold_id} {fold_label} ({preprocessing_stage.value})...",
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
            f"Cached fold {fold_id}/{fold_label}: "
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

    def load_cached_fold(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load cached fold data if available.

        If the cache files exist but are corrupt (e.g. from an interrupted
        write before atomic-write support was added), they are deleted and
        ``None`` is returned so the caller can rebuild.

        Parameters
        ----------
        fold_id : int
            Cross-validation fold ID.
        fold_label : str
            "train" or "test".
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

        # Validate that (specimen_label, participant_label) pairs in cached sequences
        # match metadata. A mismatch means the cache is stale or was built from a
        # different metadata file.
        if "specimen_label" in sequences_df.columns and "participant_label" in sequences_df.columns:
            seq_pairs = set(
                zip(sequences_df["specimen_label"], sequences_df["participant_label"])
            )
            meta_pairs = set(
                zip(self.metadata["specimen_label"], self.metadata["participant_label"])
            )
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
            # Clear all folds — delete parquet, CSV, and orphaned temp files
            data_folds_dir = self.cache_dir / "data_folds"
            if not data_folds_dir.exists():
                logger.info("No fold cache to clear")
                return
            files = (
                list(data_folds_dir.glob("fold_*.parquet"))
                + list(data_folds_dir.glob("fold_*.csv"))
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
            "folds": {}
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

        # Fold cache info
        data_folds_dir = self.cache_dir / "data_folds"
        fold_files = list(data_folds_dir.glob("fold_*.parquet")) if data_folds_dir.exists() else []
        info["folds"] = {
            "count": len(fold_files),
            "metadata": self._read_cache_metadata("data_folds")
        }

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
