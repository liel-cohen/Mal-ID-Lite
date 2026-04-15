"""Abstract base class for immune repertoire data loaders."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Iterator
from pathlib import Path
from enum import Enum
from datetime import datetime
import pandas as pd
import logging
import json
import shutil

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
        data_dir: Path,
        metadata_path: Path,
        gene_locus: str = "TCR",
        verbose: int = 1,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Path to directory containing repertoire files
            metadata_path: Path to metadata TSV file
            gene_locus: Gene locus to load ("TCR" or "BCR")
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
            cache_dir: Optional directory for caching preprocessed data
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = Path(metadata_path)
        self.gene_locus = gene_locus
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Validate gene locus
        if gene_locus not in ["TCR", "BCR"]:
            raise ValueError(f"gene_locus must be 'TCR' or 'BCR', got: {gene_locus}")

        # Accumulate preprocessing statistics
        self._preprocessing_stats: List[Dict] = []

        # Metadata in memory (lightweight, lazy loaded)
        self._metadata: Optional[pd.DataFrame] = None

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
        Load all data for a fold into memory (convenience method).

        ⚠️  Warning: May use large amounts of memory for big datasets.
        Prefer iter_fold_specimens() for memory-efficient loading.

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

        return (
            pd.concat(all_sequences, ignore_index=True),
            pd.DataFrame(all_metadata),
        )

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
        participant_df = self.load_participant_data(
            participant_label, preprocessing_stage
        )
        return participant_df[participant_df["repertoire_id"] == specimen_label]

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

    # ========== Caching Methods ==========

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

    def _write_cache_metadata(self, cache_type: str, **extra_info):
        """Write cache metadata (timestamp, version, etc.)."""
        if self.cache_dir is None:
            return

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

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose >= 2:
            logger.info(f"Wrote cache metadata to {metadata_path}")

    def _read_cache_metadata(self, cache_type: str) -> Optional[Dict]:
        """Read cache metadata if it exists."""
        if self.cache_dir is None:
            return None

        metadata_path = self._get_cache_metadata_path(cache_type)
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

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
        """
        Cache preprocessed participant data (CLEAN stage) with stats.

        Args:
            participant_label: Participant identifier
            df: Preprocessed dataframe to cache
            preprocessing_stats: Stats from preprocessing (e.g., etl_stats)
            update_metadata: Whether to update cache metadata file
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        cache_file, stats_file = self.get_participant_cache_path(participant_label)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        df.to_parquet(cache_file, index=False)

        # Save preprocessing stats if provided
        if preprocessing_stats:
            with open(stats_file, "w") as f:
                json.dump(preprocessing_stats, f, indent=2, default=str)

        if self.verbose >= 2:
            logger.info(f"Cached participant {participant_label}: {len(df)} sequences")

        # Update metadata on first write
        if update_metadata and not self._get_cache_metadata_path("participants").exists():
            self._write_cache_metadata("participants", preprocessing_stage="CLEAN")

    def load_cached_participant(self, participant_label: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Load cached participant data and stats if available.

        Args:
            participant_label: Participant identifier

        Returns:
            Tuple of (dataframe, preprocessing_stats) or None if not cached
        """
        if self.cache_dir is None:
            return None

        cache_file, stats_file = self.get_participant_cache_path(participant_label)
        if not cache_file.exists():
            return None

        # Load data
        df = pd.read_parquet(cache_file)

        # Load stats if available
        preprocessing_stats = {}
        if stats_file.exists():
            with open(stats_file, "r") as f:
                preprocessing_stats = json.load(f)

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

    def cache_fold(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
        use_participant_cache: bool = True,
    ):
        """
        Cache preprocessed fold data to disk (parquet format).

        Efficiently builds fold cache from participant-level caches when available.

        Args:
            fold_id: Fold ID
            fold_label: Fold label
            preprocessing_stage: Preprocessing stage to cache
            use_participant_cache: If True, loads from participant caches (faster)
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir not set")

        (self.cache_dir / "data_folds").mkdir(parents=True, exist_ok=True)
        sequences_file, metadata_file = self.get_cache_path(
            fold_id, fold_label, preprocessing_stage
        )

        if self.verbose >= 1:
            logger.info(
                f"Caching fold {fold_id} {fold_label} ({preprocessing_stage.value})..."
            )

        sequences_df, metadata_df = self.get_fold_data(
            fold_id, fold_label, preprocessing_stage
        )

        # Convert string/object columns to avoid Parquet type inference issues
        # (Parquet tries to convert strings like "HHC 4" to integers)
        sequences_df = sequences_df.copy()
        for col in sequences_df.columns:
            if sequences_df[col].dtype == 'object' or str(sequences_df[col].dtype).startswith('string'):
                # Convert to object dtype with plain strings
                sequences_df[col] = sequences_df[col].astype(str).astype('object')

        # Save sequences as parquet (large, benefits from compression)
        # Save metadata as CSV (small, avoids type inference issues)
        sequences_df.to_parquet(sequences_file, index=False)
        metadata_df.to_csv(metadata_file, index=False)

        if self.verbose >= 1:
            logger.info(
                f"Cached {len(sequences_df)} sequences to {sequences_file.name}"
            )

        # Update metadata on first write
        metadata_path = self._get_cache_metadata_path("data_folds")
        if not metadata_path.exists():
            self._write_cache_metadata("data_folds", preprocessing_stage=preprocessing_stage.value)

    def load_cached_fold(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load cached fold data if available.

        Args:
            fold_id: Fold ID
            fold_label: Fold label
            preprocessing_stage: Preprocessing stage

        Returns:
            Tuple of (sequences_df, metadata_df) if cache exists, else None
        """
        if self.cache_dir is None:
            return None

        sequences_file, metadata_file = self.get_cache_path(
            fold_id, fold_label, preprocessing_stage
        )

        if not sequences_file.exists() or not metadata_file.exists():
            return None

        sequences_df = pd.read_parquet(sequences_file)
        metadata_df = pd.read_csv(metadata_file)

        if self.verbose >= 1:
            logger.info(f"Loaded {len(sequences_df):,} sequences from cache")

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
            # Clear all folds — delete both parquet and metadata CSV files
            data_folds_dir = self.cache_dir / "data_folds"
            if not data_folds_dir.exists():
                logger.info("No fold cache to clear")
                return
            files = list(data_folds_dir.glob("fold_*.parquet")) + list(data_folds_dir.glob("fold_*.csv"))
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
