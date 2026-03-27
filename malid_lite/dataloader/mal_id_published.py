"""Data loader for Mal-ID published AIRR format data."""

from pathlib import Path
from typing import Optional, Dict, Tuple, Iterator
import pandas as pd
import numpy as np
import logging

from .base import BaseDataLoader, PreprocessingStage

logger = logging.getLogger(__name__)


class MalIDPublishedDataLoader(BaseDataLoader):
    """
    Data loader for Mal-ID published data in AIRR format.

    Features:
    - Loads gzip-compressed AIRR format TSV files (per participant)
    - Implements 2-stage preprocessing (cleaning + downsampling)
    - Memory-efficient iteration over specimens
    - Automatic statistics accumulation
    - Caching support

    File format:
    - Input: part_table_{participant_label}.tsv.gz (one file per participant)
    - AIRR format (produced by clean_tcr_data_to_airr.py)
    - May contain multiple specimens per file (distinguished by repertoire_id)
    - All column names are AIRR format (e.g. v_call, repertoire_id, cdr3_aa)
    """

    # Gene allele corrections (exact match, includes allele)
    # E.g., "TRBV6-2*02" -> "TRBV6-2*01"
    # [From original malid repo: Our old IgBLAST can generate TRBV6-2*02 calls, but no CDR1+2 information is available for this allele from get_tcr_v_gene_annotations, because it has been renamed:
    # https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV - see (40) ]
    GENE_ALLELE_FIXES = {
        "TRBV6-2*02": "TRBV6-2*01",
    }

    # Gene corrections (prefix match, will replace gene+any allele with target+*01)
    # E.g., "TRBV12-4*03" -> "TRBV12-3*01", "TRBV12-4*01" -> "TRBV12-3*01"
    # [From original malid repo: Replace indistinguishable TRBV gene names with the version that we use in our data.
    # https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4 ]
    GENE_FIXES = {
        "TRBV12-4": "TRBV12-3",
        "TRBV6-3": "TRBV6-2",
    }

    # Minimum requirements per specimen (TCR)
    # TODO: BCR has different thresholds
    MIN_CLONES = {"TCRB": 500}
    MIN_SEQUENCES = 1000

    # Quality score thresholds
    V_SCORE_THRESHOLD = {"TCR": 80, "BCR": 200}

    # Standard 20 amino acids — CDR3 rows containing other characters are dropped
    VALID_AMINO_ACIDS: frozenset = frozenset("ACDEFGHIKLMNPQRSTVWY")

    # V gene names to remove entirely (pseudogenes with no reference data).
    # These genes are absent from the gene reference table, so their CDR1/CDR2
    # sequences can never be filled. Matches original Mal-ID etl.py behavior.
    GENES_TO_REMOVE: frozenset = frozenset({"TRBV25/OR9-2*01"})

    # AIRR boolean columns that may arrive as 't'/'f' (lowercase) instead of
    # 'T'/'F'. Uppercased on load for consistent downstream comparisons.
    AIRR_BOOLEAN_COLS: list = ["productive", "stop_codon", "vj_in_frame"]

    def __init__(
        self,
        data_dir: Path,
        metadata_path: Path,
        gene_locus: str = "TCR",
        verbose: int = 1,
        gene_reference_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize Mal-ID published data loader.

        Args:
            data_dir: Path to airr_format_clean/TCR/ directory
                     (contains AIRR format part_table_{participant_label}.tsv.gz files)
            metadata_path: Path to metadata.tsv
            gene_locus: "TCR" or "BCR" (only TCR fully supported initially)
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
            gene_reference_path: Path to tcrb_v_gene_cdrs.generated.tsv
                                (Required for FR/CDR extraction)
            cache_dir: Optional directory for caching preprocessed data
        """
        super().__init__(data_dir, metadata_path, gene_locus, verbose, cache_dir)

        self.gene_reference_path = gene_reference_path
        self._gene_reference = None  # Lazy load

        if gene_reference_path is None:
            logger.warning(
                "gene_reference_path not provided. "
                "FR1-FR3 and CDR1-CDR2 sequences will NOT be extracted."
            )
        elif not Path(gene_reference_path).exists():
            logger.warning(
                f"Gene reference file not found: {gene_reference_path}. "
                "FR/CDR extraction will be skipped."
            )

    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate metadata.tsv.

        Logs statistics about data availability.

        Returns:
            DataFrame with metadata
        """
        self._log("Loading metadata...", level=1)

        # Load metadata
        metadata = pd.read_csv(self.metadata_path, sep="\t")

        # Log statistics
        self._log(f"Total samples in metadata: {len(metadata)}", level=1)

        # Filter to gene locus
        if "available_gene_loci" in metadata.columns:
            # Check for samples with our gene locus
            has_locus = metadata["available_gene_loci"].str.contains(
                self.gene_locus, na=False
            )
            n_with_locus = has_locus.sum()
            self._log(
                f"Samples with {self.gene_locus} data: {n_with_locus}", level=1
            )

            # Filter metadata to only samples with this gene locus
            metadata = metadata[has_locus].copy()

        # Enforce one-disease-per-participant constraint.
        # Models fundamentally require this: stratified CV splits are by participant disease,
        # binary pair filtering is participant-level, and Model 2's Fisher test counts
        # participants per disease. Participants with multiple disease labels cannot be
        # handled correctly and indicate a metadata problem.
        if "participant_label" in metadata.columns and "disease" in metadata.columns:
            multi_disease = metadata.groupby("participant_label")["disease"].nunique()
            bad_participants = multi_disease[multi_disease > 1].index.tolist()
            if bad_participants:
                raise ValueError(
                    f"Participants with multiple disease labels found — models require exactly "
                    f"one disease per participant: {bad_participants}"
                )

        # Check how many files exist
        if self.verbose >= 1:
            existing_files = 0
            missing_files = 0
            for participant_label in metadata["participant_label"].unique():
                file_path_gz = self.data_dir / f"part_table_{participant_label}.tsv.gz"
                file_path = self.data_dir / f"part_table_{participant_label}"
                if file_path_gz.exists() or file_path.exists():
                    existing_files += 1
                else:
                    missing_files += 1
                    self._log(
                        f"File not found for participant: {participant_label}",
                        level=2,
                    )

            self._log(
                f"Data files: {existing_files} found, {missing_files} missing",
                level=1,
            )

        # Log fold distribution
        if "malid_cross_validation_fold_id_when_in_test_set" in metadata.columns:
            fold_counts = metadata[
                "malid_cross_validation_fold_id_when_in_test_set"
            ].value_counts()
            self._log(f"Fold distribution:\n{fold_counts}", level=1)

        self._log("Metadata loaded successfully", level=1)
        return metadata

    def iter_fold_specimens(
        self,
        fold_id: int,
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        """
        Iterate over specimens in a fold (memory-efficient).

        Args:
            fold_id: Cross-validation fold ID
            fold_label: "train" (all except fold_id) or "test" (only fold_id)
            preprocessing_stage: Level of preprocessing to apply

        Yields:
            Tuple of (specimen_label, specimen_sequences, specimen_metadata)
        """
        # Get specimens for this fold
        if fold_label == "train":
            fold_specimens = self.metadata[
                self.metadata["malid_cross_validation_fold_id_when_in_test_set"]
                != fold_id
            ]
        elif fold_label == "test":
            fold_specimens = self.metadata[
                self.metadata["malid_cross_validation_fold_id_when_in_test_set"]
                == fold_id
            ]
        else:
            raise ValueError(
                f"fold_label must be 'train' or 'test', got: {fold_label}"
            )

        self._log(
            f"Loading fold {fold_id} {fold_label}: {len(fold_specimens)} specimens",
            level=1,
        )

        # Group by participant (files are per participant)
        for participant_label in fold_specimens["participant_label"].unique():
            # Load participant data (may contain multiple specimens)
            try:
                participant_df = self.load_participant_data(
                    participant_label, preprocessing_stage
                )
            except Exception as e:
                logger.error(
                    f"Error loading participant {participant_label}: {e}",
                    exc_info=self.verbose >= 2,
                )
                continue

            if participant_df.empty:
                continue

            # Get specimens for this participant in this fold
            participant_specimens = fold_specimens[
                fold_specimens["participant_label"] == participant_label
            ]

            # Yield each specimen separately
            for _, specimen_row in participant_specimens.iterrows():
                specimen_label = specimen_row["specimen_label"]
                specimen_df = participant_df[
                    participant_df["repertoire_id"] == specimen_label
                ]

                if not specimen_df.empty:
                    yield specimen_label, specimen_df, specimen_row

    def load_participant_data(
        self,
        participant_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> pd.DataFrame:
        """
        Load data for one participant.

        Handles:
            - Automatic .tsv.gz decompression (via pandas)
            - Falls back to uncompressed files
            - Applies requested preprocessing stage

        Args:
            participant_label: Participant identifier
            preprocessing_stage: Level of preprocessing to apply

        Returns:
            DataFrame with sequence-level data (may contain multiple specimens)
        """
        # For RAW stage, always load from original file
        if preprocessing_stage == PreprocessingStage.RAW:
            # Try .tsv.gz first, fall back to uncompressed
            file_path = self.data_dir / f"part_table_{participant_label}.tsv.gz"
            if not file_path.exists():
                file_path = self.data_dir / f"part_table_{participant_label}"
                if not file_path.exists():
                    self._log(
                        f"File not found for participant: {participant_label}", level=1
                    )
                    return pd.DataFrame()

            self._log(f"Loading RAW participant: {participant_label}", level=2)

            # Load file (pandas auto-detects .gz compression)
            try:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
                return self._normalize_boolean_cols(df)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return pd.DataFrame()

        # For CLEAN or DOWNSAMPLED: try participant cache first
        cached_result = self.load_cached_participant(participant_label)
        if cached_result is not None:
            df, etl_stats = cached_result
            self._log(f"Loaded participant {participant_label} from cache", level=2)
        else:
            # Cache miss: load raw file and preprocess
            self._log(f"Cache miss - preprocessing participant: {participant_label}", level=2)

            # Try .tsv.gz first, fall back to uncompressed
            file_path = self.data_dir / f"part_table_{participant_label}.tsv.gz"
            if not file_path.exists():
                file_path = self.data_dir / f"part_table_{participant_label}"
                if not file_path.exists():
                    self._log(
                        f"File not found for participant: {participant_label}", level=1
                    )
                    return pd.DataFrame()

            # Load file (pandas auto-detects .gz compression)
            try:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return pd.DataFrame()

            # Normalize boolean columns (uppercase t/f → T/F) before preprocessing
            df = self._normalize_boolean_cols(df)

            # Stage 1: Clean
            df, etl_stats = self.preprocess_clean(df, participant_label)

            # Cache the cleaned data with stats
            if not df.empty:
                self.cache_participant(
                    participant_label,
                    df,
                    preprocessing_stats=etl_stats,
                    update_metadata=True
                )

        if preprocessing_stage == PreprocessingStage.CLEAN:
            return df

        # Stage 2: Downsample (per specimen)
        if "repertoire_id" not in df.columns:
            logger.warning(
                f"No repertoire_id column for participant {participant_label}"
            )
            return pd.DataFrame()

        processed_specimens = []
        for specimen_label, specimen_df in df.groupby("repertoire_id"):
            sampled_df, sample_stats = self.preprocess_downsample(
                specimen_df.copy(), specimen_label
            )

            # Get fold_id from metadata
            fold_id = None
            if specimen_label in self.metadata["specimen_label"].values:
                fold_id = self.metadata.loc[
                    self.metadata["specimen_label"] == specimen_label,
                    "malid_cross_validation_fold_id_when_in_test_set",
                ].iloc[0]

            # Accumulate stats
            self._preprocessing_stats.append(
                {
                    "participant_label": participant_label,
                    "specimen_label": specimen_label,
                    "fold_id": fold_id,
                    **etl_stats,
                    **sample_stats,
                }
            )

            if not sampled_df.empty:
                processed_specimens.append(sampled_df)

        if not processed_specimens:
            return pd.DataFrame()

        return pd.concat(processed_specimens, ignore_index=True)

    def preprocess_clean(
        self,
        df: pd.DataFrame,
        participant_label: str,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Stage 1: Cleaning and validation.

        Steps:
            1.  Filter productive sequences
            2.  Filter v_score > 80 (TCR) or > 200 (BCR)
            3.  Clean IgBLAST sequences: strip spaces and uppercase (cdr3_aa/fwr4_aa/fwr3_aa)
            4.  Drop sequences with non-standard amino acid characters in CDR3
            5.  Deduplicate identical sequences, sum num_reads
            6.  Fix gene names
            7.  Remove sequences with V genes not in reference (e.g. TRBV25/OR9-2*01)
            8.  Extract FR/CDR regions from reference (if gene_reference provided)
            9.  Create v_gene/j_gene (no allele) + v_gene_w_allele/j_gene_w_allele
            10. Drop sequences with missing V/J/CDR
            11. Add isotype_supergroup = "TCRB"

        Returns:
            Tuple of (cleaned_df, stats)
        """
        stats = {}
        original_count = len(df)
        stats["original_count"] = original_count

        # Step 1: Filter productive
        if "productive" in df.columns:
            df = df[df["productive"] == "T"].copy()
            stats["productive_filter"] = original_count - len(df)
            self._log(
                f"After productive filter: {len(df)}/{original_count} sequences",
                level=2,
            )

        # Step 2: Filter v_score
        v_threshold = self.V_SCORE_THRESHOLD[self.gene_locus]
        if "v_score" in df.columns:
            before = len(df)
            df = df[df["v_score"] > v_threshold].copy()
            stats["v_score_filter"] = before - len(df)
            self._log(
                f"After v_score>{v_threshold} filter: {len(df)}/{before} sequences",
                level=2,
            )

        # Step 3: Clean IgBLAST sequences - strip spaces and uppercase.
        # Only process columns IgBLAST actually fills for TCR data (cdr3/fwr4/fwr3).
        # cdr_fr_cols is the full list of AA region columns, used later in step 8.
        # NOTE: Only spaces are stripped here. Dots, dashes, and asterisks are
        # intentionally left intact — non-standard characters in CDR3 will cause
        # rows to be dropped in step 4 below.
        cdr_fr_cols = [
            "fwr1_aa",
            "cdr1_aa",
            "fwr2_aa",
            "cdr2_aa",
            "fwr3_aa",
            "cdr3_aa",
            "fwr4_aa",
        ]
        igblast_cols = ["cdr3_aa", "fwr4_aa", "fwr3_aa"]
        seq_cleaning_changes: Dict[str, int] = {}
        for col in igblast_cols:
            if col in df.columns:
                original_strings = df[col].astype(str)
                # Strip spaces only — preserve non-standard chars for step 4 filter
                df[col] = (
                    original_strings
                    .str.replace(" ", "", regex=False)
                    .str.upper()
                )
                # Replace empty strings and "NAN" with actual NaN
                df[col] = df[col].replace({"": np.nan, "NAN": np.nan})

                # Count rows where the sequence actually changed
                cleaned_strings = df[col].fillna("__NAN__")
                original_normalized = original_strings.replace(
                    "", "__NAN__"
                ).fillna("__NAN__")
                n_changed = (original_normalized != cleaned_strings).sum()
                if n_changed > 0:
                    seq_cleaning_changes[col] = int(n_changed)

        stats["seq_cleaning_changes"] = seq_cleaning_changes

        # Step 4: Drop sequences with non-standard amino acid characters in CDR3.
        # Characters outside the 20 standard AAs (e.g. *, X, ., -) indicate
        # ambiguous or non-productive sequences. Matches clean_tcr_data.py behavior.
        _cdr3_check_col = "cdr3_aa" if "cdr3_aa" in df.columns else None

        non_standard_aa_chars: Dict[str, int] = {}
        non_standard_aa_rows_removed = 0
        if _cdr3_check_col:
            before = len(df)
            bad_rows = []
            for idx, seq in df[_cdr3_check_col].items():
                if pd.isna(seq) or seq == "":
                    continue
                bad_chars = set(str(seq).upper()) - self.VALID_AMINO_ACIDS
                if bad_chars:
                    bad_rows.append(idx)
                    for ch in bad_chars:
                        non_standard_aa_chars[ch] = non_standard_aa_chars.get(ch, 0) + 1
            non_standard_aa_rows_removed = len(bad_rows)
            if bad_rows:
                df = df.drop(index=bad_rows).copy()
            self._log(
                f"After non-standard AA filter: {len(df)}/{before} sequences",
                level=2,
            )
            if non_standard_aa_chars:
                for ch, count in sorted(
                    non_standard_aa_chars.items(), key=lambda x: -x[1]
                ):
                    self._log(
                        f"  Non-standard char '{ch}': {count} rows dropped", level=2
                    )

        stats["non_standard_aa_chars_found"] = non_standard_aa_chars
        stats["non_standard_aa_rows_removed"] = non_standard_aa_rows_removed

        # Step 5: Deduplicate
        if "sequence" in df.columns and "replicate_label" in df.columns:
            stats["sequences_before_dedup"] = len(df)

            # Group by identical sequences
            dedup_cols = ["sequence", "replicate_label"]
            if "extracted_isotype" in df.columns:
                dedup_cols.append("extracted_isotype")

            # Sum num_reads for duplicates
            if "num_reads" not in df.columns:
                df["num_reads"] = 1

            # Aggregate: sum num_reads, keep first of other columns
            df = df.groupby(dedup_cols, as_index=False, dropna=False).agg(
                {
                    col: "first" if col not in ["num_reads"] else "sum"
                    for col in df.columns
                    if col not in dedup_cols
                }
            )

            stats["sequences_after_dedup"] = len(df)
            self._log(
                f"After deduplication: {len(df)}/{stats['sequences_before_dedup']} sequences",
                level=2,
            )

        # Step 6: Fix gene names
        gene_name_fixes = {}
        gene_name_fixes_total = 0

        if "v_call" in df.columns:
            # 6a. Gene allele fixes (exact match)
            for old_name, new_name in self.GENE_ALLELE_FIXES.items():
                mask = df["v_call"] == old_name
                n_fixed = mask.sum()
                if n_fixed > 0:
                    df.loc[mask, "v_call"] = new_name
                    gene_name_fixes[f"{old_name}->{new_name}"] = n_fixed
                    gene_name_fixes_total += n_fixed

            # 6b. Gene fixes (prefix match with wildcard allele replacement)
            # For each gene fix, find all alleles and replace with target+*01
            for old_gene, new_gene in self.GENE_FIXES.items():
                prefix = f"{old_gene}*"
                mask = df["v_call"].str.startswith(prefix, na=False)

                if mask.any():
                    # Track each unique allele variant and its count BEFORE replacement
                    allele_counts = df.loc[mask, "v_call"].value_counts()

                    # Replace all with new_gene + "*01"
                    new_name = f"{new_gene}*01"
                    df.loc[mask, "v_call"] = new_name

                    # Log each specific allele replacement
                    for old_allele, count in allele_counts.items():
                        fix_key = f"{old_allele}->{new_name}"
                        gene_name_fixes[fix_key] = count
                        gene_name_fixes_total += count

        stats["gene_name_fixes"] = gene_name_fixes_total
        stats["gene_name_fixes_detail"] = gene_name_fixes
        if gene_name_fixes_total > 0:
            self._log(f"Fixed {gene_name_fixes_total} gene names", level=2)
            for fix, count in gene_name_fixes.items():
                self._log(f"  {fix}: {count} sequences", level=2)

        # Step 7: Remove sequences with V genes that have no reference data.
        # These are pseudogenes or genes absent from the reference table — their
        # CDR1/CDR2 sequences cannot be filled and they cannot be used downstream.
        # Matches clean_tcr_data.py step 8 and original Mal-ID etl.py behavior.
        genes_removed: Dict[str, int] = {}
        if "v_call" in df.columns and self.GENES_TO_REMOVE:
            mask = df["v_call"].isin(self.GENES_TO_REMOVE)
            if mask.any():
                for gene, count in df.loc[mask, "v_call"].value_counts().items():
                    genes_removed[str(gene)] = int(count)
                df = df[~mask].copy()
                total_removed = sum(genes_removed.values())
                self._log(
                    f"Removed {total_removed} sequences with no-reference V genes",
                    level=2,
                )
                for gene, count in genes_removed.items():
                    self._log(f"  {gene}: {count} sequences", level=2)

        stats["genes_removed"] = genes_removed
        stats["total_genes_removed"] = sum(genes_removed.values())

        # Step 8: Extract FR/CDR regions from reference (always overwrites raw values)
        stats["genes_missing_from_reference"] = {}
        stats["n_genes_missing_from_reference"] = 0

        if self.gene_reference_path and Path(self.gene_reference_path).exists():
            reference = self._load_gene_reference()
            if reference is not None:
                df, missing_genes = self._extract_fr_cdr_sequences(df, reference)
                stats["genes_missing_from_reference"] = missing_genes
                stats["n_genes_missing_from_reference"] = len(missing_genes)

                if missing_genes:
                    total_missing_rows = sum(missing_genes.values())
                    logger.warning(
                        f"{participant_label}: {len(missing_genes)} V gene(s) missing from "
                        f"reference table ({total_missing_rows} rows affected). "
                        f"FR/CDR sequences will be NaN for these rows."
                    )
                    for gene, count in sorted(missing_genes.items(), key=lambda x: -x[1]):
                        logger.warning(f"  {gene}: {count} rows")

                # Clean the reference-sourced sequences in-place.
                # Always overwrite: reference values replace whatever was set in step 3.
                # (Rows where the gene is missing from reference will have NaN,
                #  so those rows keep their step-3 cleaned value unchanged.)
                for col in cdr_fr_cols:
                    if col not in df.columns:
                        continue

                    # Overwrite where the reference has a value (col not NaN)
                    needs_update = df[col].notna()
                    if needs_update.any():
                        df.loc[needs_update, col] = (
                            df.loc[needs_update, col]
                            .astype(str)
                            .str.replace(".", "", regex=False)
                            .str.replace("-", "", regex=False)
                            .str.replace(" ", "", regex=False)
                            .str.replace("*", "", regex=False)
                            .str.upper()
                            .replace(["", "NAN"], np.nan)
                        )

        # Step 9: Create gene columns (with and without alleles)
        if "v_call" in df.columns:
            df["v_gene_w_allele"] = df["v_call"]
            # Remove allele (everything after *)
            df["v_gene"] = df["v_call"].str.split("*").str[0]

        if "j_call" in df.columns:
            df["j_gene_w_allele"] = df["j_call"]
            df["j_gene"] = df["j_call"].str.split("*").str[0]

        # Step 10: Drop sequences with missing critical fields
        required_cols = ["v_gene", "j_gene"]
        if "cdr3_aa" in df.columns:
            required_cols.append("cdr3_aa")

        before = len(df)
        missing_by_field = {}
        for col in required_cols:
            if col in df.columns:
                # Count missing before dropping
                n_missing = (~df[col].notna() | (df[col] == "")).sum()
                if n_missing > 0:
                    missing_by_field[col] = n_missing
                # Drop rows with missing values
                df = df[df[col].notna() & (df[col] != "")].copy()

        stats["missing_fields"] = before - len(df)
        stats["missing_fields_detail"] = missing_by_field
        if stats["missing_fields"] > 0:
            self._log(
                f"Dropped {stats['missing_fields']} sequences with missing critical fields",
                level=2,
            )
            for field, count in missing_by_field.items():
                self._log(f"  {field}: {count} sequences", level=2)

        # Step 11: Add isotype_supergroup
        # For TCR, always "TCRB"
        if self.gene_locus == "TCR":
            df["isotype_supergroup"] = "TCRB"
        # TODO: BCR needs proper isotype mapping

        # Compute CDR3 length from cleaned column
        cdr3_col = "cdr3_aa" if "cdr3_aa" in df.columns else None

        if cdr3_col:
            df["cdr3_aa_sequence_trim_len"] = df[cdr3_col].str.len()

        # Map clone ID column
        if "clone_id" in df.columns:
            df["igh_or_tcrb_clone_id"] = df["clone_id"]

        stats["after_clean"] = len(df)
        stats["total_dropped"] = original_count - len(df)

        return df, stats

    def preprocess_downsample(
        self,
        df: pd.DataFrame,
        specimen_label: str,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Stage 2: Downsampling and filtering (per specimen).

        Steps:
            1. Filter CDR3 length >= 8
            2. Filter to isotype_supergroup == "TCRB"
            3. Check >= 500 clones (drop specimen if fails)
            4. Check >= 1000 sequences (drop specimen if fails)
            5. Downsample: 1 seq per (specimen, amplification, clone, isotype)

        Returns:
            Tuple of (downsampled_df, stats)
            If specimen fails: (empty_df, stats with kept=False)
        """
        stats = {}
        stats["sequences_before_downsample"] = len(df)

        # Step 1: Filter CDR3 length
        if "cdr3_aa_sequence_trim_len" in df.columns:
            before = len(df)
            df = df[df["cdr3_aa_sequence_trim_len"] >= 8].copy()
            stats["after_cdr3_filter"] = len(df)
            self._log(
                f"{specimen_label}: After CDR3 length filter: {len(df)}/{before}",
                level=2,
            )
        else:
            stats["after_cdr3_filter"] = len(df)

        # Step 2: Filter to correct isotype
        if "isotype_supergroup" in df.columns:
            before = len(df)
            if self.gene_locus == "TCR":
                df = df[df["isotype_supergroup"] == "TCRB"].copy()
            # TODO: BCR filtering
            stats["after_isotype_filter"] = len(df)
            self._log(
                f"{specimen_label}: After isotype filter: {len(df)}/{before}",
                level=2,
            )
        else:
            stats["after_isotype_filter"] = len(df)

        # Count clones
        if "igh_or_tcrb_clone_id" in df.columns:
            n_clones = df["igh_or_tcrb_clone_id"].nunique()
        else:
            n_clones = 0

        stats["n_clones"] = n_clones

        # Step 3: Check clone threshold
        min_clones = self.MIN_CLONES.get("TCRB", 500)
        stats["meets_clone_threshold"] = n_clones >= min_clones

        if not stats["meets_clone_threshold"]:
            stats["kept"] = False
            stats["drop_reason"] = f"insufficient_clones ({n_clones} < {min_clones})"
            stats["after_downsample"] = 0
            self._log(
                f"{specimen_label}: DROPPED - insufficient clones ({n_clones} < {min_clones})",
                level=1,
            )
            return pd.DataFrame(), stats

        # Step 4: Check sequence threshold
        n_sequences = len(df)
        stats["meets_sequence_threshold"] = n_sequences >= self.MIN_SEQUENCES

        if not stats["meets_sequence_threshold"]:
            stats["kept"] = False
            stats["drop_reason"] = (
                f"insufficient_sequences ({n_sequences} < {self.MIN_SEQUENCES})"
            )
            stats["after_downsample"] = 0
            self._log(
                f"{specimen_label}: DROPPED - insufficient sequences ({n_sequences} < {self.MIN_SEQUENCES})",
                level=1,
            )
            return pd.DataFrame(), stats

        # Step 5: Downsample to 1 sequence per clone per amplification
        group_cols = ["repertoire_id", "igh_or_tcrb_clone_id", "isotype_supergroup"]
        if "amplification_label" in df.columns:
            group_cols.insert(1, "amplification_label")

        # Check all group columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"{specimen_label}: Missing columns for grouping: {missing_cols}"
            )
            stats["kept"] = False
            stats["drop_reason"] = f"missing_columns: {missing_cols}"
            stats["after_downsample"] = 0
            return pd.DataFrame(), stats

        # Group and select sequence with max num_reads
        if "num_reads" not in df.columns:
            df["num_reads"] = 1

        # Compute clone statistics before downsampling
        clone_stats = (
            df.groupby(group_cols)
            .agg(
                num_clone_members=("num_reads", "count"),
                total_clone_num_reads=("num_reads", "sum"),
                max_reads_idx=("num_reads", "idxmax"),
            )
            .reset_index()
        )

        # Select one sequence per clone (the one with max num_reads)
        downsampled_df = df.loc[clone_stats["max_reads_idx"]].copy()

        # Add clone statistics
        downsampled_df = downsampled_df.merge(
            clone_stats[group_cols + ["num_clone_members", "total_clone_num_reads"]],
            on=group_cols,
            how="left",
        )

        stats["after_downsample"] = len(downsampled_df)
        stats["kept"] = True
        stats["drop_reason"] = None

        self._log(
            f"{specimen_label}: KEPT - {len(downsampled_df)} sequences ({n_clones} clones)",
            level=1,
        )

        return downsampled_df, stats

    def _normalize_boolean_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize AIRR boolean columns to uppercase.

        AIRR files may store boolean values as 't'/'f' (lowercase) or 'T'/'F'.
        This normalizes all values to uppercase for consistent downstream comparisons.

        Applied immediately after loading from file. Loading from the participant
        cache skips this step since cached data has already been normalized.
        """
        for col in self.AIRR_BOOLEAN_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()
        return df

    def _load_gene_reference(self) -> Optional[pd.DataFrame]:
        """Lazy load gene reference table."""
        if self._gene_reference is None and self.gene_reference_path:
            try:
                self._gene_reference = pd.read_csv(
                    self.gene_reference_path, sep="\t"
                )
                self._log(
                    f"Loaded gene reference: {len(self._gene_reference)} entries",
                    level=2,
                )
            except Exception as e:
                logger.error(f"Error loading gene reference: {e}")
                return None

        return self._gene_reference

    def _extract_fr_cdr_sequences(
        self,
        df: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Extract FR1, CDR1, FR2, CDR2, FR3 from reference table.

        Joins on v_call (with allele). Always overwrites existing FR/CDR values
        with reference values — raw IgBLAST FR/CDR calls are replaced entirely.
        Rows whose V gene is absent from the reference will have NaN in all FR/CDR
        columns; the caller is responsible for logging/reporting these cases.

        Args:
            df: Sequence dataframe
            reference: Gene reference table

        Returns:
            Tuple of (df_with_fr_cdr, missing_genes)
            - df_with_fr_cdr: DataFrame with FR/CDR columns set from reference
            - missing_genes: Dict mapping V gene name -> row count for genes not
              found in the reference table
        """
        if "v_call" not in df.columns:
            return df, {}

        # Prepare reference for merging — rename AA cols to temp names to avoid
        # clashing with existing data columns during the merge.
        ref_merge = reference[
            ["v_call", "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa"]
        ].copy()
        ref_merge = ref_merge.rename(
            columns={
                "fwr1_aa": "fwr1_aa_ref",
                "cdr1_aa": "cdr1_aa_ref",
                "fwr2_aa": "fwr2_aa_ref",
                "cdr2_aa": "cdr2_aa_ref",
                "fwr3_aa": "fwr3_aa_ref",
            }
        )

        # Detect genes present in data but absent from reference (before merge)
        genes_in_ref = set(ref_merge["v_call"].dropna())
        missing_genes: Dict[str, int] = {}
        for gene, count in df["v_call"].dropna().value_counts().items():
            if gene not in genes_in_ref:
                missing_genes[gene] = int(count)

        # Left join: rows with missing genes will have NaN in all ref columns
        df = df.merge(ref_merge, on="v_call", how="left")

        # Always overwrite FR/CDR columns with reference values
        for seq_type in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3"]:
            col_name = f"{seq_type}_aa"
            ref_col_name = f"{seq_type}_aa_ref"

            if ref_col_name in df.columns:
                # Unconditional overwrite — reference is the authoritative source
                df[col_name] = df[ref_col_name]
                df = df.drop(columns=[ref_col_name])

        self._log(
            f"Replaced FR/CDR sequences from reference "
            f"({len(missing_genes)} V genes missing from reference)",
            level=2,
        )
        return df, missing_genes
