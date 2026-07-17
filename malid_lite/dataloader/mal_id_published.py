"""Data loader for Mal-ID published AIRR format data."""

import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator
import pandas as pd
import numpy as np
import logging

from .base import (
    BaseDataLoader,
    PreprocessingStage,
    FOLD_COL,
    normalize_fold_column,
    normalize_identifier_columns,
)
from ..utils.assign_repertoire_clones import (
    CLONE_ID_COL,
    CLONE_ID_ORIGINAL_COL,
    compute_participant_clone_id,
    resolve_identity_threshold,
)

logger = logging.getLogger(__name__)


def _preprocess_and_cache_participant(loader, participant_label: str) -> None:
    """Worker function for parallel precompute_clone_ids.

    Loads raw data for one participant, runs full preprocess_clean (including
    clone_id computation at step 10.5), and caches the result. If already
    cached, returns immediately.

    Module-level function so it can be pickled by joblib.
    """
    loader.load_participant_data(participant_label, PreprocessingStage.CLEAN)


class MalIDPublishedDataLoader(BaseDataLoader):
    """
    Data loader for Mal-ID published data in AIRR format.

    Features:
    - Loads gzip-compressed AIRR format TSV files (per participant)
    - Implements 2-stage preprocessing (cleaning + downsampling)
    - Memory-efficient iteration over specimens
    - Automatic statistics accumulation
    - Caching support
    - Upfront column validation with clear error messages

    File format:
    - Input: part_table_{participant_label}.tsv.gz (one file per participant)
    - AIRR format (produced by clean_tcr_data_to_airr.py)
    - May contain multiple specimens per file (distinguished by repertoire_id)
    - All column names are AIRR format (e.g. v_call, repertoire_id, cdr3_aa)

    Metadata required columns (validated at load time):
    - ``participant_label``: unique participant identifier
    - ``specimen_label``: unique specimen identifier (must match repertoire_id
      in the sequence files)
    - ``disease``: disease class label (one per participant)
    - ``CV_fold``: CV fold assignment (legacy name
      ``malid_cross_validation_fold_id_when_in_test_set`` is also accepted)

    Metadata optional columns:
    - ``available_gene_loci``: used to filter specimens by gene locus

    Sequence file required columns (validated at preprocess_clean time):
    - ``repertoire_id``: specimen identifier (must match specimen_label in metadata)
    - ``v_call``: V gene call with allele (e.g. "TRBV7-2*01")
    - ``j_call``: J gene call with allele (e.g. "TRBJ2-1*01")
    - ``cdr3_aa``: CDR3 amino acid sequence

    Sequence file conditionally required columns:
    - ``clone_id``: clone identifier (used for downsampling: 1 seq per clone).
      If absent, automatically computed via hierarchical clustering on CDR3
      sequences (step 10.5 of preprocess_clean). When ``force_clone_id=True``,
      a new clone_id is computed even if the column exists (original preserved
      as ``clone_id_original``).
    - ``cdr3`` (nucleotide CDR3): required when clone_id is being computed
      and ``clone_id_use_aa`` is False or unspecified. Not needed when clone_id
      exists in the data or when using AA CDR3 for clone assignment.

    Clone ID parameter handling:
    - Clone_id clustering parameters (``clone_id_use_aa``,
      ``clone_id_identity_threshold``, ``clone_id_linkage_method``) accept None
      to mean "unspecified by the user."
    - **Building cache:** None params resolve to defaults (use_aa=False,
      linkage="single", threshold per locus). Resolved values are stored in
      the participant stats JSON.
    - **Loading from cache:** Only explicitly-provided (non-None) params are
      validated against cached values. Unspecified (None) params are accepted
      as-is. This allows the natural workflow: set params once at cache build
      time, then omit them on subsequent training/embedding runs.
    - ``force_clone_id`` is a build-time action flag, not a clustering
      parameter. It can always be omitted after the cache is built.

    Sequence file quality columns (warn if absent, filtering skipped):
    - ``productive``: productive sequence flag ("T"/"F"). If absent, non-productive
      sequences are kept (may add noise).
    - ``v_score``: V gene alignment score. If absent, quality filtering is skipped.

    Sequence file optional columns (warn if absent, handled gracefully):
    - ``sequence``: full nucleotide sequence (used for deduplication)
    - ``num_reads``: read count (defaults to 1 if absent)
    - ``extracted_isotype``: isotype call (used in deduplication grouping)
    - ``amplification_label``: amplification protocol (used in downsampling grouping)
    - ``replicate_label``: replicate identifier (used with sequence for deduplication)
    - ``stop_codon``, ``vj_in_frame``: boolean flags (normalized but not used for filtering)
    - ``fwr1_aa`` through ``fwr4_aa``, ``cdr1_aa``, ``cdr2_aa``: framework/CDR regions
      (only used when gene_reference_path is provided)

    Specimen identifier mapping:
    - Raw/clean sequence data uses ``repertoire_id`` (AIRR standard column name)
    - Metadata uses ``specimen_label``
    - These hold **identical values** — the data loader matches sequences to metadata
      by ``repertoire_id == specimen_label`` (see iter_fold_specimens, line ~247)
    - Downstream of the data loader (fold caches, models, ensemble), all code uses
      ``specimen_label`` exclusively. The rename from ``repertoire_id`` →
      ``specimen_label`` happens in iter_fold_specimens() before yielding.
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
        data_dir: Optional[Path],
        metadata_path: Optional[Path] = None,
        gene_locus: str = "TCR",
        verbose: int = 1,
        gene_reference_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        # --- Clone ID parameters ---
        force_clone_id: bool = False,
        clone_id_identity_threshold: Optional[float] = None,
        clone_id_linkage_method: Optional[str] = None,
        clone_id_cdr3_nt_col: str = "cdr3",
        clone_id_use_aa: Optional[bool] = None,
    ):
        """
        Initialize Mal-ID published data loader.

        Args:
            data_dir: Path to airr_format_clean/TCR/ directory
                     (contains AIRR format part_table_{participant_label}.tsv.gz files).
                     Can be None for metadata-only use (e.g. --feature-matrices-dir).
            metadata_path: Path to metadata.tsv. Optional if the cache already
                contains a copy (cache_dir/metadata.tsv).
            gene_locus: "TCR" or "BCR" (only TCR fully supported initially)
            verbose: Verbosity level (0=silent, 1=normal, 2=debug)
            gene_reference_path: Path to tcrb_v_gene_cdrs.generated.tsv
                                (Required for FR/CDR extraction)
            cache_dir: Optional directory for caching preprocessed data
            force_clone_id: If True, compute clone_id even when the column already
                exists in the raw data. The original clone_id is preserved as
                clone_id_original. Only takes effect at first cache build — once
                cached, can be omitted on subsequent runs.
            clone_id_identity_threshold: Override the default CDR3 identity
                threshold for clone assignment. When None (default), uses
                per-locus/CDR3-type defaults: TCR-NT=0.95, BCR-NT=0.90,
                TCR-AA=0.90, BCR-AA=0.85. Only needs to be specified at cache
                build time.
            clone_id_linkage_method: Hierarchical clustering linkage method
                ("single", "complete", or "average"). None means unspecified
                (resolves to "single" when building cache). Only needs to be
                specified at cache build time.
            clone_id_cdr3_nt_col: AIRR column name for nucleotide CDR3
                (default "cdr3"). Only used when clone_id_use_aa is False/None.
            clone_id_use_aa: If True, use amino acid CDR3 (cdr3_aa) instead of
                nucleotide for clone assignment. None means unspecified (resolves
                to False when building cache). Only needs to be specified at
                cache build time.

        Clone ID parameter semantics:
            Clone_id clustering parameters (clone_id_use_aa,
            clone_id_identity_threshold, clone_id_linkage_method) use
            None to mean "unspecified by the user." This distinguishes
            between "the user didn't say" and "the user explicitly chose
            a value."

            - **Building cache (first run):** None params are resolved to
              defaults (use_aa=False, linkage="single", threshold per locus).
              The resolved values are stored in the participant stats JSON.
            - **Loading from cache (subsequent runs):** None params are not
              validated — the cached values are accepted as-is. Only
              explicitly-provided (non-None) params are compared against
              the cache. If they conflict, a ValueError is raised.

            This allows the natural workflow: specify clone_id parameters
            once when building the cache, then omit them on all subsequent
            training and embedding commands.
        """
        super().__init__(data_dir, metadata_path, gene_locus, verbose, cache_dir)

        self.gene_reference_path = gene_reference_path
        self._gene_reference = None  # Lazy load

        # --- Clone ID parameter validation and resolution ---
        if clone_id_use_aa is not None and not isinstance(clone_id_use_aa, bool):
            raise ValueError(
                f"clone_id_use_aa must be a bool (True/False), "
                f"got {type(clone_id_use_aa).__name__}: {clone_id_use_aa!r}"
            )

        valid_linkage_methods = ("single", "complete", "average")
        if (clone_id_linkage_method is not None
                and clone_id_linkage_method not in valid_linkage_methods):
            raise ValueError(
                f"clone_id_linkage_method must be one of {valid_linkage_methods}, "
                f"got '{clone_id_linkage_method}'"
            )

        self.force_clone_id = force_clone_id
        self.clone_id_cdr3_nt_col = clone_id_cdr3_nt_col

        # Resolve effective values: None → default
        effective_use_aa = clone_id_use_aa if clone_id_use_aa is not None else False
        effective_linkage = (
            clone_id_linkage_method
            if clone_id_linkage_method is not None
            else "single"
        )

        # Store resolved effective values for use in preprocess_clean
        self.clone_id_linkage_method = effective_linkage
        self.clone_id_use_aa = effective_use_aa

        # Resolve identity threshold using effective use_aa
        resolved_threshold = resolve_identity_threshold(
            gene_locus=self.gene_locus,
            use_aa=effective_use_aa,
            override=clone_id_identity_threshold,
        )

        # _clone_id_params: RESOLVED values for cache storage and computation.
        # Stored in participant stats JSON when clone_id is computed.
        # NOTE: force_clone_id is intentionally excluded. It is a one-time
        # build action ("should I override existing clone_id?"), not a
        # clustering parameter. Once the cache is built, the clone_id values
        # are fixed regardless of whether force was used.
        self._clone_id_params = {
            "clone_id_use_aa": effective_use_aa,
            "clone_id_identity_threshold": resolved_threshold,
            "clone_id_linkage_method": effective_linkage,
            "clone_id_cdr3_nt_col": clone_id_cdr3_nt_col,
        }

        # _clone_id_params_specified: only params the user EXPLICITLY set
        # (non-None). Used for cache validation — only these are compared
        # against cached values. Unspecified (None) params are accepted as-is.
        self._clone_id_params_specified = {}
        if clone_id_use_aa is not None:
            self._clone_id_params_specified["clone_id_use_aa"] = clone_id_use_aa
        if clone_id_identity_threshold is not None:
            self._clone_id_params_specified[
                "clone_id_identity_threshold"
            ] = resolved_threshold
        if clone_id_linkage_method is not None:
            self._clone_id_params_specified[
                "clone_id_linkage_method"
            ] = clone_id_linkage_method

        # --- Upfront cache validation (fail-fast) ---
        # If the user specified clone_id params or force_clone_id AND a cache
        # already exists, validate immediately against the first cached
        # participant rather than waiting until load_cached_participant() is
        # called. This makes parameter mismatches surface at loader
        # construction time, so the entire run fails at the beginning with a
        # clear error.
        if (self._clone_id_params_specified or self.force_clone_id) and self.cache_dir is not None:
            self._validate_clone_id_params_against_cache()

        # Global-once warning flags: these are set to True after the first
        # warning is emitted, so that repeated calls to preprocess_clean()
        # (one per participant) don't spam the same message.
        self._warned_missing_productive = False
        self._warned_missing_v_score = False
        self._warned_missing_sequence = False
        self._warned_missing_num_reads = False
        self._warned_missing_extracted_isotype = False

        if gene_reference_path is None:
            logger.info(
                "gene_reference_path not provided. "
                "FR1-FR3 and CDR1-CDR2 sequences will NOT be extracted. "
                "This is fine — these regions are reserved for future use "
                "and are not required by any current model."
            )
        elif not Path(gene_reference_path).exists():
            logger.warning(
                f"Gene reference file not found: {gene_reference_path}. "
                "FR/CDR extraction will be skipped."
            )

    def _check_clone_id_params_against_stats(
        self, cached_stats: Dict, context_label: str
    ) -> None:
        """Validate clone_id params against a single participant's cached stats.

        Shared validation logic used by both upfront (constructor-time) and
        per-participant (load-time) checks. Validates four conditions:

        1. **Missing clone_id_computed key:** If the stats JSON lacks the
           ``clone_id_computed`` key (e.g., old cache format from before
           clone_id tracking was added), raises an error so the user knows
           to rebuild the cache.
        2. **force_clone_id vs non-computed cache:** If the cache was built
           using pre-existing clone_id (clone_id_computed=False) but
           force_clone_id=True is now set, raises an error.
        3. **Clustering params vs non-computed cache:** If the user specified
           clustering params (use_aa, threshold, linkage) but the cache used
           pre-existing clone_id (never computed), raises an error — the
           params have no effect on the cached data.
        4. **Parameter mismatch:** If the user explicitly specified clone_id
           params that conflict with the cached values, raises an error.
           Only explicitly-set params (in ``_clone_id_params_specified``) are
           compared; unspecified (None) params are accepted as-is.

        Args:
            cached_stats: The participant's stats dict loaded from the JSON.
            context_label: Human-readable label for error messages, e.g.
                ``"existing participant cache"`` (upfront) or
                ``"participant cache for 'P001'"`` (per-participant).

        Raises:
            ValueError: On any of the four conditions above.
        """
        cached_clone_computed = cached_stats.get("clone_id_computed")

        # Check 1: Stats JSON predates clone_id tracking
        if cached_clone_computed is None and (
            self._clone_id_params_specified or self.force_clone_id
        ):
            raise ValueError(
                f"Cannot validate clone_id parameters: {context_label} "
                f"was built before clone_id tracking was added (no "
                f"'clone_id_computed' key in stats). To ensure correct "
                f"clone_id assignments, clear the cache and rebuild:\n"
                f"  python scripts/data/manage_cache.py clear-all "
                f"--cache-dir {self.cache_dir}\n"
                f"  rm -r trained_models/<dataset>/\n"
                f"  Then re-run your pipeline."
            )

        # Check 2: force_clone_id vs cache that used pre-existing clone_id
        if cached_clone_computed is False and self.force_clone_id:
            raise ValueError(
                f"force_clone_id=True but {context_label} was built using "
                f"the pre-existing clone_id column from the data. "
                f"To recompute clone_id, either:\n"
                f"  Option A: Clear everything for this dataset:\n"
                f"    python scripts/data/manage_cache.py clear-all "
                f"--cache-dir {self.cache_dir}\n"
                f"    rm -r trained_models/<dataset>/\n"
                f"    Then re-run your pipeline.\n"
                f"  Option B: Use a different dataset name:\n"
                f"    --dataset-name <new-name>"
            )

        # Check 3: Clone_id clustering params vs cache that used pre-existing
        # clone_id. The cache was built WITHOUT computing clone_id (the data
        # already had a clone_id column), so clustering params like use_aa,
        # threshold, linkage are irrelevant — they were never applied. If the
        # user is now specifying them, they likely intend to recompute clone_id,
        # which requires clearing the cache and rebuilding with --force-clone-id.
        if cached_clone_computed is False and self._clone_id_params_specified:
            raise ValueError(
                f"Clone ID clustering parameters were specified but "
                f"{context_label} was built using the pre-existing clone_id "
                f"column from the data (clone_id was not computed, so "
                f"clustering parameters have no effect).\n"
                f"  You specified:    {self._clone_id_params_specified}\n\n"
                f"  If you want to compute clone_id with these parameters, "
                f"clear the cache and rebuild with --force-clone-id:\n"
                f"    Option A: Clear everything for this dataset and "
                f"rebuild:\n"
                f"      python scripts/data/manage_cache.py clear-all "
                f"--cache-dir {self.cache_dir}\n"
                f"      rm -r trained_models/<dataset>/\n"
                f"      Then re-run with --force-clone-id plus your "
                f"clustering flags.\n"
                f"    Option B: Use a different dataset name:\n"
                f"      --dataset-name <new-name>"
            )

        # Check 4: Explicitly-specified clone_id params vs cached values
        if cached_clone_computed is True and self._clone_id_params_specified:
            # Old caches may still have force_clone_id stored — drop it
            cached_params = {
                k: v
                for k, v in cached_stats.get("clone_id_params", {}).items()
                if k != "force_clone_id"
            }

            mismatches = {}
            for param, current_val in self._clone_id_params_specified.items():
                if param in cached_params and cached_params[param] != current_val:
                    mismatches[param] = {
                        "cached": cached_params[param],
                        "specified": current_val,
                    }

            if mismatches:
                raise ValueError(
                    f"Clone ID parameters conflict with {context_label}.\n"
                    f"  Cached params:    {cached_params}\n"
                    f"  You specified:    {self._clone_id_params_specified}\n"
                    f"  Mismatches:       {mismatches}\n\n"
                    f"  Clone ID parameters are locked once the cache is built.\n"
                    f"  Changing them invalidates ALL downstream artifacts:\n"
                    f"    - Participant cache, fold cache, and embeddings\n"
                    f"    - Trained model artifacts in trained_models/ (if any "
                    f"were trained\n"
                    f"      with the old clone assignments, they will produce "
                    f"unreliable\n"
                    f"      predictions and should be deleted)\n\n"
                    f"  To rebuild with new parameters, either:\n"
                    f"    Option A: Clear everything for this dataset and "
                    f"rebuild:\n"
                    f"      python scripts/data/manage_cache.py clear-all "
                    f"--cache-dir {self.cache_dir}\n"
                    f"      rm -r trained_models/<dataset>/\n"
                    f"      Then re-run your pipeline.\n"
                    f"    Option B: Use a different dataset name (preserves "
                    f"existing artifacts):\n"
                    f"      --dataset-name <new-name>\n"
                    f"      This creates a separate cache and model artifact "
                    f"directory."
                )

    def _validate_clone_id_params_against_cache(self) -> None:
        """Upfront fail-fast validation of clone_id params against the cache.

        Called from __init__ when the user explicitly specified clone_id
        parameters or force_clone_id AND a cache directory exists. Reads the
        first available participant stats JSON and validates against it.
        This makes mismatches surface immediately at loader construction time,
        rather than waiting until load_cached_participant() processes the
        first participant (which may happen much later in the pipeline).

        No-op if no cached participant stats files exist yet (first run).
        """
        import json

        participants_dir = self.cache_dir / "participants"
        if not participants_dir.exists():
            return

        stats_files = sorted(participants_dir.glob("*_stats.json"))
        if not stats_files:
            return

        try:
            with open(stats_files[0]) as f:
                cached_stats = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        self._check_clone_id_params_against_stats(
            cached_stats, f"existing participant cache (checked: {stats_files[0].stem})"
        )

    def load_metadata(self) -> pd.DataFrame:
        """
        Load, validate, and filter metadata.

        Processing steps:
        1. Load raw metadata TSV
        2. Validate required columns exist and have no NaN values
        3. Validate one-disease-per-participant constraint
        4. Filter to participants with raw data files on disk (first run only;
           on subsequent runs the filtered version is loaded from cache).
           The filtered metadata is saved to cache as metadata_processed.tsv,
           which is gene-locus-agnostic so both TCR and BCR can use it.
        5. Filter to requested gene locus (TCR/BCR) — applied every time

        Participants without raw data files are excluded from all downstream
        processing (splits, training, evaluation).

        Raises:
            ValueError: If required columns are missing or contain NaN values.
            ValueError: If any participant has multiple disease labels.

        Returns:
            DataFrame with metadata, filtered to participants with data
        """
        self._log("Loading metadata...", level=1)

        # Load metadata
        metadata = pd.read_csv(self.metadata_path, sep="\t")

        # Normalize legacy fold column name → "CV_fold"
        metadata = normalize_fold_column(metadata)

        # Normalize identifier columns (int64 → str) so that numeric-looking
        # labels (e.g. 310101) are consistent with sequence data identifiers
        metadata = normalize_identifier_columns(metadata)

        # Log statistics
        self._log(f"Total samples in metadata: {len(metadata)}", level=1)

        # --- Validate required metadata columns ---
        # These columns are used throughout the pipeline (specimen matching,
        # disease labels) and cannot be recovered.
        #
        # CV_fold is NOT required here: it is only needed for cross-validation
        # (fold_label="train"/"test"), not for train-all (fold_label="all").
        # Validating it here would run at lazy metadata-load time — before the
        # caller has chosen CV vs train-all — so it is validated at the point of
        # use instead (iter_fold_specimens / _generate_splits raise a clear error
        # if a CV fold is requested but CV_fold is absent). When the column IS
        # present, we still NaN-check it below so partial fold assignments are
        # caught early.
        required_metadata_cols = [
            "participant_label",
            "specimen_label",
            "disease",
        ]
        missing_metadata_cols = [
            col for col in required_metadata_cols if col not in metadata.columns
        ]
        if missing_metadata_cols:
            raise ValueError(
                f"Metadata file is missing required column(s): {missing_metadata_cols}. "
                f"Available columns: {list(metadata.columns)}. "
                f"See PIPELINE_GUIDE.md section 4.1 for the required metadata format."
            )

        # Check for NaN values in required columns (plus CV_fold if present).
        cols_to_nan_check = required_metadata_cols + (
            [FOLD_COL] if FOLD_COL in metadata.columns else []
        )
        for col in cols_to_nan_check:
            n_nan = metadata[col].isna().sum()
            if n_nan > 0:
                nan_examples = metadata.loc[metadata[col].isna()].index[:5].tolist()
                raise ValueError(
                    f"Metadata column '{col}' has {n_nan} NaN value(s) "
                    f"(row indices: {nan_examples}{'...' if n_nan > 5 else ''}). "
                    f"All rows must have non-null values for required columns."
                )

        # Normalize CV_fold dtype. Fold ids may be stored as strings ("0","1",...) in
        # some metadata files, but get_fold_data compares `CV_fold == fold_id` against an
        # int — on a string column that comparison is element-wise False, silently
        # yielding an empty test set (and the whole dataset as "train"). Coerce to int so
        # the comparison is dtype-correct; a non-integer value fails loudly. (NaNs were
        # already rejected above, so astype(int) is safe.)
        if FOLD_COL in metadata.columns:
            try:
                metadata[FOLD_COL] = metadata[FOLD_COL].astype(int)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Metadata column '{FOLD_COL}' has non-integer fold value(s) that "
                    f"cannot be coerced to int: {e}. CV fold ids must be integers."
                )

        if FOLD_COL not in metadata.columns:
            self._log(
                f"No '{FOLD_COL}' column in metadata — cross-validation is "
                f"unavailable; only train-all (fold_label='all') can be used.",
                level=1,
            )

        # Enforce one-disease-per-participant constraint.
        # Models fundamentally require this: stratified CV splits are by participant disease,
        # binary pair filtering is participant-level, and Model 2's Fisher test counts
        # participants per disease. Participants with multiple disease labels cannot be
        # handled correctly and indicate a metadata problem.
        multi_disease = metadata.groupby("participant_label")["disease"].nunique()
        bad_participants = multi_disease[multi_disease > 1].index.tolist()
        if bad_participants:
            raise ValueError(
                f"Participants with multiple disease labels found — models require exactly "
                f"one disease per participant: {bad_participants}"
            )

        # Filter metadata to participants with raw data files on disk.
        # Already-processed metadata (loaded from metadata_processed.tsv) was
        # filtered when it was first created, so we can skip the scan.
        # When data_dir is None (metadata-only mode), skip the scan entirely —
        # all participants in the metadata file are retained.
        # This filter is applied BEFORE the gene locus filter so the processed
        # cache file is locus-agnostic (both TCR and BCR can use the same file).
        if self._metadata_needs_filtering and self.data_dir is not None:
            all_participants = metadata["participant_label"].unique()
            n_total = len(all_participants)

            # Scan raw data files for every participant in metadata
            has_raw_data = []
            missing_labels = []
            for participant_label in all_participants:
                file_path_gz = self.data_dir / f"part_table_{participant_label}.tsv.gz"
                file_path = self.data_dir / f"part_table_{participant_label}"
                if file_path_gz.exists() or file_path.exists():
                    has_raw_data.append(participant_label)
                else:
                    missing_labels.append(participant_label)

            n_found = len(has_raw_data)
            n_missing = len(missing_labels)

            if n_missing > 0:
                # Always log missing participant count (level=0), regardless of verbose
                self._log(
                    f"Metadata has {n_total} participants but {n_missing} have no raw "
                    f"data files in {self.data_dir} — these {n_missing} participants "
                    f"will be excluded from all downstream processing.",
                    level=0,
                )
                # Log individual missing labels at debug level
                for label in missing_labels:
                    self._log(
                        f"  No raw data file for participant: {label}",
                        level=2,
                    )

                # Filter metadata to only participants with raw data
                has_raw_set = set(has_raw_data)
                metadata = metadata[
                    metadata["participant_label"].isin(has_raw_set)
                ].copy()
                self._log(
                    f"Metadata filtered: {n_found} participants with raw data retained "
                    f"(out of {n_total} in metadata file).",
                    level=0,
                )
            else:
                self._log(
                    f"All {n_total} metadata participants have raw data files.",
                    level=1,
                )

            if len(metadata) == 0:
                raise ValueError(
                    f"No participants have raw data files in {self.data_dir}. "
                    f"All {n_total} participants from metadata were filtered out. "
                    f"Check that data_dir points to the correct directory."
                )

            # Record filtering info for downstream summaries
            self.metadata_filter_info = {
                "n_original": n_total,
                "n_filtered_out": n_missing,
                "n_retained": n_found,
            }

            # Save processed metadata to cache (locus-agnostic) and invalidate
            # old splits (which may have been generated from unfiltered metadata)
            if self.cache_dir is not None:
                self._save_metadata_to_cache(metadata)
                if n_missing > 0:
                    splits_dir = self.cache_dir / "splits"
                    if splits_dir.exists():
                        shutil.rmtree(splits_dir)
                        self._log(
                            "Cleared cached splits (metadata was filtered — splits "
                            "will be regenerated from filtered participants only).",
                            level=0,
                        )
        else:
            self._log(
                "Loaded pre-processed metadata (already filtered to participants "
                "with raw data).",
                level=1,
            )
            self.metadata_filter_info = None

        # Filter to gene locus (applied every time, even on processed cache,
        # since the processed file is locus-agnostic)
        if "available_gene_loci" in metadata.columns:
            has_locus = metadata["available_gene_loci"].str.contains(
                self.gene_locus, na=False
            )
            n_with_locus = has_locus.sum()
            self._log(
                f"Samples with {self.gene_locus} data: {n_with_locus}", level=1
            )
            metadata = metadata[has_locus].copy()

        # Log fold distribution
        if FOLD_COL in metadata.columns:
            fold_counts = metadata[FOLD_COL].value_counts()
            self._log(f"Fold distribution:\n{fold_counts}", level=1)

        return metadata

    def iter_fold_specimens(
        self,
        fold_id: Optional[int],
        fold_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        """
        Iterate over specimens (memory-efficient).

        Sequences are matched to metadata via repertoire_id (AIRR column in
        raw/clean data) == specimen_label (metadata column). For DOWNSAMPLED
        stage, load_participant_data already renames to specimen_label. For
        RAW/CLEAN stages, the rename happens here before yielding. Either way,
        yielded DataFrames always have a ``specimen_label`` column.

        Args:
            fold_id: Cross-validation fold ID (None when fold_label == "all").
            fold_label: "train" (all except fold_id), "test" (only fold_id), or
                "all" (every specimen, no CV_fold filtering — for train-all).
            preprocessing_stage: Level of preprocessing to apply

        Yields:
            Tuple of (specimen_label, specimen_sequences, specimen_metadata)
            where specimen_sequences always has a ``specimen_label`` column.

        Raises:
            ValueError: if a CV fold is requested ("train"/"test") but the
                metadata has no CV_fold column.
            RuntimeError: for fold_label="all" only, if any expected participant
                fails to load (so train-all never silently trains on a shrunken
                dataset). Participants with no data after QC are reported, not
                raised.
        """
        # --- Select specimens for this fold / the whole dataset ---
        if fold_label == "all":
            fold_specimens = self.metadata
        elif fold_label in ("train", "test"):
            if fold_id is None:
                # Guard: a None fold_id here would make `metadata[FOLD_COL] != None`
                # element-wise True and silently return the whole dataset as "train"
                # (and an empty "test"). Require an explicit fold id for CV.
                raise ValueError(
                    f"fold_label={fold_label!r} requires an integer fold_id, got None. "
                    f"To load the whole dataset with no fold, use fold_label='all' "
                    f"(or get_all_data())."
                )
            if FOLD_COL not in self.metadata.columns:
                raise ValueError(
                    f"Cannot load fold_label={fold_label!r}: metadata has no "
                    f"'{FOLD_COL}' column, so cross-validation is unavailable. "
                    f"Use fold_label='all' (train-all) instead, or supply metadata "
                    f"with fold assignments."
                )
            if fold_label == "train":
                fold_specimens = self.metadata[self.metadata[FOLD_COL] != fold_id]
            else:
                fold_specimens = self.metadata[self.metadata[FOLD_COL] == fold_id]
        else:
            raise ValueError(
                f"fold_label must be 'train', 'test', or 'all', got: {fold_label}"
            )

        self._log(
            f"Loading {self._fold_label_desc(fold_id, fold_label)}: "
            f"{len(fold_specimens)} specimens",
            level=1,
        )

        # For train-all ("all"), enforce completeness. A participant that yields no
        # data can be one of two very different things, and we must NOT conflate them:
        #   (a) a genuine LOAD FAILURE — load_participant_data raised, OR returned an
        #       empty frame WITHOUT preprocessing having run (missing/corrupt cache or raw
        #       file, missing repertoire_id on a non-empty clean frame). This is the
        #       silent dataset-shrink the check exists to prevent → fail loud.
        #   (b) a legitimate QC-DROP — the participant loaded and preprocessing ran, but
        #       every sequence/specimen was removed by QC. This covers BOTH the
        #       downsampling thresholds (per-specimen stats recorded in the downsample
        #       loop) AND a CLEAN-stage total drop (recorded as a participant-level stat
        #       with specimen_label=None + clean_stage_total_drop=True). Reported, not fatal.
        # We distinguish (a) from (b) by whether preprocessing recorded ANY stat for the
        # participant during THIS load: load_participant_data appends a stat when it reaches
        # the downsample step OR when the clean stage drops everything, but appends nothing
        # when it fails to load. A third case, (c) participant loaded non-empty but NONE of
        # its metadata specimen_labels match the data (specimen-id mismatch), is also a
        # genuine failure → fail loud.
        strict = fold_label == "all"
        expected_participants = list(fold_specimens["participant_label"].unique())
        failed_participants: List[str] = []   # cases (a) and (c) — fail loud
        qc_dropped_participants: List[str] = []  # case (b) — report only
        yielded_participants: set = set()

        # Group by participant (files are per participant)
        for participant_label in expected_participants:
            stats_before = len(self._preprocessing_stats)
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
                if strict:
                    failed_participants.append(participant_label)
                continue

            if participant_df.empty:
                if strict:
                    # Did preprocessing run for this participant (clean and/or downsample)?
                    # If any stat was recorded during THIS load, it's a legitimate QC-drop
                    # (b) — either all specimens failed the downsampling thresholds or the
                    # whole participant was dropped at the clean stage. If NO stat was
                    # recorded, the load failed silently (a) → treat as a failure.
                    reached_preprocessing = any(
                        s.get("participant_label") == participant_label
                        for s in self._preprocessing_stats[stats_before:]
                    )
                    (qc_dropped_participants if reached_preprocessing
                     else failed_participants).append(participant_label)
                continue

            # Get specimens for this participant in this fold
            participant_specimens = fold_specimens[
                fold_specimens["participant_label"] == participant_label
            ]

            # Yield each specimen separately
            yielded_any = False
            for _, specimen_row in participant_specimens.iterrows():
                specimen_label = specimen_row["specimen_label"]

                # Filter to this specimen. DOWNSAMPLED data already has
                # specimen_label; RAW/CLEAN data has repertoire_id (AIRR column).
                if "specimen_label" in participant_df.columns:
                    specimen_df = participant_df[
                        participant_df["specimen_label"] == specimen_label
                    ]
                else:
                    specimen_df = participant_df[
                        participant_df["repertoire_id"] == specimen_label
                    ]
                    # Rename for downstream consistency
                    if not specimen_df.empty:
                        specimen_df = specimen_df.rename(
                            columns={"repertoire_id": "specimen_label"}
                        )

                if not specimen_df.empty:
                    yielded_participants.add(participant_label)
                    yielded_any = True
                    yield specimen_label, specimen_df, specimen_row

            # Case (c): the participant's data loaded but none of its metadata
            # specimens matched — a specimen-id mismatch, not a QC-drop.
            if strict and not yielded_any:
                failed_participants.append(participant_label)

        # --- Completeness check (train-all only) ---
        if strict:
            if failed_participants:
                raise RuntimeError(
                    f"Failed to load {len(failed_participants)} participant(s) while "
                    f"loading the full dataset (fold_label='all'): "
                    f"{sorted(failed_participants)[:20]}"
                    f"{' ...' if len(failed_participants) > 20 else ''}. "
                    f"Cause is a missing/corrupt participant cache or raw file, a "
                    f"missing repertoire_id column, or a specimen-id mismatch between "
                    f"metadata and data — NOT normal QC. Training on ALL data requires "
                    f"every participant to load; rebuild the participant cache for these "
                    f"participants (e.g. re-run compute_model3_embeddings.py or "
                    f"cache_and_report_all_data.py) and retry."
                )
            # Participants that loaded but had every specimen dropped by downsampling
            # QC are legitimate — report them clearly (never silently), don't fail.
            if qc_dropped_participants:
                self._log(
                    f"{len(qc_dropped_participants)} of {len(expected_participants)} "
                    f"participant(s) contributed no data after QC (all specimens dropped "
                    f"by downsampling thresholds) and were excluded from the full dataset: "
                    f"{sorted(qc_dropped_participants)[:20]}"
                    f"{' ...' if len(qc_dropped_participants) > 20 else ''}",
                    level=0,
                )

    def load_cached_participant(
        self, participant_label: str
    ) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """Load cached participant data with clone_id parameter validation.

        Extends the base class to validate that explicitly-specified clone_id
        parameters are consistent with the cache. Only parameters that the
        user actually set (non-None in the constructor) are compared against
        cached values. Unspecified parameters (None) are accepted as-is,
        allowing the natural workflow: set clone_id params once when building
        the cache, then omit them on subsequent training/embedding commands.

        Parameters
        ----------
        participant_label : str
            Participant identifier.

        Returns
        -------
        tuple or None
            ``(dataframe, preprocessing_stats)`` or ``None`` if not cached.

        Raises
        ------
        ValueError
            If explicitly-specified clone_id parameters conflict with cached
            values.
        ValueError
            If cache was built without clone_id computation but
            force_clone_id=True is now set.
        ValueError
            If cache predates clone_id tracking (no ``clone_id_computed`` key
            in stats) and clone_id params or force_clone_id were specified.
        """
        result = super().load_cached_participant(participant_label)
        if result is None:
            return None

        df, cached_stats = result

        self._check_clone_id_params_against_stats(
            cached_stats, f"participant cache for '{participant_label}'"
        )

        return df, cached_stats

    def load_participant_data(
        self,
        participant_label: str,
        preprocessing_stage: PreprocessingStage = PreprocessingStage.DOWNSAMPLED,
    ) -> pd.DataFrame:
        """
        Load data for one participant.

        For CLEAN/DOWNSAMPLED stages, tries the participant cache first. If the
        cache hits, data_dir is not needed (supports metadata-only mode). On
        cache miss, falls back to loading the raw file from data_dir and
        preprocessing it (raises RuntimeError if data_dir is None).

        For RAW stage, always reads from disk (requires data_dir).

        Args:
            participant_label: Participant identifier
            preprocessing_stage: Level of preprocessing to apply

        Returns:
            DataFrame with sequence-level data (may contain multiple specimens)
        """
        # For RAW stage, always load from original file (no cache)
        if preprocessing_stage == PreprocessingStage.RAW:
            if self.data_dir is None:
                raise RuntimeError(
                    "Cannot load RAW participant data: data_dir is None (metadata-only mode). "
                    "Initialize the loader with a valid data_dir to load sequence data."
                )
            # Try .tsv.gz first, fall back to uncompressed
            file_path = self.data_dir / f"part_table_{participant_label}.tsv.gz"
            if not file_path.exists():
                file_path = self.data_dir / f"part_table_{participant_label}"
                if not file_path.exists():
                    self._log(
                        f"Raw data file not found for participant: {participant_label}", level=1
                    )
                    return pd.DataFrame()

            self._log(f"Loading RAW participant: {participant_label}", level=2)

            # Load file (pandas auto-detects .gz compression)
            try:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
                df = self._normalize_boolean_cols(df)
                # Numeric-looking labels (e.g. 310101) are read as int64;
                # metadata always stores them as str → normalize to match
                return normalize_identifier_columns(df)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return pd.DataFrame()

        # For CLEAN or DOWNSAMPLED: try participant cache first
        cached_result = self.load_cached_participant(participant_label)
        if cached_result is not None:
            df, etl_stats = cached_result
            self._log(f"Loaded participant {participant_label} from cache", level=2)
        else:
            # Cache miss: need raw data to preprocess
            if self.data_dir is None:
                raise RuntimeError(
                    f"Cannot load participant '{participant_label}': data_dir is None "
                    f"(metadata-only mode) and participant is not in cache. "
                    f"Initialize the loader with a valid data_dir, or build the "
                    f"participant cache first with cache_and_report_all_data.py."
                )
            self._log(f"Cache miss - preprocessing participant: {participant_label}", level=2)

            # Try .tsv.gz first, fall back to uncompressed
            file_path = self.data_dir / f"part_table_{participant_label}.tsv.gz"
            if not file_path.exists():
                file_path = self.data_dir / f"part_table_{participant_label}"
                if not file_path.exists():
                    self._log(
                        f"Raw data file not found for participant: {participant_label}", level=1
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

            # Cache the cleaned data with stats (skip if no cache_dir)
            if not df.empty and self.cache_dir is not None:
                self.cache_participant(
                    participant_label,
                    df,
                    preprocessing_stats=etl_stats,
                    update_metadata=True
                )

        # Numeric-looking labels stored as int64 in parquet/CSV; metadata
        # stores them as str.  Normalize so downstream joins always match.
        if not df.empty:
            df = normalize_identifier_columns(df)

        if preprocessing_stage == PreprocessingStage.CLEAN:
            return df

        # `df` here is the post-CLEAN frame. If it is empty, every sequence was removed
        # by CLEAN-stage QC (non-productive, V-score, dedup, etc.) — a LEGITIMATE QC
        # outcome, NOT a load failure. Record a participant-level QC-drop stat before
        # returning empty so downstream completeness checks (e.g. iter_fold_specimens
        # for train-all) can distinguish this from a genuine load failure (missing/
        # corrupt file, missing repertoire_id): a recorded stat means "preprocessing ran,
        # data was legitimately dropped"; no stat means "could not load". `etl_stats` is
        # defined on both the cache-hit and cache-miss paths above, and a clean-total-drop
        # only occurs on cache-miss (empty results are never cached).
        if df.empty:
            self._preprocessing_stats.append({
                "participant_label": participant_label,
                "specimen_label": None,
                "fold_id": None,
                **etl_stats,
                "clean_stage_total_drop": True,
            })
            return df

        # Stage 2: Downsample (per specimen). A NON-empty clean frame must carry
        # repertoire_id; its absence is a STRUCTURAL problem (malformed clean output),
        # not QC → return empty WITHOUT recording a stat, so the completeness check
        # treats it as a load failure.
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

            # Get fold_id from metadata (for the preprocessing report only).
            # Absent when the dataset has no CV_fold column (train-all datasets) —
            # leave it None rather than raising, since cache building must work
            # without folds.
            fold_id = None
            if (
                FOLD_COL in self.metadata.columns
                and specimen_label in self.metadata["specimen_label"].values
            ):
                fold_id = self.metadata.loc[
                    self.metadata["specimen_label"] == specimen_label,
                    FOLD_COL,
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

        result = pd.concat(processed_specimens, ignore_index=True)

        # Rename repertoire_id → specimen_label for downstream consistency
        # (internally, preprocessing uses repertoire_id from AIRR raw data)
        if "repertoire_id" in result.columns and "specimen_label" not in result.columns:
            result = result.rename(columns={"repertoire_id": "specimen_label"})

        return result

    def precompute_clone_ids(self, n_jobs: int = 4) -> None:
        """Pre-compute clone_id for all participants that need it.

        Separate step that runs BEFORE the main training pipeline. For each
        participant without a cached CLEAN result, loads raw data, runs full
        preprocess_clean (including clone_id computation at step 10.5), and
        caches the result. Subsequent calls to load_participant_data will
        load from cache and skip recomputation.

        When n_jobs > 1, the first participant is processed sequentially
        (so global-once warnings are emitted to the console), then the
        remaining participants are processed in parallel.

        Args:
            n_jobs: Number of parallel worker processes. Each participant
                gets its own process. Default 4. Set to 1 for sequential.

        Raises:
            ValueError: If cache_dir is None (caching is required).
            ValueError: If any cached participant has clone_id params that
                differ from the current configuration, or if the cache was
                built without clone_id computation but force_clone_id=True
                is now set. All cached participants are validated upfront
                before processing new ones.
        """
        if self.cache_dir is None:
            raise ValueError(
                "precompute_clone_ids requires caching to be enabled "
                "(cache_dir must not be None). Remove --dont-use-cache "
                "to enable caching."
            )

        if self.metadata is None:
            self.load_metadata()

        all_participants = self.metadata["participant_label"].unique().tolist()

        # Identify which participants need processing vs already cached
        needs_processing = []
        already_cached = []
        for p in all_participants:
            cache_file, _ = self.get_participant_cache_path(p)
            if cache_file.exists():
                already_cached.append(p)
            else:
                needs_processing.append(p)

        # Validate ALL cached participants' clone_id params to catch
        # mismatches early (before processing new participants). This
        # catches heterogeneous caches from interrupted runs where params
        # changed between builds.
        for p in already_cached:
            self.load_cached_participant(p)

        if not needs_processing:
            logger.info(
                f"All {len(all_participants)} participants already cached, "
                f"nothing to precompute."
            )
            return

        logger.info(
            f"Precomputing clone IDs for {len(needs_processing)}/{len(all_participants)} "
            f"participants ({len(already_cached)} already cached)"
        )

        if n_jobs == 1 or len(needs_processing) == 1:
            # Sequential processing
            for i, p in enumerate(needs_processing, 1):
                if i % 50 == 0 or i == 1:
                    logger.info(f"Processing {i}/{len(needs_processing)}: {p}")
                _preprocess_and_cache_participant(self, p)
        else:
            # Process first participant sequentially for warning output
            logger.info(
                f"Processing first participant sequentially: "
                f"{needs_processing[0]}"
            )
            _preprocess_and_cache_participant(self, needs_processing[0])

            remaining = needs_processing[1:]
            if remaining:
                from joblib import Parallel, delayed

                logger.info(
                    f"Processing {len(remaining)} remaining participants "
                    f"with n_jobs={n_jobs}"
                )
                Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(_preprocess_and_cache_participant)(self, p)
                    for p in remaining
                )

        logger.info(
            f"Precomputation complete: {len(needs_processing)} participants "
            f"processed and cached."
        )

    def preprocess_clean(
        self,
        df: pd.DataFrame,
        participant_label: str,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Stage 1: Cleaning and validation.

        Steps:
            0.  Validate required columns exist; warn about missing optional columns
            1.  Filter productive sequences (skipped with warning if column absent)
            2.  Filter v_score > 80 (TCR) or > 200 (BCR) (skipped with warning if column absent)
            3.  Clean IgBLAST sequences: strip spaces and uppercase (cdr3_aa/fwr4_aa/fwr3_aa)
            4.  Drop sequences with non-standard amino acid characters in CDR3
            5.  Deduplicate identical sequences, sum num_reads
            6.  Fix gene names
            7.  Remove sequences with V genes not in reference (e.g. TRBV25/OR9-2*01)
            8.  Extract FR/CDR regions from reference (if gene_reference provided)
            9.  Create v_gene/j_gene (no allele) + v_gene_w_allele/j_gene_w_allele
            10. Drop sequences with missing V/J/CDR
            10.5. Compute clone_id if missing or force_clone_id=True
                  (CDR3 NT validation, hierarchical clustering)
            11. Add isotype_supergroup = "TCRB"

        Raises:
            ValueError: If required columns (repertoire_id, v_call, j_call,
                cdr3_aa) are missing from the input DataFrame.
            ValueError: If clone_id computation is needed but caching is
                disabled (--dont-use-cache).
            ValueError: If nucleotide CDR3 column is missing when needed for
                clone_id computation and clone_id_use_aa=False.

        Returns:
            Tuple of (cleaned_df, stats)
        """
        stats = {}
        original_count = len(df)
        stats["original_count"] = original_count

        # --- Step 0: Validate required columns ---
        # Required columns: pipeline errors or produces garbage without these.
        # - repertoire_id: specimen identification during downsampling
        # - v_call, j_call: V/J gene extraction (used by all models)
        # - cdr3_aa: CDR3 sequence (used by Models 2, 3 and for length filtering)
        # clone_id is NOT required here — auto-computed in step 10.5 if absent
        required_seq_cols = ["repertoire_id", "v_call", "j_call", "cdr3_aa"]
        missing_required = [col for col in required_seq_cols if col not in df.columns]
        if missing_required:
            raise ValueError(
                f"Participant '{participant_label}': sequence file is missing required "
                f"column(s): {missing_required}. "
                f"Available columns: {list(df.columns)}. "
                f"See PIPELINE_GUIDE.md section 4.2 for the required sequence file format."
            )

        # Quality filter columns: filtering is skipped if absent, but the user
        # should know. Warn once globally (not per participant).
        if "productive" not in df.columns and not self._warned_missing_productive:
            self._warned_missing_productive = True
            logger.warning(
                "Column 'productive' not found in sequence data. "
                "Productive-sequence filtering will be SKIPPED for all participants. "
                "Non-productive sequences (stop codons, frameshifts) will be included, "
                "which may add noise to model predictions."
            )
        if "v_score" not in df.columns and not self._warned_missing_v_score:
            self._warned_missing_v_score = True
            v_threshold = self.V_SCORE_THRESHOLD[self.gene_locus]
            logger.warning(
                f"Column 'v_score' not found in sequence data. "
                f"V-score quality filtering (>{v_threshold}) will be SKIPPED for all "
                f"participants. Low-confidence V gene assignments will be included, "
                f"which may add noise to model predictions."
            )

        # Optional columns: used when present, handled gracefully when absent.
        # Warn once globally so the user is aware.
        if "sequence" not in df.columns and not self._warned_missing_sequence:
            self._warned_missing_sequence = True
            logger.warning(
                "Column 'sequence' not found in sequence data. "
                "Deduplication of identical sequences will be SKIPPED. "
                "This is acceptable — downsampling (1 seq per clone) handles "
                "most redundancy — but identical sequences within a clone will "
                "be counted separately in pre-downsampling statistics."
            )
        if "num_reads" not in df.columns and not self._warned_missing_num_reads:
            self._warned_missing_num_reads = True
            logger.warning(
                "Column 'num_reads' not found in sequence data. "
                "All sequences will be assigned num_reads=1. "
                "Downsampling will pick an arbitrary sequence per clone instead "
                "of the one with the most reads."
            )
        if "extracted_isotype" not in df.columns and not self._warned_missing_extracted_isotype:
            self._warned_missing_extracted_isotype = True
            logger.warning(
                "Column 'extracted_isotype' not found in sequence data. "
                "Isotype-aware deduplication will not be performed. "
                "This is fine for TCR data (single isotype)."
            )

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

        # --- Step 10.5: Compute clone_id if needed ---
        clone_id_needs_computation = (
            CLONE_ID_COL not in df.columns or self.force_clone_id
        )

        if clone_id_needs_computation:
            # Clone_id computation requires caching — hierarchical clustering
            # per participant is too expensive to repeat on every data access.
            if self.cache_dir is None:
                raise ValueError(
                    "Clone ID computation requires caching — computing hierarchical "
                    "clustering for each participant on every data access is "
                    "prohibitively expensive. Either:\n"
                    "  - Remove --dont-use-cache (recommended), or\n"
                    "  - Provide data that already has a clone_id column"
                )

            # Determine CDR3 column
            if self.clone_id_use_aa:
                clone_cdr3_col = "cdr3_aa"
            else:
                clone_cdr3_col = self.clone_id_cdr3_nt_col
                if clone_cdr3_col not in df.columns:
                    raise ValueError(
                        f"Column '{clone_cdr3_col}' (nucleotide CDR3) is required "
                        f"to compute clone_id but was not found in participant "
                        f"'{participant_label}'. "
                        f"Available columns: {list(df.columns)[:20]}. "
                        f"To use amino acid CDR3 instead (with adjusted thresholds), "
                        f"set clone_id_use_aa=True."
                    )

            # Compute clone_id (validates CDR3, clusters, logs summary)
            df, clone_stats = compute_participant_clone_id(
                df,
                participant_label=participant_label,
                cdr3_col=clone_cdr3_col,
                identity_threshold=self._clone_id_params[
                    "clone_id_identity_threshold"
                ],
                linkage_method=self.clone_id_linkage_method,
                use_aa=self.clone_id_use_aa,
                force=self.force_clone_id,
            )

            stats["clone_id_computed"] = True
            stats["clone_id_params"] = self._clone_id_params.copy()
            stats["clone_id_stats"] = clone_stats
        else:
            stats["clone_id_computed"] = False

        # Step 11: Add isotype_supergroup
        # For TCR, always "TCRB"
        if self.gene_locus == "TCR":
            df["isotype_supergroup"] = "TCRB"
        # TODO: BCR needs proper isotype mapping

        # Compute CDR3 length from cleaned column
        cdr3_col = "cdr3_aa" if "cdr3_aa" in df.columns else None

        if cdr3_col:
            df["cdr3_aa_sequence_trim_len"] = df[cdr3_col].str.len()

        # Map clone ID column for downstream use (downsampling, embeddings)
        if CLONE_ID_COL in df.columns:
            df["igh_or_tcrb_clone_id"] = df[CLONE_ID_COL]

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
