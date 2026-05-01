"""Clone assignment for immune repertoire sequences.

Computes clone IDs by hierarchical clustering of CDR3 sequences grouped by
(V gene, J gene, CDR3 length) using Hamming distance. Migrated from the
bcr_clones package and adapted for the Mal-ID-Lite pipeline.

Key functions:
    compute_participant_clone_id: Pipeline entry point for Mal-ID-Lite.
        Validates CDR3, calls assign_clones(), logs a single summary line.
    assign_clones: Assigns clone IDs for one or both chains.
    hierarchical_linkage_clonotypes: Core clustering engine — groups by
        (V gene, J gene, CDR3 length), clusters within each group.

Algorithm:
    1. Group sequences by (V gene, J gene, CDR3 length)
    2. Within each group: compute condensed Hamming distance matrix (scipy pdist)
    3. Hierarchical linkage clustering (scipy)
    4. Cut dendrogram at (1 - identity_threshold)
    5. Remap clone IDs by descending size (Clone_1 = largest clone)
    6. Assign "Unknown" to rows with missing genes/CDR3

The algorithm is fully deterministic (no random seed needed): Hamming distance,
scipy linkage, and internal sorting by (V, J, CDR3) ensure reproducible output.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

# --- Column name constants ---
CLONE_ID_COL = "clone_id"
CLONE_ID_ORIGINAL_COL = "clone_id_original"

# Default identity thresholds: (gene_locus, use_aa) -> threshold
DEFAULT_IDENTITY_THRESHOLDS = {
    ("TCR", False): 0.95,  # TCR nucleotide
    ("BCR", False): 0.90,  # BCR nucleotide
    ("TCR", True): 0.90,   # TCR amino acid
    ("BCR", True): 0.85,   # BCR amino acid
}


# ============================================================================
# Hamming distance functions (from bcr_clones.core.hamming)
# ============================================================================


def hamming_identity_fraction(seq1: str, seq2: str) -> float:
    """Compute the fraction of matching positions between two equal-length sequences.

    Args:
        seq1: First sequence.
        seq2: Second sequence (must be same length as seq1).

    Returns:
        Fraction of matching positions (1 - Hamming distance / length).

    Raises:
        ValueError: If sequences have different lengths.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length!")
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def strings_to_character_arrays(
    strs: Union[np.ndarray, List[str], pd.Series],
    validate_equal_lengths: bool = True,
) -> np.ndarray:
    """Create character matrix by viewing strings as 1-character byte arrays.

    Args:
        strs: Array-like of strings to convert.
        validate_equal_lengths: If True, raise on unequal string lengths.

    Returns:
        2D numpy array of single-byte characters, shape (n_strings, max_len).

    Raises:
        ValueError: If strings have unequal lengths and validation is enabled.
    """
    char_matrix = np.array(strs)
    char_matrix = (
        char_matrix.astype("bytes").view("S1").reshape((char_matrix.shape[0], -1))
    )

    if validate_equal_lengths and b"" in char_matrix:
        raise ValueError("Input strings must be of equal lengths.")

    return char_matrix


def strings_to_numeric_vectors(
    strs: Union[np.ndarray, List[str], pd.Series],
    validate_equal_lengths: bool = True,
) -> np.ndarray:
    """Convert strings to numeric vectors (one uint8 entry per character).

    Blank characters (from unequal-length padding) are replaced with NaN.

    Args:
        strs: Array-like of strings to convert.
        validate_equal_lengths: If True, raise on unequal string lengths.

    Returns:
        2D numeric array, shape (n_strings, max_len).
    """
    numeric_arr = strings_to_character_arrays(
        strs, validate_equal_lengths=validate_equal_lengths
    ).view(np.uint8)

    if 0 in numeric_arr:
        # Replace blanks (0s from \x00) with np.nan; cast to float first
        numeric_arr = numeric_arr.astype(float)
        np.place(numeric_arr, numeric_arr == 0.0, np.nan)

    return numeric_arr


def get_condensed_hamming_distmat_pdist(
    seqs: Union[np.ndarray, List[str], pd.Series],
) -> np.ndarray:
    """Compute condensed Hamming distance fraction matrix via scipy pdist.

    All sequences must have the same length (ensured by upstream grouping
    on CDR3 length).

    Args:
        seqs: Array-like of equal-length sequences.

    Returns:
        Condensed distance matrix (1D array, length n*(n-1)/2).

    Raises:
        ValueError: If input is not a pandas Series, numpy array, or list.
    """
    if isinstance(seqs, pd.Series):
        seqs_arr = seqs.values
    elif isinstance(seqs, (np.ndarray, list)):
        seqs_arr = seqs
    else:
        raise ValueError("Input must be a pandas Series, numpy array, or list")

    sequences_as_vectors = strings_to_numeric_vectors(seqs_arr)
    dist_mat_condensed = pdist(sequences_as_vectors, metric="hamming")

    return dist_mat_condensed


# ============================================================================
# Clustering functions (from bcr_clones.core.clustering)
# ============================================================================


def remap_cluster_ids_by_size(
    df: pd.DataFrame,
    cluster_col: str,
    output_col: Optional[str] = None,
    ignore_value: Any = "Unknown",
    prefix: str = "Clone_",
    make_string_ids: bool = True,
) -> pd.DataFrame:
    """Remap cluster IDs by descending cluster size (Clone_1 = largest).

    Creates a mapping from original cluster IDs to new IDs ordered by size,
    then validates the mapping is one-to-one.

    Args:
        df: Input DataFrame.
        cluster_col: Column containing original cluster IDs.
        output_col: Column to store remapped IDs. If None, overwrites cluster_col.
        ignore_value: Value to exclude from remapping (e.g. "Unknown" or NaN).
        prefix: Prefix for string IDs (only used when make_string_ids=True).
        make_string_ids: If True, create "Clone_1"-style IDs; if False, numeric.

    Returns:
        DataFrame with remapped cluster IDs.
    """
    df = df.copy()

    backup_col = f"{cluster_col}_remap_backup"
    df[backup_col] = df[cluster_col].copy()

    if output_col is None:
        output_col = cluster_col

    # Mask for valid (non-ignored) values
    if pd.isna(ignore_value):
        valid_mask = ~df[cluster_col].isna()
    else:
        valid_mask = df[cluster_col] != ignore_value

    if not valid_mask.any():
        logger.warning(
            f'All values in column "{cluster_col}" are {ignore_value}. '
            f"No remapping performed."
        )
        return df

    # Map old IDs to new IDs ordered by descending cluster size
    valid_clusters = df.loc[valid_mask, cluster_col]
    cluster_sizes = valid_clusters.value_counts()

    if make_string_ids:
        mapping = {
            old: f"{prefix}{i}" for i, old in enumerate(cluster_sizes.index, 1)
        }
    else:
        mapping = {old: i for i, old in enumerate(cluster_sizes.index, 1)}

    if make_string_ids:
        df[output_col] = df[cluster_col].astype("object")
    else:
        df[output_col] = df[cluster_col].copy()

    df.loc[valid_mask, output_col] = df.loc[valid_mask, cluster_col].map(mapping)

    # --- Validation: all values mapped correctly, one-to-one ---
    expected_values = set(mapping.values())
    actual_values = df.loc[valid_mask, output_col]
    assert actual_values.isin(expected_values).all(), (
        f"Some cluster IDs were not remapped correctly. "
        f"Unexpected: {actual_values[~actual_values.isin(expected_values)].unique()}"
    )
    assert expected_values.issubset(set(actual_values)), (
        f"Some expected cluster IDs are missing. "
        f"Missing: {expected_values - set(actual_values)}"
    )

    id_pairs = df.loc[
        df[backup_col] != ignore_value, [backup_col, output_col]
    ].drop_duplicates()
    counts = id_pairs[backup_col].value_counts()
    assert counts.eq(1).all(), (
        "Some original IDs map to multiple new IDs. This should not happen!"
    )
    counts = id_pairs[output_col].value_counts()
    assert counts.eq(1).all(), (
        "Some new IDs map to multiple original IDs. This should not happen!"
    )

    df.drop(columns=backup_col, inplace=True)

    return df


def hierarchical_linkage_clonotypes(
    df: pd.DataFrame,
    v_gene_col: str = "v_gene",
    j_gene_col: Optional[str] = "j_gene",
    cdr3_col: str = "cdr3",
    identity_threshold: float = 0.90,
    linkage_method: str = "single",
    new_clone_col_name: str = "clone_id",
    clone_id_prefix: str = "Clone_",
    verbose: bool = False,
    verbose_unknown: bool = True,
) -> pd.DataFrame:
    """Cluster sequences into clones using hierarchical linkage on Hamming distance.

    Groups rows by (V gene, J gene [optional], CDR3 length), then clusters
    within each group at the specified identity threshold. Rows with missing
    values are assigned clone ID "Unknown".

    The function sorts internally for deterministic clone ID assignment but
    writes results back using the original DataFrame index, preserving the
    caller's row order.

    Args:
        df: DataFrame with columns for V gene, J gene, and CDR3.
        v_gene_col: Column name for V gene (required).
        j_gene_col: Column name for J gene (optional; None to skip).
        cdr3_col: Column name for CDR3 sequence (required).
        identity_threshold: Fraction identity to be in the same cluster
            (distance cutoff = 1 - identity_threshold).
        linkage_method: Scipy linkage method ("single", "complete", "average").
        new_clone_col_name: Name of the output clone ID column.
        clone_id_prefix: Prefix for string clone IDs (e.g. "Clone_").
        verbose: If True, log per-gene-group progress at DEBUG level.
        verbose_unknown: If True, log count of rows assigned "Unknown".

    Returns:
        Copy of the input DataFrame with the clone ID column added.

    Raises:
        ValueError: If cdr3_col or v_gene_col is None.
        RuntimeError: If any rows remain unassigned after clustering.
    """
    if cdr3_col is None:
        raise ValueError("CDR3 column must be provided")
    if v_gene_col is None:
        raise ValueError("v_gene_col column must be provided")

    distance_threshold = 1.0 - identity_threshold

    df = df.copy()
    # Initialize as object dtype so it can hold both numeric IDs and "Unknown"
    df[new_clone_col_name] = pd.Series(np.nan, index=df.index, dtype=object)

    # Columns used for grouping (excluding None)
    all_cols = [col for col in [v_gene_col, j_gene_col, cdr3_col] if col is not None]
    gene_cols = [col for col in [v_gene_col, j_gene_col] if col is not None]

    # --- Assign "Unknown" to rows with missing values ---
    missing_mask = df[all_cols].isna().any(axis=1) | df[all_cols].eq("").any(axis=1)
    if missing_mask.any():
        if verbose_unknown:
            n_missing = missing_mask.sum()
            logger.debug(
                f"Found {n_missing} rows with missing values in columns "
                f"{all_cols}, assigning clone ID 'Unknown'"
            )
        df.loc[missing_mask, new_clone_col_name] = "Unknown"

    # Filter to rows that need clustering
    df_to_process = df[df[new_clone_col_name].isna()].copy()
    if len(df_to_process) == 0:
        return df

    # Sort for reproducible clone ID assignment
    df_to_process.sort_values(by=all_cols, inplace=True)

    global_clone_id = 0

    # Group by (V gene, J gene [optional], CDR3 length)
    df_to_process[cdr3_col + "_length"] = df_to_process[cdr3_col].str.len()
    group_cols = gene_cols + [cdr3_col + "_length"]
    grouped = df_to_process.groupby(group_cols, as_index=False, group_keys=True)

    # Progress tracking (only used when verbose=True)
    gene_progress = {}
    current_genes = None
    current_genes_count = 0

    for _, group_indices in grouped.groups.items():
        df_subset = df_to_process.loc[group_indices].copy()

        # Gene combination key for progress tracking
        v_gene = df_subset[v_gene_col].iloc[0]
        if j_gene_col is None:
            genes_key = v_gene
        else:
            j_gene = df_subset[j_gene_col].iloc[0]
            genes_key = (v_gene, j_gene)

        # Progress logging (verbose only)
        if verbose:
            if genes_key != current_genes:
                if current_genes is not None:
                    sequences_processed = sum(gene_progress.values())
                    logger.debug(
                        f"Finished processing gene combination: {current_genes} | "
                        f"{current_genes_count} sequences | "
                        f"Total processed: {sequences_processed}"
                    )
                current_genes = genes_key
                current_genes_count = 0
                if genes_key not in gene_progress:
                    gene_progress[genes_key] = 0

            current_genes_count += len(group_indices)
            gene_progress[genes_key] += len(group_indices)

        if len(df_subset) == 1:
            # Single sequence = its own clone
            global_clone_id += 1
            df.loc[df_subset.index, new_clone_col_name] = global_clone_id
        else:
            condensed_dist = get_condensed_hamming_distmat_pdist(df_subset[cdr3_col])
            Z = linkage(condensed_dist, method=linkage_method)
            cluster_labels = fcluster(Z, t=distance_threshold, criterion="distance")

            # Map local cluster labels to global clone IDs
            cluster_offset = global_clone_id
            new_clone_ids = cluster_labels + cluster_offset
            df.loc[df_subset.index, new_clone_col_name] = new_clone_ids

            global_clone_id += cluster_labels.max()

    # Final progress log for the last gene combination
    if verbose and current_genes is not None:
        sequences_processed = sum(gene_progress.values())
        logger.debug(
            f"Finished processing gene combination: {current_genes} | "
            f"{current_genes_count} sequences | "
            f"Total processed: {sequences_processed}"
        )

    # Convert numeric clone IDs to int, keep "Unknown" as string
    df[new_clone_col_name] = df[new_clone_col_name].apply(
        lambda x: int(x) if pd.notna(x) and x != "Unknown" else x
    )

    # Verify no unexpected NaNs remain
    unassigned_mask = df[new_clone_col_name].isna()
    if unassigned_mask.any():
        n_unassigned = unassigned_mask.sum()
        problem_rows = df.loc[unassigned_mask, all_cols]
        raise RuntimeError(
            f"{n_unassigned} rows were not assigned a clone ID unexpectedly. "
            f"Example unassigned rows:\n{problem_rows.head(10)}"
        )

    # Remap clone IDs by descending cluster size (Clone_1 = largest)
    df = remap_cluster_ids_by_size(
        df,
        cluster_col=new_clone_col_name,
        output_col=new_clone_col_name,
        ignore_value="Unknown",
        prefix=clone_id_prefix,
        make_string_ids=True,
    )

    return df


def assign_clones(
    df: pd.DataFrame,
    heavy_v_gene_col: Optional[str] = None,
    heavy_j_gene_col: Optional[str] = None,
    heavy_cdr3_col: Optional[str] = None,
    light_v_gene_col: Optional[str] = None,
    light_j_gene_col: Optional[str] = None,
    light_cdr3_col: Optional[str] = None,
    identity_threshold: float = 0.90,
    linkage_method: str = "single",
    heavy_clone_output_col_name: str = "heavy_clone_id",
    light_clone_output_col_name: str = "light_clone_id",
    paired_clone_output_col_name: str = "paired_clone_id",
    clone_id_cols_suffix: Optional[str] = None,
    clone_id_prefix: str = "Clone_",
    verbose: bool = False,
) -> pd.DataFrame:
    """Assign clone IDs to sequences, handling single-chain and paired-chain cases.

    For single-chain usage (e.g. Mal-ID-Lite with heavy chain only), provide
    only the heavy_* parameters. For paired chains, provide both heavy_* and
    light_* parameters to get independent clone IDs per chain plus a combined
    paired clone ID.

    The identity threshold is applied to CDR3 sequences via Hamming distance
    within groups of (V gene, J gene [optional], CDR3 length).

    Args:
        df: DataFrame with sequence data.
        heavy_v_gene_col: Heavy chain V gene column (None to skip heavy chain).
        heavy_j_gene_col: Heavy chain J gene column (optional within chain).
        heavy_cdr3_col: Heavy chain CDR3 column.
        light_v_gene_col: Light chain V gene column (None to skip light chain).
        light_j_gene_col: Light chain J gene column (optional within chain).
        light_cdr3_col: Light chain CDR3 column.
        identity_threshold: Fraction identity for same-clone assignment.
        linkage_method: Scipy linkage method ("single", "complete", "average").
        heavy_clone_output_col_name: Output column for heavy chain clone IDs.
        light_clone_output_col_name: Output column for light chain clone IDs.
        paired_clone_output_col_name: Output column for paired clone IDs.
        clone_id_cols_suffix: Optional suffix for all clone ID column names.
        clone_id_prefix: Prefix for clone ID strings (e.g. "Clone_").
        verbose: If True, log per-gene-group progress at DEBUG level.

    Returns:
        DataFrame with added clone ID column(s).

    Raises:
        ValueError: If no chain columns are provided.
    """
    df = df.copy()

    # --- Validate: at least one chain ---
    heavy_cols = [heavy_v_gene_col, heavy_j_gene_col, heavy_cdr3_col]
    light_cols = [light_v_gene_col, light_j_gene_col, light_cdr3_col]

    if all(col is None for col in heavy_cols + light_cols):
        raise ValueError("At least one chain (heavy or light) must be provided")

    # Apply column name suffix if provided
    if clone_id_cols_suffix is not None and clone_id_cols_suffix != "":
        heavy_clone_output_col_name = (
            f"{heavy_clone_output_col_name}_{clone_id_cols_suffix}"
        )
        light_clone_output_col_name = (
            f"{light_clone_output_col_name}_{clone_id_cols_suffix}"
        )
        paired_clone_output_col_name = (
            f"{paired_clone_output_col_name}_{clone_id_cols_suffix}"
        )

    # --- Heavy chain ---
    if heavy_v_gene_col is not None:
        df = hierarchical_linkage_clonotypes(
            df,
            v_gene_col=heavy_v_gene_col,
            j_gene_col=heavy_j_gene_col,
            cdr3_col=heavy_cdr3_col,
            identity_threshold=identity_threshold,
            linkage_method=linkage_method,
            new_clone_col_name=heavy_clone_output_col_name,
            clone_id_prefix=clone_id_prefix,
            verbose=verbose,
        )

    # --- Light chain ---
    if light_v_gene_col is not None:
        df = hierarchical_linkage_clonotypes(
            df,
            v_gene_col=light_v_gene_col,
            j_gene_col=light_j_gene_col,
            cdr3_col=light_cdr3_col,
            identity_threshold=identity_threshold,
            linkage_method=linkage_method,
            new_clone_col_name=light_clone_output_col_name,
            clone_id_prefix=clone_id_prefix,
            verbose=verbose,
        )

    # --- Paired clone IDs (when both chains provided) ---
    if heavy_v_gene_col is not None and light_v_gene_col is not None:
        unknown_mask = (df[heavy_clone_output_col_name] == "Unknown") | (
            df[light_clone_output_col_name] == "Unknown"
        )

        df[paired_clone_output_col_name] = "Unknown"

        valid_rows = ~unknown_mask
        if valid_rows.any():
            df.loc[valid_rows, paired_clone_output_col_name] = (
                df.loc[valid_rows, heavy_clone_output_col_name].astype(str)
                + "_"
                + df.loc[valid_rows, light_clone_output_col_name].astype(str)
            )

            df = remap_cluster_ids_by_size(
                df,
                cluster_col=paired_clone_output_col_name,
                output_col=paired_clone_output_col_name,
                ignore_value="Unknown",
                prefix=clone_id_prefix,
                make_string_ids=True,
            )

    return df


# ============================================================================
# Mal-ID-Lite pipeline integration
# ============================================================================


def resolve_identity_threshold(
    gene_locus: str,
    use_aa: bool,
    override: Optional[float] = None,
) -> float:
    """Resolve the identity threshold for clone assignment.

    When override is provided, it takes precedence over the default. Otherwise,
    the default is looked up from DEFAULT_IDENTITY_THRESHOLDS based on locus
    and CDR3 type.

    Args:
        gene_locus: "TCR" or "BCR".
        use_aa: True for amino acid CDR3, False for nucleotide.
        override: User-specified threshold. If provided, overrides the default.

    Returns:
        Identity threshold in (0, 1].

    Raises:
        ValueError: If gene_locus is not "TCR" or "BCR".
        ValueError: If override is not in (0, 1].
    """
    if override is not None:
        if not 0 < override <= 1:
            raise ValueError(
                f"clone_id_identity_threshold must be in (0, 1], got {override}"
            )
        return override

    key = (gene_locus.upper(), use_aa)
    if key not in DEFAULT_IDENTITY_THRESHOLDS:
        raise ValueError(
            f"No default identity threshold for gene_locus='{gene_locus}', "
            f"use_aa={use_aa}. Expected gene_locus in ['TCR', 'BCR']."
        )
    return DEFAULT_IDENTITY_THRESHOLDS[key]


def compute_participant_clone_id(
    df: pd.DataFrame,
    participant_label: str,
    cdr3_col: str,
    identity_threshold: float,
    linkage_method: str = "single",
    use_aa: bool = False,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute clone_id for a single participant's sequence data.

    Main entry point for clone assignment in the Mal-ID-Lite pipeline.
    Validates CDR3 sequences (when using NT), calls assign_clones() with
    the correct parameters, and logs a single summary line.

    Expected to be called on a DataFrame that already has ``v_gene`` and
    ``j_gene`` columns (no-allele versions, created in step 9 of
    preprocess_clean).

    CDR3 NT validation (only when ``use_aa=False``):
        - Uppercase all characters
        - Strip dashes (AIRR alignment artifacts)
        - Replace empty strings with NaN
        - Drop rows with NaN or non-ACGT characters

    Args:
        df: Participant DataFrame with v_gene, j_gene, and CDR3 columns.
        participant_label: Participant identifier (for logging).
        cdr3_col: Column name for CDR3 sequences (e.g. "cdr3" for NT,
            "cdr3_aa" for AA).
        identity_threshold: Clustering identity threshold in (0, 1].
        linkage_method: Hierarchical clustering linkage method.
        use_aa: If True, skip CDR3 NT validation (AA is already cleaned
            by earlier preprocessing steps).
        force: If True and clone_id column exists, rename existing column
            to clone_id_original and compute new clone_id.

    Returns:
        Tuple of (df_with_clone_id, clone_stats). clone_stats contains:
            - clone_id_original_preserved (bool, if force rename happened)
            - cdr3_nt_validation (dict, if NT validation was applied)
            - n_sequences (int)
            - n_clones (int)
            - n_unknown (int)

    Raises:
        ValueError: If required columns (v_gene, j_gene, cdr3_col) are missing.

    Warning:
        When ``force=True``, the existing ``clone_id`` column is renamed to
        ``clone_id_original``. If ``clone_id_original`` already exists in the
        DataFrame (e.g. from a previous force computation), it will be silently
        overwritten. Callers should guard against this if there is any risk of
        double-calling with ``force=True`` on the same data.
    """
    df = df.copy()
    clone_stats: Dict[str, Any] = {}

    # --- Input validation ---
    for required_col in ["v_gene", "j_gene"]:
        if required_col not in df.columns:
            raise ValueError(
                f"Column '{required_col}' is required for clone assignment but "
                f"was not found. Available columns: {list(df.columns)[:20]}"
            )
    if cdr3_col not in df.columns:
        raise ValueError(
            f"Column '{cdr3_col}' is required for clone assignment but was not "
            f"found. Available columns: {list(df.columns)[:20]}"
        )

    # --- Handle force rename of existing clone_id ---
    if force and CLONE_ID_COL in df.columns:
        df = df.rename(columns={CLONE_ID_COL: CLONE_ID_ORIGINAL_COL})
        logger.info(
            f"Participant {participant_label}: existing '{CLONE_ID_COL}' "
            f"renamed to '{CLONE_ID_ORIGINAL_COL}' (force=True)"
        )
        clone_stats["clone_id_original_preserved"] = True

    # --- CDR3 NT validation (only when using nucleotide CDR3) ---
    if not use_aa:
        # Uppercase to prevent mixed-case issues
        df[cdr3_col] = df[cdr3_col].str.upper()

        # Strip dashes (AIRR alignment artifacts) — sequences are kept
        n_dashes = df[cdr3_col].str.contains("-", na=False).sum()
        if n_dashes > 0:
            df[cdr3_col] = df[cdr3_col].str.replace("-", "", regex=False)

        # Replace empty strings with NaN for uniform missing-value handling
        df[cdr3_col] = df[cdr3_col].replace("", np.nan)

        # Identify rows with invalid CDR3 NT sequences
        n_nan = df[cdr3_col].isna().sum()
        non_acgt_mask = df[cdr3_col].str.contains(r"[^ACGT]", regex=True, na=False)
        n_non_acgt = non_acgt_mask.sum()

        invalid_mask = df[cdr3_col].isna() | non_acgt_mask
        n_invalid = invalid_mask.sum()

        if n_invalid > 0:
            df = df[~invalid_mask].copy()

        clone_stats["cdr3_nt_validation"] = {
            "dashes_stripped": int(n_dashes),
            "rows_dropped_nan": int(n_nan),
            "rows_dropped_non_acgt": int(n_non_acgt),
            "rows_dropped_total": int(n_invalid),
        }

    # --- Compute clone_id ---
    n_sequences = len(df)

    df = assign_clones(
        df,
        heavy_v_gene_col="v_gene",
        heavy_j_gene_col="j_gene",
        heavy_cdr3_col=cdr3_col,
        linkage_method=linkage_method,
        identity_threshold=identity_threshold,
        heavy_clone_output_col_name=CLONE_ID_COL,
        verbose=False,
    )

    # --- Clone statistics ---
    n_unknown = int((df[CLONE_ID_COL] == "Unknown").sum())
    # Don't count "Unknown" as a clone
    n_clones = df[CLONE_ID_COL].nunique() - (1 if n_unknown > 0 else 0)

    clone_stats["n_sequences"] = n_sequences
    clone_stats["n_clones"] = n_clones
    clone_stats["n_unknown"] = n_unknown

    # --- Single summary log line (only non-zero categories) ---
    parts = [f"{n_sequences} sequences", f"{n_clones} clones assigned"]

    if not use_aa and "cdr3_nt_validation" in clone_stats:
        val = clone_stats["cdr3_nt_validation"]
        if val["dashes_stripped"] > 0:
            parts.append(f"{val['dashes_stripped']} dashes stripped")
        dropped_details = []
        if val["rows_dropped_nan"] > 0:
            dropped_details.append(f"{val['rows_dropped_nan']} NaN")
        if val["rows_dropped_non_acgt"] > 0:
            dropped_details.append(f"{val['rows_dropped_non_acgt']} non-ACGT")
        if dropped_details:
            parts.append(
                f"{val['rows_dropped_total']} rows dropped "
                f"({', '.join(dropped_details)})"
            )

    if n_unknown > 0:
        parts.append(f"{n_unknown} unknown")

    logger.info(f"Participant {participant_label}: {', '.join(parts)}")

    return df, clone_stats
