"""Model 2: Convergent Cluster Classifier

Identifies disease-associated CDR3 sequence clusters shared across multiple patients
(convergent sequences), then classifies specimens by counting how many such clusters
their sequences match.

Data splits (one fold):
    train_smaller1  (~12/27 of N, i.e. ~2/3 of train): clustering + Fisher's test + GLM training (always)
    train_smaller2  (~6/27 of N,  i.e. ~1/3 of train): p-value threshold selection (MCC grid search);
                                  optionally added to GLM training pool after p-value selection
                                  (only with retrain_on_full_train=True)
    validation      (not used in Mal-ID-Lite): would be ~2/9 of N in original Mal-ID
    test            (~1/3  of N): final held-out evaluation only

    Note on fractions: Mal-ID-Lite does not implement the validation split. As a result,
    train_smaller1/2 are proportionally larger than in the original Mal-ID (which reserves
    ~6/27 for validation). Here, train_smaller1 and train_smaller2 cover the full train fold.

    All splits are participant-level and stratified by disease. train_smaller1/2 are derived from
    the train fold via stratified train_test_split(test_size=1/3, random_state=0) at training time.

Training pipeline (one fold):
    1. Cluster train_smaller1 CDR3 sequences (all sequences - cross-individual clustering):
       group by (V gene, J gene, CDR3 length),
       then single-linkage hierarchical clustering on normalized Hamming distance.
    2. Fisher's exact test: for each cluster × disease class, count how many participants with and
       without the disease have sequences in the cluster. Right-tail p-value tests enrichment.
    3. Pre-filter: discard clusters not significant for any disease at any candidate p-value.
    4. Centroid computation: weighted majority-vote consensus CDR3 per cluster (weighted
       by clone size). Only computed for pre-filtered significant clusters (~100× speedup).
    5. P-value grid search: for each candidate p_value threshold, featurize both train_smaller1
       and train_smaller2, train each model type on train_smaller1's feature vectors, evaluate
       MCC-with-abstention on train_smaller2. Repeat for each candidate model type.
    6. Select best p_value per model (maximizes MCC-with-abstention on train_smaller2).
    7. Final GLM: train on train_smaller1 only (default, matching original Mal-ID), or
       optionally on train_smaller1+train_smaller2 combined (retrain_on_full_train=True).
       Clusters and Fisher p-values are always frozen from train_smaller1 regardless.

    Why training the GLM on train_smaller1 only is the default:
       The default matches original Mal-ID behavior exactly, providing a reproducible baseline.

    Why training the GLM on A+B is available as an option (retrain_on_full_train=True):
       Once the best p-value is selected using train_smaller2, there is no remaining reason to
       withhold train_smaller2 from regression training. Cluster definitions are frozen from
       train_smaller1 — adding train_smaller2 to the regression pool increases training data
       without introducing new bias beyond what already exists (train_smaller1 specimens
       contribute to both cluster definitions and regression features). The original Mal-ID
       does not perform this step — it saves the grid-search regression directly — which is a
       simplification rather than a principled choice.

Hyperparameter tuning:
    Lambda (regularization strength): tuned automatically by glmnet's internal CV. glmnet fits
        all 100 lambda values in one pass; internal cross-validation uses StratifiedGroupKFold
        (n_splits=5, patient-aware) on train_smaller1 with deviance (log-loss) as the scoring
        metric. The lambda with the best CV score (lambda_max) is selected.
    Alpha (L1/L2 ratio): fixed per model type; not cross-validated. Five variants are defined in
        _CLASSIFIER_ALPHAS: lasso_cv (1.0), elasticnet_cv0.75, elasticnet_cv (0.5),
        elasticnet_cv0.25, ridge_cv (0.0). By default, train_all_folds() trains only
        BEST_MODEL_FOR_METAMODEL[gene_locus] (lasso_cv for TCR, ridge_cv for BCR).
        Pass explicit model_names to train_convergent_cluster_classifier() or --model-names
        to the CLI to train all 5 variants (original Mal-ID behavior).
    p-value threshold: grid search over [0.0005, 0.001, 0.005, 0.01, 0.05], evaluated on
        train_smaller2. Selection criterion: MCC-with-abstention, matching the original Mal-ID
        (crosseval) approach: abstained specimens are appended to y_true/y_pred with predicted
        label "Unknown", then sklearn.metrics.matthews_corrcoef is called on the full array.
        This treats abstentions as misclassifications.
    Model type selection: in the original Mal-ID, all 5 types are trained, evaluated on the
        validation set, and the best is manually hardcoded into config (lasso_cv for TCR,
        ridge_cv for BCR). Mal-ID-Lite skips this step and uses the pre-made result via
        BEST_MODEL_FOR_METAMODEL, training only that one model by default.

Notes on original Mal-ID vs. paper discrepancies:
    - The paper states "models were fit on each train-2 set and evaluated on the validation set",
      suggesting the final LR is fit on train_smaller2. The code fits on train_smaller1 (larger,
      also used for clustering). We follow the code — the paper description appears to be a
      simplification.
    - The paper mentions decision threshold tuning only for external cohort validation. The
      original code also applies it to the internal test set evaluation, but this step is not
      described in the paper's methods. Mal-ID-Lite does not implement threshold tuning.

Inference pipeline (one specimen):
    1. Assign each sequence to its nearest training cluster centroid (same V/J/len
       supergroup; distance <= 1 - sequence_identity_threshold).
    2. Featurize: per specimen, count unique matched clusters per disease class.
    3. Predict: pass feature vector through the fitted StandardScaler → GlmnetLogitNetWrapper.
    Abstention: specimens that match no significant clusters produce no feature vector and
    are excluded from prediction.

Key implementation notes:
    - cdr3_aa is our column; original Mal-ID uses cdr3_seq_aa_q_trim.
    - specimen_label is our specimen identifier (same column name as original Mal-ID).
    - Fisher test uses scipy.stats.hypergeom.sf, which is mathematically identical to the
      right-tail Fisher exact test: P(X >= k) where X ~ Hypergeom(M, n, N), computed as
      hypergeom.sf(k-1, M, n, N). The original uses the 'fisher' package's pvalue_npy for
      the same computation. We use scipy to avoid an extra dependency.
    - Column order in the feature matrix is always sorted(disease_classes), enforced by
      reindex() in featurize(). No MatchVariables step is needed.
    - Counting unit for Fisher test and featurization: unique PARTICIPANTS (not sequences,
      not specimens). One participant appearing in a cluster counts as one, regardless of
      how many sequences they contribute.
    - Counting unit for specimen scores: unique CLUSTER IDs matched per disease class (not
      sequence count). A specimen matching 3 different COVID clusters scores 3, regardless
      of how many sequences fell into each cluster.

References:
    - Original: malid/trained_model_wrappers/convergent_cluster_classifier.py
    - Original: malid/train/train_convergent_cluster_classifier.py
"""

import dataclasses
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from scipy.stats import hypergeom
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper

from malid_lite.utils.arrays import (
    make_consensus_sequence,
    masked_argmin,
    strings_to_numeric_vectors,
)

logger = logging.getLogger(__name__)

# StratifiedGroupKFold gained shuffle/random_state support in sklearn 1.2.
# Warn early so the user knows before hitting a cryptic TypeError at training time.
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
if _sklearn_version < (1, 2):
    warnings.warn(
        f"sklearn {sklearn.__version__} detected. StratifiedGroupKFold(shuffle=True) "
        f"requires sklearn >= 1.2. Upgrade with: conda install -c conda-forge scikit-learn",
        UserWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Column name constants
# (maps our column names to original Mal-ID equivalents in comments)
# ---------------------------------------------------------------------------

CDR3_COL = "cdr3_aa"                    # = cdr3_seq_aa_q_trim in original
SPECIMEN_COL = "specimen_label"
PARTICIPANT_COL = "participant_label"
DISEASE_COL = "disease"
V_GENE_COL = "v_gene"
J_GENE_COL = "j_gene"
CDR3_LEN_COL = "cdr3_aa_sequence_trim_len"
CLONE_MEMBERS_COL = "num_clone_members"

HIGHER_ORDER_GROUP_COLS = [V_GENE_COL, J_GENE_COL, CDR3_LEN_COL]
CLUSTER_ID_COL = "cluster_id_within_clustering_group"
CENTROID_COL = "centroid_sequence"

# ---------------------------------------------------------------------------
# Configuration constants (from original malid/config.py)
# ---------------------------------------------------------------------------

SEQUENCE_IDENTITY_THRESHOLDS: Dict[str, float] = {
    "TCR": 0.90,   # max normalized Hamming distance: 10%
    "BCR": 0.85,   # max normalized Hamming distance: 15%
}

DEFAULT_P_VALUES: List[float] = [0.0005, 0.001, 0.005, 0.01, 0.05]

# glmnet alpha (L1/L2 ratio) for each model name.
# alpha=1 is pure lasso, alpha=0 is pure ridge.
# Notes:
# - No class_weight: Model 2 intentionally ignores class imbalance (by design from original).
# - Lambda (regularization strength) is tuned automatically by glmnet's internal CV (100 values).
# - Internal CV uses StratifiedGroupKFold(n_splits=5), patient-aware (matching original Mal-ID).
# - Deviance (log-loss) is the CV scoring metric (glmnet default).
_CLASSIFIER_ALPHAS: Dict[str, float] = {
    "lasso_cv":          1.0,   # pure L1
    "elasticnet_cv0.75": 0.75,
    "elasticnet_cv":     0.5,   # equal L1+L2
    "elasticnet_cv0.25": 0.25,
    "ridge_cv":          0.0,   # pure L2
}

_GLMNET_N_LAMBDA = 100   # lambda path size (matches original Mal-ID DEFAULT_N_LAMBDAS_FOR_TUNING)
_GLMNET_CV_N_SPLITS = 5  # internal CV folds (matches original Mal-ID make_internal_cv())

BEST_MODEL_FOR_METAMODEL: Dict[str, str] = {
    "TCR": "lasso_cv",
    "BCR": "ridge_cv",
}


# ---------------------------------------------------------------------------
# Artifact path helper (single source of truth for on-disk filenames)
# ---------------------------------------------------------------------------

def get_artifact_paths(
    model_dir: Path,
    fold_id: int,
    model_name: str,
    retrain_on_full_train: bool = False,
) -> Dict[str, Path]:
    """Return the canonical paths for all artifacts of one fold + model variant.

    Parameters
    ----------
    model_dir : Directory containing saved artifacts.
    fold_id : Fold identifier.
    model_name : Model variant name (e.g., "lasso_cv").
    retrain_on_full_train : Whether the GLM (sklearn Pipeline) was trained on
        train_smaller1+2 combined (True) or on train_smaller1 only (False, default —
        matches original Mal-ID behavior). Clusters and Fisher p-values are always
        frozen from train_smaller1 regardless of this flag.
        Controls the suffix on the pipeline and metrics filenames; the clusters and
        p_value files are the same either way.

    Returns
    -------
    Dict with keys: "clusters", "p_value", "pipeline", "metrics".
    """
    train_suffix = "full" if retrain_on_full_train else "split1"
    return {
        "clusters": model_dir / f"fold_{fold_id}_clusters.joblib",
        "p_value":  model_dir / f"fold_{fold_id}_{model_name}_p_value.joblib",
        "pipeline": model_dir / f"fold_{fold_id}_{model_name}_model_{train_suffix}.joblib",
        "metrics":  model_dir / f"fold_{fold_id}_{model_name}_results_{train_suffix}.json",
    }


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FeaturizedData:
    """Output of featurize(). Separates scored specimens from abstaining ones.

    A specimen abstains if none of its sequences match any significant cluster
    (i.e., no cluster with p <= p_value_threshold for any disease class).

    Attributes
    ----------
    X : Feature matrix (n_scored_specimens × n_disease_classes).
        Values are unique cluster-hit counts per disease class (integers >= 0).
        Indexed by specimen ID (SPECIMEN_COL). Columns are sorted disease class names.
    y : True disease labels for scored specimens (matches X.index).
    participant_labels : Participant label for each scored specimen (matches X.index).
        Used as groups in glmnet's internal cross-validation to prevent within-patient leakage.
    sample_names : Specimen IDs for scored specimens (= X.index).
    abstained_sample_y : True disease labels for abstaining specimens.
    abstained_sample_names : Specimen IDs for abstaining specimens.
    p_value_threshold : The p-value cutoff used to select significant clusters.
    """

    X: pd.DataFrame
    y: pd.Series
    participant_labels: pd.Series  # indexed by specimen ID, values are participant labels
    sample_names: pd.Index
    abstained_sample_y: pd.Series
    abstained_sample_names: pd.Index
    p_value_threshold: float

    @property
    def n_scored(self) -> int:
        return len(self.sample_names)

    @property
    def n_abstained(self) -> int:
        return len(self.abstained_sample_names)

    @property
    def abstention_rate(self) -> float:
        total = self.n_scored + self.n_abstained
        return self.n_abstained / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Phase 1: Clustering
# ---------------------------------------------------------------------------

def cluster_training_set(
    df: pd.DataFrame,
    sequence_identity_threshold: float,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Cluster CDR3 sequences using single-linkage hierarchical clustering.

    Sequences are first grouped by (v_gene, j_gene, cdr3_aa_sequence_trim_len).
    Within each group, pairwise normalized Hamming distances are computed and
    sequences are clustered using scipy single-linkage hierarchical clustering,
    cut at distance = (1 - sequence_identity_threshold).

    Groups are processed in parallel using joblib. Each group is fully independent,
    so parallelism scales well. The default n_jobs=4 is safe on most machines;
    increase it if you have more cores and RAM available (see n_jobs parameter).

    Parameters
    ----------
    df : DataFrame with CDR3_COL, HIGHER_ORDER_GROUP_COLS, CLONE_MEMBERS_COL,
         PARTICIPANT_COL columns.
    sequence_identity_threshold : float
        Minimum required sequence identity (0.90 for TCR, 0.85 for BCR).
        Sequences with Hamming distance <= (1 - threshold) end up in the same cluster.
    n_jobs : int, default 4
        Number of parallel workers for the per-supergroup clustering loop.
        Each worker handles one (v_gene, j_gene, cdr3_len) supergroup independently.
        Set to 1 to disable parallelism. Set to -1 to use all available CPU cores.
        Higher values reduce runtime but increase peak memory usage, since each worker
        holds its own distance matrix (O(N²) floats for a group of N unique CDR3s).
        Recommended range: 2–8 for most workstations. On a machine with 16+ cores and
        ≥32 GB RAM, n_jobs=8 or higher is safe.

    Returns
    -------
    df copy with two new columns:
    - cluster_id_within_clustering_group (int): local cluster ID within each V-J-len group.
      scipy fcluster returns 1-indexed integers (1, 2, 3, ...).
      Exception: groups with only one unique CDR3 sequence are assigned cluster_id = 0
      (to avoid the ValueError that scipy raises on degenerate distance matrices).
      Cluster IDs are only meaningful within their (v_gene, j_gene, cdr3_len) group.
    - global_resulting_cluster_ID (tuple): (v_gene, j_gene, cdr3_len, local_id), globally unique.
      Used to cross-reference clusters between clustered_df and pvalue_df.
    """
    df = df.copy().reset_index(drop=True)  # ensure 0..N-1 index for index-based assembly

    # Fill missing clone member counts (treat as clone of size 1)
    if CLONE_MEMBERS_COL not in df.columns:
        df[CLONE_MEMBERS_COL] = 1
    df[CLONE_MEMBERS_COL] = df[CLONE_MEMBERS_COL].fillna(1)

    cut_distance = 1.0 - sequence_identity_threshold  # e.g., 0.10 for 90% identity

    def _cluster_one_group(
        key: Tuple, group_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster one (v_gene, j_gene, cdr3_len) supergroup via single-linkage.

        Returns (original_index_values, cluster_label_array).
        """
        cdr3_vals = group_df[CDR3_COL].values
        idx = group_df.index.values

        if pd.Series(cdr3_vals).nunique() == 1:
            # scipy.cluster.hierarchy.linkage raises ValueError for a single unique obs.
            # Assign all sequences to cluster 0.
            return idx, np.zeros(len(cdr3_vals), dtype=int)

        # Convert AA strings to integer ordinal vectors (same-length guaranteed by grouping)
        vectors = strings_to_numeric_vectors(cdr3_vals)

        # Pairwise normalized Hamming distances (condensed upper-triangle form)
        dist_condensed = pdist(vectors, metric="hamming")

        # Single-linkage hierarchical clustering
        Z = linkage(dist_condensed, method="single")

        # Cut dendrogram: sequences within cut_distance are in the same cluster.
        # Returns 1-indexed cluster labels.
        labels = fcluster(Z, t=cut_distance, criterion="distance")
        return idx, labels

    groups = list(df.groupby(HIGHER_ORDER_GROUP_COLS, observed=True, sort=False))

    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(_cluster_one_group)(key, group_df)
        for key, group_df in groups
    )

    # Assemble results back into df in original row order
    cluster_labels = np.empty(len(df), dtype=object)
    for idx_vals, labels in results:
        cluster_labels[idx_vals] = labels
    df[CLUSTER_ID_COL] = cluster_labels.astype(int)

    assert not df[CLUSTER_ID_COL].isna().any(), (
        "BUG: Some sequences have NaN cluster_id after clustering."
    )

    # Build globally unique cluster IDs as tuples
    df["global_resulting_cluster_ID"] = list(zip(
        df[V_GENE_COL],
        df[J_GENE_COL],
        df[CDR3_LEN_COL],
        df[CLUSTER_ID_COL],
    ))

    return df


# ---------------------------------------------------------------------------
# Phase 2: Fisher's exact test
# ---------------------------------------------------------------------------

def compute_fisher_scores(
    clustered_df: pd.DataFrame,
    disease_col: str = DISEASE_COL,
) -> pd.DataFrame:
    """Compute right-tail Fisher's exact test p-values for each (cluster, disease) pair.

    For each cluster, tests separately for each disease class: "is this cluster enriched
    for this disease?" This is a one-vs-rest test: the target disease is compared against
    all other diseases combined.

    Counting unit: unique PARTICIPANTS (not sequences, not specimens). If a participant
    has 10 sequences in a cluster, they still count as 1. This prevents high-depth
    participants from dominating the enrichment signal.

    For each (cluster, disease) pair, the 2×2 contingency table is:

                       | NOT in cluster | In cluster |
        This disease   |       c        |     a      |
        All others     |       d        |     b      |

    Where a, b, c, d are counts of unique participants. The right-tail p-value answers:
    given the cluster size (a+b) and the total number of patients with this disease (a+c),
    what is the probability of seeing an overlap of `a` or more by chance alone?

    Example: 100 total participants (40 COVID, 60 non-COVID). Cluster #42 contains
    18 COVID patients and 2 non-COVID patients (20 total in cluster).
        a=18, b=2, c=22, d=58
    This cluster is heavily COVID-enriched — the p-value will be very small.
    If instead 8 COVID and 12 non-COVID were in the cluster, it is not enriched —
    the p-value will be close to 1.

    Computed via the hypergeometric survival function (equivalent to Fisher exact):
        p = hypergeom.sf(a - 1, M=total_participants, n=cluster_size, N=n_disease)
          = P(X >= a), where X ~ Hypergeom(M, n, N)
    When a=0: sf(-1) = 1.0 (no enrichment — correct).

    This is run on all clusters before centroid computation, because its result is used
    to pre-filter clusters (discard those not significant for any disease), making the
    subsequent centroid computation much faster.

    Parameters
    ----------
    clustered_df : DataFrame with cluster assignments, PARTICIPANT_COL, and disease_col.
    disease_col : Name of the disease label column (default: "disease").

    Returns
    -------
    DataFrame indexed by (V_GENE_COL, J_GENE_COL, CDR3_LEN_COL, CLUSTER_ID_COL)
    with one column per disease class (alphabetically sorted), values = right-tail p-values.
    """
    group_cols = HIGHER_ORDER_GROUP_COLS + [CLUSTER_ID_COL]

    # Count unique participants per (cluster, disease) — counting unit is participants
    counts = (
        clustered_df
        .groupby(group_cols + [disease_col], observed=True)[PARTICIPANT_COL]
        .nunique()
        .unstack(fill_value=0)
    )

    disease_classes = sorted(counts.columns.tolist())
    counts = counts[disease_classes]

    # Total unique participants per disease class
    n_per_disease = (
        clustered_df.groupby(disease_col)[PARTICIPANT_COL]
        .nunique()
        .reindex(disease_classes, fill_value=0)
    )
    n_total = int(n_per_disease.sum())

    # Cluster sizes = total unique participants in each cluster (across all diseases)
    cluster_sizes = counts.sum(axis=1)

    # Vectorized right-tail Fisher test via hypergeometric survival function.
    # sf(k-1) = P(X >= k). When k=0: sf(-1) = 1.0 (not enriched — correct).
    pvalue_df = pd.DataFrame(index=counts.index)
    for disease in disease_classes:
        k = counts[disease].values          # overlap: cluster ∩ disease (unique participants)
        n = cluster_sizes.values            # cluster size
        N = int(n_per_disease[disease])     # total participants with this disease
        M = n_total                         # total participants

        pvalue_df[disease] = hypergeom.sf(k - 1, M=M, n=n, N=N)

    return pvalue_df


# ---------------------------------------------------------------------------
# Phase 3: Cluster centroids
# ---------------------------------------------------------------------------

def get_cluster_centroids(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the consensus (centroid) CDR3 sequence for each cluster.

    For each cluster, finds the weighted-majority-vote amino acid at every CDR3
    position. The weight of each unique CDR3 sequence within the cluster is:
        weight = (number of times this exact CDR3 appears in this cluster)
                 × (num_clone_members for this CDR3)

    This weights frequent sequences and large clones more heavily.

    Note: In the training pipeline, call this AFTER pre-filtering to significant
    clusters only. Computing centroids for all ~1M clusters is ~100× wasted effort
    if only ~10K are significant.

    Parameters
    ----------
    clustered_df : DataFrame output of cluster_training_set().

    Returns
    -------
    DataFrame with columns:
    [V_GENE_COL, J_GENE_COL, CDR3_LEN_COL, CLUSTER_ID_COL, CENTROID_COL]
    One row per cluster.
    """
    group_cols = HIGHER_ORDER_GROUP_COLS + [CLUSTER_ID_COL]
    dedup_cols = group_cols + [CDR3_COL, CLONE_MEMBERS_COL]

    # Deduplicate within each (cluster, sequence, clone_size) group, counting occurrences.
    # This aggregates identical sequences before computing the consensus.
    sizes = (
        clustered_df
        .groupby(dedup_cols, observed=True, sort=False)
        .size()
        .rename("occurrence_count")
        .reset_index()
    )

    # Total weight per unique sequence = occurrence_count × clone_size
    sizes["weight"] = sizes["occurrence_count"] * sizes[CLONE_MEMBERS_COL]

    def _consensus(grp: pd.DataFrame) -> str:
        return make_consensus_sequence(grp[CDR3_COL].values, grp["weight"].values)

    centroids = (
        sizes
        .groupby(group_cols, observed=True, sort=False)
        .apply(_consensus)
        .rename(CENTROID_COL)
        .reset_index()
    )

    return centroids


# ---------------------------------------------------------------------------
# Merge centroids with p-values
# ---------------------------------------------------------------------------

def merge_centroids_with_scores(
    centroids: pd.DataFrame,
    pvalue_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge centroid sequences with per-class Fisher p-values into one DataFrame.

    Returns
    -------
    DataFrame with columns:
    [V_GENE_COL, J_GENE_COL, CDR3_LEN_COL, CLUSTER_ID_COL, CENTROID_COL,
     <disease_class_1>, <disease_class_2>, ...]
    """
    group_cols = HIGHER_ORDER_GROUP_COLS + [CLUSTER_ID_COL]
    return pd.merge(
        centroids,
        pvalue_df.reset_index(),
        on=group_cols,
        how="inner",
        validate="1:1",
    )


# ---------------------------------------------------------------------------
# Cluster assignment helpers
# ---------------------------------------------------------------------------

def wrap_centroids_by_supergroup(
    centroids_df: pd.DataFrame,
) -> Dict[Tuple, pd.DataFrame]:
    """Build dict mapping (v_gene, j_gene, cdr3_len) → centroid DataFrame.

    Used for O(1) supergroup lookup during test-set cluster assignment.
    Each value is a DataFrame with columns [CLUSTER_ID_COL, CENTROID_COL].
    """
    result: Dict[Tuple, pd.DataFrame] = {}
    for key, grp in centroids_df.groupby(
        HIGHER_ORDER_GROUP_COLS, observed=True, sort=False
    ):
        result[key] = grp[[CLUSTER_ID_COL, CENTROID_COL]].reset_index(drop=True)
    return result


def assign_sequences_to_known_clusters(
    df: pd.DataFrame,
    centroids_by_supergroup: Dict[Tuple, pd.DataFrame],
    sequence_identity_threshold: float,
) -> pd.DataFrame:
    """Assign each sequence to its nearest training cluster centroid.

    Sequences whose nearest centroid exceeds the distance threshold receive
    CLUSTER_ID_COL = NaN (abstain). Deduplicates sequences before distance
    computation for efficiency, then re-expands to all rows.

    Parameters
    ----------
    df : Sequences DataFrame with CDR3_COL, HIGHER_ORDER_GROUP_COLS.
    centroids_by_supergroup : Output of wrap_centroids_by_supergroup().
    sequence_identity_threshold : float

    Returns
    -------
    df with CLUSTER_ID_COL added (float, NaN for unassigned sequences).
    """
    max_distance = 1.0 - sequence_identity_threshold

    # Deduplicate: one distance computation per unique (v, j, len, cdr3_aa) combination
    unique_seqs = (
        df[HIGHER_ORDER_GROUP_COLS + [CDR3_COL]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    assigned_parts: List[pd.DataFrame] = []

    for key, group_df in unique_seqs.groupby(
        HIGHER_ORDER_GROUP_COLS, observed=True, sort=False
    ):
        cdr3_vals = group_df[CDR3_COL].values
        group_df = group_df.copy()

        if key not in centroids_by_supergroup:
            # No training clusters for this (v, j, len) supergroup → can't assign to any cluster
            group_df[CLUSTER_ID_COL] = np.nan
            assigned_parts.append(group_df)
            continue

        centroids = centroids_by_supergroup[key]

        # Pairwise distances: (n_test_seqs, n_centroids)
        test_vecs = strings_to_numeric_vectors(cdr3_vals)
        centroid_vecs = strings_to_numeric_vectors(centroids[CENTROID_COL].values)
        dist_mat = cdist(test_vecs, centroid_vecs, metric="hamming")

        # Mask distances that exceed the threshold (too different to match)
        masked = np.ma.MaskedArray(dist_mat, mask=(dist_mat > max_distance))

        # For each test sequence, index of nearest unmasked centroid (NaN if all masked)
        nearest_idx = masked_argmin(masked, axis=1)

        # Map centroid index → cluster_id (NaN if no centroid within threshold)
        cluster_ids = centroids[CLUSTER_ID_COL].values
        assignments = [
            float(cluster_ids[int(i)]) if not np.isnan(i) else np.nan
            for i in nearest_idx
        ]
        group_df[CLUSTER_ID_COL] = assignments
        assigned_parts.append(group_df)

    unique_assigned = pd.concat(assigned_parts, ignore_index=True)

    # Merge assignments back to the full (possibly non-unique) df
    df = df.merge(
        unique_assigned[HIGHER_ORDER_GROUP_COLS + [CDR3_COL, CLUSTER_ID_COL]],
        on=HIGHER_ORDER_GROUP_COLS + [CDR3_COL],
        how="left",
    )

    return df


# ---------------------------------------------------------------------------
# Featurization (used during both training grid search and inference)
# ---------------------------------------------------------------------------

def featurize(
    df: pd.DataFrame,
    p_value_threshold: float,
    centroids_with_scores: pd.DataFrame,
    sequence_identity_threshold: float,
    disease_classes: List[str],
    disease_col: str = DISEASE_COL,
) -> FeaturizedData:
    """Featurize sequences into a specimen × disease_class cluster-hit count matrix.

    This is the core inference step of Model 2. The output feature matrix X has one
    row per specimen and one column per disease class. Each cell X[specimen, disease]
    is the number of distinct disease-associated convergent clusters that the specimen
    has at least one sequence in. Specimens with zero matches to any predictive cluster
    are called "abstentions" — they are excluded from X and tracked separately.

    The scoring unit is UNIQUE CLUSTERS, not sequences or clone members. If a specimen
    has 10 sequences all falling into the same cluster C, that still counts as 1. This
    makes the feature robust to sequencing depth variation across specimens.

    Pipeline steps
    --------------
    Step 1 — Filter predictive clusters:
        Keep only clusters whose Fisher p-value is <= p_value_threshold for at least one
        disease class. The input centroids_with_scores contains ALL significant clusters
        (pre-filtered at training time to the loosest candidate threshold); this step
        applies the specific threshold chosen for this call.

    Step 2 — Assign test sequences to known clusters:
        For each test sequence, find the nearest training cluster centroid within the same
        (v_gene, j_gene, cdr3_len) supergroup. If the nearest centroid's Hamming distance
        exceeds (1 - sequence_identity_threshold), the sequence is unassigned (NaN).
        Sequences that belong to supergroups not seen during training are also unassigned.

    Step 3 — Build cluster → disease association table (wide → long):
        centroids_filtered has one row per cluster and one column per disease class (p-values).
        Melt to long format: one row per (cluster, disease_class) pair. Re-filter to keep only
        pairs where p <= p_value_threshold. Result: lookup table of which disease(s) each
        cluster is predictive for. A cluster can be associated with multiple disease classes
        simultaneously (especially at permissive thresholds).

    Step 4 — Collapse to unique (specimen, cluster) hits:
        Drop unassigned sequences (NaN cluster_id). Group by (specimen, cluster_identity) to
        deduplicate — we only care whether a specimen hits a cluster, not how many sequences
        fall into it. Then inner-join with the Step 3 association table to annotate each
        (specimen, cluster) hit with the disease class(es) that cluster predicts.
        Invariant: the inner join must not lose rows (every cluster in specimen_cluster_pairs
        came from centroids_filtered, so it must exist in cluster_disease_assoc).

    Step 5 — Build globally unique cluster IDs:
        cluster_id_within_clustering_group is only locally unique within a (v_gene, j_gene,
        cdr3_len) supergroup. Build a globally unique tuple identifier
        (v_gene, j_gene, cdr3_len, local_cluster_id) for use in the nunique() count below.

    Step 6 — Score each specimen per disease class:
        For each (specimen, disease_class) pair, count the number of DISTINCT global cluster
        IDs. This is the feature value: "how many disease-D-associated convergent clusters
        does this specimen have sequences in?"

    Step 7 — Pivot to feature matrix:
        Convert from long format (one row per specimen×disease_class score) to wide format
        (one row per specimen, one column per disease class). Fill any (specimen, disease_class)
        combination with no cluster hits with 0. Ensure all disease_classes columns are present
        even if no specimen scored for that class. Any specimen appearing in X must have at
        least one non-zero cell — all-zero rows are a bug (they should have abstained).

    Step 8 — Separate abstaining specimens:
        Specimens present in the input df but absent from X had zero sequences match any
        predictive cluster. They abstain. Both scored and abstained specimens are returned
        in FeaturizedData so callers can track abstention rates.

    Parameters
    ----------
    df : Sequences DataFrame with CDR3_COL, HIGHER_ORDER_GROUP_COLS,
         SPECIMEN_COL, PARTICIPANT_COL, and disease_col columns.
         Must not already contain CLUSTER_ID_COL (that column is added here in Step 2).
    p_value_threshold : float
        Fisher p-value cutoff. A cluster is "predictive for disease D" if its
        Fisher p-value for disease D is <= this threshold. Smaller values are stricter
        (fewer clusters qualify, more specimens abstain, but hits are higher confidence).
    centroids_with_scores : DataFrame from merge_centroids_with_scores().
        Columns: HIGHER_ORDER_GROUP_COLS + [CLUSTER_ID_COL, CENTROID_COL] + disease_classes.
        One row per cluster; disease_class columns contain Fisher p-values.
        Pre-filtered to clusters significant at the loosest candidate threshold.
    sequence_identity_threshold : float
        Minimum sequence identity for assigning a test sequence to a centroid.
        Max Hamming distance allowed = 1 - sequence_identity_threshold.
        Must match the value used during training (0.90 for TCR, 0.85 for BCR).
    disease_classes : Sorted list of all disease class names (determines column order of X).
    disease_col : Name of the disease label column in df (default: DISEASE_COL).

    Returns
    -------
    FeaturizedData
        .X                   : DataFrame, shape (n_scored_specimens, n_disease_classes).
                               Integer counts; each cell = number of predictive clusters hit.
        .y                   : Series of disease labels for scored specimens.
        .participant_labels  : Series of participant labels for scored specimens.
        .sample_names        : Index of scored specimen labels.
        .abstained_sample_y  : Series of disease labels for abstaining specimens.
        .abstained_sample_names : Index of abstaining specimen labels.
        .p_value_threshold   : The threshold used (stored for traceability).
    """
    df = df.copy()

    # Record ground-truth disease label and participant label for every specimen upfront.
    # These are needed at the end to populate y and participant_labels, and to identify
    # which specimens abstained (present in input but absent from the scored feature matrix).
    specimen_disease = df.groupby(SPECIMEN_COL)[disease_col].first()
    specimen_participant = df.groupby(SPECIMEN_COL)[PARTICIPANT_COL].first()
    all_specimens = specimen_disease.index

    # -------------------------------------------------------------------------
    # Step 1: Filter to predictive clusters
    # -------------------------------------------------------------------------
    # centroids_with_scores has one column per disease class containing Fisher p-values.
    # Keep clusters where the minimum p-value across all disease classes is <= threshold,
    # i.e. clusters that are significantly enriched for at least one disease.
    sig_mask = centroids_with_scores[disease_classes].min(axis=1) <= p_value_threshold
    centroids_filtered = centroids_with_scores[sig_mask]

    if len(centroids_filtered) == 0:
        # No significant clusters → all specimens abstain.
        # During training this almost certainly indicates a problem (threshold too tight,
        # or Fisher test produced no significant results on the input data).
        warnings.warn(
            f"featurize(): no clusters passed the p-value threshold ({p_value_threshold}). "
            "All specimens will abstain. If this occurs on train_smaller1, the fold result "
            "will be unusable — check that clustering produced valid output.",
            RuntimeWarning,
            stacklevel=2,
        )
        return FeaturizedData(
            X=pd.DataFrame(columns=disease_classes, dtype=float),
            y=pd.Series(dtype=str),
            participant_labels=pd.Series(dtype=str),
            sample_names=pd.Index([]),
            abstained_sample_y=specimen_disease,
            abstained_sample_names=all_specimens,
            p_value_threshold=p_value_threshold,
        )

    # -------------------------------------------------------------------------
    # Step 2: Assign test sequences to known training clusters
    # -------------------------------------------------------------------------
    # For each test sequence, find the nearest centroid in the same (v_gene, j_gene, cdr3_len)
    # supergroup. Sequences whose nearest centroid exceeds max Hamming distance, and sequences
    # in supergroups not seen during training, receive CLUSTER_ID_COL = NaN (abstain).
    centroids_dict = wrap_centroids_by_supergroup(centroids_filtered)
    df = assign_sequences_to_known_clusters(df, centroids_dict, sequence_identity_threshold)

    # -------------------------------------------------------------------------
    # Step 3: Build cluster → disease association table (wide → long format)
    # -------------------------------------------------------------------------
    # centroids_filtered is wide: one row per cluster, one column per disease class (p-values).
    # Melt to long format so each row is one (cluster, disease_class) pair.
    # Re-filter: Step 1 kept clusters significant for >= 1 class, but after melt each cluster
    # has one row per disease class — most of those rows have high p-values for the other
    # classes and must be dropped. Result: every row is a confirmed (cluster, disease) hit.
    # A cluster can appear multiple times here if it is predictive for multiple diseases
    # (rare at tight thresholds; more common at permissive ones like p=1.0).
    group_cols = HIGHER_ORDER_GROUP_COLS + [CLUSTER_ID_COL]
    cluster_disease_assoc = (
        centroids_filtered
        .melt(
            id_vars=group_cols + [CENTROID_COL],   # keep cluster identity columns intact
            value_vars=disease_classes,             # melt the per-disease p-value columns
            var_name="cluster_dominant_label",
            value_name="p_value",
        )
        .query("p_value <= @p_value_threshold")    # drop (cluster, disease) pairs that are not significant
        [group_cols + ["cluster_dominant_label"]]  # drop p_value and centroid — not needed downstream
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Step 4: Collapse sequences → unique (specimen, cluster) hits, then annotate
    # -------------------------------------------------------------------------
    # Drop sequences that were not assigned to any cluster (NaN cluster_id).
    assigned_df = df[~df[CLUSTER_ID_COL].isna()].copy()

    if len(assigned_df) == 0:
        # Clusters existed but no test sequence was close enough to any centroid.
        return FeaturizedData(
            X=pd.DataFrame(columns=disease_classes, dtype=float),
            y=pd.Series(dtype=str),
            participant_labels=pd.Series(dtype=str),
            sample_names=pd.Index([]),
            abstained_sample_y=specimen_disease,
            abstained_sample_names=all_specimens,
            p_value_threshold=p_value_threshold,
        )

    # Deduplicate to unique (specimen, cluster) pairs.
    # A specimen may have many sequences in the same cluster — that still counts as 1 hit.
    # We use groupby+size as the deduplication idiom; the size column is discarded immediately.
    specimen_cluster_pairs = (
        assigned_df
        .groupby([SPECIMEN_COL, disease_col] + group_cols, observed=True)
        .size()
        .reset_index(name="_count")
        [[SPECIMEN_COL, disease_col] + group_cols]  # drop _count — only presence matters
    )

    # Inner join: annotate each (specimen, cluster) hit with which disease class(es) the
    # cluster predicts. This can expand rows if a cluster is associated with multiple diseases.
    # Invariant: row count must not decrease. Every cluster in specimen_cluster_pairs came
    # from centroids_filtered, so it must have at least one entry in cluster_disease_assoc.
    # A decrease would mean a cluster slipped through Step 1 without any significant disease
    # association, which is a bug.
    n_pairs_before_merge = len(specimen_cluster_pairs)
    membership_annot = pd.merge(
        specimen_cluster_pairs,
        cluster_disease_assoc,
        on=group_cols,
        how="inner",
    )
    if len(membership_annot) < n_pairs_before_merge:
        raise ValueError(
            f"BUG: inner join lost rows in featurize(). "
            f"Expected >= {n_pairs_before_merge} rows after merging specimen-cluster pairs "
            f"with cluster-disease associations, but got {len(membership_annot)}. "
            f"A cluster in the assigned sequences has no entry in cluster_disease_assoc."
        )

    # -------------------------------------------------------------------------
    # Step 5: Build globally unique cluster IDs
    # -------------------------------------------------------------------------
    # cluster_id_within_clustering_group is only locally unique within a supergroup
    # (v_gene, j_gene, cdr3_len). Two different supergroups can both have a cluster #3.
    # Build a tuple (v_gene, j_gene, cdr3_len, local_id) that is globally unique across
    # all supergroups, so that nunique() in Step 6 counts correctly.
    membership_annot["_global_cluster_id"] = list(zip(
        membership_annot[V_GENE_COL],
        membership_annot[J_GENE_COL],
        membership_annot[CDR3_LEN_COL],
        membership_annot[CLUSTER_ID_COL],
    ))

    # -------------------------------------------------------------------------
    # Step 6: Score each specimen per disease class
    # -------------------------------------------------------------------------
    # For each (specimen, disease_class) combination: count how many DISTINCT clusters
    # that specimen has sequences in (where "distinct" = unique global_cluster_id).
    # This is the final feature value. It is an integer >= 1 for every row present here.
    specimen_scores = (
        membership_annot
        .groupby([SPECIMEN_COL, "cluster_dominant_label"], observed=True)
        ["_global_cluster_id"]
        .nunique()
        .rename("score")
    )

    # -------------------------------------------------------------------------
    # Step 7: Pivot to specimen × disease_class feature matrix
    # -------------------------------------------------------------------------
    # Convert from long format (one row per specimen×disease_class score) to wide format
    # (one row per specimen, one column per disease class).
    # reindex(columns=disease_classes) ensures all disease columns are present with a
    # consistent order even if no specimen scored for a particular disease class.
    # fillna(0): specimens that hit no cluster for a given disease class score 0 there.
    X = (
        specimen_scores
        .reset_index()
        .pivot(index=SPECIMEN_COL, columns="cluster_dominant_label", values="score")
        .reindex(columns=disease_classes)
        .fillna(0)
        .astype(int)
    )
    X.columns.name = None

    # Every row in X must have at least one non-zero cell. A specimen with all zeros
    # should never have made it into X — it had no cluster hits and should have abstained.
    if len(X) > 0 and (X.values == 0).all(axis=1).any():
        raise ValueError(
            "BUG: Some scored specimens have all-zero feature rows. "
            "They should have been abstained."
        )

    # -------------------------------------------------------------------------
    # Step 8: Separate scored vs abstaining specimens
    # -------------------------------------------------------------------------
    # Specimens in the input df that are absent from X had zero sequences match any
    # predictive cluster — they abstain. Both groups are returned so callers can track
    # abstention rates and include abstaining specimens in downstream evaluation.
    scored_specimens = X.index
    abstained_mask = ~specimen_disease.index.isin(scored_specimens)

    y = specimen_disease.reindex(scored_specimens)
    y_abstained = specimen_disease[abstained_mask]
    participant_labels = specimen_participant.reindex(scored_specimens)

    return FeaturizedData(
        X=X,
        y=y,
        participant_labels=participant_labels,
        sample_names=scored_specimens,
        abstained_sample_y=y_abstained,
        abstained_sample_names=y_abstained.index,
        p_value_threshold=p_value_threshold,
    )


# ---------------------------------------------------------------------------
# P-value threshold selection
# ---------------------------------------------------------------------------

# Candidate abstain labels tried in order. The first one not present in the
# actual disease classes is used. "UNKNOWN99" is the default; "UNKNOWN_99" is
# the fallback for the unlikely case that "UNKNOWN99" is a real disease label.
_ABSTAIN_LABEL_CANDIDATES = ["UNKNOWN99", "UNKNOWN_99"]


def _choose_abstain_label(disease_classes: list) -> str:
    """Return the first candidate abstain label not present in disease_classes.

    Raises ValueError if all candidates conflict with actual class names.
    """
    for candidate in _ABSTAIN_LABEL_CANDIDATES:
        if candidate not in disease_classes:
            return candidate
    raise ValueError(
        f"All abstain label candidates {_ABSTAIN_LABEL_CANDIDATES} are present in "
        f"disease_classes {disease_classes}. Add a new candidate to _ABSTAIN_LABEL_CANDIDATES."
    )


def compute_mcc_with_abstention(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_abstained: pd.Series,
    abstain_label: str,
) -> float:
    """MCC penalized by abstentions, matching the original Mal-ID (crosseval) approach.

    Abstained specimens are appended to y_true/y_pred with a predicted label that
    never matches any real disease class (abstain_label). They are therefore treated
    as misclassifications in the MCC calculation.

    This matches crosseval's ModelSingleFoldPerformance.scores(with_abstention=True)["mcc"],
    which expands:
        y_true <- hstack(y_true_scored, y_true_abstained)
        y_pred <- hstack(y_pred_scored, [abstain_label] * n_abstained)
    then calls sklearn.metrics.matthews_corrcoef(y_true, y_pred).
    (crosseval passes sample_weight=None for Model 2, so no sample weights involved.)

    If all specimens abstain: returns 0.0 (no scored predictions to evaluate).
    If none abstain: returns standard MCC on scored specimens only.
    """
    n_scored = len(y_true)
    if n_scored == 0:
        return 0.0

    if len(y_abstained) == 0:
        return float(matthews_corrcoef(y_true, y_pred))

    y_true_full = np.hstack([y_true.values, y_abstained.values])
    y_pred_full = np.hstack([y_pred, [abstain_label] * len(y_abstained)])
    return float(matthews_corrcoef(y_true_full, y_pred_full))


def build_pipeline(model_name: str) -> Pipeline:
    """Build sklearn pipeline: StandardScaler → GlmnetLogitNetWrapper.

    Uses StratifiedGroupKFold internal CV (patient-aware, matching original Mal-ID).
    Groups (participant_label per specimen) must be passed to fit() via classifier__groups.
    """
    if model_name not in _CLASSIFIER_ALPHAS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {sorted(_CLASSIFIER_ALPHAS.keys())}"
        )
    internal_cv = StratifiedGroupKFold(
        n_splits=_GLMNET_CV_N_SPLITS,
        shuffle=True,
        random_state=0,
    )
    classifier = GlmnetLogitNetWrapper(
        alpha=_CLASSIFIER_ALPHAS[model_name],
        n_lambda=_GLMNET_N_LAMBDA,
        internal_cv=internal_cv,
        scoring=GlmnetLogitNetWrapper.deviance_scorer,
        standardize=False,      # StandardScaler in pipeline handles this
        use_lambda_1se=False,   # use lambda_max (best CV score), not the 1se shrinkage
        require_cv_group_labels=True,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train_convergent_cluster_classifier(
    train_smaller1_df: pd.DataFrame,
    train_smaller2_df: pd.DataFrame,
    sequence_identity_threshold: float,
    model_names: Optional[List[str]] = None,
    p_values: Optional[List[float]] = None,
    disease_col: str = DISEASE_COL,
    retrain_on_full_train: bool = False,
    n_jobs: int = 4,
    verbose: int = 1,
) -> Dict:
    """Full Model 2 training pipeline for one fold.

    Phases:
    1. Cluster train_smaller1 sequences.
    2. Fisher's exact test on all clusters (fast; run before centroid computation).
    3. Pre-filter: discard clusters not significant for any disease at any candidate p-value.
       Major speedup: ~1M clusters → ~10K. Centroids only computed for survivors.
    4. Compute centroids for pre-filtered clusters only (expensive; ~100× faster after pre-filter).
    5. P-value grid search: featurize + train + evaluate on train_smaller2 for each (p_value, model).
    6. Select best p_value per model (maximizes MCC-with-abstention on train_smaller2).
    7. Re-train final model. If retrain_on_full_train=False (default): train GLM on train_smaller1
       only (matches original Mal-ID behavior). If True: train GLM on train_smaller1 + train_smaller2
       combined. Clusters and Fisher p-values are always frozen from train_smaller1 regardless.

    Parameters
    ----------
    train_smaller1_df : Sequences for clustering and final model training (~2/3 of train).
    train_smaller2_df : Sequences for p-value threshold selection (~1/3 of train).
    sequence_identity_threshold : Clustering and assignment threshold.
    model_names : List of model names to train (keys of _CLASSIFIER_ALPHAS).
        Defaults to all 5 variants. Note: train_all_folds() in train_model2.py defaults to
        [BEST_MODEL_FOR_METAMODEL[gene_locus]] (lasso_cv for TCR, ridge_cv for BCR).
    p_values : P-value candidates for the threshold grid search. Defaults to
        DEFAULT_P_VALUES = [0.0005, 0.001, 0.005, 0.01, 0.05], matching the original Mal-ID.
    disease_col : Name of the disease label column.
    retrain_on_full_train : If False (default), train the final GLM on train_smaller1
        only, matching original Mal-ID behavior. If True, re-train GLM on train_smaller1 +
        train_smaller2 combined after p-value selection — clusters and Fisher p-values are
        always frozen from train_smaller1 regardless.
    n_jobs : int, default 4
        Number of parallel workers for Phase 1 (clustering) only. The p-value grid search
        and GLM fitting (Phases 4–6) are sequential. Each (v_gene, j_gene, cdr3_len)
        supergroup is fully independent and processed in a separate thread. Set to 1 to
        disable parallelism, -1 to use all CPU cores. Higher values reduce runtime but
        increase peak memory (each worker holds its own O(N²) distance matrix for groups
        of N unique CDR3s). Recommended: 2–8 on most workstations; 8–16 on high-core
        machines with ≥32 GB RAM.
    verbose : Verbosity level.

    Returns
    -------
    dict with keys:
    - "centroids_with_scores": DataFrame (cached; shared across all model names)
    - "disease_classes": sorted list of disease class names
    - "results": dict mapping model_name → {
          "best_p_value": float,
          "pipeline": fitted sklearn Pipeline,
          "all_p_value_metrics": list of per-p-value metric dicts,
      }
    """
    if model_names is None:
        model_names = list(_CLASSIFIER_ALPHAS.keys())
    if p_values is None:
        p_values = DEFAULT_P_VALUES

    disease_classes = sorted(train_smaller1_df[disease_col].unique().tolist())

    # Pick the abstain label to use for MCC computation — must not conflict with any real class.
    abstain_label = _choose_abstain_label(disease_classes)

    # -----------------------------------------------------------------------
    # Phase 1: Cluster train_smaller1
    # -----------------------------------------------------------------------
    if verbose >= 1:
        logger.info("Phase 1: Clustering train_smaller1 sequences...")
    clustered_df = cluster_training_set(train_smaller1_df, sequence_identity_threshold, n_jobs=n_jobs)

    n_clusters = clustered_df["global_resulting_cluster_ID"].nunique()
    if verbose >= 1:
        logger.info(f"  Created {n_clusters:,} clusters from {len(clustered_df):,} sequences")

    # -----------------------------------------------------------------------
    # Phase 2: Fisher's exact test (before centroid computation)
    # -----------------------------------------------------------------------
    # Run Fisher test on ALL clusters to get p-values. This is fast.
    # We use the result to pre-filter before centroid computation (which is slow).
    if verbose >= 1:
        logger.info("Phase 2: Running Fisher's exact test...")
    pvalue_df = compute_fisher_scores(clustered_df, disease_col=disease_col)
    if verbose >= 1:
        logger.info(f"  Fisher scores computed for {len(pvalue_df):,} clusters × {len(disease_classes)} disease classes")

    # Pre-filter: discard clusters that are not significant for any disease at any candidate p-value.
    # This is a major performance optimization: on the full dataset, ~1M clusters are created
    # but only ~10K pass this filter. Computing centroids on all would be ~100× wasted effort.
    max_p = max(p_values)
    sig_cluster_ids = pvalue_df.index[pvalue_df[disease_classes].min(axis=1) <= max_p]
    clustered_df_sig = clustered_df[
        clustered_df["global_resulting_cluster_ID"].isin(sig_cluster_ids)
    ]
    pvalue_df_sig = pvalue_df.loc[sig_cluster_ids]
    if verbose >= 1:
        logger.info(
            f"  Pre-filtered to {len(sig_cluster_ids):,} significant clusters out of {len(pvalue_df):,} total clusters "
            f"(min p <= {max_p} for at least one disease class)"
        )

    # -----------------------------------------------------------------------
    # Phase 3: Compute cluster centroids (on pre-filtered clusters only)
    # -----------------------------------------------------------------------
    if verbose >= 1:
        logger.info("Phase 3: Computing cluster centroids (significant clusters only)...")
    centroids = get_cluster_centroids(clustered_df_sig)
    if verbose >= 1:
        logger.info(f"  {len(centroids):,} centroid sequences computed")

    # Merge centroids with their per-disease p-values into one artifact
    centroids_prefiltered = merge_centroids_with_scores(centroids, pvalue_df_sig)
    assert len(centroids_prefiltered) == len(centroids)

    if verbose >= 1:
        logger.info(f"  Centroids merged with Fisher scores: {len(centroids_prefiltered):,} rows")

    # -----------------------------------------------------------------------
    # Phase 4: P-value grid search
    # -----------------------------------------------------------------------
    if verbose >= 1:
        logger.info(
            f"Phase 4: P-value grid search over {p_values} "
            f"× {len(model_names)} models..."
        )

    # Collect all_metrics[model_name][p_value] = metric dict
    all_metrics: Dict[str, List[Dict]] = {m: [] for m in model_names}

    for p_val in p_values:
        if verbose >= 2:
            logger.info(f"  Trying p_value={p_val}...")

        # Featurize both splits
        fd_train = featurize(
            train_smaller1_df, p_val, centroids_prefiltered,
            sequence_identity_threshold, disease_classes, disease_col,
        )
        fd_val = featurize(
            train_smaller2_df, p_val, centroids_prefiltered,
            sequence_identity_threshold, disease_classes, disease_col,
        )

        # Skip if either split yields no scored specimens or only one class
        if fd_train.n_scored == 0 or fd_val.n_scored == 0:
            if verbose >= 2:
                logger.info(f"    Skipping p={p_val}: empty feature matrix")
            for m in model_names:
                all_metrics[m].append({"p_value": p_val, "skipped": True})
            continue

        if fd_train.y.nunique() < 2:
            if verbose >= 2:
                logger.info(f"    Skipping p={p_val}: only 1 class in training set")
            for m in model_names:
                all_metrics[m].append({"p_value": p_val, "skipped": True})
            continue

        # Count clusters significant at this specific p_val (subset of pre-filtered sig_cluster_ids)
        n_sig_at_p_val = int((pvalue_df_sig[disease_classes].min(axis=1) <= p_val).sum())

        for model_name in model_names:
            pipeline = build_pipeline(model_name)
            pipeline.fit(fd_train.X, fd_train.y, classifier__groups=fd_train.participant_labels)

            y_pred = pipeline.predict(fd_val.X)
            mcc_abs = compute_mcc_with_abstention(fd_val.y, y_pred, fd_val.abstained_sample_y, abstain_label)

            all_metrics[model_name].append({
                "p_value": p_val,
                "skipped": False,
                "n_train_scored": fd_train.n_scored,
                "n_train_abstained": fd_train.n_abstained,
                "n_val_scored": fd_val.n_scored,
                "n_val_abstained": fd_val.n_abstained,
                "abstention_rate_val": fd_val.abstention_rate,
                "mcc_with_abstention": mcc_abs,
                "n_significant_clusters": n_sig_at_p_val,
            })

            if verbose >= 2:
                logger.info(
                    f"    {model_name} p={p_val}: "
                    f"MCC_abs={mcc_abs:.4f} "
                    f"(scored {fd_val.n_scored}/{fd_val.n_scored + fd_val.n_abstained})"
                )

    # -----------------------------------------------------------------------
    # Phase 5-6: Select best p-value and re-train final model
    # -----------------------------------------------------------------------
    final_train_label = "train_smaller1 + train_smaller2" if retrain_on_full_train else "train_smaller1"
    if verbose >= 1:
        logger.info(
            f"Phase 5-6: Selecting best p-value and training final models "
            f"(final training on {final_train_label})..."
        )

    # Pre-combine train splits if retraining on full A+B
    if retrain_on_full_train:
        train_full_df = pd.concat([train_smaller1_df, train_smaller2_df], ignore_index=True)
    else:
        train_full_df = train_smaller1_df

    results: Dict[str, Dict] = {}

    for model_name in model_names:
        # Select p-value with highest MCC-with-abstention (skip failed runs)
        valid_metrics = [m for m in all_metrics[model_name] if not m.get("skipped", False)]

        if not valid_metrics:
            logger.warning(f"  {model_name}: all p-values were skipped — no valid run")
            results[model_name] = {
                "best_p_value": None,
                "pipeline": None,
                "all_p_value_metrics": all_metrics[model_name],
            }
            continue

        best_metric = max(valid_metrics, key=lambda m: m["mcc_with_abstention"])
        best_p = best_metric["p_value"]

        if verbose >= 1:
            logger.info(
                f"  {model_name}: best p={best_p} "
                f"(MCC_abs={best_metric['mcc_with_abstention']:.4f})"
            )

        # Final training with best p-value. Clusters are always fixed from train_smaller1.
        # If retrain_on_full_train=True, regression is trained on train_smaller1 + train_smaller2.
        fd_final = featurize(
            train_full_df, best_p, centroids_prefiltered,
            sequence_identity_threshold, disease_classes, disease_col,
        )

        final_pipeline = build_pipeline(model_name)
        final_pipeline.fit(fd_final.X, fd_final.y, classifier__groups=fd_final.participant_labels)

        results[model_name] = {
            "best_p_value": best_p,
            "pipeline": final_pipeline,
            "all_p_value_metrics": all_metrics[model_name],
        }

    return {
        "centroids_with_scores": centroids_prefiltered,
        "disease_classes": disease_classes,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class ConvergentClusterClassifier:
    """Model 2: Convergent Cluster Classifier — inference interface.

    Loads pre-trained artifacts (centroids, best p-value, fitted pipeline) from disk
    and provides featurize / predict_proba / predict methods.

    Parameters
    ----------
    gene_locus : {"TCR", "BCR"}
        Gene locus (determines sequence identity threshold).
    model_name : str
        Which trained model to use for inference (e.g., "lasso_cv").
    """

    def __init__(self, gene_locus: str = "TCR", model_name: str = "lasso_cv"):
        if gene_locus not in SEQUENCE_IDENTITY_THRESHOLDS:
            raise ValueError(f"gene_locus must be one of {list(SEQUENCE_IDENTITY_THRESHOLDS)}")

        self.gene_locus = gene_locus
        self.model_name = model_name
        self.sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]

        # Set after load_from_dir() or load_artifacts()
        self.centroids_with_scores_: Optional[pd.DataFrame] = None
        self.disease_classes_: Optional[List[str]] = None
        self.pipeline_: Optional[Pipeline] = None
        self.best_p_value_: Optional[float] = None
        self._is_loaded: bool = False

        # Binary models only: the (disease, reference_class) pair saved at training time.
        # None for multiclass models.
        self.disease_class_: Optional[str] = None
        self.reference_class_: Optional[str] = None

    @classmethod
    def load_from_dir(
        cls,
        model_dir: Path,
        fold_id: int,
        gene_locus: str = "TCR",
        model_name: Optional[str] = None,
        retrain_on_full_train: bool = False,
    ) -> "ConvergentClusterClassifier":
        """Load a trained ConvergentClusterClassifier from disk.

        Parameters
        ----------
        model_dir : Path to the directory containing saved artifacts.
        fold_id : Fold identifier.
        gene_locus : Gene locus.
        model_name : Model name. Defaults to BEST_MODEL_FOR_METAMODEL[gene_locus].
        retrain_on_full_train : Whether the GLM was trained on train_smaller1+2 combined (True)
            or train_smaller1 only (False, default). Clusters are always from train_smaller1
            regardless. Must match what was used during training.
        """
        if model_name is None:
            model_name = BEST_MODEL_FOR_METAMODEL[gene_locus]

        obj = cls(gene_locus=gene_locus, model_name=model_name)
        obj.load_artifacts(model_dir, fold_id, model_name, retrain_on_full_train)
        return obj

    def load_artifacts(
        self,
        model_dir: Path,
        fold_id: int,
        model_name: str,
        retrain_on_full_train: bool = False,
    ) -> None:
        """Load all artifacts for a specific fold and model from disk.

        Parameters
        ----------
        retrain_on_full_train : Whether the GLM was trained on train_smaller1+2 combined (True)
            or train_smaller1 only (False, default). Clusters are always from train_smaller1
            regardless. Must match what was used during training.

        After loading, binary models will have disease_class_ and reference_class_ set,
        and predict_proba will return columns in [P(reference), P(disease)] order.
        Multiclass models leave both attributes as None and column order is alphabetical.
        Artifacts from before binary_pair was saved are treated as multiclass (no reordering).
        """
        paths = get_artifact_paths(model_dir, fold_id, model_name, retrain_on_full_train)

        for key in ("clusters", "p_value", "pipeline"):
            if not paths[key].exists():
                raise FileNotFoundError(f"Artifact not found: {paths[key]}")

        artifact = joblib.load(paths["clusters"])
        self.centroids_with_scores_ = artifact["centroids_with_scores"]
        self.disease_classes_ = artifact["disease_classes"]

        # binary_pair is None for multiclass artifacts and for older artifacts that predate
        # this field (treated as multiclass — no column reordering applied).
        binary_pair = artifact.get("binary_pair", None)
        if binary_pair is not None:
            self.disease_class_ = binary_pair["disease"]
            self.reference_class_ = binary_pair["reference_class"]

        self.best_p_value_ = joblib.load(paths["p_value"])
        self.pipeline_ = joblib.load(paths["pipeline"])
        self._is_loaded = True

    def _check_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError(
                "Classifier is not loaded. Call load_artifacts() or load_from_dir() first."
            )

    def featurize(self, df: pd.DataFrame, disease_col: str = DISEASE_COL) -> FeaturizedData:
        """Featurize sequences using the best p-value threshold for this model."""
        self._check_loaded()
        return featurize(
            df,
            p_value_threshold=self.best_p_value_,
            centroids_with_scores=self.centroids_with_scores_,
            sequence_identity_threshold=self.sequence_identity_threshold,
            disease_classes=self.disease_classes_,
            disease_col=disease_col,
        )

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order used by predict_proba columns.

        For multiclass models: alphabetical order (sklearn default).
        For binary models: always [reference_class, disease_class], so that
        predict_proba[:, 0] = P(reference/negative) and
        predict_proba[:, 1] = P(disease/positive).
        """
        self._check_loaded()
        if self.reference_class_ is not None:
            return np.array([self.reference_class_, self.disease_class_])
        return self.pipeline_.classes_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities on featurized feature matrix.

        For multiclass models: columns are in alphabetical class order (sklearn default).
        For binary models: columns are always [P(reference/negative), P(disease/positive)],
        regardless of how sklearn ordered the classes internally during training.
        Use classes_ to map column indices to class names.
        """
        self._check_loaded()
        proba = self.pipeline_.predict_proba(X)
        if self.reference_class_ is not None:
            if self.disease_class_ is None:
                raise RuntimeError(
                    "BUG: reference_class_ is set but disease_class_ is None. "
                    "The clusters artifact may be corrupted."
                )
            # Reorder so that reference=col0, disease=col1.
            pipeline_classes = list(self.pipeline_.classes_)
            ref_idx = pipeline_classes.index(self.reference_class_)
            disease_idx = pipeline_classes.index(self.disease_class_)
            proba = proba[:, [ref_idx, disease_idx]]
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels on featurized feature matrix.

        Returns an array of class name strings (e.g. "Covid19", "Healthy/Background").
        Labels are always the original string class names from training — never 0/1 integers.
        The label for each specimen will be one of the values in classes_.
        """
        self._check_loaded()
        return self.pipeline_.predict(X)

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"ConvergentClusterClassifier("
            f"gene_locus={self.gene_locus!r}, "
            f"model_name={self.model_name!r}, "
            f"{status})"
        )
