"""Model 3: V-gene-specific sequence-level classifier with ESM-2 embeddings.

Two-stage design:
  Stage 1 — Per-V-gene-group sequence classifiers trained on train_smaller1.
             Each group gets a separate StandardScaler + classifier trained on
             the raw ESM-2 embeddings (640-dim) of CDR3 sequences in that group.
  Stage 2 — Specimen-level rollup model trained on train_smaller2.
             Stage 1 predictions are aggregated per (specimen, group) via a
             configurable aggregation strategy, then concatenated into a wide
             feature vector. A second-stage RandomForest with per-class feature
             subsetting (BinaryOvRClassifierWithFeatureSubsettingByClass) is
             trained on these specimen-level features.

Paper-best configurations:
  BCR: Stage 1 = RandomForest, aggregation = mean, reweigh = True
  TCR: Stage 1 = OvR-Ridge (glmnet), aggregation = entropy_cutoff (0.20),
       reweigh = True

References (relative to Maxim-malid-release-202408/):
  malid/trained_model_wrappers/vj_gene_specific_sequence_classifier.py
  malid/trained_model_wrappers/vj_gene_specific_sequence_model_rollup_classifier.py
  malid/trained_model_wrappers/rollup_sequence_classifier.py
  malid/train/train_vj_gene_specific_sequence_model.py
  malid/train/train_vj_gene_specific_sequence_model_rollup.py
  malid/train/vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr.py
  malid/config.py:94-103
"""
from __future__ import annotations

import functools
import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.base
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from malid_lite.models.model3_classifier_utils import (
    CustomOneVsRestClassifier,
    BinaryOvRClassifierWithFeatureSubsettingByClass,
    _InnerEstimator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDR3_COL = "cdr3_aa"           # CDR3 amino acid sequence (= cdr3_seq_aa_q_trim)
V_GENE_COL = "v_gene"
J_GENE_COL = "j_gene"
ISOTYPE_COL = "isotype_supergroup"
PARTICIPANT_COL = "participant_label"
SPECIMEN_COL = "specimen_label"
DISEASE_COL = "disease"
V_MUT_COL = "v_mut"            # BCR only: somatic hypermutation rate

EMBEDDING_DIM = 640             # ESM2_t30_150M_UR50D output dimension
MIN_SEQUENCES_PER_GROUP = 10    # Skip groups with fewer training sequences
                                # (matches original: train_vj_gene_specific_sequence_model.py:291)

# ---------------------------------------------------------------------------
# Configuration constants (from original malid/config.py, model_definitions.py)
# ---------------------------------------------------------------------------

# Stage 1 TCR: glmnet ridge (L2) via CustomOneVsRestClassifier
# Reference: malid/config.py:94-96 ('ridge_cv_ovr'), model_definitions.py:517-522
_GLMNET_ALPHA = 0.0               # pure ridge (L2, no L1)
_GLMNET_N_LAMBDA = 100            # lambda path size (matches original)
_GLMNET_CV_N_SPLITS = 5           # internal CV folds for lambda selection
_GLMNET_STANDARDIZE = False       # scaler handled externally by GroupSequenceClassifier
_GLMNET_USE_LAMBDA_1SE = False    # use lambda_max (best CV), not 1-SE rule
_GLMNET_CLASS_WEIGHT = "balanced" # balance classes within each binary sub-problem
_GLMNET_RANDOM_STATE = 0

# Stage 1 TCR: OvR wrapper settings
# Reference: model_definitions.py:517-522
_OVR_ALLOW_FAILURE = True                    # skip classes that fail to train
_OVR_NORMALIZE_PROBABILITIES = False         # independent binary outputs, no normalization

# Stage 1: V-gene group parallelization
# Reference: train_vj_gene_specific_sequence_model.py:200-232
_VGENE_PARALLEL_N_JOBS = 4                   # default parallel workers for V-gene groups

# Stage 1 BCR: RandomForest
# Reference: malid/config.py:94-96, model_definitions.py:524-529
_RF_STAGE1_BCR_CONFIG = dict(
    class_weight="balanced_subsample",
    random_state=0,
    n_jobs=1,
)

# Stage 2 base classifier (both loci): RandomForest
# Reference: malid/config.py:101-103, model_definitions.py:524-529
# n_jobs=1: each inner RF is single-threaded; parallelism is at the OvR level
# (BinaryOvRClassifierWithFeatureSubsettingByClass trains N binary clfs in parallel).
# Original uses n_jobs=min(2, n_jobs) for each inner RF.
_RF_STAGE2_CONFIG = dict(
    class_weight="balanced_subsample",
    random_state=0,
    n_jobs=1,
)


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

class AggregationStrategy(Enum):
    """Aggregation strategies for rolling sequence predictions up to specimen level.

    Reference: malid/trained_model_wrappers/
               vj_gene_specific_sequence_model_rollup_classifier.py:62-74
    """
    mean = "mean_aggregated"
    median = "median_aggregated"
    trim_bottom_five_percent = "trim_bottom_five_percent_aggregated"
    # Generic entropy cutoff: threshold is set externally via
    # SequenceLevelClassifier.entropy_max_fraction (default 0.80).
    entropy_cutoff = "entropy_cutoff_aggregated"
    # Data-driven entropy cutoff: threshold is the x-th percentile of the
    # training entropy distribution, stored as entropy_percentile_threshold_.
    entropy_percentile_cutoff = "entropy_percentile_cutoff_aggregated"
    # Legacy fixed-threshold variants (kept for backward compat with saved models)
    entropy_ten_percent_cutoff = "entropy_ten_percent_cutoff_aggregated"
    entropy_twenty_percent_cutoff = "entropy_twenty_percent_cutoff_aggregated"


# ---------------------------------------------------------------------------
# ESM-2 embedding
# ---------------------------------------------------------------------------

def compute_esm2_embeddings(
    sequences: List[str],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute ESM-2 (ESM2_t30_150M_UR50D) embeddings for CDR3 amino acid sequences.

    Uses the last-layer mean representation. Empty/invalid sequences receive
    a zero vector.

    Parameters
    ----------
    sequences : List of CDR3 amino acid sequences (stripped, uppercase).
    batch_size : Number of sequences per forward pass (reduce if OOM).
    device : "cuda", "cpu", or None (auto-detect).

    Returns
    -------
    embeddings : float32 array of shape (n_sequences, 640).

    Reference: malid/apply_embedding.py, embedder "esm2_cdr3"
               Model: ESM2_t30_150M_UR50D (esm.pretrained.esm2_t30_150M_UR50D)
    """
    try:
        import esm
        import torch
    except ImportError as exc:
        raise ImportError(
            "ESM-2 embedding requires the 'fair-esm' and 'torch' packages. "
            "Install via: pip install fair-esm torch"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

    # Sanity checks: verify we loaded the expected model
    num_layers = model.num_layers
    embed_dim = getattr(model, "embed_dim", None) or model.args.embed_dim
    if num_layers != 30:
        raise RuntimeError(
            f"Expected ESM2_t30_150M_UR50D with 30 layers, got {num_layers} layers. "
            f"Wrong model loaded?"
        )
    if embed_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {embed_dim}. "
            f"Wrong model loaded?"
        )

    repr_layer = num_layers  # last layer (30 for this model)

    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    n = len(sequences)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_seqs = sequences[start : start + batch_size]
            data = [(f"seq{i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            # Mean over sequence positions (excluding BOS/EOS tokens)
            reps = results["representations"][repr_layer]  # (batch, L+2, 640)
            for j, seq in enumerate(batch_seqs):
                L = len(seq)
                embeddings[start + j] = reps[j, 1 : L + 1].mean(0).cpu().float().numpy()

    return embeddings


# ---------------------------------------------------------------------------
# Rare V gene filtering
# ---------------------------------------------------------------------------

def find_non_rare_v_genes(sequences_df: pd.DataFrame) -> List[str]:
    """Return V genes with above-median max-frequency-across-disease-classes.

    Keeps the top 50% of V genes by prevalence. Applied when split_on
    includes v_gene (which is always the case for Model 3).

    Reference: malid/helpers.py, malid/train/train_vj_gene_specific_sequence_model.py:113-115
    """
    if DISEASE_COL not in sequences_df.columns:
        raise ValueError(
            f"Column '{DISEASE_COL}' not found in sequences_df. "
            f"Available columns: {list(sequences_df.columns)[:15]}..."
        )

    if len(sequences_df) == 0:
        return []

    # Per-V-gene frequency within each disease class, then max across classes
    counts = sequences_df.groupby(
        [V_GENE_COL, DISEASE_COL], observed=True
    ).size().unstack(fill_value=0)
    freq = counts / counts.sum(axis=0)
    max_freqs = freq.max(axis=1)
    median_freq = max_freqs.median()
    kept = max_freqs[max_freqs >= median_freq].index.tolist()
    logger.info(
        f"  Rare V gene filtering: keeping {len(kept)}/{len(max_freqs)} V genes "
        f"(max-freq-across-diseases >= median {median_freq:.4f})"
    )
    return kept


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def _entropy_threshold_aggregate(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
    max_fraction: float,
    n_classes: int,
    return_survival_count: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Entropy-thresholded weighted mean of per-sequence probabilities.

    Keeps only sequences whose Shannon entropy (in nats) is below
    max_fraction * max_entropy. When no sequences pass the threshold,
    returns the uniform distribution (1/n_classes), NOT a plain mean.

    Parameters
    ----------
    max_fraction : Fraction of max entropy to use as cutoff (0-1 scale).
        E.g. 0.80 means keep sequences with entropy < 80% of max entropy.
    return_survival_count : If True, return (agg, n_survived) instead of just
        agg.  Used by verbose >= 2 diagnostics to avoid recomputing entropy.

    Reference: malid/trained_model_wrappers/rollup_sequence_classifier.py:121-188
    Entropy fallback (uniform): rollup_sequence_classifier.py:154-158
    """
    # Maximum possible entropy (uniform distribution over n_classes), in nats
    max_entropy = scipy.stats.entropy(np.ones(n_classes) / n_classes)
    # max_fraction=0.80 means "keep sequences with entropy < 80% of max entropy"
    # (i.e., cut off the top 20% most uncertain).
    # Reference: vj_gene_specific_sequence_model_rollup_classifier.py:743-765
    #   reduction_factor = 0.9 if ten_percent else 0.8
    #   reduced_cutoff = max_entropy_cutoff * reduction_factor
    threshold = max_fraction * max_entropy

    # Per-sequence entropy (Shannon, nats) — vectorized across all rows.
    # scipy.stats.entropy normalizes each distribution to sum to 1 internally,
    # which is required for OvR classifiers whose probs don't sum to 1.
    # probs is (n_sequences, n_classes). scipy computes entropy along axis=0
    # by default, treating each column as a separate distribution. Transposing
    # to (n_classes, n_sequences) makes each column a single sequence's probs,
    # so the result is a (n_sequences,) array of per-sequence entropies.
    seq_entropies = scipy.stats.entropy(probs.T)
    # Keep only high-confidence (low-entropy) sequences
    mask = seq_entropies < threshold
    n_survived = int(mask.sum())

    if n_survived == 0:
        # No sequences pass the threshold: return uniform prior (not plain mean)
        agg = np.ones(n_classes) / n_classes
    else:
        filtered_probs = probs[mask]
        if weights is not None:
            filtered_weights = weights[mask]
            if filtered_weights.sum() > 0:
                agg = np.average(filtered_probs, weights=filtered_weights, axis=0)
            else:
                agg = filtered_probs.mean(axis=0)
        else:
            agg = filtered_probs.mean(axis=0)

    if return_survival_count:
        return agg, n_survived
    return agg


def _entropy_abs_threshold_aggregate(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
    abs_threshold: float,
    n_classes: int,
    return_survival_count: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Entropy-thresholded aggregation using an absolute threshold in nats.

    Same logic as _entropy_threshold_aggregate but the threshold is a fixed
    value (in nats) rather than a fraction of max entropy. The threshold is
    computed from the training entropy distribution (e.g., the 0.1th percentile)
    and applied identically at train and test time.

    Parameters
    ----------
    abs_threshold : Absolute entropy cutoff in nats. Sequences with entropy
        >= this value are filtered out.
    return_survival_count : If True, return (agg, n_survived) tuple.
    """
    # Per-sequence entropy (Shannon, nats) — same vectorized computation.
    # See _entropy_threshold_aggregate for the transpose explanation.
    seq_entropies = scipy.stats.entropy(probs.T)
    mask = seq_entropies < abs_threshold
    n_survived = int(mask.sum())

    if n_survived == 0:
        agg = np.ones(n_classes) / n_classes
    else:
        filtered_probs = probs[mask]
        if weights is not None:
            filtered_weights = weights[mask]
            if filtered_weights.sum() > 0:
                agg = np.average(filtered_probs, weights=filtered_weights, axis=0)
            else:
                agg = filtered_probs.mean(axis=0)
        else:
            agg = filtered_probs.mean(axis=0)

    if return_survival_count:
        return agg, n_survived
    return agg


def _weighted_mean(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    if weights is not None and weights.sum() > 0:
        return np.average(probs, weights=weights, axis=0)
    return probs.mean(axis=0)


def _weighted_median(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """Weighted median per class (approximated via sorted cumsum).

    Reference: malid/trained_model_wrappers/rollup_sequence_classifier.py
    """
    n_classes = probs.shape[1]
    result = np.empty(n_classes)
    if weights is None or weights.sum() == 0:
        weights = np.ones(len(probs))
    # Normalize weights to sum to 1 for cumulative lookup
    w = weights / weights.sum()
    # Compute weighted median independently for each class column
    for j in range(n_classes):
        order = np.argsort(probs[:, j])         # sort sequences by probability
        cumw = np.cumsum(w[order])               # cumulative weight in sorted order
        idx = np.searchsorted(cumw, 0.5)         # find the 50th percentile
        idx = min(idx, len(probs) - 1)           # clamp to valid index
        result[j] = probs[order[idx], j]
    return result


def _trim_bottom_five_percent(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """Trim bottom 5% of sequences (by row-sum probability) then weighted mean.

    Reference: malid/trained_model_wrappers/rollup_sequence_classifier.py
    """
    # Score each sequence by its total probability mass (row sum)
    row_sums = probs.sum(axis=1)
    # Remove the bottom 5% of sequences by total probability
    threshold = np.percentile(row_sums, 5)
    mask = row_sums >= threshold
    if mask.sum() == 0:
        mask = np.ones(len(probs), dtype=bool)  # keep all if threshold removes everything
    filtered = probs[mask]
    if weights is not None:
        filtered_weights = weights[mask]
        if filtered_weights.sum() > 0:
            return np.average(filtered, weights=filtered_weights, axis=0)
    return filtered.mean(axis=0)


def aggregate_group(
    probs: np.ndarray,
    weights: Optional[np.ndarray],
    strategy: AggregationStrategy,
    n_classes: int,
    entropy_max_fraction: float = 0.80,
    entropy_abs_threshold: Optional[float] = None,
    return_survival_count: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Apply aggregation strategy to a matrix of per-sequence probabilities.

    Parameters
    ----------
    probs    : (n_seqs, n_classes) probability matrix.
    weights  : (n_seqs,) sample weights or None for uniform.
    strategy : AggregationStrategy enum value.
    n_classes: Number of disease classes.
    entropy_max_fraction : Fraction of max entropy to use as cutoff (0-1 scale).
        Only used when strategy is entropy_cutoff. E.g. 0.80 means keep
        sequences with entropy < 80% of max. Ignored for non-entropy strategies.
    entropy_abs_threshold : Absolute entropy threshold in nats. Only used when
        strategy is entropy_percentile_cutoff. Computed from training data
        percentile and stored as a fitted attribute. None for other strategies.
    return_survival_count : If True AND strategy is an entropy variant, return
        (agg, n_survived) so the caller can track filter stats without
        recomputing entropy.  For non-entropy strategies the count is always
        len(probs) (no filtering).

    Returns
    -------
    agg : (n_classes,) aggregated probability vector (or (agg, n_survived) tuple).
    """
    if len(probs) == 0:
        result = np.ones(n_classes) / n_classes
        return (result, 0) if return_survival_count else result

    if strategy == AggregationStrategy.mean:
        result = _weighted_mean(probs, weights)
    elif strategy == AggregationStrategy.median:
        result = _weighted_median(probs, weights)
    elif strategy == AggregationStrategy.trim_bottom_five_percent:
        result = _trim_bottom_five_percent(probs, weights)
    elif strategy == AggregationStrategy.entropy_cutoff:
        return _entropy_threshold_aggregate(
            probs, weights, entropy_max_fraction, n_classes,
            return_survival_count=return_survival_count,
        )
    elif strategy == AggregationStrategy.entropy_ten_percent_cutoff:
        # Legacy fixed threshold: 0.90 = keep below 90% of max entropy
        return _entropy_threshold_aggregate(
            probs, weights, 0.90, n_classes,
            return_survival_count=return_survival_count,
        )
    elif strategy == AggregationStrategy.entropy_twenty_percent_cutoff:
        # Legacy fixed threshold: 0.80 = keep below 80% of max entropy
        return _entropy_threshold_aggregate(
            probs, weights, 0.80, n_classes,
            return_survival_count=return_survival_count,
        )
    elif strategy == AggregationStrategy.entropy_percentile_cutoff:
        if entropy_abs_threshold is None:
            raise ValueError(
                "entropy_abs_threshold is required for entropy_percentile_cutoff. "
                "This should be computed during fit_stage2 from training data "
                "and stored as entropy_percentile_threshold_."
            )
        return _entropy_abs_threshold_aggregate(
            probs, weights, entropy_abs_threshold, n_classes,
            return_survival_count=return_survival_count,
        )
    else:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}. "
            f"Valid options: {[s.name for s in AggregationStrategy]}"
        )
    # Non-entropy strategies: all sequences survive (no filtering)
    return (result, len(probs)) if return_survival_count else result


# ---------------------------------------------------------------------------
# Stage 1: per-V-gene-group sequence classifier
# ---------------------------------------------------------------------------

class GroupSequenceClassifier:
    """One StandardScaler + classifier for a single V-gene [+ isotype] group.

    Reference:
      malid/trained_model_wrappers/vj_gene_specific_sequence_classifier.py
      malid/train/train_vj_gene_specific_sequence_model.py
    """

    def __init__(self, clf):
        self.scaler = StandardScaler()
        self.clf = clf
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "GroupSequenceClassifier":
        """Fit scaler + classifier on raw embeddings for this group.

        Parameters
        ----------
        X : Raw ESM-2 embeddings for this group's sequences.
        y : Disease labels.
        sample_weight : Per-sequence weights (BCR isotype rebalancing), or None.
        groups : Per-sequence group labels for CV grouping (e.g. participant_label).
                 Passed through to inner clf (e.g. CustomOneVsRestClassifier →
                 GlmnetLogitNetWrapper for participant-level internal CV).

        Reference: malid/train/train_vj_gene_specific_sequence_model.py:358-363
        (StandardScaler prepended to pipeline, fitted on raw embeddings)
        """
        X_scaled = self.scaler.fit_transform(X)
        # Only pass non-None kwargs — allows inner classifiers that don't accept
        # these params (e.g. RandomForestClassifier) to work when they're None.
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if groups is not None:
            fit_kwargs["groups"] = groups
        self.clf.fit(X_scaled, y, **fit_kwargs)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
        """Predict probabilities, aligned to all_classes.

        Missing classes (not seen during fit) receive probability 0.

        Reference:
          vj_gene_specific_sequence_classifier.py:127-202
          (reindex columns, fillna(0))
        """
        X_scaled = self.scaler.transform(X)
        # probs columns match self.clf.classes_ (only classes seen during fit)
        probs = self.clf.predict_proba(X_scaled)

        # Map to the global class order (all_classes).
        # Example: clf saw ["COVID-19", "Healthy"] but all_classes is
        # ["COVID-19", "HIV", "Healthy"] → HIV column stays 0.
        full_probs = np.zeros((len(X), len(all_classes)), dtype=np.float32)
        for i, cls in enumerate(self.clf.classes_):
            global_idx = np.where(all_classes == cls)[0]
            assert len(global_idx) > 0, (
                f"Class '{cls}' from group classifier not found in all_classes: {all_classes}"
            )
            full_probs[:, global_idx[0]] = probs[:, i]
        return full_probs


# ---------------------------------------------------------------------------
# Module-level helper for V-gene group parallel training (joblib loky pickling)
# ---------------------------------------------------------------------------

def _train_one_group(
    group_key: tuple,
    group_features: np.ndarray,
    group_labels: np.ndarray,
    group_sample_weights: Optional[np.ndarray],
    group_participant_labels: Optional[np.ndarray],
    clf: GroupSequenceClassifier,
) -> Optional[tuple]:
    """Train a single V-gene group classifier. Module-level for joblib loky pickling.

    Returns (group_key, fitted_clf) on success, or None on failure.
    Reference: train_vj_gene_specific_sequence_model.py:200-232
    """
    try:
        clf.fit(
            group_features, group_labels,
            sample_weight=group_sample_weights,
            groups=group_participant_labels,
        )
        return (group_key, clf)
    except Exception as e:
        logger.warning(f"  Stage 1: failed to train group {group_key}: {e}")
        return None


def _fit_one_binary_ovr_job(
    group_key: tuple,
    class_idx: int,
    X_scaled: np.ndarray,
    y_binary: np.ndarray,
    positive_class: str,
    negative_class: str,
    sample_weight: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    clf,
    allow_failure: bool,
) -> Optional[tuple]:
    """Fit a single binary classifier for one (group, class) pair.

    Used by the flattened parallelism path in fit_stage1. Each call trains
    one binary classifier (e.g., "Covid19 vs rest" for V-gene TRBV5-1).
    This is equivalent to CustomOneVsRestClassifier._fit_binary but as a
    module-level function so joblib's loky backend can pickle it.

    Parameters
    ----------
    group_key      : V-gene group tuple, e.g. ("TRBV5-1",)
    class_idx      : Index of this class within the group's binary problems
    X_scaled       : Pre-scaled feature matrix for this group (from StandardScaler)
    y_binary       : Binary labels (1 = positive_class, 0 = rest)
    positive_class : Disease class this binary clf detects
    negative_class : The "rest" label (for logging only)
    sample_weight  : Per-sequence weights, or None (TCR: always None)
    groups         : Per-sequence participant labels (for internal CV grouping)
    clf            : Cloned, unfitted estimator (GlmnetLogitNetWrapper)
    allow_failure  : If True, return None on error instead of raising

    Returns
    -------
    (group_key, class_idx, positive_class, negative_class, fitted_clf)
    on success, or None on failure.
    """
    try:
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if groups is not None:
            fit_kwargs["groups"] = groups
        clf.fit(X_scaled, y_binary, **fit_kwargs)
        return (group_key, class_idx, positive_class, negative_class, clf)
    except Exception as e:
        msg = (
            f"Stage 1: binary clf failed for group {group_key}, "
            f"class '{positive_class}' vs '{negative_class}': {e}"
        )
        if allow_failure:
            logger.warning(f"  {msg}")
            return None
        raise RuntimeError(msg) from e


# ---------------------------------------------------------------------------
# Main model: SequenceLevelClassifier
# ---------------------------------------------------------------------------

class SequenceLevelClassifier:
    """Two-stage sequence-level classifier (Model 3).

    Stage 1: per-V-gene-group classifiers on ESM-2 embeddings (train_smaller1)
    Stage 2: specimen-level rollup model with feature subsetting (train_smaller2)

    Reference:
      malid/trained_model_wrappers/vj_gene_specific_sequence_model_rollup_classifier.py
      malid/train/train_vj_gene_specific_sequence_model_rollup.py
      malid/config.py:94-103
    """

    def __init__(
        self,
        locus: str = "TCR",
        aggregation_strategy: AggregationStrategy = AggregationStrategy.entropy_cutoff,
        entropy_max_fraction: float = 0.80,
        entropy_bottom_percentile: float = 0.1,
        exclude_rare_v_genes: bool = True,
        min_sequences_per_group: int = MIN_SEQUENCES_PER_GROUP,
        reweigh_by_subset_frequencies: bool = True,
        n_estimators_stage1: int = 100,
        n_estimators_stage2: int = 100,
        n_jobs: int = _VGENE_PARALLEL_N_JOBS,
        reference_class: Optional[str] = None,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        locus : "TCR" or "BCR".
        aggregation_strategy : How to aggregate per-sequence predictions to specimen level.
            Paper best: TCR = entropy_cutoff (0.80), BCR = mean.
            Use entropy_cutoff + entropy_max_fraction for custom thresholds.
            Use entropy_percentile_cutoff + entropy_bottom_percentile for
            data-driven thresholds based on training entropy distribution.
        entropy_max_fraction : Fraction of max entropy to use as cutoff (0-1 scale).
            Only used when aggregation_strategy is entropy_cutoff. E.g. 0.80 means
            keep sequences with entropy < 80% of max entropy. Ignored for other strategies.
        entropy_bottom_percentile : Percentile of training entropy distribution to use
            as cutoff (0-100 scale). Only used when aggregation_strategy is
            entropy_percentile_cutoff. E.g. 0.1 means keep only sequences with
            entropy in the bottom 0.1% of what was observed in training.
            The threshold is computed during fit_stage2 and stored as
            entropy_percentile_threshold_. Ignored for other strategies.
        exclude_rare_v_genes : Filter V genes below median max-frequency.
        min_sequences_per_group : Minimum training sequences per group.
        reweigh_by_subset_frequencies : Multiply aggregated features by per-specimen
            V-gene group frequency. Paper best for both loci.
        n_estimators_stage1 : RF trees for BCR Stage 1.
        n_estimators_stage2 : RF trees for Stage 2.
        n_jobs : Parallel workers for V-gene group training (Stage 1),
            Stage 1 prediction (generate_sequence_predictions), and
            binary OvR classifier training (Stage 2). Each inner RF in Stage 2
            uses n_jobs=1; parallelism is at the OvR level instead.
            Default 4 (reasonable for a personal laptop).
            Reference: original uses Joblib(n_jobs=n_jobs, backend="loky")
            in train_vj_gene_specific_sequence_model.py:200-232 (Stage 1)
            and one_vs_rest_except_negative_class_classifier.py:233 (Stage 2).
        reference_class : For binary/multi-binary mode: the reference (negative)
            class (e.g. "Healthy"). In binary mode, the Stage 2 OvR classifier
            uses the disease (non-reference) class's features. None for multiclass.
        verbose : Verbosity level.
        """
        if locus not in ("TCR", "BCR"):
            raise ValueError(f"locus must be 'TCR' or 'BCR', got '{locus}'")
        self.locus = locus
        self.aggregation_strategy = aggregation_strategy
        self.entropy_max_fraction = entropy_max_fraction
        self.entropy_bottom_percentile = entropy_bottom_percentile
        self.exclude_rare_v_genes = exclude_rare_v_genes
        self.min_sequences_per_group = min_sequences_per_group
        self.reweigh_by_subset_frequencies = reweigh_by_subset_frequencies
        self.n_estimators_stage1 = n_estimators_stage1
        self.n_estimators_stage2 = n_estimators_stage2
        self.n_jobs = n_jobs
        self.reference_class = reference_class
        self.verbose = verbose

        # Set after fit_stage1
        self.group_models_: Dict[Tuple, GroupSequenceClassifier] = {}
        self.classes_: Optional[np.ndarray] = None
        self.non_rare_v_genes_: Optional[List[str]] = None

        # Set after fit_stage2
        self.stage2_clf_: Optional[BinaryOvRClassifierWithFeatureSubsettingByClass] = None
        self.stage2_scaler_: Optional[StandardScaler] = None
        self.feature_columns_: Optional[List[str]] = None
        # Absolute entropy threshold (in nats) learned from training data.
        # Only set when aggregation_strategy is entropy_percentile_cutoff.
        self.entropy_percentile_threshold_: Optional[float] = None
        # Per-(specimen, group) entropy filter survival stats from the last
        # featurize_specimens call. DataFrame with columns: specimen_label,
        # group, total_sequences, survived_sequences, survival_pct.
        # None when aggregation strategy doesn't use entropy filtering.
        self.last_entropy_survival_stats_: Optional[pd.DataFrame] = None

        # For reweigh_by_subset_frequencies: pre-aggregation scaler
        self.preagg_scaler_: Optional[StandardScaler] = None

    # ------------------------------------------------------------------ #
    # Group key helpers                                                    #
    # ------------------------------------------------------------------ #

    def _split_on_cols(self) -> List[str]:
        """Columns used to define V-gene groups.

        BCR: (v_gene, isotype_supergroup) — VGeneIsotypeSpecificSequenceClassifier
        TCR: (v_gene,)

        Reference: vj_gene_specific_sequence_classifier.py:286-293
        """
        if self.locus == "BCR":
            return [V_GENE_COL, ISOTYPE_COL]
        return [V_GENE_COL]

    def _get_group_key(self, row: pd.Series) -> Tuple:
        cols = self._split_on_cols()
        return tuple(row[c] for c in cols)

    def _get_group_keys_series(self, sequences_df: pd.DataFrame) -> pd.Series:
        """Return a Series of group key tuples aligned with sequences_df.

        Each row gets a tuple identifying its V-gene group:
          TCR: ("TRBV5-1",)
          BCR: ("IGHV3-23", "IGHG")
        """
        # TCR: [v_gene], BCR: [v_gene, isotype_supergroup]
        cols = self._split_on_cols()

        missing = [c for c in cols if c not in sequences_df.columns]
        if missing:
            raise ValueError(
                f"Group key columns missing from sequences_df: {missing}. "
                f"Available columns: {list(sequences_df.columns)[:15]}"
            )

        # Wrap single-column keys in a tuple for consistent (tuple,) format
        if len(cols) == 1:
            return sequences_df[cols[0]].apply(lambda v: (v,))
        return sequences_df[cols].apply(tuple, axis=1)

    def _sanitize_group_columns(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Replace '_' with '-' in V-gene and isotype column values.

        Feature column names use '_' as the delimiter between class name and
        group key (e.g. "Covid19_TRBV5-1"). If V-gene or isotype values
        themselves contain '_', the delimiter becomes ambiguous and downstream
        parsing in _groups_from_feature_columns / _compute_subset_frequencies
        would break.

        Standard IMGT gene names use hyphens (TRBV5-1, IGHV3-23) and isotype
        names (IGHG, IGHA, IGHD-M) don't contain underscores, so this is a
        defensive measure for non-standard data.
        """
        for col in self._split_on_cols():
            if col in sequences_df.columns:
                has_underscore = sequences_df[col].str.contains("_", na=False)
                if has_underscore.any():
                    n = has_underscore.sum()
                    logger.warning(
                        f"  {n} values in column '{col}' contain '_', which "
                        f"conflicts with the feature column delimiter. "
                        f"Replacing '_' with '-' in these values."
                    )
                    sequences_df[col] = sequences_df[col].str.replace("_", "-", regex=False)
        return sequences_df

    def _group_key_to_str(self, key: Tuple) -> str:
        """Convert group key tuple to string for use in column names."""
        return "_".join(str(k) for k in key)

    # ------------------------------------------------------------------ #
    # Stage 1: per-group classifiers                                       #
    # ------------------------------------------------------------------ #

    def _make_stage1_clf(self) -> object:
        """Return a new unfitted Stage 1 classifier for this locus.

        BCR: RandomForestClassifier
             Reference: malid/config.py:94-96, model_definitions.py:431-540
        TCR: CustomOneVsRestClassifier(GlmnetLogitNetWrapper(alpha=0.0))
             Reference: malid/config.py:94-96 ('ridge_cv_ovr')
             Uses glmnet ridge (L2, alpha=0.0) per binary classifier.
             The custom OvR passes groups (participant_label) through to
             GlmnetLogitNetWrapper for participant-level internal CV.
        """
        if self.locus == "BCR":
            return RandomForestClassifier(
                n_estimators=self.n_estimators_stage1,
                **_RF_STAGE1_BCR_CONFIG,
            )
        elif self.locus == "TCR":
            # TCR: ridge_cv_ovr via glmnet with participant-level CV grouping.
            # Uses StratifiedGroupKFold so sequences from the same participant
            # stay in the same internal CV fold during glmnet's lambda selection.
            # Original: StratifiedGroupKFoldRequiresGroups (separate package);
            # we use sklearn's StratifiedGroupKFold which has the same behavior.
            # Reference: malid/train/training_utils.py:56
            from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper
            from sklearn.model_selection import StratifiedGroupKFold
            return CustomOneVsRestClassifier(
                estimator=GlmnetLogitNetWrapper(
                    alpha=_GLMNET_ALPHA,
                    n_splits=_GLMNET_CV_N_SPLITS,
                    standardize=_GLMNET_STANDARDIZE,
                    use_lambda_1se=_GLMNET_USE_LAMBDA_1SE,
                    n_lambda=_GLMNET_N_LAMBDA,
                    class_weight=_GLMNET_CLASS_WEIGHT,
                    require_cv_group_labels=True,  # enforce groups passed
                    internal_cv=StratifiedGroupKFold(
                        n_splits=_GLMNET_CV_N_SPLITS,
                        shuffle=True,
                        random_state=_GLMNET_RANDOM_STATE,
                    ),
                ),
                normalize_predicted_probabilities=_OVR_NORMALIZE_PROBABILITIES,
                allow_some_classes_to_fail_to_train=_OVR_ALLOW_FAILURE,
                # n_jobs=1: binary classifiers run sequentially within each group
                # because parallelism is at the V-gene group level instead (fit_stage1).
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unexpected locus '{self.locus}' in _make_stage1_clf")

    def _get_sample_weights(self, sequences_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return per-sequence sample weights for Stage 1 fitting.

        BCR: sample_weight_isotype_rebalance column (if present).
        TCR: None (uniform).

        Reference: malid/trained_model_wrappers/sequence_classifier.py:210-253
        """
        if self.locus == "BCR" and "sample_weight_isotype_rebalance" in sequences_df.columns: # TODO: revisit when adding BCR to Model 3. is it ok to not have sample_weight_isotype_rebalance_col?
            w = sequences_df["sample_weight_isotype_rebalance"].values.astype(float)
            if np.isnan(w).all():
                return None
            w = np.where(np.isnan(w), 1.0, w)
            return w
        return None

    def _build_features(self, sequences_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """Build per-sequence feature matrix for Stage 1.

        BCR: 640 (ESM-2) + 1 (v_mut) = 641 features
        TCR: 640 (ESM-2) = 640 features

        V gene and isotype dummy variables are NOT included: they are constant
        within each group.

        Reference: malid/trained_model_wrappers/sequence_classifier.py:144-208
        """
        if self.locus == "BCR" and V_MUT_COL in sequences_df.columns: # TODO: revisit when adding BCR to Model 3. is it ok to not have v_mut_col?
            v_mut = sequences_df[V_MUT_COL].fillna(0).values.reshape(-1, 1).astype(np.float32)
            return np.hstack([embeddings.astype(np.float32), v_mut])
        return embeddings.astype(np.float32)

    def fit_stage1(
        self,
        sequences_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> "SequenceLevelClassifier":
        """Train per-V-gene-group classifiers on train_smaller1.

        Parameters
        ----------
        sequences_df : Sequences DataFrame (train_smaller1) with columns:
            v_gene, [isotype_supergroup], [disease], participant_label,
            specimen_label, [v_mut for BCR].
        embeddings   : (n_sequences, 640) float32 ESM-2 embeddings, aligned
                       row-by-row with sequences_df.

        Reference: malid/train/train_vj_gene_specific_sequence_model.py
        """
        sequences_df = sequences_df.copy().reset_index(drop=True)
        assert len(sequences_df) == len(embeddings), (
            f"sequences_df length ({len(sequences_df)}) != embeddings length ({len(embeddings)})"
        )

        # Sanitize V-gene/isotype values: replace '_' with '-' to avoid
        # ambiguity with the '_' delimiter in feature column names.
        sequences_df = self._sanitize_group_columns(sequences_df)

        self.classes_ = np.array(sorted(sequences_df[DISEASE_COL].unique()))

        # Validate that no class name is a prefix of another when followed by "_".
        # Column names use the format "{class}_{group_key}" and are matched via
        # startswith("{class}_"), so e.g. classes ["A", "A_B"] would cause "A_"
        # to incorrectly match columns belonging to "A_B".
        class_prefixes = [f"{c}_" for c in self.classes_]
        for i, prefix_i in enumerate(class_prefixes):
            for j, prefix_j in enumerate(class_prefixes):
                if i != j and prefix_j.startswith(prefix_i):
                    raise ValueError(
                        f"Class name '{self.classes_[i]}' is a prefix of "
                        f"'{self.classes_[j]}' when using '_' as delimiter. "
                        f"This would cause ambiguous feature column matching "
                        f"in Stage 2. Class names: {list(self.classes_)}"
                    )

        y = sequences_df[DISEASE_COL].values

        # Rare V gene filtering: keep only V genes with above-median prevalence
        if self.exclude_rare_v_genes:
            self.non_rare_v_genes_ = find_non_rare_v_genes(sequences_df)
            sequences_df = sequences_df[
                sequences_df[V_GENE_COL].isin(self.non_rare_v_genes_)
            ].copy()
            # After boolean filtering, .index holds original row positions →
            # use as numpy fancy index to select matching embedding rows
            embeddings = embeddings[sequences_df.index]
            y = sequences_df[DISEASE_COL].values
            # Reset index to 0..N-1 so it aligns with the new embeddings array
            sequences_df = sequences_df.reset_index(drop=True)

        # Per-sequence group key: (v_gene,) for TCR, (v_gene, isotype) for BCR
        group_keys = self._get_group_keys_series(sequences_df)
        # Feature matrix: ESM-2 embeddings (+ v_mut for BCR)
        features = self._build_features(sequences_df, embeddings)
        # BCR: isotype rebalancing weights (TODO: revisit when adding BCR to Model 3); TCR: None (uniform)
        sample_weights = self._get_sample_weights(sequences_df)
        # Participant labels for CV grouping inside inner classifiers
        # (e.g. GlmnetLogitNetWrapper uses these for StratifiedGroupKFold).
        # Reference: malid/train/train_vj_gene_specific_sequence_model.py:344-376
        if PARTICIPANT_COL not in sequences_df.columns:
            raise ValueError(
                f"Column '{PARTICIPANT_COL}' required for participant-level CV grouping "
                f"but not found. Available: {list(sequences_df.columns)[:15]}"
            )
        participant_labels = sequences_df[PARTICIPANT_COL].values

        unique_groups = group_keys.unique()
        if self.verbose >= 1:
            logger.info(
                f"  Stage 1: training models for {len(unique_groups)} V-gene groups "
                f"(n_jobs={self.n_jobs})..."
            )

        # --- Prepare per-group data: scale + binarize (sequential, fast) ---
        # For TCR, flatten all binary sub-problems across groups into one list
        # for a single Parallel call. This gives ~162 jobs (27 groups × 6 classes)
        # instead of 27, with much better load balancing.
        # For BCR, keep group-level parallelism (RandomForest, no inner OvR).
        uses_ovr = self.locus == "TCR"
        n_skipped_prefilter = 0
        # verbose >= 2: collect per-group training stats (seq count + class distribution)
        _group_train_stats: Dict[tuple, Dict] = {}

        # Check if Stage 1 classifier is an OvR wrapper (real TCR) or not (BCR / test mocks).
        # Flattened parallelism only applies to OvR classifiers.
        probe_clf = self._make_stage1_clf()
        use_flattened = uses_ovr and isinstance(probe_clf, CustomOneVsRestClassifier)

        # Memory optimization: free the DataFrame and raw embeddings early.
        # Everything needed for training has been extracted into numpy arrays:
        # features, y, group_keys, sample_weights, participant_labels.
        # For TCR, features IS the embeddings array (same object, float32), so
        # `del embeddings` only drops a reference — memory is held by features.
        # But `del sequences_df` frees the DataFrame's string columns (~5-8 GB).
        del sequences_df, embeddings

        if use_flattened:
            # ============================================================ #
            # TCR: Flattened parallelism over (group, class) binary jobs    #
            # ============================================================ #
            #
            # Why flattened: The non-flattened path parallelizes over V-gene
            # groups (~28 jobs), but each group trains 6 binary classifiers
            # sequentially. Large groups (millions of sequences) take 20+ min
            # while small groups finish in seconds, leaving workers idle.
            # Flattened parallelism decomposes ALL groups into individual
            # (group, class) binary jobs (~168 total = 28 groups × 6 classes),
            # submitted to a single Parallel call for optimal load balancing.
            #
            # Equivalence guarantee: This produces the exact same fitted
            # GroupSequenceClassifier objects as the non-flattened path. Each
            # step below corresponds to what GroupSequenceClassifier.fit and
            # CustomOneVsRestClassifier.fit do internally:
            #   - StandardScaler per group (same as GroupSequenceClassifier.fit)
            #   - LabelBinarizer per group (same as OvR.fit steps 1-2)
            #   - clone(estimator).fit per binary job (same as OvR.fit step 3)
            #   - Reassembly mirrors OvR.fit steps 4-5
            #
            # Memory lifecycle:
            #   Before loop: features (~14 GB) + small arrays (~0.3 GB)
            #   During loop: features + accumulated X_scaled copies (→ ~28 GB)
            #                + one transient group slice (~1 GB, freed each iter)
            #   After loop:  del features → X_scaled copies only (~14 GB)
            #   During Parallel: main process ~14 GB + 2 workers ~8 GB each
            #
            # ============================================================ #

            # --- Step 1: Build flat job list (sequential) ---
            # For each group: fit StandardScaler, binarize labels, create one
            # job per binary sub-problem. This replicates what happens inside
            # GroupSequenceClassifier.fit + CustomOneVsRestClassifier.fit
            # before the actual classifier training.

            group_scalers: Dict[tuple, StandardScaler] = {}
            group_classes_map: Dict[tuple, np.ndarray] = {}  # group_key → sorted class labels
            flat_jobs = []
            allow_failure = _OVR_ALLOW_FAILURE
            # Prototype estimator: each binary job gets an independent clone
            base_estimator = probe_clf.estimator  # GlmnetLogitNetWrapper

            for group_key in unique_groups:
                mask = (group_keys == group_key).values
                group_features, group_labels = features[mask], y[mask]

                if len(group_features) < self.min_sequences_per_group or len(np.unique(group_labels)) < 2:
                    n_skipped_prefilter += 1
                    continue

                group_sample_weights = sample_weights[mask] if sample_weights is not None else None
                group_participant_labels = participant_labels[mask]

                # Collect per-group training stats for verbose >= 2 diagnostics
                if self.verbose >= 2:
                    unique_labels, label_counts = np.unique(group_labels, return_counts=True)
                    _group_train_stats[group_key] = {
                        "n_sequences": len(group_labels),
                        "class_distribution": dict(zip(unique_labels.tolist(), label_counts.tolist())),
                    }

                # Fit StandardScaler on this group's raw embeddings.
                # (= GroupSequenceClassifier.fit: self.scaler.fit_transform(X))
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(group_features)
                del group_features  # free unscaled copy (~0.5-2 GB)
                group_scalers[group_key] = scaler

                # Binarize multiclass labels into binary sub-problems.
                # (= CustomOneVsRestClassifier.fit steps 1-2)
                # LabelBinarizer sorts classes alphabetically.
                # Binary (2 classes): 1 column, indicator for classes_[1].
                # Multiclass (K classes): K columns, column i = indicator for classes_[i].
                lb = LabelBinarizer(sparse_output=True)
                Y = lb.fit_transform(group_labels).tocsc()
                group_classes = lb.classes_
                group_classes_map[group_key] = group_classes
                is_binary = len(group_classes) == 2
                columns = [col.toarray().ravel() for col in Y.T]

                if is_binary:
                    # Single indicator column: 1 = classes_[1], 0 = classes_[0]
                    binary_jobs = [(columns[0], group_classes[1], group_classes[0])]
                else:
                    # K columns: column i → classes_[i] vs rest
                    binary_jobs = [
                        (columns[i], group_classes[i], f"not {group_classes[i]}")
                        for i in range(len(columns))
                    ]

                # Append one flat job per binary sub-problem.
                # All binary jobs for the same group share the same X_scaled
                # array reference (no duplication within a group).
                for class_idx, (y_bin, pos_cls, neg_cls) in enumerate(binary_jobs):
                    flat_jobs.append((
                        group_key, class_idx,
                        X_scaled, y_bin, pos_cls, neg_cls,
                        group_sample_weights, group_participant_labels,
                        sklearn.base.clone(base_estimator),
                        allow_failure,
                    ))

            # Free the full-size arrays. Only the per-group X_scaled copies
            # (stored in flat_jobs) remain in memory (~14 GB total).
            del features, y, participant_labels, sample_weights, group_keys

            if self.verbose >= 1:
                logger.info(
                    f"  Stage 1: {len(flat_jobs)} binary classifier jobs "
                    f"across {len(group_scalers)} groups (flattened parallelism)"
                )

            # --- Step 2: Train all binary classifiers in parallel ---
            # Each job: clone(GlmnetLogitNetWrapper).fit(X_scaled, y_binary, ...)
            # (= CustomOneVsRestClassifier.fit step 3, but across ALL groups)
            # verbose=10: prints "[Parallel]: Done X out of 168" per completed job.
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10 if self.verbose >= 1 else 0)(
                delayed(_fit_one_binary_ovr_job)(*job) for job in flat_jobs
            )

            # --- Step 3: Reassemble into GroupSequenceClassifier objects ---
            # Each result is (group_key, class_idx, pos_cls, neg_cls, fitted_clf)
            # or None on failure. We group by group_key and reconstruct the
            # CustomOneVsRestClassifier state that predict_proba expects.
            # (= CustomOneVsRestClassifier.fit steps 4-5)

            # 3a. Group results by V-gene group
            group_binary_results: Dict[tuple, List] = defaultdict(list)
            n_binary_failed = 0
            for result in results:
                if result is not None:
                    gk, cls_idx, pos_cls, neg_cls, fitted_clf = result
                    group_binary_results[gk].append(
                        _InnerEstimator(clf=fitted_clf, positive_class=pos_cls, negative_class=neg_cls)
                    )
                else:
                    n_binary_failed += 1

            # 3b. Build GroupSequenceClassifier for each group
            n_failed = 0
            for group_key in group_scalers:
                estimators = group_binary_results.get(group_key, [])
                if not estimators:
                    # All binary classifiers failed for this group — skip it.
                    # (= _train_one_group returning None)
                    n_failed += 1
                    if n_binary_failed > 0:
                        logger.warning(
                            f"  Stage 1: all binary classifiers failed for group {group_key}"
                        )
                    continue

                group_classes = group_classes_map[group_key]
                is_binary = len(group_classes) == 2

                # Create a fresh GroupSequenceClassifier and inject fitted state.
                # _make_stage1_clf() provides a new CustomOneVsRestClassifier with
                # correct settings (normalize_probs, allow_failure). We bypass its
                # .fit() and set the fitted attributes directly.
                gsc = GroupSequenceClassifier(self._make_stage1_clf())
                gsc.scaler = group_scalers[group_key]
                ovr = gsc.clf  # the CustomOneVsRestClassifier instance

                if is_binary:
                    # Binary: classes_ = both classes from LabelBinarizer (sorted).
                    # One estimator predicts P(classes_[1]).
                    # (= OvR.fit step 4, binary branch: classes_ = lb.classes_)
                    ovr.classes_ = group_classes
                    ovr.estimators_ = estimators
                else:
                    # Multiclass: only keep classes that trained successfully,
                    # sorted for deterministic ordering.
                    # (= OvR.fit step 4, multiclass branch: sorted trained classes)
                    trained_classes = sorted([e.positive_class for e in estimators])
                    ovr.classes_ = np.array(trained_classes)
                    est_by_class = {e.positive_class: e for e in estimators}
                    ovr.estimators_ = [est_by_class[cls] for cls in ovr.classes_]

                # Copy sklearn metadata from first estimator.
                # (= OvR.fit step 5)
                first_clf = ovr.estimators_[0].clf
                if hasattr(first_clf, "n_features_in_"):
                    ovr.n_features_in_ = first_clf.n_features_in_
                if hasattr(first_clf, "feature_names_in_"):
                    ovr.feature_names_in_ = first_clf.feature_names_in_

                gsc.classes_ = ovr.classes_
                self.group_models_[group_key] = gsc

        else:
            # --- Group-level parallelism (BCR RandomForest or non-OvR TCR) ---
            jobs = []
            for group_key in unique_groups:
                mask = (group_keys == group_key).values
                group_features, group_labels = features[mask], y[mask]

                if len(group_features) < self.min_sequences_per_group or len(np.unique(group_labels)) < 2:
                    n_skipped_prefilter += 1
                    continue

                group_sample_weights = sample_weights[mask] if sample_weights is not None else None
                group_participant_labels = participant_labels[mask] if uses_ovr else None

                # Collect per-group training stats for verbose >= 2 diagnostics
                if self.verbose >= 2:
                    unique_labels, label_counts = np.unique(group_labels, return_counts=True)
                    _group_train_stats[group_key] = {
                        "n_sequences": len(group_labels),
                        "class_distribution": dict(zip(unique_labels.tolist(), label_counts.tolist())),
                    }

                clf = GroupSequenceClassifier(self._make_stage1_clf())
                jobs.append((group_key, group_features, group_labels,
                             group_sample_weights, group_participant_labels, clf))

            del features, y, participant_labels, sample_weights, group_keys

            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10 if self.verbose >= 1 else 0)(
                delayed(_train_one_group)(
                    group_key, group_features, group_labels,
                    group_sample_weights, group_participant_labels, clf,
                )
                for group_key, group_features, group_labels,
                    group_sample_weights, group_participant_labels, clf in jobs
            )

            n_failed = 0
            for result in results:
                if result is not None:
                    group_key, clf = result
                    self.group_models_[group_key] = clf
                else:
                    n_failed += 1

        n_trained = len(self.group_models_)
        n_skipped = n_skipped_prefilter + n_failed
        if self.verbose >= 1:
            logger.info(
                f"  Stage 1: trained {n_trained}/{len(unique_groups)} group models "
                f"(skipped {n_skipped}: {n_skipped_prefilter} pre-filtered "
                f"[< {self.min_sequences_per_group} seqs or < 2 classes], "
                f"{n_failed} fit errors)"
            )

        # Diagnostic #2 + #3: per-group details (verbose >= 2)
        if self.verbose >= 2:
            self._log_stage1_group_diagnostics(_group_train_stats)

        return self

    # ------------------------------------------------------------------ #
    # Verbose >= 2 diagnostics                                            #
    # ------------------------------------------------------------------ #

    def _log_stage1_group_diagnostics(
        self,
        group_train_stats: Optional[Dict[tuple, Dict]] = None,
    ) -> None:
        """Log per-group details after Stage 1 training or loading.

        Includes:
        - #2: Which classes each group's classifier covers (and which are missing).
        - #3: Per-group sequence counts and class distribution (only when
              group_train_stats is provided, i.e. Stage 1 was trained, not loaded).

        Can be called after fit_stage1 or after load_stage1_artifacts.
        """
        if not self.group_models_:
            return

        all_classes = set(str(c) for c in self.classes_)
        logger.info(f"\n  [verbose=2] Stage 1 group diagnostics ({len(self.group_models_)} groups):")

        for gk in sorted(self.group_models_, key=str):
            gsc = self.group_models_[gk]
            # Classes this group's classifier covers
            trained_classes = set(str(c) for c in gsc.classes_) if gsc.classes_ is not None else set()
            missing_classes = sorted(all_classes - trained_classes)

            parts = [f"    {gk}: classes={sorted(trained_classes)}"]
            if missing_classes:
                parts.append(f"missing={missing_classes}")

            # Training stats (only available when Stage 1 was trained, not loaded)
            if group_train_stats and gk in group_train_stats:
                stats = group_train_stats[gk]
                parts.append(f"n_seqs={stats['n_sequences']:,}")
                dist_str = ", ".join(
                    f"{cls}={cnt:,}" for cls, cnt in sorted(stats["class_distribution"].items())
                )
                parts.append(f"dist=[{dist_str}]")

            logger.info("  ".join(parts))

    def _compute_entropy_percentile_threshold(
        self,
        seq_preds: pd.DataFrame,
    ) -> None:
        """Compute and store the absolute entropy threshold from training data.

        Calculates per-sequence entropy for all valid (has_prediction=True)
        sequences in the training set, then takes the entropy_bottom_percentile-th
        percentile as the absolute threshold. This threshold is stored in
        entropy_percentile_threshold_ and used by _entropy_abs_threshold_aggregate
        during featurize_specimens.

        Parameters
        ----------
        seq_preds : Output of generate_sequence_predictions() on training data.
        """
        prob_cols = [f"prob_{c}" for c in self.classes_]
        valid = seq_preds[seq_preds["has_prediction"]]

        if len(valid) == 0:
            raise ValueError(
                "No valid sequences with predictions in training data. "
                "Cannot compute entropy percentile threshold."
            )

        # Compute entropy for all valid training sequences
        probs = valid[prob_cols].values
        # scipy.stats.entropy along axis=0 on transposed array → per-sequence entropy
        all_entropies = scipy.stats.entropy(probs.T)

        # Compute the percentile threshold
        self.entropy_percentile_threshold_ = float(
            np.percentile(all_entropies, self.entropy_bottom_percentile)
        )

        if self.verbose >= 1:
            n_below = int((all_entropies < self.entropy_percentile_threshold_).sum())
            max_entropy = scipy.stats.entropy(np.ones(len(self.classes_)) / len(self.classes_))
            logger.info(
                f"  Entropy percentile threshold: {self.entropy_percentile_threshold_:.6f} nats "
                f"(percentile {self.entropy_bottom_percentile}% of {len(all_entropies):,} "
                f"training sequences, {self.entropy_percentile_threshold_ / max_entropy:.2%} "
                f"of max entropy, {n_below:,} sequences below threshold)"
            )

    def _log_entropy_filter_stats(
        self,
        filter_stats: Dict[tuple, Dict],
    ) -> None:
        """Log entropy filter survival rates per V-gene group.

        Parameters
        ----------
        filter_stats : {group_key: {"total": int, "survived": int}} aggregated
            across all specimens.
        """
        if not filter_stats:
            return
        logger.info(f"\n  [verbose=2] Entropy filter survival rates per group:")
        for gk in sorted(filter_stats, key=str):
            s = filter_stats[gk]
            total = s["total"]
            survived = s["survived"]
            pct = 100.0 * survived / total if total > 0 else 0.0
            logger.info(f"    {gk}: {survived:,}/{total:,} ({pct:.1f}%) sequences passed filter")

    def _log_stage2_feature_importance(self, top_n: int = 20) -> None:
        """Log top N most important features from Stage 2 RandomForest classifiers.

        Each binary classifier in the Stage 2 OvR has its own feature importance
        over its subset of features. We report the top features per class.
        """
        if self.stage2_clf_ is None:
            return
        logger.info(f"\n  [verbose=2] Stage 2 feature importance (top {top_n} per class):")
        for cls_name, clf in self.stage2_clf_.classifiers_.items():
            feat_cols = self.stage2_clf_.feature_subsets_.get(cls_name, [])
            if not feat_cols or not hasattr(clf, "feature_importances_"):
                continue
            importances = clf.feature_importances_
            # Sort by importance descending
            indices = np.argsort(importances)[::-1][:top_n]
            logger.info(f"    {cls_name}:")
            for idx in indices:
                logger.info(f"      {feat_cols[idx]}: {importances[idx]:.4f}")

    def _log_prediction_confidence_stats(
        self,
        proba_df: pd.DataFrame,
        context: str = "test",
    ) -> None:
        """Log prediction confidence statistics.

        Reports per-class mean/std of predicted probabilities and the fraction
        of specimens with low-confidence predictions (max prob < 0.5).
        """
        proba_vals = proba_df.values
        n_specimens = len(proba_df)
        max_probs = proba_vals.max(axis=1)
        low_conf = (max_probs < 0.5).sum()
        low_conf_pct = 100.0 * low_conf / n_specimens if n_specimens > 0 else 0.0

        logger.info(f"\n  [verbose=2] Prediction confidence ({context}, {n_specimens} specimens):")
        logger.info(
            f"    Max prob across classes: "
            f"mean={max_probs.mean():.4f}, std={max_probs.std():.4f}, "
            f"min={max_probs.min():.4f}, max={max_probs.max():.4f}"
        )
        logger.info(f"    Low-confidence (max prob < 0.5): {low_conf}/{n_specimens} ({low_conf_pct:.1f}%)")
        logger.info(f"    Per-class probability stats:")
        for i, cls in enumerate(proba_df.columns):
            col_vals = proba_vals[:, i]
            logger.info(
                f"      {cls}: mean={col_vals.mean():.4f}, std={col_vals.std():.4f}, "
                f"min={col_vals.min():.4f}, max={col_vals.max():.4f}"
            )

    def _log_reweighing_stats(
        self,
        freq_df: pd.DataFrame,
        context: str = "train",
    ) -> None:
        """Log V-gene group frequency statistics used for reweighing.

        Reports min/median/max frequency across specimens for each group,
        so the user can see if any groups dominate or are near-zero.
        """
        # Recover group keys from column names — all classes for the same group
        # share the same frequency, so pick one class prefix to get unique groups.
        if self.classes_ is None or len(self.classes_) == 0:
            return
        first_class = str(self.classes_[0])
        prefix = f"{first_class}_"
        group_cols = [c for c in freq_df.columns if c.startswith(prefix)]

        if not group_cols:
            return

        logger.info(f"\n  [verbose=2] Reweighing frequency stats ({context}, {len(freq_df)} specimens):")
        for col in sorted(group_cols):
            gk_str = col[len(prefix):]
            vals = freq_df[col].values
            logger.info(
                f"    {gk_str}: min={vals.min():.4f}, median={np.median(vals):.4f}, "
                f"max={vals.max():.4f}, mean={vals.mean():.4f}"
            )

    # ------------------------------------------------------------------ #
    # Sequence-level inference                                            #
    # ------------------------------------------------------------------ #

    def generate_sequence_predictions(
        self,
        sequences_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Run Stage 1 on all sequences and return per-sequence probability DataFrame.

        Sequences in groups without a trained model receive NaN probabilities.

        Parameters
        ----------
        sequences_df : DataFrame aligned with embeddings.
        embeddings   : (n_sequences, 640) ESM-2 embeddings. #TODO: when adding BCR to Model 3: isn't this 641?

        Returns
        -------
        DataFrame with columns:
            prob_<class1>, prob_<class2>, ...   (float, NaN if no model)
            specimen_label, participant_label
            <split_on_cols>
            weight                              (sample weight or 1.0)
            has_prediction                      (bool)

        Reference: vj_gene_specific_sequence_classifier.py:127-202
        """
        sequences_df = sequences_df.copy().reset_index(drop=True)
        assert len(sequences_df) == len(embeddings)

        # Sanitize V-gene/isotype values to match training (same '_' → '-' replacement)
        sequences_df = self._sanitize_group_columns(sequences_df)

        n = len(sequences_df)
        n_classes = len(self.classes_)
        prob_cols = [f"prob_{c}" for c in self.classes_]

        # Initialize all probabilities to NaN (groups without a model stay NaN)
        seq_probs = np.full((n, n_classes), np.nan, dtype=np.float32)
        group_keys = self._get_group_keys_series(sequences_df)
        features = self._build_features(sequences_df, embeddings) # TCR: embeddings, BCR: embeddings + v_mut

        # Run each trained group model on its sequences.
        # Pre-compute boolean masks per group to avoid redundant comparisons.
        group_items = []
        for gk, clf in self.group_models_.items():
            mask = (group_keys == gk).values
            if mask.sum() > 0:
                group_items.append((gk, clf, mask))

        if self.n_jobs == 1 or len(group_items) <= 1:
            # Sequential path (n_jobs=1 or single group)
            for g_idx, (gk, clf, mask) in enumerate(group_items, 1):
                if self.verbose >= 1 and (g_idx % 5 == 0 or g_idx == 1 or g_idx == len(group_items)):
                    logger.info(f"  Predicting group {g_idx}/{len(group_items)}: {gk} ({mask.sum():,} sequences)")
                probs = clf.predict_proba(features[mask], self.classes_)
                seq_probs[mask] = probs
        else:
            # Parallel path: each group's predict_proba is independent and
            # writes to non-overlapping rows of seq_probs. Use threading
            # backend because predict is read-only (no model mutation) and
            # avoids serializing models+features across processes.
            if self.verbose >= 1:
                logger.info(
                    f"  Predicting {len(group_items)} groups in parallel "
                    f"(n_jobs={self.n_jobs}, threading backend)..."
                )

            def _predict_group(gk, clf, mask):
                """Predict one group and scatter results into shared seq_probs."""
                probs = clf.predict_proba(features[mask], self.classes_)
                seq_probs[mask] = probs

            Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(_predict_group)(gk, clf, mask)
                for gk, clf, mask in group_items
            )

        # Assemble result DataFrame with probabilities + metadata for aggregation
        result = pd.DataFrame(seq_probs, columns=prob_cols)
        result[SPECIMEN_COL] = sequences_df[SPECIMEN_COL].values
        if PARTICIPANT_COL not in sequences_df.columns:
            raise ValueError(
                f"Column '{PARTICIPANT_COL}' required in sequences_df "
                f"but not found. Available: {list(sequences_df.columns)[:15]}"
            )
        result[PARTICIPANT_COL] = sequences_df[PARTICIPANT_COL].values
        for col in self._split_on_cols():
            if col in sequences_df.columns:
                result[col] = sequences_df[col].values

        # Sample weights for weighted aggregation in Stage 2
        sample_weights = self._get_sample_weights(sequences_df)
        result["weight"] = sample_weights if sample_weights is not None else 1.0

        # Flag sequences that received a prediction (vs NaN from missing group model)
        result["has_prediction"] = ~np.isnan(seq_probs[:, 0])
        return result

    # ------------------------------------------------------------------ #
    # Stage 2: specimen-level aggregation and rollup model                #
    # ------------------------------------------------------------------ #

    def featurize_specimens(
        self,
        seq_preds: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate per-sequence Stage 1 predictions to a specimen-level feature matrix.

        Each sequence has a probability vector over disease classes (from Stage 1).
        This method groups sequences by (specimen, group), applies the
        aggregation strategy (e.g. entropy-thresholded weighted mean) within each
        group, and concatenates the resulting vectors into one wide feature row
        per specimen.

        Grouping differs by locus:
        - TCR: split_on = ["v_gene"]. Each V-gene (e.g. "TRBV1") is one group.
          Column names: "{class}_{v_gene}", e.g. "COVID-19_TRBV1".
        - BCR: split_on = ["v_gene", "isotype_supergroup"]. Each (V-gene, isotype)
          pair is one group. Column names: "{class}_{v_gene}_{isotype}",
          e.g. "COVID-19_TRBV1_IGHG".

        For each (specimen, group):
        1. Select sequences belonging to that group within that specimen.
        2. Apply self.aggregation_strategy (e.g. entropy_cutoff, mean):
           for entropy strategies, filters out high-entropy (uncertain) sequences
           then computes a weighted mean of the survivors; for mean, averages all.
           Result: one (n_classes,) vector per group.
        3. If a group has no sequences for a specimen, fill with uniform
           prior (1/n_classes).

        The output is a wide DataFrame: one row per specimen, columns are
        "{class}_{group_key}" for every (class, group) combination.

        Missing value behavior:
        - No sequences for a group in a specimen: the group's columns are filled
          with uniform 1/n_classes (no information → assume equal probability).
        - Group exists in training but absent at test time: when feature_columns
          is provided, missing columns are added and filled with 1/n_classes.
        - No Stage 1 model for a group: sequences in that group have NaN
          probabilities (set in generate_sequence_predictions), and are excluded
          via the has_prediction filter before aggregation. If this removes all
          sequences for a (specimen, group), the first case above applies.
        - A group's classifier never saw a certain class during training (e.g.,
          no HIV sequences in TRBV5-1): that class gets probability 0 in
          GroupSequenceClassifier.predict_proba (zeros initialization, not
          1/n_classes). These zeros flow into the aggregation as real values.
          Matches original Mal-ID (reindex + fillna(0)).
        Note: BinaryOvRClassifierWithFeatureSubsettingByClass raises ValueError
        if any class has no matching features, so all classes are guaranteed to
        have trained classifiers at predict time.

        Parameters
        ----------
        seq_preds      : Output of generate_sequence_predictions(). Contains
                         per-sequence probabilities, specimen/participant labels,
                         split_on columns, sample weights, and has_prediction flag.
        feature_columns: If provided, align output to this column order and fill
                         missing groups with uniform prior. Used at test time to
                         match the training feature set.

        Returns
        -------
        DataFrame indexed by specimen_label.
        Columns = "{class}_{group_key}" for each (disease class, V-gene group).
        Shape: (n_specimens, n_classes * n_groups).

        Reference: original malid code:
          vj_gene_specific_sequence_model_rollup_classifier.py:626-749
          (_featurize_sequence_predictions)
          Fill value: 1/n_classes (line 455)
        """
        n_classes = len(self.classes_)
        prob_cols = [f"prob_{c}" for c in self.classes_]
        split_cols = [c for c in self._split_on_cols() if c in seq_preds.columns] # TCR: [v_gene], BCR: [v_gene, isotype_supergroup]

        # Drop sequences with no Stage 1 prediction (NaN = no model for their group)
        valid = seq_preds[seq_preds["has_prediction"]].copy()

        # Determine the set of V-gene groups to include as features
        if feature_columns is not None:
            # Test time: use the same groups that were seen during training
            all_groups = self._groups_from_feature_columns(feature_columns)
        else:
            # Train time: discover groups from the data.
            # Result is always a list of tuples, e.g. [("TRBV1",), ("TRBV2",)]
            # or [("TRBV1", "IGHG"), ...] for BCR with isotype.
            if len(valid) == 0:
                all_groups = []
            elif len(split_cols) > 1:
                # Multi-column groupby already returns tuple keys
                all_groups = sorted(valid.groupby(split_cols).groups.keys(), key=str)
            else:
                # Single-column groupby returns scalars; wrap in 1-tuples for consistency
                all_groups = sorted(
                    [(v,) for v in valid[split_cols[0]].unique()], key=str
                )

        # Track entropy filter survival per (specimen, group) pair.
        # Always enabled for entropy strategies — the per-specimen stats are stored
        # in self.last_entropy_survival_stats_ for artifact saving.
        # Uses return_survival_count in aggregate_group to piggyback on the
        # entropy computation already done inside _entropy_threshold_aggregate,
        # avoiding a duplicate O(n_sequences) pass.
        _uses_entropy = self.aggregation_strategy in (
            AggregationStrategy.entropy_cutoff,
            AggregationStrategy.entropy_ten_percent_cutoff,
            AggregationStrategy.entropy_twenty_percent_cutoff,
            AggregationStrategy.entropy_percentile_cutoff,
        )
        _track_survival = _uses_entropy
        # Per-group aggregate stats for verbose >= 2 logging
        _entropy_filter_stats: Dict[tuple, Dict] = {}  # {group_key: {"total": int, "survived": int}}
        # Per-(specimen, group) stats for artifact saving
        _per_specimen_survival: List[Dict] = []

        # --- Build specimen-level feature matrix via groupby ---
        # Instead of a nested loop (for specimen ... for group) with O(n_specimens *
        # n_sequences) boolean masking, use a single groupby to partition sequences
        # into (specimen, group) chunks in one O(n_sequences) pass.

        # All specimens in the input (including those with zero valid predictions)
        all_specimens = seq_preds[SPECIMEN_COL].unique()
        n_specimens = len(all_specimens)
        n_groups = len(all_groups)
        uniform = 1.0 / n_classes

        # Pre-allocate output array filled with uniform prior.
        # Shape: (n_specimens, n_classes * n_groups).
        # Missing (specimen, group) combinations stay at uniform.
        features_arr = np.full(
            (n_specimens, n_classes * n_groups), uniform, dtype=np.float64,
        )

        # Build index maps for fast scatter of aggregation results
        # specimen → row index in features_arr
        spec_to_row = {s: i for i, s in enumerate(all_specimens)}
        # group tuple → starting column index (each group occupies n_classes columns)
        gk_to_col = {gk: j * n_classes for j, gk in enumerate(all_groups)}

        # Build column names: "{class}_{group_key}" for each (group, class) combination
        col_names = []
        for gk in all_groups:
            gk_str = self._group_key_to_str(gk)
            for c in self.classes_:
                col_names.append(f"{c}_{gk_str}")

        # Single groupby: partitions valid sequences into (specimen, group) chunks.
        # TCR groups on [specimen_label, v_gene].
        # BCR groups on [specimen_label, v_gene, isotype_supergroup].
        groupby_cols = [SPECIMEN_COL] + split_cols

        for group_key, chunk in valid.groupby(groupby_cols, sort=False):
            # Extract specimen and group-key tuple from the composite groupby key.
            # TCR: group_key = (specimen, v_gene) → gk = (v_gene,)
            # BCR: group_key = (specimen, v_gene, isotype) → gk = (v_gene, isotype)
            specimen = group_key[0]
            gk = group_key[1:] if len(split_cols) > 1 else (group_key[1],)

            # Skip groups not in the target feature set (e.g. test-time group
            # absent from training). Their columns stay at uniform prior.
            col_start = gk_to_col.get(gk)
            if col_start is None:
                continue

            # Aggregate per-sequence probabilities → one (n_classes,) vector
            probs = chunk[prob_cols].values
            weights = chunk["weight"].values
            if np.isnan(weights).all():
                weights = None

            n_total_in_group = len(probs)
            agg_result = aggregate_group(
                probs, weights, self.aggregation_strategy, n_classes,
                entropy_max_fraction=self.entropy_max_fraction,
                entropy_abs_threshold=self.entropy_percentile_threshold_,
                return_survival_count=_track_survival,
            )
            if _track_survival:
                agg, n_survived = agg_result
                # Accumulate per-group aggregate stats (for verbose >= 2 logging)
                if gk not in _entropy_filter_stats:
                    _entropy_filter_stats[gk] = {"total": 0, "survived": 0}
                _entropy_filter_stats[gk]["total"] += n_total_in_group
                _entropy_filter_stats[gk]["survived"] += n_survived
                # Per-(specimen, group) row for artifact saving
                _per_specimen_survival.append({
                    "specimen_label": specimen,
                    "group": self._group_key_to_str(gk),
                    "total_sequences": n_total_in_group,
                    "survived_sequences": n_survived,
                    "survival_pct": (
                        100.0 * n_survived / n_total_in_group
                        if n_total_in_group > 0 else 0.0
                    ),
                })
            else:
                agg = agg_result

            # Scatter into pre-allocated array
            row = spec_to_row[specimen]
            features_arr[row, col_start:col_start + n_classes] = agg

        # Store per-specimen survival stats for artifact saving
        if _per_specimen_survival:
            self.last_entropy_survival_stats_ = pd.DataFrame(_per_specimen_survival)
        else:
            self.last_entropy_survival_stats_ = None

        # Diagnostic #1: entropy filter survival rates (verbose >= 2)
        if self.verbose >= 2 and _entropy_filter_stats:
            self._log_entropy_filter_stats(_entropy_filter_stats)

        # Detect specimens with zero valid sequences (all in rare/model-less V-genes).
        # These specimens have no groupby chunks, so their rows stay at uniform prior.
        specimens_with_preds = set(valid[SPECIMEN_COL].unique())
        no_prediction_specimens = [
            s for s in all_specimens if s not in specimens_with_preds
        ]

        # Warn about specimens with zero valid sequences
        if no_prediction_specimens:
            n_no_pred = len(no_prediction_specimens)
            pct = 100.0 * n_no_pred / n_specimens if n_specimens > 0 else 0.0
            logger.warning(
                f"  {n_no_pred}/{n_specimens} ({pct:.1f}%) specimen(s) have zero sequences "
                f"with Stage 1 predictions (all sequences belong to V-gene groups "
                f"without a trained model). This may indicate a data quality issue — "
                f"these specimens' predictions will be uninformative. "
                f"Specimens: {no_prediction_specimens[:10]}"
                + (f" (and {n_no_pred - 10} more)" if n_no_pred > 10 else "")
            )

        # Assemble into a wide DataFrame: rows = specimens, columns = "{class}_{group}"
        features_df = pd.DataFrame(
            features_arr, index=all_specimens, columns=col_names,
        )
        features_df.index.name = SPECIMEN_COL

        if feature_columns is not None:
            # Test time: align to training column order, fill missing groups with uniform
            for col in feature_columns:
                if col not in features_df.columns:
                    features_df[col] = uniform
            features_df = features_df[feature_columns]

        return features_df

    def _groups_from_feature_columns(self, feature_columns: List[str]) -> List[Tuple]:
        """Reconstruct group keys from feature column names at test time.

        Feature columns have the format "{class_name}_{group_key_str}",
        e.g., "COVID-19_TRBV5-1" (TCR) or "COVID-19_IGHV3-23_IGHG" (BCR).
        We strip the class prefix to recover the group key tuple.
        """
        groups = set()
        split_cols = self._split_on_cols()

        for col in feature_columns:
            # Try each class name as a prefix to find where the group key starts
            for c in self.classes_:
                prefix = f"{str(c)}_"
                if col.startswith(prefix):
                    gk_str = col[len(prefix):]

                    if len(split_cols) == 1:
                        # TCR: group key is just the V gene, e.g., "TRBV5-1"
                        groups.add((gk_str,))
                    else:
                        # BCR: group key is "{v_gene}_{isotype}".
                        # Can't naively split on "_" because V gene names
                        # could theoretically contain "_" (sanitized to "-"
                        # but defensive). Match known isotype suffixes instead.
                        known_isotypes = {"IGHG", "IGHA", "IGHD-M", "TCRB"}
                        for iso in known_isotypes:
                            if gk_str.endswith(f"_{iso}"):
                                v = gk_str[: -(len(iso) + 1)]
                                groups.add((v, iso))
                                break
                        else:
                            # Unknown isotype suffix — treat as single-component key
                            groups.add((gk_str,))
                    break  # found matching class prefix, move to next column

        return sorted(groups, key=str)

    def _compute_subset_frequencies(
        self,
        seq_preds: pd.DataFrame,
        feature_columns: List[str],
    ) -> pd.DataFrame:
        """Compute normalized per-(specimen, group) sequence frequency.

        Counts how many sequences each specimen has in each V-gene group, then
        normalizes to frequencies:
        - TCR: divide by total sequences across all groups → freqs sum to 1 per specimen.
        - BCR: divide by total sequences within each isotype → each isotype's freqs
          sum to 1 per specimen (so for 3 isotypes, the full row sums to 3).
          Rationale: isotype proportions are technical artifacts of sample prep,
          not biological signal; V-gene usage within an isotype IS biological.

        Returns a DataFrame with the same shape as the specimen feature matrix
        (rows = specimens, columns = feature_columns like "Covid19_TRBV5-1").
        Each feature column gets the frequency of its V-gene group — the same
        frequency is replicated across all class columns for the same group
        (e.g., "Covid19_TRBV5-1" and "Healthy_TRBV5-1" both get freq(TRBV5-1)).

        Reference: vj_gene_specific_sequence_model_rollup_classifier.py:460-588
        """
        split_cols = [c for c in self._split_on_cols() if c in seq_preds.columns]
        valid = seq_preds[seq_preds["has_prediction"]]
        all_groups = self._groups_from_feature_columns(feature_columns)

        # --- Step 1: Count sequences per (specimen, group) ---
        # Result: rows = specimens, columns = group keys, values = raw counts.
        # Example for TCR:
        #                 (TRBV5-1,)  (TRBV7-2,)
        #   specimen_S1          30          12
        #   specimen_S2           5          20
        counts = (
            valid
            .groupby([SPECIMEN_COL] + split_cols, observed=True)
            .size()
            .unstack(split_cols, fill_value=0)
        )

        # For TCR (single split_col), unstack produces scalar column labels like "TRBV5-1".
        # For BCR (two split_cols), unstack produces MultiIndex tuples like ("IGHV3-23", "IGHG").
        # Wrap scalars in 1-tuples so all_groups matching is consistent across loci.
        if len(split_cols) == 1:
            counts.columns = [(c,) for c in counts.columns]

        # Align to groups present in the feature matrix:
        # - Drop groups that appear in the data but not in the feature matrix
        #   (e.g., rare V-genes excluded during training).
        # - Add groups from the feature matrix that are absent in the data, filled with 0
        #   (e.g., a V-gene group this specimen has no sequences for).
        # Note: normalization is done AFTER dropping rare groups (matching original Mal-ID's
        # normalize_after_subsetting=True). The original code comments suggest a future improvement:
        # normalize over ALL genes first, then drop rare ones — this would better preserve
        # the distinction between uniform vs concentrated V-gene usage distributions.
        # Not yet implemented in original Mal-ID either; left as a future improvement. #TODO
        # Reference: vj_gene_specific_sequence_model_rollup_classifier.py:513-524
        counts = counts.reindex(columns=all_groups, fill_value=0)

        # --- Step 2: Normalize counts to frequencies ---
        if self.locus == "BCR" and len(split_cols) > 1:
            # BCR: normalize within each isotype so each isotype's freqs sum to 1 per specimen.
            # Example: if IGHG has groups [IGHV3-23, IGHV1-2] with counts [20, 10],
            # the normalized freqs are [2/3, 1/3]. IGHA groups are normalized separately.
            isotype_of_col = pd.Series(
                [gk[-1] for gk in counts.columns],
                index=range(len(counts.columns)),
            )
            freq = counts.copy().astype(float)
            for iso in isotype_of_col.unique():
                col_mask = (isotype_of_col == iso).values
                iso_sums = counts.iloc[:, col_mask].sum(axis=1)  # per-specimen total for this isotype
                # Divide each column in this isotype by the specimen's isotype total.
                # Where isotype total is 0 (specimen has no sequences for this isotype),
                # division produces NaN → filled to 0.0 below.
                freq.iloc[:, col_mask] = counts.iloc[:, col_mask].div(iso_sums, axis=0)
            freq = freq.fillna(0.0)
        else:
            # TCR: normalize across all groups so freqs sum to 1 per specimen.
            # Example: groups [TRBV5-1, TRBV7-2] with counts [30, 12] → [30/42, 12/42].
            row_sums = counts.sum(axis=1)
            # Where row sum is 0 (specimen has no sequences with predictions),
            # division produces NaN → filled to 0.0.
            freq = counts.div(row_sums, axis=0).fillna(0.0)

        # --- Step 3: Replicate group frequencies to match feature matrix columns ---
        # The feature matrix has columns like "Covid19_TRBV5-1", "Healthy_TRBV5-1", etc.
        # — one per (class, group) pair. The frequency for a group is the same regardless
        # of class: both "Covid19_TRBV5-1" and "Healthy_TRBV5-1" get freq(TRBV5-1) for
        # that specimen. This builds a DataFrame with the same shape as the feature matrix.
        # Example: freq(TRBV5-1) for S1 = 0.6 →
        #   freq_rows["Covid19_TRBV5-1"][S1]  = 0.6
        #   freq_rows["Healthy_TRBV5-1"][S1]  = 0.6
        gk_strs = {gk: self._group_key_to_str(gk) for gk in all_groups}
        freq_rows = {}
        for gk in all_groups:
            gk_str = gk_strs[gk]
            group_freq_series = freq[gk]  # Series: one frequency per specimen
            for c in self.classes_:
                freq_rows[f"{c}_{gk_str}"] = group_freq_series
        freq_df = pd.DataFrame(freq_rows)
        freq_df.index.name = SPECIMEN_COL

        # Align to feature_columns order; fill any missing columns with 0.0
        for col in feature_columns:
            if col not in freq_df.columns:
                freq_df[col] = 0.0
        freq_df = freq_df[feature_columns].fillna(0.0)
        return freq_df

    def fit_stage2(
        self,
        sequences_df: pd.DataFrame,
        embeddings: np.ndarray,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> "SequenceLevelClassifier":
        """Train Stage 2 rollup model on train_smaller2.

        Uses Stage 1 (already fitted) to generate per-sequence predictions on
        train_smaller2, aggregates them to specimen level, then trains a
        BinaryOvRClassifierWithFeatureSubsettingByClass (RandomForest).

        Parameters
        ----------
        sequences_df : Sequences DataFrame (train_smaller2), aligned with embeddings.
        embeddings   : (n_sequences, 640) ESM-2 embeddings.
        metadata_df  : Optional specimen-level metadata (for disease labels).

        Reference: malid/train/train_vj_gene_specific_sequence_model_rollup.py
        """
        if not self.group_models_:
            raise RuntimeError("fit_stage1() must be called before fit_stage2().")

        sequences_df = sequences_df.reset_index(drop=True)
        assert len(sequences_df) == len(embeddings)

        if self.verbose >= 1:
            logger.info(
                f"  Stage 2: generating sequence predictions on train_smaller2 "
                f"({len(sequences_df)} sequences, "
                f"{sequences_df[SPECIMEN_COL].nunique()} specimens)..."
            )

        # --- Step 1: Run Stage 1 on train_smaller2 to get per-sequence predictions ---
        seq_preds = self.generate_sequence_predictions(sequences_df, embeddings)

        # --- Step 1b: Compute entropy percentile threshold (if needed) ---
        # For entropy_percentile_cutoff: compute the x-th percentile of the
        # training entropy distribution and store it as an absolute threshold.
        # This threshold is then applied identically during featurize_specimens
        # at both train and test time.
        if self.aggregation_strategy == AggregationStrategy.entropy_percentile_cutoff:
            self._compute_entropy_percentile_threshold(seq_preds)

        # --- Step 2: Aggregate sequence predictions to specimen-level features ---
        if self.verbose >= 1:
            logger.info(f"  Stage 2: aggregating to specimen level ({self.aggregation_strategy.name})...")
        features_df = self.featurize_specimens(seq_preds)
        self.feature_columns_ = list(features_df.columns)

        # --- Step 3: Get disease labels for each specimen ---
        # (sequences_df has per-sequence rows; collapse to per-specimen disease labels)
        specimen_to_disease = (
            sequences_df
            .drop_duplicates(SPECIMEN_COL)
            .set_index(SPECIMEN_COL)[DISEASE_COL]
        )
        missing = [s for s in features_df.index if s not in specimen_to_disease.index]
        if missing:
            raise ValueError(
                f"Specimens in feature matrix have no disease label in sequences_df: "
                f"{missing[:10]}. This indicates a bug — all specimens should have "
                f"disease labels by this point (training scripts drop NaN rows upstream)."
            )
        y = np.array([specimen_to_disease[s] for s in features_df.index])

        if self.verbose >= 1:
            logger.info(
                f"  Stage 2: feature matrix: {features_df.shape[0]} specimens × "
                f"{features_df.shape[1]} features"
            )

        # --- Step 4: Optional reweighing by V-gene group frequency ---
        # Multiply each feature by how prevalent its V-gene group is in that specimen (in TCR) or specimen+isotype (in BCR).
        # This downweights rare groups and upweights common ones.
        # (Standardization is applied before reweighing to ensure fair comparison of feature magnitudes,
        #  and also after reweighing)
        if self.reweigh_by_subset_frequencies:
            # Scaler 1 of 2: standardize aggregated features BEFORE frequency reweighing
            self.preagg_scaler_ = StandardScaler()
            X_scaled = self.preagg_scaler_.fit_transform(features_df.values)
            features_scaled = pd.DataFrame(X_scaled, index=features_df.index, columns=features_df.columns)

            # Compute normalized per-(specimen, group) sequence counts.
            freq_df = self._compute_subset_frequencies(
                seq_preds,
                self.feature_columns_,
            )
            # Reorder freq_df rows to match features_df's specimen order, add rows for any specimen that's in features_df but not in freq_df (these get NaN values and are filled with 0.0). [safety net - if a specimen has zero has_prediction=True sequences, it would be absent from freq_df]
            freq_df = freq_df.reindex(features_df.index).fillna(0.0)

            # Diagnostic #6: reweighing frequency stats (verbose >= 2)
            if self.verbose >= 2:
                self._log_reweighing_stats(freq_df, context="train")

            # Element-wise multiplication: scaled features * frequency weights
            features_df = features_scaled * freq_df

        # --- Step 5: Final scaling + classifier training ---
        # Scaler 2 of 2 (or 1 of 1 if no reweighing): standardize before classifier
        self.stage2_scaler_ = StandardScaler()
        X_train = self.stage2_scaler_.fit_transform(features_df.values)
        features_df_scaled = pd.DataFrame(
            X_train, index=features_df.index, columns=features_df.columns
        )

        # Train N independent binary classifiers, each using only its own class's features
        # Use functools.partial (not a local def) so the factory is picklable
        _make_rf = functools.partial(
            RandomForestClassifier,
            n_estimators=self.n_estimators_stage2,
            **_RF_STAGE2_CONFIG,
        )

        if self.verbose >= 1:
            logger.info(
                f"  Stage 2: training BinaryOvR RandomForest "
                f"({len(self.classes_)} classifiers, {self.n_estimators_stage2} trees each)..."
            )

        self.stage2_clf_ = BinaryOvRClassifierWithFeatureSubsettingByClass(
            base_clf_factory=_make_rf,
            classes=self.classes_,
            n_jobs=self.n_jobs,
            reference_class=self.reference_class,
        )
        self.stage2_clf_.fit(features_df_scaled, y)

        # Diagnostic #4: Stage 2 feature importance (verbose >= 2)
        if self.verbose >= 2:
            self._log_stage2_feature_importance()

        return self

    def fit(
        self,
        train_smaller1_df: pd.DataFrame,
        train_smaller1_emb: np.ndarray,
        train_smaller2_df: pd.DataFrame,
        train_smaller2_emb: np.ndarray,
    ) -> "SequenceLevelClassifier":
        """Full two-stage fit.

        Parameters
        ----------
        train_smaller1_df  : Sequences DataFrame for Stage 1 (2/3 of train fold).
        train_smaller1_emb : ESM-2 embeddings aligned with train_smaller1_df.
        train_smaller2_df  : Sequences DataFrame for Stage 2 (1/3 of train fold).
        train_smaller2_emb : ESM-2 embeddings aligned with train_smaller2_df.
        """
        self.fit_stage1(train_smaller1_df, train_smaller1_emb)
        self.fit_stage2(train_smaller2_df, train_smaller2_emb)
        return self

    # ------------------------------------------------------------------ #
    # Load from saved artifacts (for resume support)                      #
    # ------------------------------------------------------------------ #

    def load_stage1_artifacts(self, data: dict) -> None:
        """Restore Stage 1 fitted state from a saved artifact dict.

        This is the counterpart to the Stage 1 save block in _run_fold_loop.
        After calling this, the model can run fit_stage2() or predict_proba()
        as if fit_stage1() had been called.

        Parameters
        ----------
        data : Dict with keys: group_models, classes, non_rare_v_genes,
               locus.  Matches the format saved by _save_stage1_artifact
               in train_model3.py. Old artifacts may also contain
               aggregation_strategy and entropy_max_fraction (ignored).
        """
        required_keys = {"group_models", "classes", "non_rare_v_genes", "locus"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"Stage 1 artifact is missing required keys: {missing}. "
                f"Available keys: {sorted(data.keys())}"
            )

        # Validate locus matches model configuration
        if data["locus"] != self.locus:
            raise ValueError(
                f"Stage 1 artifact locus '{data['locus']}' does not match "
                f"model locus '{self.locus}'"
            )

        self.group_models_ = data["group_models"]
        self.classes_ = data["classes"]
        self.non_rare_v_genes_ = data["non_rare_v_genes"]

    def load_stage2_artifacts(self, data: dict) -> None:
        """Restore Stage 2 fitted state from a saved artifact dict.

        This is the counterpart to the Stage 2 save block in _run_fold_loop.
        After calling this, the model can run predict_proba() as if
        fit_stage2() had been called.

        Requires Stage 1 to already be loaded (via fit_stage1 or
        load_stage1_artifacts).

        Parameters
        ----------
        data : Dict with keys: stage2_clf, stage2_scaler, preagg_scaler,
               feature_columns, classes, reweigh_by_subset_frequencies.
               Optional keys: entropy_percentile_threshold (float, for
               entropy_percentile_cutoff strategy), aggregation_strategy
               (str, validated against this model's strategy on load).
               Matches the format saved by _run_fold_loop in train_model3.py.
        """
        if not self.group_models_:
            raise RuntimeError(
                "Stage 1 must be loaded before Stage 2. Call fit_stage1() or "
                "load_stage1_artifacts() first."
            )

        required_keys = {"stage2_clf", "stage2_scaler", "feature_columns",
                         "classes", "reweigh_by_subset_frequencies"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"Stage 2 artifact is missing required keys: {missing}. "
                f"Available keys: {sorted(data.keys())}"
            )

        # Validate classes match Stage 1
        s2_classes = [str(c) for c in data["classes"]]
        s1_classes = [str(c) for c in self.classes_]
        if sorted(s2_classes) != sorted(s1_classes):
            raise ValueError(
                f"Stage 2 artifact classes {s2_classes} do not match "
                f"Stage 1 classes {s1_classes}"
            )

        # Validate aggregation strategy matches what was used during training.
        # The strategy determines how sequence-level predictions are aggregated
        # into specimen features — a mismatch would produce wrong features for
        # the Stage 2 classifier that was trained on a specific feature layout.
        saved_strategy = data.get("aggregation_strategy")
        if saved_strategy is not None:
            # Artifacts saved before this field was added won't have it — skip.
            if saved_strategy != self.aggregation_strategy.name:
                raise ValueError(
                    f"Aggregation strategy mismatch: the Stage 2 artifact was "
                    f"trained with '{saved_strategy}', but this model is "
                    f"configured with '{self.aggregation_strategy.name}'. "
                    f"Use --aggregation-strategy {saved_strategy} to match, "
                    f"or retrain Stage 2 with the desired strategy."
                )

        self.stage2_clf_ = data["stage2_clf"]
        self.stage2_scaler_ = data["stage2_scaler"]
        self.preagg_scaler_ = data.get("preagg_scaler")  # None if no reweighing
        self.feature_columns_ = data["feature_columns"]
        # Restore learned entropy threshold (None if not entropy_percentile_cutoff)
        self.entropy_percentile_threshold_ = data.get("entropy_percentile_threshold")

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict_proba(
        self,
        sequences_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Full pipeline: Stage 1 → aggregate → Stage 2 → probabilities.

        Parameters
        ----------
        sequences_df : Sequences DataFrame (any split), aligned with embeddings.
        embeddings   : (n_sequences, 640) ESM-2 embeddings.

        Returns
        -------
        DataFrame indexed by specimen_label, columns = disease classes.
        Shape: (n_specimens, n_classes). Does NOT sum to 1 when OvR is used.
        """
        if self.stage2_clf_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        sequences_df = sequences_df.reset_index(drop=True)
        assert len(sequences_df) == len(embeddings), (
            f"sequences_df length ({len(sequences_df)}) != embeddings length ({len(embeddings)})"
        )

        # Stage 1: per-sequence predictions from group models
        seq_preds = self.generate_sequence_predictions(sequences_df, embeddings)

        # Aggregate to specimen-level features (aligned to training column order)
        features_df = self.featurize_specimens(seq_preds, feature_columns=self.feature_columns_)

        # Apply same reweighing pipeline as during training
        if self.reweigh_by_subset_frequencies and self.preagg_scaler_ is not None:
            # Scaler 1: transform (not fit) using training-fitted preagg scaler
            X_scaled = self.preagg_scaler_.transform(features_df.values)
            features_df_scaled = pd.DataFrame(
                X_scaled, index=features_df.index, columns=features_df.columns
            )
            # Multiply by V-gene group frequencies
            freq_df = self._compute_subset_frequencies(seq_preds, self.feature_columns_)
            freq_df = freq_df.reindex(features_df.index).fillna(0.0)

            # Diagnostic #6: reweighing frequency stats on test (verbose >= 2)
            if self.verbose >= 2:
                self._log_reweighing_stats(freq_df, context="test")

            features_df = features_df_scaled * freq_df

        # Scaler 2: transform using training-fitted stage2 scaler
        X = self.stage2_scaler_.transform(features_df.values)
        features_df_final = pd.DataFrame(
            X, index=features_df.index, columns=features_df.columns
        )

        # Stage 2 classifier: predict specimen-level disease probabilities
        proba = self.stage2_clf_.predict_proba(features_df_final)
        proba_df = pd.DataFrame(proba, index=features_df.index, columns=self.classes_)

        # Diagnostic #5: prediction confidence stats (verbose >= 2)
        if self.verbose >= 2:
            self._log_prediction_confidence_stats(proba_df, context="test")

        return proba_df

    def predict(
        self,
        sequences_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Return predicted class labels."""
        proba_df = self.predict_proba(sequences_df, embeddings)
        return self.classes_[np.argmax(proba_df.values, axis=1)]


# ---------------------------------------------------------------------------
# Paper-best factory functions
# ---------------------------------------------------------------------------

def make_tcr_model(**kwargs) -> SequenceLevelClassifier:
    """Return paper-best Model 3 for TCR.

    Stage 1: CustomOneVsRestClassifier(GlmnetLogitNetWrapper(alpha=0.0)) per V-gene group
    Stage 2: RandomForest + entropy_cutoff (0.80) + reweigh_by_subset_frequencies

    Reference: malid/config.py:94-103
    """
    return SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.entropy_cutoff,
        entropy_max_fraction=0.80,
        exclude_rare_v_genes=True,
        reweigh_by_subset_frequencies=True,
        **kwargs,
    )


def make_bcr_model(**kwargs) -> SequenceLevelClassifier:
    """Return paper-best Model 3 for BCR.

    Stage 1: RandomForestClassifier per (V-gene, isotype) group
    Stage 2: RandomForest + mean + reweigh_by_subset_frequencies

    Reference: malid/config.py:94-103
    """
    return SequenceLevelClassifier(
        locus="BCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=True,
        reweigh_by_subset_frequencies=True,
        **kwargs,
    )
