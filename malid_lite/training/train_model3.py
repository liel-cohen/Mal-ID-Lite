#!/usr/bin/env python
"""Train and evaluate Model 3 (Sequence-Level Classifier) for disease classification.

Two-stage V-gene-specific sequence model:
  Stage 1: per-V-gene-[+isotype]-group classifiers trained on ESM-2 embeddings
           of CDR3 sequences from train_smaller1.
  Stage 2: specimen-level rollup model with per-class feature subsetting
           trained on train_smaller2 (using Stage 1 predictions as inputs).

Embeddings
----------
ESM-2 embeddings must be pre-computed per participant before training. Run:

    python -m malid_lite.training.compute_model3_embeddings \\
        --metadata-path /path/to/metadata.tsv

The training script loads embeddings from the --embedding-dir directory
(default: cache/<dataset-name>/embeddings/). If embeddings are not found, the
script errors with instructions.

Row alignment between fold data and pre-computed embeddings is ensured at load
time: rows are checked positionally using the downsampling unique key
(repertoire_id, igh_or_tcrb_clone_id, isotype_supergroup, amplification_label
if present). If the order differs,
embeddings are reordered to match (with a warning). A biological sanity check
(cdr3_aa, v_gene, j_gene) runs after alignment.
See load_precomputed_embeddings() and CACHING_ARCHITECTURE.md.

Alternatively, use --compute-embeddings to compute them inline during training.
This is convenient but slow for repeated runs (~3 hours per 10M sequences on
M4 Max MPS, ~14 GB storage per 10M sequences).

Classification modes
--------------------
multiclass
    Single N-class Stage 2 model over all disease classes.

binary
    One binary model for a single disease-vs-reference pair.

multi-binary
    One independent binary model per disease vs. the reference class.

Output directory structure
---------------------------
multiclass:   trained_models/<dataset_name>/model3/multiclass/<gene_locus>/
binary:       trained_models/<dataset_name>/model3/binary/<gene_locus>/<disease>_vs_<reference>/
multi-binary: trained_models/<dataset_name>/model3/binary/<gene_locus>/<disease1>_vs_<reference>/
                                                              <disease2>_vs_<reference>/
                                                              ...

Per-fold artifacts:
    fold_<id>_stage1.pkl       : Stage 1 group models dict + _meta
    fold_<id>_stage2.pkl       : Stage 2 rollup model + _meta
    fold_<id>_results.json     : Evaluation metrics
    fold_<id>_predictions.pkl  : Raw predictions for aggregation + CSV (resume support)
    <mode>_predictions.csv     : All-fold predictions (appended across folds)
    summary_<timestamp>.json   : Aggregated metrics summary
    RESULTS_<timestamp>.md     : Human-readable results table

Resume (--resume)
-----------------
Detects completed stages per fold and skips them:
  - stage1.pkl exists        → load Stage 1, skip to Stage 2 training
  - stage1.pkl + stage2.pkl  → load both, skip to evaluation
  - all 4 files              → skip fold entirely, reload results

Each artifact includes a _meta dict with timestamp, fold_id, locus, classes,
and data dimensions. On resume, these are validated against the current run
to catch stale/mismatched artifacts early.

Stage 1 validation excludes Stage-2-only parameters (aggregation_strategy,
entropy_threshold_fraction, n_estimators_stage2, reweigh_by_subset_frequencies)
since Stage 1 models are trained independently of these.

Resume from Stage 2 (--resume-from-stage2)
-------------------------------------------
Loads Stage 1 from saved artifacts and retrains Stage 2 from scratch.
Automatically removes existing Stage 2, results, and prediction artifacts
so they are regenerated with the new parameters. Use this when you want to
change Stage-2-only parameters without re-running the expensive Stage 1
training. Requires Stage 1 artifacts to exist.

Example: retrain Stage 2 with mean aggregation instead of entropy filtering:

    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --aggregation-strategy mean --resume-from-stage2

Example: try a different entropy threshold:

    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --aggregation-strategy entropy_cutoff --entropy-threshold 0.50 \\
        --resume-from-stage2

Resume from evaluation (--resume-from-evaluation)
--------------------------------------------------
Loads Stage 1 and Stage 2 from saved artifacts and re-runs evaluation only.
Automatically removes existing results and prediction artifacts so they are
regenerated. Requires both Stage 1 and Stage 2 artifacts to exist.

Example:

    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --resume-from-evaluation

Usage examples
--------------
    # Multiclass (default, TCR) — requires pre-computed embeddings
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv

    # Multi-binary: one COVID vs Healthy, one HIV vs Healthy, etc.
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy

    # Compute embeddings inline (no separate embedding step needed):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --compute-embeddings

    # Custom embedding directory:
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --embedding-dir /path/to/precomputed/embeddings/

    # Custom aggregation strategy (default: auto → paper-best per locus):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --aggregation-strategy mean
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# Add project root to path (must come before malid_lite imports)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Custom multiclass metrics that work with unnormalized probabilities
# (Model 3's BinaryOvR outputs independent per-class probabilities that
# don't sum to 1 — sklearn's multiclass roc_auc_score rejects these).
# These match what the original Mal-ID paper used for evaluation.
from malid_lite.utils import multiclass_metrics

from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.models.model3_sequence_level import (
    CDR3_COL,
    DISEASE_COL,
    EMBEDDING_DIM,
    ISOTYPE_COL,
    J_GENE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    V_GENE_COL,
    AggregationStrategy,
    SequenceLevelClassifier,
    compute_esm2_embeddings,
    make_bcr_model,
    make_tcr_model,
)
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    aggregate_fold_results,
    filter_to_binary_pair,
    generate_results_md,
    get_dataset_disease_classes,
    get_model_output_dir,
    make_pair_name,
    run_training_orchestration,
    save_per_pair_results,
    split_train_smaller,
    validate_mode_and_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_NAME = "model3"
MODEL_LABEL = "Model 3"

# Paper-best entropy threshold fraction for TCR (0.20 = keep below 80% of max
# entropy). Used for display/metadata when the resolved strategy is entropy_cutoff
# and no explicit --entropy-threshold was given.
_DEFAULT_ENTROPY_THRESHOLD = 0.20

# Parameters that only affect Stage 2 (aggregation + Stage 2 training).
# Excluded from Stage 1 artifact validation so that Stage 1 can be reused
# when only Stage-2-only params change (e.g., --resume-from-stage2).
_STAGE2_ONLY_PARAMS = frozenset({
    "aggregation_strategy",
    "entropy_threshold_fraction",
    "n_estimators_stage2",
    "reweigh_by_subset_frequencies",
})


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string (e.g. '2m 34s' or '1h 05m 12s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Artifact save / load / validate  (resume support)
# ---------------------------------------------------------------------------

def _build_model_params(
    model: SequenceLevelClassifier,
    classification_mode: Optional[str] = None,
    diseases: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
) -> dict:
    """Extract model and run parameters for artifact metadata.

    These are validated on resume to ensure loaded artifacts were trained
    with the same settings as the current run. Includes both model-level
    hyperparameters and run-level settings (classification mode, disease
    subset, dataset name) that affect training outcomes.
    """
    params = {
        "locus": model.locus,
        "aggregation_strategy": model.aggregation_strategy.name,
        "entropy_threshold_fraction": model.entropy_threshold_fraction,
        "exclude_rare_v_genes": model.exclude_rare_v_genes,
        "min_sequences_per_group": model.min_sequences_per_group,
        "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
        "n_estimators_stage1": model.n_estimators_stage1,
        "n_estimators_stage2": model.n_estimators_stage2,
        "reference_class": model.reference_class,
        "classification_mode": classification_mode,
        "diseases": sorted(diseases) if diseases else None,
        "dataset_name": dataset_name,
    }
    return params


def _save_stage1_artifact(
    model: SequenceLevelClassifier,
    path: Path,
    fold_id: int,
    ts1: pd.DataFrame,
    run_params: Optional[dict] = None,
) -> None:
    """Save Stage 1 model artifact with metadata for resume validation.

    Parameters
    ----------
    run_params : Optional dict with keys classification_mode, diseases,
                 dataset_name. Passed through to _build_model_params.
    """
    rp = run_params or {}
    meta = {
        "timestamp": datetime.now().isoformat(),
        "fold_id": fold_id,
        "locus": model.locus,
        "classes": [str(c) for c in model.classes_],
        "n_groups": len(model.group_models_),
        "n_training_sequences": len(ts1),
        "n_training_specimens": int(ts1[SPECIMEN_COL].nunique()),
        "n_training_participants": int(ts1[PARTICIPANT_COL].nunique()),
        "model_params": _build_model_params(model, **rp),
    }
    with open(path, "wb") as f:
        # Stage 1 artifact body contains only Stage-1-relevant data.
        # aggregation_strategy and entropy_threshold_fraction are Stage-2-only
        # concerns and are NOT stored here to avoid confusion when Stage 2 is
        # retrained with different params (via --resume-from-stage2).
        # They are still recorded in _meta.model_params for provenance.
        pickle.dump({
            "group_models": model.group_models_,
            "classes": model.classes_,
            "non_rare_v_genes": model.non_rare_v_genes_,
            "locus": model.locus,
            "_meta": meta,
        }, f)
    logger.info(f"  Saved Stage 1: {path}")


def _save_stage2_artifact(
    model: SequenceLevelClassifier,
    path: Path,
    fold_id: int,
    ts2: pd.DataFrame,
    run_params: Optional[dict] = None,
) -> None:
    """Save Stage 2 model artifact with metadata for resume validation.

    Parameters
    ----------
    run_params : Optional dict with keys classification_mode, diseases,
                 dataset_name. Passed through to _build_model_params.
    """
    rp = run_params or {}
    meta = {
        "timestamp": datetime.now().isoformat(),
        "fold_id": fold_id,
        "classes": [str(c) for c in model.classes_],
        "n_features": len(model.feature_columns_),
        "n_training_sequences": len(ts2),
        "n_training_specimens": int(ts2[SPECIMEN_COL].nunique()),
        "feature_columns": model.feature_columns_,
        "model_params": _build_model_params(model, **rp),
    }
    with open(path, "wb") as f:
        pickle.dump({
            "stage2_clf": model.stage2_clf_,
            "stage2_scaler": model.stage2_scaler_,
            "preagg_scaler": model.preagg_scaler_,
            "feature_columns": model.feature_columns_,
            "classes": model.classes_,
            "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
            "_meta": meta,
        }, f)
    logger.info(f"  Saved Stage 2: {path}")


def _save_predictions_artifact(
    raw_preds: dict,
    fold_pred_rows: List[Dict],
    path: Path,
    fold_id: int,
    classes: np.ndarray,
    n_test_specimens: int,
) -> None:
    """Save per-fold predictions artifact with metadata for resume."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "fold_id": fold_id,
        "classes": [str(c) for c in classes],
        "n_test_specimens": n_test_specimens,
        "n_prediction_rows": len(fold_pred_rows),
    }
    with open(path, "wb") as f:
        pickle.dump({
            "raw_preds": raw_preds,
            "predictions_rows": fold_pred_rows,
            "_meta": meta,
        }, f)
    logger.info(f"  Saved predictions: {path}")


def _validate_artifact_meta(
    meta: dict,
    stage_name: str,
    fold_id: int,
    locus: Optional[str] = None,
    expected_classes: Optional[List[str]] = None,
    current_model_params: Optional[dict] = None,
    expected_data_sizes: Optional[dict] = None,
) -> None:
    """Validate artifact metadata against current run parameters.

    Raises ValueError on hard mismatches (fold_id, locus, classes, model params,
    data sizes). These indicate stale or wrong artifacts that would produce
    incorrect results.

    Parameters
    ----------
    meta                 : The _meta dict from the loaded artifact.
    stage_name           : "Stage 1", "Stage 2", or "Predictions" (for error messages).
    fold_id              : Expected fold ID.
    locus                : Expected locus (checked for Stage 1 only).
    expected_classes     : Sorted class names from current data or previously loaded stage.
    current_model_params : Dict from _build_model_params(model) for the current run.
                           If provided, each key is compared against the artifact's
                           saved model_params.
    expected_data_sizes  : Dict of expected data size fields to validate, e.g.
                           {"n_training_sequences": 500000, "n_training_specimens": 120}.
                           Each key is checked against the corresponding field in meta.
                           Only provided when training data is loaded (not when both
                           stages are resumed).
    """
    if not meta:
        logger.warning(
            f"  {stage_name} artifact has no _meta (saved before resume metadata "
            f"was added). Skipping validation — cannot verify compatibility."
        )
        return

    # --- Hard errors: mismatches that would produce wrong results ---
    artifact_fold = meta.get("fold_id")
    if artifact_fold is not None and artifact_fold != fold_id:
        raise ValueError(
            f"{stage_name} artifact fold_id={artifact_fold} does not match "
            f"current fold_id={fold_id}. Wrong artifact file?"
        )

    if locus is not None:
        artifact_locus = meta.get("locus")
        if artifact_locus is not None and artifact_locus != locus:
            raise ValueError(
                f"{stage_name} artifact locus='{artifact_locus}' does not match "
                f"current locus='{locus}'. Wrong artifact file?"
            )

    if expected_classes is not None:
        artifact_classes = meta.get("classes")
        if artifact_classes is not None:
            if sorted(artifact_classes) != sorted(expected_classes):
                raise ValueError(
                    f"{stage_name} artifact classes {sorted(artifact_classes)} do not "
                    f"match expected classes {sorted(expected_classes)}. "
                    f"Data may have changed since the artifact was saved."
                )

    # --- Model parameter validation ---
    if current_model_params is not None:
        saved_params = meta.get("model_params")
        if saved_params is not None:
            mismatches = []
            for key, current_val in current_model_params.items():
                # Skip keys absent from saved artifact (backward compat with
                # older artifacts that didn't save this param). But if the key
                # IS present (even if its value is None), compare it.
                if key not in saved_params:
                    continue
                saved_val = saved_params[key]
                if saved_val != current_val:
                    mismatches.append(
                        f"  {key}: artifact={saved_val!r}, current={current_val!r}"
                    )
            if mismatches:
                raise ValueError(
                    f"{stage_name} artifact was trained with different model parameters "
                    f"than the current run:\n" + "\n".join(mismatches) + "\n"
                    f"Delete the artifact or use matching parameters to resume."
                )

    # --- Data size validation ---
    # Resume is for recovering interrupted runs on the same data. If training
    # data sizes don't match, the data likely changed since the artifact was
    # saved, making it stale.
    if expected_data_sizes is not None:
        size_mismatches = []
        for key, expected_val in expected_data_sizes.items():
            saved_val = meta.get(key)
            if saved_val is not None and saved_val != expected_val:
                size_mismatches.append(
                    f"  {key}: artifact={saved_val:,}, current={expected_val:,}"
                )
        if size_mismatches:
            raise ValueError(
                f"{stage_name} artifact was trained on different data than the "
                f"current run:\n" + "\n".join(size_mismatches) + "\n"
                f"The underlying data may have changed since the artifact was saved. "
                f"Delete the artifact to retrain from scratch."
            )


def _log_resumed_artifact(
    meta: dict,
    stage_name: str,
    data_sizes_validated: bool = False,
) -> None:
    """Log what was loaded and validated for a resumed artifact.

    Parameters
    ----------
    data_sizes_validated : Whether training data sizes were checked against the
        artifact. When False (both stages resumed, no training data loaded),
        size fields are omitted from the log to avoid implying they were verified.
    """
    if not meta:
        return

    ts = meta.get("timestamp", "unknown")
    logger.info(f"  {stage_name} loaded from saved artifact (saved {ts})")

    # --- Validated identity and structure ---
    identity_parts = []
    if "fold_id" in meta:
        identity_parts.append(f"fold_id={meta['fold_id']}")
    if "locus" in meta:
        identity_parts.append(f"locus={meta['locus']}")
    if "classes" in meta:
        identity_parts.append(f"classes={meta['classes']}")
    if identity_parts:
        logger.info(f"    Validated: {', '.join(identity_parts)}")

    # --- Validated data sizes (only when training data was available) ---
    if data_sizes_validated:
        size_parts = []
        if "n_training_sequences" in meta:
            size_parts.append(f"{meta['n_training_sequences']:,} training sequences")
        if "n_training_specimens" in meta:
            size_parts.append(f"{meta['n_training_specimens']:,} training specimens")
        if "n_training_participants" in meta:
            size_parts.append(f"{meta['n_training_participants']:,} training participants")
        if size_parts:
            logger.info(f"    Validated: {', '.join(size_parts)}")
    else:
        logger.info(
            f"    Data sizes not validated (training data not loaded for this stage)"
        )

    # --- Validated model params ---
    saved_params = meta.get("model_params")
    if saved_params:
        logger.info(f"    Validated: model/run params match ({len(saved_params)} params)")

    # --- Loaded structural info ---
    loaded_parts = []
    if "n_groups" in meta:
        loaded_parts.append(f"{meta['n_groups']} groups")
    if "n_features" in meta:
        loaded_parts.append(f"{meta['n_features']} features")
    if loaded_parts:
        logger.info(f"    Loaded: {', '.join(loaded_parts)}")


def _validate_resume_artifacts(
    output_dir: Path,
    fold_ids: List[int],
    resume_from_stage2: bool,
    resume_from_evaluation: bool,
) -> List[str]:
    """Check that required artifacts exist for a targeted resume mode.

    Returns a list of human-readable error strings, one per problematic fold.
    An empty list means all folds are valid.  The caller is responsible for
    collecting errors across multiple output directories (e.g. multi-binary
    pairs) and raising a single ValueError with the full picture.

    Parameters
    ----------
    output_dir           : Directory containing fold artifacts for one
                           classification target (multiclass dir or a single
                           binary-pair subdir).
    fold_ids             : Fold IDs to validate.
    resume_from_stage2   : True when Stage 2 will be retrained (needs Stage 1).
    resume_from_evaluation : True when only evaluation will re-run (needs
                             Stage 1 + Stage 2).
    """
    errors: List[str] = []
    for fold_id in fold_ids:
        stage1_path = output_dir / f"fold_{fold_id}_stage1.pkl"
        stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
        if resume_from_stage2:
            if not stage1_path.exists():
                errors.append(f"  Fold {fold_id}: missing {stage1_path.name}")
        elif resume_from_evaluation:
            missing = []
            if not stage1_path.exists():
                missing.append(stage1_path.name)
            if not stage2_path.exists():
                missing.append(stage2_path.name)
            if missing:
                errors.append(
                    f"  Fold {fold_id}: missing {', '.join(missing)}"
                )
    return errors


def _load_stage1_artifact(
    model: SequenceLevelClassifier,
    path: Path,
    fold_id: int,
    locus: str,
    expected_classes: Optional[List[str]] = None,
    run_params: Optional[dict] = None,
    ts1: Optional[pd.DataFrame] = None,
) -> dict:
    """Load Stage 1 artifact, validate metadata and model params, populate model.

    Parameters
    ----------
    expected_classes : If available (from training data), validate that artifact
                      classes match. None when both stages are resumed and no
                      training data is loaded.
    run_params      : Optional dict with keys classification_mode, diseases,
                      dataset_name. Passed through to _build_model_params.
    ts1             : train_smaller1 DataFrame. If provided, data sizes are
                      validated against the artifact's saved counts.

    Returns
    -------
    The _meta dict from the artifact (for downstream logging/validation).
    """
    rp = run_params or {}
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Build expected data sizes from training data (when available)
    expected_data_sizes = None
    if ts1 is not None:
        expected_data_sizes = {
            "n_training_sequences": len(ts1),
            "n_training_specimens": int(ts1[SPECIMEN_COL].nunique()),
            "n_training_participants": int(ts1[PARTICIPANT_COL].nunique()),
        }

    meta = data.get("_meta", {})
    # Exclude Stage-2-only params from Stage 1 validation: Stage 1 models are
    # trained independently of aggregation strategy and Stage 2 hyperparams,
    # so it's valid to load a Stage 1 artifact and retrain Stage 2 differently.
    stage1_params = _build_model_params(model, **rp)
    for key in _STAGE2_ONLY_PARAMS:
        stage1_params.pop(key, None)
    _validate_artifact_meta(
        meta, "Stage 1", fold_id, locus=locus,
        expected_classes=expected_classes,
        current_model_params=stage1_params,
        expected_data_sizes=expected_data_sizes,
    )
    model.load_stage1_artifacts(data)
    _log_resumed_artifact(meta, "Stage 1", data_sizes_validated=(ts1 is not None))
    return meta


def _load_stage2_artifact(
    model: SequenceLevelClassifier,
    path: Path,
    fold_id: int,
    expected_classes: Optional[List[str]] = None,
    run_params: Optional[dict] = None,
    ts2: Optional[pd.DataFrame] = None,
) -> dict:
    """Load Stage 2 artifact, validate metadata and model params, populate model.

    Parameters
    ----------
    run_params : Optional dict with keys classification_mode, diseases,
                 dataset_name. Passed through to _build_model_params.
    ts2        : train_smaller2 DataFrame. If provided, data sizes are
                 validated against the artifact's saved counts.

    Returns the _meta dict from the artifact.
    """
    rp = run_params or {}
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Build expected data sizes from training data (when available)
    expected_data_sizes = None
    if ts2 is not None:
        expected_data_sizes = {
            "n_training_sequences": len(ts2),
            "n_training_specimens": int(ts2[SPECIMEN_COL].nunique()),
        }

    meta = data.get("_meta", {})
    # Use Stage 1 classes as expected if not provided from data
    if expected_classes is None and model.classes_ is not None:
        expected_classes = [str(c) for c in model.classes_]
    _validate_artifact_meta(
        meta, "Stage 2", fold_id,
        expected_classes=expected_classes,
        current_model_params=_build_model_params(model, **rp),
        expected_data_sizes=expected_data_sizes,
    )
    model.load_stage2_artifacts(data)
    _log_resumed_artifact(meta, "Stage 2", data_sizes_validated=(ts2 is not None))
    return meta


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare_fold(
    loader: MalIDPublishedDataLoader,
    fold_id: int,
    fold_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fold sequences and join with disease metadata.

    Returns
    -------
    (sequences_df, metadata_df)
    sequences_df has all sequence columns plus disease (from metadata join).
    specimen_label column = repertoire_id.
    """
    sequences_df, metadata_df = loader.get_fold_data(fold_id, fold_label)

    if sequences_df.empty:
        raise ValueError(f"No sequences found for fold {fold_id} {fold_label}")

    # Add disease column to sequences_df
    disease_map = metadata_df.set_index("specimen_label")["disease"]
    sequences_df = sequences_df.copy()
    sequences_df[DISEASE_COL] = sequences_df["repertoire_id"].map(disease_map)

    n_before = len(sequences_df)
    sequences_df = sequences_df.dropna(subset=[DISEASE_COL])
    if len(sequences_df) < n_before:
        logger.warning(
            f"  Dropped {n_before - len(sequences_df)} rows with unknown disease"
        )

    # Ensure specimen_label column exists
    if SPECIMEN_COL not in sequences_df.columns and "repertoire_id" in sequences_df.columns:
        sequences_df[SPECIMEN_COL] = sequences_df["repertoire_id"]

    return sequences_df, metadata_df


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def _resolve_col(df: pd.DataFrame, col: str) -> str:
    """Return the actual column name in df, handling specimen_label/repertoire_id alias."""
    if col in df.columns:
        return col
    if col == SPECIMEN_COL and "repertoire_id" in df.columns:
        return "repertoire_id"
    if col == "repertoire_id" and SPECIMEN_COL in df.columns:
        return SPECIMEN_COL
    raise KeyError(f"Column '{col}' (or alias) not found. Available: {list(df.columns)[:15]}")


def _get_downsampling_key_cols(participant_df: pd.DataFrame) -> List[str]:
    """Return the downsampling unique key columns present in participant_df.

    This is the groupby key used in preprocess_downsample(), guaranteed unique
    per row after DOWNSAMPLED preprocessing.
    """
    key_cols = ["repertoire_id", "igh_or_tcrb_clone_id", ISOTYPE_COL]
    if "amplification_label" in participant_df.columns:
        key_cols.append("amplification_label")
    return key_cols


def _check_positional_alignment(
    fold_subset: pd.DataFrame,
    participant_df: pd.DataFrame,
    cols: List[str],
) -> bool:
    """Return True if all columns match positionally between fold and precomputed.

    Handles NaN correctly (two NaN values in the same position are considered equal).
    """
    for col in cols:
        fold_col = _resolve_col(fold_subset, col)
        precomputed_col = _resolve_col(participant_df, col)
        fold_vals = fold_subset[fold_col].values
        precomputed_vals = participant_df[precomputed_col].values
        # np.array_equal treats NaN != NaN; use pandas Series.equals which treats NaN == NaN
        if not pd.Series(fold_vals).equals(pd.Series(precomputed_vals)):
            return False
    return True


def _make_hashable_key(values: tuple) -> tuple:
    """Convert a tuple of values to a hashable key, replacing NaN with a sentinel.

    NaN != NaN in Python, so NaN values in tuples break dict lookups.
    We replace them with a sentinel string that cannot appear in the data.
    """
    return tuple("__NAN__" if pd.isna(v) else v for v in values)


def _compute_reorder_indices(
    fold_subset: pd.DataFrame,
    participant_df: pd.DataFrame,
    key_cols: List[str],
    participant: str,
) -> np.ndarray:
    """Compute indices to reorder participant_df rows to match fold_subset order.

    Builds an index on the downsampling unique key from participant_df, then
    looks up each fold row's key to find the correct source row index.

    Returns an integer index array of length len(fold_subset), where
    result[i] is the row in participant_df that corresponds to fold row i.
    """
    # Build a composite key → row index mapping from the precomputed side
    resolved_precomputed = [_resolve_col(participant_df, c) for c in key_cols]
    precomputed_keys = [
        _make_hashable_key(t)
        for t in zip(*(participant_df[c].values for c in resolved_precomputed))
    ]
    key_to_idx = {k: i for i, k in enumerate(precomputed_keys)}

    if len(key_to_idx) != len(participant_df):
        raise ValueError(
            f"Downsampling key is not unique for participant {participant}: "
            f"{len(participant_df)} rows but {len(key_to_idx)} unique keys. "
            f"This indicates a preprocessing issue."
        )

    # Look up each fold row (vectorized extraction, loop only for dict lookup)
    resolved_fold = [_resolve_col(fold_subset, c) for c in key_cols]
    fold_keys = [
        _make_hashable_key(t)
        for t in zip(*(fold_subset[c].values for c in resolved_fold))
    ]
    reorder_indices = np.empty(len(fold_subset), dtype=np.intp)
    for i, key in enumerate(fold_keys):
        idx = key_to_idx.get(key)
        if idx is None:
            raise ValueError(
                f"Fold row not found in pre-computed embeddings for participant "
                f"{participant}. Key: {key}. "
                f"This likely means the embeddings were computed from a different "
                f"preprocessing run. "
                f"Re-run compute_model3_embeddings.py to regenerate."
            )
        reorder_indices[i] = idx

    return reorder_indices


def _align_embeddings(
    fold_subset: pd.DataFrame,
    participant_df: pd.DataFrame,
    participant_emb: np.ndarray,
    participant: str,
) -> np.ndarray:
    """Align pre-computed embeddings with fold data for one participant.

    First checks if rows are already in the same order (fast path). If not,
    reorders embeddings using the downsampling unique key and warns the user.
    After alignment, runs a biological sanity check (cdr3_aa, v_gene, j_gene).

    Returns embeddings in fold_subset row order.
    """
    key_cols = _get_downsampling_key_cols(participant_df)
    sanity_cols = [CDR3_COL, V_GENE_COL, J_GENE_COL]

    # Fast path: check if already aligned on the downsampling key
    if _check_positional_alignment(fold_subset, participant_df, key_cols):
        # Already aligned — run sanity check directly (compared columns)
        if not _check_positional_alignment(fold_subset, participant_df, sanity_cols):
            raise ValueError(
                f"Biological sanity check failed for participant {participant}: "
                f"cdr3_aa/v_gene/j_gene differ despite matching on the downsampling "
                f"key. This indicates data corruption. "
                f"Re-run compute_model3_embeddings.py to regenerate."
            )
        return participant_emb

    # Slow path: reorder using the downsampling unique key
    logger.warning(
        f"  Row order mismatch for participant {participant} — "
        f"reordering embeddings to match fold data. "
        f"This is not an error, but if it happens consistently consider "
        f"re-running compute_model3_embeddings.py to avoid the overhead."
    )
    reorder_indices = _compute_reorder_indices(
        fold_subset, participant_df, key_cols, participant,
    )
    aligned_emb = participant_emb[reorder_indices]

    # Sanity check on reordered data
    reordered_df = participant_df.iloc[reorder_indices].reset_index(drop=True)
    fold_reset = fold_subset.reset_index(drop=True)
    if not _check_positional_alignment(fold_reset, reordered_df, sanity_cols):
        raise ValueError(
            f"Biological sanity check failed for participant {participant} "
            f"after reordering: cdr3_aa/v_gene/j_gene differ despite matching "
            f"on the downsampling key. This indicates data corruption. "
            f"Re-run compute_model3_embeddings.py to regenerate."
        )

    return aligned_emb


def load_precomputed_embeddings(
    sequences_df: pd.DataFrame,
    embedding_dir: Path,
) -> np.ndarray:
    """Load pre-computed ESM-2 embeddings for a set of sequences.

    Assembles embeddings from per-participant files in embedding_dir, matching
    rows by participant_label. Pre-computed embeddings may cover ALL of a
    participant's data (across all folds), while sequences_df may be a subset
    (e.g., one fold's training split). Both exact-match and subset cases are
    handled correctly.

    Row alignment uses the downsampling unique key (repertoire_id,
    igh_or_tcrb_clone_id, isotype_supergroup [, amplification_label]):
    - Exact match (same row count): fast-path positional check, then reorder
      if needed. Biological sanity check (cdr3_aa, v_gene, j_gene) after.
    - Subset (fold has fewer rows): key-based lookup into the full pre-computed
      data. Each fold row's key is looked up in the pre-computed key→index
      mapping to retrieve its embedding.

    Parameters
    ----------
    sequences_df : Sequences DataFrame with PARTICIPANT_COL. Each participant
        present must have pre-computed embeddings in embedding_dir.
    embedding_dir : Directory containing per-participant
        <label>_downsampled.parquet and <label>_embeddings.npy files.

    Returns
    -------
    embeddings : float32 array of shape (len(sequences_df), EMBEDDING_DIM),
        row-aligned with sequences_df.
    """
    participants = sequences_df[PARTICIPANT_COL].unique()
    embeddings = np.empty((len(sequences_df), EMBEDDING_DIM), dtype=np.float32)

    for participant in participants:
        # Load pre-computed files
        emb_path = embedding_dir / f"{participant}_embeddings.npy"
        parquet_path = embedding_dir / f"{participant}_downsampled.parquet"

        if not emb_path.exists() or not parquet_path.exists():
            raise FileNotFoundError(
                f"Pre-computed embeddings not found for participant '{participant}'. "
                f"Expected files:\n"
                f"  {emb_path}\n"
                f"  {parquet_path}\n"
                f"Run the embedding script first:\n"
                f"  python -m malid_lite.training.compute_model3_embeddings "
                f"--metadata-path <path>\n"
                f"Or use --compute-embeddings to compute inline."
            )

        participant_emb = np.load(str(emb_path)).astype(np.float32)  # float16 -> float32
        participant_df = pd.read_parquet(parquet_path)

        # Find which rows in sequences_df belong to this participant
        mask = sequences_df[PARTICIPANT_COL] == participant
        n_fold_rows = mask.sum()

        if len(participant_emb) != len(participant_df):
            raise ValueError(
                f"Embedding/parquet mismatch for {participant}: "
                f"{len(participant_emb)} embeddings vs {len(participant_df)} rows"
            )

        if n_fold_rows > len(participant_df):
            raise ValueError(
                f"Fold has MORE rows ({n_fold_rows}) than pre-computed embeddings "
                f"({len(participant_df)}) for participant {participant}. "
                f"Embeddings should cover at least all fold data. "
                f"Re-run compute_model3_embeddings.py to regenerate."
            )

        fold_subset = sequences_df.loc[mask]
        row_indices = np.where(mask)[0]

        if n_fold_rows == len(participant_df):
            # Exact match: all pre-computed rows are in this fold subset.
            # Use the fast alignment path (positional check, then reorder if needed).
            aligned_emb = _align_embeddings(
                fold_subset, participant_df, participant_emb, participant,
            )
            embeddings[row_indices] = aligned_emb
        else:
            # Subset: fold has fewer rows than pre-computed (e.g., one fold's
            # training split vs all-fold embeddings). Use key-based lookup.
            key_cols = _get_downsampling_key_cols(participant_df)

            # Build key → embedding row index mapping from pre-computed data
            resolved_pre = [_resolve_col(participant_df, c) for c in key_cols]
            precomputed_keys = [
                _make_hashable_key(t)
                for t in zip(*(participant_df[c].values for c in resolved_pre))
            ]
            key_to_idx = {k: i for i, k in enumerate(precomputed_keys)}

            # Look up each fold row's embedding by its downsampling key
            resolved_fold = [_resolve_col(fold_subset, c) for c in key_cols]
            fold_keys = [
                _make_hashable_key(t)
                for t in zip(*(fold_subset[c].values for c in resolved_fold))
            ]
            for i, key in enumerate(fold_keys):
                idx = key_to_idx.get(key)
                if idx is None:
                    raise ValueError(
                        f"Fold row key not found in pre-computed embeddings for "
                        f"participant {participant}. Key: {key}. "
                        f"Re-run compute_model3_embeddings.py to regenerate."
                    )
                embeddings[row_indices[i]] = participant_emb[idx]

    return embeddings


def compute_embeddings_inline(
    sequences_df: pd.DataFrame,
    device: Optional[str],
    batch_size: int,
) -> np.ndarray:
    """Compute ESM-2 embeddings inline (fallback when pre-computed not available).

    Parameters
    ----------
    sequences_df : Sequences DataFrame with CDR3_COL.
    device       : Device for ESM-2 ('cuda', 'mps', 'cpu', or None for auto).
    batch_size   : Sequences per batch.

    Returns
    -------
    embeddings : float32 array of shape (len(sequences_df), 640).
    """
    logger.info(
        f"  Computing ESM-2 embeddings inline for {len(sequences_df):,} sequences..."
    )
    n_null = sequences_df[CDR3_COL].isna().sum()
    if n_null > 0:
        logger.warning(
            f"  {n_null} sequences have NaN CDR3 — these will produce meaningless "
            f"embeddings. This indicates a data quality issue in DOWNSAMPLED data."
        )
    cdr3_seqs = sequences_df[CDR3_COL].fillna("").tolist()
    return compute_esm2_embeddings(cdr3_seqs, batch_size=batch_size, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    fold_id: int,
    model_name: str,
    n_test: int,
    reference_class: Optional[str] = None,
    n_train_sequences_stage1: Optional[int] = None,
    n_train_sequences_stage2: Optional[int] = None,
    n_train_specimens_stage1: Optional[int] = None,
    n_train_specimens_stage2: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Compute evaluation metrics for one fold.

    Model 3 never abstains (every specimen has sequences -> Stage 2 always predicts).

    Parameters
    ----------
    n_test                    : Number of *specimens* in the test fold.
    n_train_sequences_stage1  : Sequences in train_smaller1 (Stage 1). Optional metadata.
    n_train_sequences_stage2  : Sequences in train_smaller2 (Stage 2). Optional metadata.
    n_train_specimens_stage1  : Specimens in train_smaller1. Optional metadata.
    n_train_specimens_stage2  : Specimens in train_smaller2. Optional metadata.
    """
    n_correct = int(accuracy_score(y_true, y_pred, normalize=False))
    results = {
        "fold_id": fold_id,
        "model_name": model_name,
        "n_scored": n_test,
        "n_abstained": 0,
        "abstention_rate": 0.0,
        "n_train_sequences_stage1": n_train_sequences_stage1,
        "n_train_sequences_stage2": n_train_sequences_stage2,
        "n_train_specimens_stage1": n_train_specimens_stage1,
        "n_train_specimens_stage2": n_train_specimens_stage2,
        "accuracy": n_correct / n_test if n_test > 0 else 0.0,
    }

    # Multiclass metrics: only meaningful for 3+ classes.
    # For binary (2-class), these are left as None and the binary-specific
    # auroc_binary / auprc_binary below are used instead.
    if len(classes) >= 3:
        # Use custom multiclass_metrics (from the original Mal-ID paper) which
        # handle unnormalized probabilities natively — no need to normalize.
        # Model 3's BinaryOvR outputs independent per-class probabilities that
        # don't sum to 1; sklearn would reject these.
        try:
            results["auroc_ovo_weighted"] = float(multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average="weighted",
                multi_class="ovo",
                labels=classes,
            ))
        except ValueError as e:
            logger.warning(f"  AUROC OvO failed: {e}")
            results["auroc_ovo_weighted"] = None

        try:
            results["auprc_ovo_weighted"] = float(multiclass_metrics.auprc(
                y_true, y_proba,
                average="weighted",
                multi_class="ovo",
                labels=classes,
            ))
        except ValueError as e:
            logger.warning(f"  AUPRC OvO failed: {e}")
            results["auprc_ovo_weighted"] = None

        # Per-class AUROC OvR (same custom metrics — handles unnormalized probs)
        auroc_ovr_per_class = {}
        try:
            per_class_scores = multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average=None,
                multi_class="ovr",
                labels=classes,
            )
            for cls, score in zip(classes, per_class_scores):
                auroc_ovr_per_class[str(cls)] = float(score)
        except ValueError as e:
            logger.warning(f"  Per-class AUROC OvR failed: {e}")
            for cls in classes:
                auroc_ovr_per_class[str(cls)] = None
        results["auroc_ovr_per_class"] = auroc_ovr_per_class
    else:
        results["auroc_ovo_weighted"] = None
        results["auprc_ovo_weighted"] = None
        results["auroc_ovr_per_class"] = None

    # Log loss needs normalized probabilities for multiclass.
    if len(classes) >= 3:
        row_sums = y_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_proba_for_loss = y_proba / row_sums
    else:
        y_proba_for_loss = y_proba
    try:
        results["log_loss"] = float(log_loss(y_true, y_proba_for_loss, labels=classes))
    except ValueError as e:
        logger.warning(f"  Log loss failed: {e}")
        results["log_loss"] = None

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    results["classes"] = [str(c) for c in classes]

    # Binary AUROC/AUPRC: only for 2-class case (binary / multi-binary modes)
    # Use P(disease) as the score, with disease=1 and reference=0
    if len(classes) == 2 and reference_class is not None:
        str_classes = [str(c) for c in classes]
        if str(reference_class) in str_classes:
            disease_class = next(c for c in str_classes if c != str(reference_class))
            disease_idx = str_classes.index(disease_class)
            y_score = y_proba[:, disease_idx]
            y_binary = (y_true == disease_class).astype(int)
            try:
                results["auroc_binary"] = float(roc_auc_score(y_binary, y_score))
            except ValueError as e:
                logger.warning(f"  Binary AUROC failed: {e}")
                results["auroc_binary"] = None
            try:
                results["auprc_binary"] = float(average_precision_score(y_binary, y_score))
            except ValueError as e:
                logger.warning(f"  Binary AUPRC failed: {e}")
                results["auprc_binary"] = None

    raw_preds = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
    }
    return results, raw_preds


# ---------------------------------------------------------------------------
# Fold loop
# ---------------------------------------------------------------------------

def _get_fold_artifact_paths(output_dir: Path, fold_id: int) -> List[Path]:
    """Return the four artifact paths that constitute a complete fold."""
    return [
        output_dir / f"fold_{fold_id}_stage1.pkl",
        output_dir / f"fold_{fold_id}_stage2.pkl",
        output_dir / f"fold_{fold_id}_results.json",
        output_dir / f"fold_{fold_id}_predictions.pkl",
    ]


def _check_fold_complete(output_dir: Path, fold_id: int) -> bool:
    """Check whether all artifacts for a fold exist on disk and are non-trivial.

    A fold is considered complete if all four files are present and the .pkl
    files are at least 1 KB (guards against truncated files from a crash
    during pickle.dump).
    """
    _MIN_PKL_BYTES = 1024
    for f in _get_fold_artifact_paths(output_dir, fold_id):
        if not f.exists():
            return False
        # .pkl files can be corrupt if a crash happened during write
        if f.suffix == ".pkl" and f.stat().st_size < _MIN_PKL_BYTES:
            return False
    return True


def _load_fold_results(output_dir: Path, fold_id: int) -> Tuple[Dict, Optional[Dict], List[Dict]]:
    """Load saved fold artifacts for resume.

    Returns
    -------
    (eval_results, raw_preds, predictions_rows) matching what _run_fold_loop
    produces per fold during live training.
    """
    results_path = output_dir / f"fold_{fold_id}_results.json"
    with open(results_path, "r") as f:
        eval_results = json.load(f)

    preds_path = output_dir / f"fold_{fold_id}_predictions.pkl"
    with open(preds_path, "rb") as f:
        preds_data = pickle.load(f)

    raw_preds = preds_data["raw_preds"]
    predictions_rows = preds_data["predictions_rows"]
    return eval_results, raw_preds, predictions_rows


def _run_fold_loop(
    loader: MalIDPublishedDataLoader,
    fold_ids: List[int],
    output_dir: Path,
    locus: str,
    n_estimators_stage1: int,
    n_estimators_stage2: int,
    n_jobs: int,
    verbose: int,
    embedding_dir: Optional[Path] = None,
    compute_embeddings_flag: bool = False,
    device: Optional[str] = None,
    embedding_batch_size: int = 64,
    aggregation_strategy: Optional[AggregationStrategy] = None,
    entropy_threshold_fraction: Optional[float] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
    resume: bool = False,
    resume_from_stage2: bool = False,
    resume_from_evaluation: bool = False,
    run_params: Optional[dict] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Run training + evaluation for all specified folds.

    Parameters
    ----------
    disease_filter : (disease, reference_class) for binary/multi-binary; None for multiclass.
    entropy_threshold_fraction : Fraction of max entropy to use as cutoff (only
        used when aggregation_strategy is entropy_cutoff). Passed through to
        SequenceLevelClassifier. None uses the factory default (0.20).
    resume         : If True, skip folds whose artifacts already exist on disk
                     and reload their results for aggregation.
    resume_from_stage2 : If True, load Stage 1 from saved artifacts but retrain
                     Stage 2 from scratch. Automatically removes stale Stage 2,
                     results, and prediction artifacts. Use this when changing
                     Stage-2-only params (aggregation, entropy threshold,
                     n_estimators_stage2, reweigh_by_subset_frequencies).
                     Requires Stage 1 artifacts to exist.
    resume_from_evaluation : If True, load Stage 1 and Stage 2 from saved
                     artifacts and re-run evaluation only. Automatically removes
                     stale results and prediction artifacts. Requires both
                     Stage 1 and Stage 2 artifacts to exist.
    run_params     : Dict with classification_mode, diseases, dataset_name for
                     artifact metadata validation on resume.

    Returns
    -------
    (all_eval_results, aggregated_by_model)
    """
    all_eval_results: List[Dict] = []
    raw_preds_list: List[Optional[Dict]] = []
    predictions_rows: List[Dict] = []
    fold_timings: List[Dict[str, float]] = []

    # Build model kwargs once (shared by all folds)
    ref_class_for_model = disease_filter[1] if disease_filter else None
    model_kwargs = dict(
        n_estimators_stage1=n_estimators_stage1,
        n_estimators_stage2=n_estimators_stage2,
        n_jobs=n_jobs,
        reference_class=ref_class_for_model,
        verbose=verbose,
    )

    def _make_model() -> SequenceLevelClassifier:
        """Build a fresh (unfitted) model for this fold."""
        if aggregation_strategy is not None:
            # User specified an explicit aggregation strategy via CLI
            extra = {}
            if entropy_threshold_fraction is not None:
                extra["entropy_threshold_fraction"] = entropy_threshold_fraction
            return SequenceLevelClassifier(
                locus=locus,
                aggregation_strategy=aggregation_strategy,
                exclude_rare_v_genes=True,
                reweigh_by_subset_frequencies=True,
                **extra,
                **model_kwargs,
            )
        # aggregation_strategy is None (--aggregation-strategy auto):
        # use paper-best factory per locus (TCR=entropy_cutoff 0.20, BCR=mean)
        elif locus == "TCR":
            return make_tcr_model(**model_kwargs)
        else:
            return make_bcr_model(**model_kwargs)

    rp = run_params or {}

    # --- Upfront artifact cleanup for targeted resume modes ---
    # Validate required artifacts for ALL folds first (fail fast before
    # deleting anything), then delete downstream artifacts for ALL folds
    # before any training starts.  This prevents mixed artifacts from
    # different runs if a crash occurs mid-way through the fold loop
    # (e.g. fold 0 gets new Stage 2 but fold 2 still has old Stage 2
    # from a previous run).
    if resume_from_stage2 or resume_from_evaluation:
        # Pass 1: validate all folds (for multiclass/binary this is the
        # only validation; for multi-binary a cross-pair check already ran
        # in main(), but repeating per-pair is cheap and keeps the function
        # self-contained).
        validation_errors = _validate_resume_artifacts(
            output_dir, fold_ids, resume_from_stage2, resume_from_evaluation,
        )
        if validation_errors:
            mode_name = "--resume-from-stage2" if resume_from_stage2 else "--resume-from-evaluation"
            detail = "\n".join(validation_errors)
            if resume_from_stage2:
                hint = (
                    "Run without --resume-from-stage2 to train from scratch, "
                    "or use --fold-ids to resume only the folds that have "
                    "Stage 1 artifacts."
                )
            else:
                hint = (
                    "Use --resume-from-stage2 if only Stage 1 is available, "
                    "or run without resume flags to train from scratch."
                )
            raise ValueError(
                f"{mode_name} requires saved artifacts, but some folds "
                f"are missing them:\n{detail}\n{hint}"
            )

        # Pass 2: delete downstream artifacts and log plan for each fold
        for fold_id in fold_ids:
            stage1_path = output_dir / f"fold_{fold_id}_stage1.pkl"
            stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
            results_path = output_dir / f"fold_{fold_id}_results.json"
            predictions_path = output_dir / f"fold_{fold_id}_predictions.pkl"

            all_artifacts = {
                stage1_path.name: stage1_path.exists(),
                stage2_path.name: stage2_path.exists(),
                results_path.name: results_path.exists(),
                predictions_path.name: predictions_path.exists(),
            }
            found = [name for name, exists in all_artifacts.items() if exists]

            if resume_from_stage2:
                to_remove = [stage2_path, results_path, predictions_path]
                removed = [p.name for p in to_remove if p.exists()]
                for p in to_remove:
                    if p.exists():
                        p.unlink()
                logger.info(
                    f"Fold {fold_id}: --resume-from-stage2\n"
                    f"  Found on disk: {', '.join(found)}\n"
                    f"  Keeping:       {stage1_path.name} (Stage 1 models)\n"
                    f"  Deleting:      {', '.join(removed) if removed else '(none)'}\n"
                    f"  Will do:       load Stage 1 -> retrain Stage 2 -> evaluate on test"
                )
            else:  # resume_from_evaluation
                to_remove = [results_path, predictions_path]
                removed = [p.name for p in to_remove if p.exists()]
                for p in to_remove:
                    if p.exists():
                        p.unlink()
                logger.info(
                    f"Fold {fold_id}: --resume-from-evaluation\n"
                    f"  Found on disk: {', '.join(found)}\n"
                    f"  Keeping:       {stage1_path.name}, {stage2_path.name}\n"
                    f"  Deleting:      {', '.join(removed) if removed else '(none)'}\n"
                    f"  Will do:       load Stage 1 + Stage 2 -> evaluate on test"
                )

    for fold_id in fold_ids:
        stage1_path = output_dir / f"fold_{fold_id}_stage1.pkl"
        stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
        results_path = output_dir / f"fold_{fold_id}_results.json"
        predictions_path = output_dir / f"fold_{fold_id}_predictions.pkl"

        # --- Resume: skip folds with complete artifacts on disk ---
        if resume:
            if _check_fold_complete(output_dir, fold_id):
                logger.info(f"\n{'='*60}")
                logger.info(f"Fold {fold_id} — skipped (all 4 artifacts found on disk)")
                logger.info(
                    f"  Found: {stage1_path.name}, {stage2_path.name}, "
                    f"{results_path.name}, {predictions_path.name}\n"
                    f"  Will do: load existing results (no training or evaluation)"
                )
                logger.info(f"{'='*60}")

                # Validate artifacts against current run params.
                tmp_model = _make_model()
                full_params = _build_model_params(tmp_model, **rp)
                del tmp_model

                # Validate Stage 2 (full params including aggregation)
                with open(stage2_path, "rb") as f:
                    s2_meta = pickle.load(f).get("_meta", {})
                try:
                    _validate_artifact_meta(
                        s2_meta, "Stage 2", fold_id,
                        current_model_params=full_params,
                    )
                except ValueError as e:
                    raise ValueError(
                        f"{e}\n\n"
                        f"Hint: If you changed Stage-2-only parameters "
                        f"(aggregation strategy, entropy threshold, "
                        f"n_estimators_stage2, reweigh_by_subset_frequencies), "
                        f"use --resume-from-stage2 instead of --resume to "
                        f"retrain Stage 2 while keeping the saved Stage 1 models."
                    ) from None

                # Validate Stage 1 (excluding Stage-2-only params)
                with open(stage1_path, "rb") as f:
                    s1_meta = pickle.load(f).get("_meta", {})
                stage1_params = {k: v for k, v in full_params.items()
                                 if k not in _STAGE2_ONLY_PARAMS}
                _validate_artifact_meta(
                    s1_meta, "Stage 1", fold_id, locus=locus,
                    current_model_params=stage1_params,
                )
                _log_resumed_artifact(s1_meta, "Stage 1", data_sizes_validated=False)

                eval_results, raw_preds, fold_pred_rows = _load_fold_results(output_dir, fold_id)
                all_eval_results.append(eval_results)
                raw_preds_list.append(raw_preds)
                predictions_rows.extend(fold_pred_rows)
                continue
            else:
                # Log which artifacts exist vs missing, and what will actually happen.
                stage1_exists = stage1_path.exists()
                stage2_exists = stage2_path.exists()
                all_artifacts = {
                    stage1_path.name: stage1_exists,
                    stage2_path.name: stage2_exists,
                    results_path.name: results_path.exists(),
                    predictions_path.name: predictions_path.exists(),
                }
                found = [name for name, exists in all_artifacts.items() if exists]
                missing = [name for name, exists in all_artifacts.items() if not exists]
                if stage1_exists and stage2_exists:
                    will_do = "load Stage 1 + Stage 2 -> evaluate on test"
                elif stage1_exists:
                    will_do = "load Stage 1 -> train Stage 2 -> evaluate on test"
                else:
                    will_do = "train Stage 1 -> train Stage 2 -> evaluate on test"
                logger.info(
                    f"Fold {fold_id}: --resume (partial artifacts found)\n"
                    f"  Found on disk: {', '.join(found) if found else '(none)'}\n"
                    f"  Missing:       {', '.join(missing)}\n"
                    f"  Will do:       {will_do}"
                )

        t_fold_start = time.monotonic()
        timings: Dict[str, float] = {"fold_id": fold_id}

        pair_tag = (
            f" [{make_pair_name(disease_filter[0], disease_filter[1])}]"
            if disease_filter else ""
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_id}{pair_tag}")
        logger.info(f"{'='*60}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Determine resume point for this fold ---
        # Check existence AND minimum file size to guard against corrupt artifacts
        # (e.g. a crash during pickle.dump leaves a truncated file on disk).
        _MIN_ARTIFACT_BYTES = 1024  # any valid artifact is at least a few KB
        resume_stage1 = (
            resume and stage1_path.exists()
            and stage1_path.stat().st_size >= _MIN_ARTIFACT_BYTES
        )
        resume_stage2 = (
            resume and stage2_path.exists()
            and stage2_path.stat().st_size >= _MIN_ARTIFACT_BYTES
        )
        if resume:
            for tag, path, flag in [
                ("Stage 1", stage1_path, resume_stage1),
                ("Stage 2", stage2_path, resume_stage2),
            ]:
                if path.exists() and not flag:
                    size = path.stat().st_size
                    logger.warning(
                        f"  {tag} artifact exists but looks corrupt "
                        f"({size:,} bytes < {_MIN_ARTIFACT_BYTES:,}). "
                        f"Ignoring and retraining."
                    )
        # need_training_data: False only if BOTH stages are resumed
        need_training_data = not resume_stage1 or not resume_stage2

        # ------------------------------------------------------------------ #
        # Build model (before loading data/embeddings to minimize memory)     #
        # ------------------------------------------------------------------ #
        model = _make_model()

        # ------------------------------------------------------------------ #
        # Load training data (only if at least one stage needs training)      #
        # ------------------------------------------------------------------ #
        ts1 = None
        ts2 = None
        if need_training_data:
            t0 = time.monotonic()
            logger.info("Loading training data (fold=train)...")
            train_seq, train_meta = load_and_prepare_fold(loader, fold_id, "train")
            if disease_filter:
                disease, ref = disease_filter
                train_seq, train_meta = filter_to_binary_pair(
                    train_seq, train_meta, disease, ref)

            ts1, ts2 = split_train_smaller(train_seq, train_meta)
            timings["load_train_data"] = time.monotonic() - t0

            logger.info(
                f"  Train fold: {len(train_seq):,} sequences, "
                f"{train_seq[PARTICIPANT_COL].nunique()} participants "
                f"[{_fmt_elapsed(timings['load_train_data'])}]"
            )
            logger.info(
                f"  train_smaller1: {len(ts1):,} sequences, "
                f"{ts1[PARTICIPANT_COL].nunique()} participants"
            )
            logger.info(
                f"  train_smaller2: {len(ts2):,} sequences, "
                f"{ts2[PARTICIPANT_COL].nunique()} participants"
            )
            del train_seq, train_meta
        elif resume_stage1 and resume_stage2:
            logger.info("  Loading Stage 1 and Stage 2 from saved artifacts (no training data needed)")

        # ------------------------------------------------------------------ #
        # Stage 1: train or load from resume                                  #
        # ------------------------------------------------------------------ #
        if resume_stage1:
            # --- Resume Stage 1: load from disk ---
            # Validate classes against training data if available
            expected_classes = None
            if ts1 is not None:
                expected_classes = sorted(ts1[DISEASE_COL].unique().tolist())
            _load_stage1_artifact(model, stage1_path, fold_id, locus,
                                  expected_classes=expected_classes,
                                  run_params=run_params, ts1=ts1)
            # Diagnostic #2: per-group class coverage (verbose >= 2, no training stats since S1 was loaded)
            if verbose >= 2:
                model._log_stage1_group_diagnostics()
        else:
            # --- Train Stage 1 ---
            assert ts1 is not None, "ts1 must be loaded for Stage 1 training"
            ts1 = ts1.reset_index(drop=True)

            logger.info("  Loading ts1 embeddings...")
            t0 = time.monotonic()
            if compute_embeddings_flag or embedding_dir is None:
                emb_ts1 = compute_embeddings_inline(ts1, device, embedding_batch_size)
            else:
                emb_ts1 = load_precomputed_embeddings(ts1, embedding_dir)
            timings["load_ts1_embeddings"] = time.monotonic() - t0
            logger.info(
                f"  Loaded ts1 embeddings: {len(ts1):,} sequences, "
                f"{ts1[PARTICIPANT_COL].nunique()} participants "
                f"[{_fmt_elapsed(timings['load_ts1_embeddings'])}]"
            )

            t0 = time.monotonic()
            logger.info("Training Stage 1 (per-group sequence classifiers)...")
            model.fit_stage1(ts1, emb_ts1)
            timings["train_stage1"] = time.monotonic() - t0
            n_groups = len(model.group_models_)
            logger.info(
                f"  Stage 1 complete: {n_groups} group models trained "
                f"[{_fmt_elapsed(timings['train_stage1'])}]"
            )

            del emb_ts1  # free ~26 GB before Stage 2

            # Save Stage 1 artifact with metadata
            logger.info("  Saving Stage 1 artifact...")
            _save_stage1_artifact(model, stage1_path, fold_id, ts1,
                                  run_params=run_params)

        # ------------------------------------------------------------------ #
        # Stage 2: train or load from resume                                  #
        # ------------------------------------------------------------------ #
        if resume_stage2:
            # --- Resume Stage 2: load from disk ---
            expected_classes = [str(c) for c in model.classes_]
            try:
                _load_stage2_artifact(model, stage2_path, fold_id,
                                      expected_classes=expected_classes,
                                      run_params=run_params, ts2=ts2)
            except ValueError as e:
                raise ValueError(
                    f"{e}\n\n"
                    f"Hint: If you changed Stage-2-only parameters "
                    f"(aggregation strategy, entropy threshold, "
                    f"n_estimators_stage2, reweigh_by_subset_frequencies), "
                    f"use --resume-from-stage2 instead to retrain Stage 2 "
                    f"while keeping the saved Stage 1 models."
                ) from None
            # Diagnostic #4: feature importance from loaded Stage 2 (verbose >= 2)
            if verbose >= 2:
                model._log_stage2_feature_importance()
        else:
            # --- Train Stage 2 ---
            assert ts2 is not None, "ts2 must be loaded for Stage 2 training"
            ts2 = ts2.reset_index(drop=True)

            logger.info("  Loading ts2 embeddings...")
            t0 = time.monotonic()
            if compute_embeddings_flag or embedding_dir is None:
                emb_ts2 = compute_embeddings_inline(ts2, device, embedding_batch_size)
            else:
                emb_ts2 = load_precomputed_embeddings(ts2, embedding_dir)
            timings["load_ts2_embeddings"] = time.monotonic() - t0
            logger.info(
                f"  Loaded ts2 embeddings: {len(ts2):,} sequences, "
                f"{ts2[PARTICIPANT_COL].nunique()} participants "
                f"[{_fmt_elapsed(timings['load_ts2_embeddings'])}]"
            )

            t0 = time.monotonic()
            logger.info("Training Stage 2 (specimen-level rollup)...")
            model.fit_stage2(ts2, emb_ts2)
            timings["train_stage2"] = time.monotonic() - t0
            n_features = len(model.feature_columns_) if model.feature_columns_ else 0
            logger.info(
                f"  Stage 2 complete: {n_features} specimen-level features "
                f"[{_fmt_elapsed(timings['train_stage2'])}]"
            )

            del emb_ts2  # free ~13 GB before test

            # Save Stage 2 artifact with metadata
            logger.info("  Saving Stage 2 artifact...")
            _save_stage2_artifact(model, stage2_path, fold_id, ts2,
                                  run_params=run_params)

        # Capture training data counts before freeing (for evaluate_on_test metadata).
        # These are None when both stages were resumed (no training data loaded).
        n_train_seq_s1 = None
        n_train_seq_s2 = None
        n_train_spec_s1 = None
        n_train_spec_s2 = None
        if need_training_data:
            if ts1 is not None:
                n_train_seq_s1 = len(ts1)
                n_train_spec_s1 = int(ts1[SPECIMEN_COL].nunique())
            if ts2 is not None:
                n_train_seq_s2 = len(ts2)
                n_train_spec_s2 = int(ts2[SPECIMEN_COL].nunique())
        del ts1, ts2

        # ------------------------------------------------------------------ #
        # Load test data + embeddings (only now, after training frees RAM)    #
        # Memory: only test embeddings (~21 GB) in RAM during prediction      #
        # ------------------------------------------------------------------ #
        if disease_filter:
            disease, ref = disease_filter
        t0 = time.monotonic()
        logger.info("Loading test data...")
        test_seq, test_meta = load_and_prepare_fold(loader, fold_id, "test")
        if disease_filter:
            test_seq, test_meta = filter_to_binary_pair(test_seq, test_meta, disease, ref)
            if len(test_seq) == 0:
                raise ValueError(
                    f"Test fold {fold_id} has zero sequences after filtering to "
                    f"'{disease}' vs '{ref}'. This indicates a data/fold design issue — "
                    f"every fold should contain test specimens for both classes."
                )
        timings["load_test_data"] = time.monotonic() - t0

        logger.info(
            f"  Test fold: {len(test_seq):,} sequences, "
            f"{test_seq[SPECIMEN_COL].nunique()} specimens "
            f"[{_fmt_elapsed(timings['load_test_data'])}]"
        )

        logger.info("  Loading test embeddings...")
        t0 = time.monotonic()
        test_seq = test_seq.reset_index(drop=True)
        if compute_embeddings_flag or embedding_dir is None:
            emb_test = compute_embeddings_inline(test_seq, device, embedding_batch_size)
        else:
            emb_test = load_precomputed_embeddings(test_seq, embedding_dir)
        timings["load_test_embeddings"] = time.monotonic() - t0
        logger.info(
            f"  Loaded test embeddings: {len(test_seq):,} sequences "
            f"[{_fmt_elapsed(timings['load_test_embeddings'])}]"
        )

        # ------------------------------------------------------------------ #
        # Evaluate on test fold                                               #
        # ------------------------------------------------------------------ #
        t0 = time.monotonic()
        logger.info("Evaluating on test fold...")
        proba_df = model.predict_proba(test_seq, emb_test)
        timings["predict"] = time.monotonic() - t0
        logger.info(f"  Prediction complete [{_fmt_elapsed(timings['predict'])}]")

        del emb_test  # free ~21 GB after prediction
        classes = model.classes_

        # Build specimen → disease mapping for test fold
        # (proba_df is indexed by specimen_label; need ground-truth labels in same order)
        specimen_disease = (
            test_seq.drop_duplicates(SPECIMEN_COL)
            .set_index(SPECIMEN_COL)[DISEASE_COL]
        )
        missing = [s for s in proba_df.index if s not in specimen_disease.index]
        if missing:
            raise ValueError(
                f"Test specimens have no disease label in test_seq: {missing[:10]}. "
                f"This indicates a bug — all specimens should have disease labels "
                f"by this point (NaN rows are dropped upstream)."
            )
        y_true_arr = np.array([specimen_disease[s] for s in proba_df.index])
        y_proba_arr = proba_df.values
        proba_df_eval = proba_df

        # Predicted class = highest probability
        y_pred_arr = classes[np.argmax(y_proba_arr, axis=1)]

        t0 = time.monotonic()
        ref_class = disease_filter[1] if disease_filter else None
        eval_results, raw_preds = evaluate_on_test(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            classes=classes,
            fold_id=fold_id,
            model_name=MODEL_NAME,
            n_test=len(y_true_arr),
            reference_class=ref_class,
            n_train_sequences_stage1=n_train_seq_s1,
            n_train_sequences_stage2=n_train_seq_s2,
            n_train_specimens_stage1=n_train_spec_s1,
            n_train_specimens_stage2=n_train_spec_s2,
        )
        if disease_filter:
            eval_results["disease"] = disease_filter[0]
            eval_results["reference_class"] = disease_filter[1]

        timings["evaluate"] = time.monotonic() - t0
        logger.info(f"  Evaluation complete [{_fmt_elapsed(timings['evaluate'])}]")

        # Save per-fold results JSON
        results_path = output_dir / f"fold_{fold_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                eval_results, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )
        # Log primary metric: AUROC (multiclass uses OvO weighted, binary uses binary)
        auroc_val = eval_results.get("auroc_ovo_weighted") or eval_results.get("auroc_binary")
        auprc_val = eval_results.get("auprc_ovo_weighted") or eval_results.get("auprc_binary")
        metric_parts = []
        if auroc_val is not None:
            metric_parts.append(f"AUROC={auroc_val:.4f}")
        if auprc_val is not None:
            metric_parts.append(f"AUPRC={auprc_val:.4f}")
        metric_str = ", ".join(metric_parts) if metric_parts else "no metrics available"
        logger.info(f"  Fold {fold_id}: {metric_str}")

        all_eval_results.append(eval_results)
        raw_preds_list.append(raw_preds)

        # Collect per-specimen predictions for the combined CSV output
        fold_pred_rows: List[Dict] = []
        str_classes = [str(c) for c in classes]
        spec_to_part = test_meta.set_index(SPECIMEN_COL)[PARTICIPANT_COL]
        missing_parts = [s for s in proba_df_eval.index if s not in spec_to_part.index]
        if missing_parts:
            raise ValueError(
                f"Specimens in predictions but not in test_meta: {missing_parts[:10]}. "
                f"This indicates a bug in data loading."
            )

        if disease_filter:
            # Binary mode: one score column (P(disease)), binary label
            disease_class = next(c for c in str_classes if c != str(ref_class))
            disease_idx = str_classes.index(disease_class)
            for specimen, true_d, score in zip(
                proba_df_eval.index,
                y_true_arr,
                y_proba_arr[:, disease_idx],
            ):
                fold_pred_rows.append({
                    "participant_label": spec_to_part.get(specimen),
                    "specimen_label": specimen,
                    "disease_label": int(true_d == disease_class),
                    "disease_label_str": str(true_d),
                    "disease_model": disease_class,
                    "model_score": float(score),
                    "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                })
        else:
            # Multiclass mode: one score column per class
            for specimen, true_d, pred_d, proba_row in zip(
                proba_df_eval.index,
                y_true_arr,
                y_pred_arr,
                y_proba_arr,
            ):
                row = {
                    "participant_label": spec_to_part.get(specimen),
                    "specimen_label": specimen,
                    "true_disease": str(true_d),
                    "predicted_disease": str(pred_d),
                    "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                }
                for cls, score in zip(str_classes, proba_row):
                    row[f"score_{cls}"] = float(score)
                fold_pred_rows.append(row)
        predictions_rows.extend(fold_pred_rows)

        # Save per-fold predictions for resume support
        preds_path = output_dir / f"fold_{fold_id}_predictions.pkl"
        _save_predictions_artifact(
            raw_preds, fold_pred_rows, preds_path, fold_id, classes,
            n_test_specimens=len(y_true_arr),
        )

        # Fold timing summary
        timings["fold_total"] = time.monotonic() - t_fold_start
        fold_timings.append(timings)
        logger.info(f"\n  Fold {fold_id} timing breakdown:")
        for step_name in [
            "load_train_data",
            "load_ts1_embeddings", "train_stage1",
            "load_ts2_embeddings", "train_stage2",
            "load_test_data", "load_test_embeddings",
            "predict", "evaluate",
        ]:
            if step_name in timings:
                logger.info(f"    {step_name:.<30s} {_fmt_elapsed(timings[step_name])}")
        logger.info(f"    {'fold_total':.<30s} {_fmt_elapsed(timings['fold_total'])}")

    # ------------------------------------------------------------------ #
    # Save combined predictions CSV and aggregate metrics                  #
    # ------------------------------------------------------------------ #
    if predictions_rows:
        if disease_filter:
            pred_csv_path = output_dir / f"{MODEL_NAME}_binary_predictions.csv"
        else:
            pred_csv_path = output_dir / f"{MODEL_NAME}_multiclass_predictions.csv"
        pd.DataFrame(predictions_rows).to_csv(pred_csv_path, index=False)
        logger.info(f"  Saved predictions CSV: {pred_csv_path}")

    # Aggregate across folds
    ref_class_for_agg = disease_filter[1] if disease_filter else None
    disease_for_agg = disease_filter[0] if disease_filter else None
    disease_filter_for_agg = (disease_for_agg, ref_class_for_agg) if disease_filter else None
    aggregated = aggregate_fold_results(
        fold_metrics=all_eval_results,
        fold_raw_preds=raw_preds_list,
        disease_filter=disease_filter_for_agg,
    )

    # Cross-fold timing summary
    if len(fold_timings) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Timing summary across all folds:")
        logger.info(f"{'='*60}")
        step_names = [
            "load_train_data",
            "load_ts1_embeddings", "train_stage1",
            "load_ts2_embeddings", "train_stage2",
            "load_test_data", "load_test_embeddings",
            "predict", "evaluate", "fold_total",
        ]
        for step_name in step_names:
            vals = [t[step_name] for t in fold_timings if step_name in t]
            if vals:
                total = sum(vals)
                mean = total / len(vals)
                logger.info(
                    f"  {step_name:.<30s} "
                    f"mean={_fmt_elapsed(mean)}  total={_fmt_elapsed(total)}"
                )

    return all_eval_results, {MODEL_NAME: aggregated}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model 3 (sequence-level classifier) for Mal-ID-Lite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Data and cache paths ---
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to the metadata TSV file (e.g., data/metadata.tsv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Path to raw data directory (AIRR-format files). "
            "Required if the cache does not exist yet. "
            "Not needed when a complete cache is available."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for preprocessed data. "
            "Default: cache/<dataset-name>/ under the project root. "
            "Ignored when --dont-use-cache is set."
        ),
    )
    parser.add_argument(
        "--gene-reference-path",
        type=Path,
        default=None,
        # TODO: Audit whether CDR1/CDR2/FR columns from this file are actually used
        # by any model. If not, this argument can be removed (see TODO_for_release.md #4).
        help="Path to V-gene CDR reference file (e.g., tcrb_v_gene_cdrs.generated.tsv).",
    )
    parser.add_argument(
        "--dont-use-cache",
        action="store_true",
        help=(
            "Disable caching entirely. All data will be loaded and preprocessed "
            "from raw files on every run. Requires --data-dir."
        ),
    )

    # --- Dataset and mode ---
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name (used for cache and output directories).",
    )
    parser.add_argument(
        "--gene-locus",
        default="TCR",
        choices=["TCR"],
        help="Gene locus (default: TCR). Only TCR is supported at the moment.",
    )
    parser.add_argument(
        "--classification-mode",
        default="multiclass",
        choices=["multiclass", "binary", "multi-binary"],
        help="Classification mode. Default: multiclass.",
    )
    parser.add_argument(
        "--reference-class",
        default=None,
        help="Reference/negative class for binary and multi-binary modes.",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=None,
        help="Explicit disease subset for binary or multi-binary modes.",
    )
    parser.add_argument(
        "--fold-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Fold IDs to train (default: all folds found in metadata). "
            "Example: --fold-ids 0 1 2"
        ),
    )
    parser.add_argument(
        "--n-estimators-stage1",
        type=int,
        default=100,
        help="Number of RF trees in Stage 1 (BCR only; ignored for TCR which uses glmnet ridge).",
    )
    parser.add_argument(
        "--n-estimators-stage2",
        type=int,
        default=100,
        help="Number of RF trees in Stage 2 (original Mal-ID uses sklearn default=100).",
    )
    agg_choices = [s.name for s in AggregationStrategy]
    parser.add_argument(
        "--aggregation-strategy",
        default="auto",
        choices=["auto"] + agg_choices,
        help=(
            "Sequence-to-specimen aggregation strategy. "
            "'auto' (default) selects the paper-best per locus: "
            "TCR=entropy_cutoff (0.20), BCR=mean. "
            "Use entropy_cutoff with --entropy-threshold to set a custom threshold. "
            f"Options: auto, {', '.join(agg_choices)}."
        ),
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=None,
        help=(
            "Entropy threshold fraction for entropy_cutoff aggregation. "
            "E.g. 0.20 means keep sequences with entropy < 80%% of max entropy. "
            "Only used when --aggregation-strategy is entropy_cutoff. "
            "Default: 0.20 (paper setting)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Parallel workers for joblib-parallelized steps: "
            "Stage 1 V-gene group training, Stage 1 inner OvR binary classifiers, "
            "and Stage 2 binary classifiers. "
            "Default 4 (reasonable for a personal laptop)."
        ),
    )

    # --- Embedding parameters ---
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing pre-computed per-participant ESM-2 embeddings "
            "(from compute_model3_embeddings.py). "
            "Default: <cache-dir>/embeddings/. "
            "Each participant needs <label>_embeddings.npy and "
            "<label>_downsampled.parquet files."
        ),
    )
    parser.add_argument(
        "--compute-embeddings",
        action="store_true",
        help=(
            "Compute ESM-2 embeddings inline instead of loading pre-computed files. "
            "Use this if you haven't run compute_model3_embeddings.py yet. "
            "WARNING: This is slow (~3 hours per 10M sequences on Macbook Pro M4 Max MPS) "
            "and requires significant storage (~14 GB per 10M sequences). "
            "For repeated runs, pre-computing embeddings is strongly recommended."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for ESM-2 embedding: 'cuda', 'mps', 'cpu', or auto-detect. "
            "Only used with --compute-embeddings."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size for ESM-2 embedding (reduce if GPU OOM). "
            "Only used with --compute-embeddings."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress, 2=detailed).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run. Skips folds whose artifacts already exist "
            "in the output directory and reloads their results for aggregation. "
            "Validates that saved artifacts match current parameters. "
            "Useful when a long training run was interrupted mid-way."
        ),
    )
    parser.add_argument(
        "--resume-from-stage2",
        action="store_true",
        help=(
            "Load saved Stage 1 models and retrain Stage 2 from scratch. "
            "Automatically removes existing Stage 2, results, and prediction "
            "artifacts so they are regenerated. Use this when you want to "
            "change Stage-2-only parameters (aggregation strategy, entropy "
            "threshold, n_estimators_stage2, reweigh_by_subset_frequencies) "
            "without re-running the expensive Stage 1 training. "
            "Requires Stage 1 artifacts to exist."
        ),
    )
    parser.add_argument(
        "--resume-from-evaluation",
        action="store_true",
        help=(
            "Load saved Stage 1 and Stage 2 models and re-run evaluation only. "
            "Automatically removes existing results and prediction artifacts "
            "so they are regenerated. Requires both Stage 1 and Stage 2 "
            "artifacts to exist."
        ),
    )
    args = parser.parse_args()

    # --- Resolve cache and data paths ---
    if args.dont_use_cache:
        cache_dir = None
        if args.data_dir is None:
            parser.error("--data-dir is required when --dont-use-cache is set.")
    else:
        cache_dir = args.cache_dir or (PROJECT_ROOT / "cache" / args.dataset_name)
        participants_cache = cache_dir / "participants"
        cache_exists = (
            participants_cache.exists()
            and any(participants_cache.glob("*_clean.parquet"))
        )
        if not cache_exists and args.data_dir is None:
            parser.error(
                f"No existing cache found at {cache_dir}. "
                "Provide --data-dir so the cache can be built, or use --dont-use-cache "
                "to run without caching."
            )

    # --- Resolve embedding directory ---
    if args.embedding_dir is not None:
        embedding_dir = args.embedding_dir
    elif cache_dir is not None:
        embedding_dir = cache_dir / "embeddings"
    else:
        # No cache and no explicit embedding dir — only valid with --compute-embeddings
        embedding_dir = None

    # Validate embedding availability
    if not args.compute_embeddings:
        if embedding_dir is None:
            parser.error(
                "No embedding directory available (--dont-use-cache without --embedding-dir). "
                "Either provide --embedding-dir or add --compute-embeddings to compute "
                "embeddings inline.\n"
                "NOTE: Inline computation is slow (~3 hours per 10M sequences on Macbook Pro M4 Max MPS) "
                "and requires ~14 GB storage per 10M sequences. "
                "For repeated runs, pre-computing with compute_model3_embeddings.py is recommended."
            )
        elif not embedding_dir.exists() or not any(embedding_dir.glob("*_embeddings.npy")):
            parser.error(
                f"No pre-computed embeddings found in {embedding_dir}.\n"
                "Run the embedding script first:\n"
                "    python -m malid_lite.training.compute_model3_embeddings \\\n"
                f"        --metadata-path {args.metadata_path}\n\n"
                "Or add --compute-embeddings to compute embeddings inline.\n"
                "NOTE: Inline computation is slow (~3 hours per 10M sequences on Macbook Pro M4 Max MPS) "
                "and requires ~14 GB storage per 10M sequences."
            )

    if args.data_dir is not None and not args.data_dir.exists():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if not args.metadata_path.exists():
        parser.error(f"--metadata-path does not exist: {args.metadata_path}")
    if args.gene_reference_path is not None and not args.gene_reference_path.exists():
        parser.error(f"--gene-reference-path does not exist: {args.gene_reference_path}")

    t_total_start = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------ #
    # Setup loader                                                         #
    # ------------------------------------------------------------------ #
    t0 = time.monotonic()
    loader = MalIDPublishedDataLoader(
        data_dir=args.data_dir or Path("."),  # placeholder if cache covers all reads
        metadata_path=args.metadata_path,
        gene_reference_path=args.gene_reference_path,
        cache_dir=cache_dir,
        verbose=1,
    )

    disease_classes = get_dataset_disease_classes(args.metadata_path)
    fold_ids = args.fold_ids or sorted(
        loader.metadata["malid_cross_validation_fold_id_when_in_test_set"]
        .dropna().unique().astype(int).tolist()
    )
    logger.info(f"Loader setup [{_fmt_elapsed(time.monotonic() - t0)}]")

    reference_class = validate_mode_and_classes(
        classification_mode=args.classification_mode,
        disease_classes=disease_classes,
        reference_class=args.reference_class,
        diseases=args.diseases,
    )

    # Validate resume flags (at most one targeted resume mode)
    if args.resume_from_stage2 and args.resume_from_evaluation:
        parser.error(
            "--resume-from-stage2 and --resume-from-evaluation are mutually exclusive"
        )
    # Targeted resume implies --resume behavior for earlier stages
    if args.resume_from_stage2 or args.resume_from_evaluation:
        args.resume = True

    # Resolve aggregation strategy: "auto" → None (let factory pick per locus)
    if args.aggregation_strategy == "auto":
        agg_strategy = None
    else:
        agg_strategy = AggregationStrategy[args.aggregation_strategy]

    # Validate: --entropy-threshold only makes sense with entropy_cutoff
    if args.entropy_threshold is not None and agg_strategy != AggregationStrategy.entropy_cutoff:
        hint = ""
        if agg_strategy is None:
            hint = (
                " Note: 'auto' resolves to entropy_cutoff for TCR, but to "
                "use a custom threshold you must specify "
                "--aggregation-strategy entropy_cutoff explicitly."
            )
        parser.error(
            f"--entropy-threshold is only used with "
            f"--aggregation-strategy entropy_cutoff.{hint}"
        )

    # ------------------------------------------------------------------ #
    # Output directory                                                     #
    # ------------------------------------------------------------------ #
    base_dir = get_model_output_dir(
        model_name=MODEL_NAME,
        dataset_name=args.dataset_name,
        classification_mode=args.classification_mode,
        gene_locus=args.gene_locus,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Mirror all logging to a file in the output directory
    log_path = base_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting Model 3 training — {timestamp}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Reference class:     {args.reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {args.diseases or '(all)'}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Aggregation:         {agg_strategy.name if agg_strategy is not None else 'auto'}")
    logger.info(f"  Entropy threshold:   {args.entropy_threshold or 'default'}")
    logger.info(f"  Stage 1 estimators:  {args.n_estimators_stage1}")
    logger.info(f"  Stage 2 estimators:  {args.n_estimators_stage2}")
    logger.info(f"  n_jobs:              {args.n_jobs}")
    logger.info(f"  Verbose:             {args.verbose}")
    logger.info(f"  Resume:              {args.resume}")
    logger.info(f"  Resume from stage2:  {args.resume_from_stage2}")
    logger.info(f"  Resume from eval:    {args.resume_from_evaluation}")
    logger.info(f"  Embedding dir:       {embedding_dir}")
    logger.info(f"  Compute embeddings:  {args.compute_embeddings}")
    logger.info(f"  Device:              {args.device or 'auto'}")
    logger.info(f"  Base output dir:     {base_dir}")

    loop_kwargs = dict(
        loader=loader,
        fold_ids=fold_ids,
        locus=args.gene_locus,
        n_estimators_stage1=args.n_estimators_stage1,
        n_estimators_stage2=args.n_estimators_stage2,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        embedding_dir=embedding_dir,
        compute_embeddings_flag=args.compute_embeddings,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        aggregation_strategy=agg_strategy,
        entropy_threshold_fraction=args.entropy_threshold,
        resume=args.resume,
        resume_from_stage2=args.resume_from_stage2,
        resume_from_evaluation=args.resume_from_evaluation,
        run_params={
            "classification_mode": args.classification_mode,
            "diseases": args.diseases,
            "dataset_name": args.dataset_name,
        },
    )

    # ------------------------------------------------------------------ #
    # Multi-binary upfront validation for targeted resume modes           #
    # ------------------------------------------------------------------ #
    # For multi-binary, run_training_orchestration calls _run_fold_loop
    # once per disease pair.  Each call validates its own output_dir, but
    # if pair 3 of 5 fails, pairs 1-2 have already trained and deleted
    # artifacts.  To fail fast before any work starts, we validate ALL
    # pairs upfront here.  (Multiclass and single-binary only have one
    # output_dir, so _run_fold_loop's own validation is sufficient.)
    if (
        args.classification_mode == "multi-binary"
        and (args.resume_from_stage2 or args.resume_from_evaluation)
    ):
        # Resolve diseases_to_train the same way run_training_orchestration does
        if args.diseases is not None:
            _diseases_to_check = list(args.diseases)
        else:
            _diseases_to_check = [c for c in disease_classes if c != reference_class]

        mode_name = (
            "--resume-from-stage2" if args.resume_from_stage2
            else "--resume-from-evaluation"
        )
        all_errors: List[str] = []
        for _disease in _diseases_to_check:
            pair_name = make_pair_name(_disease, reference_class)
            pair_dir = base_dir / pair_name
            pair_errors = _validate_resume_artifacts(
                pair_dir, fold_ids,
                args.resume_from_stage2, args.resume_from_evaluation,
            )
            if pair_errors:
                all_errors.append(f"  {pair_name}/")
                all_errors.extend(f"    {e.strip()}" for e in pair_errors)

        if all_errors:
            detail = "\n".join(all_errors)
            if args.resume_from_stage2:
                hint = (
                    "Run without --resume-from-stage2 to train from scratch, "
                    "or use --diseases to resume only the pairs that have "
                    "Stage 1 artifacts for all folds."
                )
            else:
                hint = (
                    "Use --resume-from-stage2 if only Stage 1 is available, "
                    "or run without resume flags to train from scratch."
                )
            raise ValueError(
                f"{mode_name} requires saved artifacts, but some disease "
                f"pairs are missing them:\n{detail}\n{hint}"
            )

    # ------------------------------------------------------------------ #
    # Training orchestration (dispatches multiclass / binary / multi-bin) #
    # ------------------------------------------------------------------ #
    all_results = run_training_orchestration(
        base_dir=base_dir,
        classification_mode=args.classification_mode,
        reference_class=reference_class,
        diseases=args.diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_fold_loop,
        loop_kwargs=loop_kwargs,
    )

    # ------------------------------------------------------------------ #
    # Save summary JSON + Markdown results                                 #
    # ------------------------------------------------------------------ #
    run_info = {
        "Dataset": args.dataset_name,
        "Gene locus": args.gene_locus,
        "Classification mode": args.classification_mode,
        "Folds": ", ".join(str(f) for f in fold_ids),
        "Stage 1 classifier": (
            "glmnet ridge (OvR)" if args.gene_locus == "TCR"
            else f"RF ({args.n_estimators_stage1} trees)"
        ),
        "Aggregation strategy": (
            agg_strategy.name if agg_strategy is not None
            else f"auto ({('entropy_cutoff' if args.gene_locus == 'TCR' else 'mean')})"
        ),
        "Entropy threshold": args.entropy_threshold if args.entropy_threshold is not None else (
            _DEFAULT_ENTROPY_THRESHOLD if (agg_strategy == AggregationStrategy.entropy_cutoff or
                     (agg_strategy is None and args.gene_locus == "TCR")) else "N/A"
        ),
        "Stage 2 RF trees": args.n_estimators_stage2,
        "n_jobs (V-gene groups)": args.n_jobs,
        "Embedding source": "inline" if args.compute_embeddings else str(embedding_dir),
        "Embedding device": args.device or "auto",
    }
    if reference_class is not None:
        run_info["Reference class"] = reference_class

    # Write summary JSON (same envelope structure as Models 1/2)
    summary_path = base_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset_name": args.dataset_name,
                "classification_mode": args.classification_mode,
                "reference_class": args.reference_class,
                "diseases": args.diseases,
                "gene_locus": args.gene_locus,
                "fold_ids": fold_ids,
                "model_names": [MODEL_NAME],
                "aggregation_strategy": agg_strategy.name if agg_strategy is not None else "auto",
                "entropy_threshold_fraction": args.entropy_threshold if args.entropy_threshold is not None else (
                    _DEFAULT_ENTROPY_THRESHOLD if (agg_strategy == AggregationStrategy.entropy_cutoff or
                             (agg_strategy is None and args.gene_locus == "TCR")) else None
                ),
                "results_by_pair": {
                    key: val["fold_results"] for key, val in all_results.items()
                },
                "aggregated_by_pair": {
                    key: val["aggregated_by_model"] for key, val in all_results.items()
                },
            },
            f, indent=2,
            default=lambda x: (
                x.tolist() if isinstance(x, np.ndarray)
                else float(x) if isinstance(x, (np.floating, np.integer))
                else x
            ),
        )
    logger.info(f"\nSummary JSON: {summary_path}")

    # Write Markdown results
    md = generate_results_md(
        all_results=all_results,
        classification_mode=args.classification_mode,
        timestamp=timestamp,
        model_label=MODEL_LABEL,
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=[MODEL_NAME],
        has_abstention=False,
    )
    md_path = base_dir / f"RESULTS_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info(f"Results Markdown: {md_path}")

    # ------------------------------------------------------------------ #
    # Per-pair results (binary / multi-binary only)                        #
    # ------------------------------------------------------------------ #
    save_per_pair_results(
        base_dir=base_dir,
        all_results=all_results,
        classification_mode=args.classification_mode,
        timestamp=timestamp,
        model_label="Model 3",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=[MODEL_NAME],
        has_abstention=False,
        summary_json_extra={
            "dataset_name": args.dataset_name,
            "gene_locus": args.gene_locus,
            "aggregation_strategy": (
                agg_strategy.name if agg_strategy is not None else "auto"
            ),
            "entropy_threshold_fraction": (
                args.entropy_threshold if args.entropy_threshold is not None else (
                    _DEFAULT_ENTROPY_THRESHOLD
                    if (agg_strategy == AggregationStrategy.entropy_cutoff
                        or (agg_strategy is None and args.gene_locus == "TCR"))
                    else None
                )
            ),
        },
    )

    # ------------------------------------------------------------------ #
    # Print aggregated summary                                             #
    # ------------------------------------------------------------------ #
    logger.info("\n--- Aggregated Results ---")
    for pair_key, pair_data in all_results.items():
        for mn, agg in pair_data["aggregated_by_model"].items():
            logger.info(f"  {pair_key} / {mn}:")
            acc_global = agg.get("accuracy_global")
            acc_str = f"{acc_global:.4f}" if acc_global is not None else "N/A"
            if args.classification_mode == "multiclass":
                auroc_agg = agg.get("auroc_ovo_weighted", {})
                auroc_mean = auroc_agg.get("mean")
                auroc_str = f"{auroc_mean:.4f}" if auroc_mean is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_OvO={auroc_str}"
                )
            else:
                auroc_p = agg.get("auroc_pooled")
                auprc_p = agg.get("auprc_pooled")
                auroc_str = f"{auroc_p:.4f}" if auroc_p is not None else "N/A"
                auprc_str = f"{auprc_p:.4f}" if auprc_p is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_pooled={auroc_str} "
                    f"AUPRC_pooled={auprc_str}"
                )

    total_elapsed = time.monotonic() - t_total_start
    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed: {_fmt_elapsed(total_elapsed)}")
    logger.info("=" * 60)

    # Clean up file handler to flush and release the log file
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
