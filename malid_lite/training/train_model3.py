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
(default: cache/<dataset-name>/embeddings/). If embeddings are not found,
they are auto-computed for all participants before training begins.

Row alignment between fold data and pre-computed embeddings is ensured at load
time: rows are checked positionally using the downsampling unique key
(specimen_label, igh_or_tcrb_clone_id, isotype_supergroup, amplification_label
if present). If the order differs,
embeddings are reordered to match (with a warning). A biological sanity check
(cdr3_aa, v_gene, j_gene) runs after alignment.
See load_precomputed_embeddings() and CACHING_ARCHITECTURE.md.

If embeddings are not found and --no-cache-embeddings is set, they are computed
inline per-subset without saving to disk. This is slower for multi-fold runs
(~3 hours per 10M sequences on M4 Max MPS) because each subset is computed
independently rather than per-participant.

Classification modes
--------------------
multiclass
    Single N-class Stage 2 model over all disease classes.

binary
    One binary model for a single disease-vs-reference pair.
    Requires --reference-class. For 2-class datasets, the non-reference
    disease is auto-detected. For N-class datasets, use --diseases <disease>
    to pick one.

multi-binary
    One independent binary model per disease vs. the reference class.

Training contexts (--training-context)
--------------------------------------
A "training context" selects HOW the data is split and WHERE artifacts are written.
There are two families:

Cross-validation (CV) — train and test on the SAME dataset via K folds:
    cv_single_model (default)
        Standard K-fold CV. For each fold, the model is trained on the other folds
        and evaluated on the held-out fold. (Fold 0 as the test set means folds 1+2
        are the training data, etc.) Produces per-fold metrics.
    cv_ensemble
        Same as cv_single_model but the training folds reserve a validation third
        for the ensemble metamodel (used when this model is a base model of the
        ensemble).

Train-all — train on ONE whole dataset (no held-out test), to evaluate LATER on a
SEPARATE dataset (cross-dataset / external evaluation):
    train_all
        Train a single model on the ENTIRE dataset. There is no test set and no
        metrics — just the trained, reusable model artifacts.
    train_all_ensemble
        Like train_all, but holds out ~1/3 of participants as a validation set for
        the ensemble metamodel; the model itself trains on the remaining ~2/3.

In BOTH families Model 3 still uses train_smaller1 (ts1) and train_smaller2 (ts2)
SEPARATELY: Stage 1 is fit on ts1 and Stage 2 on Stage-1 predictions over the
disjoint ts2 (this keeps the Stage-2 features out-of-sample; see the model
docstring). For train-all, ts1+ts2 = all participants (train_all) or the 2/3 that
excludes the validation third (train_all_ensemble).

Train-all runs use train_full_dataset() (CV runs use train_all_folds()); the CLI
dispatches automatically on --training-context. Embeddings are shared with the CV
path unchanged (they are per-participant and fold/context-independent).

Output directory structure
---------------------------
multiclass:   trained_models/<dataset_name>/model3/multiclass/<gene_locus>/
binary:       trained_models/<dataset_name>/model3/binary/<gene_locus>/<disease>_vs_<reference>/
multi-binary: trained_models/<dataset_name>/model3/binary/<gene_locus>/<disease1>_vs_<reference>/
                                                              <disease2>_vs_<reference>/
                                                              ...

With --output-suffix <suffix>, the mode directory gets "__<suffix>" appended:
    trained_models/<dataset_name>/model3/multiclass__<suffix>/<gene_locus>/

With --output-dir <path>, the canonical path is replaced entirely:
    <path>/   (pair subdirs created within for binary/multi-binary)

Train-all contexts write to a parallel tree (no CV fold dimension):
    train_all:          trained_models/<dataset_name>/train_all_single_model/model3/<mode>/<gene_locus>/
    train_all_ensemble: trained_models/<dataset_name>/train_all_ensemble/base_models/<gene_locus>/model3/<mode>/

Per-fold artifacts (CV contexts):
    fold_<id>_stage1.pkl       : Stage 1 group models dict + _meta
    fold_<id>_stage2.pkl       : Stage 2 rollup model + _meta
    fold_<id>_results.json     : Evaluation metrics
    fold_<id>_predictions.pkl  : Raw predictions for aggregation + CSV (resume support)
    <mode>_predictions.csv     : All-fold predictions (appended across folds)
    summary_<timestamp>.json   : Aggregated metrics summary
    RESULTS_<timestamp>.md     : Human-readable results table

Train-all artifacts (NO fold prefix — there is no fold, no test set, no metrics):
    stage1.pkl                 : Stage 1 group models dict + _meta
    stage2.pkl                 : Stage 2 rollup model + _meta
    entropy_survival_stats.csv : Per-(specimen, group) entropy-filter survival (diagnostic)
    tuning_cv_results.csv      : Auto-tuning inner-CV results (only with auto_tuned)
    meta.json                  : Resume sentinel + provenance + training_info (written LAST)
    summary_<timestamp>.json   : No-metrics summary (training_only=True) with the config
                                 keys the ensemble / external eval read to reload the model
    RESULTS_<timestamp>.md     : Human-readable training summary (no metrics)

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
entropy_max_fraction, entropy_bottom_percentile, n_estimators_stage2,
reweigh_by_subset_frequencies, tuning_*)
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
        --aggregation-strategy entropy_cutoff --entropy-max-fraction 0.50 \\
        --resume-from-stage2

Resume from evaluation (--resume-from-evaluation)
--------------------------------------------------
Loads Stage 1 and Stage 2 from saved artifacts and re-runs evaluation only.
Automatically removes existing results and prediction artifacts so they are
regenerated. Requires both Stage 1 and Stage 2 artifacts to exist.
CV contexts only — train-all has no evaluation stage, so this flag errors when
combined with a train-all --training-context.

Example:

    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --resume-from-evaluation

Resume for train-all contexts
-----------------------------
Train-all uses a single meta.json sentinel (written LAST):
  --resume               : if the run is complete (all expected artifacts present +
                           non-empty) and its params match, skip and reload; otherwise
                           delete partial artifacts and retrain both stages.
  --resume-from-stage2   : keep a valid stage1.pkl (validating its Stage-1 params),
                           delete Stage-2 artifacts + meta.json, and retrain Stage 2
                           only — the fast way to iterate on Stage-2 / aggregation
                           knobs without repaying the expensive Stage 1.
  --resume-from-evaluation : NOT valid for train-all (no evaluation stage) — errors.
Note: --fold-ids and --stage1-dir are also CV-only and error under a train-all context.

Usage examples
--------------
    # Multiclass (default, TCR) — requires pre-computed embeddings
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv

    # Binary (2-class data, auto-detects disease)
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --classification-mode binary --reference-class Healthy

    # Binary (N-class data, pick one disease)
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --classification-mode binary --reference-class Healthy --diseases Covid19

    # Multi-binary: one COVID vs Healthy, one HIV vs Healthy, etc.
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy

    # Skip embedding caching (compute inline per-subset, don't save):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --no-cache-embeddings

    # Custom embedding directory:
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv \\
        --embedding-dir /path/to/precomputed/embeddings/

    # Custom aggregation strategy (default: entropy_percentile_cutoff, 0.01):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --aggregation-strategy mean

    # Paper-best strategy (TCR=entropy_cutoff 0.80, BCR=mean):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --aggregation-strategy paper_best

    # Auto-tuned strategy (inner CV grid search):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --aggregation-strategy auto_tuned

    # Train-all: train ONE model on the whole dataset (no CV, no test set), to be
    # evaluated later on a SEPARATE dataset. Writes stage1.pkl / stage2.pkl / meta.json
    # (no fold prefix) under .../train_all_single_model/model3/<mode>/<gene_locus>/:
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --training-context train_all

    # Train-all as an ensemble base model (holds out a validation third for the
    # metamodel; the model trains on the remaining 2/3):
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --training-context train_all_ensemble

    # Train-all: reuse the (expensive) Stage 1 and re-tune only Stage 2:
    python malid_lite/training/train_model3.py \\
        --metadata-path /path/to/metadata.tsv --training-context train_all \\
        --aggregation-strategy mean --resume-from-stage2

    # First run with custom clone_id (only needed once, when building cache):
    python malid_lite/training/train_model3.py \\
        --data-dir /path/to/data --metadata-path /path/to/metadata.tsv \\
        --force-clone-id --clone-id-use-aa

Clone ID parameters
-------------------
All training scripts accept clone_id flags (--force-clone-id, --clone-id-use-aa,
--clone-id-identity-threshold, --clone-id-linkage-method). These only need to be
specified when building the cache for the first time. On subsequent runs, omitting
them is fine -- the cached values are accepted as-is. If you explicitly specify a
value that conflicts with the cache, the run fails immediately with a clear error.
See PIPELINE_GUIDE.md > Clone ID Computation for details.
"""

import argparse
import json
import logging
import pickle
import re
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
    matthews_corrcoef,
    roc_auc_score,
)

# Add project root to path (must come before malid_lite imports)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Custom multiclass metrics that work with unnormalized probabilities
# (Model 3's BinaryOvR outputs independent per-class probabilities that
# don't sum to 1 — sklearn's multiclass roc_auc_score rejects these).
# These match what the original Mal-ID paper used for evaluation.
from malid_lite.utils import multiclass_metrics

from malid_lite.dataloader import (
    MalIDPublishedDataLoader,
    add_clone_id_args,
    get_clone_id_kwargs,
    normalize_identifier_columns,
)
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
    _DEFAULT_TUNING_MAX_FRACTIONS,
    _DEFAULT_TUNING_PERCENTILES,
    _DEFAULT_TUNING_STRATEGIES,
    compute_esm2_embeddings,
    make_bcr_model,
    make_tcr_model,
)
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    FOLD_COL,
    TRAIN_ALL_TRAINING_CONTEXTS,
    VALID_TRAINING_CONTEXTS,
    aggregate_fold_results,
    check_train_all_split,
    delete_stale_summaries,
    filter_to_binary_pair,
    generate_results_md,
    get_dataset_disease_classes,
    get_metadata_class_counts,
    get_model_classes,
    get_model_output_dir,
    make_pair_name,
    run_training_orchestration,
    save_per_pair_results,
    write_train_all_outputs,
    train_all_artifacts_complete,
    validate_mode_and_classes,
    validate_train_all_meta,
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

# Paper-best entropy max fraction for TCR (0.80 = keep below 0.8 * max possible
# entropy). Used for display/metadata when the resolved strategy is entropy_cutoff
# and no explicit --entropy-max-fraction was given.
_DEFAULT_ENTROPY_MAX_FRACTION = 0.80

# Default bottom percentile for entropy_percentile_cutoff strategy (0.01%).
_DEFAULT_ENTROPY_BOTTOM_PERCENTILE = 0.01

# Parameters that only affect Stage 2 (aggregation + Stage 2 training).
# Excluded from Stage 1 artifact validation so that Stage 1 can be reused
# when only Stage-2-only params change (e.g., --resume-from-stage2).
_STAGE2_ONLY_PARAMS = frozenset({
    "aggregation_strategy",
    "entropy_max_fraction",
    "entropy_bottom_percentile",
    "n_estimators_stage2",
    "reweigh_by_subset_frequencies",
    "tuning_enabled",
    "tuning_cv_splits",
    "tuning_strategies",
    "tuning_entropy_max_fractions",
    "tuning_entropy_percentiles",
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
    training_context: Optional[str] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
) -> dict:
    """Extract model and run parameters for artifact metadata.

    These are validated on resume to ensure loaded artifacts were trained
    with the same settings as the current run. Includes both model-level
    hyperparameters and run-level settings (classification mode, disease
    subset, dataset name, training context, disease filter) that affect
    training outcomes.
    """
    params = {
        "locus": model.locus,
        "aggregation_strategy": model.aggregation_strategy.name,
        "entropy_max_fraction": model.entropy_max_fraction,
        "entropy_bottom_percentile": model.entropy_bottom_percentile,
        "exclude_rare_v_genes": model.exclude_rare_v_genes,
        "min_sequences_per_group": model.min_sequences_per_group,
        "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
        "n_estimators_stage1": model.n_estimators_stage1,
        "n_estimators_stage2": model.n_estimators_stage2,
        "reference_class": model.reference_class,
        "tuning_enabled": model.tuning_enabled,
        "tuning_cv_splits": model.tuning_cv_splits,
        "tuning_strategies": model.tuning_strategies,
        "tuning_entropy_max_fractions": model.tuning_entropy_max_fractions,
        "tuning_entropy_percentiles": model.tuning_entropy_percentiles,
        "classification_mode": classification_mode,
        "diseases": sorted(diseases) if diseases else None,
        "dataset_name": dataset_name,
        "training_context": training_context,
        "disease_filter": disease_filter,
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
                 dataset_name, training_context. Passed through to _build_model_params.
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
        # aggregation_strategy and entropy params are Stage-2-only
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
                 dataset_name, training_context. Passed through to _build_model_params.
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
    # Build tuning-specific fields (present only when auto_tuned was used)
    tuning_data = {}
    if model.tuning_enabled_:
        tuning_data["tuning_enabled"] = True
        tuning_data["tuning_results"] = model.tuning_results_
        tuning_data["tuning_cv_splits"] = model.tuning_cv_splits
        # The winning strategy (after tuning, model.aggregation_strategy holds
        # the winner, not "auto_tuned")
        tuning_data["tuning_best_strategy"] = model.aggregation_strategy.name
        # threshold_param: the human-readable parameter (max_fraction or percentile)
        if model.tuning_results_:
            tuning_data["tuning_best_threshold_param"] = model.tuning_results_[0].get(
                "threshold_param"
            )

    with open(path, "wb") as f:
        pickle.dump({
            "stage2_clf": model.stage2_clf_,
            "stage2_scaler": model.stage2_scaler_,
            "preagg_scaler": model.preagg_scaler_,
            "feature_columns": model.feature_columns_,
            "classes": model.classes_,
            "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
            "entropy_percentile_threshold": model.entropy_percentile_threshold_,
            "aggregation_strategy": model.aggregation_strategy.name,
            **tuning_data,
            "_meta": meta,
        }, f)
    logger.info(f"  Saved Stage 2: {path}")


def _write_tuning_cv_results(model: SequenceLevelClassifier, path: Path) -> None:
    """Write the auto-tuning inner-CV results (all candidates ranked by mean MCC).

    Shared by the CV fold loop and the train-all path so both emit an identical
    ``*_tuning_cv_results.csv``. One row per candidate strategy/threshold with its
    mean/std MCC and per-inner-fold MCCs; ``fallback=True`` marks the safety-net
    default used when every candidate scored <= 0. Only called when
    ``model.tuning_enabled_`` and ``model.tuning_results_`` are set.
    """
    tuning_rows = []
    for rank, r in enumerate(model.tuning_results_, 1):
        row = {
            "rank": rank,
            "strategy": r["strategy_name"],
            "threshold_param": r["threshold_param"],
            "threshold_nats": r["threshold_nats"],
            "mean_mcc": r["mean_mcc"],
            "std_mcc": r["std_mcc"],
        }
        for fi, fs in enumerate(r.get("fold_scores", [])):
            row[f"fold_{fi}_mcc"] = fs
        if r.get("fallback"):
            row["fallback"] = True
        tuning_rows.append(row)
    pd.DataFrame(tuning_rows).to_csv(path, index=False)


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
    stage1_dir: Optional[Path] = None,
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
    stage1_dir           : If provided, look for Stage 1 artifacts here instead
                           of output_dir (for sharing Stage 1 across experiments).
    """
    errors: List[str] = []
    s1_dir = stage1_dir if stage1_dir is not None else output_dir
    for fold_id in fold_ids:
        stage1_path = s1_dir / f"fold_{fold_id}_stage1.pkl"
        stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
        if resume_from_stage2:
            if not stage1_path.exists():
                errors.append(f"  Fold {fold_id}: missing {stage1_path.name} in {s1_dir}")
        elif resume_from_evaluation:
            missing = []
            if not stage1_path.exists():
                missing.append(f"{stage1_path.name} (in {s1_dir})")
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
                      dataset_name, training_context. Passed through to _build_model_params.
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
                 dataset_name, training_context. Passed through to _build_model_params.
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

    # When tuning is enabled, the model's aggregation_strategy and entropy
    # params are set by tuning during fit_stage2 — before fit runs, they
    # still hold the factory defaults. The artifact stores the per-fold
    # tuning winner, which legitimately differs. Exclude these from the
    # meta comparison; load_stage2_artifacts does its own tuning-aware check.
    _TUNING_OUTCOME_PARAMS = frozenset({
        "aggregation_strategy", "entropy_max_fraction", "entropy_bottom_percentile",
    })
    current_params = _build_model_params(model, **rp)
    if model.tuning_enabled:
        current_params = {
            k: v for k, v in current_params.items()
            if k not in _TUNING_OUTCOME_PARAMS
        }

    _validate_artifact_meta(
        meta, "Stage 2", fold_id,
        expected_classes=expected_classes,
        current_model_params=current_params,
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
    fold_id: Optional[int],
    fold_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fold (or whole-dataset) sequences and join with disease metadata.

    Parameters
    ----------
    fold_id : CV fold ID for ``fold_label`` "train"/"test", or ``None`` when
        ``fold_label == "all"`` (train-all: the whole dataset, no CV fold).
    fold_label : "train", "test", or "all". "all" loads the entire dataset via
        ``loader.get_all_data()`` (the train-all path); "train"/"test" load the
        CV fold via ``loader.get_fold_data(fold_id, fold_label)``.

    Returns
    -------
    (sequences_df, metadata_df)
    sequences_df has all sequence columns plus disease (from metadata join).
    specimen_label column is the specimen identifier.
    """
    if fold_label == "all":
        # Train-all: the entire dataset, no CV fold (fold_id is ignored / None).
        sequences_df, metadata_df = loader.get_all_data()
    else:
        sequences_df, metadata_df = loader.get_fold_data(fold_id, fold_label)

    if sequences_df.empty:
        raise ValueError(
            f"No sequences found for "
            f"{'the whole dataset (fold_label=all)' if fold_label == 'all' else f'fold {fold_id} {fold_label}'}"
        )

    # Add disease column to sequences_df (keyed by specimen_label)
    disease_map = metadata_df.set_index("specimen_label")["disease"]
    sequences_df = sequences_df.copy()
    sequences_df[DISEASE_COL] = sequences_df[SPECIMEN_COL].map(disease_map)

    n_before = len(sequences_df)
    sequences_df = sequences_df.dropna(subset=[DISEASE_COL])
    if len(sequences_df) < n_before:
        logger.warning(
            f"  Dropped {n_before - len(sequences_df)} rows with unknown disease"
        )

    return sequences_df, metadata_df


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def _get_downsampling_key_cols(participant_df: pd.DataFrame) -> List[str]:
    """Return the downsampling unique key columns present in participant_df.

    This is the groupby key used in preprocess_downsample(), guaranteed unique
    per row after DOWNSAMPLED preprocessing.
    """
    key_cols = [SPECIMEN_COL, "igh_or_tcrb_clone_id", ISOTYPE_COL]
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
    Handles type mismatches by casting both sides to str when dtypes differ
    (e.g., int64 clone_ids in per-participant parquets vs str clone_ids in
    concatenated fold caches).
    """
    for col in cols:
        fold_vals = fold_subset[col].values
        precomputed_vals = participant_df[col].values
        # Cast to common str type when dtypes differ (e.g., int64 vs str
        # clone_ids from merging caches with different clone_id formats)
        if fold_vals.dtype != precomputed_vals.dtype:
            fold_vals = pd.Series(fold_vals).astype(str).values
            precomputed_vals = pd.Series(precomputed_vals).astype(str).values
        # np.array_equal treats NaN != NaN; use pandas Series.equals which treats NaN == NaN
        if not pd.Series(fold_vals).equals(pd.Series(precomputed_vals)):
            return False
    return True


def _make_hashable_key(values: tuple) -> tuple:
    """Convert a tuple of values to a hashable key for cross-source comparison.

    All values are cast to str so that keys from different sources match
    even when column dtypes differ (e.g., int64 clone_id ``1`` in a
    per-participant parquet vs str ``'1'`` in a concatenated fold cache).
    NaN values are replaced with a sentinel that cannot collide with real data.
    """
    return tuple("__NAN__" if pd.isna(v) else str(v) for v in values)


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
    precomputed_keys = [
        _make_hashable_key(t)
        for t in zip(*(participant_df[c].values for c in key_cols))
    ]
    key_to_idx = {k: i for i, k in enumerate(precomputed_keys)}

    if len(key_to_idx) != len(participant_df):
        raise ValueError(
            f"Downsampling key is not unique for participant {participant}: "
            f"{len(participant_df)} rows but {len(key_to_idx)} unique keys. "
            f"This indicates a preprocessing issue."
        )

    # Look up each fold row (vectorized extraction, loop only for dict lookup)
    fold_keys = [
        _make_hashable_key(t)
        for t in zip(*(fold_subset[c].values for c in key_cols))
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


def _load_participant_embedding_files(
    participant: str,
    embedding_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load and validate one participant's embedding .npy and .parquet files.

    Checks file existence, corruption (load errors), dtype, shape, NaN/Inf,
    row-count match, and required parquet columns. Raises clear errors with
    actionable messages on any failure.

    Returns
    -------
    (embeddings_float32, parquet_df) — embeddings already cast to float32.
    """
    emb_path = embedding_dir / f"{participant}_embeddings.npy"
    parquet_path = embedding_dir / f"{participant}_downsampled.parquet"

    # --- File existence ---
    missing = []
    if not emb_path.exists():
        missing.append(str(emb_path))
    if not parquet_path.exists():
        missing.append(str(parquet_path))
    if missing:
        raise FileNotFoundError(
            f"Pre-computed embeddings not found for participant '{participant}'. "
            f"Missing files:\n  " + "\n  ".join(missing) + "\n"
            f"Run the embedding script first:\n"
            f"  python -m malid_lite.training.compute_model3_embeddings "
            f"--metadata-path <path>\n"
            f"Or delete the participant's files from {embedding_dir} to have "
            f"them re-computed by the training script."
        )

    # --- Load .npy with corruption guard ---
    try:
        participant_emb = np.load(str(emb_path))
    except Exception as e:
        raise ValueError(
            f"Corrupt embedding file for participant '{participant}': "
            f"{emb_path}\nError: {e}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        ) from e

    # --- Validate .npy: shape, dtype, NaN/Inf ---
    if participant_emb.ndim != 2 or participant_emb.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding shape mismatch for participant '{participant}': "
            f"got {participant_emb.shape}, expected (N, {EMBEDDING_DIM}). "
            f"File: {emb_path}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        )
    if participant_emb.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError(
            f"Unexpected embedding dtype for participant '{participant}': "
            f"{participant_emb.dtype} (expected float16 or float32). "
            f"File: {emb_path}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        )
    participant_emb = participant_emb.astype(np.float32)  # float16 -> float32
    if not np.all(np.isfinite(participant_emb)):
        n_nan = int(np.isnan(participant_emb).any(axis=1).sum())
        n_inf = int(np.isinf(participant_emb).any(axis=1).sum())
        raise ValueError(
            f"Non-finite values in embeddings for participant '{participant}': "
            f"{n_nan} rows with NaN, {n_inf} rows with Inf. "
            f"File: {emb_path}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        )

    # --- Load .parquet with corruption guard ---
    try:
        participant_df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(
            f"Corrupt parquet file for participant '{participant}': "
            f"{parquet_path}\nError: {e}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        ) from e

    # --- Backward compat: old parquets use repertoire_id ---
    if "repertoire_id" in participant_df.columns and SPECIMEN_COL not in participant_df.columns:
        participant_df = participant_df.rename(columns={"repertoire_id": SPECIMEN_COL})

    # Normalize int64 identifiers to str (numeric labels from older caches)
    participant_df = normalize_identifier_columns(participant_df)

    # --- Validate parquet columns ---
    required_cols = {SPECIMEN_COL, "igh_or_tcrb_clone_id", ISOTYPE_COL}
    missing_cols = required_cols - set(participant_df.columns)
    if missing_cols:
        raise ValueError(
            f"Embedding parquet for participant '{participant}' is missing "
            f"columns: {sorted(missing_cols)}. "
            f"Available columns: {sorted(participant_df.columns)}. "
            f"File: {parquet_path}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the file manually to have it re-computed by the training script."
        )

    # --- Row count match between .npy and .parquet ---
    if len(participant_emb) != len(participant_df):
        raise ValueError(
            f"Embedding/parquet row mismatch for participant '{participant}': "
            f"{len(participant_emb)} embeddings vs {len(participant_df)} parquet rows. "
            f"Files:\n  {emb_path}\n  {parquet_path}\n"
            f"Re-run compute_model3_embeddings.py to regenerate, or delete "
            f"the files manually to have them re-computed by the training script."
        )

    return participant_emb, participant_df


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

    Each participant's files are validated on load: corruption is detected
    (truncated .npy, unreadable .parquet), shape/dtype/NaN are checked, and
    required parquet columns are verified. Any issue raises a clear error.

    Row alignment uses the downsampling unique key (specimen_label,
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
        # Load and validate pre-computed files (corruption, shape, dtype, NaN, columns)
        participant_emb, participant_df = _load_participant_embedding_files(
            participant, embedding_dir,
        )

        # Find which rows in sequences_df belong to this participant
        mask = sequences_df[PARTICIPANT_COL] == participant
        n_fold_rows = mask.sum()

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
            # training split vs all-fold embeddings). Reuse the key-based reorder
            # logic — which enforces downsampling-key UNIQUENESS (duplicate keys
            # would otherwise silently collapse, last-wins → wrong-participant
            # embeddings) and raises on missing keys — then run the SAME biological
            # sanity check as the exact-match path. (Previously this branch skipped
            # both guards; see the audit note.)
            key_cols = _get_downsampling_key_cols(participant_df)
            reorder_indices = _compute_reorder_indices(
                fold_subset, participant_df, key_cols, participant,
            )
            aligned_emb = participant_emb[reorder_indices]

            # Biological sanity check (cdr3_aa/v_gene/j_gene) on the matched rows —
            # catches a key match that nonetheless pairs biologically different rows.
            sanity_cols = [CDR3_COL, V_GENE_COL, J_GENE_COL]
            reordered_df = participant_df.iloc[reorder_indices].reset_index(drop=True)
            fold_reset = fold_subset.reset_index(drop=True)
            if not _check_positional_alignment(fold_reset, reordered_df, sanity_cols):
                raise ValueError(
                    f"Biological sanity check failed for participant {participant} "
                    f"(subset alignment): cdr3_aa/v_gene/j_gene differ despite matching "
                    f"on the downsampling key. This indicates data corruption. "
                    f"Re-run compute_model3_embeddings.py to regenerate."
                )

            embeddings[row_indices] = aligned_emb

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
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

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

def _stage_artifact_paths(
    output_dir: Path, fold_id: Optional[int]
) -> Tuple[Path, Path]:
    """Return the ``(stage1_path, stage2_path)`` pickle paths, fold-optional.

    Fold-optional naming shared by the CV and train-all paths (and by the ensemble's
    ``predict_model3`` when loading Model 3 base-model artifacts):

    - ``fold_id`` is an int (CV) → ``fold_<id>_stage1.pkl`` / ``fold_<id>_stage2.pkl``.
    - ``fold_id`` is ``None`` (train-all: whole dataset, no CV fold) →
      ``stage1.pkl`` / ``stage2.pkl`` (no fold prefix), matching the unprefixed
      artifact convention Models 1/2 use for train-all.
    """
    prefix = f"fold_{fold_id}_" if fold_id is not None else ""
    return (
        output_dir / f"{prefix}stage1.pkl",
        output_dir / f"{prefix}stage2.pkl",
    )


def _get_fold_artifact_paths(output_dir: Path, fold_id: int) -> List[Path]:
    """Return the four artifact paths that constitute a complete CV fold.

    CV-only (int ``fold_id``): Stage 1 + Stage 2 pickles plus the per-fold
    ``results.json`` and ``predictions.pkl``. Train-all has no test set (no
    results/predictions) and uses a ``meta.json`` sentinel instead — see
    ``_stage_artifact_paths`` and ``_run_train_all``.
    """
    stage1_path, stage2_path = _stage_artifact_paths(output_dir, fold_id)
    return [
        stage1_path,
        stage2_path,
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


def _build_model(
    *,
    locus: str,
    aggregation_strategy: Optional[AggregationStrategy],
    entropy_max_fraction: Optional[float],
    entropy_bottom_percentile: Optional[float],
    n_estimators_stage1: int,
    n_estimators_stage2: int,
    n_jobs: int,
    verbose: int,
    reference_class: Optional[str],
    tuning_enabled: bool,
    tuning_cv_splits: int,
    tuning_strategies: Optional[List[str]],
    tuning_entropy_max_fractions: Optional[List[float]],
    tuning_entropy_percentiles: Optional[List[float]],
) -> SequenceLevelClassifier:
    """Build a fresh (unfitted) Model 3 classifier from resolved training params.

    Shared by both training paths so they construct the model identically:
    the CV fold loop (``_run_fold_loop``, one model per fold) and the whole-dataset
    train-all path (``_run_train_all``). Behavior:

    - If ``aggregation_strategy`` is an explicit strategy (the user passed a fixed
      ``--aggregation-strategy`` other than ``paper_best``/``auto_tuned``), build the
      classifier directly with that strategy and the applicable entropy threshold.
      Tuning is never active in this branch.
    - If ``aggregation_strategy`` is ``None`` (``paper_best`` or ``auto_tuned``), use the
      per-locus paper-best factory (TCR: ``make_tcr_model``, BCR: ``make_bcr_model``),
      passing the tuning grid when ``tuning_enabled`` is set (``auto_tuned``).

    Parameters
    ----------
    locus : "TCR" or "BCR".
    aggregation_strategy : Explicit sequence->specimen aggregation strategy, or None to
        defer to the per-locus paper-best factory (and to tuning if enabled).
    entropy_max_fraction / entropy_bottom_percentile : Entropy-filter thresholds; only
        the one matching the chosen strategy is used. None -> factory default.
    n_estimators_stage1 / n_estimators_stage2 : Tree counts for the Stage 1 (BCR RF) and
        Stage 2 (RF) classifiers.
    n_jobs, verbose : Passed through to the classifier.
    reference_class : Negative/reference class for binary Stage-2 feature subsetting; None
        for multiclass.
    tuning_enabled : Enable auto-tuning of the aggregation strategy (inner CV on ts2).
    tuning_cv_splits / tuning_strategies / tuning_entropy_max_fractions /
        tuning_entropy_percentiles : Tuning search configuration (only used when
        ``tuning_enabled`` and ``aggregation_strategy is None``).
    """
    model_kwargs = dict(
        n_estimators_stage1=n_estimators_stage1,
        n_estimators_stage2=n_estimators_stage2,
        n_jobs=n_jobs,
        reference_class=reference_class,
        verbose=verbose,
    )

    tuning_kwargs: Dict = {}
    if tuning_enabled:
        tuning_kwargs["tuning_enabled"] = True
        tuning_kwargs["tuning_cv_splits"] = tuning_cv_splits
        if tuning_strategies is not None:
            tuning_kwargs["tuning_strategies"] = tuning_strategies
        if tuning_entropy_max_fractions is not None:
            tuning_kwargs["tuning_entropy_max_fractions"] = tuning_entropy_max_fractions
        if tuning_entropy_percentiles is not None:
            tuning_kwargs["tuning_entropy_percentiles"] = tuning_entropy_percentiles

    if aggregation_strategy is not None:
        # User specified an explicit strategy — tuning is never enabled here
        # (auto_tuned sets aggregation_strategy=None, so this branch is only
        # reached for fixed strategies where tuning_kwargs is empty).
        extra = {}
        if entropy_max_fraction is not None:
            extra["entropy_max_fraction"] = entropy_max_fraction
        if entropy_bottom_percentile is not None:
            extra["entropy_bottom_percentile"] = entropy_bottom_percentile
        return SequenceLevelClassifier(
            locus=locus,
            aggregation_strategy=aggregation_strategy,
            exclude_rare_v_genes=True,
            reweigh_by_subset_frequencies=True,
            **extra,
            **model_kwargs,
        )
    # aggregation_strategy is None (--aggregation-strategy paper_best or auto_tuned):
    # use paper-best factory per locus (TCR=entropy_cutoff 0.80, BCR=mean)
    elif locus == "TCR":
        return make_tcr_model(**tuning_kwargs, **model_kwargs)
    else:
        return make_bcr_model(**tuning_kwargs, **model_kwargs)


def _resolve_embeddings_and_compute(
    *,
    embedding_dir: Optional[Path],
    cache_dir: Optional[Path],
    data_dir: Optional[Path],
    metadata_path: Path,
    cache_embeddings: bool,
    device: Optional[str],
    embedding_batch_size: int,
    gene_locus: str,
    clone_id_kwargs: Optional[Dict],
    verbose: int,
) -> Tuple[Optional[Path], bool, bool]:
    """Resolve the embedding directory and ensure embeddings are available.

    Shared by the CV entry (``train_all_folds``) and the train-all entry
    (``train_full_dataset``). ESM-2 embeddings are per-participant and completely
    independent of CV fold / training context, so the exact same resolution +
    availability logic applies to both — factoring it here avoids duplicating ~100
    lines and keeps the two entry points in lockstep.

    Returns
    -------
    (embedding_dir, use_inline_embeddings, embedding_dir_explicit)
        embedding_dir : the resolved directory (auto-derived from ``cache_dir`` when
            not passed explicitly).
        use_inline_embeddings : True only when ``cache_embeddings=False`` and no cached
            embeddings exist, so embeddings must be computed on the fly per subset.
        embedding_dir_explicit : True when the caller passed ``embedding_dir``
            explicitly. Controls the post-loader completeness check
            (``_verify_embeddings_complete``), which only runs for an explicit dir.

    Behavior (unchanged from the original inline block):
    - ``cache_embeddings=True`` + explicit dir → trust it, error if empty.
    - ``cache_embeddings=True`` + auto-resolved dir → ``compute_all_embeddings`` (has
      built-in resume: already-computed participants are skipped).
    - ``cache_embeddings=False`` → use cached embeddings if present, else inline
      computation (or an error for an explicit-but-empty dir).
    """
    if embedding_dir is None and cache_dir is None:
        raise ValueError(
            "No embedding source: embedding_dir is None and cache_dir is None "
            "(so the default embedding path cannot be resolved). "
            "Either provide embedding_dir or cache_dir."
        )

    # --- Auto-resolve embedding_dir from cache_dir when not specified ---
    embedding_dir_explicit = embedding_dir is not None
    if embedding_dir is None and cache_dir is not None:
        embedding_dir = cache_dir / "embeddings"
        logger.info(f"  Resolved embedding_dir from cache: {embedding_dir}")

    # --- Ensure embeddings are available ---
    # When cache_embeddings=True: call compute_all_embeddings() which has built-in
    # resume logic — already-computed participants are skipped, only missing ones
    # are computed. This handles both "no embeddings at all" and "partial embeddings
    # (e.g., interrupted run)" cases correctly.
    # When cache_embeddings=False: use cached embeddings if ANY exist; only fall
    # back to inline computation if the embedding_dir is completely empty/missing.
    has_any_embeddings = (
        embedding_dir is not None
        and embedding_dir.exists()
        and any(embedding_dir.glob("*_embeddings.npy"))
    )
    use_inline_embeddings = False

    if cache_embeddings:
        if embedding_dir_explicit:
            # User explicitly provided --embedding-dir: trust it, don't auto-compute.
            # Full completeness check runs after loader construction (see below).
            if not has_any_embeddings:
                raise FileNotFoundError(
                    f"No pre-computed embeddings found in the specified "
                    f"embedding_dir: {embedding_dir}\n"
                    f"Either pre-compute embeddings with compute_model3_embeddings.py "
                    f"or remove --embedding-dir to use the default cache path "
                    f"(which supports auto-computation)."
                )
        elif cache_dir is not None:
            # embedding_dir auto-resolved from cache_dir: auto-compute missing ones.
            # compute_all_embeddings() has resume logic — already-done are skipped.
            from malid_lite.training.compute_model3_embeddings import (
                compute_all_embeddings,
            )
            if not has_any_embeddings:
                logger.info(
                    f"No pre-computed embeddings found in {embedding_dir}. "
                    "Auto-computing embeddings for all participants..."
                )
            else:
                logger.info(
                    f"Verifying all participants have embeddings in {embedding_dir} "
                    "(already-computed participants will be skipped)..."
                )
            compute_all_embeddings(
                metadata_path=metadata_path,
                cache_dir=cache_dir,
                data_dir=data_dir,
                device=device,
                batch_size=embedding_batch_size,
                verbose=verbose,
                gene_locus=gene_locus,
                clone_id_kwargs=clone_id_kwargs,
            )
            # Verify at least some embeddings exist after computation
            if not embedding_dir.exists() or not any(
                embedding_dir.glob("*_embeddings.npy")
            ):
                raise RuntimeError(
                    f"Embedding computation completed but no embedding files "
                    f"found in {embedding_dir}. Check the embedding log for errors."
                )
            logger.info(f"Embeddings verified/computed in {embedding_dir}")
        else:
            # No cache_dir and no auto-resolve possible — must have embeddings
            if not has_any_embeddings:
                raise FileNotFoundError(
                    f"No pre-computed embeddings found in {embedding_dir} and "
                    "cache_dir is None, so auto-computation is not possible.\n"
                    "Either pre-compute embeddings with compute_model3_embeddings.py "
                    "or provide --cache-dir."
                )
    else:
        # --no-cache-embeddings: use cached if available, else inline
        if has_any_embeddings:
            logger.info(
                f"Using existing pre-computed embeddings from {embedding_dir}. "
                "(--no-cache-embeddings is set but cached embeddings are available.)"
            )
        else:
            if embedding_dir_explicit:
                raise FileNotFoundError(
                    f"--embedding-dir was explicitly set to {embedding_dir} "
                    f"but it contains no embedding files (*_embeddings.npy), "
                    f"and --no-cache-embeddings prevents auto-computation.\n"
                    f"Either:\n"
                    f"  1. Pre-compute embeddings into that directory with "
                    f"compute_model3_embeddings.py "
                    f"--output-embedding-dir {embedding_dir}\n"
                    f"  2. Remove --embedding-dir to use the default cache path\n"
                    f"  3. Remove --no-cache-embeddings to allow auto-computation"
                )
            logger.info(
                "NOTE: --no-cache-embeddings is set and no pre-computed embeddings "
                "found. Embeddings will be computed inline for each subset "
                "(slower than pre-computing). Consider removing --no-cache-embeddings "
                "for multi-fold runs."
            )
            use_inline_embeddings = True

    return embedding_dir, use_inline_embeddings, embedding_dir_explicit


def _verify_embeddings_complete(
    *,
    loader: MalIDPublishedDataLoader,
    embedding_dir: Optional[Path],
    embedding_dir_explicit: bool,
    use_inline_embeddings: bool,
    metadata_path: Path,
    cache_dir: Optional[Path],
) -> None:
    """Verify every participant has complete embedding files (explicit dir only).

    Shared post-loader completeness check for the CV and train-all entry points.
    When the caller passed ``--embedding-dir`` explicitly (and embeddings aren't being
    computed inline), verify that every participant in ``loader.metadata`` has a
    complete set of embedding files BEFORE any training starts — so a partial or
    interrupted embedding run fails here with a remediation command instead of failing
    mid-training on the first missing participant. No-op when the dir was auto-resolved
    (that path auto-computes) or when using inline embeddings.
    """
    if not (embedding_dir_explicit and not use_inline_embeddings):
        return
    from malid_lite.training.compute_model3_embeddings import (
        validate_embedding_completeness,
    )
    all_participant_labels = sorted(
        loader.metadata[PARTICIPANT_COL].unique()
    )
    if not validate_embedding_completeness(
        all_participant_labels, embedding_dir, logger
    ):
        # Build remediation command with --output-embedding-dir when
        # the user's embedding_dir differs from the default location
        _default_emb_dir = cache_dir / "embeddings" if cache_dir else None
        _needs_output_flag = (embedding_dir != _default_emb_dir)
        _remediation = (
            f"  python -m malid_lite.training.compute_model3_embeddings "
            f"--metadata-path {metadata_path}"
            + (f" --cache-dir {cache_dir}" if cache_dir else "")
            + (f" --output-embedding-dir {embedding_dir}" if _needs_output_flag else "")
        )
        raise FileNotFoundError(
            f"Embedding completeness check failed for "
            f"--embedding-dir {embedding_dir}.\n"
            f"Some participants are missing embedding files "
            f"(see log above for details).\n"
            f"Complete them with:\n"
            f"{_remediation}\n"
            f"Or remove --embedding-dir to use the default cache path "
            f"(which supports auto-computation)."
        )


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
    use_inline_embeddings: bool = False,
    device: Optional[str] = None,
    embedding_batch_size: int = 64,
    aggregation_strategy: Optional[AggregationStrategy] = None,
    entropy_max_fraction: Optional[float] = None,
    entropy_bottom_percentile: Optional[float] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
    training_context: str = "cv_single_model",
    resume: bool = False,
    resume_from_stage2: bool = False,
    resume_from_evaluation: bool = False,
    stage1_source_dir: Optional[Path] = None,
    run_params: Optional[dict] = None,
    run_config_text: Optional[str] = None,
    timestamp: Optional[str] = None,
    tuning_enabled: bool = False,
    tuning_cv_splits: int = 3,
    tuning_strategies: Optional[List[str]] = None,
    tuning_entropy_max_fractions: Optional[List[float]] = None,
    tuning_entropy_percentiles: Optional[List[float]] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Run training + evaluation for all specified folds.

    Parameters
    ----------
    disease_filter : (disease, reference_class) for binary/multi-binary; None for multiclass.
    entropy_max_fraction : Fraction of max possible entropy to use as cutoff (0-1 scale).
        Only used when aggregation_strategy is entropy_cutoff. Passed through to
        SequenceLevelClassifier. None uses the factory default (0.80).
    entropy_bottom_percentile : Percentile of training entropy distribution (0-100).
        Only used when aggregation_strategy is entropy_percentile_cutoff. None
        uses the factory default (0.01).
    resume         : If True, skip folds whose artifacts already exist on disk
                     and reload their results for aggregation.
    resume_from_stage2 : If True, load Stage 1 from saved artifacts but retrain
                     Stage 2 from scratch. Automatically removes stale Stage 2,
                     results, and prediction artifacts. Use this when changing
                     Stage-2-only params (aggregation, entropy params,
                     n_estimators_stage2, reweigh_by_subset_frequencies).
                     Requires Stage 1 artifacts to exist.
    resume_from_evaluation : If True, load Stage 1 and Stage 2 from saved
                     artifacts and re-run evaluation only. Automatically removes
                     stale results and prediction artifacts. Requires both
                     Stage 1 and Stage 2 artifacts to exist.
    stage1_source_dir : If provided, read Stage 1 artifacts from this directory
                     instead of output_dir. Allows sharing one set of Stage 1
                     models across multiple Stage 2 experiments. Only valid
                     with resume_from_stage2=True. For binary/multi-binary,
                     this is the pair subdirectory (the orchestrator appends
                     the pair name before calling this function).
    run_params     : Dict with classification_mode, diseases, dataset_name,
                     training_context for artifact metadata validation on resume.
    run_config_text : Human-readable run config string built in main(). Written
                     to output_dir/run_config_<timestamp>.txt at the start of
                     the run (before any training), so it's available even if
                     the run crashes. None skips writing.
    timestamp      : Run timestamp string (YYYYMMDD_HHMMSS) for naming the
                     config file. None skips writing.
    tuning_enabled : If True, auto-tune aggregation strategy via inner CV.
    tuning_cv_splits : Number of inner CV folds for tuning.
    tuning_strategies : List of strategy names to search during tuning.
    tuning_entropy_max_fractions : Grid of max_fraction values for tuning.
    tuning_entropy_percentiles : Grid of percentile values for tuning.

    Returns
    -------
    (all_eval_results, aggregated_by_model)
    """
    all_eval_results: List[Dict] = []
    raw_preds_list: List[Optional[Dict]] = []
    predictions_rows: List[Dict] = []
    fold_timings: List[Dict[str, float]] = []

    # Reference class for the model (binary Stage-2 feature subsetting); None for multiclass.
    ref_class_for_model = disease_filter[1] if disease_filter else None

    def _make_model() -> SequenceLevelClassifier:
        """Build a fresh (unfitted) model for this fold.

        Thin wrapper over the module-level ``_build_model`` (shared with the
        train-all path) that forwards this loop's resolved training params.
        """
        return _build_model(
            locus=locus,
            aggregation_strategy=aggregation_strategy,
            entropy_max_fraction=entropy_max_fraction,
            entropy_bottom_percentile=entropy_bottom_percentile,
            n_estimators_stage1=n_estimators_stage1,
            n_estimators_stage2=n_estimators_stage2,
            n_jobs=n_jobs,
            verbose=verbose,
            reference_class=ref_class_for_model,
            tuning_enabled=tuning_enabled,
            tuning_cv_splits=tuning_cv_splits,
            tuning_strategies=tuning_strategies,
            tuning_entropy_max_fractions=tuning_entropy_max_fractions,
            tuning_entropy_percentiles=tuning_entropy_percentiles,
        )

    # Merge disease_filter into run_params for artifact metadata. In multi-binary
    # mode, disease_filter changes per pair, so it can't be set in main().
    rp = {**(run_params or {}), "disease_filter": disease_filter}

    # Directory for reading Stage 1 artifacts. Defaults to output_dir unless
    # --stage1-dir was provided (for sharing Stage 1 across experiments).
    stage1_read_dir = stage1_source_dir if stage1_source_dir is not None else output_dir

    # Save human-readable run config at the start (before any training, so
    # it's available even if the run crashes).  Each output_dir gets its own
    # copy — for multi-binary, this means each pair subdirectory.
    if run_config_text is not None and timestamp is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / f"run_config_{timestamp}.txt"
        config_path.write_text(run_config_text)
        logger.info(f"  Saved run config: {config_path}")

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
            stage1_dir=stage1_read_dir if stage1_source_dir is not None else None,
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
            stage1_path = stage1_read_dir / f"fold_{fold_id}_stage1.pkl"
            stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
            results_path = output_dir / f"fold_{fold_id}_results.json"
            predictions_path = output_dir / f"fold_{fold_id}_predictions.pkl"

            # Build "found on disk" summary across both directories
            found_items = []
            if stage1_path.exists():
                s1_label = stage1_path.name
                if stage1_source_dir is not None:
                    s1_label += f" (from {stage1_read_dir})"
                found_items.append(s1_label)
            for p in [stage2_path, results_path, predictions_path]:
                if p.exists():
                    found_items.append(p.name)

            if resume_from_stage2:
                tuning_csv_path = output_dir / f"fold_{fold_id}_tuning_cv_results.csv"
                survival_path = output_dir / f"fold_{fold_id}_entropy_survival_stats.csv"
                survival_test_path = output_dir / f"fold_{fold_id}_entropy_survival_stats_test.csv"
                to_remove = [
                    stage2_path, results_path, predictions_path,
                    tuning_csv_path, survival_path, survival_test_path,
                ]
                removed = [p.name for p in to_remove if p.exists()]
                for p in to_remove:
                    if p.exists():
                        p.unlink()
                s1_source = stage1_path.name
                if stage1_source_dir is not None:
                    s1_source += f" (from {stage1_read_dir})"
                logger.info(
                    f"Fold {fold_id}: --resume-from-stage2\n"
                    f"  Found on disk: {', '.join(found_items)}\n"
                    f"  Keeping:       {s1_source} (Stage 1 models)\n"
                    f"  Deleting:      {', '.join(removed) if removed else '(none)'}\n"
                    f"  Will do:       load Stage 1 -> retrain Stage 2 -> evaluate on test"
                )
            else:  # resume_from_evaluation
                survival_test_path = output_dir / f"fold_{fold_id}_entropy_survival_stats_test.csv"
                to_remove = [results_path, predictions_path, survival_test_path]
                removed = [p.name for p in to_remove if p.exists()]
                for p in to_remove:
                    if p.exists():
                        p.unlink()
                logger.info(
                    f"Fold {fold_id}: --resume-from-evaluation\n"
                    f"  Found on disk: {', '.join(found_items)}\n"
                    f"  Keeping:       {stage1_path.name}, {stage2_path.name}\n"
                    f"  Deleting:      {', '.join(removed) if removed else '(none)'}\n"
                    f"  Will do:       load Stage 1 + Stage 2 -> evaluate on test"
                )

    for fold_id in fold_ids:
        # Stage 1 read path: from stage1_read_dir (may differ from output_dir)
        # Stage 1 write path: always output_dir (only used when training Stage 1)
        stage1_read_path = stage1_read_dir / f"fold_{fold_id}_stage1.pkl"
        stage1_write_path = output_dir / f"fold_{fold_id}_stage1.pkl"
        stage2_path = output_dir / f"fold_{fold_id}_stage2.pkl"
        results_path = output_dir / f"fold_{fold_id}_results.json"
        predictions_path = output_dir / f"fold_{fold_id}_predictions.pkl"

        # --- Resume: skip folds with complete artifacts on disk ---
        if resume:
            if _check_fold_complete(output_dir, fold_id):
                logger.info(f"\n{'='*60}")
                logger.info(f"Fold {fold_id} — skipped (all 4 artifacts found on disk)")
                logger.info(
                    f"  Found: {stage1_read_path.name}, {stage2_path.name}, "
                    f"{results_path.name}, {predictions_path.name}\n"
                    f"  Will do: load existing results (no training or evaluation)"
                )
                logger.info(f"{'='*60}")

                # Validate artifacts against current run params.
                tmp_model = _make_model()
                full_params = _build_model_params(tmp_model, **rp)
                # When tuning is enabled, aggregation_strategy/entropy params are
                # tuning outcomes that differ per fold — exclude from resume validation.
                s2_params = full_params.copy()
                if tuning_enabled:
                    for _k in ("aggregation_strategy", "entropy_max_fraction",
                               "entropy_bottom_percentile"):
                        s2_params.pop(_k, None)
                del tmp_model

                # Validate Stage 2 (excludes tuning-outcome params when auto_tuned)
                with open(stage2_path, "rb") as f:
                    s2_meta = pickle.load(f).get("_meta", {})
                try:
                    _validate_artifact_meta(
                        s2_meta, "Stage 2", fold_id,
                        current_model_params=s2_params,
                    )
                except ValueError as e:
                    raise ValueError(
                        f"{e}\n\n"
                        f"Hint: If you changed Stage-2-only parameters "
                        f"(aggregation strategy, entropy params,"
                        f"n_estimators_stage2, reweigh_by_subset_frequencies), "
                        f"use --resume-from-stage2 instead of --resume to "
                        f"retrain Stage 2 while keeping the saved Stage 1 models."
                    ) from None

                # Validate Stage 1 (excluding Stage-2-only params)
                with open(stage1_read_path, "rb") as f:
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
                stage1_exists = stage1_read_path.exists()
                stage2_exists = stage2_path.exists()
                all_artifacts = {
                    stage1_read_path.name: stage1_exists,
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
            resume and stage1_read_path.exists()
            and stage1_read_path.stat().st_size >= _MIN_ARTIFACT_BYTES
        )
        resume_stage2 = (
            resume and stage2_path.exists()
            and stage2_path.stat().st_size >= _MIN_ARTIFACT_BYTES
        )
        if resume:
            for tag, path, flag in [
                ("Stage 1", stage1_read_path, resume_stage1),
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

            # Split into train_smaller1 and train_smaller2 using centralized splits.
            # cv_single_model: ts1+ts2 = all train participants
            # cv_ensemble: ts1+ts2 = train participants minus validation
            ts1_participants = set(loader.get_split_participants(
                fold_id, training_context, ["train_smaller1"]
            ))
            ts2_participants = set(loader.get_split_participants(
                fold_id, training_context, ["train_smaller2"]
            ))

            ts1 = train_seq[train_seq[PARTICIPANT_COL].isin(ts1_participants)].copy()
            ts2 = train_seq[train_seq[PARTICIPANT_COL].isin(ts2_participants)].copy()
            timings["load_train_data"] = time.monotonic() - t0

            # Assertions: split filtering must produce non-empty data with expected
            # participant counts. Empty splits indicate a bug in split generation or
            # a mismatch between fold data and split files.
            # In binary mode, disease_filter was applied above, so the data only has
            # 2 diseases — participant counts will be a subset of the full split.
            # In multiclass mode, counts should match exactly.
            ts1_actual = ts1[PARTICIPANT_COL].nunique()
            ts2_actual = ts2[PARTICIPANT_COL].nunique()
            assert len(ts1) > 0, (
                f"train_smaller1 is empty after split filtering (fold {fold_id}, "
                f"context={training_context}). Expected {len(ts1_participants)} participants."
            )
            assert len(ts2) > 0, (
                f"train_smaller2 is empty after split filtering (fold {fold_id}, "
                f"context={training_context}). Expected {len(ts2_participants)} participants."
            )
            if not disease_filter:
                # Multiclass: all split participants should be present in the data
                assert ts1_actual == len(ts1_participants), (
                    f"train_smaller1 participant count mismatch: got {ts1_actual}, "
                    f"expected {len(ts1_participants)} (fold {fold_id}, context={training_context})"
                )
                assert ts2_actual == len(ts2_participants), (
                    f"train_smaller2 participant count mismatch: got {ts2_actual}, "
                    f"expected {len(ts2_participants)} (fold {fold_id}, context={training_context})"
                )
            else:
                # Binary: data was filtered to 2 diseases, so only a subset of
                # split participants will be present. Just verify subset relationship.
                assert ts1_actual <= len(ts1_participants), (
                    f"train_smaller1 has MORE participants ({ts1_actual}) than split "
                    f"({len(ts1_participants)}) — impossible (fold {fold_id})"
                )
                assert ts2_actual <= len(ts2_participants), (
                    f"train_smaller2 has MORE participants ({ts2_actual}) than split "
                    f"({len(ts2_participants)}) — impossible (fold {fold_id})"
                )

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
            _load_stage1_artifact(model, stage1_read_path, fold_id, locus,
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
            if use_inline_embeddings:
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
            _save_stage1_artifact(model, stage1_write_path, fold_id, ts1,
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
                    f"(aggregation strategy, entropy params,"
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
            if use_inline_embeddings:
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

            # Remove any stale training-time diagnostics for this fold before writing
            # fresh ones, so a CSV from a prior run/config can't linger when the current
            # run doesn't regenerate it (e.g. a tuning_cv_results.csv left by an earlier
            # auto_tuned run, when the current run uses a fixed strategy). This runs only
            # when Stage 2 is (re)trained; resume-from-evaluation keeps the loaded
            # Stage 2's diagnostics untouched.
            for _stale in (
                output_dir / f"fold_{fold_id}_entropy_survival_stats.csv",
                output_dir / f"fold_{fold_id}_tuning_cv_results.csv",
            ):
                if _stale.exists():
                    _stale.unlink()

            # Save entropy filter survival stats (per-specimen, per-group)
            if model.last_entropy_survival_stats_ is not None:
                survival_path = output_dir / f"fold_{fold_id}_entropy_survival_stats.csv"
                model.last_entropy_survival_stats_.to_csv(survival_path, index=False)
                logger.info(f"  Saved entropy survival stats: {survival_path}")

            # Save tuning CV results (all candidates ranked by mean MCC)
            if model.tuning_enabled_ and model.tuning_results_:
                tuning_csv_path = output_dir / f"fold_{fold_id}_tuning_cv_results.csv"
                _write_tuning_cv_results(model, tuning_csv_path)
                logger.info(f"  Saved tuning CV results: {tuning_csv_path}")

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
        if use_inline_embeddings:
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

        # Save test-time entropy filter survival stats
        if model.last_entropy_survival_stats_ is not None:
            test_survival_path = output_dir / f"fold_{fold_id}_entropy_survival_stats_test.csv"
            model.last_entropy_survival_stats_.to_csv(test_survival_path, index=False)
            logger.info(f"  Saved test entropy survival stats: {test_survival_path}")

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

        # Add tuning selection info (per fold) to eval_results
        if model.tuning_enabled_ and model.tuning_results_:
            best = model.tuning_results_[0]
            eval_results["tuning_selected_strategy"] = best["strategy_name"]
            eval_results["tuning_selected_threshold"] = best["threshold_param"]
            eval_results["tuning_selected_mean_mcc"] = best["mean_mcc"]

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
        mcc_val = eval_results.get("mcc")
        metric_parts = []
        if auroc_val is not None:
            metric_parts.append(f"AUROC={auroc_val:.4f}")
        if auprc_val is not None:
            metric_parts.append(f"AUPRC={auprc_val:.4f}")
        if mcc_val is not None:
            metric_parts.append(f"MCC={mcc_val:.4f}")
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
                    FOLD_COL: fold_id,
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
                    FOLD_COL: fold_id,
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
# Train-all: single-pass training on the whole dataset (no evaluation)
# ---------------------------------------------------------------------------

def _run_train_all(
    output_dir: Path,
    disease_filter: Optional[Tuple[str, str]] = None,
    *,
    loader: MalIDPublishedDataLoader,
    locus: str,
    n_estimators_stage1: int,
    n_estimators_stage2: int,
    n_jobs: int,
    verbose: int,
    embedding_dir: Optional[Path],
    use_inline_embeddings: bool,
    device: Optional[str],
    embedding_batch_size: int,
    aggregation_strategy: Optional[AggregationStrategy],
    entropy_max_fraction: Optional[float],
    entropy_bottom_percentile: Optional[float],
    training_context: str,
    resume: bool,
    resume_from_stage2: bool,
    run_config_text: Optional[str],
    timestamp: Optional[str],
    tuning_enabled: bool,
    tuning_cv_splits: int,
    tuning_strategies: Optional[List[str]],
    tuning_entropy_max_fractions: Optional[List[float]],
    tuning_entropy_percentiles: Optional[List[float]],
    run_params: Optional[dict] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Train Model 3 once on the WHOLE dataset (train-all); no evaluation.

    Single-pass counterpart of ``_run_fold_loop`` for train-all contexts, and the
    ``fold_loop_fn`` used by ``train_full_dataset`` (via ``run_training_orchestration``,
    which supplies ``output_dir`` and ``disease_filter``). Like Model 2, Model 3 uses
    ``train_smaller1`` (ts1) and ``train_smaller2`` (ts2) SEPARATELY — Stage 1 (per-V-gene
    sequence classifiers) is fit on ts1, and Stage 2 (specimen-level rollup) is fit on
    Stage-1 predictions over the disjoint ts2 — so the ts1/ts2 split is mandatory (it
    keeps the Stage-2 features out-of-sample; see the model docstring). There is no test
    set and no evaluation. For ``train_all``, ts1+ts2 = all participants; for
    ``train_all_ensemble``, ts1+ts2 = the 2/3 that excludes the validation third.

    Artifacts are written WITHOUT a fold prefix (``stage1.pkl``, ``stage2.pkl``, plus
    optional ``entropy_survival_stats.csv`` / ``tuning_cv_results.csv``) and a ``meta.json``
    sentinel written LAST. Returns ``([training_info], {})``.

    Resume (see also the CLI guard in ``main`` that rejects ``--resume-from-evaluation``
    for train-all):
    - ``resume`` (all-or-nothing): if ``meta.json`` is present/valid, its
      ``expected_artifacts`` all exist and are non-empty, and its params match → reload and
      return; otherwise delete partials and retrain both stages.
    - ``resume_from_stage2``: keep a valid ``stage1.pkl`` (validating its Stage-1 params),
      delete Stage-2 artifacts + meta, and retrain Stage 2 only. Requires ``stage1.pkl``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tag = (
        f" [{make_pair_name(disease_filter[0], disease_filter[1])}]"
        if disease_filter else ""
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"Train-all ({training_context}){pair_tag}")
    logger.info(f"{'='*60}")

    # Save human-readable run config at the start (before any training), so it's
    # available even if the run crashes. Each output_dir (incl. per-pair) gets a copy.
    if run_config_text is not None and timestamp is not None:
        (output_dir / f"run_config_{timestamp}.txt").write_text(run_config_text)
        logger.info(f"  Saved run config: run_config_{timestamp}.txt")

    ref_class = disease_filter[1] if disease_filter else None

    def _fresh_model() -> SequenceLevelClassifier:
        """Build a fresh (unfitted) model via the shared _build_model helper."""
        return _build_model(
            locus=locus,
            aggregation_strategy=aggregation_strategy,
            entropy_max_fraction=entropy_max_fraction,
            entropy_bottom_percentile=entropy_bottom_percentile,
            n_estimators_stage1=n_estimators_stage1,
            n_estimators_stage2=n_estimators_stage2,
            n_jobs=n_jobs,
            verbose=verbose,
            reference_class=ref_class,
            tuning_enabled=tuning_enabled,
            tuning_cv_splits=tuning_cv_splits,
            tuning_strategies=tuning_strategies,
            tuning_entropy_max_fractions=tuning_entropy_max_fractions,
            tuning_entropy_percentiles=tuning_entropy_percentiles,
        )

    # run_params for artifact metadata (merge disease_filter, per Model 3 convention).
    rp = {**(run_params or {}), "disease_filter": disease_filter}

    meta_file = output_dir / "meta.json"
    stage1_path, stage2_path = _stage_artifact_paths(output_dir, None)
    entropy_csv = output_dir / "entropy_survival_stats.csv"
    tuning_csv = output_dir / "tuning_cv_results.csv"

    # Identity used by validate_train_all_meta on plain --resume. model_params is read
    # from a fresh (pre-fit) model, so it captures the INPUT config (incl. locus and the
    # Stage-1/2 knobs) — a mismatch on resume raises. (When tuning is enabled the recorded
    # aggregation_strategy is the pre-fit placeholder in BOTH the saved and current meta,
    # so it compares consistently.)
    meta_expected = {
        "training_context": training_context,
        "disease_filter": list(disease_filter) if disease_filter else None,
        "run_params": run_params or {},
        "model_params": _build_model_params(_fresh_model(), **rp),
    }

    # --- Resume handling ---
    if resume_from_stage2:
        # Reuse Stage 1, retrain Stage 2. Require a valid stage1.pkl.
        if not train_all_artifacts_complete([stage1_path]):
            raise ValueError(
                f"--resume-from-stage2 requires a saved Stage 1 artifact at "
                f"{stage1_path}, but none was found (or it is empty/truncated). "
                f"Run without --resume-from-stage2 to train Stage 1 from scratch."
            )
        for p in (stage2_path, meta_file, entropy_csv, tuning_csv):
            if p.exists():
                logger.info(f"  Deleting for Stage-2 retrain: {p.name}")
                p.unlink()
    elif resume:
        saved_meta = None
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    saved_meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"  Corrupt meta.json ({e}); treating as incomplete and retraining."
                )
                saved_meta = None
        expected_artifacts = saved_meta.get("expected_artifacts", []) if saved_meta else []
        # A complete run ALWAYS records at least one expected artifact (see the meta.json
        # write below). An empty/missing list means a truncated meta → retrain (don't let
        # all([]) == True mark it complete).
        complete = (
            saved_meta is not None
            and "training_info" in saved_meta
            and len(expected_artifacts) > 0
            and train_all_artifacts_complete(
                output_dir / name for name in expected_artifacts
            )
        )
        if complete:
            validate_train_all_meta(
                saved_meta, meta_expected, output_dir,
                match_keys=["training_context", "disease_filter"],
            )
            logger.info(
                "  Skipped (all artifacts present and params match); "
                "reloading saved training info."
            )
            return [saved_meta["training_info"]], {}
        # Incomplete/corrupt: delete partial artifacts before retraining.
        for p in (stage1_path, stage2_path, meta_file, entropy_csv, tuning_csv):
            if p.exists():
                logger.info(f"  Deleting incomplete artifact: {p.name}")
                p.unlink()
    else:
        # Fresh run (no resume): clear any prior train-all artifacts in this dir so a
        # stale file from a previous run/config can't linger and confuse — e.g. a
        # tuning_cv_results.csv left by an earlier auto_tuned run that the current
        # (non-tuned) run would not overwrite.
        for p in (stage1_path, stage2_path, meta_file, entropy_csv, tuning_csv):
            if p.exists():
                logger.info(f"  Removing prior artifact (fresh run): {p.name}")
                p.unlink()

    # --- Load the whole dataset + optional binary filter ---
    logger.info("Loading full dataset (train-all)...")
    seqs_df, meta_df = load_and_prepare_fold(loader, None, "all")
    if disease_filter:
        disease, reference_class = disease_filter
        seqs_df, meta_df = filter_to_binary_pair(seqs_df, meta_df, disease, reference_class)

    # --- Split into ts1 (Stage 1) and ts2 (Stage 2) by participant ---
    ts1_participants = set(loader.get_split_participants(
        None, training_context, ["train_smaller1"]
    ))
    ts2_participants = set(loader.get_split_participants(
        None, training_context, ["train_smaller2"]
    ))
    ts1 = seqs_df[seqs_df[PARTICIPANT_COL].isin(ts1_participants)].copy()
    ts2 = seqs_df[seqs_df[PARTICIPANT_COL].isin(ts2_participants)].copy()

    # --- Integrity checks on each subset (shared with Models 1/2) ---
    # `raise` (not `assert`) so these data-state checks survive `python -O`.
    if len(ts1) == 0:
        raise RuntimeError(
            f"train_smaller1 is empty after split filtering "
            f"(context={training_context}, pair={disease_filter})."
        )
    if len(ts2) == 0:
        raise RuntimeError(
            f"train_smaller2 is empty after split filtering "
            f"(context={training_context}, pair={disease_filter})."
        )
    check_train_all_split(
        loader, set(ts1[PARTICIPANT_COL].unique()), ts1_participants,
        training_context, disease_filter, role_label="train_smaller1",
    )
    check_train_all_split(
        loader, set(ts2[PARTICIPANT_COL].unique()), ts2_participants,
        training_context, disease_filter, role_label="train_smaller2",
    )
    ts1 = ts1.reset_index(drop=True)
    ts2 = ts2.reset_index(drop=True)
    logger.info(
        f"  train_smaller1: {ts1[PARTICIPANT_COL].nunique()} participants, "
        f"{len(ts1):,} sequences; "
        f"train_smaller2: {ts2[PARTICIPANT_COL].nunique()} participants, "
        f"{len(ts2):,} sequences"
    )

    model = _fresh_model()

    # --- Stage 1: load (resume-from-stage2) or train on ts1 ---
    if resume_from_stage2:
        expected_classes = sorted(ts1[DISEASE_COL].unique().tolist())
        _load_stage1_artifact(
            model, stage1_path, None, locus,
            expected_classes=expected_classes, run_params=run_params, ts1=ts1,
        )
        logger.info("  Loaded Stage 1 from saved artifact (--resume-from-stage2).")
    else:
        logger.info("  Loading ts1 embeddings...")
        if use_inline_embeddings:
            emb_ts1 = compute_embeddings_inline(ts1, device, embedding_batch_size)
        else:
            emb_ts1 = load_precomputed_embeddings(ts1, embedding_dir)
        logger.info("Training Stage 1 (per-group sequence classifiers)...")
        model.fit_stage1(ts1, emb_ts1)
        logger.info(
            f"  Stage 1 complete: {len(model.group_models_)} group models trained"
        )
        del emb_ts1
        _save_stage1_artifact(model, stage1_path, None, ts1, run_params=run_params)

    # --- Stage 2: always train on ts2 (train-all) ---
    logger.info("  Loading ts2 embeddings...")
    if use_inline_embeddings:
        emb_ts2 = compute_embeddings_inline(ts2, device, embedding_batch_size)
    else:
        emb_ts2 = load_precomputed_embeddings(ts2, embedding_dir)
    logger.info("Training Stage 2 (specimen-level rollup)...")
    model.fit_stage2(ts2, emb_ts2)
    n_features = len(model.feature_columns_) if model.feature_columns_ else 0
    logger.info(f"  Stage 2 complete: {n_features} specimen-level features")
    del emb_ts2
    _save_stage2_artifact(model, stage2_path, None, ts2, run_params=run_params)

    saved_artifacts = [stage1_path.name, stage2_path.name]

    # Save Stage-2 diagnostics without a fold prefix (mirrors the CV artifacts).
    if model.last_entropy_survival_stats_ is not None:
        model.last_entropy_survival_stats_.to_csv(entropy_csv, index=False)
        saved_artifacts.append(entropy_csv.name)
        logger.info(f"  Saved entropy survival stats: {entropy_csv.name}")
    if model.tuning_enabled_ and model.tuning_results_:
        _write_tuning_cv_results(model, tuning_csv)
        saved_artifacts.append(tuning_csv.name)
        logger.info(f"  Saved tuning CV results: {tuning_csv.name}")

    # --- Build training_info (no metrics) ---
    training_info = {
        "training_context": training_context,
        "classes": [str(c) for c in model.classes_],
        "n_train_ts1_participants": int(ts1[PARTICIPANT_COL].nunique()),
        "n_train_ts1_specimens": int(ts1[SPECIMEN_COL].nunique()),
        "n_train_ts1_sequences": int(len(ts1)),
        "n_train_ts2_participants": int(ts2[PARTICIPANT_COL].nunique()),
        "n_train_ts2_specimens": int(ts2[SPECIMEN_COL].nunique()),
        "n_train_ts2_sequences": int(len(ts2)),
        "locus": locus,
        # Effective aggregation strategy (the tuned winner when tuning ran).
        "aggregation_strategy": model.aggregation_strategy.name,
        "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
        "n_estimators_stage1": model.n_estimators_stage1,
        "n_estimators_stage2": model.n_estimators_stage2,
        "n_stage1_groups": len(model.group_models_),
        "n_stage2_features": n_features,
        "tuning_enabled": bool(model.tuning_enabled_),
    }
    if model.entropy_max_fraction is not None:
        training_info["entropy_max_fraction"] = model.entropy_max_fraction
    if model.entropy_bottom_percentile is not None:
        training_info["entropy_bottom_percentile"] = model.entropy_bottom_percentile
    if model.tuning_enabled_ and model.tuning_results_:
        training_info["tuning_best_strategy"] = model.aggregation_strategy.name
        training_info["tuning_best_threshold_param"] = model.tuning_results_[0].get(
            "threshold_param"
        )
    if disease_filter:
        training_info["disease"] = disease_filter[0]
        training_info["reference_class"] = disease_filter[1]

    # --- Write meta.json LAST (resume sentinel + provenance) ---
    meta_out = dict(meta_expected)
    meta_out["expected_artifacts"] = sorted(saved_artifacts)
    meta_out["training_info"] = training_info
    with open(meta_file, "w") as f:
        json.dump(
            meta_out, f, indent=2,
            default=lambda x: (
                x.tolist() if isinstance(x, np.ndarray)
                else float(x) if isinstance(x, (np.floating, np.integer))
                else x
            ),
        )
    logger.info(f"  Saved {len(saved_artifacts)} artifact(s) + meta.json")

    return [training_info], {}


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_training_params(
    aggregation_strategy: Optional[str] = None,
    n_estimators_stage1: Optional[int] = None,
    n_estimators_stage2: Optional[int] = None,
    entropy_max_fraction: Optional[float] = None,
    entropy_bottom_percentile: Optional[float] = None,
    tuning_cv_splits: Optional[int] = None,
    tuning_strategies: Optional[List[str]] = None,
    tuning_entropy_max_fractions: Optional[List[float]] = None,
    tuning_entropy_percentiles: Optional[List[float]] = None,
    **_kwargs,
) -> None:
    """Validate Model 3 training parameter ranges.

    Called by both the standalone main() and ensemble auto-training dispatch.
    Only non-None values are checked (None means "use model default").

    Raises ValueError with a clear message for any out-of-range value.
    """
    # Aggregation strategy name
    if aggregation_strategy is not None:
        _valid_agg_names = {"auto_tuned", "paper_best"} | {
            s.name for s in AggregationStrategy
        }
        if aggregation_strategy not in _valid_agg_names:
            raise ValueError(
                f"Model 3: unknown aggregation_strategy={aggregation_strategy!r}. "
                f"Valid values: {sorted(_valid_agg_names)}"
            )

    # RF estimator counts
    if n_estimators_stage1 is not None and n_estimators_stage1 < 1:
        raise ValueError(
            f"Model 3: n_estimators_stage1 must be >= 1, got {n_estimators_stage1}."
        )
    if n_estimators_stage2 is not None and n_estimators_stage2 < 1:
        raise ValueError(
            f"Model 3: n_estimators_stage2 must be >= 1, got {n_estimators_stage2}."
        )

    # Entropy parameters
    if entropy_max_fraction is not None:
        if not (0.0 < entropy_max_fraction <= 1.0):
            raise ValueError(
                f"Model 3: entropy_max_fraction must be in (0, 1], "
                f"got {entropy_max_fraction}."
            )
    if entropy_bottom_percentile is not None:
        if not (0.0 <= entropy_bottom_percentile <= 100.0):
            raise ValueError(
                f"Model 3: entropy_bottom_percentile must be in [0, 100], "
                f"got {entropy_bottom_percentile}."
            )

    # Tuning grid parameters
    if tuning_cv_splits is not None and tuning_cv_splits < 2:
        raise ValueError(
            f"Model 3: tuning_cv_splits must be >= 2, got {tuning_cv_splits}."
        )
    if tuning_strategies is not None:
        valid_tuning_names = {s.name for s in AggregationStrategy}
        for name in tuning_strategies:
            if name not in valid_tuning_names:
                raise ValueError(
                    f"Model 3: invalid tuning strategy {name!r}. "
                    f"Valid values: {sorted(valid_tuning_names)}"
                )
    if tuning_entropy_max_fractions is not None:
        for val in tuning_entropy_max_fractions:
            if not (0.0 < val <= 1.0):
                raise ValueError(
                    f"Model 3: tuning_entropy_max_fractions values must be in (0, 1], "
                    f"got {val}."
                )
    if tuning_entropy_percentiles is not None:
        for val in tuning_entropy_percentiles:
            if not (0.0 <= val <= 100.0):
                raise ValueError(
                    f"Model 3: tuning_entropy_percentiles values must be in [0, 100], "
                    f"got {val}."
                )


# ---------------------------------------------------------------------------
# Public API — callable from ensemble or standalone
# ---------------------------------------------------------------------------

def train_all_folds(
    fold_ids: Optional[List[int]],
    metadata_path: Path,
    output_dir: Optional[Path] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    classification_mode: str = "multiclass",
    reference_class: Optional[str] = None,
    diseases: Optional[List[str]] = None,
    gene_locus: str = "TCR",
    aggregation_strategy: str = "entropy_percentile_cutoff",
    entropy_max_fraction: Optional[float] = None,
    entropy_bottom_percentile: Optional[float] = None,
    n_estimators_stage1: int = 100,
    n_estimators_stage2: int = 100,
    n_jobs: int = 4,
    verbose: int = 1,
    embedding_dir: Optional[Path] = None,
    cache_embeddings: bool = True,
    device: Optional[str] = None,
    embedding_batch_size: int = 64,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    gene_reference_path: Optional[Path] = None,
    output_suffix: Optional[str] = None,
    training_context: str = "cv_single_model",
    resume: bool = False,
    resume_from_stage2: bool = False,
    resume_from_evaluation: bool = False,
    stage1_dir: Optional[Path] = None,
    tuning_cv_splits: int = 3,
    tuning_strategies: Optional[List[str]] = None,
    tuning_entropy_max_fractions: Optional[List[float]] = None,
    tuning_entropy_percentiles: Optional[List[float]] = None,
    clone_id_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Train Model 3 on all specified folds, with optional resume support.

    Trains a SequenceLevelClassifier (two-stage V-gene-specific sequence model)
    per fold, evaluates on the held-out test set, aggregates results, and writes
    summary JSON + Markdown results.

    With resume=True, folds with complete artifacts on disk are skipped and
    their saved results are reloaded for aggregation. Incomplete folds are
    retrained normally. Saved model parameters are validated against current
    run parameters to prevent silently mixing results from different configs.

    Parameters
    ----------
    fold_ids : List of fold IDs to train, or None for all folds in metadata.
    metadata_path : Path to the metadata TSV file.
    output_dir : Base output directory. If None, defaults to the canonical path
        under the project root. For binary/multi-binary, this is the parent of
        the per-pair subdirectories. Mutually exclusive with output_suffix.
    dataset_name : Dataset identifier used in the output path.
    classification_mode : "multiclass" | "binary" | "multi-binary".
    reference_class : Reference/negative class for binary and multi-binary modes.
    diseases : Explicit subset of disease classes to train.
    gene_locus : "TCR" or "BCR".
    aggregation_strategy : Sequence-to-specimen aggregation strategy.
        "entropy_percentile_cutoff" (default) keeps sequences in the bottom
        percentile of the training entropy distribution. "auto_tuned" searches
        a grid via inner CV on train_smaller2. "paper_best" selects the
        paper-best per locus. Otherwise, an AggregationStrategy enum name
        (e.g. "mean", "entropy_cutoff").
    entropy_max_fraction : Fraction of max possible entropy for the entropy_cutoff
        strategy (0-1). None uses the default (0.80).
    entropy_bottom_percentile : Percentile for the entropy_percentile_cutoff
        strategy (0-100). None uses the default (0.01).
    n_estimators_stage1 : Number of RF trees for Stage 1 (BCR only; ignored
        for TCR which uses glmnet ridge).
    n_estimators_stage2 : Number of RF trees for Stage 2.
    n_jobs : Parallel workers for joblib-parallelized steps.
    verbose : Verbosity level.
    embedding_dir : Directory with pre-computed ESM-2 embeddings. None uses
        the default (<cache-dir>/embeddings/).
    cache_embeddings : If True (default), pre-computed embeddings are used when
        available, and auto-computed + saved to disk when missing. If False,
        cached embeddings are still used when available, but if missing,
        embeddings are computed inline per-subset without saving to disk.
    device : Device for ESM-2 embedding ('cuda', 'mps', 'cpu', or auto).
    embedding_batch_size : Batch size for ESM-2 embedding computation.
    data_dir : Path to raw data directory. Required if cache is missing.
    cache_dir : Path to cache directory. None disables caching.
    gene_reference_path : Path to gene reference file (V-gene CDR sequences).
    output_suffix : Suffix appended to the mode directory name (e.g. "entropy_pct_01"
        produces "multiclass__entropy_pct_01"). Ignored when output_dir is set.
    training_context : Training context controlling data splits and output paths.
        "cv_single_model" (default) or "cv_ensemble".
    resume : If True, skip folds with complete artifacts on disk and reload
        their results.
    resume_from_stage2 : If True, load Stage 1 from saved artifacts and retrain
        Stage 2 from scratch. Implies resume=True for Stage 1.
    resume_from_evaluation : If True, load both stages from saved artifacts and
        re-run evaluation only. Implies resume=True for earlier stages.
    stage1_dir : Directory to read Stage 1 artifacts from instead of the output
        directory. Only valid with resume_from_stage2=True.
    tuning_cv_splits : Number of inner CV folds for auto-tuning (default 3).
    tuning_strategies : List of strategy names for auto-tuning grid search.
        None uses the model defaults.
    tuning_entropy_max_fractions : Grid of max_fraction values for auto-tuning.
        None uses the model defaults.
    tuning_entropy_percentiles : Grid of percentile values for auto-tuning.
        None uses the model defaults.
    clone_id_kwargs : Dict of clone_id parameters for the data loader
        (from get_clone_id_kwargs). None means all params unspecified —
        cached values accepted as-is.

    Returns
    -------
    Dict mapping pair/mode key to {"fold_results": List[Dict],
    "aggregated_by_model": Dict[str, Dict]}.
    """
    t_start = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Input validation (catches misuse by programmatic callers) ---
    if metadata_path is None:
        raise ValueError("metadata_path is required")
    if resume_from_stage2 and resume_from_evaluation:
        raise ValueError(
            "resume_from_stage2 and resume_from_evaluation are mutually exclusive"
        )
    if stage1_dir is not None and not resume_from_stage2:
        raise ValueError(
            "stage1_dir requires resume_from_stage2=True. "
            "It specifies where to read Stage 1 artifacts from when "
            "retraining Stage 2 in a separate output directory."
        )
    if output_dir is not None and output_suffix is not None:
        raise ValueError(
            "output_dir and output_suffix are mutually exclusive. "
            "Use output_dir for a fully custom path, or output_suffix "
            "to append to the canonical directory name."
        )
    if gene_locus not in ("TCR",):
        raise ValueError(
            f"Unsupported gene_locus={gene_locus!r}. Only 'TCR' is supported."
        )
    if training_context not in VALID_TRAINING_CONTEXTS:
        raise ValueError(
            f"Unknown training_context={training_context!r}. "
            f"Valid values: {sorted(VALID_TRAINING_CONTEXTS)}"
        )
    if fold_ids is not None and len(fold_ids) == 0:
        raise ValueError(
            "fold_ids is an empty list. Pass None to auto-detect from metadata, "
            "or provide at least one fold ID."
        )
    if diseases is not None and len(diseases) == 0:
        raise ValueError(
            "diseases is an empty list. Pass None to use all diseases, "
            "or provide at least one disease name."
        )
    # Resolve embedding_dir and ensure embeddings are available (shared helper —
    # embeddings are per-participant and fold/context-independent).
    embedding_dir, _use_inline_embeddings, _embedding_dir_explicit = (
        _resolve_embeddings_and_compute(
            embedding_dir=embedding_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            metadata_path=metadata_path,
            cache_embeddings=cache_embeddings,
            device=device,
            embedding_batch_size=embedding_batch_size,
            gene_locus=gene_locus,
            clone_id_kwargs=clone_id_kwargs,
            verbose=verbose,
        )
    )

    # Targeted resume implies resume behavior for earlier stages
    if resume_from_stage2 or resume_from_evaluation:
        resume = True

    # ------------------------------------------------------------------ #
    # Setup loader                                                         #
    # ------------------------------------------------------------------ #
    t0 = time.monotonic()
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=gene_locus,
        gene_reference_path=gene_reference_path,
        cache_dir=cache_dir,
        verbose=1,
        **(clone_id_kwargs or {}),
    )

    # Precompute clone IDs in parallel (no-op if all participants cached)
    if loader.cache_dir is not None:
        loader.precompute_clone_ids(n_jobs=n_jobs)

    disease_classes = get_dataset_disease_classes(loader.metadata)
    if fold_ids is None:
        fold_ids = sorted(
            loader.metadata[FOLD_COL]
            .dropna().unique().astype(int).tolist()
        )
        logger.info(f"  Auto-detected fold IDs from metadata: {fold_ids}")
    logger.info(f"Loader setup [{_fmt_elapsed(time.monotonic() - t0)}]")

    # Early completeness check for an explicit --embedding-dir (shared helper).
    _verify_embeddings_complete(
        loader=loader,
        embedding_dir=embedding_dir,
        embedding_dir_explicit=_embedding_dir_explicit,
        use_inline_embeddings=_use_inline_embeddings,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
    )

    reference_class = validate_mode_and_classes(
        classification_mode=classification_mode,
        disease_classes=disease_classes,
        reference_class=reference_class,
        diseases=diseases,
    )

    # ------------------------------------------------------------------ #
    # Resolve aggregation strategy + tuning grids                          #
    # ------------------------------------------------------------------ #
    _valid_agg_names = {"auto_tuned", "paper_best"} | {s.name for s in AggregationStrategy}
    if aggregation_strategy not in _valid_agg_names:
        raise ValueError(
            f"Unknown aggregation_strategy={aggregation_strategy!r}. "
            f"Valid values: {sorted(_valid_agg_names)}"
        )
    tuning_enabled = aggregation_strategy == "auto_tuned"
    if aggregation_strategy == "auto_tuned":
        agg_strategy = None
    elif aggregation_strategy == "paper_best":
        agg_strategy = None
    else:
        agg_strategy = AggregationStrategy[aggregation_strategy]

    # Resolve effective tuning grids (fill in model defaults when user didn't
    # specify custom values) so that all outputs document the actual values used.
    _eff_tuning_strategies = tuning_strategies or list(_DEFAULT_TUNING_STRATEGIES)
    _eff_tuning_max_fractions = tuning_entropy_max_fractions or list(_DEFAULT_TUNING_MAX_FRACTIONS)
    _eff_tuning_percentiles = tuning_entropy_percentiles or list(_DEFAULT_TUNING_PERCENTILES)

    # ------------------------------------------------------------------ #
    # Output directory                                                     #
    # ------------------------------------------------------------------ #
    base_dir = output_dir or get_model_output_dir(
        model_name=MODEL_NAME,
        dataset_name=dataset_name,
        classification_mode=classification_mode,
        gene_locus=gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Compute display strings for the aggregation strategy
    agg_display = (
        "auto_tuned" if tuning_enabled
        else (agg_strategy.name if agg_strategy is not None else "paper_best")
    )

    logger.info(f"Starting Model 3 training — {timestamp}")
    logger.info(f"  Dataset:             {dataset_name}")
    logger.info(f"  Training context:    {training_context}")
    logger.info(f"  Classification mode: {classification_mode}")
    logger.info(f"  Reference class:     {reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {diseases or '(all)'}")
    logger.info(f"  Gene locus:          {gene_locus}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Aggregation:         {agg_display}")
    if tuning_enabled:
        logger.info(f"  Tuning strategies:   {_eff_tuning_strategies}")
        logger.info(f"  Tuning max fractions: {_eff_tuning_max_fractions}")
        logger.info(f"  Tuning percentiles:  {_eff_tuning_percentiles}")
        logger.info(f"  Tuning CV splits:    {tuning_cv_splits}")
    else:
        logger.info(f"  Entropy max fraction:     {entropy_max_fraction or 'default'}")
        logger.info(f"  Entropy bottom pctile:    {entropy_bottom_percentile or 'default'}")
    logger.info(f"  Stage 1 estimators:  {n_estimators_stage1}")
    logger.info(f"  Stage 2 estimators:  {n_estimators_stage2}")
    logger.info(f"  n_jobs:              {n_jobs}")
    logger.info(f"  Verbose:             {verbose}")
    logger.info(f"  Resume:              {resume}")
    logger.info(f"  Resume from stage2:  {resume_from_stage2}")
    logger.info(f"  Resume from eval:    {resume_from_evaluation}")
    logger.info(f"  Embedding dir:       {embedding_dir}")
    _emb_mode = "inline (no caching)" if _use_inline_embeddings else "pre-computed"
    logger.info(f"  Embedding mode:      {_emb_mode}")
    logger.info(f"  Cache embeddings:    {cache_embeddings}")
    logger.info(f"  Device:              {device or 'auto'}")
    logger.info(f"  Base output dir:     {base_dir}")
    if output_suffix:
        logger.info(f"  Output suffix:       {output_suffix}")
    if stage1_dir:
        logger.info(f"  Stage 1 source dir:  {stage1_dir}")

    # Build human-readable run config text, saved to each output directory.
    _resume_mode = (
        "resume_from_evaluation" if resume_from_evaluation
        else "resume_from_stage2" if resume_from_stage2
        else "resume" if resume
        else "fresh"
    )
    _config_lines = [
        f"Run Configuration",
        f"{'=' * 60}",
        f"Timestamp:              {timestamp}",
        f"",
        f"Dataset:                {dataset_name}",
        f"Training context:       {training_context}",
        f"Gene locus:             {gene_locus}",
        f"Classification mode:    {classification_mode}",
        f"Reference class:        {reference_class or '(not set)'}",
        f"Diseases filter:        {diseases or '(all)'}",
        f"Folds:                  {', '.join(str(f) for f in fold_ids)}",
        f"",
        f"Stage 1:",
        f"  Classifier:           {'glmnet ridge (OvR)' if gene_locus == 'TCR' else f'RF ({n_estimators_stage1} trees)'}",
        f"  N estimators:         {n_estimators_stage1}",
        f"",
        f"Stage 2:",
        f"  Aggregation strategy: {agg_display}",
    ]
    if tuning_enabled:
        _config_lines += [
            f"  Tuning strategies:    {_eff_tuning_strategies}",
            f"  Tuning CV splits:     {tuning_cv_splits}",
            f"  Tuning max fractions: {_eff_tuning_max_fractions}",
            f"  Tuning percentiles:   {_eff_tuning_percentiles}",
        ]
    else:
        _config_lines += [
            f"  Entropy max fraction: {entropy_max_fraction or f'default ({_DEFAULT_ENTROPY_MAX_FRACTION})'}",
            f"  Entropy bottom pctile: {entropy_bottom_percentile or f'default ({_DEFAULT_ENTROPY_BOTTOM_PERCENTILE})'}",
        ]
    _config_lines += [
        f"  N estimators:         {n_estimators_stage2}",
        f"",
        f"Resume:",
        f"  Mode:                 {_resume_mode}",
        f"  Stage 1 source dir:   {stage1_dir or '(same as output)'}",
        f"  Output suffix:        {output_suffix or '(none)'}",
        f"",
        f"Embeddings:",
        f"  Embedding dir:        {embedding_dir}",
        f"  Embedding mode:       {_emb_mode}",
        f"  Cache embeddings:     {cache_embeddings}",
        f"  Device:               {device or 'auto'}",
        f"  Batch size:           {embedding_batch_size}",
        f"",
        f"Other:",
        f"  n_jobs:               {n_jobs}",
        f"  Verbose:              {verbose}",
        f"  Base output dir:      {base_dir}",
        f"",
    ]
    run_config_text = "\n".join(_config_lines)

    loop_kwargs = dict(
        loader=loader,
        fold_ids=fold_ids,
        locus=gene_locus,
        n_estimators_stage1=n_estimators_stage1,
        n_estimators_stage2=n_estimators_stage2,
        n_jobs=n_jobs,
        verbose=verbose,
        embedding_dir=embedding_dir,
        use_inline_embeddings=_use_inline_embeddings,
        device=device,
        embedding_batch_size=embedding_batch_size,
        aggregation_strategy=agg_strategy,
        entropy_max_fraction=entropy_max_fraction,
        entropy_bottom_percentile=entropy_bottom_percentile,
        training_context=training_context,
        resume=resume,
        resume_from_stage2=resume_from_stage2,
        resume_from_evaluation=resume_from_evaluation,
        run_config_text=run_config_text,
        timestamp=timestamp,
        tuning_enabled=tuning_enabled,
        tuning_cv_splits=tuning_cv_splits,
        tuning_strategies=tuning_strategies,
        tuning_entropy_max_fractions=tuning_entropy_max_fractions,
        tuning_entropy_percentiles=tuning_entropy_percentiles,
        run_params={
            "classification_mode": classification_mode,
            # Sort for a stable resume identity (Models 1/2 sort too): validate_train_all_meta
            # compares run_params, so a reordered --diseases must not spuriously mismatch.
            "diseases": sorted(diseases) if diseases else None,
            "dataset_name": dataset_name,
            "training_context": training_context,
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
        classification_mode == "multi-binary"
        and (resume_from_stage2 or resume_from_evaluation)
    ):
        if diseases is not None:
            _diseases_to_check = list(diseases)
        else:
            _diseases_to_check = [c for c in disease_classes if c != reference_class]

        mode_name = (
            "resume_from_stage2" if resume_from_stage2
            else "resume_from_evaluation"
        )
        all_errors: List[str] = []
        for _disease in _diseases_to_check:
            pair_name = make_pair_name(_disease, reference_class)
            pair_dir = base_dir / pair_name
            s1_pair = stage1_dir / pair_name if stage1_dir is not None else None
            pair_errors = _validate_resume_artifacts(
                pair_dir, fold_ids,
                resume_from_stage2, resume_from_evaluation,
                stage1_dir=s1_pair,
            )
            if pair_errors:
                all_errors.append(f"  {pair_name}/")
                all_errors.extend(f"    {e.strip()}" for e in pair_errors)

        if all_errors:
            detail = "\n".join(all_errors)
            if resume_from_stage2:
                hint = (
                    "Run without resume_from_stage2 to train from scratch, "
                    "or use diseases= to resume only the pairs that have "
                    "Stage 1 artifacts for all folds."
                )
            else:
                hint = (
                    "Use resume_from_stage2 if only Stage 1 is available, "
                    "or run without resume flags to train from scratch."
                )
            raise ValueError(
                f"{mode_name} requires saved artifacts, but some disease "
                f"pairs are missing them:\n{detail}\n{hint}"
            )

    # ------------------------------------------------------------------ #
    # Delete old summary/results/config files BEFORE training so stale   #
    # files from a prior run don't persist if this run fails partway.    #
    # Covers both the base_dir level and per-pair subdirectories         #
    # (binary/multi-binary write per-pair summaries + per-pair configs). #
    # Log files (training_*.log) are preserved — they document previous  #
    # runs and are useful when resuming.                                 #
    # ------------------------------------------------------------------ #
    for pattern in ("summary_*.json", "RESULTS_*.md", "run_config_*.txt"):
        for old_file in sorted(base_dir.glob(pattern)):
            logger.info(f"  Removing old: {old_file.name}")
            old_file.unlink()
    for subdir in sorted(base_dir.iterdir()) if base_dir.is_dir() else []:
        if subdir.is_dir():
            for pattern in ("summary_*.json", "RESULTS_*.md", "run_config_*.txt"):
                for old_file in sorted(subdir.glob(pattern)):
                    logger.info(f"  Removing old: {subdir.name}/{old_file.name}")
                    old_file.unlink()

    # ------------------------------------------------------------------ #
    # Training orchestration (dispatches multiclass / binary / multi-bin) #
    # ------------------------------------------------------------------ #
    all_results = run_training_orchestration(
        base_dir=base_dir,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_fold_loop,
        loop_kwargs=loop_kwargs,
        stage1_base_dir=stage1_dir,
    )

    # ------------------------------------------------------------------ #
    # Save summary JSON + Markdown results                                 #
    # ------------------------------------------------------------------ #

    # Dataset counts (participants and specimens per disease class)
    dataset_counts = get_metadata_class_counts(loader.metadata)
    metadata_filter_info = loader.metadata_filter_info

    run_info = {
        "Dataset": dataset_name,
        "Training context": training_context,
        "Classification mode": classification_mode,
        "Gene locus": gene_locus,
        "Folds": ", ".join(str(f) for f in fold_ids),
        "Stage 1 classifier": (
            "glmnet ridge (OvR)" if gene_locus == "TCR"
            else f"RF ({n_estimators_stage1} trees)"
        ),
        "Aggregation strategy": agg_display,
        "Stage 2 RF trees": n_estimators_stage2,
        "Reweigh by subset frequencies": True,
        "n_jobs (V-gene groups)": n_jobs,
        "Embedding source": _emb_mode,
        "Embedding dir": str(embedding_dir) if embedding_dir else "(none)",
        "Embedding device": device or "auto",
        "Output suffix": output_suffix or "(none)",
        "Total participants": dataset_counts["total_participants"],
        "Total specimens": dataset_counts["total_specimens"],
        "Participants per class": ", ".join(
            f"{k}: {v}" for k, v in dataset_counts["participants_per_class"].items()
        ),
        "Specimens per class": ", ".join(
            f"{k}: {v}" for k, v in dataset_counts["specimens_per_class"].items()
        ),
    }
    if metadata_filter_info and metadata_filter_info["n_filtered_out"] > 0:
        run_info["Metadata filtering"] = (
            f"{metadata_filter_info['n_filtered_out']} participants excluded "
            f"(no raw data files); {metadata_filter_info['n_retained']} retained "
            f"out of {metadata_filter_info['n_original']} in metadata file"
        )
    if tuning_enabled:
        run_info["Tuning strategies"] = ", ".join(_eff_tuning_strategies)
        run_info["Tuning max fractions"] = _eff_tuning_max_fractions
        run_info["Tuning percentiles"] = _eff_tuning_percentiles
        run_info["Tuning CV splits"] = tuning_cv_splits
    else:
        run_info["Entropy max fraction"] = entropy_max_fraction if entropy_max_fraction is not None else (
            _DEFAULT_ENTROPY_MAX_FRACTION if (agg_strategy == AggregationStrategy.entropy_cutoff or
                     (agg_strategy is None and gene_locus == "TCR")) else "N/A"
        )
        run_info["Entropy bottom percentile"] = (
            entropy_bottom_percentile if entropy_bottom_percentile is not None else (
                _DEFAULT_ENTROPY_BOTTOM_PERCENTILE
                if agg_strategy == AggregationStrategy.entropy_percentile_cutoff else "N/A"
            )
        )
    if reference_class is not None:
        run_info["Reference class"] = reference_class

    # Write summary JSON (same envelope structure as Models 1/2)
    summary_path = base_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset_name": dataset_name,
                "training_context": training_context,
                # Uniform "training complete; ready for inference" marker (5.H).
                "training_complete": True,
                "classification_mode": classification_mode,
                "reference_class": reference_class,
                "diseases": diseases,
                "model_classes": get_model_classes(
                    classification_mode, disease_classes, diseases, reference_class,
                ),
                "gene_locus": gene_locus,
                "output_suffix": output_suffix,
                # Resolved clone_id clustering definition (Phase 6.E cross-dataset check).
                "clone_id_params": loader.clone_id_params,
                "fold_ids": fold_ids,
                "model_names": [MODEL_NAME],
                "aggregation_strategy": agg_display,
                "n_estimators_stage1": n_estimators_stage1,
                "n_estimators_stage2": n_estimators_stage2,
                "tuning_enabled": tuning_enabled,
                "tuning_cv_splits": tuning_cv_splits if tuning_enabled else None,
                "tuning_strategies": _eff_tuning_strategies if tuning_enabled else None,
                "tuning_entropy_max_fractions": _eff_tuning_max_fractions if tuning_enabled else None,
                "tuning_entropy_percentiles": _eff_tuning_percentiles if tuning_enabled else None,
                "entropy_max_fraction": entropy_max_fraction if entropy_max_fraction is not None else (
                    _DEFAULT_ENTROPY_MAX_FRACTION if (agg_strategy == AggregationStrategy.entropy_cutoff or
                             (agg_strategy is None and not tuning_enabled and gene_locus == "TCR")) else None
                ),
                "entropy_bottom_percentile": entropy_bottom_percentile if entropy_bottom_percentile is not None else (
                    _DEFAULT_ENTROPY_BOTTOM_PERCENTILE
                    if agg_strategy == AggregationStrategy.entropy_percentile_cutoff else None
                ),
                "reweigh_by_subset_frequencies": True,
                "dataset_counts": dataset_counts,
                "metadata_filter_info": metadata_filter_info,
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
        classification_mode=classification_mode,
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
        classification_mode=classification_mode,
        timestamp=timestamp,
        model_label=MODEL_LABEL,
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=[MODEL_NAME],
        has_abstention=False,
        gene_locus=gene_locus,
        clone_id_params=loader.clone_id_params,
        summary_json_extra={
            "dataset_name": dataset_name,
            "aggregation_strategy": agg_display,
            "tuning_enabled": tuning_enabled,
            "tuning_cv_splits": tuning_cv_splits if tuning_enabled else None,
            "tuning_strategies": _eff_tuning_strategies if tuning_enabled else None,
            "tuning_entropy_max_fractions": _eff_tuning_max_fractions if tuning_enabled else None,
            "tuning_entropy_percentiles": _eff_tuning_percentiles if tuning_enabled else None,
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
            mcc_agg = agg.get("mcc", {})
            mcc_mean = mcc_agg.get("mean") if isinstance(mcc_agg, dict) else None
            mcc_str = f"{mcc_mean:.4f}" if mcc_mean is not None else "N/A"
            if classification_mode == "multiclass":
                auroc_agg = agg.get("auroc_ovo_weighted", {})
                auroc_mean = auroc_agg.get("mean")
                auroc_str = f"{auroc_mean:.4f}" if auroc_mean is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_OvO={auroc_str} MCC={mcc_str}"
                )
            else:
                auroc_p = agg.get("auroc_pooled")
                auprc_p = agg.get("auprc_pooled")
                auroc_str = f"{auroc_p:.4f}" if auroc_p is not None else "N/A"
                auprc_str = f"{auprc_p:.4f}" if auprc_p is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_pooled={auroc_str} "
                    f"AUPRC_pooled={auprc_str} MCC={mcc_str}"
                )

    elapsed = time.monotonic() - t_start
    logger.info(f"\ntrain_all_folds completed in {_fmt_elapsed(elapsed)}")

    return all_results


# ---------------------------------------------------------------------------
# Train-all entry point (whole dataset, no CV, no evaluation)
# ---------------------------------------------------------------------------

def train_full_dataset(
    metadata_path: Path,
    output_dir: Optional[Path] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    classification_mode: str = "multiclass",
    reference_class: Optional[str] = None,
    diseases: Optional[List[str]] = None,
    gene_locus: str = "TCR",
    aggregation_strategy: str = "entropy_percentile_cutoff",
    entropy_max_fraction: Optional[float] = None,
    entropy_bottom_percentile: Optional[float] = None,
    n_estimators_stage1: int = 100,
    n_estimators_stage2: int = 100,
    n_jobs: int = 4,
    verbose: int = 1,
    embedding_dir: Optional[Path] = None,
    cache_embeddings: bool = True,
    device: Optional[str] = None,
    embedding_batch_size: int = 64,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    gene_reference_path: Optional[Path] = None,
    output_suffix: Optional[str] = None,
    training_context: str = "train_all",
    resume: bool = False,
    resume_from_stage2: bool = False,
    tuning_cv_splits: int = 3,
    tuning_strategies: Optional[List[str]] = None,
    tuning_entropy_max_fractions: Optional[List[float]] = None,
    tuning_entropy_percentiles: Optional[List[float]] = None,
    clone_id_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Train Model 3 on the WHOLE dataset (train-all); no CV, no evaluation.

    The train-all counterpart of ``train_all_folds()``: instead of a per-fold loop
    with a held-out test set, it trains a single ``SequenceLevelClassifier`` on the
    entire dataset (Stage 1 on ``train_smaller1``, Stage 2 on the disjoint
    ``train_smaller2`` — see ``_run_train_all``), to be evaluated later on a separate
    dataset (external eval). Produces reusable, fold-prefix-free artifacts
    (``stage1.pkl``, ``stage2.pkl``, optional ``entropy_survival_stats.csv`` /
    ``tuning_cv_results.csv``, plus a ``meta.json`` sentinel) and a no-metrics summary.

    ``training_context`` must be a train-all context (``train_all`` or
    ``train_all_ensemble``); use ``train_all_folds()`` for CV. Reuses
    ``run_training_orchestration`` for pair dispatch, calling ``_run_train_all`` per
    pair/mode. There is no ``fold_ids`` (train-all ignores ``CV_fold``) and no
    ``resume_from_evaluation`` (there is no evaluation stage); ``resume_from_stage2``
    is supported (reuse Stage 1, retrain Stage 2). The summary JSON (and each per-pair
    summary) carries the exact config keys ``predict_model3`` /
    ``SequenceLevelClassifier.from_summary`` read when loading these artifacts later.

    Parameters mirror ``train_all_folds`` (minus ``fold_ids`` /
    ``resume_from_evaluation`` / ``stage1_dir``). See that function's docstring for
    per-parameter detail.
    """
    t_start = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Input validation (fail fast, before any loading) ---
    if gene_locus != "TCR":
        raise ValueError(
            f"Unsupported gene_locus={gene_locus!r}. Only 'TCR' is supported."
        )
    if training_context not in TRAIN_ALL_TRAINING_CONTEXTS:
        raise ValueError(
            f"train_full_dataset requires a train-all context "
            f"{TRAIN_ALL_TRAINING_CONTEXTS}, got {training_context!r}. "
            f"Use train_all_folds() for CV contexts."
        )
    if diseases is not None and len(diseases) == 0:
        raise ValueError(
            "diseases is an empty list. Pass None to use all diseases, "
            "or provide at least one disease name."
        )

    # Resolve embedding_dir and ensure embeddings are available (shared helper —
    # embeddings are per-participant and fold/context-independent).
    embedding_dir, _use_inline_embeddings, _embedding_dir_explicit = (
        _resolve_embeddings_and_compute(
            embedding_dir=embedding_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            metadata_path=metadata_path,
            cache_embeddings=cache_embeddings,
            device=device,
            embedding_batch_size=embedding_batch_size,
            gene_locus=gene_locus,
            clone_id_kwargs=clone_id_kwargs,
            verbose=verbose,
        )
    )

    # --- Setup loader (no fold auto-detection — train-all ignores CV_fold) ---
    t0 = time.monotonic()
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=gene_locus,
        gene_reference_path=gene_reference_path,
        cache_dir=cache_dir,
        verbose=1,
        **(clone_id_kwargs or {}),
    )
    if loader.cache_dir is not None:
        loader.precompute_clone_ids(n_jobs=n_jobs)
    disease_classes = get_dataset_disease_classes(loader.metadata)
    logger.info(f"Loader setup [{_fmt_elapsed(time.monotonic() - t0)}]")

    # Early completeness check for an explicit --embedding-dir (shared helper).
    _verify_embeddings_complete(
        loader=loader,
        embedding_dir=embedding_dir,
        embedding_dir_explicit=_embedding_dir_explicit,
        use_inline_embeddings=_use_inline_embeddings,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
    )

    reference_class = validate_mode_and_classes(
        classification_mode=classification_mode,
        disease_classes=disease_classes,
        reference_class=reference_class,
        diseases=diseases,
    )

    # --- Resolve aggregation strategy + tuning grids (same as the CV path) ---
    _valid_agg_names = {"auto_tuned", "paper_best"} | {s.name for s in AggregationStrategy}
    if aggregation_strategy not in _valid_agg_names:
        raise ValueError(
            f"Unknown aggregation_strategy={aggregation_strategy!r}. "
            f"Valid values: {sorted(_valid_agg_names)}"
        )
    tuning_enabled = aggregation_strategy == "auto_tuned"
    if aggregation_strategy in ("auto_tuned", "paper_best"):
        agg_strategy = None
    else:
        agg_strategy = AggregationStrategy[aggregation_strategy]
    agg_display = (
        "auto_tuned" if tuning_enabled
        else (agg_strategy.name if agg_strategy is not None else "paper_best")
    )
    _eff_tuning_strategies = tuning_strategies or list(_DEFAULT_TUNING_STRATEGIES)
    _eff_tuning_max_fractions = tuning_entropy_max_fractions or list(_DEFAULT_TUNING_MAX_FRACTIONS)
    _eff_tuning_percentiles = tuning_entropy_percentiles or list(_DEFAULT_TUNING_PERCENTILES)

    # --- Output directory ---
    base_dir = output_dir or get_model_output_dir(
        model_name=MODEL_NAME,
        dataset_name=dataset_name,
        classification_mode=classification_mode,
        gene_locus=gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Model 3 train-all — {timestamp}")
    logger.info(f"  Dataset:             {dataset_name}")
    logger.info(f"  Training context:    {training_context}")
    logger.info(f"  Classification mode: {classification_mode}")
    logger.info(f"  Reference class:     {reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {diseases or '(all)'}")
    logger.info(f"  Gene locus:          {gene_locus}")
    logger.info(f"  Aggregation:         {agg_display}")
    logger.info(f"  Resume:              {resume}")
    logger.info(f"  Resume from stage2:  {resume_from_stage2}")
    logger.info(f"  Embedding dir:       {embedding_dir}")
    logger.info(f"  Base output dir:     {base_dir}")

    # Human-readable run config saved to each output dir (train-all variant, no folds).
    run_config_text = (
        f"Run Configuration (train-all)\n{'=' * 60}\n"
        f"Timestamp:            {timestamp}\n"
        f"Dataset:              {dataset_name}\n"
        f"Training context:     {training_context}\n"
        f"Gene locus:           {gene_locus}\n"
        f"Classification mode:  {classification_mode}\n"
        f"Reference class:      {reference_class or '(not set)'}\n"
        f"Diseases filter:      {diseases or '(all)'}\n"
        f"Aggregation strategy: {agg_display}\n"
        f"Stage 1 estimators:   {n_estimators_stage1}\n"
        f"Stage 2 estimators:   {n_estimators_stage2}\n"
        f"Resume:               {'resume_from_stage2' if resume_from_stage2 else ('resume' if resume else 'fresh')}\n"
        f"Embedding dir:        {embedding_dir}\n"
        f"Output suffix:        {output_suffix or '(none)'}\n"
        f"Base output dir:      {base_dir}\n"
    )

    loop_kwargs = dict(
        loader=loader,
        locus=gene_locus,
        n_estimators_stage1=n_estimators_stage1,
        n_estimators_stage2=n_estimators_stage2,
        n_jobs=n_jobs,
        verbose=verbose,
        embedding_dir=embedding_dir,
        use_inline_embeddings=_use_inline_embeddings,
        device=device,
        embedding_batch_size=embedding_batch_size,
        aggregation_strategy=agg_strategy,
        entropy_max_fraction=entropy_max_fraction,
        entropy_bottom_percentile=entropy_bottom_percentile,
        training_context=training_context,
        resume=resume,
        resume_from_stage2=resume_from_stage2,
        run_config_text=run_config_text,
        timestamp=timestamp,
        tuning_enabled=tuning_enabled,
        tuning_cv_splits=tuning_cv_splits,
        tuning_strategies=tuning_strategies,
        tuning_entropy_max_fractions=tuning_entropy_max_fractions,
        tuning_entropy_percentiles=tuning_entropy_percentiles,
        run_params={
            "classification_mode": classification_mode,
            # Sort for a stable resume identity (Models 1/2 sort too): validate_train_all_meta
            # compares run_params, so a reordered --diseases must not spuriously mismatch.
            "diseases": sorted(diseases) if diseases else None,
            "dataset_name": dataset_name,
            "training_context": training_context,
        },
    )

    # Remove stale summary/results/config files before training (see train_all_folds).
    delete_stale_summaries(base_dir)

    all_results = run_training_orchestration(
        base_dir=base_dir,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_train_all,
        loop_kwargs=loop_kwargs,
    )

    # Config keys read by predict_model3 / SequenceLevelClassifier.from_summary when
    # loading these artifacts later (Phase 5 / external eval). gene_locus /
    # classification_mode / reference_class are already in the shared summary envelope;
    # the PER-PAIR summary does not get the envelope, so it needs the full set — hence
    # we keep them together here and drop the envelope keys from the top-level extra.
    from_summary_keys = {
        "gene_locus": gene_locus,
        "classification_mode": classification_mode,
        "reference_class": reference_class,
        "aggregation_strategy": agg_display,
        "reweigh_by_subset_frequencies": True,
        "tuning_enabled": tuning_enabled,
        "tuning_cv_splits": tuning_cv_splits if tuning_enabled else None,
        "tuning_strategies": _eff_tuning_strategies if tuning_enabled else None,
        "tuning_entropy_max_fractions": _eff_tuning_max_fractions if tuning_enabled else None,
        "tuning_entropy_percentiles": _eff_tuning_percentiles if tuning_enabled else None,
        "entropy_max_fraction": entropy_max_fraction if entropy_max_fraction is not None else (
            _DEFAULT_ENTROPY_MAX_FRACTION if (agg_strategy == AggregationStrategy.entropy_cutoff or
                     (agg_strategy is None and not tuning_enabled and gene_locus == "TCR")) else None
        ),
        "entropy_bottom_percentile": entropy_bottom_percentile if entropy_bottom_percentile is not None else (
            _DEFAULT_ENTROPY_BOTTOM_PERCENTILE
            if agg_strategy == AggregationStrategy.entropy_percentile_cutoff else None
        ),
    }
    _envelope_keys = {"gene_locus", "classification_mode", "reference_class"}

    # --- Shared no-metrics outputs: summary JSON + RESULTS.md + per-pair ---
    write_train_all_outputs(
        base_dir=base_dir,
        all_results=all_results,
        loader=loader,
        timestamp=timestamp,
        dataset_name=dataset_name,
        training_context=training_context,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        gene_locus=gene_locus,
        output_suffix=output_suffix,
        model_names=[MODEL_NAME],
        model_label=MODEL_LABEL,
        summary_extra={
            "n_estimators_stage1": n_estimators_stage1,
            "n_estimators_stage2": n_estimators_stage2,
            **{k: v for k, v in from_summary_keys.items() if k not in _envelope_keys},
        },
        run_info_extra={
            "Aggregation strategy": agg_display,
            "Stage 1 classifier": (
                "glmnet ridge (OvR)" if gene_locus == "TCR"
                else f"RF ({n_estimators_stage1} trees)"
            ),
            "Stage 2 RF trees": n_estimators_stage2,
        },
        # Per-pair summary must be self-sufficient for from_summary, so it carries the
        # full config set (incl. the envelope keys the per-pair summary otherwise lacks).
        per_pair_summary_extra={"dataset_name": dataset_name, **from_summary_keys},
    )

    elapsed = time.monotonic() - t_start
    logger.info(f"train_full_dataset completed in {_fmt_elapsed(elapsed)}")

    return all_results


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

    add_clone_id_args(parser)

    # --- Dataset and mode ---
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name (used for cache and output directories).",
    )
    parser.add_argument(
        "--training-context",
        default="cv_single_model",
        choices=list(VALID_TRAINING_CONTEXTS),
        help=(
            "Training context controlling data splits and output directory structure. "
            "CV (train+test on one dataset): 'cv_single_model' (default) — each model "
            "independently CV-evaluated; 'cv_ensemble' — base-model training for the "
            "ensemble (reserves a validation third). "
            "Train-all (train on one dataset, evaluate later on another): 'train_all' — "
            "one model on the WHOLE dataset, no test set/metrics; 'train_all_ensemble' — "
            "same but holds out a validation third for the ensemble metamodel. "
            "Train-all writes fold-prefix-free artifacts (stage1.pkl/stage2.pkl/meta.json) "
            "and rejects --fold-ids / --resume-from-evaluation / --stage1-dir."
        ),
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
        help=(
            "Explicit disease subset for binary or multi-binary modes. "
            "binary: one disease name (optional for 2-class datasets — the non-reference "
            "class is auto-detected). "
            "multi-binary: one or more disease names."
        ),
    )
    parser.add_argument(
        "--fold-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Fold(s) to hold out as the test set (default: all folds found in "
            "metadata). For each fold listed, the model is trained from scratch "
            "on all other folds pooled together, then evaluated on that held-out "
            "fold. Example: --fold-ids 0 1 2"
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
        default="entropy_percentile_cutoff",
        choices=["auto_tuned", "paper_best"] + agg_choices,
        help=(
            "Sequence-to-specimen aggregation strategy. "
            "'entropy_percentile_cutoff' (default) keeps sequences in the bottom "
            "percentile of the training entropy distribution (see "
            "--entropy-bottom-percentile, default 0.01). "
            "'auto_tuned' searches a grid of strategies/thresholds "
            "via inner CV on train_smaller2 and picks the best per fold. "
            "'paper_best' selects the paper-best per locus: "
            "TCR=entropy_cutoff (0.80), BCR=mean. "
            "Use entropy_cutoff with --entropy-max-fraction for custom thresholds. "
            f"Options: auto_tuned, paper_best, {', '.join(agg_choices)}."
        ),
    )
    parser.add_argument(
        "--entropy-max-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of max possible entropy to use as cutoff (0-1 scale). "
            "E.g. 0.80 means keep sequences with entropy < 0.80 * max possible entropy. "
            "Only used when --aggregation-strategy is entropy_cutoff. "
            "Default: 0.80 (paper setting)."
        ),
    )
    parser.add_argument(
        "--entropy-bottom-percentile",
        type=float,
        default=None,
        help=(
            "Percentile of the training entropy distribution to use as cutoff "
            "(0-100 scale). E.g. 0.01 means keep only sequences in the bottom "
            "0.01%% of the training entropy distribution. The threshold is computed "
            "from training data and applied at both train and test time. "
            "Only used when --aggregation-strategy is entropy_percentile_cutoff. "
            "Default: 0.01."
        ),
    )

    # --- Auto-tuning parameters (only used with --aggregation-strategy auto_tuned) ---
    parser.add_argument(
        "--tuning-strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated list of strategies to search during auto-tuning. "
            "Default: 'entropy_cutoff,entropy_percentile_cutoff'. "
            "Only used when --aggregation-strategy is auto_tuned."
        ),
    )
    parser.add_argument(
        "--tuning-cv-splits",
        type=int,
        default=3,
        help=(
            "Number of inner CV folds for auto-tuning (default 3). "
            "Only used when --aggregation-strategy is auto_tuned."
        ),
    )
    parser.add_argument(
        "--tuning-entropy-max-fractions",
        type=str,
        default=None,
        help=(
            "Comma-separated grid of max_fraction values (0-1) to try for "
            "entropy_cutoff during tuning. "
            "Default: '0.80,0.90,0.95'. "
            "Only used when --aggregation-strategy is auto_tuned."
        ),
    )
    parser.add_argument(
        "--tuning-entropy-percentiles",
        type=str,
        default=None,
        help=(
            "Comma-separated grid of percentile values (0-100) to try for "
            "entropy_percentile_cutoff during tuning. "
            "Default: '0.01,0.05,0.1,0.5'. "
            "Only used when --aggregation-strategy is auto_tuned."
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
        "--no-cache-embeddings",
        action="store_true",
        help=(
            "When pre-computed embeddings are not available, compute inline "
            "per-subset without saving to disk. By default (without this flag), "
            "missing embeddings are auto-computed and saved. Cached embeddings "
            "are always used when available regardless of this flag. "
            "NOTE: inline computation is slower for multi-fold runs (~3 hours "
            "per 10M sequences on Macbook Pro M4 Max MPS) because each subset "
            "is computed independently rather than per-participant."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for ESM-2 embedding: 'cuda', 'mps', 'cpu', or auto-detect. "
            "Used when embeddings need to be computed (auto or inline)."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size for ESM-2 embedding (reduce if GPU OOM). "
            "Used when embeddings need to be computed (auto or inline)."
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
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Suffix appended to the classification mode directory name. "
            "E.g. --output-suffix entropy_pct_01 produces "
            "'multiclass__entropy_pct_01' instead of 'multiclass'. "
            "Useful for running multiple Stage 2 experiments with different "
            "parameters in parallel without overwriting each other. "
            "Mutually exclusive with --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. If not provided, defaults to "
            "trained_models/<dataset_name>/model3/<mode>/<gene_locus>/ under the project root. "
            "For binary/multi-binary, each pair saves to a subdirectory of this base. "
            "Mutually exclusive with --output-suffix."
        ),
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=None,
        help=(
            "Directory to read Stage 1 artifacts from (instead of the output "
            "directory). Use this with --resume-from-stage2 and --output-suffix "
            "to share a single set of Stage 1 models across multiple Stage 2 "
            "experiments. The directory structure must match the output layout: "
            "for multiclass, Stage 1 files live directly in the directory; "
            "for binary/multi-binary, they live in <disease>_vs_<reference>/ "
            "subdirectories. Requires --resume-from-stage2."
        ),
    )
    args = parser.parse_args()

    # --- Validate --output-dir / --output-suffix mutual exclusion ---
    if args.output_dir is not None and args.output_suffix is not None:
        parser.error(
            "--output-dir and --output-suffix are mutually exclusive. "
            "Use --output-dir for a fully custom path, or --output-suffix "
            "to append to the canonical directory name."
        )

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
        # No cache and no explicit embedding dir
        embedding_dir = None

    # Validate embedding availability
    if embedding_dir is None:
        parser.error(
            "No embedding directory available (--dont-use-cache without --embedding-dir). "
            "Either provide --embedding-dir or remove --dont-use-cache."
        )
    elif not embedding_dir.exists() or not any(embedding_dir.glob("*_embeddings.npy")):
        if args.no_cache_embeddings:
            print(
                f"NOTE: No pre-computed embeddings found in {embedding_dir}.\n"
                "--no-cache-embeddings is set: embeddings will be computed inline "
                "per-subset without saving to disk."
            )
        else:
            print(
                f"NOTE: No pre-computed embeddings found in {embedding_dir}.\n"
                "Embeddings will be auto-computed for all participants before training begins.\n"
                "This is a one-time operation (~3 hours per 10M sequences on Macbook Pro M4 Max MPS)."
            )

    if args.data_dir is not None and not args.data_dir.exists():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if not args.metadata_path.exists():
        parser.error(f"--metadata-path does not exist: {args.metadata_path}")
    if args.gene_reference_path is not None and not args.gene_reference_path.exists():
        parser.error(f"--gene-reference-path does not exist: {args.gene_reference_path}")
    if args.stage1_dir is not None and not args.stage1_dir.exists():
        parser.error(f"--stage1-dir does not exist: {args.stage1_dir}")

    # Validate resume flags (at most one resume mode). The targeted modes
    # (--resume-from-stage2 / --resume-from-evaluation) are mutually exclusive
    # with each other AND with the general --resume: combining them is ambiguous
    # (the targeted mode would silently win), so error up front rather than
    # silently ignore --resume.
    if args.resume_from_stage2 and args.resume_from_evaluation:
        parser.error(
            "--resume-from-stage2 and --resume-from-evaluation are mutually exclusive"
        )
    if args.resume and (args.resume_from_stage2 or args.resume_from_evaluation):
        _targeted = "--resume-from-stage2" if args.resume_from_stage2 else "--resume-from-evaluation"
        parser.error(
            f"--resume and {_targeted} are mutually exclusive. Use --resume for a "
            f"full-run resume (skip completed folds/stages), or {_targeted} alone for "
            f"the targeted resume mode."
        )

    # Validate --stage1-dir requires --resume-from-stage2
    if args.stage1_dir is not None and not args.resume_from_stage2:
        parser.error(
            "--stage1-dir requires --resume-from-stage2. "
            "It specifies where to read Stage 1 artifacts from when "
            "retraining Stage 2 in a separate output directory."
        )

    # --- Train-all CLI guards (fail fast, before any loading) ---
    # A train-all context trains one model on the WHOLE dataset (no CV folds, no
    # test set), so fold- and evaluation-specific flags are meaningless there.
    is_train_all = args.training_context in TRAIN_ALL_TRAINING_CONTEXTS
    if is_train_all:
        if args.fold_ids is not None:
            parser.error(
                f"--fold-ids is not valid with --training-context "
                f"{args.training_context} (train-all trains on the whole dataset, "
                f"there are no CV folds). Remove --fold-ids, or use a CV context "
                f"(cv_single_model / cv_ensemble)."
            )
        if args.resume_from_evaluation:
            parser.error(
                f"--resume-from-evaluation is only valid for CV contexts; "
                f"--training-context {args.training_context} has no evaluation stage. "
                f"Use --resume or --resume-from-stage2 instead."
            )
        if args.stage1_dir is not None:
            parser.error(
                f"--stage1-dir is not supported with --training-context "
                f"{args.training_context}. For train-all, reuse a saved Stage 1 "
                f"in place with --resume-from-stage2 (cross-directory Stage-1 "
                f"sharing is a CV-only feature)."
            )

    # Sanitize --output-suffix: only allow alphanumeric, underscore, hyphen, dot.
    if args.output_suffix is not None:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "_", args.output_suffix)
        if sanitized != args.output_suffix:
            logger.warning(
                f"--output-suffix sanitized: '{args.output_suffix}' -> '{sanitized}' "
                f"(only alphanumeric, underscore, hyphen, and dot are allowed)"
            )
            args.output_suffix = sanitized
        if not sanitized:
            parser.error("--output-suffix must not be empty after sanitization.")

    # --- Aggregation strategy CLI validations ---
    tuning_enabled = args.aggregation_strategy == "auto_tuned"

    # Resolve to AggregationStrategy for validation only (train_all_folds re-resolves)
    if args.aggregation_strategy in ("auto_tuned", "paper_best"):
        _agg_strategy_for_validation = None
    else:
        _agg_strategy_for_validation = AggregationStrategy[args.aggregation_strategy]

    if tuning_enabled:
        if args.entropy_max_fraction is not None:
            parser.error(
                "--entropy-max-fraction cannot be used with "
                "--aggregation-strategy auto_tuned (the threshold is selected "
                "automatically). Use a fixed strategy like entropy_cutoff instead."
            )
        if args.entropy_bottom_percentile is not None:
            parser.error(
                "--entropy-bottom-percentile cannot be used with "
                "--aggregation-strategy auto_tuned (the threshold is selected "
                "automatically). Use a fixed strategy like "
                "entropy_percentile_cutoff instead."
            )

    if args.entropy_max_fraction is not None and _agg_strategy_for_validation != AggregationStrategy.entropy_cutoff:
        hint = ""
        if _agg_strategy_for_validation is None:
            hint = (
                " Note: 'paper_best' resolves to entropy_cutoff for TCR, but to "
                "use a custom fraction you must specify "
                "--aggregation-strategy entropy_cutoff explicitly."
            )
        parser.error(
            f"--entropy-max-fraction is only used with "
            f"--aggregation-strategy entropy_cutoff.{hint}"
        )

    if args.entropy_bottom_percentile is not None and _agg_strategy_for_validation != AggregationStrategy.entropy_percentile_cutoff:
        parser.error(
            "--entropy-bottom-percentile is only used with "
            "--aggregation-strategy entropy_percentile_cutoff."
        )

    _tuning_flags_used = any([
        args.tuning_strategies is not None,
        args.tuning_cv_splits != 3,
        args.tuning_entropy_max_fractions is not None,
        args.tuning_entropy_percentiles is not None,
    ])
    if _tuning_flags_used and not tuning_enabled:
        parser.error(
            "--tuning-* flags are only valid with "
            "--aggregation-strategy auto_tuned."
        )

    # Parse comma-separated tuning grids into lists for train_all_folds()
    tuning_strategies = None
    if args.tuning_strategies is not None:
        tuning_strategies = [s.strip() for s in args.tuning_strategies.split(",") if s.strip()]
        if not tuning_strategies:
            parser.error("--tuning-strategies is empty after parsing.")
        valid_names = {s.name for s in AggregationStrategy}
        bad = [s for s in tuning_strategies if s not in valid_names]
        if bad:
            parser.error(
                f"Unknown --tuning-strategies: {bad}. "
                f"Valid names: {sorted(valid_names)}"
            )
    tuning_entropy_max_fractions = None
    if args.tuning_entropy_max_fractions is not None:
        try:
            tuning_entropy_max_fractions = [
                float(v.strip()) for v in args.tuning_entropy_max_fractions.split(",") if v.strip()
            ]
        except ValueError as e:
            parser.error(f"--tuning-entropy-max-fractions contains non-numeric values: {e}")
        if not tuning_entropy_max_fractions:
            parser.error("--tuning-entropy-max-fractions is empty after parsing.")
    tuning_entropy_percentiles = None
    if args.tuning_entropy_percentiles is not None:
        try:
            tuning_entropy_percentiles = [
                float(v.strip()) for v in args.tuning_entropy_percentiles.split(",") if v.strip()
            ]
        except ValueError as e:
            parser.error(f"--tuning-entropy-percentiles contains non-numeric values: {e}")
        if not tuning_entropy_percentiles:
            parser.error("--tuning-entropy-percentiles is empty after parsing.")

    # --- Validate training parameter ranges ---
    validate_training_params(
        aggregation_strategy=args.aggregation_strategy,
        n_estimators_stage1=args.n_estimators_stage1,
        n_estimators_stage2=args.n_estimators_stage2,
        entropy_max_fraction=args.entropy_max_fraction,
        entropy_bottom_percentile=args.entropy_bottom_percentile,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_strategies=tuning_strategies,
        tuning_entropy_max_fractions=tuning_entropy_max_fractions,
        tuning_entropy_percentiles=tuning_entropy_percentiles,
    )

    # --- Resolve base output dir early so the log file handler captures everything ---
    base_dir = args.output_dir or get_model_output_dir(
        model_name=MODEL_NAME,
        dataset_name=args.dataset_name,
        classification_mode=args.classification_mode,
        gene_locus=args.gene_locus,
        training_context=args.training_context,
        output_suffix=args.output_suffix,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # --- Train (summary JSON, RESULTS.md, and per-pair results are written
    #     inside the entry function). Dispatch on the training context:
    #     train-all → train_full_dataset (whole dataset, no CV, no eval);
    #     CV        → train_all_folds (fold loop + evaluation). ---
    try:
        if is_train_all:
            all_results = train_full_dataset(
                metadata_path=args.metadata_path,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                classification_mode=args.classification_mode,
                reference_class=args.reference_class,
                diseases=args.diseases,
                gene_locus=args.gene_locus,
                aggregation_strategy=args.aggregation_strategy,
                entropy_max_fraction=args.entropy_max_fraction,
                entropy_bottom_percentile=args.entropy_bottom_percentile,
                n_estimators_stage1=args.n_estimators_stage1,
                n_estimators_stage2=args.n_estimators_stage2,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                embedding_dir=args.embedding_dir,
                cache_embeddings=not args.no_cache_embeddings,
                device=args.device,
                embedding_batch_size=args.embedding_batch_size,
                data_dir=args.data_dir,
                cache_dir=cache_dir,
                gene_reference_path=args.gene_reference_path,
                output_suffix=args.output_suffix,
                training_context=args.training_context,
                resume=args.resume,
                resume_from_stage2=args.resume_from_stage2,
                tuning_cv_splits=args.tuning_cv_splits,
                tuning_strategies=tuning_strategies,
                tuning_entropy_max_fractions=tuning_entropy_max_fractions,
                tuning_entropy_percentiles=tuning_entropy_percentiles,
                clone_id_kwargs=get_clone_id_kwargs(args),
            )
            # Train-all produces no evaluation metrics — log what was trained per
            # pair/model so the console output summarizes the outcome. Model 3
            # trains Stage 1 on train_smaller1 (ts1) and Stage 2 on train_smaller2
            # (ts2); "aggregation" is the effective strategy (the tuned winner when
            # --aggregation-strategy auto_tuned was used).
            logger.info("\n--- Train-all summary (no evaluation) ---")
            for pair_key, pair_data in all_results.items():
                for info in pair_data["fold_results"]:
                    logger.info(
                        f"  {pair_key}: "
                        f"ts1={info['n_train_ts1_participants']} participants / "
                        f"{info['n_train_ts1_specimens']} specimens / "
                        f"{info['n_train_ts1_sequences']:,} sequences; "
                        f"ts2={info['n_train_ts2_participants']} participants / "
                        f"{info['n_train_ts2_specimens']} specimens / "
                        f"{info['n_train_ts2_sequences']:,} sequences; "
                        f"aggregation={info['aggregation_strategy']}; "
                        f"stage1_groups={info['n_stage1_groups']}; "
                        f"classes={info['classes']}"
                    )
            logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            train_all_folds(
                fold_ids=args.fold_ids,
                metadata_path=args.metadata_path,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                classification_mode=args.classification_mode,
                reference_class=args.reference_class,
                diseases=args.diseases,
                gene_locus=args.gene_locus,
                aggregation_strategy=args.aggregation_strategy,
                entropy_max_fraction=args.entropy_max_fraction,
                entropy_bottom_percentile=args.entropy_bottom_percentile,
                n_estimators_stage1=args.n_estimators_stage1,
                n_estimators_stage2=args.n_estimators_stage2,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                embedding_dir=args.embedding_dir,
                cache_embeddings=not args.no_cache_embeddings,
                device=args.device,
                embedding_batch_size=args.embedding_batch_size,
                data_dir=args.data_dir,
                cache_dir=cache_dir,
                gene_reference_path=args.gene_reference_path,
                output_suffix=args.output_suffix,
                training_context=args.training_context,
                resume=args.resume,
                resume_from_stage2=args.resume_from_stage2,
                resume_from_evaluation=args.resume_from_evaluation,
                stage1_dir=args.stage1_dir,
                tuning_cv_splits=args.tuning_cv_splits,
                tuning_strategies=tuning_strategies,
                tuning_entropy_max_fractions=tuning_entropy_max_fractions,
                tuning_entropy_percentiles=tuning_entropy_percentiles,
                clone_id_kwargs=get_clone_id_kwargs(args),
            )
    finally:
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
