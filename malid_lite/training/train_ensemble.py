"""Ensemble (metamodel) training for Mal-ID-Lite.

Trains a ridge-regularized logistic regression metamodel on the predictions of
three base models (Model 1: repertoire stats, Model 2: convergent clusters,
Model 3: sequence-level). The metamodel learns how to combine base model outputs
for the final disease classification.

Training contexts (--training-context, REQUIRED)
------------------------------------------------
cv (→ cv_ensemble)
    Cross-validation ensemble. For each outer CV fold the metamodel is trained
    on the validation third and evaluated on the held-out test fold → reports
    test metrics. This is the standard evaluation workflow.

train_all (→ train_all_ensemble)
    Whole-dataset ensemble for later evaluation on a SEPARATE dataset. There is
    NO test fold: the metamodel is trained on base-model predictions over the
    validation third and saved (no metrics). Evaluate it later on another dataset
    with the external-evaluation workflow. Base models are auto-trained in the
    train_all_ensemble context (on ts1+ts2, validation excluded → the metamodel
    trains on base-model out-of-sample predictions, exactly as in CV).

The short CLI values map internally to the context names above. The flag is
REQUIRED (no default): CV-evaluation and train-all-for-external-eval are very
different jobs, so intent must be explicit.

Architecture (cv)
-----------------
For each outer CV fold (0, 1, 2), that fold's participants are the held-out test
set; all other folds are pooled as the train pool for that pass (fold 0 as test
means folds 1+2 are pooled as training data, etc.):
  1. Load the cv_ensemble split: test / validation / train_smaller1 / train_smaller2
  2. Load pre-trained base model artifacts (trained on train_smaller)
  3. Get base model predictions on validation specimens
  4. Build metamodel feature matrix from those predictions
  5. Train ridge meta-learner (GlmnetLogitNetWrapper, alpha=0.0, MCC scoring)
  6. Get base model predictions on test specimens
  7. Predict with metamodel, evaluate ensemble + all base models

Architecture (train_all)
------------------------
A single whole-dataset pass — steps 1-5 above KEPT (metamodel trained on the
validation third, loading the WHOLE dataset via get_all_data), steps 6-7 DROPPED
(no test → no metrics/predictions). One metamodel per pair.

Artifacts per fold (cv)
-----------------------
  fold_<id>_ridge_cv_metamodel.joblib   — fitted Pipeline
  fold_<id>_metamodel_config.json        — feature columns, classes, config
  fold_<id>_ensemble_results.json        — per-fold metrics + abstention details
  fold_<id>_feature_matrix_val.csv       — validation feature matrix with labels
  fold_<id>_feature_matrix_test.csv      — test feature matrix with labels

Artifacts (train_all) — no fold prefix, no test/metrics
-------------------------------------------------------
  ridge_cv_metamodel.joblib   — fitted Pipeline
  metamodel_config.json        — feature columns, classes, λ, validation counts
  ensemble_results.json        — validation abstention/fill details (no metrics)
  feature_matrix_val.csv        — validation feature matrix with labels
  feature_matrix_raw_val.csv    — raw (pre-fill) validation matrix
  summary_<timestamp>.json      — no-metrics training summary; carries
      training_complete: True (the "ready for inference" marker the external-eval
      workflow checks) + the inference config (base-model dirs, feature-column
      order, abstention strategy).

Classification modes
--------------------
multiclass
    A single N-class ensemble. Default.

binary
    One ensemble for a single disease-vs-reference pair.
    Requires --reference-class. For 2-class datasets, the non-reference
    disease is auto-detected. For N-class datasets, use --diseases <disease>
    to pick one.

multi-binary
    One independent binary ensemble per disease vs. the reference class.
    Default (no --diseases): trains all N-1 non-reference diseases.
    With --diseases <d1> <d2> ...: trains only the specified subset.
    Requires --reference-class when data has more than 2 classes.

Usage
-----
    # NOTE: --training-context is REQUIRED (cv or train_all).

    # Cross-validation ensemble: all 3 models, multiclass, all folds
    python malid_lite/training/train_ensemble.py \\
        --training-context cv \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data

    # Train-all ensemble: train base models + metamodel on one whole dataset,
    # to be evaluated later on a SEPARATE dataset (no test set, no metrics)
    python malid_lite/training/train_ensemble.py \\
        --training-context train_all \\
        --metadata-path cache/train-dataset/metadata.tsv \\
        --cache-dir cache/train-dataset

    # Only Models 1 and 3
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --models 1 3

    # Binary (2-class data, auto-detects disease)
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --classification-mode binary --reference-class "Healthy/Background"

    # Binary (N-class data, pick one disease)
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --classification-mode binary --diseases Covid19 \\
        --reference-class "Healthy/Background"

    # Multi-binary: one ensemble per disease vs reference
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --classification-mode multi-binary \\
        --reference-class "Healthy/Background"

    # With custom model suffixes (for different training variants)
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --model3-suffix pct_0-1

    # Fill Model 2 abstentions with 0.5 (instead of dropping specimens)
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --model2-abstention-strategy fill_0.5

    # Fill Model 2 abstentions with mean of Models 1 and 3 predictions
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --model2-abstention-strategy fill_models13_mean

Auto-training
-------------
Base models are automatically trained if their artifacts are not found. The
script detects per-model state (LOAD / TRAIN / RESUME) and dispatches to each
model's CV trainer train_all_folds() (context cv) or whole-dataset trainer
train_full_dataset() (context train_all) as needed. Use --retrain-base-models or
--retrain-models to force retraining even when artifacts exist. Training params
can be customized via --model1-*, --model2-*, --model3-* CLI flags.

Clone ID parameters
-------------------
All training scripts accept clone_id flags (--force-clone-id, --clone-id-use-aa,
--clone-id-identity-threshold, --clone-id-linkage-method). These only need to be
specified when building the cache for the first time. On subsequent runs, omitting
them is fine -- the cached values are accepted as-is. If you explicitly specify a
value that conflicts with the cache, the run fails immediately with a clear error.
See PIPELINE_GUIDE.md > Clone ID Computation for details.

Example with clone_id::

    python malid_lite/training/train_ensemble.py \\
        --data-dir /path/to/data --metadata-path /path/to/metadata.tsv \\
        --cache-dir cache/my-dataset \\
        --force-clone-id --clone-id-use-aa
"""

import argparse
import dataclasses
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss as sklearn_log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.dataloader import (
    MalIDPublishedDataLoader,
    add_clone_id_args,
    get_clone_id_kwargs,
)
from malid_lite.dataloader.base import PreprocessingStage
from malid_lite.models.model1_repertoire import (
    V_GENE_COL,
    RepertoireClassifier,
)
from malid_lite.models.model2_convergent_clusters import (
    BEST_MODEL_FOR_METAMODEL,
    SEQUENCE_IDENTITY_THRESHOLDS,
    featurize,
    get_artifact_paths,
    get_no_valid_clusters_path,
)
from malid_lite.models.model3_sequence_level import (
    AggregationStrategy,
)
from malid_lite.training.training_utils import (
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    aggregate_fold_results,
    cap_cv_splits_for_data,
    filter_to_binary_pair,
    get_dataset_disease_classes,
    get_dataset_fold_ids,
    get_ensemble_output_dir,
    get_metadata_class_counts,
    get_model_output_dir,
    make_pair_name,
    preflight_check_fold_artifacts,
    read_model_summary,
    resolve_model_artifact_dir,
    resolve_binary_disease,
    validate_mode_and_classes,
    validate_model_summary,
)
from malid_lite.utils import multiclass_metrics
from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper
from malid_lite.utils.markdown import pad_md_tables

logger = logging.getLogger(__name__)

TRAINING_CONTEXT = "cv_ensemble"

# Display names for feature column prefixes, matching original Mal-ID config
MODEL_DISPLAY_NAMES = {
    1: "repertoire_stats",
    2: "convergent_cluster_model",
    3: "sequence_model",
}

# Valid values for the model2_abstention_strategy parameter
MODEL2_ABSTENTION_STRATEGIES = ("ensemble_abstain", "fill_0.5", "fill_models13_mean")


# ====================================================================== #
# Argument validation                                                      #
# ====================================================================== #


def validate_ensemble_args(
    args: argparse.Namespace,
    retrain_set: set,
    cli_training_params: Dict[int, Dict[str, Any]],
    parser: argparse.ArgumentParser,
) -> None:
    """Validate all CLI arguments and their interactions.

    Called once at the beginning of main(), before any work starts.
    Raises parser.error() or ValueError for any invalid input.

    Checks:
    - Retrain/resume conflicts
    - output-suffix vs output-dir mutual exclusivity
    - Suffix sanitization (output-suffix, model1/2/3-suffix)
    - n_jobs range
    - diseases not supported in multiclass mode
    - metadata-path existence (if provided)
    - gene-reference-path existence (if provided)
    - Model-specific args scoped to included models
    - Model 3 cross-param interactions (tuning flags require auto_tuned,
      entropy params require matching strategy)
    - Per-model training param range validation (via model-specific validators)

    Parameters
    ----------
    args : Parsed CLI arguments.
    retrain_set : Set of model numbers flagged for retraining.
    cli_training_params : Per-model dict of CLI param name -> value.
    parser : ArgumentParser for parser.error() calls.
    """
    # --- Train-all context guards ---
    # A train-all ensemble is a single whole-dataset pass — there are no CV
    # folds, so --fold-ids is meaningless (and the dataset may not even have a
    # CV_fold column).
    is_train_all = args.training_context == "train_all"
    if is_train_all and args.fold_ids is not None:
        parser.error(
            "--fold-ids is not valid with --training-context train_all: a train-all "
            "ensemble trains a single whole-dataset pass with no CV folds. "
            "Use --training-context cv for per-fold cross-validation."
        )

    # --- Retrain / resume conflicts ---
    if args.resume and retrain_set:
        parser.error(
            "--resume and --retrain-base-models / --retrain-models are contradictory. "
            "--resume continues from partial work; retrain starts fresh."
        )

    if args.retrain_models:
        for m in args.retrain_models:
            if m not in args.models:
                parser.error(
                    f"--retrain-models includes Model {m} but --models does not "
                    f"include it. Either add {m} to --models or remove it from "
                    f"--retrain-models."
                )

    # --- output-suffix vs output-dir mutual exclusivity ---
    if args.output_dir is not None and args.output_suffix is not None:
        parser.error(
            "--output-dir and --output-suffix are mutually exclusive. "
            "Use --output-dir for a fully custom path, or --output-suffix "
            "to append a label to the canonical path."
        )

    # --- Sanitize suffixes (only allow alphanumeric, underscore, hyphen, dot) ---
    import re
    _suffix_attrs = [
        ("--output-suffix", "output_suffix"),
        ("--model1-suffix", "model1_suffix"),
        ("--model2-suffix", "model2_suffix"),
        ("--model3-suffix", "model3_suffix"),
    ]
    for flag_name, attr_name in _suffix_attrs:
        val = getattr(args, attr_name)
        if val is not None:
            sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "_", val)
            if sanitized != val:
                logger.warning(
                    f"{flag_name} sanitized: '{val}' -> '{sanitized}' "
                    f"(only alphanumeric, underscore, hyphen, and dot are allowed)"
                )
                setattr(args, attr_name, sanitized)
            if not sanitized:
                parser.error(f"{flag_name} must not be empty after sanitization.")

    # --- n_jobs range ---
    if args.n_jobs < 1:
        parser.error(f"--n-jobs must be >= 1, got {args.n_jobs}.")

    # --- metamodel CV splits range ---
    # StratifiedGroupKFold needs >= 2 folds (it is auto-capped DOWN per class at
    # fit time, but the requested value must be a valid starting point).
    if getattr(args, "metamodel_cv_n_splits", 5) < 2:
        parser.error(
            f"--metamodel-cv-n-splits must be >= 2, got {args.metamodel_cv_n_splits}."
        )

    # --- Model 2 abstention strategy validation ---
    # Skip when --feature-matrices-dir is set: models come from source config,
    # not --models. Strategy validation happens in _run_from_feature_matrices().
    fm_dir = getattr(args, "feature_matrices_dir", None)
    strategy = args.model2_abstention_strategy
    if fm_dir is None:
        if strategy == "fill_models13_mean":
            if 1 not in args.models or 3 not in args.models:
                parser.error(
                    f"--model2-abstention-strategy fill_models13_mean requires both "
                    f"Models 1 and 3 in --models, but --models is {args.models}. "
                    f"Use --model2-abstention-strategy fill_0.5 or ensemble_abstain instead."
                )
        if strategy != "ensemble_abstain" and 2 not in args.models:
            parser.error(
                f"--model2-abstention-strategy {strategy} only makes sense when Model 2 "
                f"is included in --models, but --models is {args.models}."
            )

    # --- feature-matrices-dir validation ---
    if fm_dir is not None:
        if not fm_dir.is_dir():
            parser.error(
                f"--feature-matrices-dir does not exist or is not a directory: {fm_dir}"
            )
        if args.resume:
            parser.error(
                "--feature-matrices-dir and --resume are mutually exclusive. "
                "--feature-matrices-dir loads feature matrices from an external "
                "directory; --resume loads from the current output directory."
            )
        if getattr(args, "retrain_models", None):
            parser.error(
                "--feature-matrices-dir and --retrain-models are mutually exclusive. "
                "--feature-matrices-dir skips all base model training."
            )
        if getattr(args, "retrain_base_models", False):
            parser.error(
                "--feature-matrices-dir and --retrain-base-models are mutually exclusive. "
                "--feature-matrices-dir skips all base model training."
            )
        # Require run_config.json in the source dir (directly, or in pair subdirs for multi-binary)
        src_config = fm_dir / "run_config.json"
        if not src_config.exists():
            pair_subdirs_with_config = [
                d for d in fm_dir.iterdir()
                if d.is_dir() and "_vs_" in d.name
                and (d / "run_config.json").exists()
            ]
            if not pair_subdirs_with_config:
                parser.error(
                    f"--feature-matrices-dir requires run_config.json in the source directory "
                    f"(or in pair subdirectories for multi-binary mode). "
                    f"Not found: {src_config}"
                )

    # --- diseases not supported in multiclass ---
    if args.classification_mode == "multiclass" and args.diseases is not None:
        parser.error(
            "--diseases is not supported in multiclass mode (all disease classes "
            "from the data are used). To train on a subset of diseases, use "
            "--classification-mode binary or --classification-mode multi-binary."
        )

    # --- metadata-path existence ---
    if args.metadata_path is not None and not args.metadata_path.exists():
        parser.error(
            f"--metadata-path does not exist: {args.metadata_path}"
        )

    # --- gene-reference-path existence ---
    if args.gene_reference_path is not None and not args.gene_reference_path.exists():
        parser.error(
            f"--gene-reference-path does not exist: {args.gene_reference_path}"
        )

    # --- Model-specific args must be for included models ---
    # Training params
    for num in (1, 2, 3):
        if num not in args.models:
            has_non_none = any(
                v is not None for v in cli_training_params[num].values()
            )
            if has_non_none:
                specified = [
                    k for k, v in cli_training_params[num].items()
                    if v is not None
                ]
                parser.error(
                    f"Training params specified for Model {num} ({specified}) but "
                    f"Model {num} is not in --models {args.models}. Either add {num} "
                    f"to --models or remove the Model {num} params."
                )

    # Model-specific suffixes
    _suffix_args = {
        1: ("--model1-suffix", args.model1_suffix),
        2: ("--model2-suffix", args.model2_suffix),
        3: ("--model3-suffix", args.model3_suffix),
    }
    for num, (arg_name, val) in _suffix_args.items():
        if val is not None and num not in args.models:
            parser.error(
                f"{arg_name} is set but Model {num} is not in --models {args.models}. "
                f"Either add {num} to --models or remove {arg_name}."
            )

    # Model 3 embedding/device args
    _m3_infra_args = {
        "--model3-embedding-dir": args.model3_embedding_dir,
        "--model3-no-cache-embeddings": args.model3_no_cache_embeddings or None,
        "--model3-device": args.model3_device,
        "--model3-embedding-batch-size": args.model3_embedding_batch_size,
    }
    if 3 not in args.models:
        has_m3_infra = any(v is not None for v in _m3_infra_args.values())
        if has_m3_infra:
            specified = [k for k, v in _m3_infra_args.items() if v is not None]
            parser.error(
                f"Model 3 args specified ({specified}) but Model 3 is not in "
                f"--models {args.models}. Either add 3 to --models or remove "
                f"the Model 3 args."
            )

    # --- Model 3 cross-param interaction checks ---
    # These mirror the standalone model3 main() checks.  In the ensemble,
    # --model3-aggregation-strategy defaults to None (= use model default,
    # which is entropy_percentile_cutoff).
    if 3 in args.models:
        m3 = cli_training_params[3]
        m3_agg = m3.get("aggregation_strategy")  # None when not specified

        # --- Tuning flags require explicit auto_tuned ---
        _tuning_flags_used = any([
            m3.get("tuning_strategies") is not None,
            m3.get("tuning_cv_splits") is not None,
            m3.get("tuning_entropy_max_fractions") is not None,
            m3.get("tuning_entropy_percentiles") is not None,
        ])

        if m3_agg != "auto_tuned" and _tuning_flags_used:
            if m3_agg is None:
                parser.error(
                    "--model3-tuning-* flags require "
                    "--model3-aggregation-strategy auto_tuned. The default "
                    "strategy is entropy_percentile_cutoff (not auto_tuned). "
                    "Either remove the tuning flags or set "
                    "--model3-aggregation-strategy auto_tuned."
                )
            else:
                parser.error(
                    f"--model3-tuning-* flags are only valid with "
                    f"--model3-aggregation-strategy auto_tuned, but got "
                    f"'{m3_agg}'. Either remove the tuning flags or set "
                    f"--model3-aggregation-strategy auto_tuned."
                )

        # --- Entropy param / strategy cross-validation ---
        # Resolve effective strategy for entropy param checks: None → default
        _eff_agg = m3_agg if m3_agg is not None else "entropy_percentile_cutoff"

        if _eff_agg == "auto_tuned":
            # auto_tuned selects thresholds automatically — fixed entropy
            # params conflict with the tuning process
            if m3.get("entropy_max_fraction") is not None:
                parser.error(
                    "--model3-entropy-max-fraction cannot be used with "
                    "--model3-aggregation-strategy auto_tuned (the threshold "
                    "is selected automatically). Use a fixed strategy like "
                    "entropy_cutoff instead."
                )
            if m3.get("entropy_bottom_percentile") is not None:
                parser.error(
                    "--model3-entropy-bottom-percentile cannot be used with "
                    "--model3-aggregation-strategy auto_tuned (the threshold "
                    "is selected automatically). Use a fixed strategy like "
                    "entropy_percentile_cutoff instead."
                )
        else:
            # Fixed strategy (explicit or default entropy_percentile_cutoff)
            if m3.get("entropy_max_fraction") is not None and _eff_agg != "entropy_cutoff":
                _strategy_note = (
                    " The default strategy is entropy_percentile_cutoff."
                    if m3_agg is None else ""
                )
                parser.error(
                    f"--model3-entropy-max-fraction is only used with "
                    f"--model3-aggregation-strategy entropy_cutoff, but got "
                    f"'{_eff_agg}'.{_strategy_note}"
                )

            if m3.get("entropy_bottom_percentile") is not None and _eff_agg != "entropy_percentile_cutoff":
                parser.error(
                    f"--model3-entropy-bottom-percentile is only used with "
                    f"--model3-aggregation-strategy entropy_percentile_cutoff, "
                    f"but got '{_eff_agg}'."
                )

    # --- Per-model training param range validation ---
    # Import lazily to avoid circular imports at module level
    for num in args.models:
        params = cli_training_params[num]
        has_non_none = any(v is not None for v in params.values())
        if not has_non_none:
            continue  # all defaults, nothing to validate

        if num == 1:
            from malid_lite.training.train_model1 import (
                validate_training_params as validate_m1,
            )
            validate_m1(**params)
        elif num == 2:
            from malid_lite.training.train_model2 import (
                validate_training_params as validate_m2,
            )
            validate_m2(**params)
        elif num == 3:
            from malid_lite.training.train_model3 import (
                validate_training_params as validate_m3,
            )
            validate_m3(**params)


def _validate_cross_model_disease_classes(
    summaries: Dict[int, Optional[dict]],
    label: str = "base models",
) -> None:
    """Validate that all models with summaries agree on training classes.

    Reads the 'model_classes' key from each summary (the effective classes
    the model was trained on) and raises ValueError if any two models
    have different sets.

    Parameters
    ----------
    summaries : {model_number: summary_dict_or_None}.
        None entries and summaries without 'model_classes' are skipped.
    label : Descriptive label for error messages (e.g. "LOAD models").
    """
    class_sets: Dict[int, set] = {}
    for num, bm_summary in summaries.items():
        if bm_summary is None:
            continue
        classes = bm_summary.get("model_classes")
        if classes is not None:
            class_sets[num] = set(classes)

    if len(class_sets) > 1:
        reference_num = next(iter(class_sets))
        reference_set = class_sets[reference_num]
        for num, cls_set in class_sets.items():
            if cls_set != reference_set:
                raise ValueError(
                    f"Model class mismatch between {label}: "
                    f"Model {reference_num} has {sorted(reference_set)}, "
                    f"Model {num} has {sorted(cls_set)}. "
                    f"All base models must be trained on the same classes."
                )


# ====================================================================== #
# Auto-training: mode detection and parameter comparison                  #
# ====================================================================== #

# Mapping from CLI arg name to summary JSON key, per model.
# Only these keys are compared when validating loaded models.
_PARAM_COMPARISON_KEYS = {
    1: {
        "n_pcs": "n_pcs",
        "l1_ratio": "l1_ratio",
        "model_name": "model_names",  # single str vs 1-element list; handled specially
    },
    2: {
        "p_values": "p_values",  # compared as sorted lists
        "retrain_on_full_train": "retrain_on_full_train",
        "sequence_identity_threshold": "sequence_identity_threshold",
    },
    3: {
        "aggregation_strategy": "aggregation_strategy",
        "n_estimators_stage1": "n_estimators_stage1",
        "n_estimators_stage2": "n_estimators_stage2",
        "tuning_cv_splits": "tuning_cv_splits",
        "tuning_strategies": "tuning_strategies",  # compared as sorted lists
        "tuning_entropy_max_fractions": "tuning_entropy_max_fractions",  # sorted
        "tuning_entropy_percentiles": "tuning_entropy_percentiles",  # sorted
        "entropy_max_fraction": "entropy_max_fraction",
        "entropy_bottom_percentile": "entropy_bottom_percentile",
    },
}


def compare_training_params(
    model_num: int,
    summary: dict,
    cli_params: Dict[str, Any],
) -> List[Tuple[str, Any, Any]]:
    """Compare CLI-specified training params against a loaded model's summary.

    Only non-None CLI params are compared (None = not specified by user).

    Parameters
    ----------
    model_num : Base model number (1, 2, or 3).
    summary   : Loaded summary dict from read_model_summary().
    cli_params : Dict of CLI param name → value. Only non-None entries
        are compared against the summary.

    Returns
    -------
    List of (param_name, summary_value, cli_value) tuples for each mismatch.
    Empty list means all specified params match.
    """
    if model_num not in _PARAM_COMPARISON_KEYS:
        raise ValueError(f"Unknown model_num: {model_num}. Expected 1, 2, or 3.")

    key_map = _PARAM_COMPARISON_KEYS[model_num]
    mismatches = []

    for cli_key, cli_val in cli_params.items():
        if cli_val is None:
            continue  # user didn't specify → no comparison

        if cli_key not in key_map:
            continue  # not a compared param

        summary_key = key_map[cli_key]

        # Special case: Model 1 model_name is stored as model_names (1-element list)
        if model_num == 1 and cli_key == "model_name":
            summary_val = summary.get("model_names")
            if isinstance(summary_val, list) and len(summary_val) == 1:
                summary_val = summary_val[0]
            if summary_val != cli_val:
                mismatches.append((cli_key, summary_val, cli_val))
            continue

        summary_val = summary.get(summary_key)

        # Skip if key is absent from summary (e.g., older summary format)
        if summary_key not in summary:
            continue

        # List params: compare as sorted to be order-independent
        if isinstance(cli_val, list) and isinstance(summary_val, list):
            if sorted(cli_val) != sorted(summary_val):
                mismatches.append((cli_key, summary_val, cli_val))
        elif summary_val != cli_val:
            mismatches.append((cli_key, summary_val, cli_val))

    return mismatches


def _has_partial_base_artifacts(model_dir: Path, training_context: str) -> bool:
    """True if the dir holds partial (non-summary) base-model artifacts.

    Used to distinguish RESUME (partial artifacts present, no summary yet) from
    a fresh TRAIN. CV artifacts are fold-prefixed (``fold_*``); train-all
    artifacts have no prefix, so any model pickle/joblib (but not the
    ``summary_*.json`` / ``run_config`` bookkeeping) counts as partial.
    """
    if not model_dir.exists():
        return False
    if training_context == "cv_ensemble":
        return any(model_dir.glob("fold_*"))
    # train_all_ensemble: no fold prefix — look for any model artifact file.
    for pattern in ("*.pkl", "*.joblib", "*_v_genes.json"):
        if any(model_dir.glob(pattern)):
            return True
    return False


def resolve_base_model_mode(
    model_num: int,
    retrain_set: set,
    resume_flag: bool,
    dataset_name: str,
    classification_mode: str,
    gene_locus: str,
    output_suffix: Optional[str],
    cli_training_params: Dict[str, Any],
    training_context: str = "cv_ensemble",
) -> Tuple[str, Path, Optional[dict]]:
    """Determine the mode for a base model: LOAD, TRAIN, or RESUME.

    Parameters
    ----------
    model_num : Base model number (1, 2, or 3).
    retrain_set : Set of model numbers that should be retrained.
    resume_flag : Whether --resume is active.
    dataset_name : Dataset identifier.
    classification_mode : "multiclass", "binary", or "multi-binary".
    gene_locus : "TCR" or "BCR".
    output_suffix : Model-specific suffix (from --modelN-suffix), or None.
    cli_training_params : Dict of CLI param name → value for this model.
    training_context : "cv_ensemble" (fold-prefixed base artifacts) or
        "train_all_ensemble" (whole-dataset, prefix-less base artifacts).
        Selects the artifact directory AND the partial-artifact detection.

    Returns
    -------
    (mode, artifact_dir, summary_or_none)
        mode : "LOAD", "TRAIN", or "RESUME"
        artifact_dir : Path to the model's artifact directory (may not exist yet
            for TRAIN mode)
        summary_or_none : Loaded summary dict for LOAD mode, None for TRAIN/RESUME.
    """
    if model_num in retrain_set:
        # Retrain requested — resolve target directory path
        target_dir = get_model_output_dir(
            model_name=f"model{model_num}",
            dataset_name=dataset_name,
            classification_mode=classification_mode,
            gene_locus=gene_locus,
            training_context=training_context,
            output_suffix=output_suffix,
        )
        return "TRAIN", target_dir, None

    # Try to find existing artifacts via resolve_model_artifact_dir().
    # That function only succeeds if a directory WITH summary_*.json is found.
    # If it raises, we still need to check for partial artifacts (for RESUME).
    target_dir = get_model_output_dir(
        model_name=f"model{model_num}",
        dataset_name=dataset_name,
        classification_mode=classification_mode,
        gene_locus=gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )

    try:
        resolved_dir, _suffix = resolve_model_artifact_dir(
            model_name=f"model{model_num}",
            dataset_name=dataset_name,
            classification_mode=classification_mode,
            gene_locus=gene_locus,
            training_context=training_context,
            output_suffix=output_suffix,
        )
    except ValueError:
        # Multiple suffixed directories found — ambiguity must be resolved by user
        raise
    except FileNotFoundError:
        # No complete (summary-bearing) artifacts found.
        # Check if the target dir has partial artifacts (for RESUME).
        if _has_partial_base_artifacts(target_dir, training_context):
            if resume_flag:
                return "RESUME", target_dir, None
            else:
                return "TRAIN", target_dir, None
        return "TRAIN", target_dir, None

    # resolve_model_artifact_dir succeeded → directory usually has summary_*.json.
    # Exception: explicit --modelN-suffix pointing to a dir that exists but has
    # no summary (Case 1 in resolve_model_artifact_dir doesn't check for summary).
    has_summary = any(resolved_dir.glob("summary_*.json"))

    if has_summary:
        # Fully trained — validate params match
        summary = read_model_summary(resolved_dir)
        mismatches = compare_training_params(model_num, summary, cli_training_params)
        if mismatches:
            mismatch_lines = "\n".join(
                f"  {k}: loaded={v_loaded!r}, CLI={v_cli!r}"
                for k, v_loaded, v_cli in mismatches
            )
            raise ValueError(
                f"Model {model_num} loaded from {resolved_dir.name} but CLI args "
                f"specify different training parameters:\n{mismatch_lines}\n"
                f"Either remove the conflicting CLI args or use "
                f"--retrain-models {model_num} to retrain with the new parameters."
            )
        return "LOAD", resolved_dir, summary

    # No summary — check for partial artifacts (fold-prefixed for CV,
    # prefix-less for train-all)
    has_partial = _has_partial_base_artifacts(resolved_dir, training_context)
    if has_partial and resume_flag:
        return "RESUME", resolved_dir, None
    elif has_partial and not resume_flag:
        # Partial artifacts without --resume → fresh start (overwrites)
        return "TRAIN", resolved_dir, None
    else:
        # Empty directory → train from scratch
        return "TRAIN", resolved_dir, None


def _read_fold_meta(model_dir: Path) -> Optional[dict]:
    """Read _meta from the first available fold artifact for pre-flight validation.

    Used in RESUME mode to compare the saved training params from a partially-
    completed run against current CLI params before training begins.

    All three models store _meta as a top-level key in their prediction pkl
    files: {"_meta": {"model_params": {...}, ...}, ...}. Training params
    are nested under _meta["model_params"].

    Returns None if no fold artifact with _meta is found (e.g., old artifact
    format without _meta or corrupted files).
    """
    # Try the standard prediction artifact patterns
    patterns = [
        "fold_*_predictions.pkl",  # Model 2, Model 3
        "fold_*_*.pkl",  # Model 1 (fold_<id>_lasso_cv.pkl)
    ]

    for pattern in patterns:
        for pkl_path in sorted(model_dir.glob(pattern)):
            try:
                obj = joblib.load(pkl_path)
                if hasattr(obj, "_meta"):
                    return obj._meta
                if isinstance(obj, dict) and "_meta" in obj:
                    return obj["_meta"]
            except Exception:
                continue

    return None


def preflight_validate_resume_params(
    model_num: int,
    model_dir: Path,
    cli_training_params: Dict[str, Any],
) -> None:
    """Pre-flight validation for RESUME mode: compare saved _meta against CLI params.

    Reads _meta from an existing fold artifact and compares each non-None CLI
    param against the saved value. Raises ValueError on mismatch so training
    does not start with incompatible parameters.

    Does nothing (with a warning) if no _meta is found in fold artifacts.
    """
    meta = _read_fold_meta(model_dir)

    if meta is None:
        logger.warning(
            f"  Model {model_num}: no _meta found in fold artifacts at {model_dir}. "
            f"Cannot validate resume parameters (older artifact format). Proceeding."
        )
        return

    # Training params are nested under _meta["model_params"]
    model_params = meta.get("model_params", {})
    if not model_params:
        logger.warning(
            f"  Model {model_num}: _meta has no 'model_params' in fold artifacts at "
            f"{model_dir}. Cannot validate resume parameters. Proceeding."
        )
        return

    # Compare non-None CLI params against saved model_params
    mismatches = []
    for cli_key, cli_val in cli_training_params.items():
        if cli_val is None:
            continue
        meta_val = model_params.get(cli_key)
        if meta_val is None:
            continue  # key not in model_params → can't compare

        # List comparison: order-independent
        if isinstance(cli_val, list) and isinstance(meta_val, list):
            if sorted(cli_val) != sorted(meta_val):
                mismatches.append((cli_key, meta_val, cli_val))
        elif meta_val != cli_val:
            mismatches.append((cli_key, meta_val, cli_val))

    if mismatches:
        mismatch_lines = "\n".join(
            f"  {k}: saved={v_saved!r}, CLI={v_cli!r}"
            for k, v_saved, v_cli in mismatches
        )
        raise ValueError(
            f"Model {model_num} RESUME: partially-trained artifacts in {model_dir.name} "
            f"were trained with different parameters than current CLI "
            f"(from _meta.model_params):\n{mismatch_lines}\n"
            f"Either match the original parameters, use --retrain-models {model_num} "
            f"to start fresh, or delete the partial artifacts manually."
        )


def _log_base_model_status_table(
    model_modes: Dict[int, str],
    model_dirs: Dict[int, Path],
    model_summaries: Dict[int, Optional[dict]],
    training_context: str = "cv_ensemble",
) -> None:
    """Log the base model status table after mode detection.

    Parameters
    ----------
    model_modes : {model_number: "LOAD" | "TRAIN" | "RESUME"}.
    model_dirs : {model_number: Path to model artifact directory}.
    model_summaries : {model_number: summary dict or None}.
    training_context : "cv_ensemble" or "train_all_ensemble" — selects how the
        RESUME row is described (per-fold list vs single whole-dataset pass).
    """
    is_train_all = training_context == "train_all_ensemble"
    logger.info("")
    logger.info("=" * 70)
    logger.info("BASE MODEL STATUS")
    logger.info("=" * 70)

    for num in sorted(model_modes.keys()):
        mode = model_modes[num]
        d = model_dirs[num]
        summary = model_summaries.get(num)

        if mode == "LOAD":
            ts = summary.get("timestamp", "?") if summary else "?"
            logger.info(f"  Model {num}: LOAD    {d}")
            logger.info(f"            Created: {ts}")
        elif mode == "RESUME":
            logger.info(f"  Model {num}: RESUME  {d}")
            if is_train_all:
                # Train-all is a single whole-dataset pass (no folds); the
                # resume sentinel is meta.json (present only when complete).
                logger.info(
                    f"            Whole-dataset pass "
                    f"(partial artifacts present, meta.json not yet written)"
                )
            else:
                # Count completed fold artifacts
                fold_dirs = sorted(d.glob("fold_*"))
                completed_ids = sorted({
                    p.stem.split("_")[1]
                    for p in fold_dirs
                    if p.stem.startswith("fold_") and p.stem.split("_")[1].isdigit()
                })
                logger.info(f"            Folds with artifacts: {completed_ids}")
        else:
            exists = d.exists()
            logger.info(
                f"  Model {num}: TRAIN   "
                f"{'(will overwrite ' + str(d) + ')' if exists else '(no artifacts at ' + str(d) + ')'}"
            )

    logger.info("=" * 70)
    logger.info("")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string (e.g. '12m 45s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m"


def auto_train_base_model(
    model_num: int,
    training_params: Dict[str, Any],
    output_dir: Path,
    metadata_path: Path,
    dataset_name: str,
    classification_mode: str,
    reference_class: Optional[str],
    diseases: Optional[List[str]],
    gene_locus: str,
    fold_ids: List[int],
    data_dir: Optional[Path],
    cache_dir: Optional[Path],
    gene_reference_path: Optional[Path],
    n_jobs: int,
    verbose: int,
    resume: bool,
    # Model 3 specific
    embedding_dir: Optional[Path] = None,
    no_cache_embeddings: bool = False,
    device: Optional[str] = None,
    embedding_batch_size: Optional[int] = None,
    # Clone ID (all models)
    clone_id_kwargs: Optional[Dict] = None,
    training_context: str = "cv_ensemble",
) -> None:
    """Train a base model by dispatching to its CV or train-all entry point.

    Imports the model's training module and calls, depending on
    ``training_context``:
    - ``cv_ensemble`` → ``train_all_folds(fold_ids=..., ...)`` (per-fold CV).
    - ``train_all_ensemble`` → ``train_full_dataset(...)`` (single whole-dataset
      pass; NO ``fold_ids``; Model 3 keeps the embedding args but has no
      ``resume_from_evaluation``/``stage1_dir``).
    with:
    - Shared params (metadata_path, etc.) passed directly
    - Model-specific training params unpacked from training_params dict

    Only non-None values should be in training_params, so unspecified params
    fall through to train_all_folds() defaults.

    Parameters
    ----------
    model_num : Base model number (1, 2, or 3).
    training_params : Dict of non-None CLI training params for this model.
        Keys must match train_all_folds() parameter names exactly.
    output_dir : Base output directory for the model's artifacts.
    metadata_path : Path to the metadata TSV file.
    dataset_name : Dataset identifier.
    classification_mode : "multiclass", "binary", or "multi-binary".
    reference_class : Reference/negative class for binary/multi-binary modes.
    diseases : Subset of disease classes to train, or None for all.
    gene_locus : "TCR" or "BCR".
    fold_ids : List of fold IDs to train.
    data_dir : Path to raw data directory.
    cache_dir : Path to cache directory.
    gene_reference_path : Path to gene reference file.
    n_jobs : Parallel workers (Model 2 clustering, Model 3 V-gene groups).
    verbose : Verbosity level.
    resume : Whether to resume from partial artifacts (True for RESUME mode).
    embedding_dir : Model 3 only: directory with pre-computed ESM-2 embeddings.
    no_cache_embeddings : Model 3 only: if True, don't save newly computed
        embeddings to disk (cached embeddings are still used when available).
    device : Model 3 only: device for embedding computation.
    embedding_batch_size : Model 3 only: batch size for embedding computation.
    clone_id_kwargs : Dict of clone_id parameters for the data loader
        (from get_clone_id_kwargs). None means all params unspecified —
        cached values accepted as-is.

    Raises
    ------
    ValueError
        If model_num is not 1, 2, or 3.
    """
    if model_num not in (1, 2, 3):
        raise ValueError(f"Unknown model_num: {model_num}. Expected 1, 2, or 3.")
    if training_context not in ("cv_ensemble", "train_all_ensemble"):
        raise ValueError(
            f"training_context must be 'cv_ensemble' or 'train_all_ensemble', "
            f"got {training_context!r}."
        )
    is_train_all = training_context == "train_all_ensemble"

    # Shared kwargs passed to all models' entry point. For CV we pass fold_ids
    # to train_all_folds(); for train-all we call train_full_dataset(), which
    # has no fold_ids (single whole-dataset pass).
    shared_kwargs = dict(
        metadata_path=metadata_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        gene_locus=gene_locus,
        verbose=verbose,
        data_dir=data_dir,
        cache_dir=cache_dir,
        gene_reference_path=gene_reference_path,
        training_context=training_context,
        resume=resume,
        clone_id_kwargs=clone_id_kwargs,
    )
    if not is_train_all:
        shared_kwargs["fold_ids"] = fold_ids

    if model_num == 1:
        if is_train_all:
            from malid_lite.training.train_model1 import train_full_dataset as train_m1
        else:
            from malid_lite.training.train_model1 import train_all_folds as train_m1
        train_m1(**shared_kwargs, **training_params)

    elif model_num == 2:
        if is_train_all:
            from malid_lite.training.train_model2 import train_full_dataset as train_m2
        else:
            from malid_lite.training.train_model2 import train_all_folds as train_m2
        train_m2(**shared_kwargs, n_jobs=n_jobs, **training_params)

    elif model_num == 3:
        if is_train_all:
            from malid_lite.training.train_model3 import train_full_dataset as train_m3
        else:
            from malid_lite.training.train_model3 import train_all_folds as train_m3
        # Build Model 3 infra kwargs (only include if explicitly set).
        # train_full_dataset shares these args with train_all_folds (it only
        # drops fold_ids + the CV-only resume_from_evaluation/stage1_dir).
        m3_kwargs: Dict[str, Any] = dict(
            n_jobs=n_jobs,
            embedding_dir=embedding_dir,
            cache_embeddings=not no_cache_embeddings,
        )
        if device is not None:
            m3_kwargs["device"] = device
        if embedding_batch_size is not None:
            m3_kwargs["embedding_batch_size"] = embedding_batch_size
        train_m3(**shared_kwargs, **m3_kwargs, **training_params)


def _json_default(x):
    """Custom JSON serializer for numpy types and other non-JSON-serializable objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return str(x)


# ============================================================================
# ModelPredictions dataclass
# ============================================================================

@dataclasses.dataclass
class ModelPredictions:
    """Standardized prediction output for ensemble consumption.

    All base models produce this format. Models 1 and 3 return empty abstention
    lists. Model 2 may abstain on specimens with zero cluster matches.

    Attributes
    ----------
    probabilities : DataFrame indexed by specimen_label, columns = disease classes.
        Shape (n_scored_specimens, n_classes).
    abstained_specimen_labels : Specimen labels with no prediction.
    abstained_specimen_diseases : Ground-truth disease labels of abstained specimens.
    """

    probabilities: pd.DataFrame
    abstained_specimen_labels: list
    abstained_specimen_diseases: list

    @property
    def n_scored(self) -> int:
        return len(self.probabilities)

    @property
    def n_abstained(self) -> int:
        return len(self.abstained_specimen_labels)


# ============================================================================
# Base model prediction functions
# ============================================================================

def _model1_artifact_paths(
    model_dir: Path, fold_id: Optional[int], model_name: str,
) -> Tuple[Path, Path]:
    """Resolve Model 1's (model.pkl, v_genes.json) paths, fold-optional.

    fold_id is an int for CV artifacts (``fold_<id>_<model_name>_model.pkl``)
    or None for whole-dataset train-all artifacts (``<model_name>_model.pkl``,
    no prefix). Mirrors Model 3's ``_stage_artifact_paths`` fold-optional
    convention so the ensemble can load either layout.
    """
    prefix = f"fold_{fold_id}_" if fold_id is not None else ""
    return (
        model_dir / f"{prefix}{model_name}_model.pkl",
        model_dir / f"{prefix}{model_name}_v_genes.json",
    )


def predict_model1(
    model_dir: Path,
    fold_id: Optional[int],
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    model_name: str = "lasso_cv",
    disease_filter: Optional[Tuple[str, str]] = None,
    summary: Optional[dict] = None,
) -> ModelPredictions:
    """Load Model 1 artifacts and predict on target specimens.

    Parameters
    ----------
    fold_id : CV fold ID (int) → fold-prefixed artifacts, or None →
        whole-dataset train-all artifacts (no fold prefix).
    model_dir : Directory containing the Model 1 artifacts:
        ``fold_<id>_<model_name>_model.pkl`` + ``_v_genes.json`` (CV), or
        ``<model_name>_model.pkl`` + ``_v_genes.json`` (train-all).
    target_specimens : Set of specimen_labels to predict on.
    disease_filter : (disease, reference_class) for binary mode, or None.
    summary : Pre-loaded summary dict. Used for mode validation (multiclass vs binary
        consistency with disease_filter) AND to resolve the Model 1 variant name (the
        artifacts are named ``<model_name>_model.pkl`` — a non-default
        ``--model1-model-name`` must be read from the summary, or loading fails).

    Returns
    -------
    ModelPredictions with probabilities indexed by specimen_label.
    """
    # Resolve the Model 1 variant name from the summary when available. Artifacts are
    # named ``fold_<id>_<model_name>_model.pkl``; a non-default --model1-model-name would
    # otherwise fail to load because callers default model_name to "lasso_cv".
    if summary and summary.get("model_names"):
        model_name = summary["model_names"][0]

    # --- Validate classification mode ---
    if summary:
        mode = summary.get("classification_mode")
        if disease_filter and mode and mode == "multiclass":
            raise ValueError(
                f"Model 1 was trained in multiclass mode but binary disease_filter "
                f"{disease_filter} was specified. Use binary-trained artifacts."
            )
        if not disease_filter and mode and mode in ("binary", "multi-binary"):
            raise ValueError(
                f"Model 1 was trained in {mode} mode but the ensemble is running "
                f"in multiclass mode (no disease_filter). Use multiclass-trained artifacts."
            )

    # --- Load artifacts ---
    logger.info(f"    Model 1: loading artifacts from {model_dir.name}/")
    model_path, v_genes_path = _model1_artifact_paths(model_dir, fold_id, model_name)
    # Fold-aware remediation hint: CV artifacts come from cv_ensemble training,
    # train-all (fold_id is None) from train_all_ensemble.
    _ctx_hint = "cv_ensemble" if fold_id is not None else "train_all_ensemble"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model 1 artifact not found: {model_path}. "
            f"Train Model 1 with --training-context {_ctx_hint} first."
        )
    if not v_genes_path.exists():
        raise FileNotFoundError(
            f"Model 1 V-gene list not found: {v_genes_path}. "
            f"Train Model 1 with --training-context {_ctx_hint} first."
        )

    model = RepertoireClassifier.load(model_path)
    with open(v_genes_path) as f:
        kept_v_genes = json.load(f)

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Filter to training V-genes
    seq = seq[seq[V_GENE_COL].isin(kept_v_genes)].copy()
    logger.info(f"    Model 1: predicting on {len(meta)} specimens ({len(seq)} sequences)")

    # --- Extract features and predict ---
    X = model.extract_features(
        sequences=seq,
        metadata=meta,
        train_vj_columns=model.train_vj_columns_,
    )
    y_proba = model.predict_proba(X)

    proba_df = pd.DataFrame(
        y_proba,
        index=X.index,  # specimen_label
        columns=model.classes_,
    )

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=[],
        abstained_specimen_diseases=[],
    )


def predict_model2(
    model_dir: Path,
    fold_id: Optional[int],
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    gene_locus: str = "TCR",
    model_name: Optional[str] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
    summary: Optional[dict] = None,
    n_jobs: int = 4,
) -> ModelPredictions:
    """Load Model 2 artifacts and predict on target specimens.

    Reads `retrain_on_full_train` from the training summary to load the
    correct artifact files (_split1 vs _full suffix).

    Parameters
    ----------
    model_dir : Directory containing fold_<id>_clusters.joblib, etc.
    target_specimens : Set of specimen_labels to predict on.
    model_name : GLM variant name (default: BEST_MODEL_FOR_METAMODEL[gene_locus]).
    disease_filter : (disease, reference_class) for binary mode, or None.
    summary : Pre-loaded summary dict. If None, read from model_dir.

    Returns
    -------
    ModelPredictions with probabilities for scored specimens; abstention info
    for specimens with zero cluster matches. If Model 2 training produced no
    valid clusters for this fold (NO_VALID_CLUSTERS marker file), returns
    full abstention for all target specimens.
    """
    if model_name is None:
        model_name = BEST_MODEL_FOR_METAMODEL[gene_locus]

    # --- Read config from summary ---
    if summary is None:
        summary = read_model_summary(model_dir)

    # --- Validate classification mode ---
    mode = summary.get("classification_mode")
    if disease_filter and mode and mode == "multiclass":
        raise ValueError(
            f"Model 2 was trained in multiclass mode but binary disease_filter "
            f"{disease_filter} was specified. Use binary-trained artifacts."
        )
    if not disease_filter and mode and mode in ("binary", "multi-binary"):
        raise ValueError(
            f"Model 2 was trained in {mode} mode but the ensemble is running "
            f"in multiclass mode (no disease_filter). Use multiclass-trained artifacts."
        )

    retrain_on_full_train = summary.get("retrain_on_full_train", False)

    # --- Check for "no valid clusters" marker ---
    # When Model 2 training finds no significant convergent clusters for a fold
    # (all p-values skipped), it writes a marker file instead of per-model
    # artifacts. In this case, all specimens must abstain for this fold.
    no_clusters_marker = get_no_valid_clusters_path(model_dir, fold_id, model_name)
    if no_clusters_marker.exists():
        _fold_desc = f"fold {fold_id}" if fold_id is not None else "train-all"
        logger.warning(f"    Model 2: {_fold_desc} has no valid clusters "
                       f"({no_clusters_marker.name}) — all specimens abstain")
        # Derive disease classes and specimen labels from the inputs
        meta = metadata_df[metadata_df[SPECIMEN_COL].isin(target_specimens)].copy()
        if disease_filter:
            target_diseases = {disease_filter[0], disease_filter[1]}
            meta = meta[meta[DISEASE_COL].isin(target_diseases)]
        specimen_labels = meta[SPECIMEN_COL].tolist()
        specimen_diseases = meta[DISEASE_COL].tolist()
        if disease_filter:
            disease_classes = sorted([disease_filter[0], disease_filter[1]])
        else:
            disease_classes = sorted(meta[DISEASE_COL].unique().tolist())
        return ModelPredictions(
            probabilities=pd.DataFrame(columns=disease_classes),
            abstained_specimen_labels=specimen_labels,
            abstained_specimen_diseases=specimen_diseases,
        )

    # --- Load artifacts ---
    logger.info(f"    Model 2: loading artifacts from {model_dir.name}/ "
                f"(retrain_on_full_train={retrain_on_full_train})")
    paths = get_artifact_paths(model_dir, fold_id, model_name, retrain_on_full_train=retrain_on_full_train)
    for key, path in paths.items():
        if key == "metrics":
            continue  # metrics file not needed for prediction
        if not path.exists():
            _ctx_hint = "cv_ensemble" if fold_id is not None else "train_all_ensemble"
            raise FileNotFoundError(
                f"Model 2 artifact not found: {path} (key={key}). "
                f"Train Model 2 with --training-context {_ctx_hint} first."
            )

    clusters_data = joblib.load(paths["clusters"])
    best_p_value = joblib.load(paths["p_value"])
    pipeline = joblib.load(paths["pipeline"])

    centroids_with_scores = clusters_data["centroids_with_scores"]
    disease_classes = clusters_data["disease_classes"]
    sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Model 2 featurize() requires a disease column on sequences
    if DISEASE_COL not in seq.columns:
        disease_map = meta.set_index(SPECIMEN_COL)[DISEASE_COL]
        seq[DISEASE_COL] = seq[SPECIMEN_COL].map(disease_map)

    # --- Featurize and predict ---
    logger.info(f"    Model 2: featurizing {len(meta)} specimens ({len(seq)} sequences, "
                f"n_jobs={n_jobs})")
    fd = featurize(
        seq,
        p_value_threshold=best_p_value,
        centroids_with_scores=centroids_with_scores,
        sequence_identity_threshold=sequence_identity_threshold,
        disease_classes=disease_classes,
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
    )

    if fd.n_scored == 0:
        # All specimens abstained
        return ModelPredictions(
            probabilities=pd.DataFrame(columns=disease_classes),
            abstained_specimen_labels=list(fd.abstained_sample_names),
            abstained_specimen_diseases=list(fd.abstained_sample_y),
        )

    y_proba = pipeline.predict_proba(fd.X)

    proba_df = pd.DataFrame(
        y_proba,
        index=fd.sample_names,
        columns=pipeline.classes_,
    )

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=list(fd.abstained_sample_names),
        abstained_specimen_diseases=list(fd.abstained_sample_y),
    )


def predict_model3(
    model_dir: Path,
    fold_id: Optional[int],
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    embedding_dir: Path,
    gene_locus: str = "TCR",
    disease_filter: Optional[Tuple[str, str]] = None,
    summary: Optional[dict] = None,
    n_jobs: int = 4,
) -> ModelPredictions:
    """Load Model 3 artifacts and predict on target specimens.

    Constructs the model from the training run's summary config so that
    the model object matches the artifact's configuration exactly (strategy,
    tuning params, etc.) regardless of how the model was trained.

    Parameters
    ----------
    model_dir : Directory containing fold_<id>_stage1.pkl, fold_<id>_stage2.pkl,
        and summary_*.json.
    target_specimens : Set of specimen_labels to predict on.
    embedding_dir : Directory with pre-computed ESM-2 embeddings.
    disease_filter : (disease, reference_class) for binary mode, or None.
    summary : Pre-loaded summary dict. If None, read from model_dir.
    n_jobs : Parallel workers for Stage 1 V-gene group predictions.

    Returns
    -------
    ModelPredictions with probabilities indexed by specimen_label. Never abstains.
    """
    from malid_lite.models.model3_sequence_level import (
        DISEASE_COL as M3_DISEASE_COL,
        SPECIMEN_COL as M3_SPECIMEN_COL,
        SequenceLevelClassifier,
    )
    from malid_lite.training.train_model3 import load_precomputed_embeddings

    # --- Read config and construct model to match artifacts ---
    if summary is None:
        summary = read_model_summary(model_dir)

    # --- Validate classification mode ---
    mode = summary.get("classification_mode")
    if disease_filter and mode and mode == "multiclass":
        raise ValueError(
            f"Model 3 was trained in multiclass mode but binary disease_filter "
            f"{disease_filter} was specified. Use binary-trained artifacts."
        )
    if not disease_filter and mode and mode in ("binary", "multi-binary"):
        raise ValueError(
            f"Model 3 was trained in {mode} mode but the ensemble is running "
            f"in multiclass mode (no disease_filter). Use multiclass-trained artifacts."
        )

    agg_strategy = summary.get("aggregation_strategy", "unknown")
    logger.info(f"    Model 3: loading artifacts from {model_dir.name}/ "
                f"(strategy={agg_strategy}, n_jobs={n_jobs})")
    model = SequenceLevelClassifier.from_summary(summary, n_jobs=n_jobs, verbose=0)

    # --- Load artifacts ---
    # Fold-optional naming (shared with train_model3): an int fold_id gives the CV
    # names fold_<id>_stage{1,2}.pkl; fold_id=None gives the train-all names
    # stage{1,2}.pkl (no fold prefix). This lets the ensemble consume both CV and
    # whole-dataset (train-all) Model 3 base models.
    from malid_lite.training.train_model3 import _stage_artifact_paths
    stage1_path, stage2_path = _stage_artifact_paths(model_dir, fold_id)
    _ctx_hint = (
        "cv_ensemble" if fold_id is not None else "train_all_ensemble"
    )
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Model 3 Stage 1 artifact not found: {stage1_path}. "
            f"Train Model 3 with --training-context {_ctx_hint} first."
        )
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Model 3 Stage 2 artifact not found: {stage2_path}. "
            f"Train Model 3 with --training-context {_ctx_hint} first."
        )

    with open(stage1_path, "rb") as f:
        s1_data = pickle.load(f)
    model.load_stage1_artifacts(s1_data)

    with open(stage2_path, "rb") as f:
        s2_data = pickle.load(f)
    model.load_stage2_artifacts(s2_data)

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[M3_SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[M3_SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Ensure disease column is present (needed for ground-truth alignment)
    if M3_DISEASE_COL not in seq.columns:
        disease_map = meta.set_index(M3_SPECIMEN_COL)[DISEASE_COL]
        seq[M3_DISEASE_COL] = seq[M3_SPECIMEN_COL].map(disease_map)

    # --- Load embeddings and predict ---
    logger.info(f"    Model 3: predicting on {len(meta)} specimens ({len(seq)} sequences)")
    seq = seq.reset_index(drop=True)
    embeddings = load_precomputed_embeddings(seq, embedding_dir)
    logger.info(f"    Model 3: embeddings loaded, running Stage 1 + Stage 2...")
    proba_df = model.predict_proba(seq, embeddings)

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=[],
        abstained_specimen_diseases=[],
    )


# ============================================================================
# Feature matrix construction
# ============================================================================

def build_feature_matrix(
    predictions_by_model: Dict[int, ModelPredictions],
    gene_locus: str,
    reference_class: Optional[str] = None,
    model2_abstention_strategy: str = "ensemble_abstain",
) -> Tuple[pd.DataFrame, list, list, Dict]:
    """Build the metamodel feature matrix from base model predictions.

    Steps:
    1. Exclude models that scored zero specimens (full abstention, e.g. Model 2
       found no valid convergent clusters). These contribute no features.
    2. For binary models (2 columns), keep only the non-reference class column.
    3. Rename columns: {locus}:{model_display_name}:{class_name}.
    4. Harmonize abstentions: only specimens scored by ALL contributing models
       are kept (intersection of scored sets). Specimens scored by some but not
       all models ("partially scored") are excluded — these are specimens that
       at least one model explicitly abstained on.
       Exception: when model2_abstention_strategy is "fill_0.5" or
       "fill_models13_mean", Model 2 abstentions are filled rather than dropped.
    5. Concatenate horizontally; columns kept in model-insertion order (deterministic).

    Parameters
    ----------
    predictions_by_model : {model_number: ModelPredictions}.
    gene_locus : "TCR" or "BCR".
    reference_class : For binary mode, the reference/negative class.
    model2_abstention_strategy : How to handle Model 2 abstentions.
        - "ensemble_abstain" (default): specimens with Model 2 abstention are
          dropped from the feature matrix (original behavior).
        - "fill_0.5": fill Model 2's probability columns with 0.5 for abstained
          specimens (uninformative prior).
        - "fill_models13_mean": fill Model 2's probability columns with the
          mean of Models 1 and 3's predictions for each class. Requires both
          Models 1 and 3 to be present.

    Returns
    -------
    (X, abstained_labels, abstained_diseases, fill_info)
        X : DataFrame (n_specimens, n_features), index=specimen_label.
            With "ensemble_abstain": only contains specimens scored by ALL models.
            With fill strategies: contains all specimens scored by non-M2 models
            (M2 abstentions are filled).
        abstained_labels : Specimen labels that are still excluded from X.
            With fill strategies, Model 2 abstentions are NOT in this list
            (they were filled and included in X).
        abstained_diseases : Ground-truth diseases for the abstained_labels,
            in the same order.
        fill_info : Dict with details about filled specimens. Empty dict when
            strategy is "ensemble_abstain". Otherwise contains:
            - "strategy": the strategy used
            - "n_filled": total specimens filled
            - "filled_specimen_labels": list of filled specimen labels
            - "filled_specimen_diseases": list of their ground-truth diseases
            - "filled_per_class": {disease_class: count}
    """
    assert model2_abstention_strategy in MODEL2_ABSTENTION_STRATEGIES, (
        f"Invalid model2_abstention_strategy: {model2_abstention_strategy!r}. "
        f"Must be one of {MODEL2_ABSTENTION_STRATEGIES}"
    )
    use_fill = model2_abstention_strategy != "ensemble_abstain"

    # Validate fill_models13_mean requires both Models 1 and 3
    if model2_abstention_strategy == "fill_models13_mean":
        if 1 not in predictions_by_model or 3 not in predictions_by_model:
            present = sorted(predictions_by_model.keys())
            raise ValueError(
                f"model2_abstention_strategy='fill_models13_mean' requires both "
                f"Models 1 and 3 in the ensemble, but only models {present} are "
                f"present. Use 'fill_0.5' or 'ensemble_abstain' instead."
            )

    # --- Step 1: Binary column selection + column renaming ---
    # We need the disease classes that Model 2 would produce columns for, even
    # when Model 2 has 0 scored specimens, so we can create fill columns.
    renamed_dfs = {}
    fully_abstained_models = []
    model2_disease_classes = None  # populated below if M2 is present
    for model_num, preds in sorted(predictions_by_model.items()):
        proba = preds.probabilities.copy()

        # Track Model 2's disease classes from its column names.
        # predict_model2() always sets columns even on full abstention (0 rows),
        # so proba.shape[1] > 0 covers all cases.
        if model_num == 2 and proba.shape[1] > 0:
            model2_disease_classes = list(proba.columns)

        if len(proba) == 0:
            fully_abstained_models.append(model_num)
            if model_num == 2 and use_fill:
                # Fill strategy: log info, M2 columns will be created in Step 5
                logger.info(
                    f"  Model 2 ({MODEL_DISPLAY_NAMES[2]}): scored 0 specimens "
                    f"— will create fill columns (strategy={model2_abstention_strategy})"
                )
            else:
                logger.warning(
                    f"  Model {model_num} ({MODEL_DISPLAY_NAMES[model_num]}): "
                    f"scored 0 specimens"
                )

        display_name = MODEL_DISPLAY_NAMES[model_num]

        # For binary classifiers, keep only the non-reference class column
        if proba.shape[1] == 2 and reference_class is not None:
            non_ref_cols = [c for c in proba.columns if str(c) != str(reference_class)]
            assert len(non_ref_cols) == 1, (
                f"Model {model_num}: expected 1 non-reference class, "
                f"got {non_ref_cols} (reference={reference_class})"
            )
            proba = proba[non_ref_cols]

        # Rename columns: {locus}:{model_name}:{class_name}
        proba.columns = [
            f"{gene_locus}:{display_name}:{cls}" for cls in proba.columns
        ]
        renamed_dfs[model_num] = proba

    # --- Step 2: Identify Model 2 abstained specimens for filling ---
    fill_info: Dict[str, Any] = {}
    model2_abstained_labels = []
    model2_abstained_diseases = []
    if 2 in predictions_by_model:
        model2_abstained_labels = list(predictions_by_model[2].abstained_specimen_labels)
        model2_abstained_diseases = list(predictions_by_model[2].abstained_specimen_diseases)

    # --- Step 3: Harmonize abstentions ---
    # Build the scored sets for intersection. When using a fill strategy,
    # Model 2's abstained specimens are NOT excluded — they will be filled.
    scored_sets = {}
    for model_num, df in renamed_dfs.items():
        scored_sets[model_num] = set(df.index)

    # Exclude fully-abstained models from the scored intersection.
    # A model that scored 0 specimens provides no discriminative information
    # and would otherwise force the intersection to be empty, preventing
    # training entirely. Exclude it and continue with remaining models.
    # This applies even when a fill strategy is active: a model with zero
    # predictions has nothing to fill from, so it's excluded entirely.
    excluded_models = []
    active_scored_sets = {}
    for mn, ss in scored_sets.items():
        if not ss and mn in fully_abstained_models:
            excluded_models.append(mn)
        else:
            active_scored_sets[mn] = ss

    if excluded_models:
        for mn in excluded_models:
            logger.warning(
                f"  Model {mn} ({MODEL_DISPLAY_NAMES[mn]}): fully abstained (0 scored) "
                f"— excluded from feature matrix. Ensemble will use remaining models."
            )
        # Remove excluded models from renamed_dfs so their columns aren't included
        for mn in excluded_models:
            renamed_dfs.pop(mn, None)

        # fill_models13_mean requires M1 and M3 to have real predictions.
        # Only relevant when M2 is NOT excluded (partial abstention → fill needed).
        if (use_fill and model2_abstention_strategy == "fill_models13_mean"
                and 2 not in excluded_models):
            if 1 in excluded_models or 3 in excluded_models:
                missing = [mn for mn in (1, 3) if mn in excluded_models]
                raise ValueError(
                    f"fill_models13_mean requires Models 1 and 3, but "
                    f"Model(s) {missing} fully abstained (0 scored specimens). "
                    f"Use fill_0.5 or ensemble_abstain instead."
                )

    if not active_scored_sets:
        raise ValueError(
            "All models fully abstained — no specimens with predictions from any model. "
            "Cannot build feature matrix."
        )

    # For fill strategies: treat Model 2's scored set as if it scored everything
    # that the other models scored (M2 abstentions will be filled below)
    if use_fill and 2 in active_scored_sets:
        # Compute the union of all non-M2 models' scored sets
        non_m2_scored = set()
        for mn, ss in active_scored_sets.items():
            if mn != 2:
                non_m2_scored |= ss
        # Also include any M2-scored specimens
        m2_scored = active_scored_sets.get(2, set())
        # For intersection purposes, pretend M2 scored everything non-M2 scored
        scored_sets_for_intersection = {}
        for mn, ss in active_scored_sets.items():
            if mn == 2:
                scored_sets_for_intersection[mn] = non_m2_scored | m2_scored
            else:
                scored_sets_for_intersection[mn] = ss
    else:
        scored_sets_for_intersection = active_scored_sets

    # Common scored = intersection of all contributing models' effective scored sets
    all_effective_sets = list(scored_sets_for_intersection.values())
    common_scored = all_effective_sets[0]
    for s in all_effective_sets[1:]:
        common_scored &= s

    # Collect abstention info. Skip excluded models (their abstentions don't
    # apply since the model isn't part of the feature matrix). With fill
    # strategies, Model 2's abstentions are also not reported (they are filled).
    excluded_set = set(excluded_models)
    all_abstained_labels = []
    all_abstained_diseases = []
    for model_num, preds in predictions_by_model.items():
        if model_num in excluded_set:
            continue
        if use_fill and model_num == 2:
            # M2 abstentions will be filled, not reported as abstained
            continue
        all_abstained_labels.extend(preds.abstained_specimen_labels)
        all_abstained_diseases.extend(preds.abstained_specimen_diseases)

    # Specimens scored by some models but not all (excluding M2 fill cases)
    all_scored_union = set()
    for ss in scored_sets_for_intersection.values():
        all_scored_union |= ss
    partially_scored = all_scored_union - common_scored

    if partially_scored:
        logger.info(
            f"  {len(partially_scored)} specimens scored by some models but not all "
            f"— excluded from ensemble"
        )

    # --- Step 4: Determine which M2-abstained specimens need filling ---
    # Only specimens that are in common_scored but NOT in M2's actual scored set
    # need filling (they are in common_scored because we expanded M2's set above).
    # Skip if M2 was excluded (fully abstained → nothing to fill).
    m2_specimens_to_fill = set()
    if use_fill and 2 in predictions_by_model and 2 not in set(excluded_models):
        m2_actual_scored = scored_sets.get(2, set())
        m2_specimens_to_fill = common_scored - m2_actual_scored

    # --- Step 5: Build Model 2 fill columns if needed ---
    if m2_specimens_to_fill:
        display_name = MODEL_DISPLAY_NAMES[2]
        fill_sorted = sorted(m2_specimens_to_fill)

        # Determine which columns Model 2 should contribute
        if 2 in renamed_dfs:
            # M2 had some scored specimens — use its existing column names
            m2_cols = list(renamed_dfs[2].columns)
        else:
            # M2 scored 0 specimens — derive columns from disease classes
            if model2_disease_classes is None:
                # Infer from other models' class names
                other_model = next(iter(renamed_dfs.values()))
                # Extract class names from "{locus}:{model}:{class}" format
                other_classes = [c.split(":", 2)[2] for c in other_model.columns]
                model2_disease_classes = other_classes
            # Apply binary column selection
            if reference_class is not None and len(model2_disease_classes) == 2:
                m2_classes = [c for c in model2_disease_classes if str(c) != str(reference_class)]
            else:
                m2_classes = model2_disease_classes
            m2_cols = [f"{gene_locus}:{display_name}:{cls}" for cls in m2_classes]

        # Compute fill values per column
        if model2_abstention_strategy == "fill_0.5":
            fill_values = {col: 0.5 for col in m2_cols}
        elif model2_abstention_strategy == "fill_models13_mean":
            # For each M2 column (e.g. "TCR:convergent_cluster_model:Covid19"),
            # extract the class name and average M1 and M3's values for that class
            fill_values = {}
            for m2_col in m2_cols:
                class_name = m2_col.split(":", 2)[2]
                m1_col = f"{gene_locus}:{MODEL_DISPLAY_NAMES[1]}:{class_name}"
                m3_col = f"{gene_locus}:{MODEL_DISPLAY_NAMES[3]}:{class_name}"

                # Verify M1 and M3 have the corresponding columns
                if 1 not in renamed_dfs or m1_col not in renamed_dfs[1].columns:
                    raise ValueError(
                        f"fill_models13_mean: Model 1 column '{m1_col}' not found. "
                        f"Available M1 columns: {list(renamed_dfs.get(1, pd.DataFrame()).columns)}"
                    )
                if 3 not in renamed_dfs or m3_col not in renamed_dfs[3].columns:
                    raise ValueError(
                        f"fill_models13_mean: Model 3 column '{m3_col}' not found. "
                        f"Available M3 columns: {list(renamed_dfs.get(3, pd.DataFrame()).columns)}"
                    )
                # Per-specimen mean of M1 and M3 for this class
                m1_vals = renamed_dfs[1].loc[fill_sorted, m1_col]
                m3_vals = renamed_dfs[3].loc[fill_sorted, m3_col]
                fill_values[m2_col] = (m1_vals.values + m3_vals.values) / 2.0

        # Create a DataFrame for the filled specimens
        fill_data = {}
        for col in m2_cols:
            val = fill_values[col]
            if isinstance(val, np.ndarray):
                fill_data[col] = val
            else:
                fill_data[col] = [val] * len(fill_sorted)
        m2_fill_df = pd.DataFrame(fill_data, index=fill_sorted)

        # Merge filled rows into Model 2's renamed_df
        if 2 in renamed_dfs:
            renamed_dfs[2] = pd.concat([renamed_dfs[2], m2_fill_df], axis=0)
        else:
            renamed_dfs[2] = m2_fill_df

        # Build fill_info with per-class counts.
        # Map M2-abstained specimens to their ground-truth disease.
        m2_abstained_disease_map = dict(
            zip(model2_abstained_labels, model2_abstained_diseases)
        )
        filled_diseases_for_info = []
        for spec in fill_sorted:
            disease = m2_abstained_disease_map.get(spec, "unknown")
            if disease == "unknown":
                logger.warning(
                    f"  Filled specimen '{spec}' not found in Model 2 abstention list "
                    f"— disease label unavailable"
                )
            filled_diseases_for_info.append(disease)
        filled_per_class = {}
        for disease in filled_diseases_for_info:
            filled_per_class[disease] = filled_per_class.get(disease, 0) + 1

        fill_info = {
            "strategy": model2_abstention_strategy,
            "n_filled": len(fill_sorted),
            "filled_specimen_labels": fill_sorted,
            "filled_specimen_diseases": filled_diseases_for_info,
            "filled_per_class": filled_per_class,
        }

        logger.info(
            f"  Model 2 abstention fill: {len(fill_sorted)} specimens filled "
            f"(strategy={model2_abstention_strategy})"
        )
        for disease, count in sorted(filled_per_class.items()):
            logger.info(f"    {disease}: {count} filled")

    # --- Step 6: Filter to common set and concatenate ---
    # Sort specimens (rows) for determinism, but preserve column insertion order
    # (Model 1, then Model 2, then Model 3) to match original Mal-ID behavior.
    common_sorted = sorted(common_scored)
    filtered_dfs = [df.loc[common_sorted] for mn, df in sorted(renamed_dfs.items())]
    X = pd.concat(filtered_dfs, axis=1)

    # Record excluded models in fill_info so callers can track per-fold exclusions
    if excluded_models:
        fill_info["excluded_models"] = excluded_models

    return X, all_abstained_labels, all_abstained_diseases, fill_info


def _build_raw_feature_matrix(
    X_processed: pd.DataFrame,
    predictions_by_model: Dict[int, "ModelPredictions"],
    fill_info: Dict[str, Any],
    abstained_labels: list,
    abstained_diseases: list,
    gene_locus: str,
    reference_class: Optional[str],
    model2_abstention_strategy: str,
) -> pd.DataFrame:
    """Build a strategy-agnostic "raw" feature matrix with NaN for M2 abstentions.

    The raw matrix includes ALL specimens scored by non-M2 models, with M2
    columns set to NaN where Model 2 abstained. This enables changing fill
    strategy at load time (e.g., on --resume or --feature-matrices-dir).

    For fill strategies: starts from X_processed (which has all specimens),
    sets M2 columns back to NaN for filled specimens.

    For ensemble_abstain: starts from X_processed (missing abstained specimens),
    adds rows for M2-abstained specimens with M1/M3 real predictions and M2=NaN.

    Parameters
    ----------
    X_processed : The processed feature matrix from build_feature_matrix().
    predictions_by_model : {model_number: ModelPredictions} — base model outputs.
    fill_info : Fill info dict from build_feature_matrix().
    abstained_labels : Abstained specimen labels from build_feature_matrix().
    abstained_diseases : Abstained diseases from build_feature_matrix().
    gene_locus : "TCR" or "BCR".
    reference_class : Reference class for binary mode, or None.
    model2_abstention_strategy : The strategy that was applied to produce X_processed.

    Returns
    -------
    X_raw : DataFrame with same columns as X_processed, index=specimen_label.
        M2 columns are NaN for M2-abstained specimens; all other values are real.
    """
    m2_display = MODEL_DISPLAY_NAMES[2]
    m2_cols = [c for c in X_processed.columns if f":{m2_display}:" in c]

    # If Model 2 is not in the ensemble, raw == processed (no M2 columns to null out)
    if not m2_cols or 2 not in predictions_by_model:
        return X_processed.copy()

    m2_preds = predictions_by_model[2]
    m2_abstained_set = set(m2_preds.abstained_specimen_labels)

    # No M2 abstentions → raw == processed
    if not m2_abstained_set:
        return X_processed.copy()

    use_fill = model2_abstention_strategy != "ensemble_abstain"

    if use_fill:
        # Fill mode: X_processed already has all specimens (filled ones included).
        # Set M2 columns to NaN for filled specimens to get the raw state.
        X_raw = X_processed.copy()
        filled_labels = fill_info.get("filled_specimen_labels", [])
        if filled_labels:
            X_raw.loc[filled_labels, m2_cols] = np.nan
        return X_raw
    else:
        # ensemble_abstain: X_processed is missing M2-abstained specimens.
        # Add them back with M1/M3 real values and M2=NaN.
        # Only add specimens that were M2-abstained AND scored by all other models.

        # Build rows for M2-abstained specimens from other models' predictions
        non_m2_models = {mn: p for mn, p in predictions_by_model.items() if mn != 2}
        # Intersection of non-M2 scored sets
        non_m2_scored_sets = [set(p.probabilities.index) for p in non_m2_models.values()]
        if not non_m2_scored_sets:
            return X_processed.copy()
        non_m2_common = non_m2_scored_sets[0]
        for s in non_m2_scored_sets[1:]:
            non_m2_common &= s

        # Specimens to add: M2-abstained AND in non_m2_common AND not already in X_processed
        to_add = sorted((m2_abstained_set & non_m2_common) - set(X_processed.index))

        if not to_add:
            return X_processed.copy()

        # Build feature rows for these specimens (same column naming as X_processed)
        add_data = {}
        for col in X_processed.columns:
            if f":{m2_display}:" in col:
                # M2 column → NaN
                add_data[col] = [np.nan] * len(to_add)
            else:
                # Non-M2 column → find the model and extract real values
                parts = col.split(":", 2)
                # parts = [locus, model_display_name, class_name]
                model_display = parts[1]
                class_name = parts[2]
                # Find which model this belongs to
                model_num = None
                for mn, dn in MODEL_DISPLAY_NAMES.items():
                    if dn == model_display:
                        model_num = mn
                        break
                assert model_num is not None, (
                    f"Could not map column '{col}' to a model number"
                )
                proba_df = predictions_by_model[model_num].probabilities
                # The predictions DataFrame columns may be original class names
                # (before renaming). Try both the class_name directly and as a
                # column match.
                if class_name in proba_df.columns:
                    vals = proba_df.loc[to_add, class_name].values
                else:
                    # Column was renamed; shouldn't happen if predictions are consistent
                    raise ValueError(
                        f"Cannot find class '{class_name}' in Model {model_num}'s "
                        f"probability columns: {list(proba_df.columns)}"
                    )
                add_data[col] = vals

        add_df = pd.DataFrame(add_data, index=to_add)
        X_raw = pd.concat([X_processed, add_df], axis=0)
        X_raw = X_raw.sort_index()
        return X_raw


def apply_m2_fill_strategy(
    X_raw: pd.DataFrame,
    strategy: str,
    true_diseases: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, list, list, Dict[str, Any]]:
    """Apply a Model 2 abstention fill strategy to a raw feature matrix.

    The raw feature matrix has NaN in M2 columns for M2-abstained specimens
    and real values everywhere else. This function applies the desired strategy
    and returns the processed matrix along with abstention/fill metadata.

    Parameters
    ----------
    X_raw : Raw feature matrix (specimen_label index, feature columns only —
        no "true_disease" column). M2 columns contain NaN for abstained specimens.
    strategy : One of MODEL2_ABSTENTION_STRATEGIES.
    true_diseases : Series mapping specimen_label -> disease, covering all
        specimens in X_raw. Required for fill_info disease labels. If None,
        disease labels in fill_info will be "unknown".

    Returns
    -------
    (X, abstained_labels, abstained_diseases, fill_info)
        Same semantics as build_feature_matrix() returns.
    """
    assert strategy in MODEL2_ABSTENTION_STRATEGIES, (
        f"Invalid strategy: {strategy!r}. Must be one of {MODEL2_ABSTENTION_STRATEGIES}"
    )

    m2_display = MODEL_DISPLAY_NAMES[2]
    m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]

    # Identify M2-abstained specimens: rows where ANY M2 column is NaN
    if m2_cols:
        m2_nan_mask = X_raw[m2_cols].isna().any(axis=1)
        m2_abstained_labels = sorted(X_raw.index[m2_nan_mask].tolist())
    else:
        m2_nan_mask = pd.Series(False, index=X_raw.index)
        m2_abstained_labels = []

    # If no M2 abstentions, all strategies produce the same result
    if not m2_abstained_labels:
        return X_raw.copy(), [], [], {}

    # M2 fully abstained (ALL specimens have NaN M2 columns): exclude M2
    # entirely by dropping its columns. No specimens are lost, no filling
    # needed — M2 simply had no usable predictions.
    if m2_nan_mask.all():
        X_no_m2 = X_raw.drop(columns=m2_cols).copy()
        fill_info = {"excluded_models": [2]}
        logger.warning(
            f"  Model 2 fully abstained (all {len(X_raw)} specimens have NaN M2 columns) "
            f"— dropping M2 columns from feature matrix"
        )
        return X_no_m2, [], [], fill_info

    # Get disease labels for abstained specimens
    if true_diseases is not None:
        m2_abstained_diseases = [
            true_diseases.get(lbl, "unknown") if hasattr(true_diseases, 'get')
            else true_diseases.loc[lbl] if lbl in true_diseases.index else "unknown"
            for lbl in m2_abstained_labels
        ]
    else:
        m2_abstained_diseases = ["unknown"] * len(m2_abstained_labels)

    fill_info: Dict[str, Any] = {}

    if strategy == "ensemble_abstain":
        # Drop M2-abstained specimens from the matrix
        X = X_raw.loc[~m2_nan_mask].copy()
        return X, m2_abstained_labels, m2_abstained_diseases, fill_info

    # --- Fill strategies ---
    X = X_raw.copy()

    if strategy == "fill_0.5":
        X.loc[m2_abstained_labels, m2_cols] = 0.5
    elif strategy == "fill_models13_mean":
        # For each M2 column, compute mean of corresponding M1 and M3 columns
        m1_display = MODEL_DISPLAY_NAMES[1]
        m3_display = MODEL_DISPLAY_NAMES[3]
        for m2_col in m2_cols:
            class_name = m2_col.split(":", 2)[2]
            m1_col = m2_col.replace(f":{m2_display}:", f":{m1_display}:")
            m3_col = m2_col.replace(f":{m2_display}:", f":{m3_display}:")
            if m1_col not in X.columns:
                raise ValueError(
                    f"fill_models13_mean: Model 1 column '{m1_col}' not found. "
                    f"Available columns with '{m1_display}': "
                    f"{[c for c in X.columns if m1_display in c]}"
                )
            if m3_col not in X.columns:
                raise ValueError(
                    f"fill_models13_mean: Model 3 column '{m3_col}' not found. "
                    f"Available columns with '{m3_display}': "
                    f"{[c for c in X.columns if m3_display in c]}"
                )
            m1_vals = X.loc[m2_abstained_labels, m1_col]
            m3_vals = X.loc[m2_abstained_labels, m3_col]
            X.loc[m2_abstained_labels, m2_col] = (m1_vals.values + m3_vals.values) / 2.0

    # Build fill_info
    filled_per_class: Dict[str, int] = {}
    for disease in m2_abstained_diseases:
        filled_per_class[disease] = filled_per_class.get(disease, 0) + 1

    fill_info = {
        "strategy": strategy,
        "n_filled": len(m2_abstained_labels),
        "filled_specimen_labels": m2_abstained_labels,
        "filled_specimen_diseases": m2_abstained_diseases,
        "filled_per_class": filled_per_class,
    }

    return X, [], [], fill_info


# ============================================================================
# Meta-learner training
# ============================================================================

def train_metamodel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int = 5,
) -> Pipeline:
    """Train the ridge meta-learner on validation predictions.

    Pipeline: StandardScaler -> GlmnetLogitNetWrapper(alpha=0.0, MCC scoring).
    Internal CV: StratifiedGroupKFold grouped by participant.

    Parameters
    ----------
    X_train : Feature matrix (n_validation_specimens, n_features).
    y_train : Disease labels, aligned with X_train.
    groups_train : Participant labels for group-aware CV, aligned with X_train.
    n_splits : Number of CV folds for StratifiedGroupKFold. Default 5
        (matching original Mal-ID). Use a lower value (2-3) for small datasets
        where some classes have fewer than 5 participants.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    import glmnet.scorer

    # TODO: write a custom MCC scorer with NaN guard (like deviance_scorer/rocauc_scorer
    # in glmnet_wrapper.py) to handle degenerate lambdas explicitly. Currently uses
    # glmnet's built-in make_scorer, which goes through predict() → argmax(predict_proba()).
    # If predict_proba returns NaN, argmax silently picks a wrong class instead of failing.
    # Low risk for the metamodel's small dense feature matrix, and matches original Mal-ID.
    mcc_scorer = glmnet.scorer.make_scorer(matthews_corrcoef)

    # Cap n_splits if the data doesn't have enough groups (participants) per class
    n_splits = cap_cv_splits_for_data(
        requested_n_splits=n_splits,
        y=y_train.values,
        groups=groups_train.values,
        context="metamodel CV",
    )
    cv_strategy = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GlmnetLogitNetWrapper(
            alpha=0.0,
            scoring=mcc_scorer,
            n_lambda=100,
            standardize=False,
            random_state=0,
            class_weight="balanced",
            internal_cv=cv_strategy,
            require_cv_group_labels=True,
            use_lambda_1se=False,
        )),
    ])

    pipeline.fit(
        X_train.values,
        y_train.values,
        classifier__groups=groups_train.values,
    )

    return pipeline


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    fold_id: int,
    model_label: str,
    n_scored: int,
    n_abstained: int,
    reference_class: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Compute evaluation metrics for one model on one fold.

    Shared by both the ensemble and individual base model evaluations.
    Accuracy includes abstention penalty: n_correct / (n_scored + n_abstained).

    Returns
    -------
    (metrics_dict, raw_preds_dict)
    """
    n_total = n_scored + n_abstained
    n_correct = int(accuracy_score(y_true, y_pred, normalize=False))

    results = {
        "fold_id": fold_id,
        "model_name": model_label,
        "n_scored": n_scored,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n_total if n_total > 0 else 0.0,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }

    # Multiclass metrics (3+ classes)
    # Matches original crosseval defaults: weighted + macro OvO for both AUROC and AUPRC.
    # Uses our multiclass_metrics module (verbatim copy of original Maxim-multiclass-metrics-repo).
    if len(classes) >= 3:
        for avg_method in ["weighted", "macro"]:
            key_auroc = f"auroc_ovo_{avg_method}"
            key_auprc = f"auprc_ovo_{avg_method}"
            try:
                results[key_auroc] = float(multiclass_metrics.roc_auc_score(
                    y_true, y_proba,
                    average=avg_method, multi_class="ovo", labels=classes,
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"  {key_auroc} failed for {model_label}: {e}")
                results[key_auroc] = None

            try:
                results[key_auprc] = float(multiclass_metrics.auprc(
                    y_true, y_proba,
                    average=avg_method, multi_class="ovo", labels=classes,
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"  {key_auprc} failed for {model_label}: {e}")
                results[key_auprc] = None

        # Per-class AUROC OvR (Lite addition, not in original crosseval defaults)
        auroc_ovr_per_class = {}
        try:
            per_class_scores = multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average=None, multi_class="ovr", labels=classes,
            )
            for cls, score in zip(classes, per_class_scores):
                auroc_ovr_per_class[str(cls)] = float(score)
        except (ValueError, TypeError) as e:
            logger.warning(f"  Per-class AUROC OvR failed for {model_label}: {e}")
            for cls in classes:
                auroc_ovr_per_class[str(cls)] = None
        results["auroc_ovr_per_class"] = auroc_ovr_per_class
    else:
        results["auroc_ovo_weighted"] = None
        results["auroc_ovo_macro"] = None
        results["auprc_ovo_weighted"] = None
        results["auprc_ovo_macro"] = None
        results["auroc_ovr_per_class"] = None

    # Log loss (normalize for Model 3's OvR probabilities that don't sum to 1)
    if len(classes) >= 3:
        row_sums = y_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_proba_for_loss = y_proba / row_sums
    else:
        y_proba_for_loss = y_proba
    try:
        results["log_loss"] = float(sklearn_log_loss(y_true, y_proba_for_loss, labels=classes))
    except ValueError as e:
        logger.warning(f"  Log loss failed for {model_label}: {e}")
        results["log_loss"] = None

    # Binary metrics: AUROC and AUPRC with disease as positive class
    if len(classes) == 2 and reference_class is not None:
        str_classes = [str(c) for c in classes]
        disease_class = next(c for c in str_classes if c != str(reference_class))
        disease_idx = str_classes.index(disease_class)

        y_true_binary = (np.array([str(c) for c in y_true]) == disease_class).astype(int)
        y_score = y_proba[:, disease_idx]

        try:
            results["auroc_binary"] = float(roc_auc_score(y_true_binary, y_score))
        except ValueError as e:
            logger.warning(f"  Binary AUROC failed for {model_label}: {e}")
            results["auroc_binary"] = None
        try:
            results["auprc_binary"] = float(average_precision_score(y_true_binary, y_score))
        except ValueError as e:
            logger.warning(f"  Binary AUPRC failed for {model_label}: {e}")
            results["auprc_binary"] = None

    # MCC on predicted labels
    try:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError as e:
        logger.warning(f"  MCC failed for {model_label}: {e}")
        results["mcc"] = None

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_labels"] = [str(c) for c in classes]
    except ValueError as e:
        logger.warning(f"  Confusion matrix failed for {model_label}: {e}")
        results["confusion_matrix"] = None
        results["confusion_matrix_labels"] = None

    raw_preds = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
    }

    return results, raw_preds


# ============================================================================
# Fold loop: the core orchestration
# ============================================================================

def _log_fold_metrics(
    label: str,
    metrics: Dict,
    reference_class: Optional[str],
) -> None:
    """Log a one-line summary of a model's per-fold metrics."""
    def _f(val):
        return f"{val:.4f}" if val is not None else "N/A"

    n_abstained = metrics.get("n_abstained", 0)
    abstain_str = f" (abstained={n_abstained})" if n_abstained > 0 else ""

    if reference_class is not None:
        logger.info(
            f"\n  {label}: "
            f"AUROC={_f(metrics.get('auroc_binary'))}, "
            f"AUPRC={_f(metrics.get('auprc_binary'))}, "
            f"accuracy={_f(metrics.get('accuracy'))}"
            f"{abstain_str}"
        )
    else:
        logger.info(
            f"\n  {label}: "
            f"AUROC_ovo_w={_f(metrics.get('auroc_ovo_weighted'))}, "
            f"AUPRC_ovo_w={_f(metrics.get('auprc_ovo_weighted'))}, "
            f"accuracy={_f(metrics.get('accuracy'))}"
            f"{abstain_str}"
        )


def _subsample_by_class(
    specimen_set: set,
    metadata_df: pd.DataFrame,
    max_per_class: int,
    seed: int = 0,
) -> set:
    """Subsample specimen set to at most max_per_class specimens per disease."""
    rng = np.random.RandomState(seed)
    meta_subset = metadata_df[metadata_df[SPECIMEN_COL].isin(specimen_set)]
    sampled = set()
    for _, group in meta_subset.groupby(DISEASE_COL):
        specimens = list(group[SPECIMEN_COL])
        if len(specimens) > max_per_class:
            specimens = list(rng.choice(specimens, size=max_per_class, replace=False))
        sampled.update(specimens)
    return sampled


def run_ensemble_fold(
    loader: MalIDPublishedDataLoader,
    fold_id: int,
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]] = None,
    reference_class: Optional[str] = None,
    model_summaries: Optional[Dict[int, dict]] = None,
    n_jobs: int = 4,
    max_specimens_per_class: Optional[int] = None,
    metamodel_cv_n_splits: int = 5,
    model2_abstention_strategy: str = "ensemble_abstain",
) -> Dict:
    """Run the full ensemble pipeline for one fold.

    Parameters
    ----------
    model_summaries : {model_number: summary_dict} pre-loaded summaries.
        Passed through to predict functions so they can construct models
        matching the training config. If None, each predict function reads
        its own summary from model_dir.
    max_specimens_per_class : If set, subsample validation and test specimen
        sets to at most this many per disease class. Useful for fast debugging
        or integration tests.
    metamodel_cv_n_splits : Number of CV folds for the metamodel's internal
        StratifiedGroupKFold. Default 5 (matching original Mal-ID).
    model2_abstention_strategy : How to handle Model 2 abstentions. Passed
        through to build_feature_matrix(). See build_feature_matrix() docstring.

    Returns a dict with keys: fold_id, ensemble_metrics, ensemble_raw_preds,
    base_model_metrics, base_model_raw_preds, pipeline, metamodel_config,
    predictions_rows, feature_matrix_val, feature_matrix_test,
    test_abstained_details, test_fill_info.
    """
    t_fold_start = time.monotonic()
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_id}")
    logger.info(f"{'='*70}")

    # --- Step 1: Load split participants ---
    validation_participants = set(
        loader.get_split_participants(fold_id, TRAINING_CONTEXT, ["validation"])
    )
    # Model 1 trains on ts1+ts2 (all of train_smaller); Models 2, 3 split internally
    logger.info(f"  Validation participants: {len(validation_participants)}")

    # --- Step 2: Load fold data ---
    logger.info("  Loading train fold data...")
    t0 = time.monotonic()
    train_seq, train_meta = loader.get_fold_data(
        fold_id=fold_id, fold_label="train",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
    )
    logger.info(
        f"  Train fold: {len(train_meta)} specimens, "
        f"{len(train_seq):,} sequences [{time.monotonic()-t0:.1f}s]"
    )

    logger.info("  Loading test fold data...")
    t0 = time.monotonic()
    test_seq, test_meta = loader.get_fold_data(
        fold_id=fold_id, fold_label="test",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
    )
    logger.info(
        f"  Test fold: {len(test_meta)} specimens, "
        f"{len(test_seq):,} sequences [{time.monotonic()-t0:.1f}s]"
    )

    # Identify validation and test specimen sets.
    # In binary mode, restrict to specimens from the two target diseases only —
    # specimens from other diseases are not "abstained", they are simply outside
    # the classification scope.
    validation_specimens = set(
        train_meta[train_meta[PARTICIPANT_COL].isin(validation_participants)][SPECIMEN_COL]
    )
    test_specimens = set(test_meta[SPECIMEN_COL])

    if disease_filter:
        disease, ref = disease_filter
        target_diseases = {disease, ref}
        validation_specimens = set(
            train_meta[
                train_meta[PARTICIPANT_COL].isin(validation_participants)
                & train_meta[DISEASE_COL].isin(target_diseases)
            ][SPECIMEN_COL]
        )
        test_specimens = set(
            test_meta[test_meta[DISEASE_COL].isin(target_diseases)][SPECIMEN_COL]
        )

    # Optional subsampling for fast debugging or integration tests
    if max_specimens_per_class is not None:
        validation_specimens = _subsample_by_class(
            validation_specimens, train_meta, max_specimens_per_class, seed=fold_id * 100,
        )
        test_specimens = _subsample_by_class(
            test_specimens, test_meta, max_specimens_per_class, seed=fold_id * 100 + 1,
        )
        logger.info(f"  Subsampled to max {max_specimens_per_class}/class")

    logger.info(f"  Validation specimens: {len(validation_specimens)}")
    logger.info(f"  Test specimens: {len(test_specimens)}")

    # --- Guard: classification mode mismatch ---
    if model_summaries:
        for model_num, summary in model_summaries.items():
            if summary is None:
                continue
            summary_mode = summary.get("classification_mode")
            if disease_filter and summary_mode and summary_mode == "multiclass":
                raise ValueError(
                    f"Model {model_num} was trained in multiclass mode, but the ensemble "
                    f"is running in binary mode (disease_filter={disease_filter}). "
                    f"Binary ensembles must use binary-trained base models."
                )
            if not disease_filter and summary_mode and summary_mode in ("binary", "multi-binary"):
                raise ValueError(
                    f"Model {model_num} was trained in {summary_mode} mode, but the "
                    f"ensemble is running in multiclass mode (no disease_filter). "
                    f"Multiclass ensembles must use multiclass-trained base models."
                )

    # --- Step 3: Get base model predictions on validation ---
    logger.info("\n  Collecting validation predictions...")
    val_predictions: Dict[int, ModelPredictions] = {}
    for model_num in model_nums:
        t0 = time.monotonic()
        summary = (model_summaries or {}).get(model_num)
        preds = _get_model_predictions(
            model_num, model_dirs[model_num], fold_id,
            train_seq, train_meta, validation_specimens,
            gene_locus, embedding_dir, disease_filter,
            summary=summary, n_jobs=n_jobs,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            f"    Model {model_num}: {preds.n_scored} scored, "
            f"{preds.n_abstained} abstained [{elapsed:.1f}s]"
        )
        val_predictions[model_num] = preds

    # --- Step 4: Build validation feature matrix ---
    X_val, val_abstained_labels, val_abstained_diseases, val_fill_info = build_feature_matrix(
        val_predictions, gene_locus, reference_class,
        model2_abstention_strategy=model2_abstention_strategy,
    )
    logger.info(
        f"  Validation feature matrix: {X_val.shape[0]} specimens x {X_val.shape[1]} features"
    )
    if val_fill_info.get("n_filled"):
        logger.info(f"  Validation Model 2 fills: {val_fill_info['n_filled']}")
    if val_fill_info.get("excluded_models"):
        logger.warning(
            f"  Validation: excluded models (fully abstained): "
            f"{val_fill_info['excluded_models']}"
        )
    if val_abstained_labels:
        logger.info(f"  Validation abstentions: {len(val_abstained_labels)}")

    # Build raw (pre-fill) validation feature matrix for saving
    X_val_raw = _build_raw_feature_matrix(
        X_val, val_predictions, val_fill_info,
        val_abstained_labels, val_abstained_diseases,
        gene_locus, reference_class, model2_abstention_strategy,
    )

    # Build val_abstained_details (parallel to test_abstained_details built below)
    val_abstained_details = []
    if val_abstained_labels:
        train_meta_indexed = train_meta.set_index(SPECIMEN_COL)
        for spec_label, disease in zip(val_abstained_labels, val_abstained_diseases):
            participant = (
                train_meta_indexed.loc[spec_label, PARTICIPANT_COL]
                if spec_label in train_meta_indexed.index else "unknown"
            )
            val_abstained_details.append({
                "specimen_label": spec_label,
                "participant_label": participant,
                "disease": disease,
            })

    if X_val.shape[0] == 0:
        raise ValueError(
            f"Fold {fold_id}: all validation specimens were abstained by at least one "
            f"base model — 0 specimens with complete predictions. Cannot train metamodel."
        )

    # Get validation labels and groups, aligned to the feature matrix index
    assert train_meta[SPECIMEN_COL].is_unique, (
        f"Duplicate specimen labels in train_meta: "
        f"{train_meta[SPECIMEN_COL][train_meta[SPECIMEN_COL].duplicated()].tolist()[:10]}"
    )
    val_meta_aligned = train_meta.set_index(SPECIMEN_COL).loc[X_val.index]
    assert len(val_meta_aligned) == X_val.shape[0], (
        f"val_meta_aligned length ({len(val_meta_aligned)}) != X_val rows ({X_val.shape[0]}). "
        f"Possible duplicate specimens in train_meta."
    )
    y_val = val_meta_aligned[DISEASE_COL]
    groups_val = val_meta_aligned[PARTICIPANT_COL]

    # Validate y_val contains only expected disease classes
    if disease_filter:
        expected_diseases = set(disease_filter)
    else:
        expected_diseases = set(train_meta[DISEASE_COL].unique())
    unexpected = set(y_val) - expected_diseases
    assert not unexpected, (
        f"Unexpected classes in y_val: {unexpected}. "
        f"Expected: {sorted(expected_diseases)}"
    )

    # Drop validation specimens with NaN features (can happen if a base model
    # produced degenerate probabilities). Warn with percentage so user knows
    # if data quality is degraded.
    nan_mask_val = X_val.isna().any(axis=1)
    if nan_mask_val.any():
        n_nan = nan_mask_val.sum()
        pct = 100.0 * n_nan / len(X_val)
        if nan_mask_val.all():
            raise ValueError(
                "All validation specimens have NaN features — cannot train metamodel. "
                "Check base model predictions for errors."
            )
        logger.warning(
            f"Dropping {n_nan}/{len(X_val)} validation specimens ({pct:.1f}%) with NaN features"
        )
        X_val = X_val[~nan_mask_val]
        y_val = y_val[~nan_mask_val]
        groups_val = groups_val[~nan_mask_val]

    # --- Step 5: Train metamodel ---
    logger.info("  Training metamodel...")
    t0 = time.monotonic()
    pipeline = train_metamodel(X_val, y_val, groups_val, n_splits=metamodel_cv_n_splits)
    train_time = time.monotonic() - t0
    logger.info(f"  Metamodel training done [{train_time:.1f}s]")

    # Log selected lambda
    clf = pipeline.named_steps["classifier"]
    logger.info(f"  Selected lambda: {clf.lambda_best_:.6f}")

    # --- Step 6: Get base model predictions on test ---
    logger.info("\n  Collecting test predictions...")
    test_predictions: Dict[int, ModelPredictions] = {}
    for model_num in model_nums:
        t0 = time.monotonic()
        summary = (model_summaries or {}).get(model_num)
        preds = _get_model_predictions(
            model_num, model_dirs[model_num], fold_id,
            test_seq, test_meta, test_specimens,
            gene_locus, embedding_dir, disease_filter,
            summary=summary, n_jobs=n_jobs,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            f"    Model {model_num}: {preds.n_scored} scored, "
            f"{preds.n_abstained} abstained [{elapsed:.1f}s]"
        )
        test_predictions[model_num] = preds

    # --- Step 7: Build test feature matrix (same column order as validation) ---
    X_test, test_abstained_labels, test_abstained_diseases, test_fill_info = build_feature_matrix(
        test_predictions, gene_locus, reference_class,
        model2_abstention_strategy=model2_abstention_strategy,
    )

    # Build raw (pre-fill) test feature matrix for saving
    X_test_raw = _build_raw_feature_matrix(
        X_test, test_predictions, test_fill_info,
        test_abstained_labels, test_abstained_diseases,
        gene_locus, reference_class, model2_abstention_strategy,
    )

    # Enforce same column order as validation
    missing_cols = set(X_val.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_val.columns)
    if missing_cols:
        raise ValueError(
            f"Test feature matrix is missing columns present in validation: {missing_cols}. "
            f"This indicates a class mismatch between validation and test."
        )
    if extra_cols:
        logger.warning(
            f"Test has {len(extra_cols)} extra columns not in validation — dropping: {extra_cols}"
        )
    X_test = X_test[X_val.columns]
    assert list(X_test.columns) == list(X_val.columns), (
        f"Column mismatch after reindexing: "
        f"X_test has {list(X_test.columns)[:5]}... vs X_val {list(X_val.columns)[:5]}..."
    )

    logger.info(
        f"  Test feature matrix: {X_test.shape[0]} specimens x {X_test.shape[1]} features"
    )
    n_test_abstained = len(test_specimens) - X_test.shape[0]
    if n_test_abstained > 0:
        logger.info(f"  Test abstentions: {n_test_abstained}")

    if X_test.shape[0] == 0:
        raise ValueError(
            f"Fold {fold_id}: all test specimens were abstained by at least one "
            f"base model — 0 specimens with complete predictions. Cannot evaluate."
        )

    # Drop test specimens with NaN features (counted as abstentions)
    nan_mask_test = X_test.isna().any(axis=1)
    if nan_mask_test.any():
        n_nan = nan_mask_test.sum()
        logger.warning(
            f"Dropping {n_nan}/{len(X_test)} test specimens with NaN features (counted as abstentions)"
        )
        n_test_abstained += n_nan
        X_test = X_test[~nan_mask_test]

    if X_test.shape[0] == 0:
        raise ValueError(
            f"Fold {fold_id}: all test specimens have NaN features after filtering "
            f"— 0 specimens remaining. Cannot evaluate."
        )

    # --- Step 8: Predict with metamodel ---
    y_pred = pipeline.predict(X_test.values)
    y_proba = pipeline.predict_proba(X_test.values)
    classes = pipeline.classes_

    # Align ground-truth labels to the test feature matrix index
    assert test_meta[SPECIMEN_COL].is_unique, (
        f"Duplicate specimen labels in test_meta: "
        f"{test_meta[SPECIMEN_COL][test_meta[SPECIMEN_COL].duplicated()].tolist()[:10]}"
    )
    test_meta_aligned = test_meta.set_index(SPECIMEN_COL).loc[X_test.index]
    assert len(test_meta_aligned) == X_test.shape[0], (
        f"test_meta_aligned length ({len(test_meta_aligned)}) != X_test rows ({X_test.shape[0]}). "
        f"Possible duplicate specimens in test_meta."
    )
    y_true = test_meta_aligned[DISEASE_COL].values

    # --- Step 9: Evaluate ensemble ---
    ensemble_metrics, ensemble_raw_preds = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        fold_id=fold_id,
        model_label="ensemble",
        n_scored=X_test.shape[0],
        n_abstained=n_test_abstained,
        reference_class=reference_class,
    )
    _log_fold_metrics("Ensemble", ensemble_metrics, reference_class)

    # --- Step 10: Evaluate each base model on test specimens ---
    # In ensemble_abstain mode: all models are evaluated on the same common specimen set
    # (intersection of all models' scored specimens). n_abstained is the ensemble-level
    # count, applied equally to all models so accuracy is directly comparable.
    # In fill mode: Model 2 is evaluated only on specimens it actually scored (filled
    # specimens excluded from M2's metrics). Models 1, 3 are evaluated on the full set.
    test_filled_set = set(test_fill_info.get("filled_specimen_labels", []))
    base_model_metrics = {}
    base_model_raw_preds = {}
    for model_num in model_nums:
        preds = test_predictions[model_num]

        # If a model scored 0 specimens (full abstention, e.g. Model 2 with no
        # valid clusters), it was excluded from the feature matrix and cannot be
        # evaluated on the common specimen set. Skip it.
        if len(preds.probabilities) == 0:
            logger.info(
                f"  Model {model_num}: fully abstained — skipping base model evaluation"
            )
            continue

        # Determine evaluation specimen set for this model.
        # For Model 2 in fill mode: exclude filled specimens (M2 only has real
        # predictions for non-filled specimens; fills are synthetic).
        common_specimens = X_test.index
        if model_num == 2 and test_filled_set:
            bm_eval_specimens = common_specimens.difference(test_filled_set)
            if len(bm_eval_specimens) == 0:
                logger.info(
                    f"  Model 2: all {len(test_filled_set)} specimens in the test set "
                    f"were filled — no real predictions to evaluate"
                )
                continue
            logger.info(
                f"  Model 2: evaluating on {len(bm_eval_specimens)} real predictions "
                f"({len(test_filled_set)} filled specimens excluded)"
            )
        else:
            bm_eval_specimens = common_specimens

        # Filter base model probabilities to this model's evaluation set
        proba_common = preds.probabilities.loc[
            preds.probabilities.index.isin(bm_eval_specimens)
        ].loc[bm_eval_specimens]  # enforce same order

        bm_classes = np.array(sorted(preds.probabilities.columns))
        bm_proba = proba_common[bm_classes].values
        bm_y_true = test_meta_aligned.loc[bm_eval_specimens, DISEASE_COL].values
        bm_y_pred = bm_classes[np.argmax(bm_proba, axis=1)]

        bm_metrics, bm_raw = evaluate_predictions(
            y_true=bm_y_true,
            y_pred=bm_y_pred,
            y_proba=bm_proba,
            classes=bm_classes,
            fold_id=fold_id,
            model_label=f"model{model_num}",
            n_scored=len(bm_eval_specimens),
            n_abstained=n_test_abstained,
            reference_class=reference_class,
        )
        base_model_metrics[model_num] = bm_metrics
        base_model_raw_preds[model_num] = bm_raw

        _log_fold_metrics(f"Model {model_num}", bm_metrics, reference_class)

    # --- Build per-specimen prediction rows for CSV ---
    # Scored specimens get full prediction details; abstained specimens are
    # included with ensemble_predicted="ABSTAINED" and no probabilities.
    # model2_filled marks specimens whose Model 2 predictions were filled.
    # (test_filled_set was computed above in Step 10.)
    predictions_rows = []
    for i, specimen in enumerate(X_test.index):
        row = {
            "fold_id": fold_id,
            "specimen_label": specimen,
            "participant_label": test_meta_aligned.loc[specimen, PARTICIPANT_COL],
            "true_disease": y_true[i],
            "ensemble_predicted": y_pred[i],
            "abstained": False,
            "model2_filled": specimen in test_filled_set,
        }
        for j, cls in enumerate(classes):
            row[f"ensemble_P({cls})"] = float(y_proba[i, j])
        predictions_rows.append(row)

    # --- Collect abstained specimen details and add to predictions CSV ---
    test_abstained_details = []
    if test_abstained_labels:
        test_meta_indexed = test_meta.set_index(SPECIMEN_COL)
        for spec_label, disease in zip(test_abstained_labels, test_abstained_diseases):
            participant = (
                test_meta_indexed.loc[spec_label, PARTICIPANT_COL]
                if spec_label in test_meta_indexed.index else "unknown"
            )
            test_abstained_details.append({
                "specimen_label": spec_label,
                "participant_label": participant,
                "disease": disease,
            })
            row = {
                "fold_id": fold_id,
                "specimen_label": spec_label,
                "participant_label": participant,
                "true_disease": disease,
                "ensemble_predicted": "ABSTAINED",
                "abstained": True,
                "model2_filled": False,
            }
            for cls in classes:
                row[f"ensemble_P({cls})"] = None
            predictions_rows.append(row)

    # --- Metamodel config for saving ---
    metamodel_config = {
        "feature_columns": list(X_val.columns),
        "classes": [str(c) for c in classes],
        "gene_locus": gene_locus,
        "models_included": model_nums,
        "n_features": X_val.shape[1],
        "n_validation_specimens": X_val.shape[0],
        "n_test_specimens": X_test.shape[0],
        "n_test_abstained": n_test_abstained,
        "lambda_best": float(clf.lambda_best_),
    }

    elapsed = time.monotonic() - t_fold_start
    logger.info(f"\n  Fold {fold_id} complete [{elapsed:.1f}s]")

    # --- Build feature matrices with labels for saving ---
    X_val_with_labels = X_val.copy()
    X_val_with_labels.insert(0, "true_disease", y_val)

    X_test_with_labels = X_test.copy()
    X_test_with_labels.insert(0, "true_disease", y_true)

    # --- Build raw (pre-fill) feature matrices with labels ---
    # Raw matrices include M2-abstained specimens with NaN in M2 columns.
    # Need true_disease for all specimens, including those not in X_processed.
    val_disease_map = dict(zip(X_val.index, y_val))
    for lbl, dis in zip(val_abstained_labels, val_abstained_diseases):
        val_disease_map[lbl] = dis
    X_val_raw_with_labels = X_val_raw.copy()
    X_val_raw_with_labels.insert(
        0, "true_disease", X_val_raw.index.map(val_disease_map)
    )

    test_disease_map = dict(zip(X_test.index, y_true))
    for lbl, dis in zip(test_abstained_labels, test_abstained_diseases):
        test_disease_map[lbl] = dis
    X_test_raw_with_labels = X_test_raw.copy()
    # Enforce same feature column order as validation raw matrix
    raw_feature_cols = [c for c in X_val_raw_with_labels.columns if c != "true_disease"]
    X_test_raw_with_labels = X_test_raw_with_labels[raw_feature_cols]
    X_test_raw_with_labels.insert(
        0, "true_disease", X_test_raw.index.map(test_disease_map)
    )

    return {
        "fold_id": fold_id,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_raw_preds": ensemble_raw_preds,
        "base_model_metrics": base_model_metrics,
        "base_model_raw_preds": base_model_raw_preds,
        "pipeline": pipeline,
        "metamodel_config": metamodel_config,
        "predictions_rows": predictions_rows,
        "feature_matrix_val": X_val_with_labels,
        "feature_matrix_test": X_test_with_labels,
        "feature_matrix_raw_val": X_val_raw_with_labels,
        "feature_matrix_raw_test": X_test_raw_with_labels,
        "test_abstained_details": test_abstained_details,
        "val_abstained_details": val_abstained_details,
        "test_fill_info": test_fill_info,
        "val_fill_info": val_fill_info,
    }


def run_ensemble_fold_from_features(
    fold_id: int,
    output_dir: Path,
    model_nums: List[int],
    gene_locus: str,
    loader: MalIDPublishedDataLoader,
    reference_class: Optional[str] = None,
    metamodel_cv_n_splits: int = 5,
    model2_abstention_strategy: str = "ensemble_abstain",
    source_dir: Optional[Path] = None,
) -> Dict:
    """Run ensemble fold using previously saved feature matrices.

    Loads feature matrices from source_dir (or output_dir if source_dir is None),
    applies the requested Model 2 abstention fill strategy, trains a new metamodel
    on validation features, evaluates on test features, and recomputes base model
    metrics from the per-model columns.

    When raw feature matrices (fold_*_feature_matrix_raw_*.csv) are available,
    the fill strategy can differ from the original run's strategy. When only
    processed matrices exist (backward compat), the strategy must match.

    Parameters
    ----------
    model2_abstention_strategy : How to handle Model 2 abstentions. Applied to
        the raw feature matrices at load time.
    source_dir : Directory to load feature matrices from. Defaults to output_dir.
        Used by --feature-matrices-dir to load from an external location.
    metamodel_cv_n_splits : Number of CV folds for the metamodel's internal
        StratifiedGroupKFold. Default 5 (matching original Mal-ID).

    Returns the same dict format as run_ensemble_fold.
    """
    if source_dir is None:
        source_dir = output_dir

    t_fold_start = time.monotonic()
    source_label = "external features" if source_dir != output_dir else "saved features"
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_id} (from {source_label})")
    logger.info(f"{'='*70}")

    # --- Load feature matrices (prefer raw, fall back to processed) ---
    raw_val_path = source_dir / f"fold_{fold_id}_feature_matrix_raw_val.csv"
    raw_test_path = source_dir / f"fold_{fold_id}_feature_matrix_raw_test.csv"
    proc_val_path = source_dir / f"fold_{fold_id}_feature_matrix_val.csv"
    proc_test_path = source_dir / f"fold_{fold_id}_feature_matrix_test.csv"

    using_raw = raw_val_path.exists() and raw_test_path.exists()
    using_processed = proc_val_path.exists() and proc_test_path.exists()

    if not using_raw and not using_processed:
        raise FileNotFoundError(
            f"No feature matrices found for fold {fold_id} in {source_dir}.\n"
            f"Looked for: {raw_val_path.name} (raw) and {proc_val_path.name} (processed).\n"
            f"Run the full pipeline first to generate feature matrices."
        )

    # --- Load results JSON for abstention/fill info ---
    results_json_path = source_dir / f"fold_{fold_id}_ensemble_results.json"
    if not results_json_path.exists():
        raise FileNotFoundError(
            f"Results JSON not found: {results_json_path}\n"
            f"This file is required for abstention and fill information."
        )
    with open(results_json_path) as f:
        prev_results = json.load(f)

    # Load abstention details
    test_abstained_details = prev_results.get("test_abstained_details", [])
    prev_n_abstained = prev_results.get("ensemble", {}).get("n_abstained", 0)
    if prev_n_abstained != len(test_abstained_details):
        raise ValueError(
            f"Abstention count mismatch in {results_json_path.name}: "
            f"ensemble.n_abstained={prev_n_abstained} but "
            f"test_abstained_details has {len(test_abstained_details)} entries"
        )
    val_abstained_details = prev_results.get("val_abstained_details", [])

    # Load previous fill info (describes what the source run did)
    prev_test_fill_info = prev_results.get("test_fill_info", {})
    prev_val_fill_info = prev_results.get("val_fill_info", {})
    prev_strategy = prev_test_fill_info.get("strategy") or (
        prev_val_fill_info.get("strategy") or "ensemble_abstain"
    )
    strategy_changed = model2_abstention_strategy != prev_strategy

    if strategy_changed:
        logger.info(
            f"  Fill strategy change: {prev_strategy!r} -> {model2_abstention_strategy!r}"
        )

    _rebuild_abstained_details = False

    if using_raw:
        # Load raw feature matrices and apply fill strategy at load time
        logger.info(f"  Loading raw feature matrices (strategy-agnostic)")
        raw_val_df = pd.read_csv(raw_val_path, index_col="specimen_label")
        raw_test_df = pd.read_csv(raw_test_path, index_col="specimen_label")

        y_val_raw = raw_val_df.pop("true_disease")
        y_true_raw = raw_test_df.pop("true_disease")

        # Apply the requested fill strategy to raw matrices
        X_val, val_abstained_labels, val_abstained_diseases, val_fill_info = (
            apply_m2_fill_strategy(raw_val_df, model2_abstention_strategy, y_val_raw)
        )
        X_test, test_abstained_labels, test_abstained_diseases, test_fill_info = (
            apply_m2_fill_strategy(raw_test_df, model2_abstention_strategy, y_true_raw)
        )

        # Labels aligned to the processed (post-fill) matrices
        y_val = y_val_raw.loc[X_val.index]
        y_true = y_true_raw.loc[X_test.index].values

        # n_abstained for accuracy penalty = specimens dropped by the new strategy
        n_abstained = len(test_abstained_labels)

        # Defer abstained_details rebuild until after metadata lookup (below)
        # so we can include real participant labels instead of "unknown".
        _rebuild_abstained_details = strategy_changed

        # Save raw matrices for re-saving in new output dir
        X_val_raw_with_labels = raw_val_df.copy()
        X_val_raw_with_labels.insert(0, "true_disease", y_val_raw)
        X_test_raw_with_labels = raw_test_df.copy()
        X_test_raw_with_labels.insert(0, "true_disease", y_true_raw)

    else:
        # Fall back to processed matrices (backward compat — no raw available)
        if strategy_changed:
            raise ValueError(
                f"Cannot change fill strategy from {prev_strategy!r} to "
                f"{model2_abstention_strategy!r}: raw feature matrices not found in "
                f"{source_dir}. Raw matrices (fold_*_feature_matrix_raw_*.csv) are "
                f"required to change fill strategy. Re-run the full pipeline to "
                f"generate them."
            )

        logger.info(f"  Loading processed feature matrices (no raw available)")
        val_df = pd.read_csv(proc_val_path, index_col="specimen_label")
        test_df = pd.read_csv(proc_test_path, index_col="specimen_label")

        y_val = val_df.pop("true_disease")
        y_true_series = test_df.pop("true_disease")
        X_val = val_df
        X_test = test_df
        y_true = y_true_series.values

        n_abstained = prev_n_abstained
        test_fill_info = prev_test_fill_info
        val_fill_info = prev_val_fill_info

        # Defensive: initialize abstention lists (not used in this branch since
        # _rebuild_abstained_details is False, but prevents NameError if logic changes)
        test_abstained_labels = []
        test_abstained_diseases = []
        val_abstained_labels = []
        val_abstained_diseases = []

        # No raw matrices to re-save
        X_val_raw_with_labels = None
        X_test_raw_with_labels = None

    logger.info(f"  Validation features: {X_val.shape[0]} x {X_val.shape[1]}")
    logger.info(f"  Test features: {X_test.shape[0]} x {X_test.shape[1]}")
    if n_abstained > 0:
        logger.info(f"  Test abstentions: {n_abstained}")
    if test_fill_info.get("n_filled"):
        logger.info(
            f"  Test fills: {test_fill_info['n_filled']} "
            f"(strategy={test_fill_info.get('strategy', 'N/A')})"
        )
    if test_fill_info.get("excluded_models"):
        logger.warning(
            f"  Test: excluded models (fully abstained): "
            f"{test_fill_info['excluded_models']}"
        )
    if val_fill_info.get("n_filled"):
        logger.info(
            f"  Validation fills: {val_fill_info['n_filled']} "
            f"(strategy={val_fill_info.get('strategy', 'N/A')})"
        )
    if val_fill_info.get("excluded_models"):
        logger.warning(
            f"  Validation: excluded models (fully abstained): "
            f"{val_fill_info['excluded_models']}"
        )

    # --- Get participant groups for metamodel CV ---
    metadata_df = loader.metadata
    specimen_to_participant = dict(
        zip(metadata_df[SPECIMEN_COL], metadata_df[PARTICIPANT_COL])
    )
    missing = [s for s in X_val.index if s not in specimen_to_participant]
    if missing:
        raise ValueError(
            f"{len(missing)} specimen(s) in saved feature matrix not found in current metadata. "
            f"First 5: {missing[:5]}. Metadata may have changed since the original run."
        )
    groups_val = pd.Series(
        [specimen_to_participant[s] for s in X_val.index],
        index=X_val.index,
    )

    # --- Drop rows with NaN features, for parity with the fresh run_ensemble_fold ---
    # Validation: NaN rows cannot train the metamodel → dropped (X_val/y_val/groups_val
    # together). Test: NaN rows cannot be scored → dropped AND counted as abstentions
    # (n_abstained), exactly as the fresh path does. These are non-M2 NaNs only (M2 NaNs
    # were already resolved by apply_m2_fill_strategy above). For the processed-matrix
    # branch the masks are typically empty (NaN test rows were dropped before the matrix
    # was saved), so this is a no-op there.
    nan_mask_val = X_val.isna().any(axis=1)
    if nan_mask_val.any():
        n_nan = int(nan_mask_val.sum())
        if nan_mask_val.all():
            raise ValueError(
                "All validation specimens have NaN features — cannot train metamodel. "
                "Check base model predictions for errors."
            )
        logger.warning(
            f"Dropping {n_nan}/{len(X_val)} validation specimens with NaN features"
        )
        X_val = X_val[~nan_mask_val]
        y_val = y_val[~nan_mask_val]
        groups_val = groups_val[~nan_mask_val]

    nan_mask_test = X_test.isna().any(axis=1)
    if nan_mask_test.any():
        n_nan = int(nan_mask_test.sum())
        logger.warning(
            f"Dropping {n_nan}/{len(X_test)} test specimens with NaN features "
            f"(counted as abstentions)"
        )
        # y_true is a positional numpy array aligned to X_test.index — mask both together
        # so they stay aligned (downstream code relies on this alignment).
        keep_test = (~nan_mask_test).to_numpy()
        y_true = y_true[keep_test]
        X_test = X_test[~nan_mask_test]
        n_abstained += n_nan
        if X_test.shape[0] == 0:
            raise ValueError(
                f"Fold {fold_id}: all test specimens have NaN features after filtering "
                f"— 0 specimens remaining. Cannot evaluate."
            )

    # --- Rebuild abstained_details if fill strategy changed (raw path) ---
    if _rebuild_abstained_details:
        test_abstained_details = [
            {
                "specimen_label": spec,
                "participant_label": specimen_to_participant.get(spec, "unknown"),
                "disease": disease,
            }
            for spec, disease in zip(test_abstained_labels, test_abstained_diseases)
        ]
        val_abstained_details = [
            {
                "specimen_label": spec,
                "participant_label": specimen_to_participant.get(spec, "unknown"),
                "disease": disease,
            }
            for spec, disease in zip(val_abstained_labels, val_abstained_diseases)
        ]
        logger.info(
            f"  Rebuilt abstained_details for new strategy: "
            f"{len(test_abstained_details)} test, {len(val_abstained_details)} val"
        )

    # --- Reconcile test columns to the validation column set (mirrors the fresh path) ---
    # apply_m2_fill_strategy runs on val and test INDEPENDENTLY, so if Model 2 abstains
    # fully on one split but scores some specimens on the other, their feature-column sets
    # can differ (a dropped M2 column on one side). The metamodel is trained on X_val's
    # columns, so X_test MUST be reindexed to them — otherwise pipeline.predict below hits
    # an opaque sklearn shape error instead of a clear message.
    missing_cols = set(X_val.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_val.columns)
    if missing_cols:
        raise ValueError(
            f"Fold {fold_id}: test feature matrix is missing columns present in "
            f"validation: {sorted(missing_cols)}. Likely a Model 2 abstention asymmetry "
            f"between the validation and test splits."
        )
    if extra_cols:
        logger.warning(
            f"  Test has {len(extra_cols)} extra column(s) not in validation — dropping: "
            f"{sorted(extra_cols)}"
        )
    X_test = X_test[list(X_val.columns)]

    # --- Train metamodel ---
    logger.info("  Training metamodel...")
    t0 = time.monotonic()
    pipeline = train_metamodel(X_val, y_val, groups_val, n_splits=metamodel_cv_n_splits)
    train_time = time.monotonic() - t0
    logger.info(f"  Metamodel training done [{train_time:.1f}s]")

    clf = pipeline.named_steps["classifier"]
    logger.info(f"  Selected lambda: {clf.lambda_best_:.6f}")

    # --- Predict with metamodel ---
    y_pred = pipeline.predict(X_test.values)
    y_proba = pipeline.predict_proba(X_test.values)
    classes = pipeline.classes_

    # --- Evaluate ensemble ---
    # n_abstained is loaded from the previous run's results JSON so that
    # accuracy is computed with the same abstention penalty as the original.
    ensemble_metrics, ensemble_raw_preds = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        fold_id=fold_id,
        model_label="ensemble",
        n_scored=X_test.shape[0],
        n_abstained=n_abstained,
        reference_class=reference_class,
    )
    _log_fold_metrics("Ensemble", ensemble_metrics, reference_class)

    # --- Evaluate each base model by extracting its columns from the feature matrix ---
    # Column format: {locus}:{model_display_name}:{class_name}
    # In fill mode, Model 2 is evaluated only on specimens it actually scored.
    test_filled_set = set(test_fill_info.get("filled_specimen_labels", []))
    base_model_metrics = {}
    base_model_raw_preds = {}
    for model_num in model_nums:
        display_name = MODEL_DISPLAY_NAMES[model_num]
        prefix = f"{gene_locus}:{display_name}:"
        model_cols = [c for c in X_test.columns if c.startswith(prefix)]

        if not model_cols:
            # Model fully abstained during training (e.g. Model 2 with no valid
            # clusters) — it was excluded from the feature matrix. Skip evaluation.
            logger.warning(
                f"  Model {model_num} ({display_name}): no columns in feature matrix "
                f"(fully abstained) — skipping base model evaluation"
            )
            continue

        # Determine evaluation specimen set: for Model 2 in fill mode, exclude
        # filled specimens (evaluate on real predictions only).
        if model_num == 2 and test_filled_set:
            bm_eval_idx = X_test.index.difference(test_filled_set)
            if len(bm_eval_idx) == 0:
                logger.info(
                    f"  Model 2 (resume): all specimens were filled "
                    f"— no real predictions to evaluate"
                )
                continue
            logger.info(
                f"  Model 2 (resume): evaluating on {len(bm_eval_idx)} real predictions "
                f"({len(test_filled_set)} filled specimens excluded)"
            )
            bm_eval_df = X_test.loc[bm_eval_idx]
            # y_true is aligned to X_test.index; filter to eval subset
            y_true_aligned = pd.Series(y_true, index=X_test.index)
            bm_y_true_arr = y_true_aligned.loc[bm_eval_idx].values
        else:
            bm_eval_df = X_test
            bm_y_true_arr = y_true

        # Extract class names from column names
        bm_classes = np.array([c.split(":", 2)[2] for c in model_cols])
        bm_proba = bm_eval_df[model_cols].values

        # Binary mode: feature matrix has only the non-reference class column.
        # Reconstruct the 2-class probability matrix for evaluation.
        if len(bm_classes) == 1 and reference_class is not None:
            disease_class = bm_classes[0]
            disease_proba = bm_proba[:, 0]
            ref_proba = 1.0 - disease_proba
            # Sorted class order, e.g. ["Covid19", "Healthy/Background"]
            bm_classes = np.array(sorted([disease_class, reference_class]))
            col_probas = {disease_class: disease_proba, reference_class: ref_proba}
            bm_proba = np.column_stack([col_probas[c] for c in bm_classes])

        bm_y_pred = bm_classes[np.argmax(bm_proba, axis=1)]

        bm_metrics, bm_raw = evaluate_predictions(
            y_true=bm_y_true_arr,
            y_pred=bm_y_pred,
            y_proba=bm_proba,
            classes=bm_classes,
            fold_id=fold_id,
            model_label=f"model{model_num}",
            n_scored=len(bm_eval_df),
            n_abstained=n_abstained,
            reference_class=reference_class,
        )
        base_model_metrics[model_num] = bm_metrics
        base_model_raw_preds[model_num] = bm_raw
        _log_fold_metrics(f"Model {model_num}", bm_metrics, reference_class)

    # --- Build per-specimen prediction rows for CSV ---
    predictions_rows = []
    for i, specimen in enumerate(X_test.index):
        participant = specimen_to_participant.get(specimen, "unknown")
        row = {
            "fold_id": fold_id,
            "specimen_label": specimen,
            "participant_label": participant,
            "true_disease": y_true[i],
            "ensemble_predicted": y_pred[i],
            "abstained": False,
            "model2_filled": specimen in test_filled_set,
        }
        for j, cls in enumerate(classes):
            row[f"ensemble_P({cls})"] = float(y_proba[i, j])
        predictions_rows.append(row)

    # Add abstained specimens to prediction rows (from previous run's details)
    for detail in test_abstained_details:
        row = {
            "fold_id": fold_id,
            "specimen_label": detail["specimen_label"],
            "participant_label": detail["participant_label"],
            "true_disease": detail["disease"],
            "ensemble_predicted": "ABSTAINED",
            "abstained": True,
            "model2_filled": False,
        }
        for cls in classes:
            row[f"ensemble_P({cls})"] = None
        predictions_rows.append(row)

    # --- Metamodel config ---
    metamodel_config = {
        "feature_columns": list(X_val.columns),
        "classes": [str(c) for c in classes],
        "gene_locus": gene_locus,
        "models_included": model_nums,
        "n_features": X_val.shape[1],
        "n_validation_specimens": X_val.shape[0],
        "n_test_specimens": X_test.shape[0],
        "n_test_abstained": n_abstained,
        "lambda_best": float(clf.lambda_best_),
        "resumed": True,
    }

    elapsed = time.monotonic() - t_fold_start
    logger.info(f"\n  Fold {fold_id} complete [{elapsed:.1f}s]")

    # Re-attach labels for saving updated feature matrices
    X_val_with_labels = X_val.copy()
    X_val_with_labels.insert(0, "true_disease", y_val)
    X_test_with_labels = X_test.copy()
    X_test_with_labels.insert(0, "true_disease", y_true)

    result = {
        "fold_id": fold_id,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_raw_preds": ensemble_raw_preds,
        "base_model_metrics": base_model_metrics,
        "base_model_raw_preds": base_model_raw_preds,
        "pipeline": pipeline,
        "metamodel_config": metamodel_config,
        "predictions_rows": predictions_rows,
        "feature_matrix_val": X_val_with_labels,
        "feature_matrix_test": X_test_with_labels,
        "test_abstained_details": test_abstained_details,
        "val_abstained_details": val_abstained_details,
        "test_fill_info": test_fill_info,
        "val_fill_info": val_fill_info,
    }
    # Include raw matrices if available (for re-saving in output dir)
    if X_val_raw_with_labels is not None:
        result["feature_matrix_raw_val"] = X_val_raw_with_labels
    if X_test_raw_with_labels is not None:
        result["feature_matrix_raw_test"] = X_test_raw_with_labels
    return result


# ============================================================================
# Train-all ensemble: metamodel on the validation third, no test / no metrics
# ============================================================================
#
# The train-all ensemble is the CV per-fold flow (run_ensemble_fold) with the
# validation/metamodel steps KEPT and the test/evaluation steps DROPPED. Base
# models are the train_all_ensemble variant (trained on ts1+ts2, validation
# excluded), so the metamodel trains on their out-of-sample predictions over the
# validation third — exactly like CV, minus the test set. The heavy lifting
# stays in the shared free functions (build_feature_matrix, apply_m2_fill_strategy,
# train_metamodel); only the orchestration head is separate (see plan 5.B).


def _train_all_collect_validation(
    loader: MalIDPublishedDataLoader,
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]],
    reference_class: Optional[str],
    model_summaries: Optional[Dict[int, dict]],
    n_jobs: int,
    model2_abstention_strategy: str,
) -> Dict:
    """Build the train-all validation feature matrix by predicting base models.

    The fresh path (no cached matrices). Mirrors run_ensemble_fold steps 1-5 but
    for the whole-dataset train-all context: loads the ENTIRE dataset
    (get_all_data), restricts to the train_all_ensemble "validation" third, gets
    base-model predictions on it (fold_id=None → prefix-less artifacts), and
    builds the processed + raw validation feature matrices. NO test side.

    Returns a dict with X_val, X_val_raw_with_labels, y_val, groups_val,
    val_fill_info, val_abstained_details, and n_validation_per_class.
    """
    # --- Step 1: validation participants (fold_id=None for train-all) ---
    validation_participants = set(
        loader.get_split_participants(None, "train_all_ensemble", ["validation"])
    )
    logger.info(f"  Validation participants: {len(validation_participants)}")

    # --- Step 2: load the WHOLE dataset (all participants, no CV fold) ---
    logger.info("  Loading whole dataset (train-all)...")
    t0 = time.monotonic()
    all_seq, all_meta = loader.get_all_data(
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
    )
    logger.info(
        f"  Whole dataset: {len(all_meta)} specimens, "
        f"{len(all_seq):,} sequences [{time.monotonic()-t0:.1f}s]"
    )

    # Validation specimen set = specimens of the validation participants.
    # In binary mode, restrict to the two target diseases (others are out of
    # scope, not abstentions) — identical to run_ensemble_fold.
    validation_specimens = set(
        all_meta[all_meta[PARTICIPANT_COL].isin(validation_participants)][SPECIMEN_COL]
    )
    if disease_filter:
        disease, ref = disease_filter
        target_diseases = {disease, ref}
        validation_specimens = set(
            all_meta[
                all_meta[PARTICIPANT_COL].isin(validation_participants)
                & all_meta[DISEASE_COL].isin(target_diseases)
            ][SPECIMEN_COL]
        )
    logger.info(f"  Validation specimens: {len(validation_specimens)}")

    # Fail fast with a clear cause: an empty validation set (e.g. the
    # train_all_ensemble split has no validation participants for this pair)
    # would otherwise surface later as a misleading "all specimens abstained"
    # error from the empty feature matrix.
    if not validation_specimens:
        _scope = f" for pair {disease_filter}" if disease_filter else ""
        raise ValueError(
            f"Train-all ensemble: no validation specimens{_scope}. The "
            f"train_all_ensemble split allocated {len(validation_participants)} "
            f"validation participant(s), but none have specimens in the loaded data"
            + (" for the two target diseases" if disease_filter else "")
            + ". Check the split and the dataset."
        )

    # --- Guard: classification mode mismatch (same as run_ensemble_fold) ---
    if model_summaries:
        for model_num, summary in model_summaries.items():
            if summary is None:
                continue
            summary_mode = summary.get("classification_mode")
            if disease_filter and summary_mode and summary_mode == "multiclass":
                raise ValueError(
                    f"Model {model_num} was trained in multiclass mode, but the ensemble "
                    f"is running in binary mode (disease_filter={disease_filter}). "
                    f"Binary ensembles must use binary-trained base models."
                )
            if not disease_filter and summary_mode and summary_mode in ("binary", "multi-binary"):
                raise ValueError(
                    f"Model {model_num} was trained in {summary_mode} mode, but the "
                    f"ensemble is running in multiclass mode (no disease_filter). "
                    f"Multiclass ensembles must use multiclass-trained base models."
                )

    # --- Step 3: base-model predictions on validation (fold_id=None) ---
    logger.info("\n  Collecting validation predictions...")
    val_predictions: Dict[int, ModelPredictions] = {}
    for model_num in model_nums:
        t0 = time.monotonic()
        summary = (model_summaries or {}).get(model_num)
        preds = _get_model_predictions(
            model_num, model_dirs[model_num], None,
            all_seq, all_meta, validation_specimens,
            gene_locus, embedding_dir, disease_filter,
            summary=summary, n_jobs=n_jobs,
        )
        logger.info(
            f"    Model {model_num}: {preds.n_scored} scored, "
            f"{preds.n_abstained} abstained [{time.monotonic()-t0:.1f}s]"
        )
        val_predictions[model_num] = preds

    # --- Step 4: build validation feature matrix (processed + raw) ---
    X_val, val_abstained_labels, val_abstained_diseases, val_fill_info = build_feature_matrix(
        val_predictions, gene_locus, reference_class,
        model2_abstention_strategy=model2_abstention_strategy,
    )
    logger.info(
        f"  Validation feature matrix: {X_val.shape[0]} specimens x {X_val.shape[1]} features"
    )
    if val_fill_info.get("n_filled"):
        logger.info(f"  Validation Model 2 fills: {val_fill_info['n_filled']}")
    if val_fill_info.get("excluded_models"):
        logger.warning(
            f"  Validation: excluded models (fully abstained): "
            f"{val_fill_info['excluded_models']}"
        )
    if val_abstained_labels:
        logger.info(f"  Validation abstentions: {len(val_abstained_labels)}")

    X_val_raw = _build_raw_feature_matrix(
        X_val, val_predictions, val_fill_info,
        val_abstained_labels, val_abstained_diseases,
        gene_locus, reference_class, model2_abstention_strategy,
    )

    if X_val.shape[0] == 0:
        raise ValueError(
            "Train-all ensemble: all validation specimens were abstained by at least "
            "one base model — 0 specimens with complete predictions. Cannot train metamodel."
        )

    # Align labels + participant groups to the feature-matrix index
    assert all_meta[SPECIMEN_COL].is_unique, (
        f"Duplicate specimen labels in metadata: "
        f"{all_meta[SPECIMEN_COL][all_meta[SPECIMEN_COL].duplicated()].tolist()[:10]}"
    )
    val_meta_aligned = all_meta.set_index(SPECIMEN_COL).loc[X_val.index]
    assert len(val_meta_aligned) == X_val.shape[0], (
        f"val_meta_aligned length ({len(val_meta_aligned)}) != X_val rows ({X_val.shape[0]})."
    )
    y_val = val_meta_aligned[DISEASE_COL]
    groups_val = val_meta_aligned[PARTICIPANT_COL]

    # Validate y_val contains only expected disease classes
    if disease_filter:
        expected_diseases = set(disease_filter)
    else:
        expected_diseases = set(all_meta[DISEASE_COL].unique())
    unexpected = set(y_val) - expected_diseases
    assert not unexpected, (
        f"Unexpected classes in y_val: {unexpected}. Expected: {sorted(expected_diseases)}"
    )

    # Drop validation specimens with NaN features (same policy as run_ensemble_fold)
    nan_mask_val = X_val.isna().any(axis=1)
    if nan_mask_val.any():
        n_nan = int(nan_mask_val.sum())
        pct = 100.0 * n_nan / len(X_val)
        if nan_mask_val.all():
            raise ValueError(
                "All validation specimens have NaN features — cannot train metamodel. "
                "Check base model predictions for errors."
            )
        logger.warning(
            f"Dropping {n_nan}/{len(X_val)} validation specimens ({pct:.1f}%) with NaN features"
        )
        X_val = X_val[~nan_mask_val]
        y_val = y_val[~nan_mask_val]
        groups_val = groups_val[~nan_mask_val]

    # Build val_abstained_details (specimen/participant/disease of abstentions)
    val_abstained_details = []
    if val_abstained_labels:
        meta_indexed = all_meta.set_index(SPECIMEN_COL)
        for spec_label, disease in zip(val_abstained_labels, val_abstained_diseases):
            participant = (
                meta_indexed.loc[spec_label, PARTICIPANT_COL]
                if spec_label in meta_indexed.index else "unknown"
            )
            val_abstained_details.append({
                "specimen_label": spec_label,
                "participant_label": participant,
                "disease": disease,
            })

    # Re-attach labels for saving the raw matrix (parallel to run_ensemble_fold)
    val_disease_map = dict(zip(X_val.index, y_val))
    for lbl, dis in zip(val_abstained_labels, val_abstained_diseases):
        val_disease_map[lbl] = dis
    X_val_raw_with_labels = X_val_raw.copy()
    X_val_raw_with_labels.insert(0, "true_disease", X_val_raw.index.map(val_disease_map))

    return {
        "X_val": X_val,
        "X_val_raw_with_labels": X_val_raw_with_labels,
        "y_val": y_val,
        "groups_val": groups_val,
        "val_fill_info": val_fill_info,
        "val_abstained_details": val_abstained_details,
        "n_validation_per_class": y_val.value_counts().to_dict(),
    }


def _train_all_load_validation(
    source_dir: Path,
    loader: MalIDPublishedDataLoader,
    model2_abstention_strategy: str,
) -> Dict:
    """Build the train-all validation feature matrix from a cached raw matrix.

    The resume / --feature-matrices-dir path: loads the prefix-less
    ``feature_matrix_raw_val.csv`` (NO test matrix, NO fold prefix) from
    source_dir, re-applies the requested Model 2 fill strategy, and derives
    labels + participant groups. Same return shape as
    _train_all_collect_validation. Mirrors the validation half of the CV
    run_ensemble_fold_from_features.
    """
    raw_val_path = source_dir / "feature_matrix_raw_val.csv"
    results_json_path = source_dir / "ensemble_results.json"

    if not raw_val_path.exists():
        raise FileNotFoundError(
            f"Train-all raw validation feature matrix not found: {raw_val_path}\n"
            f"Raw matrices are required to (re)train the metamodel from features. "
            f"Run the full train-all ensemble first to generate them."
        )
    if not results_json_path.exists():
        raise FileNotFoundError(
            f"Train-all results JSON not found: {results_json_path}\n"
            f"This file carries the validation abstention/fill information."
        )

    with open(results_json_path) as f:
        prev_results = json.load(f)
    prev_val_fill_info = prev_results.get("val_fill_info", {})
    prev_strategy = prev_val_fill_info.get("strategy") or "ensemble_abstain"
    if model2_abstention_strategy != prev_strategy:
        logger.info(
            f"  Fill strategy change: {prev_strategy!r} -> {model2_abstention_strategy!r}"
        )

    logger.info(f"  Loading raw validation feature matrix from {source_dir}")
    raw_val_df = pd.read_csv(raw_val_path, index_col="specimen_label")
    y_val_raw = raw_val_df.pop("true_disease")

    # Apply the requested fill strategy to the raw matrix
    X_val, val_abstained_labels, val_abstained_diseases, val_fill_info = (
        apply_m2_fill_strategy(raw_val_df, model2_abstention_strategy, y_val_raw)
    )
    y_val = y_val_raw.loc[X_val.index]

    # Participant groups for the metamodel's internal grouped CV
    metadata_df = loader.metadata
    specimen_to_participant = dict(
        zip(metadata_df[SPECIMEN_COL], metadata_df[PARTICIPANT_COL])
    )
    missing = [s for s in X_val.index if s not in specimen_to_participant]
    if missing:
        raise ValueError(
            f"{len(missing)} specimen(s) in the saved feature matrix are not in the "
            f"current metadata. First 5: {missing[:5]}. Metadata may have changed since "
            f"the original run."
        )
    groups_val = pd.Series(
        [specimen_to_participant[s] for s in X_val.index], index=X_val.index,
    )

    # Drop validation rows with NaN features, for parity with the FRESH path
    # (_train_all_collect_validation) so a resume / --feature-matrices-dir run
    # trains the metamodel on the same rows a fresh run would. After the fill
    # strategy, Model 2 NaNs are already resolved (filled or the row dropped);
    # any remaining NaN is a non-M2 degenerate probability, dropped here.
    nan_mask_val = X_val.isna().any(axis=1)
    if nan_mask_val.any():
        n_nan = int(nan_mask_val.sum())
        if nan_mask_val.all():
            raise ValueError(
                "Train-all ensemble (from features): all validation specimens have "
                "NaN features — cannot train the metamodel. Check the saved feature "
                "matrix / base-model predictions."
            )
        logger.warning(
            f"Dropping {n_nan}/{len(X_val)} validation specimens with NaN features"
        )
        X_val = X_val[~nan_mask_val]
        y_val = y_val[~nan_mask_val]
        groups_val = groups_val[~nan_mask_val]

    val_abstained_details = [
        {
            "specimen_label": spec,
            "participant_label": specimen_to_participant.get(spec, "unknown"),
            "disease": disease,
        }
        for spec, disease in zip(val_abstained_labels, val_abstained_diseases)
    ]

    # Re-attach labels for re-saving the raw matrix in the output dir
    X_val_raw_with_labels = raw_val_df.copy()
    X_val_raw_with_labels.insert(0, "true_disease", y_val_raw)

    logger.info(f"  Validation features: {X_val.shape[0]} x {X_val.shape[1]}")

    # Fail fast if the fill strategy left no scored specimens (e.g. every row was
    # M2-abstained and dropped under ensemble_abstain) — the metamodel cannot train.
    if X_val.shape[0] == 0:
        raise ValueError(
            f"Train-all ensemble (from features): 0 validation specimens remain "
            f"after applying the '{model2_abstention_strategy}' abstention strategy "
            f"to {raw_val_path}. All rows were abstained/dropped — cannot train the "
            f"metamodel. Try a fill strategy (fill_0.5 / fill_models13_mean)."
        )

    return {
        "X_val": X_val,
        "X_val_raw_with_labels": X_val_raw_with_labels,
        "y_val": y_val,
        "groups_val": groups_val,
        "val_fill_info": val_fill_info,
        "val_abstained_details": val_abstained_details,
        "n_validation_per_class": y_val.value_counts().to_dict(),
    }


def _run_train_all_ensemble(
    output_dir: Path,
    *,
    loader: MalIDPublishedDataLoader,
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]],
    reference_class: Optional[str],
    model_summaries: Optional[Dict[int, dict]],
    n_jobs: int,
    metamodel_cv_n_splits: int,
    model2_abstention_strategy: str,
    resume: bool,
    run_config: Dict,
    source_dir: Optional[Path] = None,
) -> Dict:
    """Train ONE train-all ensemble (per pair): metamodel on the validation third.

    Per-pair orchestrator — the train-all counterpart of train_ensemble() (which
    loops CV folds). It owns: old-artifact cleanup, run_config.json, building the
    validation feature matrix (fresh predictions OR cached raw matrix), metamodel
    training, prefix-less artifact saving, and the no-metrics training summary
    (5.H) that doubles as the "inference-ready" marker for Phase 6. There is NO
    test set → no metrics, no predictions CSV, no RESULTS metrics table.

    Note: like the CV ensemble, the metamodel is a glmnet fit — if the feature
    matrix is degenerate (e.g. a single feature with near-constant values, as can
    happen in binary mode with only one base model on a tiny dataset), glmnet
    raises "All predictors have zero variance". This is inherited from the shared
    train_metamodel() and is identical to the CV path; include more base models or
    more data to avoid it.

    Parameters
    ----------
    resume : Reload the cached raw-val matrix from output_dir and retrain the
        metamodel (skip base-model prediction). Mirrors the CV from-features resume.
    source_dir : External directory to load the raw-val matrix from
        (--feature-matrices-dir). When set, base-model prediction is skipped and
        the matrix comes from here instead of output_dir.
    run_config : Run configuration dict, saved as run_config.json and used for
        resume config validation.

    Returns the no-metrics training summary dict (also written to disk).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from_features = resume or source_dir is not None

    # --- Resume config validation (same key set as the CV ensemble) ---
    # Skip when source_dir is set: main() validates against the source config.
    if resume and source_dir is None:
        prev_config_path = output_dir / "run_config.json"
        if prev_config_path.exists():
            with open(prev_config_path) as f:
                prev_config = json.load(f)
            _resume_check_keys = [
                "classification_mode", "disease_filter", "reference_class",
                "diseases", "gene_locus", "models_included",
            ]
            mismatches = [
                f"  {k}: was {prev_config.get(k)!r}, now {run_config.get(k)!r}"
                for k in _resume_check_keys
                if prev_config.get(k) != run_config.get(k)
            ]
            if mismatches:
                raise ValueError(
                    "Cannot resume train-all ensemble: run configuration has changed.\n"
                    + "\n".join(mismatches)
                    + "\nRe-run without --resume to start fresh."
                )
            logger.info("  Resume config validation passed")

    # --- Clean up old artifacts (prefix-less train-all layout) ---
    # Fresh run deletes everything; resume / same-dir feature-matrices keep the
    # raw matrix + results JSON (they are inputs).
    _source_is_output = (
        source_dir is not None and source_dir.resolve() == output_dir.resolve()
    )
    _keep_inputs = resume or _source_is_output
    cleanup_patterns = [
        "summary_*.json", "RESULTS_*.md", "run_config.json",
        "ridge_cv_metamodel.joblib", "metamodel_config.json",
        "ensemble_results.json", "feature_matrix_*.csv",
    ]
    for pattern in cleanup_patterns:
        for old_file in output_dir.glob(pattern):
            if _keep_inputs and (
                "feature_matrix" in old_file.name or "ensemble_results" in old_file.name
            ):
                continue
            old_file.unlink()
            logger.info(f"Removed old artifact: {old_file.name}")

    # --- Save run config ---
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=_json_default)

    # --- Build the validation feature matrix ---
    if from_features:
        val = _train_all_load_validation(
            source_dir or output_dir, loader, model2_abstention_strategy,
        )
    else:
        val = _train_all_collect_validation(
            loader, model_nums, model_dirs, gene_locus, embedding_dir,
            disease_filter, reference_class, model_summaries, n_jobs,
            model2_abstention_strategy,
        )
    X_val = val["X_val"]
    y_val = val["y_val"]
    groups_val = val["groups_val"]

    # --- Train the metamodel ---
    logger.info("  Training metamodel...")
    t0 = time.monotonic()
    pipeline = train_metamodel(X_val, y_val, groups_val, n_splits=metamodel_cv_n_splits)
    logger.info(f"  Metamodel training done [{time.monotonic()-t0:.1f}s]")
    clf = pipeline.named_steps["classifier"]
    classes = pipeline.classes_
    logger.info(f"  Selected lambda: {clf.lambda_best_:.6f}")

    # --- Metamodel config: REUSE the CV field names, drop the test-only ones ---
    metamodel_config = {
        "feature_columns": list(X_val.columns),
        "classes": [str(c) for c in classes],
        "gene_locus": gene_locus,
        "models_included": model_nums,
        "n_features": X_val.shape[1],
        "n_validation_specimens": X_val.shape[0],
        "n_validation_per_class": {str(k): int(v) for k, v in val["n_validation_per_class"].items()},
        "lambda_best": float(clf.lambda_best_),
        # Requested n_splits (effective value auto-capped DOWN per class at fit time
        # if there aren't enough participants per class), for provenance parity with
        # the CV run_config (see L2).
        "metamodel_cv_n_splits": metamodel_cv_n_splits,
        "internal_cv": (
            f"StratifiedGroupKFold(n_splits={metamodel_cv_n_splits} "
            f"(auto-capped if needed), shuffle=True, random_state=0)"
        ),
    }

    # --- Save artifacts (prefix-less) ---
    joblib.dump(pipeline, output_dir / "ridge_cv_metamodel.joblib")
    with open(output_dir / "metamodel_config.json", "w") as f:
        json.dump(metamodel_config, f, indent=2, default=_json_default)
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(
            {
                "val_abstained_details": val["val_abstained_details"],
                "val_fill_info": val["val_fill_info"],
            },
            f, indent=2, default=_json_default,
        )
    # Processed (post-fill) validation matrix with labels
    X_val_with_labels = X_val.copy()
    X_val_with_labels.insert(0, "true_disease", y_val)
    X_val_with_labels.to_csv(output_dir / "feature_matrix_val.csv", index_label="specimen_label")
    # Raw (pre-fill) validation matrix with labels
    val["X_val_raw_with_labels"].to_csv(
        output_dir / "feature_matrix_raw_val.csv", index_label="specimen_label"
    )

    # --- Write the no-metrics training summary (5.H) — LAST, so a crashed run
    #     never looks complete. This IS the "inference-ready" marker Phase 6 reads. ---
    summary = _write_train_all_ensemble_summary(
        output_dir, run_config, metamodel_config, val["val_fill_info"],
        model2_abstention_strategy,
    )
    logger.info(f"  Train-all ensemble complete → {output_dir}")
    return summary


def _write_train_all_ensemble_summary(
    output_dir: Path,
    run_config: Dict,
    metamodel_config: Dict,
    val_fill_info: Dict,
    model2_abstention_strategy: str,
) -> Dict:
    """Write the no-metrics train-all ensemble summary_<timestamp>.json (5.H).

    Serves two purposes: (1) summarize the training stats (reusing the CV
    metamodel_config field names), and (2) be the canonical "training complete;
    ready for inference" marker Phase 6 checks. ``training_complete: True`` is
    written LAST (this is the last file written by the run), so a partial run
    never looks complete. Returns the summary dict.
    """
    timestamp = run_config.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "training_only": True,
        "training_complete": True,
        "timestamp": timestamp,
        "training_context": "train_all_ensemble",
        # Inference config: what Phase 6 needs to rebuild the feature matrix and
        # apply the metamodel on a separate dataset.
        "gene_locus": run_config.get("gene_locus"),
        "classification_mode": run_config.get("classification_mode"),
        "reference_class": run_config.get("reference_class"),
        "disease_filter": run_config.get("disease_filter"),
        "models_included": run_config.get("models_included"),
        "model2_abstention_strategy": model2_abstention_strategy,
        "clone_id_params": run_config.get("clone_id_params"),
        "base_model_paths": run_config.get("base_model_paths"),
        "base_model_suffixes": run_config.get("base_model_suffixes"),
        # Base models MUST be the train_all_ensemble variant (validation excluded);
        # Phase 6 asserts this to rule out the leaky train_all base models.
        "base_model_training_context": "train_all_ensemble",
        "embedding_dir": run_config.get("embedding_dir"),
        "dataset_name": run_config.get("dataset_name"),
        "dataset_counts": run_config.get("dataset_counts"),
        "metadata_filter_info": run_config.get("metadata_filter_info"),
        "metadata_resolved_path": run_config.get("metadata_resolved_path"),
        # Training stats (reuse CV metamodel_config field names) + M2 fill counts
        "metamodel_config": metamodel_config,
        "val_fill_info": val_fill_info,
    }
    # Human-readable RESULTS.md (no metrics — train-all has none), for parity with
    # the base-model train-all outputs. Written BEFORE the summary JSON so the JSON
    # (the training_complete marker) remains the last file written.
    (output_dir / f"RESULTS_{timestamp}.md").write_text(
        _render_train_all_ensemble_results_md(summary, metamodel_config, val_fill_info)
    )
    with open(output_dir / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return summary


def _render_train_all_ensemble_results_md(
    summary: Dict, metamodel_config: Dict, val_fill_info: Dict,
) -> str:
    """Render a no-metrics, human-readable train-all ensemble RESULTS.md.

    Documents "what was trained" and the metamodel configuration — the ensemble
    counterpart of the base models' train-all RESULTS.md. Train-all has no held-out
    test set, so there are no metrics to report.
    """
    lines: List[str] = [
        "# Ensemble (Metamodel) — Train-All Training Summary",
        "",
        "Trained on the FULL dataset (no CV fold). No held-out metrics — this "
        "document records what was trained and the metamodel configuration.",
        "",
        "## Run configuration",
        "",
        f"- **Dataset:** {summary.get('dataset_name')}",
        f"- **Training context:** {summary.get('training_context')}",
        f"- **Gene locus:** {summary.get('gene_locus')}",
        f"- **Classification mode:** {summary.get('classification_mode')}",
    ]
    if summary.get("reference_class"):
        lines.append(f"- **Reference class:** {summary['reference_class']}")
    if summary.get("disease_filter"):
        lines.append(f"- **Disease filter (pair):** {summary['disease_filter']}")
    models_included = summary.get("models_included") or []
    lines += [
        f"- **Models included:** {', '.join(f'Model {m}' for m in models_included)}",
        f"- **Model 2 abstention strategy:** {summary.get('model2_abstention_strategy')}",
        "- **Base model training context:** train_all_ensemble "
        "(validation held out — no leakage)",
    ]
    if summary.get("embedding_dir"):
        lines.append(f"- **Embedding dir:** {summary['embedding_dir']}")

    # Dataset counts (participants / specimens), guarded for absence.
    counts = summary.get("dataset_counts") or {}
    filt = summary.get("metadata_filter_info") or {}
    lines += ["", "## Dataset", ""]
    if "total_participants" in counts:
        lines.append(f"- **Total participants:** {counts['total_participants']}")
    if "total_specimens" in counts:
        lines.append(f"- **Total specimens:** {counts['total_specimens']}")
    if filt and filt.get("n_filtered_out"):
        lines.append(
            f"- **Metadata filtering:** {filt['n_filtered_out']} participants excluded "
            f"(no raw data files); {filt.get('n_retained')} retained out of "
            f"{filt.get('n_original')} in metadata file"
        )

    # Metamodel configuration (reuses the CV metamodel_config field names).
    lines += [
        "",
        "## Metamodel",
        "",
        "- **Algorithm:** ridge_cv (L2-regularized glmnet logistic regression)",
        f"- **Classes:** {', '.join(str(c) for c in metamodel_config.get('classes', []))}",
        f"- **Number of features:** {metamodel_config.get('n_features')}",
        f"- **Feature columns:** {', '.join(metamodel_config.get('feature_columns', []))}",
        f"- **Internal CV:** {metamodel_config.get('internal_cv')}",
        f"- **Selected lambda (lambda_best):** {metamodel_config.get('lambda_best')}",
        f"- **Validation specimens:** {metamodel_config.get('n_validation_specimens')}",
        f"- **Validation per class:** {metamodel_config.get('n_validation_per_class')}",
    ]

    # Model 2 abstention handling on the validation matrix (fill info), if present.
    if val_fill_info:
        lines += ["", "## Model 2 abstention handling (validation)", ""]
        for key, val in val_fill_info.items():
            lines.append(f"- **{key}:** {val}")

    # Base model artifact paths.
    base_paths = summary.get("base_model_paths") or {}
    if base_paths:
        lines += ["", "## Base model paths", ""]
        for name, path in base_paths.items():
            lines.append(f"- **{name}:** {path}")

    lines.append("")
    return "\n".join(lines)


def _get_model_predictions(
    model_num: int,
    model_dir: Path,
    fold_id: Optional[int],
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]],
    summary: Optional[dict] = None,
    n_jobs: int = 4,
) -> ModelPredictions:
    """Dispatch to the appropriate model's prediction function.

    fold_id is an int for CV artifacts or None for whole-dataset train-all
    artifacts (no fold prefix); all three predict_modelN functions are
    fold-optional.
    """
    if model_num == 1:
        return predict_model1(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, disease_filter=disease_filter,
            summary=summary,
        )
    elif model_num == 2:
        return predict_model2(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, gene_locus=gene_locus,
            disease_filter=disease_filter, summary=summary,
            n_jobs=n_jobs,
        )
    elif model_num == 3:
        if embedding_dir is None:
            raise ValueError(
                "Model 3 requires --model3-embedding-dir for loading pre-computed embeddings."
            )
        return predict_model3(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, embedding_dir=embedding_dir,
            gene_locus=gene_locus, disease_filter=disease_filter,
            summary=summary, n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown model number: {model_num}")


# ============================================================================
# Artifact saving
# ============================================================================

def save_fold_artifacts(
    output_dir: Path,
    fold_result: Dict,
):
    """Save metamodel pipeline, config, results, and feature matrices for one fold."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_id = fold_result["fold_id"]

    # Save fitted pipeline
    pipeline_path = output_dir / f"fold_{fold_id}_ridge_cv_metamodel.joblib"
    joblib.dump(fold_result["pipeline"], pipeline_path)

    # Save metamodel config
    config_path = output_dir / f"fold_{fold_id}_metamodel_config.json"
    with open(config_path, "w") as f:
        json.dump(fold_result["metamodel_config"], f, indent=2, default=_json_default)

    # Save per-fold results JSON (ensemble + base model metrics + abstention/fill details)
    results_dict = {
        "fold_id": fold_id,
        "ensemble": fold_result["ensemble_metrics"],
        "base_models": {
            f"model{num}": metrics
            for num, metrics in fold_result["base_model_metrics"].items()
        },
        "test_abstained_details": fold_result.get("test_abstained_details", []),
        "val_abstained_details": fold_result.get("val_abstained_details", []),
        "test_fill_info": fold_result.get("test_fill_info", {}),
        "val_fill_info": fold_result.get("val_fill_info", {}),
    }
    results_path = output_dir / f"fold_{fold_id}_ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=_json_default)

    # Save feature matrices (base model probability outputs)
    for split in ("val", "test"):
        key = f"feature_matrix_{split}"
        if key in fold_result:
            fm_path = output_dir / f"fold_{fold_id}_feature_matrix_{split}.csv"
            fold_result[key].to_csv(fm_path, index_label="specimen_label")
        # Save raw (pre-fill) feature matrices — strategy-agnostic, M2=NaN for abstentions
        raw_key = f"feature_matrix_raw_{split}"
        if raw_key in fold_result:
            raw_path = output_dir / f"fold_{fold_id}_feature_matrix_raw_{split}.csv"
            fold_result[raw_key].to_csv(raw_path, index_label="specimen_label")

    logger.info(
        f"  Saved: {pipeline_path.name}, {config_path.name}, "
        f"{results_path.name}, feature matrices"
    )


# ============================================================================
# Main training orchestrator
# ============================================================================

def train_ensemble(
    loader: MalIDPublishedDataLoader,
    fold_ids: List[int],
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    output_dir: Path,
    embedding_dir: Optional[Path] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
    reference_class: Optional[str] = None,
    run_config: Optional[Dict] = None,
    model_summaries: Optional[Dict[int, dict]] = None,
    n_jobs: int = 4,
    resume: bool = False,
    max_specimens_per_class: Optional[int] = None,
    metamodel_cv_n_splits: int = 5,
    model2_abstention_strategy: str = "ensemble_abstain",
    source_dir: Optional[Path] = None,
) -> Tuple[List[Dict], Dict]:
    """Train the ensemble across all folds.

    Parameters
    ----------
    model_summaries : {model_number: summary_dict} pre-loaded from each base
        model's artifact directory. Passed to predict functions for config-aware
        loading. If None, each predict function reads its own summary.
    resume : If True, skip base model predictions and load previously saved
        feature matrices from output_dir.
    max_specimens_per_class : If set, subsample val/test specimens to at most
        this many per disease class. Passed through to run_ensemble_fold.
    metamodel_cv_n_splits : Number of CV folds for the metamodel's internal
        StratifiedGroupKFold. Default 5 (matching original Mal-ID). Use a
        lower value (2-3) for small datasets where some classes have fewer
        than 5 participants.
    source_dir : External directory to load feature matrices from
        (--feature-matrices-dir mode). When set, run_ensemble_fold_from_features()
        loads matrices from this directory. When None, resume loads from output_dir.
        Incompatible args and config are validated by the caller.

    Returns
    -------
    (all_fold_results, aggregated_metrics)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # On resume, validate that key params match the previous run's config.
    # Feature matrices were built with the original config; reusing them with
    # different classification_mode, disease_filter, or models would produce
    # silently wrong results. model2_abstention_strategy is allowed to differ
    # because it's applied at load time from raw feature matrices.
    # Skip when source_dir is set: the caller already validated against the source.
    if resume and source_dir is None and run_config is not None:
        prev_config_path = output_dir / "run_config.json"
        if prev_config_path.exists():
            with open(prev_config_path) as f:
                prev_config = json.load(f)
            _resume_check_keys = [
                "classification_mode", "disease_filter", "reference_class",
                "diseases", "gene_locus", "models_included",
            ]
            mismatches = []
            for key in _resume_check_keys:
                prev_val = prev_config.get(key)
                curr_val = run_config.get(key)
                if prev_val != curr_val:
                    mismatches.append(
                        f"  {key}: was {prev_val!r}, now {curr_val!r}"
                    )
            if mismatches:
                raise ValueError(
                    f"Cannot resume: run configuration has changed.\n"
                    + "\n".join(mismatches)
                    + "\nRe-run without --resume to start fresh."
                )
            # Log if fill strategy changed (allowed, applied at load time)
            prev_strategy = prev_config.get("model2_abstention_strategy")
            curr_strategy = run_config.get("model2_abstention_strategy")
            if prev_strategy != curr_strategy:
                logger.info(
                    f"  Fill strategy changed: {prev_strategy!r} -> {curr_strategy!r} "
                    f"(will apply at load time from raw feature matrices)"
                )
            logger.info("  Resume config validation passed")

    # Remove old artifacts to prevent mixing with new results on partial failure.
    # Resume mode: keep feature matrices (inputs) and ensemble_results.json
    # (contains abstention details needed for accurate accuracy computation).
    # Fresh run: delete everything — it will all be regenerated.
    cleanup_patterns = [
        "summary_*.json",
        "RESULTS_*.md",
        "ensemble_predictions.csv",
        "run_config.json",
        "fold_*_ridge_cv_metamodel.joblib",
        "fold_*_metamodel_config.json",
        "fold_*_ensemble_results.json",
        "fold_*_feature_matrix_*.csv",
    ]
    # If source_dir resolves to the same path as output_dir, protect feature
    # matrices and results (they are our inputs, same as standard resume).
    _source_is_output = (
        source_dir is not None
        and source_dir.resolve() == output_dir.resolve()
    )
    _keep_inputs = resume or _source_is_output
    for pattern in cleanup_patterns:
        for old_file in output_dir.glob(pattern):
            if _keep_inputs and (
                "feature_matrix" in old_file.name
                or "ensemble_results" in old_file.name
            ):
                continue
            old_file.unlink()
            logger.info(f"Removed old artifact: {old_file.name}")

    # Save run configuration (base model paths, args, and settings)
    if run_config is not None:
        run_config_path = output_dir / "run_config.json"
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2, default=_json_default)
        logger.info(f"Saved run config: {run_config_path}")

    all_fold_results = []
    all_ensemble_metrics = []
    all_ensemble_raw_preds = []
    all_predictions_rows = []

    for fold_id in fold_ids:
        if resume or source_dir is not None:
            fold_result = run_ensemble_fold_from_features(
                fold_id=fold_id,
                output_dir=output_dir,
                model_nums=model_nums,
                gene_locus=gene_locus,
                loader=loader,
                reference_class=reference_class,
                metamodel_cv_n_splits=metamodel_cv_n_splits,
                model2_abstention_strategy=model2_abstention_strategy,
                source_dir=source_dir,
            )
        else:
            fold_result = run_ensemble_fold(
                loader=loader,
                fold_id=fold_id,
                model_nums=model_nums,
                model_dirs=model_dirs,
                gene_locus=gene_locus,
                embedding_dir=embedding_dir,
                disease_filter=disease_filter,
                reference_class=reference_class,
                model_summaries=model_summaries,
                n_jobs=n_jobs,
                max_specimens_per_class=max_specimens_per_class,
                metamodel_cv_n_splits=metamodel_cv_n_splits,
                model2_abstention_strategy=model2_abstention_strategy,
            )

        save_fold_artifacts(output_dir, fold_result)

        all_fold_results.append(fold_result)
        all_ensemble_metrics.append(fold_result["ensemble_metrics"])
        all_ensemble_raw_preds.append(fold_result["ensemble_raw_preds"])
        all_predictions_rows.extend(fold_result["predictions_rows"])

    # --- Save predictions CSV ---
    predictions_df = pd.DataFrame(all_predictions_rows)
    predictions_path = output_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"\nSaved predictions: {predictions_path}")

    logger.info(f"\nAll {len(fold_ids)} folds complete. Aggregating results...")

    # --- Aggregate ensemble metrics across folds ---
    aggregated = aggregate_fold_results(
        all_ensemble_metrics,
        all_ensemble_raw_preds,
        disease_filter=disease_filter,
    )

    # --- Aggregate base model metrics ---
    # Models that fully abstained on all folds (e.g. Model 2 with no valid clusters)
    # won't have entries in base_model_metrics — skip them.
    base_model_aggregated = {}
    for model_num in model_nums:
        fold_bm_metrics = [
            fr["base_model_metrics"][model_num]
            for fr in all_fold_results
            if model_num in fr["base_model_metrics"]
        ]
        if not fold_bm_metrics:
            logger.warning(
                f"  Model {model_num}: fully abstained on all folds — "
                f"excluded from ensemble (no base model metrics to aggregate)"
            )
            continue
        fold_bm_raw = [
            fr["base_model_raw_preds"][model_num]
            for fr in all_fold_results
            if model_num in fr["base_model_raw_preds"]
        ]
        base_model_aggregated[model_num] = aggregate_fold_results(
            fold_bm_metrics, fold_bm_raw, disease_filter=disease_filter,
        )

    # --- Save summary JSON ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract run-level fields from run_config for the summary
    _rc = run_config or {}
    # Identify models that were requested but fully abstained (excluded from ensemble)
    models_excluded = [
        num for num in model_nums if num not in base_model_aggregated
    ]
    # Per-fold exclusion details: {model_num: [fold_ids where excluded]}
    per_fold_exclusions: Dict[int, List[int]] = {}
    for fr in all_fold_results:
        for info_key in ("val_fill_info", "test_fill_info"):
            for mn in fr.get(info_key, {}).get("excluded_models", []):
                per_fold_exclusions.setdefault(mn, [])
                if fr["fold_id"] not in per_fold_exclusions[mn]:
                    per_fold_exclusions[mn].append(fr["fold_id"])
    summary = {
        "timestamp": timestamp,
        # Uniform "run finished; ready for inference" marker (5.H). Written at the
        # END of the run (this summary is the last file), so a crashed run never
        # looks complete. Phase 6 checks this ONE field regardless of context.
        "training_complete": True,
        "classification_mode": _rc.get("classification_mode"),
        "reference_class": _rc.get("reference_class"),
        "diseases": _rc.get("diseases"),
        "models_included": model_nums,
        "models_excluded": models_excluded,
        "models_excluded_per_fold": {
            str(mn): sorted(folds) for mn, folds in per_fold_exclusions.items()
        } if per_fold_exclusions else {},
        "gene_locus": gene_locus,
        "clone_id_params": _rc.get("clone_id_params"),
        "fold_ids": fold_ids,
        "disease_filter": list(disease_filter) if disease_filter else None,
        "dataset_counts": _rc.get("dataset_counts"),
        "metadata_filter_info": _rc.get("metadata_filter_info"),
        "ensemble": aggregated,
        "base_models": {
            f"model{num}": agg for num, agg in base_model_aggregated.items()
        },
    }
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info(f"Saved summary: {summary_path}")

    # --- Generate results MD ---
    md_content = _generate_ensemble_results_md(
        run_config=run_config,
        ensemble_agg=aggregated,
        base_model_agg=base_model_aggregated,
        model_nums=model_nums,
        all_fold_results=all_fold_results,
        timestamp=timestamp,
        model2_abstention_strategy=model2_abstention_strategy,
    )
    md_path = output_dir / f"RESULTS_{timestamp}.md"
    md_path.write_text(md_content)
    logger.info(f"Saved results MD: {md_path}")

    # --- Print comparison table ---
    _log_comparison_table(aggregated, base_model_aggregated, model_nums)

    return all_fold_results, summary


def _generate_ensemble_results_md(
    run_config: Optional[Dict],
    ensemble_agg: Dict,
    base_model_agg: Dict[int, Dict],
    model_nums: List[int],
    all_fold_results: List[Dict],
    timestamp: str,
    model2_abstention_strategy: str = "ensemble_abstain",
) -> str:
    """Generate a Markdown summary of ensemble training results."""

    def _fv(val, fmt=".4f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    def _ms(agg, key):
        d = agg.get(key, {})
        if not d or not isinstance(d, dict):
            return "N/A"
        return f"{_fv(d.get('mean'))} +/- {_fv(d.get('std'))}"

    lines: List[str] = []
    lines += ["# Ensemble Training Results", ""]
    lines.append(f"**Timestamp**: {timestamp}")

    if run_config:
        lines += ["", "## Run Configuration", ""]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in run_config.items():
            if k == "metamodel_config" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| metamodel.{mk} | {mv} |")
            elif k == "base_model_paths" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| base_model_path.{mk} | `{mv}` |")
            elif k == "base_model_suffixes" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| base_model_suffix.{mk} | {mv or '(none)'} |")
            else:
                lines.append(f"| {k} | {v} |")

    # --- Models excluded notice ---
    # Collect per-fold exclusion info from val_fill_info (which mirrors test_fill_info)
    per_fold_exclusions: Dict[int, List[int]] = {}  # {model_num: [fold_ids]}
    for fr in all_fold_results:
        fold_id_val = fr["fold_id"]
        for info_key in ("val_fill_info", "test_fill_info"):
            excluded = fr.get(info_key, {}).get("excluded_models", [])
            for mn in excluded:
                per_fold_exclusions.setdefault(mn, [])
                if fold_id_val not in per_fold_exclusions[mn]:
                    per_fold_exclusions[mn].append(fold_id_val)

    # Models excluded from ALL folds (no aggregated metrics at all)
    models_excluded_all = [num for num in model_nums if num not in base_model_agg]
    # Models excluded from SOME folds (have aggregated metrics but with gaps)
    models_excluded_some = [
        num for num in per_fold_exclusions
        if num not in models_excluded_all
    ]
    has_exclusions = models_excluded_all or models_excluded_some

    if has_exclusions:
        lines += ["", "## WARNING: Models Excluded from Ensemble", ""]
        for num in models_excluded_all:
            folds = per_fold_exclusions.get(num, [])
            lines.append(
                f"**Model {num}** was requested but produced no valid predictions on any fold. "
                f"It scored 0 specimens across all folds and was excluded from the ensemble "
                f"feature matrix. The ensemble was trained using the remaining models only."
            )
            if num == 2:
                lines.append(
                    "This typically means Model 2 (convergent clusters) found no statistically "
                    "significant convergent clusters at any p-value threshold. This can happen "
                    "with small datasets or binary classification with limited repertoire overlap."
                )
            lines.append("")
        for num in models_excluded_some:
            folds = sorted(per_fold_exclusions[num])
            n_total = len(all_fold_results)
            lines.append(
                f"**Model {num}** was excluded from {len(folds)}/{n_total} folds "
                f"(folds {folds}) due to full abstention (0 scored specimens). "
                f"On these folds, the ensemble was trained using the remaining models only. "
                f"Aggregated metrics for Model {num} are computed over the "
                f"{n_total - len(folds)} folds where it participated."
            )
            if num == 2:
                lines.append(
                    "This typically means Model 2 (convergent clusters) found no statistically "
                    "significant convergent clusters for those folds."
                )
            lines.append("")

    # --- Abstention methodology note ---
    use_fill = model2_abstention_strategy != "ensemble_abstain"
    m2_fully_excluded = 2 in models_excluded_all
    lines += ["", "## Abstention Handling", ""]
    if use_fill:
        lines.append(
            f"**Model 2 abstention strategy**: `{model2_abstention_strategy}`"
        )
        lines.append("")
        if m2_fully_excluded:
            lines.append(
                "**Note**: Model 2 was fully excluded from the ensemble (see warning above), "
                "so no filling was applied. The ensemble operates without Model 2."
            )
        elif model2_abstention_strategy == "fill_0.5":
            lines.append(
                "When Model 2 abstains (zero cluster matches), its probability features "
                "are filled with 0.5 (uninformative prior). This filling is applied to "
                "both the validation feature matrix (used for metamodel training) and the "
                "test feature matrix (used for evaluation). Filled specimens are included "
                "in the ensemble and fully evaluated."
            )
        elif model2_abstention_strategy == "fill_models13_mean":
            lines.append(
                "When Model 2 abstains (zero cluster matches), its probability features "
                "are filled with the per-class mean of Models 1 and 3's predictions for "
                "that specimen. This filling is applied to both the validation feature "
                "matrix (used for metamodel training) and the test feature matrix (used "
                "for evaluation). Filled specimens are included in the ensemble and "
                "fully evaluated."
            )
        lines.append("")
        if not m2_fully_excluded:
            lines.append(
                "**Note on base model evaluation**: Model 2 is evaluated only on specimens "
                "it actually scored (filled specimens are excluded from Model 2's standalone "
                "metrics). Models 1, 3, and the ensemble are evaluated on the full specimen "
                "set including filled specimens."
            )
        lines.append("")
        lines.append(
            "Remaining abstentions (from non-Model-2 sources, if any) are handled as: "
            "excluded from AUROC, AUPRC, MCC, and log loss; accuracy penalizes them "
            "as errors: `accuracy = n_correct / (n_scored + n_abstained)`."
        )
    else:
        lines.append(
            "Specimens for which any base model abstained (e.g., Model 2 found zero cluster "
            "matches) are excluded from AUROC, AUPRC, MCC, and log loss computation — these "
            "metrics are computed on scored specimens only. Accuracy includes abstentions as "
            "errors: `accuracy = n_correct / (n_scored + n_abstained)`. This matches the "
            "original Mal-ID crosseval `with_abstention=True` behavior."
        )
    lines.append("")

    # --- Abstention summary across folds ---
    total_scored = sum(fr["ensemble_metrics"].get("n_scored", 0) for fr in all_fold_results)
    total_abstained = sum(fr["ensemble_metrics"].get("n_abstained", 0) for fr in all_fold_results)
    total_specimens = total_scored + total_abstained
    overall_abstention_rate = total_abstained / total_specimens if total_specimens > 0 else 0.0

    lines.append(f"**Total scored**: {total_scored} | **Total abstained**: {total_abstained} | "
                 f"**Abstention rate**: {overall_abstention_rate:.2%}")
    lines.append("")

    # --- Model 2 fill summary (if using a fill strategy) ---
    total_test_filled = sum(
        fr.get("test_fill_info", {}).get("n_filled", 0) for fr in all_fold_results
    )
    total_val_filled = sum(
        fr.get("val_fill_info", {}).get("n_filled", 0) for fr in all_fold_results
    )
    total_filled = total_test_filled + total_val_filled
    if use_fill and total_filled > 0:
        lines += ["### Model 2 Filled Specimens", ""]
        lines.append(
            f"**Strategy**: `{model2_abstention_strategy}`"
        )
        lines.append("")

        # --- Test fill details ---
        if total_test_filled > 0:
            lines.append(f"**Test set**: {total_test_filled} specimens filled across all folds")
            lines.append("")

            agg_test_per_class: Dict[str, int] = {}
            for fr in all_fold_results:
                fpc = fr.get("test_fill_info", {}).get("filled_per_class", {})
                for disease, count in fpc.items():
                    agg_test_per_class[disease] = agg_test_per_class.get(disease, 0) + count

            lines.append("| Disease | Specimens Filled (test) |")
            lines.append("|---------|------------------------|")
            for disease in sorted(agg_test_per_class.keys()):
                lines.append(f"| {disease} | {agg_test_per_class[disease]} |")
            lines.append("")

            lines.append("| Fold | N Filled (test) |")
            lines.append("|------|-----------------|")
            for fr in all_fold_results:
                n_f = fr.get("test_fill_info", {}).get("n_filled", 0)
                lines.append(f"| {fr['fold_id']} | {n_f} |")
            lines.append("")

        # --- Validation fill details ---
        if total_val_filled > 0:
            lines.append(
                f"**Validation set** (used for metamodel training): "
                f"{total_val_filled} specimens filled across all folds"
            )
            lines.append("")

            agg_val_per_class: Dict[str, int] = {}
            for fr in all_fold_results:
                fpc = fr.get("val_fill_info", {}).get("filled_per_class", {})
                for disease, count in fpc.items():
                    agg_val_per_class[disease] = agg_val_per_class.get(disease, 0) + count

            lines.append("| Disease | Specimens Filled (validation) |")
            lines.append("|---------|-------------------------------|")
            for disease in sorted(agg_val_per_class.keys()):
                lines.append(f"| {disease} | {agg_val_per_class[disease]} |")
            lines.append("")

            lines.append("| Fold | N Filled (validation) |")
            lines.append("|------|-----------------------|")
            for fr in all_fold_results:
                n_f = fr.get("val_fill_info", {}).get("n_filled", 0)
                lines.append(f"| {fr['fold_id']} | {n_f} |")
            lines.append("")

    # Note: per-specimen abstention details are at the end of the report
    any_abstentions = any(fr.get("test_abstained_details") for fr in all_fold_results)
    if any_abstentions:
        lines.append(
            "See [Abstained Specimens](#abstained-specimens) at the end of "
            "this report for the full list."
        )
        lines.append("")

    lines += ["---", ""]

    # --- Comparison table (binary vs multiclass columns) ---
    is_binary = "auroc_pooled" in ensemble_agg
    lines += ["## Model Comparison", ""]
    if is_binary:
        lines.append("| Model | Accuracy (global) | AUROC (pooled) | AUPRC (pooled) | MCC |")
        lines.append("|-------|-------------------|----------------|----------------|-----|")
    else:
        lines.append("| Model | Accuracy (global) | AUROC OvO weighted | AUROC OvO macro | MCC |")
        lines.append("|-------|-------------------|--------------------|-----------------|-----|")

    comparison_entries = [("**Ensemble**", ensemble_agg)] + [
        (f"Model {num}", base_model_agg[num])
        for num in model_nums if num in base_model_agg
    ]
    # Models that fully abstained get a placeholder row
    for num in model_nums:
        if num not in base_model_agg:
            comparison_entries.append((f"Model {num} (abstained)", {}))
    for label, agg in comparison_entries:
        acc = _fv(agg.get("accuracy_global"))
        mcc_d = agg.get("mcc", {})
        mcc = _fv(mcc_d.get("mean")) if isinstance(mcc_d, dict) else "N/A"
        if is_binary:
            auroc = _fv(agg.get("auroc_pooled"))
            auprc = _fv(agg.get("auprc_pooled"))
            lines.append(f"| {label} | {acc} | {auroc} | {auprc} | {mcc} |")
        else:
            auroc_w = _ms(agg, "auroc_ovo_weighted")
            auroc_m = _ms(agg, "auroc_ovo_macro")
            lines.append(f"| {label} | {acc} | {auroc_w} | {auroc_m} | {mcc} |")
    lines += [""]

    # --- Helper: per-disease breakdown for one model ---
    def _add_disease_breakdown(lines, label, agg):
        """Add per-class AUROC (OvR) table, per-class accuracy, and confusion matrix."""
        cm = agg.get("confusion_matrix_aggregated")
        classes = agg.get("classes", [])

        # Per-class AUROC (OvR) table
        auroc_ovr = agg.get("auroc_ovr_per_class", {})
        if auroc_ovr and not is_binary:
            lines += [f"### Per-Class AUROC (OvR) — {label}", ""]
            lines.append("| Disease | AUROC (OvR) | Std Dev |")
            lines.append("|---------|-------------|---------|")
            for cls_name, cls_data in auroc_ovr.items():
                if isinstance(cls_data, dict):
                    lines.append(
                        f"| {cls_name} | {_fv(cls_data.get('mean'))} "
                        f"| +/-{_fv(cls_data.get('std'))} |"
                    )
            lines += [""]

        # Per-class accuracy from confusion matrix
        if cm and classes:
            lines += [f"### Per-Class Accuracy — {label}", ""]
            lines.append("| Disease | Correct | Total | Accuracy |")
            lines.append("|---------|---------|-------|----------|")
            for i, cls in enumerate(classes):
                total = sum(cm[i])
                correct = cm[i][i]
                acc_pct = f"{correct / total * 100:.1f}%" if total > 0 else "N/A"
                lines.append(f"| {cls} | {correct} | {total} | {acc_pct} |")
            lines += [""]

        # Aggregated confusion matrix
        if cm and classes:
            lines += [f"### Aggregated Confusion Matrix — {label}", ""]
            lines.append("| | " + " | ".join(str(c) for c in classes) + " |")
            lines.append("|-" + "-|-".join("---" for _ in classes) + "-|")
            for i, cls in enumerate(classes):
                row_vals = " | ".join(str(cm[i][j]) for j in range(len(classes)))
                lines.append(f"| **{cls}** | {row_vals} |")
            lines += [""]

    # --- Per-fold ensemble results ---
    auroc_col = "AUROC" if is_binary else "AUROC OvO weighted"
    auroc_key = "auroc_binary" if is_binary else "auroc_ovo_weighted"
    lines += ["## Ensemble Results", ""]

    lines += ["### Per-Fold Results", ""]
    lines.append(f"| Fold | Accuracy | {auroc_col} | MCC | N scored | N abstained |")
    lines.append("|------|----------|" + "-" * (len(auroc_col) + 2) + "|-----|----------|-------------|")
    for fr in all_fold_results:
        em = fr["ensemble_metrics"]
        lines.append(
            f"| {em['fold_id']} | {_fv(em.get('accuracy'))} | "
            f"{_fv(em.get(auroc_key))} | "
            f"{_fv(em.get('mcc'))} | "
            f"{em.get('n_scored', 'N/A')} | {em.get('n_abstained', 0)} |"
        )
    lines += [""]

    _add_disease_breakdown(lines, "Ensemble", ensemble_agg)

    # --- Per base model results ---
    for num in model_nums:
        if num not in base_model_agg:
            lines += [f"## Model {num} Results", ""]
            lines.append(
                f"*Model {num} fully abstained on all folds — no valid predictions were produced. "
                f"This model was excluded from the ensemble feature matrix and evaluation.*"
            )
            if num == 2:
                lines.append(
                    "*Reason: No statistically significant convergent clusters were found "
                    "at any p-value threshold (see NO_VALID_CLUSTERS.txt in the Model 2 output directory).*"
                )
            lines += [""]
            continue

        lines += [f"## Model {num} Results", ""]

        lines += ["### Per-Fold Results", ""]
        lines.append(f"| Fold | Accuracy | {auroc_col} | MCC |")
        lines.append("|------|----------|" + "-" * (len(auroc_col) + 2) + "|-----|")
        for fr in all_fold_results:
            if num not in fr["base_model_metrics"]:
                lines.append(
                    f"| {fr['fold_id']} | N/A | N/A | N/A |"
                )
                continue
            bm = fr["base_model_metrics"][num]
            lines.append(
                f"| {bm['fold_id']} | {_fv(bm.get('accuracy'))} | "
                f"{_fv(bm.get(auroc_key))} | "
                f"{_fv(bm.get('mcc'))} |"
            )
        lines += [""]

        _add_disease_breakdown(lines, f"Model {num}", base_model_agg[num])

    # --- Investigation: metrics by Model 2 fill status ---
    # Only generated when a fill strategy was used and there are filled specimens.
    if use_fill and total_filled > 0:
        lines += ["---", ""]
        lines += ["## Investigation: Ensemble Performance by Model 2 Fill Status", ""]
        lines.append(
            "This section computes ensemble metrics separately for specimens where "
            "Model 2 had real predictions vs. specimens where Model 2 predictions "
            "were filled. This helps assess whether filled specimens degrade "
            "ensemble quality."
        )
        lines.append("")

        # Collect per-specimen predictions across all folds, split by fill status.
        # prediction_rows have: specimen_label, true_disease, ensemble_predicted,
        # ensemble_P(cls), model2_filled, abstained
        all_rows = []
        for fr in all_fold_results:
            all_rows.extend(fr["predictions_rows"])

        # Filter to scored (non-abstained) specimens only
        scored_rows = [r for r in all_rows if not r.get("abstained", False)]

        real_rows = [r for r in scored_rows if not r.get("model2_filled", False)]
        filled_rows = [r for r in scored_rows if r.get("model2_filled", False)]

        # Get ensemble class list from first fold's pipeline
        ensemble_classes = None
        for fr in all_fold_results:
            ensemble_classes = fr["pipeline"].classes_
            break
        if ensemble_classes is None:
            lines.append("*Could not determine ensemble classes — skipping investigation.*")
            lines.append("")
        else:
            ensemble_classes = np.array(ensemble_classes)
            prob_cols = [f"ensemble_P({cls})" for cls in ensemble_classes]

            # Extract reference_class for binary vs multiclass metric selection
            reference_class = (run_config or {}).get("reference_class")

            def _compute_subset_metrics(rows, subset_label):
                """Compute metrics for a subset of prediction rows."""
                if not rows:
                    return {"n": 0}
                y_true_sub = np.array([r["true_disease"] for r in rows])
                y_pred_sub = np.array([r["ensemble_predicted"] for r in rows])
                y_proba_sub = np.array([[r[pc] for pc in prob_cols] for r in rows])

                result = {"n": len(rows)}

                # Accuracy
                result["accuracy"] = float(accuracy_score(y_true_sub, y_pred_sub))

                # Balanced accuracy
                result["balanced_accuracy"] = float(
                    balanced_accuracy_score(y_true_sub, y_pred_sub)
                )

                # MCC
                try:
                    result["mcc"] = float(matthews_corrcoef(y_true_sub, y_pred_sub))
                except ValueError:
                    result["mcc"] = None

                # AUROC and AUPRC — binary vs multiclass
                if len(ensemble_classes) == 2 and reference_class is not None:
                    ref_class = reference_class
                    str_classes = [str(c) for c in ensemble_classes]
                    disease_class = next(
                        c for c in str_classes if c != str(ref_class)
                    )
                    disease_idx = str_classes.index(disease_class)
                    y_true_bin = (
                        np.array([str(c) for c in y_true_sub]) == disease_class
                    ).astype(int)
                    y_score = y_proba_sub[:, disease_idx]

                    # Need both classes present for AUROC/AUPRC
                    if len(np.unique(y_true_bin)) < 2:
                        result["auroc"] = None
                        result["auprc"] = None
                    else:
                        try:
                            result["auroc"] = float(
                                roc_auc_score(y_true_bin, y_score)
                            )
                        except ValueError:
                            result["auroc"] = None
                        try:
                            result["auprc"] = float(
                                average_precision_score(y_true_bin, y_score)
                            )
                        except ValueError:
                            result["auprc"] = None
                else:
                    # Multiclass
                    unique_true = set(y_true_sub)
                    if len(unique_true) < 2:
                        result["auroc"] = None
                        result["auprc"] = None
                    else:
                        try:
                            result["auroc"] = float(
                                multiclass_metrics.roc_auc_score(
                                    y_true_sub, y_proba_sub,
                                    average="weighted", multi_class="ovo",
                                    labels=ensemble_classes,
                                )
                            )
                        except (ValueError, TypeError):
                            result["auroc"] = None
                        try:
                            result["auprc"] = float(
                                multiclass_metrics.auprc(
                                    y_true_sub, y_proba_sub,
                                    average="weighted", multi_class="ovo",
                                    labels=ensemble_classes,
                                )
                            )
                        except (ValueError, TypeError):
                            result["auprc"] = None
                return result

            real_metrics = _compute_subset_metrics(real_rows, "Real M2")
            filled_metrics = _compute_subset_metrics(filled_rows, "Filled M2")

            auroc_label = "AUROC" if is_binary else "AUROC OvO weighted"
            auprc_label = "AUPRC" if is_binary else "AUPRC OvO weighted"

            lines.append(
                f"| Subset | N | Accuracy | Balanced Acc | {auroc_label} | {auprc_label} | MCC |"
            )
            lines.append(
                "|--------|---|----------|-------------|"
                + "-" * (len(auroc_label) + 2) + "|"
                + "-" * (len(auprc_label) + 2) + "|-----|"
            )

            for label, m in [
                ("Real M2 predictions", real_metrics),
                ("Filled M2 predictions", filled_metrics),
            ]:
                if m["n"] == 0:
                    lines.append(f"| {label} | 0 | N/A | N/A | N/A | N/A | N/A |")
                else:
                    lines.append(
                        f"| {label} | {m['n']} | {_fv(m.get('accuracy'))} "
                        f"| {_fv(m.get('balanced_accuracy'))} "
                        f"| {_fv(m.get('auroc'))} "
                        f"| {_fv(m.get('auprc'))} "
                        f"| {_fv(m.get('mcc'))} |"
                    )
            lines += [""]

            # Per-class breakdown of filled specimens' prediction quality
            if filled_rows:
                filled_diseases = [r["true_disease"] for r in filled_rows]
                filled_correct = [
                    r["true_disease"] == r["ensemble_predicted"] for r in filled_rows
                ]
                disease_counts: Dict[str, Dict[str, int]] = {}
                for disease, correct in zip(filled_diseases, filled_correct):
                    if disease not in disease_counts:
                        disease_counts[disease] = {"correct": 0, "total": 0}
                    disease_counts[disease]["total"] += 1
                    if correct:
                        disease_counts[disease]["correct"] += 1

                lines.append("### Per-Class Accuracy for Filled Specimens")
                lines.append("")
                lines.append("| Disease | Correct | Total | Accuracy |")
                lines.append("|---------|---------|-------|----------|")
                for disease in sorted(disease_counts.keys()):
                    d = disease_counts[disease]
                    acc_pct = f"{d['correct'] / d['total'] * 100:.1f}%" if d["total"] > 0 else "N/A"
                    lines.append(
                        f"| {disease} | {d['correct']} | {d['total']} | {acc_pct} |"
                    )
                lines += [""]

    # --- Abstained specimen details (moved to end of report) ---
    if any_abstentions:
        lines += ["---", ""]
        lines += ["## Abstained Specimens", ""]
        lines.append("| Fold | Specimen | Participant | Disease |")
        lines.append("|------|----------|-------------|---------|")
        for fr in all_fold_results:
            fold_id_val = fr["fold_id"]
            for detail in fr.get("test_abstained_details", []):
                lines.append(
                    f"| {fold_id_val} | {detail['specimen_label']} | "
                    f"{detail['participant_label']} | {detail['disease']} |"
                )
        lines += [""]

    return pad_md_tables("\n".join(lines))


def _log_comparison_table(
    ensemble_agg: Dict,
    base_model_agg: Dict[int, Dict],
    model_nums: List[int],
):
    """Log a comparison table of ensemble vs base model performance."""
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS COMPARISON")
    logger.info(
        "Note: AUROC/AUPRC/MCC on scored specimens only; "
        "accuracy penalized for abstentions."
    )
    logger.info(f"{'='*70}")

    header = f"{'Model':<25} {'Accuracy':>10} {'AUROC':>10} {'MCC':>10}"
    logger.info(header)
    logger.info("-" * 55)

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    def _get_metric_mean(agg, key):
        d = agg.get(key, {})
        return d.get("mean") if isinstance(d, dict) else None

    is_binary = "auroc_pooled" in ensemble_agg

    def _get_auroc(agg):
        if is_binary:
            return agg.get("auroc_pooled")
        return _get_metric_mean(agg, "auroc_ovo_weighted")

    # Ensemble
    acc = ensemble_agg.get("accuracy_global", _get_metric_mean(ensemble_agg, "accuracy_per_fold"))
    auroc = _get_auroc(ensemble_agg)
    mcc_mean = _get_metric_mean(ensemble_agg, "mcc")
    logger.info(f"{'Ensemble':<25} {_fmt(acc):>10} {_fmt(auroc):>10} {_fmt(mcc_mean):>10}")

    # Base models
    for num in model_nums:
        if num not in base_model_agg:
            logger.info(f"{'Model ' + str(num) + ' (abstained)':<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        agg = base_model_agg[num]
        acc = agg.get("accuracy_global", _get_metric_mean(agg, "accuracy_per_fold"))
        auroc = _get_auroc(agg)
        mcc_mean = _get_metric_mean(agg, "mcc")
        logger.info(f"{'Model ' + str(num):<25} {_fmt(acc):>10} {_fmt(auroc):>10} {_fmt(mcc_mean):>10}")


# ============================================================================
# CLI
# ============================================================================


def _log_final_summary(
    all_pair_summaries: Dict[str, Dict],
    base_output_dir: Path,
    base_model_dirs: Dict[int, Path],
    model_modes: Dict[int, str],
    training_times: Dict[int, str],
    model_suffixes: Dict[int, Optional[str]],
    embedding_dir: Optional[Path],
    args,
    elapsed_seconds: float,
) -> None:
    """Log a final console summary after all training is complete.

    Includes a cross-disease results table (ensemble + base models),
    key configuration, paths, and total elapsed time.

    Parameters
    ----------
    all_pair_summaries : Aggregated summary dict per pair key (or single entry
                         for multiclass/binary).
    base_output_dir    : Root output directory.
    base_model_dirs    : Base model artifact directories (without pair suffix).
    model_modes        : LOAD/TRAIN/RESUME mode per model.
    training_times     : Formatted elapsed time strings per model (from auto-training).
    model_suffixes     : Per-model folder suffixes (None if default).
    embedding_dir      : Model 3 embedding directory (or None).
    args               : Parsed CLI args.
    elapsed_seconds    : Total wall-clock time from start of main().
    """
    def _fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    def _mcc_mean(agg):
        d = agg.get("mcc", {})
        return d.get("mean") if isinstance(d, dict) else None

    # Use args.models (not base_models keys) so fully-abstained models
    # still appear in the summary with "N/A" instead of being silently dropped.
    first_summary = next(iter(all_pair_summaries.values()))
    model_nums = sorted(args.models)
    is_binary = "auroc_pooled" in first_summary.get("ensemble", {})

    def _get_auroc(agg):
        if is_binary:
            return agg.get("auroc_pooled")
        d = agg.get("auroc_ovo_weighted", {})
        return d.get("mean") if isinstance(d, dict) else None

    # ==========================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    # --- Cross-disease results table ---
    if len(all_pair_summaries) > 1:
        # Multi-binary: one row per disease
        logger.info("")
        logger.info("Results by disease (ensemble):")
        logger.info(
            f"  {'Disease':<30} {'Accuracy':>10} {'AUROC':>10} {'MCC':>10}"
        )
        logger.info("  " + "-" * 62)
        for pair_key, summary in all_pair_summaries.items():
            ens = summary.get("ensemble", {})
            disease_name = (
                pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
            )
            acc = _fmt(ens.get("accuracy_global"))
            auroc = _fmt(_get_auroc(ens))
            mcc = _fmt(_mcc_mean(ens))
            logger.info(f"  {disease_name:<30} {acc:>10} {auroc:>10} {mcc:>10}")

        # Base model results per disease
        for num in model_nums:
            logger.info("")
            logger.info(f"Results by disease (Model {num}):")
            logger.info(
                f"  {'Disease':<30} {'Accuracy':>10} {'AUROC':>10} {'MCC':>10}"
            )
            logger.info("  " + "-" * 62)
            for pair_key, summary in all_pair_summaries.items():
                bm = summary.get("base_models", {}).get(f"model{num}", {})
                disease_name = (
                    pair_key.split("_vs_")[0]
                    if "_vs_" in pair_key
                    else pair_key
                )
                acc = _fmt(bm.get("accuracy_global"))
                auroc = _fmt(_get_auroc(bm))
                mcc = _fmt(_mcc_mean(bm))
                logger.info(
                    f"  {disease_name:<30} {acc:>10} {auroc:>10} {mcc:>10}"
                )
    else:
        # Single pair (multiclass or binary): just re-log the comparison table
        pair_key = next(iter(all_pair_summaries))
        summary = all_pair_summaries[pair_key]
        ens = summary.get("ensemble", {})
        base_models = summary.get("base_models", {})
        base_model_agg = {
            int(k.replace("model", "")): v for k, v in base_models.items()
        }
        _log_comparison_table(ens, base_model_agg, model_nums)

    # --- Configuration ---
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Models:              {args.models}")
    logger.info(f"  Folds:               {args.fold_ids or 'all'}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    if args.model2_abstention_strategy != "ensemble_abstain":
        logger.info(f"  M2 abstention:       {args.model2_abstention_strategy}")
    for num in args.models:
        suffix = model_suffixes.get(num)
        mode = model_modes.get(num, "?")
        time_str = training_times.get(num)
        parts = [f"Model {num}: {mode}"]
        if suffix:
            parts.append(f"suffix={suffix}")
        if time_str:
            parts.append(f"trained in {time_str}")
        logger.info(f"  {', '.join(parts)}")

    # --- Paths ---
    logger.info("")
    logger.info("Paths:")
    logger.info(f"  Ensemble output:     {base_output_dir}")
    for num in args.models:
        logger.info(f"  Model {num} artifacts:  {base_model_dirs[num]}")
    if embedding_dir:
        logger.info(f"  Embeddings:          {embedding_dir}")

    # --- Elapsed time ---
    logger.info("")
    logger.info(
        f"Total elapsed time: {_format_elapsed_time(elapsed_seconds)}"
    )
    logger.info("=" * 70)


def _save_multi_binary_summary(
    base_output_dir: Path,
    all_pair_summaries: Dict[str, Dict],
    all_pair_fold_results: Dict[str, List[Dict]],
    pairs_to_train: List[Tuple[str, str]],
    reference_class: str,
    run_config: Optional[Dict] = None,
) -> None:
    """Save a cross-pair comparison summary for multi-binary ensemble training.

    Writes both a comprehensive Markdown report and a JSON summary to the base
    binary output directory (parent of all pair subdirectories).

    Parameters
    ----------
    base_output_dir        : Parent directory for all pair subdirectories.
    all_pair_summaries     : Aggregated summary dict per pair key.
    all_pair_fold_results  : Per-fold result dicts per pair key.
    pairs_to_train         : List of (disease, reference_class) tuples.
    reference_class        : Reference/negative class name.
    run_config             : Run configuration dict (from the first pair; per-pair
                             fields like disease_filter are excluded from display).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    def _fv(val, fmt=".4f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    def _mcc_mean(agg):
        d = agg.get("mcc", {})
        return d.get("mean") if isinstance(d, dict) else None

    # Determine which base model nums are present (from first pair's summary)
    first_summary = next(iter(all_pair_summaries.values()))
    model_nums = sorted(
        int(k.replace("model", ""))
        for k in first_summary.get("base_models", {})
    )

    # ======================================================================
    # Markdown
    # ======================================================================
    lines: List[str] = ["# Multi-Binary Ensemble Summary", ""]
    lines.append(f"**Summary generated**: {timestamp}")
    lines.append(f"**Reference class**: {reference_class}")
    lines.append(f"**Pairs included**: {len(pairs_to_train)}")
    lines.append(f"**Models included**: {', '.join(str(n) for n in model_nums)}")

    # Per-pair training timestamps (so reader can tell if pairs are from the same run)
    lines += ["", "### Pairs Included", ""]
    lines.append("| Pair | Training timestamp |")
    lines.append("|------|--------------------|")
    for pair_key, pair_summary in all_pair_summaries.items():
        pair_ts = pair_summary.get("timestamp", "unknown")
        lines.append(f"| {pair_key} | {pair_ts} |")

    # --- Run Configuration ---
    if run_config:
        lines += ["", "## Run Configuration", ""]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in run_config.items():
            # Skip per-pair fields that vary across disease pairs
            if k in ("disease_filter", "timestamp"):
                continue
            if k == "metamodel_config" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| metamodel.{mk} | {mv} |")
            elif k == "base_model_paths" and isinstance(v, dict):
                for mk, mv in v.items():
                    # Show base directory (strip pair subdirectory suffix)
                    base_path = str(Path(mv).parent) if "_vs_" in str(mv) else mv
                    lines.append(f"| base_model_path.{mk} | `{base_path}` |")
            elif k == "base_model_suffixes" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| base_model_suffix.{mk} | {mv or '(none)'} |")
            elif k == "base_model_configs" and isinstance(v, dict):
                # Summarize base model configs compactly
                for mk, mv in v.items():
                    if isinstance(mv, dict):
                        for ck, cv in mv.items():
                            if ck in ("timestamp", "fold_ids"):
                                continue
                            lines.append(f"| {mk}.{ck} | {cv} |")
                    else:
                        lines.append(f"| {mk} | {mv} |")
            else:
                lines.append(f"| {k} | {v} |")

    # --- Abstention Handling ---
    lines += ["", "## Abstention Handling", ""]
    lines.append(
        "Specimens for which any base model abstained (e.g., Model 2 found zero cluster "
        "matches) are excluded from AUROC, AUPRC, MCC, and log loss computation — these "
        "metrics are computed on scored specimens only. Accuracy includes abstentions as "
        "errors: `accuracy = n_correct / (n_scored + n_abstained)`. This matches the "
        "original Mal-ID crosseval `with_abstention=True` behavior."
    )
    lines.append("")

    total_scored = 0
    total_abstained = 0
    for fold_results in all_pair_fold_results.values():
        for fr in fold_results:
            em = fr.get("ensemble_metrics", {})
            total_scored += em.get("n_scored", 0)
            total_abstained += em.get("n_abstained", 0)
    total_specimens = total_scored + total_abstained
    overall_rate = total_abstained / total_specimens if total_specimens > 0 else 0.0
    lines.append(
        f"**Total scored**: {total_scored} | "
        f"**Total abstained**: {total_abstained} | "
        f"**Abstention rate**: {overall_rate:.2%}"
    )
    lines.append("")

    # Build per-disease abstention blocks (will be appended at end of report)
    abstention_blocks: List[str] = []
    for pair_key, fold_results in all_pair_fold_results.items():
        pair_abstained = [
            (fr["fold_id"], detail)
            for fr in fold_results
            for detail in fr.get("test_abstained_details", [])
        ]
        if not pair_abstained:
            continue
        disease_name = pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
        abstention_blocks.append(f"### {disease_name} ({len(pair_abstained)} abstained)")
        abstention_blocks.append("")
        abstention_blocks.append("| Fold | Specimen | Participant | Disease |")
        abstention_blocks.append("|------|----------|-------------|---------|")
        for fold_id_val, detail in pair_abstained:
            abstention_blocks.append(
                f"| {fold_id_val} | {detail['specimen_label']} | "
                f"{detail['participant_label']} | {detail['disease']} |"
            )
        abstention_blocks.append("")
    if abstention_blocks:
        lines.append(
            "See [Abstained Specimens by Disease Model]"
            "(#abstained-specimens-by-disease-model) at the end of "
            "this report for the full specimen list."
        )
        lines.append("")

    # Per-disease fill statistics (when a fill strategy was used)
    fill_blocks: List[str] = []
    for pair_key, fold_results in all_pair_fold_results.items():
        pair_test_filled = sum(
            fr.get("test_fill_info", {}).get("n_filled", 0)
            for fr in fold_results
        )
        pair_val_filled = sum(
            fr.get("val_fill_info", {}).get("n_filled", 0)
            for fr in fold_results
        )
        if pair_test_filled > 0 or pair_val_filled > 0:
            disease_name = pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
            strategy = "unknown"
            for fr in fold_results:
                fi = fr.get("test_fill_info", {})
                if fi.get("strategy"):
                    strategy = fi["strategy"]
                    break
            fill_blocks.append(
                f"| {disease_name} | {pair_test_filled} | {pair_val_filled} | `{strategy}` |"
            )
    if fill_blocks:
        lines += ["### Model 2 Fill Statistics", ""]
        lines.append("| Disease | Test Filled | Val Filled | Strategy |")
        lines.append("|---------|-------------|------------|----------|")
        lines += fill_blocks
        lines.append("")

    lines += ["---", ""]

    # --- Cross-Pair Comparison (Ensemble) ---
    lines += ["## Cross-Pair Comparison (Ensemble)", ""]
    lines.append(
        "| Disease | Accuracy | AUROC (pooled) | AUPRC (pooled) | MCC | Abstention |"
    )
    lines.append(
        "|---------|----------|----------------|----------------|-----|------------|"
    )
    for pair_key, summary in all_pair_summaries.items():
        ens = summary.get("ensemble", {})
        disease_name = pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
        acc = _fv(ens.get("accuracy_global"))
        auroc = _fv(ens.get("auroc_pooled"))
        auprc = _fv(ens.get("auprc_pooled"))
        mcc = _fv(_mcc_mean(ens))
        # Compute abstention rate for this pair
        pair_frs = all_pair_fold_results.get(pair_key, [])
        n_s = sum(fr.get("ensemble_metrics", {}).get("n_scored", 0) for fr in pair_frs)
        n_a = sum(fr.get("ensemble_metrics", {}).get("n_abstained", 0) for fr in pair_frs)
        abs_str = f"{n_a / (n_s + n_a):.1%}" if (n_s + n_a) > 0 else "N/A"
        lines.append(f"| {disease_name} | {acc} | {auroc} | {auprc} | {mcc} | {abs_str} |")
    lines += [""]

    # --- Cross-Pair Comparison (Base Models) ---
    if model_nums:
        lines += ["## Cross-Pair Comparison (Base Models)", ""]
        for num in model_nums:
            lines += [f"### Model {num}", ""]
            lines.append(
                "| Disease | Accuracy | AUROC (pooled) | AUPRC (pooled) | MCC |"
            )
            lines.append(
                "|---------|----------|----------------|----------------|-----|"
            )
            for pair_key, summary in all_pair_summaries.items():
                bm = summary.get("base_models", {}).get(f"model{num}", {})
                disease_name = (
                    pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
                )
                acc = _fv(bm.get("accuracy_global"))
                auroc = _fv(bm.get("auroc_pooled"))
                auprc = _fv(bm.get("auprc_pooled"))
                mcc = _fv(_mcc_mean(bm))
                lines.append(
                    f"| {disease_name} | {acc} | {auroc} | {auprc} | {mcc} |"
                )
            lines += [""]

    lines += ["---", ""]

    # --- Per-Disease Detail ---
    lines += ["## Per-Disease Detail", ""]

    for pair_key, summary in all_pair_summaries.items():
        ens = summary.get("ensemble", {})
        base_models = summary.get("base_models", {})
        fold_results = all_pair_fold_results.get(pair_key, [])
        disease_name = pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key
        disease_filter = summary.get("disease_filter")
        ref_display = disease_filter[1] if disease_filter else reference_class

        lines += [f"### {disease_name} vs {ref_display}", ""]

        # -- Model comparison table --
        lines += ["#### Model Comparison", ""]
        lines.append(
            "| Model | Accuracy (global) | AUROC (pooled) | AUPRC (pooled) | MCC |"
        )
        lines.append(
            "|-------|-------------------|----------------|----------------|-----|"
        )
        for label, agg in [("**Ensemble**", ens)] + [
            (f"Model {num}", base_models.get(f"model{num}", {}))
            for num in model_nums
        ]:
            acc = _fv(agg.get("accuracy_global"))
            auroc = _fv(agg.get("auroc_pooled"))
            auprc = _fv(agg.get("auprc_pooled"))
            mcc = _fv(_mcc_mean(agg))
            lines.append(f"| {label} | {acc} | {auroc} | {auprc} | {mcc} |")
        lines += [""]

        # -- Ensemble per-fold results --
        if fold_results:
            lines += ["#### Ensemble Per-Fold Results", ""]
            lines.append(
                "| Fold | Accuracy | AUROC | AUPRC | MCC | N scored | N abstained |"
            )
            lines.append(
                "|------|----------|-------|-------|-----|----------|-------------|"
            )
            for fr in fold_results:
                em = fr.get("ensemble_metrics", {})
                lines.append(
                    f"| {em.get('fold_id', '?')} | "
                    f"{_fv(em.get('accuracy'))} | "
                    f"{_fv(em.get('auroc_binary'))} | "
                    f"{_fv(em.get('auprc_binary'))} | "
                    f"{_fv(em.get('mcc'))} | "
                    f"{em.get('n_scored', '?')} | "
                    f"{em.get('n_abstained', 0)} |"
                )
            lines += [""]

        # -- Base model per-fold results --
        for num in model_nums:
            if not fold_results:
                continue
            lines += [f"#### Model {num} Per-Fold Results", ""]
            lines.append("| Fold | Accuracy | AUROC | AUPRC | MCC |")
            lines.append("|------|----------|-------|-------|-----|")
            for fr in fold_results:
                bm = fr.get("base_model_metrics", {}).get(num, {})
                lines.append(
                    f"| {bm.get('fold_id', '?')} | "
                    f"{_fv(bm.get('accuracy'))} | "
                    f"{_fv(bm.get('auroc_binary'))} | "
                    f"{_fv(bm.get('auprc_binary'))} | "
                    f"{_fv(bm.get('mcc'))} |"
                )
            lines += [""]

        # -- Confusion matrix (ensemble, aggregated) --
        cm = ens.get("confusion_matrix_aggregated")
        classes = ens.get("classes", [])
        if cm and classes:
            lines += ["#### Per-Class Accuracy (Ensemble)", ""]
            lines.append("| Class | Correct | Total | Accuracy |")
            lines.append("|-------|---------|-------|----------|")
            for i, cls in enumerate(classes):
                total = sum(cm[i])
                correct = cm[i][i]
                acc_pct = f"{correct / total * 100:.1f}%" if total > 0 else "N/A"
                lines.append(f"| {cls} | {correct} | {total} | {acc_pct} |")
            lines += [""]

            lines += ["#### Confusion Matrix (Ensemble)", ""]
            lines.append("| | " + " | ".join(str(c) for c in classes) + " |")
            lines.append("|-" + "-|-".join("---" for _ in classes) + "-|")
            for i, cls in enumerate(classes):
                row_vals = " | ".join(str(cm[i][j]) for j in range(len(classes)))
                lines.append(f"| **{cls}** | {row_vals} |")
            lines += [""]

    # --- Abstained specimen details (at end of report) ---
    if abstention_blocks:
        lines += ["---", ""]
        lines += ["## Abstained Specimens by Disease Model", ""]
        lines += abstention_blocks

    lines += ["---", "", "*Generated by ensemble training script*", ""]

    md_path = base_output_dir / f"MULTI_BINARY_SUMMARY_{timestamp}.md"
    md_path.write_text(pad_md_tables("\n".join(lines)))
    logger.info(f"\nSaved multi-binary summary: {md_path}")

    # ======================================================================
    # JSON
    # ======================================================================
    cross_summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "reference_class": reference_class,
        "n_pairs": len(pairs_to_train),
        "models_included": model_nums,
        "total_scored": total_scored,
        "total_abstained": total_abstained,
        "overall_abstention_rate": overall_rate,
        "pairs": {},
    }
    for pair_key, summary in all_pair_summaries.items():
        ens = summary.get("ensemble", {})
        base_models = summary.get("base_models", {})
        pair_fold_results = all_pair_fold_results.get(pair_key, [])
        pair_abstained = [
            detail
            for fr in pair_fold_results
            for detail in fr.get("test_abstained_details", [])
        ]
        pair_entry: Dict[str, Any] = {
            "training_timestamp": summary.get("timestamp", "unknown"),
            "ensemble": {
                "accuracy_global": ens.get("accuracy_global"),
                "auroc_pooled": ens.get("auroc_pooled"),
                "auprc_pooled": ens.get("auprc_pooled"),
                "mcc_mean": _mcc_mean(ens),
            },
            "base_models": {},
            "n_abstained": len(pair_abstained),
            "abstained_specimens": pair_abstained,
            "n_test_filled": sum(
                fr.get("test_fill_info", {}).get("n_filled", 0)
                for fr in pair_fold_results
            ),
            "n_val_filled": sum(
                fr.get("val_fill_info", {}).get("n_filled", 0)
                for fr in pair_fold_results
            ),
        }
        for num in model_nums:
            bm = base_models.get(f"model{num}", {})
            pair_entry["base_models"][f"model{num}"] = {
                "accuracy_global": bm.get("accuracy_global"),
                "auroc_pooled": bm.get("auroc_pooled"),
                "auprc_pooled": bm.get("auprc_pooled"),
                "mcc_mean": _mcc_mean(bm),
            }
        cross_summary["pairs"][pair_key] = pair_entry

    json_path = base_output_dir / f"multi_binary_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(cross_summary, f, indent=2, default=_json_default)
    logger.info(f"Saved multi-binary summary JSON: {json_path}")


def _run_from_feature_matrices(
    args,
    clone_id_kwargs: Optional[Dict] = None,
    training_context: str = "cv_ensemble",
) -> None:
    """Handle --feature-matrices-dir mode: train metamodel from external feature matrices.

    Loads pre-computed feature matrices from the specified directory, validates
    configuration against the source run, and trains only the ensemble metamodel
    layer (skipping all base model training).

    Works for both CV (per-fold, fold-prefixed matrices) and train-all (a single
    prefix-less ``feature_matrix_raw_val.csv``, no test side). ``training_context``
    is the RESOLVED context ("cv_ensemble" / "train_all_ensemble") from the CLI;
    the source run's own ``training_context`` must match it, else we error (a CV
    source under a train-all run, or vice versa, would be nonsensical).

    For multiclass/binary sources, the source directory directly contains
    run_config.json and fold feature matrix files. For multi-binary sources,
    the source directory contains pair subdirectories (Disease_vs_Reference),
    each with their own run_config.json and feature matrices.

    Parameters
    ----------
    args : argparse.Namespace with at least feature_matrices_dir, metadata_path,
        model2_abstention_strategy, output_dir, output_suffix, dataset_name,
        fold_ids, cache_dir, verbose, n_jobs.
    clone_id_kwargs : Dict of clone_id parameters for the data loader
        (from get_clone_id_kwargs). None means all params unspecified —
        cached values accepted as-is.
    """
    import re as _re

    # Resolve cache dir default (same as main path)
    if args.cache_dir is None:
        args.cache_dir = PROJECT_ROOT / "cache" / args.dataset_name

    source_dir = args.feature_matrices_dir

    # --- Load source run configuration ---
    # For multi-binary, run_config.json is in each pair subdir, not at the base.
    src_config_path = source_dir / "run_config.json"
    pair_subdirs = []

    if src_config_path.exists():
        with open(src_config_path) as f:
            source_config = json.load(f)
        is_multi_binary_base = False
    else:
        # Multi-binary: discover pair subdirs (already validated in validate_ensemble_args)
        pair_subdirs = sorted([
            d for d in source_dir.iterdir()
            if d.is_dir() and "_vs_" in d.name
            and (d / "run_config.json").exists()
        ])
        assert pair_subdirs, (
            f"No run_config.json found in {source_dir} or pair subdirectories. "
            f"This should have been caught by validate_ensemble_args."
        )
        with open(pair_subdirs[0] / "run_config.json") as f:
            source_config = json.load(f)
        is_multi_binary_base = True

        # Verify all pair configs are consistent on shared keys
        _consistency_keys = ["gene_locus", "models_included", "reference_class",
                             "classification_mode"]
        for d in pair_subdirs[1:]:
            with open(d / "run_config.json") as f:
                other_cfg = json.load(f)
            mismatches = []
            for key in _consistency_keys:
                val_first = source_config.get(key)
                val_other = other_cfg.get(key)
                if val_first != val_other:
                    mismatches.append(
                        f"  {key}: {pair_subdirs[0].name} has {val_first!r}, "
                        f"{d.name} has {val_other!r}"
                    )
            if mismatches:
                logger.error(
                    f"Inconsistent run_config.json across pair subdirectories:\n"
                    + "\n".join(mismatches)
                    + "\nAll pairs must have been trained with the same settings."
                )
                sys.exit(1)

    # --- Extract source config values ---
    src_classification_mode = source_config["classification_mode"]
    src_gene_locus = source_config["gene_locus"]
    src_models = source_config["models_included"]
    src_diseases = source_config.get("diseases")
    src_reference_class = source_config.get("reference_class")
    src_disease_filter = source_config.get("disease_filter")
    src_strategy = source_config.get("model2_abstention_strategy", "ensemble_abstain")

    # --- Validate the source's training context matches the requested one ---
    # Legacy sources (pre-Phase-5) have no training_context key → they are CV.
    src_training_context = source_config.get("training_context", "cv_ensemble")
    if src_training_context != training_context:
        _short = {"cv_ensemble": "cv", "train_all_ensemble": "train_all"}
        logger.error(
            f"--feature-matrices-dir source was trained as {src_training_context!r} "
            f"but --training-context is {_short.get(training_context, training_context)!r} "
            f"({training_context!r}).\n"
            f"The source feature matrices and the requested ensemble must be the same "
            f"context. Pass --training-context "
            f"{_short.get(src_training_context, src_training_context)!r} to match the source, "
            f"or point to a matching source directory."
        )
        sys.exit(1)
    is_train_all = training_context == "train_all_ensemble"

    logger.info(f"\n{'='*70}")
    logger.info("FEATURE MATRICES MODE")
    logger.info(f"{'='*70}")
    logger.info(f"  Source directory:     {source_dir}")
    logger.info(f"  Classification mode:  {src_classification_mode}")
    logger.info(f"  Gene locus:           {src_gene_locus}")
    logger.info(f"  Models:               {src_models}")
    logger.info(f"  Source M2 strategy:   {src_strategy}")
    if is_multi_binary_base:
        logger.info(f"  Multi-binary pairs:   {len(pair_subdirs)}")
        for d in pair_subdirs:
            logger.info(f"    {d.name}")

    # --- Validate CLI args against source config ---
    # In --feature-matrices-dir mode, classification-mode, gene-locus, and models
    # are determined by the source run. Error if user explicitly passed conflicting values.
    _cli_conflicts = []
    if "--classification-mode" in sys.argv:
        if args.classification_mode != src_classification_mode:
            _cli_conflicts.append(
                f"  --classification-mode: passed {args.classification_mode!r}, "
                f"but source is {src_classification_mode!r}"
            )
    if "--gene-locus" in sys.argv:
        if args.gene_locus != src_gene_locus:
            _cli_conflicts.append(
                f"  --gene-locus: passed {args.gene_locus!r}, "
                f"but source is {src_gene_locus!r}"
            )
    if "--models" in sys.argv:
        if sorted(args.models) != sorted(src_models):
            _cli_conflicts.append(
                f"  --models: passed {args.models}, "
                f"but source used {src_models}"
            )
    if _cli_conflicts:
        logger.error(
            "CLI arguments conflict with source run configuration.\n"
            "In --feature-matrices-dir mode, these values are determined by the source run:\n"
            + "\n".join(_cli_conflicts)
            + "\nRemove the conflicting arguments to use the source configuration, "
            "or use a different source directory."
        )
        sys.exit(1)

    # --- Determine effective model2_abstention_strategy ---
    # If user explicitly passed --model2-abstention-strategy, use theirs.
    # Otherwise, default to the source run's strategy.
    user_specified_strategy = "--model2-abstention-strategy" in sys.argv
    if user_specified_strategy:
        effective_strategy = args.model2_abstention_strategy
        if effective_strategy != src_strategy:
            logger.info(
                f"  Strategy override:    {src_strategy!r} -> {effective_strategy!r}"
            )
    else:
        effective_strategy = src_strategy
        if effective_strategy != "ensemble_abstain":
            # Source used a non-default strategy; inform user we're inheriting it
            logger.info(
                f"  Using source strategy: {effective_strategy!r} "
                f"(pass --model2-abstention-strategy to override)"
            )

    # Validate fill_models13_mean requires Models 1 and 3
    if effective_strategy == "fill_models13_mean":
        if 1 not in src_models or 3 not in src_models:
            logger.error(
                f"Strategy fill_models13_mean requires Models 1 and 3, "
                f"but source models are {src_models}."
            )
            sys.exit(1)
    # Validate fill strategy requires Model 2
    if effective_strategy != "ensemble_abstain" and 2 not in src_models:
        logger.error(
            f"Strategy {effective_strategy} only makes sense when Model 2 "
            f"is included, but source models are {src_models}."
        )
        sys.exit(1)

    # --- Resolve metadata path ---
    if args.metadata_path is not None:
        metadata_path = args.metadata_path
        logger.info(f"  Metadata (user):      {metadata_path}")
    else:
        # Try source config's resolved path first, then original path
        src_meta_resolved = source_config.get("metadata_resolved_path")
        src_meta_original = source_config.get("metadata_path")
        metadata_path = None
        for candidate in [src_meta_resolved, src_meta_original]:
            if candidate is not None:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    metadata_path = candidate_path
                    break
        if metadata_path is None:
            logger.error(
                f"Cannot resolve metadata path from source config.\n"
                f"  metadata_resolved_path: {src_meta_resolved}\n"
                f"  metadata_path: {src_meta_original}\n"
                f"Neither exists. Provide --metadata-path explicitly."
            )
            sys.exit(1)
        logger.info(f"  Metadata (source):    {metadata_path}")

    # --- Initialize data loader ---
    # data_dir=None: metadata-only mode. No raw data loading needed — we only
    # use loader.metadata for specimen-to-participant mapping.
    loader = MalIDPublishedDataLoader(
        data_dir=None,
        metadata_path=metadata_path,
        gene_locus=src_gene_locus,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
        **(clone_id_kwargs or {}),
    )

    # --- Discover fold IDs from feature matrix files ---
    discovery_dir = pair_subdirs[0] if is_multi_binary_base else source_dir

    if is_train_all:
        # Train-all: a single prefix-less feature_matrix_raw_val.csv (no folds).
        if not (discovery_dir / "feature_matrix_raw_val.csv").exists():
            logger.error(
                f"No train-all validation feature matrix found in {discovery_dir}. "
                f"Expected feature_matrix_raw_val.csv."
            )
            sys.exit(1)
        fold_ids = [None]  # sentinel single pass (unused downstream for train-all)
        logger.info("  Train-all: single whole-dataset feature matrix (no folds)")
    else:
        # Pattern: fold_<id>_feature_matrix_raw_val.csv or fold_<id>_feature_matrix_val.csv
        available_folds = set()
        for f in discovery_dir.glob("fold_*_feature_matrix_*val.csv"):
            match = _re.match(r"fold_(\d+)_feature_matrix_", f.name)
            if match:
                available_folds.add(int(match.group(1)))

        if not available_folds:
            logger.error(
                f"No feature matrix files found in {discovery_dir}. "
                f"Expected fold_*_feature_matrix_*val.csv files."
            )
            sys.exit(1)

        available_folds = sorted(available_folds)

        # Filter by --fold-ids if specified
        if args.fold_ids is not None:
            invalid = [f for f in args.fold_ids if f not in available_folds]
            if invalid:
                logger.error(
                    f"Requested fold IDs {invalid} not found in source directory. "
                    f"Available folds: {available_folds}"
                )
                sys.exit(1)
            fold_ids = args.fold_ids
        else:
            fold_ids = available_folds

        logger.info(f"  Fold IDs:             {fold_ids}")

    # --- Resolve output directory ---
    if args.output_dir is not None:
        base_output_dir = args.output_dir
    else:
        base_output_dir = get_ensemble_output_dir(
            dataset_name=args.dataset_name,
            classification_mode=src_classification_mode,
            gene_locus=src_gene_locus,
            output_suffix=args.output_suffix,
            training_context=training_context,
        )
    logger.info(f"  Output:               {base_output_dir}")

    # --- Resolve disease pairs from source config ---
    reference_class = src_reference_class

    if src_classification_mode == "multiclass":
        pairs_to_train = [None]
    elif src_classification_mode == "binary":
        if src_disease_filter is None:
            logger.error(
                "Source config is binary mode but has no disease_filter. "
                "Cannot determine disease pair."
            )
            sys.exit(1)
        pairs_to_train = [tuple(src_disease_filter)]
    elif src_classification_mode == "multi-binary":
        if is_multi_binary_base:
            # Discover pairs from each subdir's run_config.json (NOT from directory
            # names — make_pair_name sanitizes characters like / → _, so parsing
            # back would produce wrong class names e.g. "Healthy_Background"
            # instead of "Healthy/Background").
            pairs_to_train = []
            for d in pair_subdirs:
                with open(d / "run_config.json") as f:
                    pair_cfg = json.load(f)
                pair_filter = pair_cfg.get("disease_filter")
                if pair_filter and len(pair_filter) == 2:
                    pairs_to_train.append(tuple(pair_filter))
                else:
                    logger.warning(
                        f"Cannot extract disease_filter from {d.name}/run_config.json "
                        f"(got {pair_filter!r}), skipping"
                    )
            if not pairs_to_train:
                logger.error(
                    f"No valid pair subdirectories found in {source_dir}."
                )
                sys.exit(1)
        else:
            # User pointed to a single pair subdir directly
            if src_disease_filter is None:
                logger.error(
                    "Source config is multi-binary but has no disease_filter. "
                    "Cannot determine disease pair."
                )
                sys.exit(1)
            pairs_to_train = [tuple(src_disease_filter)]
    else:
        logger.error(
            f"Unknown classification mode in source config: {src_classification_mode!r}"
        )
        sys.exit(1)

    # --- Dataset counts ---
    dataset_counts = get_metadata_class_counts(loader.metadata)
    metadata_filter_info = loader.metadata_filter_info

    # --- Add file handler for logging ---
    base_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_output_dir / "ensemble_training.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_path}")

    # --- Train each pair ---
    all_pair_summaries = {}
    all_pair_fold_results = {}
    first_run_config = None

    for pair in pairs_to_train:
        if pair is None:
            # Multiclass
            pair_output_dir = base_output_dir
            disease_filter = None
            ref_class = None
            pair_key = "multiclass"
            pair_source_dir = source_dir
        else:
            disease, ref = pair
            pair_key = make_pair_name(disease, ref)
            disease_filter = pair
            ref_class = ref
            # Binary/multi-binary always use pair subdirectory (matches normal flow)
            pair_output_dir = base_output_dir / pair_key
            if is_multi_binary_base:
                pair_source_dir = source_dir / pair_key
            else:
                pair_source_dir = source_dir

        if src_classification_mode == "multi-binary":
            logger.info(f"\n{'*'*60}")
            logger.info(f"Binary pair: {pair_key}")
            logger.info(f"{'*'*60}")

        # Load pair-specific source config if multi-binary
        if is_multi_binary_base:
            with open(pair_source_dir / "run_config.json") as f:
                pair_source_config = json.load(f)
        else:
            pair_source_config = source_config

        # Build run config for this pair (records provenance)
        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "training_context": training_context,
            "dataset_name": args.dataset_name,
            "classification_mode": src_classification_mode,
            "gene_locus": src_gene_locus,
            # Resolved clone_id clustering definition (Phase 6.E cross-dataset check).
            "clone_id_params": loader.clone_id_params,
            "models_included": src_models,
            "fold_ids": fold_ids,
            "reference_class": ref_class,
            "diseases": src_diseases,
            "disease_filter": list(disease_filter) if disease_filter else None,
            "output_suffix": args.output_suffix,
            "resume": False,
            "feature_matrices_dir": str(source_dir),
            "model2_abstention_strategy": effective_strategy,
            "source_model2_abstention_strategy": src_strategy,
            "dataset_counts": dataset_counts,
            "metadata_filter_info": metadata_filter_info,
            "metadata_path": str(args.metadata_path) if args.metadata_path else None,
            "metadata_resolved_path": str(loader.metadata_path),
            "base_model_paths": pair_source_config.get("base_model_paths"),
            "base_model_suffixes": pair_source_config.get("base_model_suffixes"),
            "embedding_dir": pair_source_config.get("embedding_dir"),
            "metamodel_config": pair_source_config.get("metamodel_config"),
            "base_model_training_mode": {
                f"model{num}": "external_features"
                for num in src_models
            },
            "base_model_training_params": pair_source_config.get("base_model_training_params"),
            "base_model_training_times": None,
            "base_model_configs": pair_source_config.get("base_model_configs"),
        }

        if is_train_all:
            # Train-all: load the source's prefix-less raw-val matrix and retrain
            # the metamodel only (no base models, no test/metrics).
            summary = _run_train_all_ensemble(
                pair_output_dir,
                loader=loader,
                model_nums=src_models,
                model_dirs={},
                gene_locus=src_gene_locus,
                embedding_dir=None,
                disease_filter=disease_filter,
                reference_class=ref_class,
                model_summaries=None,
                n_jobs=args.n_jobs,
                metamodel_cv_n_splits=args.metamodel_cv_n_splits,
                model2_abstention_strategy=effective_strategy,
                resume=False,
                run_config=run_config,
                source_dir=pair_source_dir,
            )
            all_pair_summaries[pair_key] = summary
            all_pair_fold_results[pair_key] = []
        else:
            fold_results, summary = train_ensemble(
                loader=loader,
                fold_ids=fold_ids,
                model_nums=src_models,
                model_dirs={},
                gene_locus=src_gene_locus,
                output_dir=pair_output_dir,
                disease_filter=disease_filter,
                reference_class=ref_class,
                run_config=run_config,
                model_summaries=None,
                n_jobs=args.n_jobs,
                resume=False,
                metamodel_cv_n_splits=args.metamodel_cv_n_splits,
                model2_abstention_strategy=effective_strategy,
                source_dir=pair_source_dir,
            )
            all_pair_summaries[pair_key] = summary
            all_pair_fold_results[pair_key] = fold_results
        if first_run_config is None:
            first_run_config = run_config

    # --- Multi-binary cross-pair summary (metrics-based → CV only) ---
    if src_classification_mode == "multi-binary" and not is_train_all:
        if len(all_pair_summaries) > 1:
            _save_multi_binary_summary(
                base_output_dir, all_pair_summaries, all_pair_fold_results,
                pairs_to_train, reference_class, run_config=first_run_config,
            )
        else:
            logger.info(
                "Single disease pair — skipping cross-pair comparison summary. "
                "Per-pair results are in the pair subdirectory."
            )

    logger.info(f"\nDone. Output: {base_output_dir}")


def main():
    """Parse CLI arguments and orchestrate ensemble training.

    Handles three modes per base model (LOAD / TRAIN / RESUME),
    auto-trains base models as needed, then trains the metamodel
    (ensemble) on their combined predictions. Supports both
    multiclass and multi-binary classification.
    """
    parser = argparse.ArgumentParser(
        description="Train ensemble (metamodel) for Mal-ID-Lite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Data paths ---
    parser.add_argument(
        "--metadata-path", type=Path, default=None,
        help="Path to metadata.tsv (optional if cache has a copy).",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=None,
        help="Cache directory with preprocessed data. "
             "Default: cache/<dataset-name>/ under the project root.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to AIRR data directory (optional when using cache).",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="mal-id-orig-data",
        help="Dataset name for output directory structure.",
    )

    # --- Training context (CV vs whole-dataset train-all) ---
    parser.add_argument(
        "--training-context", type=str, required=True,
        choices=["cv", "train_all"],
        help=(
            "REQUIRED. Which ensemble to train:\n"
            "  cv        — cross-validation ensemble (per-fold; reports test metrics). "
            "Maps internally to 'cv_ensemble'.\n"
            "  train_all — whole-dataset ensemble for later evaluation on a SEPARATE "
            "dataset (no test set, no metrics; trains the metamodel on the validation "
            "third of the data). Maps internally to 'train_all_ensemble'.\n"
            "No default: the two are very different long-running jobs, so intent must be "
            "explicit (existing CV commands must now pass --training-context cv)."
        ),
    )

    # --- Classification mode ---
    parser.add_argument(
        "--classification-mode", type=str, default="multiclass",
        choices=["multiclass", "binary", "multi-binary"],
        help="Classification mode. multi-binary trains one ensemble per disease vs reference.",
    )
    parser.add_argument(
        "--reference-class", type=str, default="Healthy/Background",
        help="Reference/negative class. Required for binary and multi-binary modes.",
    )
    parser.add_argument(
        "--diseases", nargs="+", type=str, default=None,
        help=(
            "Disease classes to include (default: all from metadata). "
            "binary: one disease name (optional for 2-class datasets — the non-reference "
            "class is auto-detected; required for N-class datasets to pick one disease). "
            "multi-binary: one or more disease names."
        ),
    )

    # --- Model selection ---
    parser.add_argument(
        "--models", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which base models to include (default: 1 2 3).",
    )
    parser.add_argument(
        "--gene-locus", type=str, default="TCR", choices=["TCR", "BCR"],
        help="Gene locus (default: TCR).",
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, default=None,
        help=(
            "Fold(s) to hold out as the test set (default: all folds found in "
            "metadata). For each fold listed, the model is trained from scratch "
            "on all other folds pooled together, then evaluated on that held-out "
            "fold. CV context only -- rejected under --training-context train_all."
        ),
    )

    # --- Model-specific suffixes ---
    parser.add_argument(
        "--model1-suffix", type=str, default=None,
        help="Output suffix for Model 1 artifacts.",
    )
    parser.add_argument(
        "--model2-suffix", type=str, default=None,
        help="Output suffix for Model 2 artifacts.",
    )
    parser.add_argument(
        "--model3-suffix", type=str, default=None,
        help="Output suffix for Model 3 artifacts.",
    )

    # --- Model 3 specific (embeddings) ---
    parser.add_argument(
        "--model3-embedding-dir", type=Path, default=None,
        help="Directory with pre-computed ESM-2 embeddings. "
             "Defaults to cache_dir/embeddings.",
    )

    # --- Base model retrain control ---
    parser.add_argument(
        "--retrain-base-models", action="store_true",
        help="Force retrain ALL included base models (ignores existing artifacts).",
    )
    parser.add_argument(
        "--retrain-models", nargs="+", type=int, default=None,
        choices=[1, 2, 3],
        help="Force retrain specific base models (e.g., --retrain-models 1 3).",
    )

    # --- Model 1 training parameters (None = not specified, use model default) ---
    parser.add_argument(
        "--model1-n-pcs", type=int, default=None,
        help="Number of PCA components for Model 1 (default in train_model1: 15).",
    )
    parser.add_argument(
        "--model1-l1-ratio", type=float, default=None,
        help="Elastic net L1/L2 ratio for Model 1 (default in train_model1: None -> 1.0 for TCR).",
    )
    parser.add_argument(
        "--model1-model-name", type=str, default=None,
        help="Model variant label for Model 1 (default in train_model1: 'lasso_cv').",
    )

    # --- Model 2 training parameters (None = not specified, use model default) ---
    parser.add_argument(
        "--model2-p-values", nargs="+", type=float, default=None,
        help="P-value candidates for Model 2 threshold grid search "
             "(default in train_model2: [0.0005, 0.001, 0.005, 0.01, 0.05]).",
    )
    parser.add_argument(
        "--model2-retrain-on-full-train", action=argparse.BooleanOptionalAction,
        default=None,
        help="Retrain Model 2 final GLM on ts1+ts2 combined "
             "(default in train_model2: False). Use --no-model2-retrain-on-full-train to disable.",
    )
    parser.add_argument(
        "--model2-sequence-identity-threshold", type=float, default=None,
        help="CDR3 clustering identity threshold for Model 2 "
             "(default in train_model2: per-locus constant).",
    )

    # --- Model 3 training parameters (None = not specified, use model default) ---
    parser.add_argument(
        "--model3-aggregation-strategy", type=str, default=None,
        choices=["auto_tuned", "paper_best"] + [s.name for s in AggregationStrategy],
        help="Aggregation strategy for Model 3 (default in train_model3: 'entropy_percentile_cutoff').",
    )
    parser.add_argument(
        "--model3-n-estimators-stage1", type=int, default=None,
        help="Number of RF trees in Model 3 Stage 1 (default: 100).",
    )
    parser.add_argument(
        "--model3-n-estimators-stage2", type=int, default=None,
        help="Number of RF trees in Model 3 Stage 2 (default: 100).",
    )
    parser.add_argument(
        "--model3-entropy-max-fraction", type=float, default=None,
        help="Fraction of max possible entropy as cutoff for Model 3 "
             "entropy_cutoff strategy (default in train_model3: 0.80).",
    )
    parser.add_argument(
        "--model3-entropy-bottom-percentile", type=float, default=None,
        help="Percentile of training entropy distribution for Model 3 "
             "entropy_percentile_cutoff strategy (default in train_model3: 0.01).",
    )
    parser.add_argument(
        "--model3-tuning-strategies", type=str, default=None,
        help="Comma-separated strategies for Model 3 auto-tuning grid.",
    )
    parser.add_argument(
        "--model3-tuning-cv-splits", type=int, default=None,
        help="Number of inner CV folds for Model 3 auto-tuning (default: 3).",
    )
    parser.add_argument(
        "--model3-tuning-entropy-max-fractions", type=str, default=None,
        help="Comma-separated max_fraction values for Model 3 tuning grid.",
    )
    parser.add_argument(
        "--model3-tuning-entropy-percentiles", type=str, default=None,
        help="Comma-separated percentile values for Model 3 tuning grid.",
    )
    parser.add_argument(
        "--model3-no-cache-embeddings", action="store_true",
        help="Don't save newly computed embeddings for Model 3 (cached embeddings "
             "are still used when available). Without this flag, missing embeddings "
             "are auto-computed and saved to disk.",
    )
    parser.add_argument(
        "--model3-device", type=str, default=None,
        help="Device for ESM-2 embedding: 'cuda', 'mps', 'cpu', or auto.",
    )
    parser.add_argument(
        "--model3-embedding-batch-size", type=int, default=None,
        help="Batch size for Model 3 ESM-2 embedding computation (default: 64).",
    )

    # --- Gene reference ---
    parser.add_argument(
        "--gene-reference-path", type=Path, default=None,
        help="Path to V-gene CDR reference file (e.g., tcrb_v_gene_cdrs.generated.tsv). "
             "Optional; passed to base model training when auto-training.",
    )

    # --- Ensemble output ---
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help="Suffix for ensemble output directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (ignores canonical path).",
    )

    # --- Resume ---
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume the entire pipeline. For base models: partially-trained models "
            "resume from completed folds (skip re-training); fully-trained models "
            "are loaded as-is. For metamodel: retrains from saved feature matrices "
            "if available. Without --resume, partial base model artifacts are "
            "overwritten (fresh start)."
        ),
    )

    # --- External feature matrices ---
    parser.add_argument(
        "--feature-matrices-dir", type=Path, default=None,
        help=(
            "Load pre-computed feature matrices from this directory and train only "
            "the ensemble metamodel layer. Skips all base model training. "
            "The directory must contain fold_*_feature_matrix_raw_*.csv files, "
            "fold_*_ensemble_results.json, and run_config.json from a previous run. "
            "Mutually exclusive with --resume, --retrain-models, --retrain-base-models."
        ),
    )

    # --- Model 2 abstention handling ---
    parser.add_argument(
        "--model2-abstention-strategy", type=str,
        default="ensemble_abstain",
        choices=list(MODEL2_ABSTENTION_STRATEGIES),
        help=(
            "How to handle Model 2 abstentions (specimens with zero cluster matches). "
            "ensemble_abstain (default): drop specimen from ensemble (original behavior). "
            "fill_0.5: fill Model 2's features with 0.5 (uninformative prior). "
            "fill_models13_mean: fill with mean of Models 1 and 3 predictions per class "
            "(requires --models to include both 1 and 3)."
        ),
    )

    # --- Runtime ---
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help=(
            "Number of parallel workers for Model 2 cluster assignment "
            "and Model 3 V-gene group predictions. "
            "Set to 1 to disable parallelism. Default: 4."
        ),
    )
    parser.add_argument(
        "--metamodel-cv-n-splits", type=int, default=5,
        help=(
            "Number of folds for the metamodel's internal StratifiedGroupKFold "
            "(the ridge meta-learner's own cross-validation for lambda selection). "
            "Default 5 (matching original Mal-ID); auto-capped down if a class has "
            "fewer participants. Lower it (e.g. 2-3) for small datasets. Applies to "
            "both cv and train_all contexts."
        ),
    )
    parser.add_argument("--verbose", type=int, default=1)

    add_clone_id_args(parser)

    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    t_main_start = time.monotonic()

    # --- Resolve training context (map short CLI value → internal name) ---
    # 'cv' → 'cv_ensemble', 'train_all' → 'train_all_ensemble'. The redundant
    # '_ensemble' suffix never appears on the CLI (everything here is an ensemble).
    training_context = "train_all_ensemble" if args.training_context == "train_all" else "cv_ensemble"
    is_train_all = training_context == "train_all_ensemble"

    # --- Collect per-model CLI training params ---
    # Only non-None values will be compared against saved summaries / _meta.
    cli_training_params: Dict[int, Dict[str, Any]] = {
        1: {
            "n_pcs": args.model1_n_pcs,
            "l1_ratio": args.model1_l1_ratio,
            "model_name": args.model1_model_name,
        },
        2: {
            "p_values": args.model2_p_values,
            "retrain_on_full_train": args.model2_retrain_on_full_train,
            "sequence_identity_threshold": args.model2_sequence_identity_threshold,
        },
        3: {
            "aggregation_strategy": args.model3_aggregation_strategy,
            "n_estimators_stage1": args.model3_n_estimators_stage1,
            "n_estimators_stage2": args.model3_n_estimators_stage2,
            "entropy_max_fraction": args.model3_entropy_max_fraction,
            "entropy_bottom_percentile": args.model3_entropy_bottom_percentile,
            "tuning_cv_splits": args.model3_tuning_cv_splits,
            "tuning_strategies": None,
            "tuning_entropy_max_fractions": None,
            "tuning_entropy_percentiles": None,
        },
    }

    # Parse comma-separated Model 3 tuning args (safe conversion with clear errors)
    if args.model3_tuning_strategies is not None:
        parsed = [s.strip() for s in args.model3_tuning_strategies.split(",") if s.strip()]
        if not parsed:
            parser.error("--model3-tuning-strategies is empty after parsing.")
        cli_training_params[3]["tuning_strategies"] = parsed
    if args.model3_tuning_entropy_max_fractions is not None:
        try:
            cli_training_params[3]["tuning_entropy_max_fractions"] = [
                float(v.strip())
                for v in args.model3_tuning_entropy_max_fractions.split(",")
                if v.strip()
            ]
        except ValueError as e:
            parser.error(
                f"--model3-tuning-entropy-max-fractions contains non-numeric values: {e}"
            )
        if not cli_training_params[3]["tuning_entropy_max_fractions"]:
            parser.error(
                "--model3-tuning-entropy-max-fractions is empty after parsing."
            )
    if args.model3_tuning_entropy_percentiles is not None:
        try:
            cli_training_params[3]["tuning_entropy_percentiles"] = [
                float(v.strip())
                for v in args.model3_tuning_entropy_percentiles.split(",")
                if v.strip()
            ]
        except ValueError as e:
            parser.error(
                f"--model3-tuning-entropy-percentiles contains non-numeric values: {e}"
            )
        if not cli_training_params[3]["tuning_entropy_percentiles"]:
            parser.error(
                "--model3-tuning-entropy-percentiles is empty after parsing."
            )

    # --- Build retrain set ---
    retrain_set: set = set()
    if args.retrain_base_models:
        retrain_set = set(args.models)
    if args.retrain_models:
        for m in args.retrain_models:
            retrain_set.add(m)

    # --- Validate all args up front ---
    validate_ensemble_args(args, retrain_set, cli_training_params, parser)

    # --- Handle --feature-matrices-dir (early exit: skip all base model logic) ---
    if args.feature_matrices_dir is not None:
        _run_from_feature_matrices(
            args, clone_id_kwargs=get_clone_id_kwargs(args),
            training_context=training_context,
        )
        return

    # --- Resolve cache dir (default: cache/<dataset-name>/) ---
    if args.cache_dir is None:
        args.cache_dir = PROJECT_ROOT / "cache" / args.dataset_name

    # --- Resolve and validate paths ---
    if args.data_dir is not None:
        data_dir = args.data_dir
        if not data_dir.exists():
            logger.error(f"--data-dir does not exist: {data_dir}")
            sys.exit(1)
    else:
        # Check that the cache exists; if not, data_dir is needed
        participants_cache = args.cache_dir / "participants"
        cache_exists = (
            participants_cache.exists()
            and any(participants_cache.glob("*_clean.parquet"))
        )
        if not cache_exists:
            logger.error(
                f"No existing cache found at {args.cache_dir} and --data-dir was not provided. "
                f"Either provide --data-dir to the raw AIRR data directory, or build the cache first."
            )
            sys.exit(1)
        data_dir = None
    metadata_path = args.metadata_path

    # --- Initialize data loader ---
    clone_id_kwargs = get_clone_id_kwargs(args)
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=args.gene_locus,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
        **(clone_id_kwargs or {}),
    )

    # Precompute clone IDs in parallel (no-op if all participants cached)
    if loader.cache_dir is not None:
        loader.precompute_clone_ids(n_jobs=args.n_jobs)

    # --- Resolve fold IDs ---
    # Train-all is a single whole-dataset pass with no CV folds (the dataset may
    # not even have a CV_fold column) → use a single sentinel "fold" (None) and
    # skip get_dataset_fold_ids. --fold-ids is already rejected for train-all.
    if is_train_all:
        fold_ids = [None]
        logger.info("Train-all: single whole-dataset pass (no CV folds)")
    elif args.fold_ids is not None:
        fold_ids = args.fold_ids
        logger.info(f"Fold IDs: {fold_ids}")
    else:
        fold_ids = get_dataset_fold_ids(loader.metadata)
        logger.info(f"Fold IDs: {fold_ids}")

    # --- Resolve per-model modes (LOAD / TRAIN / RESUME) ---
    model_suffixes = {
        1: args.model1_suffix,
        2: args.model2_suffix,
        3: args.model3_suffix,
    }
    model_modes: Dict[int, str] = {}
    base_model_dirs: Dict[int, Path] = {}
    base_model_summaries: Dict[int, Optional[dict]] = {}

    for num in args.models:
        mode, artifact_dir, summary = resolve_base_model_mode(
            model_num=num,
            retrain_set=retrain_set,
            resume_flag=args.resume,
            dataset_name=args.dataset_name,
            classification_mode=args.classification_mode,
            gene_locus=args.gene_locus,
            output_suffix=model_suffixes.get(num),
            cli_training_params=cli_training_params.get(num, {}),
            training_context=training_context,
        )
        model_modes[num] = mode
        base_model_dirs[num] = artifact_dir
        base_model_summaries[num] = summary

    # --- Display base model status ---
    _log_base_model_status_table(
        model_modes, base_model_dirs, base_model_summaries,
        training_context=training_context,
    )

    # --- Pre-flight meta validation for RESUME models ---
    # Merge model hyperparams with run-level params so that changes to
    # classification_mode, reference_class, or diseases also trigger a mismatch.
    _resume_run_params = {
        "classification_mode": args.classification_mode,
        "reference_class": args.reference_class,
        "diseases": args.diseases,
    }
    for num in args.models:
        if model_modes[num] == "RESUME":
            merged_params = {**cli_training_params.get(num, {}), **_resume_run_params}
            preflight_validate_resume_params(
                model_num=num,
                model_dir=base_model_dirs[num],
                cli_training_params=merged_params,
            )

    # Check if any models need training
    models_to_train = [num for num in args.models if model_modes[num] in ("TRAIN", "RESUME")]

    # --- Resolve embedding directory ---
    embedding_dir = args.model3_embedding_dir
    # Only validate embeddings for LOAD mode (prediction needs them).
    # For TRAIN/RESUME, train_model3.train_all_folds() auto-computes if missing.
    if 3 in args.models and model_modes[3] == "LOAD":
        load_embedding_dir = embedding_dir or (args.cache_dir / "embeddings")
        if not load_embedding_dir.exists():
            logger.error(
                f"Model 3 is in LOAD mode but embedding directory not found: {load_embedding_dir}\n"
                f"Embeddings are needed for prediction. Compute with compute_model3_embeddings.py."
            )
            sys.exit(1)

        # Completeness check: verify ALL participants have embedding files.
        # In LOAD mode there is no auto-compute fallback — if any participant
        # is missing, prediction will fail mid-loop. Catch it upfront.
        from malid_lite.training.compute_model3_embeddings import (
            validate_embedding_completeness,
        )
        all_participant_labels = sorted(
            loader.metadata[PARTICIPANT_COL].unique()
        )
        if not validate_embedding_completeness(
            all_participant_labels, load_embedding_dir, logger
        ):
            # Include --output-embedding-dir when the user's embedding dir
            # differs from the default (cache_dir/embeddings/)
            _default_emb_dir = args.cache_dir / "embeddings" if args.cache_dir else None
            _needs_output_flag = (load_embedding_dir != _default_emb_dir)
            _remediation = (
                f"  python -m malid_lite.training.compute_model3_embeddings "
                f"--metadata-path {args.metadata_path}"
                + (f" --cache-dir {args.cache_dir}" if args.cache_dir else "")
                + (f" --output-embedding-dir {load_embedding_dir}" if _needs_output_flag else "")
            )
            logger.error(
                f"Embedding completeness check failed for Model 3 (LOAD mode).\n"
                f"Embedding directory: {load_embedding_dir}\n"
                f"Some participants are missing embedding files "
                f"(see log above for details).\n"
                f"Complete them with:\n"
                f"{_remediation}"
            )
            sys.exit(1)

    # --- Resolve base output directory ---
    # For multi-binary, each pair gets a subdirectory under this base.
    if args.output_dir is not None:
        base_output_dir = args.output_dir
    else:
        base_output_dir = get_ensemble_output_dir(
            dataset_name=args.dataset_name,
            classification_mode=args.classification_mode,
            gene_locus=args.gene_locus,
            output_suffix=args.output_suffix,
            training_context=training_context,
        )

    # --- Resolve disease pairs to train ---
    reference_class = None
    if args.classification_mode == "multiclass":
        if args.reference_class != "Healthy/Background":
            # User explicitly set --reference-class, but multiclass ignores it
            logger.warning(
                f"--reference-class '{args.reference_class}' is ignored in multiclass mode "
                f"(only used in binary/multi-binary mode)."
            )
        pairs_to_train = [None]

    elif args.classification_mode == "binary":
        disease_classes = get_dataset_disease_classes(loader.metadata)
        # validate_mode_and_classes handles:
        # - reference_class required and must exist in data
        # - if diseases is None, data must have exactly 2 classes
        reference_class = validate_mode_and_classes(
            "binary", disease_classes, args.reference_class, args.diseases,
        )
        disease = resolve_binary_disease(
            args.diseases, disease_classes, reference_class,
        )
        if args.diseases is None:
            logger.info(
                f"Binary mode: auto-detected disease class '{disease}' "
                f"(2-class dataset, reference='{reference_class}')"
            )
        pairs_to_train = [(disease, reference_class)]

    elif args.classification_mode == "multi-binary":
        disease_classes = get_dataset_disease_classes(loader.metadata)
        reference_class = validate_mode_and_classes(
            "multi-binary", disease_classes, args.reference_class, args.diseases,
        )
        if args.diseases is not None:
            invalid = [d for d in args.diseases if d not in disease_classes]
            if invalid:
                raise ValueError(
                    f"--diseases {invalid} not found in data: {disease_classes}"
                )
            as_ref = [d for d in args.diseases if d == reference_class]
            if as_ref:
                raise ValueError(
                    f"--diseases includes reference class '{reference_class}'. "
                    f"Remove it from --diseases or change --reference-class."
                )
            diseases_to_train = list(args.diseases)
        else:
            diseases_to_train = [c for c in disease_classes if c != reference_class]
        pairs_to_train = [(d, reference_class) for d in diseases_to_train]

    # Warn if multi-binary output directory already has pair subdirectories
    # that won't be covered by this run (stale cross-pair summary risk)
    if args.classification_mode == "multi-binary" and base_output_dir.exists():
        current_pair_keys = {make_pair_name(d, r) for d, r in pairs_to_train}
        existing_pair_dirs = {
            d.name for d in base_output_dir.iterdir()
            if d.is_dir() and "_vs_" in d.name
        }
        uncovered = existing_pair_dirs - current_pair_keys
        if uncovered:
            logger.warning(
                f"Output directory has existing pair subdirectories not included in "
                f"this run: {sorted(uncovered)}. The cross-pair summary will only "
                f"cover the current {len(current_pair_keys)} pair(s). "
                f"Re-run with all diseases to regenerate a complete summary."
            )

    # --- Add file handler so the full log is saved to disk ---
    base_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_output_dir / "ensemble_training.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_path}")

    # --- Log configuration ---
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Models:              {args.models}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Output:              {base_output_dir}")
    if args.resume:
        logger.info(f"  Resume:              YES")
    if args.model2_abstention_strategy != "ensemble_abstain":
        logger.info(f"  M2 abstention strat: {args.model2_abstention_strategy}")
    if models_to_train:
        mode_info = {num: model_modes[num] for num in models_to_train}
        logger.info(f"  Models to train:     {mode_info}")
    if args.classification_mode == "multi-binary":
        logger.info(f"  Reference class:     {reference_class}")
        logger.info(f"  Pairs to train:      {len(pairs_to_train)}")
        for d, r in pairs_to_train:
            logger.info(f"    {make_pair_name(d, r)}")
    elif args.classification_mode == "binary":
        logger.info(f"  Disease filter:      {pairs_to_train[0]}")

    # --- Validate LOAD model summaries ---
    # LOAD models have summaries from mode detection; validate config consistency.
    # Check gene_locus, training_context, classification_mode, reference_class,
    # and diseases to catch mismatches before hours of ensemble training.
    # Note: expecting training_context == "train_all_ensemble" here also enforces
    # the leakage guard — base models trained as the leaky "train_all" (which saw
    # the validation set) or as "cv_ensemble" are rejected for a train-all ensemble.
    expected_config = {
        "gene_locus": args.gene_locus,
        "training_context": training_context,
        "classification_mode": args.classification_mode,
        "reference_class": reference_class,
        "diseases": args.diseases,
    }
    loaded_models = [num for num in args.models if model_modes[num] == "LOAD"]
    for num in loaded_models:
        bm_summary = base_model_summaries[num]
        validate_model_summary(
            bm_summary, expected_config,
            model_label=f"Model {num} ({base_model_dirs[num].name})",
        )
        logger.info(f"  Model {num} config validated: {base_model_dirs[num].name}")

    # --- Early cross-model disease class validation (LOAD models only) ---
    # Fail fast before hours of training if LOAD models disagree on disease classes.
    # A second pass after Stage 4c re-checks including freshly-trained models.
    if loaded_models:
        load_summaries = {num: base_model_summaries[num] for num in loaded_models}
        _validate_cross_model_disease_classes(load_summaries, label="LOAD models")
        logger.info("  Cross-model disease class validation passed (LOAD models)")

    # ================================================================== #
    # Stage 4c: auto-train base models that need it (TRAIN / RESUME)     #
    # ================================================================== #
    training_times: Dict[int, str] = {}
    if models_to_train:
        logger.info("")
        logger.info("=" * 70)
        logger.info("BASE MODEL TRAINING")
        logger.info("=" * 70)

        # Train sequentially in model order (1 → 2 → 3)
        for num in sorted(models_to_train):
            non_none_params = {
                k: v for k, v in cli_training_params[num].items()
                if v is not None
            }
            is_resume = (model_modes[num] == "RESUME")

            logger.info(
                f"  Model {num}: {'resuming' if is_resume else 'training from scratch'}..."
            )
            if non_none_params:
                logger.info(f"            params: {non_none_params}")

            t_start = time.time()
            auto_train_base_model(
                model_num=num,
                training_params=non_none_params,
                output_dir=base_model_dirs[num],
                metadata_path=loader.metadata_path,
                dataset_name=args.dataset_name,
                classification_mode=args.classification_mode,
                reference_class=reference_class,
                diseases=args.diseases,
                gene_locus=args.gene_locus,
                fold_ids=fold_ids,
                data_dir=data_dir,
                cache_dir=args.cache_dir,
                gene_reference_path=args.gene_reference_path,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                resume=is_resume,
                # Model 3 specific (ignored by Model 1/2 dispatch)
                embedding_dir=embedding_dir,
                no_cache_embeddings=args.model3_no_cache_embeddings,
                device=args.model3_device,
                embedding_batch_size=args.model3_embedding_batch_size,
                clone_id_kwargs=clone_id_kwargs,
                training_context=training_context,
            )
            elapsed = time.time() - t_start
            training_times[num] = _format_elapsed_time(elapsed)

            logger.info(
                f"  Model {num}: complete in {training_times[num]}, "
                f"artifacts at {base_model_dirs[num]}"
            )

        # Post-training summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("BASE MODEL TRAINING COMPLETE")
        logger.info("=" * 70)
        for num in sorted(models_to_train):
            mode_label = "resumed" if model_modes[num] == "RESUME" else "trained"
            logger.info(f"  Model {num}: {mode_label} in {training_times[num]}")
            logger.info(f"            Artifacts: {base_model_dirs[num]}")
        logger.info("=" * 70)

    # --- Collect all summaries ---
    # LOAD models: already in base_model_summaries from mode detection.
    # TRAIN/RESUME models: read freshly-written summary after training.
    for num in args.models:
        if model_modes[num] in ("TRAIN", "RESUME"):
            if base_model_dirs[num].exists() and any(
                base_model_dirs[num].glob("summary_*.json")
            ):
                base_model_summaries[num] = read_model_summary(base_model_dirs[num])
            else:
                raise RuntimeError(
                    f"Model {num} was {model_modes[num].lower()}d but no summary_*.json "
                    f"found at {base_model_dirs[num]}. This indicates a training failure."
                )

    # Cross-model validation including newly-trained models
    _validate_cross_model_disease_classes(base_model_summaries, label="base models")

    # --- Pre-flight: verify fold artifacts exist (all models) ---
    # After auto-training (Stage 4c), all models should have complete artifacts.
    disease_pairs_for_preflight = None
    if args.classification_mode in ("binary", "multi-binary"):
        disease_pairs_for_preflight = [
            p for p in pairs_to_train if p is not None
        ]
    preflight_check_fold_artifacts(
        model_dirs=dict(base_model_dirs),
        fold_ids=fold_ids,
        classification_mode=args.classification_mode,
        disease_pairs=disease_pairs_for_preflight,
        training_context=training_context,
    )
    logger.info("  Pre-flight check passed: all model artifacts found.")

    # Resolve embedding_dir now that Model 3 training (if any) is complete.
    # train_model3.train_all_folds() writes to cache_dir/embeddings when
    # embedding_dir=None; downstream prediction requires an explicit path.
    if 3 in args.models and embedding_dir is None:
        embedding_dir = args.cache_dir / "embeddings"

    # --- Dataset counts (participants and specimens per disease class) ---
    dataset_counts = get_metadata_class_counts(loader.metadata)
    metadata_filter_info = loader.metadata_filter_info

    # --- Train each pair (single iteration for multiclass/binary, N for multi-binary) ---
    all_pair_summaries = {}
    all_pair_fold_results = {}
    first_run_config = None
    for pair in pairs_to_train:
        if pair is None:
            # Multiclass: base_model_dirs already point to .../multiclass/
            model_dirs = dict(base_model_dirs)
            output_dir = base_output_dir
            disease_filter = None
            ref_class = None
            pair_key = "multiclass"
        else:
            # Binary pair: append pair subdirectory to each base model dir
            disease, ref = pair
            pair_key = make_pair_name(disease, ref)
            model_dirs = {num: d / pair_key for num, d in base_model_dirs.items()}
            output_dir = base_output_dir / pair_key
            disease_filter = pair
            ref_class = ref

        if args.classification_mode == "multi-binary":
            logger.info(f"\n{'*'*60}")
            logger.info(f"Binary pair: {pair_key}")
            logger.info(f"{'*'*60}")

        # Validate artifact directories exist
        for num, d in model_dirs.items():
            if not d.exists():
                raise RuntimeError(
                    f"Model {num} artifact directory not found: {d}\n"
                    f"All models should have artifacts after auto-training. "
                    f"Check training output above for errors."
                )
            logger.info(f"  Model {num} artifacts: {d}")

        # Build run config for this pair
        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "training_context": training_context,
            "dataset_name": args.dataset_name,
            "classification_mode": args.classification_mode,
            "gene_locus": args.gene_locus,
            # Resolved clone_id clustering definition (Phase 6.E cross-dataset check).
            "clone_id_params": loader.clone_id_params,
            "models_included": args.models,
            "fold_ids": fold_ids,
            "reference_class": ref_class,
            "diseases": args.diseases,
            "disease_filter": list(disease_filter) if disease_filter else None,
            "output_suffix": args.output_suffix,
            "resume": args.resume,
            "model2_abstention_strategy": args.model2_abstention_strategy,
            "dataset_counts": dataset_counts,
            "metadata_filter_info": metadata_filter_info,
            "metadata_path": str(args.metadata_path) if args.metadata_path else None,
            "metadata_resolved_path": str(loader.metadata_path),
            "base_model_paths": {
                f"model{num}": str(d) for num, d in model_dirs.items()
            },
            "base_model_suffixes": {
                f"model{num}": model_suffixes.get(num) for num in args.models
            },
            "embedding_dir": str(embedding_dir) if embedding_dir else None,
            "metamodel_config": {
                "algorithm": "ridge_cv",
                "alpha": 0.0,
                "n_lambda": 100,
                "scoring": "MCC",
                # Requested n_splits (the effective value is auto-capped DOWN per
                # class at fit time if there aren't enough participants per class).
                "metamodel_cv_n_splits": args.metamodel_cv_n_splits,
                "internal_cv": (
                    f"StratifiedGroupKFold(n_splits={args.metamodel_cv_n_splits} "
                    f"(auto-capped if needed), shuffle=True, random_state=0)"
                ),
                "class_weight": "balanced",
                "use_lambda_1se": False,
            },
            "base_model_training_mode": {
                f"model{num}": model_modes[num].lower()
                for num in args.models
            },
            "base_model_training_params": {
                f"model{num}": (
                    {k: v for k, v in cli_training_params[num].items()
                     if v is not None}
                    if model_modes[num] in ("TRAIN", "RESUME") else None
                )
                for num in args.models
            },
            "base_model_training_times": {
                f"model{num}": training_times.get(num)
                for num in args.models
            },
            "base_model_configs": {
                f"model{num}": (
                    {
                        k: v for k, v in bm_summary.items()
                        if k not in ("results_by_pair", "aggregated_by_pair")
                    }
                    if bm_summary is not None else None
                )
                for num, bm_summary in base_model_summaries.items()
            },
        }

        if is_train_all:
            # Train-all: single whole-dataset pass, metamodel on the validation
            # third, NO test → no metrics, no aggregation, no predictions CSV.
            summary = _run_train_all_ensemble(
                output_dir,
                loader=loader,
                model_nums=args.models,
                model_dirs=model_dirs,
                gene_locus=args.gene_locus,
                embedding_dir=embedding_dir,
                disease_filter=disease_filter,
                reference_class=ref_class,
                model_summaries=base_model_summaries,
                n_jobs=args.n_jobs,
                metamodel_cv_n_splits=args.metamodel_cv_n_splits,
                model2_abstention_strategy=args.model2_abstention_strategy,
                resume=args.resume,
                run_config=run_config,
            )
            all_pair_summaries[pair_key] = summary
            all_pair_fold_results[pair_key] = []
        else:
            fold_results, summary = train_ensemble(
                loader=loader,
                fold_ids=fold_ids,
                model_nums=args.models,
                model_dirs=model_dirs,
                gene_locus=args.gene_locus,
                output_dir=output_dir,
                embedding_dir=embedding_dir,
                disease_filter=disease_filter,
                reference_class=ref_class,
                run_config=run_config,
                model_summaries=base_model_summaries,
                n_jobs=args.n_jobs,
                resume=args.resume,
                metamodel_cv_n_splits=args.metamodel_cv_n_splits,
                model2_abstention_strategy=args.model2_abstention_strategy,
            )
            all_pair_summaries[pair_key] = summary
            all_pair_fold_results[pair_key] = fold_results
        if first_run_config is None:
            first_run_config = run_config

    # --- Multi-binary cross-pair summary (metrics-based → CV only) ---
    if args.classification_mode == "multi-binary" and not is_train_all:
        if len(all_pair_summaries) > 1:
            _save_multi_binary_summary(
                base_output_dir, all_pair_summaries, all_pair_fold_results,
                pairs_to_train, reference_class, run_config=first_run_config,
            )
        else:
            logger.info(
                "Single disease pair — skipping cross-pair comparison summary. "
                "Per-pair results are in the pair subdirectory."
            )

    # --- Final console summary ---
    elapsed = time.monotonic() - t_main_start
    if is_train_all:
        # No metrics to summarize — report what was trained and where.
        logger.info(f"\n{'='*70}")
        logger.info("TRAIN-ALL ENSEMBLE COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"  Pairs trained:  {list(all_pair_summaries.keys())}")
        logger.info(f"  Elapsed:        {_format_elapsed_time(elapsed)}")
        logger.info(
            "  No test set → no metrics. Evaluate on a separate dataset with the "
            "external-evaluation script (Phase 6)."
        )
    else:
        _log_final_summary(
            all_pair_summaries=all_pair_summaries,
            base_output_dir=base_output_dir,
            base_model_dirs=base_model_dirs,
            model_modes=model_modes,
            training_times=training_times,
            model_suffixes=model_suffixes,
            embedding_dir=embedding_dir,
            args=args,
            elapsed_seconds=elapsed,
        )

    logger.info(f"\nDone. Output: {base_output_dir}")


if __name__ == "__main__":
    main()
