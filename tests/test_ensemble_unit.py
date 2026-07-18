"""Unit tests for the ensemble (metamodel) pipeline.

Tests individual components with synthetic data — no real data, cache, or
pre-trained base model artifacts required. All probability matrices are
randomly generated.

Test groups
-----------
Core pipeline:
  1.  ModelPredictions dataclass: properties, empty case
  2.  build_feature_matrix: multiclass column naming and concatenation
  3.  build_feature_matrix: binary column selection (non-reference class)
  4.  build_feature_matrix: abstention harmonization (partial scoring)
  5.  build_feature_matrix: single model (no harmonization needed)
  6.  train_metamodel: fit on small synthetic feature matrix
  7.  evaluate_predictions: multiclass metrics (3+ classes)
  8.  evaluate_predictions: binary metrics (2 classes + reference_class)
  9.  evaluate_predictions: all-correct edge case (perfect accuracy)
  10. evaluate_predictions: abstention penalty (n_abstained > 0)
  11. Column alignment: test reindexing and assertion
  12. aggregate_fold_results: multiclass with macro/weighted + MCC
  13. aggregate_fold_results: binary mode with MCC and log_loss
  14. _generate_ensemble_results_md: multiclass and binary output format
  15. _log_comparison_table: output contains model names and metrics
  16. _save_multi_binary_summary: cross-pair MD and JSON
  17. Path construction: make_pair_name, model/ensemble output dirs
  18. validate_mode_and_classes: error paths for multi-binary/binary
  19. train_ensemble binary (mocked folds): full flow + artifact verification
  20. Multi-binary orchestration (mocked folds): 2 pairs + cross-pair summary
  21. Binary specimen filtering: only target-disease specimens counted
  22. Resume mode: metrics match original run (multiclass + binary + round-trip)

Auto-training:
  23. compare_training_params: matching, mismatch, None skip, list order, M3
  24. resolve_base_model_mode: retrain, no artifacts, LOAD, LOAD mismatch, RESUME
  25. CLI arg interaction validation (via subprocess)
  26. preflight_validate_resume_params: no _meta, _meta mismatch

Dispatch:
  27. _format_elapsed_time: seconds to human-readable
  28. auto_train_base_model: invalid model, dispatch M1/M2/M3, optional kwargs

Argument validation:
  29. validate_ensemble_args: valid defaults, conflicts, excluded models, ranges
  30. validate_ensemble_args: suffix sanitization, file existence
  31. validate_ensemble_args: M3 cross-param interactions (tuning/entropy)

Cross-model validation:
  32. _validate_cross_model_disease_classes: match, mismatch, edge cases
  33. _log_base_model_status_table: LOAD/TRAIN/RESUME modes

New tests (U1-U6):
  U1. Resume config mismatch detection
  U2. LOAD model expected_config with reference_class/diseases
  U3. Base model RESUME with run-level param mismatch
  U4. Ensemble summary includes classification_mode/reference_class/diseases
  U5. Feature matrix with all specimens abstained
  U6. Feature matrix column mismatch (test has extra/missing vs val)

Raw feature matrix and load-time fill:
  R1. _build_raw_feature_matrix: from fill_0.5, ensemble_abstain, no abstentions, no M2
  R2. apply_m2_fill_strategy: ensemble_abstain, fill_0.5, fill_models13_mean, no abstentions
  R3. Strategy round-trips: fill->fill, abstain->fill, fill->abstain, CSV round-trip

Feature matrices dir and source_dir:
  F1. validate_ensemble_args: conflicts, missing config, multi-binary pair subdirs
  F2. train_ensemble(source_dir=...): routing, cleanup, config validation skip

Cache-dir resolution:
  C1. --cache-dir defaults to cache/<dataset-name>/ when omitted
  C2. Explicit --cache-dir overrides --dataset-name

Requirements
------------
numpy, pandas, scikit-learn, glmnet

Expected runtime: ~30-60 seconds

Running
-------
    python -m pytest tests/test_ensemble_unit.py -v
    python -m pytest tests/test_ensemble_unit.py -v -k "compare_training_params"
"""

import json
import logging
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_ensemble_unit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from malid_lite.training.training_utils import (
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    aggregate_fold_results,
    get_ensemble_output_dir,
    get_model_output_dir,
    make_pair_name,
    read_model_summary,
    resolve_model_artifact_dir,
    validate_mode_and_classes,
    validate_model_summary,
)
from malid_lite.training.train_ensemble import (
    MODEL_DISPLAY_NAMES,
    MODEL2_ABSTENTION_STRATEGIES,
    TRAINING_CONTEXT,
    ModelPredictions,
    _build_raw_feature_matrix,
    apply_m2_fill_strategy,
    auto_train_base_model,
    build_feature_matrix,
    compare_training_params,
    evaluate_predictions,
    preflight_validate_resume_params,
    resolve_base_model_mode,
    run_ensemble_fold_from_features,
    save_fold_artifacts,
    train_ensemble,
    train_metamodel,
    validate_ensemble_args,
    _format_elapsed_time,
    _generate_ensemble_results_md,
    _log_base_model_status_table,
    _log_comparison_table,
    _save_multi_binary_summary,
    _validate_cross_model_disease_classes,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

DISEASE_CLASSES = ["Covid19", "HIV", "Healthy/Background", "Influenza", "Lupus", "T1D"]
BINARY_DISEASE = "Covid19"
BINARY_REFERENCE = "Healthy/Background"


def _make_mock_proba(
    specimen_labels: List[str],
    classes: List[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Create random probability matrix, softmax-normalized."""
    logits = rng.randn(len(specimen_labels), len(classes))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp / exp.sum(axis=1, keepdims=True)
    return pd.DataFrame(proba, index=specimen_labels, columns=classes)


def _make_mock_predictions(
    n_specimens: int,
    classes: List[str],
    prefix: str = "spec",
    n_abstained: int = 0,
    rng: np.random.RandomState = None,
) -> ModelPredictions:
    """Create a ModelPredictions with random probabilities."""
    if rng is None:
        rng = np.random.RandomState(42)
    scored_labels = [f"{prefix}_{i:03d}" for i in range(n_specimens)]
    abstained_labels = [f"{prefix}_abs_{i:03d}" for i in range(n_abstained)]
    abstained_diseases = list(rng.choice(classes, size=n_abstained))
    proba = _make_mock_proba(scored_labels, classes, rng)
    return ModelPredictions(
        probabilities=proba,
        abstained_specimen_labels=abstained_labels,
        abstained_specimen_diseases=abstained_diseases,
    )


def _get_test_output_dir(test_name: str) -> Path:
    """Create a clean test output subdirectory."""
    test_dir = OUTPUT_DIR / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


# ---------------------------------------------------------------------------
# Helper: _make_mock_fold_result_with_features (for resume tests)
# ---------------------------------------------------------------------------

def _make_mock_fold_result_with_features(
    fold_id: int,
    gene_locus: str,
    classes: np.ndarray,
    model_nums: List[int],
    n_val: int = 30,
    n_test: int = 20,
    n_abstained: int = 0,
    reference_class: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Dict, pd.DataFrame]:
    """Build a synthetic fold result with consistent feature matrices and metrics.

    Uses train_metamodel (real GlmnetLogitNet) so metrics are perfectly
    consistent between saved artifacts and what resume mode recomputes.
    """
    rng = np.random.RandomState(seed)
    str_classes = np.array([str(c) for c in classes])

    val_specimens = [f"val_{fold_id}_{i:03d}" for i in range(n_val)]
    test_specimens = [f"test_{fold_id}_{i:03d}" for i in range(n_test)]
    specimen_to_participant = {s: f"part_{s}" for s in val_specimens + test_specimens}

    y_val = np.array([str_classes[i % len(str_classes)] for i in range(n_val)])
    y_test = np.array([str_classes[i % len(str_classes)] for i in range(n_test)])

    # Build feature columns
    is_binary = reference_class is not None and len(classes) == 2
    feature_cols = []
    for model_num in model_nums:
        display_name = MODEL_DISPLAY_NAMES[model_num]
        if is_binary:
            non_ref = [str(c) for c in classes if str(c) != str(reference_class)][0]
            feature_cols.append(f"{gene_locus}:{display_name}:{non_ref}")
        else:
            for cls in sorted(str_classes):
                feature_cols.append(f"{gene_locus}:{display_name}:{cls}")

    # Generate and normalize features
    X_val_data = rng.rand(n_val, len(feature_cols)).astype(np.float64)
    X_test_data = rng.rand(n_test, len(feature_cols)).astype(np.float64)
    if not is_binary:
        for model_num in model_nums:
            display_name = MODEL_DISPLAY_NAMES[model_num]
            prefix = f"{gene_locus}:{display_name}:"
            col_idx = [i for i, c in enumerate(feature_cols) if c.startswith(prefix)]
            X_val_data[:, col_idx] /= X_val_data[:, col_idx].sum(axis=1, keepdims=True)
            X_test_data[:, col_idx] /= X_test_data[:, col_idx].sum(axis=1, keepdims=True)

    X_val = pd.DataFrame(X_val_data, index=val_specimens, columns=feature_cols)
    X_test = pd.DataFrame(X_test_data, index=test_specimens, columns=feature_cols)

    # Train metamodel
    y_val_series = pd.Series(y_val, index=X_val.index)
    groups_val = pd.Series(
        [specimen_to_participant[s] for s in val_specimens], index=X_val.index,
    )
    pipeline = train_metamodel(X_val, y_val_series, groups_val)
    clf = pipeline.named_steps["classifier"]

    # Predict
    y_pred = pipeline.predict(X_test.values)
    y_proba = pipeline.predict_proba(X_test.values)
    pipeline_classes = pipeline.classes_

    # Evaluate ensemble
    ens_metrics, ens_raw = evaluate_predictions(
        y_true=y_test, y_pred=y_pred, y_proba=y_proba,
        classes=pipeline_classes, fold_id=fold_id, model_label="ensemble",
        n_scored=n_test, n_abstained=n_abstained, reference_class=reference_class,
    )

    # Evaluate base models
    bm_metrics, bm_raw = {}, {}
    for model_num in model_nums:
        display_name = MODEL_DISPLAY_NAMES[model_num]
        prefix = f"{gene_locus}:{display_name}:"
        model_col_names = [c for c in feature_cols if c.startswith(prefix)]
        bm_classes = np.array([c.split(":", 2)[2] for c in model_col_names])
        bm_proba_vals = X_test[model_col_names].values
        if len(bm_classes) == 1 and reference_class is not None:
            disease_class = bm_classes[0]
            disease_proba = bm_proba_vals[:, 0]
            ref_proba = 1.0 - disease_proba
            bm_classes = np.array(sorted([disease_class, str(reference_class)]))
            col_probas = {disease_class: disease_proba, str(reference_class): ref_proba}
            bm_proba_vals = np.column_stack([col_probas[c] for c in bm_classes])
        bm_y_pred = bm_classes[np.argmax(bm_proba_vals, axis=1)]
        m, r = evaluate_predictions(
            y_true=y_test, y_pred=bm_y_pred, y_proba=bm_proba_vals,
            classes=bm_classes, fold_id=fold_id, model_label=f"model{model_num}",
            n_scored=n_test, n_abstained=n_abstained, reference_class=reference_class,
        )
        bm_metrics[model_num] = m
        bm_raw[model_num] = r

    # Abstention details
    abstained_details = []
    for i in range(n_abstained):
        abstained_details.append({
            "specimen_label": f"abstained_{fold_id}_{i:03d}",
            "participant_label": f"part_abstained_{fold_id}_{i:03d}",
            "disease": str(str_classes[i % len(str_classes)]),
        })

    # Prediction rows
    rows = []
    for i, spec in enumerate(test_specimens):
        row = {
            "fold_id": fold_id,
            "specimen_label": spec,
            "participant_label": specimen_to_participant[spec],
            "true_disease": str(y_test[i]),
            "ensemble_predicted": str(y_pred[i]),
            "abstained": False,
        }
        for j, cls in enumerate(pipeline_classes):
            row[f"ensemble_P({cls})"] = float(y_proba[i, j])
        rows.append(row)
    for detail in abstained_details:
        row = {
            "fold_id": fold_id,
            "specimen_label": detail["specimen_label"],
            "participant_label": detail["participant_label"],
            "true_disease": detail["disease"],
            "ensemble_predicted": "ABSTAINED",
            "abstained": True,
        }
        for cls in pipeline_classes:
            row[f"ensemble_P({cls})"] = None
        rows.append(row)

    # Feature matrices with labels
    X_val_with_labels = X_val.copy()
    X_val_with_labels.insert(0, "true_disease", y_val)
    X_test_with_labels = X_test.copy()
    X_test_with_labels.insert(0, "true_disease", y_test)

    fold_result = {
        "fold_id": fold_id,
        "ensemble_metrics": ens_metrics,
        "ensemble_raw_preds": ens_raw,
        "base_model_metrics": bm_metrics,
        "base_model_raw_preds": bm_raw,
        "pipeline": pipeline,
        "metamodel_config": {
            "feature_columns": list(X_val.columns),
            "classes": [str(c) for c in pipeline_classes],
            "gene_locus": gene_locus,
            "models_included": model_nums,
            "n_features": X_val.shape[1],
            "n_validation_specimens": n_val,
            "n_test_specimens": n_test,
            "n_test_abstained": n_abstained,
            "lambda_best": float(clf.lambda_best_),
        },
        "predictions_rows": rows,
        "feature_matrix_val": X_val_with_labels,
        "feature_matrix_test": X_test_with_labels,
        "test_abstained_details": abstained_details,
    }

    meta_rows = [{SPECIMEN_COL: s, PARTICIPANT_COL: p}
                 for s, p in specimen_to_participant.items()]
    for detail in abstained_details:
        meta_rows.append({
            SPECIMEN_COL: detail["specimen_label"],
            PARTICIPANT_COL: detail["participant_label"],
        })
    metadata_df = pd.DataFrame(meta_rows)

    return fold_result, metadata_df


# ---------------------------------------------------------------------------
# Helper: _make_mock_fold_result (simpler, for mocked train_ensemble tests)
# ---------------------------------------------------------------------------

def _make_mock_fold_result(
    fold_id: int,
    classes: np.ndarray,
    n_specimens: int,
    model_nums: List[int],
    reference_class: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Build a synthetic fold result matching run_ensemble_fold output."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed if seed is not None else fold_id)
    y_true = np.array([classes[i % len(classes)] for i in range(n_specimens)])

    def _synth_preds(boost: float):
        proba = rng.dirichlet(np.ones(len(classes)), size=n_specimens)
        for i in range(n_specimens):
            true_idx = np.where(classes == y_true[i])[0][0]
            proba[i, true_idx] += boost
        proba /= proba.sum(axis=1, keepdims=True)
        pred = classes[np.argmax(proba, axis=1)]
        return pred, proba

    ens_pred, ens_proba = _synth_preds(boost=2.0)
    ens_metrics, ens_raw = evaluate_predictions(
        y_true=y_true, y_pred=ens_pred, y_proba=ens_proba,
        classes=classes, fold_id=fold_id, model_label="ensemble",
        n_scored=n_specimens, n_abstained=0, reference_class=reference_class,
    )

    bm_metrics, bm_raw = {}, {}
    for num in model_nums:
        bp, bproba = _synth_preds(boost=0.8)
        m, r = evaluate_predictions(
            y_true=y_true, y_pred=bp, y_proba=bproba,
            classes=classes, fold_id=fold_id, model_label=f"model{num}",
            n_scored=n_specimens, n_abstained=0, reference_class=reference_class,
        )
        bm_metrics[num] = m
        bm_raw[num] = r

    n_per_class = max(2, 10 // len(classes))
    X_tiny = rng.randn(n_per_class * len(classes), 3)
    y_tiny = np.array([str(c) for c in classes] * n_per_class)
    pipe = SkPipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    pipe.fit(X_tiny, y_tiny)

    specimen_labels = [f"spec_{fold_id}_{i:03d}" for i in range(n_specimens)]
    rows = []
    for i, spec in enumerate(specimen_labels):
        row = {
            "fold_id": fold_id,
            "specimen_label": spec,
            "participant_label": f"part_{fold_id}_{i:03d}",
            "true_disease": str(y_true[i]),
            "ensemble_predicted": str(ens_pred[i]),
        }
        for j, cls in enumerate(classes):
            row[f"ensemble_P({cls})"] = float(ens_proba[i, j])
        rows.append(row)

    return {
        "fold_id": fold_id,
        "ensemble_metrics": ens_metrics,
        "ensemble_raw_preds": ens_raw,
        "base_model_metrics": bm_metrics,
        "base_model_raw_preds": bm_raw,
        "pipeline": pipe,
        "metamodel_config": {
            "feature_columns": [f"feat_{i}" for i in range(3)],
            "classes": [str(c) for c in classes],
            "gene_locus": "TCR",
            "models_included": model_nums,
            "n_features": 3,
            "n_validation_specimens": 30,
            "n_test_specimens": n_specimens,
            "n_test_abstained": 0,
            "lambda_best": 0.01,
        },
        "predictions_rows": rows,
    }


# ---------------------------------------------------------------------------
# Helpers for validate_ensemble_args tests
# ---------------------------------------------------------------------------

import argparse


def _make_base_namespace(**overrides) -> argparse.Namespace:
    defaults = dict(
        models=[1, 2, 3], resume=False, retrain_models=None,
        retrain_base_models=False, output_dir=None, output_suffix=None,
        model1_suffix=None, model2_suffix=None, model3_suffix=None,
        n_jobs=4, classification_mode="multiclass", diseases=None,
        metadata_path=None, gene_reference_path=None,
        model3_embedding_dir=None, model3_no_cache_embeddings=False,
        model3_device=None, model3_embedding_batch_size=None,
        model2_abstention_strategy="ensemble_abstain",
        feature_matrices_dir=None,
        # Phase 5: --training-context is REQUIRED on the CLI; default the unit
        # test args to "cv" (the pre-Phase-5 implicit behavior).
        training_context="cv", fold_ids=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_base_cli_params(**m3_overrides) -> Dict[int, Dict]:
    m3 = dict(
        aggregation_strategy=None, n_estimators_stage1=None,
        n_estimators_stage2=None, entropy_max_fraction=None,
        entropy_bottom_percentile=None, tuning_cv_splits=None,
        tuning_strategies=None, tuning_entropy_max_fractions=None,
        tuning_entropy_percentiles=None,
    )
    m3.update(m3_overrides)
    return {
        1: {"n_pcs": None, "l1_ratio": None},
        2: {"p_values": None, "sequence_identity_threshold": None},
        3: m3,
    }


class _FakeParser:
    def __init__(self):
        self.error_message = None

    def error(self, message: str):
        self.error_message = message
        raise SystemExit(message)


# =========================================================================
# Core pipeline tests
# =========================================================================

class TestModelPredictions:
    def test_properties_and_empty(self):
        """Test 1: ModelPredictions properties and empty edge case."""
        rng = np.random.RandomState(0)
        preds = _make_mock_predictions(10, DISEASE_CLASSES, n_abstained=3, rng=rng)
        assert preds.n_scored == 10
        assert preds.n_abstained == 3
        assert list(preds.probabilities.columns) == DISEASE_CLASSES

        empty = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=["s1", "s2"],
            abstained_specimen_diseases=["Covid19", "HIV"],
        )
        assert empty.n_scored == 0
        assert empty.n_abstained == 2


class TestBuildFeatureMatrix:
    def test_multiclass(self):
        """Test 2: multiclass column naming and concatenation."""
        rng = np.random.RandomState(1)
        specimens = [f"spec_{i:03d}" for i in range(20)]
        predictions = {}
        for model_num in [1, 2, 3]:
            proba = _make_mock_proba(specimens, DISEASE_CLASSES, rng)
            predictions[model_num] = ModelPredictions(
                probabilities=proba, abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X, abstained_labels, abstained_diseases, _fill_info = build_feature_matrix(
            predictions, gene_locus="TCR", reference_class=None,
        )
        assert X.shape == (20, 18)
        for col in X.columns:
            parts = col.split(":")
            assert len(parts) == 3
            assert parts[0] == "TCR"
        assert len(abstained_labels) == 0
        assert not X.isna().any().any()

    def test_binary(self):
        """Test 3: binary column selection (non-reference class)."""
        rng = np.random.RandomState(2)
        specimens = [f"spec_{i:03d}" for i in range(15)]
        binary_classes = [BINARY_DISEASE, BINARY_REFERENCE]
        predictions = {}
        for model_num in [1, 2, 3]:
            proba = _make_mock_proba(specimens, binary_classes, rng)
            predictions[model_num] = ModelPredictions(
                probabilities=proba, abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X, _, _, _ = build_feature_matrix(
            predictions, gene_locus="TCR", reference_class=BINARY_REFERENCE,
        )
        assert X.shape == (15, 3)
        for col in X.columns:
            assert BINARY_REFERENCE not in col
            assert BINARY_DISEASE in col

    def test_abstention_harmonization(self):
        """Test 4: abstention harmonization across models."""
        rng = np.random.RandomState(3)
        all_specimens = [f"spec_{i:03d}" for i in range(20)]
        scored_specimens = all_specimens[:15]
        abstained_specimens = all_specimens[15:]

        preds1 = ModelPredictions(
            probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[], abstained_specimen_diseases=[],
        )
        preds2 = ModelPredictions(
            probabilities=_make_mock_proba(scored_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=abstained_specimens,
            abstained_specimen_diseases=["Covid19"] * 5,
        )
        preds3 = ModelPredictions(
            probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[], abstained_specimen_diseases=[],
        )
        X, abstained_labels, _, _fill_info = build_feature_matrix(
            {1: preds1, 2: preds2, 3: preds3}, gene_locus="TCR", reference_class=None,
        )
        assert X.shape[0] == 15
        assert len(abstained_labels) == 5
        assert not X.isna().any().any()

    def test_single_model(self):
        """Test 5: single model (e.g., --models 3)."""
        rng = np.random.RandomState(4)
        preds = _make_mock_predictions(10, DISEASE_CLASSES, rng=rng)
        preds.probabilities.index = [f"spec_{i:03d}" for i in range(10)]
        X, _, _, _ = build_feature_matrix({3: preds}, gene_locus="TCR", reference_class=None)
        assert X.shape == (10, 6)

    def test_model_fully_abstained_excluded(self):
        """U5: Model fully abstains -> excluded from feature matrix, other models used."""
        rng = np.random.RandomState(50)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds1 = ModelPredictions(
            probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[], abstained_specimen_diseases=[],
        )
        # Model 2 abstains on ALL specimens
        preds2 = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=specimens,
            abstained_specimen_diseases=list(rng.choice(DISEASE_CLASSES, size=10)),
        )
        X, abstained_labels, _, fill_info = build_feature_matrix(
            {1: preds1, 2: preds2}, gene_locus="TCR", reference_class=None,
        )
        # M2 is excluded, M1 provides features for all 10 specimens
        assert X.shape[0] == 10
        assert len(abstained_labels) == 0
        # M2 columns should NOT be in the feature matrix
        m2_cols = [c for c in X.columns if "convergent_cluster_model" in c]
        assert len(m2_cols) == 0
        # M1 columns should be present
        m1_cols = [c for c in X.columns if "repertoire_stats" in c]
        assert len(m1_cols) > 0
        # Excluded models tracked in fill_info
        assert fill_info.get("excluded_models") == [2]

    def test_model_fully_abstained_excluded_with_fill(self):
        """U5a2: M2 fully abstains with fill strategy -> still excluded (not filled)."""
        rng = np.random.RandomState(51)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds1 = ModelPredictions(
            probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[], abstained_specimen_diseases=[],
        )
        # Model 2 abstains on ALL specimens
        preds2 = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=specimens,
            abstained_specimen_diseases=list(rng.choice(DISEASE_CLASSES, size=10)),
        )
        X, abstained_labels, _, fill_info = build_feature_matrix(
            {1: preds1, 2: preds2}, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )
        # M2 fully abstained -> excluded even with fill strategy
        assert X.shape[0] == 10
        assert len(abstained_labels) == 0
        m2_cols = [c for c in X.columns if "convergent_cluster_model" in c]
        assert len(m2_cols) == 0, f"M2 columns should be absent, got: {m2_cols}"
        assert fill_info.get("excluded_models") == [2]
        # No fill info (M2 was excluded, not filled)
        assert "n_filled" not in fill_info

    def test_all_models_fully_abstained(self):
        """U5b: ALL models fully abstain -> ValueError."""
        preds1 = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=["s1"], abstained_specimen_diseases=["D1"],
        )
        preds2 = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=["s1"], abstained_specimen_diseases=["D1"],
        )
        with pytest.raises(ValueError, match="All models fully abstained"):
            build_feature_matrix(
                {1: preds1, 2: preds2}, gene_locus="TCR", reference_class=None,
            )

    def test_column_mismatch(self):
        """U6: Column reindexing — missing columns detected."""
        rng = np.random.RandomState(60)
        val_specimens = [f"val_{i:03d}" for i in range(20)]
        val_predictions = {}
        for model_num in [1, 2, 3]:
            val_predictions[model_num] = ModelPredictions(
                probabilities=_make_mock_proba(val_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        X_val, _, _, _ = build_feature_matrix(val_predictions, "TCR", None)

        test_specimens = [f"test_{i:03d}" for i in range(15)]
        test_predictions = {}
        for model_num in [1, 2, 3]:
            test_predictions[model_num] = ModelPredictions(
                probabilities=_make_mock_proba(test_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        X_test, _, _, _ = build_feature_matrix(test_predictions, "TCR", None)

        # Drop a column from test — detect missing
        X_test_missing = X_test.drop(columns=[X_test.columns[0]])
        missing_cols = set(X_val.columns) - set(X_test_missing.columns)
        assert len(missing_cols) == 1

        # Reindex should work if all columns present
        X_test_reindexed = X_test[X_val.columns]
        assert list(X_test_reindexed.columns) == list(X_val.columns)

    def test_fill_0_5_multiclass(self):
        """Fill strategy fill_0.5: M2 abstentions filled with 0.5, all specimens included."""
        rng = np.random.RandomState(100)
        all_specimens = [f"spec_{i:03d}" for i in range(20)]
        m2_scored = all_specimens[:15]
        m2_abstained = all_specimens[15:]
        m2_abstained_diseases = ["Covid19", "HIV", "Healthy/Background",
                                 "Influenza", "Lupus"]

        preds = {}
        for num in [1, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        preds[2] = ModelPredictions(
            probabilities=_make_mock_proba(m2_scored, DISEASE_CLASSES, rng),
            abstained_specimen_labels=m2_abstained,
            abstained_specimen_diseases=m2_abstained_diseases,
        )

        X, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )

        # All 20 specimens included (fills brought M2 abstentions back)
        assert X.shape[0] == 20
        # 18 columns: 6 classes x 3 models
        assert X.shape[1] == 18
        # M2 abstentions are NOT reported as abstained (they were filled)
        assert len(abstained) == 0
        # fill_info populated correctly
        assert fill_info["strategy"] == "fill_0.5"
        assert fill_info["n_filled"] == 5
        assert set(fill_info["filled_specimen_labels"]) == set(m2_abstained)
        assert fill_info["filled_per_class"] == {
            "Covid19": 1, "HIV": 1, "Healthy/Background": 1,
            "Influenza": 1, "Lupus": 1,
        }

        # Verify M2 columns have 0.5 for filled specimens
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X.columns if m2_display in c]
        for spec in m2_abstained:
            for col in m2_cols:
                assert X.loc[spec, col] == 0.5, (
                    f"Expected 0.5 for filled specimen {spec}, col {col}, "
                    f"got {X.loc[spec, col]}"
                )
        # M1/M3 columns should NOT be 0.5 for filled specimens (they have real values)
        m1_display = MODEL_DISPLAY_NAMES[1]
        m1_cols = [c for c in X.columns if m1_display in c]
        for spec in m2_abstained:
            vals = X.loc[spec, m1_cols].values
            assert not np.allclose(vals, 0.5), (
                f"M1 values for filled specimen {spec} should not all be 0.5"
            )

    def test_fill_0_5_binary(self):
        """Fill strategy fill_0.5 in binary mode: single column per model, 0.5 fill."""
        rng = np.random.RandomState(101)
        binary_classes = [BINARY_DISEASE, BINARY_REFERENCE]
        all_specimens = [f"spec_{i:03d}" for i in range(18)]
        m2_scored = all_specimens[:15]
        m2_abstained = all_specimens[15:]

        preds = {}
        for num in [1, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(all_specimens, binary_classes, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        preds[2] = ModelPredictions(
            probabilities=_make_mock_proba(m2_scored, binary_classes, rng),
            abstained_specimen_labels=m2_abstained,
            abstained_specimen_diseases=[BINARY_DISEASE] * 3,
        )

        X, abstained, _, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=BINARY_REFERENCE,
            model2_abstention_strategy="fill_0.5",
        )

        # All 18 specimens, 3 columns (1 non-reference column per model)
        assert X.shape == (18, 3)
        assert fill_info["n_filled"] == 3
        # M2 column for filled specimens should be 0.5
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_col = [c for c in X.columns if m2_display in c]
        assert len(m2_col) == 1
        for spec in m2_abstained:
            assert X.loc[spec, m2_col[0]] == 0.5

    def test_fill_models13_mean_multiclass(self):
        """Fill strategy fill_models13_mean: M2 filled with per-class mean of M1+M3."""
        rng = np.random.RandomState(102)
        all_specimens = [f"spec_{i:03d}" for i in range(20)]
        m2_scored = all_specimens[:15]
        m2_abstained = all_specimens[15:]

        preds = {}
        for num in [1, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        preds[2] = ModelPredictions(
            probabilities=_make_mock_proba(m2_scored, DISEASE_CLASSES, rng),
            abstained_specimen_labels=m2_abstained,
            abstained_specimen_diseases=["Covid19"] * 5,
        )

        X, abstained, _, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_models13_mean",
        )

        assert X.shape[0] == 20
        assert fill_info["strategy"] == "fill_models13_mean"
        assert fill_info["n_filled"] == 5

        # Verify per-class mean computation for each filled specimen
        m1_display = MODEL_DISPLAY_NAMES[1]
        m2_display = MODEL_DISPLAY_NAMES[2]
        m3_display = MODEL_DISPLAY_NAMES[3]
        for cls in DISEASE_CLASSES:
            m1_col = f"TCR:{m1_display}:{cls}"
            m2_col = f"TCR:{m2_display}:{cls}"
            m3_col = f"TCR:{m3_display}:{cls}"
            for spec in m2_abstained:
                expected = (X.loc[spec, m1_col] + X.loc[spec, m3_col]) / 2.0
                actual = X.loc[spec, m2_col]
                assert np.isclose(actual, expected, atol=1e-10), (
                    f"Specimen {spec}, class {cls}: expected mean "
                    f"{expected:.6f}, got {actual:.6f}"
                )

    def test_fill_models13_mean_requires_m1_and_m3(self):
        """fill_models13_mean must error when M1 or M3 is missing."""
        rng = np.random.RandomState(103)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = {
            1: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
            2: ModelPredictions(
                probabilities=_make_mock_proba(specimens[:7], DISEASE_CLASSES, rng),
                abstained_specimen_labels=specimens[7:],
                abstained_specimen_diseases=["Covid19"] * 3,
            ),
        }
        with pytest.raises(ValueError, match="Models 1 and 3"):
            build_feature_matrix(
                preds, gene_locus="TCR", reference_class=None,
                model2_abstention_strategy="fill_models13_mean",
            )

    def test_fill_full_m2_abstention(self):
        """Fill with full M2 abstention (0 scored): M2 excluded, not filled."""
        rng = np.random.RandomState(104)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = {
            1: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
            2: ModelPredictions(
                probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
                abstained_specimen_labels=specimens,
                abstained_specimen_diseases=list(rng.choice(DISEASE_CLASSES, size=10)),
            ),
            3: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
        }
        X, abstained, _, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )
        # All 10 specimens included (M2 excluded, M1+M3 score all)
        assert X.shape[0] == 10
        assert len(abstained) == 0
        # M2 excluded entirely — no M2 columns, no fill
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X.columns if m2_display in c]
        assert len(m2_cols) == 0, f"M2 columns should be absent, got: {m2_cols}"
        assert fill_info.get("excluded_models") == [2]
        assert "n_filled" not in fill_info

    def test_fill_models13_mean_m1_fully_abstains(self):
        """fill_models13_mean with M1 fully abstaining raises clear error."""
        rng = np.random.RandomState(106)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = {
            # M1 fully abstains (0 scored)
            1: ModelPredictions(
                probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
                abstained_specimen_labels=specimens,
                abstained_specimen_diseases=list(rng.choice(DISEASE_CLASSES, size=10)),
            ),
            # M2 partially abstains (fill needed)
            2: ModelPredictions(
                probabilities=_make_mock_proba(specimens[:7], DISEASE_CLASSES, rng),
                abstained_specimen_labels=specimens[7:],
                abstained_specimen_diseases=["Covid19"] * 3,
            ),
            3: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
        }
        with pytest.raises(ValueError, match="fill_models13_mean requires Models 1 and 3"):
            build_feature_matrix(
                preds, gene_locus="TCR", reference_class=None,
                model2_abstention_strategy="fill_models13_mean",
            )

    def test_fill_models13_mean_m2_fully_abstains_m1_ok(self):
        """fill_models13_mean with M2 fully abstaining: M2 excluded, no error about M1/M3."""
        rng = np.random.RandomState(107)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = {
            1: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
            # M2 fully abstains
            2: ModelPredictions(
                probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
                abstained_specimen_labels=specimens,
                abstained_specimen_diseases=list(rng.choice(DISEASE_CLASSES, size=10)),
            ),
            3: ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            ),
        }
        # Should NOT raise — M2 is excluded, fill_models13_mean validation skipped
        X, abstained, _, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_models13_mean",
        )
        assert X.shape[0] == 10
        assert fill_info.get("excluded_models") == [2]

    def test_fill_no_abstentions_noop(self):
        """Fill strategy active but M2 has no abstentions: fill_info is empty."""
        rng = np.random.RandomState(105)
        specimens = [f"spec_{i:03d}" for i in range(15)]
        preds = {}
        for num in [1, 2, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[], abstained_specimen_diseases=[],
            )
        X, abstained, _, fill_info = build_feature_matrix(
            preds, gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )
        assert X.shape[0] == 15
        assert fill_info == {}
        assert len(abstained) == 0


class TestTrainMetamodel:
    def test_fit_and_predict(self):
        """Test 6: train_metamodel on small synthetic data."""
        rng = np.random.RandomState(5)
        n, nf = 60, 18
        X = pd.DataFrame(rng.randn(n, nf),
                         index=[f"spec_{i:03d}" for i in range(n)],
                         columns=[f"feat_{i}" for i in range(nf)])
        y = pd.Series([cls for cls in DISEASE_CLASSES for _ in range(10)], index=X.index)
        groups = pd.Series([f"P{i // 2:03d}" for i in range(n)], index=X.index)

        pipeline = train_metamodel(X, y, groups)
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")
        assert set(pipeline.classes_) == set(DISEASE_CLASSES)

        y_proba = pipeline.predict_proba(X.values)
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)


class TestEvaluatePredictions:
    def test_multiclass(self):
        """Test 7: multiclass metrics."""
        rng = np.random.RandomState(6)
        classes = np.array(sorted(DISEASE_CLASSES))
        n = 60
        y_true = np.array([classes[i % len(classes)] for i in range(n)])
        y_proba = np.zeros((n, len(classes)))
        for i in range(n):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i] = rng.dirichlet(np.ones(len(classes)))
            y_proba[i, true_idx] += 0.5
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = classes[np.argmax(y_proba, axis=1)]

        metrics, raw_preds = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_id=0, model_label="test_model",
            n_scored=n, n_abstained=0, reference_class=None,
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["auroc_ovo_weighted"] is not None
        assert metrics["mcc"] is not None
        assert len(raw_preds["y_true"]) == n

    def test_binary(self):
        """Test 8: binary metrics (2 classes + reference_class)."""
        rng = np.random.RandomState(7)
        classes = np.array(sorted([BINARY_DISEASE, BINARY_REFERENCE]))
        n = 40
        y_true = np.array([classes[i % 2] for i in range(n)])
        y_proba = np.zeros((n, 2))
        for i in range(n):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i] = rng.dirichlet(np.ones(2))
            y_proba[i, true_idx] += 0.3
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = classes[np.argmax(y_proba, axis=1)]

        metrics, _ = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_id=0, model_label="binary_model",
            n_scored=n, n_abstained=0, reference_class=BINARY_REFERENCE,
        )
        assert "auroc_binary" in metrics
        assert metrics["auroc_binary"] is not None
        assert metrics["auroc_ovo_weighted"] is None

    def test_perfect(self):
        """Test 9: perfect predictions."""
        classes = np.array(sorted(DISEASE_CLASSES))
        n = 30
        y_true = np.array([classes[i % len(classes)] for i in range(n)])
        y_proba = np.zeros((n, len(classes)))
        for i in range(n):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i, true_idx] = 1.0
        metrics, _ = evaluate_predictions(
            y_true=y_true, y_pred=y_true.copy(), y_proba=y_proba,
            classes=classes, fold_id=0, model_label="perfect",
            n_scored=n, n_abstained=0,
        )
        assert metrics["accuracy"] == 1.0
        assert metrics["mcc"] == 1.0

    def test_abstention_penalty(self):
        """Test 10: abstention penalty."""
        classes = np.array(sorted(DISEASE_CLASSES))
        n_scored, n_abstained = 20, 10
        y_true = np.array([classes[i % len(classes)] for i in range(n_scored)])
        y_proba = np.zeros((n_scored, len(classes)))
        for i in range(n_scored):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i, true_idx] = 1.0
        metrics, _ = evaluate_predictions(
            y_true=y_true, y_pred=y_true.copy(), y_proba=y_proba,
            classes=classes, fold_id=0, model_label="with_abstention",
            n_scored=n_scored, n_abstained=n_abstained,
        )
        expected_acc = n_scored / (n_scored + n_abstained)
        assert abs(metrics["accuracy"] - expected_acc) < 1e-6


class TestAggregateFoldResults:
    def test_multiclass_with_mcc(self):
        """Tests 12+13: multiclass aggregation with macro/weighted + MCC."""
        classes = np.array(sorted(DISEASE_CLASSES))
        rng = np.random.RandomState(12)
        fold_metrics, fold_raw = [], []
        for fold_id in range(3):
            n = 30
            y_true = np.array([classes[i % len(classes)] for i in range(n)])
            y_proba = rng.dirichlet(np.ones(len(classes)), size=n)
            for i in range(n):
                true_idx = np.where(classes == y_true[i])[0][0]
                y_proba[i, true_idx] += 1.0
            y_proba /= y_proba.sum(axis=1, keepdims=True)
            y_pred = classes[np.argmax(y_proba, axis=1)]
            m, r = evaluate_predictions(
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                classes=classes, fold_id=fold_id, model_label="test",
                n_scored=n, n_abstained=0,
            )
            fold_metrics.append(m)
            fold_raw.append(r)

        agg = aggregate_fold_results(fold_metrics, fold_raw, disease_filter=None)
        assert "accuracy_global" in agg
        for key in ["auroc_ovo_weighted", "auroc_ovo_macro", "mcc"]:
            assert key in agg
            assert isinstance(agg[key], dict)
            assert "mean" in agg[key]
            assert len(agg[key]["per_fold"]) == 3

    def test_binary_with_mcc_and_log_loss(self):
        """Test 13 (binary): includes MCC, log_loss, auroc_pooled."""
        classes = np.array(["Covid19", "Healthy/Background"])
        rng = np.random.RandomState(16)
        fold_metrics, fold_raw = [], []
        for fold_id in range(3):
            n = 20
            y_true = np.array([classes[i % 2] for i in range(n)])
            y_proba = rng.dirichlet(np.ones(2), size=n)
            for i in range(n):
                true_idx = np.where(classes == y_true[i])[0][0]
                y_proba[i, true_idx] += 1.0
            y_proba /= y_proba.sum(axis=1, keepdims=True)
            y_pred = classes[np.argmax(y_proba, axis=1)]
            m, r = evaluate_predictions(
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                classes=classes, fold_id=fold_id, model_label="binary",
                n_scored=n, n_abstained=0, reference_class="Healthy/Background",
            )
            fold_metrics.append(m)
            fold_raw.append(r)
        agg = aggregate_fold_results(
            fold_metrics, fold_raw, disease_filter=("Covid19", "Healthy/Background"),
        )
        assert "auroc_pooled" in agg
        assert "mcc" in agg
        assert "log_loss" in agg
        assert agg["disease"] == "Covid19"


class TestResultsGeneration:
    def test_multiclass_md(self):
        """Test 14: _generate_ensemble_results_md output format."""
        ensemble_agg = {
            "accuracy_global": 0.75,
            "accuracy_per_fold": {"mean": 0.75, "std": 0.02, "per_fold": [0.73, 0.75, 0.77]},
            "auroc_ovo_weighted": {"mean": 0.90, "std": 0.01, "per_fold": [0.89, 0.90, 0.91]},
            "auroc_ovo_macro": {"mean": 0.88, "std": 0.02, "per_fold": [0.86, 0.88, 0.90]},
            "mcc": {"mean": 0.60, "std": 0.03, "per_fold": [0.57, 0.60, 0.63]},
            "confusion_matrix_aggregated": [[10, 2], [3, 15]],
            "classes": ["Covid19", "Healthy/Background"],
        }
        base_model_agg = {1: {
            "accuracy_global": 0.70,
            "auroc_ovo_weighted": {"mean": 0.85, "std": 0.02},
            "auroc_ovo_macro": {"mean": 0.83, "std": 0.03},
            "mcc": {"mean": 0.50, "std": 0.04},
        }}
        all_fold_results = [{
            "fold_id": 0,
            "ensemble_metrics": {
                "fold_id": 0, "accuracy": 0.75, "auroc_ovo_weighted": 0.90,
                "mcc": 0.60, "n_scored": 30, "n_abstained": 0,
            },
            "base_model_metrics": {1: {
                "fold_id": 0, "accuracy": 0.70, "auroc_ovo_weighted": 0.85, "mcc": 0.50,
            }},
            "test_fill_info": {},
            "val_fill_info": {},
        }]
        run_config = {"dataset_name": "test-dataset", "classification_mode": "multiclass",
                      "gene_locus": "TCR", "models_included": [1]}
        md = _generate_ensemble_results_md(
            run_config=run_config, ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg, model_nums=[1],
            all_fold_results=all_fold_results, timestamp="20260422_120000",
        )
        assert "# Ensemble Training Results" in md
        assert "Model Comparison" in md

    def test_binary_md(self):
        """Test 17+23: Binary MD with accuracy, MCC, pooled AUROC, no OvO."""
        ensemble_agg = {
            "accuracy_global": 0.85,
            "accuracy_per_fold": {"mean": 0.85, "std": 0.02, "per_fold": [0.83, 0.85, 0.87]},
            "auroc_pooled": 0.92, "auprc_pooled": 0.88,
            "mcc": {"mean": 0.70, "std": 0.03, "per_fold": [0.67, 0.70, 0.73]},
            "log_loss": {"mean": 0.35, "std": 0.02, "per_fold": [0.33, 0.35, 0.37]},
            "confusion_matrix_aggregated": [[18, 2], [4, 16]],
            "classes": ["Covid19", "Healthy/Background"],
            "disease": "Covid19", "reference_class": "Healthy/Background",
        }
        base_model_agg = {1: {
            "accuracy_global": 0.80, "auroc_pooled": 0.88, "auprc_pooled": 0.84,
            "mcc": {"mean": 0.60, "std": 0.04},
        }}
        all_fold_results = [{
            "fold_id": 0,
            "ensemble_metrics": {
                "fold_id": 0, "accuracy": 0.85, "auroc_binary": 0.92,
                "mcc": 0.70, "n_scored": 40, "n_abstained": 0,
            },
            "base_model_metrics": {1: {
                "fold_id": 0, "accuracy": 0.80, "auroc_binary": 0.88, "mcc": 0.60,
            }},
            "test_fill_info": {},
            "val_fill_info": {},
        }]
        run_config = {"dataset_name": "test-dataset", "classification_mode": "binary",
                      "gene_locus": "TCR", "disease_filter": ["Covid19", "Healthy/Background"]}
        md = _generate_ensemble_results_md(
            run_config=run_config, ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg, model_nums=[1],
            all_fold_results=all_fold_results, timestamp="20260422_130000",
        )
        assert "AUROC (pooled)" in md
        assert "AUROC OvO" not in md

    def test_log_comparison_table(self):
        """Test 15: _log_comparison_table output."""
        import io
        ensemble_agg = {
            "accuracy_global": 0.80,
            "auroc_ovo_weighted": {"mean": 0.92, "std": 0.01},
            "mcc": {"mean": 0.65, "std": 0.02},
        }
        base_model_agg = {
            1: {"accuracy_global": 0.75,
                "auroc_ovo_weighted": {"mean": 0.88, "std": 0.02},
                "mcc": {"mean": 0.55, "std": 0.03}},
        }
        log_handler = logging.StreamHandler(io.StringIO())
        log_handler.setLevel(logging.INFO)
        comp_logger = logging.getLogger("malid_lite.training.train_ensemble")
        old_level = comp_logger.level
        comp_logger.setLevel(logging.INFO)
        comp_logger.addHandler(log_handler)
        try:
            _log_comparison_table(ensemble_agg, base_model_agg, [1])
            log_output = log_handler.stream.getvalue()
        finally:
            comp_logger.removeHandler(log_handler)
            comp_logger.setLevel(old_level)
        assert "Ensemble" in log_output

    def test_save_multi_binary_summary(self):
        """Test 18: _save_multi_binary_summary writes MD and JSON."""
        tmp_dir = _get_test_output_dir("multi_binary_summary")
        pairs = [("Covid19", "Healthy/Background"), ("HIV", "Healthy/Background")]
        summaries = {}
        fold_results_by_pair = {}
        for disease, ref in pairs:
            pk = make_pair_name(disease, ref)
            summaries[pk] = {
                "ensemble": {
                    "accuracy_global": 0.85,
                    "auroc_pooled": 0.90,
                    "auprc_pooled": 0.87,
                    "mcc": {"mean": 0.65, "std": 0.03},
                },
            }
            fold_results_by_pair[pk] = [{
                "fold_id": 0,
                "ensemble_metrics": {
                    "fold_id": 0, "accuracy": 0.85, "auroc_binary": 0.90,
                    "auprc_binary": 0.88, "mcc": 0.70, "n_scored": 10, "n_abstained": 0,
                },
                "base_model_metrics": {},
                "test_abstained_details": [],
            }]

        _save_multi_binary_summary(tmp_dir, summaries, fold_results_by_pair, pairs, "Healthy/Background")
        md_files = list(tmp_dir.glob("MULTI_BINARY_SUMMARY_*.md"))
        json_files = list(tmp_dir.glob("multi_binary_summary_*.json"))
        assert len(md_files) == 1
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            cross_json = json.load(f)
        assert cross_json["n_pairs"] == 2

    def test_md_with_fill_strategy(self):
        """MD generation with fill strategy: investigation section and fill stats."""
        import types

        classes = ["Covid19", "Healthy/Background"]
        # Mock pipeline with classes_ attribute
        mock_pipeline = types.SimpleNamespace(classes_=np.array(classes))

        # Build prediction rows: 4 real, 2 filled
        predictions_rows = []
        for i in range(4):
            predictions_rows.append({
                "specimen_label": f"real_{i}",
                "true_disease": classes[i % 2],
                "ensemble_predicted": classes[i % 2],
                "ensemble_P(Covid19)": 0.8 if i % 2 == 0 else 0.2,
                "ensemble_P(Healthy/Background)": 0.2 if i % 2 == 0 else 0.8,
                "model2_filled": False,
                "abstained": False,
            })
        for i in range(2):
            predictions_rows.append({
                "specimen_label": f"filled_{i}",
                "true_disease": classes[i % 2],
                "ensemble_predicted": classes[i % 2],
                "ensemble_P(Covid19)": 0.6 if i % 2 == 0 else 0.4,
                "ensemble_P(Healthy/Background)": 0.4 if i % 2 == 0 else 0.6,
                "model2_filled": True,
                "abstained": False,
            })

        ensemble_agg = {
            "accuracy_global": 0.85,
            "accuracy_per_fold": {"mean": 0.85, "std": 0.02, "per_fold": [0.85]},
            "auroc_pooled": 0.92, "auprc_pooled": 0.88,
            "mcc": {"mean": 0.70, "std": 0.03, "per_fold": [0.70]},
            "log_loss": {"mean": 0.35, "std": 0.02, "per_fold": [0.35]},
            "confusion_matrix_aggregated": [[2, 0], [0, 4]],
            "classes": classes,
            "disease": "Covid19", "reference_class": "Healthy/Background",
        }
        base_model_agg = {1: {
            "accuracy_global": 0.80, "auroc_pooled": 0.88, "auprc_pooled": 0.84,
            "mcc": {"mean": 0.60, "std": 0.04},
        }, 2: {
            "accuracy_global": 0.75, "auroc_pooled": 0.85, "auprc_pooled": 0.82,
            "mcc": {"mean": 0.55, "std": 0.05},
        }}

        all_fold_results = [{
            "fold_id": 0,
            "ensemble_metrics": {
                "fold_id": 0, "accuracy": 0.85, "auroc_binary": 0.92,
                "mcc": 0.70, "n_scored": 6, "n_abstained": 0,
            },
            "base_model_metrics": {
                1: {"fold_id": 0, "accuracy": 0.80, "auroc_binary": 0.88, "mcc": 0.60},
                2: {"fold_id": 0, "accuracy": 0.75, "auroc_binary": 0.85, "mcc": 0.55},
            },
            "test_fill_info": {
                "strategy": "fill_0.5", "n_filled": 2,
                "filled_specimen_labels": ["filled_0", "filled_1"],
                "filled_per_class": {"Covid19": 1, "Healthy/Background": 1},
            },
            "val_fill_info": {
                "strategy": "fill_0.5", "n_filled": 3,
                "filled_specimen_labels": ["val_f0", "val_f1", "val_f2"],
                "filled_per_class": {"Covid19": 2, "Healthy/Background": 1},
            },
            "predictions_rows": predictions_rows,
            "pipeline": mock_pipeline,
        }]

        run_config = {
            "dataset_name": "test-dataset",
            "classification_mode": "binary",
            "gene_locus": "TCR",
            "disease_filter": ["Covid19", "Healthy/Background"],
            "reference_class": "Healthy/Background",
        }

        md = _generate_ensemble_results_md(
            run_config=run_config, ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg, model_nums=[1, 2],
            all_fold_results=all_fold_results, timestamp="20260426_120000",
            model2_abstention_strategy="fill_0.5",
        )

        # Verify fill-specific sections appear
        assert "fill_0.5" in md
        assert "Model 2 Filled Specimens" in md
        assert "Investigation" in md
        assert "Fill Status" in md
        # Verify fill counts
        assert "2 specimens filled" in md  # test fills
        assert "3 specimens filled" in md  # validation fills
        # Verify abstention handling section mentions fill
        assert "Model 2 abstention strategy" in md


class TestPathConstruction:
    def test_make_pair_name(self):
        """Test 19: make_pair_name and model/ensemble output dirs."""
        assert make_pair_name("Covid19", "Healthy/Background") == "Covid19_vs_Healthy_Background"
        dir_bin = get_model_output_dir(
            model_name="model1", dataset_name="test-data",
            classification_mode="binary", gene_locus="TCR",
            training_context="cv_ensemble",
        )
        dir_mb = get_model_output_dir(
            model_name="model1", dataset_name="test-data",
            classification_mode="multi-binary", gene_locus="TCR",
            training_context="cv_ensemble",
        )
        assert dir_bin == dir_mb
        assert dir_bin.name == "binary"


class TestValidateModeAndClasses:
    def test_error_paths(self):
        """Test 20: validate_mode_and_classes error paths."""
        classes_4 = ["Covid19", "HIV", "Healthy/Background", "Lupus"]
        classes_2 = ["Covid19", "Healthy/Background"]

        with pytest.raises(ValueError, match="reference-class"):
            validate_mode_and_classes("multi-binary", classes_4, None, None)
        with pytest.raises(ValueError, match="not found"):
            validate_mode_and_classes("multi-binary", classes_4, "InvalidClass", None)
        ref = validate_mode_and_classes("multi-binary", classes_4, "Healthy/Background", None)
        assert ref == "Healthy/Background"

        with pytest.raises(ValueError, match="reference-class"):
            validate_mode_and_classes("multi-binary", classes_2, None, None)
        with pytest.raises(ValueError, match="reference-class"):
            validate_mode_and_classes("binary", classes_2, None, None)


class TestMockedTrainEnsemble:
    def test_binary_flow(self):
        """Test 21: Full binary train_ensemble() with mocked folds."""
        classes = np.array(["Covid19", "Healthy/Background"])
        ref_class = "Healthy/Background"
        model_nums = [1, 2]
        fold_ids = [0, 1]

        def _mock_run_fold(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=classes,
                n_specimens=20, model_nums=model_nums, reference_class=ref_class,
            )

        tmp = _get_test_output_dir("binary_mocked")
        with patch("malid_lite.training.train_ensemble.run_ensemble_fold",
                    side_effect=_mock_run_fold):
            fold_results, summary = train_ensemble(
                loader=None, fold_ids=fold_ids, model_nums=model_nums,
                model_dirs={1: Path("dummy"), 2: Path("dummy")},
                gene_locus="TCR", output_dir=tmp,
                disease_filter=("Covid19", ref_class), reference_class=ref_class,
                run_config={"classification_mode": "binary", "test": True},
            )
        assert len(fold_results) == 2
        assert "auroc_pooled" in summary["ensemble"]

    def test_multi_binary_orchestration(self):
        """Test 22: Multi-binary orchestration with 2 pairs."""
        ref_class = "Healthy/Background"
        model_nums = [1, 3]
        fold_ids = [0, 1]
        pairs = [("Covid19", ref_class), ("HIV", ref_class)]
        base_output = _get_test_output_dir("multi_binary_orchestration")

        all_pair_summaries, all_pair_fold_results = {}, {}
        for disease, ref in pairs:
            classes = np.array(sorted([disease, ref]))
            pk = make_pair_name(disease, ref)
            output_dir = base_output / pk

            def _mock_run_fold(disease_=disease, classes_=classes, **kwargs):
                return _make_mock_fold_result(
                    fold_id=kwargs["fold_id"], classes=classes_,
                    n_specimens=16, model_nums=model_nums, reference_class=ref,
                )

            with patch("malid_lite.training.train_ensemble.run_ensemble_fold",
                        side_effect=_mock_run_fold):
                fold_results, summary = train_ensemble(
                    loader=None, fold_ids=fold_ids, model_nums=model_nums,
                    model_dirs={n: Path("dummy") for n in model_nums},
                    gene_locus="TCR", output_dir=output_dir,
                    disease_filter=(disease, ref), reference_class=ref,
                    run_config={"classification_mode": "multi-binary",
                                "disease_filter": [disease, ref]},
                )
            all_pair_summaries[pk] = summary
            all_pair_fold_results[pk] = fold_results

        _save_multi_binary_summary(
            base_output, all_pair_summaries, all_pair_fold_results, pairs, ref_class,
        )
        mb_json = list(base_output.glob("multi_binary_summary_*.json"))
        assert len(mb_json) == 1
        with open(mb_json[0]) as f:
            cross = json.load(f)
        assert cross["n_pairs"] == 2


class TestBinarySpecimenFiltering:
    def test_disease_filter_restricts_specimens(self):
        """Test 24: In binary mode, only target-disease specimens counted."""
        test_meta = pd.DataFrame({
            SPECIMEN_COL: [f"spec_{i:03d}" for i in range(30)],
            DISEASE_COL: ["Covid19"] * 10 + ["Healthy/Background"] * 10 + ["HIV"] * 10,
        })
        disease_filter = ("Covid19", "Healthy/Background")
        filtered = set(test_meta[test_meta[DISEASE_COL].isin(set(disease_filter))][SPECIMEN_COL])
        assert len(filtered) == 20

        n_scored = 15
        n_abstained_correct = len(filtered) - n_scored
        assert n_abstained_correct == 5


class TestResumeMatchesOriginal:
    def test_multiclass_resume(self):
        """Test 25a: Multiclass resume produces identical metrics."""
        classes = np.array(["Covid19", "HIV", "Healthy"])
        model_nums = [1, 3]
        fold_result, metadata_df = _make_mock_fold_result_with_features(
            fold_id=0, gene_locus="TCR", classes=classes, model_nums=model_nums,
            n_val=30, n_test=18, n_abstained=2, seed=42,
        )
        tmp = _get_test_output_dir("resume_multiclass")
        save_fold_artifacts(tmp, fold_result)

        class MockLoader:
            @property
            def metadata(self):
                return metadata_df

        resume_result = run_ensemble_fold_from_features(
            fold_id=0, output_dir=tmp, model_nums=model_nums,
            gene_locus="TCR", loader=MockLoader(), reference_class=None,
        )
        for model_num in model_nums:
            orig = fold_result["base_model_metrics"][model_num]
            resu = resume_result["base_model_metrics"][model_num]
            for key in ["accuracy", "mcc", "n_scored", "n_abstained"]:
                assert orig.get(key) == resu.get(key), f"M{model_num} {key} mismatch"

    def test_binary_resume(self):
        """Test 25b: Binary resume produces identical metrics."""
        classes = np.array(["Covid19", "Healthy/Background"])
        ref_class = "Healthy/Background"
        model_nums = [1, 2]
        fold_result, metadata_df = _make_mock_fold_result_with_features(
            fold_id=0, gene_locus="TCR", classes=classes, model_nums=model_nums,
            n_val=20, n_test=16, n_abstained=1, reference_class=ref_class, seed=99,
        )
        tmp = _get_test_output_dir("resume_binary")
        save_fold_artifacts(tmp, fold_result)

        class MockLoader:
            @property
            def metadata(self):
                return metadata_df

        resume_result = run_ensemble_fold_from_features(
            fold_id=0, output_dir=tmp, model_nums=model_nums,
            gene_locus="TCR", loader=MockLoader(), reference_class=ref_class,
        )
        for key in ["accuracy", "mcc", "n_scored", "n_abstained"]:
            assert fold_result["ensemble_metrics"].get(key) == resume_result["ensemble_metrics"].get(key)

    def test_round_trip(self):
        """Test 25c: Full train_ensemble round-trip via mocked folds."""
        classes = np.array(["A", "B", "C"])
        model_nums = [1, 3]
        fold_ids = [0, 1]

        fold_results_map = {}
        metadata_dfs = []
        for fid in fold_ids:
            fr, meta = _make_mock_fold_result_with_features(
                fold_id=fid, gene_locus="TCR", classes=classes,
                model_nums=model_nums, n_val=30, n_test=18,
                n_abstained=1, seed=fid * 10 + 7,
            )
            fold_results_map[fid] = fr
            metadata_dfs.append(meta)

        combined_metadata = pd.concat(metadata_dfs, ignore_index=True).drop_duplicates(
            subset=[SPECIMEN_COL],
        )

        class MockLoader:
            @property
            def metadata(self):
                return combined_metadata

        def _mock_run_fold(**kwargs):
            return fold_results_map[kwargs["fold_id"]]

        tmp = _get_test_output_dir("resume_roundtrip")
        with patch("malid_lite.training.train_ensemble.run_ensemble_fold",
                    side_effect=_mock_run_fold):
            _, original_summary = train_ensemble(
                loader=MockLoader(), fold_ids=fold_ids, model_nums=model_nums,
                model_dirs={n: Path("dummy") for n in model_nums},
                gene_locus="TCR", output_dir=tmp,
                run_config={"test": True, "classification_mode": "multiclass"},
            )

        _, resume_summary = train_ensemble(
            loader=MockLoader(), fold_ids=fold_ids, model_nums=model_nums,
            model_dirs={}, gene_locus="TCR", output_dir=tmp, resume=True,
            run_config={"test": True, "classification_mode": "multiclass", "resume": True},
        )

        assert abs(original_summary["ensemble"]["accuracy_global"]
                    - resume_summary["ensemble"]["accuracy_global"]) < 1e-10


# =========================================================================
# Auto-training tests (consolidated)
# =========================================================================

class TestCompareTrainingParams:
    """Consolidated tests 31-36b: compare_training_params."""

    def test_matching(self):
        summary = {"n_pcs": 15, "l1_ratio": 1.0, "model_names": ["lasso_cv"]}
        cli = {"n_pcs": 15, "l1_ratio": 1.0, "model_name": "lasso_cv"}
        assert compare_training_params(1, summary, cli) == []

    def test_mismatch(self):
        summary = {"n_pcs": 15, "l1_ratio": 1.0}
        cli = {"n_pcs": 20, "l1_ratio": 0.5}
        mismatches = compare_training_params(1, summary, cli)
        assert len(mismatches) == 2
        assert set(m[0] for m in mismatches) == {"n_pcs", "l1_ratio"}

    def test_none_skip(self):
        summary = {"n_pcs": 15, "l1_ratio": 1.0}
        cli = {"n_pcs": None, "l1_ratio": 1.0, "model_name": None}
        assert compare_training_params(1, summary, cli) == []

    def test_model1_model_name(self):
        summary = {"model_names": ["lasso_cv"]}
        assert compare_training_params(1, summary, {"model_name": "lasso_cv"}) == []
        mismatches = compare_training_params(1, summary, {"model_name": "ridge_cv"})
        assert len(mismatches) == 1

    def test_list_order_independent(self):
        summary = {"p_values": [0.05, 0.001, 0.01, 0.005, 0.0005]}
        cli = {"p_values": [0.0005, 0.001, 0.005, 0.01, 0.05]}
        assert compare_training_params(2, summary, cli) == []
        assert len(compare_training_params(2, summary, {"p_values": [0.001]})) == 1

    def test_model3_all_params(self):
        summary = {
            "aggregation_strategy": "auto_tuned", "n_estimators_stage1": 100,
            "n_estimators_stage2": 100, "tuning_cv_splits": 3,
            "tuning_strategies": ["entropy_cutoff", "entropy_percentile_cutoff"],
        }
        cli = {
            "aggregation_strategy": "auto_tuned", "n_estimators_stage1": 100,
            "n_estimators_stage2": None,
            "tuning_strategies": ["entropy_percentile_cutoff", "entropy_cutoff"],
        }
        assert compare_training_params(3, summary, cli) == []

    def test_missing_summary_key_skipped(self):
        summary = {"aggregation_strategy": "auto_tuned"}
        cli = {"aggregation_strategy": "auto_tuned", "n_estimators_stage1": 100}
        assert compare_training_params(3, summary, cli) == []


class TestResolveBaseModelMode:
    """Consolidated tests 37-41: resolve_base_model_mode."""

    def test_retrain(self):
        mode, _, summary = resolve_base_model_mode(
            model_num=1, retrain_set={1}, resume_flag=False,
            dataset_name="test-dataset", classification_mode="multiclass",
            gene_locus="TCR", output_suffix=None, cli_training_params={},
        )
        assert mode == "TRAIN"
        assert summary is None

    def test_no_artifacts(self):
        mode, _, _ = resolve_base_model_mode(
            model_num=2, retrain_set=set(), resume_flag=False,
            dataset_name="nonexistent-xyz", classification_mode="multiclass",
            gene_locus="TCR", output_suffix=None, cli_training_params={},
        )
        assert mode == "TRAIN"

    def test_load_with_summary(self):
        import malid_lite.training.training_utils as tu
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        summary_data = {
            "gene_locus": "TCR", "training_context": "cv_ensemble",
            "classification_mode": "multiclass", "n_pcs": 15,
        }
        with open(model_dir / "summary_20260425.json", "w") as f:
            json.dump(summary_data, f)
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            mode, _, summary = resolve_base_model_mode(
                model_num=1, retrain_set=set(), resume_flag=False,
                dataset_name="test-ds", classification_mode="multiclass",
                gene_locus="TCR", output_suffix=None, cli_training_params={"n_pcs": 15},
            )
            assert mode == "LOAD"
            assert summary["n_pcs"] == 15
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)

    def test_load_param_mismatch(self):
        import malid_lite.training.training_utils as tu
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        with open(model_dir / "summary_20260425.json", "w") as f:
            json.dump({"gene_locus": "TCR", "training_context": "cv_ensemble",
                        "classification_mode": "multiclass", "n_pcs": 15}, f)
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            with pytest.raises(ValueError, match="n_pcs"):
                resolve_base_model_mode(
                    model_num=1, retrain_set=set(), resume_flag=False,
                    dataset_name="test-ds", classification_mode="multiclass",
                    gene_locus="TCR", output_suffix=None, cli_training_params={"n_pcs": 20},
                )
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)

    def test_resume(self):
        import malid_lite.training.training_utils as tu
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        (model_dir / "fold_0_lasso_cv.pkl").touch()
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            mode, _, _ = resolve_base_model_mode(
                model_num=1, retrain_set=set(), resume_flag=True,
                dataset_name="test-ds", classification_mode="multiclass",
                gene_locus="TCR", output_suffix=None, cli_training_params={},
            )
            assert mode == "RESUME"
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)


class TestCLIArgValidation:
    """Tests 42-43: CLI arg interaction (subprocess)."""

    def test_retrain_not_in_models(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "malid_lite.training.train_ensemble",
             "--models", "1", "2", "--retrain-models", "3",
             "--metadata-path", "/nonexistent"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_resume_retrain_conflict(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "malid_lite.training.train_ensemble",
             "--resume", "--retrain-base-models",
             "--metadata-path", "/nonexistent"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


class TestPreflightValidateResume:
    """Tests 44-45: preflight_validate_resume_params."""

    def test_no_meta_no_error(self):
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "fold_0_lasso_cv.pkl").touch()
        preflight_validate_resume_params(
            model_num=1, model_dir=tmpdir, cli_training_params={"n_pcs": 20},
        )
        shutil.rmtree(tmpdir)

    def test_meta_mismatch(self):
        import joblib as jl
        tmpdir = Path(tempfile.mkdtemp())
        artifact = {"_meta": {"model_params": {"n_pcs": 15, "l1_ratio": 1.0}}}
        jl.dump(artifact, tmpdir / "fold_0_predictions.pkl")
        with pytest.raises(ValueError, match="n_pcs"):
            preflight_validate_resume_params(
                model_num=1, model_dir=tmpdir, cli_training_params={"n_pcs": 20},
            )
        shutil.rmtree(tmpdir)


# =========================================================================
# Dispatch tests (consolidated)
# =========================================================================

class TestFormatElapsedTime:
    def test_formats(self):
        """Test 46: _format_elapsed_time."""
        assert _format_elapsed_time(5) == "5s"
        assert _format_elapsed_time(90) == "1m 30s"
        assert _format_elapsed_time(3600) == "1h 0m"


class TestAutoTrainBaseModel:
    """Tests 47-52: auto_train_base_model dispatch."""

    def test_invalid_model_num(self):
        with pytest.raises(ValueError, match="4"):
            auto_train_base_model(
                model_num=4, training_params={}, output_dir=Path("/tmp/fake"),
                metadata_path=Path("/tmp/fake.tsv"), dataset_name="test",
                classification_mode="multiclass", reference_class=None,
                diseases=None, gene_locus="TCR", fold_ids=[0],
                data_dir=None, cache_dir=None, gene_reference_path=None,
                n_jobs=1, verbose=0, resume=False,
            )

    def test_model1_dispatch(self):
        mock_train = MagicMock()
        with patch("malid_lite.training.train_model1.train_all_folds", mock_train):
            auto_train_base_model(
                model_num=1, training_params={"n_pcs": 20, "l1_ratio": 0.5},
                output_dir=Path("/tmp/out"), metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds", classification_mode="multiclass",
                reference_class=None, diseases=None, gene_locus="TCR",
                fold_ids=[0, 1, 2], data_dir=None, cache_dir=Path("/tmp/cache"),
                gene_reference_path=None, n_jobs=4, verbose=1, resume=False,
            )
        kw = mock_train.call_args[1]
        assert kw["n_pcs"] == 20
        assert "n_jobs" not in kw

    def test_model2_dispatch(self):
        mock_train = MagicMock()
        with patch("malid_lite.training.train_model2.train_all_folds", mock_train):
            auto_train_base_model(
                model_num=2, training_params={"p_values": [0.001, 0.01]},
                output_dir=Path("/tmp/out"), metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds", classification_mode="multiclass",
                reference_class=None, diseases=None, gene_locus="TCR",
                fold_ids=[0], data_dir=None, cache_dir=None,
                gene_reference_path=None, n_jobs=4, verbose=1, resume=True,
            )
        kw = mock_train.call_args[1]
        assert kw["n_jobs"] == 4
        assert kw["resume"] is True

    def test_model3_dispatch(self):
        mock_train = MagicMock()
        with patch("malid_lite.training.train_model3.train_all_folds", mock_train):
            auto_train_base_model(
                model_num=3, training_params={"aggregation_strategy": "mean"},
                output_dir=Path("/tmp/out"), metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds", classification_mode="multiclass",
                reference_class=None, diseases=None, gene_locus="TCR",
                fold_ids=[0, 1], data_dir=None, cache_dir=Path("/tmp/cache"),
                gene_reference_path=None, n_jobs=4, verbose=1, resume=False,
                embedding_dir=Path("/tmp/embeddings"), no_cache_embeddings=False,
                device="cpu", embedding_batch_size=32,
            )
        kw = mock_train.call_args[1]
        assert kw["device"] == "cpu"
        assert kw["embedding_batch_size"] == 32
        assert kw["cache_embeddings"] is True

    def test_model3_optional_kwargs_omitted(self):
        mock_train = MagicMock()
        with patch("malid_lite.training.train_model3.train_all_folds", mock_train):
            auto_train_base_model(
                model_num=3, training_params={},
                output_dir=Path("/tmp/out"), metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds", classification_mode="multiclass",
                reference_class=None, diseases=None, gene_locus="TCR",
                fold_ids=[0], data_dir=None, cache_dir=None,
                gene_reference_path=None, n_jobs=4, verbose=1, resume=False,
            )
        kw = mock_train.call_args[1]
        assert "device" not in kw
        assert "embedding_batch_size" not in kw

    def test_empty_training_params(self):
        mock_train = MagicMock()
        with patch("malid_lite.training.train_model1.train_all_folds", mock_train):
            auto_train_base_model(
                model_num=1, training_params={},
                output_dir=Path("/tmp/out"), metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds", classification_mode="binary",
                reference_class="Healthy", diseases=["Covid19"],
                gene_locus="TCR", fold_ids=[0, 1, 2],
                data_dir=Path("/tmp/data"), cache_dir=Path("/tmp/cache"),
                gene_reference_path=None, n_jobs=4, verbose=1, resume=False,
            )
        kw = mock_train.call_args[1]
        assert kw["classification_mode"] == "binary"
        assert kw["reference_class"] == "Healthy"
        for k in ("n_pcs", "l1_ratio", "model_name"):
            assert k not in kw


# =========================================================================
# Argument validation tests (consolidated)
# =========================================================================

class TestValidateEnsembleArgs:
    """Consolidated tests 53-68: validate_ensemble_args."""

    def test_valid_defaults(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
        assert parser.error_message is None

    def test_resume_retrain_conflict(self):
        args = _make_base_namespace(resume=True)
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, {1}, _make_base_cli_params(), parser)
        assert "contradictory" in parser.error_message.lower()

    def test_retrain_not_in_models(self):
        args = _make_base_namespace(models=[1, 2], retrain_models=[3])
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_output_suffix_dir_conflict(self):
        args = _make_base_namespace(output_dir=Path("/tmp/out"), output_suffix="v1")
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_n_jobs_zero(self):
        args = _make_base_namespace(n_jobs=0)
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_diseases_in_multiclass(self):
        args = _make_base_namespace(diseases=["Covid19"])
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_model_specific_for_excluded(self):
        args = _make_base_namespace(models=[1, 3])
        parser = _FakeParser()
        cli_params = _make_base_cli_params()
        cli_params[2]["p_values"] = [0.01, 0.05]
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_suffix_for_excluded_model(self):
        args = _make_base_namespace(models=[1, 3], model2_suffix="v1")
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_m3_infra_excluded(self):
        args = _make_base_namespace(models=[1, 2], model3_device="cpu")
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_per_model_range(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params()
        cli_params[1]["l1_ratio"] = 1.5
        with pytest.raises(ValueError, match="l1_ratio"):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_suffix_sanitization(self):
        args = _make_base_namespace(output_suffix="my run/v1")
        parser = _FakeParser()
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
        assert args.output_suffix == "my_run_v1"

    def test_suffix_clean_passthrough(self):
        args = _make_base_namespace(output_suffix="my_run-v1.2")
        parser = _FakeParser()
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
        assert args.output_suffix == "my_run-v1.2"

    def test_metadata_path_missing(self):
        args = _make_base_namespace(metadata_path=Path("/nonexistent/meta.tsv"))
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_gene_reference_path_missing(self):
        args = _make_base_namespace(gene_reference_path=Path("/nonexistent/genes.csv"))
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_metadata_path_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".tsv") as tmp:
            args = _make_base_namespace(metadata_path=Path(tmp.name))
            parser = _FakeParser()
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
            assert parser.error_message is None

    def test_fill_models13_mean_without_m1_m3(self):
        """fill_models13_mean requires Models 1 and 3 in --models."""
        # Missing Model 3
        args = _make_base_namespace(models=[1, 2], model2_abstention_strategy="fill_models13_mean")
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
        assert "Models 1 and 3" in parser.error_message

        # Missing Model 1
        args = _make_base_namespace(models=[2, 3], model2_abstention_strategy="fill_models13_mean")
        parser = _FakeParser()
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
        assert "Models 1 and 3" in parser.error_message

    def test_fill_strategy_without_model2(self):
        """Fill strategies require Model 2 in --models."""
        for strategy in ("fill_0.5", "fill_models13_mean"):
            args = _make_base_namespace(models=[1, 3], model2_abstention_strategy=strategy)
            parser = _FakeParser()
            with pytest.raises(SystemExit):
                validate_ensemble_args(args, set(), _make_base_cli_params(), parser)
            assert "Model 2" in parser.error_message


class TestValidateEnsembleArgsM3:
    """Consolidated tests 69-76: M3 cross-param interactions."""

    def test_tuning_with_fixed_strategy_error(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean", tuning_strategies=["mean", "median"],
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_entropy_max_wrong_strategy_error(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean", entropy_max_fraction=0.5,
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_entropy_percentile_wrong_strategy_error(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean", entropy_bottom_percentile=10,
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_fixed_entropy_with_auto_tuned_error(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="auto_tuned", entropy_max_fraction=0.5,
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)

    def test_entropy_cutoff_valid(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="entropy_cutoff", entropy_max_fraction=0.5,
        )
        validate_ensemble_args(args, set(), cli_params, parser)
        assert parser.error_message is None

    def test_auto_tuned_with_tuning_flags_valid(self):
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="auto_tuned", tuning_strategies=["mean"],
        )
        validate_ensemble_args(args, set(), cli_params, parser)
        assert parser.error_message is None

    def test_unspecified_strategy_with_tuning_rejected(self):
        """Tuning flags with strategy=None (default entropy_percentile_cutoff) → rejected.

        The default is not auto_tuned, so tuning flags must be explicitly
        paired with --model3-aggregation-strategy auto_tuned.
        """
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy=None, tuning_strategies=["mean"],
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)
        assert parser.error_message is not None
        assert "auto_tuned" in parser.error_message

    def test_unspecified_strategy_with_entropy_max_fraction_rejected(self):
        """entropy_max_fraction with strategy=None → rejected.

        Default is entropy_percentile_cutoff, so entropy_max_fraction (which
        belongs to entropy_cutoff) is invalid without explicit strategy.
        """
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy=None, entropy_max_fraction=0.9,
        )
        with pytest.raises(SystemExit):
            validate_ensemble_args(args, set(), cli_params, parser)
        assert parser.error_message is not None
        assert "entropy_cutoff" in parser.error_message

    def test_unspecified_strategy_with_entropy_bottom_percentile_valid(self):
        """entropy_bottom_percentile with strategy=None → accepted.

        Default is entropy_percentile_cutoff, so entropy_bottom_percentile
        is valid (overrides the default 0.01 threshold).
        """
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy=None, entropy_bottom_percentile=0.05,
        )
        validate_ensemble_args(args, set(), cli_params, parser)
        assert parser.error_message is None


# =========================================================================
# Cross-model validation (consolidated)
# =========================================================================

class TestValidateCrossModelDiseaseClasses:
    """Consolidated tests 77-82."""

    def test_matching_classes(self):
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy", "HIV"]},
            2: {"model_classes": ["HIV", "Healthy", "Covid19"]},
        }
        _validate_cross_model_disease_classes(summaries, label="test")

    def test_mismatched_classes(self):
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: {"model_classes": ["Covid19", "Healthy", "HIV"]},
        }
        with pytest.raises(ValueError, match="mismatch"):
            _validate_cross_model_disease_classes(summaries, label="test")

    def test_none_summaries_skipped(self):
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: None,
            3: {"model_classes": ["Covid19", "Healthy"]},
        }
        _validate_cross_model_disease_classes(summaries, label="test")

    def test_missing_key_skipped(self):
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: {"other_key": "value"},
        }
        _validate_cross_model_disease_classes(summaries, label="test")

    def test_single_model(self):
        _validate_cross_model_disease_classes(
            {1: {"model_classes": ["Covid19"]}}, label="test",
        )

    def test_no_class_keys(self):
        _validate_cross_model_disease_classes(
            {1: {"other": "val"}, 2: {"another": 42}}, label="test",
        )


class TestLogBaseModelStatusTable:
    def test_load_train_resume(self):
        """Test 83: _log_base_model_status_table with all modes."""
        import io
        tmpdir = Path(tempfile.mkdtemp())
        resume_dir = tmpdir / "model2"
        resume_dir.mkdir()
        (resume_dir / "fold_0_clusters.pkl").touch()
        (resume_dir / "fold_3_clusters.pkl").touch()
        (resume_dir / "fold_info.json").touch()

        model_modes = {1: "LOAD", 2: "RESUME", 3: "TRAIN"}
        model_dirs = {1: tmpdir / "model1", 2: resume_dir, 3: tmpdir / "model3"}
        model_summaries = {1: {"timestamp": "2026-01-15"}, 2: None, 3: None}

        log_handler = logging.StreamHandler(io.StringIO())
        log_handler.setLevel(logging.DEBUG)
        _logger = logging.getLogger("malid_lite.training.train_ensemble")
        _logger.addHandler(log_handler)
        old_level = _logger.level
        _logger.setLevel(logging.DEBUG)
        try:
            _log_base_model_status_table(model_modes, model_dirs, model_summaries)
            log_output = log_handler.stream.getvalue()
        finally:
            _logger.removeHandler(log_handler)
            _logger.setLevel(old_level)
        shutil.rmtree(tmpdir)

        assert "LOAD" in log_output
        assert "RESUME" in log_output
        assert "TRAIN" in log_output


# =========================================================================
# New unit tests U1-U4
# =========================================================================

class TestResumeConfigMismatch:
    """U1: Resume config mismatch detection in train_ensemble."""

    def test_classification_mode_change(self):
        tmp = _get_test_output_dir("u1_resume_config_mismatch")
        # Write a run_config.json with multiclass
        prev_config = {
            "classification_mode": "multiclass",
            "gene_locus": "TCR",
            "models_included": [1, 2, 3],
        }
        with open(tmp / "run_config.json", "w") as f:
            json.dump(prev_config, f)

        # Resume with binary -> should raise
        with pytest.raises(ValueError, match="Cannot resume"):
            train_ensemble(
                loader=None, fold_ids=[0], model_nums=[1, 2, 3],
                model_dirs={}, gene_locus="TCR", output_dir=tmp,
                resume=True,
                run_config={"classification_mode": "binary", "gene_locus": "TCR",
                            "models_included": [1, 2, 3]},
            )

    def test_gene_locus_change(self):
        tmp = _get_test_output_dir("u1_resume_gene_locus_mismatch")
        with open(tmp / "run_config.json", "w") as f:
            json.dump({"classification_mode": "multiclass", "gene_locus": "TCR"}, f)

        with pytest.raises(ValueError, match="Cannot resume"):
            train_ensemble(
                loader=None, fold_ids=[0], model_nums=[1],
                model_dirs={}, gene_locus="BCR", output_dir=tmp,
                resume=True,
                run_config={"classification_mode": "multiclass", "gene_locus": "BCR"},
            )


class TestLoadModelExpectedConfig:
    """U2: validate_model_summary with reference_class/diseases."""

    def test_reference_class_mismatch(self):
        summary = {
            "gene_locus": "TCR",
            "training_context": "cv_ensemble",
            "classification_mode": "binary",
            "reference_class": "Healthy/Background",
            "diseases": ["Covid19"],
        }
        expected = {**summary, "reference_class": "HIV"}
        with pytest.raises(ValueError, match="reference_class"):
            validate_model_summary(summary, expected, model_label="test model")

    def test_diseases_mismatch(self):
        summary = {
            "gene_locus": "TCR",
            "training_context": "cv_ensemble",
            "classification_mode": "binary",
            "reference_class": "Healthy/Background",
            "diseases": ["Covid19"],
        }
        expected = {**summary, "diseases": ["HIV"]}
        with pytest.raises(ValueError, match="diseases"):
            validate_model_summary(summary, expected, model_label="test model")

    def test_matching_config_passes(self):
        summary = {
            "gene_locus": "TCR",
            "training_context": "cv_ensemble",
            "classification_mode": "binary",
            "reference_class": "Healthy/Background",
            "diseases": ["Covid19"],
        }
        validate_model_summary(summary, summary, model_label="test model")


class TestBaseModelResumeRunParams:
    """U3: Base model RESUME with run-level param mismatch."""

    def test_classification_mode_in_resume_params(self):
        import joblib as jl
        tmpdir = Path(tempfile.mkdtemp())
        artifact = {"_meta": {"model_params": {
            "n_pcs": 15,
            "classification_mode": "multiclass",
        }}}
        jl.dump(artifact, tmpdir / "fold_0_predictions.pkl")
        with pytest.raises(ValueError, match="classification_mode"):
            preflight_validate_resume_params(
                model_num=1, model_dir=tmpdir,
                cli_training_params={"n_pcs": 15, "classification_mode": "binary"},
            )
        shutil.rmtree(tmpdir)


class TestEnsembleSummaryFields:
    """U4: Ensemble summary includes classification_mode/reference_class/diseases."""

    def test_summary_has_run_config_fields(self):
        classes = np.array(["Covid19", "Healthy/Background"])
        ref_class = "Healthy/Background"
        model_nums = [1, 2]

        def _mock_run_fold(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=classes,
                n_specimens=20, model_nums=model_nums, reference_class=ref_class,
            )

        tmp = _get_test_output_dir("u4_summary_fields")
        run_config = {
            "classification_mode": "binary",
            "reference_class": ref_class,
            "diseases": ["Covid19"],
        }

        with patch("malid_lite.training.train_ensemble.run_ensemble_fold",
                    side_effect=_mock_run_fold):
            _, summary = train_ensemble(
                loader=None, fold_ids=[0], model_nums=model_nums,
                model_dirs={1: Path("dummy"), 2: Path("dummy")},
                gene_locus="TCR", output_dir=tmp,
                disease_filter=("Covid19", ref_class), reference_class=ref_class,
                run_config=run_config,
            )

        assert summary["classification_mode"] == "binary"
        assert summary["reference_class"] == ref_class
        assert summary["diseases"] == ["Covid19"]

        # Verify persisted JSON also has these fields
        summary_files = list(tmp.glob("summary_*.json"))
        all_files = sorted(f.name for f in tmp.iterdir()) if tmp.exists() else []
        assert len(summary_files) == 1, (
            f"Expected 1 summary_*.json in {tmp}, found {len(summary_files)}. "
            f"Directory exists: {tmp.exists()}. "
            f"Files in dir: {all_files}"
        )
        with open(summary_files[0]) as f:
            saved = json.load(f)
        assert saved["classification_mode"] == "binary"
        assert saved["reference_class"] == ref_class
        assert saved["diseases"] == ["Covid19"]


# =========================================================================
# Raw feature matrix and load-time fill strategy tests
# =========================================================================

def _build_fill_test_data(
    n_specimens: int = 20,
    n_m2_abstained: int = 5,
    classes: List[str] = None,
    seed: int = 200,
) -> Tuple[Dict[int, "ModelPredictions"], List[str], List[str]]:
    """Build synthetic predictions for raw matrix / fill strategy tests.

    Returns (predictions_dict, m2_abstained_labels, m2_abstained_diseases).
    Models 1 and 3 score all specimens; Model 2 abstains on the last n_m2_abstained.
    """
    if classes is None:
        classes = DISEASE_CLASSES
    rng = np.random.RandomState(seed)
    all_specimens = [f"spec_{i:03d}" for i in range(n_specimens)]
    m2_scored = all_specimens[: n_specimens - n_m2_abstained]
    m2_abstained = all_specimens[n_specimens - n_m2_abstained :]
    m2_abstained_diseases = [classes[i % len(classes)] for i in range(n_m2_abstained)]

    preds = {}
    for num in [1, 3]:
        preds[num] = ModelPredictions(
            probabilities=_make_mock_proba(all_specimens, classes, rng),
            abstained_specimen_labels=[],
            abstained_specimen_diseases=[],
        )
    preds[2] = ModelPredictions(
        probabilities=_make_mock_proba(m2_scored, classes, rng),
        abstained_specimen_labels=m2_abstained,
        abstained_specimen_diseases=m2_abstained_diseases,
    )
    return preds, m2_abstained, m2_abstained_diseases


class TestBuildRawFeatureMatrix:
    """Tests for _build_raw_feature_matrix: strategy-agnostic raw matrix construction."""

    def test_raw_from_fill_0_5(self):
        """Raw matrix from fill_0.5: filled M2 values replaced with NaN."""
        preds, m2_abstained, _ = _build_fill_test_data()
        X_proc, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="fill_0.5",
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abstained, abstained_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )

        # Same shape (fill mode: all specimens present in both)
        assert X_raw.shape == X_proc.shape
        assert set(X_raw.index) == set(X_proc.index)

        # M2 columns are NaN for abstained specimens in raw, 0.5 in processed
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]
        for spec in m2_abstained:
            for col in m2_cols:
                assert np.isnan(X_raw.loc[spec, col]), (
                    f"Expected NaN for abstained {spec} col {col}, got {X_raw.loc[spec, col]}"
                )
                assert X_proc.loc[spec, col] == 0.5

        # Non-M2 columns unchanged
        non_m2_cols = [c for c in X_raw.columns if f":{m2_display}:" not in c]
        pd.testing.assert_frame_equal(X_raw[non_m2_cols], X_proc[non_m2_cols])

    def test_raw_from_ensemble_abstain(self):
        """Raw matrix from ensemble_abstain: adds back M2-abstained rows with M1/M3 real values."""
        preds, m2_abstained, _ = _build_fill_test_data()
        X_proc, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="ensemble_abstain",
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abstained, abstained_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="ensemble_abstain",
        )

        # Raw has MORE rows: includes M2-abstained specimens
        assert X_raw.shape[0] == X_proc.shape[0] + len(m2_abstained)
        assert set(m2_abstained).issubset(set(X_raw.index))

        # M2 columns are NaN for the added specimens
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]
        for spec in m2_abstained:
            for col in m2_cols:
                assert np.isnan(X_raw.loc[spec, col])

        # M1/M3 columns have real values for added specimens (not NaN)
        m1_display = MODEL_DISPLAY_NAMES[1]
        m3_display = MODEL_DISPLAY_NAMES[3]
        non_m2_cols = [c for c in X_raw.columns if f":{m2_display}:" not in c]
        for spec in m2_abstained:
            vals = X_raw.loc[spec, non_m2_cols].values
            assert not np.any(np.isnan(vals)), (
                f"Non-M2 columns should have real values for {spec}"
            )

        # Original scored rows unchanged
        pd.testing.assert_frame_equal(
            X_raw.loc[X_proc.index], X_proc, check_like=True,
        )

    def test_raw_no_abstentions(self):
        """No M2 abstentions: raw == processed."""
        rng = np.random.RandomState(201)
        specimens = [f"spec_{i:03d}" for i in range(15)]
        preds = {}
        for num in [1, 2, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X_proc, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None,
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abstained, abstained_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="ensemble_abstain",
        )
        pd.testing.assert_frame_equal(X_raw, X_proc)

    def test_raw_no_model2(self):
        """No Model 2 in predictions: raw == processed."""
        rng = np.random.RandomState(202)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = {}
        for num in [1, 3]:
            preds[num] = ModelPredictions(
                probabilities=_make_mock_proba(specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X_proc, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None,
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abstained, abstained_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="ensemble_abstain",
        )
        pd.testing.assert_frame_equal(X_raw, X_proc)


class TestApplyM2FillStrategy:
    """Tests for apply_m2_fill_strategy: load-time strategy application on raw matrices."""

    def _make_raw_matrix(
        self,
        n_specimens: int = 20,
        n_abstained: int = 5,
        classes: List[str] = None,
        seed: int = 300,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build a raw feature matrix with NaN in M2 columns for abstained specimens.

        n_abstained must be < n_specimens (partial abstention). For full M2
        abstention (all NaN), build the matrix directly — see
        test_apply_m2_full_abstention for an example.

        Returns (X_raw, true_diseases).
        """
        assert n_abstained < n_specimens, (
            f"_make_raw_matrix requires partial abstention (n_abstained < n_specimens), "
            f"got n_abstained={n_abstained}, n_specimens={n_specimens}. "
            f"Full M2 abstention causes M2 to be excluded from build_feature_matrix, "
            f"producing a raw matrix without M2 columns. Build the matrix directly instead."
        )
        if classes is None:
            classes = DISEASE_CLASSES
        preds, m2_abstained, _ = _build_fill_test_data(
            n_specimens=n_specimens, n_m2_abstained=n_abstained,
            classes=classes, seed=seed,
        )
        X_proc, abstained, abstained_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="fill_0.5",
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abstained, abstained_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )
        # Verify M2 columns are present (sanity check for partial abstention)
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]
        assert m2_cols, "Raw matrix should have M2 columns for partial abstention"
        # Build true_diseases series
        all_specimens = sorted(X_raw.index)
        true_diseases = pd.Series(
            [classes[i % len(classes)] for i in range(len(all_specimens))],
            index=all_specimens,
        )
        return X_raw, true_diseases

    def test_apply_ensemble_abstain(self):
        """ensemble_abstain: drops M2-NaN rows."""
        X_raw, true_diseases = self._make_raw_matrix()
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]
        n_abstained = X_raw[m2_cols].isna().any(axis=1).sum()
        assert n_abstained == 5

        X, abstained, abstained_diseases, fill_info = apply_m2_fill_strategy(
            X_raw, "ensemble_abstain", true_diseases,
        )
        assert X.shape[0] == X_raw.shape[0] - n_abstained
        assert len(abstained) == n_abstained
        assert len(abstained_diseases) == n_abstained
        assert not X.isna().any().any(), "No NaN should remain after ensemble_abstain"

    def test_apply_fill_0_5(self):
        """fill_0.5: NaN in M2 columns replaced with 0.5."""
        X_raw, true_diseases = self._make_raw_matrix()
        X, abstained, abstained_diseases, fill_info = apply_m2_fill_strategy(
            X_raw, "fill_0.5", true_diseases,
        )

        # All specimens kept
        assert X.shape[0] == X_raw.shape[0]
        assert len(abstained) == 0
        assert fill_info["strategy"] == "fill_0.5"
        assert fill_info["n_filled"] == 5

        # M2 columns have 0.5 for filled specimens
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X.columns if f":{m2_display}:" in c]
        for spec in fill_info["filled_specimen_labels"]:
            for col in m2_cols:
                assert X.loc[spec, col] == 0.5

        # No NaN remaining
        assert not X.isna().any().any()

    def test_apply_fill_models13_mean(self):
        """fill_models13_mean: M2 columns filled with (M1 + M3) / 2."""
        X_raw, true_diseases = self._make_raw_matrix()
        X, abstained, abstained_diseases, fill_info = apply_m2_fill_strategy(
            X_raw, "fill_models13_mean", true_diseases,
        )

        assert X.shape[0] == X_raw.shape[0]
        assert fill_info["strategy"] == "fill_models13_mean"
        assert fill_info["n_filled"] == 5
        assert not X.isna().any().any()

        # Verify M2 values = (M1 + M3) / 2 for filled specimens
        m1_display = MODEL_DISPLAY_NAMES[1]
        m2_display = MODEL_DISPLAY_NAMES[2]
        m3_display = MODEL_DISPLAY_NAMES[3]
        m2_cols = [c for c in X.columns if f":{m2_display}:" in c]
        for m2_col in m2_cols:
            class_name = m2_col.split(":", 2)[2]
            m1_col = m2_col.replace(f":{m2_display}:", f":{m1_display}:")
            m3_col = m2_col.replace(f":{m2_display}:", f":{m3_display}:")
            for spec in fill_info["filled_specimen_labels"]:
                expected = (X_raw.loc[spec, m1_col] + X_raw.loc[spec, m3_col]) / 2.0
                assert np.isclose(X.loc[spec, m2_col], expected), (
                    f"M2 fill for {spec}, {class_name}: "
                    f"expected {expected:.6f}, got {X.loc[spec, m2_col]:.6f}"
                )

    def test_apply_no_abstentions(self):
        """No M2 abstentions: all strategies return same result."""
        X_raw, true_diseases = self._make_raw_matrix(n_abstained=0)
        for strategy in MODEL2_ABSTENTION_STRATEGIES:
            X, abstained, _, fill_info = apply_m2_fill_strategy(
                X_raw, strategy, true_diseases,
            )
            assert X.shape == X_raw.shape
            assert len(abstained) == 0
            assert fill_info == {}

    def test_apply_m2_full_abstention(self):
        """M2 fully abstains (all NaN): M2 columns dropped, no specimens lost."""
        # Build a raw matrix directly (can't use _make_raw_matrix because
        # build_feature_matrix now excludes fully-abstained M2)
        rng = np.random.RandomState(400)
        n = 20
        specimens = [f"spec_{i:03d}" for i in range(n)]
        classes = DISEASE_CLASSES
        m1_display = MODEL_DISPLAY_NAMES[1]
        m2_display = MODEL_DISPLAY_NAMES[2]
        m3_display = MODEL_DISPLAY_NAMES[3]

        data = {}
        for cls in classes:
            data[f"TCR:{m1_display}:{cls}"] = rng.rand(n)
            data[f"TCR:{m2_display}:{cls}"] = [np.nan] * n  # ALL M2 NaN
            data[f"TCR:{m3_display}:{cls}"] = rng.rand(n)
        X_raw = pd.DataFrame(data, index=specimens)
        X_raw.index.name = "specimen_label"
        true_diseases = pd.Series(
            [classes[i % len(classes)] for i in range(n)], index=specimens,
        )

        m2_cols = [c for c in X_raw.columns if f":{m2_display}:" in c]
        assert len(m2_cols) > 0, "Test setup: M2 columns should exist"

        for strategy in MODEL2_ABSTENTION_STRATEGIES:
            X, abstained, abstained_diseases, fill_info = apply_m2_fill_strategy(
                X_raw, strategy, true_diseases,
            )
            # All specimens kept (M2 excluded, not specimens dropped)
            assert X.shape[0] == n, f"{strategy}: expected {n} specimens, got {X.shape[0]}"
            # M2 columns removed
            m2_cols_out = [c for c in X.columns if f":{m2_display}:" in c]
            assert len(m2_cols_out) == 0, f"{strategy}: M2 columns should be removed"
            # No abstentions (ensemble gets predictions for all specimens)
            assert len(abstained) == 0, f"{strategy}: expected 0 abstentions"
            # Excluded models tracked
            assert fill_info.get("excluded_models") == [2], (
                f"{strategy}: expected excluded_models=[2], got {fill_info}"
            )

    def test_invalid_strategy(self):
        """Invalid strategy raises AssertionError."""
        X_raw, true_diseases = self._make_raw_matrix()
        with pytest.raises(AssertionError, match="Invalid strategy"):
            apply_m2_fill_strategy(X_raw, "invalid_strategy", true_diseases)


class TestRawMatrixRoundTrip:
    """Strategy switching via raw matrices: build with one strategy, apply another."""

    def test_fill_to_different_fill(self):
        """Build with fill_0.5, extract raw, apply fill_models13_mean."""
        preds, m2_abstained, _ = _build_fill_test_data(seed=400)

        # Build with fill_0.5
        X_fill05, abs_labels, abs_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="fill_0.5",
        )
        X_raw = _build_raw_feature_matrix(
            X_fill05, preds, fill_info, abs_labels, abs_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )

        # Apply fill_models13_mean to the raw matrix
        true_diseases = pd.Series(
            [DISEASE_CLASSES[i % len(DISEASE_CLASSES)] for i in range(X_raw.shape[0])],
            index=X_raw.index,
        )
        X_m13, abstained, _, fill_info_m13 = apply_m2_fill_strategy(
            X_raw, "fill_models13_mean", true_diseases,
        )

        # All specimens still present, M2 values now are (M1+M3)/2 not 0.5
        assert X_m13.shape[0] == X_raw.shape[0]
        assert fill_info_m13["strategy"] == "fill_models13_mean"
        assert not X_m13.isna().any().any()

        # Verify M2 values differ from 0.5 fill
        m2_display = MODEL_DISPLAY_NAMES[2]
        m2_cols = [c for c in X_m13.columns if f":{m2_display}:" in c]
        for spec in m2_abstained:
            fill05_vals = X_fill05.loc[spec, m2_cols].values
            m13_vals = X_m13.loc[spec, m2_cols].values
            # They could coincidentally match, but with random data they shouldn't all be 0.5
            assert not np.allclose(m13_vals, 0.5), (
                f"fill_models13_mean values should differ from fill_0.5 for {spec}"
            )

    def test_abstain_to_fill(self):
        """Build with ensemble_abstain, extract raw, apply fill_0.5."""
        preds, m2_abstained, _ = _build_fill_test_data(seed=401)

        # Build with ensemble_abstain (drops M2-abstained)
        X_abstain, abs_labels, abs_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="ensemble_abstain",
        )
        X_raw = _build_raw_feature_matrix(
            X_abstain, preds, fill_info, abs_labels, abs_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="ensemble_abstain",
        )

        # Raw has all specimens (abstained rows added back)
        assert X_raw.shape[0] == X_abstain.shape[0] + len(m2_abstained)

        # Apply fill_0.5 to raw
        true_diseases = pd.Series(
            [DISEASE_CLASSES[i % len(DISEASE_CLASSES)] for i in range(X_raw.shape[0])],
            index=X_raw.index,
        )
        X_filled, abstained, _, fill_info_fill = apply_m2_fill_strategy(
            X_raw, "fill_0.5", true_diseases,
        )

        # All specimens present, none abstained
        assert X_filled.shape[0] == X_raw.shape[0]
        assert len(abstained) == 0
        assert fill_info_fill["n_filled"] == len(m2_abstained)
        assert not X_filled.isna().any().any()

    def test_fill_to_abstain(self):
        """Build with fill_0.5, extract raw, apply ensemble_abstain."""
        preds, m2_abstained, _ = _build_fill_test_data(seed=402)

        # Build with fill_0.5
        X_fill, abs_labels, abs_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="fill_0.5",
        )
        X_raw = _build_raw_feature_matrix(
            X_fill, preds, fill_info, abs_labels, abs_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )

        # Apply ensemble_abstain to raw (drops M2-NaN rows)
        true_diseases = pd.Series(
            [DISEASE_CLASSES[i % len(DISEASE_CLASSES)] for i in range(X_raw.shape[0])],
            index=X_raw.index,
        )
        X_abstained, abstained, abstained_diseases, _ = apply_m2_fill_strategy(
            X_raw, "ensemble_abstain", true_diseases,
        )

        # Fewer specimens: M2-abstained dropped
        assert X_abstained.shape[0] == X_raw.shape[0] - len(m2_abstained)
        assert len(abstained) == len(m2_abstained)
        assert set(abstained) == set(m2_abstained)
        assert not X_abstained.isna().any().any()

    def test_csv_round_trip(self):
        """Save raw matrix to CSV, reload, apply strategy — simulates --feature-matrices-dir."""
        preds, m2_abstained, _ = _build_fill_test_data(seed=403)

        # Build with fill_0.5 and get raw
        X_proc, abs_labels, abs_diseases, fill_info = build_feature_matrix(
            preds, "TCR", None, model2_abstention_strategy="fill_0.5",
        )
        X_raw = _build_raw_feature_matrix(
            X_proc, preds, fill_info, abs_labels, abs_diseases,
            gene_locus="TCR", reference_class=None,
            model2_abstention_strategy="fill_0.5",
        )

        # Add true_disease column (as saved by save_fold_artifacts)
        true_diseases = pd.Series(
            [DISEASE_CLASSES[i % len(DISEASE_CLASSES)] for i in range(X_raw.shape[0])],
            index=X_raw.index,
        )
        X_raw_with_label = X_raw.copy()
        X_raw_with_label["true_disease"] = true_diseases

        # Save to CSV and reload
        tmp_dir = _get_test_output_dir("raw_csv_round_trip")
        csv_path = tmp_dir / "fold_0_feature_matrix_raw_val.csv"
        X_raw_with_label.to_csv(csv_path, index_label="specimen_label")

        # Reload (same logic as run_ensemble_fold_from_features)
        loaded = pd.read_csv(csv_path, index_col="specimen_label")
        loaded_diseases = loaded.pop("true_disease")
        loaded_X = loaded

        # Apply fill_models13_mean after reload
        X_result, abstained, _, fill_info_result = apply_m2_fill_strategy(
            loaded_X, "fill_models13_mean", loaded_diseases,
        )

        assert X_result.shape[0] == X_raw.shape[0]
        assert fill_info_result["strategy"] == "fill_models13_mean"
        assert not X_result.isna().any().any()


class TestValidateFeatureMatricesDir:
    """Tests for validate_ensemble_args with --feature-matrices-dir."""

    def test_conflict_with_resume(self):
        """--feature-matrices-dir and --resume are mutually exclusive."""
        tmp = _get_test_output_dir("fm_conflict_resume")
        (tmp / "run_config.json").write_text("{}")
        args = _make_base_namespace(feature_matrices_dir=tmp, resume=True)
        parser = _FakeParser()
        with pytest.raises(SystemExit, match="mutually exclusive"):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_conflict_with_retrain_models(self):
        """--feature-matrices-dir and --retrain-models are mutually exclusive."""
        tmp = _get_test_output_dir("fm_conflict_retrain")
        (tmp / "run_config.json").write_text("{}")
        args = _make_base_namespace(feature_matrices_dir=tmp, retrain_models=[1])
        parser = _FakeParser()
        with pytest.raises(SystemExit, match="mutually exclusive"):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_conflict_with_retrain_base_models(self):
        """--feature-matrices-dir and --retrain-base-models are mutually exclusive."""
        tmp = _get_test_output_dir("fm_conflict_retrain_base")
        (tmp / "run_config.json").write_text("{}")
        args = _make_base_namespace(
            feature_matrices_dir=tmp, retrain_base_models=True,
        )
        parser = _FakeParser()
        with pytest.raises(SystemExit, match="mutually exclusive"):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_missing_run_config(self):
        """Error when no run_config.json in source dir or pair subdirs."""
        tmp = _get_test_output_dir("fm_no_config")
        tmp.mkdir(parents=True, exist_ok=True)
        args = _make_base_namespace(feature_matrices_dir=tmp)
        parser = _FakeParser()
        with pytest.raises(SystemExit, match="run_config.json"):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_valid_with_run_config(self):
        """No error when run_config.json exists in source dir."""
        tmp = _get_test_output_dir("fm_valid_config")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "run_config.json").write_text("{}")
        args = _make_base_namespace(feature_matrices_dir=tmp)
        parser = _FakeParser()
        # Should not raise
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_multi_binary_pair_subdirs(self):
        """Accepts multi-binary source dir with run_config.json in pair subdirs."""
        tmp = _get_test_output_dir("fm_multi_binary")
        tmp.mkdir(parents=True, exist_ok=True)
        # No run_config.json at top level, but pair subdirs have it
        pair1 = tmp / "Covid19_vs_Healthy-Background"
        pair2 = tmp / "HIV_vs_Healthy-Background"
        pair1.mkdir(parents=True, exist_ok=True)
        pair2.mkdir(parents=True, exist_ok=True)
        (pair1 / "run_config.json").write_text("{}")
        (pair2 / "run_config.json").write_text("{}")

        args = _make_base_namespace(feature_matrices_dir=tmp)
        parser = _FakeParser()
        # Should not raise — pair subdirs have configs
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_nonexistent_dir(self):
        """Error when --feature-matrices-dir doesn't exist."""
        args = _make_base_namespace(
            feature_matrices_dir=Path("/nonexistent/path"),
        )
        parser = _FakeParser()
        with pytest.raises(SystemExit, match="not.*directory"):
            validate_ensemble_args(args, set(), _make_base_cli_params(), parser)

    def test_fill_strategy_not_validated_against_cli_models(self):
        """With --feature-matrices-dir, fill strategy isn't validated against --models."""
        tmp = _get_test_output_dir("fm_strategy_skip")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "run_config.json").write_text("{}")
        # models=[1, 3] but strategy=fill_0.5 — would normally error because
        # fill requires Model 2. With --feature-matrices-dir, this check is
        # skipped (actual models come from source config).
        args = _make_base_namespace(
            feature_matrices_dir=tmp, models=[1, 3],
            model2_abstention_strategy="fill_0.5",
        )
        parser = _FakeParser()
        # Should not raise — strategy validation deferred to _run_from_feature_matrices
        validate_ensemble_args(args, set(), _make_base_cli_params(), parser)


class TestTrainEnsembleSourceDir:
    """Tests for train_ensemble() with source_dir parameter."""

    def test_source_dir_routes_to_from_features(self):
        """source_dir causes train_ensemble to call run_ensemble_fold_from_features."""
        tmp_output = _get_test_output_dir("te_source_dir_routing")
        tmp_source = _get_test_output_dir("te_source_dir_routing_src")

        calls = []

        def _mock_from_features(**kwargs):
            calls.append(kwargs)
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=np.array(DISEASE_CLASSES),
                n_specimens=20, model_nums=[1, 2, 3],
            )

        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold_from_features",
            side_effect=_mock_from_features,
        ):
            fold_results, summary = train_ensemble(
                loader=None, fold_ids=[0, 1],
                model_nums=[1, 2, 3], model_dirs={},
                gene_locus="TCR", output_dir=tmp_output,
                source_dir=tmp_source,
            )

        # run_ensemble_fold_from_features called (not run_ensemble_fold)
        assert len(calls) == 2
        # source_dir passed through
        assert calls[0]["source_dir"] == tmp_source
        assert calls[1]["source_dir"] == tmp_source

    def test_resume_without_source_dir_routes_to_from_features(self):
        """resume=True without source_dir also uses run_ensemble_fold_from_features."""
        tmp = _get_test_output_dir("te_resume_no_source")

        calls = []

        def _mock_from_features(**kwargs):
            calls.append(kwargs)
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=np.array(DISEASE_CLASSES),
                n_specimens=20, model_nums=[1, 2, 3],
            )

        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold_from_features",
            side_effect=_mock_from_features,
        ):
            fold_results, _ = train_ensemble(
                loader=None, fold_ids=[0],
                model_nums=[1, 2, 3], model_dirs={},
                gene_locus="TCR", output_dir=tmp,
                resume=True,
            )

        assert len(calls) == 1
        # source_dir is None (default, not external)
        assert calls[0]["source_dir"] is None

    def test_cleanup_doesnt_protect_with_external_source(self):
        """With external source_dir, cleanup removes feature matrices from output_dir."""
        tmp_output = _get_test_output_dir("te_cleanup_external")
        tmp_source = _get_test_output_dir("te_cleanup_external_src")

        # Pre-populate output_dir with old artifacts for a fold NOT being trained,
        # so save_fold_artifacts won't overwrite them.
        tmp_output.mkdir(parents=True, exist_ok=True)
        old_matrix = tmp_output / "fold_99_feature_matrix_val.csv"
        old_results = tmp_output / "fold_99_ensemble_results.json"
        old_matrix.write_text("old")
        old_results.write_text("old")

        def _mock_from_features(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=np.array(DISEASE_CLASSES),
                n_specimens=20, model_nums=[1, 2, 3],
            )

        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold_from_features",
            side_effect=_mock_from_features,
        ):
            train_ensemble(
                loader=None, fold_ids=[0],
                model_nums=[1, 2, 3], model_dirs={},
                gene_locus="TCR", output_dir=tmp_output,
                source_dir=tmp_source,
            )

        # Old artifacts for fold 99 were cleaned (not protected like resume mode)
        assert not old_matrix.exists(), (
            "Old feature matrix should be cleaned when source_dir is external"
        )
        assert not old_results.exists(), (
            "Old results should be cleaned when source_dir is external"
        )

    def test_cleanup_protects_when_source_equals_output(self):
        """When source_dir == output_dir, cleanup protects feature matrices."""
        tmp = _get_test_output_dir("te_cleanup_same_dir")
        tmp.mkdir(parents=True, exist_ok=True)

        # Pre-populate with artifacts that should be kept
        matrix = tmp / "fold_0_feature_matrix_val.csv"
        results = tmp / "fold_0_ensemble_results.json"
        summary = tmp / "summary_old.json"
        matrix.write_text("keep")
        results.write_text("keep")
        summary.write_text("remove")

        def _mock_from_features(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=np.array(DISEASE_CLASSES),
                n_specimens=20, model_nums=[1, 2, 3],
            )

        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold_from_features",
            side_effect=_mock_from_features,
        ):
            train_ensemble(
                loader=None, fold_ids=[0],
                model_nums=[1, 2, 3], model_dirs={},
                gene_locus="TCR", output_dir=tmp,
                source_dir=tmp,
            )

        # Feature matrices and results protected (source == output)
        assert matrix.exists(), (
            "Feature matrix should be protected when source_dir == output_dir"
        )
        assert results.exists(), (
            "Results should be protected when source_dir == output_dir"
        )
        # Non-input artifacts cleaned
        assert not summary.exists(), "Old summary should be cleaned"

    def test_resume_config_validation_skipped_with_source_dir(self):
        """Resume config validation is skipped when source_dir is provided."""
        tmp_output = _get_test_output_dir("te_skip_resume_check")
        tmp_source = _get_test_output_dir("te_skip_resume_check_src")

        # Put a mismatched run_config.json in output_dir. With resume=True
        # and source_dir=None this would fail. With source_dir set, it should
        # be skipped.
        tmp_output.mkdir(parents=True, exist_ok=True)
        config = {
            "classification_mode": "multiclass",
            "disease_filter": None,
            "reference_class": None,
            "diseases": None,
            "gene_locus": "TCR",
            "models_included": [1, 2, 3],
        }
        with open(tmp_output / "run_config.json", "w") as f:
            json.dump(config, f)

        new_config = dict(config)
        new_config["gene_locus"] = "BCR"  # Mismatched

        def _mock_from_features(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"], classes=np.array(DISEASE_CLASSES),
                n_specimens=20, model_nums=[1, 2, 3],
            )

        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold_from_features",
            side_effect=_mock_from_features,
        ):
            # Should NOT raise even though gene_locus differs
            train_ensemble(
                loader=None, fold_ids=[0],
                model_nums=[1, 2, 3], model_dirs={},
                gene_locus="BCR", output_dir=tmp_output,
                source_dir=tmp_source,
                run_config=new_config,
            )


class TestCacheDirResolution:
    """Verify that --cache-dir defaults to cache/<dataset-name>/ under project root."""

    def test_cache_dir_resolves_from_dataset_name(self):
        """When --cache-dir is omitted, main() resolves it from --dataset-name."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "malid_lite.training.train_ensemble",
             "--training-context", "cv",
             "--dataset-name", "test-nonexistent-xyz",
             "--classification-mode", "multiclass"],
            capture_output=True, text=True, timeout=30,
        )
        # main() should resolve cache dir to cache/test-nonexistent-xyz/
        # and fail because the cache doesn't exist — error message contains the path
        assert "cache/test-nonexistent-xyz" in result.stderr

    def test_explicit_cache_dir_overrides_dataset_name(self):
        """When --cache-dir is provided, it is used as-is."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "malid_lite.training.train_ensemble",
             "--training-context", "cv",
             "--cache-dir", "/tmp/my-explicit-cache",
             "--dataset-name", "should-be-ignored",
             "--classification-mode", "multiclass"],
            capture_output=True, text=True, timeout=30,
        )
        # Error message should reference the explicit path, not the dataset name
        assert "/tmp/my-explicit-cache" in result.stderr
        assert "should-be-ignored" not in result.stderr
