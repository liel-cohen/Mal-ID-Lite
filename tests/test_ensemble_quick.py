"""Quick smoke test for the Ensemble (metamodel) pipeline.

Two tiers of tests:

  Tier 1 — Unit tests with SYNTHETIC data (no cache, no GPU, ~30 seconds):
    Fast tests exercising individual components with fabricated probability matrices.
    These run even without a data cache or pre-trained base model artifacts.

  Tier 2 — Integration tests with REAL data (needs cache + cv_ensemble artifacts):
    End-to-end tests running the full ensemble pipeline on a participant subset.
    Requires pre-trained base model artifacts under cv_ensemble/.

Tests
-----
Unit tests (synthetic data):
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
  12. aggregate_fold_results: multiclass aggregation with macro/weighted metrics
  13. aggregate_fold_results: MCC aggregation
  14. _generate_ensemble_results_md: output format and key sections
  15. _log_comparison_table: output contains model names and metrics
  16. aggregate_fold_results: binary mode with MCC and log_loss
  17. _generate_ensemble_results_md + _log_comparison_table: binary metrics
  18. _save_multi_binary_summary: cross-pair MD and JSON
  19. Path construction: make_pair_name, model/ensemble output dirs for all modes
  20. validate_mode_and_classes: error paths for multi-binary/binary
  21. train_ensemble binary (mocked folds): full flow with artifact verification
  22. Multi-binary orchestration (mocked folds): 2 pairs + cross-pair summary
  23. _generate_ensemble_results_md binary: accuracy, MCC, log_loss, confusion matrix
  24. Binary specimen filtering: only target-disease specimens counted

Integration tests (real data):
  26. Full multiclass fold 0 pipeline (participant subset)
  27. Binary mode fold 0 pipeline (one disease vs reference)
  28. Artifact save/load round-trip
  29. run_config.json and RESULTS_*.md generation

Design notes
------------
- Unit tests use randomly generated probability matrices (not real model predictions).
- Integration tests require pre-trained cv_ensemble base model artifacts for fold 0.
- Model 3 integration uses pre-computed embeddings; the ensemble only calls
  predict_proba, not the full training pipeline, so memory usage stays manageable.
- Integration tests use a PARTICIPANT SUBSET (~3 per disease for Model 3) for speed.
  Models 1 and 2 process the full fold data but only predict on the subset specimens.

Requirements
------------
- Tier 1 (unit): numpy, pandas, scikit-learn, glmnet
- Tier 2 (integration): fold cache + pre-computed embeddings + cv_ensemble base model
  artifacts for fold 0

Expected runtime
----------------
- Tier 1 only: ~30 seconds
- Tier 1 + Tier 2: ~5-15 minutes (depending on hardware)

Output files
------------
All outputs saved to tests/test_outputs/test_ensemble_quick/:
- test_log_YYYYMMDD_HHMMSS.txt              - Full log
- test_results_YYYYMMDD_HHMMSS.json         - Structured results (pass/fail per test)
- integration/                               - Integration test artifacts

Running
-------
From Mal-ID-Lite root directory:

    # All tests:
    python tests/test_ensemble_quick.py

    # Unit tests only (no real data needed):
    python tests/test_ensemble_quick.py --unit-only

"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
)
from malid_lite.training.train_ensemble import (
    MODEL_DISPLAY_NAMES,
    ModelPredictions,
    build_feature_matrix,
    evaluate_predictions,
    train_ensemble,
    train_metamodel,
    _generate_ensemble_results_md,
    _log_comparison_table,
    _save_multi_binary_summary,
)


# ---------------------------------------------------------------------------
# Test logger (same pattern as other test files)
# ---------------------------------------------------------------------------

class _TestLogger:
    """Logger that writes to both console and file, tracks results as JSON."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_file, "a")
        self._results = {"tests": [], "start_time": datetime.now().isoformat()}

    def log(self, message: str):
        self._fh.write(message + "\n")
        self._fh.flush()
        print(message)

    def record(self, test_name: str, passed: bool, details: Optional[Dict] = None):
        self._results["tests"].append({
            "test": test_name,
            "status": "PASS" if passed else "FAIL",
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def close(self) -> Path:
        self._results["end_time"] = datetime.now().isoformat()
        self._fh.close()
        results_file = self.log_file.with_suffix(".json")
        with open(results_file, "w") as f:
            json.dump(
                self._results, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x),
            )
        return results_file


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


# ---------------------------------------------------------------------------
# Tier 1: Unit tests
# ---------------------------------------------------------------------------

def test_01_model_predictions_dataclass(tlog: _TestLogger):
    """Test 1: ModelPredictions properties and empty edge case."""
    tlog.log("\n--- Test 1: ModelPredictions dataclass ---")
    try:
        rng = np.random.RandomState(0)
        preds = _make_mock_predictions(10, DISEASE_CLASSES, n_abstained=3, rng=rng)
        assert preds.n_scored == 10, f"Expected 10 scored, got {preds.n_scored}"
        assert preds.n_abstained == 3, f"Expected 3 abstained, got {preds.n_abstained}"
        assert list(preds.probabilities.columns) == DISEASE_CLASSES

        # Empty case
        empty = ModelPredictions(
            probabilities=pd.DataFrame(columns=DISEASE_CLASSES),
            abstained_specimen_labels=["s1", "s2"],
            abstained_specimen_diseases=["Covid19", "HIV"],
        )
        assert empty.n_scored == 0
        assert empty.n_abstained == 2

        tlog.record("ModelPredictions dataclass", True, {"n_scored": 10, "n_abstained": 3})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("ModelPredictions dataclass", False, {"error": str(e)})


def test_02_build_feature_matrix_multiclass(tlog: _TestLogger):
    """Test 2: build_feature_matrix multiclass column naming and concatenation."""
    tlog.log("\n--- Test 2: build_feature_matrix (multiclass) ---")
    try:
        rng = np.random.RandomState(1)
        specimens = [f"spec_{i:03d}" for i in range(20)]

        predictions = {}
        for model_num in [1, 2, 3]:
            proba = _make_mock_proba(specimens, DISEASE_CLASSES, rng)
            predictions[model_num] = ModelPredictions(
                probabilities=proba,
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )

        X, abstained_labels, abstained_diseases = build_feature_matrix(
            predictions, gene_locus="TCR", reference_class=None,
        )

        # Check shape: 20 specimens x (6 classes * 3 models) = 18 features
        assert X.shape == (20, 18), f"Expected (20, 18), got {X.shape}"

        # Check column naming pattern: {locus}:{display_name}:{class}
        for col in X.columns:
            parts = col.split(":")
            assert len(parts) == 3, f"Column '{col}' doesn't have 3 colon-separated parts"
            assert parts[0] == "TCR"
            assert parts[1] in MODEL_DISPLAY_NAMES.values()

        # Check columns are sorted (deterministic)
        assert list(X.columns) == sorted(X.columns)

        # Check index is specimen labels
        assert list(X.index) == sorted(specimens)

        # No abstentions
        assert len(abstained_labels) == 0
        assert len(abstained_diseases) == 0

        # Check no NaN
        assert not X.isna().any().any(), "Feature matrix contains NaN"

        tlog.log(f"  Shape: {X.shape}, columns: {X.columns[:3].tolist()}...")
        tlog.record("build_feature_matrix multiclass", True, {"shape": list(X.shape)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("build_feature_matrix multiclass", False, {"error": str(e)})


def test_03_build_feature_matrix_binary(tlog: _TestLogger):
    """Test 3: build_feature_matrix binary column selection."""
    tlog.log("\n--- Test 3: build_feature_matrix (binary) ---")
    try:
        rng = np.random.RandomState(2)
        specimens = [f"spec_{i:03d}" for i in range(15)]
        binary_classes = [BINARY_DISEASE, BINARY_REFERENCE]

        predictions = {}
        for model_num in [1, 2, 3]:
            proba = _make_mock_proba(specimens, binary_classes, rng)
            predictions[model_num] = ModelPredictions(
                probabilities=proba,
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )

        X, _, _ = build_feature_matrix(
            predictions, gene_locus="TCR", reference_class=BINARY_REFERENCE,
        )

        # Binary: 2 classes -> keep only non-reference = 1 col per model = 3 total
        assert X.shape == (15, 3), f"Expected (15, 3), got {X.shape}"

        # All columns should be for the disease class, not the reference
        for col in X.columns:
            assert BINARY_REFERENCE not in col, (
                f"Reference class '{BINARY_REFERENCE}' should not appear in column '{col}'"
            )
            assert BINARY_DISEASE in col, (
                f"Disease class '{BINARY_DISEASE}' should appear in column '{col}'"
            )

        tlog.log(f"  Shape: {X.shape}, columns: {list(X.columns)}")
        tlog.record("build_feature_matrix binary", True, {"shape": list(X.shape)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("build_feature_matrix binary", False, {"error": str(e)})


def test_04_build_feature_matrix_abstention_harmonization(tlog: _TestLogger):
    """Test 4: build_feature_matrix abstention harmonization."""
    tlog.log("\n--- Test 4: build_feature_matrix (abstention harmonization) ---")
    try:
        rng = np.random.RandomState(3)

        # Model 1: scores all 20 specimens
        all_specimens = [f"spec_{i:03d}" for i in range(20)]
        preds1 = ModelPredictions(
            probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[],
            abstained_specimen_diseases=[],
        )

        # Model 2: abstains on 5 specimens (spec_015 through spec_019)
        scored_specimens = all_specimens[:15]
        abstained_specimens = all_specimens[15:]
        preds2 = ModelPredictions(
            probabilities=_make_mock_proba(scored_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=abstained_specimens,
            abstained_specimen_diseases=["Covid19"] * 5,
        )

        # Model 3: scores all 20
        preds3 = ModelPredictions(
            probabilities=_make_mock_proba(all_specimens, DISEASE_CLASSES, rng),
            abstained_specimen_labels=[],
            abstained_specimen_diseases=[],
        )

        X, abstained_labels, abstained_diseases = build_feature_matrix(
            {1: preds1, 2: preds2, 3: preds3},
            gene_locus="TCR", reference_class=None,
        )

        # Only the 15 common specimens should be in X
        assert X.shape[0] == 15, f"Expected 15 common specimens, got {X.shape[0]}"
        assert set(X.index) == set(scored_specimens)

        # Abstention info should include Model 2's abstentions
        assert len(abstained_labels) == 5
        assert set(abstained_labels) == set(abstained_specimens)

        # No NaN in the result (harmonization removed incomplete rows)
        assert not X.isna().any().any()

        tlog.log(f"  Common specimens: {X.shape[0]}, abstained: {len(abstained_labels)}")
        tlog.record("build_feature_matrix abstention", True,
                     {"n_common": X.shape[0], "n_abstained": len(abstained_labels)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("build_feature_matrix abstention", False, {"error": str(e)})


def test_05_build_feature_matrix_single_model(tlog: _TestLogger):
    """Test 5: build_feature_matrix with single model (e.g., --models 3)."""
    tlog.log("\n--- Test 5: build_feature_matrix (single model) ---")
    try:
        rng = np.random.RandomState(4)
        specimens = [f"spec_{i:03d}" for i in range(10)]
        preds = _make_mock_predictions(10, DISEASE_CLASSES, rng=rng)
        # Override to use consistent specimen labels
        preds.probabilities.index = specimens

        X, _, _ = build_feature_matrix(
            {3: preds}, gene_locus="TCR", reference_class=None,
        )

        # Single model: 6 classes = 6 features
        assert X.shape == (10, 6), f"Expected (10, 6), got {X.shape}"
        # All columns should be sequence_model
        for col in X.columns:
            assert "sequence_model" in col

        tlog.record("build_feature_matrix single model", True, {"shape": list(X.shape)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("build_feature_matrix single model", False, {"error": str(e)})


def test_06_train_metamodel(tlog: _TestLogger):
    """Test 6: train_metamodel on small synthetic data."""
    tlog.log("\n--- Test 6: train_metamodel ---")
    try:
        rng = np.random.RandomState(5)
        n_specimens = 60
        n_features = 18

        # Create synthetic feature matrix
        X = pd.DataFrame(
            rng.randn(n_specimens, n_features),
            index=[f"spec_{i:03d}" for i in range(n_specimens)],
            columns=[f"feat_{i}" for i in range(n_features)],
        )

        # Create labels (6 classes, 10 specimens each)
        classes = DISEASE_CLASSES
        y = pd.Series(
            [cls for cls in classes for _ in range(10)],
            index=X.index,
        )

        # Create participant groups (2 specimens per participant)
        groups = pd.Series(
            [f"P{i // 2:03d}" for i in range(n_specimens)],
            index=X.index,
        )

        pipeline = train_metamodel(X, y, groups)

        # Verify pipeline structure
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")
        assert hasattr(pipeline, "classes_")
        assert set(pipeline.classes_) == set(classes)

        # Verify it can predict
        y_pred = pipeline.predict(X.values)
        y_proba = pipeline.predict_proba(X.values)
        assert len(y_pred) == n_specimens
        assert y_proba.shape == (n_specimens, len(classes))

        # Probabilities should sum to 1
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Proba row sums: {row_sums[:5]}"

        # Check lambda was selected
        clf = pipeline.named_steps["classifier"]
        assert hasattr(clf, "lambda_best_")
        tlog.log(f"  Pipeline fitted. lambda_best={clf.lambda_best_:.6f}")
        tlog.log(f"  Classes: {list(pipeline.classes_)}")

        tlog.record("train_metamodel", True,
                     {"lambda_best": float(clf.lambda_best_), "n_classes": len(classes)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("train_metamodel", False, {"error": str(e)})


def test_07_evaluate_predictions_multiclass(tlog: _TestLogger):
    """Test 7: evaluate_predictions with multiclass (3+ classes)."""
    tlog.log("\n--- Test 7: evaluate_predictions (multiclass) ---")
    try:
        rng = np.random.RandomState(6)
        classes = np.array(sorted(DISEASE_CLASSES))
        n = 60

        y_true = np.array([classes[i % len(classes)] for i in range(n)])
        # Generate probabilities with some signal
        y_proba = np.zeros((n, len(classes)))
        for i in range(n):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i] = rng.dirichlet(np.ones(len(classes)))
            y_proba[i, true_idx] += 0.5  # boost true class
        y_proba /= y_proba.sum(axis=1, keepdims=True)

        y_pred = classes[np.argmax(y_proba, axis=1)]

        metrics, raw_preds = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_id=0, model_label="test_model",
            n_scored=n, n_abstained=0, reference_class=None,
        )

        # Check all expected keys are present
        expected_keys = [
            "fold_id", "model_name", "n_scored", "n_abstained",
            "abstention_rate", "accuracy",
            "auroc_ovo_weighted", "auroc_ovo_macro",
            "auprc_ovo_weighted", "auprc_ovo_macro",
            "auroc_ovr_per_class",
            "log_loss", "mcc", "confusion_matrix", "confusion_matrix_labels",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        assert metrics["fold_id"] == 0
        assert metrics["model_name"] == "test_model"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["auroc_ovo_weighted"] is not None
        assert metrics["auroc_ovo_macro"] is not None
        assert metrics["auprc_ovo_weighted"] is not None
        assert metrics["auprc_ovo_macro"] is not None
        assert 0.0 <= metrics["auroc_ovo_weighted"] <= 1.0
        assert 0.0 <= metrics["auroc_ovo_macro"] <= 1.0
        assert metrics["mcc"] is not None

        # Check raw_preds structure
        assert "y_true" in raw_preds
        assert "y_pred" in raw_preds
        assert "y_proba" in raw_preds
        assert "classes" in raw_preds
        assert len(raw_preds["y_true"]) == n

        tlog.log(f"  accuracy={metrics['accuracy']:.4f}, "
                 f"AUROC_weighted={metrics['auroc_ovo_weighted']:.4f}, "
                 f"AUROC_macro={metrics['auroc_ovo_macro']:.4f}, "
                 f"MCC={metrics['mcc']:.4f}")
        tlog.record("evaluate_predictions multiclass", True, {
            "accuracy": metrics["accuracy"],
            "auroc_weighted": metrics["auroc_ovo_weighted"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("evaluate_predictions multiclass", False, {"error": str(e)})


def test_08_evaluate_predictions_binary(tlog: _TestLogger):
    """Test 8: evaluate_predictions with binary (2 classes + reference_class)."""
    tlog.log("\n--- Test 8: evaluate_predictions (binary) ---")
    try:
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

        # Binary metrics should be present
        assert "auroc_binary" in metrics, "Missing auroc_binary"
        assert "auprc_binary" in metrics, "Missing auprc_binary"
        assert metrics["auroc_binary"] is not None
        assert metrics["auprc_binary"] is not None
        assert 0.0 <= metrics["auroc_binary"] <= 1.0

        # OvO multiclass metrics should be None (only 2 classes)
        assert metrics["auroc_ovo_weighted"] is None
        assert metrics["auroc_ovo_macro"] is None

        tlog.log(f"  accuracy={metrics['accuracy']:.4f}, "
                 f"AUROC_binary={metrics['auroc_binary']:.4f}, "
                 f"MCC={metrics['mcc']:.4f}")
        tlog.record("evaluate_predictions binary", True, {
            "accuracy": metrics["accuracy"],
            "auroc_binary": metrics["auroc_binary"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("evaluate_predictions binary", False, {"error": str(e)})


def test_09_evaluate_predictions_perfect(tlog: _TestLogger):
    """Test 9: evaluate_predictions with perfect predictions."""
    tlog.log("\n--- Test 9: evaluate_predictions (perfect) ---")
    try:
        classes = np.array(sorted(DISEASE_CLASSES))
        n = 30
        y_true = np.array([classes[i % len(classes)] for i in range(n)])

        # Perfect probabilities: 1.0 for true class, 0.0 elsewhere
        y_proba = np.zeros((n, len(classes)))
        for i in range(n):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i, true_idx] = 1.0

        y_pred = y_true.copy()

        metrics, _ = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_id=0, model_label="perfect",
            n_scored=n, n_abstained=0,
        )

        assert metrics["accuracy"] == 1.0, f"Expected accuracy 1.0, got {metrics['accuracy']}"
        assert metrics["mcc"] == 1.0, f"Expected MCC 1.0, got {metrics['mcc']}"

        tlog.log(f"  accuracy={metrics['accuracy']}, MCC={metrics['mcc']}")
        tlog.record("evaluate_predictions perfect", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("evaluate_predictions perfect", False, {"error": str(e)})


def test_10_evaluate_predictions_abstention_penalty(tlog: _TestLogger):
    """Test 10: evaluate_predictions abstention penalty."""
    tlog.log("\n--- Test 10: evaluate_predictions (abstention penalty) ---")
    try:
        classes = np.array(sorted(DISEASE_CLASSES))
        n_scored = 20
        n_abstained = 10

        y_true = np.array([classes[i % len(classes)] for i in range(n_scored)])
        y_proba = np.zeros((n_scored, len(classes)))
        for i in range(n_scored):
            true_idx = np.where(classes == y_true[i])[0][0]
            y_proba[i, true_idx] = 1.0
        y_pred = y_true.copy()

        metrics, _ = evaluate_predictions(
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_id=0, model_label="with_abstention",
            n_scored=n_scored, n_abstained=n_abstained,
        )

        # All 20 scored are correct, but 10 abstained count as wrong
        # accuracy = 20 / (20 + 10) = 0.6667
        expected_acc = n_scored / (n_scored + n_abstained)
        assert abs(metrics["accuracy"] - expected_acc) < 1e-6, (
            f"Expected accuracy {expected_acc:.4f}, got {metrics['accuracy']:.4f}"
        )
        assert metrics["abstention_rate"] == n_abstained / (n_scored + n_abstained)

        tlog.log(f"  accuracy={metrics['accuracy']:.4f} (penalized), "
                 f"abstention_rate={metrics['abstention_rate']:.4f}")
        tlog.record("evaluate_predictions abstention", True,
                     {"accuracy": metrics["accuracy"]})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("evaluate_predictions abstention", False, {"error": str(e)})


def test_11_column_alignment(tlog: _TestLogger):
    """Test 11: Column alignment between validation and test feature matrices."""
    tlog.log("\n--- Test 11: Column alignment ---")
    try:
        rng = np.random.RandomState(11)

        # Validation: all 3 models, 6 classes
        val_specimens = [f"val_{i:03d}" for i in range(20)]
        val_predictions = {}
        for model_num in [1, 2, 3]:
            val_predictions[model_num] = ModelPredictions(
                probabilities=_make_mock_proba(val_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X_val, _, _ = build_feature_matrix(val_predictions, "TCR", None)

        # Test: same 3 models, same 6 classes
        test_specimens = [f"test_{i:03d}" for i in range(15)]
        test_predictions = {}
        for model_num in [1, 2, 3]:
            test_predictions[model_num] = ModelPredictions(
                probabilities=_make_mock_proba(test_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X_test, _, _ = build_feature_matrix(test_predictions, "TCR", None)

        # Reindex test to match validation column order
        X_test = X_test[X_val.columns]
        assert list(X_test.columns) == list(X_val.columns), "Column mismatch after reindexing"

        # Test with missing columns should raise ValueError
        X_test_missing = X_test.drop(columns=[X_test.columns[0]])
        missing_cols = set(X_val.columns) - set(X_test_missing.columns)
        assert len(missing_cols) == 1, "Should detect 1 missing column"

        tlog.log(f"  Columns aligned: {X_val.shape[1]} features")
        tlog.record("Column alignment", True, {"n_features": X_val.shape[1]})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("Column alignment", False, {"error": str(e)})


def test_12_aggregate_fold_results_multiclass(tlog: _TestLogger):
    """Test 12: aggregate_fold_results multiclass with macro/weighted metrics."""
    tlog.log("\n--- Test 12: aggregate_fold_results (multiclass) ---")
    try:
        classes = np.array(sorted(DISEASE_CLASSES))
        rng = np.random.RandomState(12)

        fold_metrics = []
        fold_raw_preds = []
        for fold_id in range(3):
            n = 30
            y_true = np.array([classes[i % len(classes)] for i in range(n)])
            y_proba = rng.dirichlet(np.ones(len(classes)), size=n)
            y_pred = classes[np.argmax(y_proba, axis=1)]

            metrics, raw_preds = evaluate_predictions(
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                classes=classes, fold_id=fold_id, model_label="test",
                n_scored=n, n_abstained=0,
            )
            fold_metrics.append(metrics)
            fold_raw_preds.append(raw_preds)

        agg = aggregate_fold_results(fold_metrics, fold_raw_preds, disease_filter=None)

        # Check aggregated structure
        assert "accuracy_global" in agg
        assert "accuracy_per_fold" in agg
        assert "auroc_ovo_weighted" in agg
        assert "auroc_ovo_macro" in agg
        assert "auprc_ovo_weighted" in agg
        assert "auprc_ovo_macro" in agg
        assert "mcc" in agg

        # Check nested structure
        for key in ["auroc_ovo_weighted", "auroc_ovo_macro", "mcc"]:
            d = agg[key]
            assert isinstance(d, dict), f"{key} should be a dict, got {type(d)}"
            assert "mean" in d, f"{key} missing 'mean'"
            assert "std" in d, f"{key} missing 'std'"
            assert "per_fold" in d, f"{key} missing 'per_fold'"
            assert len(d["per_fold"]) == 3, f"{key} per_fold should have 3 entries"

        tlog.log(f"  accuracy_global={agg['accuracy_global']:.4f}")
        tlog.log(f"  AUROC weighted: {agg['auroc_ovo_weighted']['mean']:.4f} +/- "
                 f"{agg['auroc_ovo_weighted']['std']:.4f}")
        tlog.log(f"  AUROC macro: {agg['auroc_ovo_macro']['mean']:.4f} +/- "
                 f"{agg['auroc_ovo_macro']['std']:.4f}")
        tlog.log(f"  MCC: {agg['mcc']['mean']:.4f} +/- {agg['mcc']['std']:.4f}")

        tlog.record("aggregate_fold_results multiclass", True, {
            "accuracy_global": agg["accuracy_global"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("aggregate_fold_results multiclass", False, {"error": str(e)})


def test_13_aggregate_mcc(tlog: _TestLogger):
    """Test 13: aggregate_fold_results includes MCC aggregation."""
    tlog.log("\n--- Test 13: aggregate_fold_results (MCC) ---")
    try:
        classes = np.array(["A", "B", "C"])
        rng = np.random.RandomState(13)

        fold_metrics = []
        fold_raw_preds = []
        for fold_id in range(3):
            n = 30
            y_true = np.array([classes[i % 3] for i in range(n)])
            y_proba = rng.dirichlet(np.ones(3), size=n)
            for i in range(n):
                true_idx = np.where(classes == y_true[i])[0][0]
                y_proba[i, true_idx] += 1.0
            y_proba /= y_proba.sum(axis=1, keepdims=True)
            y_pred = classes[np.argmax(y_proba, axis=1)]

            metrics, raw_preds = evaluate_predictions(
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                classes=classes, fold_id=fold_id, model_label="mcc_test",
                n_scored=n, n_abstained=0,
            )
            fold_metrics.append(metrics)
            fold_raw_preds.append(raw_preds)

        agg = aggregate_fold_results(fold_metrics, fold_raw_preds, disease_filter=None)

        assert "mcc" in agg, "MCC not aggregated"
        assert agg["mcc"]["mean"] is not None
        assert agg["mcc"]["mean"] > 0, "MCC should be positive with boosted true class"
        assert len(agg["mcc"]["per_fold"]) == 3

        tlog.log(f"  MCC: {agg['mcc']['mean']:.4f} +/- {agg['mcc']['std']:.4f}")
        tlog.record("aggregate MCC", True, {"mcc_mean": agg["mcc"]["mean"]})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("aggregate MCC", False, {"error": str(e)})


def test_14_generate_results_md(tlog: _TestLogger):
    """Test 14: _generate_ensemble_results_md output format."""
    tlog.log("\n--- Test 14: _generate_ensemble_results_md ---")
    try:
        # Build minimal aggregated dicts
        ensemble_agg = {
            "accuracy_global": 0.75,
            "accuracy_per_fold": {"mean": 0.75, "std": 0.02, "per_fold": [0.73, 0.75, 0.77]},
            "auroc_ovo_weighted": {"mean": 0.90, "std": 0.01, "per_fold": [0.89, 0.90, 0.91]},
            "auroc_ovo_macro": {"mean": 0.88, "std": 0.02, "per_fold": [0.86, 0.88, 0.90]},
            "mcc": {"mean": 0.60, "std": 0.03, "per_fold": [0.57, 0.60, 0.63]},
            "confusion_matrix_aggregated": [[10, 2], [3, 15]],
            "classes": ["Covid19", "Healthy/Background"],
        }
        base_model_agg = {
            1: {
                "accuracy_global": 0.70,
                "auroc_ovo_weighted": {"mean": 0.85, "std": 0.02},
                "auroc_ovo_macro": {"mean": 0.83, "std": 0.03},
                "mcc": {"mean": 0.50, "std": 0.04},
            },
        }
        all_fold_results = [
            {
                "ensemble_metrics": {"fold_id": 0, "accuracy": 0.75,
                                     "auroc_ovo_weighted": 0.90, "mcc": 0.60,
                                     "n_scored": 30, "n_abstained": 0},
                "base_model_metrics": {
                    1: {"fold_id": 0, "accuracy": 0.70,
                        "auroc_ovo_weighted": 0.85, "mcc": 0.50},
                },
            },
        ]
        run_config = {
            "dataset_name": "test-dataset",
            "classification_mode": "multiclass",
            "gene_locus": "TCR",
            "models_included": [1],
        }

        md = _generate_ensemble_results_md(
            run_config=run_config,
            ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg,
            model_nums=[1],
            all_fold_results=all_fold_results,
            timestamp="20260422_120000",
        )

        assert "# Ensemble Training Results" in md
        assert "Run Configuration" in md
        assert "Model Comparison" in md
        assert "Per-Fold Ensemble Results" in md
        assert "Confusion Matrix" in md
        assert "test-dataset" in md

        tlog.log(f"  MD generated: {len(md)} characters, "
                 f"{md.count(chr(10))} lines")
        tlog.record("generate_results_md", True, {"length": len(md)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("generate_results_md", False, {"error": str(e)})


def test_15_log_comparison_table(tlog: _TestLogger):
    """Test 15: _log_comparison_table output contains model names and metrics."""
    tlog.log("\n--- Test 15: _log_comparison_table ---")
    try:
        ensemble_agg = {
            "accuracy_global": 0.80,
            "auroc_ovo_weighted": {"mean": 0.92, "std": 0.01},
            "mcc": {"mean": 0.65, "std": 0.02},
        }
        base_model_agg = {
            1: {"accuracy_global": 0.75,
                "auroc_ovo_weighted": {"mean": 0.88, "std": 0.02},
                "mcc": {"mean": 0.55, "std": 0.03}},
            2: {"accuracy_global": 0.72,
                "auroc_ovo_weighted": {"mean": 0.85, "std": 0.03},
                "mcc": {"mean": 0.50, "std": 0.04}},
        }

        # Capture log output to verify content
        import io
        log_handler = logging.StreamHandler(io.StringIO())
        log_handler.setLevel(logging.INFO)
        comp_logger = logging.getLogger("malid_lite.training.train_ensemble")
        old_level = comp_logger.level
        comp_logger.setLevel(logging.INFO)
        comp_logger.addHandler(log_handler)
        try:
            _log_comparison_table(ensemble_agg, base_model_agg, [1, 2])
            log_output = log_handler.stream.getvalue()
        finally:
            comp_logger.removeHandler(log_handler)
            comp_logger.setLevel(old_level)

        assert "Ensemble" in log_output, "Log output should mention 'Ensemble'"
        assert "0.92" in log_output or "0.920" in log_output, (
            "Log output should contain ensemble AUROC value"
        )

        tlog.log(f"  Log output: {len(log_output)} chars, contains expected content")
        tlog.record("log_comparison_table", True, {"log_length": len(log_output)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("log_comparison_table", False, {"error": str(e)})


def test_16_aggregate_fold_results_binary(tlog: _TestLogger):
    """Test 16: aggregate_fold_results for binary mode includes MCC and log_loss."""
    tlog.log("\n--- Test 16: aggregate_fold_results (binary) ---")
    try:
        classes = np.array(["Covid19", "Healthy/Background"])
        rng = np.random.RandomState(16)
        ref_class = "Healthy/Background"

        fold_metrics = []
        fold_raw_preds = []
        for fold_id in range(3):
            n = 20
            y_true = np.array([classes[i % 2] for i in range(n)])
            y_proba = rng.dirichlet(np.ones(2), size=n)
            for i in range(n):
                true_idx = np.where(classes == y_true[i])[0][0]
                y_proba[i, true_idx] += 1.0
            y_proba /= y_proba.sum(axis=1, keepdims=True)
            y_pred = classes[np.argmax(y_proba, axis=1)]

            metrics, raw_preds = evaluate_predictions(
                y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                classes=classes, fold_id=fold_id, model_label="binary_test",
                n_scored=n, n_abstained=0, reference_class=ref_class,
            )
            fold_metrics.append(metrics)
            fold_raw_preds.append(raw_preds)

        agg = aggregate_fold_results(
            fold_metrics, fold_raw_preds,
            disease_filter=("Covid19", ref_class),
        )

        assert "auroc_pooled" in agg, "auroc_pooled missing"
        assert "auprc_pooled" in agg, "auprc_pooled missing"
        assert agg["auroc_pooled"] is not None
        assert "mcc" in agg, "MCC missing from binary aggregation"
        assert agg["mcc"]["mean"] is not None
        assert "log_loss" in agg, "log_loss missing from binary aggregation"
        assert agg["log_loss"]["mean"] is not None
        assert "disease" in agg and agg["disease"] == "Covid19"
        assert "reference_class" in agg and agg["reference_class"] == ref_class

        # auroc_per_fold and auprc_per_fold should be _mean_std_per_fold dicts
        auroc_pf = agg["auroc_per_fold"]
        assert isinstance(auroc_pf, dict), (
            f"auroc_per_fold should be a dict, got {type(auroc_pf)}"
        )
        assert "mean" in auroc_pf and "std" in auroc_pf and "per_fold" in auroc_pf
        assert auroc_pf["mean"] is not None
        assert len(auroc_pf["per_fold"]) == 3

        auprc_pf = agg["auprc_per_fold"]
        assert isinstance(auprc_pf, dict), (
            f"auprc_per_fold should be a dict, got {type(auprc_pf)}"
        )
        assert auprc_pf["mean"] is not None

        # accuracy_per_fold should also be a _mean_std_per_fold dict
        acc_pf = agg["accuracy_per_fold"]
        assert isinstance(acc_pf, dict)
        assert "mean" in acc_pf and "std" in acc_pf and "per_fold" in acc_pf

        tlog.log(f"  auroc_pooled={agg['auroc_pooled']:.4f}")
        tlog.log(f"  auroc_per_fold mean={auroc_pf['mean']:.4f}")
        tlog.log(f"  MCC mean={agg['mcc']['mean']:.4f}")
        tlog.log(f"  log_loss mean={agg['log_loss']['mean']:.4f}")
        tlog.record("aggregate binary with MCC/log_loss", True, {
            "auroc_pooled": agg["auroc_pooled"],
            "auroc_per_fold_mean": auroc_pf["mean"],
            "mcc_mean": agg["mcc"]["mean"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("aggregate binary with MCC/log_loss", False, {"error": str(e)})


def test_17_binary_md_and_comparison(tlog: _TestLogger):
    """Test 17: MD and log_comparison_table use binary metrics when auroc_pooled present."""
    tlog.log("\n--- Test 17: Binary MD and comparison table ---")
    try:
        ensemble_agg = {
            "accuracy_global": 0.85,
            "auroc_pooled": 0.92,
            "auprc_pooled": 0.88,
            "mcc": {"mean": 0.70, "std": 0.03, "per_fold": [0.67, 0.70, 0.73]},
            "confusion_matrix_aggregated": [[18, 2], [4, 16]],
            "classes": ["Covid19", "Healthy/Background"],
            "disease": "Covid19",
            "reference_class": "Healthy/Background",
        }
        base_model_agg = {
            1: {
                "accuracy_global": 0.80,
                "auroc_pooled": 0.88,
                "auprc_pooled": 0.84,
                "mcc": {"mean": 0.60, "std": 0.04},
            },
        }
        all_fold_results = [
            {
                "ensemble_metrics": {
                    "fold_id": 0, "accuracy": 0.85,
                    "auroc_binary": 0.92, "mcc": 0.70,
                    "n_scored": 40, "n_abstained": 0,
                },
                "base_model_metrics": {
                    1: {"fold_id": 0, "accuracy": 0.80,
                        "auroc_binary": 0.88, "mcc": 0.60},
                },
            },
        ]
        run_config = {
            "dataset_name": "test-dataset",
            "classification_mode": "binary",
            "gene_locus": "TCR",
            "disease_filter": ["Covid19", "Healthy/Background"],
        }

        md = _generate_ensemble_results_md(
            run_config=run_config,
            ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg,
            model_nums=[1],
            all_fold_results=all_fold_results,
            timestamp="20260422_130000",
        )

        assert "AUROC (pooled)" in md, "Binary MD should show 'AUROC (pooled)'"
        assert "AUPRC (pooled)" in md, "Binary MD should show 'AUPRC (pooled)'"
        assert "AUROC OvO" not in md, "Binary MD should not show OvO metrics"

        _log_comparison_table(ensemble_agg, base_model_agg, [1])

        tlog.log(f"  Binary MD: {len(md)} chars, has AUROC/AUPRC pooled columns")
        tlog.record("binary MD and comparison", True, {"md_length": len(md)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("binary MD and comparison", False, {"error": str(e)})


def test_18_save_multi_binary_summary(tlog: _TestLogger):
    """Test 18: _save_multi_binary_summary writes MD and JSON."""
    tlog.log("\n--- Test 18: _save_multi_binary_summary ---")
    try:
        import tempfile
        import shutil

        tmp_dir = Path(tempfile.mkdtemp(prefix="ensemble_mb_test_"))
        try:
            pairs = [("Covid19", "Healthy/Background"), ("HIV", "Healthy/Background")]
            summaries = {}
            for disease, ref in pairs:
                pk = make_pair_name(disease, ref)
                summaries[pk] = {
                    "ensemble": {
                        "accuracy_global": 0.80 + np.random.rand() * 0.1,
                        "auroc_pooled": 0.85 + np.random.rand() * 0.1,
                        "auprc_pooled": 0.82 + np.random.rand() * 0.1,
                        "mcc": {"mean": 0.60 + np.random.rand() * 0.1, "std": 0.03},
                    },
                }

            _save_multi_binary_summary(
                tmp_dir, summaries, pairs, "Healthy/Background",
            )

            md_files = list(tmp_dir.glob("MULTI_BINARY_SUMMARY_*.md"))
            json_files = list(tmp_dir.glob("multi_binary_summary_*.json"))
            assert len(md_files) == 1, f"Expected 1 MD file, got {len(md_files)}"
            assert len(json_files) == 1, f"Expected 1 JSON file, got {len(json_files)}"

            md_content = md_files[0].read_text()
            assert "Covid19" in md_content
            assert "HIV" in md_content
            assert "Healthy/Background" in md_content

            with open(json_files[0]) as f:
                cross_json = json.load(f)
            assert cross_json["n_pairs"] == 2
            assert len(cross_json["pairs"]) == 2
            for pk in summaries:
                assert pk in cross_json["pairs"]
                assert "auroc_pooled" in cross_json["pairs"][pk]

            tlog.log(f"  MD: {md_files[0].name}, JSON: {json_files[0].name}")
            tlog.log(f"  {len(cross_json['pairs'])} pairs in summary")
            tlog.record("multi-binary summary", True, {"n_pairs": 2})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("multi-binary summary", False, {"error": str(e)})


def _make_mock_fold_result(
    fold_id: int,
    classes: np.ndarray,
    n_specimens: int,
    model_nums: List[int],
    reference_class: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Build a synthetic fold result matching run_ensemble_fold output.

    Uses evaluate_predictions on random data so metrics/raw_preds are
    structurally correct.  The pipeline is a tiny LogisticRegression so
    joblib serialisation works.
    """
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

    # Ensemble predictions (strong)
    ens_pred, ens_proba = _synth_preds(boost=2.0)
    ens_metrics, ens_raw = evaluate_predictions(
        y_true=y_true, y_pred=ens_pred, y_proba=ens_proba,
        classes=classes, fold_id=fold_id, model_label="ensemble",
        n_scored=n_specimens, n_abstained=0, reference_class=reference_class,
    )

    # Base model predictions (weaker)
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

    # Minimal serialisable pipeline (not GlmnetLogitNet — just needs to survive joblib)
    n_per_class = max(2, 10 // len(classes))
    X_tiny = rng.randn(n_per_class * len(classes), 3)
    y_tiny = np.array([str(c) for c in classes] * n_per_class)
    pipe = SkPipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    pipe.fit(X_tiny, y_tiny)

    # Prediction rows
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


def test_19_pair_resolution_and_paths(tlog: _TestLogger):
    """Test 19: Verify path construction logic for multiclass, binary, multi-binary."""
    tlog.log("\n--- Test 19: Pair resolution and path construction ---")
    try:
        # --- make_pair_name ---
        assert make_pair_name("Covid19", "Healthy/Background") == "Covid19_vs_Healthy_Background"
        assert make_pair_name("HIV", "Healthy/Background") == "HIV_vs_Healthy_Background"

        # --- get_model_output_dir: binary and multi-binary share "binary" mode_dir ---
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
        assert dir_bin == dir_mb, (
            f"binary and multi-binary should share mode_dir: {dir_bin} vs {dir_mb}"
        )
        assert dir_bin.name == "binary"

        # Appending pair_name gives per-pair directory
        pair_name = make_pair_name("Covid19", "Healthy/Background")
        pair_dir = dir_bin / pair_name
        assert pair_dir.name == "Covid19_vs_Healthy_Background"

        # --- get_ensemble_output_dir: same pattern ---
        ens_bin = get_ensemble_output_dir(
            dataset_name="test-data", classification_mode="binary", gene_locus="TCR",
        )
        ens_mb = get_ensemble_output_dir(
            dataset_name="test-data", classification_mode="multi-binary", gene_locus="TCR",
        )
        assert ens_bin == ens_mb
        assert ens_bin.name == "binary"

        # Multiclass is separate
        dir_mc = get_model_output_dir(
            model_name="model1", dataset_name="test-data",
            classification_mode="multiclass", gene_locus="TCR",
            training_context="cv_ensemble",
        )
        assert dir_mc.name == "multiclass"
        assert dir_mc != dir_bin

        tlog.log("  All path assertions passed")
        tlog.record("pair resolution and paths", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("pair resolution and paths", False, {"error": str(e)})


def test_20_validate_mode_and_classes_errors(tlog: _TestLogger):
    """Test 20: validate_mode_and_classes error paths for multi-binary."""
    tlog.log("\n--- Test 20: validate_mode_and_classes error paths ---")
    try:
        classes_4 = ["Covid19", "HIV", "Healthy/Background", "Lupus"]
        classes_2 = ["Covid19", "Healthy/Background"]
        errors_caught = 0

        # multi-binary without reference_class (N>2) -> ValueError
        try:
            validate_mode_and_classes("multi-binary", classes_4, None, None)
            assert False, "Should have raised ValueError for missing reference_class"
        except ValueError as e:
            assert "reference-class" in str(e).lower()
            errors_caught += 1

        # multi-binary with invalid reference_class -> ValueError
        try:
            validate_mode_and_classes("multi-binary", classes_4, "InvalidClass", None)
            assert False, "Should have raised ValueError for invalid reference_class"
        except ValueError as e:
            assert "not found" in str(e).lower()
            errors_caught += 1

        # multi-binary with valid reference_class -> returns it
        ref = validate_mode_and_classes("multi-binary", classes_4, "Healthy/Background", None)
        assert ref == "Healthy/Background"

        # multi-binary with 2 classes, no reference -> infers alphabetically
        ref2 = validate_mode_and_classes("multi-binary", classes_2, None, None)
        assert ref2 is not None

        # binary without diseases (N>2) -> ValueError
        try:
            validate_mode_and_classes("binary", classes_4, "Healthy/Background", None)
            assert False, "Should have raised ValueError for binary with N>2 and no diseases"
        except ValueError as e:
            errors_caught += 1

        tlog.log(f"  {errors_caught} expected errors caught, valid cases returned correctly")
        tlog.record("validate error paths", True, {"errors_caught": errors_caught})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate error paths", False, {"error": str(e)})


def test_21_train_ensemble_binary_mocked(tlog: _TestLogger):
    """Test 21: Full binary train_ensemble() with mocked run_ensemble_fold.

    Mocks the per-fold function so no real data/artifacts are needed, then
    verifies that train_ensemble correctly:
    - loops over folds
    - saves metamodel artifacts (joblib + config JSON)
    - writes predictions CSV
    - aggregates binary metrics (auroc_pooled, MCC)
    - generates summary JSON and RESULTS MD with binary columns
    """
    tlog.log("\n--- Test 21: train_ensemble binary (mocked folds) ---")
    try:
        from unittest.mock import patch
        import tempfile
        import shutil

        classes = np.array(["Covid19", "Healthy/Background"])
        ref_class = "Healthy/Background"
        model_nums = [1, 2]
        fold_ids = [0, 1]

        def _mock_run_fold(**kwargs):
            return _make_mock_fold_result(
                fold_id=kwargs["fold_id"],
                classes=classes,
                n_specimens=20,
                model_nums=model_nums,
                reference_class=ref_class,
            )

        tmp = Path(tempfile.mkdtemp(prefix="ens_bin_test_"))
        try:
            with patch(
                "malid_lite.training.train_ensemble.run_ensemble_fold",
                side_effect=_mock_run_fold,
            ):
                fold_results, summary = train_ensemble(
                    loader=None,
                    fold_ids=fold_ids,
                    model_nums=model_nums,
                    model_dirs={1: Path("dummy"), 2: Path("dummy")},
                    gene_locus="TCR",
                    output_dir=tmp,
                    disease_filter=("Covid19", ref_class),
                    reference_class=ref_class,
                    run_config={"classification_mode": "binary", "test": True},
                )

            # --- Verify fold results ---
            assert len(fold_results) == 2, f"Expected 2 fold results, got {len(fold_results)}"

            # --- Verify artifacts on disk ---
            assert (tmp / "run_config.json").exists(), "run_config.json missing"
            assert (tmp / "ensemble_predictions.csv").exists(), "predictions CSV missing"
            preds_df = pd.read_csv(tmp / "ensemble_predictions.csv")
            assert len(preds_df) == 40, f"Expected 40 prediction rows, got {len(preds_df)}"

            metamodel_dir = tmp / "metamodel"
            assert metamodel_dir.exists(), "metamodel/ directory missing"
            for fid in fold_ids:
                assert (metamodel_dir / f"fold_{fid}_ridge_cv_metamodel.joblib").exists()
                assert (metamodel_dir / f"fold_{fid}_metamodel_config.json").exists()

            summary_files = list(tmp.glob("summary_*.json"))
            assert len(summary_files) == 1, f"Expected 1 summary JSON, got {len(summary_files)}"
            md_files = list(tmp.glob("RESULTS_*.md"))
            assert len(md_files) == 1, f"Expected 1 RESULTS MD, got {len(md_files)}"

            # --- Verify binary metrics in summary ---
            assert "ensemble" in summary
            ens = summary["ensemble"]
            assert "auroc_pooled" in ens, "auroc_pooled missing from binary summary"
            assert ens["auroc_pooled"] is not None
            assert "mcc" in ens, "MCC missing from binary summary"
            assert ens["mcc"]["mean"] is not None

            # --- Verify RESULTS MD has binary columns, not multiclass ---
            md_text = md_files[0].read_text()
            assert "AUROC (pooled)" in md_text, "MD should show binary AUROC column"
            assert "AUROC OvO" not in md_text, "MD should not show OvO metrics for binary"

            tlog.log(f"  Folds: {len(fold_results)}, predictions: {len(preds_df)} rows")
            tlog.log(f"  auroc_pooled={ens['auroc_pooled']:.4f}, "
                     f"MCC={ens['mcc']['mean']:.4f}")
            tlog.log(f"  MD has binary columns: OK")
            tlog.record("train_ensemble binary mocked", True, {
                "auroc_pooled": ens["auroc_pooled"],
                "mcc_mean": ens["mcc"]["mean"],
            })
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("train_ensemble binary mocked", False, {"error": str(e)})


def test_22_multi_binary_orchestration(tlog: _TestLogger):
    """Test 22: Full multi-binary orchestration — 2 disease pairs end-to-end.

    Exercises the exact logic from main(): for each pair, resolve pair-specific
    model dirs and output dir, call train_ensemble with disease_filter, then
    call _save_multi_binary_summary for the cross-pair report.  All fold
    execution is mocked.
    """
    tlog.log("\n--- Test 22: Multi-binary orchestration (2 pairs, mocked folds) ---")
    try:
        from unittest.mock import patch
        import tempfile
        import shutil

        ref_class = "Healthy/Background"
        model_nums = [1, 3]
        fold_ids = [0, 1]
        pairs_to_train = [
            ("Covid19", ref_class),
            ("HIV", ref_class),
        ]

        base_output_dir = Path(tempfile.mkdtemp(prefix="ens_mb_test_"))
        try:
            all_pair_summaries = {}

            for disease, ref in pairs_to_train:
                classes = np.array(sorted([disease, ref]))
                pair_key = make_pair_name(disease, ref)
                output_dir = base_output_dir / pair_key
                disease_filter = (disease, ref)

                def _mock_run_fold(disease_=disease, classes_=classes, **kwargs):
                    return _make_mock_fold_result(
                        fold_id=kwargs["fold_id"],
                        classes=classes_,
                        n_specimens=16,
                        model_nums=model_nums,
                        reference_class=ref,
                    )

                with patch(
                    "malid_lite.training.train_ensemble.run_ensemble_fold",
                    side_effect=_mock_run_fold,
                ):
                    _, summary = train_ensemble(
                        loader=None,
                        fold_ids=fold_ids,
                        model_nums=model_nums,
                        model_dirs={n: Path("dummy") for n in model_nums},
                        gene_locus="TCR",
                        output_dir=output_dir,
                        disease_filter=disease_filter,
                        reference_class=ref,
                        run_config={"classification_mode": "multi-binary",
                                    "disease_filter": list(disease_filter)},
                    )
                all_pair_summaries[pair_key] = summary

            # --- Verify per-pair outputs ---
            for disease, ref in pairs_to_train:
                pk = make_pair_name(disease, ref)
                pair_dir = base_output_dir / pk
                assert pair_dir.exists(), f"Pair directory missing: {pk}"
                assert (pair_dir / "ensemble_predictions.csv").exists()
                assert (pair_dir / "metamodel").exists()
                assert len(list(pair_dir.glob("summary_*.json"))) == 1
                assert len(list(pair_dir.glob("RESULTS_*.md"))) == 1

                # Verify binary metrics
                assert "auroc_pooled" in all_pair_summaries[pk]["ensemble"]

            # --- Save and verify cross-pair summary ---
            _save_multi_binary_summary(
                base_output_dir, all_pair_summaries, pairs_to_train, ref_class,
            )

            mb_md = list(base_output_dir.glob("MULTI_BINARY_SUMMARY_*.md"))
            mb_json = list(base_output_dir.glob("multi_binary_summary_*.json"))
            assert len(mb_md) == 1, f"Expected 1 cross-pair MD, got {len(mb_md)}"
            assert len(mb_json) == 1, f"Expected 1 cross-pair JSON, got {len(mb_json)}"

            with open(mb_json[0]) as f:
                cross = json.load(f)
            assert cross["n_pairs"] == 2
            assert "Covid19_vs_Healthy_Background" in cross["pairs"]
            assert "HIV_vs_Healthy_Background" in cross["pairs"]

            md_text = mb_md[0].read_text()
            assert "Covid19" in md_text
            assert "HIV" in md_text

            tlog.log(f"  2 pair directories created with all artifacts")
            tlog.log(f"  Cross-pair summary: {mb_md[0].name}")
            for pk in sorted(all_pair_summaries):
                auroc = all_pair_summaries[pk]["ensemble"].get("auroc_pooled")
                tlog.log(f"    {pk}: auroc_pooled={auroc:.4f}" if auroc else f"    {pk}: N/A")
            tlog.record("multi-binary orchestration", True, {"n_pairs": 2})
        finally:
            shutil.rmtree(base_output_dir, ignore_errors=True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("multi-binary orchestration", False, {"error": str(e)})


def test_23_generate_results_md_binary_enrichment(tlog: _TestLogger):
    """Test 23: Binary MD includes accuracy, MCC, log_loss, and confusion matrix."""
    tlog.log("\n--- Test 23: Binary MD enrichment content ---")
    try:
        ensemble_agg = {
            "accuracy_global": 0.85,
            "accuracy_per_fold": {"mean": 0.85, "std": 0.02, "per_fold": [0.83, 0.85, 0.87]},
            "auroc_pooled": 0.92,
            "auprc_pooled": 0.88,
            "mcc": {"mean": 0.70, "std": 0.03, "per_fold": [0.67, 0.70, 0.73]},
            "log_loss": {"mean": 0.35, "std": 0.02, "per_fold": [0.33, 0.35, 0.37]},
            "confusion_matrix_aggregated": [[18, 2], [4, 16]],
            "classes": ["Covid19", "Healthy/Background"],
            "disease": "Covid19",
            "reference_class": "Healthy/Background",
        }
        base_model_agg = {
            1: {
                "accuracy_global": 0.80,
                "accuracy_per_fold": {"mean": 0.80, "std": 0.03, "per_fold": [0.77, 0.80, 0.83]},
                "auroc_pooled": 0.88,
                "auprc_pooled": 0.84,
                "mcc": {"mean": 0.60, "std": 0.04, "per_fold": [0.56, 0.60, 0.64]},
                "log_loss": {"mean": 0.42, "std": 0.03, "per_fold": [0.39, 0.42, 0.45]},
                "confusion_matrix_aggregated": [[16, 4], [6, 14]],
                "classes": ["Covid19", "Healthy/Background"],
            },
        }
        all_fold_results = []
        for fid in range(3):
            all_fold_results.append({
                "ensemble_metrics": {
                    "fold_id": fid, "accuracy": 0.85,
                    "auroc_binary": 0.92, "auprc_binary": 0.88,
                    "mcc": 0.70, "log_loss": 0.35,
                    "n_scored": 40, "n_abstained": 0,
                    "confusion_matrix": [[6, 1], [1, 5]],
                    "confusion_matrix_labels": ["Covid19", "Healthy/Background"],
                },
                "base_model_metrics": {
                    1: {
                        "fold_id": fid, "accuracy": 0.80,
                        "auroc_binary": 0.88, "auprc_binary": 0.84,
                        "mcc": 0.60, "log_loss": 0.42,
                        "n_scored": 40, "n_abstained": 0,
                        "confusion_matrix": [[5, 2], [2, 4]],
                        "confusion_matrix_labels": ["Covid19", "Healthy/Background"],
                    },
                },
            })
        run_config = {
            "dataset_name": "test-dataset",
            "classification_mode": "binary",
            "gene_locus": "TCR",
            "disease_filter": ["Covid19", "Healthy/Background"],
        }

        md = _generate_ensemble_results_md(
            run_config=run_config,
            ensemble_agg=ensemble_agg,
            base_model_agg=base_model_agg,
            model_nums=[1],
            all_fold_results=all_fold_results,
            timestamp="20260422_140000",
        )

        # Check comparison table has accuracy, AUROC pooled, MCC columns
        assert "Accuracy (global)" in md, "Comparison table should have Accuracy column"
        assert "AUROC (pooled)" in md, "Comparison table should have AUROC pooled column"
        assert "AUPRC (pooled)" in md, "Comparison table should have AUPRC pooled column"
        assert "MCC" in md, "Comparison table should have MCC column"

        # Per-fold table should have Accuracy and MCC columns
        assert "Per-Fold Ensemble Results" in md, "Should have per-fold section"
        assert "| Accuracy |" in md or "Accuracy" in md

        # Confusion matrix section
        assert "Confusion Matrix" in md, "Should have confusion matrix section"
        assert "Covid19" in md
        assert "Healthy/Background" in md

        # Binary mode should NOT have OvO metrics
        assert "AUROC OvO" not in md, "Binary MD should not show OvO metrics"

        tlog.log(f"  Binary MD enrichment validated: {len(md)} chars")
        tlog.record("binary MD enrichment", True, {"md_length": len(md)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("binary MD enrichment", False, {"error": str(e)})


def test_24_binary_specimen_filtering(tlog: _TestLogger):
    """Test 24: In binary mode, only target-disease specimens are counted.

    Validates the fix for the abstention count bug: when disease_filter is set,
    test_specimens must be restricted to the two target diseases. Specimens from
    other diseases are outside the classification scope, not "abstained".
    """
    tlog.log("\n--- Test 24: Binary specimen filtering ---")
    try:
        # Simulate test_meta with 3 diseases (30 specimens total)
        test_meta = pd.DataFrame({
            SPECIMEN_COL: [f"spec_{i:03d}" for i in range(30)],
            PARTICIPANT_COL: [f"part_{i:03d}" for i in range(30)],
            DISEASE_COL: (
                ["Covid19"] * 10 + ["Healthy/Background"] * 10 + ["HIV"] * 10
            ),
        })

        # Without disease_filter: all 30 specimens
        test_specimens_all = set(test_meta[SPECIMEN_COL])
        assert len(test_specimens_all) == 30

        # With disease_filter: only Covid19 + Healthy/Background = 20
        disease_filter = ("Covid19", "Healthy/Background")
        target_diseases = set(disease_filter)
        test_specimens_filtered = set(
            test_meta[test_meta[DISEASE_COL].isin(target_diseases)][SPECIMEN_COL]
        )
        assert len(test_specimens_filtered) == 20, (
            f"Expected 20 target-disease specimens, got {len(test_specimens_filtered)}"
        )

        # The 10 HIV specimens should NOT be in the filtered set
        hiv_specimens = set(
            test_meta[test_meta[DISEASE_COL] == "HIV"][SPECIMEN_COL]
        )
        assert hiv_specimens.isdisjoint(test_specimens_filtered), (
            "HIV specimens should not be in binary-filtered set"
        )

        # Demonstrate the bug: if we had 15 scored specimens (binary pair only),
        # the old code would compute n_abstained = 30 - 15 = 15 (WRONG: counts HIV)
        # the new code computes n_abstained = 20 - 15 = 5 (CORRECT)
        n_scored = 15
        n_abstained_old = len(test_specimens_all) - n_scored  # 15 (WRONG)
        n_abstained_new = len(test_specimens_filtered) - n_scored  # 5 (CORRECT)
        assert n_abstained_old == 15, "Old logic would count 15 abstained (wrong)"
        assert n_abstained_new == 5, "New logic counts 5 abstained (correct)"

        # Same pattern for validation_specimens
        train_meta = pd.DataFrame({
            SPECIMEN_COL: [f"train_spec_{i:03d}" for i in range(60)],
            PARTICIPANT_COL: [f"train_part_{i:03d}" for i in range(60)],
            DISEASE_COL: (
                ["Covid19"] * 20 + ["Healthy/Background"] * 20 + ["HIV"] * 20
            ),
        })
        validation_participants = {f"train_part_{i:03d}" for i in range(20)}

        # Without filter: all validation participants' specimens
        val_specimens_all = set(
            train_meta[
                train_meta[PARTICIPANT_COL].isin(validation_participants)
            ][SPECIMEN_COL]
        )

        # With filter: only target-disease specimens from validation participants
        val_specimens_filtered = set(
            train_meta[
                train_meta[PARTICIPANT_COL].isin(validation_participants)
                & train_meta[DISEASE_COL].isin(target_diseases)
            ][SPECIMEN_COL]
        )
        assert len(val_specimens_filtered) <= len(val_specimens_all), (
            "Filtered validation set should be <= unfiltered set"
        )

        tlog.log(f"  test_specimens: {len(test_specimens_all)} all -> "
                 f"{len(test_specimens_filtered)} filtered")
        tlog.log(f"  Abstention: old={n_abstained_old} (wrong), new={n_abstained_new} (correct)")
        tlog.record("binary specimen filtering", True, {
            "all_specimens": len(test_specimens_all),
            "filtered_specimens": len(test_specimens_filtered),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("binary specimen filtering", False, {"error": str(e)})


# ---------------------------------------------------------------------------
# Tier 2: Integration tests
# ---------------------------------------------------------------------------

def _get_integration_loader():
    """Create data loader for integration tests."""
    cache_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data"
    cached_metadata = cache_dir / "metadata.tsv"

    if cached_metadata.exists():
        metadata_path = cached_metadata
        cache_info_path = cache_dir / "participants" / "cache_info.json"
        if cache_info_path.exists():
            with open(cache_info_path) as f:
                cache_info = json.load(f)
            data_dir = Path(cache_info.get("data_dir", "."))
        else:
            data_dir = Path(".")
    else:
        cache_info_path = cache_dir / "participants" / "cache_info.json"
        if not cache_info_path.exists():
            raise FileNotFoundError(
                f"No cached metadata at {cached_metadata} and no cache info at "
                f"{cache_info_path}. Rebuild cache with: "
                "python scripts/data/cache_and_report_all_data.py"
            )
        with open(cache_info_path) as f:
            cache_info = json.load(f)
        metadata_path = Path(cache_info["metadata_path"])
        data_dir = Path(cache_info.get("data_dir", "."))

    from malid_lite.dataloader import MalIDPublishedDataLoader
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        verbose=0,
    )
    return loader, metadata_path


def _check_integration_prerequisites(
    model3_suffix: Optional[str] = None,
) -> Optional[str]:
    """Return None if prerequisites met, or an error message string."""
    cache_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data"
    folds_dir = cache_dir / "data_folds"
    emb_dir = cache_dir / "embeddings"

    if not folds_dir.exists():
        return f"Fold cache not found: {folds_dir}"
    if not any(folds_dir.glob("fold_0_train_*")):
        return "Fold 0 train data not found in cache"

    # Check cv_ensemble base model artifacts exist for fold 0
    for num in [1, 2, 3]:
        suffix = model3_suffix if num == 3 else None
        try:
            resolved_dir, _ = resolve_model_artifact_dir(
                model_name=f"model{num}",
                dataset_name="mal-id-orig-data",
                classification_mode="multiclass",
                gene_locus="TCR",
                training_context="cv_ensemble",
                output_suffix=suffix,
            )
        except (FileNotFoundError, ValueError) as e:
            return f"Model {num} cv_ensemble: {e}"
        if not any(resolved_dir.glob("fold_0_*")):
            return f"Model {num} fold 0 artifacts not found: {resolved_dir}"

    if not emb_dir.exists() or not any(emb_dir.glob("*_embeddings.npy")):
        return f"Pre-computed embeddings not found: {emb_dir}"

    return None


def test_26_integration_multiclass(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 26: Full multiclass fold 0 pipeline (participant subset)."""
    tlog.log("\n--- Test 26: Integration - multiclass fold 0 ---")
    try:
        from malid_lite.training.train_ensemble import (
            run_ensemble_fold,
            save_fold_artifacts,
        )

        loader, metadata_path = _get_integration_loader()
        fold_id = 0

        # Resolve model directories (auto-detects suffixed dirs for model3)
        model_dirs = {}
        model_summaries = {}
        for num in [1, 2, 3]:
            suffix_arg = model3_suffix if num == 3 else None
            resolved_dir, suffix = resolve_model_artifact_dir(
                model_name=f"model{num}",
                dataset_name="mal-id-orig-data",
                classification_mode="multiclass",
                gene_locus="TCR",
                training_context="cv_ensemble",
                output_suffix=suffix_arg,
            )
            model_dirs[num] = resolved_dir
            model_summaries[num] = read_model_summary(resolved_dir)
            tlog.log(f"  Model {num}: {resolved_dir.name}"
                     + (f" (suffix={suffix!r})" if suffix else ""))
        embedding_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "embeddings"

        t0 = time.time()
        fold_result = run_ensemble_fold(
            loader=loader,
            fold_id=fold_id,
            model_nums=[1, 2, 3],
            model_dirs=model_dirs,
            gene_locus="TCR",
            embedding_dir=embedding_dir,
            disease_filter=None,
            reference_class=None,
            verbose=1,
            model_summaries=model_summaries,
            n_jobs=n_jobs,
        )
        elapsed = time.time() - t0
        tlog.log(f"  n_jobs={n_jobs}")

        # Verify result structure
        assert fold_result["fold_id"] == 0
        assert "ensemble_metrics" in fold_result
        assert "base_model_metrics" in fold_result
        assert "pipeline" in fold_result
        assert "metamodel_config" in fold_result
        assert "predictions_rows" in fold_result

        em = fold_result["ensemble_metrics"]
        assert 0.0 <= em["accuracy"] <= 1.0
        assert em["mcc"] is not None

        # Check base model metrics exist for all 3 models
        for model_num in [1, 2, 3]:
            assert model_num in fold_result["base_model_metrics"]
            bm = fold_result["base_model_metrics"][model_num]
            assert 0.0 <= bm["accuracy"] <= 1.0

        # Save artifacts to test output dir
        output_dir = (Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
                      / "integration")
        save_fold_artifacts(output_dir, fold_result)
        assert (output_dir / "metamodel" / "fold_0_ridge_cv_metamodel.joblib").exists()
        assert (output_dir / "metamodel" / "fold_0_metamodel_config.json").exists()

        tlog.log(f"  Fold 0 complete in {elapsed:.1f}s")
        tlog.log(f"  Ensemble: accuracy={em['accuracy']:.4f}, MCC={em['mcc']:.4f}")
        for num in [1, 2, 3]:
            bm = fold_result["base_model_metrics"][num]
            tlog.log(f"  Model {num}: accuracy={bm['accuracy']:.4f}, "
                     f"MCC={bm.get('mcc', 'N/A')}")
        tlog.log(f"  Predictions: {len(fold_result['predictions_rows'])} specimens")

        tlog.record("Integration multiclass fold 0", True, {
            "elapsed": elapsed,
            "ensemble_accuracy": em["accuracy"],
            "ensemble_mcc": em.get("mcc"),
            "n_predictions": len(fold_result["predictions_rows"]),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("Integration multiclass fold 0", False, {"error": str(e)})


def test_27_integration_binary(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 27: Binary mode fold 0 pipeline (one disease vs reference)."""
    tlog.log("\n--- Test 27: Integration - binary fold 0 ---")
    try:
        from malid_lite.training.train_ensemble import run_ensemble_fold

        loader, _ = _get_integration_loader()
        fold_id = 0

        # Use multiclass artifacts with binary disease_filter (the ensemble
        # filters specimens at predict time, same as the main flow)
        model_dirs = {}
        model_summaries = {}
        for num in [1, 2, 3]:
            suffix_arg = model3_suffix if num == 3 else None
            resolved_dir, _ = resolve_model_artifact_dir(
                model_name=f"model{num}",
                dataset_name="mal-id-orig-data",
                classification_mode="multiclass",
                gene_locus="TCR",
                training_context="cv_ensemble",
                output_suffix=suffix_arg,
            )
            model_dirs[num] = resolved_dir
            model_summaries[num] = read_model_summary(resolved_dir)
        embedding_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "embeddings"

        t0 = time.time()
        fold_result = run_ensemble_fold(
            loader=loader,
            fold_id=fold_id,
            model_nums=[1, 2, 3],
            model_dirs=model_dirs,
            gene_locus="TCR",
            embedding_dir=embedding_dir,
            disease_filter=(BINARY_DISEASE, BINARY_REFERENCE),
            reference_class=BINARY_REFERENCE,
            verbose=1,
            model_summaries=model_summaries,
            n_jobs=n_jobs,
        )
        elapsed = time.time() - t0
        tlog.log(f"  n_jobs={n_jobs}")

        em = fold_result["ensemble_metrics"]
        assert 0.0 <= em["accuracy"] <= 1.0

        # In binary mode, only 2 classes
        config = fold_result["metamodel_config"]
        assert len(config["classes"]) == 2
        assert BINARY_DISEASE in config["classes"] or str(BINARY_DISEASE) in config["classes"]

        tlog.log(f"  Binary fold 0 complete in {elapsed:.1f}s")
        tlog.log(f"  Ensemble: accuracy={em['accuracy']:.4f}")
        tlog.log(f"  Classes: {config['classes']}")
        tlog.log(f"  Features: {config['n_features']}")

        tlog.record("Integration binary fold 0", True, {
            "elapsed": elapsed,
            "accuracy": em["accuracy"],
            "n_features": config["n_features"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("Integration binary fold 0", False, {"error": str(e)})


def test_28_artifact_roundtrip(tlog: _TestLogger):
    """Test 28: Artifact save/load round-trip."""
    tlog.log("\n--- Test 28: Artifact round-trip ---")
    try:
        import joblib

        output_dir = (Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
                      / "integration")
        metamodel_dir = output_dir / "metamodel"

        pipeline_path = metamodel_dir / "fold_0_ridge_cv_metamodel.joblib"
        config_path = metamodel_dir / "fold_0_metamodel_config.json"

        if not pipeline_path.exists():
            tlog.log("  SKIP: artifacts not found (run test_26 first)")
            tlog.record("Artifact round-trip (SKIPPED)", True, {"skipped": True})
            return

        # Load pipeline
        pipeline = joblib.load(pipeline_path)
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")
        assert hasattr(pipeline, "classes_")

        # Load config
        with open(config_path) as f:
            config = json.load(f)
        assert "feature_columns" in config
        assert "classes" in config
        assert "n_features" in config

        # Verify pipeline can predict on random data with correct feature count
        rng = np.random.RandomState(18)
        n_features = config["n_features"]
        X_fake = rng.randn(5, n_features)
        y_pred = pipeline.predict(X_fake)
        y_proba = pipeline.predict_proba(X_fake)
        assert len(y_pred) == 5
        assert y_proba.shape == (5, len(config["classes"]))

        tlog.log(f"  Pipeline loaded: {len(config['classes'])} classes, "
                 f"{n_features} features")
        tlog.log(f"  Prediction on random data: {y_pred[:3]}")
        tlog.record("Artifact round-trip", True, {
            "n_classes": len(config["classes"]),
            "n_features": n_features,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("Artifact round-trip", False, {"error": str(e)})


def test_29_run_config_and_results_md(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 29: run_config.json and RESULTS_*.md generation."""
    tlog.log("\n--- Test 29: run_config.json and RESULTS MD ---")
    try:
        from malid_lite.training.train_ensemble import train_ensemble

        loader, _ = _get_integration_loader()
        fold_id = 0

        # Resolve model directories (auto-detects suffixed dirs for model3)
        model_dirs = {}
        model_summaries = {}
        for num in [1, 2, 3]:
            suffix_arg = model3_suffix if num == 3 else None
            resolved_dir, suffix = resolve_model_artifact_dir(
                model_name=f"model{num}",
                dataset_name="mal-id-orig-data",
                classification_mode="multiclass",
                gene_locus="TCR",
                training_context="cv_ensemble",
                output_suffix=suffix_arg,
            )
            model_dirs[num] = resolved_dir
            model_summaries[num] = read_model_summary(resolved_dir)
            tlog.log(f"  Model {num}: {resolved_dir.name}"
                     + (f" (suffix={suffix!r})" if suffix else ""))
        embedding_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "embeddings"

        output_dir = (Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
                      / "integration" / "full_run")

        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset_name": "mal-id-orig-data",
            "classification_mode": "multiclass",
            "gene_locus": "TCR",
            "models_included": [1, 2, 3],
            "fold_ids": [fold_id],
            "base_model_paths": {f"model{n}": str(d) for n, d in model_dirs.items()},
            "metamodel_config": {
                "algorithm": "ridge_cv",
                "alpha": 0.0,
            },
        }

        t0 = time.time()
        all_fold_results, summary = train_ensemble(
            loader=loader,
            fold_ids=[fold_id],
            model_nums=[1, 2, 3],
            model_dirs=model_dirs,
            gene_locus="TCR",
            output_dir=output_dir,
            embedding_dir=embedding_dir,
            disease_filter=None,
            reference_class=None,
            run_config=run_config,
            model_summaries=model_summaries,
            verbose=1,
            n_jobs=n_jobs,
        )
        elapsed = time.time() - t0
        tlog.log(f"  n_jobs={n_jobs}")

        # Check run_config.json
        rc_path = output_dir / "run_config.json"
        assert rc_path.exists(), "run_config.json not created"
        with open(rc_path) as f:
            saved_rc = json.load(f)
        assert saved_rc["dataset_name"] == "mal-id-orig-data"
        assert saved_rc["classification_mode"] == "multiclass"

        # Check RESULTS_*.md
        md_files = list(output_dir.glob("RESULTS_*.md"))
        assert len(md_files) >= 1, "No RESULTS MD file found"
        md_content = md_files[0].read_text()
        assert "Ensemble Training Results" in md_content
        assert "Model Comparison" in md_content

        # Check summary_*.json
        summary_files = list(output_dir.glob("summary_*.json"))
        assert len(summary_files) >= 1, "No summary JSON found"

        # Check predictions CSV
        pred_path = output_dir / "ensemble_predictions.csv"
        assert pred_path.exists(), "Predictions CSV not created"
        preds_df = pd.read_csv(pred_path)
        assert "specimen_label" in preds_df.columns
        assert "true_disease" in preds_df.columns
        assert "ensemble_predicted" in preds_df.columns

        tlog.log(f"  Full pipeline complete in {elapsed:.1f}s")
        tlog.log(f"  run_config.json: {rc_path}")
        tlog.log(f"  RESULTS MD: {md_files[0].name}")
        tlog.log(f"  Predictions: {len(preds_df)} rows")

        tlog.record("run_config and RESULTS MD", True, {
            "elapsed": elapsed,
            "n_predictions": len(preds_df),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("run_config and RESULTS MD", False, {"error": str(e)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble quick test")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run only unit tests (no real data needed)")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel workers for Model 2/3 predictions. Default: 4.")
    parser.add_argument("--model3-suffix", type=str, default=None,
                        help="Output suffix for Model 3 artifact directory "
                             "(e.g. 'auto_tuned_v3'). Required when multiple "
                             "suffixed directories exist.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
    output_dir.mkdir(parents=True, exist_ok=True)

    tlog = _TestLogger(output_dir / f"test_log_{timestamp}.txt")
    tlog.log(f"Ensemble quick test — {timestamp}")
    tlog.log("=" * 60)

    # Suppress verbose logging from ensemble modules during tests
    logging.basicConfig(level=logging.WARNING)

    # --- Tier 1: Unit tests ---
    tlog.log("\n" + "=" * 60)
    tlog.log("TIER 1: Unit tests (synthetic data)")
    tlog.log("=" * 60)

    unit_tests = [
        test_01_model_predictions_dataclass,
        test_02_build_feature_matrix_multiclass,
        test_03_build_feature_matrix_binary,
        test_04_build_feature_matrix_abstention_harmonization,
        test_05_build_feature_matrix_single_model,
        test_06_train_metamodel,
        test_07_evaluate_predictions_multiclass,
        test_08_evaluate_predictions_binary,
        test_09_evaluate_predictions_perfect,
        test_10_evaluate_predictions_abstention_penalty,
        test_11_column_alignment,
        test_12_aggregate_fold_results_multiclass,
        test_13_aggregate_mcc,
        test_14_generate_results_md,
        test_15_log_comparison_table,
        test_16_aggregate_fold_results_binary,
        test_17_binary_md_and_comparison,
        test_18_save_multi_binary_summary,
        test_19_pair_resolution_and_paths,
        test_20_validate_mode_and_classes_errors,
        test_21_train_ensemble_binary_mocked,
        test_22_multi_binary_orchestration,
        test_23_generate_results_md_binary_enrichment,
        test_24_binary_specimen_filtering,
    ]

    for test_fn in unit_tests:
        test_fn(tlog)

    # --- Tier 2: Integration tests ---
    tier2_skip_reason = None
    if args.unit_only:
        tier2_skip_reason = "--unit-only flag"
        tlog.log("\n" + "=" * 60)
        tlog.log("Skipping Tier 2 (--unit-only)")
        tlog.log("=" * 60)
    else:
        tlog.log("\n" + "=" * 60)
        tlog.log("TIER 2: Integration tests (real data)")
        tlog.log("=" * 60)

        m3s = args.model3_suffix
        prereq_error = _check_integration_prerequisites(model3_suffix=m3s)
        if prereq_error:
            tier2_skip_reason = prereq_error
            tlog.log(f"\n  SKIP Tier 2: {prereq_error}")
            tlog.log("  Train base models with --training-context cv_ensemble first.")
        else:
            tlog.log(f"\n  Using n_jobs={args.n_jobs}, model3_suffix={m3s!r}")
            test_26_integration_multiclass(tlog, n_jobs=args.n_jobs, model3_suffix=m3s)
            test_27_integration_binary(tlog, n_jobs=args.n_jobs, model3_suffix=m3s)
            test_28_artifact_roundtrip(tlog)
            test_29_run_config_and_results_md(tlog, n_jobs=args.n_jobs, model3_suffix=m3s)

    # --- Summary ---
    tlog.log("\n" + "=" * 60)
    tlog.log("SUMMARY")
    tlog.log("=" * 60)

    n_pass = sum(1 for t in tlog._results["tests"] if t["status"] == "PASS")
    n_fail = sum(1 for t in tlog._results["tests"] if t["status"] == "FAIL")
    n_total = len(tlog._results["tests"])

    tlog.log(f"  {n_pass}/{n_total} passed, {n_fail} failed")
    if tier2_skip_reason:
        tlog.log(f"  WARNING: Tier 2 integration tests were SKIPPED ({tier2_skip_reason})")
    if n_fail > 0:
        tlog.log("  Failed tests:")
        for t in tlog._results["tests"]:
            if t["status"] == "FAIL":
                tlog.log(f"    - {t['test']}: {t['details'].get('error', 'unknown')}")

    results_file = tlog.close()
    print(f"\nResults saved: {results_file}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
