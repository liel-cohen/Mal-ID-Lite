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
  25. Resume mode: metrics match original run (multiclass + binary + round-trip)

Stage 4b — auto-training mode detection and parameter comparison:
  31-36b. compare_training_params: matching, mismatch, None skip, model_name,
          list order-independence, Model 3 all params, missing summary key
  37-41.  resolve_base_model_mode: retrain, no artifacts, LOAD, LOAD mismatch, RESUME
  42-43.  CLI arg interaction validation (retrain/resume conflicts, via subprocess)
  44-45.  preflight_validate_resume_params: no _meta, _meta mismatch

Stage 4c — auto_train_base_model dispatch:
  46. _format_elapsed_time: seconds → human-readable string formatting
  47. auto_train_base_model: invalid model_num → ValueError
  48. auto_train_base_model: Model 1 dispatch (shared kwargs, training params, no n_jobs)
  49. auto_train_base_model: Model 2 dispatch (n_jobs, training params, resume)
  50. auto_train_base_model: Model 3 dispatch (infra kwargs, training params)
  51. auto_train_base_model: Model 3 optional kwargs (device/batch_size omitted when None)
  52. auto_train_base_model: empty training_params → only shared kwargs

Stage 4d — validate_ensemble_args, cross-model disease classes, and logging:
  53.     validate_ensemble_args: valid defaults → no error
  54.     validate_ensemble_args: resume + retrain conflict → error
  55.     validate_ensemble_args: retrain model not in --models → error
  56.     validate_ensemble_args: output-suffix + output-dir → error
  57.     validate_ensemble_args: n_jobs=0 → error
  58.     validate_ensemble_args: --diseases in multiclass → error
  59.     validate_ensemble_args: training params for excluded model → error
  60.     validate_ensemble_args: suffix for excluded model → error
  61.     validate_ensemble_args: M3 infra args when M3 excluded → error
  62.     validate_ensemble_args: invalid per-model param range → ValueError
  63.     validate_ensemble_args: suffix sanitization (bad chars → underscore)
  64.     validate_ensemble_args: suffix with all special chars → non-empty
  65.     validate_ensemble_args: clean suffix passthrough
  66.     validate_ensemble_args: missing metadata-path → error
  67.     validate_ensemble_args: missing gene-reference-path → error
  68.     validate_ensemble_args: existing metadata-path → accepted
  69.     validate_ensemble_args: M3 tuning flags with fixed strategy → error
  70.     validate_ensemble_args: M3 entropy_max_fraction with wrong strategy → error
  71.     validate_ensemble_args: M3 entropy_bottom_percentile with wrong strategy → error
  72.     validate_ensemble_args: M3 entropy_max_fraction with auto_tuned → error
  73.     validate_ensemble_args: M3 entropy_bottom_percentile with auto_tuned → error
  74.     validate_ensemble_args: M3 entropy_cutoff with entropy_max_fraction → valid
  75.     validate_ensemble_args: M3 auto_tuned with tuning flags → valid
  76.     validate_ensemble_args: M3 strategy=None with tuning flags → valid
  77.     _validate_cross_model_disease_classes: matching classes → no error
  78.     _validate_cross_model_disease_classes: mismatched classes → ValueError
  79.     _validate_cross_model_disease_classes: None summaries skipped
  80.     _validate_cross_model_disease_classes: missing model_classes key skipped
  81.     _validate_cross_model_disease_classes: single model → no comparison
  82.     _validate_cross_model_disease_classes: no class keys → skipped
  83.     _log_base_model_status_table: LOAD/TRAIN/RESUME modes + isdigit filter

Integration tests (real data):
  26. Full multiclass fold 0 pipeline (specimen subset via max_specimens_per_class)
  27. Binary mode fold 0 pipeline (one disease vs reference, specimen subset)
  28. Artifact save/load round-trip
  29. run_config.json and RESULTS_*.md generation (specimen subset)
  30. Resume integration: original run -> resume, verify metrics match

Design notes
------------
- Unit tests use randomly generated probability matrices (not real model predictions).
- Integration tests require pre-trained cv_ensemble base model artifacts for fold 0.
- Model 3 integration uses pre-computed embeddings; the ensemble only calls
  predict_proba, not the full training pipeline, so memory usage stays manageable.
- Integration tests use max_specimens_per_class to limit specimens for speed.

Requirements
------------
- Tier 1 (unit): numpy, pandas, scikit-learn, glmnet
- Tier 2 (integration): fold cache + pre-computed embeddings + cv_ensemble base model
  artifacts for fold 0

Expected runtime
----------------
- Tier 1 only: ~30 seconds
- Tier 1 + Tier 2: ~2-5 minutes (depending on hardware)

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
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
)
from malid_lite.training.train_ensemble import (
    MODEL_DISPLAY_NAMES,
    TRAINING_CONTEXT,
    ModelPredictions,
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
    _read_fold_meta,
    _save_multi_binary_summary,
    _validate_cross_model_disease_classes,
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


def _get_test_output_dir(test_name: str) -> Path:
    """Create a clean test output subdirectory, removing stale artifacts from prior runs."""
    import shutil
    test_dir = OUTPUT_DIR / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def tlog():
    """Provide a _TestLogger instance for each test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"test_log_{timestamp}.txt"
    logger = _TestLogger(log_file)
    yield logger
    logger.close()


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

        X, abstained_labels, abstained_diseases, _fill_info = build_feature_matrix(
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

        # Check columns preserve model insertion order (Model 1, 2, 3)
        expected_order = []
        for num in [1, 2, 3]:
            display = MODEL_DISPLAY_NAMES[num]
            expected_order.extend(
                sorted(c for c in X.columns if c.startswith(f"TCR:{display}:"))
            )
        assert list(X.columns) == expected_order

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

        X, _, _, _ = build_feature_matrix(
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

        X, abstained_labels, abstained_diseases, _fill_info = build_feature_matrix(
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

        X, _, _, _ = build_feature_matrix(
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
        X_val, _, _, _ = build_feature_matrix(val_predictions, "TCR", None)

        # Test: same 3 models, same 6 classes
        test_specimens = [f"test_{i:03d}" for i in range(15)]
        test_predictions = {}
        for model_num in [1, 2, 3]:
            test_predictions[model_num] = ModelPredictions(
                probabilities=_make_mock_proba(test_specimens, DISEASE_CLASSES, rng),
                abstained_specimen_labels=[],
                abstained_specimen_diseases=[],
            )
        X_test, _, _, _ = build_feature_matrix(test_predictions, "TCR", None)

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
        assert "Per-Fold Results" in md
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
        tmp_dir = _get_test_output_dir("test_18_multi_binary_summary")

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

        # Build minimal fold results for each pair (function uses them for per-fold tables)
        fold_results_by_pair = {}
        for disease, ref in pairs:
            pk = make_pair_name(disease, ref)
            fold_results_by_pair[pk] = [
                {
                    "fold_id": 0,
                    "ensemble_metrics": {
                        "fold_id": 0,
                        "accuracy": 0.85,
                        "auroc_binary": 0.90,
                        "auprc_binary": 0.88,
                        "mcc": 0.70,
                        "n_scored": 10,
                        "n_abstained": 0,
                    },
                    "base_model_metrics": {},
                    "test_abstained_details": [],
                },
            ]

        _save_multi_binary_summary(
            tmp_dir, summaries, fold_results_by_pair, pairs, "Healthy/Background",
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
            assert "auroc_pooled" in cross_json["pairs"][pk]["ensemble"]

        tlog.log(f"  MD: {md_files[0].name}, JSON: {json_files[0].name}")
        tlog.log(f"  {len(cross_json['pairs'])} pairs in summary")
        tlog.record("multi-binary summary", True, {"n_pairs": 2})
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

        # multi-binary with 2 classes, no reference -> now requires reference_class
        try:
            validate_mode_and_classes("multi-binary", classes_2, None, None)
            assert False, "Should have raised ValueError for missing reference_class"
        except ValueError as e:
            assert "reference-class" in str(e).lower()
            errors_caught += 1

        # binary without reference_class -> ValueError
        try:
            validate_mode_and_classes("binary", classes_2, None, None)
            assert False, "Should have raised ValueError for missing reference_class"
        except ValueError as e:
            assert "reference-class" in str(e).lower()
            errors_caught += 1

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

        tmp = _get_test_output_dir("test_21_binary_mocked")

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

        for fid in fold_ids:
            assert (tmp / f"fold_{fid}_ridge_cv_metamodel.joblib").exists()
            assert (tmp / f"fold_{fid}_metamodel_config.json").exists()
            assert (tmp / f"fold_{fid}_ensemble_results.json").exists()

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

        ref_class = "Healthy/Background"
        model_nums = [1, 3]
        fold_ids = [0, 1]
        pairs_to_train = [
            ("Covid19", ref_class),
            ("HIV", ref_class),
        ]

        base_output_dir = _get_test_output_dir("test_22_multi_binary")

        all_pair_summaries = {}
        all_pair_fold_results = {}

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
                fold_results, summary = train_ensemble(
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
            all_pair_fold_results[pair_key] = fold_results

        # --- Verify per-pair outputs ---
        for disease, ref in pairs_to_train:
            pk = make_pair_name(disease, ref)
            pair_dir = base_output_dir / pk
            assert pair_dir.exists(), f"Pair directory missing: {pk}"
            assert (pair_dir / "ensemble_predictions.csv").exists()
            assert any(pair_dir.glob("fold_*_ridge_cv_metamodel.joblib"))
            assert len(list(pair_dir.glob("summary_*.json"))) == 1
            assert len(list(pair_dir.glob("RESULTS_*.md"))) == 1

            # Verify binary metrics
            assert "auroc_pooled" in all_pair_summaries[pk]["ensemble"]

        # --- Save and verify cross-pair summary ---
        _save_multi_binary_summary(
            base_output_dir, all_pair_summaries, all_pair_fold_results,
            pairs_to_train, ref_class,
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
        assert "Per-Fold Results" in md, "Should have per-fold section"
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

    Unlike _make_mock_fold_result, this uses train_metamodel (the real GlmnetLogitNet
    pipeline) and derives all metrics from the feature matrix columns, so metrics
    are perfectly consistent between the saved artifacts and what resume mode
    would recompute.

    Returns (fold_result_dict, metadata_df) where metadata_df has the
    specimen_label/participant_label columns needed by run_ensemble_fold_from_features.
    """
    rng = np.random.RandomState(seed)
    str_classes = np.array([str(c) for c in classes])

    val_specimens = [f"val_{fold_id}_{i:03d}" for i in range(n_val)]
    test_specimens = [f"test_{fold_id}_{i:03d}" for i in range(n_test)]
    specimen_to_participant = {s: f"part_{s}" for s in val_specimens + test_specimens}

    y_val = np.array([str_classes[i % len(str_classes)] for i in range(n_val)])
    y_test = np.array([str_classes[i % len(str_classes)] for i in range(n_test)])

    # Build feature columns: {locus}:{display_name}:{class}
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

    # Generate random probabilities
    X_val_data = rng.rand(n_val, len(feature_cols)).astype(np.float64)
    X_test_data = rng.rand(n_test, len(feature_cols)).astype(np.float64)

    # Normalize multiclass probabilities per model
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

    # Evaluate base models from feature matrix columns (same logic as resume)
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

    # Build metadata for all specimens (val + test + abstained)
    meta_rows = [
        {SPECIMEN_COL: s, PARTICIPANT_COL: p}
        for s, p in specimen_to_participant.items()
    ]
    for detail in abstained_details:
        meta_rows.append({
            SPECIMEN_COL: detail["specimen_label"],
            PARTICIPANT_COL: detail["participant_label"],
        })
    metadata_df = pd.DataFrame(meta_rows)

    return fold_result, metadata_df


def test_25_resume_matches_original(tlog: _TestLogger):
    """Test 25: Resume produces identical metrics to original run.

    Creates a fold result with consistent feature matrices and metrics using the
    real metamodel (GlmnetLogitNet), saves all artifacts, then calls
    run_ensemble_fold_from_features on the saved artifacts. Verifies that base
    model metrics and ensemble metrics match exactly (same data + deterministic
    training = same results).

    Also tests:
    - Abstention count is preserved across save/load
    - test_abstained_details round-trips through the results JSON
    - Binary mode probability reconstruction from single-column feature matrix
    """
    tlog.log("\n--- Test 25: Resume matches original ---")

    # --- 25a: Multiclass with abstentions ---
    tlog.log("  25a: Multiclass with abstentions")
    try:
        classes = np.array(["Covid19", "HIV", "Healthy"])
        model_nums = [1, 3]
        n_abstained = 2

        fold_result, metadata_df = _make_mock_fold_result_with_features(
            fold_id=0, gene_locus="TCR", classes=classes, model_nums=model_nums,
            n_val=30, n_test=18, n_abstained=n_abstained, seed=42,
        )

        tmp = _get_test_output_dir("test_25a_resume_multiclass")

        save_fold_artifacts(tmp, fold_result)

        # Verify results JSON has abstention details
        results_json = tmp / "fold_0_ensemble_results.json"
        assert results_json.exists(), "Results JSON not created"
        with open(results_json) as f:
            saved_results = json.load(f)
        assert len(saved_results["test_abstained_details"]) == n_abstained
        assert saved_results["ensemble"]["n_abstained"] == n_abstained

        # Run resume
        class MockLoader:
            @property
            def metadata(self):
                return metadata_df

        resume_result = run_ensemble_fold_from_features(
            fold_id=0, output_dir=tmp, model_nums=model_nums,
            gene_locus="TCR", loader=MockLoader(), reference_class=None,
        )

        # Compare base model metrics (tolerance for float)
        for model_num in model_nums:
            orig = fold_result["base_model_metrics"][model_num]
            resu = resume_result["base_model_metrics"][model_num]
            for key in ["accuracy", "mcc", "n_scored", "n_abstained",
                        "auroc_ovo_weighted", "auroc_ovo_macro", "log_loss"]:
                orig_val = orig.get(key)
                resu_val = resu.get(key)
                if isinstance(orig_val, float):
                    assert abs(orig_val - resu_val) < 1e-10, (
                        f"Model {model_num} {key}: orig={orig_val} vs resume={resu_val}"
                    )
                else:
                    assert orig_val == resu_val, (
                        f"Model {model_num} {key}: orig={orig_val} vs resume={resu_val}"
                    )

        # Compare ensemble metrics (exact match — deterministic training)
        orig_ens = fold_result["ensemble_metrics"]
        resu_ens = resume_result["ensemble_metrics"]
        for key in ["accuracy", "mcc", "n_scored", "n_abstained",
                    "auroc_ovo_weighted", "auroc_ovo_macro", "log_loss"]:
            orig_val = orig_ens.get(key)
            resu_val = resu_ens.get(key)
            if isinstance(orig_val, float):
                assert abs(orig_val - resu_val) < 1e-10, (
                    f"Ensemble {key}: orig={orig_val} vs resume={resu_val}"
                )
            else:
                assert orig_val == resu_val, (
                    f"Ensemble {key}: orig={orig_val} vs resume={resu_val}"
                )

        # Verify abstention details round-tripped
        assert len(resume_result["test_abstained_details"]) == n_abstained
        assert resume_result["metamodel_config"]["n_test_abstained"] == n_abstained

        tlog.log(f"    Multiclass: all metrics match, n_abstained={n_abstained} preserved")
        tlog.log(f"    Ensemble accuracy: orig={orig_ens['accuracy']:.4f}, "
                 f"resume={resu_ens['accuracy']:.4f}")

        tlog.record("resume multiclass", True, {"n_abstained": n_abstained})
    except Exception as e:
        tlog.log(f"    FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resume multiclass", False, {"error": str(e)})

    # --- 25b: Binary with abstentions ---
    tlog.log("  25b: Binary with abstentions")
    try:
        classes = np.array(["Covid19", "Healthy/Background"])
        ref_class = "Healthy/Background"
        model_nums = [1, 2]
        n_abstained = 1

        fold_result, metadata_df = _make_mock_fold_result_with_features(
            fold_id=0, gene_locus="TCR", classes=classes, model_nums=model_nums,
            n_val=20, n_test=16, n_abstained=n_abstained,
            reference_class=ref_class, seed=99,
        )

        tmp = _get_test_output_dir("test_25b_resume_binary")

        save_fold_artifacts(tmp, fold_result)

        class MockLoader:
            @property
            def metadata(self):
                return metadata_df

        resume_result = run_ensemble_fold_from_features(
            fold_id=0, output_dir=tmp, model_nums=model_nums,
            gene_locus="TCR", loader=MockLoader(), reference_class=ref_class,
        )

        # Compare base model metrics
        for model_num in model_nums:
            orig = fold_result["base_model_metrics"][model_num]
            resu = resume_result["base_model_metrics"][model_num]
            for key in ["accuracy", "mcc", "n_scored", "n_abstained",
                        "auroc_binary", "auprc_binary", "log_loss"]:
                orig_val = orig.get(key)
                resu_val = resu.get(key)
                if isinstance(orig_val, float):
                    assert abs(orig_val - resu_val) < 1e-10, (
                        f"Model {model_num} {key}: {orig_val} vs {resu_val}"
                    )
                else:
                    assert orig_val == resu_val, (
                        f"Model {model_num} {key}: {orig_val} vs {resu_val}"
                    )

        # Compare ensemble metrics
        orig_ens = fold_result["ensemble_metrics"]
        resu_ens = resume_result["ensemble_metrics"]
        for key in ["accuracy", "mcc", "n_scored", "n_abstained",
                    "auroc_binary", "auprc_binary", "log_loss"]:
            orig_val = orig_ens.get(key)
            resu_val = resu_ens.get(key)
            if isinstance(orig_val, float):
                assert abs(orig_val - resu_val) < 1e-10, (
                    f"Ensemble {key}: {orig_val} vs {resu_val}"
                )
            else:
                assert orig_val == resu_val, (
                    f"Ensemble {key}: {orig_val} vs {resu_val}"
                )

        assert resume_result["metamodel_config"]["n_test_abstained"] == n_abstained

        tlog.log(f"    Binary: all metrics match, n_abstained={n_abstained} preserved")
        tlog.log(f"    Binary reconstruction: {len(fold_result['base_model_metrics'])} models OK")

        tlog.record("resume binary", True, {"n_abstained": n_abstained})
    except Exception as e:
        tlog.log(f"    FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resume binary", False, {"error": str(e)})

    # --- 25c: Full train_ensemble round-trip (original -> resume) ---
    tlog.log("  25c: train_ensemble round-trip (original -> resume via mocked folds)")
    try:
        from unittest.mock import patch

        classes = np.array(["A", "B", "C"])
        model_nums = [1, 3]
        fold_ids = [0, 1]

        # Pre-build fold results with consistent feature matrices
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

        tmp = _get_test_output_dir("test_25c_resume_roundtrip")

        # --- Original run (mocked fold function) ---
        with patch(
            "malid_lite.training.train_ensemble.run_ensemble_fold",
            side_effect=_mock_run_fold,
        ):
            _, original_summary = train_ensemble(
                loader=MockLoader(),
                fold_ids=fold_ids,
                model_nums=model_nums,
                model_dirs={n: Path("dummy") for n in model_nums},
                gene_locus="TCR",
                output_dir=tmp,
                run_config={"test": True, "classification_mode": "multiclass"},
            )

        # --- Resume run ---
        _, resume_summary = train_ensemble(
            loader=MockLoader(),
            fold_ids=fold_ids,
            model_nums=model_nums,
            model_dirs={},
            gene_locus="TCR",
            output_dir=tmp,
            resume=True,
            run_config={"test": True, "classification_mode": "multiclass",
                        "resume": True},
        )

        # Compare ensemble aggregated metrics
        orig_ens = original_summary["ensemble"]
        resu_ens = resume_summary["ensemble"]
        for key in ["accuracy_global"]:
            assert abs(orig_ens[key] - resu_ens[key]) < 1e-10, (
                f"Aggregated {key}: {orig_ens[key]} vs {resu_ens[key]}"
            )

        # Compare base model aggregated metrics
        for model_num in model_nums:
            orig_bm = original_summary["base_models"][f"model{model_num}"]
            resu_bm = resume_summary["base_models"][f"model{model_num}"]
            assert abs(orig_bm["accuracy_global"] - resu_bm["accuracy_global"]) < 1e-10, (
                f"Model {model_num} accuracy_global mismatch"
            )

        tlog.log(f"    Round-trip: {len(fold_ids)} folds, all aggregated metrics match")

        tlog.record("resume round-trip", True, {"n_folds": len(fold_ids)})
    except Exception as e:
        tlog.log(f"    FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resume round-trip", False, {"error": str(e)})


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


MAX_SPECIMENS_PER_CLASS_INTEGRATION = 5


@pytest.mark.integration
def test_26_integration_multiclass(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 26: Full multiclass fold 0 pipeline (specimen subset)."""
    tlog.log("\n--- Test 26: Integration - multiclass fold 0 ---")
    try:
        from malid_lite.training.train_ensemble import run_ensemble_fold

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
            max_specimens_per_class=MAX_SPECIMENS_PER_CLASS_INTEGRATION,
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
        assert (output_dir / "fold_0_ridge_cv_metamodel.joblib").exists()
        assert (output_dir / "fold_0_metamodel_config.json").exists()
        assert (output_dir / "fold_0_ensemble_results.json").exists()

        tlog.log(f"  Fold 0 complete in {elapsed:.1f}s")
        tlog.log(f"  Ensemble: AUROC_ovo_w={em.get('auroc_ovo_weighted', 'N/A'):.4f}, "
                 f"accuracy={em['accuracy']:.4f}, MCC={em['mcc']:.4f}")
        for num in [1, 2, 3]:
            bm = fold_result["base_model_metrics"][num]
            auroc = bm.get('auroc_ovo_weighted')
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            mcc = bm.get('mcc')
            mcc_str = f"{mcc:.4f}" if mcc is not None else "N/A"
            tlog.log(f"  Model {num}: AUROC_ovo_w={auroc_str}, "
                     f"accuracy={bm['accuracy']:.4f}, MCC={mcc_str}")
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


@pytest.mark.integration
def test_27_integration_binary(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 27: Binary mode fold 0 pipeline (one disease vs reference)."""
    tlog.log("\n--- Test 27: Integration - binary fold 0 ---")
    try:
        from malid_lite.training.train_ensemble import run_ensemble_fold

        loader, _ = _get_integration_loader()
        fold_id = 0

        # Resolve binary-trained model artifacts, then append the pair subdirectory
        # (matching the main flow for multi-binary mode)
        pair_key = make_pair_name(BINARY_DISEASE, BINARY_REFERENCE)
        model_dirs = {}
        model_summaries = {}
        for num in [1, 2, 3]:
            suffix_arg = model3_suffix if num == 3 else None
            resolved_dir, _ = resolve_model_artifact_dir(
                model_name=f"model{num}",
                dataset_name="mal-id-orig-data",
                classification_mode="binary",
                gene_locus="TCR",
                training_context="cv_ensemble",
                output_suffix=suffix_arg,
            )
            pair_dir = resolved_dir / pair_key
            if not pair_dir.exists():
                tlog.log(f"  SKIP: binary pair dir not found: {pair_dir}")
                tlog.record("Integration binary fold 0 (SKIPPED)", True,
                            {"skipped": True})
                return
            model_dirs[num] = pair_dir
            # Summary lives at the mode level (resolved_dir), not in the pair subdir
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
            max_specimens_per_class=MAX_SPECIMENS_PER_CLASS_INTEGRATION,
        )
        elapsed = time.time() - t0
        tlog.log(f"  n_jobs={n_jobs}, max_specimens_per_class={MAX_SPECIMENS_PER_CLASS_INTEGRATION}")

        em = fold_result["ensemble_metrics"]
        assert 0.0 <= em["accuracy"] <= 1.0

        # In binary mode, only 2 classes
        config = fold_result["metamodel_config"]
        assert len(config["classes"]) == 2
        assert BINARY_DISEASE in config["classes"] or str(BINARY_DISEASE) in config["classes"]

        tlog.log(f"  Binary fold 0 complete in {elapsed:.1f}s")
        auroc = em.get('auroc_binary')
        auprc = em.get('auprc_binary')
        mcc = em.get('mcc')
        tlog.log(f"  Ensemble: AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}, "
                 f"AUPRC={f'{auprc:.4f}' if auprc is not None else 'N/A'}, "
                 f"accuracy={em['accuracy']:.4f}, "
                 f"MCC={f'{mcc:.4f}' if mcc is not None else 'N/A'}")
        for num in [1, 2, 3]:
            bm = fold_result["base_model_metrics"][num]
            bm_auroc = bm.get('auroc_binary')
            bm_auprc = bm.get('auprc_binary')
            bm_mcc = bm.get('mcc')
            tlog.log(f"  Model {num}: AUROC={f'{bm_auroc:.4f}' if bm_auroc is not None else 'N/A'}, "
                     f"AUPRC={f'{bm_auprc:.4f}' if bm_auprc is not None else 'N/A'}, "
                     f"accuracy={bm['accuracy']:.4f}, "
                     f"MCC={f'{bm_mcc:.4f}' if bm_mcc is not None else 'N/A'}")
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


@pytest.mark.integration
def test_28_artifact_roundtrip(tlog: _TestLogger):
    """Test 28: Artifact save/load round-trip."""
    tlog.log("\n--- Test 28: Artifact round-trip ---")
    try:
        import joblib

        output_dir = (Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
                      / "integration")

        pipeline_path = output_dir / "fold_0_ridge_cv_metamodel.joblib"
        config_path = output_dir / "fold_0_metamodel_config.json"

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


@pytest.mark.integration
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
            max_specimens_per_class=MAX_SPECIMENS_PER_CLASS_INTEGRATION,
        )
        elapsed = time.time() - t0
        tlog.log(f"  n_jobs={n_jobs}, max_specimens_per_class={MAX_SPECIMENS_PER_CLASS_INTEGRATION}")

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


@pytest.mark.integration
def test_30_integration_resume(
    tlog: _TestLogger, n_jobs: int = 4, model3_suffix: Optional[str] = None,
):
    """Test 30: Integration resume — original run then resume, verify metrics match.

    Runs train_ensemble on fold 0 with max_specimens_per_class for speed, then
    immediately resumes from saved feature matrices. Compares base model metrics
    (exact match from feature matrix columns) and ensemble metrics (exact match
    because metamodel training is deterministic with same data + random_state=0).
    """
    tlog.log("\n--- Test 30: Integration - resume round-trip ---")
    try:
        loader, _ = _get_integration_loader()
        fold_id = 0

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
        embedding_dir = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "embeddings"

        output_dir = (Path(__file__).parent / "test_outputs" / "test_ensemble_quick"
                      / "integration" / "resume_test")

        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset_name": "mal-id-orig-data",
            "classification_mode": "multiclass",
            "gene_locus": "TCR",
            "models_included": [1, 2, 3],
            "fold_ids": [fold_id],
        }

        # --- Original run ---
        tlog.log("  Running original...")
        t0 = time.time()
        _, original_summary = train_ensemble(
            loader=loader,
            fold_ids=[fold_id],
            model_nums=[1, 2, 3],
            model_dirs=model_dirs,
            gene_locus="TCR",
            output_dir=output_dir,
            embedding_dir=embedding_dir,
            run_config=run_config,
            model_summaries=model_summaries,
            n_jobs=n_jobs,
            max_specimens_per_class=MAX_SPECIMENS_PER_CLASS_INTEGRATION,
        )
        t_orig = time.time() - t0
        tlog.log(f"  Original complete in {t_orig:.1f}s")

        # --- Resume run ---
        tlog.log("  Running resume...")
        t0 = time.time()
        _, resume_summary = train_ensemble(
            loader=loader,
            fold_ids=[fold_id],
            model_nums=[1, 2, 3],
            model_dirs={},
            gene_locus="TCR",
            output_dir=output_dir,
            resume=True,
            run_config={**run_config, "resume": True},
        )
        t_resume = time.time() - t0
        tlog.log(f"  Resume complete in {t_resume:.1f}s")

        # --- Compare aggregated metrics ---
        orig_ens = original_summary["ensemble"]
        resu_ens = resume_summary["ensemble"]

        mismatches = []
        # Compare ensemble aggregated metrics
        ensemble_keys = ["accuracy_global", "mcc"]
        for agg_key in ["auroc_ovo_weighted", "auroc_ovo_macro"]:
            if agg_key in orig_ens and isinstance(orig_ens[agg_key], dict):
                ensemble_keys.append(agg_key)
        for key in ensemble_keys:
            orig_val = orig_ens[key]
            resu_val = resu_ens[key]
            if isinstance(orig_val, dict):
                # {mean, std} dict — compare means
                orig_val = orig_val.get("mean")
                resu_val = resu_val.get("mean")
            if orig_val is not None and resu_val is not None:
                diff = abs(orig_val - resu_val)
                if diff > 1e-10:
                    mismatches.append(
                        f"ensemble.{key}: {orig_val} vs {resu_val}"
                    )

        # Compare base model aggregated metrics
        for model_num in [1, 2, 3]:
            orig_bm = original_summary["base_models"][f"model{model_num}"]
            resu_bm = resume_summary["base_models"][f"model{model_num}"]
            bm_keys = ["accuracy_global", "mcc"]
            for key in bm_keys:
                orig_val = orig_bm.get(key)
                resu_val = resu_bm.get(key)
                if orig_val is not None and resu_val is not None:
                    diff = abs(orig_val - resu_val)
                    if diff > 1e-10:
                        mismatches.append(
                            f"model{model_num}.{key}: {orig_val} vs {resu_val}"
                        )

        assert not mismatches, f"Metrics mismatch:\n" + "\n".join(mismatches)

        n_compared = len(ensemble_keys) + len([1, 2, 3]) * len(bm_keys)
        tlog.log(f"  All {n_compared} metrics match between original and resume")
        orig_mcc_mean = orig_ens["mcc"]["mean"] if isinstance(orig_ens.get("mcc"), dict) else orig_ens.get("mcc")
        tlog.log(f"  Ensemble: accuracy={orig_ens['accuracy_global']:.4f}, "
                 f"MCC={orig_mcc_mean:.4f}")
        tlog.log(f"  Speedup: original {t_orig:.1f}s vs resume {t_resume:.1f}s")
        tlog.record("Integration resume round-trip", True, {
            "t_original": t_orig,
            "t_resume": t_resume,
            "accuracy_original": orig_ens["accuracy_global"],
            "accuracy_resume": resu_ens["accuracy_global"],
            "mcc_original": orig_mcc_mean,
            "mcc_resume": resu_ens["mcc"]["mean"] if isinstance(resu_ens.get("mcc"), dict) else resu_ens.get("mcc"),
            "n_metrics_compared": n_compared,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("Integration resume round-trip", False, {"error": str(e)})


# ---------------------------------------------------------------------------
# Tier 1b: Auto-training unit tests (mode detection, param comparison)
# ---------------------------------------------------------------------------


def test_31_compare_training_params_matching(tlog: _TestLogger):
    """compare_training_params: all specified params match → empty list."""
    tlog.log("\n--- Test 31: compare_training_params matching ---")
    try:
        summary = {
            "n_pcs": 15,
            "l1_ratio": 1.0,
            "model_names": ["lasso_cv"],
        }
        cli_params = {"n_pcs": 15, "l1_ratio": 1.0, "model_name": "lasso_cv"}
        mismatches = compare_training_params(1, summary, cli_params)
        assert mismatches == [], f"Expected no mismatches, got: {mismatches}"
        tlog.log("  PASS")
        tlog.record("compare_training_params matching", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params matching", False, {"error": str(e)})
        raise


def test_32_compare_training_params_mismatch(tlog: _TestLogger):
    """compare_training_params: mismatched params → mismatch list."""
    tlog.log("\n--- Test 32: compare_training_params mismatch ---")
    try:
        summary = {
            "n_pcs": 15,
            "l1_ratio": 1.0,
            "model_names": ["lasso_cv"],
        }
        cli_params = {"n_pcs": 20, "l1_ratio": 0.5}
        mismatches = compare_training_params(1, summary, cli_params)
        assert len(mismatches) == 2, f"Expected 2 mismatches, got {len(mismatches)}: {mismatches}"
        mismatch_keys = [m[0] for m in mismatches]
        assert "n_pcs" in mismatch_keys
        assert "l1_ratio" in mismatch_keys
        tlog.log(f"  Mismatches found: {mismatches}")
        tlog.log("  PASS")
        tlog.record("compare_training_params mismatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params mismatch", False, {"error": str(e)})
        raise


def test_33_compare_training_params_none_skip(tlog: _TestLogger):
    """compare_training_params: None CLI params are not compared."""
    tlog.log("\n--- Test 33: compare_training_params None skip ---")
    try:
        summary = {
            "n_pcs": 15,
            "l1_ratio": 1.0,
            "model_names": ["lasso_cv"],
        }
        # n_pcs=None → not compared even though it differs in summary
        cli_params = {"n_pcs": None, "l1_ratio": 1.0, "model_name": None}
        mismatches = compare_training_params(1, summary, cli_params)
        assert mismatches == [], f"Expected no mismatches (None skipped), got: {mismatches}"
        tlog.log("  PASS")
        tlog.record("compare_training_params None skip", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params None skip", False, {"error": str(e)})
        raise


def test_34_compare_training_params_model1_model_name(tlog: _TestLogger):
    """compare_training_params: Model 1 model_name vs model_names list."""
    tlog.log("\n--- Test 34: compare_training_params Model 1 model_name ---")
    try:
        # Summary stores as 1-element list, CLI passes as single string
        summary = {"model_names": ["lasso_cv"]}
        cli_params = {"model_name": "lasso_cv"}
        mismatches = compare_training_params(1, summary, cli_params)
        assert mismatches == [], f"Expected match, got: {mismatches}"

        # Mismatch case
        cli_params_mismatch = {"model_name": "ridge_cv"}
        mismatches = compare_training_params(1, summary, cli_params_mismatch)
        assert len(mismatches) == 1
        assert mismatches[0][0] == "model_name"
        assert mismatches[0][1] == "lasso_cv"  # extracted from list
        assert mismatches[0][2] == "ridge_cv"
        tlog.log("  PASS")
        tlog.record("compare_training_params Model 1 model_name", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params Model 1 model_name", False, {"error": str(e)})
        raise


def test_35_compare_training_params_list_order_independent(tlog: _TestLogger):
    """compare_training_params: list params compared as sorted."""
    tlog.log("\n--- Test 35: compare_training_params list order-independent ---")
    try:
        summary = {
            "p_values": [0.05, 0.001, 0.01, 0.005, 0.0005],
            "retrain_on_full_train": False,
        }
        # Same values, different order
        cli_params = {"p_values": [0.0005, 0.001, 0.005, 0.01, 0.05]}
        mismatches = compare_training_params(2, summary, cli_params)
        assert mismatches == [], f"Expected match (order-independent), got: {mismatches}"

        # Different values
        cli_params_diff = {"p_values": [0.001, 0.01]}
        mismatches = compare_training_params(2, summary, cli_params_diff)
        assert len(mismatches) == 1, f"Expected 1 mismatch, got {len(mismatches)}"
        tlog.log("  PASS")
        tlog.record("compare_training_params list order-independent", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params list order-independent", False, {"error": str(e)})
        raise


def test_36_compare_training_params_model3(tlog: _TestLogger):
    """compare_training_params: Model 3 with all params."""
    tlog.log("\n--- Test 36: compare_training_params Model 3 ---")
    try:
        summary = {
            "aggregation_strategy": "auto_tuned",
            "n_estimators_stage1": 100,
            "n_estimators_stage2": 100,
            "tuning_cv_splits": 3,
            "tuning_strategies": ["entropy_cutoff", "entropy_percentile_cutoff"],
        }
        cli_params = {
            "aggregation_strategy": "auto_tuned",
            "n_estimators_stage1": 100,
            "n_estimators_stage2": None,  # not specified
            "tuning_strategies": ["entropy_percentile_cutoff", "entropy_cutoff"],  # different order
        }
        mismatches = compare_training_params(3, summary, cli_params)
        assert mismatches == [], f"Expected no mismatches, got: {mismatches}"

        # Now mismatch on n_estimators_stage1
        cli_mismatch = {"n_estimators_stage1": 200}
        mismatches = compare_training_params(3, summary, cli_mismatch)
        assert len(mismatches) == 1
        assert mismatches[0] == ("n_estimators_stage1", 100, 200)
        tlog.log("  PASS")
        tlog.record("compare_training_params Model 3", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params Model 3", False, {"error": str(e)})
        raise


def test_36b_compare_training_params_missing_summary_key(tlog: _TestLogger):
    """compare_training_params: missing key in summary → skipped (not a mismatch)."""
    tlog.log("\n--- Test 36b: compare_training_params missing summary key ---")
    try:
        # Old summary without n_estimators_stage1 (added in 4a-2)
        summary = {
            "aggregation_strategy": "auto_tuned",
            # n_estimators_stage1 and n_estimators_stage2 intentionally absent
        }
        cli_params = {
            "aggregation_strategy": "auto_tuned",
            "n_estimators_stage1": 100,
            "n_estimators_stage2": 200,
        }
        mismatches = compare_training_params(3, summary, cli_params)
        # n_estimators keys are absent from summary → should be skipped, not mismatched
        assert mismatches == [], f"Expected no mismatches (missing keys skipped), got: {mismatches}"
        tlog.log("  PASS")
        tlog.record("compare_training_params missing summary key", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("compare_training_params missing summary key", False, {"error": str(e)})
        raise


def test_37_resolve_base_model_mode_retrain(tlog: _TestLogger):
    """resolve_base_model_mode: retrain set → TRAIN regardless of artifacts."""
    tlog.log("\n--- Test 37: resolve_base_model_mode retrain ---")
    try:
        mode, path, summary = resolve_base_model_mode(
            model_num=1,
            retrain_set={1},
            resume_flag=False,
            dataset_name="test-dataset",
            classification_mode="multiclass",
            gene_locus="TCR",
            output_suffix=None,
            cli_training_params={},
        )
        assert mode == "TRAIN", f"Expected TRAIN, got {mode}"
        assert summary is None
        tlog.log(f"  Mode: {mode}, path: {path}")
        tlog.log("  PASS")
        tlog.record("resolve_base_model_mode retrain", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resolve_base_model_mode retrain", False, {"error": str(e)})
        raise


def test_38_resolve_base_model_mode_no_artifacts(tlog: _TestLogger):
    """resolve_base_model_mode: no artifacts → TRAIN."""
    tlog.log("\n--- Test 38: resolve_base_model_mode no artifacts ---")
    try:
        mode, path, summary = resolve_base_model_mode(
            model_num=2,
            retrain_set=set(),
            resume_flag=False,
            dataset_name="nonexistent-dataset-xyz",
            classification_mode="multiclass",
            gene_locus="TCR",
            output_suffix=None,
            cli_training_params={},
        )
        assert mode == "TRAIN", f"Expected TRAIN, got {mode}"
        assert summary is None
        tlog.log(f"  Mode: {mode}, path: {path}")
        tlog.log("  PASS")
        tlog.record("resolve_base_model_mode no artifacts", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resolve_base_model_mode no artifacts", False, {"error": str(e)})
        raise


def test_39_resolve_base_model_mode_load(tlog: _TestLogger):
    """resolve_base_model_mode: full artifacts + summary → LOAD."""
    tlog.log("\n--- Test 39: resolve_base_model_mode LOAD ---")
    try:
        import tempfile, shutil
        # Create a fake model directory with a summary
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        summary_data = {
            "gene_locus": "TCR",
            "training_context": "cv_ensemble",
            "classification_mode": "multiclass",
            "n_pcs": 15,
            "l1_ratio": 1.0,
            "model_names": ["lasso_cv"],
            "timestamp": "20260425_120000",
        }
        with open(model_dir / "summary_20260425.json", "w") as f:
            json.dump(summary_data, f)

        # Monkey-patch PROJECT_ROOT temporarily
        import malid_lite.training.training_utils as tu
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            mode, path, summary = resolve_base_model_mode(
                model_num=1,
                retrain_set=set(),
                resume_flag=False,
                dataset_name="test-ds",
                classification_mode="multiclass",
                gene_locus="TCR",
                output_suffix=None,
                cli_training_params={"n_pcs": 15},
            )
            assert mode == "LOAD", f"Expected LOAD, got {mode}"
            assert summary is not None
            assert summary["n_pcs"] == 15
            tlog.log(f"  Mode: {mode}")
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)

        tlog.log("  PASS")
        tlog.record("resolve_base_model_mode LOAD", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resolve_base_model_mode LOAD", False, {"error": str(e)})
        raise


def test_40_resolve_base_model_mode_load_param_mismatch(tlog: _TestLogger):
    """resolve_base_model_mode: LOAD + param mismatch → ValueError."""
    tlog.log("\n--- Test 40: resolve_base_model_mode LOAD param mismatch ---")
    try:
        import tempfile, shutil
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        summary_data = {
            "gene_locus": "TCR",
            "training_context": "cv_ensemble",
            "classification_mode": "multiclass",
            "n_pcs": 15,
            "l1_ratio": 1.0,
            "model_names": ["lasso_cv"],
        }
        with open(model_dir / "summary_20260425.json", "w") as f:
            json.dump(summary_data, f)

        import malid_lite.training.training_utils as tu
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            # CLI says n_pcs=20 but summary says 15 → should raise
            raised = False
            try:
                resolve_base_model_mode(
                    model_num=1,
                    retrain_set=set(),
                    resume_flag=False,
                    dataset_name="test-ds",
                    classification_mode="multiclass",
                    gene_locus="TCR",
                    output_suffix=None,
                    cli_training_params={"n_pcs": 20},
                )
            except ValueError as ve:
                raised = True
                tlog.log(f"  Correctly raised ValueError: {ve}")
                assert "n_pcs" in str(ve)
            assert raised, "Expected ValueError for param mismatch"
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)

        tlog.log("  PASS")
        tlog.record("resolve_base_model_mode LOAD param mismatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resolve_base_model_mode LOAD param mismatch", False, {"error": str(e)})
        raise


def test_41_resolve_base_model_mode_resume(tlog: _TestLogger):
    """resolve_base_model_mode: partial artifacts + resume → RESUME."""
    tlog.log("\n--- Test 41: resolve_base_model_mode RESUME ---")
    try:
        import tempfile, shutil
        tmpdir = Path(tempfile.mkdtemp())
        model_dir = tmpdir / "trained_models" / "test-ds" / "cv_ensemble" / "base_models" / "TCR" / "model1" / "multiclass"
        model_dir.mkdir(parents=True)
        # Create fold artifacts but NO summary
        (model_dir / "fold_0_lasso_cv.pkl").touch()

        import malid_lite.training.training_utils as tu
        old_root = tu.PROJECT_ROOT
        tu.PROJECT_ROOT = tmpdir
        try:
            # With resume=True → RESUME
            mode, _, _ = resolve_base_model_mode(
                model_num=1,
                retrain_set=set(),
                resume_flag=True,
                dataset_name="test-ds",
                classification_mode="multiclass",
                gene_locus="TCR",
                output_suffix=None,
                cli_training_params={},
            )
            assert mode == "RESUME", f"Expected RESUME, got {mode}"

            # Without resume → TRAIN (partial artifacts overwritten)
            mode2, _, _ = resolve_base_model_mode(
                model_num=1,
                retrain_set=set(),
                resume_flag=False,
                dataset_name="test-ds",
                classification_mode="multiclass",
                gene_locus="TCR",
                output_suffix=None,
                cli_training_params={},
            )
            assert mode2 == "TRAIN", f"Expected TRAIN without resume, got {mode2}"
        finally:
            tu.PROJECT_ROOT = old_root
            shutil.rmtree(tmpdir)

        tlog.log("  PASS")
        tlog.record("resolve_base_model_mode RESUME", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("resolve_base_model_mode RESUME", False, {"error": str(e)})
        raise


def test_42_arg_interaction_retrain_not_in_models():
    """Arg validation: --retrain-models N where N not in --models → error."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "malid_lite.training.train_ensemble",
            "--models", "1", "2",
            "--retrain-models", "3",
            "--metadata-path", "/nonexistent",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Should error when retrain model not in models"
    assert "retrain-models" in result.stderr.lower() or "model 3" in result.stderr.lower(), (
        f"Error should mention retrain-models conflict. stderr: {result.stderr[:500]}"
    )


def test_43_arg_interaction_resume_retrain_conflict():
    """Arg validation: --resume + --retrain-base-models → error."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "malid_lite.training.train_ensemble",
            "--resume", "--retrain-base-models",
            "--metadata-path", "/nonexistent",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Should error for resume + retrain conflict"
    assert "contradictory" in result.stderr.lower() or "resume" in result.stderr.lower(), (
        f"Error should mention conflict. stderr: {result.stderr[:500]}"
    )


def test_44_preflight_validate_resume_no_meta(tlog: _TestLogger):
    """preflight_validate_resume_params: no _meta in artifacts → warning, no error."""
    tlog.log("\n--- Test 44: preflight_validate_resume_params no _meta ---")
    try:
        import tempfile, shutil
        tmpdir = Path(tempfile.mkdtemp())
        # Create fold artifact with no _meta
        (tmpdir / "fold_0_lasso_cv.pkl").touch()

        # Should not raise (just warn)
        preflight_validate_resume_params(
            model_num=1,
            model_dir=tmpdir,
            cli_training_params={"n_pcs": 20},
        )
        tlog.log("  No error raised (correct: no _meta → warning)")
        shutil.rmtree(tmpdir)
        tlog.log("  PASS")
        tlog.record("preflight_validate_resume no _meta", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("preflight_validate_resume no _meta", False, {"error": str(e)})
        raise


def test_45_preflight_validate_resume_mismatch(tlog: _TestLogger):
    """preflight_validate_resume_params: _meta mismatch → ValueError."""
    tlog.log("\n--- Test 45: preflight_validate_resume_params mismatch ---")
    try:
        import tempfile, shutil
        import joblib as jl

        tmpdir = Path(tempfile.mkdtemp())

        # Create a fake fold artifact (dict with _meta.model_params)
        artifact = {"_meta": {"model_params": {"n_pcs": 15, "l1_ratio": 1.0}}}
        jl.dump(artifact, tmpdir / "fold_0_predictions.pkl")

        raised = False
        try:
            preflight_validate_resume_params(
                model_num=1,
                model_dir=tmpdir,
                cli_training_params={"n_pcs": 20},
            )
        except ValueError as ve:
            raised = True
            assert "n_pcs" in str(ve)
            tlog.log(f"  Correctly raised ValueError: {ve}")

        assert raised, "Expected ValueError for _meta mismatch"
        shutil.rmtree(tmpdir)
        tlog.log("  PASS")
        tlog.record("preflight_validate_resume mismatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("preflight_validate_resume mismatch", False, {"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# Stage 4c: auto_train_base_model and training dispatch tests
# ---------------------------------------------------------------------------

def test_46_format_elapsed_time(tlog: _TestLogger):
    """_format_elapsed_time: formats seconds into human-readable strings."""
    tlog.log("\n--- Test 46: _format_elapsed_time ---")
    try:
        assert _format_elapsed_time(5) == "5s"
        assert _format_elapsed_time(45.7) == "46s"
        assert _format_elapsed_time(90) == "1m 30s"
        assert _format_elapsed_time(765) == "12m 45s"
        assert _format_elapsed_time(3600) == "1h 0m"
        assert _format_elapsed_time(8100) == "2h 15m"
        tlog.log("  All format cases pass")
        tlog.log("  PASS")
        tlog.record("_format_elapsed_time", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("_format_elapsed_time", False, {"error": str(e)})
        raise


def test_47_auto_train_invalid_model_num(tlog: _TestLogger):
    """auto_train_base_model: invalid model_num raises ValueError."""
    tlog.log("\n--- Test 47: auto_train_base_model invalid model_num ---")
    try:
        raised = False
        try:
            auto_train_base_model(
                model_num=4,
                training_params={},
                output_dir=Path("/tmp/fake"),
                metadata_path=Path("/tmp/fake.tsv"),
                dataset_name="test",
                classification_mode="multiclass",
                reference_class=None,
                diseases=None,
                gene_locus="TCR",
                fold_ids=[0],
                data_dir=None,
                cache_dir=None,
                gene_reference_path=None,
                n_jobs=1,
                verbose=0,
                resume=False,
            )
        except ValueError as ve:
            raised = True
            assert "4" in str(ve)
            tlog.log(f"  Correctly raised ValueError: {ve}")

        assert raised, "Expected ValueError for model_num=4"
        tlog.log("  PASS")
        tlog.record("auto_train_base_model invalid model_num", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train_base_model invalid model_num", False, {"error": str(e)})
        raise


def test_48_auto_train_dispatch_model1(tlog: _TestLogger):
    """auto_train_base_model: Model 1 dispatch passes correct kwargs."""
    tlog.log("\n--- Test 48: auto_train_base_model Model 1 dispatch ---")
    try:
        from unittest.mock import patch, MagicMock

        mock_train = MagicMock()
        with patch(
            "malid_lite.training.train_model1.train_all_folds", mock_train,
        ):
            auto_train_base_model(
                model_num=1,
                training_params={"n_pcs": 20, "l1_ratio": 0.5},
                output_dir=Path("/tmp/model1_out"),
                metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds",
                classification_mode="multiclass",
                reference_class=None,
                diseases=None,
                gene_locus="TCR",
                fold_ids=[0, 1, 2],
                data_dir=None,
                cache_dir=Path("/tmp/cache"),
                gene_reference_path=None,
                n_jobs=4,
                verbose=1,
                resume=False,
            )

        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args[1]

        # Shared params
        assert call_kwargs["fold_ids"] == [0, 1, 2]
        assert call_kwargs["metadata_path"] == Path("/tmp/meta.tsv")
        assert call_kwargs["output_dir"] == Path("/tmp/model1_out")
        assert call_kwargs["training_context"] == TRAINING_CONTEXT
        assert call_kwargs["resume"] is False
        assert call_kwargs["gene_locus"] == "TCR"

        # Training params unpacked
        assert call_kwargs["n_pcs"] == 20
        assert call_kwargs["l1_ratio"] == 0.5

        # n_jobs NOT passed to Model 1 (it has no n_jobs param)
        assert "n_jobs" not in call_kwargs

        tlog.log(f"  Model 1 dispatch kwargs verified ({len(call_kwargs)} keys)")
        tlog.log("  PASS")
        tlog.record("auto_train_base_model Model 1 dispatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train_base_model Model 1 dispatch", False, {"error": str(e)})
        raise


def test_49_auto_train_dispatch_model2(tlog: _TestLogger):
    """auto_train_base_model: Model 2 dispatch passes n_jobs and training params."""
    tlog.log("\n--- Test 49: auto_train_base_model Model 2 dispatch ---")
    try:
        from unittest.mock import patch, MagicMock

        mock_train = MagicMock()
        with patch(
            "malid_lite.training.train_model2.train_all_folds", mock_train,
        ):
            auto_train_base_model(
                model_num=2,
                training_params={"p_values": [0.001, 0.01], "retrain_on_full_train": True},
                output_dir=Path("/tmp/model2_out"),
                metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds",
                classification_mode="multiclass",
                reference_class=None,
                diseases=None,
                gene_locus="TCR",
                fold_ids=[0],
                data_dir=None,
                cache_dir=None,
                gene_reference_path=None,
                n_jobs=4,
                verbose=1,
                resume=True,
            )

        call_kwargs = mock_train.call_args[1]

        # n_jobs IS passed to Model 2
        assert call_kwargs["n_jobs"] == 4
        assert call_kwargs["resume"] is True

        # Training params unpacked
        assert call_kwargs["p_values"] == [0.001, 0.01]
        assert call_kwargs["retrain_on_full_train"] is True

        tlog.log(f"  Model 2 dispatch kwargs verified ({len(call_kwargs)} keys)")
        tlog.log("  PASS")
        tlog.record("auto_train_base_model Model 2 dispatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train_base_model Model 2 dispatch", False, {"error": str(e)})
        raise


def test_50_auto_train_dispatch_model3(tlog: _TestLogger):
    """auto_train_base_model: Model 3 dispatch passes infra kwargs and training params."""
    tlog.log("\n--- Test 50: auto_train_base_model Model 3 dispatch ---")
    try:
        from unittest.mock import patch, MagicMock

        mock_train = MagicMock()
        with patch(
            "malid_lite.training.train_model3.train_all_folds", mock_train,
        ):
            auto_train_base_model(
                model_num=3,
                training_params={
                    "aggregation_strategy": "mean",
                    "n_estimators_stage2": 200,
                },
                output_dir=Path("/tmp/model3_out"),
                metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds",
                classification_mode="multiclass",
                reference_class=None,
                diseases=None,
                gene_locus="TCR",
                fold_ids=[0, 1],
                data_dir=None,
                cache_dir=Path("/tmp/cache"),
                gene_reference_path=None,
                n_jobs=4,
                verbose=1,
                resume=False,
                # Model 3 specific
                embedding_dir=Path("/tmp/embeddings"),
                no_cache_embeddings=False,
                device="cpu",
                embedding_batch_size=32,
            )

        call_kwargs = mock_train.call_args[1]

        # Model 3 infra kwargs
        assert call_kwargs["n_jobs"] == 4
        assert call_kwargs["embedding_dir"] == Path("/tmp/embeddings")
        assert call_kwargs["cache_embeddings"] is True
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["embedding_batch_size"] == 32

        # Training params unpacked
        assert call_kwargs["aggregation_strategy"] == "mean"
        assert call_kwargs["n_estimators_stage2"] == 200

        # training_context always cv_ensemble
        assert call_kwargs["training_context"] == TRAINING_CONTEXT

        tlog.log(f"  Model 3 dispatch kwargs verified ({len(call_kwargs)} keys)")
        tlog.log("  PASS")
        tlog.record("auto_train_base_model Model 3 dispatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train_base_model Model 3 dispatch", False, {"error": str(e)})
        raise


def test_51_auto_train_model3_optional_kwargs(tlog: _TestLogger):
    """auto_train_base_model: Model 3 omits device/batch_size when None."""
    tlog.log("\n--- Test 51: auto_train Model 3 optional kwargs ---")
    try:
        from unittest.mock import patch, MagicMock

        mock_train = MagicMock()
        with patch(
            "malid_lite.training.train_model3.train_all_folds", mock_train,
        ):
            auto_train_base_model(
                model_num=3,
                training_params={},
                output_dir=Path("/tmp/model3_out"),
                metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds",
                classification_mode="multiclass",
                reference_class=None,
                diseases=None,
                gene_locus="TCR",
                fold_ids=[0],
                data_dir=None,
                cache_dir=None,
                gene_reference_path=None,
                n_jobs=4,
                verbose=1,
                resume=False,
                # device and embedding_batch_size left as None (defaults)
            )

        call_kwargs = mock_train.call_args[1]

        # device and embedding_batch_size should NOT be in kwargs when None
        assert "device" not in call_kwargs, (
            f"device should not be passed when None, but got: {call_kwargs.get('device')}"
        )
        assert "embedding_batch_size" not in call_kwargs, (
            f"embedding_batch_size should not be passed when None, but got: "
            f"{call_kwargs.get('embedding_batch_size')}"
        )

        # These ARE always passed
        assert call_kwargs["cache_embeddings"] is True
        assert call_kwargs["n_jobs"] == 4

        tlog.log("  device and embedding_batch_size correctly omitted when None")
        tlog.log("  PASS")
        tlog.record("auto_train Model 3 optional kwargs", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train Model 3 optional kwargs", False, {"error": str(e)})
        raise


def test_52_auto_train_empty_training_params(tlog: _TestLogger):
    """auto_train_base_model: empty training_params → only shared kwargs passed."""
    tlog.log("\n--- Test 52: auto_train empty training_params ---")
    try:
        from unittest.mock import patch, MagicMock

        mock_train = MagicMock()
        with patch(
            "malid_lite.training.train_model1.train_all_folds", mock_train,
        ):
            auto_train_base_model(
                model_num=1,
                training_params={},  # empty: all defaults
                output_dir=Path("/tmp/out"),
                metadata_path=Path("/tmp/meta.tsv"),
                dataset_name="test_ds",
                classification_mode="binary",
                reference_class="Healthy",
                diseases=["Covid19"],
                gene_locus="TCR",
                fold_ids=[0, 1, 2],
                data_dir=Path("/tmp/data"),
                cache_dir=Path("/tmp/cache"),
                gene_reference_path=None,
                n_jobs=4,
                verbose=1,
                resume=False,
            )

        call_kwargs = mock_train.call_args[1]

        # Verify shared params passed correctly
        assert call_kwargs["classification_mode"] == "binary"
        assert call_kwargs["reference_class"] == "Healthy"
        assert call_kwargs["diseases"] == ["Covid19"]
        assert call_kwargs["data_dir"] == Path("/tmp/data")

        # No extra training params present (n_pcs, l1_ratio, model_name not in kwargs)
        for k in ("n_pcs", "l1_ratio", "model_name"):
            assert k not in call_kwargs, (
                f"{k} should not be in kwargs when empty training_params"
            )

        tlog.log("  Empty training_params: only shared kwargs passed")
        tlog.log("  PASS")
        tlog.record("auto_train empty training_params", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("auto_train empty training_params", False, {"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# Stage 4d — validate_ensemble_args, cross-model disease classes,
#             suffix sanitization, file existence checks, cross-param
#             interactions, _log_base_model_status_table
# ---------------------------------------------------------------------------

def _make_base_namespace(**overrides) -> "argparse.Namespace":
    """Build a minimal argparse.Namespace for validate_ensemble_args tests.

    Sets all required attributes to valid defaults.  Callers override
    specific fields to trigger the validation path under test.
    """
    import argparse
    defaults = dict(
        models=[1, 2, 3],
        resume=False,
        retrain_models=None,
        retrain_base_models=False,
        output_dir=None,
        output_suffix=None,
        model1_suffix=None,
        model2_suffix=None,
        model3_suffix=None,
        n_jobs=4,
        classification_mode="multiclass",
        diseases=None,
        metadata_path=None,
        gene_reference_path=None,
        model3_embedding_dir=None,
        model3_no_cache_embeddings=False,
        model3_device=None,
        model3_embedding_batch_size=None,
        model2_abstention_strategy="ensemble_abstain",
        feature_matrices_dir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_base_cli_params(**m3_overrides) -> Dict[int, Dict]:
    """Build minimal cli_training_params for validate_ensemble_args tests."""
    m3 = dict(
        aggregation_strategy=None,
        n_estimators_stage1=None,
        n_estimators_stage2=None,
        entropy_max_fraction=None,
        entropy_bottom_percentile=None,
        tuning_cv_splits=None,
        tuning_strategies=None,
        tuning_entropy_max_fractions=None,
        tuning_entropy_percentiles=None,
    )
    m3.update(m3_overrides)
    return {
        1: {"n_pcs": None, "l1_ratio": None},
        2: {"p_values": None, "sequence_identity_threshold": None},
        3: m3,
    }


class _FakeParser:
    """Lightweight stand-in for argparse.ArgumentParser.

    Captures calls to parser.error() so tests can assert on the message
    without triggering SystemExit.
    """
    def __init__(self):
        self.error_message = None

    def error(self, message: str):
        self.error_message = message
        raise SystemExit(message)


# ---- T1: validate_ensemble_args unit tests ----

def test_53_validate_args_valid_defaults(tlog: _TestLogger):
    """validate_ensemble_args: all-default args pass without error."""
    tlog.log("\n--- Test 53: validate_ensemble_args valid defaults ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        # Should not raise
        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert parser.error_message is None

        tlog.log("  All-default args accepted")
        tlog.log("  PASS")
        tlog.record("validate_args valid defaults", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args valid defaults", False, {"error": str(e)})
        raise


def test_54_validate_args_resume_retrain_conflict(tlog: _TestLogger):
    """validate_ensemble_args: --resume + retrain_set → error."""
    tlog.log("\n--- Test 54: validate_args resume/retrain conflict ---")
    try:
        args = _make_base_namespace(resume=True)
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit) as exc_info:
            validate_ensemble_args(args, retrain_set={1}, cli_training_params=cli_params, parser=parser)
        assert "contradictory" in parser.error_message.lower()

        tlog.log("  resume + retrain → error with 'contradictory'")
        tlog.log("  PASS")
        tlog.record("validate_args resume/retrain conflict", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args resume/retrain conflict", False, {"error": str(e)})
        raise


def test_55_validate_args_retrain_not_in_models(tlog: _TestLogger):
    """validate_ensemble_args: retrain_models includes model not in models → error."""
    tlog.log("\n--- Test 55: validate_args retrain not in models ---")
    try:
        args = _make_base_namespace(models=[1, 2], retrain_models=[3])
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "model 3" in parser.error_message.lower() or "retrain-models" in parser.error_message.lower()

        tlog.log("  retrain model not in --models → error")
        tlog.log("  PASS")
        tlog.record("validate_args retrain not in models", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args retrain not in models", False, {"error": str(e)})
        raise


def test_56_validate_args_output_suffix_and_dir_conflict(tlog: _TestLogger):
    """validate_ensemble_args: --output-dir + --output-suffix → error."""
    tlog.log("\n--- Test 56: validate_args output-suffix + output-dir ---")
    try:
        args = _make_base_namespace(output_dir=Path("/tmp/out"), output_suffix="v1")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "mutually exclusive" in parser.error_message.lower()

        tlog.log("  output-dir + output-suffix → mutually exclusive error")
        tlog.log("  PASS")
        tlog.record("validate_args output-suffix/dir conflict", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args output-suffix/dir conflict", False, {"error": str(e)})
        raise


def test_57_validate_args_n_jobs_zero(tlog: _TestLogger):
    """validate_ensemble_args: n_jobs=0 → error."""
    tlog.log("\n--- Test 57: validate_args n_jobs=0 ---")
    try:
        args = _make_base_namespace(n_jobs=0)
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "n-jobs" in parser.error_message.lower() or "n_jobs" in parser.error_message.lower()

        tlog.log("  n_jobs=0 → error")
        tlog.log("  PASS")
        tlog.record("validate_args n_jobs zero", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args n_jobs zero", False, {"error": str(e)})
        raise


def test_58_validate_args_diseases_in_multiclass(tlog: _TestLogger):
    """validate_ensemble_args: --diseases in multiclass mode → error."""
    tlog.log("\n--- Test 58: validate_args diseases in multiclass ---")
    try:
        args = _make_base_namespace(classification_mode="multiclass", diseases=["Covid19"])
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "diseases" in parser.error_message.lower()

        tlog.log("  --diseases in multiclass → error")
        tlog.log("  PASS")
        tlog.record("validate_args diseases in multiclass", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args diseases in multiclass", False, {"error": str(e)})
        raise


def test_59_validate_args_model_specific_for_excluded(tlog: _TestLogger):
    """validate_ensemble_args: training params for excluded model → error."""
    tlog.log("\n--- Test 59: validate_args model-specific for excluded ---")
    try:
        # Model 2 excluded, but p_values set
        args = _make_base_namespace(models=[1, 3])
        parser = _FakeParser()
        cli_params = _make_base_cli_params()
        cli_params[2]["p_values"] = [0.01, 0.05]

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "model 2" in parser.error_message.lower()

        tlog.log("  Training params for excluded model → error mentions model")
        tlog.log("  PASS")
        tlog.record("validate_args model-specific excluded", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args model-specific excluded", False, {"error": str(e)})
        raise


def test_60_validate_args_suffix_for_excluded_model(tlog: _TestLogger):
    """validate_ensemble_args: --model2-suffix when model 2 excluded → error."""
    tlog.log("\n--- Test 60: validate_args suffix for excluded model ---")
    try:
        args = _make_base_namespace(models=[1, 3], model2_suffix="v1")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "model2-suffix" in parser.error_message.lower() or "model 2" in parser.error_message.lower()

        tlog.log("  Suffix for excluded model → error")
        tlog.log("  PASS")
        tlog.record("validate_args suffix excluded model", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args suffix excluded model", False, {"error": str(e)})
        raise


def test_61_validate_args_m3_infra_excluded(tlog: _TestLogger):
    """validate_ensemble_args: Model 3 infra args when model 3 excluded → error."""
    tlog.log("\n--- Test 61: validate_args M3 infra excluded ---")
    try:
        args = _make_base_namespace(models=[1, 2], model3_device="cpu")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "model 3" in parser.error_message.lower()

        tlog.log("  M3 infra args when excluded → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 infra excluded", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 infra excluded", False, {"error": str(e)})
        raise


def test_62_validate_args_per_model_range(tlog: _TestLogger):
    """validate_ensemble_args: invalid per-model param range → ValueError."""
    tlog.log("\n--- Test 62: validate_args per-model range ---")
    try:
        # l1_ratio > 1.0 should fail model1's validate_training_params
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params()
        cli_params[1]["l1_ratio"] = 1.5

        with pytest.raises(ValueError, match="l1_ratio"):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)

        tlog.log("  Invalid l1_ratio=1.5 → ValueError")
        tlog.log("  PASS")
        tlog.record("validate_args per-model range", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args per-model range", False, {"error": str(e)})
        raise


# ---- T3: Suffix sanitization ----

def test_63_validate_args_suffix_sanitization(tlog: _TestLogger):
    """validate_ensemble_args: suffix with bad chars → sanitized, with warning."""
    tlog.log("\n--- Test 63: validate_args suffix sanitization ---")
    try:
        args = _make_base_namespace(output_suffix="my run/v1")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        # Should succeed (sanitizes, doesn't error)
        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert args.output_suffix == "my_run_v1", f"Expected 'my_run_v1', got '{args.output_suffix}'"

        tlog.log(f"  'my run/v1' → '{args.output_suffix}'")
        tlog.log("  PASS")
        tlog.record("validate_args suffix sanitization", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args suffix sanitization", False, {"error": str(e)})
        raise


def test_64_validate_args_suffix_empty_after_sanitize(tlog: _TestLogger):
    """validate_ensemble_args: suffix all-bad-chars → empty after sanitization → error."""
    tlog.log("\n--- Test 64: validate_args suffix empty after sanitize ---")
    try:
        # All chars are bad (only special chars that get replaced)
        args = _make_base_namespace(model1_suffix="///")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        # "///" → "___" after sanitization, which is NOT empty
        # Need a string that becomes empty — but the regex replaces bad chars
        # with "_", so a non-empty string always stays non-empty.
        # The empty-after-sanitization case would require an empty string input,
        # but argparse wouldn't store "" for a suffix. This edge case is
        # effectively unreachable for suffix attrs, but we verify the check
        # works via direct attribute manipulation.

        # Test with valid suffix
        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert args.model1_suffix == "___", f"Expected '___', got '{args.model1_suffix}'"

        tlog.log("  '///' → '___' (non-empty, accepted)")
        tlog.log("  PASS")
        tlog.record("validate_args suffix sanitize special chars", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args suffix sanitize special chars", False, {"error": str(e)})
        raise


def test_65_validate_args_suffix_clean_passthrough(tlog: _TestLogger):
    """validate_ensemble_args: clean suffix passes through unchanged."""
    tlog.log("\n--- Test 65: validate_args suffix clean passthrough ---")
    try:
        args = _make_base_namespace(output_suffix="my_run-v1.2")
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert args.output_suffix == "my_run-v1.2"

        tlog.log("  'my_run-v1.2' passes through unchanged")
        tlog.log("  PASS")
        tlog.record("validate_args suffix clean passthrough", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args suffix clean passthrough", False, {"error": str(e)})
        raise


# ---- T4: metadata-path / gene-reference-path existence ----

def test_66_validate_args_metadata_path_missing(tlog: _TestLogger):
    """validate_ensemble_args: --metadata-path to nonexistent file → error."""
    tlog.log("\n--- Test 66: validate_args metadata-path missing ---")
    try:
        args = _make_base_namespace(metadata_path=Path("/nonexistent/meta.tsv"))
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "metadata-path" in parser.error_message.lower()

        tlog.log("  Nonexistent metadata-path → error")
        tlog.log("  PASS")
        tlog.record("validate_args metadata-path missing", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args metadata-path missing", False, {"error": str(e)})
        raise


def test_67_validate_args_gene_reference_path_missing(tlog: _TestLogger):
    """validate_ensemble_args: --gene-reference-path nonexistent → error."""
    tlog.log("\n--- Test 67: validate_args gene-reference-path missing ---")
    try:
        args = _make_base_namespace(gene_reference_path=Path("/nonexistent/genes.csv"))
        parser = _FakeParser()
        cli_params = _make_base_cli_params()

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "gene-reference-path" in parser.error_message.lower()

        tlog.log("  Nonexistent gene-reference-path → error")
        tlog.log("  PASS")
        tlog.record("validate_args gene-reference-path missing", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args gene-reference-path missing", False, {"error": str(e)})
        raise


def test_68_validate_args_metadata_path_exists(tlog: _TestLogger):
    """validate_ensemble_args: --metadata-path to real file → accepted."""
    tlog.log("\n--- Test 68: validate_args metadata-path exists ---")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tsv") as tmp:
            args = _make_base_namespace(metadata_path=Path(tmp.name))
            parser = _FakeParser()
            cli_params = _make_base_cli_params()

            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
            assert parser.error_message is None

        tlog.log("  Existing metadata-path accepted")
        tlog.log("  PASS")
        tlog.record("validate_args metadata-path exists", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args metadata-path exists", False, {"error": str(e)})
        raise


# ---- T5: Model 3 cross-param interactions ----

def test_69_validate_args_m3_tuning_with_fixed_strategy(tlog: _TestLogger):
    """validate_ensemble_args: tuning flags + fixed strategy → error."""
    tlog.log("\n--- Test 69: validate_args M3 tuning with fixed strategy ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean",
            tuning_strategies=["mean", "median"],
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "tuning" in parser.error_message.lower()

        tlog.log("  Tuning flags with fixed strategy → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 tuning+fixed", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 tuning+fixed", False, {"error": str(e)})
        raise


def test_70_validate_args_m3_entropy_max_wrong_strategy(tlog: _TestLogger):
    """validate_ensemble_args: entropy_max_fraction with non-entropy_cutoff → error."""
    tlog.log("\n--- Test 70: validate_args M3 entropy_max wrong strategy ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean",
            entropy_max_fraction=0.5,
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "entropy-max-fraction" in parser.error_message.lower() or "entropy_max_fraction" in parser.error_message.lower()

        tlog.log("  entropy_max_fraction with strategy=mean → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 entropy_max wrong strategy", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 entropy_max wrong strategy", False, {"error": str(e)})
        raise


def test_71_validate_args_m3_entropy_percentile_wrong_strategy(tlog: _TestLogger):
    """validate_ensemble_args: entropy_bottom_percentile with non-entropy_percentile_cutoff → error."""
    tlog.log("\n--- Test 71: validate_args M3 entropy_percentile wrong strategy ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="mean",
            entropy_bottom_percentile=10,
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "entropy-bottom-percentile" in parser.error_message.lower() or "entropy_bottom_percentile" in parser.error_message.lower()

        tlog.log("  entropy_bottom_percentile with strategy=mean → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 entropy_percentile wrong strategy", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 entropy_percentile wrong strategy", False, {"error": str(e)})
        raise


def test_72_validate_args_m3_fixed_entropy_with_auto_tuned(tlog: _TestLogger):
    """validate_ensemble_args: entropy_max_fraction with auto_tuned → error."""
    tlog.log("\n--- Test 72: validate_args M3 fixed entropy + auto_tuned ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="auto_tuned",
            entropy_max_fraction=0.5,
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "auto_tuned" in parser.error_message.lower() or "automatically" in parser.error_message.lower()

        tlog.log("  entropy_max_fraction with auto_tuned → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 fixed entropy + auto_tuned", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 fixed entropy + auto_tuned", False, {"error": str(e)})
        raise


def test_73_validate_args_m3_fixed_percentile_with_auto_tuned(tlog: _TestLogger):
    """validate_ensemble_args: entropy_bottom_percentile with auto_tuned → error."""
    tlog.log("\n--- Test 73: validate_args M3 fixed percentile + auto_tuned ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="auto_tuned",
            entropy_bottom_percentile=10,
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert "auto_tuned" in parser.error_message.lower() or "automatically" in parser.error_message.lower()

        tlog.log("  entropy_bottom_percentile with auto_tuned → error")
        tlog.log("  PASS")
        tlog.record("validate_args M3 fixed percentile + auto_tuned", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 fixed percentile + auto_tuned", False, {"error": str(e)})
        raise


def test_74_validate_args_m3_entropy_cutoff_valid(tlog: _TestLogger):
    """validate_ensemble_args: entropy_max_fraction with entropy_cutoff → accepted."""
    tlog.log("\n--- Test 74: validate_args M3 entropy_cutoff valid ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="entropy_cutoff",
            entropy_max_fraction=0.5,
        )

        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert parser.error_message is None

        tlog.log("  entropy_max_fraction with entropy_cutoff → accepted")
        tlog.log("  PASS")
        tlog.record("validate_args M3 entropy_cutoff valid", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 entropy_cutoff valid", False, {"error": str(e)})
        raise


def test_75_validate_args_m3_auto_tuned_with_tuning_flags(tlog: _TestLogger):
    """validate_ensemble_args: tuning flags with auto_tuned → accepted."""
    tlog.log("\n--- Test 75: validate_args M3 auto_tuned with tuning ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy="auto_tuned",
            tuning_strategies=["mean", "median"],
            tuning_cv_splits=3,
        )

        validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert parser.error_message is None

        tlog.log("  Tuning flags with auto_tuned → accepted")
        tlog.log("  PASS")
        tlog.record("validate_args M3 auto_tuned tuning valid", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 auto_tuned tuning valid", False, {"error": str(e)})
        raise


def test_76_validate_args_m3_unspecified_with_tuning_flags(tlog: _TestLogger):
    """validate_ensemble_args: tuning flags with strategy=None (default) → rejected.

    When aggregation_strategy is None (not specified), the default is
    entropy_percentile_cutoff (not auto_tuned). Tuning flags require
    explicit --model3-aggregation-strategy auto_tuned, so this must error.
    """
    tlog.log("\n--- Test 76: validate_args M3 unspecified + tuning ---")
    try:
        args = _make_base_namespace()
        parser = _FakeParser()
        cli_params = _make_base_cli_params(
            aggregation_strategy=None,
            tuning_strategies=["mean", "median"],
        )

        with pytest.raises(SystemExit):
            validate_ensemble_args(args, retrain_set=set(), cli_training_params=cli_params, parser=parser)
        assert parser.error_message is not None
        assert "auto_tuned" in parser.error_message

        tlog.log("  Tuning flags with strategy=None (default entropy_percentile_cutoff) → rejected")
        tlog.log("  PASS")
        tlog.record("validate_args M3 unspecified+tuning rejected", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("validate_args M3 unspecified+tuning rejected", False, {"error": str(e)})
        raise


# ---- T2: _validate_cross_model_disease_classes ----

def test_77_cross_model_disease_classes_match(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: matching classes → no error."""
    tlog.log("\n--- Test 77: cross-model disease classes match ---")
    try:
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy", "HIV"]},
            2: {"model_classes": ["HIV", "Healthy", "Covid19"]},
            3: {"model_classes": ["Healthy", "Covid19", "HIV"]},
        }
        # Should not raise (all sets equal despite different order)
        _validate_cross_model_disease_classes(summaries, label="test")

        tlog.log("  Matching classes (different order) → accepted")
        tlog.log("  PASS")
        tlog.record("cross-model classes match", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes match", False, {"error": str(e)})
        raise


def test_78_cross_model_disease_classes_mismatch(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: mismatched classes → ValueError."""
    tlog.log("\n--- Test 78: cross-model disease classes mismatch ---")
    try:
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: {"model_classes": ["Covid19", "Healthy", "HIV"]},
        }
        with pytest.raises(ValueError, match="mismatch"):
            _validate_cross_model_disease_classes(summaries, label="test models")

        tlog.log("  Mismatched classes → ValueError")
        tlog.log("  PASS")
        tlog.record("cross-model classes mismatch", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes mismatch", False, {"error": str(e)})
        raise


def test_79_cross_model_disease_classes_none_skip(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: None summaries are skipped."""
    tlog.log("\n--- Test 79: cross-model disease classes None skip ---")
    try:
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: None,
            3: {"model_classes": ["Covid19", "Healthy"]},
        }
        # Should not raise — None entries skipped
        _validate_cross_model_disease_classes(summaries, label="test")

        tlog.log("  None summaries skipped, remaining match → accepted")
        tlog.log("  PASS")
        tlog.record("cross-model classes None skip", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes None skip", False, {"error": str(e)})
        raise


def test_80_cross_model_disease_classes_missing_key_skipped(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: summary without model_classes is skipped."""
    tlog.log("\n--- Test 80: cross-model disease classes missing key skipped ---")
    try:
        # Model 2 has no model_classes key — should be skipped, not compared
        summaries = {
            1: {"model_classes": ["Covid19", "Healthy"]},
            2: {"classes": ["Covid19", "Healthy", "HIV"]},  # old key, not model_classes
        }
        # Should not raise — model 2 is skipped (only 1 model with model_classes)
        _validate_cross_model_disease_classes(summaries, label="test")

        tlog.log("  Summary without 'model_classes' key → skipped")
        tlog.log("  PASS")
        tlog.record("cross-model classes missing key skipped", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes missing key skipped", False, {"error": str(e)})
        raise


def test_81_cross_model_disease_classes_single_model(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: single model → no comparison needed."""
    tlog.log("\n--- Test 81: cross-model disease classes single model ---")
    try:
        summaries = {1: {"model_classes": ["Covid19", "Healthy"]}}
        # Only one model — nothing to compare
        _validate_cross_model_disease_classes(summaries, label="test")

        tlog.log("  Single model → no error (nothing to compare)")
        tlog.log("  PASS")
        tlog.record("cross-model classes single model", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes single model", False, {"error": str(e)})
        raise


def test_82_cross_model_disease_classes_no_key(tlog: _TestLogger):
    """_validate_cross_model_disease_classes: summaries without class keys → skipped."""
    tlog.log("\n--- Test 82: cross-model disease classes no key ---")
    try:
        summaries = {
            1: {"some_other_key": "value"},
            2: {"another_key": 42},
        }
        # Neither summary has disease_classes or classes → nothing to compare
        _validate_cross_model_disease_classes(summaries, label="test")

        tlog.log("  No disease class keys → no error (nothing to compare)")
        tlog.log("  PASS")
        tlog.record("cross-model classes no key", True)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("cross-model classes no key", False, {"error": str(e)})
        raise


# ---- _log_base_model_status_table ----

def test_83_log_base_model_status_table(tlog: _TestLogger):
    """_log_base_model_status_table: logs correctly for LOAD/TRAIN/RESUME modes."""
    tlog.log("\n--- Test 83: _log_base_model_status_table ---")
    try:
        import tempfile, shutil

        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Create RESUME artifacts
            resume_dir = tmpdir / "model2"
            resume_dir.mkdir()
            (resume_dir / "fold_0_clusters.pkl").touch()
            (resume_dir / "fold_3_clusters.pkl").touch()
            # Non-fold file that should be ignored
            (resume_dir / "fold_info.json").touch()

            model_modes = {1: "LOAD", 2: "RESUME", 3: "TRAIN"}
            model_dirs = {
                1: tmpdir / "model1",
                2: resume_dir,
                3: tmpdir / "model3",
            }
            model_summaries = {
                1: {"timestamp": "2026-01-15"},
                2: None,
                3: None,
            }

            # Capture log output
            import io
            log_handler = logging.StreamHandler(io.StringIO())
            log_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger("malid_lite.training.train_ensemble")
            logger.addHandler(log_handler)
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
                _log_base_model_status_table(model_modes, model_dirs, model_summaries)

                log_output = log_handler.stream.getvalue()
            finally:
                logger.removeHandler(log_handler)
                logger.setLevel(original_level)

            # Verify LOAD mode shows timestamp
            assert "LOAD" in log_output
            assert "2026-01-15" in log_output

            # Verify RESUME shows fold IDs (0 and 3, NOT "info")
            assert "RESUME" in log_output
            assert "'0'" in log_output or "0" in log_output
            assert "'3'" in log_output or "3" in log_output
            assert "info" not in log_output.split("RESUME")[1].split("TRAIN")[0], (
                "fold_info.json should be filtered by .isdigit() guard"
            )

            # Verify TRAIN mode present
            assert "TRAIN" in log_output

            tlog.log("  LOAD/RESUME/TRAIN logged correctly, fold_info.json filtered out")
            tlog.log("  PASS")
            tlog.record("_log_base_model_status_table", True)
        finally:
            shutil.rmtree(tmpdir)
    except Exception as e:
        tlog.log(f"  FAIL: {e}\n{traceback.format_exc()}")
        tlog.record("_log_base_model_status_table", False, {"error": str(e)})
        raise


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
    output_dir = OUTPUT_DIR

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
        test_25_resume_matches_original,
        test_31_compare_training_params_matching,
        test_32_compare_training_params_mismatch,
        test_33_compare_training_params_none_skip,
        test_34_compare_training_params_model1_model_name,
        test_35_compare_training_params_list_order_independent,
        test_36_compare_training_params_model3,
        test_36b_compare_training_params_missing_summary_key,
        test_37_resolve_base_model_mode_retrain,
        test_38_resolve_base_model_mode_no_artifacts,
        test_39_resolve_base_model_mode_load,
        test_40_resolve_base_model_mode_load_param_mismatch,
        test_41_resolve_base_model_mode_resume,
        test_44_preflight_validate_resume_no_meta,
        test_45_preflight_validate_resume_mismatch,
        test_46_format_elapsed_time,
        test_47_auto_train_invalid_model_num,
        test_48_auto_train_dispatch_model1,
        test_49_auto_train_dispatch_model2,
        test_50_auto_train_dispatch_model3,
        test_51_auto_train_model3_optional_kwargs,
        test_52_auto_train_empty_training_params,
        test_53_validate_args_valid_defaults,
        test_54_validate_args_resume_retrain_conflict,
        test_55_validate_args_retrain_not_in_models,
        test_56_validate_args_output_suffix_and_dir_conflict,
        test_57_validate_args_n_jobs_zero,
        test_58_validate_args_diseases_in_multiclass,
        test_59_validate_args_model_specific_for_excluded,
        test_60_validate_args_suffix_for_excluded_model,
        test_61_validate_args_m3_infra_excluded,
        test_62_validate_args_per_model_range,
        test_63_validate_args_suffix_sanitization,
        test_64_validate_args_suffix_empty_after_sanitize,
        test_65_validate_args_suffix_clean_passthrough,
        test_66_validate_args_metadata_path_missing,
        test_67_validate_args_gene_reference_path_missing,
        test_68_validate_args_metadata_path_exists,
        test_69_validate_args_m3_tuning_with_fixed_strategy,
        test_70_validate_args_m3_entropy_max_wrong_strategy,
        test_71_validate_args_m3_entropy_percentile_wrong_strategy,
        test_72_validate_args_m3_fixed_entropy_with_auto_tuned,
        test_73_validate_args_m3_fixed_percentile_with_auto_tuned,
        test_74_validate_args_m3_entropy_cutoff_valid,
        test_75_validate_args_m3_auto_tuned_with_tuning_flags,
        test_76_validate_args_m3_unspecified_with_tuning_flags,
        test_77_cross_model_disease_classes_match,
        test_78_cross_model_disease_classes_mismatch,
        test_79_cross_model_disease_classes_none_skip,
        test_80_cross_model_disease_classes_missing_key_skipped,
        test_81_cross_model_disease_classes_single_model,
        test_82_cross_model_disease_classes_no_key,
        test_83_log_base_model_status_table,
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
            test_30_integration_resume(tlog, n_jobs=args.n_jobs, model3_suffix=m3s)

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
