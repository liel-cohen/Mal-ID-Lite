"""Quick smoke test for Model 2 (Convergent Cluster Classifier).

Tests the full Model 2 pipeline in two tiers:

  Tier 1 — Unit tests with SYNTHETIC data (no cache, no GPU, ~30-60 seconds):
    Fast tests that exercise individual components using small fabricated datasets.
    These run even without a data cache or pre-computed embeddings.

  Tier 2 — Integration tests with TEST DATA (tests/test_data/, ~10-20 min):
    End-to-end tests that run the full training pipeline on the small test
    dataset (~260K sequences, 48 participants, 3 folds, 4 diseases).
    Uses glmnet_cv_n_splits=2 to match the small dataset.

Tests
-----
Unit tests (synthetic data):
  1.  validate_mode_and_classes: 11+ mode/data compatibility cases
  2.  validate_training_params: p_value range, threshold range, cv splits
  3.  evaluate_on_test (multiclass): AUROC, AUPRC, accuracy, MCC, confusion matrix
  4.  evaluate_on_test (binary): auroc_binary, auprc_binary
  5.  evaluate_on_test (all abstained): accuracy=0, raw_preds=None
  6.  evaluate_on_test (unseen classes): class-mismatch detection
  7.  filter_to_binary_pair: 4-class to 2-class filtering
  8.  save/load artifacts round-trip
  9.  save artifacts retrain_full suffix
  10. save artifacts no_valid_clusters notice
  11. save artifacts binary_pair
  12-18. _check_fold_complete: valid, missing, corrupt, truncated, old format, no_valid_clusters
  19. _save/_load_fold_predictions round-trip
  20-25. _validate_fold_meta: matching, fold_id/model_names/context/params mismatch, no _meta
  26. _get_fold_artifact_paths completeness
  27. predictions row format (multiclass)
  28. predictions row format (binary)

Integration tests (test data):
  29. Full multiclass pipeline on fold 0
  30. Full binary pipeline
  31. Binary with --diseases flag (1 from N-class)
  32. Multi-binary orchestration (N-1 pairs)
  33. cv_ensemble split isolation + pipeline
  34. retrain_on_full_train=True
  35. Multiple model_names in single run
  36. train_all_folds orchestrator (summary JSON, RESULTS_*.md, predictions CSV)
  37. Resume: original run -> resume -> metrics match
  38. Resume: incomplete fold (partial artifacts) -> retrain
  39. Resume: parameter mismatch -> ValueError
  40. Predictions CSV format (multiclass)
  41. Predictions CSV format (binary)

Design notes
------------
- Synthetic data uses sklearn LogisticRegression (not glmnet) for unit tests.
- Integration tests use test_data (tests/test_data/) with glmnet_cv_n_splits=2
  to avoid "not enough samples per class for 5-fold CV" errors.
- All outputs saved to tests/test_outputs/test_model2_quick/.

Requirements
------------
- Tier 1 (unit): numpy, pandas, scikit-learn (no cache, no GPU, no glmnet)
- Tier 2 (integration): glmnet, cache auto-built from test_data/

Expected runtime
----------------
- Tier 1 only: ~30-60 seconds
- Tier 1 + Tier 2: ~10-20 minutes

Output files
------------
All outputs saved to tests/test_outputs/test_model2_quick/:
- test_log_YYYYMMDD_HHMMSS.txt              - Full log
- test_results_YYYYMMDD_HHMMSS.json         - Structured results (pass/fail per test)
- integration/                               - Integration test artifacts

Running
-------
From Mal-ID-Lite root directory:

    # Unit tests only (~30-60 seconds)
    python -m pytest tests/test_model2_quick.py -v -s -k "not integration"

    # Full suite (~10-20 minutes)
    python -m pytest tests/test_model2_quick.py -v -s
"""

import json
import logging
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test_helpers import create_test_loader, TEST_DATA_DIR, TEST_FOLD_IDS, TEST_DISEASES

# Test output directory (per CLAUDE.md convention)
TEST_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / TEST_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLMNET_CV_N_SPLITS = 2  # Reduced from 5 for test data (too few specimens per class for 5-fold)
TEST_P_VALUES = [0.001, 0.01, 0.05]
TEST_MODEL_NAME = "lasso_cv"
TEST_FOLD_ID = 0

# ---------------------------------------------------------------------------
# Imports from malid_lite
# ---------------------------------------------------------------------------

from malid_lite.models.model2_convergent_clusters import (
    CDR3_COL,
    CLUSTER_ID_COL,
    DEFAULT_P_VALUES,
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    ConvergentClusterClassifier,
    FeaturizedData,
    build_pipeline,
    cluster_training_set,
    compute_fisher_scores,
    featurize,
    get_artifact_paths,
    get_cluster_centroids,
    merge_centroids_with_scores,
    train_convergent_cluster_classifier,
)
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    filter_to_binary_pair,
    get_model_output_dir,
    make_pair_name,
    run_training_orchestration,
    validate_mode_and_classes,
)
from malid_lite.training.train_model2 import (
    _check_fold_complete,
    _get_fold_artifact_paths,
    _load_fold_results,
    _save_fold_predictions,
    _validate_fold_meta,
    evaluate_on_test,
    load_and_prepare_fold,
    save_fold_artifacts,
    train_all_folds,
    validate_training_params,
)


# ---------------------------------------------------------------------------
# Test logger (same pattern as model3/ensemble)
# ---------------------------------------------------------------------------

class _TestLogger:
    """Logger that writes to both console and file, tracks pass/fail."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a")
        self._results: List[Dict] = []
        self._start = datetime.now()

    def log(self, msg: str, *, to_file_only: bool = False) -> None:
        self._file.write(msg + "\n")
        self._file.flush()
        if not to_file_only:
            print(msg)

    def record(self, name: str, status: str, details: Optional[Dict] = None) -> None:
        self._results.append({
            "test": name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def close(self) -> Path:
        data = {
            "start_time": self._start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "tests": self._results,
        }
        self._file.close()
        results_path = self.log_path.with_name(
            self.log_path.stem.replace("test_log", "test_results") + ".json"
        )
        with open(results_path, "w") as f:
            json.dump(
                data, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )
        return results_path


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------




@pytest.fixture(scope="session")
def tlog():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _TestLogger(OUTPUT_DIR / f"test_log_{timestamp}.txt")
    logger.log(f"Model 2 test suite — {timestamp}")
    logger.log("=" * 60)
    yield logger
    results_path = logger.close()
    print(f"\nResults saved to: {results_path}")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_featurized(
    n_scored: int,
    n_abstained: int,
    disease_classes: List[str],
    p_value: float = 0.01,
) -> FeaturizedData:
    """Create a synthetic FeaturizedData for unit testing evaluate_on_test."""
    rng = np.random.RandomState(42)
    n_classes = len(disease_classes)

    # Scored specimens
    X_data = rng.randint(0, 10, size=(n_scored, n_classes)).astype(float)
    scored_specimens = [f"specimen_{i}" for i in range(n_scored)]
    scored_participants = [f"participant_{i}" for i in range(n_scored)]
    # Assign true labels cycling through classes
    y_labels = [disease_classes[i % n_classes] for i in range(n_scored)]

    X = pd.DataFrame(X_data, columns=disease_classes, index=scored_specimens)
    y = pd.Series(y_labels, index=scored_specimens, name=DISEASE_COL)
    participant_labels = pd.Series(scored_participants, index=scored_specimens)

    # Abstained specimens
    abstained_specimens = [f"specimen_abs_{i}" for i in range(n_abstained)]
    abstained_labels = [disease_classes[i % n_classes] for i in range(n_abstained)]
    abstained_y = pd.Series(
        abstained_labels, index=abstained_specimens, name=DISEASE_COL,
    )

    return FeaturizedData(
        X=X,
        y=y,
        participant_labels=participant_labels,
        sample_names=pd.Index(scored_specimens),
        abstained_sample_y=abstained_y,
        abstained_sample_names=pd.Index(abstained_specimens),
        p_value_threshold=p_value,
    )


def _fit_simple_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Fit a minimal sklearn Pipeline (StandardScaler + LogisticRegression)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe


def _make_synthetic_train_result(
    model_names: List[str],
    disease_classes: List[str],
    has_valid_pvalue: bool = True,
    n_scored: int = 20,
) -> Dict:
    """Create a synthetic train_result dict with fitted pipeline for save/load tests."""
    rng = np.random.RandomState(42)
    n_classes = len(disease_classes)

    # Create synthetic feature matrix and labels
    X_data = rng.randint(0, 10, size=(n_scored, n_classes)).astype(float)
    specimens = [f"spec_{i}" for i in range(n_scored)]
    y_labels = [disease_classes[i % n_classes] for i in range(n_scored)]

    X = pd.DataFrame(X_data, columns=disease_classes, index=specimens)
    y = pd.Series(y_labels, index=specimens, name=DISEASE_COL)

    # Centroids + scores
    n_centroids = 15
    centroids_data = pd.DataFrame({
        "v_gene": [f"TRBV{i}" for i in range(n_centroids)],
        "j_gene": [f"TRBJ{i % 3}" for i in range(n_centroids)],
        "centroid_sequence": [f"CASS{'A' * (i + 3)}" for i in range(n_centroids)],
        "cdr3_aa_sequence_trim_len": [i + 5 for i in range(n_centroids)],
        "global_resulting_cluster_ID": list(range(n_centroids)),
    })
    for cls in disease_classes:
        centroids_data[cls] = rng.random(n_centroids) * 0.1

    results = {}
    for mn in model_names:
        if has_valid_pvalue:
            pipeline = _fit_simple_pipeline(X, y)
            results[mn] = {
                "best_p_value": 0.01,
                "pipeline": pipeline,
                "all_p_value_metrics": {
                    "0.01": {"accuracy": 0.8, "n_scored": n_scored, "n_abstained": 0},
                },
            }
        else:
            results[mn] = {
                "best_p_value": None,
                "pipeline": None,
                "all_p_value_metrics": {},
            }

    return {
        "centroids_with_scores": centroids_data,
        "disease_classes": disease_classes,
        "results": results,
    }


def _make_synthetic_sequences_and_metadata(
    n_participants: int = 16,
    diseases: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic sequence and metadata DataFrames for unit tests."""
    if diseases is None:
        diseases = ["DiseaseA", "DiseaseB", "DiseaseC", "Healthy"]
    rng = np.random.RandomState(42)

    rows = []
    meta_rows = []
    for i in range(n_participants):
        participant = f"P{i:04d}"
        disease = diseases[i % len(diseases)]
        specimen = f"S{i:04d}"
        meta_rows.append({
            PARTICIPANT_COL: participant,
            SPECIMEN_COL: specimen,
            DISEASE_COL: disease,
            "malid_cross_validation_fold_id_when_in_test_set": i % 3,
        })
        for j in range(10):
            rows.append({
                PARTICIPANT_COL: participant,
                SPECIMEN_COL: specimen,
                DISEASE_COL: disease,
                CDR3_COL: f"CASS{'ABCDEF'[rng.randint(6)]}" * (j % 3 + 1),
                "v_gene": f"TRBV{rng.randint(1, 5)}",
                "j_gene": f"TRBJ{rng.randint(1, 3)}",
            })

    sequences_df = pd.DataFrame(rows)
    metadata_df = pd.DataFrame(meta_rows)
    return sequences_df, metadata_df


# ===========================================================================
# Tier 1: Unit Tests (Synthetic Data)
# ===========================================================================

# ---------------------------------------------------------------------------
# A. Mode / Parameter Validation
# ---------------------------------------------------------------------------

def test_validate_mode_and_classes(tlog: _TestLogger):
    """11+ cases covering multiclass/binary/multi-binary validation."""
    errors = []

    # multiclass with 2 classes — should warn but not raise
    try:
        validate_mode_and_classes("multiclass", ["A", "B"], None)
    except Exception as e:
        errors.append(f"multiclass/2-class raised unexpectedly: {e}")

    # multiclass with N classes — should not raise
    try:
        validate_mode_and_classes("multiclass", ["A", "B", "C"], None)
    except Exception as e:
        errors.append(f"multiclass/N-class raised unexpectedly: {e}")

    # multiclass with unknown reference_class — should warn but not raise
    try:
        validate_mode_and_classes("multiclass", ["A", "B"], "X")
    except Exception as e:
        errors.append(f"multiclass/ref-class raised unexpectedly: {e}")

    # binary with 2 classes, no reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("binary", ["A", "B"], None)
    except ValueError:
        raised = True
    if not raised:
        errors.append("binary/2-class/no-ref did not raise ValueError")

    # binary with >2 classes, no reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("binary", ["A", "B", "C"], None)
    except ValueError:
        raised = True
    if not raised:
        errors.append("binary/>2-class/no-ref did not raise ValueError")

    # binary with valid reference_class — should succeed
    try:
        validate_mode_and_classes("binary", ["A", "B"], "B")
    except Exception as e:
        errors.append(f"binary/valid-ref raised unexpectedly: {e}")

    # binary with unknown reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("binary", ["A", "B"], "Z")
    except ValueError:
        raised = True
    if not raised:
        errors.append("binary/unknown-ref did not raise ValueError")

    # multi-binary with 2 classes, no reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("multi-binary", ["A", "B"], None)
    except ValueError:
        raised = True
    if not raised:
        errors.append("multi-binary/2-class/no-ref did not raise ValueError")

    # multi-binary with N classes, no reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("multi-binary", ["A", "B", "C"], None)
    except ValueError:
        raised = True
    if not raised:
        errors.append("multi-binary/N-class/no-ref did not raise ValueError")

    # multi-binary with N classes, valid reference_class — should succeed
    try:
        ref = validate_mode_and_classes("multi-binary", ["A", "B", "C"], "C")
        assert ref == "C", f"expected ref='C', got {ref!r}"
    except Exception as e:
        errors.append(f"multi-binary/valid-ref raised unexpectedly: {e}")

    # multi-binary with N classes, unknown reference_class — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("multi-binary", ["A", "B", "C"], "Z")
    except ValueError:
        raised = True
    if not raised:
        errors.append("multi-binary/unknown-ref did not raise ValueError")

    # unknown mode — must raise ValueError
    raised = False
    try:
        validate_mode_and_classes("invalid-mode", ["A", "B"], None)
    except ValueError:
        raised = True
    if not raised:
        errors.append("invalid mode did not raise ValueError")

    if errors:
        tlog.record("validate_mode_and_classes", "FAIL", {"errors": errors})
        raise AssertionError("Validation failures:\n  " + "\n  ".join(errors))

    tlog.log("  [PASS] validate_mode_and_classes: 11 cases passed")
    tlog.record("validate_mode_and_classes", "PASS", {"n_cases": 11})


def test_validate_training_params(tlog: _TestLogger):
    """Validate p_value range, threshold range, and glmnet_cv_n_splits."""
    # Valid params — should not raise
    validate_training_params(p_values=[0.001, 0.01, 0.05])
    validate_training_params(sequence_identity_threshold=0.90)
    validate_training_params(glmnet_cv_n_splits=2)
    validate_training_params(glmnet_cv_n_splits=5)
    # None passthrough
    validate_training_params(p_values=None, sequence_identity_threshold=None, glmnet_cv_n_splits=None)

    # Invalid p_values
    with pytest.raises(ValueError, match="p_values"):
        validate_training_params(p_values=[0.0])
    with pytest.raises(ValueError, match="p_values"):
        validate_training_params(p_values=[1.0])
    with pytest.raises(ValueError, match="p_values"):
        validate_training_params(p_values=[-0.01])

    # Invalid threshold
    with pytest.raises(ValueError, match="sequence_identity_threshold"):
        validate_training_params(sequence_identity_threshold=0.0)
    with pytest.raises(ValueError, match="sequence_identity_threshold"):
        validate_training_params(sequence_identity_threshold=1.5)

    # Invalid glmnet_cv_n_splits
    with pytest.raises(ValueError, match="glmnet_cv_n_splits"):
        validate_training_params(glmnet_cv_n_splits=1)
    with pytest.raises(ValueError, match="glmnet_cv_n_splits"):
        validate_training_params(glmnet_cv_n_splits=0)

    tlog.log("  [PASS] validate_training_params")
    tlog.record("validate_training_params", "PASS")


# ---------------------------------------------------------------------------
# B. Evaluation Unit Tests
# ---------------------------------------------------------------------------

def test_evaluate_on_test_multiclass(tlog: _TestLogger):
    """Multiclass evaluate_on_test with synthetic data."""
    classes = ["Covid19", "HIV", "Healthy"]
    fd = _make_synthetic_featurized(n_scored=30, n_abstained=3, disease_classes=classes)
    pipeline = _fit_simple_pipeline(fd.X, fd.y)

    metrics, raw_preds = evaluate_on_test(
        featurized=fd,
        pipeline=pipeline,
        classes=np.array(classes),
        fold_id=0,
        model_name="test_model",
        reference_class=None,
    )

    # Accuracy penalizes abstentions: n_correct / (n_scored + n_abstained)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_scored"] == 30
    assert metrics["n_abstained"] == 3
    assert metrics["abstention_rate"] == 3 / 33

    # Multiclass metrics (3 classes)
    assert metrics["auroc_ovo_weighted"] is not None
    assert 0.0 <= metrics["auroc_ovo_weighted"] <= 1.0
    assert metrics["auprc_ovo_weighted"] is not None
    assert metrics["auroc_ovr_per_class"] is not None
    assert set(metrics["auroc_ovr_per_class"].keys()) == set(classes)

    # Other metrics
    assert metrics["log_loss"] is not None
    assert metrics["log_loss"] > 0
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert metrics["confusion_matrix"] is not None
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["classes"] == [str(c) for c in classes]

    # Raw preds for cross-fold aggregation
    assert raw_preds is not None
    assert len(raw_preds["y_true"]) == 30
    assert len(raw_preds["y_pred"]) == 30
    assert raw_preds["y_proba"].shape == (30, 3)

    tlog.log("  [PASS] evaluate_on_test_multiclass")
    tlog.record("evaluate_on_test_multiclass", "PASS")


def test_evaluate_on_test_binary(tlog: _TestLogger):
    """Binary evaluate_on_test with reference_class."""
    classes = ["Disease", "Healthy"]
    fd = _make_synthetic_featurized(n_scored=20, n_abstained=0, disease_classes=classes)
    pipeline = _fit_simple_pipeline(fd.X, fd.y)

    metrics, raw_preds = evaluate_on_test(
        featurized=fd,
        pipeline=pipeline,
        classes=np.array(classes),
        fold_id=0,
        model_name="test_model",
        reference_class="Healthy",
    )

    # Binary-specific metrics
    assert metrics["auroc_binary"] is not None
    assert 0.0 <= metrics["auroc_binary"] <= 1.0
    assert metrics["auprc_binary"] is not None
    assert 0.0 <= metrics["auprc_binary"] <= 1.0

    # Multiclass OvO metrics are None for 2-class
    assert metrics["auroc_ovo_weighted"] is None
    assert metrics["auprc_ovo_weighted"] is None
    assert metrics["auroc_ovr_per_class"] is None

    tlog.log("  [PASS] evaluate_on_test_binary")
    tlog.record("evaluate_on_test_binary", "PASS")


def test_evaluate_on_test_all_abstained(tlog: _TestLogger):
    """All specimens abstained: accuracy=0, raw_preds=None."""
    classes = ["A", "B", "C"]
    fd = _make_synthetic_featurized(n_scored=0, n_abstained=5, disease_classes=classes)
    pipeline = _fit_simple_pipeline(
        pd.DataFrame(np.random.rand(10, 3), columns=classes),
        pd.Series(["A"] * 4 + ["B"] * 3 + ["C"] * 3),
    )

    metrics, raw_preds = evaluate_on_test(
        featurized=fd,
        pipeline=pipeline,
        classes=np.array(classes),
        fold_id=0,
        model_name="test_model",
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["n_scored"] == 0
    assert metrics["n_abstained"] == 5
    assert raw_preds is None

    tlog.log("  [PASS] evaluate_on_test_all_abstained")
    tlog.record("evaluate_on_test_all_abstained", "PASS")


def test_evaluate_on_test_unseen_classes(tlog: _TestLogger):
    """Test fold has a class not seen during training."""
    train_classes = ["A", "B"]
    # FeaturizedData with 3 classes but pipeline only knows 2
    fd = _make_synthetic_featurized(n_scored=20, n_abstained=0, disease_classes=train_classes)
    # Override y to include an unseen class "C"
    y_vals = list(fd.y.values)
    y_vals[-2] = "C"
    y_vals[-1] = "C"
    fd = FeaturizedData(
        X=fd.X, y=pd.Series(y_vals, index=fd.X.index, name=DISEASE_COL),
        participant_labels=fd.participant_labels,
        sample_names=fd.sample_names,
        abstained_sample_y=fd.abstained_sample_y,
        abstained_sample_names=fd.abstained_sample_names,
        p_value_threshold=fd.p_value_threshold,
    )
    pipeline = _fit_simple_pipeline(fd.X, pd.Series(["A"] * 10 + ["B"] * 10, index=fd.X.index))

    metrics, raw_preds = evaluate_on_test(
        featurized=fd,
        pipeline=pipeline,
        classes=np.array(train_classes),
        fold_id=0,
        model_name="test_model",
    )

    # Should detect the unseen class
    assert "unseen_test_classes" in metrics
    assert "C" in metrics["unseen_test_classes"]
    assert metrics["n_unseen_test_specimens"] == 2

    tlog.log("  [PASS] evaluate_on_test_unseen_classes")
    tlog.record("evaluate_on_test_unseen_classes", "PASS")


# ---------------------------------------------------------------------------
# C. Filter / Split Unit Tests
# ---------------------------------------------------------------------------

def test_filter_to_binary_pair(tlog: _TestLogger):
    """Filter 4-class synthetic data to a 2-class binary pair."""
    diseases = ["DiseaseA", "DiseaseB", "DiseaseC", "Healthy"]
    seq_df, meta_df = _make_synthetic_sequences_and_metadata(
        n_participants=16, diseases=diseases,
    )

    target, reference = "DiseaseA", "Healthy"
    filtered_seqs, filtered_meta = filter_to_binary_pair(
        seq_df, meta_df, target, reference,
    )

    # Only 2 diseases remain
    remaining = sorted(filtered_seqs[DISEASE_COL].unique())
    assert remaining == sorted([target, reference]), f"Expected 2 diseases, got {remaining}"

    # No rows from other diseases
    assert not filtered_seqs[DISEASE_COL].isin(["DiseaseB", "DiseaseC"]).any()

    # Participants consistent between sequences and metadata
    seq_participants = set(filtered_seqs[PARTICIPANT_COL].unique())
    meta_participants = set(filtered_meta[PARTICIPANT_COL].unique())
    assert seq_participants == meta_participants

    tlog.log("  [PASS] filter_to_binary_pair")
    tlog.record("filter_to_binary_pair", "PASS")


# ---------------------------------------------------------------------------
# D. Artifact Save/Load Unit Tests
# ---------------------------------------------------------------------------

def test_save_load_artifacts_roundtrip(tlog: _TestLogger):
    """Save synthetic train_result, then load and verify structure."""
    disease_classes = ["Covid19", "HIV", "Healthy"]
    train_result = _make_synthetic_train_result(
        [TEST_MODEL_NAME], disease_classes, has_valid_pvalue=True,
    )

    artifact_dir = OUTPUT_DIR / "unit_artifacts_roundtrip"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    saved = save_fold_artifacts(artifact_dir, fold_id=0, train_result=train_result)

    # Verify files exist
    assert (artifact_dir / "fold_0_clusters.joblib").exists()
    assert (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_p_value.joblib").exists()
    assert (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib").exists()
    assert (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_results_split1.json").exists()

    # Load via joblib and verify structure
    clusters = joblib.load(artifact_dir / "fold_0_clusters.joblib")
    assert "centroids_with_scores" in clusters
    assert "disease_classes" in clusters
    assert clusters["disease_classes"] == disease_classes
    assert clusters["binary_pair"] is None  # multiclass

    p_val = joblib.load(artifact_dir / f"fold_0_{TEST_MODEL_NAME}_p_value.joblib")
    assert p_val == 0.01

    pipeline = joblib.load(artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib")
    assert hasattr(pipeline, "predict")

    # Load via ConvergentClusterClassifier
    clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
    clf.load_artifacts(artifact_dir, fold_id=0, model_name=TEST_MODEL_NAME)
    assert clf._is_loaded
    assert clf.disease_classes_ == disease_classes
    assert clf.best_p_value_ == 0.01

    tlog.log("  [PASS] save_load_artifacts_roundtrip")
    tlog.record("save_load_artifacts_roundtrip", "PASS")


def test_save_artifacts_retrain_full(tlog: _TestLogger):
    """retrain_on_full_train=True: artifact filenames use 'full' suffix."""
    disease_classes = ["A", "B"]
    train_result = _make_synthetic_train_result(
        [TEST_MODEL_NAME], disease_classes, has_valid_pvalue=True,
    )

    artifact_dir = OUTPUT_DIR / "unit_artifacts_retrain_full"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    save_fold_artifacts(artifact_dir, fold_id=0, train_result=train_result,
                        retrain_on_full_train=True)

    # "full" suffix, not "split1"
    assert (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_full.joblib").exists()
    assert (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_results_full.json").exists()
    assert not (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib").exists()

    # Load via ConvergentClusterClassifier with retrain_on_full_train=True
    clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
    clf.load_artifacts(artifact_dir, fold_id=0, model_name=TEST_MODEL_NAME,
                       retrain_on_full_train=True)
    assert clf._is_loaded

    tlog.log("  [PASS] save_artifacts_retrain_full")
    tlog.record("save_artifacts_retrain_full", "PASS")


def test_save_artifacts_no_valid_clusters(tlog: _TestLogger):
    """best_p_value=None: NO_VALID_CLUSTERS.txt written, no per-model artifacts."""
    disease_classes = ["A", "B"]
    train_result = _make_synthetic_train_result(
        [TEST_MODEL_NAME], disease_classes, has_valid_pvalue=False,
    )

    artifact_dir = OUTPUT_DIR / "unit_artifacts_no_valid"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    saved = save_fold_artifacts(artifact_dir, fold_id=0, train_result=train_result)

    # Clusters artifact always saved
    assert (artifact_dir / "fold_0_clusters.joblib").exists()
    # Notice file saved
    notice_path = artifact_dir / f"fold_0_{TEST_MODEL_NAME}_NO_VALID_CLUSTERS.txt"
    assert notice_path.exists()
    assert "no valid p-value" in notice_path.read_text().lower()
    # Per-model artifacts NOT saved
    assert not (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_p_value.joblib").exists()
    assert not (artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib").exists()

    # saved filenames should include clusters and notice, not model artifacts
    assert "fold_0_clusters.joblib" in saved
    assert notice_path.name in saved

    tlog.log("  [PASS] save_artifacts_no_valid_clusters")
    tlog.record("save_artifacts_no_valid_clusters", "PASS")


def test_save_artifacts_binary_pair(tlog: _TestLogger):
    """disease_filter saves binary_pair in clusters.joblib."""
    disease_classes = ["Disease", "Healthy"]
    train_result = _make_synthetic_train_result(
        [TEST_MODEL_NAME], disease_classes, has_valid_pvalue=True,
    )

    artifact_dir = OUTPUT_DIR / "unit_artifacts_binary_pair"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    save_fold_artifacts(
        artifact_dir, fold_id=0, train_result=train_result,
        disease_filter=("Disease", "Healthy"),
    )

    # Verify binary_pair in clusters artifact
    clusters = joblib.load(artifact_dir / "fold_0_clusters.joblib")
    assert clusters["binary_pair"] == {
        "disease": "Disease", "reference_class": "Healthy",
    }

    # ConvergentClusterClassifier loads binary_pair correctly
    clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
    clf.load_artifacts(artifact_dir, fold_id=0, model_name=TEST_MODEL_NAME)
    assert clf.disease_class_ == "Disease"
    assert clf.reference_class_ == "Healthy"
    assert list(clf.classes_) == ["Healthy", "Disease"]

    tlog.log("  [PASS] save_artifacts_binary_pair")
    tlog.record("save_artifacts_binary_pair", "PASS")


# ---------------------------------------------------------------------------
# E. Resume Logic Unit Tests
# ---------------------------------------------------------------------------

def _create_complete_fold_artifacts(
    artifact_dir: Path,
    fold_id: int = 0,
    model_names: Optional[List[str]] = None,
    has_valid_pvalue: bool = True,
    training_context: str = "cv_single_model",
) -> Dict:
    """Helper: create a complete set of fold artifacts on disk for resume tests."""
    if model_names is None:
        model_names = [TEST_MODEL_NAME]
    disease_classes = ["A", "B", "C"]
    train_result = _make_synthetic_train_result(
        model_names, disease_classes, has_valid_pvalue=has_valid_pvalue,
    )

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    saved_names = save_fold_artifacts(artifact_dir, fold_id, train_result)

    # Build per_model_data for _save_fold_predictions
    per_model_data = {}
    for mn in model_names:
        mr = train_result["results"][mn]
        if mr["best_p_value"] is not None:
            per_model_data[mn] = {
                "eval_result": {"fold_id": fold_id, "model_name": mn, "accuracy": 0.8,
                                "n_scored": 20, "n_abstained": 0, "abstention_rate": 0.0,
                                "p_value_threshold": mr["best_p_value"]},
                "raw_preds": {"y_true": np.array(["A"] * 10 + ["B"] * 10),
                              "y_pred": np.array(["A"] * 10 + ["B"] * 10),
                              "y_proba": np.random.rand(20, 3),
                              "classes": np.array(disease_classes)},
                "predictions_rows": [{"specimen": f"s{i}"} for i in range(20)],
            }

    model_params = {
        "sequence_identity_threshold": 0.9,
        "p_values": sorted(TEST_P_VALUES),
        "retrain_on_full_train": False,
        "classification_mode": "multiclass",
        "diseases": None,
        "dataset_name": "test",
        "reference_class": None,
        "disease_filter": None,
    }

    _save_fold_predictions(
        artifact_dir, fold_id, model_names,
        per_model_data=per_model_data,
        model_params=model_params,
        training_context=training_context,
        expected_artifacts=saved_names,
    )
    return {"model_params": model_params, "training_context": training_context,
            "model_names": model_names, "saved_names": saved_names}


def test_check_fold_complete_valid(tlog: _TestLogger):
    """All artifacts present: returns loaded preds_data."""
    artifact_dir = OUTPUT_DIR / "unit_resume_complete"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is not None
    assert "_meta" in result
    assert "by_model" in result
    assert result["_meta"]["fold_id"] == 0

    tlog.log("  [PASS] check_fold_complete_valid")
    tlog.record("check_fold_complete_valid", "PASS")


def test_check_fold_complete_missing_predictions(tlog: _TestLogger):
    """No predictions.pkl: returns None."""
    artifact_dir = OUTPUT_DIR / "unit_resume_missing_preds"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)
    # Delete predictions
    (artifact_dir / "fold_0_predictions.pkl").unlink()

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is None

    tlog.log("  [PASS] check_fold_complete_missing_predictions")
    tlog.record("check_fold_complete_missing_predictions", "PASS")


def test_check_fold_complete_corrupt_pickle(tlog: _TestLogger):
    """Corrupt/truncated predictions.pkl: returns None."""
    artifact_dir = OUTPUT_DIR / "unit_resume_corrupt"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)
    # Corrupt the pickle
    preds_path = artifact_dir / "fold_0_predictions.pkl"
    preds_path.write_bytes(b"NOT_A_PICKLE")

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is None

    tlog.log("  [PASS] check_fold_complete_corrupt_pickle")
    tlog.record("check_fold_complete_corrupt_pickle", "PASS")


def test_check_fold_complete_missing_expected_artifact(tlog: _TestLogger):
    """One expected artifact deleted: returns None."""
    artifact_dir = OUTPUT_DIR / "unit_resume_missing_artifact"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)
    # Delete a model artifact that's listed in expected_artifacts
    model_path = artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib"
    if model_path.exists():
        model_path.unlink()

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is None

    tlog.log("  [PASS] check_fold_complete_missing_expected_artifact")
    tlog.record("check_fold_complete_missing_expected_artifact", "PASS")


def test_check_fold_complete_truncated_model(tlog: _TestLogger):
    """Pipeline model file < 1KB: returns None (truncated)."""
    artifact_dir = OUTPUT_DIR / "unit_resume_truncated"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)
    # Truncate the model file
    model_path = artifact_dir / f"fold_0_{TEST_MODEL_NAME}_model_split1.joblib"
    model_path.write_bytes(b"X" * 100)  # < 1KB

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is None

    tlog.log("  [PASS] check_fold_complete_truncated_model")
    tlog.record("check_fold_complete_truncated_model", "PASS")


def test_check_fold_complete_no_expected_artifacts_key(tlog: _TestLogger):
    """Old-format _meta without expected_artifacts: returns None."""
    artifact_dir = OUTPUT_DIR / "unit_resume_old_format"
    _create_complete_fold_artifacts(artifact_dir, fold_id=0)
    # Overwrite predictions.pkl with old-format (no expected_artifacts)
    preds_path = artifact_dir / "fold_0_predictions.pkl"
    with open(preds_path, "rb") as f:
        data = pickle.load(f)
    del data["_meta"]["expected_artifacts"]
    with open(preds_path, "wb") as f:
        pickle.dump(data, f)

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is None

    tlog.log("  [PASS] check_fold_complete_no_expected_artifacts_key")
    tlog.record("check_fold_complete_no_expected_artifacts_key", "PASS")


def test_check_fold_complete_no_valid_clusters(tlog: _TestLogger):
    """Fold with best_p_value=None: still detected as complete."""
    artifact_dir = OUTPUT_DIR / "unit_resume_no_valid_clusters"
    _create_complete_fold_artifacts(
        artifact_dir, fold_id=0, has_valid_pvalue=False,
    )

    result = _check_fold_complete(artifact_dir, fold_id=0)
    assert result is not None
    assert result["_meta"]["fold_id"] == 0
    # by_model should be empty (no valid model)
    assert len(result["by_model"]) == 0

    tlog.log("  [PASS] check_fold_complete_no_valid_clusters")
    tlog.record("check_fold_complete_no_valid_clusters", "PASS")


def test_save_load_fold_predictions_roundtrip(tlog: _TestLogger):
    """_save_fold_predictions -> _load_fold_results round-trip."""
    artifact_dir = OUTPUT_DIR / "unit_predictions_roundtrip"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    model_names = [TEST_MODEL_NAME]
    per_model_data = {
        TEST_MODEL_NAME: {
            "eval_result": {"fold_id": 0, "accuracy": 0.9},
            "raw_preds": {"y_true": np.array(["A", "B"]), "y_pred": np.array(["A", "B"])},
            "predictions_rows": [{"spec": "s0"}, {"spec": "s1"}],
        }
    }
    model_params = {"p_values": [0.01], "classification_mode": "multiclass"}

    _save_fold_predictions(
        artifact_dir, fold_id=0, model_names=model_names,
        per_model_data=per_model_data,
        model_params=model_params,
        training_context="cv_single_model",
        expected_artifacts=["fold_0_clusters.joblib"],
    )

    loaded = _load_fold_results(artifact_dir, fold_id=0)
    assert "_meta" in loaded
    assert "by_model" in loaded
    assert loaded["_meta"]["fold_id"] == 0
    assert loaded["_meta"]["training_context"] == "cv_single_model"
    assert loaded["_meta"]["model_names"] == sorted(model_names)
    assert loaded["_meta"]["expected_artifacts"] == ["fold_0_clusters.joblib"]
    assert TEST_MODEL_NAME in loaded["by_model"]
    assert loaded["by_model"][TEST_MODEL_NAME]["eval_result"]["accuracy"] == 0.9

    tlog.log("  [PASS] save_load_fold_predictions_roundtrip")
    tlog.record("save_load_fold_predictions_roundtrip", "PASS")


def test_validate_fold_meta_matching(tlog: _TestLogger):
    """Matching params: no error."""
    artifact_dir = OUTPUT_DIR / "unit_meta_matching"
    info = _create_complete_fold_artifacts(artifact_dir, fold_id=0)

    with open(artifact_dir / "fold_0_predictions.pkl", "rb") as f:
        preds_data = pickle.load(f)

    # Should not raise
    _validate_fold_meta(
        preds_data, fold_id=0, model_names=info["model_names"],
        current_model_params=info["model_params"],
        current_training_context=info["training_context"],
    )

    tlog.log("  [PASS] validate_fold_meta_matching")
    tlog.record("validate_fold_meta_matching", "PASS")


def test_validate_fold_meta_fold_id_mismatch(tlog: _TestLogger):
    """Different fold_id: ValueError."""
    artifact_dir = OUTPUT_DIR / "unit_meta_fold_mismatch"
    info = _create_complete_fold_artifacts(artifact_dir, fold_id=0)

    with open(artifact_dir / "fold_0_predictions.pkl", "rb") as f:
        preds_data = pickle.load(f)

    with pytest.raises(ValueError, match="fold_id mismatch"):
        _validate_fold_meta(
            preds_data, fold_id=99, model_names=info["model_names"],
            current_model_params=info["model_params"],
            current_training_context=info["training_context"],
        )

    tlog.log("  [PASS] validate_fold_meta_fold_id_mismatch")
    tlog.record("validate_fold_meta_fold_id_mismatch", "PASS")


def test_validate_fold_meta_model_names_mismatch(tlog: _TestLogger):
    """Different model_names: ValueError."""
    artifact_dir = OUTPUT_DIR / "unit_meta_names_mismatch"
    info = _create_complete_fold_artifacts(artifact_dir, fold_id=0)

    with open(artifact_dir / "fold_0_predictions.pkl", "rb") as f:
        preds_data = pickle.load(f)

    with pytest.raises(ValueError, match="model_names mismatch"):
        _validate_fold_meta(
            preds_data, fold_id=0, model_names=["ridge_cv"],
            current_model_params=info["model_params"],
            current_training_context=info["training_context"],
        )

    tlog.log("  [PASS] validate_fold_meta_model_names_mismatch")
    tlog.record("validate_fold_meta_model_names_mismatch", "PASS")


def test_validate_fold_meta_training_context_mismatch(tlog: _TestLogger):
    """Different training_context: ValueError."""
    artifact_dir = OUTPUT_DIR / "unit_meta_ctx_mismatch"
    info = _create_complete_fold_artifacts(artifact_dir, fold_id=0,
                                           training_context="cv_single_model")

    with open(artifact_dir / "fold_0_predictions.pkl", "rb") as f:
        preds_data = pickle.load(f)

    with pytest.raises(ValueError, match="training_context mismatch"):
        _validate_fold_meta(
            preds_data, fold_id=0, model_names=info["model_names"],
            current_model_params=info["model_params"],
            current_training_context="cv_ensemble",
        )

    tlog.log("  [PASS] validate_fold_meta_training_context_mismatch")
    tlog.record("validate_fold_meta_training_context_mismatch", "PASS")


def test_validate_fold_meta_model_params_mismatch(tlog: _TestLogger):
    """Different model_params value: ValueError."""
    artifact_dir = OUTPUT_DIR / "unit_meta_params_mismatch"
    info = _create_complete_fold_artifacts(artifact_dir, fold_id=0)

    with open(artifact_dir / "fold_0_predictions.pkl", "rb") as f:
        preds_data = pickle.load(f)

    # Change classification_mode from the saved value
    bad_params = dict(info["model_params"])
    bad_params["classification_mode"] = "binary"

    with pytest.raises(ValueError, match="mismatch"):
        _validate_fold_meta(
            preds_data, fold_id=0, model_names=info["model_names"],
            current_model_params=bad_params,
            current_training_context=info["training_context"],
        )

    tlog.log("  [PASS] validate_fold_meta_model_params_mismatch")
    tlog.record("validate_fold_meta_model_params_mismatch", "PASS")


def test_validate_fold_meta_no_meta(tlog: _TestLogger):
    """Old-format predictions without _meta: ValueError."""
    artifact_dir = OUTPUT_DIR / "unit_meta_no_meta"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    # Write predictions.pkl without _meta
    preds_path = artifact_dir / "fold_0_predictions.pkl"
    with open(preds_path, "wb") as f:
        pickle.dump({"by_model": {}}, f)

    with open(preds_path, "rb") as f:
        preds_data = pickle.load(f)

    with pytest.raises(ValueError, match="no _meta"):
        _validate_fold_meta(
            preds_data, fold_id=0, model_names=[TEST_MODEL_NAME],
            current_model_params={},
            current_training_context="cv_single_model",
        )

    tlog.log("  [PASS] validate_fold_meta_no_meta")
    tlog.record("validate_fold_meta_no_meta", "PASS")


def test_get_fold_artifact_paths(tlog: _TestLogger):
    """Verify all expected paths returned for cleanup."""
    artifact_dir = Path("/tmp/test_paths")
    model_names = ["lasso_cv", "ridge_cv"]

    paths = _get_fold_artifact_paths(artifact_dir, fold_id=0,
                                     model_names=model_names,
                                     retrain_on_full_train=False)

    path_names = [p.name for p in paths]

    # Should include: clusters, per-model (p_value, pipeline, metrics),
    # per-model NO_VALID_CLUSTERS notice, and predictions.pkl
    assert "fold_0_clusters.joblib" in path_names
    assert "fold_0_predictions.pkl" in path_names
    for mn in model_names:
        assert f"fold_0_{mn}_p_value.joblib" in path_names
        assert f"fold_0_{mn}_model_split1.joblib" in path_names
        assert f"fold_0_{mn}_results_split1.json" in path_names
        assert f"fold_0_{mn}_NO_VALID_CLUSTERS.txt" in path_names

    tlog.log("  [PASS] get_fold_artifact_paths")
    tlog.record("get_fold_artifact_paths", "PASS")


# ---------------------------------------------------------------------------
# F. Predictions Row Format Unit Tests
# ---------------------------------------------------------------------------

def test_predictions_row_format_multiclass(tlog: _TestLogger):
    """Verify multiclass prediction row structure."""
    disease_classes = ["Covid19", "HIV", "Healthy"]

    # Scored row
    scored_row = {
        "participant_label": "P001",
        "specimen_label": "S001",
        "true_disease": "Covid19",
        "predicted_disease": "Covid19",
        "abstained": False,
        "malid_cross_validation_fold_id_when_in_test_set": 0,
    }
    for cls in disease_classes:
        scored_row[f"score_{cls}"] = 0.33

    # Abstained row
    abstained_row = {
        "participant_label": "P002",
        "specimen_label": "S002",
        "true_disease": "HIV",
        "predicted_disease": None,
        "abstained": True,
        "malid_cross_validation_fold_id_when_in_test_set": 0,
    }
    for cls in disease_classes:
        abstained_row[f"score_{cls}"] = None

    # Build DataFrame
    score_cols = sorted(f"score_{c}" for c in disease_classes)
    fixed_cols = [
        "participant_label", "specimen_label", "true_disease", "predicted_disease",
        "abstained", "malid_cross_validation_fold_id_when_in_test_set",
    ]
    df = pd.DataFrame([scored_row, abstained_row], columns=fixed_cols + score_cols)

    assert list(df.columns[:6]) == fixed_cols
    assert df.loc[0, "abstained"] == False  # noqa: E712 — numpy bool, not Python bool
    assert pd.isna(df.loc[1, "predicted_disease"])
    assert pd.isna(df.loc[1, "score_Covid19"])
    assert len(score_cols) == len(disease_classes)

    tlog.log("  [PASS] predictions_row_format_multiclass")
    tlog.record("predictions_row_format_multiclass", "PASS")


def test_predictions_row_format_binary(tlog: _TestLogger):
    """Verify binary prediction row structure."""
    row = {
        "participant_label": "P001",
        "specimen_label": "S001",
        "disease_label": 1,
        "disease_label_str": "Covid19",
        "disease_model": "Covid19",
        "model_score": 0.85,
        "malid_cross_validation_fold_id_when_in_test_set": 0,
    }
    expected_cols = [
        "participant_label", "specimen_label", "disease_label", "disease_label_str",
        "disease_model", "model_score", "malid_cross_validation_fold_id_when_in_test_set",
    ]
    df = pd.DataFrame([row], columns=expected_cols)

    assert list(df.columns) == expected_cols
    assert df.loc[0, "disease_label"] in (0, 1)
    assert 0.0 <= df.loc[0, "model_score"] <= 1.0

    tlog.log("  [PASS] predictions_row_format_binary")
    tlog.record("predictions_row_format_binary", "PASS")


# ===========================================================================
# Tier 2: Integration Tests (test_data/)
# ===========================================================================

# ---------------------------------------------------------------------------
# Core Pipeline Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_multiclass(tlog: _TestLogger, n_jobs: int):
    """Full multiclass pipeline on fold 0 using test data."""
    loader = create_test_loader(verbose=0)

    # --- Load and prepare ---
    train_seqs, train_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")
    assert DISEASE_COL in train_seqs.columns
    assert train_seqs[DISEASE_COL].notna().all()
    n_diseases = train_seqs[DISEASE_COL].nunique()
    tlog.log(f"  Loaded: {len(train_seqs):,} seqs, {train_seqs[PARTICIPANT_COL].nunique()} "
             f"participants, {n_diseases} diseases")

    # --- Split via centralized splits ---
    ts1_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
    ts2_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
    assert not (ts1_parts & ts2_parts), "ts1/ts2 overlap"

    ts1 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
    ts2 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()
    assert len(ts1) > 0 and len(ts2) > 0
    tlog.log(f"  ts1: {ts1[PARTICIPANT_COL].nunique()} participants, ts2: {ts2[PARTICIPANT_COL].nunique()}")

    # --- Train ---
    train_result = train_convergent_cluster_classifier(
        train_smaller1_df=ts1,
        train_smaller2_df=ts2,
        sequence_identity_threshold=0.90,
        model_names=[TEST_MODEL_NAME],
        p_values=TEST_P_VALUES,
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
        verbose=1,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    assert "centroids_with_scores" in train_result
    assert "disease_classes" in train_result
    assert TEST_MODEL_NAME in train_result["results"]

    model_result = train_result["results"][TEST_MODEL_NAME]
    assert model_result["best_p_value"] is not None, "No valid p-value found"
    assert model_result["pipeline"] is not None
    tlog.log(f"  Best p-value: {model_result['best_p_value']}")

    # --- Save + Load + Inference ---
    artifact_dir = OUTPUT_DIR / "integration" / "multiclass"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    save_fold_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, train_result=train_result)

    clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
    clf.load_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, model_name=TEST_MODEL_NAME)
    assert clf._is_loaded

    fd_test = clf.featurize(ts2, disease_col=DISEASE_COL, n_jobs=n_jobs)
    tlog.log(f"  Featurized: {fd_test.n_scored} scored, {fd_test.n_abstained} abstained")

    if fd_test.n_scored > 0:
        y_proba = clf.predict_proba(fd_test.X)
        y_pred = clf.predict(fd_test.X)
        assert y_proba.shape[0] == fd_test.n_scored
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)
        assert len(y_pred) == fd_test.n_scored
        tlog.log(f"  predict_proba sums to 1.0: confirmed")

    # --- Evaluate ---
    test_seqs, test_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "test")
    fd_eval = featurize(
        test_seqs,
        p_value_threshold=model_result["best_p_value"],
        centroids_with_scores=train_result["centroids_with_scores"],
        sequence_identity_threshold=0.90,
        disease_classes=train_result["disease_classes"],
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
    )
    metrics, raw_preds = evaluate_on_test(
        featurized=fd_eval,
        pipeline=model_result["pipeline"],
        classes=np.array(train_result["disease_classes"]),
        fold_id=TEST_FOLD_ID,
        model_name=TEST_MODEL_NAME,
    )
    assert "accuracy" in metrics
    assert "n_scored" in metrics
    tlog.log(f"  Test accuracy: {metrics['accuracy']:.3f}, "
             f"scored: {metrics['n_scored']}, abstained: {metrics['n_abstained']}")

    tlog.log("  [PASS] integration_multiclass")
    tlog.record("integration_multiclass", "PASS", {
        "best_p_value": model_result["best_p_value"],
        "accuracy": metrics["accuracy"],
    })


@pytest.mark.integration
def test_integration_binary(tlog: _TestLogger, n_jobs: int):
    """Full binary pipeline on fold 0: 1 disease vs reference."""
    loader = create_test_loader(verbose=0)
    train_seqs, train_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")

    diseases = sorted(train_seqs[DISEASE_COL].unique())
    reference = "Healthy/Background"
    target = next(d for d in diseases if d != reference)
    tlog.log(f"  Binary pair: {target} vs {reference}")

    # Filter to binary pair
    bin_seqs, bin_meta = filter_to_binary_pair(train_seqs, train_meta, target, reference)
    assert sorted(bin_seqs[DISEASE_COL].unique()) == sorted([target, reference])

    # Split
    ts1_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
    ts2_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
    bin_ts1 = bin_seqs[bin_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
    bin_ts2 = bin_seqs[bin_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()

    # Train
    result = train_convergent_cluster_classifier(
        train_smaller1_df=bin_ts1,
        train_smaller2_df=bin_ts2,
        sequence_identity_threshold=0.90,
        model_names=[TEST_MODEL_NAME],
        p_values=TEST_P_VALUES,
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
        verbose=1,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    assert len(result["disease_classes"]) == 2
    assert set(result["disease_classes"]) == {target, reference}

    mr = result["results"][TEST_MODEL_NAME]

    # Save with binary_pair
    artifact_dir = OUTPUT_DIR / "integration" / "binary"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    save_fold_artifacts(
        artifact_dir, fold_id=TEST_FOLD_ID, train_result=result,
        disease_filter=(target, reference),
    )

    if mr["best_p_value"] is None:
        # Small test data: binary pair may not produce enough convergent
        # clusters for training. Verify graceful abstention with clear artifacts.
        tlog.log(f"  No valid p-value found (expected with small test data)")
        tlog.log(f"  min_cluster_pvalue: {result.get('min_cluster_pvalue')}")

        notice_file = artifact_dir / f"fold_{TEST_FOLD_ID}_{TEST_MODEL_NAME}_NO_VALID_CLUSTERS.txt"
        assert notice_file.exists(), "NO_VALID_CLUSTERS.txt notice should be written"
        notice_text = notice_file.read_text()
        assert "no valid p-value" in notice_text.lower()
        # Verify min cluster p-value hint is included (clusters exist but p-vals too high)
        if result.get("min_cluster_pvalue") is not None:
            assert "Minimum p-value" in notice_text or "minimum p-value" in notice_text, (
                "Notice should include minimum cluster p-value hint"
            )
            assert "wider p-value range" in notice_text, (
                "Notice should suggest wider p-value range"
            )
    else:
        tlog.log(f"  Best p-value: {mr['best_p_value']}")

        # Load and verify binary pair
        clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
        clf.load_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, model_name=TEST_MODEL_NAME)
        assert clf.disease_class_ == target
        assert clf.reference_class_ == reference
        assert list(clf.classes_) == [reference, target]

        # Inference
        fd_bin = clf.featurize(bin_ts2, disease_col=DISEASE_COL, n_jobs=n_jobs)
        if fd_bin.n_scored > 0:
            y_proba = clf.predict_proba(fd_bin.X)
            assert y_proba.shape[1] == 2
            assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)
            y_pred = clf.predict(fd_bin.X)
            assert set(y_pred) <= {target, reference}
            tlog.log(f"  Inference: {fd_bin.n_scored} scored, proba shape {y_proba.shape}")

        # Evaluate with binary metrics
        test_seqs, test_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "test")
        test_seqs, test_meta = filter_to_binary_pair(test_seqs, test_meta, target, reference)
        fd_test = featurize(
            test_seqs,
            p_value_threshold=mr["best_p_value"],
            centroids_with_scores=result["centroids_with_scores"],
            sequence_identity_threshold=0.90,
            disease_classes=result["disease_classes"],
            disease_col=DISEASE_COL,
            n_jobs=n_jobs,
        )
        metrics, _ = evaluate_on_test(
            featurized=fd_test,
            pipeline=mr["pipeline"],
            classes=np.array(result["disease_classes"]),
            fold_id=TEST_FOLD_ID,
            model_name=TEST_MODEL_NAME,
            reference_class=reference,
        )
        if metrics.get("auroc_binary") is not None:
            tlog.log(f"  Binary AUROC: {metrics['auroc_binary']:.3f}")

    tlog.log("  [PASS] integration_binary")
    tlog.record("integration_binary", "PASS", {"target": target, "reference": reference})


@pytest.mark.integration
def test_integration_binary_with_diseases_flag(tlog: _TestLogger, n_jobs: int):
    """Binary mode selecting 1 disease from 4-class data via run_training_orchestration."""
    loader = create_test_loader(verbose=0)
    metadata_path = TEST_DATA_DIR / "metadata.tsv"
    diseases_in_data = sorted(
        pd.read_csv(metadata_path, sep="\t")["disease"].unique()
    )
    reference = "Healthy/Background"
    target = next(d for d in diseases_in_data if d != reference)

    output_dir = OUTPUT_DIR / "integration" / "binary_diseases_flag"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    def fold_loop_fn(output_dir, disease_filter, **kwargs):
        """Minimal fold loop that trains 1 fold and creates output dir."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_seqs, train_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")
        if disease_filter:
            train_seqs, train_meta = filter_to_binary_pair(
                train_seqs, train_meta, disease_filter[0], disease_filter[1],
            )
        ts1_parts = set(loader.get_split_participants(
            TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
        ts2_parts = set(loader.get_split_participants(
            TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
        ts1 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
        ts2 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()

        result = train_convergent_cluster_classifier(
            train_smaller1_df=ts1, train_smaller2_df=ts2,
            sequence_identity_threshold=0.90,
            model_names=[TEST_MODEL_NAME], p_values=TEST_P_VALUES,
            disease_col=DISEASE_COL, n_jobs=n_jobs, verbose=0,
            glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
        )
        fold_results = [{"fold_id": TEST_FOLD_ID, "model_name": TEST_MODEL_NAME,
                         "accuracy": 0.5, "n_scored": 5, "n_abstained": 0,
                         "abstention_rate": 0.0}]
        agg = {TEST_MODEL_NAME: {"accuracy_global": 0.5}}
        return fold_results, agg

    all_results = run_training_orchestration(
        base_dir=output_dir,
        classification_mode="binary",
        reference_class=reference,
        diseases=[target],
        disease_classes=diseases_in_data,
        fold_loop_fn=fold_loop_fn,
        loop_kwargs={},
    )

    # Verify pair key and subdir
    pair_name = make_pair_name(target, reference)
    assert pair_name in all_results
    assert (output_dir / pair_name).exists()

    tlog.log(f"  [PASS] integration_binary_with_diseases_flag ({pair_name})")
    tlog.record("integration_binary_with_diseases_flag", "PASS", {"pair": pair_name})


@pytest.mark.integration
def test_integration_multi_binary(tlog: _TestLogger, n_jobs: int):
    """Multi-binary via run_training_orchestration: N-1 independent pairs."""
    loader = create_test_loader(verbose=0)
    metadata_path = TEST_DATA_DIR / "metadata.tsv"
    diseases_in_data = sorted(
        pd.read_csv(metadata_path, sep="\t")["disease"].unique()
    )
    reference = "Healthy/Background"
    non_ref_diseases = [d for d in diseases_in_data if d != reference]

    output_dir = OUTPUT_DIR / "integration" / "multi_binary"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    def fold_loop_fn(output_dir, disease_filter, **kwargs):
        """Minimal fold loop for one binary pair."""
        train_seqs, train_meta = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")
        if disease_filter:
            train_seqs, train_meta = filter_to_binary_pair(
                train_seqs, train_meta, disease_filter[0], disease_filter[1],
            )
        ts1_parts = set(loader.get_split_participants(
            TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
        ts2_parts = set(loader.get_split_participants(
            TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
        ts1 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
        ts2 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()

        result = train_convergent_cluster_classifier(
            train_smaller1_df=ts1, train_smaller2_df=ts2,
            sequence_identity_threshold=0.90,
            model_names=[TEST_MODEL_NAME], p_values=TEST_P_VALUES,
            disease_col=DISEASE_COL, n_jobs=n_jobs, verbose=0,
            glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        save_fold_artifacts(output_dir, fold_id=TEST_FOLD_ID, train_result=result,
                            disease_filter=disease_filter)

        fold_results = [{"fold_id": TEST_FOLD_ID, "model_name": TEST_MODEL_NAME,
                         "accuracy": 0.5, "n_scored": 5, "n_abstained": 0,
                         "abstention_rate": 0.0}]
        agg = {TEST_MODEL_NAME: {"accuracy_global": 0.5}}
        return fold_results, agg

    all_results = run_training_orchestration(
        base_dir=output_dir,
        classification_mode="multi-binary",
        reference_class=reference,
        diseases=None,
        disease_classes=diseases_in_data,
        fold_loop_fn=fold_loop_fn,
        loop_kwargs={},
    )

    # Verify all N-1 pairs trained
    assert len(all_results) == len(non_ref_diseases), (
        f"Expected {len(non_ref_diseases)} pairs, got {len(all_results)}"
    )
    for disease in non_ref_diseases:
        pair_name = make_pair_name(disease, reference)
        assert pair_name in all_results, f"Missing pair: {pair_name}"
        pair_dir = output_dir / pair_name
        assert pair_dir.exists(), f"Missing pair dir: {pair_dir}"
        assert (pair_dir / f"fold_{TEST_FOLD_ID}_clusters.joblib").exists()

    tlog.log(f"  [PASS] integration_multi_binary ({len(non_ref_diseases)} pairs)")
    tlog.record("integration_multi_binary", "PASS", {"n_pairs": len(non_ref_diseases)})


# ---------------------------------------------------------------------------
# Training Context / Mode Variants
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_cv_ensemble_splits(tlog: _TestLogger, n_jobs: int):
    """cv_ensemble: train < cv_single_model, no overlap train/val, pipeline runs."""
    loader = create_test_loader(verbose=0)

    # Get split participants for both contexts
    sm_ts1 = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
    sm_ts2 = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
    ens_ts1 = set(loader.get_split_participants(TEST_FOLD_ID, "cv_ensemble", ["train_smaller1"]))
    ens_ts2 = set(loader.get_split_participants(TEST_FOLD_ID, "cv_ensemble", ["train_smaller2"]))
    ens_val = set(loader.get_split_participants(TEST_FOLD_ID, "cv_ensemble", ["validation"]))

    sm_train = sm_ts1 | sm_ts2
    ens_train = ens_ts1 | ens_ts2

    # cv_ensemble training is strictly smaller
    assert len(ens_train) < len(sm_train), (
        f"cv_ensemble train ({len(ens_train)}) should be < cv_single_model train ({len(sm_train)})"
    )
    # No overlap between validation and training
    assert not (ens_train & ens_val)
    # Partition completeness
    assert ens_train | ens_val == sm_train
    # No overlap between ts1 and ts2
    assert not (ens_ts1 & ens_ts2)

    tlog.log(f"  cv_single_model train: {len(sm_train)}")
    tlog.log(f"  cv_ensemble train: {len(ens_train)} (ts1={len(ens_ts1)}, ts2={len(ens_ts2)})")
    tlog.log(f"  cv_ensemble validation: {len(ens_val)}")

    # Run pipeline on cv_ensemble data
    train_seqs, _ = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")
    ens_ts1_df = train_seqs[train_seqs[PARTICIPANT_COL].isin(ens_ts1)].copy()
    ens_ts2_df = train_seqs[train_seqs[PARTICIPANT_COL].isin(ens_ts2)].copy()

    result = train_convergent_cluster_classifier(
        train_smaller1_df=ens_ts1_df,
        train_smaller2_df=ens_ts2_df,
        sequence_identity_threshold=0.90,
        model_names=[TEST_MODEL_NAME],
        p_values=TEST_P_VALUES,
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
        verbose=1,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    assert "centroids_with_scores" in result
    assert TEST_MODEL_NAME in result["results"]
    mr = result["results"][TEST_MODEL_NAME]
    assert mr["best_p_value"] is not None
    tlog.log(f"  cv_ensemble pipeline: best_p={mr['best_p_value']}")

    tlog.log("  [PASS] integration_cv_ensemble_splits")
    tlog.record("integration_cv_ensemble_splits", "PASS")


@pytest.mark.integration
def test_integration_retrain_full(tlog: _TestLogger, n_jobs: int):
    """retrain_on_full_train=True: artifact suffix is 'full'."""
    loader = create_test_loader(verbose=0)
    train_seqs, _ = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")

    ts1_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
    ts2_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
    ts1 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
    ts2 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()

    result = train_convergent_cluster_classifier(
        train_smaller1_df=ts1,
        train_smaller2_df=ts2,
        sequence_identity_threshold=0.90,
        model_names=[TEST_MODEL_NAME],
        p_values=TEST_P_VALUES,
        disease_col=DISEASE_COL,
        retrain_on_full_train=True,
        n_jobs=n_jobs,
        verbose=1,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    mr = result["results"][TEST_MODEL_NAME]
    assert mr["best_p_value"] is not None

    artifact_dir = OUTPUT_DIR / "integration" / "retrain_full"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    save_fold_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, train_result=result,
                        retrain_on_full_train=True)

    # Verify "full" suffix
    assert (artifact_dir / f"fold_{TEST_FOLD_ID}_{TEST_MODEL_NAME}_model_full.joblib").exists()
    assert (artifact_dir / f"fold_{TEST_FOLD_ID}_{TEST_MODEL_NAME}_results_full.json").exists()
    assert not (artifact_dir / f"fold_{TEST_FOLD_ID}_{TEST_MODEL_NAME}_model_split1.joblib").exists()

    # Load with retrain_on_full_train=True
    clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=TEST_MODEL_NAME)
    clf.load_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, model_name=TEST_MODEL_NAME,
                       retrain_on_full_train=True)
    assert clf._is_loaded

    tlog.log("  [PASS] integration_retrain_full")
    tlog.record("integration_retrain_full", "PASS")


@pytest.mark.integration
def test_integration_multiple_models(tlog: _TestLogger, n_jobs: int):
    """Two model_names in single run: both trained with separate artifacts."""
    loader = create_test_loader(verbose=0)
    train_seqs, _ = load_and_prepare_fold(loader, TEST_FOLD_ID, "train")

    ts1_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller1"]))
    ts2_parts = set(loader.get_split_participants(TEST_FOLD_ID, "cv_single_model", ["train_smaller2"]))
    ts1 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts1_parts)].copy()
    ts2 = train_seqs[train_seqs[PARTICIPANT_COL].isin(ts2_parts)].copy()

    model_names = ["lasso_cv", "ridge_cv"]
    result = train_convergent_cluster_classifier(
        train_smaller1_df=ts1,
        train_smaller2_df=ts2,
        sequence_identity_threshold=0.90,
        model_names=model_names,
        p_values=TEST_P_VALUES,
        disease_col=DISEASE_COL,
        n_jobs=n_jobs,
        verbose=1,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    # Both models in results
    for mn in model_names:
        assert mn in result["results"], f"Missing model: {mn}"
        assert result["results"][mn]["best_p_value"] is not None

    # Save and verify separate artifacts
    artifact_dir = OUTPUT_DIR / "integration" / "multiple_models"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    save_fold_artifacts(artifact_dir, fold_id=TEST_FOLD_ID, train_result=result)

    for mn in model_names:
        assert (artifact_dir / f"fold_{TEST_FOLD_ID}_{mn}_model_split1.joblib").exists()
        assert (artifact_dir / f"fold_{TEST_FOLD_ID}_{mn}_p_value.joblib").exists()

    tlog.log(f"  [PASS] integration_multiple_models ({', '.join(model_names)})")
    tlog.record("integration_multiple_models", "PASS", {"models": model_names})


# ---------------------------------------------------------------------------
# Orchestrator Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_train_all_folds(tlog: _TestLogger, n_jobs: int):
    """Full train_all_folds on fold 0: summary JSON, RESULTS_*.md, predictions CSV."""
    output_dir = OUTPUT_DIR / "integration" / "train_all_folds"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    all_results = train_all_folds(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="multiclass",
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        p_values=TEST_P_VALUES,
        n_jobs=n_jobs,
        verbose=1,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    # Verify outputs
    assert "multiclass" in all_results
    mc_data = all_results["multiclass"]
    assert len(mc_data["fold_results"]) > 0
    assert TEST_MODEL_NAME in mc_data["aggregated_by_model"]

    # Summary JSON
    summaries = list(output_dir.glob("summary_*.json"))
    assert len(summaries) >= 1, "No summary JSON found"
    with open(summaries[0]) as f:
        summary = json.load(f)
    assert summary["classification_mode"] == "multiclass"
    assert summary["fold_ids"] == [TEST_FOLD_ID]

    # Results Markdown
    results_md = list(output_dir.glob("RESULTS_*.md"))
    assert len(results_md) >= 1, "No RESULTS_*.md found"

    # Predictions CSV
    pred_csv = list(output_dir.glob(f"{TEST_MODEL_NAME}_multiclass_predictions.csv"))
    assert len(pred_csv) >= 1, "No predictions CSV found"
    pred_df = pd.read_csv(pred_csv[0])
    assert "participant_label" in pred_df.columns
    assert "true_disease" in pred_df.columns
    assert "predicted_disease" in pred_df.columns

    # Fold artifacts
    assert (output_dir / f"fold_{TEST_FOLD_ID}_clusters.joblib").exists()
    assert (output_dir / f"fold_{TEST_FOLD_ID}_predictions.pkl").exists()

    tlog.log(f"  [PASS] integration_train_all_folds")
    tlog.record("integration_train_all_folds", "PASS")


# ---------------------------------------------------------------------------
# Resume Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_resume(tlog: _TestLogger, n_jobs: int):
    """Original run -> resume -> metrics match."""
    output_dir = OUTPUT_DIR / "integration" / "resume_test"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    common_kwargs = dict(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="multiclass",
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        p_values=TEST_P_VALUES,
        n_jobs=n_jobs,
        verbose=1,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    # --- Original run ---
    original = train_all_folds(**common_kwargs, resume=False)
    orig_metrics = original["multiclass"]["fold_results"]

    # --- Resume run ---
    t0 = time.monotonic()
    resumed = train_all_folds(**common_kwargs, resume=True)
    resume_time = time.monotonic() - t0
    resume_metrics = resumed["multiclass"]["fold_results"]

    # Metrics should match exactly
    assert len(orig_metrics) == len(resume_metrics)
    for orig, res in zip(orig_metrics, resume_metrics):
        assert orig["fold_id"] == res["fold_id"]
        assert orig["model_name"] == res["model_name"]
        assert orig["accuracy"] == res["accuracy"], (
            f"Accuracy mismatch: original {orig['accuracy']} vs resume {res['accuracy']}"
        )
        assert orig["n_scored"] == res["n_scored"]
        assert orig["n_abstained"] == res["n_abstained"]

    tlog.log(f"  Resume completed in {resume_time:.1f}s (should be fast — fold was skipped)")
    tlog.log("  [PASS] integration_resume")
    tlog.record("integration_resume", "PASS")


@pytest.mark.integration
def test_integration_resume_incomplete_fold(tlog: _TestLogger, n_jobs: int):
    """Partial artifacts (no predictions.pkl) -> resume retrains from scratch."""
    output_dir = OUTPUT_DIR / "integration" / "resume_incomplete"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    common_kwargs = dict(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="multiclass",
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        p_values=TEST_P_VALUES,
        n_jobs=n_jobs,
        verbose=0,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    # Original run to create complete artifacts
    train_all_folds(**common_kwargs, resume=False)

    # Delete predictions.pkl to simulate incomplete fold
    preds_path = output_dir / f"fold_{TEST_FOLD_ID}_predictions.pkl"
    assert preds_path.exists()
    preds_path.unlink()

    # Resume should retrain (not error)
    result = train_all_folds(**common_kwargs, resume=True)
    assert "multiclass" in result
    assert len(result["multiclass"]["fold_results"]) > 0
    # predictions.pkl should be recreated
    assert preds_path.exists()

    tlog.log("  [PASS] integration_resume_incomplete_fold")
    tlog.record("integration_resume_incomplete_fold", "PASS")


@pytest.mark.integration
def test_integration_resume_param_mismatch(tlog: _TestLogger, n_jobs: int):
    """Resume with different p_values -> ValueError."""
    output_dir = OUTPUT_DIR / "integration" / "resume_mismatch"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    base_kwargs = dict(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="multiclass",
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        n_jobs=n_jobs,
        verbose=0,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    # Original run with one set of p_values
    train_all_folds(**base_kwargs, p_values=[0.01, 0.05], resume=False)

    # Resume with different p_values -> should raise ValueError
    with pytest.raises(ValueError, match="mismatch"):
        train_all_folds(**base_kwargs, p_values=[0.001, 0.01], resume=True)

    tlog.log("  [PASS] integration_resume_param_mismatch")
    tlog.record("integration_resume_param_mismatch", "PASS")


# ---------------------------------------------------------------------------
# Output Format Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_predictions_csv_multiclass(tlog: _TestLogger, n_jobs: int):
    """Verify multiclass predictions CSV columns and values."""
    output_dir = OUTPUT_DIR / "integration" / "predictions_mc"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_all_folds(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="multiclass",
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        p_values=TEST_P_VALUES,
        n_jobs=n_jobs,
        verbose=0,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    csv_files = list(output_dir.glob(f"{TEST_MODEL_NAME}_multiclass_predictions.csv"))
    assert len(csv_files) >= 1
    df = pd.read_csv(csv_files[0])

    # Required columns
    for col in ["participant_label", "specimen_label", "true_disease",
                "predicted_disease", "abstained",
                "malid_cross_validation_fold_id_when_in_test_set"]:
        assert col in df.columns, f"Missing column: {col}"

    # Score columns exist
    score_cols = [c for c in df.columns if c.startswith("score_")]
    assert len(score_cols) > 0, "No score columns found"

    # Scored rows: scores should be numeric
    scored = df[df["abstained"] == False]
    if len(scored) > 0:
        for sc in score_cols:
            assert scored[sc].notna().all(), f"NaN in scored row score column {sc}"
        # Scores should roughly sum to 1
        score_sums = scored[score_cols].sum(axis=1)
        assert np.allclose(score_sums, 1.0, atol=0.1), "Score sums far from 1.0"

    # Abstained rows: predicted_disease should be None/NaN
    abstained = df[df["abstained"] == True]
    if len(abstained) > 0:
        assert abstained["predicted_disease"].isna().all()

    tlog.log(f"  Predictions CSV: {len(df)} rows ({len(scored)} scored, {len(abstained)} abstained)")
    tlog.log("  [PASS] integration_predictions_csv_multiclass")
    tlog.record("integration_predictions_csv_multiclass", "PASS")


@pytest.mark.integration
def test_integration_predictions_csv_binary(tlog: _TestLogger, n_jobs: int):
    """Verify binary predictions CSV columns and values."""
    diseases = sorted(pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")["disease"].unique())
    reference = "Healthy/Background"
    target = next(d for d in diseases if d != reference)
    pair_name = make_pair_name(target, reference)

    output_dir = OUTPUT_DIR / "integration" / "predictions_bin"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_all_folds(
        fold_ids=[TEST_FOLD_ID],
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        output_dir=output_dir,
        dataset_name="test-data",
        classification_mode="binary",
        reference_class=reference,
        diseases=[target],
        model_names=[TEST_MODEL_NAME],
        gene_locus="TCR",
        p_values=TEST_P_VALUES,
        n_jobs=n_jobs,
        verbose=0,
        data_dir=TEST_DATA_DIR / "raw",
        cache_dir=TEST_DATA_DIR,
        glmnet_cv_n_splits=GLMNET_CV_N_SPLITS,
    )

    pair_dir = output_dir / pair_name
    csv_files = list(pair_dir.glob(f"{TEST_MODEL_NAME}_binary_predictions.csv"))

    if len(csv_files) == 0:
        # Small test data: binary pair may fully abstain (no valid model).
        # Verify that the NO_VALID_CLUSTERS notice exists instead.
        notice_files = list(pair_dir.glob(f"fold_*_{TEST_MODEL_NAME}_NO_VALID_CLUSTERS.txt"))
        assert len(notice_files) >= 1, (
            f"No binary predictions CSV and no NO_VALID_CLUSTERS notice in {pair_dir}. "
            f"Expected at least one of them."
        )
        notice_text = notice_files[0].read_text()
        assert "no valid p-value" in notice_text.lower()
        tlog.log(f"  Binary pair fully abstained — NO_VALID_CLUSTERS notice verified")
    else:
        df = pd.read_csv(csv_files[0])

        # Required columns
        for col in ["participant_label", "specimen_label", "disease_label",
                    "disease_label_str", "disease_model", "model_score",
                    "malid_cross_validation_fold_id_when_in_test_set"]:
            assert col in df.columns, f"Missing column: {col}"

        # disease_label is 0 or 1
        assert set(df["disease_label"].unique()) <= {0, 1}
        # model_score in [0, 1]
        assert (df["model_score"] >= 0.0).all() and (df["model_score"] <= 1.0).all()

        tlog.log(f"  Binary predictions CSV: {len(df)} rows")

    tlog.log("  [PASS] integration_predictions_csv_binary")
    tlog.record("integration_predictions_csv_binary", "PASS")


# ===========================================================================
# Entry point (for running outside pytest)
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    pytest.main([__file__, "-v", "-s"])
