#!/usr/bin/env python
"""Quick tests for Model 1 per-fold resume support.

Tests the resume helper functions (artifact detection, save/load round-trip,
metadata validation) and the resume behavior in _run_fold_loop (skip completed
folds, retrain incomplete folds, error on parameter mismatch, backward
compatibility with legacy artifacts).

Tests
-----
Unit tests (no data loader):
  1.  _get_fold_artifact_paths returns 4 paths with correct names
  2.  _check_fold_complete: all present + pkl >= 1KB = True
  3.  _check_fold_complete: missing file = False
  4.  _check_fold_complete: truncated pkl = False
  5.  _check_fold_has_legacy_artifacts: model + results but no predictions.pkl
  6.  _save_fold_predictions / _load_fold_results round-trip
  7.  _validate_fold_meta: matching params passes silently
  8.  _validate_fold_meta: model_name mismatch raises ValueError
  9.  _validate_fold_meta: training_context mismatch raises ValueError
  10. _validate_fold_meta: model_params key mismatch raises ValueError
      (also verifies backward compat: extra keys in current skip gracefully)
  11. _validate_fold_meta: missing _meta raises ValueError
  11b. _validate_fold_meta: fold_id mismatch raises ValueError
  11c. _validate_fold_meta: run-level params (classification_mode, disease_filter,
      reference_class, dataset_name) mismatch raises ValueError

Integration tests (real data, single fold):
  12. Full run fold 0 (multiclass), then resume — resumed fold is skipped
      and loaded results match the original run's results exactly
  13. Resume with incomplete artifacts — partial files are deleted before
      retraining to prevent mixing old and new artifacts
  14. Resume with legacy artifacts (no predictions.pkl) — fold is retrained
  15. Multi-binary resume round-trip: train 2 binary pairs (Covid19 and HIV
      vs Healthy/Background), then resume — resumed results match originals
      for every pair

Requirements
------------
- Fold cache built: cache/mal-id-orig-data/data_folds/fold_*.parquet
- python-glmnet installed
- All dependencies from requirements.txt

Expected runtime
----------------
- Unit tests: <5 seconds
- Integration tests: ~3-5 minutes (trains fold 0 in multiclass and
  2 binary pairs, then resumes each)

Output files
------------
All outputs saved to tests/test_outputs/test_model1_resume_quick/:
- test_model1_resume_quick_YYYYMMDD_HHMMSS.log   - Full log
- test_model1_resume_quick_YYYYMMDD_HHMMSS.json  - Structured results
- test_12_multiclass/                             - Multiclass model artifacts
- test_13_incomplete/                             - Incomplete artifact test
- test_14_mismatch/                               - Param mismatch test
- test_15_multi_binary/<pair_name>/               - Multi-binary model artifacts
"""

import importlib.util
import json
import logging
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Project root (tests/ -> project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage

# Import model-specific functions directly from the training script
_script = project_root / "malid_lite" / "training" / "train_model1.py"
_spec = importlib.util.spec_from_file_location("train_model1", _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_get_fold_artifact_paths = _module._get_fold_artifact_paths
_check_fold_complete = _module._check_fold_complete
_check_fold_has_legacy_artifacts = _module._check_fold_has_legacy_artifacts
_save_fold_predictions = _module._save_fold_predictions
_load_fold_results = _module._load_fold_results
_validate_fold_meta = _module._validate_fold_meta
_run_fold_loop = _module._run_fold_loop
_MIN_PKL_BYTES = _module._MIN_PKL_BYTES
make_pair_name = _module.make_pair_name


# ---------------------------------------------------------------------------
# Minimal logger (same pattern as other test scripts)
# ---------------------------------------------------------------------------

class TestLogger:
    """Writes to both console and file, and accumulates structured results."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, "a")
        self.results = {"tests": [], "start_time": datetime.now().isoformat()}

    def log(self, message: str, to_file_only: bool = False) -> None:
        self.file.write(message + "\n")
        self.file.flush()
        if not to_file_only:
            print(message)

    def add_result(self, test_name: str, status: str, details: dict = None) -> None:
        self.results["tests"].append({
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def close(self) -> Path:
        self.results["end_time"] = datetime.now().isoformat()
        self.file.close()
        results_file = self.log_file.with_suffix(".json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        return results_file


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

MODEL_NAME = "lasso_cv"
MODEL_PARAMS = {"gene_locus": "TCR", "n_pcs": 15, "l1_ratio": 1.0}
TRAINING_CONTEXT = "cv_single_model"
FOLD_ID = 0

# Output directory for unit test artifacts
_UNIT_TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem

# Full meta_model_params: model hyperparams + run-level settings (as saved in _meta)
META_MODEL_PARAMS = {
    **MODEL_PARAMS,
    "classification_mode": "multiclass",
    "diseases": None,
    "dataset_name": "mal-id-orig-data",
    "reference_class": None,
    "disease_filter": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_complete_fold(output_dir: Path, fold_id: int, model_name: str,
                             model_params: dict = None,
                             training_context: str = TRAINING_CONTEXT):
    """Create all 4 fold artifacts with realistic sizes for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _get_fold_artifact_paths(output_dir, fold_id, model_name)

    # model.pkl — needs to be >= _MIN_PKL_BYTES
    model_data = {"weights": np.random.randn(100).tolist(), "padding": "x" * 2000}
    with open(paths[0], "wb") as f:
        pickle.dump(model_data, f)

    # v_genes.json
    with open(paths[1], "w") as f:
        json.dump(["TRBV1", "TRBV2", "TRBV3"], f)

    # results.json
    eval_results = {
        "fold_id": fold_id, "model_name": model_name,
        "accuracy": 0.75, "mcc": 0.60,
        "auroc_ovo_weighted": 0.90,
        "n_scored": 50, "n_abstained": 0,
    }
    with open(paths[2], "w") as f:
        json.dump(eval_results, f)

    # predictions.pkl — with _meta
    # Use enough data so the pickled file exceeds _MIN_PKL_BYTES (1024)
    n_samples = 50
    params = model_params or MODEL_PARAMS
    raw_preds = {
        "y_true": np.array(["A", "B"] * (n_samples // 2)),
        "y_pred": np.array(["A", "B"] * (n_samples // 2)),
        "y_proba": np.random.rand(n_samples, 2),
        "classes": np.array(["A", "B"]),
    }
    predictions_rows = [
        {"specimen_label": f"s{i}", "true_disease": "A" if i % 2 == 0 else "B",
         "predicted_disease": "A"}
        for i in range(n_samples)
    ]
    preds_data = {
        "raw_preds": raw_preds,
        "predictions_rows": predictions_rows,
        "_meta": {
            "model_name": model_name,
            "model_params": params,
            "training_context": training_context,
            "fold_id": fold_id,
        },
    }
    with open(paths[3], "wb") as f:
        pickle.dump(preds_data, f)

    # Verify pkl sizes are above threshold
    for p in [paths[0], paths[3]]:
        assert p.stat().st_size >= _MIN_PKL_BYTES, (
            f"Test setup bug: {p.name} is {p.stat().st_size} bytes, "
            f"expected >= {_MIN_PKL_BYTES}"
        )

    return paths


def _get_test_output_dir(base_output_dir: Path, test_name: str) -> Path:
    """Create a clean test output subdirectory, removing stale artifacts from prior runs."""
    test_dir = base_output_dir / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_01_artifact_paths(tlog: TestLogger):
    """_get_fold_artifact_paths returns 4 paths with correct names."""
    tlog.log("\n1. _get_fold_artifact_paths returns correct paths")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_01_artifact_paths")

    paths = _get_fold_artifact_paths(tmp, fold_id=2, model_name="lasso_cv")
    assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"

    expected_names = [
        "fold_2_lasso_cv_model.pkl",
        "fold_2_lasso_cv_v_genes.json",
        "fold_2_lasso_cv_results.json",
        "fold_2_lasso_cv_predictions.pkl",
    ]
    actual_names = [p.name for p in paths]
    assert actual_names == expected_names, (
        f"Path names mismatch.\nExpected: {expected_names}\nActual:   {actual_names}"
    )

    # All paths should be under output_dir
    for p in paths:
        assert p.parent == tmp, f"Path {p} not under {tmp}"

    tlog.log("  PASS")
    tlog.add_result("artifact_paths", "PASS", {"paths": actual_names})


def test_02_check_fold_complete_all_present(tlog: TestLogger):
    """_check_fold_complete returns True when all 4 artifacts exist and pkls >= 1KB."""
    tlog.log("\n2. _check_fold_complete: all present + valid sizes = True")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_02_complete")

    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)
    assert _check_fold_complete(tmp, FOLD_ID, MODEL_NAME) is True
    tlog.log("  PASS")
    tlog.add_result("check_fold_complete_all_present", "PASS")


def test_03_check_fold_complete_missing_file(tlog: TestLogger):
    """_check_fold_complete returns False when any artifact is missing."""
    tlog.log("\n3. _check_fold_complete: missing file = False")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_03_missing")

    paths = _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)

    # Remove each file one at a time and verify incompleteness
    for i, path in enumerate(paths):
        path.unlink()
        assert _check_fold_complete(tmp, FOLD_ID, MODEL_NAME) is False, (
            f"Expected False after removing {path.name}"
        )
        # Recreate for next iteration
        _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)

    tlog.log("  PASS (tested removal of each of 4 artifacts)")
    tlog.add_result("check_fold_complete_missing_file", "PASS",
                    {"n_variants_tested": 4})


def test_04_check_fold_complete_truncated_pkl(tlog: TestLogger):
    """_check_fold_complete returns False when a pkl is below _MIN_PKL_BYTES."""
    tlog.log("\n4. _check_fold_complete: truncated pkl = False")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_04_truncated")

    paths = _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)
    pkl_paths = [p for p in paths if p.suffix == ".pkl"]
    assert len(pkl_paths) == 2, f"Expected 2 pkl files, got {len(pkl_paths)}"

    for pkl_path in pkl_paths:
        # Write a tiny file (well below threshold)
        pkl_path.write_bytes(b"x" * 10)
        assert pkl_path.stat().st_size < _MIN_PKL_BYTES
        assert _check_fold_complete(tmp, FOLD_ID, MODEL_NAME) is False, (
            f"Expected False with truncated {pkl_path.name} "
            f"({pkl_path.stat().st_size} bytes)"
        )
        # Restore
        _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)

    tlog.log("  PASS (tested truncation of model.pkl and predictions.pkl)")
    tlog.add_result("check_fold_complete_truncated_pkl", "PASS")


def test_05_check_fold_has_legacy_artifacts(tlog: TestLogger):
    """_check_fold_has_legacy_artifacts detects model + results without predictions.pkl."""
    tlog.log("\n5. _check_fold_has_legacy_artifacts: backward compat detection")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_05_legacy")

    # Start with complete fold — should NOT be legacy
    paths = _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)
    assert _check_fold_has_legacy_artifacts(tmp, FOLD_ID, MODEL_NAME) is False, (
        "Complete fold should not be detected as legacy"
    )

    # Remove predictions.pkl — now it's legacy
    preds_pkl = paths[3]
    assert preds_pkl.name.endswith("_predictions.pkl")
    preds_pkl.unlink()
    assert _check_fold_has_legacy_artifacts(tmp, FOLD_ID, MODEL_NAME) is True, (
        "Fold with model + results but no predictions.pkl should be legacy"
    )

    # Remove results.json too — not legacy (no results)
    results_json = paths[2]
    assert results_json.name.endswith("_results.json")
    results_json.unlink()
    assert _check_fold_has_legacy_artifacts(tmp, FOLD_ID, MODEL_NAME) is False, (
        "Fold with only model.pkl (no results) should not be legacy"
    )

    tlog.log("  PASS")
    tlog.add_result("check_fold_has_legacy_artifacts", "PASS")


def test_06_save_load_roundtrip(tlog: TestLogger):
    """_save_fold_predictions / _load_fold_results round-trip preserves data exactly."""
    tlog.log("\n6. save/load round-trip")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_06_roundtrip")

    # Also need a results JSON for _load_fold_results to read
    eval_results = {
        "fold_id": FOLD_ID, "model_name": MODEL_NAME,
        "accuracy": 0.8123, "mcc": 0.6543,
        "auroc_ovo_weighted": 0.9234,
        "n_scored": 42, "n_abstained": 0,
    }
    results_path = tmp / f"fold_{FOLD_ID}_{MODEL_NAME}_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f)

    # Save predictions
    raw_preds = {
        "y_true": np.array(["Covid19", "HIV", "Healthy/Background"]),
        "y_pred": np.array(["Covid19", "HIV", "Covid19"]),
        "y_proba": np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.5, 0.1, 0.4]]),
        "classes": np.array(["Covid19", "HIV", "Healthy/Background"]),
    }
    predictions_rows = [
        {"specimen_label": "s1", "true_disease": "Covid19", "predicted_disease": "Covid19"},
        {"specimen_label": "s2", "true_disease": "HIV", "predicted_disease": "HIV"},
        {"specimen_label": "s3", "true_disease": "Healthy/Background",
         "predicted_disease": "Covid19"},
    ]

    preds_path = _save_fold_predictions(
        tmp, FOLD_ID, MODEL_NAME,
        raw_preds=raw_preds,
        predictions_rows=predictions_rows,
        model_params=MODEL_PARAMS,
        training_context=TRAINING_CONTEXT,
    )
    assert preds_path.exists(), f"Predictions file not created at {preds_path}"
    assert preds_path.stat().st_size >= _MIN_PKL_BYTES, (
        f"Predictions file too small: {preds_path.stat().st_size} bytes"
    )

    # Load back
    loaded_eval, loaded_raw, loaded_rows = _load_fold_results(
        tmp, FOLD_ID, MODEL_NAME,
    )

    # Verify eval_results round-trip
    assert loaded_eval == eval_results, (
        f"eval_results mismatch.\nSaved: {eval_results}\nLoaded: {loaded_eval}"
    )

    # Verify raw_preds round-trip (numpy arrays)
    for key in ["y_true", "y_pred", "y_proba", "classes"]:
        np.testing.assert_array_equal(
            loaded_raw[key], raw_preds[key],
            err_msg=f"raw_preds['{key}'] mismatch",
        )

    # Verify predictions_rows round-trip
    assert loaded_rows == predictions_rows, (
        f"predictions_rows mismatch.\nSaved: {predictions_rows}\nLoaded: {loaded_rows}"
    )

    tlog.log("  PASS (eval_results, raw_preds arrays, predictions_rows all match)")
    tlog.add_result("save_load_roundtrip", "PASS")


def test_07_validate_meta_matching(tlog: TestLogger):
    """_validate_fold_meta passes silently when all params match."""
    tlog.log("\n7. _validate_fold_meta: matching params = no error")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_07_meta_match")

    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=MODEL_PARAMS,
                            training_context=TRAINING_CONTEXT)

    # Should not raise
    _validate_fold_meta(
        tmp, FOLD_ID, MODEL_NAME,
        current_model_params=MODEL_PARAMS,
        current_training_context=TRAINING_CONTEXT,
    )

    tlog.log("  PASS")
    tlog.add_result("validate_meta_matching", "PASS")


def test_08_validate_meta_model_name_mismatch(tlog: TestLogger):
    """_validate_fold_meta raises ValueError on model_name mismatch."""
    tlog.log("\n8. _validate_fold_meta: model_name mismatch = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_08_model_name")

    # Save with model_name="lasso_cv"
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)

    # Validate with a different model_name — the pkl was saved with "lasso_cv"
    # but we're now claiming the fold was trained by "ridge_cv".
    # We need to load the preds pkl and modify the model_name in _meta.
    preds_path = tmp / f"fold_{FOLD_ID}_{MODEL_NAME}_predictions.pkl"
    with open(preds_path, "rb") as f:
        data = pickle.load(f)
    data["_meta"]["model_name"] = "ridge_cv"
    with open(preds_path, "wb") as f:
        pickle.dump(data, f)

    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=MODEL_PARAMS,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for model_name mismatch")
    except ValueError as e:
        assert "model_name mismatch" in str(e).lower(), (
            f"Error message should mention model_name mismatch, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError: {e}")

    tlog.log("  PASS")
    tlog.add_result("validate_meta_model_name_mismatch", "PASS")


def test_09_validate_meta_training_context_mismatch(tlog: TestLogger):
    """_validate_fold_meta raises ValueError on training_context mismatch."""
    tlog.log("\n9. _validate_fold_meta: training_context mismatch = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_09_context")

    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            training_context="cv_single_model")

    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=MODEL_PARAMS,
            current_training_context="cv_ensemble",
        )
        raise AssertionError("Expected ValueError for training_context mismatch")
    except ValueError as e:
        assert "training_context mismatch" in str(e).lower(), (
            f"Error message should mention training_context mismatch, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError: {e}")

    tlog.log("  PASS")
    tlog.add_result("validate_meta_training_context_mismatch", "PASS")


def test_10_validate_meta_model_params_mismatch(tlog: TestLogger):
    """_validate_fold_meta raises ValueError on model_params key mismatch."""
    tlog.log("\n10. _validate_fold_meta: model_params mismatch = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_10_params")

    # Save with n_pcs=15
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME, model_params=MODEL_PARAMS)

    # Try to validate with n_pcs=20
    different_params = {**MODEL_PARAMS, "n_pcs": 20}
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_params,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for n_pcs mismatch")
    except ValueError as e:
        assert "n_pcs" in str(e), f"Error message should mention n_pcs, got: {e}"
        assert "15" in str(e) and "20" in str(e), (
            f"Error message should show both values, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError (n_pcs): {e}")

    # Try with l1_ratio mismatch
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME, model_params=MODEL_PARAMS)
    different_l1 = {**MODEL_PARAMS, "l1_ratio": 0.5}
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_l1,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for l1_ratio mismatch")
    except ValueError as e:
        assert "l1_ratio" in str(e), f"Error should mention l1_ratio, got: {e}"
        tlog.log(f"  Caught expected ValueError (l1_ratio): {e}")

    # Extra key in current but NOT in saved → skipped (backward compat).
    # Older artifacts won't have keys added later. Must not raise.
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME, model_params=MODEL_PARAMS)
    extra_key_params = {**MODEL_PARAMS, "new_param": "value"}
    _validate_fold_meta(
        tmp, FOLD_ID, MODEL_NAME,
        current_model_params=extra_key_params,
        current_training_context=TRAINING_CONTEXT,
    )
    tlog.log("  Extra key in current (not in saved): correctly skipped (no error)")

    tlog.log("  PASS (tested n_pcs, l1_ratio mismatches + backward compat skip)")
    tlog.add_result("validate_meta_model_params_mismatch", "PASS")


def test_11_validate_meta_missing_meta(tlog: TestLogger):
    """_validate_fold_meta raises ValueError when predictions.pkl has no _meta."""
    tlog.log("\n11. _validate_fold_meta: missing _meta = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_11_missing_meta")

    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME)

    # Overwrite predictions.pkl without _meta
    preds_path = tmp / f"fold_{FOLD_ID}_{MODEL_NAME}_predictions.pkl"
    data_no_meta = {
        "raw_preds": {"y_true": np.array(["A"]), "y_pred": np.array(["A"]),
                      "y_proba": np.array([[1.0]]), "classes": np.array(["A"])},
        "predictions_rows": [{"specimen_label": "s1"}],
    }
    with open(preds_path, "wb") as f:
        pickle.dump(data_no_meta, f)

    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=MODEL_PARAMS,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for missing _meta")
    except ValueError as e:
        assert "_meta" in str(e).lower() or "no _meta" in str(e), (
            f"Error message should mention missing _meta, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError: {e}")

    tlog.log("  PASS")
    tlog.add_result("validate_meta_missing_meta", "PASS")


def test_11b_validate_meta_fold_id_mismatch(tlog: TestLogger):
    """_validate_fold_meta raises ValueError on fold_id mismatch."""
    tlog.log("\n11b. _validate_fold_meta: fold_id mismatch = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_11b_fold_id")

    # Save fold 0 artifacts
    _make_fake_complete_fold(tmp, 0, MODEL_NAME)

    # Tamper: change fold_id in _meta to 1
    preds_path = tmp / f"fold_0_{MODEL_NAME}_predictions.pkl"
    with open(preds_path, "rb") as f:
        data = pickle.load(f)
    data["_meta"]["fold_id"] = 1
    with open(preds_path, "wb") as f:
        pickle.dump(data, f)

    try:
        _validate_fold_meta(
            tmp, 0, MODEL_NAME,
            current_model_params=MODEL_PARAMS,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for fold_id mismatch")
    except ValueError as e:
        assert "fold_id" in str(e).lower(), f"Error should mention fold_id, got: {e}"
        tlog.log(f"  Caught expected ValueError: {e}")

    tlog.log("  PASS")
    tlog.add_result("validate_meta_fold_id_mismatch", "PASS")


def test_11c_validate_meta_run_params_mismatch(tlog: TestLogger):
    """_validate_fold_meta catches mismatches in run-level params."""
    tlog.log("\n11c. _validate_fold_meta: run-level params mismatch = ValueError")
    tmp = _get_test_output_dir(_UNIT_TEST_OUTPUT_DIR, "test_11c_run_params")

    # Save with full meta_model_params
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=META_MODEL_PARAMS)

    # classification_mode mismatch
    different_mode = {**META_MODEL_PARAMS, "classification_mode": "binary"}
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_mode,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for classification_mode mismatch")
    except ValueError as e:
        assert "classification_mode" in str(e), (
            f"Error should mention classification_mode, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError (classification_mode): {e}")

    # disease_filter mismatch (None vs tuple)
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=META_MODEL_PARAMS)
    different_filter = {
        **META_MODEL_PARAMS,
        "disease_filter": ("Covid19", "Healthy/Background"),
    }
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_filter,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for disease_filter mismatch")
    except ValueError as e:
        assert "disease_filter" in str(e), (
            f"Error should mention disease_filter, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError (disease_filter): {e}")

    # reference_class mismatch
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=META_MODEL_PARAMS)
    different_ref = {**META_MODEL_PARAMS, "reference_class": "Healthy/Background"}
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_ref,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for reference_class mismatch")
    except ValueError as e:
        assert "reference_class" in str(e), (
            f"Error should mention reference_class, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError (reference_class): {e}")

    # dataset_name mismatch
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=META_MODEL_PARAMS)
    different_ds = {**META_MODEL_PARAMS, "dataset_name": "other-dataset"}
    try:
        _validate_fold_meta(
            tmp, FOLD_ID, MODEL_NAME,
            current_model_params=different_ds,
            current_training_context=TRAINING_CONTEXT,
        )
        raise AssertionError("Expected ValueError for dataset_name mismatch")
    except ValueError as e:
        assert "dataset_name" in str(e), (
            f"Error should mention dataset_name, got: {e}"
        )
        tlog.log(f"  Caught expected ValueError (dataset_name): {e}")

    # Matching full params — no error
    _make_fake_complete_fold(tmp, FOLD_ID, MODEL_NAME,
                            model_params=META_MODEL_PARAMS)
    _validate_fold_meta(
        tmp, FOLD_ID, MODEL_NAME,
        current_model_params=META_MODEL_PARAMS,
        current_training_context=TRAINING_CONTEXT,
    )
    tlog.log("  Full META_MODEL_PARAMS match: no error (correct)")

    tlog.log("  PASS")
    tlog.add_result("validate_meta_run_params_mismatch", "PASS")


# ---------------------------------------------------------------------------
# Integration tests (require real data loader)
# ---------------------------------------------------------------------------

def _create_loader():
    """Create a MalIDPublishedDataLoader for integration tests."""
    cache_dir = project_root / "cache" / "mal-id-orig-data"
    if not cache_dir.exists() or not any((cache_dir / "data_folds").glob("fold_*.parquet")):
        raise RuntimeError(
            f"Integration tests require fold cache at {cache_dir}. "
            "Run scripts/data/cache_and_report_all_data.py first."
        )

    loader = MalIDPublishedDataLoader(
        data_dir=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data_clean/airr_format_clean/TCR/"
        ),
        metadata_path=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv"
        ),
        gene_reference_path=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/tcrb_v_gene_cdrs.generated.tsv"
        ),
        gene_locus="TCR",
        cache_dir=cache_dir,
        verbose=0,
    )
    return loader


def test_12_resume_skips_completed_fold(tlog: TestLogger, base_output_dir: Path):
    """Full multiclass run + resume: resumed fold is skipped, results match original exactly."""
    tlog.log("\n12. Integration: multiclass resume round-trip (fold 0)")

    loader = _create_loader()
    test_dir = _get_test_output_dir(base_output_dir, "test_12_multiclass")

    fold_ids = [0]
    run_params = {
        "classification_mode": "multiclass",
        "diseases": None,
        "dataset_name": "mal-id-orig-data",
        "reference_class": None,
    }

    # --- Original run (resume=False) ---
    tlog.log("  Running original training (fold 0, resume=False)...")
    t0 = time.time()
    orig_results, orig_agg = _run_fold_loop(
        loader=loader,
        fold_ids=fold_ids,
        output_dir=test_dir,
        model_name=MODEL_NAME,
        model_params=MODEL_PARAMS,
        verbose=0,
        training_context=TRAINING_CONTEXT,
        resume=False,
        run_params=run_params,
    )
    orig_time = time.time() - t0

    assert len(orig_results) == 1, f"Expected 1 fold result, got {len(orig_results)}"
    assert _check_fold_complete(test_dir, 0, MODEL_NAME), "Fold should be complete after run"

    # Record original metrics
    orig_accuracy = orig_results[0]["accuracy"]
    orig_mcc = orig_results[0]["mcc"]
    orig_auroc = orig_results[0].get("auroc_ovo_weighted")
    auroc_str = f"{orig_auroc:.4f}" if orig_auroc is not None else "N/A"
    tlog.log(
        f"  Original: accuracy={orig_accuracy:.4f} mcc={orig_mcc:.4f} "
        f"auroc_ovo={auroc_str} ({orig_time:.1f}s)"
    )

    # --- Resume run (resume=True) ---
    tlog.log("  Running resume (fold 0, resume=True)...")
    t0 = time.time()
    resume_results, resume_agg = _run_fold_loop(
        loader=loader,
        fold_ids=fold_ids,
        output_dir=test_dir,
        model_name=MODEL_NAME,
        model_params=MODEL_PARAMS,
        verbose=0,
        training_context=TRAINING_CONTEXT,
        resume=True,
        run_params=run_params,
    )
    resume_time = time.time() - t0

    assert len(resume_results) == 1, f"Expected 1 fold result, got {len(resume_results)}"

    # Verify resumed results match original exactly
    resume_accuracy = resume_results[0]["accuracy"]
    resume_mcc = resume_results[0]["mcc"]
    resume_auroc = resume_results[0].get("auroc_ovo_weighted")

    assert orig_accuracy == resume_accuracy, (
        f"Accuracy mismatch: orig={orig_accuracy}, resume={resume_accuracy}"
    )
    assert orig_mcc == resume_mcc, (
        f"MCC mismatch: orig={orig_mcc}, resume={resume_mcc}"
    )
    assert orig_auroc == resume_auroc, (
        f"AUROC mismatch: orig={orig_auroc}, resume={resume_auroc}"
    )
    resume_auroc_str = f"{resume_auroc:.4f}" if resume_auroc is not None else "N/A"
    tlog.log(
        f"  Resumed:  accuracy={resume_accuracy:.4f} mcc={resume_mcc:.4f} "
        f"auroc_ovo={resume_auroc_str} ({resume_time:.1f}s)"
    )

    # Verify aggregated metrics match too
    orig_agg_metrics = orig_agg[MODEL_NAME]
    resume_agg_metrics = resume_agg[MODEL_NAME]
    assert orig_agg_metrics["accuracy_global"] == resume_agg_metrics["accuracy_global"], (
        "Aggregated accuracy_global mismatch"
    )

    tlog.log("  PASS (all metrics match between original and resumed run)")
    tlog.add_result("resume_skips_completed_fold", "PASS", {
        "orig_accuracy": orig_accuracy,
        "resume_accuracy": resume_accuracy,
        "output_dir": str(test_dir),
    })


def test_13_resume_retrains_incomplete_fold(tlog: TestLogger, base_output_dir: Path):
    """Resume with incomplete artifacts: partial files cleaned up, fold retrained."""
    tlog.log("\n13. Integration: resume retrains incomplete fold")

    loader = _create_loader()
    test_dir = _get_test_output_dir(base_output_dir, "test_13_incomplete")

    fold_ids = [0]

    # First do a real run to get valid artifacts
    tlog.log("  Running original training (fold 0)...")
    orig_results, _ = _run_fold_loop(
        loader=loader,
        fold_ids=fold_ids,
        output_dir=test_dir,
        model_name=MODEL_NAME,
        model_params=MODEL_PARAMS,
        verbose=0,
        training_context=TRAINING_CONTEXT,
        resume=False,
    )
    orig_accuracy = orig_results[0]["accuracy"]

    # Delete predictions.pkl to make it incomplete (legacy-like)
    preds_path = test_dir / f"fold_0_{MODEL_NAME}_predictions.pkl"
    assert preds_path.exists()
    preds_path.unlink()
    assert not _check_fold_complete(test_dir, 0, MODEL_NAME), (
        "Fold should be incomplete after removing predictions.pkl"
    )
    assert _check_fold_has_legacy_artifacts(test_dir, 0, MODEL_NAME), (
        "Should be detected as legacy artifacts"
    )

    # Resume — should retrain the fold
    tlog.log("  Resuming (fold 0 is incomplete — should retrain)...")
    resume_results, _ = _run_fold_loop(
        loader=loader,
        fold_ids=fold_ids,
        output_dir=test_dir,
        model_name=MODEL_NAME,
        model_params=MODEL_PARAMS,
        verbose=0,
        training_context=TRAINING_CONTEXT,
        resume=True,
    )

    assert len(resume_results) == 1, f"Expected 1 fold result, got {len(resume_results)}"

    # After retraining, fold should now be complete
    assert _check_fold_complete(test_dir, 0, MODEL_NAME), (
        "Fold should be complete after resume retrained it"
    )

    # Accuracy should be the same (deterministic model on same data)
    resume_accuracy = resume_results[0]["accuracy"]
    assert orig_accuracy == resume_accuracy, (
        f"Accuracy should match after retrain: orig={orig_accuracy}, "
        f"resume={resume_accuracy}"
    )

    tlog.log(
        f"  Retrained accuracy={resume_accuracy:.4f} matches original={orig_accuracy:.4f}"
    )
    tlog.log("  PASS (incomplete fold retrained successfully)")
    tlog.add_result("resume_retrains_incomplete_fold", "PASS", {
        "accuracy_match": orig_accuracy == resume_accuracy,
        "output_dir": str(test_dir),
    })


def test_14_resume_param_mismatch_integration(tlog: TestLogger, base_output_dir: Path):
    """Resume with parameter mismatch raises ValueError (integration)."""
    tlog.log("\n14. Integration: resume with parameter mismatch raises ValueError")

    loader = _create_loader()
    test_dir = _get_test_output_dir(base_output_dir, "test_14_mismatch")

    fold_ids = [0]

    # Train with n_pcs=15
    tlog.log("  Running original training (fold 0, n_pcs=15)...")
    _run_fold_loop(
        loader=loader,
        fold_ids=fold_ids,
        output_dir=test_dir,
        model_name=MODEL_NAME,
        model_params=MODEL_PARAMS,
        verbose=0,
        training_context=TRAINING_CONTEXT,
        resume=False,
    )
    assert _check_fold_complete(test_dir, 0, MODEL_NAME)

    # Try to resume with different n_pcs
    different_params = {**MODEL_PARAMS, "n_pcs": 20}
    tlog.log("  Resuming with n_pcs=20 (should error)...")
    try:
        _run_fold_loop(
            loader=loader,
            fold_ids=fold_ids,
            output_dir=test_dir,
            model_name=MODEL_NAME,
            model_params=different_params,
            verbose=0,
            training_context=TRAINING_CONTEXT,
            resume=True,
        )
        raise AssertionError("Expected ValueError for n_pcs mismatch on resume")
    except ValueError as e:
        assert "n_pcs" in str(e), f"Error should mention n_pcs, got: {e}"
        tlog.log(f"  Caught expected ValueError: {e}")

    tlog.log("  PASS")
    tlog.add_result("resume_param_mismatch_integration", "PASS", {
        "output_dir": str(test_dir),
    })


def test_15_resume_multi_binary(tlog: TestLogger, base_output_dir: Path):
    """Multi-binary resume: original run and resumed run produce identical per-pair results."""
    tlog.log("\n15. Integration: multi-binary resume round-trip (2 pairs, fold 0)")

    loader = _create_loader()
    test_dir = _get_test_output_dir(base_output_dir, "test_15_multi_binary")

    fold_ids = [0]
    reference_class = "Healthy/Background"
    diseases = ["Covid19", "HIV"]

    run_params = {
        "classification_mode": "multi-binary",
        "diseases": sorted(diseases),
        "dataset_name": "mal-id-orig-data",
        "reference_class": reference_class,
    }

    # --- Original run (resume=False) for all pairs ---
    orig_results_by_pair = {}
    for disease in diseases:
        pair_name = make_pair_name(disease, reference_class)
        pair_dir = test_dir / pair_name
        disease_filter = (disease, reference_class)

        tlog.log(f"  Training {pair_name} (original, resume=False)...")
        t0 = time.time()
        fold_results, agg = _run_fold_loop(
            loader=loader,
            fold_ids=fold_ids,
            output_dir=pair_dir,
            model_name=MODEL_NAME,
            model_params=MODEL_PARAMS,
            verbose=0,
            disease_filter=disease_filter,
            training_context=TRAINING_CONTEXT,
            resume=False,
            run_params=run_params,
        )
        elapsed = time.time() - t0

        assert len(fold_results) == 1, (
            f"Expected 1 fold result for {pair_name}, got {len(fold_results)}"
        )
        assert _check_fold_complete(pair_dir, 0, MODEL_NAME), (
            f"Fold should be complete after training {pair_name}"
        )

        orig = fold_results[0]
        orig_results_by_pair[pair_name] = orig
        auroc_val = orig.get("auroc_binary")
        auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A"
        tlog.log(
            f"    accuracy={orig['accuracy']:.4f} mcc={orig['mcc']:.4f} "
            f"auroc_binary={auroc_str} ({elapsed:.1f}s)"
        )

    # --- Resume run (resume=True) for all pairs ---
    tlog.log("  Resuming all pairs (resume=True)...")
    for disease in diseases:
        pair_name = make_pair_name(disease, reference_class)
        pair_dir = test_dir / pair_name
        disease_filter = (disease, reference_class)

        t0 = time.time()
        resume_results, resume_agg = _run_fold_loop(
            loader=loader,
            fold_ids=fold_ids,
            output_dir=pair_dir,
            model_name=MODEL_NAME,
            model_params=MODEL_PARAMS,
            verbose=0,
            disease_filter=disease_filter,
            training_context=TRAINING_CONTEXT,
            resume=True,
            run_params=run_params,
        )
        elapsed = time.time() - t0

        assert len(resume_results) == 1, (
            f"Expected 1 fold result for {pair_name}, got {len(resume_results)}"
        )

        orig = orig_results_by_pair[pair_name]
        resumed = resume_results[0]

        # Compare all numeric metric keys present in both
        metric_keys = [
            "accuracy", "mcc", "log_loss", "n_scored", "n_abstained",
            "auroc_binary", "auprc_binary",
            "auroc_ovo_weighted", "auprc_ovo_weighted",
        ]
        for key in metric_keys:
            if key in orig:
                assert orig[key] == resumed.get(key), (
                    f"{pair_name}: {key} mismatch: orig={orig[key]}, "
                    f"resume={resumed.get(key)}"
                )

        # Also verify confusion matrix matches
        assert orig["confusion_matrix"] == resumed["confusion_matrix"], (
            f"{pair_name}: confusion_matrix mismatch"
        )

        auroc_val = resumed.get("auroc_binary")
        auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A"
        tlog.log(
            f"  {pair_name}: accuracy={resumed['accuracy']:.4f} "
            f"mcc={resumed['mcc']:.4f} auroc_binary={auroc_str} "
            f"-- MATCH ({elapsed:.1f}s)"
        )

    tlog.log("  PASS (all pairs match between original and resumed run)")
    tlog.add_result("resume_multi_binary", "PASS", {
        "pairs_tested": list(orig_results_by_pair.keys()),
        "output_dir": str(test_dir),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).stem
    output_dir = Path(__file__).parent / "test_outputs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{script_name}_{timestamp}.log"

    # Configure Python logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    tlog = TestLogger(log_file)

    tlog.log("=" * 60)
    tlog.log("MODEL 1 RESUME TESTS")
    tlog.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tlog.log("=" * 60)

    n_pass = 0
    n_fail = 0

    # Unit tests (no data loader needed)
    unit_tests = [
        test_01_artifact_paths,
        test_02_check_fold_complete_all_present,
        test_03_check_fold_complete_missing_file,
        test_04_check_fold_complete_truncated_pkl,
        test_05_check_fold_has_legacy_artifacts,
        test_06_save_load_roundtrip,
        test_07_validate_meta_matching,
        test_08_validate_meta_model_name_mismatch,
        test_09_validate_meta_training_context_mismatch,
        test_10_validate_meta_model_params_mismatch,
        test_11_validate_meta_missing_meta,
        test_11b_validate_meta_fold_id_mismatch,
        test_11c_validate_meta_run_params_mismatch,
    ]

    tlog.log("\n--- Unit Tests ---")
    for test_fn in unit_tests:
        try:
            test_fn(tlog)
            n_pass += 1
        except Exception as e:
            n_fail += 1
            tlog.log(f"  FAIL: {e}")
            tlog.add_result(test_fn.__name__, "FAIL", {"error": str(e)})
            import traceback
            tlog.log(traceback.format_exc(), to_file_only=True)

    # Integration tests (require data loader)
    integration_tests = [
        test_12_resume_skips_completed_fold,
        test_13_resume_retrains_incomplete_fold,
        test_14_resume_param_mismatch_integration,
        test_15_resume_multi_binary,
    ]

    tlog.log("\n--- Integration Tests ---")
    for test_fn in integration_tests:
        try:
            test_fn(tlog, output_dir)
            n_pass += 1
        except RuntimeError as e:
            # Missing cache / data — skip gracefully
            tlog.log(f"  SKIP ({test_fn.__name__}): {e}")
            tlog.add_result(test_fn.__name__, "SKIP", {"reason": str(e)})
        except Exception as e:
            n_fail += 1
            tlog.log(f"  FAIL: {e}")
            tlog.add_result(test_fn.__name__, "FAIL", {"error": str(e)})
            import traceback
            tlog.log(traceback.format_exc(), to_file_only=True)

    # Summary
    tlog.log(f"\n{'=' * 60}")
    if n_fail == 0:
        tlog.log(f"ALL TESTS PASSED ({n_pass}/{n_pass})")
    else:
        tlog.log(f"TESTS: {n_pass} passed, {n_fail} failed")
    tlog.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tlog.log("=" * 60)

    results_file = tlog.close()

    rel_output_dir = output_dir.relative_to(Path(__file__).parent)
    print(f"\nTest outputs saved to: {rel_output_dir}/")
    print(f"  - Log file: {log_file.name}")
    print(f"  - Results JSON: {results_file.name}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
