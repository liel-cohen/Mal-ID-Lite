"""Phase 6 tests: external evaluation (evaluate_external.py + _reporting.py).

Unit tests (fast, no training) cover the readiness/leakage checks, class alignment,
preprocessing-consistency, the model-source CLI guards, test-fold filtering, and the
rich-metrics computation. Integration tests (Model 1 only — cheap) train a tiny model
on the bundled tests/test_data and evaluate it: train-all ensemble, standalone
base-model, binary/multi-binary, --test-on-folds, and --model-fold-id (CV).

Uses the bundled tests/test_data/ (72 participants, 4 diseases, CV folds 0/1/2).
Output: tests/test_outputs/test_evaluate_external/. Expected runtime: ~1-2 min.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR

from malid_lite.evaluation import evaluate_external as ee
from malid_lite.evaluation import evaluate_external_reporting as rep
from malid_lite.training.training_utils import PROJECT_ROOT

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED = TEST_DATA_DIR / "embeddings"
REF = "Healthy/Background"


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_dataset(ds: str):
    d = PROJECT_ROOT / "trained_models" / ds
    if d.exists():
        shutil.rmtree(d)


def _train(ds, *, context, mode="multiclass", models=("1",), extra=()):
    """Train a tiny ensemble and return its ensemble dir (Model 1 by default)."""
    argv = [
        sys.executable, "-m", "malid_lite.training.train_ensemble",
        "--training-context", context, "--models", *models,
        "--classification-mode", mode, "--gene-locus", "TCR",
        "--dataset-name", ds, "--cache-dir", str(TEST_DATA_DIR),
        "--data-dir", str(TEST_RAW_DIR), "--model1-n-pcs", "10",
        "--metamodel-cv-n-splits", "2", "--n-jobs", "4", "--verbose", "0",
    ]
    if "3" in models:
        argv += ["--model3-embedding-dir", str(EMBED)]
    if mode in ("binary", "multi-binary"):
        argv += ["--reference-class", REF]
    argv += list(extra)
    subprocess.run(argv, check=True, capture_output=True, timeout=600)
    ctx_dir = "cv_ensemble" if context == "cv" else "train_all_ensemble"
    mode_dir = "binary" if mode in ("binary", "multi-binary") else "multiclass"
    return PROJECT_ROOT / "trained_models" / ds / ctx_dir / "ensemble" / "TCR" / mode_dir


def _run_eval(argv_extra):
    """Invoke evaluate_external.main() in-process with a patched argv."""
    argv = [
        "evaluate_external", "--test-cache-dir", str(TEST_DATA_DIR),
        "--test-dataset-name", "self", "--gene-locus", "TCR",
        "--n-jobs", "4", "--verbose", "0",
    ] + list(argv_extra)
    old = sys.argv
    try:
        sys.argv = argv
        ee.main()
    finally:
        sys.argv = old


# ===========================================================================
# Unit tests — readiness / leakage
# ===========================================================================


class TestLoadTrainedModel:
    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            ee.load_trained_model(tmp_path / "nope", is_ensemble=False)

    def test_no_summary(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a completed model"):
            ee.load_trained_model(tmp_path, is_ensemble=False)

    def test_not_complete(self, tmp_path):
        (tmp_path / "summary_x.json").write_text(json.dumps({"gene_locus": "TCR"}))
        with pytest.raises(ValueError, match="not a COMPLETED"):
            ee.load_trained_model(tmp_path, is_ensemble=False)

    def test_ensemble_leakage_rejected(self, tmp_path):
        # An ensemble whose base models were the leaky train_all must be rejected.
        (tmp_path / "summary_x.json").write_text(json.dumps({
            "training_complete": True, "gene_locus": "TCR",
            "metamodel_config": {"feature_columns": ["a"], "classes": ["X", "Y"],
                                 "models_included": [1]},
            "base_model_training_context": "train_all",  # leaky
        }))
        (tmp_path / "ridge_cv_metamodel.joblib").write_text("x")
        with pytest.raises(ValueError, match="leakage|train_all_ensemble"):
            ee.load_trained_model(tmp_path, is_ensemble=True)


# ===========================================================================
# Unit tests — class alignment
# ===========================================================================


class TestAlignTestClasses:
    def _meta(self, diseases):
        return pd.DataFrame({"specimen_label": [f"s{i}" for i in range(len(diseases))],
                             "disease": diseases})

    def test_exact_match(self):
        meta = self._meta(["A", "B", "A"])
        out, info = ee.align_test_classes(meta, ["A", "B"], allow_unknown_test_classes=False,
                                          pair_label="mc")
        assert len(out) == 3
        assert info["extra_test_classes"] == []

    def test_extra_class_errors(self):
        meta = self._meta(["A", "B", "C"])
        with pytest.raises(ValueError, match="never saw|allow-unknown-test-classes"):
            ee.align_test_classes(meta, ["A", "B"], allow_unknown_test_classes=False,
                                  pair_label="mc")

    def test_extra_class_filtered_with_flag(self):
        meta = self._meta(["A", "B", "C", "C"])
        out, info = ee.align_test_classes(meta, ["A", "B"], allow_unknown_test_classes=True,
                                          pair_label="mc")
        assert len(out) == 2  # the two C specimens dropped
        assert info["extra_test_classes"] == ["C"]

    def test_missing_training_class_proceeds(self):
        meta = self._meta(["A", "A"])  # B has no test support
        out, info = ee.align_test_classes(meta, ["A", "B"], allow_unknown_test_classes=False,
                                          pair_label="mc")
        assert len(out) == 2
        assert info["missing_training_classes"] == ["B"]

    def test_no_overlap_errors(self):
        meta = self._meta(["X", "Y"])
        with pytest.raises(ValueError, match="No overlap"):
            ee.align_test_classes(meta, ["A", "B"], allow_unknown_test_classes=True,
                                  pair_label="mc")


# ===========================================================================
# Unit tests — preprocessing consistency
# ===========================================================================


class _FakeLoader:
    def __init__(self, gene_locus="TCR", clone_id_params=None):
        self.gene_locus = gene_locus
        self._cip = clone_id_params or {"clone_id_computed": False}

    @property
    def clone_id_params(self):
        return self._cip


class TestPreprocessingConsistency:
    def test_locus_mismatch_errors(self):
        with pytest.raises(ValueError, match="gene_locus mismatch"):
            ee.check_preprocessing_consistency(_FakeLoader("BCR"), "TCR", {"clone_id_computed": True})

    def test_preexisting_clone_id_skips(self, caplog):
        # Pre-existing clone_id on either side -> skip comparison (no warning).
        loader = _FakeLoader("TCR", {"clone_id_computed": False})
        ee.check_preprocessing_consistency(loader, "TCR", {"clone_id_computed": True,
                                                           "clone_id_use_aa": True})
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_computed_mismatch_warns(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        loader = _FakeLoader("TCR", {"clone_id_computed": True, "clone_id_use_aa": True})
        ee.check_preprocessing_consistency(loader, "TCR", {"clone_id_computed": True,
                                                           "clone_id_use_aa": False})
        assert any("clone_id" in r.message for r in caplog.records)


# ===========================================================================
# Unit tests — rich metrics
# ===========================================================================


class TestRichMetrics:
    def test_multiclass_keys(self):
        rng = np.random.RandomState(0)
        classes = ["A", "B", "C"]
        y_true = np.array(["A", "B", "C", "A", "B", "C"])
        proba = rng.dirichlet([1, 1, 1], size=6)
        y_pred = np.array(classes)[proba.argmax(1)]
        m = rep.compute_rich_metrics(y_true, y_pred, proba, classes, reference_class=None,
                                     model_label="m", n_scored=6, n_abstained=0)
        for k in ("accuracy", "balanced_accuracy", "mcc", "classification_report",
                  "confusion_matrix_normalized", "top_confusions", "roc_curves", "pr_curves"):
            assert k in m

    def test_binary_operating_point(self):
        classes = ["Covid19", "Healthy/Background"]
        y_true = np.array(["Covid19", "Covid19", "Healthy/Background", "Healthy/Background"])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])
        m = rep.compute_rich_metrics(y_true, y_pred=np.array(classes)[proba.argmax(1)],
                                     y_proba=proba, classes=classes, reference_class="Healthy/Background",
                                     model_label="m", n_scored=4, n_abstained=0)
        op = m["binary_operating_point"]
        assert op["positive_class"] == "Covid19"
        assert 0.0 <= op["youden_optimal"]["sensitivity"] <= 1.0


# ===========================================================================
# Unit tests — CLI model-source guards (subprocess: exercises argparse errors)
# ===========================================================================


class TestModelSourceGuards:
    def _err(self, extra):
        r = subprocess.run(
            [sys.executable, "-m", "malid_lite.evaluation.evaluate_external",
             "--test-cache-dir", str(TEST_DATA_DIR)] + extra,
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode != 0
        return r.stderr

    def test_ensemble_and_model_dir_conflict(self):
        assert "mutually exclusive" in self._err(["--ensemble-dir", "/x", "--model1-dir", "/y"])

    def test_explicit_and_convention_conflict(self):
        assert "mutually exclusive" in self._err(["--ensemble-dir", "/x",
                                                  "--train-dataset-name", "z"])

    def test_no_model_specified(self):
        assert "No model specified" in self._err([])

    def test_abstention_strategy_standalone_errors(self):
        assert "only meaningful for an ensemble" in self._err(
            ["--model1-dir", "/x", "--model2-abstention-strategy", "fill_0.5"])

    def test_inference_only_with_allow_unknown_errors(self):
        assert "contradictory" in self._err(
            ["--ensemble-dir", "/x", "--inference-only", "--allow-unknown-test-classes"])

    def test_inline_embeddings_without_model3_errors(self):
        # Standalone Model-1 source (no Model 3) + --inline-embeddings → clear error,
        # checked up front (no model loading needed for the standalone peek).
        assert "does NOT include Model 3" in self._err(
            ["--model1-dir", "/x", "--inline-embeddings"])


class TestLoaderRequireDisease:
    """The Phase-6 loader change enabling label-free inference (--inference-only)."""

    def _write_no_disease_meta(self, tmp_path):
        md = pd.DataFrame({"participant_label": ["p1", "p2"],
                           "specimen_label": ["s1", "s2"]})
        p = tmp_path / "meta_nodisease.tsv"
        md.to_csv(p, sep="\t", index=False)
        return p

    def test_require_disease_false_loads(self, tmp_path):
        from malid_lite.dataloader import MalIDPublishedDataLoader
        mdp = self._write_no_disease_meta(tmp_path)
        ldr = MalIDPublishedDataLoader(data_dir=None, metadata_path=mdp,
                                       cache_dir=tmp_path / "cache", verbose=0,
                                       require_disease=False)
        assert len(ldr.metadata) == 2
        assert "disease" not in ldr.metadata.columns

    def test_require_disease_true_errors(self, tmp_path):
        from malid_lite.dataloader import MalIDPublishedDataLoader
        mdp = self._write_no_disease_meta(tmp_path)
        ldr = MalIDPublishedDataLoader(data_dir=None, metadata_path=mdp,
                                       cache_dir=tmp_path / "cache", verbose=0)
        with pytest.raises(ValueError, match="disease"):
            _ = ldr.metadata


# ===========================================================================
# Integration tests (Model 1 — cheap)
# ===========================================================================


@pytest.mark.integration
class TestExternalEvalIntegration:
    def test_train_all_ensemble_multiclass(self):
        ds = "test-data-ee-mc"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all")
            out = _out("mc")
            _run_eval(["--ensemble-dir", str(ens), "--output-dir", str(out)])
            r = json.loads(next(out.glob("results_*.json")).read_text())
            assert r["ensemble"] is not None
            assert r["base_models"]["model1"]["n_scored"] == 76
            assert (out / "figures").is_dir() and any((out / "figures").glob("*.png"))
            assert any(out.glob("predictions_*.csv"))
        finally:
            _cleanup_dataset(ds)

    def test_standalone_base_model(self):
        ds = "test-data-ee-standalone"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all")
            base1 = ens.parent.parent.parent / "base_models" / "TCR" / "model1" / "multiclass"
            out = _out("standalone")
            _run_eval(["--model1-dir", str(base1), "--output-dir", str(out)])
            r = json.loads(next(out.glob("results_*.json")).read_text())
            assert r["ensemble"] is None
            assert "model1" in r["base_models"]
        finally:
            _cleanup_dataset(ds)

    def test_multi_binary(self):
        ds = "test-data-ee-mb"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all", mode="multi-binary", models=("1", "3"))
            out = _out("mb")
            _run_eval(["--ensemble-dir", str(ens), "--test-embedding-dir", str(EMBED),
                       "--output-dir", str(out)])
            for disease in ("HIV", "Covid19", "T1D"):
                pair = out / f"{disease}_vs_Healthy_Background"
                assert any(pair.glob("results_*.json")), f"missing {pair}"
        finally:
            _cleanup_dataset(ds)

    def test_test_on_folds(self):
        ds = "test-data-ee-tof"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all")
            out = _out("tof")
            _run_eval(["--ensemble-dir", str(ens), "--test-on-folds", "2",
                       "--output-dir", str(out)])
            r = json.loads(next(out.glob("results_*.json")).read_text())
            assert r["n_test_specimens"] == 24  # fold 2 has 24 specimens
        finally:
            _cleanup_dataset(ds)

    def test_model_fold_id_cv(self):
        ds = "test-data-ee-cv"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="cv")
            out = _out("cvfold")
            _run_eval(["--ensemble-dir", str(ens), "--model-fold-id", "2",
                       "--test-on-folds", "2", "--output-dir", str(out)])
            r = json.loads(next(out.glob("results_*.json")).read_text())
            assert r["ensemble"] is not None
            assert r["n_test_specimens"] == 24
        finally:
            _cleanup_dataset(ds)

    def test_inference_only(self):
        # Inference-only: per-specimen predictions, NO metrics / true_disease / figures.
        ds = "test-data-ee-inf"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all")
            out = _out("inference")
            _run_eval(["--ensemble-dir", str(ens), "--inference-only", "--output-dir", str(out)])
            r = json.loads(next(out.glob("results_*.json")).read_text())
            assert r["inference_only"] is True
            assert r["ensemble"] is None and r["base_models"] == {}
            preds = pd.read_csv(next(out.glob("predictions_*.csv")))
            assert "true_disease" not in preds.columns
            assert "ensemble_predicted" in preds.columns
            assert not (out / "figures").exists()  # no metrics → no figures
        finally:
            _cleanup_dataset(ds)

    def test_inference_only_binary_model_errors(self):
        # Inference-only is multiclass-only; a binary model must error clearly (not crash
        # deep in predict_modelN's mode guard).
        ds = "test-data-ee-infbin"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all", mode="binary", models=("1",),
                         extra=["--diseases", "HIV"])
            out = _out("infbin")
            with pytest.raises(ValueError, match="only supported for MULTICLASS"):
                _run_eval(["--ensemble-dir", str(ens), "--inference-only",
                           "--output-dir", str(out)])
        finally:
            _cleanup_dataset(ds)

    def test_binary_operating_point_in_output(self):
        # Binary eval writes the sensitivity/specificity operating point.
        ds = "test-data-ee-bin"
        _cleanup_dataset(ds)
        try:
            ens = _train(ds, context="train_all", mode="binary", models=("1",),
                         extra=["--diseases", "HIV"])
            out = _out("binary")
            _run_eval(["--ensemble-dir", str(ens), "--output-dir", str(out)])
            pair_out = out / "HIV_vs_Healthy_Background"
            r = json.loads(next(pair_out.glob("results_*.json")).read_text())
            assert "binary_operating_point" in r["ensemble"]
            assert r["ensemble"]["binary_operating_point"]["positive_class"] == "HIV"
        finally:
            _cleanup_dataset(ds)
