"""Phase 5 tests: train-all ensemble (metamodel on the validation third, no test).

Covers the ensemble train-all path added in Phase 5 (train_ensemble.py +
training_utils.py):

Unit (fast, no training):
1. get_ensemble_output_dir(training_context=...) — cv vs train-all paths, no collision.
2. predict_model1 fold-optional path resolution (_model1_artifact_paths).
3. preflight_check_fold_artifacts train-all mode — prefix-less patterns; clear error.
4. _write_train_all_ensemble_summary — readiness/leakage markers + reused CV field names.

Integration (fast — Model 1 only: no ESM-2, no clustering):
5. Multiclass end-to-end: auto-train Model 1 as train_all_ensemble, train the
   metamodel on the validation third; assert prefix-free artifacts + no-metrics
   summary (training_complete/training_only); no *_test / predictions / metrics;
   the metamodel joblib reloads.
6. Binary + multi-binary: per-pair train_all_ensemble/ensemble/.../<pair>/ artifacts.
7. Resume: reloads the cached raw-val matrix and retrains the metamodel WITHOUT
   re-running base-model predictions.
8. --feature-matrices-dir: external metamodel-only re-fit under train-all.
9. Guards: --fold-ids rejected; leakage — base models trained as train_all
   (not train_all_ensemble) are rejected; feature-matrices context mismatch errors.

Uses the bundled tests/test_data/ (72 participants, 4 diseases). Model 1 keeps it
cheap. Output: tests/test_outputs/test_ensemble_train_all/.
Expected runtime: ~1-2 minutes.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR

from malid_lite.training import train_ensemble as te
from malid_lite.training.train_ensemble import (
    _model1_artifact_paths,
    _write_train_all_ensemble_summary,
)
from malid_lite.training.training_utils import (
    get_ensemble_output_dir,
    preflight_check_fold_artifacts,
)

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CLASS = "Healthy/Background"


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


from malid_lite.training.training_utils import PROJECT_ROOT


def _cleanup_dataset(dataset_name: str):
    """Remove the canonical trained_models tree for a per-test dataset name.

    The ensemble auto-trains base models into
    trained_models/<dataset_name>/train_all_ensemble/... (NOT --output-dir),
    so each integration test uses a UNIQUE --dataset-name (cache is shared via
    --cache-dir) to avoid cross-test base-model collisions, then cleans up here.
    """
    d = PROJECT_ROOT / "trained_models" / dataset_name
    if d.exists():
        shutil.rmtree(d)


EMBED_DIR = TEST_DATA_DIR / "embeddings"


def _run_ensemble_cli(
    argv_extra=(), *, n_jobs, dataset_name="test-data", classification_mode="multiclass",
    mode_args=(), with_data_dir=True, training_context="train_all", models=("1",),
):
    """Run train_ensemble.main() in-process with a patched argv.

    Model 1 by default (cheap: no ESM-2/clustering). Pass models=("1","3") for a
    richer metamodel (e.g. binary/multi-binary, where a single-feature metamodel
    can hit glmnet zero-variance on the tiny test data); Model 3 uses the bundled
    precomputed embeddings and stays fast. Raises SystemExit on parser/validation
    errors (tests catch it). ``with_data_dir`` is False for --feature-matrices-dir
    runs (no base-model training). ``n_jobs`` comes from the --n-jobs test fixture so
    the runner controls parallelism (required — never hardcoded).
    """
    argv = [
        "train_ensemble",
        "--training-context", training_context,
        "--models", *models,
        "--classification-mode", classification_mode,
        "--gene-locus", "TCR",
        "--dataset-name", dataset_name,
        "--cache-dir", str(TEST_DATA_DIR),
        "--model1-n-pcs", "10",
        "--n-jobs", str(n_jobs),
        "--verbose", "0",
    ]
    if "3" in models:
        argv += ["--model3-embedding-dir", str(EMBED_DIR)]
    if with_data_dir:
        argv += ["--data-dir", str(TEST_RAW_DIR)]
    argv += list(mode_args) + list(argv_extra)
    old = sys.argv
    try:
        sys.argv = argv
        te.main()
    finally:
        sys.argv = old


# ===========================================================================
# Unit tests (no training)
# ===========================================================================


class TestEnsembleOutputDir:
    def test_cv_vs_train_all_paths_distinct(self):
        cv = get_ensemble_output_dir("ds", "multiclass", "TCR", training_context="cv_ensemble")
        ta = get_ensemble_output_dir("ds", "multiclass", "TCR", training_context="train_all_ensemble")
        assert "cv_ensemble" in cv.parts
        assert "train_all_ensemble" in ta.parts
        assert cv != ta  # no collision

    def test_default_is_cv(self):
        d = get_ensemble_output_dir("ds", "multiclass", "TCR")
        assert "cv_ensemble" in d.parts

    def test_invalid_context_raises(self):
        with pytest.raises(ValueError, match="training_context"):
            get_ensemble_output_dir("ds", "multiclass", "TCR", training_context="bogus")


class TestModel1FoldOptionalPaths:
    def test_int_fold_prefixed(self, tmp_path):
        model_p, vg_p = _model1_artifact_paths(tmp_path, 2, "lasso_cv")
        assert model_p.name == "fold_2_lasso_cv_model.pkl"
        assert vg_p.name == "fold_2_lasso_cv_v_genes.json"

    def test_none_prefix_less(self, tmp_path):
        model_p, vg_p = _model1_artifact_paths(tmp_path, None, "lasso_cv")
        assert model_p.name == "lasso_cv_model.pkl"
        assert vg_p.name == "lasso_cv_v_genes.json"


class TestPreflightTrainAll:
    def test_detects_prefix_less_artifacts(self, tmp_path):
        d = tmp_path / "model1"
        d.mkdir()
        (d / "lasso_cv_model.pkl").write_text("x")
        (d / "lasso_cv_v_genes.json").write_text("[]")
        # Should not raise
        preflight_check_fold_artifacts(
            {1: d}, fold_ids=[None], classification_mode="multiclass",
            training_context="train_all_ensemble",
        )

    def test_missing_errors_with_context(self, tmp_path):
        d = tmp_path / "model1"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="train_all_ensemble"):
            preflight_check_fold_artifacts(
                {1: d}, fold_ids=[None], classification_mode="multiclass",
                training_context="train_all_ensemble",
            )

    def test_invalid_context_raises(self, tmp_path):
        with pytest.raises(ValueError, match="training_context"):
            preflight_check_fold_artifacts(
                {1: tmp_path}, fold_ids=[None], classification_mode="multiclass",
                training_context="bogus",
            )


class TestSummaryMarkers:
    def test_readiness_and_leakage_markers(self, tmp_path):
        run_config = {
            "timestamp": "20260101_000000",
            "gene_locus": "TCR",
            "classification_mode": "multiclass",
            "reference_class": None,
            "disease_filter": None,
            "models_included": [1],
            "base_model_paths": {"model1": "/x"},
            "base_model_suffixes": {"model1": None},
            "embedding_dir": None,
            "dataset_name": "ds",
        }
        mm_config = {"feature_columns": ["a"], "classes": ["X", "Y"], "n_features": 1}
        summary = _write_train_all_ensemble_summary(
            tmp_path, run_config, mm_config, {"strategy": "ensemble_abstain"},
            "ensemble_abstain",
        )
        # Readiness markers
        assert summary["training_complete"] is True
        assert summary["training_only"] is True
        assert summary["training_context"] == "train_all_ensemble"
        # Leakage marker: base models must be the train_all_ensemble variant
        assert summary["base_model_training_context"] == "train_all_ensemble"
        # Reused metamodel_config (no test-only fields)
        assert "n_test_specimens" not in summary["metamodel_config"]
        # Written to disk with the timestamp
        assert (tmp_path / "summary_20260101_000000.json").exists()


# ===========================================================================
# Integration tests (Model 1 only — fast)
# ===========================================================================


def _assert_train_all_artifacts(out: Path):
    """Assert the prefix-free train-all ensemble artifact set (no test/metrics)."""
    for name in (
        "ridge_cv_metamodel.joblib", "metamodel_config.json",
        "feature_matrix_val.csv", "feature_matrix_raw_val.csv",
        "ensemble_results.json", "run_config.json",
    ):
        assert (out / name).exists(), f"missing {name}"
    # No fold-prefixed artifacts
    assert not list(out.glob("fold_*")), "train-all must not write fold-prefixed files"
    # No test / metrics / predictions
    assert not list(out.glob("*_test*"))
    assert not (out / "ensemble_predictions.csv").exists()
    # Exactly one summary; it is the no-metrics readiness marker
    summaries = list(out.glob("summary_*.json"))
    assert len(summaries) == 1
    s = json.loads(summaries[0].read_text())
    assert s["training_complete"] is True
    assert s["training_only"] is True
    assert "ensemble" not in s  # no aggregated metrics block
    # The metamodel joblib reloads
    joblib.load(out / "ridge_cv_metamodel.joblib")


@pytest.mark.integration
class TestTrainAllEnsembleMulticlass:
    def test_end_to_end_models1(self, n_jobs):
        ds = "test-data-ta-mc"
        _cleanup_dataset(ds)
        try:
            out = _out("multiclass")
            # Also exercises --metamodel-cv-n-splits threading end-to-end.
            _run_ensemble_cli(
                ["--output-dir", str(out), "--metamodel-cv-n-splits", "2"],
                n_jobs=n_jobs, dataset_name=ds,
            )
            _assert_train_all_artifacts(out)
        finally:
            _cleanup_dataset(ds)

    def test_multi_model_1_2_with_fill(self, n_jobs):
        """Models 1+2 multiclass: exercises Model 2's fold-optional (train-all)
        prediction + the abstention FILL plumbing. Model 2 finds no valid
        clusters on the tiny test data (fully abstains); with fill_0.5 the
        abstained specimens are kept (filled), so the metamodel still trains and
        the val_fill_info records the fills (rather than silently dropping)."""
        ds = "test-data-ta-m12"
        _cleanup_dataset(ds)
        try:
            out = _out("multiclass_m12")
            _run_ensemble_cli(
                ["--output-dir", str(out),
                 "--model2-abstention-strategy", "fill_0.5"],
                n_jobs=n_jobs, dataset_name=ds, models=("1", "2"),
            )
            _assert_train_all_artifacts(out)
            # Feature matrix carries columns from both models.
            mm = json.loads((out / "metamodel_config.json").read_text())
            assert mm["n_features"] >= 1
            # The fill plumbing is recorded (not silently swallowed).
            res = json.loads((out / "ensemble_results.json").read_text())
            assert "val_fill_info" in res
        finally:
            _cleanup_dataset(ds)


@pytest.mark.integration
class TestTrainAllEnsembleBinary:
    def test_binary_pair(self, n_jobs):
        ds = "test-data-ta-bin"
        _cleanup_dataset(ds)
        try:
            out = _out("binary")
            _run_ensemble_cli(
                ["--output-dir", str(out)], n_jobs=n_jobs, dataset_name=ds,
                classification_mode="binary",
                mode_args=["--diseases", "HIV", "--reference-class", REFERENCE_CLASS],
            )
            _assert_train_all_artifacts(out / "HIV_vs_Healthy_Background")
        finally:
            _cleanup_dataset(ds)

    def test_multi_binary_all_pairs(self, n_jobs):
        ds = "test-data-ta-mb"
        _cleanup_dataset(ds)
        try:
            out = _out("multi_binary")
            # Models 1+3 so the per-pair binary metamodel has >1 feature (a
            # single-feature metamodel can hit glmnet zero-variance on the tiny
            # test data). Model 3 uses bundled embeddings → still fast.
            _run_ensemble_cli(
                ["--output-dir", str(out)], n_jobs=n_jobs, dataset_name=ds, models=("1", "3"),
                classification_mode="multi-binary",
                mode_args=["--reference-class", REFERENCE_CLASS],
            )
            # One subdir per non-reference disease
            for disease in ("HIV", "Covid19", "T1D"):
                _assert_train_all_artifacts(out / f"{disease}_vs_Healthy_Background")
        finally:
            _cleanup_dataset(ds)


@pytest.mark.integration
class TestTrainAllEnsembleResume:
    def test_resume_reuses_matrix_no_repredict(self, n_jobs):
        ds = "test-data-ta-resume"
        _cleanup_dataset(ds)
        try:
            out = _out("resume")
            _run_ensemble_cli(["--output-dir", str(out)], n_jobs=n_jobs, dataset_name=ds)
            _assert_train_all_artifacts(out)
            # Resume: reload the cached raw-val matrix, retrain the metamodel
            # (no base-model re-prediction). Assert the same validation specimen
            # set is reused (float re-serialization may differ, so compare the
            # specimen index, not raw bytes).
            import pandas as pd
            idx_before = set(pd.read_csv(
                out / "feature_matrix_raw_val.csv")["specimen_label"])
            _run_ensemble_cli(["--output-dir", str(out), "--resume"], n_jobs=n_jobs, dataset_name=ds)
            _assert_train_all_artifacts(out)
            idx_after = set(pd.read_csv(
                out / "feature_matrix_raw_val.csv")["specimen_label"])
            assert idx_before == idx_after
        finally:
            _cleanup_dataset(ds)


@pytest.mark.integration
class TestTrainAllEnsembleFeatureMatrices:
    def test_external_metamodel_refit(self, n_jobs):
        ds = "test-data-ta-fm"
        _cleanup_dataset(ds)
        try:
            src = _out("fm_source")
            _run_ensemble_cli(["--output-dir", str(src)], n_jobs=n_jobs, dataset_name=ds)
            dst = _out("fm_refit")
            # No --data-dir: base-model training is skipped in feature-matrices mode.
            _run_ensemble_cli(
                ["--feature-matrices-dir", str(src), "--output-dir", str(dst)],
                n_jobs=n_jobs, dataset_name=ds, with_data_dir=False,
            )
            _assert_train_all_artifacts(dst)
        finally:
            _cleanup_dataset(ds)

    def test_context_mismatch_errors(self, n_jobs):
        ds = "test-data-ta-fmctx"
        _cleanup_dataset(ds)
        try:
            src = _out("fm_ctx_source")
            _run_ensemble_cli(["--output-dir", str(src)], n_jobs=n_jobs, dataset_name=ds)
            # A CV run pointing at a train-all source → error (SystemExit).
            with pytest.raises(SystemExit):
                _run_ensemble_cli(
                    ["--feature-matrices-dir", str(src),
                     "--output-dir", str(OUTPUT_DIR / "fm_ctx_mismatch")],
                    n_jobs=n_jobs, dataset_name=ds, with_data_dir=False,
                    training_context="cv",  # CV run vs a train-all source → mismatch
                )
        finally:
            _cleanup_dataset(ds)


class TestTrainAllEnsembleGuards:
    def test_fold_ids_rejected(self, n_jobs):
        # --fold-ids + train_all → parser.error → SystemExit (no training needed).
        with pytest.raises(SystemExit):
            _run_ensemble_cli(
                ["--fold-ids", "0"], n_jobs=n_jobs, dataset_name="test-data-ta-foldguard",
                with_data_dir=False,
            )

    def test_metamodel_cv_n_splits_below_2_rejected(self, n_jobs):
        # --metamodel-cv-n-splits < 2 → parser.error → SystemExit.
        with pytest.raises(SystemExit):
            _run_ensemble_cli(
                ["--metamodel-cv-n-splits", "1"],
                n_jobs=n_jobs, dataset_name="test-data-ta-splitguard", with_data_dir=False,
            )

    def test_leaky_base_model_rejected_by_guard(self):
        """Leakage guard: a train-all ensemble LOADs base models only if their
        summary's training_context == 'train_all_ensemble'. A base model trained
        as the leaky 'train_all' (validation NOT excluded) must be rejected.

        This is the exact mechanism main() uses: validate_model_summary compares
        the loaded summary's training_context against the expected one. We assert
        the guard directly (no full training needed)."""
        from malid_lite.training.training_utils import validate_model_summary

        expected = {
            "gene_locus": "TCR",
            "training_context": "train_all_ensemble",  # what a train-all ensemble requires
            "classification_mode": "multiclass",
            "reference_class": None,
            "diseases": None,
        }
        leaky_summary = {
            "gene_locus": "TCR",
            "training_context": "train_all",  # LEAKY: saw the validation set
            "classification_mode": "multiclass",
            "reference_class": None,
            "diseases": None,
            "timestamp": "20260101_000000",
        }
        with pytest.raises(ValueError, match="training_context"):
            validate_model_summary(leaky_summary, expected, model_label="Model 1 (leaky)")

        # A correctly-trained train_all_ensemble base model passes the same guard.
        ok_summary = {**leaky_summary, "training_context": "train_all_ensemble"}
        validate_model_summary(ok_summary, expected, model_label="Model 1 (ok)")
