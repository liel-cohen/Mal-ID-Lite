"""Phase 3 tests: Model 2 train-all (train on the whole dataset, no evaluation).

Covers `train_full_dataset()` / `_run_train_all()` in train_model2.py and the
fold-optional artifact paths in model2_convergent_clusters.py:
1. Multiclass: clusters + per-variant artifacts + meta.json written WITHOUT a fold
   prefix; no predictions/CSV; no-metrics summary; load_from_dir(fold_id=None) works.
2. --retrain-full → _full-suffixed artifacts.
3. Binary + multi-binary: per-pair subdirectories (each with clusters + meta).
4. train_all vs train_all_ensemble: the ensemble context uses strictly fewer
   ts1+ts2 participants (validation excluded).
5. get_artifact_paths(fold_id=None) unit test.
6. CLI: --training-context train_all --fold-ids 0 errors; programmatic CV rejection.
7. Resume: skip / param-mismatch-raises / corrupt-meta-retrains.
8. NO_VALID_CLUSTERS edge (ultra-strict p-value) → marker written, run completes.

Uses the bundled tests/test_data/ (72 participants, 4 diseases, 3 folds).

Output: tests/test_outputs/test_model2_train_all/
Expected runtime: ~2-3 minutes.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR, TEST_DISEASES
from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.models.model2_convergent_clusters import (
    ConvergentClusterClassifier,
    get_artifact_paths,
    get_clusters_path,
    get_no_valid_clusters_path,
)
from malid_lite.training.train_model2 import train_full_dataset

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CLASS = "Healthy/Background"


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


# The runner's --n-jobs (default 2) controls parallelism for Model 2 clustering /
# featurize. An autouse fixture captures it into this module value that the helpers
# below read, so every test scales with --n-jobs without threading the knob through all
# ~13 call sites.
_N_JOBS = 2


@pytest.fixture(autouse=True)
def _capture_n_jobs(n_jobs):
    global _N_JOBS
    _N_JOBS = n_jobs


def _common_kwargs(output_dir: Path, **overrides) -> dict:
    kw = dict(
        metadata_path=None,             # from cache_dir/metadata.tsv
        output_dir=output_dir,
        dataset_name="test-data",
        gene_locus="TCR",
        # A single, permissive p-value keeps clustering fast and (for multiclass)
        # reliably yields valid clusters on the small test data.
        p_values=[0.05],
        n_jobs=_N_JOBS,
        verbose=0,
        cache_dir=TEST_DATA_DIR,
        data_dir=TEST_RAW_DIR,
    )
    kw.update(overrides)
    return kw


def _loader() -> MalIDPublishedDataLoader:
    return MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR, metadata_path=None, gene_locus="TCR",
        cache_dir=TEST_DATA_DIR, verbose=0,
    )


# ---------------------------------------------------------------------------
# Unit: fold-optional artifact paths (3.B)
# ---------------------------------------------------------------------------

class TestArtifactPathsFoldOptional:
    def test_cv_vs_train_all_names(self):
        d = Path("x")
        cv = get_artifact_paths(d, 0, "lasso_cv")
        ta = get_artifact_paths(d, None, "lasso_cv")
        assert cv["clusters"].name == "fold_0_clusters.joblib"
        assert ta["clusters"].name == "clusters.joblib"
        assert cv["pipeline"].name == "fold_0_lasso_cv_model_split1.joblib"
        assert ta["pipeline"].name == "lasso_cv_model_split1.joblib"
        assert get_artifact_paths(d, None, "lasso_cv", True)["pipeline"].name == \
            "lasso_cv_model_full.joblib"
        assert get_clusters_path(d, None).name == "clusters.joblib"
        assert get_no_valid_clusters_path(d, 0, "lasso_cv").name == \
            "fold_0_lasso_cv_NO_VALID_CLUSTERS.txt"
        assert get_no_valid_clusters_path(d, None, "lasso_cv").name == \
            "lasso_cv_NO_VALID_CLUSTERS.txt"


# ---------------------------------------------------------------------------
# Integration: multiclass
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel2TrainAllMulticlass:
    def test_artifacts_no_metrics_and_loads(self):
        out = _out("multiclass")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        info = results["multiclass"]["fold_results"][0]
        assert results["multiclass"]["aggregated_by_model"] == {}
        assert set(info["classes"]) == set(TEST_DISEASES)
        assert info["n_train_ts1_participants"] > 0
        assert info["n_train_ts2_participants"] > 0
        assert "lasso_cv" in info["models"]

        # Artifacts: no fold prefix, no predictions/CSV.
        assert (out / "clusters.joblib").exists()
        assert (out / "lasso_cv_p_value.joblib").exists()
        assert (out / "lasso_cv_model_split1.joblib").exists()
        assert (out / "lasso_cv_results_split1.json").exists()
        assert (out / "meta.json").exists()
        assert not list(out.glob("fold_*")), "train-all must not write fold-prefixed artifacts"
        assert not list(out.glob("*_predictions.csv"))
        assert not list(out.glob("*predictions.pkl"))

        # Summary: training-only, no metrics.
        summary = json.loads(next(out.glob("summary_*.json")).read_text())
        assert summary["training_only"] is True
        assert "aggregated_by_pair" not in summary
        # Config keys predict_model2 needs when loading later:
        assert summary["retrain_on_full_train"] is False
        assert summary["classification_mode"] == "multiclass"
        assert summary["model_names"] == ["lasso_cv"]

        # meta.json records params + expected_artifacts.
        meta = json.loads((out / "meta.json").read_text())
        assert meta["training_context"] == "train_all"
        assert "clusters.joblib" in meta["expected_artifacts"]

        # Loads via the fold-optional loader and predicts on held-in data.
        clf = ConvergentClusterClassifier.load_from_dir(out, fold_id=None,
                                                        gene_locus="TCR", model_name="lasso_cv")
        assert clf.best_p_value_ is not None
        loader = _loader()
        seqs, meta_df = loader.get_all_data()
        disease_map = meta_df.set_index("specimen_label")["disease"]
        seqs = seqs.copy()
        seqs["disease"] = seqs["specimen_label"].map(disease_map)
        # Featurize a small slice (few specimens) and predict.
        some = seqs["specimen_label"].dropna().unique()[:3]
        sub = seqs[seqs["specimen_label"].isin(some)].dropna(subset=["disease"])
        fd = clf.featurize(sub, n_jobs=_N_JOBS)
        if fd.n_scored > 0:
            proba = clf.predict_proba(fd.X)
            assert proba.shape[0] == fd.n_scored
            assert proba.shape[1] == len(clf.classes_)

    def test_retrain_full_suffix(self):
        out = _out("retrain_full")
        train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
            retrain_on_full_train=True,
        )
        assert (out / "lasso_cv_model_full.joblib").exists()
        assert (out / "lasso_cv_results_full.json").exists()
        assert not (out / "lasso_cv_model_split1.joblib").exists()


@pytest.mark.integration
class TestModel2TrainAllBinary:
    def test_binary_and_multi_binary_pairs(self):
        out = _out("binary")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="binary",
            reference_class=REFERENCE_CLASS,
            diseases=["Covid19"],
            training_context="train_all",
        )
        pair = f"Covid19_vs_{REFERENCE_CLASS.replace('/', '_')}"
        assert pair in results
        # Every pair dir has the shared clusters + meta (per-variant artifacts exist
        # only if the variant found valid clusters).
        assert (out / pair / "clusters.joblib").exists()
        assert (out / pair / "meta.json").exists()

    def test_multi_binary(self):
        out = _out("multi_binary")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multi-binary",
            reference_class=REFERENCE_CLASS,
            training_context="train_all",
        )
        non_ref = [d for d in TEST_DISEASES if d != REFERENCE_CLASS]
        assert len(results) == len(non_ref)
        for pair_key in results:
            assert (out / pair_key / "clusters.joblib").exists()
            assert (out / pair_key / "meta.json").exists()


@pytest.mark.integration
class TestModel2TrainAllVsEnsemble:
    def test_ensemble_uses_fewer_participants(self):
        loader = _loader()
        ts12_all = set(loader.get_split_participants(
            None, "train_all", ["train_smaller1", "train_smaller2"]))
        ts12_ens = set(loader.get_split_participants(
            None, "train_all_ensemble", ["train_smaller1", "train_smaller2"]))
        validation = set(loader.get_split_participants(
            None, "train_all_ensemble", ["validation"]))
        # train_all covers everyone; ensemble holds out validation.
        assert ts12_ens < ts12_all
        assert not (ts12_ens & validation)

        out = _out("vs_ensemble")
        r = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all_ensemble",
        )
        info = r["multiclass"]["fold_results"][0]
        # ts1+ts2 participants used == the ensemble split's ts1+ts2 count.
        n_used = info["n_train_ts1_participants"] + info["n_train_ts2_participants"]
        assert n_used == len(ts12_ens)


@pytest.mark.integration
class TestModel2TrainAllValidation:
    def test_programmatic_rejects_cv_context(self):
        out = _out("reject_cv")
        with pytest.raises(ValueError, match="train-all context"):
            train_full_dataset(
                **_common_kwargs(out),
                classification_mode="multiclass",
                training_context="cv_single_model",
            )

    def test_cli_fold_ids_with_train_all_errors(self, monkeypatch):
        from malid_lite.training import train_model2
        argv = [
            "train_model2.py",
            "--metadata-path", str(TEST_DATA_DIR / "metadata.tsv"),
            "--cache-dir", str(TEST_DATA_DIR),
            "--dataset-name", "test-data",
            "--training-context", "train_all",
            "--fold-ids", "0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            train_model2.main()


@pytest.mark.integration
class TestModel2TrainAllResume:
    def test_resume_skip_mismatch_corrupt(self):
        out = _out("resume")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"
        mtime = meta_path.stat().st_mtime_ns

        # Resume with identical params → skip (meta untouched).
        train_full_dataset(**kwargs, resume=True)
        assert meta_path.stat().st_mtime_ns == mtime

        # Corrupt meta → retrain (no crash).
        meta_path.write_text("{corrupt")
        train_full_dataset(**kwargs, resume=True)
        assert "training_info" in json.loads(meta_path.read_text())

        # Param mismatch → raise.
        mism = dict(kwargs)
        mism["retrain_on_full_train"] = True  # differs from the saved False
        with pytest.raises(ValueError, match="mismatch"):
            train_full_dataset(**mism, resume=True)

    def test_resume_empty_expected_artifacts_retrains(self):
        """B1: a meta.json with empty/missing expected_artifacts is incomplete.

        all([]) is vacuously True, so an empty expected_artifacts list must not be
        mistaken for "all artifacts present". A truncated meta (e.g. one predating
        artifact tracking) must trigger a retrain, which repopulates the list.
        """
        out = _out("resume_empty_artifacts")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"

        # Wipe the expected_artifacts list to simulate an untracked/truncated meta.
        meta = json.loads(meta_path.read_text())
        meta["expected_artifacts"] = []
        meta_path.write_text(json.dumps(meta))

        # Resume must retrain (not skip) → expected_artifacts repopulated.
        train_full_dataset(**kwargs, resume=True)
        meta_after = json.loads(meta_path.read_text())
        assert len(meta_after["expected_artifacts"]) > 0
        assert "training_info" in meta_after

    def test_resume_zero_byte_artifact_retrains(self):
        """B4: a zero-byte artifact (crash mid-write) is treated as incomplete.

        Existence alone is not enough — a truncated 0-byte artifact must force a
        retrain that regenerates the file with real content.
        """
        out = _out("resume_zero_byte")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"
        expected = json.loads(meta_path.read_text())["expected_artifacts"]

        # Truncate one non-meta artifact to 0 bytes.
        target = next(
            out / name for name in expected if name != "meta.json"
        )
        target.write_bytes(b"")
        assert target.stat().st_size == 0

        # Resume must retrain → the artifact is rewritten with real content.
        train_full_dataset(**kwargs, resume=True)
        assert target.stat().st_size > 0

    def test_resume_gene_locus_mismatch_raises(self):
        """B3: a resume with a different gene_locus is caught via run_params.

        gene_locus is not a Model 2 classifier hyperparameter (it is resolved into
        model_names / sequence_identity_threshold), so it is not otherwise part of
        the resume identity. run_params must carry gene_locus so a locus mismatch
        raises rather than silently reusing artifacts trained on the other locus.

        We tamper the SAVED meta's run_params.gene_locus (rather than constructing a
        real BCR loader on TCR test data, which would fail for unrelated reasons):
        this exercises exactly the run_params comparison in validate_train_all_meta.
        """
        out = _out("resume_locus_mismatch")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"

        # First, confirm gene_locus is actually recorded in run_params.
        meta = json.loads(meta_path.read_text())
        assert meta["run_params"]["gene_locus"] == "TCR"

        # Tamper the saved locus → resume (still TCR) must detect the mismatch.
        meta["run_params"]["gene_locus"] = "BCR"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="mismatch"):
            train_full_dataset(**kwargs, resume=True)


@pytest.mark.integration
class TestModel2TrainAllNoValidClusters:
    def test_no_valid_clusters_marker(self):
        out = _out("no_valid")
        # An ultra-strict p-value: no cluster passes → NO_VALID_CLUSTERS marker.
        results = train_full_dataset(
            **_common_kwargs(out, p_values=[1e-12]),
            classification_mode="multiclass",
            training_context="train_all",
        )
        marker = out / "lasso_cv_NO_VALID_CLUSTERS.txt"
        assert marker.exists(), "expected a NO_VALID_CLUSTERS marker (no fold prefix)"
        assert not (out / "lasso_cv_model_split1.joblib").exists()
        # Run still completed; training_info records the abstention.
        info = results["multiclass"]["fold_results"][0]
        assert info["models"]["lasso_cv"]["abstained"] is True


@pytest.mark.integration
class TestModel2TrainAllStaleMarkerCleanup:
    """A fresh (non-resume) re-run must clear a stale NO_VALID_CLUSTERS marker so the
    ensemble (which checks the marker first) doesn't silently abstain a freshly-trained
    valid model (audit H2)."""

    def test_fresh_rerun_clears_stale_no_valid_clusters_marker(self):
        out = _out("stale_marker")
        # Run 1: ultra-strict p-value → no valid clusters → marker written, no pipeline.
        train_full_dataset(
            **_common_kwargs(out, p_values=[1e-12]),
            classification_mode="multiclass",
            training_context="train_all",
        )
        marker = out / "lasso_cv_NO_VALID_CLUSTERS.txt"
        assert marker.exists(), "run 1 should have written the abstain marker"
        assert not (out / "lasso_cv_model_split1.joblib").exists()

        # Run 2: fresh (no --resume) with a permissive p-value → valid model. The stale
        # marker must be gone and the trained pipeline present (mutual exclusion).
        train_full_dataset(
            **_common_kwargs(out, p_values=[0.05]),
            classification_mode="multiclass",
            training_context="train_all",
        )
        assert not marker.exists(), "fresh re-run must clear the stale NO_VALID_CLUSTERS marker"
        assert (out / "lasso_cv_model_split1.joblib").exists(), "valid pipeline must be written"


@pytest.mark.integration
class TestModel2TrainAllPerPairSummaryKeys:
    """Each per-pair train-all summary must carry the config keys predict_model2 reads
    (classification_mode + retrain_on_full_train), so a pair reloads on its own (M3)."""

    def test_binary_per_pair_summary_self_sufficient(self):
        out = _out("perpair_keys")
        train_full_dataset(
            **_common_kwargs(out, retrain_on_full_train=True),
            classification_mode="binary",
            reference_class=REFERENCE_CLASS,
            diseases=["Covid19"],
            training_context="train_all",
        )
        pair = f"Covid19_vs_{REFERENCE_CLASS.replace('/', '_')}"
        pair_summary = json.loads(next((out / pair).glob("summary_*.json")).read_text())
        assert pair_summary["classification_mode"] == "binary"
        assert pair_summary["retrain_on_full_train"] is True
        assert pair_summary["reference_class"] == REFERENCE_CLASS
        assert "model_names" in pair_summary
