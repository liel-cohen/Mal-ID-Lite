"""Phase 2 tests: Model 1 train-all (train on the whole dataset, no evaluation).

Covers `train_full_dataset()` / `_run_train_all()` in train_model1.py:
1. Multiclass: artifacts written without fold prefix (model/v_genes/meta), a
   no-metrics summary + RESULTS.md, and the saved model loads.
2. Binary + multi-binary: per-pair subdirectories with their own artifacts.
3. train_all vs train_all_ensemble: the ensemble context trains on strictly fewer
   participants (validation excluded) — the subset/leakage invariant.
4. CLI: `--training-context train_all --fold-ids 0` errors; the programmatic
   `train_full_dataset` rejects CV contexts.
5. Resume: a second run with artifacts present skips; a param mismatch raises.

Uses the bundled tests/test_data/ (72 participants, 4 diseases, 3 folds).

Output: tests/test_outputs/test_model1_train_all/
Expected runtime: ~1-2 minutes.
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
from malid_lite.models.model1_repertoire import RepertoireClassifier
from malid_lite.training.train_model1 import train_full_dataset

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_CLASS = "Healthy/Background"


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _common_kwargs(output_dir: Path) -> dict:
    """Shared train_full_dataset kwargs backed by the bundled test data."""
    return dict(
        metadata_path=None,             # picked up from cache_dir/metadata.tsv
        output_dir=output_dir,
        dataset_name="test-data",
        model_name="lasso_cv",
        gene_locus="TCR",
        n_pcs=10,
        verbose=0,
        cache_dir=TEST_DATA_DIR,
        data_dir=TEST_RAW_DIR,
    )


@pytest.mark.integration
class TestTrainAllMulticlass:
    def test_multiclass_artifacts_no_metrics(self):
        out = _out("multiclass")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )

        # Result structure: training-info, no aggregated metrics.
        assert "multiclass" in results
        fold_results = results["multiclass"]["fold_results"]
        assert results["multiclass"]["aggregated_by_model"] == {}
        assert len(fold_results) == 1
        info = fold_results[0]
        assert info["n_train_participants"] > 0
        assert info["n_train_sequences"] > 0
        assert set(info["classes"]) == set(TEST_DISEASES)

        # Artifacts: no fold prefix.
        assert (out / "lasso_cv_model.pkl").exists()
        assert (out / "lasso_cv_v_genes.json").exists()
        assert (out / "lasso_cv_meta.json").exists()
        assert not list(out.glob("fold_*")), "train-all must not write fold-prefixed artifacts"
        assert any(out.glob("summary_*.json"))
        assert any(out.glob("RESULTS_*.md"))

        # Summary JSON: training-only, no metric keys.
        summary = json.loads(next(out.glob("summary_*.json")).read_text())
        assert summary["training_only"] is True
        assert summary["training_context"] == "train_all"
        assert "aggregated_by_pair" not in summary
        assert "results_by_pair" not in summary

        # meta.json: resume params + training info.
        meta = json.loads((out / "lasso_cv_meta.json").read_text())
        assert meta["training_context"] == "train_all"
        assert meta["disease_filter"] is None
        assert meta["training_info"]["n_train_participants"] == info["n_train_participants"]

        # Saved model loads and exposes classes.
        model = RepertoireClassifier.load(out / "lasso_cv_model.pkl")
        assert set(str(c) for c in model.classes_) == set(TEST_DISEASES)

    def test_multiclass_trains_on_all_participants(self):
        out = _out("multiclass_all_participants")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        n_trained = results["multiclass"]["fold_results"][0]["n_train_participants"]

        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR, metadata_path=None, gene_reference_path=None,
            gene_locus="TCR", cache_dir=TEST_DATA_DIR, verbose=0,
        )
        n_all = loader.metadata.drop_duplicates("participant_label").shape[0]
        # train_all trains on every participant (no QC drops in the test data).
        assert n_trained == n_all


@pytest.mark.integration
class TestTrainAllBinary:
    def test_binary_pair_subdir(self):
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
        pair_dir = out / pair
        assert (pair_dir / "lasso_cv_model.pkl").exists()
        assert (pair_dir / "lasso_cv_v_genes.json").exists()
        assert (pair_dir / "lasso_cv_meta.json").exists()
        assert any(pair_dir.glob("summary_*.json"))
        info = results[pair]["fold_results"][0]
        assert set(info["classes"]) == {"Covid19", REFERENCE_CLASS}

    def test_multi_binary_all_pairs(self):
        out = _out("multi_binary")
        results = train_full_dataset(
            **_common_kwargs(out),
            classification_mode="multi-binary",
            reference_class=REFERENCE_CLASS,
            training_context="train_all",
        )
        # One pair per non-reference disease.
        non_ref = [d for d in TEST_DISEASES if d != REFERENCE_CLASS]
        assert len(results) == len(non_ref)
        for pair_key in results:
            pair_dir = out / pair_key
            assert (pair_dir / "lasso_cv_model.pkl").exists()
            assert (pair_dir / "lasso_cv_meta.json").exists()


@pytest.mark.integration
class TestTrainAllVsEnsemble:
    """train_all_ensemble excludes the validation third (no leakage)."""

    def test_ensemble_trains_on_fewer_participants(self):
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR, metadata_path=None, gene_reference_path=None,
            gene_locus="TCR", cache_dir=TEST_DATA_DIR, verbose=0,
        )
        n_all = loader.metadata.drop_duplicates("participant_label").shape[0]
        ts12 = set(loader.get_split_participants(
            None, "train_all_ensemble", ["train_smaller1", "train_smaller2"]))
        validation = set(loader.get_split_participants(
            None, "train_all_ensemble", ["validation"]))

        out_all = _out("vs_train_all")
        r_all = train_full_dataset(
            **_common_kwargs(out_all),
            classification_mode="multiclass", training_context="train_all",
        )
        out_ens = _out("vs_train_all_ensemble")
        r_ens = train_full_dataset(
            **_common_kwargs(out_ens),
            classification_mode="multiclass", training_context="train_all_ensemble",
        )

        n_all_trained = r_all["multiclass"]["fold_results"][0]["n_train_participants"]
        n_ens_trained = r_ens["multiclass"]["fold_results"][0]["n_train_participants"]

        # train_all uses everyone; ensemble uses only ts1+ts2 (validation held out).
        assert n_all_trained == n_all
        assert n_ens_trained == len(ts12)
        assert n_ens_trained < n_all_trained, "ensemble context must train on fewer participants"
        # Leakage invariant: ts1+ts2 and validation are disjoint (Phase 1), so the
        # ensemble training set cannot include validation participants.
        assert not (ts12 & validation)


@pytest.mark.integration
class TestTrainAllValidation:
    def test_programmatic_rejects_cv_context(self):
        out = _out("reject_cv")
        with pytest.raises(ValueError, match="train-all context"):
            train_full_dataset(
                **_common_kwargs(out),
                classification_mode="multiclass",
                training_context="cv_single_model",
            )

    def test_cli_fold_ids_with_train_all_errors(self, monkeypatch):
        """`--training-context train_all --fold-ids 0` must fail fast (SystemExit)."""
        from malid_lite.training import train_model1
        argv = [
            "train_model1.py",
            "--metadata-path", str(TEST_DATA_DIR / "metadata.tsv"),
            "--cache-dir", str(TEST_DATA_DIR),
            "--dataset-name", "test-data",
            "--training-context", "train_all",
            "--fold-ids", "0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            train_model1.main()


@pytest.mark.integration
class TestTrainAllResume:
    def test_resume_skips_and_mismatch_raises(self):
        out = _out("resume")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "lasso_cv_meta.json"
        assert meta_path.exists()
        mtime_before = meta_path.stat().st_mtime_ns

        # Resume with identical params → skip (artifacts untouched).
        train_full_dataset(**kwargs, resume=True)
        assert meta_path.stat().st_mtime_ns == mtime_before, "resume should not rewrite artifacts"

        # Resume with a changed param → raise (don't silently mix configs).
        mismatched = dict(kwargs)
        mismatched["n_pcs"] = 12  # differs from the saved 10
        with pytest.raises(ValueError, match="mismatch"):
            train_full_dataset(**mismatched, resume=True)

    def test_resume_corrupt_meta_retrains(self):
        """A corrupt meta.json on --resume is treated as incomplete → retrain (no crash)."""
        out = _out("resume_corrupt_meta")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "lasso_cv_meta.json"
        meta_path.write_text("{corrupt json")  # simulate a truncated/crashed write

        # Must not raise — retrains and rewrites a valid meta.json.
        results = train_full_dataset(**kwargs, resume=True)
        assert results["multiclass"]["fold_results"][0]["n_train_participants"] > 0
        reloaded = json.loads(meta_path.read_text())
        assert "training_info" in reloaded

    def test_resume_zero_byte_artifact_retrains(self):
        """B4: a zero-byte artifact (crash mid-write) is treated as incomplete.

        Existence alone is not enough — a truncated 0-byte model file must force a
        retrain that regenerates it with real content (parity with the CV size-check).
        """
        out = _out("resume_zero_byte")
        kwargs = dict(
            **_common_kwargs(out),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        model_path = out / "lasso_cv_model.pkl"
        model_path.write_bytes(b"")  # truncate to 0 bytes
        assert model_path.stat().st_size == 0

        # Resume must retrain → the model file is rewritten with real content.
        train_full_dataset(**kwargs, resume=True)
        assert model_path.stat().st_size > 0
