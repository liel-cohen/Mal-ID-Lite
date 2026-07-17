"""Phase 4 tests: Model 3 train-all (train on the whole dataset, no evaluation).

Covers `train_full_dataset()` / `_run_train_all()` in train_model3.py, the
fold-optional `_stage_artifact_paths`, and the fold-optional `predict_model3`
loader in train_ensemble.py:

1. `_stage_artifact_paths` fold-optional (unit): int -> fold_<id>_stage{1,2}.pkl;
   None -> stage{1,2}.pkl (no prefix).
2. Multiclass train-all (integration): stage1.pkl + stage2.pkl + meta.json written
   WITHOUT a fold prefix; no fold_* / results.json / predictions.pkl; no-metrics
   summary (training_only); training_info ts1/ts2 counts; reload via from_summary.
3. train_all vs train_all_ensemble: the ensemble context trains on strictly fewer
   ts1+ts2 participants (validation excluded); no validation leakage.
4. Binary: per-pair subdirectory with stage1/stage2/meta + reference_class recorded;
   the per-pair summary carries the from_summary config keys.
5. Resume: skip / param-mismatch-raises / corrupt-meta-retrains / empty-expected
   -retrains / zero-byte-retrains; --resume-from-stage2 reuses Stage 1 and retrains
   Stage 2; missing Stage 1 -> raises.
6. Tuning (auto_tuned): tuning_cv_results.csv written; training_info records the
   selected strategy.
7. CLI guards: --fold-ids / --resume-from-evaluation / --stage1-dir with a train-all
   context error early; programmatic CV-context rejection.
8. predict_model3(fold_id=None) loads the train-all artifacts (Phase 5 enabler).

Efficiency (per project convention — small data subset for integration tests):
Instead of the full 72-participant bundled dataset (which makes glmnet Stage 1
very slow when repeated across tests), these tests build a small SUBSET cache once
(``_subset`` fixture: 6 participants/disease = 24, safely above the >=3 ensemble
stratification minimum) in a dedicated cache dir, reusing the bundled per-participant
clean parquets (so downsampling matches the bundled precomputed embeddings) and the
bundled embeddings (so no ESM-2 is needed). Read-only assertions share ONE trained
multiclass model via the ``_trained_multiclass`` fixture.

Output: tests/test_outputs/test_model3_train_all/
Expected runtime: a few minutes.
"""

import json
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR, TEST_DISEASES
from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.models.model3_sequence_level import SequenceLevelClassifier
from malid_lite.training.train_model3 import (
    _stage_artifact_paths,
    train_full_dataset,
)

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIR = TEST_DATA_DIR / "embeddings"
REFERENCE_CLASS = "Healthy/Background"
N_PER_DISEASE = 4  # 4 x 4 diseases = 16; > the >=3 train_all_ensemble minimum.
# The auto-tuning test runs an inner CV over ts2 (~1/3 of participants), so it needs
# more participants per class than the other tests to avoid a degenerate inner fold.
N_PER_DISEASE_TUNING = 8  # 8 x 4 = 32


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_subset(n_per_disease: int, name: str) -> dict:
    """Build a small subset cache for fast integration tests; return loader kwargs.

    Selects the first ``n_per_disease`` participants of each disease, copies their
    bundled clean parquets into a DEDICATED cache dir (so downsampling reproduces
    exactly what the bundled embeddings were built from, and train-all splits
    regenerate for the subset rather than colliding with any full-dataset split
    files), and writes a subset metadata TSV. Embeddings are the shared bundled dir.
    """
    cache = OUTPUT_DIR / name
    if cache.exists():
        shutil.rmtree(cache)
    (cache / "participants").mkdir(parents=True)

    full = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
    parts = full.drop_duplicates("participant_label")[["participant_label", "disease"]]
    chosen = []
    for _disease, grp in parts.groupby("disease"):
        chosen.extend(sorted(grp["participant_label"])[:n_per_disease])

    src_participants = TEST_DATA_DIR / "participants"
    for lbl in chosen:
        for suffix in ("_clean.parquet", "_stats.json"):
            src = src_participants / f"{lbl}{suffix}"
            if src.exists():
                shutil.copy(src, cache / "participants" / src.name)

    meta_path = OUTPUT_DIR / f"{name}_metadata.tsv"
    full[full["participant_label"].isin(chosen)].to_csv(meta_path, sep="\t", index=False)

    return dict(
        metadata_path=meta_path,
        cache_dir=cache,
        data_dir=TEST_RAW_DIR,
        embedding_dir=EMBEDDING_DIR,
    )


@pytest.fixture(scope="module")
def _subset():
    """Small subset cache (16 participants) shared by most integration tests."""
    return _build_subset(N_PER_DISEASE, "_subset_cache")


@pytest.fixture(scope="module")
def _subset_tuning():
    """Larger subset cache (32 participants) for the auto-tuning inner-CV test."""
    return _build_subset(N_PER_DISEASE_TUNING, "_subset_tuning_cache")


def _common_kwargs(output_dir: Path, subset: dict, **overrides) -> dict:
    """Fast train-all kwargs against the subset cache + bundled embeddings.

    cache_embeddings=False → use the cached embeddings as-is (never recompute /
    invoke ESM-2). Small Stage-2 forest keeps the run quick.
    """
    kw = dict(
        metadata_path=subset["metadata_path"],
        output_dir=output_dir,
        dataset_name="test-data-subset",
        gene_locus="TCR",
        embedding_dir=subset["embedding_dir"],
        cache_embeddings=False,
        n_estimators_stage2=10,
        n_jobs=2,
        verbose=0,
        cache_dir=subset["cache_dir"],
        data_dir=subset["data_dir"],
    )
    kw.update(overrides)
    return kw


@pytest.fixture(scope="module")
def _trained_multiclass(_subset):
    """Train ONE multiclass train_all model, shared by read-only tests."""
    out = OUTPUT_DIR / "_shared_multiclass"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    results = train_full_dataset(
        **_common_kwargs(out, _subset),
        classification_mode="multiclass",
        training_context="train_all",
    )
    return out, results


# ---------------------------------------------------------------------------
# Unit: fold-optional artifact paths (4.C)
# ---------------------------------------------------------------------------

class TestStageArtifactPathsFoldOptional:
    def test_cv_vs_train_all_names(self):
        d = Path("x")
        cv1, cv2 = _stage_artifact_paths(d, 3)
        assert cv1.name == "fold_3_stage1.pkl"
        assert cv2.name == "fold_3_stage2.pkl"
        ta1, ta2 = _stage_artifact_paths(d, None)
        assert ta1.name == "stage1.pkl"
        assert ta2.name == "stage2.pkl"


# ---------------------------------------------------------------------------
# Integration: multiclass train-all (shared trained model)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllMulticlass:
    def test_artifacts_no_metrics_and_reload(self, _trained_multiclass):
        out, results = _trained_multiclass

        # Result structure: training-info, no aggregated metrics.
        assert "multiclass" in results
        assert results["multiclass"]["aggregated_by_model"] == {}
        info = results["multiclass"]["fold_results"][0]
        assert info["n_train_ts1_participants"] > 0
        assert info["n_train_ts2_participants"] > 0
        assert info["n_train_ts1_sequences"] > 0
        assert set(info["classes"]) == set(TEST_DISEASES)
        assert info["n_stage1_groups"] > 0

        # Artifacts: NO fold prefix, no eval outputs.
        assert (out / "stage1.pkl").exists()
        assert (out / "stage2.pkl").exists()
        assert (out / "meta.json").exists()
        assert not list(out.glob("fold_*")), "train-all must not write fold-prefixed artifacts"
        assert not list(out.glob("*_results.json"))
        assert not list(out.glob("*_predictions.pkl"))
        assert any(out.glob("summary_*.json"))
        assert any(out.glob("RESULTS_*.md"))

        # Summary JSON: training_only + config keys predict_model3/from_summary read.
        summary = json.loads(next(out.glob("summary_*.json")).read_text())
        assert summary["training_only"] is True
        assert summary["training_context"] == "train_all"
        assert summary["gene_locus"] == "TCR"
        assert summary["classification_mode"] == "multiclass"
        assert "aggregation_strategy" in summary
        assert "reweigh_by_subset_frequencies" in summary
        assert "tuning_enabled" in summary

        # meta.json: resume sentinel with expected_artifacts + training_info.
        meta = json.loads((out / "meta.json").read_text())
        assert meta["training_context"] == "train_all"
        assert meta["disease_filter"] is None
        assert set(meta["expected_artifacts"]) >= {"stage1.pkl", "stage2.pkl"}
        assert "training_info" in meta

        # Saved model reloads via from_summary + stage loaders and exposes classes.
        model = SequenceLevelClassifier.from_summary(summary, n_jobs=1, verbose=0)
        with open(out / "stage1.pkl", "rb") as f:
            model.load_stage1_artifacts(pickle.load(f))
        with open(out / "stage2.pkl", "rb") as f:
            model.load_stage2_artifacts(pickle.load(f))
        assert set(str(c) for c in model.classes_) == set(TEST_DISEASES)

    def test_predict_model3_loads_train_all_artifacts(self, _subset, _trained_multiclass):
        """4.I enabler: predict_model3(fold_id=None) loads the no-prefix artifacts."""
        out, _ = _trained_multiclass
        summary = json.loads(next(out.glob("summary_*.json")).read_text())

        from malid_lite.training.train_ensemble import predict_model3

        loader = MalIDPublishedDataLoader(
            data_dir=_subset["data_dir"], metadata_path=_subset["metadata_path"],
            gene_locus="TCR", cache_dir=_subset["cache_dir"], verbose=0,
        )
        seqs, meta = loader.get_all_data()
        # A small set of target specimens keeps the prediction quick.
        target = set(meta["specimen_label"].unique()[:4])
        preds = predict_model3(
            out, None, seqs, meta, target,
            embedding_dir=_subset["embedding_dir"], summary=summary, n_jobs=1,
        )
        # Predictions returned for the requested specimens (never abstains).
        assert preds is not None
        assert len(preds.probabilities) == len(target)


# ---------------------------------------------------------------------------
# Integration: train_all vs train_all_ensemble
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllVsEnsemble:
    def test_ensemble_uses_fewer_participants(self, _subset, _trained_multiclass):
        # Reuse the shared train_all model; only the ensemble context trains here.
        out_all, r_all = _trained_multiclass
        out_ens = _out("ctx_train_all_ensemble")
        r_ens = train_full_dataset(
            **_common_kwargs(out_ens, _subset),
            classification_mode="multiclass",
            training_context="train_all_ensemble",
        )
        info_all = r_all["multiclass"]["fold_results"][0]
        info_ens = r_ens["multiclass"]["fold_results"][0]
        n_all = info_all["n_train_ts1_participants"] + info_all["n_train_ts2_participants"]
        n_ens = info_ens["n_train_ts1_participants"] + info_ens["n_train_ts2_participants"]
        # train_all_ensemble holds out ~1/3 for the metamodel validation set.
        assert n_ens < n_all, (
            f"train_all_ensemble ({n_ens}) should use fewer ts1+ts2 participants "
            f"than train_all ({n_all})"
        )


# ---------------------------------------------------------------------------
# Integration: binary
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllBinary:
    def test_binary_pair_subdir_and_summary_keys(self, _subset):
        out = _out("binary")
        results = train_full_dataset(
            **_common_kwargs(out, _subset),
            classification_mode="binary",
            reference_class=REFERENCE_CLASS,
            diseases=["HIV"],
            training_context="train_all",
        )
        # Locate the per-pair subdir (exact separator handling is in make_pair_name).
        pair_dirs = [d for d in out.iterdir() if d.is_dir()]
        assert len(pair_dirs) == 1, f"expected one pair subdir, got {[d.name for d in pair_dirs]}"
        pair_dir = pair_dirs[0]
        assert (pair_dir / "stage1.pkl").exists()
        assert (pair_dir / "stage2.pkl").exists()
        assert (pair_dir / "meta.json").exists()

        # training_info records the binary pair.
        key = next(iter(results))
        info = results[key]["fold_results"][0]
        assert info["disease"] == "HIV"
        assert info["reference_class"] == REFERENCE_CLASS

        # Per-pair summary carries the from_summary config keys (Phase 5 self-sufficiency).
        pair_summary = json.loads(next(pair_dir.glob("summary_*.json")).read_text())
        for k in ("gene_locus", "aggregation_strategy", "tuning_enabled",
                  "reweigh_by_subset_frequencies", "reference_class"):
            assert k in pair_summary, f"per-pair summary missing '{k}'"
        assert pair_summary["reference_class"] == REFERENCE_CLASS


# ---------------------------------------------------------------------------
# Integration: multi-binary
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllMultiBinary:
    def test_multi_binary_pairs(self, _subset):
        """One independent binary model per disease, each in its own pair subdir."""
        out = _out("multi_binary")
        results = train_full_dataset(
            **_common_kwargs(out, _subset),
            classification_mode="multi-binary",
            reference_class=REFERENCE_CLASS,
            diseases=["HIV", "Covid19"],
            training_context="train_all",
        )
        # Two pairs (HIV vs ref, Covid19 vs ref), each with its own artifacts + summary.
        pair_dirs = [d for d in out.iterdir() if d.is_dir()]
        assert len(pair_dirs) == 2, f"expected 2 pair subdirs, got {[d.name for d in pair_dirs]}"
        for d in pair_dirs:
            assert (d / "stage1.pkl").exists(), f"{d.name} missing stage1.pkl"
            assert (d / "stage2.pkl").exists(), f"{d.name} missing stage2.pkl"
            assert (d / "meta.json").exists(), f"{d.name} missing meta.json"
            assert any(d.glob("summary_*.json")), f"{d.name} missing per-pair summary"

        # results keyed per pair; each training_info records its disease + reference.
        assert len(results) == 2
        diseases_seen = set()
        for _key, val in results.items():
            info = val["fold_results"][0]
            assert info["reference_class"] == REFERENCE_CLASS
            diseases_seen.add(info["disease"])
        assert diseases_seen == {"HIV", "Covid19"}


# ---------------------------------------------------------------------------
# Integration: fresh-run cleanup of stale diagnostics (F3)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllFreshCleanup:
    def test_fresh_run_removes_stale_diagnostic_csv(self, _subset):
        """A fresh (non-resume) run clears a stale diagnostic left by a prior run.

        Simulates a tuning_cv_results.csv left by an earlier auto_tuned run; the new
        (non-tuned) run does not regenerate it, so the fresh-run cleanup must remove it
        rather than leave a misleading orphan.
        """
        out = _out("fresh_cleanup")
        stale = out / "tuning_cv_results.csv"
        stale.write_text("stale,leftover\n1,2\n")
        train_full_dataset(
            **_common_kwargs(out, _subset),
            classification_mode="multiclass",
            training_context="train_all",
        )
        assert not stale.exists(), "fresh run must remove a stale tuning_cv_results.csv"
        # The run itself still produced the real artifacts.
        assert (out / "stage1.pkl").exists()
        assert (out / "stage2.pkl").exists()


# ---------------------------------------------------------------------------
# Integration: resume
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllResume:
    def test_resume_skip_mismatch_corrupt(self, _subset):
        out = _out("resume")
        kwargs = dict(
            **_common_kwargs(out, _subset),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"
        mtime = meta_path.stat().st_mtime_ns

        # Resume with identical params → skip (meta untouched).
        train_full_dataset(**kwargs, resume=True)
        assert meta_path.stat().st_mtime_ns == mtime

        # Corrupt meta → retrain (no crash), meta rewritten with training_info.
        meta_path.write_text("{corrupt")
        train_full_dataset(**kwargs, resume=True)
        assert "training_info" in json.loads(meta_path.read_text())

        # Param mismatch → raise (n_estimators_stage2 differs from saved).
        mism = dict(kwargs)
        mism["n_estimators_stage2"] = 20
        with pytest.raises(ValueError, match="mismatch"):
            train_full_dataset(**mism, resume=True)

    def test_resume_empty_and_zero_byte_retrain(self, _subset):
        out = _out("resume_incomplete")
        kwargs = dict(
            **_common_kwargs(out, _subset),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        meta_path = out / "meta.json"

        # Empty expected_artifacts (B1 parity) → retrain repopulates it.
        meta = json.loads(meta_path.read_text())
        meta["expected_artifacts"] = []
        meta_path.write_text(json.dumps(meta))
        train_full_dataset(**kwargs, resume=True)
        assert len(json.loads(meta_path.read_text())["expected_artifacts"]) > 0

        # Zero-byte stage1.pkl (B4 parity) → retrain rewrites it non-empty.
        (out / "stage1.pkl").write_bytes(b"")
        train_full_dataset(**kwargs, resume=True)
        assert (out / "stage1.pkl").stat().st_size > 0

    def test_resume_from_stage2(self, _subset):
        out = _out("resume_from_stage2")
        kwargs = dict(
            **_common_kwargs(out, _subset),
            classification_mode="multiclass",
            training_context="train_all",
        )
        train_full_dataset(**kwargs, resume=False)
        stage1_mtime = (out / "stage1.pkl").stat().st_mtime_ns

        # Change only a Stage-2 knob and resume-from-stage2 → Stage 1 reused
        # (mtime unchanged), Stage 2 rewritten.
        s2 = dict(kwargs)
        s2["aggregation_strategy"] = "mean"
        train_full_dataset(**s2, resume_from_stage2=True)
        assert (out / "stage1.pkl").stat().st_mtime_ns == stage1_mtime, (
            "--resume-from-stage2 must reuse the existing Stage 1 artifact"
        )
        assert (out / "stage2.pkl").exists()
        summary = json.loads(next(out.glob("summary_*.json")).read_text())
        assert summary["aggregation_strategy"] == "mean"

    def test_resume_from_stage2_missing_stage1_raises(self, _subset):
        out = _out("resume_from_stage2_missing")
        kwargs = dict(
            **_common_kwargs(out, _subset),
            classification_mode="multiclass",
            training_context="train_all",
        )
        with pytest.raises(ValueError, match="requires a saved Stage 1"):
            train_full_dataset(**kwargs, resume_from_stage2=True)


# ---------------------------------------------------------------------------
# Integration: tuning
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModel3TrainAllTuning:
    def test_auto_tuned_writes_results_and_records_winner(self, _subset_tuning):
        out = _out("tuning")
        results = train_full_dataset(
            **_common_kwargs(out, _subset_tuning),
            classification_mode="multiclass",
            training_context="train_all",
            aggregation_strategy="auto_tuned",
            tuning_cv_splits=2,
        )
        assert (out / "tuning_cv_results.csv").exists()
        info = results["multiclass"]["fold_results"][0]
        assert info["tuning_enabled"] is True
        # The recorded aggregation strategy is the tuned winner (a real enum name,
        # not the literal "auto_tuned").
        assert info["aggregation_strategy"] != "auto_tuned"
        assert "tuning_best_strategy" in info


# ---------------------------------------------------------------------------
# CLI guards + programmatic validation
# ---------------------------------------------------------------------------

class TestModel3TrainAllValidation:
    def test_programmatic_rejects_cv_context(self, _subset):
        out = _out("reject_cv")
        with pytest.raises(ValueError, match="train-all context"):
            train_full_dataset(
                **_common_kwargs(out, _subset),
                classification_mode="multiclass",
                training_context="cv_single_model",
            )

    def test_cli_fold_ids_with_train_all_errors(self, monkeypatch):
        from malid_lite.training import train_model3
        argv = [
            "train_model3.py",
            "--metadata-path", str(TEST_DATA_DIR / "metadata.tsv"),
            "--cache-dir", str(TEST_DATA_DIR),
            "--dataset-name", "test-data",
            "--training-context", "train_all",
            "--fold-ids", "0",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            train_model3.main()

    def test_cli_resume_from_evaluation_with_train_all_errors(self, monkeypatch):
        from malid_lite.training import train_model3
        argv = [
            "train_model3.py",
            "--metadata-path", str(TEST_DATA_DIR / "metadata.tsv"),
            "--cache-dir", str(TEST_DATA_DIR),
            "--dataset-name", "test-data",
            "--training-context", "train_all",
            "--resume-from-evaluation",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            train_model3.main()
