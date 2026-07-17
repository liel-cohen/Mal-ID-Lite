"""Test the train-all data-loading path (Phase 1: cross-dataset support).

Uses the bundled test data (tests/test_data/) with 72 participants, 3 folds,
4 diseases. Each test creates a fresh temporary cache directory.

Tests:
1. get_all_data() returns the whole dataset (= union of all CV test folds)
2. get_all_data() caches to data_folds/all_<stage>_* and reloads identically
3. get_all_data() equals get_fold_data over all folds combined
4. iter_all_specimens() yields every specimen
5. CV_fold-optional: metadata without a CV_fold column loads and supports
   train-all, but fold-based loading raises a clear error
6. get_fold_data rejects fold_label="all" (must use get_all_data)

Output: tests/test_outputs/test_train_all_loader/

Expected runtime: <30 seconds
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR, TEST_FOLD_IDS
from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.dataloader.base import PreprocessingStage

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_test_output_dir(test_name: str) -> Path:
    test_dir = OUTPUT_DIR / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def _fresh_loader(cache_dir: Path, metadata_path: Path = None) -> MalIDPublishedDataLoader:
    """Loader over the bundled raw test data with a fresh cache_dir."""
    return MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=metadata_path or (TEST_DATA_DIR / "metadata.tsv"),
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=cache_dir,
        verbose=0,
    )


def _participants(df: pd.DataFrame) -> set:
    return set(df["participant_label"].unique())


@pytest.mark.integration
class TestGetAllData:
    """get_all_data() loads the entire dataset regardless of folds."""

    def test_all_equals_union_of_test_folds(self):
        """The whole dataset == union of every CV fold's test partition."""
        cache_dir = _get_test_output_dir("test_all_equals_union_of_test_folds") / "cache"
        loader = _fresh_loader(cache_dir)

        all_seq, all_meta = loader.get_all_data()
        assert len(all_seq) > 0 and len(all_meta) > 0

        # Union of test folds should reproduce the full participant set.
        union_participants = set()
        total_test_specimens = 0
        for fold_id in TEST_FOLD_IDS:
            _, test_meta = loader.get_fold_data(fold_id, "test")
            union_participants |= _participants(test_meta)
            total_test_specimens += len(test_meta)

        assert _participants(all_meta) == union_participants, (
            "get_all_data participants must equal the union of per-fold test participants"
        )
        # Each specimen appears in exactly one test fold, so counts must match.
        assert len(all_meta) == total_test_specimens

    def test_all_data_caches_and_reloads(self):
        """get_all_data caches to data_folds/all_<stage>_* and reloads identically."""
        cache_dir = _get_test_output_dir("test_all_data_caches_and_reloads") / "cache"
        loader = _fresh_loader(cache_dir)

        seq1, meta1 = loader.get_all_data()

        stage = PreprocessingStage.DOWNSAMPLED.value
        seq_cache = cache_dir / "data_folds" / f"all_{stage}_sequences.parquet"
        meta_cache = cache_dir / "data_folds" / f"all_{stage}_metadata.csv"
        assert seq_cache.exists(), "all_<stage>_sequences.parquet not cached"
        assert meta_cache.exists(), "all_<stage>_metadata.csv not cached"

        # Reload (cache hit) yields the same participants and row counts.
        seq2, meta2 = loader.get_all_data()
        assert len(seq1) == len(seq2)
        assert _participants(meta1) == _participants(meta2)

    def test_iter_all_specimens_covers_dataset(self):
        """iter_all_specimens yields every specimen present in get_all_data."""
        cache_dir = _get_test_output_dir("test_iter_all_specimens_covers_dataset") / "cache"
        loader = _fresh_loader(cache_dir)

        specimen_labels = [s for s, _, _ in loader.iter_all_specimens()]
        assert len(specimen_labels) == len(set(specimen_labels)), "duplicate specimens yielded"

        _, all_meta = loader.get_all_data()
        assert set(specimen_labels) == set(all_meta["specimen_label"])

    def test_get_fold_data_rejects_all_label(self):
        """get_fold_data must not accept fold_label='all' (use get_all_data)."""
        cache_dir = _get_test_output_dir("test_get_fold_data_rejects_all_label") / "cache"
        loader = _fresh_loader(cache_dir)
        with pytest.raises(ValueError, match="get_all_data"):
            loader.get_fold_data(0, "all")

    def test_all_fails_loudly_on_load_error(self):
        """get_all_data must raise (not silently shrink) if a participant errors.

        Uses metadata-only mode (data_dir=None) with an empty cache: every
        participant load raises (cache miss + no data_dir), so the completeness
        check must fail loudly rather than return a partial/empty dataset.
        """
        cache_dir = _get_test_output_dir("test_all_fails_loudly_on_load_error") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        loader = MalIDPublishedDataLoader(
            data_dir=None,  # no raw data + empty cache → participant loads raise
            metadata_path=TEST_DATA_DIR / "metadata.tsv",
            gene_reference_path=None,
            gene_locus="TCR",
            cache_dir=cache_dir,
            verbose=0,
        )
        with pytest.raises(RuntimeError, match="Failed to load"):
            loader.get_all_data()


@pytest.mark.integration
class TestTrainAllCacheManagement:
    """The train-all 'all_*' cache is counted and cleared like fold caches."""

    def test_all_cache_counted_and_cleared(self):
        cache_dir = _get_test_output_dir("test_all_cache_counted_and_cleared") / "cache"
        loader = _fresh_loader(cache_dir)

        # Build the train-all cache.
        loader.get_all_data()
        stage = PreprocessingStage.DOWNSAMPLED.value
        all_seq = cache_dir / "data_folds" / f"all_{stage}_sequences.parquet"
        manifest = cache_dir / "data_folds" / f"all_{stage}_cache_manifest.json"
        assert all_seq.exists()
        assert manifest.exists(), "cache manifest not written"

        # get_cache_info must count the all_* cache (regression: previously it
        # only globbed fold_*).
        info = loader.get_cache_info()
        assert info["folds"]["count"] >= 1, "train-all cache not counted by get_cache_info"

        # clear_fold_cache must remove the all_* cache AND its manifest (regression:
        # previously it only globbed fold_*, leaving a stale train-all cache behind).
        loader.clear_fold_cache(confirm=False)
        assert not all_seq.exists(), "clear_fold_cache left the train-all cache behind"
        assert not manifest.exists(), "clear_fold_cache left the cache manifest behind"


@pytest.mark.integration
class TestAllCacheManifest:
    """The whole-dataset cache manifest detects metadata changes on a cache hit."""

    def _manifest_path(self, cache_dir: Path) -> Path:
        stage = PreprocessingStage.DOWNSAMPLED.value
        return cache_dir / "data_folds" / f"all_{stage}_cache_manifest.json"

    def test_manifest_content(self):
        cache_dir = _get_test_output_dir("test_manifest_content") / "cache"
        loader = _fresh_loader(cache_dir)
        _, all_meta = loader.get_all_data()

        manifest = json.loads(self._manifest_path(cache_dir).read_text())
        # metadata_pairs == all metadata (specimen, participant) pairs
        meta_pairs = {
            (str(s), str(p))
            for s, p in zip(loader.metadata["specimen_label"],
                            loader.metadata["participant_label"])
        }
        recorded = {tuple(x) for x in manifest["metadata_pairs"]}
        assert recorded == meta_pairs
        assert manifest["n_metadata_pairs"] == len(meta_pairs)
        # QC-dropped = metadata pairs absent from the cached data
        cached_pairs = {
            (str(s), str(p))
            for s, p in zip(all_meta["specimen_label"], all_meta["participant_label"])
        }
        assert {tuple(x) for x in manifest["qc_dropped_pairs"]} == (meta_pairs - cached_pairs)

    def test_unchanged_metadata_reloads_without_error(self):
        """A cache hit with unchanged metadata must NOT raise (no false alarm)."""
        cache_dir = _get_test_output_dir("test_unchanged_metadata_reloads") / "cache"
        loader = _fresh_loader(cache_dir)
        loader.get_all_data()          # build
        seq2, meta2 = loader.get_all_data()  # cache hit — must not raise
        assert len(seq2) > 0

    def test_added_participant_detected(self):
        """Simulate a metadata addition (manifest missing a current pair) → raise."""
        cache_dir = _get_test_output_dir("test_added_participant_detected") / "cache"
        loader = _fresh_loader(cache_dir)
        loader.get_all_data()

        # Drop one pair from the manifest → as if metadata gained it after build.
        mpath = self._manifest_path(cache_dir)
        manifest = json.loads(mpath.read_text())
        dropped = manifest["metadata_pairs"].pop()
        manifest["n_metadata_pairs"] = len(manifest["metadata_pairs"])
        mpath.write_text(json.dumps(manifest))

        loader2 = _fresh_loader(cache_dir)
        with pytest.raises(ValueError, match="stale"):
            loader2.get_all_data()

    def test_removed_participant_detected(self):
        """Simulate a metadata removal (manifest has an extra pair) → raise."""
        cache_dir = _get_test_output_dir("test_removed_participant_detected") / "cache"
        loader = _fresh_loader(cache_dir)
        loader.get_all_data()

        mpath = self._manifest_path(cache_dir)
        manifest = json.loads(mpath.read_text())
        manifest["metadata_pairs"].append(["__ghost_specimen__", "__ghost_participant__"])
        mpath.write_text(json.dumps(manifest))

        loader2 = _fresh_loader(cache_dir)
        with pytest.raises(ValueError, match="stale"):
            loader2.get_all_data()

    def test_missing_manifest_backward_compatible(self):
        """No manifest (older cache) → fall back to the parquet check, no raise."""
        cache_dir = _get_test_output_dir("test_missing_manifest_backward_compatible") / "cache"
        loader = _fresh_loader(cache_dir)
        loader.get_all_data()

        self._manifest_path(cache_dir).unlink()  # simulate a pre-manifest cache
        loader2 = _fresh_loader(cache_dir)
        seq, _ = loader2.get_all_data()  # must not raise
        assert len(seq) > 0

    def test_corrupt_manifest_triggers_rebuild(self):
        """A corrupt manifest → discard the cache, rebuild, regenerate the manifest.

        A corrupt manifest cannot verify that no participant was silently added,
        so the loader must NOT trust the cached parquet. Instead it discards the
        cached bundle and rebuilds from scratch, which re-runs the completeness
        check and writes a fresh, valid manifest. This must not raise, must return
        the full dataset, and must leave a valid manifest behind (so the next hit
        validates normally rather than rebuilding again).
        """
        cache_dir = _get_test_output_dir("test_corrupt_manifest_rebuild") / "cache"
        loader = _fresh_loader(cache_dir)
        seq_orig, _ = loader.get_all_data()

        mpath = self._manifest_path(cache_dir)
        mpath.write_text("{not valid json")  # corrupt it
        loader2 = _fresh_loader(cache_dir)
        seq, _ = loader2.get_all_data()  # must not raise; cache is rebuilt
        assert len(seq) == len(seq_orig), "rebuilt cache must cover the full dataset"

        # A fresh, valid manifest must have been regenerated by the rebuild.
        assert mpath.exists(), "rebuild should regenerate a valid manifest"
        manifest = json.loads(mpath.read_text())  # must parse cleanly now
        assert manifest["n_metadata_pairs"] > 0

        # The regenerated manifest is valid, so the next hit validates without rebuilding.
        loader3 = _fresh_loader(cache_dir)
        seq3, _ = loader3.get_all_data()
        assert len(seq3) == len(seq_orig)

    def test_fresh_loader_unchanged_no_false_alarm(self):
        """A fresh loader reloading an unchanged cache must NOT false-alarm.

        Exercises the manifest comparison across loader instances (fresh loader
        re-derives metadata from cache), guarding against pair dtype / round-trip
        drift between build and validate.
        """
        cache_dir = _get_test_output_dir("test_fresh_loader_unchanged") / "cache"
        loader1 = _fresh_loader(cache_dir)
        loader1.get_all_data()  # build + manifest

        loader2 = _fresh_loader(cache_dir)  # fresh instance, same cache + metadata
        seq, _ = loader2.get_all_data()  # cache hit → manifest validated → no raise
        assert len(seq) > 0


@pytest.mark.integration
class TestCvFoldOptional:
    """Metadata without a CV_fold column supports train-all but not CV."""

    def _metadata_without_fold(self, dest_dir: Path) -> Path:
        """Write a copy of the test metadata with the CV_fold column removed."""
        meta = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
        # Drop canonical + legacy fold columns if present
        for col in ("CV_fold", "malid_cross_validation_fold_id_when_in_test_set"):
            if col in meta.columns:
                meta = meta.drop(columns=[col])
        out = dest_dir / "metadata_no_fold.tsv"
        meta.to_csv(out, sep="\t", index=False)
        return out

    def test_metadata_without_fold_loads(self):
        test_dir = _get_test_output_dir("test_metadata_without_fold_loads")
        meta_path = self._metadata_without_fold(test_dir)
        loader = _fresh_loader(test_dir / "cache", metadata_path=meta_path)

        # Metadata loads fine (no CV_fold required)
        assert "CV_fold" not in loader.metadata.columns
        assert len(loader.metadata) > 0

    def test_train_all_works_without_fold(self):
        test_dir = _get_test_output_dir("test_train_all_works_without_fold")
        meta_path = self._metadata_without_fold(test_dir)
        loader = _fresh_loader(test_dir / "cache", metadata_path=meta_path)

        # get_all_data + train-all splits work without any CV_fold column
        all_seq, all_meta = loader.get_all_data()
        assert len(all_seq) > 0
        splits = loader.load_splits(None, "train_all")
        assert set(splits["split_role"].unique()) == {"train_smaller1", "train_smaller2"}

    def test_cv_loading_without_fold_raises(self):
        test_dir = _get_test_output_dir("test_cv_loading_without_fold_raises")
        meta_path = self._metadata_without_fold(test_dir)
        loader = _fresh_loader(test_dir / "cache", metadata_path=meta_path)

        # Fold-based loading must raise a clear, actionable error
        with pytest.raises(ValueError, match="CV_fold"):
            loader.get_fold_data(0, "train")
        with pytest.raises(ValueError, match="CV_fold|fold assignments"):
            loader.load_splits(0, "cv_single_model")


@pytest.mark.integration
class TestCleanStageQCDrop:
    """A participant whose sequences are ALL dropped at the CLEAN stage is a
    legitimate QC-drop (reported + skipped), not a fatal load failure (audit H1)."""

    def test_clean_total_drop_reported_not_fatal(self):
        cache_dir = _get_test_output_dir("test_clean_total_drop") / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        loader = _fresh_loader(cache_dir)

        # Make CLEAN drop every sequence for one target participant (returns an empty
        # frame). Before the fix, get_all_data misclassified this as a load failure and
        # raised RuntimeError; it must now be treated as a QC-drop.
        target = sorted(loader.metadata["participant_label"].unique())[0]
        orig_clean = loader.preprocess_clean

        def _patched_clean(df, participant_label):
            cleaned, stats = orig_clean(df, participant_label)
            if participant_label == target:
                return cleaned.iloc[0:0], stats  # simulate total clean-stage drop
            return cleaned, stats

        loader.preprocess_clean = _patched_clean

        # Must NOT raise, and must load the rest of the dataset.
        seqs, meta = loader.get_all_data()
        assert len(seqs) > 0
        loaded_participants = set(meta["participant_label"])
        # The target contributed no data and was excluded; everyone else remains.
        assert target not in loaded_participants
        assert loaded_participants == (
            set(loader.metadata["participant_label"].unique()) - {target}
        )


@pytest.mark.integration
class TestStaleSplitRegeneration:
    """load_splits regenerates when a persisted split's participant set no longer
    matches the current metadata (audit M4)."""

    def test_stale_split_regenerates(self):
        cache_dir = _get_test_output_dir("test_stale_split") / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        loader = _fresh_loader(cache_dir)
        roles = ["train_smaller1", "train_smaller2"]
        parts_full = set(loader.get_split_participants(None, "train_all", roles))
        assert parts_full == set(loader.metadata["participant_label"].unique())

        # Corrupt the persisted split: drop one participant row to simulate metadata
        # drift that bypassed the __init__ filecmp guard.
        split_path = cache_dir / "splits" / "train_all.csv"
        assert split_path.exists()
        s = pd.read_csv(split_path)
        s.iloc[1:].to_csv(split_path, index=False)  # drop the first participant

        # A fresh loader (same cache + metadata) must detect the stale split and
        # regenerate it from the current metadata → the full participant set is restored.
        loader2 = _fresh_loader(cache_dir)
        parts_after = set(loader2.get_split_participants(None, "train_all", roles))
        assert parts_after == set(loader2.metadata["participant_label"].unique())
        assert parts_after == parts_full


def test_train_all_json_default_numpy_types():
    """_train_all_json_default converts every numpy scalar type to JSON-native (M5).

    In particular np.bool_ must be handled — it is not an np.integer/np.floating, so
    without an explicit branch json.dump would crash at the end of a completed run.
    """
    import numpy as np
    import json as _json
    from malid_lite.training.training_utils import _train_all_json_default

    assert _train_all_json_default(np.bool_(True)) is True
    assert isinstance(_train_all_json_default(np.int64(42)), int)
    assert _train_all_json_default(np.int64(42)) == 42        # int, not 42.0
    assert isinstance(_train_all_json_default(np.float64(1.5)), float)
    assert _train_all_json_default(np.array([1, 2])) == [1, 2]
    # Full round-trip with a numpy bool + int must not raise and must stay native.
    s = _json.dumps(
        {"flag": np.bool_(False), "n": np.int64(3)}, default=_train_all_json_default
    )
    assert _json.loads(s) == {"flag": False, "n": 3}


class TestCvFoldDtypeCoercion:
    """CV_fold stored as strings is coerced to int, so `CV_fold == fold_id` (int)
    doesn't silently yield an empty test set (audit L5)."""

    def test_string_cv_fold_coerced_to_int(self):
        test_dir = _get_test_output_dir("test_string_cv_fold")
        meta = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
        fold_col = next(
            c for c in ("CV_fold", "malid_cross_validation_fold_id_when_in_test_set")
            if c in meta.columns
        )
        # Store fold ids as strings ("0","1","2").
        meta[fold_col] = meta[fold_col].astype(int).astype(str)
        out = test_dir / "metadata_str_fold.tsv"
        meta.to_csv(out, sep="\t", index=False)

        loader = _fresh_loader(test_dir / "cache", metadata_path=out)
        # Coerced to an integer dtype on load.
        assert loader.metadata["CV_fold"].dtype.kind in ("i", "u")
        # An int fold filter now matches rows (would be 0 if the column stayed string).
        assert int((loader.metadata["CV_fold"] == 0).sum()) > 0
