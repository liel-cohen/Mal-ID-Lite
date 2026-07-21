"""Tests for scripts/data/create_subset_cache.py.

The subset-cache tool builds a new dataset cache from a subset of an existing
(reference) cache's participants — copying or symlinking their participant and
embedding files. It is the first step of the cross-dataset "train on a fold
subset" workflow (see PIPELINE_GUIDE.md Section 10): filter the metadata to the
desired folds, build a subset cache, then train train-all on it.

Unit tests (fast, no marker) cover the pure helper functions on tiny synthetic
caches. The integration test (marked ``integration``, fast — symlinks only)
runs the real CLI against tests/test_data/ (72 participants, folds
0/1/2) to build a folds-0+1 subset and verify the output layout + error cases.

Output: tests/test_outputs/test_create_subset_cache/. Expected runtime: <30 s.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
# The subset-cache tool lives in scripts/data/ (not a package) — add it to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "data"))

from test_helpers import TEST_DATA_DIR

import create_subset_cache as csc

SCRIPT = Path(__file__).parent.parent / "scripts" / "data" / "create_subset_cache.py"
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem


def _out(name: str) -> Path:
    d = OUTPUT_DIR / name
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_ref_cache(root: Path, participants, folds=None) -> Path:
    """Build a tiny synthetic reference cache with the participant + embedding
    files create_subset_cache expects (2 files in participants/, 3 in embeddings/
    per participant), plus a metadata_processed.tsv. Returns the cache dir."""
    folds = folds or {p: 0 for p in participants}
    (root / "participants").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    for p in participants:
        pd.DataFrame({"x": [1]}).to_parquet(root / "participants" / f"{p}_clean.parquet")
        (root / "participants" / f"{p}_stats.json").write_text("{}")
        np.save(root / "embeddings" / f"{p}_embeddings.npy", np.zeros((1, 2)))
        pd.DataFrame({"x": [1]}).to_parquet(root / "embeddings" / f"{p}_downsampled.parquet")
        (root / "embeddings" / f"{p}_stats.json").write_text("{}")
    meta = pd.DataFrame({
        "participant_label": participants,
        "specimen_label": [f"{p}_s1" for p in participants],
        "disease": ["Covid19"] * len(participants),
        "CV_fold": [folds[p] for p in participants],
    })
    meta.to_csv(root / "metadata_processed.tsv", sep="\t", index=False)
    return root


# ===========================================================================
# Unit tests — pure helpers on synthetic inputs (fast, no marker)
# ===========================================================================


class TestValidateMetadata:
    def test_valid_subset_returns_df(self, tmp_path):
        meta = pd.DataFrame({
            "participant_label": ["p1", "p2"], "specimen_label": ["s1", "s2"],
            "disease": ["Covid19", "HIV"], "CV_fold": [0, 1],
        })
        p = tmp_path / "subset.tsv"
        meta.to_csv(p, sep="\t", index=False)
        out = csc.validate_metadata(p)
        assert set(out["participant_label"]) == {"p1", "p2"}

    def test_missing_required_column_errors(self, tmp_path):
        # No CV_fold column — required by the subset tool.
        meta = pd.DataFrame({
            "participant_label": ["p1"], "specimen_label": ["s1"], "disease": ["HIV"],
        })
        p = tmp_path / "bad.tsv"
        meta.to_csv(p, sep="\t", index=False)
        with pytest.raises((ValueError, SystemExit, KeyError)):
            csc.validate_metadata(p)

    def test_legacy_fold_column_normalized(self, tmp_path):
        meta = pd.DataFrame({
            "participant_label": ["p1"], "specimen_label": ["s1"], "disease": ["HIV"],
            "malid_cross_validation_fold_id_when_in_test_set": [2],
        })
        p = tmp_path / "legacy.tsv"
        meta.to_csv(p, sep="\t", index=False)
        out = csc.validate_metadata(p)
        assert "CV_fold" in out.columns and int(out["CV_fold"].iloc[0]) == 2


class TestCopyOrLink:
    def test_symlink_creates_links(self, tmp_path):
        ref = _make_ref_cache(tmp_path / "ref", ["p1", "p2"])
        out = tmp_path / "out"
        csc.copy_or_link_files(["p1"], ref, out, use_symlinks=True)
        f = out / "participants" / "p1_clean.parquet"
        assert f.exists() and f.is_symlink()
        assert (out / "embeddings" / "p1_embeddings.npy").exists()
        # Only the requested participant is copied, not p2.
        assert not (out / "participants" / "p2_clean.parquet").exists()

    def test_copy_creates_real_files(self, tmp_path):
        ref = _make_ref_cache(tmp_path / "ref", ["p1"])
        out = tmp_path / "out"
        csc.copy_or_link_files(["p1"], ref, out, use_symlinks=False)
        f = out / "participants" / "p1_clean.parquet"
        assert f.exists() and not f.is_symlink()

    def test_missing_ref_file_detected(self, tmp_path):
        ref = _make_ref_cache(tmp_path / "ref", ["p1"])
        # Remove one required embedding file → validation must flag it.
        (ref / "embeddings" / "p1_embeddings.npy").unlink()
        with pytest.raises((ValueError, SystemExit)):
            csc.validate_ref_files(["p1"], ref)


# ===========================================================================
# Integration — real CLI against tests/test_data/ (fast: symlinks only)
# ===========================================================================


@pytest.mark.integration
class TestSubsetCLI:
    @pytest.fixture(autouse=True)
    def _ref_cache_ready(self):
        """create_subset_cache reads the reference cache's participants/ AND embeddings/.
        Make this test order-independent: (1) build the participant clean cache via the
        loader (another test may have cleared it), and (2) skip if precomputed embeddings
        are absent (they are gitignored — run_all_tests.py builds them via the slow
        embedding test; a bare unordered `pytest` on a fresh checkout would not have them).
        """
        from test_helpers import create_test_loader
        from malid_lite.dataloader.base import PreprocessingStage
        if not any((TEST_DATA_DIR / "embeddings").glob("*_embeddings.npy")):
            pytest.skip(
                "precomputed embeddings absent in tests/test_data/embeddings/ (gitignored) "
                "— see PIPELINE_GUIDE.md Section 3.4"
            )
        create_test_loader().get_all_data(PreprocessingStage.DOWNSAMPLED)

    def _folds01_metadata(self, out_dir: Path) -> Path:
        meta = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
        fold_col = next(
            c for c in ("CV_fold", "malid_cross_validation_fold_id_when_in_test_set")
            if c in meta.columns
        )
        sub = meta[meta[fold_col].astype(int).isin([0, 1])].copy()
        p = out_dir / "folds01_metadata.tsv"
        sub.to_csv(p, sep="\t", index=False)
        return p, sub

    def _run(self, args):
        return subprocess.run(
            [sys.executable, str(SCRIPT)] + args,
            capture_output=True, text=True, timeout=300,
        )

    def test_build_folds01_subset_symlinked(self):
        out = _out("folds01")
        subset_meta, sub_df = self._folds01_metadata(out)
        cache = out / "cache"
        r = self._run([
            "--ref-cache-dir", str(TEST_DATA_DIR),
            "--metadata-subset", str(subset_meta),
            "--dataset-name", "test_folds01",
            "--output-cache-dir", str(cache),
            "--symlink",
        ])
        assert r.returncode == 0, f"stderr:\n{r.stderr}"

        expected = sorted(sub_df["participant_label"].unique())
        # Every subset participant has both participant-cache files (symlinked).
        for p in expected:
            for suffix in ("_clean.parquet", "_stats.json"):
                f = cache / "participants" / f"{p}{suffix}"
                assert f.exists() and f.is_symlink(), f"missing {f}"
            assert (cache / "embeddings" / f"{p}_embeddings.npy").exists()
        # No fold-2 participant leaked into the subset cache.
        all_meta = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
        fold_col = "CV_fold" if "CV_fold" in all_meta.columns else \
            "malid_cross_validation_fold_id_when_in_test_set"
        fold2 = set(all_meta[all_meta[fold_col].astype(int) == 2]["participant_label"]) - set(expected)
        for p in list(fold2)[:5]:
            assert not (cache / "participants" / f"{p}_clean.parquet").exists()
        # Subset metadata written for the loader to pick up.
        assert (cache / "metadata_processed.tsv").exists() or (cache / "metadata.tsv").exists()

    def test_participant_not_in_ref_errors(self):
        out = _out("bad_participant")
        meta = pd.DataFrame({
            "participant_label": ["NOT_A_REAL_PARTICIPANT"], "specimen_label": ["s1"],
            "disease": ["HIV"], "CV_fold": [0],
        })
        subset_meta = out / "bad.tsv"
        meta.to_csv(subset_meta, sep="\t", index=False)
        r = self._run([
            "--ref-cache-dir", str(TEST_DATA_DIR),
            "--metadata-subset", str(subset_meta),
            "--dataset-name", "test_bad",
            "--output-cache-dir", str(out / "cache"),
            "--symlink",
        ])
        assert r.returncode != 0
        assert "not found" in (r.stderr + r.stdout).lower()

    def test_existing_output_dir_without_force_errors(self):
        out = _out("existing")
        subset_meta, _ = self._folds01_metadata(out)
        cache = out / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "sentinel.txt").write_text("x")  # non-empty → must refuse without --force
        r = self._run([
            "--ref-cache-dir", str(TEST_DATA_DIR),
            "--metadata-subset", str(subset_meta),
            "--dataset-name", "test_existing",
            "--output-cache-dir", str(cache),
            "--symlink",
        ])
        assert r.returncode != 0, f"expected refusal; stdout/stderr:\n{r.stdout}\n{r.stderr}"
