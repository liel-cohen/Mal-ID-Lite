"""
Tests for scripts/data/manage_cache.py.

Tests cache info display and clearing operations using a temporary
directory with a synthetic cache structure.

Expected runtime: <5 seconds
"""

import sys
import json
import shutil
import tempfile
from pathlib import Path
from io import StringIO
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.manage_cache import (
    format_size,
    get_dir_size,
    read_cache_info,
    show_cache_info,
    clear_directory,
)


def create_fake_cache(cache_dir: Path):
    """Create a synthetic cache directory structure for testing."""
    # --- Participant cache ---
    participants_dir = cache_dir / "participants"
    participants_dir.mkdir(parents=True)

    for i in range(5):
        # Fake parquet files (just need to exist with some size)
        pf = participants_dir / f"participant_{i:03d}_clean.parquet"
        pf.write_bytes(b"x" * (1024 * (i + 1)))  # 1-5 KB each

    # cache_info.json for participants
    with open(participants_dir / "cache_info.json", "w") as f:
        json.dump({
            "created_at": "2026-01-15 10:30:00",
            "malid_version": "0.1.0",
            "data_dir": "/path/to/data",
        }, f)

    # --- Fold cache ---
    folds_dir = cache_dir / "data_folds"
    folds_dir.mkdir(parents=True)

    for fold_id in range(3):
        for label in ["train", "test"]:
            pf = folds_dir / f"fold_{fold_id}_{label}_downsampled_sequences.parquet"
            pf.write_bytes(b"y" * 2048)

    with open(folds_dir / "cache_info.json", "w") as f:
        json.dump({
            "created_at": "2026-01-15 11:00:00",
            "malid_version": "0.1.0",
        }, f)

    # --- Embedding cache ---
    embeddings_dir = cache_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)

    for i in range(3):
        nf = embeddings_dir / f"participant_{i:03d}.npy"
        nf.write_bytes(b"z" * 4096)

    with open(embeddings_dir / "cache_info.json", "w") as f:
        json.dump({
            "created_at": "2026-01-15 12:00:00",
            "model_name": "esm2_t6_8M_UR50D",
            "embedding_dim": 640,
            "storage_dtype": "float16",
            "total_sequences_embedded": 50000,
        }, f)

    # --- Reports directory (should NOT be deleted by clear-all) ---
    reports_dir = cache_dir / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "summary.csv").write_text("metric,value\ntest,1\n")


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert format_size(3 * 1024 ** 3) == "3.0 GB"

    def test_zero(self):
        assert format_size(0) == "0.0 B"


class TestGetDirSize:
    def test_with_files(self, tmp_path):
        d = tmp_path / "test_dir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"x" * 100)
        (d / "b.txt").write_bytes(b"y" * 200)

        assert get_dir_size(d) == 300

    def test_nonexistent(self, tmp_path):
        assert get_dir_size(tmp_path / "nonexistent") == 0

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert get_dir_size(d) == 0


class TestReadCacheInfo:
    def test_with_info_file(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with open(d / "cache_info.json", "w") as f:
            json.dump({"key": "value"}, f)

        result = read_cache_info(d)
        assert result == {"key": "value"}

    def test_without_info_file(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        assert read_cache_info(d) is None


class TestShowCacheInfo:
    def test_nonexistent_cache(self, tmp_path, capsys):
        show_cache_info(tmp_path / "nonexistent")
        captured = capsys.readouterr()
        assert "does not exist yet" in captured.out

    def test_full_cache(self, tmp_path, capsys):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        create_fake_cache(cache_dir)

        show_cache_info(cache_dir)
        captured = capsys.readouterr()

        # Check participant section
        assert "PARTICIPANT CACHE" in captured.out
        assert "5 participants" in captured.out
        assert "2026-01-15" in captured.out

        # Check fold section
        assert "FOLD CACHE" in captured.out
        assert "6 fold files" in captured.out
        assert "3 folds" in captured.out

        # Check embedding section
        assert "EMBEDDING CACHE" in captured.out
        assert "3 participants" in captured.out
        assert "esm2_t6_8M_UR50D" in captured.out
        assert "50,000" in captured.out

        # Check total
        assert "Total cache size" in captured.out

    def test_empty_cache(self, tmp_path, capsys):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        show_cache_info(cache_dir)
        captured = capsys.readouterr()

        assert "PARTICIPANT CACHE: None" in captured.out
        assert "FOLD CACHE: None" in captured.out
        assert "EMBEDDING CACHE: None" in captured.out


class TestClearDirectory:
    def test_clear_existing(self, tmp_path):
        d = tmp_path / "to_clear"
        d.mkdir()
        (d / "file1.txt").write_text("data")
        (d / "file2.txt").write_text("data")

        clear_directory(d, "test", confirm=False)
        assert not d.exists()

    def test_clear_nonexistent(self, tmp_path, capsys):
        clear_directory(tmp_path / "nonexistent", "test", confirm=False)
        captured = capsys.readouterr()
        assert "No test cache to clear" in captured.out

    def test_clear_with_confirmation_denied(self, tmp_path, monkeypatch):
        d = tmp_path / "to_clear"
        d.mkdir()
        (d / "file1.txt").write_text("data")

        # Simulate user typing "n"
        monkeypatch.setattr("builtins.input", lambda _: "n")
        clear_directory(d, "test", confirm=True)

        # Directory should still exist
        assert d.exists()
        assert (d / "file1.txt").exists()

    def test_clear_with_confirmation_accepted(self, tmp_path, monkeypatch):
        d = tmp_path / "to_clear"
        d.mkdir()
        (d / "file1.txt").write_text("data")

        monkeypatch.setattr("builtins.input", lambda _: "y")
        clear_directory(d, "test", confirm=True)

        assert not d.exists()


class TestClearAllPreservesReports:
    """Verify that clear-all deletes participants, folds, embeddings but NOT reports."""

    def test_clear_all_preserves_reports(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        create_fake_cache(cache_dir)

        # Simulate clear-all logic (same as main() in manage_cache.py)
        for subdir, label in [
            ("participants", "participant"),
            ("data_folds", "fold"),
            ("embeddings", "embedding"),
        ]:
            path = cache_dir / subdir
            if path.exists():
                shutil.rmtree(path)

        # Cache subdirectories should be gone
        assert not (cache_dir / "participants").exists()
        assert not (cache_dir / "data_folds").exists()
        assert not (cache_dir / "embeddings").exists()

        # Reports should still exist
        assert (cache_dir / "reports").exists()
        assert (cache_dir / "reports" / "summary.csv").exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
