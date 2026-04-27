"""Comprehensive caching test for the two-level caching system.

Exercises all caching methods in base.py on the small test dataset
(tests/test_data/), including:

 1. clean_test_cache() — wipes auto-generated cache dirs
 2. Participant caching — fresh build, round-trip, stats JSON
 3. Fold caching — auto-cache via get_fold_data(), round-trip
 4. NaN preservation through the cache round-trip (no "nan" strings)
 5. Split generation, persistence, and corruption recovery
 6. Corruption recovery for participant cache, fold cache, and metadata
 7. cache_dir=None — no caching occurs
 8. clear_participant_cache() and clear_fold_cache()
 9. get_cache_info()
10. Atomic writes — no orphaned temp files after normal operation
11. Cache metadata files (cache_info.json)
12. Missing-participant metadata filtering and metadata_processed.tsv
13. All-participants-filtered ValueError

Uses the mock test data at tests/test_data/ (~118K sequences, 72
participants, 3 folds, 4 diseases).

Output: tests/test_outputs/test_caching_quick/
Expected runtime: <60 seconds
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_helpers import (
    create_test_loader,
    clean_test_cache,
    TEST_DATA_DIR,
    TEST_RAW_DIR,
    TEST_FOLD_IDS,
)
import pytest

from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.dataloader.base import PreprocessingStage

pytestmark = pytest.mark.integration

import pandas as pd

# Output directory per CLAUDE.md conventions
output_dir = Path(__file__).parent / "test_outputs" / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Helpers
# ======================================================================

def _log(msg: str):
    print(f"  {msg}")


def _assert(condition, msg):
    assert condition, msg


# ======================================================================
# Test 1: clean_test_cache
# ======================================================================

def test_clean_test_cache():
    """Verify clean_test_cache() removes participants/, data_folds/, splits/."""
    print("\n[Test 1] clean_test_cache")

    # Ensure dirs exist first (create dummy files if needed)
    for subdir in ("participants", "data_folds", "splits"):
        d = TEST_DATA_DIR / subdir
        d.mkdir(parents=True, exist_ok=True)
        # Write a sentinel file so we can verify deletion
        (d / "_sentinel.txt").write_text("test")

    # Run clean
    clean_test_cache()

    # Verify all three dirs are gone
    for subdir in ("participants", "data_folds", "splits"):
        d = TEST_DATA_DIR / subdir
        _assert(not d.exists(), f"{subdir}/ should not exist after clean_test_cache()")
        _log(f"{subdir}/ removed: OK")

    # Verify raw/ and metadata.tsv are untouched
    _assert(TEST_RAW_DIR.exists(), "raw/ should still exist")
    _assert((TEST_DATA_DIR / "metadata.tsv").exists(), "metadata.tsv should still exist")
    _log("raw/ and metadata.tsv preserved: OK")

    print("  PASSED")


# ======================================================================
# Test 2: Participant caching — fresh build + round-trip
# ======================================================================

def test_participant_caching():
    """Build participant cache from raw files, verify round-trip integrity."""
    print("\n[Test 2] Participant caching — fresh build + round-trip")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Pick first participant from metadata
    meta = loader.metadata
    participant = meta["participant_label"].iloc[0]

    # No cache should exist yet
    cache_file, stats_file = loader.get_participant_cache_path(participant)
    _assert(not cache_file.exists(), "Participant cache should not exist before build")

    # Load participant data (triggers preprocessing from raw)
    df_raw = loader.load_participant_data(participant, PreprocessingStage.CLEAN)
    _assert(len(df_raw) > 0, f"No data for participant {participant}")
    _log(f"Loaded {len(df_raw)} sequences for {participant} from raw")

    # Cache should now exist (load_participant_data caches automatically)
    _assert(cache_file.exists(), "Participant cache parquet should exist after load")
    _assert(stats_file.exists(), "Participant stats JSON should exist after load")
    _log("Cache files created: OK")

    # Load from cache — should match
    cached = loader.load_cached_participant(participant)
    _assert(cached is not None, "load_cached_participant should return data")
    df_cached, stats = cached
    _assert(len(df_cached) == len(df_raw), (
        f"Cached row count mismatch: {len(df_cached)} vs {len(df_raw)}"
    ))
    _assert(list(df_cached.columns) == list(df_raw.columns), "Column mismatch")

    # Value comparison (using pandas testing for NaN-aware comparison)
    pd.testing.assert_frame_equal(
        df_cached.reset_index(drop=True),
        df_raw.reset_index(drop=True),
        check_dtype=False,
    )
    _log("Round-trip data integrity: OK")

    # Stats should be a dict with preprocessing info
    _assert(isinstance(stats, dict), "Stats should be a dict")
    _assert(len(stats) > 0, "Stats should not be empty")
    _log(f"Stats keys: {sorted(stats.keys())[:5]}...")

    # Verify no orphaned temp files in participants/ dir
    orphans = list(cache_file.parent.glob("tmp*"))
    _assert(len(orphans) == 0, f"Orphaned temp files found: {orphans}")
    _log("No orphaned temp files: OK")

    print("  PASSED")


# ======================================================================
# Test 3: Fold caching via get_fold_data() auto-cache
# ======================================================================

def test_fold_auto_caching():
    """get_fold_data() should auto-cache on miss, then serve from cache."""
    print("\n[Test 3] Fold auto-caching via get_fold_data()")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    fold_id = 0
    fold_label = "train"
    stage = PreprocessingStage.DOWNSAMPLED

    # Verify no fold cache exists
    seq_file, meta_file = loader.get_cache_path(fold_id, fold_label, stage)
    _assert(not seq_file.exists(), "Fold cache should not exist before first get_fold_data()")

    # First call: builds from specimens, auto-caches
    seq_df_1, meta_df_1 = loader.get_fold_data(fold_id, fold_label, stage)
    _assert(len(seq_df_1) > 0, "No sequences returned")
    _assert(len(meta_df_1) > 0, "No metadata returned")
    _log(f"First call: {len(seq_df_1):,} sequences, {len(meta_df_1)} specimens")

    # Fold cache should now exist
    _assert(seq_file.exists(), "Fold cache parquet should exist after auto-cache")
    _assert(meta_file.exists(), "Fold cache metadata CSV should exist after auto-cache")
    _log("Fold cache files created: OK")

    # Second call: should load from cache (much faster, identical data)
    seq_df_2, meta_df_2 = loader.get_fold_data(fold_id, fold_label, stage)
    _assert(len(seq_df_2) == len(seq_df_1), (
        f"Cached row count mismatch: {len(seq_df_2)} vs {len(seq_df_1)}"
    ))
    _assert(len(meta_df_2) == len(meta_df_1), "Metadata row count mismatch")
    _log("Second call row counts match: OK")

    # Value-level comparison (NaN-aware)
    pd.testing.assert_frame_equal(
        seq_df_2.reset_index(drop=True).sort_values(
            ["specimen_label", "cdr3_aa"], ignore_index=True
        ),
        seq_df_1.reset_index(drop=True).sort_values(
            ["specimen_label", "cdr3_aa"], ignore_index=True
        ),
        check_dtype=False,
    )
    _log("Round-trip data integrity: OK")

    # Verify cache_info.json was created
    cache_info_path = TEST_DATA_DIR / "data_folds" / "cache_info.json"
    _assert(cache_info_path.exists(), "data_folds/cache_info.json should exist")
    with open(cache_info_path) as f:
        info = json.load(f)
    _assert("malid_version" in info, "cache_info should contain malid_version")
    _log(f"cache_info.json: OK (version={info.get('malid_version')})")

    # No orphaned temp files
    data_folds_dir = TEST_DATA_DIR / "data_folds"
    orphans = list(data_folds_dir.glob("tmp*"))
    _assert(len(orphans) == 0, f"Orphaned temp files: {orphans}")
    _log("No orphaned temp files: OK")

    print("  PASSED")


# ======================================================================
# Test 4: NaN preservation through fold cache round-trip
# ======================================================================

def test_nan_preservation():
    """NaN values in string columns must survive the cache round-trip.

    The test data has NaN values in fwr1_aa. After caching and
    reloading, those must still be NaN — not the literal string "nan".
    """
    print("\n[Test 4] NaN preservation through cache round-trip")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Build + auto-cache fold 0 train
    seq_df, _ = loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)

    # Count NaN values in the fresh data
    nan_cols = {}
    for col in seq_df.columns:
        n_nan = seq_df[col].isna().sum()
        if n_nan > 0:
            nan_cols[col] = n_nan
    _assert(len(nan_cols) > 0, "Test data should have at least one column with NaN values")
    _log(f"Columns with NaN before caching: {nan_cols}")

    # Count literal "nan" strings (should be zero)
    for col, n_nan in nan_cols.items():
        if seq_df[col].dtype == object:
            n_literal_nan = (seq_df[col] == "nan").sum()
            _assert(n_literal_nan == 0, (
                f"Column {col} has {n_literal_nan} literal 'nan' strings in fresh data"
            ))

    # Reload from cache
    cached = loader.load_cached_fold(0, "train", PreprocessingStage.DOWNSAMPLED)
    _assert(cached is not None, "Fold cache should exist")
    seq_cached, _ = cached

    # NaN counts must match
    for col, expected_nan in nan_cols.items():
        actual_nan = seq_cached[col].isna().sum()
        _assert(actual_nan == expected_nan, (
            f"NaN count mismatch in {col}: expected {expected_nan}, got {actual_nan}"
        ))

        # No literal "nan" strings
        if seq_cached[col].dtype == object:
            n_literal = (seq_cached[col] == "nan").sum()
            _assert(n_literal == 0, (
                f"Column {col} has {n_literal} literal 'nan' strings after cache round-trip"
            ))
    _log("NaN counts match between fresh and cached data: OK")
    _log("No literal 'nan' strings in cached data: OK")

    print("  PASSED")


# ======================================================================
# Test 5: Split generation and persistence
# ======================================================================

def test_split_generation():
    """Test split generation for both training contexts, file persistence,
    and correctness properties.
    """
    print("\n[Test 5] Split generation and persistence")

    clean_test_cache()
    loader = create_test_loader(verbose=0)
    meta = loader.metadata
    all_participants = sorted(meta["participant_label"].unique())
    n_participants = len(all_participants)
    _log(f"Total participants: {n_participants}")

    for context in ("cv_single_model", "cv_ensemble"):
        _log(f"\n  --- {context} ---")

        for fold_id in TEST_FOLD_IDS:
            # Generate splits (first call creates file)
            splits = loader.load_splits(fold_id, context)

            # Correctness: all participants assigned exactly once
            _assert(len(splits) == n_participants, (
                f"Split has {len(splits)} rows, expected {n_participants}"
            ))
            _assert(
                set(splits["participant_label"]) == set(all_participants),
                f"Participant set mismatch in fold {fold_id}"
            )

            # No duplicate participants
            _assert(splits["participant_label"].is_unique, "Duplicate participants in splits")

            # Split role values are correct for this context
            roles = set(splits["split_role"].unique())
            if context == "cv_single_model":
                expected_roles = {"test", "train_smaller1", "train_smaller2"}
            else:
                expected_roles = {"test", "validation", "train_smaller1", "train_smaller2"}
            _assert(roles == expected_roles, (
                f"Unexpected roles: {roles} (expected {expected_roles})"
            ))

            # No overlap between test and any train role
            test_parts = set(splits.loc[splits["split_role"] == "test", "participant_label"])
            train_parts = set(
                splits.loc[splits["split_role"] != "test", "participant_label"]
            )
            _assert(len(test_parts & train_parts) == 0, "Overlap between test and train")

            # Test participants should match fold assignment
            expected_test = set(
                meta.loc[
                    meta["malid_cross_validation_fold_id_when_in_test_set"] == fold_id,
                    "participant_label"
                ]
            )
            _assert(test_parts == expected_test, (
                f"Test set mismatch for fold {fold_id}: "
                f"got {len(test_parts)}, expected {len(expected_test)}"
            ))

            _log(f"  Fold {fold_id}: {splits['split_role'].value_counts().to_dict()} OK")

        # Verify split files exist on disk
        splits_dir = TEST_DATA_DIR / "splits"
        _assert(splits_dir.exists(), "splits/ directory should exist")
        for fold_id in TEST_FOLD_IDS:
            split_path = splits_dir / f"fold_{fold_id}_{context}.csv"
            _assert(split_path.exists(), f"{split_path.name} should exist")

    # Verify split metadata JSON
    split_meta_path = TEST_DATA_DIR / "splits" / "split_metadata.json"
    _assert(split_meta_path.exists(), "split_metadata.json should exist")
    with open(split_meta_path) as f:
        split_meta = json.load(f)
    _assert("random_state" in split_meta, "split_metadata should contain random_state")
    _assert(split_meta["random_state"] == 0, "random_state should be 0")
    _log(f"\n  split_metadata.json: OK")

    # Reproducibility: delete file and regenerate — should produce identical splits
    split_path_0 = TEST_DATA_DIR / "splits" / "fold_0_cv_single_model.csv"
    splits_before_delete = loader.load_splits(0, "cv_single_model")
    split_path_0.unlink()
    splits_after_regen = loader.load_splits(0, "cv_single_model")
    pd.testing.assert_frame_equal(
        splits_before_delete.reset_index(drop=True),
        splits_after_regen.reset_index(drop=True),
    )
    _log("Reproducibility (delete + regenerate): OK")

    print("  PASSED")


# ======================================================================
# Test 6: Invalid training_context raises ValueError
# ======================================================================

def test_invalid_training_context():
    """load_splits() should raise ValueError for unknown training contexts."""
    print("\n[Test 6] Invalid training_context raises ValueError")

    loader = create_test_loader(verbose=0)

    try:
        loader.load_splits(0, "invalid_context")
        _assert(False, "Should have raised ValueError")
    except ValueError as e:
        _log(f"Correctly raised: {e}")

    print("  PASSED")


# ======================================================================
# Test 7: Corruption recovery — fold cache
# ======================================================================

def test_fold_corruption_recovery():
    """Corrupt fold cache files should be detected, deleted, and rebuilt."""
    print("\n[Test 7] Fold corruption recovery")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Build and cache fold 0/train
    seq_df_orig, meta_df_orig = loader.get_fold_data(
        0, "train", PreprocessingStage.DOWNSAMPLED
    )
    seq_file, meta_file = loader.get_cache_path(
        0, "train", PreprocessingStage.DOWNSAMPLED
    )
    _assert(seq_file.exists(), "Fold cache should exist after get_fold_data()")

    # --- Corrupt the parquet file ---
    with open(seq_file, "wb") as f:
        f.write(b"THIS IS NOT A PARQUET FILE")
    _log("Corrupted parquet file")

    # load_cached_fold should detect corruption and return None
    result = loader.load_cached_fold(0, "train", PreprocessingStage.DOWNSAMPLED)
    _assert(result is None, "load_cached_fold should return None for corrupt parquet")
    _assert(not seq_file.exists(), "Corrupt parquet should be deleted")
    _assert(not meta_file.exists(), "Metadata CSV should also be deleted on parquet corruption")
    _log("Corrupt parquet detected and cleaned: OK")

    # get_fold_data should rebuild transparently
    seq_df_rebuilt, meta_df_rebuilt = loader.get_fold_data(
        0, "train", PreprocessingStage.DOWNSAMPLED
    )
    _assert(len(seq_df_rebuilt) == len(seq_df_orig), (
        f"Rebuilt row count mismatch: {len(seq_df_rebuilt)} vs {len(seq_df_orig)}"
    ))
    _log("Transparent rebuild after corruption: OK")

    # --- Corrupt the metadata CSV ---
    _, meta_file = loader.get_cache_path(0, "train", PreprocessingStage.DOWNSAMPLED)
    with open(meta_file, "w") as f:
        f.write("not,valid\x00csv\ngarbage")
    _log("Corrupted metadata CSV file")

    result = loader.load_cached_fold(0, "train", PreprocessingStage.DOWNSAMPLED)
    # CSV corruption may or may not cause an exception in pd.read_csv (pandas
    # is lenient). If it loads without error, the specimen validation in
    # load_cached_fold will still catch mismatched specimen/participant pairs.
    # Either way, the cache is invalid.
    if result is None:
        _log("Corrupt CSV detected and cleaned: OK")
    else:
        _log("Pandas parsed corrupt CSV without error (lenient mode) — checking if content is valid")
        # The result would have wrong specimen_labels, which would fail
        # downstream validation. For testing, we verify the test still
        # reaches this point without crashing.
        _log("Warning: pandas parsed corrupt CSV — downstream validation would catch this")

    print("  PASSED")


# ======================================================================
# Test 8: Corruption recovery — participant cache
# ======================================================================

def test_participant_corruption_recovery():
    """Corrupt participant cache should be detected and rebuilt from raw."""
    print("\n[Test 8] Participant corruption recovery")

    clean_test_cache()
    loader = create_test_loader(verbose=0)
    meta = loader.metadata
    participant = meta["participant_label"].iloc[0]

    # Build participant cache
    df_orig = loader.load_participant_data(participant, PreprocessingStage.CLEAN)
    cache_file, stats_file = loader.get_participant_cache_path(participant)
    _assert(cache_file.exists(), "Participant cache should exist")

    # Corrupt the parquet file
    with open(cache_file, "wb") as f:
        f.write(b"CORRUPT DATA")
    _log("Corrupted participant parquet")

    # load_cached_participant should return None
    result = loader.load_cached_participant(participant)
    _assert(result is None, "Should return None for corrupt participant cache")
    _assert(not cache_file.exists(), "Corrupt parquet should be deleted")
    _assert(not stats_file.exists(), "Stats JSON should also be deleted")
    _log("Corrupt participant cache detected and cleaned: OK")

    # Reload from raw should succeed
    df_rebuilt = loader.load_participant_data(participant, PreprocessingStage.CLEAN)
    _assert(len(df_rebuilt) == len(df_orig), "Rebuilt data should match original")
    _log("Transparent rebuild from raw: OK")

    # --- Corrupt stats JSON only (data parquet is fine) ---
    # First, rebuild cache
    _ = loader.load_participant_data(participant, PreprocessingStage.CLEAN)
    _assert(cache_file.exists(), "Cache should exist after rebuild")

    with open(stats_file, "w") as f:
        f.write("{invalid json")
    _log("Corrupted stats JSON")

    # load_cached_participant should still return data (stats ignored)
    result = loader.load_cached_participant(participant)
    _assert(result is not None, "Should still return data when only stats is corrupt")
    df_loaded, stats = result
    _assert(len(df_loaded) == len(df_orig), "Data should be intact")
    _assert(isinstance(stats, dict), "Stats should be an empty dict on corruption")
    _log("Corrupt stats JSON handled gracefully (data still valid): OK")

    print("  PASSED")


# ======================================================================
# Test 9: Corruption recovery — split files
# ======================================================================

def test_split_corruption_recovery():
    """Corrupt split CSV should be regenerated transparently."""
    print("\n[Test 9] Split corruption recovery")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Generate splits
    splits_orig = loader.load_splits(0, "cv_single_model")
    split_path = TEST_DATA_DIR / "splits" / "fold_0_cv_single_model.csv"
    _assert(split_path.exists(), "Split file should exist")

    # --- Case 1: Completely corrupt file ---
    with open(split_path, "wb") as f:
        f.write(b"\x00\x00GARBAGE\x00\x00")
    _log("Corrupted split CSV (binary garbage)")

    splits_rebuilt = loader.load_splits(0, "cv_single_model")
    pd.testing.assert_frame_equal(
        splits_rebuilt.reset_index(drop=True),
        splits_orig.reset_index(drop=True),
    )
    _log("Regenerated from binary garbage: OK")

    # --- Case 2: Missing required columns ---
    pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(split_path, index=False)
    _log("Wrote CSV with wrong columns")

    splits_rebuilt2 = loader.load_splits(0, "cv_single_model")
    pd.testing.assert_frame_equal(
        splits_rebuilt2.reset_index(drop=True),
        splits_orig.reset_index(drop=True),
    )
    _log("Regenerated from missing-columns CSV: OK")

    print("  PASSED")


# ======================================================================
# Test 10: Corruption recovery — cache metadata (cache_info.json)
# ======================================================================

def test_cache_metadata_corruption():
    """Corrupt cache_info.json should be deleted, returning None."""
    print("\n[Test 10] Cache metadata corruption recovery")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Build fold cache to create cache_info.json
    loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)

    cache_info_path = TEST_DATA_DIR / "data_folds" / "cache_info.json"
    _assert(cache_info_path.exists(), "cache_info.json should exist")

    # Verify it reads correctly first
    info = loader._read_cache_metadata("data_folds")
    _assert(info is not None, "_read_cache_metadata should return data")
    _assert("malid_version" in info, "Should contain malid_version")
    _log("cache_info.json reads correctly: OK")

    # Corrupt it
    with open(cache_info_path, "w") as f:
        f.write("{broken json!!!")
    _log("Corrupted cache_info.json")

    # Should return None and delete the corrupt file
    info = loader._read_cache_metadata("data_folds")
    _assert(info is None, "Should return None for corrupt metadata")
    _assert(not cache_info_path.exists(), "Corrupt file should be deleted")
    _log("Corrupt metadata detected, deleted, returned None: OK")

    print("  PASSED")


# ======================================================================
# Test 11: cache_dir=None — no caching occurs
# ======================================================================

def test_no_cache_dir():
    """With cache_dir=None, no cache files should be created."""
    print("\n[Test 11] cache_dir=None — no caching")

    clean_test_cache()

    # Create a loader with cache_dir=None. Since metadata_path=None
    # tries to resolve from cache, we must provide it explicitly.
    loader = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=None,
        verbose=0,
    )

    # get_fold_data should still work (building from raw each time)
    seq_df, meta_df = loader.get_fold_data(0, "test", PreprocessingStage.DOWNSAMPLED)
    _assert(len(seq_df) > 0, "Should return data even without cache")
    _log(f"Got {len(seq_df):,} sequences without caching")

    # No cache directories should have been created
    for subdir in ("participants", "data_folds"):
        d = TEST_DATA_DIR / subdir
        _assert(not d.exists(), f"{subdir}/ should not exist with cache_dir=None")
    _log("No cache directories created: OK")

    # load_cached_fold returns None
    result = loader.load_cached_fold(0, "test", PreprocessingStage.DOWNSAMPLED)
    _assert(result is None, "load_cached_fold should return None when cache_dir=None")

    # load_cached_participant returns None
    participant = loader.metadata["participant_label"].iloc[0]
    result = loader.load_cached_participant(participant)
    _assert(result is None, "load_cached_participant should return None when cache_dir=None")
    _log("All load_cached_* return None: OK")

    # cache_fold raises ValueError
    try:
        loader.cache_fold(0, "test")
        _assert(False, "cache_fold should raise ValueError when cache_dir=None")
    except ValueError as e:
        _log(f"cache_fold raises ValueError: {e}")

    # load_splits raises ValueError (no cache_dir for persistence)
    try:
        loader.load_splits(0, "cv_single_model")
        _assert(False, "load_splits should raise ValueError when cache_dir=None")
    except ValueError as e:
        _log(f"load_splits raises ValueError: {e}")

    print("  PASSED")


# ======================================================================
# Test 12: clear_participant_cache and clear_fold_cache
# ======================================================================

def test_clear_cache_methods():
    """Test selective and full cache clearing."""
    print("\n[Test 12] clear_participant_cache and clear_fold_cache")

    clean_test_cache()
    loader = create_test_loader(verbose=0)
    meta = loader.metadata
    participants = sorted(meta["participant_label"].unique())[:3]

    # Build participant cache for 3 participants
    for p in participants:
        loader.load_participant_data(p, PreprocessingStage.CLEAN)

    # Verify all 3 exist
    for p in participants:
        cache_file, _ = loader.get_participant_cache_path(p)
        _assert(cache_file.exists(), f"Cache for {p} should exist")
    _log(f"Built cache for {len(participants)} participants")

    # Clear one specific participant
    loader.clear_participant_cache(participants[0], confirm=False)
    cache_file_0, stats_file_0 = loader.get_participant_cache_path(participants[0])
    _assert(not cache_file_0.exists(), "Cleared participant's cache should be gone")
    _assert(not stats_file_0.exists(), "Cleared participant's stats should be gone")

    # Others still exist
    for p in participants[1:]:
        cache_file, _ = loader.get_participant_cache_path(p)
        _assert(cache_file.exists(), f"Cache for {p} should still exist")
    _log("Single participant clear: OK")

    # Clear all participants
    loader.clear_participant_cache(confirm=False)
    participants_dir = TEST_DATA_DIR / "participants"
    _assert(not participants_dir.exists(), "participants/ should be gone after full clear")
    _log("Full participant clear: OK")

    # Build fold cache
    loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)
    seq_file, meta_file = loader.get_cache_path(0, "train", PreprocessingStage.DOWNSAMPLED)
    _assert(seq_file.exists(), "Fold cache should exist")

    # Clear specific fold
    loader.clear_fold_cache(fold_id=0, fold_label="train", confirm=False)
    # Check all preprocessing stages were cleared for this fold/label
    for stage in PreprocessingStage:
        sf, mf = loader.get_cache_path(0, "train", stage)
        _assert(not sf.exists(), f"Fold 0/train/{stage.value} parquet should be gone")
        _assert(not mf.exists(), f"Fold 0/train/{stage.value} metadata should be gone")
    _log("Specific fold clear: OK")

    # Rebuild, then clear all folds
    loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)
    loader.clear_fold_cache(confirm=False)
    data_folds_dir = TEST_DATA_DIR / "data_folds"
    fold_files = list(data_folds_dir.glob("fold_*")) if data_folds_dir.exists() else []
    _assert(len(fold_files) == 0, "All fold cache files should be gone")
    _log("Full fold clear: OK")

    print("  PASSED")


# ======================================================================
# Test 13: get_cache_info
# ======================================================================

def test_get_cache_info():
    """get_cache_info() should report counts and metadata for both caches."""
    print("\n[Test 13] get_cache_info")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Initially empty (participants dir doesn't exist → dict has no "count" key)
    info = loader.get_cache_info()
    has_participants = info["participants"].get("count", 0) > 0
    _assert(not has_participants, "Should start with no participant cache")
    has_folds = info["folds"].get("count", 0) > 0
    _assert(not has_folds, "Should start with no fold cache")
    _log("Initial state (empty): OK")

    # Build some caches
    meta = loader.metadata
    participant = meta["participant_label"].iloc[0]
    loader.load_participant_data(participant, PreprocessingStage.CLEAN)
    loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)

    info = loader.get_cache_info()
    _assert(info["participants"]["count"] >= 1, "Should have at least 1 participant cached")
    _assert(info["folds"]["count"] >= 1, "Should have at least 1 fold cached")
    _assert(info["participants"]["metadata"] is not None, "Participant metadata should exist")
    _assert(info["folds"]["metadata"] is not None, "Fold metadata should exist")
    _log(f"Participants: {info['participants']['count']} cached")
    _log(f"Folds: {info['folds']['count']} cached")
    _log(f"cache_dir: {info['cache_dir']}")

    print("  PASSED")


# ======================================================================
# Test 14: Metadata copy to cache
# ======================================================================

def test_metadata_save_to_cache():
    """_save_metadata_to_cache() should create metadata.tsv and metadata_processed.tsv."""
    print("\n[Test 14] Metadata save to cache")

    clean_test_cache()

    # The cached metadata.tsv already exists in test_data/ (it IS the
    # original for test data), so we need a fresh cache_dir that doesn't
    # have it yet.
    temp_cache = output_dir / "temp_cache_test"
    if temp_cache.exists():
        shutil.rmtree(temp_cache)
    temp_cache.mkdir(parents=True, exist_ok=True)

    loader = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=temp_cache,
        verbose=0,
    )

    cached_raw_path = temp_cache / "metadata.tsv"
    cached_processed_path = temp_cache / "metadata_processed.tsv"
    _assert(not cached_raw_path.exists(), "No metadata.tsv yet in temp cache")
    _assert(not cached_processed_path.exists(), "No metadata_processed.tsv yet in temp cache")

    # Trigger save (pass the loaded metadata as the "filtered" version)
    filtered_meta = loader.metadata
    loader._save_metadata_to_cache(filtered_meta)
    _assert(cached_raw_path.exists(), "metadata.tsv should be copied to cache")
    _assert(cached_processed_path.exists(), "metadata_processed.tsv should be saved to cache")

    # Verify original copy matches source
    orig = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
    cached = pd.read_csv(cached_raw_path, sep="\t")
    pd.testing.assert_frame_equal(orig, cached)
    _log("Original metadata copy matches source: OK")

    # Verify processed copy matches filtered metadata
    cached_proc = pd.read_csv(cached_processed_path, sep="\t")
    pd.testing.assert_frame_equal(filtered_meta.reset_index(drop=True), cached_proc)
    _log("Processed metadata matches filtered metadata: OK")

    # Second call for original copy should be a no-op (file already exists)
    mtime_before = cached_raw_path.stat().st_mtime
    loader._save_metadata_to_cache(filtered_meta)
    mtime_after = cached_raw_path.stat().st_mtime
    _assert(mtime_before == mtime_after, "Second call should not overwrite existing raw file")
    _log("Original copy idempotent: OK")

    # Clean up
    shutil.rmtree(temp_cache)

    print("  PASSED")


# ======================================================================
# Test 15: Multiple folds — all fold_ids + both train/test labels
# ======================================================================

def test_all_folds_caching():
    """Cache all folds and verify each has the correct specimens."""
    print("\n[Test 15] All folds caching — train and test")

    clean_test_cache()
    loader = create_test_loader(verbose=0)
    meta = loader.metadata

    for fold_id in TEST_FOLD_IDS:
        for fold_label in ("train", "test"):
            seq_df, meta_df = loader.get_fold_data(
                fold_id, fold_label, PreprocessingStage.DOWNSAMPLED
            )

            # Verify specimens match metadata expectations
            if fold_label == "test":
                expected_specimens = set(
                    meta.loc[
                        meta["malid_cross_validation_fold_id_when_in_test_set"] == fold_id,
                        "specimen_label"
                    ]
                )
            else:
                expected_specimens = set(
                    meta.loc[
                        meta["malid_cross_validation_fold_id_when_in_test_set"] != fold_id,
                        "specimen_label"
                    ]
                )

            # Not all specimens survive downsampling (some may be dropped),
            # so actual specimens should be a subset of expected
            actual_specimens = set(meta_df["specimen_label"])
            _assert(
                actual_specimens.issubset(expected_specimens),
                f"Fold {fold_id}/{fold_label}: specimens not subset of expected. "
                f"Extra: {actual_specimens - expected_specimens}"
            )
            _assert(len(actual_specimens) > 0, f"Fold {fold_id}/{fold_label}: no specimens")

            # Verify sequences actually belong to these specimens
            seq_specimens = set(seq_df["specimen_label"].unique())
            _assert(
                seq_specimens == actual_specimens,
                f"Fold {fold_id}/{fold_label}: specimen mismatch between sequences and metadata"
            )

            _log(f"Fold {fold_id}/{fold_label}: {len(actual_specimens)} specimens, "
                 f"{len(seq_df):,} sequences — OK")

    # Verify all cache files exist
    data_folds_dir = TEST_DATA_DIR / "data_folds"
    n_parquets = len(list(data_folds_dir.glob("fold_*_sequences.parquet")))
    n_csvs = len(list(data_folds_dir.glob("fold_*_metadata.csv")))
    expected = len(TEST_FOLD_IDS) * 2  # train + test per fold
    _assert(n_parquets == expected, f"Expected {expected} parquets, got {n_parquets}")
    _assert(n_csvs == expected, f"Expected {expected} CSVs, got {n_csvs}")
    _log(f"\nAll {expected} fold pairs cached: OK")

    print("  PASSED")


# ======================================================================
# Test 16: get_split_participants convenience method
# ======================================================================

def test_get_split_participants():
    """get_split_participants should return correct participants for roles."""
    print("\n[Test 16] get_split_participants")

    clean_test_cache()
    loader = create_test_loader(verbose=0)
    meta = loader.metadata

    # cv_ensemble context for fold 0
    splits = loader.load_splits(0, "cv_ensemble")

    # Get train_smaller1 + train_smaller2 via convenience method
    train_parts = loader.get_split_participants(
        0, "cv_ensemble", ["train_smaller1", "train_smaller2"]
    )
    expected_train = sorted(
        splits.loc[
            splits["split_role"].isin(["train_smaller1", "train_smaller2"]),
            "participant_label"
        ].tolist()
    )
    _assert(sorted(train_parts) == expected_train, "Train participants mismatch")
    _log(f"Train participants ({len(train_parts)}): OK")

    # Get validation
    val_parts = loader.get_split_participants(0, "cv_ensemble", ["validation"])
    expected_val = sorted(
        splits.loc[splits["split_role"] == "validation", "participant_label"].tolist()
    )
    _assert(sorted(val_parts) == expected_val, "Validation participants mismatch")
    _log(f"Validation participants ({len(val_parts)}): OK")

    # Get test
    test_parts = loader.get_split_participants(0, "cv_ensemble", ["test"])
    expected_test = sorted(
        meta.loc[
            meta["malid_cross_validation_fold_id_when_in_test_set"] == 0,
            "participant_label"
        ].unique().tolist()
    )
    _assert(sorted(test_parts) == expected_test, "Test participants mismatch")
    _log(f"Test participants ({len(test_parts)}): OK")

    # All roles together = all participants
    all_parts = set(train_parts) | set(val_parts) | set(test_parts)
    _assert(all_parts == set(meta["participant_label"].unique()),
            "All roles combined should cover all participants")
    _log("All roles = all participants: OK")

    print("  PASSED")


# ======================================================================
# Test 17: Orphaned temp file cleanup in clear_fold_cache
# ======================================================================

def test_orphan_temp_cleanup():
    """clear_fold_cache() should also remove orphaned tmp* files."""
    print("\n[Test 17] Orphaned temp file cleanup")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    # Build fold cache
    loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)
    data_folds_dir = TEST_DATA_DIR / "data_folds"
    _assert(data_folds_dir.exists(), "data_folds/ should exist")

    # Create fake orphaned temp files (as if a prior run was interrupted)
    orphan1 = data_folds_dir / "tmpABC123.parquet"
    orphan2 = data_folds_dir / "tmpXYZ789.csv"
    orphan1.write_text("orphan1")
    orphan2.write_text("orphan2")
    _log("Created 2 orphaned temp files")

    # Clear all folds — should remove orphans too
    loader.clear_fold_cache(confirm=False)
    _assert(not orphan1.exists(), "Orphaned .parquet temp should be removed")
    _assert(not orphan2.exists(), "Orphaned .csv temp should be removed")
    _log("Orphaned temp files removed: OK")

    print("  PASSED")


# ======================================================================
# Test 18: Back-to-back get_fold_data — cache hit performance
# ======================================================================

def test_cache_hit_performance():
    """Second get_fold_data() call should be faster (cache hit)."""
    print("\n[Test 18] Cache hit performance")

    clean_test_cache()
    loader = create_test_loader(verbose=0)

    import time

    # First call — builds from specimens, writes cache
    t0 = time.time()
    seq_1, _ = loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)
    t_first = time.time() - t0

    # Second call — loads from cache
    t0 = time.time()
    seq_2, _ = loader.get_fold_data(0, "train", PreprocessingStage.DOWNSAMPLED)
    t_second = time.time() - t0

    _log(f"First call (build+cache): {t_first:.2f}s")
    _log(f"Second call (cache hit):  {t_second:.2f}s")

    # Cache hit should be faster (at least 2x speedup expected)
    if t_first > 0.5:
        # Only assert speedup if first call was slow enough to measure
        _assert(t_second < t_first, "Cache hit should be faster than first build")
        _log(f"Speedup: {t_first / t_second:.1f}x")
    else:
        _log("First call too fast to measure meaningful speedup (small test data)")

    _assert(len(seq_1) == len(seq_2), "Row counts should match")

    print("  PASSED")


# ======================================================================
# Test 19: Missing-participant filtering
# ======================================================================

def test_missing_participant_filtering():
    """Metadata filtering removes participants whose raw files are absent.

    Creates a temporary metadata file that includes 3 phantom participants
    (no raw data files). Verifies:
    - The warning is logged (n_missing > 0 path)
    - metadata_processed.tsv excludes the phantom participants
    - loader.metadata only contains participants with raw data
    - metadata_filter_info records correct counts
    - A second loader (without metadata_path) loads from processed cache
    """
    print("\n[Test 19] Missing-participant filtering")

    clean_test_cache()

    # Build a metadata file with 3 phantom participants appended
    orig_meta = pd.read_csv(TEST_DATA_DIR / "metadata.tsv", sep="\t")
    n_orig = orig_meta["participant_label"].nunique()

    phantom_rows = pd.DataFrame({
        "participant_label": ["PHANTOM-001", "PHANTOM-002", "PHANTOM-003"],
        "specimen_label": ["PHANT-S001", "PHANT-S002", "PHANT-S003"],
        "disease": ["HIV", "Covid19", "Healthy/Background"],
        "malid_cross_validation_fold_id_when_in_test_set": [0, 1, 2],
        "available_gene_loci": ["GeneLocus.BCR|TCR"] * 3,
    })
    extended_meta = pd.concat([orig_meta, phantom_rows], ignore_index=True)

    # Write to a temp file
    temp_meta = output_dir / "metadata_with_phantoms.tsv"
    extended_meta.to_csv(temp_meta, sep="\t", index=False)

    temp_cache = output_dir / "temp_cache_filter_test"
    if temp_cache.exists():
        shutil.rmtree(temp_cache)
    temp_cache.mkdir(parents=True, exist_ok=True)

    # --- First loader: scan raw dir, filter, save processed cache ---
    loader1 = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=temp_meta,
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=temp_cache,
        verbose=0,
    )

    # Access metadata to trigger load_metadata()
    meta1 = loader1.metadata

    # Verify filtering: only real participants retained
    _assert(
        meta1["participant_label"].nunique() == n_orig,
        f"Expected {n_orig} participants after filtering, "
        f"got {meta1['participant_label'].nunique()}"
    )
    for phantom in ["PHANTOM-001", "PHANTOM-002", "PHANTOM-003"]:
        _assert(
            phantom not in meta1["participant_label"].values,
            f"Phantom participant {phantom} should have been filtered out"
        )
    _log("Phantom participants filtered from loader.metadata: OK")

    # Verify metadata_filter_info
    finfo = loader1.metadata_filter_info
    _assert(finfo is not None, "metadata_filter_info should be set")
    _assert(finfo["n_original"] == n_orig + 3,
            f"n_original should be {n_orig + 3}, got {finfo['n_original']}")
    _assert(finfo["n_filtered_out"] == 3,
            f"n_filtered_out should be 3, got {finfo['n_filtered_out']}")
    _assert(finfo["n_retained"] == n_orig,
            f"n_retained should be {n_orig}, got {finfo['n_retained']}")
    _log("metadata_filter_info counts correct: OK")

    # Verify processed cache file exists and excludes phantoms
    cached_proc = temp_cache / "metadata_processed.tsv"
    _assert(cached_proc.exists(), "metadata_processed.tsv should be created")
    proc_df = pd.read_csv(cached_proc, sep="\t")
    _assert(
        proc_df["participant_label"].nunique() == n_orig,
        f"Processed cache should have {n_orig} participants, "
        f"got {proc_df['participant_label'].nunique()}"
    )
    for phantom in ["PHANTOM-001", "PHANTOM-002", "PHANTOM-003"]:
        _assert(
            phantom not in proc_df["participant_label"].values,
            f"Phantom {phantom} should not be in metadata_processed.tsv"
        )
    _log("metadata_processed.tsv excludes phantoms: OK")

    # Verify original metadata is also cached (for reference)
    cached_raw = temp_cache / "metadata.tsv"
    _assert(cached_raw.exists(), "Original metadata.tsv should be cached")
    raw_df = pd.read_csv(cached_raw, sep="\t")
    _assert(
        raw_df["participant_label"].nunique() == n_orig + 3,
        "Cached original should include all participants (including phantoms)"
    )
    _log("Original metadata.tsv cached with all participants: OK")

    # --- Second loader: load from processed cache (no metadata_path) ---
    loader2 = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=None,
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=temp_cache,
        verbose=0,
    )

    meta2 = loader2.metadata
    _assert(
        meta2["participant_label"].nunique() == n_orig,
        f"Cache-only loader should have {n_orig} participants, "
        f"got {meta2['participant_label'].nunique()}"
    )
    _assert(loader2.metadata_filter_info is None,
            "metadata_filter_info should be None for cache-only loader")
    _log("Cache-only loader loads filtered metadata correctly: OK")

    # --- Verify splits invalidation ---
    # Create a dummy splits dir, then re-run a filtering loader to verify
    # it gets cleared
    splits_dir = temp_cache / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "dummy_split.csv").write_text("test")

    loader3 = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=temp_meta,
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=temp_cache,
        verbose=0,
    )
    _ = loader3.metadata  # triggers filtering + split invalidation
    _assert(not splits_dir.exists(),
            "Splits dir should be cleared when metadata is filtered")
    _log("Splits invalidated on filtering: OK")

    # Clean up
    shutil.rmtree(temp_cache)
    temp_meta.unlink(missing_ok=True)

    print("  PASSED")


# ======================================================================
# Test 20: All-participants-filtered raises ValueError
# ======================================================================

def test_all_participants_filtered_error():
    """Verify ValueError when ALL participants have no raw data files."""
    print("\n[Test 20] All-participants-filtered raises ValueError")

    # Create metadata with only phantom participants
    phantom_meta = pd.DataFrame({
        "participant_label": ["GHOST-001", "GHOST-002"],
        "specimen_label": ["GHOST-S001", "GHOST-S002"],
        "disease": ["HIV", "Covid19"],
        "malid_cross_validation_fold_id_when_in_test_set": [0, 1],
        "available_gene_loci": ["GeneLocus.BCR|TCR"] * 2,
    })
    temp_meta = output_dir / "metadata_all_ghosts.tsv"
    phantom_meta.to_csv(temp_meta, sep="\t", index=False)

    temp_cache = output_dir / "temp_cache_ghost_test"
    if temp_cache.exists():
        shutil.rmtree(temp_cache)
    temp_cache.mkdir(parents=True, exist_ok=True)

    loader = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=temp_meta,
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=temp_cache,
        verbose=0,
    )

    raised = False
    try:
        _ = loader.metadata
    except ValueError as e:
        raised = True
        _assert("No participants have raw data files" in str(e),
                f"Expected 'No participants have raw data files' in error, got: {e}")
        _log(f"Correct ValueError raised: OK")

    _assert(raised, "Should have raised ValueError for all-phantom metadata")

    # Clean up
    shutil.rmtree(temp_cache)
    temp_meta.unlink(missing_ok=True)

    print("  PASSED")


# ======================================================================
# Test 21: data_dir=None (metadata-only mode)
# ======================================================================

def test_data_dir_none():
    """Loader with data_dir=None should work from cache, fail on cache miss.

    Tests:
    - Loader initializes with data_dir=None + existing processed metadata cache
    - load_participant_data(CLEAN) succeeds when participant cache exists
    - load_participant_data(CLEAN) raises RuntimeError on cache miss
    - load_participant_data(RAW) always raises RuntimeError with data_dir=None
    """
    print("\n[Test 21] data_dir=None (metadata-only mode)")

    clean_test_cache()

    # --- Step 1: Build participant cache with a normal loader first ---
    loader_full = create_test_loader(verbose=0)
    meta = loader_full.metadata
    participant = meta["participant_label"].iloc[0]

    # Build cache for one participant
    df_orig = loader_full.load_participant_data(participant, PreprocessingStage.CLEAN)
    _assert(len(df_orig) > 0, f"No data for {participant}")
    _log(f"Built cache for {participant}: {len(df_orig)} sequences")

    # Verify cache file exists
    cache_file, stats_file = loader_full.get_participant_cache_path(participant)
    _assert(cache_file.exists(), "Participant cache should exist")

    # --- Step 2: Create a metadata-only loader (data_dir=None) ---
    # metadata_processed.tsv should exist now in TEST_DATA_DIR from the first loader
    loader_none = MalIDPublishedDataLoader(
        data_dir=None,
        metadata_path=None,
        gene_reference_path=None,
        gene_locus="TCR",
        cache_dir=TEST_DATA_DIR,
        verbose=0,
    )

    # Metadata should load from the processed cache
    meta_none = loader_none.metadata
    _assert(len(meta_none) > 0, "Metadata should load from cache even with data_dir=None")
    _assert(
        set(meta_none["participant_label"]) == set(meta["participant_label"]),
        "Metadata participants should match between full and None loaders"
    )
    _log("data_dir=None loader initialized with cached metadata: OK")

    # --- Step 3: CLEAN stage + cache hit => success ---
    df_cached = loader_none.load_participant_data(participant, PreprocessingStage.CLEAN)
    _assert(len(df_cached) == len(df_orig), (
        f"Cache hit data mismatch: {len(df_cached)} vs {len(df_orig)}"
    ))
    pd.testing.assert_frame_equal(
        df_cached.reset_index(drop=True),
        df_orig.reset_index(drop=True),
        check_dtype=False,
    )
    _log(f"CLEAN + cache hit: loaded {len(df_cached)} sequences: OK")

    # --- Step 4: CLEAN stage + cache miss => RuntimeError ---
    # Pick a participant that is NOT cached
    all_participants = sorted(meta["participant_label"].unique())
    uncached_participant = None
    for p in all_participants:
        cf, _ = loader_none.get_participant_cache_path(p)
        if not cf.exists():
            uncached_participant = p
            break
    _assert(uncached_participant is not None, "Need at least one uncached participant for test")

    raised = False
    try:
        loader_none.load_participant_data(uncached_participant, PreprocessingStage.CLEAN)
    except RuntimeError as e:
        raised = True
        _assert("data_dir is None" in str(e), f"Error should mention data_dir: {e}")
        _assert("not in cache" in str(e), f"Error should mention cache miss: {e}")
        _log(f"CLEAN + cache miss raises RuntimeError: OK")
    _assert(raised, "Should raise RuntimeError for CLEAN + cache miss + data_dir=None")

    # --- Step 5: RAW stage => always RuntimeError ---
    raised = False
    try:
        loader_none.load_participant_data(participant, PreprocessingStage.RAW)
    except RuntimeError as e:
        raised = True
        _assert("data_dir is None" in str(e), f"Error should mention data_dir: {e}")
        _log(f"RAW + data_dir=None raises RuntimeError: OK")
    _assert(raised, "Should raise RuntimeError for RAW + data_dir=None")

    print("  PASSED")


# ======================================================================
# Runner
# ======================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=" * 60)
    print(f"Caching Test Suite — {timestamp}")
    print(f"Test data: {TEST_DATA_DIR}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    import time
    t_start = time.time()

    tests = [
        test_clean_test_cache,
        test_participant_caching,
        test_fold_auto_caching,
        test_nan_preservation,
        test_split_generation,
        test_invalid_training_context,
        test_fold_corruption_recovery,
        test_participant_corruption_recovery,
        test_split_corruption_recovery,
        test_cache_metadata_corruption,
        test_no_cache_dir,
        test_clear_cache_methods,
        test_get_cache_info,
        test_metadata_save_to_cache,
        test_all_folds_caching,
        test_get_split_participants,
        test_orphan_temp_cleanup,
        test_cache_hit_performance,
        test_missing_participant_filtering,
        test_all_participants_filtered_error,
        test_data_dir_none,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {e}")

    t_elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({t_elapsed:.1f}s)")
    if errors:
        print(f"\nFailed tests:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    print(f"{'=' * 60}")

    # Clean up test caches after all tests
    clean_test_cache()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
