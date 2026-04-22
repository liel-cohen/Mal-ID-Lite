"""Test split persistence in the data loader.

Tests:
1. Split generation for cv_single_model and cv_ensemble contexts
2. Split file persistence (save/load)
3. Correctness: no overlap between splits, all participants assigned
4. Reproducibility: same splits on repeated calls
5. Consistency with existing split_train_smaller() function

Expected runtime: <10 seconds
"""

import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader

# Output directory per CLAUDE.md conventions
output_dir = Path(__file__).parent / "test_outputs" / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def create_loader(cache_dir: Path) -> MalIDPublishedDataLoader:
    """Create a data loader with the standard paths."""
    return MalIDPublishedDataLoader(
        data_dir=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/"
            "data_clean/airr_format_clean/TCR/"
        ),
        metadata_path=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/"
            "data/metadata.tsv"
        ),
        gene_reference_path=Path(
            "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/"
            "data/tcrb_v_gene_cdrs.generated.tsv"
        ),
        gene_locus="TCR",
        cache_dir=cache_dir,
        verbose=1,
    )


def test_cv_single_model_splits():
    """Test cv_single_model split generation and properties."""
    print("\n=== Test: cv_single_model splits ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        for fold_id in [0, 1, 2]:
            splits = loader.load_splits(fold_id, "cv_single_model")

            # Check columns
            assert list(splits.columns) == ["participant_label", "disease", "split_role"], \
                f"Unexpected columns: {list(splits.columns)}"

            # Check roles
            roles = set(splits["split_role"].unique())
            assert roles == {"test", "train_smaller1", "train_smaller2"}, \
                f"Fold {fold_id}: unexpected roles {roles}"

            # Check no duplicates
            assert splits["participant_label"].is_unique, \
                f"Fold {fold_id}: duplicate participants"

            # Check no NaN
            assert not splits.isna().any().any(), \
                f"Fold {fold_id}: NaN values in splits"

            # Check approximate proportions
            n_total = len(splits)
            n_test = (splits["split_role"] == "test").sum()
            n_ts1 = (splits["split_role"] == "train_smaller1").sum()
            n_ts2 = (splits["split_role"] == "train_smaller2").sum()

            # test ~= 1/3 of total
            assert 0.2 < n_test / n_total < 0.45, \
                f"Fold {fold_id}: test proportion {n_test/n_total:.2f} out of range"
            # ts1 ~= 2/3 of train, ts2 ~= 1/3 of train
            n_train = n_ts1 + n_ts2
            assert 0.55 < n_ts1 / n_train < 0.78, \
                f"Fold {fold_id}: ts1/train proportion {n_ts1/n_train:.2f} out of range"

            print(f"  Fold {fold_id}: test={n_test}, ts1={n_ts1}, ts2={n_ts2}, total={n_total}")

        # Check split file was saved
        split_file = cache_dir / "splits" / "fold_0_cv_single_model.csv"
        assert split_file.exists(), "Split file not saved"

        # Check metadata was saved
        metadata_file = cache_dir / "splits" / "split_metadata.json"
        assert metadata_file.exists(), "Split metadata not saved"

    print("  PASSED")


def test_cv_ensemble_splits():
    """Test cv_ensemble split generation and properties."""
    print("\n=== Test: cv_ensemble splits ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        for fold_id in [0, 1, 2]:
            splits = loader.load_splits(fold_id, "cv_ensemble")

            # Check roles
            roles = set(splits["split_role"].unique())
            assert roles == {"test", "validation", "train_smaller1", "train_smaller2"}, \
                f"Fold {fold_id}: unexpected roles {roles}"

            # Check no duplicates
            assert splits["participant_label"].is_unique, \
                f"Fold {fold_id}: duplicate participants"

            # Check approximate proportions (see ENSEMBLE_ARCHITECTURE.md section 2)
            n_total = len(splits)
            n_test = (splits["split_role"] == "test").sum()
            n_val = (splits["split_role"] == "validation").sum()
            n_ts1 = (splits["split_role"] == "train_smaller1").sum()
            n_ts2 = (splits["split_role"] == "train_smaller2").sum()

            # validation ~= 1/3 of train ~= 6/27 of total ~= 0.22
            n_train = n_val + n_ts1 + n_ts2
            assert 0.2 < n_val / n_train < 0.45, \
                f"Fold {fold_id}: validation/train proportion {n_val/n_train:.2f} out of range"

            # ts1 ~= 2/3 of train_smaller, ts2 ~= 1/3 of train_smaller
            n_train_smaller = n_ts1 + n_ts2
            assert 0.55 < n_ts1 / n_train_smaller < 0.78, \
                f"Fold {fold_id}: ts1/train_smaller proportion {n_ts1/n_train_smaller:.2f} out of range"

            print(
                f"  Fold {fold_id}: test={n_test}, val={n_val}, "
                f"ts1={n_ts1}, ts2={n_ts2}, total={n_total}"
            )

    print("  PASSED")


def test_reproducibility():
    """Test that loading splits twice gives identical results."""
    print("\n=== Test: reproducibility ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        # Generate
        splits1 = loader.load_splits(0, "cv_ensemble")
        # Load from file
        splits2 = loader.load_splits(0, "cv_ensemble")

        assert splits1.equals(splits2), "Splits differ on reload!"

        # Also test with a fresh loader instance (same cache_dir)
        loader2 = create_loader(cache_dir)
        splits3 = loader2.load_splits(0, "cv_ensemble")
        assert splits1.equals(splits3), "Splits differ with fresh loader!"

    print("  PASSED")


def test_no_overlap_between_splits():
    """Test that no participant appears in multiple split roles."""
    print("\n=== Test: no overlap between splits ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        for context in ["cv_single_model", "cv_ensemble"]:
            for fold_id in [0, 1, 2]:
                splits = loader.load_splits(fold_id, context)

                # Group by role
                by_role = splits.groupby("split_role")["participant_label"].apply(set)

                # Check pairwise disjointness
                roles = list(by_role.index)
                for i, r1 in enumerate(roles):
                    for r2 in roles[i + 1:]:
                        overlap = by_role[r1] & by_role[r2]
                        assert not overlap, (
                            f"Fold {fold_id} ({context}): "
                            f"{r1} and {r2} share {len(overlap)} participants"
                        )

    print("  PASSED")


def test_disease_stratification():
    """Test that disease distribution is roughly preserved across splits."""
    print("\n=== Test: disease stratification ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        splits = loader.load_splits(0, "cv_ensemble")

        # Get overall disease distribution
        overall = splits["disease"].value_counts(normalize=True).sort_index()

        # Check each split role has all diseases represented
        for role in splits["split_role"].unique():
            role_diseases = splits[splits["split_role"] == role]["disease"].unique()
            missing = set(overall.index) - set(role_diseases)
            # Some small roles might miss rare diseases, but test + validation should have all
            if role in ("test", "validation"):
                assert not missing, (
                    f"Role '{role}' is missing diseases: {missing}"
                )
            if missing:
                print(f"  WARNING: Role '{role}' missing diseases: {missing}")

        print(f"  Disease classes: {sorted(overall.index.tolist())}")
        print(f"  Overall distribution: {overall.to_dict()}")

    print("  PASSED")


def test_consistency_with_existing_split_function():
    """Test that cv_single_model ts1/ts2 match the existing split_train_smaller().

    The existing function in training_utils.py operates on sequence DataFrames,
    but uses the same train_test_split(test_size=1/3, random_state=0, stratify=disease)
    logic. The participant assignments should be identical.
    """
    print("\n=== Test: consistency with split_train_smaller() ===")

    from malid_lite.training.training_utils import split_train_smaller

    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache" / "mal-id-orig-data"
    loader = create_loader(cache_dir)

    # Only test fold 0 to keep it quick
    fold_id = 0

    # --- New split persistence approach ---
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_cache = Path(tmpdir) / "cache"
        temp_loader = create_loader(temp_cache)
        new_splits = temp_loader.load_splits(fold_id, "cv_single_model")

    new_ts1 = set(
        new_splits[new_splits["split_role"] == "train_smaller1"]["participant_label"]
    )
    new_ts2 = set(
        new_splits[new_splits["split_role"] == "train_smaller2"]["participant_label"]
    )

    # --- Existing approach (operates on sequence data) ---
    # Load fold data from existing cache
    from malid_lite.dataloader import PreprocessingStage
    sequences_df, metadata_df = loader.get_fold_data(
        fold_id, "train", PreprocessingStage.DOWNSAMPLED
    )

    ts1_seqs, ts2_seqs = split_train_smaller(sequences_df, metadata_df)
    old_ts1 = set(ts1_seqs["participant_label"].unique())
    old_ts2 = set(ts2_seqs["participant_label"].unique())

    # Compare
    assert new_ts1 == old_ts1, (
        f"ts1 mismatch: {len(new_ts1 - old_ts1)} only in new, "
        f"{len(old_ts1 - new_ts1)} only in old"
    )
    assert new_ts2 == old_ts2, (
        f"ts2 mismatch: {len(new_ts2 - old_ts2)} only in new, "
        f"{len(old_ts2 - new_ts2)} only in old"
    )

    print(f"  ts1: {len(new_ts1)} participants (match)")
    print(f"  ts2: {len(new_ts2)} participants (match)")
    print("  PASSED")


def test_get_split_participants():
    """Test the convenience method get_split_participants()."""
    print("\n=== Test: get_split_participants() ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        # Model 1 training set = ts1 + ts2
        model1_train = loader.get_split_participants(
            0, "cv_ensemble", ["train_smaller1", "train_smaller2"]
        )
        # Validation set
        val = loader.get_split_participants(0, "cv_ensemble", ["validation"])
        # Test set
        test = loader.get_split_participants(0, "cv_ensemble", ["test"])

        # Check they partition all participants
        all_participants = set(model1_train) | set(val) | set(test)
        meta = loader.metadata
        expected = set(
            meta.drop_duplicates(subset=["participant_label"])["participant_label"]
        )
        assert all_participants == expected, (
            f"Participants mismatch: "
            f"{len(all_participants - expected)} extra, "
            f"{len(expected - all_participants)} missing"
        )

        print(f"  Model 1 train: {len(model1_train)}")
        print(f"  Validation: {len(val)}")
        print(f"  Test: {len(test)}")
        print(f"  Total: {len(all_participants)}")

    print("  PASSED")


def test_cv_ensemble_is_subset_of_cv_single_model():
    """Test that cv_ensemble training participants are a strict subset of cv_single_model's.

    cv_single_model uses all train participants for ts1+ts2.
    cv_ensemble holds out a validation split, so ts1+ts2 should be strictly smaller.
    The validation participants must not appear in ts1 or ts2.
    """
    print("\n=== Test: cv_ensemble training is subset of cv_single_model ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        for fold_id in [0, 1, 2]:
            # cv_single_model: ts1+ts2 = all train participants
            sm_train = set(loader.get_split_participants(
                fold_id, "cv_single_model", ["train_smaller1", "train_smaller2"]
            ))

            # cv_ensemble: ts1+ts2 = train minus validation
            ens_train = set(loader.get_split_participants(
                fold_id, "cv_ensemble", ["train_smaller1", "train_smaller2"]
            ))
            ens_val = set(loader.get_split_participants(
                fold_id, "cv_ensemble", ["validation"]
            ))

            # cv_ensemble training must be a strict subset of cv_single_model training
            assert ens_train < sm_train, (
                f"Fold {fold_id}: cv_ensemble train ({len(ens_train)}) "
                f"is not a strict subset of cv_single_model train ({len(sm_train)})"
            )

            # Validation participants must not overlap with training
            assert not (ens_train & ens_val), (
                f"Fold {fold_id}: {len(ens_train & ens_val)} participants in both "
                f"cv_ensemble train and validation"
            )

            # Validation + training should equal the full train set
            assert ens_train | ens_val == sm_train, (
                f"Fold {fold_id}: cv_ensemble train+val does not equal "
                f"cv_single_model train"
            )

            print(
                f"  Fold {fold_id}: sm_train={len(sm_train)}, "
                f"ens_train={len(ens_train)}, ens_val={len(ens_val)}"
            )

    print("  PASSED")


def test_invalid_context():
    """Test that invalid training_context raises ValueError."""
    print("\n=== Test: invalid context ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        loader = create_loader(cache_dir)

        try:
            loader.load_splits(0, "invalid_context")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "training_context" in str(e)
            print(f"  Correctly raised: {e}")

    print("  PASSED")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Split persistence tests - {timestamp}")
    print("=" * 60)

    test_cv_single_model_splits()
    test_cv_ensemble_splits()
    test_reproducibility()
    test_no_overlap_between_splits()
    test_disease_stratification()
    test_consistency_with_existing_split_function()
    test_get_split_participants()
    test_cv_ensemble_is_subset_of_cv_single_model()
    test_invalid_context()

    print("\n" + "=" * 60)
    print("All tests PASSED")
