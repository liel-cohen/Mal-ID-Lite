"""Test split persistence in the data loader.

Uses the bundled test data (tests/test_data/) with 72 participants,
76 specimens, 4 diseases, 3 folds. Each test creates a fresh temporary
cache directory so split files are generated from scratch (not loaded
from a pre-existing cache).

Tests:
1. cv_single_model split generation and properties
2. cv_ensemble split generation and properties
3. Reproducibility (reload, fresh loader)
4. No overlap between split roles
5. Disease stratification across split roles
6. get_split_participants() convenience method
7. cv_ensemble training is a strict subset of cv_single_model training
8. Invalid training_context raises ValueError

Output: tests/test_outputs/test_splits/

Expected runtime: <10 seconds
"""

import sys
import shutil
from pathlib import Path

import pytest

# Ensure project root and tests/ are on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR, TEST_FOLD_IDS, TEST_DISEASES
from malid_lite.dataloader import MalIDPublishedDataLoader

# Output directory per CLAUDE.md conventions
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_test_output_dir(test_name: str) -> Path:
    """Create a clean test output subdirectory, removing stale artifacts from prior runs."""
    test_dir = OUTPUT_DIR / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def _create_fresh_loader(cache_dir: Path) -> MalIDPublishedDataLoader:
    """Create a loader with a fresh (empty) cache_dir for split generation.

    Points data_dir at the bundled test raw files and metadata_path at
    the bundled metadata.tsv. The cache_dir is expected to be an empty
    temporary directory so that split files are generated from scratch.

    Parameters
    ----------
    cache_dir : Path
        An empty directory where splits and metadata cache will be written.

    Returns
    -------
    MalIDPublishedDataLoader
    """
    return MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=TEST_DATA_DIR / "metadata.tsv",
        gene_reference_path=None,  # FR/CDR extraction not needed for split tests
        gene_locus="TCR",
        cache_dir=cache_dir,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCvSingleModelSplits:
    """Test cv_single_model split generation and properties."""

    def test_cv_single_model_splits(self):
        """Split roles, proportions, uniqueness, and file persistence."""
        test_dir = _get_test_output_dir("test_cv_single_model_splits")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        for fold_id in TEST_FOLD_IDS:
            splits = loader.load_splits(fold_id, "cv_single_model")

            # --- Column check ---
            assert list(splits.columns) == ["participant_label", "disease", "split_role"], \
                f"Unexpected columns: {list(splits.columns)}"

            # --- Role check ---
            roles = set(splits["split_role"].unique())
            assert roles == {"test", "train_smaller1", "train_smaller2"}, \
                f"Fold {fold_id}: unexpected roles {roles}"

            # --- No duplicates ---
            assert splits["participant_label"].is_unique, \
                f"Fold {fold_id}: duplicate participants"

            # --- No NaN ---
            assert not splits.isna().any().any(), \
                f"Fold {fold_id}: NaN values in splits"

            # --- Proportion checks ---
            n_total = len(splits)
            n_test = (splits["split_role"] == "test").sum()
            n_ts1 = (splits["split_role"] == "train_smaller1").sum()
            n_ts2 = (splits["split_role"] == "train_smaller2").sum()

            # test ~= 1/3 of total (range allows for rounding with small datasets)
            assert 0.2 < n_test / n_total < 0.45, \
                f"Fold {fold_id}: test proportion {n_test / n_total:.2f} out of range"
            # ts1 ~= 2/3 of train, ts2 ~= 1/3 of train
            n_train = n_ts1 + n_ts2
            assert 0.55 < n_ts1 / n_train < 0.78, \
                f"Fold {fold_id}: ts1/train proportion {n_ts1 / n_train:.2f} out of range"

        # --- Split file saved to disk ---
        split_file = cache_dir / "splits" / "fold_0_cv_single_model.csv"
        assert split_file.exists(), "Split file not saved"

        # --- Split metadata JSON saved ---
        metadata_file = cache_dir / "splits" / "split_metadata.json"
        assert metadata_file.exists(), "Split metadata not saved"


@pytest.mark.integration
class TestCvEnsembleSplits:
    """Test cv_ensemble split generation and properties."""

    def test_cv_ensemble_splits(self):
        """Split roles, proportions, uniqueness for cv_ensemble context."""
        test_dir = _get_test_output_dir("test_cv_ensemble_splits")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        for fold_id in TEST_FOLD_IDS:
            splits = loader.load_splits(fold_id, "cv_ensemble")

            # --- Role check: ensemble has an extra "validation" role ---
            roles = set(splits["split_role"].unique())
            assert roles == {"test", "validation", "train_smaller1", "train_smaller2"}, \
                f"Fold {fold_id}: unexpected roles {roles}"

            # --- No duplicates ---
            assert splits["participant_label"].is_unique, \
                f"Fold {fold_id}: duplicate participants"

            # --- Proportion checks ---
            n_total = len(splits)
            n_test = (splits["split_role"] == "test").sum()
            n_val = (splits["split_role"] == "validation").sum()
            n_ts1 = (splits["split_role"] == "train_smaller1").sum()
            n_ts2 = (splits["split_role"] == "train_smaller2").sum()

            # validation ~= 1/3 of train
            n_train = n_val + n_ts1 + n_ts2
            assert 0.2 < n_val / n_train < 0.45, \
                f"Fold {fold_id}: validation/train proportion {n_val / n_train:.2f} out of range"

            # ts1 ~= 2/3 of train_smaller, ts2 ~= 1/3 of train_smaller
            n_train_smaller = n_ts1 + n_ts2
            assert 0.55 < n_ts1 / n_train_smaller < 0.78, \
                f"Fold {fold_id}: ts1/train_smaller proportion {n_ts1 / n_train_smaller:.2f} out of range"


@pytest.mark.integration
class TestReproducibility:
    """Test that splits are deterministic: reload and fresh-loader give same results."""

    def test_reproducibility(self):
        test_dir = _get_test_output_dir("test_reproducibility")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        # First call generates and saves
        splits1 = loader.load_splits(0, "cv_ensemble")
        # Second call loads from file
        splits2 = loader.load_splits(0, "cv_ensemble")
        assert splits1.equals(splits2), "Splits differ on reload!"

        # Fresh loader instance, same cache_dir -> should load identical splits
        loader2 = _create_fresh_loader(cache_dir)
        splits3 = loader2.load_splits(0, "cv_ensemble")
        assert splits1.equals(splits3), "Splits differ with fresh loader!"


@pytest.mark.integration
class TestNoOverlap:
    """Test that no participant appears in multiple split roles."""

    def test_no_overlap_between_splits(self):
        test_dir = _get_test_output_dir("test_no_overlap_between_splits")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        for context in ["cv_single_model", "cv_ensemble"]:
            for fold_id in TEST_FOLD_IDS:
                splits = loader.load_splits(fold_id, context)

                # Group by role -> set of participant labels
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


@pytest.mark.integration
class TestDiseaseStratification:
    """Test that disease distribution is roughly preserved across splits."""

    def test_disease_stratification(self):
        test_dir = _get_test_output_dir("test_disease_stratification")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        splits = loader.load_splits(0, "cv_ensemble")

        # Overall disease distribution
        overall = splits["disease"].value_counts(normalize=True).sort_index()

        # Check each split role has all diseases represented.
        # With 72 balanced participants (6 per disease per fold), every
        # role should have all 4 diseases even after splitting.
        for role in splits["split_role"].unique():
            role_diseases = splits[splits["split_role"] == role]["disease"].unique()
            missing = set(overall.index) - set(role_diseases)
            # test and validation must have all diseases; ts1/ts2 might miss
            # a rare disease in very small datasets, but with our balanced
            # test data they should all be present
            assert not missing, (
                f"Role '{role}' is missing diseases: {missing}"
            )


@pytest.mark.integration
class TestGetSplitParticipants:
    """Test the convenience method get_split_participants()."""

    def test_get_split_participants(self):
        test_dir = _get_test_output_dir("test_get_split_participants")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        # Model 1 training set = ts1 + ts2
        model1_train = loader.get_split_participants(
            0, "cv_ensemble", ["train_smaller1", "train_smaller2"]
        )
        # Validation set
        val = loader.get_split_participants(0, "cv_ensemble", ["validation"])
        # Test set
        test = loader.get_split_participants(0, "cv_ensemble", ["test"])

        # These three sets should partition all participants
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

        # No overlap between the three sets
        assert not (set(model1_train) & set(val)), "model1_train and validation overlap"
        assert not (set(model1_train) & set(test)), "model1_train and test overlap"
        assert not (set(val) & set(test)), "validation and test overlap"


@pytest.mark.integration
class TestCvEnsembleSubsetProperty:
    """Test that cv_ensemble training participants are a strict subset of cv_single_model's.

    cv_single_model uses all train participants for ts1+ts2.
    cv_ensemble holds out a validation split, so ts1+ts2 should be strictly smaller.
    """

    def test_cv_ensemble_is_subset_of_cv_single_model(self):
        test_dir = _get_test_output_dir("test_cv_ensemble_is_subset_of_cv_single_model")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        for fold_id in TEST_FOLD_IDS:
            # cv_single_model: ts1 + ts2 = all train participants
            sm_train = set(loader.get_split_participants(
                fold_id, "cv_single_model", ["train_smaller1", "train_smaller2"]
            ))

            # cv_ensemble: ts1 + ts2 = train minus validation
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

            # Validation participants must not overlap with ensemble training
            assert not (ens_train & ens_val), (
                f"Fold {fold_id}: {len(ens_train & ens_val)} participants in both "
                f"cv_ensemble train and validation"
            )

            # Validation + training should equal the full cv_single_model train set
            assert ens_train | ens_val == sm_train, (
                f"Fold {fold_id}: cv_ensemble train+val does not equal "
                f"cv_single_model train"
            )


@pytest.mark.integration
class TestInvalidContext:
    """Test that invalid training_context raises ValueError."""

    def test_invalid_context(self):
        test_dir = _get_test_output_dir("test_invalid_context")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        with pytest.raises(ValueError, match="training_context"):
            loader.load_splits(0, "invalid_context")


# ---------------------------------------------------------------------------
# Train-all split tests (Phase 1)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTrainAllSplits:
    """Test train_all split generation (no test fold, no fold id)."""

    def test_train_all_splits(self):
        test_dir = _get_test_output_dir("test_train_all_splits")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        # fold_id must be None for train-all contexts
        splits = loader.load_splits(None, "train_all")

        # --- Roles: only ts1/ts2, no test, no validation ---
        roles = set(splits["split_role"].unique())
        assert roles == {"train_smaller1", "train_smaller2"}, \
            f"Unexpected train_all roles: {roles}"

        # --- Covers ALL participants (disease-agnostic, fold-agnostic) ---
        meta = loader.metadata
        expected = set(meta.drop_duplicates(subset=["participant_label"])["participant_label"])
        assert set(splits["participant_label"]) == expected, \
            "train_all splits do not cover all participants"
        assert splits["participant_label"].is_unique
        assert not splits.isna().any().any()

        # --- Proportions: ts1 ~= 2/3, ts2 ~= 1/3 ---
        n_ts1 = (splits["split_role"] == "train_smaller1").sum()
        n_ts2 = (splits["split_role"] == "train_smaller2").sum()
        assert 0.55 < n_ts1 / (n_ts1 + n_ts2) < 0.78

        # --- Split CSV named without a fold prefix ---
        assert (cache_dir / "splits" / "train_all.csv").exists(), \
            "train_all.csv split file not saved"
        # --- Per-context human-readable summary written ---
        assert (cache_dir / "splits" / "train_all_summary.txt").exists(), \
            "train_all_summary.txt not written"


@pytest.mark.integration
class TestTrainAllEnsembleSplits:
    """Test train_all_ensemble split generation (validation + ts1/ts2, no test)."""

    def test_train_all_ensemble_splits(self):
        test_dir = _get_test_output_dir("test_train_all_ensemble_splits")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        splits = loader.load_splits(None, "train_all_ensemble")

        roles = set(splits["split_role"].unique())
        assert roles == {"validation", "train_smaller1", "train_smaller2"}, \
            f"Unexpected train_all_ensemble roles: {roles}"

        # Covers ALL participants
        meta = loader.metadata
        expected = set(meta.drop_duplicates(subset=["participant_label"])["participant_label"])
        assert set(splits["participant_label"]) == expected

        # validation ~= 1/3 of all; ts1+ts2 ~= 2/3
        n_val = (splits["split_role"] == "validation").sum()
        n_ts1 = (splits["split_role"] == "train_smaller1").sum()
        n_ts2 = (splits["split_role"] == "train_smaller2").sum()
        n_total = n_val + n_ts1 + n_ts2
        assert 0.2 < n_val / n_total < 0.45
        assert 0.55 < n_ts1 / (n_ts1 + n_ts2) < 0.78

        assert (cache_dir / "splits" / "train_all_ensemble.csv").exists()
        assert (cache_dir / "splits" / "train_all_ensemble_summary.txt").exists()


@pytest.mark.integration
class TestTrainAllReproducibility:
    """Train-all splits are deterministic across reload and fresh loader."""

    def test_train_all_reproducibility(self):
        test_dir = _get_test_output_dir("test_train_all_reproducibility")
        cache_dir = test_dir / "cache"
        loader = _create_fresh_loader(cache_dir)

        splits1 = loader.load_splits(None, "train_all_ensemble")   # generate
        splits2 = loader.load_splits(None, "train_all_ensemble")   # reload
        assert splits1.equals(splits2)

        loader2 = _create_fresh_loader(cache_dir)
        splits3 = loader2.load_splits(None, "train_all_ensemble")
        assert splits1.equals(splits3)


@pytest.mark.integration
class TestTrainAllContextFoldIdValidation:
    """fold_id must be None for train-all and an int for CV — else clear error."""

    def test_train_all_rejects_fold_id(self):
        test_dir = _get_test_output_dir("test_train_all_rejects_fold_id")
        loader = _create_fresh_loader(test_dir / "cache")
        with pytest.raises(ValueError, match="no fold concept"):
            loader.load_splits(0, "train_all")

    def test_cv_requires_fold_id(self):
        test_dir = _get_test_output_dir("test_cv_requires_fold_id")
        loader = _create_fresh_loader(test_dir / "cache")
        with pytest.raises(ValueError, match="requires a fold_id"):
            loader.load_splits(None, "cv_single_model")


@pytest.mark.integration
class TestTrainAllVsCvNoLeakage:
    """train_all uses ALL participants; train_all_ensemble holds out validation."""

    def test_ensemble_train_is_strict_subset_of_train_all(self):
        test_dir = _get_test_output_dir("test_ensemble_train_is_strict_subset_of_train_all")
        loader = _create_fresh_loader(test_dir / "cache")

        # train_all ts1+ts2 = ALL participants
        all_train = set(loader.get_split_participants(
            None, "train_all", ["train_smaller1", "train_smaller2"]
        ))
        meta = loader.metadata
        expected = set(meta.drop_duplicates(subset=["participant_label"])["participant_label"])
        assert all_train == expected, "train_all ts1+ts2 should be every participant"

        # train_all_ensemble ts1+ts2 = 2/3 (validation held out) → strict subset
        ens_train = set(loader.get_split_participants(
            None, "train_all_ensemble", ["train_smaller1", "train_smaller2"]
        ))
        ens_val = set(loader.get_split_participants(
            None, "train_all_ensemble", ["validation"]
        ))
        assert ens_train < all_train, \
            "train_all_ensemble train must be a strict subset of train_all train"
        assert not (ens_train & ens_val), "ensemble train and validation overlap (leakage!)"
        assert ens_train | ens_val == all_train, \
            "ensemble train + validation should equal all participants"


@pytest.mark.integration
class TestStratificationGuard:
    """The upfront per-disease count guard fails fast with a clear message."""

    def _make_pool(self, disease_counts):
        import pandas as pd
        rows = []
        for disease, n in disease_counts.items():
            for i in range(n):
                rows.append({"participant_label": f"{disease}_{i}", "disease": disease})
        return pd.DataFrame(rows)

    def test_train_all_needs_two_per_disease(self):
        loader = _create_fresh_loader(_get_test_output_dir("strat_train_all") / "cache")
        pool = self._make_pool({"A": 5, "B": 1})  # B too small for one split
        with pytest.raises(ValueError, match="at least 2 participant"):
            loader._validate_stratification_counts(pool, is_ensemble=False, context_label="train_all")
        # >= 2 each is fine
        ok = self._make_pool({"A": 5, "B": 2})
        loader._validate_stratification_counts(ok, is_ensemble=False, context_label="train_all")

    def test_ensemble_needs_three_per_disease(self):
        loader = _create_fresh_loader(_get_test_output_dir("strat_ensemble") / "cache")
        pool = self._make_pool({"A": 5, "B": 2})  # B ok for single split, too small for nested
        with pytest.raises(ValueError, match="at least 3 participant"):
            loader._validate_stratification_counts(pool, is_ensemble=True, context_label="train_all_ensemble")
        ok = self._make_pool({"A": 5, "B": 3})
        loader._validate_stratification_counts(ok, is_ensemble=True, context_label="train_all_ensemble")
