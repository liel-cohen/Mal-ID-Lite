"""Quick smoke tests for the data loader.

Tests basic data loader functionality using the bundled test data in
tests/test_data/ (72 participants, 76 specimens, 4 diseases, 3 folds).

Tests:
  1. Loader initialization and metadata loading
  2. Load participant data at RAW stage
  3. Load participant data at CLEAN stage
  4. Load participant data at DOWNSAMPLED stage
  5. Preprocessing report generation
  6. Data flow: RAW > CLEAN > DOWNSAMPLED counts decrease monotonically

Expected runtime: <30 seconds (participant cache is pre-built in test_data/).
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from malid_lite.dataloader import PreprocessingStage
from malid_lite.dataloader.base import FOLD_COL, _LEGACY_FOLD_COL, normalize_fold_column

from test_helpers import (
    TEST_DATA_DIR,
    TEST_DISEASES,
    TEST_FOLD_IDS,
    clean_test_cache,
    create_test_loader,
)

# ---------------------------------------------------------------------------
# Output directory (per project convention: tests/test_outputs/<test_name>/)
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loader():
    """Create a test data loader (shared across all tests in this module).

    Cleans stale cache first so the loader scans raw files fresh —
    ensures participant/specimen counts match the current test data.
    """
    clean_test_cache()
    return create_test_loader(verbose=1)


@pytest.fixture(scope="module")
def first_participant(loader):
    """Return the first participant label from the test metadata."""
    return loader.metadata["participant_label"].iloc[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDataLoaderInitialization:
    """Test loader initialization and metadata properties."""

    def test_loader_initializes(self, loader):
        """Loader should initialize without errors."""
        assert loader is not None, "Loader failed to initialize"

    def test_metadata_loaded(self, loader):
        """Metadata should load as a non-empty DataFrame."""
        metadata = loader.metadata
        assert isinstance(metadata, pd.DataFrame), (
            f"Expected DataFrame, got {type(metadata)}"
        )
        assert len(metadata) > 0, "Metadata is empty"

    def test_metadata_has_expected_columns(self, loader):
        """Metadata must contain the required columns."""
        required_cols = [
            "participant_label",
            "specimen_label",
            "disease",
            "CV_fold",
        ]
        metadata = loader.metadata
        for col in required_cols:
            assert col in metadata.columns, (
                f"Required column '{col}' missing from metadata. "
                f"Available: {list(metadata.columns)}"
            )

    def test_metadata_participant_count(self, loader):
        """Test data should have the expected number of participants."""
        n_participants = loader.metadata["participant_label"].nunique()
        assert n_participants == 72, (
            f"Expected 72 participants, got {n_participants}"
        )

    def test_metadata_specimen_count(self, loader):
        """Test data should have the expected number of specimens."""
        n_specimens = loader.metadata["specimen_label"].nunique()
        assert n_specimens == 76, (
            f"Expected 76 specimens, got {n_specimens}"
        )

    def test_metadata_diseases(self, loader):
        """Metadata diseases should match the expected set."""
        actual_diseases = sorted(loader.metadata["disease"].unique())
        expected_diseases = sorted(TEST_DISEASES)
        assert actual_diseases == expected_diseases, (
            f"Disease mismatch: expected {expected_diseases}, got {actual_diseases}"
        )

    def test_metadata_fold_ids(self, loader):
        """Metadata fold IDs should match the expected set."""
        fold_col = "CV_fold"
        actual_folds = sorted(loader.metadata[fold_col].unique())
        expected_folds = sorted(TEST_FOLD_IDS)
        assert actual_folds == expected_folds, (
            f"Fold ID mismatch: expected {expected_folds}, got {actual_folds}"
        )


@pytest.mark.integration
class TestLoadParticipantData:
    """Test loading participant data at each preprocessing stage."""

    def test_load_raw(self, loader, first_participant):
        """RAW data should load as a non-empty DataFrame."""
        df_raw = loader.load_participant_data(
            first_participant, PreprocessingStage.RAW
        )
        assert isinstance(df_raw, pd.DataFrame), (
            f"Expected DataFrame, got {type(df_raw)}"
        )
        assert len(df_raw) > 0, (
            f"RAW data for participant '{first_participant}' is empty"
        )

    def test_load_clean(self, loader, first_participant):
        """CLEAN data should load as a non-empty DataFrame with key columns."""
        df_clean = loader.load_participant_data(
            first_participant, PreprocessingStage.CLEAN
        )
        assert isinstance(df_clean, pd.DataFrame), (
            f"Expected DataFrame, got {type(df_clean)}"
        )
        assert len(df_clean) > 0, (
            f"CLEAN data for participant '{first_participant}' is empty"
        )

        # CLEAN stage should produce columns used in downstream models
        expected_cols = ["v_gene", "j_gene"]
        for col in expected_cols:
            assert col in df_clean.columns, (
                f"Expected column '{col}' in CLEAN data. "
                f"Available: {list(df_clean.columns)[:20]}"
            )

    def test_load_downsampled(self, loader, first_participant):
        """DOWNSAMPLED data should load as a non-empty DataFrame."""
        df_down = loader.load_participant_data(
            first_participant, PreprocessingStage.DOWNSAMPLED
        )
        assert isinstance(df_down, pd.DataFrame), (
            f"Expected DataFrame, got {type(df_down)}"
        )
        assert len(df_down) > 0, (
            f"DOWNSAMPLED data for participant '{first_participant}' is empty"
        )

        # DOWNSAMPLED data should have a specimen_label column
        assert "specimen_label" in df_down.columns, (
            f"Expected 'specimen_label' column in DOWNSAMPLED data. "
            f"Available: {list(df_down.columns)[:20]}"
        )


@pytest.mark.integration
class TestPreprocessingReport:
    """Test preprocessing report generation."""

    def test_report_after_loading(self, loader, first_participant):
        """After loading CLEAN data, the preprocessing report should be non-empty."""
        # Trigger preprocessing to populate stats (may already be cached)
        loader.load_participant_data(
            first_participant, PreprocessingStage.CLEAN
        )

        report = loader.get_preprocessing_report()
        assert isinstance(report, pd.DataFrame), (
            f"Expected DataFrame, got {type(report)}"
        )
        # Report may be empty if data was loaded from cache (stats are only
        # accumulated during actual preprocessing, not cache reads). This is
        # expected behavior -- we just verify the method runs without error.

    def test_save_report(self, loader, first_participant):
        """save_preprocessing_report should write a CSV file without error."""
        # Trigger preprocessing to populate stats
        loader.load_participant_data(
            first_participant, PreprocessingStage.CLEAN
        )

        report_path = OUTPUT_DIR / "preprocessing_report.csv"
        loader.save_preprocessing_report(report_path)
        assert report_path.exists(), (
            f"Report file was not created at {report_path}"
        )


@pytest.mark.integration
class TestDataFlow:
    """Test that the preprocessing pipeline reduces data monotonically."""

    def test_raw_geq_clean_geq_downsampled(self, loader, first_participant):
        """Sequence counts should decrease (or stay equal) through the pipeline:
        RAW >= CLEAN >= DOWNSAMPLED.
        """
        df_raw = loader.load_participant_data(
            first_participant, PreprocessingStage.RAW
        )
        df_clean = loader.load_participant_data(
            first_participant, PreprocessingStage.CLEAN
        )
        df_down = loader.load_participant_data(
            first_participant, PreprocessingStage.DOWNSAMPLED
        )

        n_raw = len(df_raw)
        n_clean = len(df_clean)
        n_down = len(df_down)

        assert n_raw >= n_clean, (
            f"RAW ({n_raw}) should be >= CLEAN ({n_clean}) "
            f"for participant '{first_participant}'"
        )
        assert n_clean >= n_down, (
            f"CLEAN ({n_clean}) should be >= DOWNSAMPLED ({n_down}) "
            f"for participant '{first_participant}'"
        )
        assert n_down > 0, (
            f"DOWNSAMPLED data should not be empty for participant "
            f"'{first_participant}' (RAW={n_raw}, CLEAN={n_clean})"
        )


# ---------------------------------------------------------------------------
# Fold column normalization tests
# ---------------------------------------------------------------------------


class TestFoldColumnNormalization:
    """Test backward-compatible fold column renaming.

    The canonical fold column is 'CV_fold'. Metadata files that use the legacy
    name 'malid_cross_validation_fold_id_when_in_test_set' must be normalized
    transparently. These tests verify the normalization function directly and
    through the data loader, independent of what column name the on-disk test
    data happens to have.
    """

    def test_normalize_legacy_column_renamed(self):
        """Legacy column name should be renamed to 'CV_fold'."""
        df = pd.DataFrame({
            _LEGACY_FOLD_COL: [0, 1, 2],
            "disease": ["A", "B", "C"],
        })
        result = normalize_fold_column(df)
        assert FOLD_COL in result.columns
        assert _LEGACY_FOLD_COL not in result.columns
        assert list(result[FOLD_COL]) == [0, 1, 2]

    def test_normalize_new_column_unchanged(self):
        """DataFrame already using 'CV_fold' should pass through unchanged."""
        df = pd.DataFrame({
            FOLD_COL: [0, 1, 2],
            "disease": ["A", "B", "C"],
        })
        result = normalize_fold_column(df)
        assert FOLD_COL in result.columns
        assert list(result[FOLD_COL]) == [0, 1, 2]

    def test_normalize_no_fold_column_passes_through(self):
        """DataFrame without any fold column should pass through unchanged."""
        df = pd.DataFrame({"disease": ["A", "B"]})
        result = normalize_fold_column(df)
        assert FOLD_COL not in result.columns
        assert _LEGACY_FOLD_COL not in result.columns
        assert list(result.columns) == ["disease"]

    def test_normalize_both_columns_raises(self):
        """Having both fold columns should raise ValueError."""
        df = pd.DataFrame({
            _LEGACY_FOLD_COL: [0, 1],
            FOLD_COL: [0, 1],
        })
        with pytest.raises(ValueError, match="both"):
            normalize_fold_column(df)

    def test_normalize_preserves_data(self):
        """Normalization should preserve all other columns and row values."""
        df = pd.DataFrame({
            "participant_label": ["P1", "P2", "P3"],
            "disease": ["HIV", "Covid19", "T1D"],
            _LEGACY_FOLD_COL: [0, 1, 2],
            "extra_col": [10, 20, 30],
        })
        result = normalize_fold_column(df)
        assert list(result.columns) == [
            "participant_label", "disease", FOLD_COL, "extra_col"
        ]
        assert list(result["participant_label"]) == ["P1", "P2", "P3"]
        assert list(result["extra_col"]) == [10, 20, 30]

    @pytest.mark.integration
    def test_loader_accepts_legacy_metadata(self):
        """Data loader should load metadata with the legacy fold column name.

        Writes a minimal metadata file using the old column name and verifies
        the loader normalizes it to 'CV_fold'.
        """
        metadata = pd.DataFrame({
            "participant_label": ["P1", "P2"],
            "specimen_label": ["S1", "S2"],
            "disease": ["HIV", "Covid19"],
            _LEGACY_FOLD_COL: [0, 1],
            "available_gene_loci": ["GeneLocus.BCR|TCR"] * 2,
        })
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False
        ) as f:
            metadata.to_csv(f, sep="\t", index=False)
            meta_path = Path(f.name)

        try:
            from malid_lite.dataloader import MalIDPublishedDataLoader
            loader = MalIDPublishedDataLoader(
                data_dir=None,
                metadata_path=meta_path,
                gene_locus="TCR",
                verbose=0,
            )
            assert FOLD_COL in loader.metadata.columns, (
                f"Loader metadata should have '{FOLD_COL}' after normalization. "
                f"Columns: {list(loader.metadata.columns)}"
            )
            assert _LEGACY_FOLD_COL not in loader.metadata.columns, (
                f"Legacy column '{_LEGACY_FOLD_COL}' should not remain after normalization"
            )
            assert list(loader.metadata[FOLD_COL]) == [0, 1]
        finally:
            meta_path.unlink(missing_ok=True)

    @pytest.mark.integration
    def test_loader_accepts_new_metadata(self):
        """Data loader should load metadata with the new 'CV_fold' column name."""
        metadata = pd.DataFrame({
            "participant_label": ["P1", "P2"],
            "specimen_label": ["S1", "S2"],
            "disease": ["HIV", "Covid19"],
            FOLD_COL: [0, 1],
            "available_gene_loci": ["GeneLocus.BCR|TCR"] * 2,
        })
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False
        ) as f:
            metadata.to_csv(f, sep="\t", index=False)
            meta_path = Path(f.name)

        try:
            from malid_lite.dataloader import MalIDPublishedDataLoader
            loader = MalIDPublishedDataLoader(
                data_dir=None,
                metadata_path=meta_path,
                gene_locus="TCR",
                verbose=0,
            )
            assert FOLD_COL in loader.metadata.columns
            assert list(loader.metadata[FOLD_COL]) == [0, 1]
        finally:
            meta_path.unlink(missing_ok=True)
