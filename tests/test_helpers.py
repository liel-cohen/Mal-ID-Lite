"""Shared test utilities for Mal-ID-Lite integration tests.

Provides a lightweight data loader backed by the mock test data in
tests/test_data/ (~118K sequences across 72 participants, 3 folds,
4 diseases, 6 participants per disease per fold). This replaces the
full-dataset loaders that used to be hardcoded in each test script.

Usage::

    from test_helpers import create_test_loader, TEST_DATA_DIR, TEST_FOLD_IDS

    loader = create_test_loader()
    sequences, metadata = loader.get_fold_data(0, "train")
"""

import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path so malid_lite imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from malid_lite.dataloader import MalIDPublishedDataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
"""Root of the mock test data directory (contains metadata.tsv, raw/, and
will contain auto-generated participants/ cache after first use)."""

TEST_RAW_DIR = TEST_DATA_DIR / "raw"
"""Directory with the raw participant .tsv.gz files."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_FOLD_IDS = [0, 1, 2]
"""The three external cross-validation folds present in the test data."""

TEST_DISEASES = ["HIV", "Covid19", "T1D", "Healthy/Background"]
"""Diseases present in the test data (6 participants per disease per fold)."""


def clean_test_cache() -> None:
    """Remove all auto-generated cache from the test data directory.

    Deletes ``participants/``, ``data_folds/``, ``splits/``, and
    ``metadata_processed.tsv`` under ``tests/test_data/``.
    The raw files (``raw/``) and ``metadata.tsv`` are never touched.

    Call this at the start of caching-specific tests that need a clean
    slate. Model tests generally do NOT need to call this — they can
    rely on auto-caching via ``get_fold_data()``.
    """
    for subdir in ("participants", "data_folds", "splits"):
        path = TEST_DATA_DIR / subdir
        if path.exists():
            shutil.rmtree(path)
    # Remove processed metadata cache (auto-generated from metadata.tsv)
    proc_meta = TEST_DATA_DIR / "metadata_processed.tsv"
    if proc_meta.exists():
        proc_meta.unlink()


def create_test_loader(verbose: int = 0) -> MalIDPublishedDataLoader:
    """Create a MalIDPublishedDataLoader backed by mock test data.

    The loader points at tests/test_data/ for both raw files and cache.
    On first use the loader preprocesses raw .tsv.gz files into a
    participant cache (test_data/participants/) — this takes a few seconds.
    Subsequent loads reuse the cached participant data.

    Parameters
    ----------
    verbose : int
        Verbosity level for the loader (0=silent, 1=normal, 2=debug).

    Returns
    -------
    MalIDPublishedDataLoader
        A loader that only reads from the test data directory — it never
        touches the real dataset or its cache.
    """
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test data directory not found: {TEST_DATA_DIR}\n"
            "Generate it with: python scripts/data/create_test_data.py"
        )
    if not TEST_RAW_DIR.exists() or not any(TEST_RAW_DIR.glob("part_table_*.tsv.gz")):
        raise FileNotFoundError(
            f"No raw participant files in {TEST_RAW_DIR}\n"
            "Generate them with: python scripts/data/create_test_data.py"
        )

    loader = MalIDPublishedDataLoader(
        data_dir=TEST_RAW_DIR,
        metadata_path=None,  # picks up TEST_DATA_DIR/metadata.tsv via cache_dir
        gene_reference_path=None,  # FR/CDR extraction not needed for tests
        gene_locus="TCR",
        cache_dir=TEST_DATA_DIR,
        verbose=verbose,
    )
    return loader
