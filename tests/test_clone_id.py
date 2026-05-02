"""Tests for clone_id auto-computation.

Tests both the standalone clone assignment functions (unit tests) and
their integration with the data loader pipeline.

Tests:
  Unit tests (assign_repertoire_clones.py):
    1. resolve_identity_threshold: defaults and overrides
    2. hamming_identity_fraction: basic distance computation
    3. hierarchical_linkage_clonotypes: clustering correctness, linkage methods
    4. assign_clones: single-chain wrapper
    5. compute_participant_clone_id: CDR3 NT validation (uppercase, ACGT-only),
       missing column errors, summary logging
    6. Determinism: same input -> same output
    7. remap_cluster_ids_by_size: Clone_1 = largest

  Validation tests (data loader constructor):
    8. Invalid linkage method -> ValueError

  Integration tests (data loader pipeline):
    9.  Default path: existing clone_id used as-is (no computation)
    10. force_clone_id + use_aa: rename original, compute new from cdr3_aa
    11. Cache param validation: mismatch -> error
    12. force on existing cache -> error
    13. No-cache mode works when clone_id exists
    14. precompute_clone_ids: caching + idempotency
    15. Downsampled stage flow-through

  Synthetic data tests (no clone_id in input):
    16. Auto-compute from cdr3 NT and AA
    17. No-cache + missing clone_id -> error
    18. Parallel vs sequential equivalence

Expected runtime: <30 seconds
"""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite.utils.assign_repertoire_clones import (
    CLONE_ID_COL,
    CLONE_ID_ORIGINAL_COL,
    DEFAULT_IDENTITY_THRESHOLDS,
    assign_clones,
    compute_participant_clone_id,
    hamming_identity_fraction,
    hierarchical_linkage_clonotypes,
    remap_cluster_ids_by_size,
    resolve_identity_threshold,
)

from test_helpers import TEST_DATA_DIR, TEST_RAW_DIR, clean_test_cache

TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.tsv"

# ---------------------------------------------------------------------------
# Output directory (per project convention)
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    """Small DataFrame for unit testing clone assignment."""
    return pd.DataFrame({
        "v_gene": ["TRBV1", "TRBV1", "TRBV1", "TRBV1", "TRBV2", "TRBV2"],
        "j_gene": ["TRBJ1", "TRBJ1", "TRBJ1", "TRBJ1", "TRBJ2", "TRBJ2"],
        "cdr3": [
            "ACGTACGT",  # group 1: identical to seq 1
            "ACGTACGT",  # group 1: identical to seq 0
            "ACGTACGA",  # group 1: 1/8 distance from 0,1
            "TTTTTTTT",  # group 1: very different
            "AAAA",      # group 2: short, different v_gene
            "AAAC",      # group 2: 1/4 distance from seq 4
        ],
        "cdr3_aa": ["CASS", "CASS", "CASX", "CAXX", "DASS", "DASX"],
    })


@pytest.fixture
def dirty_cdr3_df():
    """DataFrame with CDR3 NT issues: dashes, NaN, non-ACGT."""
    return pd.DataFrame({
        "v_gene": ["TRBV1"] * 7,
        "j_gene": ["TRBJ1"] * 7,
        "cdr3": [
            "ACGT-ACGT",  # dash (should be stripped)
            "acgtacgt",   # lowercase (should be uppercased)
            "ACGTACGT",   # clean
            "ACGN",       # non-ACGT (N) — should be dropped
            np.nan,       # NaN — should be dropped
            "",           # empty string — should be dropped
            "ACGT",       # clean, different length
        ],
        "cdr3_aa": ["CA", "CA", "CA", "CA", "CA", "CA", "CA"],
    })


@pytest.fixture
def tmp_cache_dir():
    """Create a temporary directory for cache tests, cleaned up after."""
    tmpdir = tempfile.mkdtemp(prefix="test_clone_id_cache_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Unit tests: resolve_identity_threshold
# ===========================================================================


class TestResolveIdentityThreshold:
    def test_defaults(self):
        assert resolve_identity_threshold("TCR", False) == 0.95
        assert resolve_identity_threshold("BCR", False) == 0.90
        assert resolve_identity_threshold("TCR", True) == 0.90
        assert resolve_identity_threshold("BCR", True) == 0.85

    def test_override(self):
        assert resolve_identity_threshold("TCR", False, override=0.88) == 0.88
        assert resolve_identity_threshold("BCR", True, override=0.99) == 0.99

    def test_invalid_override(self):
        with pytest.raises(ValueError, match="must be in"):
            resolve_identity_threshold("TCR", False, override=0.0)
        with pytest.raises(ValueError, match="must be in"):
            resolve_identity_threshold("TCR", False, override=1.5)

    def test_invalid_locus(self):
        with pytest.raises(ValueError, match="No default"):
            resolve_identity_threshold("XYZ", False)


# ===========================================================================
# Unit tests: hamming_identity_fraction
# ===========================================================================


class TestHammingIdentityFraction:
    def test_identical(self):
        assert hamming_identity_fraction("ACGT", "ACGT") == 1.0

    def test_one_mismatch(self):
        assert hamming_identity_fraction("ACGT", "ACGA") == 0.75

    def test_all_different(self):
        assert hamming_identity_fraction("AAAA", "CCCC") == 0.0

    def test_unequal_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            hamming_identity_fraction("ACGT", "AC")


# ===========================================================================
# Unit tests: hierarchical_linkage_clonotypes
# ===========================================================================


class TestHierarchicalLinkageClonotypes:
    def test_basic_clustering(self, small_df):
        result = hierarchical_linkage_clonotypes(
            small_df,
            v_gene_col="v_gene",
            j_gene_col="j_gene",
            cdr3_col="cdr3",
            identity_threshold=0.95,
        )
        assert CLONE_ID_COL in result.columns
        assert len(result) == len(small_df)
        # Identical sequences (rows 0, 1) should be in the same clone
        assert result.iloc[0][CLONE_ID_COL] == result.iloc[1][CLONE_ID_COL]
        # Very different sequence (row 3) should be in a different clone
        assert result.iloc[0][CLONE_ID_COL] != result.iloc[3][CLONE_ID_COL]

    def test_threshold_effect(self, small_df):
        """Lower threshold should merge more sequences into clones."""
        # At 0.95: seqs 0,1 same clone; seq 2 different (1/8=0.125 dist)
        strict = hierarchical_linkage_clonotypes(
            small_df, identity_threshold=0.95, cdr3_col="cdr3",
        )
        n_clones_strict = strict[CLONE_ID_COL].nunique()

        # At 0.80: seqs 0,1,2 should merge (0.125 < 0.20 threshold)
        lenient = hierarchical_linkage_clonotypes(
            small_df, identity_threshold=0.80, cdr3_col="cdr3",
        )
        n_clones_lenient = lenient[CLONE_ID_COL].nunique()

        # Strictly fewer clones with lenient threshold (merges more)
        assert n_clones_lenient < n_clones_strict

    def test_linkage_methods(self, small_df):
        """All supported linkage methods should produce valid output."""
        for method in ("single", "complete", "average"):
            result = hierarchical_linkage_clonotypes(
                small_df,
                cdr3_col="cdr3",
                identity_threshold=0.90,
                linkage_method=method,
            )
            assert CLONE_ID_COL in result.columns
            assert len(result) == len(small_df)
            assert result[CLONE_ID_COL].notna().all()

    def test_preserves_row_order(self, small_df):
        """Output should have the same index as input."""
        result = hierarchical_linkage_clonotypes(
            small_df, cdr3_col="cdr3", identity_threshold=0.90,
        )
        pd.testing.assert_index_equal(result.index, small_df.index)

    def test_missing_values_get_unknown(self):
        df = pd.DataFrame({
            "v_gene": ["TRBV1", np.nan, "TRBV1"],
            "j_gene": ["TRBJ1", "TRBJ1", "TRBJ1"],
            "cdr3": ["ACGT", "ACGT", ""],
        })
        result = hierarchical_linkage_clonotypes(
            df, cdr3_col="cdr3", identity_threshold=0.90,
        )
        # Row 1 (NaN v_gene) and row 2 (empty cdr3) should be Unknown
        assert result.iloc[1][CLONE_ID_COL] == "Unknown"
        assert result.iloc[2][CLONE_ID_COL] == "Unknown"
        # Row 0 should have a valid clone ID
        assert result.iloc[0][CLONE_ID_COL] != "Unknown"

    def test_clone_ids_ordered_by_size(self, small_df):
        """Clone_1 should be the largest clone."""
        result = hierarchical_linkage_clonotypes(
            small_df, cdr3_col="cdr3", identity_threshold=0.95,
        )
        clone_sizes = result[CLONE_ID_COL].value_counts()
        # Clone_1 should have the most members
        assert clone_sizes.index[0] == "Clone_1"

    def test_single_sequence_groups(self):
        """Each unique (V, J, cdr3_len) with 1 sequence gets its own clone."""
        df = pd.DataFrame({
            "v_gene": ["TRBV1", "TRBV2", "TRBV3"],
            "j_gene": ["TRBJ1", "TRBJ2", "TRBJ3"],
            "cdr3": ["ACGT", "AAAA", "CCCC"],
        })
        result = hierarchical_linkage_clonotypes(
            df, cdr3_col="cdr3", identity_threshold=0.90,
        )
        assert result[CLONE_ID_COL].nunique() == 3


# ===========================================================================
# Unit tests: assign_clones
# ===========================================================================


class TestAssignClones:
    def test_single_chain(self, small_df):
        result = assign_clones(
            small_df,
            heavy_v_gene_col="v_gene",
            heavy_j_gene_col="j_gene",
            heavy_cdr3_col="cdr3",
            identity_threshold=0.95,
            heavy_clone_output_col_name=CLONE_ID_COL,
        )
        assert CLONE_ID_COL in result.columns
        assert len(result) == len(small_df)

    def test_no_chain_raises(self, small_df):
        with pytest.raises(ValueError, match="At least one chain"):
            assign_clones(small_df)


# ===========================================================================
# Unit tests: compute_participant_clone_id
# ===========================================================================


class TestComputeParticipantCloneId:
    def test_basic_computation(self, small_df):
        result_df, stats = compute_participant_clone_id(
            small_df, "test_p1", "cdr3",
            identity_threshold=0.95,
        )
        assert CLONE_ID_COL in result_df.columns
        assert stats["n_sequences"] == len(small_df)
        assert stats["n_clones"] > 0
        assert stats["n_unknown"] == 0

    def test_cdr3_nt_validation(self, dirty_cdr3_df):
        """CDR3 NT validation: uppercase, strip dashes, drop invalid."""
        result_df, stats = compute_participant_clone_id(
            dirty_cdr3_df, "test_dirty", "cdr3",
            identity_threshold=0.90,
            use_aa=False,
        )
        val = stats["cdr3_nt_validation"]

        # 1 dash stripped (row 0)
        assert val["dashes_stripped"] == 1
        # 3 rows dropped: NaN (row 4), empty (row 5), non-ACGT N (row 3)
        assert val["rows_dropped_nan"] == 2  # NaN + empty→NaN
        assert val["rows_dropped_non_acgt"] == 1
        assert val["rows_dropped_total"] == 3
        # 4 rows remaining: dash-stripped (0), uppercased (1), clean (2), short (6)
        assert len(result_df) == 4
        assert CLONE_ID_COL in result_df.columns

        # Verify all surviving CDR3 NT values are uppercase ACGT only
        valid_chars = set("ACGT")
        for cdr3_val in result_df["cdr3"]:
            assert cdr3_val == cdr3_val.upper(), f"CDR3 not uppercased: {cdr3_val}"
            assert set(cdr3_val).issubset(valid_chars), f"Non-ACGT in CDR3: {cdr3_val}"

    def test_no_validation_when_use_aa(self, dirty_cdr3_df):
        """When use_aa=True, CDR3 NT validation should NOT run."""
        result_df, stats = compute_participant_clone_id(
            dirty_cdr3_df, "test_aa", "cdr3_aa",
            identity_threshold=0.90,
            use_aa=True,
        )
        assert "cdr3_nt_validation" not in stats
        # All rows should survive (no NT validation)
        assert len(result_df) == len(dirty_cdr3_df)

    def test_force_rename(self, small_df):
        """force=True should rename existing clone_id to clone_id_original."""
        df_with_clone = small_df.copy()
        df_with_clone[CLONE_ID_COL] = ["C1", "C1", "C2", "C3", "C4", "C4"]

        result_df, stats = compute_participant_clone_id(
            df_with_clone, "test_force", "cdr3",
            identity_threshold=0.95,
            force=True,
        )
        assert CLONE_ID_ORIGINAL_COL in result_df.columns
        assert CLONE_ID_COL in result_df.columns
        assert stats["clone_id_original_preserved"] is True
        # Original values preserved
        assert result_df[CLONE_ID_ORIGINAL_COL].tolist() == [
            "C1", "C1", "C2", "C3", "C4", "C4"
        ]

    def test_missing_columns_raise(self, small_df):
        df_no_v = small_df.drop(columns=["v_gene"])
        with pytest.raises(ValueError, match="v_gene"):
            compute_participant_clone_id(
                df_no_v, "test", "cdr3", identity_threshold=0.95,
            )

    def test_missing_cdr3_col_raises(self, small_df):
        """Requesting a CDR3 column that doesn't exist should raise."""
        with pytest.raises(ValueError, match="nonexistent_col"):
            compute_participant_clone_id(
                small_df, "test", "nonexistent_col", identity_threshold=0.95,
            )

    def test_determinism(self, small_df):
        """Same input should always produce the same clone assignments."""
        results = []
        for _ in range(3):
            df_copy = small_df.copy()
            result_df, _ = compute_participant_clone_id(
                df_copy, "det_test", "cdr3", identity_threshold=0.95,
            )
            results.append(result_df[CLONE_ID_COL].tolist())

        assert results[0] == results[1] == results[2]

    def test_determinism_shuffled(self, small_df):
        """Shuffled input should produce equivalent clone assignments."""
        result1, _ = compute_participant_clone_id(
            small_df.copy(), "det1", "cdr3", identity_threshold=0.95,
        )

        # Shuffle the DataFrame
        shuffled = small_df.sample(frac=1, random_state=42).reset_index(drop=True)
        result2, _ = compute_participant_clone_id(
            shuffled, "det2", "cdr3", identity_threshold=0.95,
        )

        # Clone sizes should match (even if IDs differ due to row order)
        sizes1 = sorted(result1[CLONE_ID_COL].value_counts().values)
        sizes2 = sorted(result2[CLONE_ID_COL].value_counts().values)
        assert sizes1 == sizes2


# ===========================================================================
# Unit tests: remap_cluster_ids_by_size
# ===========================================================================


class TestRemapClusterIdsBySize:
    def test_basic_remap(self):
        df = pd.DataFrame({"cluster": [1, 1, 1, 2, 2, 3]})
        result = remap_cluster_ids_by_size(df, "cluster")
        # Cluster 1 has 3 members -> Clone_1, cluster 2 has 2 -> Clone_2
        assert (result.loc[result["cluster"] == "Clone_1"]).shape[0] == 3
        assert (result.loc[result["cluster"] == "Clone_2"]).shape[0] == 2
        assert (result.loc[result["cluster"] == "Clone_3"]).shape[0] == 1

    def test_all_unknown(self):
        df = pd.DataFrame({"cluster": ["Unknown", "Unknown"]})
        result = remap_cluster_ids_by_size(df, "cluster")
        assert (result["cluster"] == "Unknown").all()


# ===========================================================================
# Integration tests: data loader pipeline
# ===========================================================================


class TestDataLoaderCloneIdValidation:
    """Tests for clone_id parameter validation in the data loader constructor."""

    def test_invalid_linkage_method_raises(self):
        """Invalid linkage method should raise ValueError at construction."""
        with pytest.raises(ValueError, match="clone_id_linkage_method must be"):
            MalIDPublishedDataLoader(
                data_dir=None,
                metadata_path=TEST_METADATA_PATH,
                clone_id_linkage_method="invalid_method",
            )

    def test_invalid_use_aa_type_raises(self):
        """Non-bool clone_id_use_aa should raise ValueError at construction."""
        with pytest.raises(ValueError, match="clone_id_use_aa must be a bool"):
            MalIDPublishedDataLoader(
                data_dir=None,
                metadata_path=TEST_METADATA_PATH,
                clone_id_use_aa="yes",
            )

        with pytest.raises(ValueError, match="clone_id_use_aa must be a bool"):
            MalIDPublishedDataLoader(
                data_dir=None,
                metadata_path=TEST_METADATA_PATH,
                clone_id_use_aa=1,
            )


class TestDataLoaderCloneIdIntegration:
    """Integration tests using the bundled test data in tests/test_data/."""

    @pytest.fixture(autouse=True)
    def _check_test_data(self):
        if not TEST_DATA_DIR.exists() or not TEST_RAW_DIR.exists():
            pytest.skip("Test data not available (run create_test_data.py)")

    def test_existing_clone_id_used_as_is(self, tmp_cache_dir):
        """When data has clone_id and force=False, no computation happens."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=False,
        )
        participants = loader.metadata["participant_label"].unique()
        p = participants[0]

        df = loader.load_participant_data(p, PreprocessingStage.CLEAN)
        assert not df.empty
        assert CLONE_ID_COL in df.columns
        assert "igh_or_tcrb_clone_id" in df.columns

        # Verify stats show clone_id was NOT computed
        _, stats_file = loader.get_participant_cache_path(p)
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        assert stats["clone_id_computed"] is False

    def test_force_clone_id_with_aa(self, tmp_cache_dir):
        """force_clone_id=True with use_aa recomputes from cdr3_aa."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        participants = loader.metadata["participant_label"].unique()
        p = participants[0]

        df = loader.load_participant_data(p, PreprocessingStage.CLEAN)
        assert not df.empty
        assert CLONE_ID_COL in df.columns
        assert CLONE_ID_ORIGINAL_COL in df.columns

        # Verify stats show clone_id WAS computed
        _, stats_file = loader.get_participant_cache_path(p)
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        assert stats["clone_id_computed"] is True
        assert stats["clone_id_params"]["clone_id_use_aa"] is True
        # force_clone_id is a build-time action flag, not a clustering param
        # — it should NOT be stored in clone_id_params
        assert "force_clone_id" not in stats["clone_id_params"]

    def test_cache_param_mismatch_raises_at_construction(self, tmp_cache_dir):
        """Explicitly conflicting clone_id params should raise at construction.

        The error surfaces at loader construction time (fail-fast), not when
        data is first loaded.
        """
        # Build cache with specific params
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
            clone_id_identity_threshold=0.90,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Create new loader with DIFFERENT threshold — should raise at
        # construction time (upfront validation), not at load time
        with pytest.raises(ValueError, match="Clone ID parameters conflict"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_use_aa=True,
                clone_id_identity_threshold=0.85,
            )

    def test_old_cache_without_clone_id_computed_raises(self, tmp_cache_dir):
        """Cache missing clone_id_computed key should raise when params specified.

        Old caches built before clone_id tracking lack the 'clone_id_computed'
        key in their stats JSON. If the user explicitly specifies clone_id
        params (or force_clone_id), the loader should raise rather than
        silently skipping validation.
        """
        # Build cache normally
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Tamper with the stats JSON to remove clone_id_computed key
        # (simulates an old cache format)
        import json
        participants_dir = tmp_cache_dir / "participants"
        stats_files = sorted(participants_dir.glob("*_stats.json"))
        assert len(stats_files) > 0

        for sf in stats_files:
            with open(sf) as f:
                stats = json.load(f)
            stats.pop("clone_id_computed", None)
            stats.pop("clone_id_params", None)
            with open(sf, "w") as f:
                json.dump(stats, f)

        # Upfront check: specifying clone_id params should raise at construction
        with pytest.raises(ValueError, match="clone_id tracking"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_use_aa=True,
            )

        # Upfront check: force_clone_id should also raise at construction
        with pytest.raises(ValueError, match="clone_id tracking"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                force_clone_id=True,
            )

        # No clone_id params specified — should succeed (no validation needed)
        loader2 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        df = loader2.load_participant_data(p, PreprocessingStage.CLEAN)
        assert not df.empty

    def test_unspecified_params_not_validated(self, tmp_cache_dir):
        """Omitting clone_id params on subsequent runs should not raise.

        The natural workflow: set clone_id params once at cache build time,
        then omit them on all subsequent training/embedding commands.
        Unspecified (None) params are accepted as-is — only explicitly-
        provided params that conflict with the cache trigger an error.
        """
        # Build cache with non-default params
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        df1 = loader1.load_participant_data(p, PreprocessingStage.CLEAN)
        assert CLONE_ID_COL in df1.columns

        # Load with NO clone_id params (all None) — should succeed
        loader2 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        df2 = loader2.load_participant_data(p, PreprocessingStage.CLEAN)

        assert len(df2) == len(df1)
        assert CLONE_ID_COL in df2.columns

    def test_matching_explicit_params_accepted(self, tmp_cache_dir):
        """Explicitly-specified params that match the cache should pass."""
        # Build cache with use_aa=True
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        df1 = loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Explicitly specify the SAME use_aa — should succeed
        loader2 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            clone_id_use_aa=True,
        )
        df2 = loader2.load_participant_data(p, PreprocessingStage.CLEAN)

        assert len(df2) == len(df1)
        assert CLONE_ID_COL in df2.columns

    def test_force_on_existing_cache_raises_at_construction(self, tmp_cache_dir):
        """force_clone_id=True on cache built without computation raises at construction."""
        # Build cache WITHOUT force (data has clone_id, used as-is)
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=False,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Now try with force_clone_id=True — should raise at construction
        with pytest.raises(ValueError, match="force_clone_id=True but"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                force_clone_id=True,
            )

    def test_clone_params_on_non_computed_cache_raises(self, tmp_cache_dir):
        """Specifying clone_id params when cache used pre-existing clone_id raises.

        If the cache was built without computing clone_id (the data already had
        one), passing clustering params like --clone-id-use-aa is an error —
        they have no effect on the cached data and likely indicate the user
        intends to recompute clone_id.
        """
        # Build cache WITHOUT force (data has clone_id, used as-is)
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Specifying use_aa should raise at construction
        with pytest.raises(ValueError, match="clustering parameters"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_use_aa=True,
            )

        # Specifying threshold should also raise
        with pytest.raises(ValueError, match="clustering parameters"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_identity_threshold=0.85,
            )

        # Specifying linkage should also raise
        with pytest.raises(ValueError, match="clustering parameters"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_linkage_method="complete",
            )

    def test_force_flag_not_required_on_subsequent_loads(self, tmp_cache_dir):
        """Cache built with --force-clone-id should load without the flag.

        force_clone_id is a build-time action flag, not a clustering param.
        The user's natural workflow is: build cache with --force-clone-id,
        then run training without it (no clone args at all).
        """
        # Build cache WITH force + use_aa
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        df1 = loader1.load_participant_data(p, PreprocessingStage.CLEAN)
        assert CLONE_ID_COL in df1.columns
        assert CLONE_ID_ORIGINAL_COL in df1.columns

        # Load with NO clone args at all — should succeed
        loader2 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        df2 = loader2.load_participant_data(p, PreprocessingStage.CLEAN)

        assert len(df2) == len(df1)
        assert CLONE_ID_COL in df2.columns
        assert CLONE_ID_ORIGINAL_COL in df2.columns
        pd.testing.assert_series_equal(
            df1[CLONE_ID_COL].reset_index(drop=True),
            df2[CLONE_ID_COL].reset_index(drop=True),
            check_dtype=False,
        )

    def test_no_cache_works_when_clone_id_exists(self):
        """No-cache mode works when data already has clone_id."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=None,  # no caching
            verbose=0,
        )
        p = loader.metadata["participant_label"].unique()[0]
        df = loader.load_participant_data(p, PreprocessingStage.CLEAN)
        assert not df.empty
        assert CLONE_ID_COL in df.columns

    def test_use_aa_mismatch_raises(self, tmp_cache_dir):
        """Explicitly specifying use_aa=False when cache was built with True raises."""
        # Build cache with use_aa=True
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        p = loader1.metadata["participant_label"].unique()[0]
        loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Explicitly specify use_aa=False — should raise at construction
        with pytest.raises(ValueError, match="Clone ID parameters conflict"):
            MalIDPublishedDataLoader(
                data_dir=TEST_RAW_DIR,
                metadata_path=TEST_METADATA_PATH,
                gene_locus="TCR",
                cache_dir=tmp_cache_dir,
                verbose=0,
                clone_id_use_aa=False,
            )

    def test_per_participant_validation_catches_tampered_stats(self, tmp_cache_dir):
        """Per-participant validation catches mismatches that upfront missed.

        Simulates a partially-rebuilt cache where the first participant (checked
        by upfront validation) has matching params, but a later participant has
        different params. The per-participant check in load_cached_participant
        should catch it at load time.
        """
        # Build cache with use_aa=True for all participants
        loader1 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        participants = sorted(loader1.metadata["participant_label"].unique())
        assert len(participants) >= 2, "Need at least 2 participants for this test"
        for p in participants:
            loader1.load_participant_data(p, PreprocessingStage.CLEAN)

        # Tamper with the SECOND participant's stats to simulate a different
        # use_aa value (as if it was cached with different params)
        import json
        _, stats_file = loader1.get_participant_cache_path(participants[1])
        with open(stats_file) as f:
            stats = json.load(f)
        stats["clone_id_params"]["clone_id_use_aa"] = False
        with open(stats_file, "w") as f:
            json.dump(stats, f)

        # Construction succeeds — upfront checks the first participant (sorted),
        # which still has matching params
        loader2 = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            clone_id_use_aa=True,
        )

        # Loading the first participant succeeds
        df_ok = loader2.load_participant_data(
            participants[0], PreprocessingStage.CLEAN
        )
        assert not df_ok.empty

        # Loading the tampered second participant raises
        with pytest.raises(ValueError, match="Clone ID parameters conflict"):
            loader2.load_participant_data(
                participants[1], PreprocessingStage.CLEAN
            )

    def test_precompute_clone_ids(self, tmp_cache_dir):
        """precompute_clone_ids should cache all participants."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        # Use sequential for test predictability
        loader.precompute_clone_ids(n_jobs=1)

        # All participants should now have cache files
        participants = loader.metadata["participant_label"].unique()
        for p in participants:
            cache_file, stats_file = loader.get_participant_cache_path(p)
            assert cache_file.exists(), f"Missing cache for {p}"
            assert stats_file.exists(), f"Missing stats for {p}"

    def test_precompute_idempotent(self, tmp_cache_dir):
        """Calling precompute_clone_ids twice should be a no-op the second time."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )
        loader.precompute_clone_ids(n_jobs=1)

        # Get modification times
        p = loader.metadata["participant_label"].unique()[0]
        cache_file, _ = loader.get_participant_cache_path(p)
        mtime1 = cache_file.stat().st_mtime

        # Second call should not reprocess
        loader.precompute_clone_ids(n_jobs=1)
        mtime2 = cache_file.stat().st_mtime

        assert mtime1 == mtime2, "Cache file was rewritten on second precompute"

    def test_downsampled_uses_computed_clone_id(self, tmp_cache_dir):
        """Computed clone_id should flow through to downsampled stage."""
        loader = MalIDPublishedDataLoader(
            data_dir=TEST_RAW_DIR,
            metadata_path=TEST_METADATA_PATH,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            force_clone_id=True,
            clone_id_use_aa=True,
        )
        # Find a participant that survives downsampling (has enough clones)
        participants = loader.metadata["participant_label"].unique()
        found_nonempty = False
        for p in participants:
            df = loader.load_participant_data(p, PreprocessingStage.DOWNSAMPLED)
            if not df.empty:
                found_nonempty = True
                # DOWNSAMPLED stage must have clone_id
                assert CLONE_ID_COL in df.columns, (
                    f"Missing {CLONE_ID_COL} in downsampled data for {p}. "
                    f"Columns: {list(df.columns)}"
                )
                break

        if not found_nonempty:
            pytest.skip(
                "No participants survived downsampling — cannot verify "
                "clone_id flows through to DOWNSAMPLED stage"
            )


# ===========================================================================
# Integration test: synthetic data without clone_id
# ===========================================================================


class TestSyntheticNoCloneId:
    """Test auto-computation with synthetic data that lacks clone_id."""

    def _create_synthetic_data(self, tmp_dir: Path, n_participants: int = 2):
        """Create minimal synthetic AIRR data without clone_id column."""
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for i in range(n_participants):
            p_label = f"SYNTH_{i:04d}"
            s_label = f"spec_{i:04d}"
            disease = "Healthy" if i % 2 == 0 else "Disease"
            fold = i % 3

            # Create synthetic sequences (no clone_id column)
            n_seqs = 200
            np.random.seed(i)
            v_genes = np.random.choice(
                ["TRBV1*01", "TRBV2*01", "TRBV3*01"], n_seqs
            )
            j_genes = np.random.choice(
                ["TRBJ1-1*01", "TRBJ2-1*01"], n_seqs
            )
            # Generate random CDR3 NT sequences (4-12 chars, ACGT only)
            cdr3_nt = []
            cdr3_aa = []
            for _ in range(n_seqs):
                length = np.random.choice([8, 10, 12])
                nt_seq = "".join(np.random.choice(list("ACGT"), length))
                cdr3_nt.append(nt_seq)
                aa_len = length // 3
                aa_seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), aa_len))
                cdr3_aa.append(aa_seq)

            df = pd.DataFrame({
                "repertoire_id": [s_label] * n_seqs,
                "productive": ["T"] * n_seqs,
                "v_score": [100.0] * n_seqs,
                "v_call": v_genes,
                "j_call": j_genes,
                "cdr3_aa": cdr3_aa,
                "cdr3": cdr3_nt,
                "sequence": [f"SEQ_{j}" for j in range(n_seqs)],
            })

            # Save as gzipped TSV (no clone_id column!)
            file_path = raw_dir / f"part_table_{p_label}.tsv.gz"
            df.to_csv(file_path, sep="\t", index=False, compression="gzip")

            metadata_rows.append({
                "participant_label": p_label,
                "specimen_label": s_label,
                "disease": disease,
                "CV_fold": fold,
            })

        # Save metadata
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(tmp_dir / "metadata.tsv", sep="\t", index=False)

        return raw_dir, tmp_dir / "metadata.tsv"

    def test_auto_compute_from_nt(self, tmp_cache_dir):
        """Data without clone_id should get it computed from cdr3 NT."""
        raw_dir, meta_path = self._create_synthetic_data(tmp_cache_dir)

        loader = MalIDPublishedDataLoader(
            data_dir=raw_dir,
            metadata_path=meta_path,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
        )

        p = loader.metadata["participant_label"].iloc[0]
        df = loader.load_participant_data(p, PreprocessingStage.CLEAN)

        assert not df.empty
        assert CLONE_ID_COL in df.columns
        assert "igh_or_tcrb_clone_id" in df.columns

        # All rows should have a clone_id (no NaN)
        assert df[CLONE_ID_COL].notna().all()

        # Verify stats
        import json
        _, stats_file = loader.get_participant_cache_path(p)
        with open(stats_file) as f:
            stats = json.load(f)
        assert stats["clone_id_computed"] is True
        assert stats["clone_id_params"]["clone_id_identity_threshold"] == 0.95

    def test_auto_compute_from_aa(self, tmp_cache_dir):
        """Data without clone_id + use_aa=True should compute from cdr3_aa."""
        raw_dir, meta_path = self._create_synthetic_data(tmp_cache_dir)

        loader = MalIDPublishedDataLoader(
            data_dir=raw_dir,
            metadata_path=meta_path,
            gene_locus="TCR",
            cache_dir=tmp_cache_dir,
            verbose=0,
            clone_id_use_aa=True,
        )

        p = loader.metadata["participant_label"].iloc[0]
        df = loader.load_participant_data(p, PreprocessingStage.CLEAN)

        assert not df.empty
        assert CLONE_ID_COL in df.columns

        # Verify AA threshold was used
        import json
        _, stats_file = loader.get_participant_cache_path(p)
        with open(stats_file) as f:
            stats = json.load(f)
        assert stats["clone_id_params"]["clone_id_use_aa"] is True
        assert stats["clone_id_params"]["clone_id_identity_threshold"] == 0.90

    def test_no_cache_missing_clone_id_raises(self, tmp_cache_dir):
        """No cache + missing clone_id should raise ValueError."""
        raw_dir, meta_path = self._create_synthetic_data(tmp_cache_dir)

        loader = MalIDPublishedDataLoader(
            data_dir=raw_dir,
            metadata_path=meta_path,
            gene_locus="TCR",
            cache_dir=None,  # no caching
            verbose=0,
        )

        p = loader.metadata["participant_label"].iloc[0]
        with pytest.raises(ValueError, match="Clone ID computation requires caching"):
            loader.load_participant_data(p, PreprocessingStage.CLEAN)

    def test_precompute_parallel(self, tmp_cache_dir):
        """Parallel precompute should produce same results as sequential."""
        raw_dir, meta_path = self._create_synthetic_data(
            tmp_cache_dir, n_participants=4,
        )

        # Sequential
        seq_cache = tmp_cache_dir / "seq_cache"
        seq_cache.mkdir()
        loader_seq = MalIDPublishedDataLoader(
            data_dir=raw_dir,
            metadata_path=meta_path,
            gene_locus="TCR",
            cache_dir=seq_cache,
            verbose=0,
        )
        loader_seq.precompute_clone_ids(n_jobs=1)

        # Parallel
        par_cache = tmp_cache_dir / "par_cache"
        par_cache.mkdir()
        loader_par = MalIDPublishedDataLoader(
            data_dir=raw_dir,
            metadata_path=meta_path,
            gene_locus="TCR",
            cache_dir=par_cache,
            verbose=0,
        )
        loader_par.precompute_clone_ids(n_jobs=2)

        # Compare results for each participant
        for p in loader_seq.metadata["participant_label"].unique():
            df_seq = loader_seq.load_participant_data(p, PreprocessingStage.CLEAN)
            df_par = loader_par.load_participant_data(p, PreprocessingStage.CLEAN)

            assert len(df_seq) == len(df_par), f"Row count mismatch for {p}"

            # Clone sizes should match (IDs are deterministic given same input order)
            sizes_seq = sorted(df_seq[CLONE_ID_COL].value_counts().values)
            sizes_par = sorted(df_par[CLONE_ID_COL].value_counts().values)
            assert sizes_seq == sizes_par, f"Clone size mismatch for {p}"
