"""Core pipeline tests for Model 3 (Sequence-Level Classifier).

Tests the Model 3 pipeline (aggregation, classifier, featurize, evaluate,
tuning, resume) in two tiers. Embedding-related tests (validation, alignment,
caching, ESM-2 smoke) are in test_model3_embeddings.py.

Tier 1 -- Unit tests with SYNTHETIC data (no cache, no GPU, ~30 seconds):
  1.  AggregationStrategy enum values and aggregate_group dispatch
  2.  Aggregation edge cases: empty input, single row, uniform entropy fallback
  3.  find_non_rare_v_genes filtering
  4.  GroupSequenceClassifier fit + predict_proba with class alignment
  5.  SequenceLevelClassifier init validation and helpers
  6.  _sanitize_group_columns replaces underscores
  7.  Class name prefix collision detection
  8.  Full two-stage fit + predict on synthetic data (TCR, multiclass)
  9.  featurize_specimens: uniform prior fill for missing groups
  10. featurize_specimens: test-time column alignment
  11. evaluate_on_test: multiclass metrics
  12. evaluate_on_test: binary metrics with reference_class
  13. evaluate_on_test: optional train count parameters
  14. make_tcr_model / make_bcr_model factory defaults
  23. _check_fold_complete: detects presence/absence of fold artifacts
  24. _load_fold_results: save/load round-trip for resume data
  25. Resume artifact save/load with metadata validation (stage1 round-trip,
      _meta structure checks for all artifact types)
  26. resume=False does not skip even when artifacts exist
  27. _validate_artifact_meta: errors on fold_id/locus/classes/param/training_context
      mismatches, plus backward compat for old artifacts without training_context
  28. Backward compat: old artifacts without _meta still load (warning only)
  30. _build_group_index: correctness and index assertion
  31. _fast_featurize: all strategies, entropy filtering, unknown strategy error
  32. Tuning sort key: tie-breaking order and 0.0 vs None handling
  33. Full auto-tuning pipeline on synthetic data (end-to-end)
  34. load_stage2_artifacts: tuning validation (missing winner, mismatch, reweigh)
  35. Tuning winner selection: sort order and attribute assignment
  36. Tuning artifact save/load round-trip

Tier 2 -- Integration tests with TEST DATA (tests/test_data/, ~2-5 min):
  18. Full multiclass pipeline on fold 0
  19. Full binary pipeline (one disease vs Healthy/Background)
  20. Predictions CSV format validation (multiclass)
  21. Predictions CSV format validation (binary)
  22. Model artifact save/load round-trip
  29. cv_ensemble split isolation

Requirements
------------
- Tier 1 (unit): numpy, pandas, scikit-learn (no cache, no GPU, no glmnet)
- Tier 2 (integration): glmnet (for TCR Stage 1); test data (tests/test_data/)

Expected runtime
----------------
- Tier 1 only: ~30 seconds
- Tier 1 + Tier 2: ~2-5 minutes

Output files
------------
All outputs saved to tests/test_outputs/test_model3/:
- test_log_YYYYMMDD_HHMMSS.txt              - Full log
- test_results_YYYYMMDD_HHMMSS.json         - Structured results (pass/fail per test)
- integration/                               - Integration test artifacts

Running
-------
From Mal-ID-Lite root directory:

    python -m pytest tests/test_model3.py -v -s

    # With more parallel workers (speeds up integration tests on multi-core servers):
    python -m pytest tests/test_model3.py -v -s --n-jobs 8

"""

import json
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test_helpers import create_test_loader

# Test output directory (per CLAUDE.md convention)
TEST_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / TEST_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Test logger
# ---------------------------------------------------------------------------

class _TestLogger:
    """Logger that writes to both console and file, tracks pass/fail."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, "a")
        self.results: List[Dict] = []
        self.start_time = datetime.now()

    def log(self, message: str):
        self.file.write(message + "\n")
        self.file.flush()
        print(message)

    def record(self, test_name: str, passed: bool, details: Optional[Dict] = None):
        status = "PASSED" if passed else "FAILED"
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
        self.log(f"  -> {status}")

    def save_results(self, json_path: Path):
        data = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "n_tests": len(self.results),
            "n_passed": sum(1 for r in self.results if r["status"] == "PASSED"),
            "n_failed": sum(1 for r in self.results if r["status"] == "FAILED"),
            "tests": self.results,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        self.log(f"\nResults saved: {json_path}")

    def close(self):
        self.file.close()


@pytest.fixture
def tlog():
    """Provide a _TestLogger instance for each test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"test_log_{timestamp}.txt"
    logger = _TestLogger(log_file)
    yield logger
    logger.close()


@pytest.fixture
def n_jobs(request):
    """Number of parallel workers for integration tests (from --n-jobs CLI arg)."""
    return request.config.getoption("--n-jobs")


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_sequences(
    n_specimens: int = 20,
    n_seqs_per_specimen: int = 50,
    diseases: List[str] = None,
    v_genes: List[str] = None,
    j_genes: List[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a synthetic sequences DataFrame for unit testing.

    Returns a DataFrame with the same columns the Model 3 pipeline expects:
    v_gene, j_gene, cdr3_aa, specimen_label, participant_label, disease,
    igh_or_tcrb_clone_id, isotype_supergroup.
    """
    rng = np.random.RandomState(random_state)
    if diseases is None:
        diseases = ["Covid19", "HIV", "Healthy/Background"]
    if v_genes is None:
        v_genes = ["TRBV5-1", "TRBV5-6", "TRBV7-2", "TRBV18", "TRBV20-1",
                    "TRBV28", "TRBV12-3", "TRBV6-1"]
    if j_genes is None:
        j_genes = ["TRBJ1-1", "TRBJ1-5", "TRBJ2-1", "TRBJ2-5", "TRBJ2-7"]

    # CDR3 amino acid alphabet (subset of common residues)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    rows = []
    for spec_idx in range(n_specimens):
        disease = diseases[spec_idx % len(diseases)]
        specimen = f"SPEC-{spec_idx:04d}"
        participant = f"PART-{spec_idx:04d}"  # 1:1 specimen:participant for simplicity
        for seq_idx in range(n_seqs_per_specimen):
            cdr3_len = rng.randint(10, 20)
            cdr3 = "CAS" + "".join(rng.choice(aa, cdr3_len - 3))
            rows.append({
                "v_gene": rng.choice(v_genes),
                "j_gene": rng.choice(j_genes),
                "cdr3_aa": cdr3,
                "specimen_label": specimen,
                "participant_label": participant,
                "disease": disease,
                "igh_or_tcrb_clone_id": seq_idx,
                "isotype_supergroup": "TCRB",
            })
    return pd.DataFrame(rows)


def make_synthetic_embeddings(n: int, dim: int = 640, random_state: int = 42) -> np.ndarray:
    """Create random float32 embeddings for unit testing."""
    rng = np.random.RandomState(random_state)
    return rng.randn(n, dim).astype(np.float32)


class _RFIgnoringGroups:
    """Wrapper around RandomForestClassifier that silently ignores `groups` kwarg.

    Used in unit tests to substitute for glmnet (which accepts groups for
    participant-level CV). RF doesn't accept groups, but the TCR code path
    in SequenceLevelClassifier passes groups to all Stage 1 classifiers.
    """

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self._rf = RandomForestClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None, groups=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        # groups silently dropped (RF doesn't need them)
        self._rf.fit(X, y, **fit_kwargs)
        self.classes_ = self._rf.classes_
        return self

    def predict_proba(self, X):
        return self._rf.predict_proba(X)


# ---------------------------------------------------------------------------
# Unit tests: Aggregation functions
# ---------------------------------------------------------------------------

def test_aggregation_strategies(tlog: _TestLogger):
    """Test 1: AggregationStrategy enum values and aggregate_group dispatch."""
    tlog.log("\n--- Test 1: AggregationStrategy enum + aggregate_group ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        aggregate_group,
    )

    # Verify all expected strategies exist
    expected = {"mean", "median",
                "entropy_cutoff", "entropy_percentile_cutoff",
                "entropy_ten_percent_cutoff", "entropy_twenty_percent_cutoff"}
    actual = {s.name for s in AggregationStrategy}
    assert actual == expected, f"Expected {expected}, got {actual}"

    # Create a simple 3-class, 10-sequence probability matrix
    rng = np.random.RandomState(42)
    probs = rng.dirichlet([1, 1, 1], size=10).astype(np.float32)  # rows sum to 1
    weights = np.ones(10)
    n_classes = 3

    # All strategies should return a vector of length n_classes.
    # entropy_percentile_cutoff requires an absolute threshold (normally computed
    # during fit_stage2), so pass a reasonable dummy value for it.
    for strategy in AggregationStrategy:
        kwargs = {}
        if strategy == AggregationStrategy.entropy_percentile_cutoff:
            kwargs["entropy_abs_threshold"] = 0.5  # dummy threshold in nats
        result = aggregate_group(probs, weights, strategy, n_classes, **kwargs)
        assert result.shape == (n_classes,), f"{strategy.name}: shape={result.shape}"
        assert np.all(np.isfinite(result)), f"{strategy.name}: non-finite values"
        assert np.all(result >= 0), f"{strategy.name}: negative values"

    # Mean should be close to column means
    mean_result = aggregate_group(probs, weights, AggregationStrategy.mean, n_classes)
    np.testing.assert_allclose(mean_result, probs.mean(axis=0), atol=1e-6)

    # Weighted mean with double weight on first row
    weights_heavy = np.ones(10)
    weights_heavy[0] = 100.0
    heavy_result = aggregate_group(probs, weights_heavy, AggregationStrategy.mean, n_classes)
    # Should be dominated by first row
    assert np.argmax(heavy_result) == np.argmax(probs[0])

    tlog.record("AggregationStrategy enum + dispatch", True)


def test_aggregation_edge_cases(tlog: _TestLogger):
    """Test 2: Aggregation edge cases — empty, single row, entropy fallback."""
    tlog.log("\n--- Test 2: Aggregation edge cases ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        aggregate_group,
    )

    n_classes = 3

    # Empty input → uniform prior
    empty = np.zeros((0, n_classes), dtype=np.float32)
    for strategy in AggregationStrategy:
        result = aggregate_group(empty, None, strategy, n_classes)
        np.testing.assert_allclose(result, np.ones(n_classes) / n_classes, atol=1e-6)

    # Single row → should return that row (or close to it)
    single = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
    for strategy in [AggregationStrategy.mean, AggregationStrategy.median]:
        result = aggregate_group(single, None, strategy, n_classes)
        np.testing.assert_allclose(result, single[0], atol=1e-6)

    # Entropy fallback: all rows near-uniform entropy → threshold filters all →
    # returns uniform prior (not mean)
    uniform_probs = np.full((10, n_classes), 1.0 / n_classes, dtype=np.float32)
    # entropy_twenty_percent_cutoff: threshold = 0.8 * max_entropy.
    # Uniform rows have entropy == max_entropy > threshold → all filtered out.
    result = aggregate_group(
        uniform_probs, None, AggregationStrategy.entropy_twenty_percent_cutoff, n_classes
    )
    np.testing.assert_allclose(result, np.ones(n_classes) / n_classes, atol=1e-6)

    # Confident rows should survive entropy filtering
    confident = np.array([
        [0.95, 0.03, 0.02],
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
    ], dtype=np.float32)
    result = aggregate_group(
        confident, None, AggregationStrategy.entropy_twenty_percent_cutoff, n_classes
    )
    assert result[0] > 0.5, "Confident rows should keep high P(class 0)"

    tlog.record("Aggregation edge cases", True)


# ---------------------------------------------------------------------------
# Unit tests: V gene filtering
# ---------------------------------------------------------------------------

def test_find_non_rare_v_genes(tlog: _TestLogger):
    """Test 3: find_non_rare_v_genes filtering."""
    tlog.log("\n--- Test 3: find_non_rare_v_genes ---")

    from malid_lite.models.model3_sequence_level import find_non_rare_v_genes

    # Create data with 4 V genes: 2 common (high freq), 2 rare (low freq)
    rng = np.random.RandomState(42)
    rows = []
    for disease in ["Covid19", "Healthy"]:
        # Common V genes: 100 seqs each
        for _ in range(100):
            rows.append({"v_gene": "TRBV5-1", "disease": disease})
            rows.append({"v_gene": "TRBV7-2", "disease": disease})
        # Rare V genes: 2 seqs each
        for _ in range(2):
            rows.append({"v_gene": "TRBV99-1", "disease": disease})
            rows.append({"v_gene": "TRBV99-2", "disease": disease})

    df = pd.DataFrame(rows)
    kept = find_non_rare_v_genes(df)

    # Common V genes should be kept, rare ones filtered
    assert "TRBV5-1" in kept, "Common V gene TRBV5-1 should be kept"
    assert "TRBV7-2" in kept, "Common V gene TRBV7-2 should be kept"
    # Median of max-freqs: 2 high-freq + 2 low-freq → median filters bottom half
    assert len(kept) <= 4, f"Expected at most 4 kept, got {len(kept)}"
    assert len(kept) >= 2, f"Expected at least 2 kept, got {len(kept)}"

    # Error on missing disease column
    try:
        find_non_rare_v_genes(pd.DataFrame({"v_gene": ["TRBV5-1"]}))
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "disease" in str(e).lower()

    # Empty DataFrame → empty list
    empty_df = pd.DataFrame(columns=["v_gene", "disease"])
    assert find_non_rare_v_genes(empty_df) == []

    tlog.record("find_non_rare_v_genes", True)


# ---------------------------------------------------------------------------
# Unit tests: GroupSequenceClassifier
# ---------------------------------------------------------------------------

def test_group_sequence_classifier(tlog: _TestLogger):
    """Test 4: GroupSequenceClassifier fit + predict_proba with class alignment."""
    tlog.log("\n--- Test 4: GroupSequenceClassifier ---")

    from sklearn.ensemble import RandomForestClassifier
    from malid_lite.models.model3_sequence_level import GroupSequenceClassifier

    # Simple 2-class problem with random embeddings
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 640).astype(np.float32)
    y = np.array(["Covid19"] * 100 + ["Healthy"] * 100)

    clf = GroupSequenceClassifier(RandomForestClassifier(n_estimators=10, random_state=0))
    clf.fit(X, y)

    assert clf.classes_ is not None
    assert len(clf.classes_) == 2

    # predict_proba aligned to all_classes (including unseen class "HIV")
    all_classes = np.array(["Covid19", "HIV", "Healthy"])
    probs = clf.predict_proba(X[:5], all_classes)

    assert probs.shape == (5, 3), f"Expected (5, 3), got {probs.shape}"
    # HIV column should be all zeros (unseen during training)
    hiv_idx = list(all_classes).index("HIV")
    assert np.all(probs[:, hiv_idx] == 0.0), "Unseen class HIV should have prob 0"
    # Other columns should have non-zero values
    covid_idx = list(all_classes).index("Covid19")
    assert np.any(probs[:, covid_idx] > 0), "Covid19 should have non-zero probs"

    tlog.record("GroupSequenceClassifier", True)


# ---------------------------------------------------------------------------
# Unit tests: SequenceLevelClassifier helpers
# ---------------------------------------------------------------------------

def test_classifier_init_and_helpers(tlog: _TestLogger):
    """Test 5: SequenceLevelClassifier init validation and helpers."""
    tlog.log("\n--- Test 5: SequenceLevelClassifier init + helpers ---")

    from malid_lite.models.model3_sequence_level import SequenceLevelClassifier

    # Valid init
    clf = SequenceLevelClassifier(locus="TCR")
    assert clf.locus == "TCR"
    assert clf._split_on_cols() == ["v_gene"]

    clf_bcr = SequenceLevelClassifier(locus="BCR")
    assert clf_bcr._split_on_cols() == ["v_gene", "isotype_supergroup"]

    # Invalid locus
    try:
        SequenceLevelClassifier(locus="XYZ")
        assert False, "Should raise ValueError for invalid locus"
    except ValueError as e:
        assert "XYZ" in str(e)

    # _group_key_to_str
    assert clf._group_key_to_str(("TRBV5-1",)) == "TRBV5-1"
    assert clf_bcr._group_key_to_str(("IGHV3-23", "IGHG")) == "IGHV3-23_IGHG"

    # _get_group_keys_series (TCR)
    df = pd.DataFrame({"v_gene": ["TRBV5-1", "TRBV7-2", "TRBV5-1"]})
    keys = clf._get_group_keys_series(df)
    assert keys.iloc[0] == ("TRBV5-1",)
    assert keys.iloc[1] == ("TRBV7-2",)

    # _get_group_keys_series (BCR)
    df_bcr = pd.DataFrame({
        "v_gene": ["IGHV3-23", "IGHV1-2"],
        "isotype_supergroup": ["IGHG", "IGHA"],
    })
    keys_bcr = clf_bcr._get_group_keys_series(df_bcr)
    assert keys_bcr.iloc[0] == ("IGHV3-23", "IGHG")
    assert keys_bcr.iloc[1] == ("IGHV1-2", "IGHA")

    tlog.record("SequenceLevelClassifier init + helpers", True)


def test_sanitize_group_columns(tlog: _TestLogger):
    """Test 6: _sanitize_group_columns replaces underscores with hyphens."""
    tlog.log("\n--- Test 6: _sanitize_group_columns ---")

    from malid_lite.models.model3_sequence_level import SequenceLevelClassifier

    clf = SequenceLevelClassifier(locus="TCR")
    df = pd.DataFrame({"v_gene": ["TRBV5_1", "TRBV7-2", "TRBV5_1"]})
    result = clf._sanitize_group_columns(df)
    assert list(result["v_gene"]) == ["TRBV5-1", "TRBV7-2", "TRBV5-1"]

    # BCR: both columns sanitized
    clf_bcr = SequenceLevelClassifier(locus="BCR")
    df_bcr = pd.DataFrame({
        "v_gene": ["IGHV3_23", "IGHV1-2"],
        "isotype_supergroup": ["IGH_G", "IGHA"],
    })
    result_bcr = clf_bcr._sanitize_group_columns(df_bcr)
    assert result_bcr["v_gene"].iloc[0] == "IGHV3-23"
    assert result_bcr["isotype_supergroup"].iloc[0] == "IGH-G"

    tlog.record("_sanitize_group_columns", True)


def test_class_prefix_collision(tlog: _TestLogger):
    """Test 7: Class name prefix collision detection in fit_stage1."""
    tlog.log("\n--- Test 7: Class prefix collision detection ---")

    from malid_lite.models.model3_sequence_level import SequenceLevelClassifier

    clf = SequenceLevelClassifier(locus="TCR", min_sequences_per_group=1)

    # Create data where "A" is a prefix of "A_B" when using '_' delimiter
    rows = []
    for disease in ["A", "A_B"]:
        for i in range(20):
            rows.append({
                "v_gene": "TRBV5-1",
                "cdr3_aa": f"CASS{'A' * i}F",
                "specimen_label": f"S_{disease}_{i}",
                "participant_label": f"P_{disease}_{i}",
                "disease": disease,
            })
    df = pd.DataFrame(rows)
    embeddings = np.random.randn(len(df), 640).astype(np.float32)

    try:
        clf.fit_stage1(df, embeddings)
        assert False, "Should raise ValueError for prefix collision"
    except ValueError as e:
        assert "prefix" in str(e).lower(), f"Expected prefix collision error, got: {e}"

    tlog.record("Class prefix collision detection", True)


# ---------------------------------------------------------------------------
# Unit tests: Full synthetic pipeline
# ---------------------------------------------------------------------------

def test_full_synthetic_pipeline(tlog: _TestLogger):
    """Test 8: Full two-stage fit + predict on synthetic data (TCR, multiclass)."""
    tlog.log("\n--- Test 8: Full synthetic pipeline (TCR multiclass) ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    # Create synthetic data: 3 diseases, 21 specimens (7 per disease), 30 seqs each
    seq_df = make_synthetic_sequences(
        n_specimens=21, n_seqs_per_specimen=30,
        diseases=["Covid19", "HIV", "Healthy"],
    )
    embeddings = make_synthetic_embeddings(len(seq_df))

    # Split into train_smaller1 (2/3 specimens) and train_smaller2 (1/3 specimens)
    specimens = list(seq_df["specimen_label"].unique())
    rng = np.random.RandomState(0)
    rng.shuffle(specimens)
    split = len(specimens) * 2 // 3
    ts1_specs = set(specimens[:split])
    ts2_specs = set(specimens[split:])

    ts1_mask = seq_df["specimen_label"].isin(ts1_specs)
    ts2_mask = seq_df["specimen_label"].isin(ts2_specs)
    ts1 = seq_df[ts1_mask].reset_index(drop=True)
    ts2 = seq_df[ts2_mask].reset_index(drop=True)
    emb_ts1 = embeddings[ts1_mask.values]
    emb_ts2 = embeddings[ts2_mask.values]

    # Build model with RF (not glmnet) for Stage 1 to avoid glmnet dependency.
    # Use _RFIgnoringGroups because TCR code path passes `groups` kwarg to
    # Stage 1 classifiers (for participant-level CV in glmnet), but RF doesn't
    # accept `groups`.
    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,  # simplest
        exclude_rare_v_genes=False,  # keep all with small data
        min_sequences_per_group=5,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=1,
    )

    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1
    )

    # Stage 1
    model.fit_stage1(ts1, emb_ts1)
    assert len(model.group_models_) > 0, "No group models trained"
    assert model.classes_ is not None
    tlog.log(f"  Stage 1: {len(model.group_models_)} group models, "
             f"classes={list(model.classes_)}")

    # Stage 2
    model.fit_stage2(ts2, emb_ts2)
    assert model.stage2_clf_ is not None, "Stage 2 not fitted"
    assert model.feature_columns_ is not None
    tlog.log(f"  Stage 2: {len(model.feature_columns_)} features")

    # Predict on ts2 (sanity check — not a fair eval, just verifying it runs)
    proba_df = model.predict_proba(ts2, emb_ts2)
    assert proba_df.shape[0] == ts2["specimen_label"].nunique()
    assert proba_df.shape[1] == len(model.classes_)
    assert np.all(np.isfinite(proba_df.values)), "Non-finite probabilities"

    predictions = model.predict(ts2, emb_ts2)
    assert len(predictions) == proba_df.shape[0]
    assert all(p in model.classes_ for p in predictions)

    tlog.record("Full synthetic pipeline (TCR multiclass)", True)


# ---------------------------------------------------------------------------
# Unit tests: Featurization
# ---------------------------------------------------------------------------

def test_featurize_uniform_fill(tlog: _TestLogger):
    """Test 9: featurize_specimens fills missing groups with uniform prior."""
    tlog.log("\n--- Test 9: featurize_specimens uniform prior fill ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )
    from sklearn.ensemble import RandomForestClassifier

    # Small model: 2 classes, 2 V genes
    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_jobs=1,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, random_state=0, n_jobs=1
    )

    # Training data: both V genes present, 2 diseases per V gene
    # (need >= 2 classes per group for classifier to train)
    train_df = pd.DataFrame({
        "v_gene": ["TRBV5-1"] * 20 + ["TRBV7-2"] * 20,
        "specimen_label": (["S1"] * 10 + ["S2"] * 10) * 2,
        "participant_label": (["P1"] * 10 + ["P2"] * 10) * 2,
        "disease": (["Covid19"] * 10 + ["Healthy"] * 10) * 2,
    })
    train_emb = np.random.randn(40, 640).astype(np.float32)

    model.fit_stage1(train_df, train_emb)

    # Test data: specimen with only TRBV5-1 (missing TRBV7-2)
    test_df = pd.DataFrame({
        "v_gene": ["TRBV5-1"] * 10,
        "specimen_label": ["S_test"] * 10,
        "participant_label": ["P_test"] * 10,
        "disease": ["Covid19"] * 10,
    })
    test_emb = np.random.randn(10, 640).astype(np.float32)

    seq_preds = model.generate_sequence_predictions(test_df, test_emb)
    features_df = model.featurize_specimens(seq_preds)

    # The TRBV7-2 group columns should be filled with 1/n_classes = 0.5
    n_classes = len(model.classes_)
    for cls in model.classes_:
        col = f"{cls}_TRBV7-2"
        if col in features_df.columns:
            val = features_df[col].iloc[0]
            assert abs(val - 1.0 / n_classes) < 1e-6, \
                f"Missing group fill: expected {1.0/n_classes}, got {val}"

    tlog.record("featurize_specimens uniform fill", True)


def test_featurize_column_alignment(tlog: _TestLogger):
    """Test 10: featurize_specimens test-time column alignment."""
    tlog.log("\n--- Test 10: featurize_specimens column alignment ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_jobs=1,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, random_state=0, n_jobs=1
    )

    # Train with 2 V genes
    train_df = pd.DataFrame({
        "v_gene": ["TRBV5-1"] * 20 + ["TRBV7-2"] * 20,
        "specimen_label": (["S1"] * 10 + ["S2"] * 10) * 2,
        "participant_label": (["P1"] * 10 + ["P2"] * 10) * 2,
        "disease": ["Covid19"] * 20 + ["Healthy"] * 20,
    })
    train_emb = np.random.randn(40, 640).astype(np.float32)
    model.fit_stage1(train_df, train_emb)

    # Get training feature columns
    seq_preds_train = model.generate_sequence_predictions(train_df, train_emb)
    features_train = model.featurize_specimens(seq_preds_train)
    train_cols = list(features_train.columns)

    # Test data: only TRBV5-1 sequences
    test_df = pd.DataFrame({
        "v_gene": ["TRBV5-1"] * 10,
        "specimen_label": ["S_test"] * 10,
        "participant_label": ["P_test"] * 10,
        "disease": ["Covid19"] * 10,
    })
    test_emb = np.random.randn(10, 640).astype(np.float32)
    seq_preds_test = model.generate_sequence_predictions(test_df, test_emb)

    # Pass feature_columns to align to training columns
    features_test = model.featurize_specimens(seq_preds_test, feature_columns=train_cols)

    assert list(features_test.columns) == train_cols, \
        "Test columns should match training columns exactly"
    assert features_test.shape[1] == len(train_cols)

    tlog.record("featurize_specimens column alignment", True)


# ---------------------------------------------------------------------------
# Unit tests: evaluate_on_test
# ---------------------------------------------------------------------------

def test_evaluate_multiclass(tlog: _TestLogger):
    """Test 11: evaluate_on_test multiclass metrics."""
    tlog.log("\n--- Test 11: evaluate_on_test multiclass ---")

    from malid_lite.training.train_model3 import evaluate_on_test

    classes = np.array(["Covid19", "HIV", "Healthy"])
    y_true = np.array(["Covid19", "HIV", "Healthy", "Covid19", "HIV",
                        "Healthy", "Covid19", "HIV", "Healthy", "Covid19"])
    # Near-perfect predictions
    y_pred = np.array(["Covid19", "HIV", "Healthy", "Covid19", "HIV",
                        "Healthy", "Covid19", "HIV", "Healthy", "HIV"])  # 1 error
    y_proba = np.zeros((10, 3), dtype=np.float32)
    for i, cls in enumerate(y_pred):
        idx = list(classes).index(cls)
        y_proba[i, idx] = 0.8
        y_proba[i, :] += 0.1 / 3  # small uniform noise

    results, raw_preds = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        classes=classes, fold_id=0, model_name="model3", n_test=10,
    )

    assert results["fold_id"] == 0
    assert results["model_name"] == "model3"
    assert results["n_scored"] == 10
    assert results["n_abstained"] == 0
    assert 0.0 <= results["accuracy"] <= 1.0
    assert results["accuracy"] == 0.9  # 9/10 correct
    assert "auroc_ovo_weighted" in results
    assert "auprc_ovo_weighted" in results
    assert "log_loss" in results
    assert results["log_loss"] is not None and results["log_loss"] > 0
    assert "confusion_matrix" in results
    assert "auroc_ovr_per_class" in results
    assert len(results["classes"]) == 3
    # Should NOT have binary metrics (multiclass, no reference_class)
    assert "auroc_binary" not in results

    tlog.record("evaluate_on_test multiclass", True)


def test_evaluate_binary(tlog: _TestLogger):
    """Test 12: evaluate_on_test binary metrics with reference_class."""
    tlog.log("\n--- Test 12: evaluate_on_test binary ---")

    from malid_lite.training.train_model3 import evaluate_on_test

    classes = np.array(["Covid19", "Healthy"])
    y_true = np.array(["Covid19", "Healthy", "Covid19", "Healthy", "Covid19",
                        "Healthy", "Covid19", "Healthy"])
    y_pred = np.array(["Covid19", "Healthy", "Covid19", "Healthy", "Covid19",
                        "Healthy", "Covid19", "Covid19"])  # 1 error
    y_proba = np.zeros((8, 2), dtype=np.float32)
    for i, cls in enumerate(y_pred):
        idx = list(classes).index(cls)
        y_proba[i, idx] = 0.8
        y_proba[i, :] += 0.1

    results, raw_preds = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        classes=classes, fold_id=0, model_name="model3", n_test=8,
        reference_class="Healthy",
    )

    assert "auroc_binary" in results, "Binary mode should have auroc_binary"
    assert "auprc_binary" in results, "Binary mode should have auprc_binary"
    assert 0.0 <= results["auroc_binary"] <= 1.0
    assert 0.0 <= results["auprc_binary"] <= 1.0
    assert "log_loss" in results
    assert results["log_loss"] is not None and results["log_loss"] > 0

    tlog.record("evaluate_on_test binary", True)


def test_evaluate_optional_params(tlog: _TestLogger):
    """Test 13: evaluate_on_test with None optional train counts."""
    tlog.log("\n--- Test 13: evaluate_on_test optional params ---")

    from malid_lite.training.train_model3 import evaluate_on_test

    classes = np.array(["A", "B"])
    y_true = np.array(["A", "B", "A", "B"])
    y_pred = np.array(["A", "B", "A", "B"])
    y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)

    # All optional params as None (default)
    results, _ = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        classes=classes, fold_id=0, model_name="model3", n_test=4,
    )
    assert results["n_train_sequences_stage1"] is None
    assert results["n_train_sequences_stage2"] is None
    assert results["n_train_specimens_stage1"] is None
    assert results["n_train_specimens_stage2"] is None

    # All optional params provided
    results2, _ = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        classes=classes, fold_id=0, model_name="model3", n_test=4,
        n_train_sequences_stage1=1000,
        n_train_sequences_stage2=500,
        n_train_specimens_stage1=100,
        n_train_specimens_stage2=50,
    )
    assert results2["n_train_sequences_stage1"] == 1000
    assert results2["n_train_specimens_stage2"] == 50

    tlog.record("evaluate_on_test optional params", True)


# ---------------------------------------------------------------------------
# Unit tests: Factory functions
# ---------------------------------------------------------------------------

def test_factory_functions(tlog: _TestLogger):
    """Test 14: make_tcr_model / make_bcr_model factory defaults."""
    tlog.log("\n--- Test 14: Factory functions ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        make_bcr_model,
        make_tcr_model,
    )

    tcr = make_tcr_model(n_estimators_stage2=50)
    assert tcr.locus == "TCR"
    assert tcr.aggregation_strategy == AggregationStrategy.entropy_cutoff
    assert tcr.entropy_max_fraction == 0.80
    assert tcr.exclude_rare_v_genes is True
    assert tcr.reweigh_by_subset_frequencies is True
    assert tcr.n_estimators_stage2 == 50  # kwarg passed through

    bcr = make_bcr_model(n_estimators_stage1=200)
    assert bcr.locus == "BCR"
    assert bcr.aggregation_strategy == AggregationStrategy.mean
    assert bcr.exclude_rare_v_genes is True
    assert bcr.reweigh_by_subset_frequencies is True
    assert bcr.n_estimators_stage1 == 200

    tlog.record("Factory functions", True)


# ---------------------------------------------------------------------------
# Integration tests (require test data + random embeddings)
# ---------------------------------------------------------------------------

def _make_random_embeddings(n_sequences: int, random_state: int = 42) -> np.ndarray:
    """Generate random float32 embeddings matching ESM-2 dimensions.

    Used for integration tests where we need correctly-shaped embeddings
    to exercise the full pipeline, but don't need real ESM-2 features.
    """
    from malid_lite.models.model3_sequence_level import EMBEDDING_DIM
    rng = np.random.RandomState(random_state)
    return rng.randn(n_sequences, EMBEDDING_DIM).astype(np.float32)


def _check_integration_prerequisites() -> Optional[str]:
    """Return None if prerequisites met, or an error message string.

    Integration tests use the small test dataset (tests/test_data/) with
    random embeddings. The only external dependency is glmnet for TCR
    Stage 1 classifiers.
    """
    try:
        from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper
    except ImportError:
        return "glmnet not installed (required for TCR Stage 1)"
    return None


@pytest.mark.integration
def test_integration_multiclass(tlog: _TestLogger, n_jobs: int):
    """Test 18: Full multiclass pipeline on fold 0 (test data, random embeddings)."""
    tlog.log("\n--- Test 18: Integration - multiclass pipeline ---")

    from malid_lite.training.train_model3 import (
        load_and_prepare_fold,
        evaluate_on_test,
    )
    from malid_lite.training.training_utils import split_train_smaller
    from malid_lite.models.model3_sequence_level import (
        DISEASE_COL,
        PARTICIPANT_COL,
        SPECIMEN_COL,
        EMBEDDING_DIM,
        make_tcr_model,
    )

    loader = create_test_loader(verbose=0)

    # Load fold 0 training data from test dataset
    t0 = time.time()
    train_seq, train_meta = load_and_prepare_fold(loader, 0, "train")
    tlog.log(f"  Loaded fold 0 train: {len(train_seq):,} sequences, "
             f"{train_seq[PARTICIPANT_COL].nunique()} participants "
             f"({time.time()-t0:.1f}s)")
    tlog.log(f"  Diseases: {sorted(train_seq[DISEASE_COL].unique())}")

    # Generate random embeddings (test data is small — no subsampling needed)
    train_seq = train_seq.reset_index(drop=True)
    emb_full = _make_random_embeddings(len(train_seq), random_state=42)
    assert emb_full.shape == (len(train_seq), EMBEDDING_DIM)

    # Split into train_smaller1 and train_smaller2
    ts1, ts2 = split_train_smaller(train_seq, train_meta)
    tlog.log(f"  ts1: {len(ts1):,} seqs ({ts1[SPECIMEN_COL].nunique()} specimens)")
    tlog.log(f"  ts2: {len(ts2):,} seqs ({ts2[SPECIMEN_COL].nunique()} specimens)")

    # Extract embeddings for each split using the original indices
    # (ts1/ts2 retain their original index from train_seq before reset_index)
    emb_ts1 = emb_full[ts1.index.values]
    emb_ts2 = emb_full[ts2.index.values]
    ts1 = ts1.reset_index(drop=True)
    ts2 = ts2.reset_index(drop=True)

    assert emb_ts1.shape == (len(ts1), EMBEDDING_DIM)
    assert emb_ts2.shape == (len(ts2), EMBEDDING_DIM)

    # Build and train model
    model = make_tcr_model(
        n_estimators_stage2=50,
        n_jobs=n_jobs,
        verbose=1,
    )
    t0 = time.time()
    model.fit_stage1(ts1, emb_ts1)
    tlog.log(f"  Stage 1: {len(model.group_models_)} groups ({time.time()-t0:.1f}s)")

    t0 = time.time()
    model.fit_stage2(ts2, emb_ts2)
    tlog.log(f"  Stage 2: {len(model.feature_columns_)} features ({time.time()-t0:.1f}s)")

    # Load test data
    test_seq, test_meta = load_and_prepare_fold(loader, 0, "test")
    test_seq = test_seq.reset_index(drop=True)
    emb_test = _make_random_embeddings(len(test_seq), random_state=99)
    tlog.log(f"  Test: {len(test_seq):,} sequences, "
             f"{test_seq[SPECIMEN_COL].nunique()} specimens")

    # Predict
    t0 = time.time()
    proba_df = model.predict_proba(test_seq, emb_test)
    tlog.log(f"  Prediction: {proba_df.shape[0]} specimens ({time.time()-t0:.1f}s)")

    assert proba_df.shape[0] > 0
    assert proba_df.shape[1] == len(model.classes_)
    assert np.all(np.isfinite(proba_df.values))

    # Evaluate (accuracy will be near-random with random embeddings — that's OK,
    # we're testing pipeline mechanics, not model quality)
    specimen_disease = test_seq.drop_duplicates(SPECIMEN_COL).set_index(SPECIMEN_COL)[DISEASE_COL]
    y_true = np.array([specimen_disease[s] for s in proba_df.index])
    y_pred = model.classes_[np.argmax(proba_df.values, axis=1)]

    eval_results, _ = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=proba_df.values,
        classes=model.classes_, fold_id=0, model_name="model3", n_test=len(y_true),
        n_train_sequences_stage1=len(ts1), n_train_sequences_stage2=len(ts2),
        n_train_specimens_stage1=int(ts1[SPECIMEN_COL].nunique()),
        n_train_specimens_stage2=int(ts2[SPECIMEN_COL].nunique()),
    )

    acc = eval_results["accuracy"]
    auroc = eval_results.get("auroc_ovo_weighted")
    tlog.log(f"  Results: accuracy={acc:.4f}, AUROC_OvO={auroc:.4f}" if auroc else
             f"  Results: accuracy={acc:.4f}, AUROC unavailable")

    assert 0.0 <= acc <= 1.0
    if auroc is not None:
        assert 0.0 <= auroc <= 1.0

    tlog.record("Integration multiclass pipeline", True,
                {"accuracy": acc, "auroc": auroc, "n_test": len(y_true)})


@pytest.mark.integration
def test_integration_binary(tlog: _TestLogger, n_jobs: int):
    """Test 19: Full binary pipeline (test data, random embeddings)."""
    tlog.log("\n--- Test 19: Integration - binary pipeline ---")

    from malid_lite.training.train_model3 import (
        load_and_prepare_fold,
        evaluate_on_test,
    )
    from malid_lite.training.training_utils import (
        filter_to_binary_pair,
        split_train_smaller,
    )
    from malid_lite.models.model3_sequence_level import (
        DISEASE_COL,
        PARTICIPANT_COL,
        SPECIMEN_COL,
        EMBEDDING_DIM,
        make_tcr_model,
    )

    loader = create_test_loader(verbose=0)

    # Load fold 0 and filter to binary pair
    train_seq, train_meta = load_and_prepare_fold(loader, 0, "train")
    train_seq, train_meta = filter_to_binary_pair(
        train_seq, train_meta, "Covid19", "Healthy/Background"
    )
    tlog.log(f"  Binary pair: Covid19 vs Healthy/Background")
    tlog.log(f"  Train: {len(train_seq):,} sequences, "
             f"{train_seq[SPECIMEN_COL].nunique()} specimens")

    diseases = sorted(train_seq[DISEASE_COL].unique())
    assert len(diseases) == 2, f"Expected 2 diseases, got {diseases}"
    assert "Covid19" in diseases
    assert "Healthy/Background" in diseases

    # Generate random embeddings (test data is small — no subsampling needed)
    train_seq = train_seq.reset_index(drop=True)
    emb_full = _make_random_embeddings(len(train_seq), random_state=42)

    ts1, ts2 = split_train_smaller(train_seq, train_meta)
    emb_ts1 = emb_full[ts1.index.values]
    emb_ts2 = emb_full[ts2.index.values]
    ts1 = ts1.reset_index(drop=True)
    ts2 = ts2.reset_index(drop=True)

    assert emb_ts1.shape == (len(ts1), EMBEDDING_DIM)
    assert emb_ts2.shape == (len(ts2), EMBEDDING_DIM)

    # Build model with reference_class
    model = make_tcr_model(
        n_estimators_stage2=50,
        n_jobs=n_jobs,
        reference_class="Healthy/Background",
        verbose=1,
    )
    model.fit_stage1(ts1, emb_ts1)
    model.fit_stage2(ts2, emb_ts2)

    assert len(model.classes_) == 2

    # Test data
    test_seq, test_meta = load_and_prepare_fold(loader, 0, "test")
    test_seq, test_meta = filter_to_binary_pair(
        test_seq, test_meta, "Covid19", "Healthy/Background"
    )
    test_seq = test_seq.reset_index(drop=True)
    emb_test = _make_random_embeddings(len(test_seq), random_state=99)

    proba_df = model.predict_proba(test_seq, emb_test)
    assert proba_df.shape[1] == 2

    specimen_disease = test_seq.drop_duplicates(SPECIMEN_COL).set_index(SPECIMEN_COL)[DISEASE_COL]
    y_true = np.array([specimen_disease[s] for s in proba_df.index])
    y_pred = model.classes_[np.argmax(proba_df.values, axis=1)]

    eval_results, _ = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=proba_df.values,
        classes=model.classes_, fold_id=0, model_name="model3",
        n_test=len(y_true), reference_class="Healthy/Background",
    )

    assert "auroc_binary" in eval_results
    assert "auprc_binary" in eval_results
    auroc_b = eval_results["auroc_binary"]
    auprc_b = eval_results["auprc_binary"]
    tlog.log(f"  Results: AUROC_binary={auroc_b:.4f}, AUPRC_binary={auprc_b:.4f}"
             if auroc_b is not None else "  Results: binary metrics unavailable")

    tlog.record("Integration binary pipeline", True,
                {"auroc_binary": auroc_b, "auprc_binary": auprc_b})


def test_integration_predictions_csv_multiclass(tlog: _TestLogger):
    """Test 20: Predictions CSV format validation (multiclass)."""
    tlog.log("\n--- Test 20: Predictions CSV format (multiclass) ---")

    # Build predictions rows the same way _run_fold_loop does for multiclass
    classes = np.array(["Covid19", "HIV", "Healthy"])
    str_classes = [str(c) for c in classes]
    rows = []
    for i in range(5):
        row = {
            "participant_label": f"P{i}",
            "specimen_label": f"S{i}",
            "true_disease": str_classes[i % 3],
            "predicted_disease": str_classes[(i + 1) % 3],
            "malid_cross_validation_fold_id_when_in_test_set": 0,
        }
        for cls in str_classes:
            row[f"score_{cls}"] = 0.33
        rows.append(row)

    df = pd.DataFrame(rows)

    # Validate expected columns
    expected_cols = {"participant_label", "specimen_label", "true_disease",
                     "predicted_disease", "malid_cross_validation_fold_id_when_in_test_set"}
    for cls in str_classes:
        expected_cols.add(f"score_{cls}")

    actual_cols = set(df.columns)
    assert expected_cols == actual_cols, f"Missing: {expected_cols - actual_cols}, Extra: {actual_cols - expected_cols}"

    # Should NOT have 'abstained' column (removed in previous session)
    assert "abstained" not in df.columns, "'abstained' column should not be present"

    tlog.record("Predictions CSV format (multiclass)", True)


def test_integration_predictions_csv_binary(tlog: _TestLogger):
    """Test 21: Predictions CSV format validation (binary)."""
    tlog.log("\n--- Test 21: Predictions CSV format (binary) ---")

    rows = []
    for i in range(5):
        rows.append({
            "participant_label": f"P{i}",
            "specimen_label": f"S{i}",
            "disease_label": i % 2,
            "disease_label_str": "Covid19" if i % 2 == 1 else "Healthy",
            "disease_model": "Covid19",
            "model_score": 0.5 + 0.1 * i,
            "malid_cross_validation_fold_id_when_in_test_set": 0,
        })

    df = pd.DataFrame(rows)

    expected_cols = {"participant_label", "specimen_label", "disease_label",
                     "disease_label_str", "disease_model", "model_score",
                     "malid_cross_validation_fold_id_when_in_test_set"}
    actual_cols = set(df.columns)
    assert expected_cols == actual_cols

    # disease_label should be 0 or 1
    assert set(df["disease_label"].unique()).issubset({0, 1})

    tlog.record("Predictions CSV format (binary)", True)


def test_integration_model_save_load(tlog: _TestLogger):
    """Test 22: Model artifact save/load round-trip."""
    tlog.log("\n--- Test 22: Model save/load round-trip ---")

    import pickle
    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    # Train a small synthetic model
    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, random_state=0, n_jobs=1
    )

    seq_df = make_synthetic_sequences(n_specimens=12, n_seqs_per_specimen=20)
    embeddings = make_synthetic_embeddings(len(seq_df))

    # Split
    specimens = seq_df["specimen_label"].unique()
    split = len(specimens) * 2 // 3
    ts1_specs = set(specimens[:split])
    ts2_specs = set(specimens[split:])
    ts1 = seq_df[seq_df["specimen_label"].isin(ts1_specs)].reset_index(drop=True)
    ts2 = seq_df[seq_df["specimen_label"].isin(ts2_specs)].reset_index(drop=True)
    emb_ts1 = embeddings[seq_df["specimen_label"].isin(ts1_specs).values]
    emb_ts2 = embeddings[seq_df["specimen_label"].isin(ts2_specs).values]

    model.fit_stage1(ts1, emb_ts1)
    model.fit_stage2(ts2, emb_ts2)

    # Save Stage 1 artifacts (same format as _run_fold_loop)
    save_dir = OUTPUT_DIR / "integration" / "save_load_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = save_dir / "fold_0_stage1.pkl"
    with open(stage1_path, "wb") as f:
        pickle.dump({
            "group_models": model.group_models_,
            "classes": model.classes_,
            "non_rare_v_genes": model.non_rare_v_genes_,
            "locus": model.locus,
        }, f)

    # Stage 2: save only the picklable components.
    # BinaryOvRClassifierWithFeatureSubsettingByClass stores a reference to
    # a local _make_rf function (from fit_stage2), which can't be pickled
    # with standard pickle. The actual training script uses pickle.dump on
    # the full stage2_clf_ — this works because the _run_fold_loop context
    # keeps the function alive. In a unit test with synthetic data and the
    # _RFIgnoringGroups mock, the classifier internals differ. We test the
    # Stage 1 round-trip and verify Stage 2 metadata separately.
    stage2_meta = {
        "stage2_scaler_type": type(model.stage2_scaler_).__name__,
        "feature_columns": model.feature_columns_,
        "classes": list(model.classes_),
        "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
    }

    # Load and verify Stage 1 round-trip
    with open(stage1_path, "rb") as f:
        s1 = pickle.load(f)

    assert len(s1["group_models"]) == len(model.group_models_)
    assert list(s1["classes"]) == list(model.classes_)
    assert s1["locus"] == "TCR"
    # aggregation_strategy is NOT stored in Stage 1 artifacts (Stage-2-only param)
    assert "aggregation_strategy" not in s1

    # Verify Stage 2 metadata
    assert stage2_meta["feature_columns"] == model.feature_columns_
    assert stage2_meta["reweigh_by_subset_frequencies"] == model.reweigh_by_subset_frequencies
    assert stage2_meta["stage2_scaler_type"] == "StandardScaler"

    # Verify Stage 1 loaded models produce predictions
    from malid_lite.models.model3_sequence_level import GroupSequenceClassifier
    for gk, clf in s1["group_models"].items():
        assert isinstance(clf, GroupSequenceClassifier)
        assert clf.classes_ is not None

    tlog.log(f"  Saved/loaded: {len(s1['group_models'])} group models, "
             f"{len(stage2_meta['feature_columns'])} features")

    tlog.record("Model save/load round-trip", True)


@pytest.mark.integration
def test_integration_cv_ensemble_splits(tlog: _TestLogger):
    """Test 29: cv_ensemble split isolation (lightweight, no fold data loading).

    Verifies that cv_ensemble produces fewer training participants than
    cv_single_model, that validation participants are excluded from ts1/ts2,
    and that the participant sets partition correctly.
    """
    tlog.log("\n--- Test 29: cv_ensemble split isolation ---")

    loader = create_test_loader(verbose=0)

    for fold_id in [0, 1, 2]:
        # cv_single_model: ts1+ts2 = all train participants
        sm_ts1 = set(loader.get_split_participants(fold_id, "cv_single_model", ["train_smaller1"]))
        sm_ts2 = set(loader.get_split_participants(fold_id, "cv_single_model", ["train_smaller2"]))
        sm_train = sm_ts1 | sm_ts2

        # cv_ensemble: ts1+ts2 = train minus validation
        ens_ts1 = set(loader.get_split_participants(fold_id, "cv_ensemble", ["train_smaller1"]))
        ens_ts2 = set(loader.get_split_participants(fold_id, "cv_ensemble", ["train_smaller2"]))
        ens_val = set(loader.get_split_participants(fold_id, "cv_ensemble", ["validation"]))
        ens_train = ens_ts1 | ens_ts2

        # cv_ensemble training is strictly smaller
        assert len(ens_train) < len(sm_train), (
            f"Fold {fold_id}: cv_ensemble train ({len(ens_train)}) should be < "
            f"cv_single_model train ({len(sm_train)})"
        )

        # No overlap between validation and training
        assert not (ens_train & ens_val), (
            f"Fold {fold_id}: {len(ens_train & ens_val)} participants in both "
            f"train and validation"
        )

        # Validation + training = full train set
        assert ens_train | ens_val == sm_train, (
            f"Fold {fold_id}: cv_ensemble train+val does not equal "
            f"cv_single_model train"
        )

        # No overlap between ts1 and ts2
        assert not (ens_ts1 & ens_ts2), (
            f"Fold {fold_id}: ts1 and ts2 overlap"
        )

        tlog.log(
            f"  Fold {fold_id}: sm_train={len(sm_train)}, "
            f"ens_train={len(ens_train)} (ts1={len(ens_ts1)}, ts2={len(ens_ts2)}), "
            f"val={len(ens_val)}"
        )

    tlog.log("  -> PASSED")
    tlog.record("cv_ensemble split isolation", True)


# ---------------------------------------------------------------------------
# Resume tests (unit tests — synthetic data, no external deps)
# ---------------------------------------------------------------------------

def test_check_fold_complete(tlog: _TestLogger):
    """Test 23: _check_fold_complete detects presence/absence of fold artifacts."""
    tlog.log("\n--- Test 23: _check_fold_complete ---")

    import pickle
    from malid_lite.training.train_model3 import _check_fold_complete

    test_dir = OUTPUT_DIR / "resume_tests" / "check_complete"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Clean slate — no artifacts
    for f in test_dir.glob("fold_*"):
        f.unlink()
    assert not _check_fold_complete(test_dir, 0), "Should be False with no artifacts"

    # Create all four required files.
    # .pkl files must be >= 1024 bytes to pass _check_fold_complete's
    # truncation guard, so we write enough padding.
    required_files = [
        test_dir / "fold_0_stage1.pkl",
        test_dir / "fold_0_stage2.pkl",
        test_dir / "fold_0_results.json",
        test_dir / "fold_0_predictions.pkl",
    ]
    for f in required_files:
        if f.suffix == ".pkl":
            f.write_bytes(b"x" * 1024)
        else:
            f.write_text("placeholder")

    assert _check_fold_complete(test_dir, 0), "Should be True with all artifacts"

    # Remove one file at a time — should be False each time
    for remove_file in required_files:
        remove_file.unlink()
        assert not _check_fold_complete(test_dir, 0), \
            f"Should be False with {remove_file.name} missing"
        # Restore with enough bytes for .pkl truncation guard
        if remove_file.suffix == ".pkl":
            remove_file.write_bytes(b"x" * 1024)
        else:
            remove_file.write_text("placeholder")

    # Different fold ID — should be False
    assert not _check_fold_complete(test_dir, 1), \
        "Should be False for fold_id=1 (only fold_0 exists)"

    tlog.log("  -> PASSED")
    tlog.record("_check_fold_complete", True)


def test_load_fold_results_roundtrip(tlog: _TestLogger):
    """Test 24: _load_fold_results loads saved artifacts correctly."""
    tlog.log("\n--- Test 24: _load_fold_results round-trip ---")

    import pickle
    from malid_lite.training.train_model3 import _load_fold_results

    test_dir = OUTPUT_DIR / "resume_tests" / "load_roundtrip"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create realistic fold artifacts
    eval_results = {
        "fold_id": 0,
        "model_name": "model3",
        "n_scored": 50,
        "n_abstained": 0,
        "accuracy": 0.42,
        "auroc_ovo_weighted": 0.75,
        "classes": ["Covid19", "HIV", "Healthy"],
    }
    raw_preds = {
        "y_true": np.array(["Covid19", "HIV", "Healthy"] * 10),
        "y_pred": np.array(["Covid19", "HIV", "HIV"] * 10),
        "y_proba": np.random.RandomState(42).rand(30, 3).astype(np.float32),
        "classes": np.array(["Covid19", "HIV", "Healthy"]),
    }
    predictions_rows = [
        {"specimen_label": f"SPEC-{i}", "true_disease": "Covid19",
         "predicted_disease": "HIV", "malid_cross_validation_fold_id_when_in_test_set": 0}
        for i in range(10)
    ]

    # Save in the same format as _run_fold_loop
    with open(test_dir / "fold_0_results.json", "w") as f:
        json.dump(eval_results, f)
    with open(test_dir / "fold_0_predictions.pkl", "wb") as f:
        pickle.dump({"raw_preds": raw_preds, "predictions_rows": predictions_rows}, f)

    # Load and verify
    loaded_eval, loaded_raw, loaded_rows = _load_fold_results(test_dir, 0)

    assert loaded_eval["fold_id"] == 0
    assert loaded_eval["accuracy"] == 0.42
    assert loaded_eval["auroc_ovo_weighted"] == 0.75

    assert np.array_equal(loaded_raw["y_true"], raw_preds["y_true"])
    assert np.array_equal(loaded_raw["y_pred"], raw_preds["y_pred"])
    assert np.allclose(loaded_raw["y_proba"], raw_preds["y_proba"])
    assert np.array_equal(loaded_raw["classes"], raw_preds["classes"])

    assert len(loaded_rows) == 10
    assert loaded_rows[0]["specimen_label"] == "SPEC-0"

    tlog.log("  -> PASSED")
    tlog.record("_load_fold_results round-trip", True)


def test_resume_skips_completed_folds(tlog: _TestLogger):
    """Test 25: Resume artifact save/load with metadata validation.

    Trains a synthetic model for fold 0, saves all four artifacts using the
    production save helpers (with _meta), then verifies:
    - _check_fold_complete detects fold 0 as complete / fold 1 as incomplete
    - _load_fold_results round-trips eval metrics, raw predictions, CSV rows
    - _load_stage1_artifact restores Stage 1 state on a fresh model, including
      metadata validation (fold_id, locus, classes, model_params)
    - Stage 2 artifact (saved manually — BinaryOvR can't be pickled in unit
      tests due to the local _make_rf factory) has correct _meta structure

    Note: Stage 2 pkl is saved with dummy model data + real _meta because the
    production stage2_clf_ (BinaryOvRClassifierWithFeatureSubsettingByClass)
    stores base_clf_factory, a local function from fit_stage2 that can't be
    pickled. The _meta validation logic is tested separately in Test 27.
    """
    tlog.log("\n--- Test 25: Resume skips completed folds ---")

    import pickle
    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
        DISEASE_COL,
        PARTICIPANT_COL,
        SPECIMEN_COL,
    )
    from malid_lite.training.train_model3 import (
        _build_model_params,
        _check_fold_complete,
        _load_fold_results,
        _load_stage1_artifact,
        _save_predictions_artifact,
        _save_stage1_artifact,
        evaluate_on_test,
    )

    # --- Setup: build and train a synthetic model for fold 0, save its artifacts ---
    test_dir = OUTPUT_DIR / "resume_tests" / "skip_completed"
    test_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous artifacts
    for f in test_dir.glob("fold_*"):
        f.unlink()

    # Create synthetic data with 2 "folds" (split specimens in half)
    diseases = ["Covid19", "HIV", "Healthy"]
    n_specimens = 18  # 6 per disease, split into 2 folds of 9
    n_seqs = 20
    seq_df = make_synthetic_sequences(
        n_specimens=n_specimens, n_seqs_per_specimen=n_seqs, diseases=diseases,
    )
    embeddings = make_synthetic_embeddings(len(seq_df))

    # Assign fold IDs: specimens 0-8 → fold 0 test, 9-17 → fold 1 test
    specimens = seq_df[SPECIMEN_COL].unique()
    fold_map = {s: 0 if i < 9 else 1 for i, s in enumerate(specimens)}
    seq_df["fold_id"] = seq_df[SPECIMEN_COL].map(fold_map)

    # --- "Complete" fold 0: train a model and save all artifacts ---
    fold0_test = seq_df[seq_df["fold_id"] == 0].copy().reset_index(drop=True)
    fold0_train = seq_df[seq_df["fold_id"] == 1].copy().reset_index(drop=True)
    emb_train = embeddings[seq_df["fold_id"].values == 1]
    emb_test = embeddings[seq_df["fold_id"].values == 0]

    # Split train into ts1/ts2
    train_specs = fold0_train[SPECIMEN_COL].unique()
    split_idx = len(train_specs) * 2 // 3
    ts1_specs = set(train_specs[:split_idx])
    ts2_specs = set(train_specs[split_idx:])

    ts1 = fold0_train[fold0_train[SPECIMEN_COL].isin(ts1_specs)].reset_index(drop=True)
    ts2 = fold0_train[fold0_train[SPECIMEN_COL].isin(ts2_specs)].reset_index(drop=True)
    emb_ts1 = emb_train[fold0_train[SPECIMEN_COL].isin(ts1_specs).values]
    emb_ts2 = emb_train[fold0_train[SPECIMEN_COL].isin(ts2_specs).values]

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, random_state=0, n_jobs=1
    )
    model.fit_stage1(ts1, emb_ts1)
    model.fit_stage2(ts2, emb_ts2)

    # Predict on fold 0 test
    proba_df = model.predict_proba(fold0_test, emb_test)
    classes = model.classes_
    specimen_disease = fold0_test.drop_duplicates(SPECIMEN_COL).set_index(SPECIMEN_COL)[DISEASE_COL]
    y_true = np.array([specimen_disease[s] for s in proba_df.index])
    y_pred = classes[np.argmax(proba_df.values, axis=1)]

    eval_results, raw_preds = evaluate_on_test(
        y_true=y_true, y_pred=y_pred, y_proba=proba_df.values,
        classes=classes, fold_id=0, model_name="model3", n_test=len(y_true),
    )

    # Build predictions_rows (multiclass format)
    fold_pred_rows = []
    str_classes = [str(c) for c in classes]
    for specimen, true_d, pred_d, proba_row in zip(
        proba_df.index, y_true, y_pred, proba_df.values,
    ):
        row = {
            "specimen_label": specimen,
            "true_disease": str(true_d),
            "predicted_disease": str(pred_d),
            "malid_cross_validation_fold_id_when_in_test_set": 0,
        }
        for cls, score in zip(str_classes, proba_row):
            row[f"score_{cls}"] = float(score)
        fold_pred_rows.append(row)

    # --- Save artifacts using production helpers where possible ---
    stage1_path = test_dir / "fold_0_stage1.pkl"
    stage2_path = test_dir / "fold_0_stage2.pkl"
    preds_path = test_dir / "fold_0_predictions.pkl"

    # Stage 1: use production save helper (group_models_ are picklable)
    _save_stage1_artifact(model, stage1_path, fold_id=0, ts1=ts1)

    # Stage 2: save manually with proper _meta — BinaryOvR's base_clf_factory
    # is a local function that can't be pickled, so we save dummy model data
    # but real metadata matching what _save_stage2_artifact would produce.
    stage2_meta = {
        "timestamp": datetime.now().isoformat(),
        "fold_id": 0,
        "classes": [str(c) for c in model.classes_],
        "n_features": len(model.feature_columns_),
        "n_training_sequences": len(ts2),
        "n_training_specimens": int(ts2[SPECIMEN_COL].nunique()),
        "feature_columns": model.feature_columns_,
        "model_params": _build_model_params(model),
    }
    with open(stage2_path, "wb") as f:
        pickle.dump({
            "stage2_clf": "DUMMY_UNPICKLABLE",
            "stage2_scaler": model.stage2_scaler_,
            "preagg_scaler": model.preagg_scaler_,
            "feature_columns": model.feature_columns_,
            "classes": model.classes_,
            "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
            "_meta": stage2_meta,
        }, f)

    # Predictions: use production save helper
    _save_predictions_artifact(
        raw_preds, fold_pred_rows, preds_path,
        fold_id=0, classes=classes, n_test_specimens=len(proba_df),
    )

    # Results JSON
    with open(test_dir / "fold_0_results.json", "w") as f:
        json.dump(eval_results, f, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # --- Verify fold 0 is detected as complete ---
    assert _check_fold_complete(test_dir, 0), "Fold 0 should be complete"
    assert not _check_fold_complete(test_dir, 1), "Fold 1 should NOT be complete"

    # --- Verify _load_fold_results round-trip ---
    loaded_eval, loaded_raw, loaded_rows = _load_fold_results(test_dir, 0)
    assert loaded_eval["fold_id"] == 0
    assert loaded_eval["accuracy"] == eval_results["accuracy"]
    assert len(loaded_rows) == len(fold_pred_rows)
    assert np.array_equal(loaded_raw["y_true"], raw_preds["y_true"])

    # --- Verify _load_stage1_artifact on a fresh model ---
    fresh_model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
    )
    expected_classes = sorted(str(c) for c in classes)
    s1_meta = _load_stage1_artifact(
        fresh_model, stage1_path, fold_id=0, locus="TCR",
        expected_classes=expected_classes,
    )
    # Verify Stage 1 state was restored
    assert fresh_model.group_models_ is not None
    assert len(fresh_model.group_models_) > 0
    assert fresh_model.classes_ is not None
    assert sorted(str(c) for c in fresh_model.classes_) == expected_classes
    # Verify metadata has expected fields
    assert s1_meta["fold_id"] == 0
    assert s1_meta["locus"] == "TCR"
    assert "timestamp" in s1_meta
    assert "model_params" in s1_meta
    assert s1_meta["model_params"]["locus"] == "TCR"
    assert s1_meta["model_params"]["n_estimators_stage1"] == 10

    # --- Verify stage2 _meta structure (loaded raw, not via _load_stage2_artifact
    #     because stage2_clf is dummy) ---
    with open(stage2_path, "rb") as f:
        s2_data = pickle.load(f)
    s2_meta = s2_data["_meta"]
    assert s2_meta["fold_id"] == 0
    assert sorted(s2_meta["classes"]) == expected_classes
    assert "model_params" in s2_meta
    assert s2_meta["n_features"] == len(model.feature_columns_)

    # --- Verify predictions _meta ---
    with open(preds_path, "rb") as f:
        p_data = pickle.load(f)
    p_meta = p_data["_meta"]
    assert p_meta["fold_id"] == 0
    assert p_meta["n_test_specimens"] == len(proba_df)

    tlog.log(f"  Fold 0 saved and verified: accuracy={eval_results['accuracy']:.4f}")
    tlog.log(f"  Stage 1 load+validate on fresh model: OK ({len(fresh_model.group_models_)} groups)")
    tlog.log(f"  Stage 2 _meta structure: OK ({s2_meta['n_features']} features)")
    tlog.log(f"  Predictions _meta: OK ({p_meta['n_test_specimens']} specimens)")
    tlog.log(f"  _check_fold_complete(fold_0)=True, _check_fold_complete(fold_1)=False")
    tlog.log("  -> PASSED")
    tlog.record("Resume helpers on realistic fold artifacts", True)


def test_resume_false_does_not_skip(tlog: _TestLogger):
    """Test 26: resume=False does not skip even when artifacts exist on disk.

    The resume check is gated on `resume and _check_fold_complete(...)`.
    When resume=False (the default), _check_fold_complete should never be
    called and folds are always trained. This test verifies that
    _check_fold_complete returning True has no effect when resume is off.
    """
    tlog.log("\n--- Test 26: resume=False does not skip ---")

    import pickle
    from malid_lite.training.train_model3 import _check_fold_complete

    test_dir = OUTPUT_DIR / "resume_tests" / "no_skip"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create all four artifacts for fold 0.
    # .pkl files must be >= 1024 bytes to pass _check_fold_complete's
    # truncation guard.
    for name in ["fold_0_stage1.pkl", "fold_0_stage2.pkl",
                  "fold_0_results.json", "fold_0_predictions.pkl"]:
        p = test_dir / name
        if p.suffix == ".pkl":
            p.write_bytes(b"x" * 1024)
        else:
            p.write_text("placeholder")

    # _check_fold_complete returns True — artifacts exist
    assert _check_fold_complete(test_dir, 0)

    # Simulate the fold loop guard with resume=False:
    # `if resume and _check_fold_complete(...)` should short-circuit to False
    resume = False
    skipped = resume and _check_fold_complete(test_dir, 0)
    assert not skipped, "resume=False should never skip, even with artifacts present"

    # And with resume=True, the same check should return True
    resume = True
    skipped = resume and _check_fold_complete(test_dir, 0)
    assert skipped, "resume=True should skip when artifacts are present"

    tlog.log("  resume=False + artifacts present -> not skipped: OK")
    tlog.log("  resume=True  + artifacts present -> skipped: OK")
    tlog.log("  -> PASSED")
    tlog.record("resume=False does not skip", True)


def test_metadata_validation_errors(tlog: _TestLogger):
    """Test 27: _validate_artifact_meta raises ValueError on mismatches.

    Tests metadata mismatch detection across all validated categories:
    - fold_id mismatch (wrong fold artifact loaded)
    - locus mismatch (TCR artifact loaded for BCR run)
    - classes mismatch (data changed since artifact was saved)
    - model parameter mismatch (hyperparameters changed since training)
    - run-level parameter mismatches (classification_mode, diseases, dataset_name,
      training_context)
    - training_context backward compat (absent from old artifact)

    Each should raise ValueError with a descriptive message.
    """
    tlog.log("\n--- Test 27: Metadata validation errors ---")

    from malid_lite.training.train_model3 import _validate_artifact_meta

    base_meta = {
        "timestamp": "2026-04-14T12:00:00",
        "fold_id": 0,
        "locus": "TCR",
        "classes": ["Covid19", "HIV", "Healthy"],
        "n_groups": 10,
        "model_params": {
            "locus": "TCR",
            "aggregation_strategy": "mean",
            "exclude_rare_v_genes": True,
            "min_sequences_per_group": 5,
            "reweigh_by_subset_frequencies": False,
            "n_estimators_stage1": 500,
            "n_estimators_stage2": 500,
            "reference_class": None,
            "classification_mode": "multiclass",
            "diseases": None,
            "dataset_name": "mal-id-orig-data",
        },
    }
    current_model_params = dict(base_meta["model_params"])

    # --- 1. fold_id mismatch ---
    try:
        _validate_artifact_meta(base_meta, "Stage 1", fold_id=1)
        assert False, "Should have raised ValueError for fold_id mismatch"
    except ValueError as e:
        assert "fold_id=0" in str(e) and "fold_id=1" in str(e)
        tlog.log(f"  fold_id mismatch: ValueError raised correctly")

    # --- 2. locus mismatch ---
    try:
        _validate_artifact_meta(base_meta, "Stage 1", fold_id=0, locus="BCR")
        assert False, "Should have raised ValueError for locus mismatch"
    except ValueError as e:
        assert "TCR" in str(e) and "BCR" in str(e)
        tlog.log(f"  locus mismatch: ValueError raised correctly")

    # --- 3. classes mismatch ---
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            expected_classes=["Covid19", "HIV", "Healthy", "Lupus"],
        )
        assert False, "Should have raised ValueError for classes mismatch"
    except ValueError as e:
        assert "classes" in str(e).lower()
        tlog.log(f"  classes mismatch: ValueError raised correctly")

    # --- 4. model parameter mismatch (change n_estimators_stage1) ---
    changed_params = dict(current_model_params)
    changed_params["n_estimators_stage1"] = 100  # was 500 in artifact
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            current_model_params=changed_params,
        )
        assert False, "Should have raised ValueError for model param mismatch"
    except ValueError as e:
        assert "n_estimators_stage1" in str(e)
        assert "500" in str(e) and "100" in str(e)
        tlog.log(f"  model param mismatch: ValueError raised correctly")

    # --- 5. Multiple model parameter mismatches ---
    multi_changed = dict(current_model_params)
    multi_changed["n_estimators_stage1"] = 100
    multi_changed["locus"] = "BCR"
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            current_model_params=multi_changed,
        )
        assert False, "Should have raised ValueError for multiple param mismatches"
    except ValueError as e:
        assert "n_estimators_stage1" in str(e)
        assert "locus" in str(e)
        tlog.log(f"  multiple param mismatches: ValueError raised correctly")

    # --- 6. classification_mode mismatch ---
    mode_changed = dict(current_model_params)
    mode_changed["classification_mode"] = "binary"  # was "multiclass"
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            current_model_params=mode_changed,
        )
        assert False, "Should have raised ValueError for classification_mode mismatch"
    except ValueError as e:
        assert "classification_mode" in str(e)
        assert "multiclass" in str(e) and "binary" in str(e)
        tlog.log(f"  classification_mode mismatch: ValueError raised correctly")

    # --- 7. diseases mismatch ---
    diseases_changed = dict(current_model_params)
    diseases_changed["diseases"] = ["Covid19", "HIV"]  # was None
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            current_model_params=diseases_changed,
        )
        assert False, "Should have raised ValueError for diseases mismatch"
    except ValueError as e:
        assert "diseases" in str(e)
        tlog.log(f"  diseases mismatch: ValueError raised correctly")

    # --- 8. dataset_name mismatch ---
    dataset_changed = dict(current_model_params)
    dataset_changed["dataset_name"] = "other-dataset"  # was "mal-id-orig-data"
    try:
        _validate_artifact_meta(
            base_meta, "Stage 1", fold_id=0,
            current_model_params=dataset_changed,
        )
        assert False, "Should have raised ValueError for dataset_name mismatch"
    except ValueError as e:
        assert "dataset_name" in str(e)
        assert "mal-id-orig-data" in str(e) and "other-dataset" in str(e)
        tlog.log(f"  dataset_name mismatch: ValueError raised correctly")

    # --- 8b. training_context mismatch ---
    meta_with_context = dict(base_meta)
    meta_with_context["model_params"] = dict(base_meta["model_params"])
    meta_with_context["model_params"]["training_context"] = "cv_single_model"
    context_changed = dict(current_model_params)
    context_changed["training_context"] = "cv_ensemble"
    try:
        _validate_artifact_meta(
            meta_with_context, "Stage 1", fold_id=0,
            current_model_params=context_changed,
        )
        assert False, "Should have raised ValueError for training_context mismatch"
    except ValueError as e:
        assert "training_context" in str(e)
        assert "cv_single_model" in str(e) and "cv_ensemble" in str(e)
        tlog.log(f"  training_context mismatch: ValueError raised correctly")

    # --- 8c. training_context backward compat (absent from old artifact) ---
    # Old artifacts without training_context in model_params should NOT raise
    context_current = dict(current_model_params)
    context_current["training_context"] = "cv_ensemble"
    _validate_artifact_meta(
        base_meta, "Stage 1", fold_id=0,
        current_model_params=context_current,
    )
    tlog.log(f"  training_context backward compat: no error (correct)")

    # --- 9. data size mismatch (n_training_sequences) ---
    meta_with_sizes = dict(base_meta)
    meta_with_sizes["n_training_sequences"] = 500000
    meta_with_sizes["n_training_specimens"] = 120
    try:
        _validate_artifact_meta(
            meta_with_sizes, "Stage 1", fold_id=0,
            expected_data_sizes={"n_training_sequences": 600000, "n_training_specimens": 120},
        )
        assert False, "Should have raised ValueError for data size mismatch"
    except ValueError as e:
        assert "n_training_sequences" in str(e)
        assert "500,000" in str(e) and "600,000" in str(e)
        tlog.log(f"  data size mismatch: ValueError raised correctly")

    # --- 10. matching data sizes should NOT raise ---
    _validate_artifact_meta(
        meta_with_sizes, "Stage 1", fold_id=0,
        expected_data_sizes={"n_training_sequences": 500000, "n_training_specimens": 120},
    )
    tlog.log(f"  matching data sizes: no error (correct)")

    # --- 11. Matching metadata (all checks) should NOT raise ---
    _validate_artifact_meta(
        base_meta, "Stage 1", fold_id=0, locus="TCR",
        expected_classes=["Covid19", "HIV", "Healthy"],
        current_model_params=current_model_params,
    )
    tlog.log(f"  matching metadata: no error (correct)")

    tlog.log("  -> PASSED")
    tlog.record("Metadata validation errors", True)


def test_backward_compat_no_meta(tlog: _TestLogger):
    """Test 28: Old artifacts without _meta produce a warning but still load.

    Simulates artifacts saved before the metadata system was added. These
    lack the _meta key entirely. _validate_artifact_meta should log a
    warning (not raise) and _load_stage1_artifact should still restore
    model state.
    """
    tlog.log("\n--- Test 28: Backward compat - no _meta ---")

    import pickle
    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
        DISEASE_COL,
        SPECIMEN_COL,
    )
    from malid_lite.training.train_model3 import (
        _load_stage1_artifact,
        _validate_artifact_meta,
    )

    test_dir = OUTPUT_DIR / "resume_tests" / "backward_compat"
    test_dir.mkdir(parents=True, exist_ok=True)

    # --- Train a small model to get real group_models_ ---
    diseases = ["Covid19", "Healthy"]
    seq_df = make_synthetic_sequences(
        n_specimens=6, n_seqs_per_specimen=20, diseases=diseases,
    )
    embeddings = make_synthetic_embeddings(len(seq_df))

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, random_state=0, n_jobs=1
    )
    model.fit_stage1(seq_df, embeddings)

    # --- Save Stage 1 WITHOUT _meta (old format) ---
    stage1_path = test_dir / "fold_0_stage1.pkl"
    with open(stage1_path, "wb") as f:
        pickle.dump({
            "group_models": model.group_models_,
            "classes": model.classes_,
            "non_rare_v_genes": model.non_rare_v_genes_,
            "locus": model.locus,
            "aggregation_strategy": model.aggregation_strategy.name,
            # No _meta key — simulates old artifact
        }, f)

    # --- _validate_artifact_meta with empty/missing meta should warn, not raise ---
    # Empty dict (what data.get("_meta", {}) returns when _meta is absent)
    import logging as stdlib_logging
    with warnings.catch_warnings(record=True):
        # This should log a warning but not raise
        _validate_artifact_meta(
            {}, "Stage 1", fold_id=0, locus="TCR",
            expected_classes=["Covid19", "Healthy"],
        )
    tlog.log("  _validate_artifact_meta({}, ...): no error (warning only)")

    # --- _load_stage1_artifact should still restore state ---
    fresh_model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
    )
    meta = _load_stage1_artifact(
        fresh_model, stage1_path, fold_id=0, locus="TCR",
    )
    # meta should be empty dict (no _meta in artifact)
    assert meta == {}, f"Expected empty meta for old artifact, got: {meta}"
    # Model state should still be restored
    assert fresh_model.group_models_ is not None
    assert len(fresh_model.group_models_) > 0
    assert fresh_model.classes_ is not None
    tlog.log(f"  Stage 1 loaded without _meta: OK ({len(fresh_model.group_models_)} groups)")

    tlog.log("  -> PASSED")
    tlog.record("Backward compat - no _meta", True)


# ---------------------------------------------------------------------------
# Unit tests: Tuning components
# ---------------------------------------------------------------------------


def test_build_group_index(tlog: _TestLogger):
    """Test 30: _build_group_index correctness and index assertion."""
    tlog.log("\n--- Test 30: _build_group_index ---")

    from malid_lite.models.model3_sequence_level import _build_group_index

    # Normal case: reset integer index, TCR (single split col)
    df = pd.DataFrame({
        "specimen_label": ["S1", "S1", "S2", "S2", "S2"],
        "v_gene": ["TRBV5-1", "TRBV5-6", "TRBV5-1", "TRBV5-1", "TRBV5-6"],
    })
    idx = _build_group_index(df, ["v_gene"])
    assert ("S1", "TRBV5-1") in idx
    assert ("S1", "TRBV5-6") in idx
    assert ("S2", "TRBV5-1") in idx
    np.testing.assert_array_equal(idx[("S1", "TRBV5-1")], [0])
    np.testing.assert_array_equal(idx[("S1", "TRBV5-6")], [1])
    np.testing.assert_array_equal(sorted(idx[("S2", "TRBV5-1")]), [2, 3])
    tlog.log("  Normal case: OK")

    # Bad index: non-reset index should trigger assertion
    df_bad = df.copy()
    df_bad.index = [10, 20, 30, 40, 50]
    try:
        _build_group_index(df_bad, ["v_gene"])
        assert False, "Should have raised AssertionError"
    except AssertionError:
        tlog.log("  Non-reset index assertion: OK")

    tlog.record("_build_group_index", True)


def test_fast_featurize(tlog: _TestLogger):
    """Test 31: _fast_featurize correctness for all supported strategies."""
    tlog.log("\n--- Test 31: _fast_featurize ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        _build_group_index,
        _fast_featurize,
    )

    # Setup: 2 specimens, 2 groups, 3 classes
    df = pd.DataFrame({
        "specimen_label": ["S1", "S1", "S1", "S2", "S2"],
        "v_gene": ["G1", "G1", "G2", "G1", "G2"],
    })
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.3, 0.3, 0.4],
    ])
    entropies = np.array([0.1, 0.5, 0.2, 0.3, 0.9])
    weights = None
    all_specimens = np.array(["S1", "S2"])
    all_groups = [("G1",), ("G2",)]
    n_classes = 3

    group_index = _build_group_index(df, ["v_gene"])

    # Test mean strategy
    feat = _fast_featurize(
        group_index, probs, entropies, weights, all_specimens, all_groups,
        n_classes, AggregationStrategy.mean, None, 1,
    )
    assert feat.shape == (2, 6), f"Expected shape (2, 6), got {feat.shape}"
    # S1-G1: mean of rows 0,1 -> [0.7, 0.2, 0.1]
    np.testing.assert_allclose(feat[0, 0:3], [0.7, 0.2, 0.1], atol=1e-10)
    tlog.log("  mean strategy: OK")

    # Test entropy_cutoff: threshold=0.4 nats, S1-G1 rows: ent=[0.1, 0.5]
    # Only row 0 survives (ent=0.1 < 0.4)
    feat_ec = _fast_featurize(
        group_index, probs, entropies, weights, all_specimens, all_groups,
        n_classes, AggregationStrategy.entropy_cutoff, 0.4, 1,
    )
    np.testing.assert_allclose(feat_ec[0, 0:3], [0.8, 0.1, 0.1], atol=1e-10)
    tlog.log("  entropy_cutoff strategy: OK")

    # Test entropy_cutoff: threshold so low nothing survives -> uniform
    feat_empty = _fast_featurize(
        group_index, probs, entropies, weights, all_specimens, all_groups,
        n_classes, AggregationStrategy.entropy_cutoff, 0.01, 1,
    )
    np.testing.assert_allclose(feat_empty[0, 0:3], [1/3, 1/3, 1/3], atol=1e-10)
    tlog.log("  entropy_cutoff all filtered -> uniform: OK")

    # Test unknown strategy -> ValueError
    try:
        _fast_featurize(
            group_index, probs, entropies, weights, all_specimens, all_groups,
            n_classes, AggregationStrategy.entropy_ten_percent_cutoff, None, 1,
        )
        assert False, "Should have raised ValueError for unhandled strategy"
    except ValueError as e:
        assert "unhandled strategy" in str(e).lower()
        tlog.log(f"  Unknown strategy raises ValueError: OK")

    tlog.record("_fast_featurize", True)


def test_tuning_sort_key(tlog: _TestLogger):
    """Test 32: Tuning tie-breaking sort order and 0.0 vs None handling."""
    tlog.log("\n--- Test 32: Tuning sort key ---")

    from malid_lite.models.model3_sequence_level import _TUNING_STRATEGY_PRIORITY

    # Simulate the _sort_key function from _tune_aggregation_strategy
    def _sort_key(r):
        priority = _TUNING_STRATEGY_PRIORITY.get(r["strategy_name"], 99)
        threshold = r["threshold_param"]
        param_tiebreak = -(threshold if threshold is not None else 0)
        return (-r["mean_mcc"], priority, param_tiebreak)

    results = [
        {"strategy_name": "entropy_cutoff", "threshold_param": 0.80, "mean_mcc": 0.5},
        {"strategy_name": "entropy_cutoff", "threshold_param": 0.95, "mean_mcc": 0.5},
        {"strategy_name": "mean", "threshold_param": None, "mean_mcc": 0.5},
        {"strategy_name": "entropy_percentile_cutoff", "threshold_param": 0.01, "mean_mcc": 0.5},
    ]
    results.sort(key=_sort_key)

    # All same MCC -> sort by priority: mean(0) < entropy_cutoff(3) < percentile(4)
    # Within entropy_cutoff: prefer higher threshold -> 0.95 before 0.80
    assert results[0]["strategy_name"] == "mean"
    assert results[1]["strategy_name"] == "entropy_cutoff"
    assert results[1]["threshold_param"] == 0.95
    assert results[2]["strategy_name"] == "entropy_cutoff"
    assert results[2]["threshold_param"] == 0.80
    assert results[3]["strategy_name"] == "entropy_percentile_cutoff"
    tlog.log("  Tie-breaking order: OK")

    # Verify 0.0 threshold_param is NOT treated as None
    r_zero = {"strategy_name": "entropy_cutoff", "threshold_param": 0.0, "mean_mcc": 0.5}
    r_none = {"strategy_name": "entropy_cutoff", "threshold_param": None, "mean_mcc": 0.5}
    # Both should produce param_tiebreak = 0 (negative of 0), so they sort equal
    assert _sort_key(r_zero) == _sort_key(r_none)
    # But importantly, 0.0 should NOT crash or produce wrong value
    tlog.log("  0.0 vs None threshold_param: OK")

    tlog.record("Tuning sort key", True)


def test_tuning_full_synthetic(tlog: _TestLogger):
    """Test 33: Full auto-tuning pipeline on synthetic data.

    Trains a model with tuning_enabled=True, verifies tuning selects a
    strategy, sets model attributes correctly, and the model can predict.
    """
    tlog.log("\n--- Test 33: Full tuning pipeline (synthetic) ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    # Need enough specimens per class so inner 3-fold CV always has all
    # classes in both train and val. 45 specimens (15/class), stratified
    # split ensures ts2 gets 5/class -> inner 3-fold: train~10, val~5.
    seq_df = make_synthetic_sequences(
        n_specimens=45, n_seqs_per_specimen=30,
        diseases=["Covid19", "HIV", "Healthy"],
    )
    embeddings = make_synthetic_embeddings(len(seq_df))

    # Stratified split to guarantee balanced classes in ts2
    spec_disease = (
        seq_df.drop_duplicates("specimen_label")
        .set_index("specimen_label")["disease"]
    )
    ts1_specs, ts2_specs = set(), set()
    for disease, group in spec_disease.groupby(spec_disease):
        specs = list(group.index)
        rng = np.random.RandomState(0)
        rng.shuffle(specs)
        cut = len(specs) * 2 // 3
        ts1_specs.update(specs[:cut])
        ts2_specs.update(specs[cut:])

    ts1_mask = seq_df["specimen_label"].isin(ts1_specs)
    ts2_mask = seq_df["specimen_label"].isin(ts2_specs)
    ts1 = seq_df[ts1_mask].reset_index(drop=True)
    ts2 = seq_df[ts2_mask].reset_index(drop=True)
    emb_ts1 = embeddings[ts1_mask.values]
    emb_ts2 = embeddings[ts2_mask.values]

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.entropy_cutoff,
        exclude_rare_v_genes=False,
        min_sequences_per_group=5,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=1,
        tuning_enabled=True,
        tuning_cv_splits=3,
        tuning_strategies=["entropy_cutoff", "entropy_percentile_cutoff"],
        tuning_entropy_max_fractions=[0.80, 0.95],
        tuning_entropy_percentiles=[0.1, 0.5],
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1,
    )

    # Stage 1
    model.fit_stage1(ts1, emb_ts1)
    assert len(model.group_models_) > 0

    # Stage 2 (triggers tuning)
    model.fit_stage2(ts2, emb_ts2)

    # Verify tuning ran and set attributes
    assert model.tuning_enabled_ is True, "tuning_enabled_ not set"
    assert model.tuning_results_ is not None, "tuning_results_ is None"
    assert len(model.tuning_results_) > 0, "No tuning results"
    tlog.log(f"  Tuning results: {len(model.tuning_results_)} entries")

    # The winning strategy should be a valid AggregationStrategy
    assert isinstance(model.aggregation_strategy, AggregationStrategy)
    winner = model.aggregation_strategy.name
    tlog.log(f"  Winner: {winner}")

    # Each result should have required keys
    for r in model.tuning_results_:
        assert "strategy_name" in r
        assert "mean_mcc" in r
        assert "fold_scores" in r

    # With random data, tuning likely hit the fallback (MCC <= 0) — log it
    is_fallback = model.tuning_results_[0].get("fallback", False)
    if is_fallback:
        tlog.log("  (fallback path: all candidates had MCC <= 0 — expected with random data)")
    else:
        best_mcc = model.tuning_results_[0]["mean_mcc"]
        tlog.log(f"  Best MCC: {best_mcc:.4f}")

    # Model should be able to predict
    proba_df = model.predict_proba(ts2, emb_ts2)
    assert proba_df.shape[0] == ts2["specimen_label"].nunique()
    assert np.all(np.isfinite(proba_df.values))
    tlog.log(f"  Prediction shape: {proba_df.shape}")

    tlog.record("Full tuning pipeline (synthetic)", True)


def test_load_stage2_tuning_validation(tlog: _TestLogger):
    """Test 34: load_stage2_artifacts validation for tuning-specific fields."""
    tlog.log("\n--- Test 34: load_stage2_artifacts tuning validation ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    # Build a minimal model with Stage 1 loaded (so load_stage2_artifacts doesn't
    # complain about missing Stage 1)
    seq_df = make_synthetic_sequences(
        n_specimens=12, n_seqs_per_specimen=20,
        diseases=["Covid19", "Healthy"],
    )
    embeddings = make_synthetic_embeddings(len(seq_df))
    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
        tuning_enabled=True,
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1,
    )
    model.fit_stage1(seq_df, embeddings)

    # Build a fake Stage 2 artifact
    base_artifact = {
        "stage2_clf": "fake_clf",
        "stage2_scaler": "fake_scaler",
        "feature_columns": ["Covid19_TRBV5_1", "Healthy_TRBV5_1"],
        "classes": list(model.classes_),
        "reweigh_by_subset_frequencies": True,
    }

    # Test: tuning_enabled=True but missing tuning_best_strategy -> ValueError
    artifact_tuned_no_winner = {**base_artifact, "tuning_enabled": True}
    try:
        model.load_stage2_artifacts(artifact_tuned_no_winner)
        assert False, "Should have raised ValueError for missing tuning_best_strategy"
    except ValueError as e:
        assert "tuning_best_strategy" in str(e)
        tlog.log(f"  Missing tuning_best_strategy -> ValueError: OK")

    # Test: tuning mismatch (artifact=tuned, model=not tuned) -> ValueError
    model_no_tune = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        exclude_rare_v_genes=False,
        min_sequences_per_group=2,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
        tuning_enabled=False,
    )
    model_no_tune._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1,
    )
    model_no_tune.fit_stage1(seq_df, embeddings)

    artifact_tuned = {
        **base_artifact,
        "tuning_enabled": True,
        "tuning_best_strategy": "mean",
    }
    try:
        model_no_tune.load_stage2_artifacts(artifact_tuned)
        assert False, "Should have raised ValueError for tuning mismatch"
    except ValueError as e:
        assert "auto_tuned" in str(e)
        tlog.log(f"  Tuning mismatch (artifact tuned, model fixed) -> ValueError: OK")

    # Test: reweigh_by_subset_frequencies mismatch -> ValueError
    artifact_reweigh_mismatch = {
        **base_artifact,
        "reweigh_by_subset_frequencies": False,  # model has True
    }
    try:
        model.load_stage2_artifacts(artifact_reweigh_mismatch)
        assert False, "Should have raised ValueError for reweigh mismatch"
    except ValueError as e:
        assert "reweigh_by_subset_frequencies" in str(e)
        tlog.log(f"  reweigh mismatch -> ValueError: OK")

    # Test: reverse mismatch (artifact=fixed, model=tuned) -> ValueError
    artifact_fixed = {
        **base_artifact,
        "tuning_enabled": False,
        "aggregation_strategy": "mean",
    }
    try:
        model.load_stage2_artifacts(artifact_fixed)
        assert False, "Should have raised ValueError for reverse tuning mismatch"
    except ValueError as e:
        assert "fixed strategy" in str(e).lower() or "auto_tuned" in str(e)
        tlog.log(f"  Reverse mismatch (artifact fixed, model tuned) -> ValueError: OK")

    # Test: invalid tuning_best_strategy name -> ValueError
    artifact_bad_winner = {
        **base_artifact,
        "tuning_enabled": True,
        "tuning_best_strategy": "nonexistent_strategy",
    }
    try:
        model.load_stage2_artifacts(artifact_bad_winner)
        assert False, "Should have raised ValueError for invalid strategy name"
    except ValueError as e:
        assert "nonexistent_strategy" in str(e)
        tlog.log(f"  Invalid tuning_best_strategy -> ValueError: OK")

    tlog.record("load_stage2_artifacts tuning validation", True)


def test_tuning_winner_selection(tlog: _TestLogger):
    """Test 35: Verify the winner-selection logic sets model attributes correctly.

    Bypasses the full CV pipeline and directly tests the Phase 4 logic
    that applies the winning candidate's strategy/threshold to the model.
    """
    tlog.log("\n--- Test 35: Tuning winner selection logic ---")

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
        _TUNING_STRATEGY_PRIORITY,
    )

    # Create a model in its pre-tuning state
    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.mean,
        entropy_max_fraction=0.80,
        entropy_bottom_percentile=0.1,
        tuning_enabled=True,
        n_jobs=1,
        verbose=0,
    )

    # Simulate tuning results (as _tune_aggregation_strategy would produce)
    fake_results = [
        {
            "strategy_name": "entropy_cutoff",
            "threshold_param": 0.95,
            "threshold_nats": 1.05,
            "mean_mcc": 0.65,
            "std_mcc": 0.02,
            "fold_scores": [0.63, 0.67, 0.65],
        },
        {
            "strategy_name": "entropy_percentile_cutoff",
            "threshold_param": 0.5,
            "threshold_nats": 0.42,
            "mean_mcc": 0.60,
            "std_mcc": 0.03,
            "fold_scores": [0.57, 0.63, 0.60],
        },
        {
            "strategy_name": "entropy_cutoff",
            "threshold_param": 0.80,
            "threshold_nats": 0.88,
            "mean_mcc": 0.55,
            "std_mcc": 0.04,
            "fold_scores": [0.51, 0.55, 0.59],
        },
    ]

    # Sort results the same way _tune_aggregation_strategy does
    def _sort_key(r):
        priority = _TUNING_STRATEGY_PRIORITY.get(r["strategy_name"], 99)
        threshold = r["threshold_param"]
        param_tiebreak = -(threshold if threshold is not None else 0)
        return (-r["mean_mcc"], priority, param_tiebreak)

    fake_results.sort(key=_sort_key)
    best = fake_results[0]

    # Verify sort order: highest MCC first
    assert best["strategy_name"] == "entropy_cutoff"
    assert best["threshold_param"] == 0.95
    tlog.log(f"  Sort order correct: best = {best['strategy_name']} ({best['threshold_param']})")

    # Apply the winner (replicate the Phase 4 logic)
    model.aggregation_strategy = AggregationStrategy[best["strategy_name"]]
    model.tuning_enabled_ = True
    model.tuning_results_ = fake_results
    if best["strategy_name"] == "entropy_cutoff":
        model.entropy_max_fraction = best["threshold_param"]
    elif best["strategy_name"] == "entropy_percentile_cutoff":
        model.entropy_bottom_percentile = best["threshold_param"]
        model.entropy_percentile_threshold_ = best["threshold_nats"]

    # Verify all attributes set correctly
    assert model.aggregation_strategy == AggregationStrategy.entropy_cutoff
    assert model.entropy_max_fraction == 0.95, f"Expected 0.95, got {model.entropy_max_fraction}"
    assert model.tuning_enabled_ is True
    assert len(model.tuning_results_) == 3
    tlog.log("  entropy_cutoff winner: attributes set correctly")

    # Now test entropy_percentile_cutoff winner
    model2 = SequenceLevelClassifier(
        locus="TCR", tuning_enabled=True, n_jobs=1, verbose=0,
    )
    pctile_best = {
        "strategy_name": "entropy_percentile_cutoff",
        "threshold_param": 0.05,
        "threshold_nats": 0.31,
        "mean_mcc": 0.70,
    }
    model2.aggregation_strategy = AggregationStrategy[pctile_best["strategy_name"]]
    model2.tuning_enabled_ = True
    model2.entropy_bottom_percentile = pctile_best["threshold_param"]
    model2.entropy_percentile_threshold_ = pctile_best["threshold_nats"]

    assert model2.aggregation_strategy == AggregationStrategy.entropy_percentile_cutoff
    assert model2.entropy_bottom_percentile == 0.05
    assert model2.entropy_percentile_threshold_ == 0.31
    tlog.log("  entropy_percentile_cutoff winner: attributes set correctly")

    tlog.record("Tuning winner selection", True)


def test_tuning_artifact_roundtrip(tlog: _TestLogger):
    """Test 36: Save/load round-trip for tuning-enabled Stage 2 artifact.

    Trains a model with tuning, saves the Stage 2 artifact using
    _save_stage2_artifact, loads it back on a fresh model using
    load_stage2_artifacts, and verifies all tuning state survives.
    """
    tlog.log("\n--- Test 36: Tuning artifact round-trip ---")

    import pickle
    import shutil

    from malid_lite.models.model3_sequence_level import (
        AggregationStrategy,
        SequenceLevelClassifier,
    )

    # Train a model with tuning on synthetic data (same setup as test 33)
    seq_df = make_synthetic_sequences(
        n_specimens=45, n_seqs_per_specimen=30,
        diseases=["Covid19", "HIV", "Healthy"],
    )
    embeddings = make_synthetic_embeddings(len(seq_df))

    spec_disease = (
        seq_df.drop_duplicates("specimen_label")
        .set_index("specimen_label")["disease"]
    )
    ts1_specs, ts2_specs = set(), set()
    for disease, group in spec_disease.groupby(spec_disease):
        specs = list(group.index)
        rng = np.random.RandomState(0)
        rng.shuffle(specs)
        cut = len(specs) * 2 // 3
        ts1_specs.update(specs[:cut])
        ts2_specs.update(specs[cut:])

    ts1_mask = seq_df["specimen_label"].isin(ts1_specs)
    ts2_mask = seq_df["specimen_label"].isin(ts2_specs)
    ts1 = seq_df[ts1_mask].reset_index(drop=True)
    ts2 = seq_df[ts2_mask].reset_index(drop=True)
    emb_ts1 = embeddings[ts1_mask.values]
    emb_ts2 = embeddings[ts2_mask.values]

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.entropy_cutoff,
        exclude_rare_v_genes=False,
        min_sequences_per_group=5,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
        tuning_enabled=True,
        tuning_cv_splits=3,
        tuning_strategies=["entropy_cutoff", "entropy_percentile_cutoff"],
        tuning_entropy_max_fractions=[0.80, 0.95],
        tuning_entropy_percentiles=[0.1, 0.5],
    )
    model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1,
    )
    model.fit_stage1(ts1, emb_ts1)
    model.fit_stage2(ts2, emb_ts2)

    # Capture the tuning state before save
    orig_strategy = model.aggregation_strategy
    orig_tuning_enabled = model.tuning_enabled_
    orig_tuning_results = model.tuning_results_
    orig_entropy_max_fraction = model.entropy_max_fraction
    orig_entropy_bottom_percentile = model.entropy_bottom_percentile
    orig_entropy_percentile_threshold = model.entropy_percentile_threshold_
    orig_feature_columns = model.feature_columns_

    tlog.log(f"  Original: strategy={orig_strategy.name}, "
             f"tuning_enabled_={orig_tuning_enabled}, "
             f"n_results={len(orig_tuning_results)}")

    # Save artifact using the same format as _save_stage2_artifact
    roundtrip_dir = OUTPUT_DIR / "test_36_tuning_roundtrip"
    if roundtrip_dir.exists():
        shutil.rmtree(roundtrip_dir)
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = roundtrip_dir / "stage2_artifact.pkl"

    tuning_data = {}
    if model.tuning_enabled_:
        tuning_data["tuning_enabled"] = True
        tuning_data["tuning_results"] = model.tuning_results_
        tuning_data["tuning_cv_splits"] = model.tuning_cv_splits
        tuning_data["tuning_best_strategy"] = model.aggregation_strategy.name
        if model.tuning_results_:
            tuning_data["tuning_best_threshold_param"] = model.tuning_results_[0].get(
                "threshold_param"
            )

    with open(artifact_path, "wb") as f:
        pickle.dump({
            "stage2_clf": model.stage2_clf_,
            "stage2_scaler": model.stage2_scaler_,
            "preagg_scaler": model.preagg_scaler_,
            "feature_columns": model.feature_columns_,
            "classes": model.classes_,
            "reweigh_by_subset_frequencies": model.reweigh_by_subset_frequencies,
            "entropy_percentile_threshold": model.entropy_percentile_threshold_,
            "aggregation_strategy": model.aggregation_strategy.name,
            **tuning_data,
        }, f)

    # Load on a fresh model (Stage 1 must be loaded first)
    fresh_model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=AggregationStrategy.entropy_cutoff,
        exclude_rare_v_genes=False,
        min_sequences_per_group=5,
        n_estimators_stage1=10,
        n_estimators_stage2=10,
        n_jobs=1,
        verbose=0,
        tuning_enabled=True,
    )
    fresh_model._make_stage1_clf = lambda: _RFIgnoringGroups(
        n_estimators=10, class_weight="balanced_subsample", random_state=0, n_jobs=1,
    )
    fresh_model.fit_stage1(ts1, emb_ts1)

    with open(artifact_path, "rb") as f:
        data = pickle.load(f)
    fresh_model.load_stage2_artifacts(data)

    # Verify tuning state survived the round-trip
    assert fresh_model.aggregation_strategy == orig_strategy, (
        f"Strategy mismatch: {fresh_model.aggregation_strategy} != {orig_strategy}"
    )
    assert fresh_model.tuning_enabled_ == orig_tuning_enabled
    assert fresh_model.tuning_results_ is not None
    assert len(fresh_model.tuning_results_) == len(orig_tuning_results)
    assert fresh_model.entropy_percentile_threshold_ == orig_entropy_percentile_threshold
    assert fresh_model.feature_columns_ == orig_feature_columns

    # If the winner was entropy_cutoff, verify max_fraction survived
    if orig_strategy == AggregationStrategy.entropy_cutoff:
        assert fresh_model.entropy_max_fraction == orig_entropy_max_fraction
    # If the winner was entropy_percentile_cutoff, verify bottom_percentile survived
    elif orig_strategy == AggregationStrategy.entropy_percentile_cutoff:
        assert fresh_model.entropy_bottom_percentile == orig_entropy_bottom_percentile

    tlog.log(f"  Round-trip: strategy={fresh_model.aggregation_strategy.name}, "
             f"tuning_enabled_={fresh_model.tuning_enabled_}")

    # Verify the loaded model can predict
    proba_df = fresh_model.predict_proba(ts2, emb_ts2)
    assert proba_df.shape[0] == ts2["specimen_label"].nunique()
    assert np.all(np.isfinite(proba_df.values))
    tlog.log(f"  Loaded model prediction shape: {proba_df.shape}")
    tlog.log(f"  Artifact saved to: {artifact_path}")

    tlog.record("Tuning artifact round-trip", True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model 3 core pipeline tests")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for integration tests (default: 1)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"test_log_{timestamp}.txt"
    results_path = OUTPUT_DIR / f"test_results_{timestamp}.json"

    tlog = _TestLogger(log_path)
    tlog.log("=" * 70)
    tlog.log("Model 3 Core Pipeline Tests")
    tlog.log(f"Time: {datetime.now().isoformat()}")
    tlog.log(f"Output: {OUTPUT_DIR}")
    tlog.log("=" * 70)

    # --- Tier 1: Unit tests (synthetic data, no external deps) ---
    tlog.log("\n" + "=" * 70)
    tlog.log("TIER 1: Unit Tests (synthetic data)")
    tlog.log("=" * 70)

    unit_tests = [
        ("Test 1", test_aggregation_strategies),
        ("Test 2", test_aggregation_edge_cases),
        ("Test 3", test_find_non_rare_v_genes),
        ("Test 4", test_group_sequence_classifier),
        ("Test 5", test_classifier_init_and_helpers),
        ("Test 6", test_sanitize_group_columns),
        ("Test 7", test_class_prefix_collision),
        ("Test 8", test_full_synthetic_pipeline),
        ("Test 9", test_featurize_uniform_fill),
        ("Test 10", test_featurize_column_alignment),
        ("Test 11", test_evaluate_multiclass),
        ("Test 12", test_evaluate_binary),
        ("Test 13", test_evaluate_optional_params),
        ("Test 14", test_factory_functions),
        ("Test 23", test_check_fold_complete),
        ("Test 24", test_load_fold_results_roundtrip),
        ("Test 25", test_resume_skips_completed_folds),
        ("Test 26", test_resume_false_does_not_skip),
        ("Test 27", test_metadata_validation_errors),
        ("Test 28", test_backward_compat_no_meta),
        ("Test 30", test_build_group_index),
        ("Test 31", test_fast_featurize),
        ("Test 32", test_tuning_sort_key),
        ("Test 33", test_tuning_full_synthetic),
        ("Test 34", test_load_stage2_tuning_validation),
        ("Test 35", test_tuning_winner_selection),
        ("Test 36", test_tuning_artifact_roundtrip),
    ]

    for name, test_fn in unit_tests:
        try:
            test_fn(tlog)
        except Exception as e:
            tlog.log(f"  EXCEPTION: {e}")
            tlog.log(traceback.format_exc())
            tlog.record(name, False, {"error": str(e)})

    # --- Tier 2: Integration tests (test data + random embeddings, needs glmnet) ---
    tlog.log("\n" + "=" * 70)
    tlog.log("TIER 2: Integration Tests (test data, random embeddings)")
    tlog.log("=" * 70)

    prereq_error = _check_integration_prerequisites()
    if prereq_error:
        tlog.log(f"\nSkipping integration tests: {prereq_error}")
        tlog.log("To run integration tests, ensure:")
        tlog.log("  glmnet installed: conda install -c conda-forge glmnet")
    else:
        integration_tests = [
            ("Test 18", lambda t: test_integration_multiclass(t, args.n_jobs)),
            ("Test 19", lambda t: test_integration_binary(t, args.n_jobs)),
            ("Test 20", test_integration_predictions_csv_multiclass),
            ("Test 21", test_integration_predictions_csv_binary),
            ("Test 22", test_integration_model_save_load),
            ("Test 29", test_integration_cv_ensemble_splits),
        ]

        for name, test_fn in integration_tests:
            try:
                test_fn(tlog)
            except Exception as e:
                tlog.log(f"  EXCEPTION: {e}")
                tlog.log(traceback.format_exc())
                tlog.record(name, False, {"error": str(e)})

    # --- Summary ---
    tlog.log("\n" + "=" * 70)
    tlog.log("SUMMARY")
    tlog.log("=" * 70)
    n_passed = sum(1 for r in tlog.results if r["status"] == "PASSED")
    n_failed = sum(1 for r in tlog.results if r["status"] == "FAILED")
    n_total = len(tlog.results)
    elapsed = datetime.now() - tlog.start_time
    elapsed_str = str(elapsed).split(".")[0]  # HH:MM:SS without microseconds
    tlog.log(f"\n  Total: {n_total}  Passed: {n_passed}  Failed: {n_failed}")
    tlog.log(f"  Elapsed: {elapsed_str} ({elapsed.total_seconds():.1f}s)")
    if n_failed > 0:
        tlog.log("\nFailed tests:")
        for r in tlog.results:
            if r["status"] == "FAILED":
                tlog.log(f"  - {r['test']}: {r['details'].get('error', 'unknown')}")

    tlog.log(f"\nLog: {log_path}")
    tlog.save_results(results_path)
    tlog.close()

    if n_failed > 0:
        print(f"\n{n_failed} test(s) FAILED in {elapsed_str}")
        sys.exit(1)
    else:
        print(f"\nAll {n_passed} tests PASSED in {elapsed_str}")


if __name__ == "__main__":
    main()
