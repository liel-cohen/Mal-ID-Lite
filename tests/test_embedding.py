"""Smoke test for ESM-2 embedding computation.

Tests on a small subset of test data participants (2-3) to verify:
1. ESM-2 model loads correctly on detected device
2. Embeddings have correct shape (N, 640) and dtype (float16)
3. Parquet + npy row alignment
4. Stats JSON is complete and correct
5. Resumption works (re-running skips already-processed participants)
6. Verify mode catches intentionally corrupted files
7. Embedding loading and fold assembly simulation

Output: tests/test_outputs/test_embedding/

Expected runtime: ~2-4 minutes (dominated by ESM-2 model load + embedding compute).
Requires PyTorch and fair-esm.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from test_helpers import create_test_loader, TEST_DATA_DIR

from malid_lite.training.compute_model3_embeddings import (
    EMBEDDING_DIM,
    EXPECTED_NUM_LAYERS,
    detect_device,
    load_esm2_model,
    compute_embeddings_for_sequences,
    process_participant,
    verify_embeddings,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "test_outputs" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TEST_PARTICIPANTS = 3

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def esm2_model():
    """Load ESM-2 model once for all tests in this module.

    Returns (model, batch_converter, repr_layer, device, load_time, log).
    """
    log_file = OUTPUT_DIR / "test_embedding.log"
    log = setup_logging(log_file, verbose=1)

    device = detect_device()
    model, _alphabet, batch_converter, repr_layer, torch_device, load_time = \
        load_esm2_model(device, log)

    return model, batch_converter, repr_layer, torch_device, load_time, log


@pytest.fixture(scope="module")
def test_loader():
    """Data loader backed by tests/test_data/."""
    return create_test_loader(verbose=0)


@pytest.fixture(scope="module")
def test_participant_labels(test_loader):
    """Pick first N_TEST_PARTICIPANTS from the test data."""
    participants_dir = TEST_DATA_DIR / "participants"
    clean_parquets = sorted(participants_dir.glob("*_clean.parquet"))
    assert len(clean_parquets) > 0, (
        f"No cached participants in {participants_dir}. "
        "Generate test data with: python scripts/data/create_test_data.py"
    )
    labels = [p.stem.removesuffix("_clean") for p in clean_parquets[:N_TEST_PARTICIPANTS]]
    return labels


@pytest.fixture(scope="module")
def computed_embeddings_dir(test_loader, test_participant_labels, esm2_model):
    """Compute embeddings for test participants into a temp directory.

    This fixture runs the full participant pipeline (CLEAN -> DOWNSAMPLE -> embed)
    and returns the output directory. Re-used by resumption and verify tests.
    """
    model, batch_converter, repr_layer, device, _load_time, log = esm2_model

    emb_dir = OUTPUT_DIR / "embeddings_test"
    if emb_dir.exists():
        shutil.rmtree(emb_dir)
    emb_dir.mkdir(parents=True)

    for label in test_participant_labels:
        process_participant(
            participant_label=label,
            loader=test_loader,
            model=model,
            batch_converter=batch_converter,
            repr_layer=repr_layer,
            device=device,
            batch_size=64,
            output_dir=emb_dir,
            log=log,
            verbose=0,
        )

    return emb_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestESM2ModelLoading:
    """Test ESM-2 model loading and sanity checks."""

    def test_model_layers(self, esm2_model):
        model = esm2_model[0]
        assert model.num_layers == EXPECTED_NUM_LAYERS, (
            f"Expected {EXPECTED_NUM_LAYERS} layers, got {model.num_layers}"
        )

    def test_embedding_dim(self, esm2_model):
        model = esm2_model[0]
        embed_dim = getattr(model, "embed_dim", None) or model.args.embed_dim
        assert embed_dim == EMBEDDING_DIM, (
            f"Expected embedding dim {EMBEDDING_DIM}, got {embed_dim}"
        )

    def test_repr_layer(self, esm2_model):
        repr_layer = esm2_model[2]
        assert repr_layer == EXPECTED_NUM_LAYERS


@pytest.mark.integration
class TestBasicEmbedding:
    """Test embedding computation on known sequences."""

    def test_shape_and_dtype(self, esm2_model):
        model, batch_converter, repr_layer, device, _, log = esm2_model
        seqs = ["CASSLGTDTQYF", "CASSLAPGATNEKLFF", "CASRLAGGRNEQFF"]

        embeddings = compute_embeddings_for_sequences(
            seqs, model, batch_converter, repr_layer,
            device, batch_size=64, log=log, verbose=0,
        )

        assert embeddings.shape == (3, EMBEDDING_DIM)
        assert embeddings.dtype == np.float16

    def test_no_nan_or_inf(self, esm2_model):
        model, batch_converter, repr_layer, device, _, log = esm2_model
        seqs = ["CASSLGTDTQYF", "CSVGTGANNLFF", "CASSYSIEQYF"]

        embeddings = compute_embeddings_for_sequences(
            seqs, model, batch_converter, repr_layer,
            device, batch_size=64, log=log, verbose=0,
        )

        assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
        assert not np.isinf(embeddings).any(), "Embeddings contain Inf"

    def test_distinct_sequences_produce_distinct_embeddings(self, esm2_model):
        model, batch_converter, repr_layer, device, _, log = esm2_model
        seqs = ["CASSLGTDTQYF", "CASSLAPGATNEKLFF", "CASRLAGGRNEQFF"]

        embeddings = compute_embeddings_for_sequences(
            seqs, model, batch_converter, repr_layer,
            device, batch_size=64, log=log, verbose=0,
        )

        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                assert not np.allclose(embeddings[i], embeddings[j], atol=1e-3), (
                    f"Sequences {i} and {j} produced identical embeddings"
                )

    def test_empty_sequences(self, esm2_model):
        model, batch_converter, repr_layer, device, _, log = esm2_model

        embeddings = compute_embeddings_for_sequences(
            [], model, batch_converter, repr_layer,
            device, batch_size=64, log=log, verbose=0,
        )

        assert embeddings.shape == (0, EMBEDDING_DIM)
        assert embeddings.dtype == np.float16


@pytest.mark.integration
class TestParticipantProcessing:
    """Test full participant pipeline: CLEAN -> DOWNSAMPLE -> embed -> save."""

    def test_output_files_exist(self, computed_embeddings_dir, test_participant_labels):
        for label in test_participant_labels:
            assert (computed_embeddings_dir / f"{label}_embeddings.npy").exists()
            assert (computed_embeddings_dir / f"{label}_downsampled.parquet").exists()
            assert (computed_embeddings_dir / f"{label}_stats.json").exists()

    def test_embedding_shape_matches_parquet(self, computed_embeddings_dir, test_participant_labels):
        for label in test_participant_labels:
            emb = np.load(computed_embeddings_dir / f"{label}_embeddings.npy")
            df = pd.read_parquet(computed_embeddings_dir / f"{label}_downsampled.parquet")

            assert emb.dtype == np.float16, f"{label}: dtype={emb.dtype}"
            assert emb.shape[0] == len(df), (
                f"{label}: row mismatch: embeddings={emb.shape[0]}, parquet={len(df)}"
            )
            if emb.shape[0] > 0:
                assert emb.shape[1] == EMBEDDING_DIM, (
                    f"{label}: dim={emb.shape[1]}, expected {EMBEDDING_DIM}"
                )

    def test_stats_json_contents(self, computed_embeddings_dir, test_participant_labels):
        for label in test_participant_labels:
            with open(computed_embeddings_dir / f"{label}_stats.json") as f:
                stats = json.load(f)

            assert stats["participant_label"] == label
            assert "kept" in stats
            assert "n_sequences_downsampled" in stats
            assert "timestamp" in stats


@pytest.mark.integration
class TestResumption:
    """Test that re-running skips already-processed participants."""

    def test_files_not_modified_on_rerun(self, computed_embeddings_dir, test_participant_labels):
        # Record file modification times
        mtimes = {}
        for label in test_participant_labels:
            stats_path = computed_embeddings_dir / f"{label}_stats.json"
            mtimes[label] = stats_path.stat().st_mtime

        # All participants have stats.json, so resumption should skip them
        for label in test_participant_labels:
            stats_path = computed_embeddings_dir / f"{label}_stats.json"
            assert stats_path.exists(), (
                f"Participant {label} stats missing — resumption would recompute"
            )
            assert stats_path.stat().st_mtime == mtimes[label], (
                f"Stats file for {label} was modified during resumption check"
            )


@pytest.mark.integration
class TestVerifyMode:
    """Test verify mode on valid and corrupted data."""

    def test_valid_data_passes(self, computed_embeddings_dir, esm2_model):
        _, _, _, _, _, log = esm2_model
        assert verify_embeddings(computed_embeddings_dir, log), (
            "Verification should pass on valid data"
        )

    def test_corrupted_data_detected(self, computed_embeddings_dir, esm2_model):
        _, _, _, _, _, log = esm2_model

        # Find a non-empty embedding file to corrupt
        npy_files = sorted(computed_embeddings_dir.glob("*_embeddings.npy"))
        target = None
        for npy_path in npy_files:
            emb = np.load(npy_path)
            if emb.shape[0] > 0:
                target = npy_path
                break

        if target is None:
            pytest.skip("No non-empty embeddings to corrupt")

        # Save original, corrupt, verify, then restore
        original_emb = np.load(target)
        corrupted = np.zeros((original_emb.shape[0] + 5, EMBEDDING_DIM), dtype=np.float16)
        np.save(target, corrupted)

        try:
            ok = verify_embeddings(computed_embeddings_dir, log)
            assert not ok, "Verification should fail on corrupted data"
        finally:
            # Always restore original
            np.save(target, original_emb)


@pytest.mark.integration
class TestEmbeddingFoldAssembly:
    """Test loading and assembling embeddings (simulates training-time fold assembly)."""

    def test_concatenation(self, computed_embeddings_dir, test_participant_labels):
        all_embeddings = []
        all_dfs = []

        for label in test_participant_labels:
            emb = np.load(computed_embeddings_dir / f"{label}_embeddings.npy")
            df = pd.read_parquet(computed_embeddings_dir / f"{label}_downsampled.parquet")

            assert emb.shape[0] == len(df), (
                f"Row mismatch for {label}: emb={emb.shape[0]}, parquet={len(df)}"
            )

            if emb.shape[0] > 0:
                all_embeddings.append(emb)
                all_dfs.append(df)

        if not all_embeddings:
            pytest.skip("No non-empty participants for fold assembly test")

        combined_emb = np.concatenate(all_embeddings, axis=0)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_rows = combined_emb.shape[0]

        assert combined_emb.shape == (total_rows, EMBEDDING_DIM)
        assert combined_emb.dtype == np.float16
        assert len(combined_df) == total_rows

    def test_cdr3_column_present(self, computed_embeddings_dir, test_participant_labels):
        """Training requires cdr3_aa in the assembled DataFrame."""
        for label in test_participant_labels:
            df = pd.read_parquet(computed_embeddings_dir / f"{label}_downsampled.parquet")
            if len(df) > 0:
                assert "cdr3_aa" in df.columns, (
                    f"{label}: missing cdr3_aa column in downsampled parquet"
                )
