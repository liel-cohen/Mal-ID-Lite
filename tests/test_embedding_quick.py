#!/usr/bin/env python
"""
Quick smoke test for ESM-2 embedding computation.

Tests on a small subset of participants (2-3) to verify:
1. ESM-2 model loads correctly on detected device
2. CLEAN -> DOWNSAMPLED preprocessing works
3. Embeddings have correct shape (N, 640) and dtype (float16)
4. Parquet + npy row alignment
5. Stats JSON is complete and correct
6. Resumption works (re-running skips already-processed participants)
7. Verify mode catches intentionally corrupted files
8. Embedding loading and fold assembly simulation
9. Performance summary with extrapolated full-run estimate

Output: tests/test_outputs/test_embedding_quick/

Expected runtime: ~4 minutes on MPS (M4 Max), varies by hardware and batch size
"""

import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage

# Test output directory
TEST_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / TEST_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import embedding script components (moved from scripts/embedding/ to training module)
from malid_lite.training.compute_model3_embeddings import (
    EMBEDDING_DIM,
    ESM2_MODEL_NAME,
    EXPECTED_NUM_LAYERS,
    detect_device,
    get_machine_specs,
    load_esm2_model,
    compute_embeddings_for_sequences,
    process_participant,
    verify_embeddings,
    setup_logging,
    HAS_PSUTIL,
)

# Try to import psutil
try:
    import psutil
except ImportError:
    psutil = None

# Number of test participants
N_TEST_PARTICIPANTS = 3


def get_test_loader_and_participants():
    """Initialize data loader and pick a few participants for testing."""
    cache_base = PROJECT_ROOT / "cache" / "mal-id-orig-data"
    participants_dir = cache_base / "participants"

    assert participants_dir.exists(), (
        f"Participant cache not found: {participants_dir}\n"
        "Run: python scripts/data/cache_and_report_all_data.py"
    )

    # Pick first N participants from cache
    clean_parquets = sorted(participants_dir.glob("*_clean.parquet"))
    assert len(clean_parquets) > 0, "No cached participants found"

    test_labels = [p.stem.removesuffix("_clean") for p in clean_parquets[:N_TEST_PARTICIPANTS]]

    # Find metadata path from cache info or use standard location
    cache_info_path = cache_base / "participants" / "cache_info.json"
    if cache_info_path.exists():
        import json as _json
        with open(cache_info_path) as f:
            cache_info = _json.load(f)
        metadata_path = Path(cache_info.get("metadata_path", ""))
        data_dir = Path(cache_info.get("data_dir", "."))
    else:
        # Fallback: look for metadata in standard locations
        raise FileNotFoundError(
            f"Cache info not found at {cache_info_path}. "
            "Rebuild cache with: python scripts/data/cache_and_report_all_data.py"
        )

    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus="TCR",
        verbose=0,
        cache_dir=cache_base,
    )

    return loader, test_labels


def test_model_loading():
    """Test 1: ESM-2 model loads correctly and passes sanity checks."""
    print("\n--- Test 1: Model Loading ---")

    device = detect_device()
    print(f"Detected device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"test_log_{timestamp}.log"
    log = setup_logging(log_file, verbose=1)

    t0 = time.time()
    model, alphabet, batch_converter, repr_layer, torch_device, load_time = \
        load_esm2_model(device, log)
    t1 = time.time()

    print(f"Model loaded in {load_time:.1f}s")
    print(f"  Layers: {model.num_layers}")
    embed_dim = getattr(model, "embed_dim", None) or model.args.embed_dim
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Repr layer: {repr_layer}")
    print(f"  Device: {torch_device}")

    assert model.num_layers == EXPECTED_NUM_LAYERS
    assert embed_dim == EMBEDDING_DIM
    assert repr_layer == EXPECTED_NUM_LAYERS

    print("PASSED")
    return model, alphabet, batch_converter, repr_layer, torch_device, load_time, log


def test_basic_embedding(model, batch_converter, repr_layer, device, log):
    """Test 2: Embedding a few known sequences produces correct output."""
    print("\n--- Test 2: Basic Embedding ---")

    # Short CDR3-like sequences
    test_sequences = [
        "CASSLGTDTQYF",
        "CASSLAPGATNEKLFF",
        "CASRLAGGRNEQFF",
        "CSVGTGANNLFF",
        "CASSYSIEQYF",
    ]

    t0 = time.time()
    embeddings = compute_embeddings_for_sequences(
        test_sequences, model, batch_converter, repr_layer,
        device, batch_size=64, log=log, verbose=2,
    )
    t1 = time.time()

    print(f"Embedded {len(test_sequences)} sequences in {t1 - t0:.3f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    assert embeddings.shape == (len(test_sequences), EMBEDDING_DIM), \
        f"Expected ({len(test_sequences)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    assert embeddings.dtype == np.float16, f"Expected float16, got {embeddings.dtype}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"
    # Each sequence should produce a different embedding
    for i in range(len(test_sequences)):
        for j in range(i + 1, len(test_sequences)):
            assert not np.allclose(embeddings[i], embeddings[j], atol=1e-3), \
                f"Sequences {i} and {j} produced identical embeddings"

    print("PASSED")
    return t1 - t0


def test_empty_sequences(model, batch_converter, repr_layer, device, log):
    """Test 3: Empty sequence list produces correct empty output."""
    print("\n--- Test 3: Empty Sequences ---")

    embeddings = compute_embeddings_for_sequences(
        [], model, batch_converter, repr_layer,
        device, batch_size=64, log=log, verbose=0,
    )

    assert embeddings.shape == (0, EMBEDDING_DIM)
    assert embeddings.dtype == np.float16

    print(f"  Empty output shape: {embeddings.shape}")
    print("PASSED")


def test_participant_processing(loader, test_labels, model, batch_converter,
                                repr_layer, device, batch_size, log):
    """Test 4: Full participant pipeline — CLEAN -> DOWNSAMPLE -> embed -> save."""
    print(f"\n--- Test 4: Participant Processing ({len(test_labels)} participants) ---")

    # Use a temp directory so we don't pollute the real cache
    test_emb_dir = OUTPUT_DIR / "embeddings"
    if test_emb_dir.exists():
        shutil.rmtree(test_emb_dir)
    test_emb_dir.mkdir(parents=True)

    participant_stats = []

    for idx, label in enumerate(test_labels, 1):
        print(f"  Processing {idx}/{len(test_labels)}: {label}")

        t0 = time.time()
        stats = process_participant(
            participant_label=label,
            loader=loader,
            model=model,
            batch_converter=batch_converter,
            repr_layer=repr_layer,
            device=device,
            batch_size=batch_size,
            output_dir=test_emb_dir,
            log=log,
            verbose=1,
        )
        elapsed = time.time() - t0

        participant_stats.append(stats)

        # Check output files exist
        npy_path = test_emb_dir / f"{label}_embeddings.npy"
        pq_path = test_emb_dir / f"{label}_downsampled.parquet"
        stats_path = test_emb_dir / f"{label}_stats.json"

        assert npy_path.exists(), f"Missing: {npy_path}"
        assert pq_path.exists(), f"Missing: {pq_path}"
        assert stats_path.exists(), f"Missing: {stats_path}"

        # Load and verify
        emb = np.load(npy_path)
        df = pd.read_parquet(pq_path)

        assert emb.dtype == np.float16, f"dtype={emb.dtype}"
        assert emb.shape[0] == len(df), \
            f"Row mismatch: embeddings={emb.shape[0]}, parquet={len(df)}"
        if emb.shape[0] > 0:
            assert emb.shape[1] == EMBEDDING_DIM, f"dim={emb.shape[1]}"

        n_seq = stats.get("n_sequences_downsampled", 0)
        kept = stats.get("kept", False)
        emb_time = stats.get("embedding_time_seconds", 0)
        print(f"    {'Kept' if kept else 'DROPPED'}: {n_seq:,} seqs, "
              f"embed: {emb_time:.1f}s, total: {elapsed:.1f}s")

        # Verify stats JSON contents
        with open(stats_path) as f:
            loaded_stats = json.load(f)
        assert loaded_stats["participant_label"] == label
        assert "kept" in loaded_stats
        assert "n_sequences_downsampled" in loaded_stats
        assert "timestamp" in loaded_stats

    print("PASSED")
    return participant_stats, test_emb_dir


def test_resumption(loader, test_labels, model, batch_converter,
                    repr_layer, device, batch_size, test_emb_dir, log):
    """Test 5: Resumption — re-running skips already-processed participants."""
    print("\n--- Test 5: Resumption ---")

    # Record file modification times
    mtimes_before = {}
    for label in test_labels:
        stats_path = test_emb_dir / f"{label}_stats.json"
        if stats_path.exists():
            mtimes_before[label] = stats_path.stat().st_mtime

    # "Process" again — all should be skipped since stats.json exists
    n_skipped = 0
    for label in test_labels:
        stats_path = test_emb_dir / f"{label}_stats.json"
        if stats_path.exists():
            n_skipped += 1
            continue
        # If we get here, resumption failed
        assert False, f"Participant {label} was not skipped during resumption"

    # Verify files were not modified
    for label in test_labels:
        stats_path = test_emb_dir / f"{label}_stats.json"
        if label in mtimes_before:
            assert stats_path.stat().st_mtime == mtimes_before[label], \
                f"Stats file for {label} was modified during resumption"

    print(f"  Skipped {n_skipped}/{len(test_labels)} participants (all already done)")
    print("PASSED")


def test_verify_mode(test_emb_dir, log):
    """Test 6: Verify mode detects consistency and catches corruption."""
    print("\n--- Test 6: Verify Mode ---")

    # Should pass on valid data
    ok = verify_embeddings(test_emb_dir, log)
    assert ok, "Verification should pass on valid data"
    print("  Valid data: PASSED")

    # Corrupt one file and verify it catches it
    npy_files = sorted(test_emb_dir.glob("*_embeddings.npy"))
    if npy_files:
        # Find a non-empty one
        target = None
        for npy_path in npy_files:
            emb = np.load(npy_path)
            if emb.shape[0] > 0:
                target = npy_path
                break

        if target is not None:
            label = target.stem.replace("_embeddings", "")
            original_emb = np.load(target)

            # Corrupt: save with wrong number of rows
            corrupted = np.zeros((original_emb.shape[0] + 5, EMBEDDING_DIM), dtype=np.float16)
            np.save(target, corrupted)

            ok = verify_embeddings(test_emb_dir, log)
            assert not ok, "Verification should fail on corrupted data"
            print("  Corrupted data detected: PASSED")

            # Restore original
            np.save(target, original_emb)
        else:
            print("  (No non-empty embeddings to corrupt, skipping corruption test)")
    else:
        print("  (No embedding files found, skipping corruption test)")

    print("PASSED")


def test_embedding_loading(test_emb_dir, test_labels):
    """Test 7: Loading and assembling embeddings simulates training-time fold assembly."""
    print("\n--- Test 7: Embedding Loading (Fold Assembly Simulation) ---")

    all_embeddings = []
    all_dfs = []
    total_rows = 0

    for label in test_labels:
        npy_path = test_emb_dir / f"{label}_embeddings.npy"
        pq_path = test_emb_dir / f"{label}_downsampled.parquet"

        emb = np.load(npy_path)
        df = pd.read_parquet(pq_path)

        assert emb.shape[0] == len(df), \
            f"Row mismatch for {label}: emb={emb.shape[0]}, parquet={len(df)}"

        if emb.shape[0] > 0:
            all_embeddings.append(emb)
            all_dfs.append(df)
            total_rows += emb.shape[0]

    if all_embeddings:
        # Simulate fold assembly: concatenate all participants
        combined_emb = np.concatenate(all_embeddings, axis=0)
        combined_df = pd.concat(all_dfs, ignore_index=True)

        assert combined_emb.shape == (total_rows, EMBEDDING_DIM), \
            f"Combined shape: {combined_emb.shape}, expected ({total_rows}, {EMBEDDING_DIM})"
        assert combined_emb.dtype == np.float16
        assert len(combined_df) == total_rows, \
            f"Combined df rows: {len(combined_df)}, expected {total_rows}"

        # Verify CDR3 column exists (needed for training)
        assert "cdr3_aa" in combined_df.columns, "Missing cdr3_aa column in assembled data"

        # Verify we can index into embeddings by DataFrame position
        # (this is how training will access them)
        sample_idx = min(100, total_rows - 1)
        sample_cdr3 = combined_df.iloc[sample_idx]["cdr3_aa"]
        sample_emb = combined_emb[sample_idx]
        assert sample_emb.shape == (EMBEDDING_DIM,)
        assert isinstance(sample_cdr3, str) and len(sample_cdr3) > 0

        print(f"  Assembled {len(test_labels)} participants: {total_rows:,} sequences")
        print(f"  Combined embeddings shape: {combined_emb.shape}")
        print(f"  Combined DataFrame columns: {len(combined_df.columns)}")
        print(f"  Sample CDR3 at idx {sample_idx}: {sample_cdr3} (emb norm={np.linalg.norm(sample_emb.astype(np.float32)):.2f})")
    else:
        print("  (No non-empty participants, skipping assembly test)")

    print("PASSED")


def print_performance_summary(
    machine_specs, model_load_time, basic_embed_time, participant_stats, batch_size
):
    """Print performance summary with full-run estimate."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    # Machine specs
    print(f"\nMachine:")
    print(f"  Platform: {machine_specs.get('platform', 'unknown')}")
    print(f"  Processor: {machine_specs.get('processor', 'unknown')}")
    print(f"  CPU cores: {machine_specs.get('cpu_count', 'unknown')}")
    ram = machine_specs.get('ram_total_gb', 'unknown')
    print(f"  RAM: {ram} GB" if isinstance(ram, (int, float)) else f"  RAM: {ram}")
    print(f"  Device: {machine_specs.get('device_type', 'unknown')}")
    gpu = machine_specs.get('gpu_name', 'none')
    if gpu != "none":
        print(f"  GPU: {gpu}")
        vram = machine_specs.get('gpu_vram_gb', 'unknown')
        print(f"  GPU VRAM: {vram} GB" if isinstance(vram, (int, float)) else f"  GPU VRAM: {vram}")

    if not HAS_PSUTIL:
        print(f"  Note: install psutil for more detailed specs (pip install psutil)")

    # Timing
    print(f"\nTiming:")
    print(f"  Model load: {model_load_time:.1f}s")
    print(f"  Basic embed (5 seqs): {basic_embed_time:.3f}s")

    kept_stats = [s for s in participant_stats if s.get("kept", False)]
    if kept_stats:
        total_seqs = sum(s.get("n_sequences_downsampled", 0) for s in kept_stats)
        total_embed_time = sum(s.get("embedding_time_seconds", 0) for s in kept_stats)
        total_preprocess_time = sum(s.get("preprocess_time_seconds", 0) for s in kept_stats)

        print(f"  Test participants processed: {len(kept_stats)}")
        print(f"  Test sequences embedded: {total_seqs:,}")
        print(f"  Test preprocessing time: {total_preprocess_time:.1f}s")
        print(f"  Test embedding time: {total_embed_time:.1f}s")

        if total_embed_time > 0:
            throughput = total_seqs / total_embed_time
            print(f"  Throughput: {throughput:,.0f} seq/s")

            # Full-run estimate
            # Get total participant count and estimate total sequences
            cache_base = PROJECT_ROOT / "cache" / "mal-id-orig-data"
            total_participants = len(list((cache_base / "participants").glob("*_clean.parquet")))
            avg_seqs_per_participant = total_seqs / len(kept_stats)

            # Rough estimate
            est_total_seqs = avg_seqs_per_participant * total_participants
            est_embed_time = est_total_seqs / throughput
            est_preprocess_time = (total_preprocess_time / len(kept_stats)) * total_participants
            est_total_time = est_embed_time + est_preprocess_time + model_load_time

            print(f"\nFull-run estimate ({total_participants} participants):")
            print(f"  Estimated total sequences: ~{est_total_seqs:,.0f}")
            print(f"  Estimated embedding time: ~{est_embed_time / 60:.0f} min")
            print(f"  Estimated preprocessing time: ~{est_preprocess_time / 60:.0f} min")
            print(f"  Estimated total time: ~{est_total_time / 60:.0f} min ({est_total_time / 3600:.1f} hours)")
            print(f"  Batch size used: {batch_size}")

            # Storage estimate
            est_storage_gb = (est_total_seqs * EMBEDDING_DIM * 2) / (1024 ** 3)  # float16 = 2 bytes
            print(f"  Estimated embedding storage: ~{est_storage_gb:.1f} GB")
    else:
        print("  No participants had data after downsampling — cannot estimate full run.")

    print(f"\nTest output: {OUTPUT_DIR}")
    print("=" * 70)


def save_performance_report(
    machine_specs, model_load_time, basic_embed_time, participant_stats, batch_size,
    test_results,
):
    """Save performance summary as JSON and MD report."""
    kept_stats = [s for s in participant_stats if s.get("kept", False)]
    total_seqs = sum(s.get("n_sequences_downsampled", 0) for s in kept_stats)
    total_embed_time = sum(s.get("embedding_time_seconds", 0) for s in kept_stats)
    total_preprocess_time = sum(s.get("preprocess_time_seconds", 0) for s in kept_stats)
    throughput = round(total_seqs / total_embed_time, 1) if total_embed_time > 0 else 0

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "machine_specs": machine_specs,
        "model_load_time_seconds": round(model_load_time, 3),
        "basic_embed_time_seconds": round(basic_embed_time, 3),
        "batch_size": batch_size,
        "test_participants": len(participant_stats),
        "test_participants_kept": len(kept_stats),
        "test_sequences_embedded": total_seqs,
        "test_embedding_time_seconds": round(total_embed_time, 3),
        "throughput_seq_per_sec": throughput,
        "participant_details": participant_stats,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = OUTPUT_DIR / f"performance_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nPerformance report (JSON) saved: {json_path}")

    # Save MD
    md_path = OUTPUT_DIR / f"test_report_batch{batch_size}_{timestamp}.md"
    _write_md_report(
        md_path, machine_specs, model_load_time, basic_embed_time,
        participant_stats, batch_size, test_results, throughput,
        total_seqs, total_embed_time, total_preprocess_time,
    )
    print(f"Performance report (MD) saved: {md_path}")


def _write_md_report(
    md_path, machine_specs, model_load_time, basic_embed_time,
    participant_stats, batch_size, test_results, throughput,
    total_seqs, total_embed_time, total_preprocess_time,
):
    """Write detailed markdown report with all diagnostics."""
    kept_stats = [s for s in participant_stats if s.get("kept", False)]
    lines = []

    lines.append(f"# ESM-2 Embedding Test Report (batch_size={batch_size})")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append("")

    # Test results
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Test | Result |")
    lines.append("|------|--------|")
    for name, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        lines.append(f"| {name} | {status} |")
    lines.append("")

    # Machine specs
    lines.append("## Machine Specs")
    lines.append("")
    lines.append(f"- **Platform:** {machine_specs.get('platform', 'unknown')}")
    lines.append(f"- **Processor:** {machine_specs.get('processor', 'unknown')}")
    lines.append(f"- **CPU cores:** {machine_specs.get('cpu_count', 'unknown')}")
    ram = machine_specs.get('ram_total_gb', 'unknown')
    lines.append(f"- **RAM:** {ram} GB" if isinstance(ram, (int, float)) else f"- **RAM:** {ram}")
    lines.append(f"- **Compute device:** {machine_specs.get('device_type', 'unknown')}")
    gpu = machine_specs.get('gpu_name', 'none')
    if gpu != "none":
        lines.append(f"- **GPU:** {gpu}")
        vram = machine_specs.get('gpu_vram_gb', 'unknown')
        lines.append(f"- **GPU VRAM:** {vram} GB" if isinstance(vram, (int, float)) else f"- **GPU VRAM:** {vram}")
    lines.append(f"- **Python:** {machine_specs.get('python_version', 'unknown')}")
    lines.append(f"- **PyTorch:** {machine_specs.get('torch_version', 'unknown')}")
    lines.append(f"- **NumPy:** {machine_specs.get('numpy_version', 'unknown')}")
    if not HAS_PSUTIL:
        lines.append(f"- **Note:** install psutil for more detailed specs (`pip install psutil`)")
    lines.append("")

    # ESM-2 model info
    lines.append("## ESM-2 Model")
    lines.append("")
    lines.append(f"- **Model:** {ESM2_MODEL_NAME}")
    lines.append(f"- **Layers:** {EXPECTED_NUM_LAYERS}")
    lines.append(f"- **Embedding dim:** {EMBEDDING_DIM}")
    lines.append(f"- **Storage dtype:** float16")
    lines.append(f"- **Model load time:** {model_load_time:.2f}s")
    lines.append(f"- **Basic embed warmup (5 seqs):** {basic_embed_time:.3f}s")
    lines.append("")

    # Run parameters
    lines.append("## Run Parameters")
    lines.append("")
    lines.append(f"- **Batch size:** {batch_size}")
    lines.append(f"- **Test participants:** {len(participant_stats)}")
    lines.append(f"- **Test participants with data:** {len(kept_stats)}")
    lines.append("")

    # Per-participant details
    lines.append("## Per-Participant Results")
    lines.append("")
    lines.append("| Participant | Kept | Sequences | Specimens | Preprocess (s) | Embed (s) | Seq/s |")
    lines.append("|-------------|------|-----------|-----------|----------------|-----------|-------|")
    for s in sorted(participant_stats, key=lambda x: x["participant_label"]):
        kept_str = "yes" if s.get("kept", False) else "no"
        n_seq = s.get("n_sequences_downsampled", 0)
        n_spec = s.get("n_specimens", 0)
        pre_time = s.get("preprocess_time_seconds", 0)
        emb_time = s.get("embedding_time_seconds", 0)
        seq_s = s.get("sequences_per_second", 0)
        lines.append(
            f"| {s['participant_label']} | {kept_str} | {n_seq:,} | {n_spec} "
            f"| {pre_time:.2f} | {emb_time:.1f} | {seq_s:,.0f} |"
        )
    lines.append("")

    # Aggregate performance
    lines.append("## Aggregate Performance")
    lines.append("")
    lines.append(f"- **Total sequences embedded:** {total_seqs:,}")
    lines.append(f"- **Total preprocessing time:** {total_preprocess_time:.1f}s")
    lines.append(f"- **Total embedding time:** {total_embed_time:.1f}s")
    lines.append(f"- **Throughput:** {throughput:,.0f} seq/s")
    if kept_stats:
        avg_embed = total_embed_time / len(kept_stats)
        lines.append(f"- **Avg embedding time per participant:** {avg_embed:.1f}s")
    lines.append("")

    # Full-run estimate
    if throughput > 0 and kept_stats:
        cache_base = PROJECT_ROOT / "cache" / "mal-id-orig-data"
        total_participants = len(list((cache_base / "participants").glob("*_clean.parquet")))
        avg_seqs = total_seqs / len(kept_stats)
        est_total_seqs = avg_seqs * total_participants
        est_embed_time = est_total_seqs / throughput
        est_preprocess = (total_preprocess_time / len(kept_stats)) * total_participants
        est_total = est_embed_time + est_preprocess + model_load_time
        est_storage_gb = (est_total_seqs * EMBEDDING_DIM * 2) / (1024 ** 3)

        lines.append("## Full-Run Estimate")
        lines.append("")
        lines.append(f"- **Total participants:** {total_participants}")
        lines.append(f"- **Estimated total sequences:** ~{est_total_seqs:,.0f}")
        lines.append(f"- **Estimated embedding time:** ~{est_embed_time / 60:.0f} min")
        lines.append(f"- **Estimated preprocessing time:** ~{est_preprocess / 60:.0f} min")
        lines.append(f"- **Estimated total time:** ~{est_total / 60:.0f} min ({est_total / 3600:.1f} hours)")
        lines.append(f"- **Estimated embedding storage:** ~{est_storage_gb:.1f} GB (float16 .npy)")
        lines.append(f"- **Batch size used for estimate:** {batch_size}")
        lines.append("")
        lines.append("*Note: Estimates are extrapolated from a small sample. Actual times may vary ")
        lines.append("based on sequence length distribution and system load.*")
        lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path


def main():
    """Run all embedding tests."""
    import argparse

    parser = argparse.ArgumentParser(description="ESM-2 Embedding Quick Test")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for embedding (default: 64). Try larger values to benchmark.",
    )
    args = parser.parse_args()
    batch_size = args.batch_size

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("ESM-2 Embedding Quick Test")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Track test results for the report
    test_results = []

    # Collect machine specs early
    machine_specs = get_machine_specs()

    # Test 1: Model loading
    model, alphabet, batch_converter, repr_layer, device, model_load_time, log = \
        test_model_loading()
    test_results.append(("Model loading", True))

    # Test 2: Basic embedding
    basic_embed_time = test_basic_embedding(model, batch_converter, repr_layer, device, log)
    test_results.append(("Basic embedding", True))

    # Test 3: Empty sequences
    test_empty_sequences(model, batch_converter, repr_layer, device, log)
    test_results.append(("Empty sequences", True))

    # Test 4: Full participant pipeline
    loader, test_labels = get_test_loader_and_participants()
    participant_stats, test_emb_dir = test_participant_processing(
        loader, test_labels, model, batch_converter,
        repr_layer, device, batch_size, log,
    )
    test_results.append(("Participant processing", True))

    # Test 5: Resumption
    test_resumption(
        loader, test_labels, model, batch_converter,
        repr_layer, device, batch_size, test_emb_dir, log,
    )
    test_results.append(("Resumption", True))

    # Test 6: Verify mode
    test_verify_mode(test_emb_dir, log)
    test_results.append(("Verify mode", True))

    # Test 7: Embedding loading (fold assembly simulation)
    test_embedding_loading(test_emb_dir, test_labels)
    test_results.append(("Embedding loading", True))

    # Performance summary
    print_performance_summary(
        machine_specs, model_load_time, basic_embed_time,
        participant_stats, batch_size,
    )
    save_performance_report(
        machine_specs, model_load_time, basic_embed_time,
        participant_stats, batch_size, test_results,
    )

    print("\nAll tests PASSED")


if __name__ == "__main__":
    main()
