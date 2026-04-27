# Mal-ID-Lite Pipeline Guide

How to set up, run, and verify the Mal-ID-Lite training pipeline.

All commands should be run from the **mal-id-lite project root** directory (the cloned repo).

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Key Concepts](#2-key-concepts)
  - [2.1 Classification Modes](#21-classification-modes)
  - [2.2 Common Arguments](#22-common-arguments)
  - [2.3 Model-Specific Arguments](#23-model-specific-arguments)
- [3. Prerequisites](#3-prerequisites)
  - [3.4 Run the Test Suite](#34-run-the-test-suite)
- [4. Data Setup](#4-data-setup)
  - [4.1 Metadata File](#41-metadata-file)
  - [4.2 Participant Sequence Files](#42-participant-sequence-files)
  - [4.3 Directory Layout and Paths](#43-directory-layout-and-paths)
  - [4.4 Data Cache](#44-data-cache)
- [5. Pre-Computing Data and Embeddings (Optional)](#5-pre-computing-data-and-embeddings-optional)
  - [5.1 Building the Data Cache](#51-building-the-data-cache)
  - [5.2 Pre-Computing ESM-2 Embeddings](#52-pre-computing-esm-2-embeddings)
  - [5.3 Cache Management](#53-cache-management)
- [6. Running the Full Pipeline (Ensemble)](#6-running-the-full-pipeline-ensemble)
- [7. Running Individual Models](#7-running-individual-models)
- [8. Resume Logic](#8-resume-logic)
- [9. Verifying Results](#9-verifying-results)
- [Appendix A: Hardware and Performance](#appendix-a-hardware-and-performance)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)

---

## 1. Overview

Mal-ID-Lite classifies disease from TCR repertoire data using three models, each capturing signal at a different level. An ensemble meta-learner combines their predictions.

```
Raw AIRR data + Metadata
        |
        v
  [Data Cache]  .............. preprocess + cache (auto or manual)
   |    |    |
   v    v    v
  M1   M2   M3 (ESM-2 embeddings auto-computed if missing)
   |    |    |
   v    v    v
  [Ensemble meta-learner]  .. ridge regression on base model predictions
        |
        v
  Final predictions + metrics
```

| Model | What it captures | Approach |
| --- | --- | --- |
| **Model 1** | Repertoire-level V/J gene usage and CDR3 length distributions | PCA of gene frequencies, then elastic net logistic regression (glmnet) |
| **Model 2** | Convergent CDR3 clusters shared across patients | Fisher exact test for disease-associated clusters, then logistic regression |
| **Model 3** | Individual CDR3 sequence features via protein language model | ESM-2 embeddings, per-V-gene binary classifiers, then specimen-level random forest |
| **Ensemble** | Combined signal from all three models | Ridge logistic regression meta-learner on base model probability predictions |

Models are trained and evaluated using **participant-level cross-validation** (stratified by disease). With the original Mal-ID dataset this is 3-fold CV.

---

## 2. Key Concepts

### 2.1 Classification Modes

All training scripts (individual and ensemble) support three modes via `--classification-mode`:

**Multiclass** (default) -- A single N-class classifier on all disease classes simultaneously. Standard mode for the original Mal-ID dataset (6 classes).

```bash
--classification-mode multiclass
```

**Binary** -- One binary classifier for a single disease-vs-reference pair.

```bash
# Two-class dataset (auto-detects classes):
--classification-mode binary

# Pick one disease from an N-class dataset:
--classification-mode binary \
    --reference-class "Healthy/Background" \
    --diseases Covid19
```

**Multi-binary** -- One independent binary classifier per disease vs. a shared reference class. Trains N-1 separate models (one per non-reference disease).

```bash
# All diseases vs Healthy/Background:
--classification-mode multi-binary \
    --reference-class "Healthy/Background"

# Only specific diseases:
--classification-mode multi-binary \
    --reference-class "Healthy/Background" \
    --diseases Covid19 HIV Lupus
```

### 2.2 Common Arguments

These arguments are shared across all training scripts:

| Argument | Description |
| --- | --- |
| `--metadata-path` | Path to the metadata TSV file. See [Section 4.1](#41-metadata-file) for required columns. |
| `--data-dir` | Path to the directory containing participant sequence files (e.g., `$DATA_DIR`). The directory must contain the `part_table_*` files directly -- not in subdirectories. See [Section 4.2](#42-participant-sequence-files) for naming and format requirements. Only needed on first run to build the cache; subsequent runs can omit it. |
| `--dataset-name` | Dataset identifier used in output folder names (default: `mal-id-orig-data`). |
| `--cache-dir` | Directory where the pipeline stores the preprocessed data cache and Model 3 embeddings cache (default: `cache/<dataset-name>` relative to project root). Recommended: set to a location outside the repo (see [Section 4.3](#43-directory-layout-and-paths)). See [Section 4.4](#44-data-cache) for details. |
| `--classification-mode` | `multiclass`, `binary`, or `multi-binary` (default: `multiclass`). See [Section 2.1](#21-classification-modes). |
| `--reference-class` | Reference/negative class for binary and multi-binary modes (e.g., `"Healthy/Background"`). |
| `--diseases` | Subset of disease classes to include (space-separated). Default: all classes from metadata. |
| `--fold-ids` | Train only specific folds (e.g., `--fold-ids 0 2`). Default: all folds from metadata. |
| `--gene-locus` | `TCR` (default; BCR not yet fully supported). |
| `--n-jobs` | Parallel workers (default: 4). Never use -1. |
| `--verbose` | 0 = silent, 1 = progress (default), 2 = diagnostics. |
| `--resume` | Resume from partial artifacts after crash/interruption. See [Section 8](#8-resume-logic). |

### 2.3 Model-Specific Arguments

**Model 1:**

| Argument | Default | Description |
| --- | --- | --- |
| `--n-pcs` | 15 | Number of PCA components |
| `--l1-ratio` | (auto) | Elastic net L1/L2 ratio |
| `--model-name` | lasso_cv | Model variant label |

**Model 2:**

| Argument | Default | Description |
| --- | --- | --- |
| `--p-values` | 0.0005 0.001 0.005 0.01 0.05 | P-value grid for Fisher's exact test threshold search |
| `--retrain-full` | off (original Mal-ID behavior) | Train final GLM on combined train set (train_smaller1 + train_smaller2) after p-value search |

The CDR3 clustering identity threshold defaults to 0.90 for TCR and is not configurable via the standalone `train_model2.py` CLI.

**Model 3:**

| Argument | Default | Description |
| --- | --- | --- |
| `--aggregation-strategy` | `entropy_percentile_cutoff` | Sequence-to-specimen aggregation (see below) |
| `--entropy-bottom-percentile` | 0.01 | Percentile cutoff for `entropy_percentile_cutoff` (0-100 scale). 0.01 = keep sequences in the bottom 0.01% of the training entropy distribution. |
| `--entropy-max-fraction` | 0.80 | Fraction cutoff for `entropy_cutoff` (0-1 scale). 0.80 = keep sequences below 0.8 * max possible entropy. |
| `--n-estimators-stage1` | 100 | RF trees in Stage 1 (BCR only; TCR uses glmnet ridge) |
| `--n-estimators-stage2` | 100 | RF trees in Stage 2 |
| `--device` | (auto) | Device for ESM-2 embeddings: `cuda`, `mps`, or `cpu` |
| `--embedding-batch-size` | 64 | Batch size for ESM-2 embedding computation |
| `--embedding-dir` | (auto) | Directory with pre-computed ESM-2 embeddings |
| `--resume-from-stage2` | off | Reload Stage 1, retrain Stage 2 only |
| `--resume-from-evaluation` | off | Reload Stage 1 + 2, re-run evaluation only |

**Aggregation strategies** (`--aggregation-strategy`):

| Strategy | Description |
| --- | --- |
| `entropy_percentile_cutoff` (default) | Keep sequences below x-th percentile of the training entropy distribution, then compute weighted mean of surviving sequence probabilities. Threshold set via `--entropy-bottom-percentile` (default 0.01). |
| `entropy_cutoff` | Keep sequences with entropy below a fraction of max possible entropy, then compute weighted mean of surviving sequence probabilities. Threshold set via `--entropy-max-fraction` (default 0.80). |
| `auto_tuned` | Inner CV grid search over multiple strategies to find the best one (see below). |
| `paper_best` | Paper-best per locus: TCR = `entropy_cutoff` (0.80), BCR = `mean`. |
| `mean` | Weighted mean of all sequence probabilities. |
| `median` | Weighted median of all sequence probabilities. |

**Auto-tuning details** (`--aggregation-strategy auto_tuned`):

Searches a grid of strategies and thresholds via inner CV on the train_smaller2 split, selects by mean MCC. Default tuning grid:

- **Strategies:** `entropy_cutoff`, `entropy_percentile_cutoff`
- **Entropy max fractions** (for `entropy_cutoff`): 0.80, 0.90, 0.95
- **Entropy percentiles** (for `entropy_percentile_cutoff`): 0.01, 0.05, 0.1, 0.5
- **Inner CV splits:** 3

The grid can be customized via `--tuning-strategies`, `--tuning-entropy-max-fractions`, `--tuning-entropy-percentiles`, and `--tuning-cv-splits`. When training through the ensemble, use the `--model3-` prefix (e.g., `--model3-tuning-strategies`).

---

## 3. Prerequisites

### 3.1 Create the conda environment

```bash
conda create -n mal_id_lite python=3.12
conda activate mal_id_lite
```

### 3.2 Install dependencies

```bash
# Core scientific stack (conda-forge)
conda install -c conda-forge pandas numpy pyarrow scikit-learn scipy psutil pytest

# python-glmnet (pip only -- not on conda-forge)
pip install python-glmnet

# PyTorch
# With CUDA GPU:
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
# NOTE: pytorch-cuda version must be <= your driver's CUDA version.
# Run `nvidia-smi` and check "CUDA Version" in the top right.

# CPU only:
conda install pytorch cpuonly -c pytorch

# fair-esm (pip only)
pip install fair-esm
```

**Alternative:** `pip install -r requirements.txt` (not recommended -- conda handles CUDA/version compatibility better).

### 3.3 Verify installation

```bash
python -c "
import pandas, numpy, sklearn, glmnet, torch, esm, scipy
print('All imports OK')
print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

### 3.4 Run the test suite

After installing dependencies, run the test suite to verify everything works end-to-end. The repo includes a small mock dataset (`tests/test_data/`) that exercises the full pipeline -- no external data needed.

```bash
# Full suite (unit + integration, ~10-30 min):
python tests/run_all_tests.py

# Unit tests only (fast, ~2-5 min):
python tests/run_all_tests.py --skip-integration

# Custom parallel workers for Models 2, 3, and ensemble integration tests (default: 2):
python tests/run_all_tests.py --n-jobs 4
```

The `--n-jobs` flag controls parallelism in integration tests that use it (Models 2, 3, and ensemble). Model 1 tests are single-threaded and ignore this flag. The default (2) is conservative; increase it on machines with more RAM and CPU cores.

The test runner executes test groups in dependency order (data loading, Model 1, Model 2, Model 3, ensemble) and reports a summary at the end. All groups should pass before proceeding.

---

## 4. Data Setup

The examples below (number of participants, disease classes, fold counts) are for the **original Mal-ID published dataset**, which contains TCR repertoire data for 6 disease classes: **Covid19, HIV, Healthy/Background, Influenza, Lupus, and T1D** (542 participants, 616 specimens, 3-fold CV).

The package works with any dataset that follows the same format.

### 4.1 Metadata File

A TSV file with one row per specimen. Passed via `--metadata-path`.

All required columns are validated at load time. The pipeline raises a clear error if any required column is missing or contains NaN values.

**Required columns:**

| Column | Description | What happens if missing |
| --- | --- | --- |
| `participant_label` | Unique participant identifier. | Error at load time. |
| `specimen_label` | Unique specimen identifier. Must match the `repertoire_id` column in the participant's sequence file. | Error at load time. |
| `disease` | Disease class label. Each participant must have exactly one disease label. | Error at load time. |
| `malid_cross_validation_fold_id_when_in_test_set` | CV fold assignment (integer). Determines which fold this participant is held out in for testing. | Error at load time. |

**Optional columns:**

| Column | Description | What happens if missing |
| --- | --- | --- |
| `available_gene_loci` | Gene loci available for this specimen (e.g., "TCRB"). If present, used to filter specimens to the requested `--gene-locus`. | All specimens are kept regardless of locus. |

### 4.2 Participant Sequence Files

One AIRR-format TSV file per participant, placed in a single flat directory (no subdirectories). This directory is passed via `--data-dir`.

**File naming:** `part_table_{participant_label}.tsv.gz` (gzip-compressed) or `part_table_{participant_label}.tsv` (uncompressed). The loader checks for the `.tsv.gz` file first, then falls back to `.tsv`.

All required columns are validated on the first participant file processed. The pipeline raises a clear error if any required column is missing.

**Required columns** -- pipeline errors immediately if any of these are absent:

| Column | Description | What it's used for |
| --- | --- | --- |
| `repertoire_id` | Specimen identifier. Must match `specimen_label` in the metadata. A participant file may contain multiple specimens (grouped by `repertoire_id`). | Specimen identification and matching to metadata; downsampling grouping. |
| `v_call` | V gene call with allele (e.g., "TRBV7-2*01"). | V gene extraction (used by all three models for feature computation). |
| `j_call` | J gene call with allele (e.g., "TRBJ2-1*01"). | J gene extraction (used by all three models for feature computation). |
| `cdr3_aa` | CDR3 amino acid sequence. Sequences with non-standard amino acids are dropped during preprocessing. | CDR3 length filtering, sequence clustering (Model 2), ESM-2 embeddings (Model 3). |
| `clone_id` | Clone identifier. | Clone counting (specimens with < 500 clones are dropped) and downsampling (1 sequence per clone). Without this column, all specimens would be dropped as having 0 clones. |

**Quality columns** -- filtering is skipped with a loud warning if absent:

| Column | Description | What happens if missing |
| --- | --- | --- |
| `productive` | Whether the sequence is productive ("T" or "F"). Only productive sequences are kept. | Non-productive sequences (stop codons, frameshifts) are kept, which may add noise to model predictions. Warning logged once. |
| `v_score` | V gene alignment score. Sequences below the threshold (80 for TCR, 200 for BCR) are dropped. | Low-confidence V gene assignments are kept, which may add noise to model predictions. Warning logged once. |

**Optional columns** -- used when present, handled gracefully when absent. A warning is logged once for columns marked with (*):

| Column | Description | What happens if missing |
| --- | --- | --- |
| `sequence` (*) | Full nucleotide sequence. | Deduplication of identical sequences is skipped. Downsampling (1 seq per clone) still handles most redundancy. |
| `num_reads` (*) | Read count. Summed during deduplication. | All sequences assigned num_reads=1. Downsampling picks an arbitrary sequence per clone instead of the highest-read one. |
| `extracted_isotype` (*) | Isotype call. | Isotype-aware deduplication is not performed. Fine for TCR data (single isotype). |
| `replicate_label` | Replicate identifier. | Used with `sequence` for deduplication. If either is absent, deduplication is skipped. |
| `amplification_label` | Amplification protocol label. | Not used in downsampling grouping. Single amplification assumed. |
| `stop_codon` | Whether a stop codon is present ("T" or "F"). | Normalized to uppercase but not used for filtering. |
| `vj_in_frame` | Whether V-J junction is in frame ("T" or "F"). | Normalized to uppercase but not used for filtering. |
| `fwr1_aa` through `fwr4_aa`, `cdr1_aa`, `cdr2_aa` | Framework and CDR region amino acid sequences. | Only used when `--gene-reference-path` is provided. Not required by any current model. |

**Not used by the pipeline** (safe to omit):

| Column | Notes |
| --- | --- |
| `d_call` | D gene call. Present in AIRR files but not read or used by any processing step. |
| `locus` | Gene locus column in sequence files. Locus filtering uses the metadata's `available_gene_loci` column instead. |

### 4.3 Directory Layout and Paths

Set these variables once and `cd` to the project root. All commands in this guide should be run from there.

```bash
# -- Edit these to match your setup --
export MALID_CODE="$HOME/mal-id-lite"                         # the cloned repo
export DATA_DIR="$HOME/project/data/TCR"                      # folder with part_table_* files
export METADATA="$HOME/project/data/metadata.tsv"             # metadata TSV file
export DATASET_NAME="mal-id-orig-data"                        # dataset identifier
export CACHE_DIR="$HOME/project/data_cache/$DATASET_NAME"     # cache output directory

cd "$MALID_CODE"
```

`DATA_DIR` must point to the directory that **directly contains** the `part_table_*` files. In the original Mal-ID dataset, sequence files are organized in a `TCR/` subfolder by locus, so `DATA_DIR` points there -- not to the parent.

`CACHE_DIR` can be anywhere on disk. Keeping it outside the repo is recommended -- the cache can be tens of GB and should not be tracked by git.

**Example directory layout** (original Mal-ID dataset):

```
$HOME/project/
├── data/
│   ├── metadata.tsv                          # 68 KB -- $METADATA
│   └── TCR/                                  # 11 GB -- $DATA_DIR
│       ├── part_table_BFI-0000234.tsv.gz
│       ├── part_table_BFI-0000254.tsv.gz
│       └── ...                               # 542 participant files
└── data_cache/
    └── mal-id-orig-data/                     # ~56 GB -- $CACHE_DIR (created by the pipeline)
```

```bash
# Verify (original dataset)
wc -l "$METADATA"                    # Expected: 616-617 (616 samples + header)
ls "$DATA_DIR"/*.tsv.gz | wc -l      # Expected: 542
```

If your dataset does not use locus subfolders, just point `DATA_DIR` to whichever directory contains the `part_table_*` files.

### 4.4 Data Cache

The `--cache-dir` directory is where the pipeline stores all cached data: preprocessed participant data, fold-level training data, and Model 3 ESM-2 embeddings.

**Cache structure** (original dataset):

```
$CACHE_DIR/
├── participants/       #  4.8 GB -- preprocessed sequences (CLEAN stage)
├── data_folds/         #  11 GB  -- fold-level data (DOWNSAMPLED stage)
├── embeddings/         #  40 GB  -- ESM-2 embeddings (Model 3 only)
└── reports/            #  data quality reports
```

Total cache: ~56 GB (participants + folds + embeddings). Embeddings are only needed for Model 3 and are auto-computed if missing.

The cache can be built in two ways:

1. **Automatically** -- the pipeline builds and caches data on first run when you provide `--data-dir`. Subsequent runs load from cache and don't need `--data-dir`.
2. **Manually** -- you can pre-build the cache and pre-compute embeddings before training. This is useful for separating the heavy I/O and GPU work from the training itself, or when running on different machines (e.g. compute embeddings on a GPU node, train on a CPU node). See [Section 5](#5-pre-computing-data-and-embeddings-optional).

---

## 5. Pre-Computing Data and Embeddings (Optional)

This section covers how to prepare data caches and ESM-2 embeddings **before** running the training pipeline. This step is optional -- the pipeline handles it automatically when `--data-dir` is provided. Pre-computing is useful when you want to:

- **Separate concerns**: run heavy I/O (caching) and GPU work (embeddings) independently of model training
- **Use different machines**: compute embeddings on a GPU node, then transfer the cache to a CPU-only training server
- **Inspect data quality**: the caching script generates detailed preprocessing reports
- **Speed up repeated training runs**: once cached, training starts instantly without re-reading raw files or re-computing embeddings

### 5.1 Building the Data Cache

The caching script preprocesses all participant files and builds fold-level training data:

```bash
python scripts/data/cache_and_report_all_data.py \
    --data-dir "$DATA_DIR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME"
```

This runs in two phases:

1. **Phase 1 -- Participant cache**: reads each raw participant file, applies Stage 1 preprocessing (productive filter, V-score filter, deduplication, gene name cleaning), and saves the cleaned data as `$CACHE_DIR/participants/<label>_clean.parquet`. If the participant cache is already complete, this phase is skipped.

2. **Phase 2 -- Fold cache**: builds cross-validation fold data from the participant cache, applying Stage 2 preprocessing (clone/sequence thresholds, 1 sequence per clone downsampling). Saves as `$CACHE_DIR/data_folds/fold_<id>_<label>_downsampled_sequences.parquet`. If a fold is already cached, it is loaded directly.

**Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `--data-dir` | (required) | Path to directory containing `part_table_*` files |
| `--metadata-path` | (required) | Path to the metadata TSV file |
| `--cache-dir` | `cache/<dataset-name>/` | Cache output directory |
| `--dataset-name` | `mal-id-orig-data` | Dataset identifier |
| `--gene-locus` | `TCR` | Gene locus |
| `--force-reprocess` | off | Delete all existing caches and rebuild from scratch |

**Runtime:** ~30-45 minutes for the original dataset (542 participants).

**Output:**

```
$CACHE_DIR/
├── participants/                    # ~4.8 GB
│   ├── <label>_clean.parquet        # preprocessed sequences per participant
│   ├── <label>_stats.json           # preprocessing stats per participant
│   └── cache_info.json
├── data_folds/                      # ~11 GB
│   ├── fold_0_train_downsampled_sequences.parquet
│   ├── fold_0_train_metadata.csv
│   ├── fold_0_test_downsampled_sequences.parquet
│   ├── fold_0_test_metadata.csv
│   ├── ...                          # (one pair per fold x split)
│   └── cache_info.json
├── metadata.tsv                     # copy of original metadata
├── metadata_processed.tsv           # filtered to participants with data files
└── reports/
    ├── summary_report_<ts>.csv      # high-level statistics
    ├── preprocessing_report_full_<ts>.csv
    ├── global_stats_<ts>.json
    └── caching_log_<ts>.txt
```

Once the data cache is built, all training scripts can run without `--data-dir`:

```bash
# First run (builds cache):
python malid_lite/training/train_ensemble.py \
    --data-dir "$DATA_DIR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR"

# Subsequent runs (uses cache, --data-dir not needed):
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR"
```

### 5.2 Pre-Computing ESM-2 Embeddings

Model 3 requires ESM-2 embeddings for every downsampled CDR3 sequence. These are the most expensive artifact to compute (~40 GB, hours of GPU time). You can pre-compute them independently:

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --device cuda
```

**Prerequisites:** The participant cache must exist (built by the caching script above, or by a prior training run with `--data-dir`). If the participant cache does not exist, provide `--data-dir` and the script will build it automatically.

```bash
# Without existing participant cache (builds it first):
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --data-dir "$DATA_DIR" \
    --device cuda

# With existing participant cache (no --data-dir needed):
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --device cuda
```

**Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `--metadata-path` | (required) | Path to the metadata TSV file |
| `--cache-dir` | `cache/<dataset-name>/` | Cache directory (embeddings saved in `<cache-dir>/embeddings/`) |
| `--dataset-name` | `mal-id-orig-data` | Dataset identifier (used when `--cache-dir` is omitted) |
| `--data-dir` | (none) | Raw data directory. Only needed if participant cache doesn't exist |
| `--device` | (auto) | `cuda`, `mps`, or `cpu`. Auto-detected if omitted |
| `--batch-size` | (auto) | Sequences per batch. Auto-selected per device (mps=64, cuda=4000, cpu=64) |
| `--gene-locus` | `TCR` | Gene locus |
| `--verbose` | 1 | 0=silent, 1=per-participant progress, 2=also per-batch |
| `--verify` | off | Only verify existing embeddings (no computation) |

**Device selection and performance** (original dataset, ~30M sequences):

| Device | Time | Notes |
| --- | --- | --- |
| CUDA (A100/H100) | ~30-60 min | `--batch-size 4000` (default). Reduce if GPU OOM |
| MPS (M4 Max) | ~10 hours | `--batch-size 64` (default) |
| CPU | ~24+ hours | `--batch-size 64` (default) |

Use `--device cuda` for NVIDIA GPUs, `--device mps` for Apple Silicon, `--device cpu` as fallback.

**Resume behavior:** The script automatically skips participants whose embeddings already exist (all three files present and verified: `_embeddings.npy`, `_downsampled.parquet`, `_stats.json`). Orphaned partial files from interrupted runs are cleaned up automatically. No `--resume` flag needed -- just re-run the same command.

**Output** (~40 GB for the original dataset):

```
$CACHE_DIR/embeddings/
├── <participant>_embeddings.npy         # float16, shape (N, 640)
├── <participant>_downsampled.parquet    # exact sequences that were embedded
├── <participant>_stats.json             # per-participant metadata
├── ...                                  # (one triplet per participant)
├── cache_info.json                      # run-level metadata
├── embedding_report_<ts>.md            # human-readable summary
└── embedding_log_<ts>.log              # full log
```

**Verifying embeddings:**

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --verify
```

This checks that every `_embeddings.npy` has matching companion files, correct shape `(N, 640)`, float16 dtype, no NaN/Inf values, and row counts match the stats JSON. Exits with code 0 if all checks pass.

### 5.3 Cache Management

```bash
python scripts/data/manage_cache.py info                 # view cache status
python scripts/data/manage_cache.py clear-participants   # clear participant cache
python scripts/data/manage_cache.py clear-folds          # clear fold cache only
python scripts/data/manage_cache.py clear-embeddings     # clear embeddings only
python scripts/data/manage_cache.py clear-all            # clear everything
```

All `clear-*` commands prompt for confirmation. Pass `-y` to skip:

```bash
python scripts/data/manage_cache.py clear-all --cache-dir "$CACHE_DIR" -y
```

**When to clear caches:**

| Situation | Command |
| --- | --- |
| Changed CLEAN preprocessing logic | `clear-participants` (fold cache auto-rebuilds) |
| Changed DOWNSAMPLED preprocessing logic | `clear-folds` |
| Changed ESM-2 model or embedding approach | `clear-embeddings` |
| Changed source metadata | `clear-all` |
| Major code changes | `clear-all` |

**Recommended preparation workflow:**

```bash
# Step 1: Build the data cache (CPU-bound, ~30-45 min)
python scripts/data/cache_and_report_all_data.py \
    --data-dir "$DATA_DIR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR"

# Step 2: Pre-compute ESM-2 embeddings (GPU-bound, ~30 min on A100)
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --device cuda

# Step 3: Verify everything
python scripts/data/manage_cache.py info --cache-dir "$CACHE_DIR"
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --verify

# Step 4: Train (no --data-dir needed, no GPU needed for training itself)
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --classification-mode multiclass \
    --n-jobs 8
```

---

## 6. Running the Full Pipeline (Ensemble)

The **recommended way** to train the full pipeline is with `train_ensemble.py`. It handles everything end-to-end: data caching, ESM-2 embedding computation, base model training (Models 1-3), and ensemble meta-learner training.

If you have already pre-built the data cache and embeddings (see [Section 5](#5-pre-computing-data-and-embeddings-optional)), the ensemble will detect them and skip those steps automatically.

### 6.1 Quick Start

```bash
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 4
```

On first run, provide `--data-dir` so the cache can be built. Subsequent runs can omit it.

The ensemble will:

1. Build the data cache if it doesn't exist
2. Train Models 1, 2, and 3 sequentially (or load them if already trained)
3. Auto-compute ESM-2 embeddings for Model 3 if missing
4. Combine base model predictions and train the ensemble meta-learner
5. Evaluate and write results

### 6.2 Ensemble-Specific Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--models` | `1 2 3` | Which base models to include (e.g., `--models 1 3`) |
| `--retrain-base-models` | off | Force retrain ALL base models from scratch |
| `--retrain-models` | (none) | Force retrain specific models (e.g., `--retrain-models 2 3`) |
| `--model2-abstention-strategy` | `ensemble_abstain` | How to handle Model 2 abstentions (see below) |
| `--output-suffix` | (none) | Suffix for the output directory name |

**Model 2 abstention strategies:**

Model 2 can abstain from prediction when a specimen has no significant cluster matches. The ensemble handles this via `--model2-abstention-strategy`:

| Strategy | Behavior |
| --- | --- |
| `ensemble_abstain` | Drop the specimen from ensemble prediction (default) |
| `fill_0.5` | Fill with uninformative prior (0.5) |
| `fill_models13_mean` | Fill with the mean of Models 1 and 3 predictions |

**Passing model-specific parameters through the ensemble:**

The ensemble forwards model-specific arguments with a prefix. For example:

```bash
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --model1-n-pcs 20 \
    --model3-aggregation-strategy paper_best \
    --model3-device cuda
```

Full list of forwarded arguments:
- Model 1: `--model1-n-pcs`, `--model1-l1-ratio`, `--model1-model-name`, `--model1-suffix`
- Model 2: `--model2-p-values`, `--model2-retrain-on-full-train`, `--model2-sequence-identity-threshold`, `--model2-suffix`
- Model 3: `--model3-aggregation-strategy`, `--model3-n-estimators-stage1`, `--model3-n-estimators-stage2`, `--model3-entropy-max-fraction`, `--model3-entropy-bottom-percentile`, `--model3-device`, `--model3-embedding-batch-size`, `--model3-embedding-dir`, `--model3-no-cache-embeddings`, `--model3-suffix`, `--model3-tuning-strategies`, `--model3-tuning-cv-splits`, `--model3-tuning-entropy-max-fractions`, `--model3-tuning-entropy-percentiles`

### 6.3 Ensemble Output

```
trained_models/<dataset>/cv_ensemble/
├── base_models/TCR/
│   ├── model1/multiclass/          # Model 1 artifacts
│   ├── model2/multiclass/          # Model 2 artifacts
│   └── model3/multiclass/          # Model 3 artifacts
└── ensemble/TCR/multiclass/
    ├── fold_0_ridge_cv_metamodel.joblib       # fitted meta-learner
    ├── fold_0_metamodel_config.json           # feature columns, classes
    ├── fold_0_ensemble_results.json           # per-fold metrics
    ├── fold_0_feature_matrix_val.csv          # validation features
    ├── fold_0_feature_matrix_test.csv         # test features
    ├── fold_1_...
    ├── fold_2_...
    ├── summary_<timestamp>.json
    ├── RESULTS_<timestamp>.md
    └── ensemble_training.log
```

### 6.4 Using Pre-Trained Base Models

If base models were already trained (by a previous ensemble run or by individual training scripts), the ensemble will detect them and skip training. It checks for a valid `summary_*.json` file and verifies the saved configuration matches the current CLI arguments.

To force retraining specific models:

```bash
# Retrain only Model 2 and Model 3, keep existing Model 1
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --retrain-models 2 3 \
    --n-jobs 8
```

---

## 7. Running Individual Models

While the ensemble can train everything end-to-end, you can also run each model independently. This is useful for debugging, experimentation, or when you only need one model's results.

Standalone model artifacts are saved under `cv_single_model/` (vs. `cv_ensemble/base_models/` when trained via the ensemble).

All individual training scripts support `--data-dir` for the first run (to build the cache) and can omit it on subsequent runs when the cache exists. See [Section 5](#5-pre-computing-data-and-embeddings-optional) for pre-building the cache.

### 7.1 Model 1 -- Repertoire Classifier

PCA of gene frequencies followed by elastic net logistic regression. The fastest model.

```bash
python malid_lite/training/train_model1.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass
```

**Runtime:** ~2-5 minutes. **Memory:** < 4 GB.

Output: `trained_models/<dataset>/cv_single_model/model1/multiclass/TCR/`

### 7.2 Model 2 -- Convergent Cluster Classifier

CDR3 sequence clustering with Fisher's exact test, then logistic regression on cluster features. Can abstain when a specimen has no significant cluster matches.

```bash
python malid_lite/training/train_model2.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 4
```

**Runtime:** ~1-3 hours (clustering dominates). **Memory:** ~10-30 GB depending on `--n-jobs`.

Output: `trained_models/<dataset>/cv_single_model/model2/multiclass/TCR/`

### 7.3 ESM-2 Embeddings (Pre-Computation)

Model 3 requires ESM-2 embeddings for every downsampled CDR3 sequence. Embeddings are auto-computed if missing (by both the ensemble and standalone Model 3 training). You can also pre-compute them separately -- see [Section 5.2](#52-pre-computing-esm-2-embeddings) for full details, arguments, and performance estimates.

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --device cuda
```

### 7.4 Model 3 -- Sequence-Level Classifier

Two-stage model: (1) per-V-gene binary classifiers on ESM-2 embeddings produce sequence-level disease probabilities, (2) aggregates to specimen-level features and trains a random forest. This is the most compute-intensive model.

```bash
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8
```

For long-running training, use `tmux` or `nohup`:

```bash
nohup python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    > training_model3.log 2>&1 &
```

**Runtime and n-jobs tuning:**

| Machine | RAM | Recommended `--n-jobs` | Approx. time (3 folds) |
| --- | --- | --- | --- |
| Laptop (16 CPU cores) | 64 GB | up to 2 | ~30-40 hours |
| Server (256 CPU cores) | 1 TB | up to 200 | ~12 hours |

Stage 1 dominates runtime (>90%). **Memory:** ~30-50 GB per fold for the main process, plus ~2-8 GB per worker.

Output: `trained_models/<dataset>/cv_single_model/model3/multiclass/TCR/`

---

## 8. Resume Logic

All training scripts support `--resume` for recovering from crashes or interruptions.

### 8.1 How Resume Works

**Individual models (Models 1-3):** Resume checks each fold for complete artifacts on disk. Complete folds are skipped and their results are reloaded. Incomplete folds have their partial artifacts deleted and are retrained from scratch.

Saved artifacts include a `_meta` block with the training parameters used. On resume, parameters that were saved in the original run are validated against the current CLI arguments. If any saved parameter doesn't match, training raises a `ValueError` to prevent accidentally mixing results from different configurations. Parameters added in newer code versions that weren't present in the saved artifact are ignored.

**Ensemble:** The ensemble detects each base model's state independently:

| State | Condition | Action |
| --- | --- | --- |
| **LOAD** | Complete model found (`summary_*.json` + matching config) | Skip training, load predictions directly |
| **TRAIN** | No artifacts or incomplete without `--resume` | Train from scratch |
| **RESUME** | Partial artifacts found with `--resume` | Resume from last checkpoint |

### 8.2 Resume Examples

**Resume after a crash (any model or ensemble):**

```bash
# Individual model
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume

# Ensemble
python malid_lite/training/train_ensemble.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume
```

**Model 3 stage-specific resume:**

Model 3 has additional resume modes since its two-stage training is expensive:

- `--resume-from-stage2` -- Reload saved Stage 1 models, retrain Stage 2 from scratch. Use this to change Stage-2-only parameters (aggregation strategy, entropy threshold, n_estimators_stage2) without repeating the expensive Stage 1 training.
- `--resume-from-evaluation` -- Reload both stages, re-run evaluation only.

```bash
# Retrain Stage 2 with a different aggregation strategy
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume-from-stage2 \
    --aggregation-strategy paper_best
```

### 8.3 Important Notes

- `--resume` and `--retrain-base-models` / `--retrain-models` are **mutually exclusive** in the ensemble.
- Resume validates saved parameters against CLI arguments. Changing a parameter that was used in the original run will raise an error. To change parameters, use `--retrain-models` instead.
- The embedding computation script (`compute_model3_embeddings`) is inherently resumable -- it skips already-computed participants automatically, no `--resume` flag needed.

---

## 9. Verifying Results

All training scripts produce a human-readable `RESULTS_<timestamp>.md` and a machine-readable `summary_<timestamp>.json` in the output directory.

```bash
# Read the results summary for any model
cat trained_models/$DATASET_NAME/cv_single_model/model1/multiclass/TCR/RESULTS_*.md

# Ensemble results
cat trained_models/$DATASET_NAME/cv_ensemble/ensemble/TCR/multiclass/RESULTS_*.md
```

**What to look for:**

- All folds completed (3 for the original dataset)
- No warnings about missing data or failed folds in the log
- Model 2: abstention rate is reported
- Model 3: Stage 1 reports 168 binary classifier jobs per fold (28 V-gene groups x 6 classes, for multiclass on the original dataset)

**Verify cache status:**

```bash
python scripts/data/manage_cache.py info --cache-dir "$CACHE_DIR"
```

**Verify embeddings:**

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --verify
```

---

## Appendix A: Hardware and Performance

### Minimum Requirements

| Component | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| RAM | 4 GB | 16 GB | 64 GB |
| CPU cores | 1 | 4+ | 4+ |
| GPU | -- | -- | For embeddings only |
| Disk | 20 GB | 20 GB | 60 GB (with embeddings) |

### Recommended (original dataset: 542 participants, ~30M sequences)

- **RAM:** 128+ GB (comfortable `--n-jobs 8` for Models 2 and 3)
- **CPU:** 16+ cores
- **GPU:** Any CUDA GPU for embeddings (A100/H100 for speed)
- **Disk:** 100+ GB (cache + embeddings + model artifacts)

### Runtime Estimates (original dataset, 3 folds)

| Step | Laptop (M4 Max, n_jobs=2) | Server (64 cores, n_jobs=16) |
| --- | --- | --- |
| Build cache | ~45 min | ~30 min |
| Compute embeddings | ~10 hours (MPS) | ~45 min (A100) |
| Model 1 | ~3 min | ~2 min |
| Model 2 | ~2 hours | ~30 min |
| Model 3 | ~35 hours | ~3 hours |
| Ensemble | ~1 min | ~1 min |

---

## Appendix B: Troubleshooting

### Process killed with no error message (OOM)

```bash
# Linux
dmesg | grep -i "oom\|kill" | tail -20

# macOS
log show --predicate 'eventMessage contains "Jetsam"' --last 1h
```

Fix: reduce `--n-jobs` or free memory.

### "Loaded 0 sequences from cache"

Fold cache is empty or corrupted. Delete and rebuild:

```bash
python scripts/data/manage_cache.py clear-folds --cache-dir "$CACHE_DIR" -y
# Re-run training with --data-dir to rebuild
```

### "No embedding file found for participant X"

Embedding cache is incomplete. Re-run the embedding script -- it skips already-computed participants:

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --device cuda
```

### PyTorch: `undefined symbol: iJIT_NotifyEvent`

MKL 2025+ breaks a PyTorch symbol. Fix:

```bash
conda install "mkl<2025.0.0"
```

### PyTorch: `No module named 'torch'` after conda install

Force reinstall:

```bash
conda install pytorch cpuonly -c pytorch --force-reinstall
```

If that fails, use pip:

```bash
conda remove pytorch cpuonly --force
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### glmnet install or import error

Use `python-glmnet` (Replica HQ fork), not `glmnet`:

```bash
pip install python-glmnet
```

`pip install glmnet` installs an old version that fails on Python 3.12+. `conda install glmnet` installs the R package, not the Python one.

If building from source fails, install a Fortran compiler: `sudo apt install gfortran` (Linux) or `brew install gcc` (macOS).

### CUDA out of memory during embedding computation

Reduce the batch size: `--batch-size 1000` (or lower).

### Model 3 crashed mid-training

Use `--resume` to pick up where it left off (see [Resume Logic](#8-resume-logic)).

### Training seems stuck

Model 3 Stage 1 individual jobs can take 10-30 minutes each with low `--n-jobs`. If workers show 100% CPU, training is proceeding normally -- it's just slow.
