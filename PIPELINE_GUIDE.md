# Mal-ID-Lite Pipeline Guide

How to set up, run, and verify each step of the Mal-ID-Lite training pipeline.

This guide covers the original Mal-ID dataset with 3-fold cross-validation.
Each model is trained and evaluated independently. Ensemble training and
external test set evaluation will be added in a future update.

---

## Table of Contents

- [0. Overview](#0-overview)
- [1. Prerequisites](#1-prerequisites)
- [2. Data Setup](#2-data-setup)
- [3. Build the Data Cache](#3-build-the-data-cache)
- [4. Model 1 — Repertoire Classifier](#4-model-1--repertoire-classifier)
- [5. Model 2 — Convergent Cluster Classifier](#5-model-2--convergent-cluster-classifier)
- [6. Compute ESM-2 Embeddings](#6-compute-esm-2-embeddings)
- [7. Model 3 — Sequence-Level Classifier](#7-model-3--sequence-level-classifier)
- [Appendix A: Classification Modes](#appendix-a-classification-modes)
- [Appendix B: Hardware and Performance](#appendix-b-hardware-and-performance)
- [Appendix C: Troubleshooting](#appendix-c-troubleshooting)

---

## 0. Overview

Mal-ID-Lite classifies disease from TCR repertoire data using three
independent models, each capturing signal at a different level:

```
Raw AIRR data + Metadata
        |
        v
  [Data Cache]  .............. Step 3: preprocess + cache
   |    |    |
   v    v    v
  M1   M2   M3 (ESM-2 embeddings computed first)
   |    |    |
   v    v    v
 Per-fold results (accuracy, AUROC, confusion matrices)
```


| Model       | What it captures                                              | Approach                                                                        |
| ----------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Model 1** | Repertoire-level V/J gene usage and CDR3 length distributions | Logistic regression (elastic net) on PCA of gene frequencies                    |
| **Model 2** | Convergent CDR3 clusters shared across patients               | Fisher exact test for disease-associated clusters, then logistic regression     |
| **Model 3** | Individual CDR3 sequence features via protein language model  | ESM-2 embeddings → per-V-gene binary classifiers → specimen-level random forest |


Each model is trained and evaluated using **3-fold participant-level
cross-validation** (stratified by disease). The models are independent —
you can train them in any order (except Model 3 requires pre-computed
embeddings).

---

## 1. Prerequisites

### 1.1 Create the conda environment

```bash
conda create -n mal_id_lite python=3.12
conda activate mal_id_lite
```

### 1.2 Install dependencies

```bash
# Core scientific stack (conda-forge)
conda install -c conda-forge pandas numpy pyarrow scikit-learn scipy psutil pytest

# python-glmnet (pip only — not on conda-forge)
pip install python-glmnet

# PyTorch — use the pytorch channel (bundles the right CUDA libraries)
# If the machine has a CUDA GPU:
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
# NOTE: pytorch-cuda version must be <= your driver's CUDA version.
# Run `nvidia-smi` and check "CUDA Version" in the top right.
# Common values: 12.4, 12.1, 11.8. Pick the highest that fits.

# If no GPU (CPU only):
conda install pytorch cpuonly -c pytorch

# fair-esm (pip only — not on conda)
pip install fair-esm
```

**Alternative (not recommended):** install everything from requirements.txt
using pip only. Conda is preferred because it handles compiled dependencies
(CUDA for torch) and version compatibility automatically. With pip you must
manage these yourself:

```bash
pip install -r requirements.txt
```

### 1.3 Verify installation

```bash
python -c "
import pandas, numpy, sklearn, glmnet, torch, esm, scipy
print('All imports OK')
print(f'  pandas {pandas.__version__}')
print(f'  numpy {numpy.__version__}')
print(f'  sklearn {sklearn.__version__}')
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"
```

---

## 2. Data Setup

### 2.1 Define paths

Set these variables once. All commands in this guide reference them.

```bash
# -- Edit these to match your setup --
export MALID_CODE="$HOME/mal-id-lite"           # the cloned repo
export MALID_DATA="$HOME/mal-id-data"           # raw data directory (outside the repo)

# -- Derived paths (no need to edit) --
export METADATA="$MALID_DATA/metadata.tsv"
export CACHE_DIR="$MALID_CODE/cache/mal-id-orig-data"
export DATASET_NAME="mal-id-orig-data"
```

### 2.2 Raw data

```
$MALID_DATA/
├── metadata.tsv                              # 68 KB — required by all models
└── TCR/                                      # 11 GB — raw AIRR-format files
    ├── part_table_BFI-0000234.tsv.gz
    ├── part_table_BFI-0000254.tsv.gz
    ├── ...                                   # 542 participant files
```

### 2.3 Verify raw data

```bash
# Check metadata
wc -l "$METADATA"
# Expected: 616 or 617 (616 samples + header; wc -l may show 616 if no trailing newline)

# Check raw AIRR files
ls "$MALID_DATA/TCR/"*.tsv.gz | wc -l
# Expected: 542
```

---

## 3. Build the Data Cache

The cache preprocesses all 542 participants and creates fold-level files
for fast training. It is stored inside the repo (`$MALID_CODE/cache/`,
git-ignored).

There are two ways to build the cache:

### Option A: Let the training scripts build it on first run

Each training script accepts `--data-dir` pointing to the raw AIRR files.
On first run, if no cache exists, the script automatically preprocesses
the data and creates the cache. This is the simplest approach — just add
`--data-dir` to your first training command and the cache will be built
before training starts.

```bash
# Example: first Model 1 run also builds the cache
python malid_lite/training/train_model1.py \
    --metadata-path "$METADATA" \
    --data-dir "$MALID_DATA/TCR" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass
```

Subsequent runs (including other models) will use the cache automatically
and don't need `--data-dir`.

### Option B: Dedicated cache-building script

The script `scripts/data/cache_and_report_all_data.py` builds the full
cache and generates data quality reports:

```bash
cd "$MALID_CODE"

python scripts/data/cache_and_report_all_data.py \
    --data-dir "$MALID_DATA/TCR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME"
```

**Expected runtime:** ~30-60 minutes (depends on disk speed).

### Cache structure

Once built, the cache looks like this:

```
$MALID_CODE/cache/
└── mal-id-orig-data/
    ├── participants/                         #  4.8 GB — preprocessed sequences (CLEAN stage)
    │   ├── BFI-0000234_clean.parquet
    │   ├── BFI-0000234_stats.json
    │   ├── ...                               # 542 participants x 2 files each
    │   └── cache_info.json
    ├── data_folds/                           #  11 GB — fold-level data (DOWNSAMPLED stage)
    │   ├── fold_0_train_downsampled_sequences.parquet
    │   ├── fold_0_train_downsampled_metadata.csv
    │   ├── fold_0_test_downsampled_sequences.parquet
    │   ├── fold_0_test_downsampled_metadata.csv
    │   ├── fold_1_train_...                  # 3 folds x 2 splits x 2 files
    │   ├── fold_2_...
    │   └── cache_info.json
    ├── embeddings/                           #  40 GB — ESM-2 embeddings (Model 3 only)
    │   ├── BFI-0000234_embeddings.npy
    │   ├── BFI-0000234_downsampled.parquet
    │   ├── BFI-0000234_stats.json
    │   ├── ...                               # 542 participants x 3 files each
    │   └── cache_info.json
    └── reports/                              # data quality reports
```

**Total cache size: ~56 GB** (participants + folds + embeddings).

Embeddings are only needed for Model 3. You can train Models 1 and 2
without them. See [Step 6](#6-compute-esm-2-embeddings) for the embedding
computation command.

### Verify the cache

```bash
# Quick summary (file counts, sizes, creation dates)
python scripts/data/manage_cache.py info --cache-dir "$CACHE_DIR"
# Expected: 542 participants, 6 fold files, 542 embedding files (if computed)

# Or check manually
echo "Participants: $(ls "$CACHE_DIR/participants/"*.parquet 2>/dev/null | wc -l)"
echo "Fold files:   $(ls "$CACHE_DIR/data_folds/"*.parquet 2>/dev/null | wc -l)"
echo "Embeddings:   $(ls "$CACHE_DIR/embeddings/"*.npy 2>/dev/null | wc -l)"
```

### ESM-2 Embeddings

Embeddings are only needed for Model 3. You can train Models 1 and 2
without them. See [Step 6](#6-compute-esm-2-embeddings) for the embedding
computation command.

---

## 4. Model 1 — Repertoire Classifier

### What it does

Model 1 represents each specimen (patient sample) as a vector of V-gene
frequencies, J-gene frequencies, and CDR3 amino acid length distributions,
then runs PCA followed by logistic regression with elastic net
regularization (L1 via `glmnet`). It is the fastest model to train.

### Train (multiclass, all folds)

```bash
cd "$MALID_CODE"

python malid_lite/training/train_model1.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass
```

**Expected runtime:** ~2-5 minutes total (all 3 folds).

**Expected memory:** < 4 GB.

### Output

```
trained_models/mal-id-orig-data/model1/multiclass/TCR/
├── fold_0_lasso_cv_model.pkl             # fitted model
├── fold_0_lasso_cv_v_genes.json          # V genes used (after filtering)
├── fold_0_lasso_cv_results.json          # per-fold metrics
├── fold_1_lasso_cv_model.pkl
├── fold_1_lasso_cv_v_genes.json
├── fold_1_lasso_cv_results.json
├── fold_2_lasso_cv_model.pkl
├── fold_2_lasso_cv_v_genes.json
├── fold_2_lasso_cv_results.json
├── lasso_cv_multiclass_predictions.csv   # all test predictions (all folds)
├── summary_<timestamp>.json              # aggregated metrics
├── RESULTS_<timestamp>.md                # human-readable results report
└── training_<timestamp>.log              # full training log
```

### Verify results

```bash
# Quick check: read the results summary
cat trained_models/$DATASET_NAME/model1/multiclass/TCR/RESULTS_*.md

# Programmatic check: extract key metrics from summary JSON
python -c "
import json, glob
f = glob.glob('trained_models/$DATASET_NAME/model1/multiclass/TCR/summary_*.json')[0]
d = json.load(open(f))
m = d['aggregated_by_pair']['multiclass']['lasso_cv']
print(f\"Accuracy:      {m['accuracy_global']:.3f}\")
auroc = m['auroc_ovo_weighted']
auprc = m['auprc_ovo_weighted']
print(f\"AUROC (OvO):   {auroc['mean']:.3f} +/- {auroc['std']:.3f}\")
print(f\"AUPRC (OvO):   {auprc['mean']:.3f} +/- {auprc['std']:.3f}\")
"
```

**What to look for:**

- All 3 folds completed (3 `fold_*_results.json` files)
- `RESULTS_*.md` contains confusion matrices for each fold
- No warnings about missing data or failed folds in the log

---

## 5. Model 2 — Convergent Cluster Classifier

### What it does

Model 2 identifies CDR3 amino acid sequences that cluster together
(edit distance <= 2) more often than expected by chance. It uses Fisher's
exact test to find disease-associated clusters, then trains a logistic
regression on the cluster presence/absence matrix. Specimens with no
significant clusters abstain from prediction.

### Train (multiclass, all folds)

```bash
cd "$MALID_CODE"

python malid_lite/training/train_model2.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 4
```

**Expected runtime:** ~1-3 hours (dominated by the clustering phase).
On a server with many cores, increase `--n-jobs` (e.g., `--n-jobs 16`).

**Expected memory:** ~10-30 GB depending on `--n-jobs`. Each worker holds
pairwise distance matrices for its assigned (V gene, J gene, CDR3 length)
group.

### Output

```
trained_models/mal-id-orig-data/model2/multiclass/TCR/
├── fold_0_clusters.joblib                    # cluster centroids
├── fold_0_lasso_cv_model_split1.joblib       # fitted GLM
├── fold_0_lasso_cv_p_value.joblib            # selected p-value threshold
├── fold_0_lasso_cv_results_split1.json       # per-fold metrics
├── fold_1_clusters.joblib
├── fold_1_lasso_cv_model_split1.joblib
├── fold_1_lasso_cv_p_value.joblib
├── fold_1_lasso_cv_results_split1.json
├── fold_2_clusters.joblib
├── fold_2_lasso_cv_model_split1.joblib
├── fold_2_lasso_cv_p_value.joblib
├── fold_2_lasso_cv_results_split1.json
├── lasso_cv_multiclass_predictions.csv
├── summary_<timestamp>.json
├── RESULTS_<timestamp>.md
└── training_<timestamp>.log
```

### Verify results

```bash
# Quick check
cat trained_models/$DATASET_NAME/model2/multiclass/TCR/RESULTS_*.md

# Programmatic check
python -c "
import json, glob
f = glob.glob('trained_models/$DATASET_NAME/model2/multiclass/TCR/summary_*.json')[0]
d = json.load(open(f))
m = d['aggregated_by_pair']['multiclass']['lasso_cv']
print(f\"Accuracy:      {m['accuracy_global']:.3f}\")
auroc = m['auroc_ovo_weighted']
print(f\"AUROC (OvO):   {auroc['mean']:.3f} +/- {auroc['std']:.3f}\")
"
```

**What to look for:**

- All 3 folds completed
- Abstention rate is reported (Model 2 can abstain when a specimen has no
significant cluster matches)
- AUROC should be in the 0.8-0.9 range for the original dataset
- Check the log for warnings about empty cluster groups or degenerate folds

---

## 6. Compute ESM-2 Embeddings

### What it does

Pre-computes ESM-2 protein language model embeddings for every downsampled
CDR3 sequence. These are stored as per-participant `.npy` files and loaded
by Model 3 during training. This is a one-time computation — once
embeddings are cached, Model 3 loads them directly.

> **Skip this step if you copied the pre-built embedding cache**
> (`$CACHE_DIR/embeddings/` with 542 `.npy` files).

### Compute

```bash
cd "$MALID_CODE"

python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --device cuda
```

Use `--device cuda` on a GPU server, `--device mps` on Apple Silicon,
`--device cpu` as fallback.

For CUDA, you can increase the batch size for faster throughput:

```bash
    --batch-size 4000    # CUDA default; reduce if GPU OOM
```

**Expected runtime:**


| Device           | Time (30M sequences) | Notes           |
| ---------------- | -------------------- | --------------- |
| CUDA (A100/H100) | ~30-60 min           | batch_size=4000 |
| MPS (M4 Max)     | ~10 hours            | batch_size=64   |
| CPU              | ~24+ hours           | batch_size=64   |


**Expected output size:** ~40 GB (float16, 640 dimensions per sequence).

### Verify

```bash
# Check file count
ls "$CACHE_DIR/embeddings/"*.npy | wc -l
# Expected: 542

# Run the built-in verification (checks row alignment between
# embeddings and their source sequences)
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --verify
```

The `--verify` flag checks every participant's embedding file for:

- Correct shape (N sequences x 640 dimensions)
- Row alignment with the corresponding downsampled parquet file
- No NaN or Inf values

---

## 7. Model 3 — Sequence-Level Classifier

### What it does

Model 3 operates at the individual sequence level using ESM-2 embeddings:

- **Stage 1:** For each V-gene group, trains one-vs-rest binary classifiers
(ridge regression via `glmnet`) on sequence embeddings. Produces per-sequence
disease probability vectors.
- **Stage 2:** Aggregates sequence-level probabilities to specimen-level
features (configurable strategy; default for TCR: entropy-based filtering
that keeps only high-confidence sequences), then trains a random forest
for final specimen classification.

This is the most compute-intensive model. It processes ~10-17 million
sequences per fold.

### Model 3 options

Beyond the standard options (`--metadata-path`, `--cache-dir`, etc.),
Model 3 has several specific parameters:

**Aggregation strategy** (`--aggregation-strategy`):

Controls how per-sequence probabilities from Stage 1 are aggregated into
specimen-level features for Stage 2. The default `auto` selects the
paper-best strategy per locus:

| Strategy                         | Description                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| `auto` (default)                 | TCR = `entropy_cutoff` (0.20), BCR = `mean`                     |
| `entropy_cutoff`                 | Keep sequences with entropy < (1 - threshold) * max; configurable via `--entropy-threshold` |
| `entropy_ten_percent_cutoff`     | Legacy: fixed 0.10 threshold (keep below 90% of max entropy)    |
| `entropy_twenty_percent_cutoff`  | Legacy: fixed 0.20 threshold (keep below 80% of max entropy)    |
| `mean`                           | Weighted mean of all sequences                                   |
| `median`                         | Weighted median                                                  |
| `trim_bottom_five_percent`       | Drop lowest-weight 5% then weighted mean                         |

**Entropy threshold** (`--entropy-threshold`):

Only used with `--aggregation-strategy entropy_cutoff`. Sets the
fraction of maximum entropy to cut off. For example, `0.20` (the
default) means keep sequences whose entropy is below 80% of the
maximum possible entropy. Lower values are more aggressive (keep
fewer, more confident sequences).

```bash
# Example: use a stricter entropy cutoff
--aggregation-strategy entropy_cutoff --entropy-threshold 0.30
```

**Other options:**

| Option                                | Default | Description                                                            |
| ------------------------------------- | ------- | ---------------------------------------------------------------------- |
| `--n-jobs`                            | 4       | Parallel workers for Stage 1 group training and Stage 2 classifiers    |
| `--n-estimators-stage1`               | 100     | RF trees in Stage 1 (BCR only; TCR uses glmnet ridge)                  |
| `--n-estimators-stage2`               | 100     | RF trees in Stage 2                                                    |
| `--reweigh-by-subset-frequencies`     | (auto)  | Multiply Stage 2 features by V-gene group frequencies (TCR default: on)|
| `--fold-ids`                          | all     | Train only specific folds, e.g. `--fold-ids 0 2`                      |
| `--verbose`                           | 1       | 0 = silent, 1 = progress, 2 = diagnostics                  |


### Train (multiclass, all folds)

For long-running training, use `nohup` or run inside `tmux`:

```bash
cd "$MALID_CODE"

# Option A: run in foreground (inside tmux)
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8

# Option B: run in background with nohup
nohup python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    > training_model3.log 2>&1 &

echo "PID: $!"
```

### Resuming after interruption

Model 3 saves per-fold artifacts after each stage completes. Three
resume modes let you pick up from different points:

**`--resume`** — Skip completed work after a crash or interruption.
Detects which stages and folds have saved artifacts on disk and picks
up where it left off:

```bash
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume
```

Resume detects completed stages per fold and skips them:

- `fold_<id>_stage1.pkl` exists -> loads Stage 1, skips to Stage 2
- `fold_<id>_stage1.pkl` + `stage2.pkl` exist -> skips to evaluation
- All 4 artifacts exist -> skips fold entirely, reloads results

**`--resume-from-stage2`** — Load saved Stage 1 models and retrain
Stage 2 from scratch. Use this when you want to change Stage-2-only
parameters (aggregation strategy, entropy threshold, n_estimators_stage2,
reweigh_by_subset_frequencies) without re-running the expensive Stage 1
training. Existing Stage 2, prediction, and result artifacts are
automatically deleted and regenerated.

```bash
# Example: retrain Stage 2 with a different entropy threshold
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume-from-stage2 \
    --aggregation-strategy entropy_cutoff \
    --entropy-threshold 0.15 \
    --verbose 2
```

**`--resume-from-evaluation`** — Load saved Stage 1 and Stage 2 models
and re-run only the evaluation phase. Useful for regenerating results
or predictions without retraining anything. Requires both Stage 1 and
Stage 2 artifacts to exist.

```bash
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --resume-from-evaluation \
    --verbose 2
```

**Tuning `--n-jobs`:**


| Machine          | RAM     | Recommended `--n-jobs` | Approx. time (3 folds) |
| ---------------- | ------- | ---------------------- | ---------------------- |
| Laptop (16 CPU cores) | 64 GB   | up to 2                      | ~30-40 hours           |
| Server (256 CPU cores) | 1 TB | up to 200                  | ~12 hours           |

Times are rough estimates. Stage 1 dominates the runtime (>90%).

**Expected memory:** ~30-50 GB per fold for the main process (loading
embeddings + feature arrays), plus ~2-8 GB per worker (job).

With a 64 GB RAM machine, memory will be the bottleneck. It is advised to run the code on a server with more RAM and higher n-jobs for faster computation.

### Monitor progress (if running in background)

```bash
# Check if still running
ps aux | grep train_model3 | grep -v grep

# Follow the log
tail -f training_model3.log

# Check how many Stage 1 jobs are done (168 total per fold for multiclass)
grep -c "Done.*tasks" training_model3.log
```

### Output

```
trained_models/mal-id-orig-data/model3/multiclass/TCR/
├── fold_0_stage1.pkl                         # Stage 1 classifiers (per V-gene group)
├── fold_0_stage2.pkl                         # Stage 2 random forest
├── fold_0_results.json                       # per-fold metrics
├── fold_0_predictions.pkl                    # raw predictions (for ensemble use)
├── fold_1_stage1.pkl
├── fold_1_stage2.pkl
├── fold_1_results.json
├── fold_1_predictions.pkl
├── fold_2_stage1.pkl
├── fold_2_stage2.pkl
├── fold_2_results.json
├── fold_2_predictions.pkl
├── model3_multiclass_predictions.csv         # all test predictions (all folds)
├── summary_<timestamp>.json
├── RESULTS_<timestamp>.md
└── training_<timestamp>.log
```

### Verify results

```bash
# Quick check
cat trained_models/$DATASET_NAME/model3/multiclass/TCR/RESULTS_*.md

# Programmatic check
python -c "
import json, glob
f = glob.glob('trained_models/$DATASET_NAME/model3/multiclass/TCR/summary_*.json')[0]
d = json.load(open(f))
m = d['aggregated_by_pair']['multiclass']['model3']
print(f\"Accuracy:      {m['accuracy_global']:.3f}\")
auroc = m['auroc_ovo_weighted']
auprc = m['auprc_ovo_weighted']
print(f\"AUROC (OvO):   {auroc['mean']:.3f} +/- {auroc['std']:.3f}\")
print(f\"AUPRC (OvO):   {auprc['mean']:.3f} +/- {auprc['std']:.3f}\")
"
```

**What to look for:**

- All 3 folds completed (check for "Fold 2" in the log)
- Stage 1 reports 168 binary classifier jobs per fold (28 V-gene groups x 6 classes)
- Stage 2 trains and evaluates successfully
- No OOM kills (check `dmesg | grep -i kill` on Linux)

---

## Appendix A: Classification Modes

All three training scripts support three classification modes via
`--classification-mode`:

### Multiclass (default)

A single N-class classifier trained on all disease classes simultaneously.
This is the standard mode for the original Mal-ID dataset (6 classes:
Healthy/Background, HIV, T1D, Lupus, Covid19, Influenza).

```bash
--classification-mode multiclass
```

### Binary

One binary classifier for a single disease-vs-reference pair.

```bash
# Two-class dataset (auto-detects classes):
--classification-mode binary

# Pick one disease from an N-class dataset:
--classification-mode binary \
    --reference-class "Healthy/Background" \
    --diseases Covid19
```

### Multi-binary

One independent binary classifier per disease vs. a shared reference class.
Trains N-1 separate models (one per non-reference disease).

```bash
# All diseases vs Healthy/Background:
--classification-mode multi-binary \
    --reference-class "Healthy/Background"

# Only specific diseases:
--classification-mode multi-binary \
    --reference-class "Healthy/Background" \
    --diseases Covid19 HIV Lupus
```

---

## Appendix B: Hardware and Performance

### Minimum requirements


| Component | Model 1 | Model 2 | Model 3                 |
| --------- | ------- | ------- | ----------------------- |
| RAM       | 4 GB    | 16 GB   | 64 GB                   |
| CPU cores | 1       | 4+      | 4+                      |
| GPU       | --      | --      | For embeddings only     |
| Disk      | 20 GB   | 20 GB   | 60 GB (with embeddings) |


### Recommended for the original dataset (542 participants, ~30M sequences)

- **RAM:** 128+ GB (allows comfortable `--n-jobs 8` for Models 2 and 3)
- **CPU:** 16+ cores
- **GPU:** Any CUDA GPU for embedding computation (A100/H100 for speed)
- **Disk:** 100+ GB (cache + embeddings + model artifacts)

### Runtime estimates (original dataset, 3 folds)


| Step               | Laptop (M4 Max, n_jobs=2) | Server (64 cores, n_jobs=16) |
| ------------------ | ------------------------- | ---------------------------- |
| Build cache        | ~45 min                   | ~30 min                      |
| Compute embeddings | ~10 hours (MPS)           | ~45 min (A100)               |
| Model 1            | ~3 min                    | ~2 min                       |
| Model 2            | ~2 hours                  | ~30 min                      |
| Model 3            | ~35 hours                 | ~3 hours                     |


---

## Appendix C: Troubleshooting

### Process killed with no error message (OOM)

On Linux, check `dmesg`:

```bash
dmesg | grep -i "oom\|kill" | tail -20
```

On macOS, check Console.app or:

```bash
log show --predicate 'eventMessage contains "Jetsam"' --last 1h
```

**Fix:** Reduce `--n-jobs` or free memory by closing other applications.

### "Loaded 0 sequences from cache"

The fold cache exists but is empty or corrupted. Delete the fold cache
and let the training script rebuild it on next run:

```bash
rm "$CACHE_DIR/data_folds/"*.parquet "$CACHE_DIR/data_folds/"*.csv
# Then re-run the training command with --data-dir to rebuild
```

### "No embedding file found for participant X"

The embedding cache is incomplete. Re-run the embedding script — it skips
already-computed participants:

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --device cuda
```

### PyTorch import error: `undefined symbol: iJIT_NotifyEvent`

This is caused by MKL 2025+ breaking a symbol that PyTorch's CPU build
links against. Fix by pinning MKL to an older version:

```bash
conda install "mkl<2025.0.0"
```

### PyTorch import error: `No module named 'torch'` after conda install

If `conda install pytorch` says "already installed" but `import torch`
fails, conda may have stale metadata. Force reinstall:

```bash
conda install pytorch cpuonly -c pytorch --force-reinstall
```

If that still fails, use pip's pre-built CPU wheel instead:

```bash
conda remove pytorch cpuonly --force
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### glmnet install or import error

The correct pip package is `python-glmnet` (the Replica HQ fork), not
`glmnet` (the old Civis 2.2.1 release which fails on modern Python/numpy):

```bash
pip install python-glmnet
```

Common pitfalls:

- `pip install glmnet` installs the old Civis v2.2.1 which uses
`numpy.distutils` (removed in NumPy 2.0) and fails on Python 3.12+.
- `conda install -c conda-forge glmnet` installs the **R** glmnet
package, not the Python one.
- The `python-glmnet` package is not on conda-forge — pip is the only
option.

If building from source fails, ensure a Fortran compiler is available:

```bash
# Linux
sudo apt install gfortran

# macOS
brew install gcc
```

### CUDA out of memory during embedding computation

Reduce the batch size:

```bash
--batch-size 1000   # or lower
```

### Model 3 crashed or was killed mid-training

Use `--resume` to pick up where it left off without re-training completed
folds and stages:

```bash
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8 \
    --resume
```

### Training seems stuck (no progress for hours)

For Model 3, Stage 1 individual jobs can take 10-30 minutes each with
`--n-jobs 2`. Check that workers are using CPU:

```bash
# Linux
top -H -p $(pgrep -f train_model3)

# macOS
ps aux | grep train_model3
```

If workers show 100% CPU, training is proceeding normally — it's just slow.

---

*This guide covers single-model training with cross-validation. Ensemble
training (`train_ensemble.py`) and external test set evaluation will be
documented here when implemented.*