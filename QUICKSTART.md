# Quick Start

Get Mal-ID-Lite running end-to-end: install, verify, train.

For full details on all arguments, resume logic, cache management, and troubleshooting, see [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).

---

## 1. Install

```bash
conda create -n mal_id_lite python=3.12
conda activate mal_id_lite

# Core stack
conda install -c conda-forge pandas numpy pyarrow scikit-learn scipy psutil pytest

# glmnet (pip only)
pip install python-glmnet

# PyTorch (pick one)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia   # CUDA GPU
conda install pytorch cpuonly -c pytorch                        # CPU only

# ESM-2 (pip only)
pip install fair-esm
```

Verify:

```bash
python -c "
import pandas, numpy, sklearn, glmnet, torch, esm, scipy
print('All imports OK')
print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

---

## 2. Data Requirements

You need two things: a **metadata file** and a directory of **participant sequence files**.

### Metadata (TSV, one row per specimen)

A participant may have multiple specimens (e.g., different time points or tissue sites). CV splits are performed at the participant level to prevent data leakage.

| Column                                            | Description                                                       |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| `participant_label`                               | Unique participant ID (may have multiple specimens)               |
| `specimen_label`                                  | Unique specimen ID (must match `repertoire_id` in sequence files) |
| `disease`                                         | Disease class label (one per participant)                         |
| `CV_fold`                                         | CV fold assignment (integer). Legacy name `malid_cross_validation_fold_id_when_in_test_set` is also accepted. |

### Sequence files (one per participant)

Named `part_table_{participant_label}.tsv.gz`, placed in a single flat directory.

**Required columns:** `repertoire_id`, `v_call`, `j_call`, `cdr3_aa`

**Auto-computed if missing:** `clone_id` (computed via CDR3 hierarchical clustering; requires `cdr3` nucleotide column by default, or `cdr3_aa` with `--clone-id-use-aa`)

**Recommended columns:** `productive`, `v_score` (filtering is skipped with a warning if absent)

See [PIPELINE_GUIDE.md, Section 4](PIPELINE_GUIDE.md#4-data-setup) for the full column reference and optional columns.

---

## 3. Set Up Paths

```bash
# -- Edit these to match your setup --
export MALID_CODE="$HOME/mal-id-lite"                         # the cloned repo
export DATA_DIR="$HOME/project/data/TCR"                      # folder with part_table_* files
export METADATA="$HOME/project/data/metadata.tsv"             # metadata TSV file
export DATASET_NAME="mal-id-orig-data"                        # dataset identifier
export CACHE_DIR="$HOME/project/data_cache/$DATASET_NAME"     # cache output directory

cd "$MALID_CODE"
```

Expected layout:

```
$HOME/project/
├── data/
│   ├── metadata.tsv                     # $METADATA
│   └── TCR/                             # $DATA_DIR
│       ├── part_table_BFI-0000234.tsv.gz
│       └── ...
└── data_cache/
    └── mal-id-orig-data/                # $CACHE_DIR (created by the pipeline)
```

---

## 4. Run Tests

Verify your setup with the **validity check** -- a small end-to-end run on the built-in mock
dataset (`tests/test_data/`, no external data needed) that confirms your install, Python
environment, and data format all work:

```bash
python tests/run_all_tests.py --validity
```

This takes ~4-6 minutes. Options: add `--skip-slow` to skip the Model 3 / ESM-2 step for a
much quicker check (~1-2 min), or `--n-jobs 8` to use more parallel workers. If it passes,
you're ready to train.

> The more thorough test tiers (unit-only, thorough, and the full CI suite) are described in
> [PIPELINE_GUIDE.md, Section 3.4](PIPELINE_GUIDE.md#34-run-the-test-suite).

---

## 5. Train the Full Pipeline (Ensemble)

The ensemble trains all three base models and combines their predictions:

```bash
python malid_lite/training/train_ensemble.py \
    --data-dir "$DATA_DIR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 4
```

On first run, provide `--data-dir` so the data cache and ESM-2 embeddings can be built. Subsequent runs can omit it -- the pipeline loads from cache.

**Classification modes:**

```bash
# Multiclass (default): all diseases simultaneously
--classification-mode multiclass

# Binary (2-class data: auto-detects disease)
--classification-mode binary \
    --reference-class "Healthy/Background"

# Binary (N-class data: pick one disease)
--classification-mode binary \
    --reference-class "Healthy/Background" \
    --diseases Covid19

# Multi-binary: each disease vs reference (N-1 independent classifiers)
--classification-mode multi-binary \
    --reference-class "Healthy/Background"
```

**Results** are written to `trained_models/<dataset>/cv_ensemble/`:

```bash
cat trained_models/$DATASET_NAME/cv_ensemble/ensemble/TCR/multiclass/RESULTS_*.md
```

---

## 6. Train Individual Models

You can also run each model independently. Artifacts are saved under `cv_single_model/`.

**Model 1** -- Repertoire-level gene usage (fast, ~2-5 min):

```bash
python malid_lite/training/train_model1.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass
```

**Model 2** -- Convergent CDR3 clusters (~1-3 hours):

```bash
python malid_lite/training/train_model2.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 4
```

**Model 3** -- Sequence-level ESM-2 classifier (slowest, hours to days):

```bash
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8
```

---

## 7. Common Options

| Flag             | Description                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `--n-jobs N`     | Parallel workers (default: 4). Never use -1.                                                |
| `--fold-ids 0 2` | Fold(s) to hold out as the test set. For each fold listed, the model is trained from scratch on all other folds pooled together, then evaluated on that held-out fold. |
| `--resume`       | Resume after a crash (see [PIPELINE_GUIDE.md, Section 8](PIPELINE_GUIDE.md#8-resume-logic)) |
| `--verbose 2`    | Diagnostics-level logging                                                                   |
| `--force-clone-id` | Recompute clone_id even when it exists in the data (original preserved as `clone_id_original`) |
| `--clone-id-use-aa` | Use amino acid CDR3 for clone assignment (use when nucleotide CDR3 is unavailable)        |

Clone_id flags (`--clone-id-use-aa`, `--clone-id-identity-threshold`, `--clone-id-linkage-method`) only need to be specified once when building the cache. Subsequent training and embedding commands do not need to repeat them -- the cached values are accepted automatically. For details, see [PIPELINE_GUIDE.md, Clone ID computation](PIPELINE_GUIDE.md#clone-id-computation).

---

## Next Steps

- **Pre-compute embeddings separately** (GPU node): see [PIPELINE_GUIDE.md, Section 5.2](PIPELINE_GUIDE.md#52-pre-computing-esm-2-embeddings)
- **Manage the cache**: `python scripts/data/manage_cache.py info --cache-dir "$CACHE_DIR"`
- **Train on one dataset, evaluate on another** (or on a held-out fold subset): see [PIPELINE_GUIDE.md, Section 10](PIPELINE_GUIDE.md#10-cross-dataset-training--external-evaluation) and [malid_lite/evaluation/README.md](malid_lite/evaluation/README.md)
- **Train on a subset of participants**: `python scripts/data/create_subset_cache.py --help`
- **Resume after crash**: add `--resume` to the same command
- **Full reference**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
