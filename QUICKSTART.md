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

| Column                                            | Description                                                       |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| `participant_label`                               | Unique participant ID                                             |
| `specimen_label`                                  | Unique specimen ID (must match `repertoire_id` in sequence files) |
| `disease`                                         | Disease class label (one per participant)                         |
| `malid_cross_validation_fold_id_when_in_test_set` | CV fold assignment (integer)                                      |

### Sequence files (one per participant)

Named `part_table_{participant_label}.tsv.gz`, placed in a single flat directory.

**Required columns:** `repertoire_id`, `v_call`, `j_call`, `cdr3_aa`, `clone_id`

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

The repo includes a mock dataset (`tests/test_data/`) -- no external data needed.

```bash
# Full suite (unit + integration, ~5-10 min) -- highly recommended:
python tests/run_all_tests.py

# Unit tests only (fast, ~1-2 min):
python tests/run_all_tests.py --skip-integration
```

**We highly recommend running the full suite including integration tests.** It exercises the entire pipeline end-to-end on the built-in mock dataset and takes only ~5-10 minutes. All groups should pass before proceeding to training.

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

# Binary: one disease vs reference
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
| `--fold-ids 0 2` | Train only specific folds                                                                   |
| `--resume`       | Resume after a crash (see [PIPELINE_GUIDE.md, Section 8](PIPELINE_GUIDE.md#8-resume-logic)) |
| `--verbose 2`    | Diagnostics-level logging                                                                   |

---

## Next Steps

- **Pre-compute embeddings separately** (GPU node): see [PIPELINE_GUIDE.md, Section 5.2](PIPELINE_GUIDE.md#52-pre-computing-esm-2-embeddings)
- **Manage the cache**: `python scripts/data/manage_cache.py info --cache-dir "$CACHE_DIR"`
- **Resume after crash**: add `--resume` to the same command
- **Full reference**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
