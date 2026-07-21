# Training Scripts

This directory contains scripts for training and evaluating Mal-ID models.

## Model 1: Repertoire Classifier — `train_model1.py`

Unified training script for Model 1 (logistic regression with elastic net).
Supports multiclass, binary, and multi-binary classification modes via `--classification-mode`.

### Quick Start

```bash
# Multiclass (default): single N-class model
python malid_lite/training/train_model1.py

# Multi-binary: one model per disease vs Healthy/Background
python malid_lite/training/train_model1.py \
    --classification-mode multi-binary --reference-class "Healthy/Background"

# Binary: one specific disease pair
python malid_lite/training/train_model1.py \
    --classification-mode binary --reference-class "Healthy/Background" --diseases Covid19

# Train only fold 0
python malid_lite/training/train_model1.py --fold-ids 0

# Train-all: train on the WHOLE dataset (no CV holdout), for later evaluation
# on a separate dataset. Produces artifacts with no fold prefix and no metrics.
python malid_lite/training/train_model1.py --training-context train_all
```

### Cross-validation vs. train-all (`--training-context`)

- **CV** (default) evaluates the model via cross-validation on this one dataset:
  `cv_single_model` (standalone) or `cv_ensemble` (base models for the ensemble).
  A `CV_fold` column is required.
- **Train-all** trains on the entire dataset with no held-out test fold, for
  scoring later on a *separate* dataset (see external evaluation): `train_all`
  (standalone) or `train_all_ensemble` (base models — holds out a validation
  third for the metamodel). No `CV_fold` column is required, `--fold-ids` is not
  allowed, and **no evaluation metrics are produced** — the summary documents
  what was trained. Artifacts have no `fold_<id>_` prefix (`<model>_model.pkl`,
  `<model>_v_genes.json`, `<model>_meta.json`) and live under
  `trained_models/<dataset>/train_all_single_model/...` (or
  `train_all_ensemble/base_models/...`).

### What It Does

1. **Loads cached data** from `cache/<dataset_name>/` (fast, ~seconds per fold)
2. **Filters rare V genes** (bottom 50% by frequency) on training set only
3. **Extracts features** (V-J gene pair frequencies + PCA)
4. **Trains Model 1** (RepertoireClassifier with glmnet elastic net, internal CV to tune lambda)
5. **Evaluates** on held-out test fold
6. **Computes metrics**:
   - Accuracy (global pooled)
   - **AUROC (OvO weighted)** ← Primary multiclass metric
   - **AUPRC (OvR weighted)**
   - Per-class AUROC OvR
   - Log loss, confusion matrix
   - For binary: `auroc_binary`, `auprc_binary` (disease as positive)
7. **Saves outputs** (see Output Structure below)

### Command-Line Options

```
--dataset-name STR        Dataset identifier (default: mal-id-orig-data). Used in output path.
--training-context        cv_single_model | cv_ensemble | train_all | train_all_ensemble
                          (default: cv_single_model). train_all* = train on the whole
                          dataset, no evaluation; incompatible with --fold-ids.
--classification-mode     multiclass | binary | multi-binary (default: multiclass)
--reference-class STR     Reference/negative class for binary/multi-binary modes
--diseases STR [STR ...]  Explicit disease subset (binary: one; multi-binary: any subset)
--fold-ids INT [INT ...]  Fold IDs to train (default: 0 1 2)
--model-name STR          Label for this model variant (default: lasso_cv)
--l1-ratio FLOAT          Elastic net L1/L2 ratio (default: 1.0 for TCR, 0.25 for BCR)
--n-pcs INT               PCA components (default: 15)
--gene-locus TCR|BCR      Gene locus (default: TCR)
--output-dir PATH         Override canonical output path (mutually exclusive with --output-suffix)
--output-suffix STR       Suffix appended to mode dir (e.g. "no_pca" → multiclass__no_pca).
                          Mutually exclusive with --output-dir.
--verbose {0,1,2}         Verbosity level (default: 1)
```

### Output Structure

With `--output-suffix <suffix>`, the mode directory becomes `<mode>__<suffix>` (e.g. `multiclass__no_pca`).

**Multiclass:**
```
trained_models/<dataset_name>/model1/multiclass/<gene_locus>/
├── fold_<id>_<model_name>_model.pkl       # Fitted RepertoireClassifier
├── fold_<id>_<model_name>_v_genes.json    # V genes kept after frequency filtering
├── fold_<id>_<model_name>_results.json    # Per-fold evaluation metrics
├── <model_name>_multiclass_predictions.csv # All folds pooled; columns: participant_label,
│                                           #   specimen_label, true_disease, predicted_disease,
│                                           #   score_<class1>, score_<class2>, ..., fold_id
├── summary_<timestamp>.json              # Full run summary (all folds aggregated)
└── training_<timestamp>.log              # Log file
```

**Binary / Multi-binary:**
```
trained_models/<dataset_name>/model1/binary/<gene_locus>/
├── <disease>_vs_<reference>/
│   ├── fold_<id>_<model_name>_model.pkl
│   ├── fold_<id>_<model_name>_v_genes.json
│   ├── fold_<id>_<model_name>_results.json
│   └── <model_name>_binary_predictions.csv   # All folds pooled; columns: participant_label,
│                                              #   specimen_label, disease_label (0/1),
│                                              #   disease_label_str, disease_model,
│                                              #   model_score, fold_id
├── <disease2>_vs_<reference>/
│   └── ...
├── summary_<timestamp>.json                  # Covers all pairs
└── training_<timestamp>.log
```

**Train-all** (`--training-context train_all`) — no evaluation, so artifacts have
no `fold_<id>_` prefix and there are no `results`/`predictions` files:
```
trained_models/<dataset_name>/train_all_single_model/model1/multiclass/<gene_locus>/
├── <model_name>_model.pkl        # Fitted on the whole dataset
├── <model_name>_v_genes.json
├── <model_name>_meta.json        # Run params (for --resume) + training info
├── summary_<timestamp>.json      # Training summary (training_only=true; no metrics)
├── RESULTS_<timestamp>.md        # Human-readable training summary (no metrics)
└── training_<timestamp>.log
```
(Ensemble base models use `train_all_ensemble/base_models/<locus>/model1/<mode>/`.)

### Summary JSON Structure

```json
{
  "timestamp": "...",
  "dataset_name": "mal-id-orig-data",
  "classification_mode": "multiclass",
  "reference_class": null,
  "diseases": null,
  "gene_locus": "TCR",
  "output_suffix": null,
  "fold_ids": [0, 1, 2],
  "model_names": ["lasso_cv"],
  "results_by_pair": {
    "multiclass": [
      {"fold_id": 0, "model_name": "lasso_cv", "n_scored": 184, "n_abstained": 0, "accuracy": 0.70, ...}
    ]
  },
  "aggregated_by_pair": {
    "multiclass": {
      "lasso_cv": {
        "n_folds": 3, "accuracy_global": 0.70,
        "auroc_ovo_weighted": {"mean": 0.9423, "std": 0.0063, "per_fold": [...], "n_folds_valid": 3},
        ...
      }
    }
  }
}
```

### Requirements

Before running, ensure:
1. Cache is built (`python scripts/data/cache_and_report_all_data.py` first)
2. `glmnet` installed (python-glmnet, requires R)
3. All dependencies from `requirements.txt` installed

### Performance

| Condition          | Time per fold  |
| ------------------ | -------------- |
| With fold cache    | ~1-2 minutes   |
| Without fold cache | ~10-15 minutes |

### Model Variants

| `--model-name`      | `--l1-ratio`      | Description                             |
| ------------------- | ----------------- | --------------------------------------- |
| `lasso_cv`          | 1.0 (default TCR) | Pure L1 — best for TCR per paper        |
| `elasticnet_cv0.75` | 0.75              | 75% L1, 25% L2                          |
| `elasticnet_cv`     | 0.5               | 50% L1, 50% L2                          |
| `elasticnet_cv0.25` | 0.25              | 25% L1, 75% L2 — best for BCR per paper |
| `ridge_cv`          | 0.0               | Pure L2                                 |

### Troubleshooting

**Issue**: `FileNotFoundError: Cache not found`
- **Fix**: Run `python scripts/data/cache_and_report_all_data.py` first

**Issue**: `ModuleNotFoundError: No module named 'glmnet'`
- **Fix**: `conda install -c conda-forge glmnet` (preferred) or `pip install glmnet`

**Issue**: Slow training (>10 min per fold)
- **Check**: Cache should make this ~1-2 min per fold
- **Fix**: Verify cache exists (check log for "Fold cache files: N")

---

## Model 2: Convergent Cluster Classifier — `train_model2.py`

See module docstring (`python malid_lite/training/train_model2.py --help`) for full documentation.
Supports the same `--classification-mode` / `--reference-class` / `--diseases` interface as Model 1,
and the same `--training-context` (`cv_single_model` | `cv_ensemble` | `train_all` |
`train_all_ensemble`).

Key differences from Model 1:
- Uses train_smaller1 / train_smaller2 inner split (2/3 + 1/3 of train fold)
- Clusters CDR3 sequences by (V gene, J gene, CDR3 length, Hamming distance ≤ 1)
- Selects significant clusters via Fisher's exact test (p-value grid search on train_smaller2)
- Trains logistic regression on cluster hit counts (sparse, low-dimensional feature matrix)
- Specimens matching no clusters produce no prediction (abstention)
- `--retrain-full` flag: after p-value selection, re-trains GLM on train_smaller1 + train_smaller2

**Train-all** (`--training-context train_all`): like Model 1, trains on the whole
dataset with no evaluation, writing no-fold-prefix artifacts under
`train_all_single_model/model2/<mode>/<locus>/`: `clusters.joblib`,
`<model>_p_value.joblib`, `<model>_model_<suffix>.joblib`, `<model>_results_<suffix>.json`
(or `<model>_NO_VALID_CLUSTERS.txt`), plus `meta.json` (resume) and a no-metrics
summary. Model 2 still uses ts1 (cluster) and ts2 (p-value select) — both are
training, not evaluation. `--retrain-full` is recommended for a standalone `train_all`
model so the final GLM uses all the data.

---

## Model 3: Sequence-Level Classifier — `train_model3.py`

Two-stage V-gene-specific sequence model using ESM-2 embeddings of CDR3 sequences.
Supports multiclass, binary, and multi-binary classification modes.

### Pre-requisite: Compute Embeddings

ESM-2 embeddings must be pre-computed per participant before training:

```bash
python -m malid_lite.training.compute_model3_embeddings \
    --metadata-path /path/to/metadata.tsv
```

Resource estimates (approximate, M4 Max MPS): ~3 hours per 10M downsampled sequences,
~14 GB storage per 10M sequences.

### Quick Start

```bash
# Multiclass (default, requires pre-computed embeddings)
python malid_lite/training/train_model3.py \
    --metadata-path /path/to/metadata.tsv

# Skip embedding caching (compute inline per-subset, don't save)
python malid_lite/training/train_model3.py \
    --metadata-path /path/to/metadata.tsv --no-cache-embeddings

# Multi-binary
python malid_lite/training/train_model3.py \
    --metadata-path /path/to/metadata.tsv \
    --classification-mode multi-binary --reference-class "Healthy/Background"

# Custom aggregation strategy (default: entropy_percentile_cutoff, 0.01)
python malid_lite/training/train_model3.py \
    --metadata-path /path/to/metadata.tsv --aggregation-strategy mean

# Train-all: train ONE model on the whole dataset (no CV, no test set)
python malid_lite/training/train_model3.py \
    --metadata-path /path/to/metadata.tsv --training-context train_all
```

### What It Does

1. **Loads pre-computed ESM-2 embeddings** (auto-computed and cached if missing; `--no-cache-embeddings` for inline without saving)
2. **Stage 1**: Trains per-V-gene classifiers on CDR3 embeddings (train_smaller1 split)
3. **Stage 2**: Trains specimen-level rollup model on Stage 1 predictions (train_smaller2 split)
4. **Evaluates** on held-out test fold (CV contexts only)
5. **Saves outputs** to `trained_models/<dataset_name>/model3/<mode>/<gene_locus>/`

**Train-all** (`--training-context train_all`): like Models 1/2, trains on the whole
dataset with no held-out test and no metrics — a reusable model to score later on a
*separate* dataset. Model 3 still uses ts1/ts2 separately (Stage 1 on ts1, Stage 2 on
ts2). Artifacts have no fold prefix and go to
`train_all_single_model/model3/<mode>/<locus>/`: `stage1.pkl`, `stage2.pkl`, a `meta.json`
resume sentinel, optional `entropy_survival_stats.csv` / `tuning_cv_results.csv`, and a
no-metrics `summary_*.json`. Use `train_all_ensemble` for ensemble base models (holds out
a validation third). Resume: `--resume` (all-or-nothing) or `--resume-from-stage2` (reuse
Stage 1, retrain Stage 2); `--fold-ids` / `--resume-from-evaluation` / `--stage1-dir` are
CV-only and error under a train-all context.

### Embedding Computation — `compute_model3_embeddings.py`

Standalone script for pre-computing per-participant ESM-2 embeddings from DOWNSAMPLED
CDR3 sequences. Supports resumption (skips already-processed participants), verification
(`--verify`), and generates a detailed report with timing and storage statistics.

Output per participant (in `cache/<dataset>/embeddings/` by default, or a custom
directory via `--output-embedding-dir`):
- `<label>_embeddings.npy` — float16 embeddings array (N x 640), row-aligned with parquet
- `<label>_downsampled.parquet` — exact DOWNSAMPLED sequences that were embedded
- `<label>_stats.json` — per-participant processing statistics

### Embedding-to-Fold Alignment

At training time, per-participant embeddings are assembled into fold-level arrays.
Row alignment is ensured using the downsampling unique key (`repertoire_id`,
`igh_or_tcrb_clone_id`, `isotype_supergroup`, `amplification_label` if present).
If rows are already in the same order, embeddings are used directly (fast path).
If the order differs, embeddings are automatically reordered to match (with a warning).
A biological sanity check (`cdr3_aa`, `v_gene`, `j_gene`) runs after alignment.
See `CACHING_ARCHITECTURE.md` for details.

---

## Ensemble Meta-Learner — `train_ensemble.py`

Trains a ridge-regularized logistic-regression metamodel on the three base models'
per-specimen disease probabilities. Base models are auto-trained if their artifacts
are missing (per-model LOAD / TRAIN / RESUME detection).

### Cross-validation vs. train-all (`--training-context`, REQUIRED)

Unlike the base-model scripts (which default to `cv_single_model`), the ensemble
requires `--training-context` explicitly — the short values `cv` / `train_all` map
internally to `cv_ensemble` / `train_all_ensemble`:

- **`cv`** — per-fold cross-validation. The metamodel trains on each fold's
  validation third and is evaluated on the held-out test fold → reports test
  metrics, aggregates across folds, writes `ensemble_predictions.csv`.
- **`train_all`** — a single whole-dataset pass for later evaluation on a
  *separate* dataset. The metamodel trains on base-model predictions over the
  validation third; there is **no test fold → no metrics**. Base models are
  auto-trained as `train_all_ensemble` (on ts1+ts2, validation excluded), so the
  metamodel sees base-model *out-of-sample* predictions — exactly as in CV. Once
  the base models are `train_all_ensemble`, this is a leakage guard: the ensemble
  rejects base models trained as the leaky `train_all` (which saw the validation
  set) or as `cv_ensemble`.

The flag has **no default**: CV-evaluation and train-all-for-external-eval are very
different long-running jobs, so intent must be explicit.

### Quick Start

```bash
# Cross-validation ensemble (all 3 models, multiclass)
python malid_lite/training/train_ensemble.py \
    --training-context cv \
    --metadata-path cache/mal-id-orig-data/metadata.tsv \
    --cache-dir cache/mal-id-orig-data

# Train-all ensemble: base models + metamodel on one whole dataset, to be
# evaluated later on a SEPARATE dataset (no test set, no metrics)
python malid_lite/training/train_ensemble.py \
    --training-context train_all \
    --metadata-path cache/train-dataset/metadata.tsv \
    --cache-dir cache/train-dataset
```

### Output Structure

- **`cv`** → `trained_models/<dataset>/cv_ensemble/ensemble/<locus>/<mode>/` with
  per-fold artifacts (`fold_<id>_ridge_cv_metamodel.joblib`,
  `fold_<id>_metamodel_config.json`, `fold_<id>_feature_matrix_{val,test}.csv`),
  aggregated metrics + `ensemble_predictions.csv`.
- **`train_all`** → `trained_models/<dataset>/train_all_ensemble/ensemble/<locus>/<mode>/`,
  no fold prefix, no test/metrics:
  - `ridge_cv_metamodel.joblib`, `metamodel_config.json`
  - `feature_matrix_val.csv` + `feature_matrix_raw_val.csv`
  - `ensemble_results.json` (validation abstention/fill details)
  - `summary_<timestamp>.json` — no-metrics training summary carrying
    `training_complete: True` (the "ready for inference" marker downstream external
    evaluation checks) plus the inference config (base-model dirs, feature-column
    order, abstention strategy, `base_model_training_context`).

Binary/multi-binary write per-pair subdirectories (`<disease>_vs_<reference>/`),
each self-sufficient. `train_all` writes no cross-pair metrics summary (no metrics).

### Other options

- `--fold-ids` is CV-only (rejected under `train_all`).
- `--resume` (train-all): reloads the cached `feature_matrix_raw_val.csv` and
  retrains the metamodel without re-running base-model predictions.
- `--feature-matrices-dir DIR`: trains only the metamodel from an external run's
  saved feature matrices (works for both contexts; the source run's
  `training_context` must match `--training-context`).
- `--model2-abstention-strategy {ensemble_abstain,fill_0.5,fill_models13_mean}`:
  how to handle Model 2 abstentions (drop vs. fill).
- `--metamodel-cv-n-splits N` (default 5, must be ≥2): folds for the metamodel's
  internal StratifiedGroupKFold (auto-capped down for small classes). Lower it for
  small datasets. Applies to both contexts.

The `training_complete: True` readiness marker is also written to every base-model
and CV-ensemble `summary_<timestamp>.json`, so downstream scripts check one uniform
field regardless of how a model was trained.

---

## External Evaluation

External evaluation (applying a trained model to a separate dataset) now lives in the
**`malid_lite.evaluation`** package. See **`malid_lite/evaluation/README.md`** for the
full guide, or run:

```bash
python -m malid_lite.evaluation.evaluate_external --help
```

---

## Shared Utilities — `training_utils.py`

All shared code used by the training scripts lives here. Key exports:

| Name                                                                                                                                      | Type     | Purpose                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DISEASE_COL`                                                                                                                             | constant | Column name for disease label                                                                                                                                           |
| `SPECIMEN_COL`                                                                                                                            | constant | Column name for specimen label                                                                                                                                          |
| `PARTICIPANT_COL`                                                                                                                         | constant | Column name for participant label                                                                                                                                       |
| `DEFAULT_DATASET_NAME`                                                                                                                    | constant | Default dataset identifier for output paths                                                                                                                             |
| `PROJECT_ROOT`                                                                                                                            | constant | `Path(__file__).parent.parent.parent` — project root                                                                                                                    |
| `make_pair_name(disease, ref)`                                                                                                            | function | Filesystem-safe `<disease>_vs_<ref>` string                                                                                                                             |
| `get_model_output_dir(model_name, dataset_name, classification_mode, gene_locus, training_context="cv_single_model", output_suffix=None)` | function | Canonical `trained_models/...` output path; training_context controls directory structure (suffix appends `__<suffix>` to mode dir)                                     |
| `get_dataset_disease_classes(metadata_path)`                                                                                              | function | All disease classes in metadata                                                                                                                                         |
| `validate_mode_and_classes(classification_mode, disease_classes, reference_class, diseases)`                                              | function | CLI argument validation                                                                                                                                                 |
| `filter_to_binary_pair(sequences_df, metadata_df, disease, reference_class)`                                                              | function | Filter data to one binary pair                                                                                                                                          |
| `split_train_smaller(sequences_df, metadata_df)`                                                                                          | function | Split train fold into train_smaller1 (2/3) and train_smaller2 (1/3). **Deprecated** — use `loader.get_split_participants()` instead for centralized, persistent splits. |
| `aggregate_fold_results(fold_metrics, fold_raw_preds, disease_filter)`                                                                    | function | Cross-fold metric aggregation                                                                                                                                           |
| `run_training_orchestration(classification_mode, disease_classes, reference_class, fold_loop_fn, ...)`                                    | function | Dispatches training across classification modes                                                                                                                         |
| `generate_results_md(all_results, classification_mode, timestamp, model_label, run_info, fold_ids, model_names, has_abstention)`          | function | Generates `RESULTS_<timestamp>.md` from training run results                                                                                                            |

---

## Data Processing Scripts

See `scripts/data/README.md` for data preprocessing and caching scripts.
