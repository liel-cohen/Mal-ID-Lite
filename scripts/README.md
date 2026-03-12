# Training Scripts

This directory contains scripts for training and evaluating Mal-ID models.

## Model 1: Repertoire Classifier

**Script**: `train_model1.py`

### Quick Start

```bash
# Train on all 3 folds with default settings (lasso for TCR, l1_ratio=1.0)
python scripts/train_model1.py

# Train only fold 0
python scripts/train_model1.py --fold-ids 0

# Use elastic net (l1_ratio=0.5) instead of lasso
python scripts/train_model1.py --model-name elasticnet_cv --l1-ratio 0.5

# Customize PCA components and output directory
python scripts/train_model1.py --n-pcs 10 --output-dir results/model1_custom
```

### Requirements

Before running, ensure:
1. ✅ Cache is built (run `python scripts/data/cache_and_report_all_data.py` first)
2. ✅ `wrap-glmnet` is installed (`pip install wrap-glmnet`)
3. ✅ All dependencies from `requirements.txt` are installed

### What It Does

1. **Loads cached data** from `cache/` (fast, ~seconds per fold)
2. **Filters rare V genes** (bottom 50% by frequency) on training set only
3. **Extracts features** (V-J gene pair frequencies)
4. **Trains Model 1** with internal CV to tune lambda
5. **Evaluates** on held-out test fold
6. **Computes metrics**:
   - Accuracy
   - **AUROC (multiclass one-vs-one, weighted)** ← Primary metric
   - Log loss
   - Confusion matrix
7. **Saves outputs**:
   - Trained models (`*.pkl`)
   - Predictions (`*.predictions.csv`)
   - V gene lists (`*.v_genes.json`)
   - Aggregated results (`*.results_*.json`)
   - Training log (`training_*.log`)

### Command-Line Options

```
--fold-ids 0 1 2          Fold IDs to train on (default: all 3 folds)
--model-name STR          Model variant name (default: lasso_cv for TCR)
--l1-ratio FLOAT          L1/L2 ratio: 0=ridge, 1=lasso (default: None = 1.0 for TCR, 0.25 for BCR)
--n-pcs INT               PCA components per isotype (default: 15)
--output-dir PATH         Output directory (default: models/model1)
--cache-dir PATH          Cache directory (default: cache)
--verbose {0,1,2}         Verbosity: 0=silent, 1=basic, 2=detailed (default: 1)
```

### Output Structure

```
models/model1/
├── elasticnet_cv.fold0.pkl                        # Trained model (fold 0)
├── elasticnet_cv.fold0.predictions.csv            # Predictions with probabilities
├── elasticnet_cv.fold0.v_genes.json               # V genes kept for training
├── elasticnet_cv.fold1.pkl                        # Trained model (fold 1)
├── elasticnet_cv.fold1.predictions.csv
├── elasticnet_cv.fold1.v_genes.json
├── elasticnet_cv.fold2.pkl                        # Trained model (fold 2)
├── elasticnet_cv.fold2.predictions.csv
├── elasticnet_cv.fold2.v_genes.json
├── elasticnet_cv.results_20260304_123456.json     # Aggregated results
└── training_20260304_123456.log                   # Training log
```

### Predictions CSV Format

```csv
specimen_label,y_true,y_pred,prob_Covid19,prob_HIV,prob_Healthy/Background,prob_Influenza,prob_Lupus,prob_T1D
M124-S014,Healthy/Background,Healthy/Background,0.05,0.10,0.75,0.03,0.05,0.02
...
```

### Results JSON Format

```json
{
  "model_name": "lasso_cv",
  "timestamp": "20260304_123456",
  "parameters": {
    "gene_locus": "TCR",
    "n_pcs": 15,
    "l1_ratio": 1.0
  },
  "aggregated_metrics": {
    "n_folds": 3,
    "fold_ids": [0, 1, 2],
    "accuracy_per_fold": {
      "mean": 0.707,
      "std": 0.015,
      "per_fold": [0.700, 0.720, 0.701]
    },
    "accuracy_global": 0.707,
    "auroc_ovo_weighted": {
      "mean": 0.942,
      "std": 0.006,
      "per_fold": [0.933, 0.947, 0.946]
    },
    "auprc_ovr_weighted": {
      "mean": 0.803,
      "std": 0.028,
      "per_fold": [0.775, 0.831, 0.803]
    },
    "log_loss": {
      "mean": 0.654,
      "std": 0.032,
      "per_fold": [0.643, 0.621, 0.698]
    },
    "auroc_ovr_per_class": {
      "Covid19": {"mean": 0.98, "std": 0.01, "per_fold": [0.97, 0.99, 0.98]},
      "HIV": {"mean": 0.97, "std": 0.01, "per_fold": [0.96, 0.98, 0.97]},
      "..."
    }
  },
  "fold_results": [...]
}
```

### Performance

**With cache** (recommended):
- Fold loading: ~5-10 seconds
- Feature extraction: ~10-20 seconds
- Training: ~30-60 seconds
- Total per fold: ~1-2 minutes

**Without cache** (not recommended):
- Fold loading: ~5-10 minutes
- Total per fold: ~10-15 minutes

### Model Variants

The script supports different model variants via `--l1-ratio`:

| Variant | `--l1-ratio` | Description |
|---------|--------------|-------------|
| Ridge | 0.0 | L2 regularization only |
| Elastic Net (0.25) | 0.25 | 25% L1, 75% L2 (BCR best in paper) |
| Elastic Net (0.5) | 0.5 | 50% L1, 50% L2 |
| Elastic Net (0.75) | 0.75 | 75% L1, 25% L2 |
| **Lasso** | **1.0** | **L1 regularization only (TCR default, TCR best in paper)** |

**Defaults based on paper**:
- **TCR**: `l1_ratio=1.0` (pure lasso) ← Default for our TCR-only implementation
- **BCR** (future): `l1_ratio=0.25` (elastic net 0.25)

### Examples

**Train with default lasso (recommended for TCR)**:
```bash
python scripts/train_model1.py
```

**Train with elastic net instead**:
```bash
python scripts/train_model1.py --model-name elasticnet_cv --l1-ratio 0.5
```

**Train only on fold 0 for quick testing**:
```bash
python scripts/train_model1.py --fold-ids 0 --verbose 2
```

**Custom PCA dimensions**:
```bash
python scripts/train_model1.py --n-pcs 10
```

**Save to custom directory**:
```bash
python scripts/train_model1.py --output-dir experiments/model1_test1
```

### Troubleshooting

**Issue**: `FileNotFoundError: Cache not found`
- **Fix**: Run `python scripts/data/cache_and_report_all_data.py` first

**Issue**: `ModuleNotFoundError: No module named 'wrap_glmnet'`
- **Fix**: `pip install wrap-glmnet`

**Issue**: Low AUROC scores (<0.6)
- **Check**: Are you using the cache? Without cache, preprocessing might differ
- **Check**: Is the fold data loaded correctly? Look at n_train and n_test in logs
- **Check**: Are features being extracted? Look at n_features in logs

**Issue**: Training is slow (>10 min per fold)
- **Check**: Cache should make this ~1-2 min per fold
- **Fix**: Verify cache exists and is being used (check logs for "Using cache directory")

### Next Steps

After training:
1. Review results in `*.results_*.json`
2. Analyze predictions in `*.predictions.csv`
3. Compare to expected AUROC from paper (TCR ~0.88, BCR ~0.91)
4. Train Models 2 & 3 (coming soon)
5. Build metamodel ensemble (coming soon)

---

## Data Processing Scripts

See `scripts/data/README.md` for data preprocessing and caching scripts.
