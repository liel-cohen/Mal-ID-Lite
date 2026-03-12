# Model 1 Quick Test - Ready to Run ✅

## Test Script: `tests/test_model1_quick.py`

### What It Tests

1. **Initialize Data Loader** - With cache support
2. **Load Training Data** - Fold 0 train (DOWNSAMPLED stage)
3. **Extract Features** - V-J gene pair frequencies
4. **Prepare Labels** - Disease labels and participant groups
5. **Train Model** - Fit Model 1 with internal CV
6. **Predict on Training** - Sanity check
7. **Predict on Test Set** - Evaluate on held-out fold
8. **Save and Load Model** - Persistence test

### ✅ Fixed: Now Uses Cache

**Before:**
```python
loader = MalIDPublishedDataLoader(
    data_dir=...,
    metadata_path=...,
    # No cache_dir - would reprocess everything!
)
```

**After (Fixed):**
```python
cache_dir = project_root / "cache"  # Use existing cache

loader = MalIDPublishedDataLoader(
    data_dir=...,
    metadata_path=...,
    cache_dir=cache_dir,  # ✅ Enable caching for fast loading
    verbose=1,
)
```

### Cache Status Check

The test now checks and reports:
- Participant cache files available
- Fold cache files available
- Uses cached data if present (MUCH faster!)

### Expected Performance

**With cache (542 participant files + 12 fold files):**
- Loading fold 0 train: ~5-10 seconds
- Total test time: ~1-2 minutes

**Without cache:**
- Loading fold 0 train: ~5-10 minutes
- Total test time: ~10-15 minutes

### Output Files

All outputs saved to `tests/test_outputs/test_model1_quick/`:
- `test_model1_quick_YYYYMMDD_HHMMSS.log` - Full log
- `test_model1_quick_YYYYMMDD_HHMMSS.json` - Structured results
- `features_fold0_train_YYYYMMDD_HHMMSS.csv` - Extracted features
- `model_fold0_YYYYMMDD_HHMMSS.pkl` - Trained model

**Organization:** Each test script creates its own subfolder to keep outputs organized.

### How to Run

```bash
# Run from project root
python tests/test_model1_quick.py

# Or with pytest
pytest tests/test_model1_quick.py -v
```

### What to Expect

**Console output:**
```
============================================================
QUICK MODEL 1 SMOKE TEST
Started: 2026-03-03 HH:MM:SS
============================================================

1. Initializing data loader...
✓ Using cache directory: /Users/.../Mal-ID-Lite/cache
  - Participant cache files: 542
  - Fold cache files: 12
✓ Loader initialized

2. Loading fold 0 training data (downsampled)...
✓ Loaded fold 0 train data
  - 366 specimens
  - 16,301,506 sequences
  - Diseases: ['Healthy/Background', 'HIV', 'T1D', 'Lupus', 'Covid19', 'Influenza']

3. Extracting features with Model 1...
✓ Features extracted
  - Shape: 366 specimens × ~700 features  (V-J pair frequencies, varies by fold)
  - Column examples: ['TRBV10-1|TRBJ1-1:TCRB', 'TRBV10-1|TRBJ1-2:TCRB', ...]

4. Preparing labels and groups...
✓ Labels prepared
  - Unique labels: [...]
  - Label counts:
      Healthy/Background: 131
      HIV: 65
      ...

5. Training Model 1...
✓ Model trained
  - Fitted classes: [...]
  - Features in: ~700

6. Making predictions on training set...
✓ Predictions made
  - Training accuracy: 0.XXX
  - Probability shape: (366, N_classes)

7. Loading and predicting on test set...
✓ Loaded fold 0 test data: 184 specimens
✓ Test features extracted: (184, ~700)
✓ Test predictions made
  - Test accuracy: 0.XXX

8. Testing model save/load...
✓ Model saved to: test_outputs/test_model1_quick/model_fold0_YYYYMMDD_HHMMSS.pkl
✓ Model loaded successfully
✓ Loaded model predictions match: True

============================================================
✓ ALL TESTS PASSED
Completed: 2026-03-03 HH:MM:SS
============================================================

📝 Test outputs saved to: test_outputs/test_model1_quick/
  - Log file: test_model1_quick_YYYYMMDD_HHMMSS.log
  - Results JSON: test_model1_quick_YYYYMMDD_HHMMSS.json
  - Features CSV: features_fold0_train_YYYYMMDD_HHMMSS.csv
  - Model PKL: model_fold0_YYYYMMDD_HHMMSS.pkl
```

### Verification Checklist

Before running:
- ✅ Cache directory exists at `cache/`
- ✅ Participant cache populated (542 files)
- ✅ Fold cache populated (12 files)
- ✅ wrap-glmnet installed
- ✅ All dependencies installed

### Notes

- **First run**: May be slow if cache needs to be built
- **Subsequent runs**: Should be fast (~1-2 minutes)
- **Memory usage**: ~2-3 GB during training
- **Disk usage**: Outputs ~100 MB total

---

**Status**: ✅ Ready to run
**Last Updated**: 2026-03-03
