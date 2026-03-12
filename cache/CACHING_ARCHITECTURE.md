# Caching Architecture

Mal-ID-Lite uses a **two-level caching strategy** to optimize data preprocessing and loading.

## Overview

```
Raw Data (CSV files)
    ↓
[Level 1] Participant Cache (CLEAN stage) ← Preprocess once per participant
    ↓
[Level 2] Fold Cache (DOWNSAMPLED stage) ← Assemble from participant caches
    ↓
Model Training
```

## Why Two Levels?

**Problem with single-level fold caching:**
- 3 folds × 2 splits = 6 fold combinations
- Each participant appears in ~5 train sets + 1 test set
- With single-level caching: each participant preprocessed **6 times**
- For 542 participants: ~3,252 preprocessing operations!

**Solution with two-level caching:**
- Level 1: Cache each participant after CLEAN stage (542 operations)
- Level 2: Build folds from participant caches (fast assembly, no reprocessing)
- **Result: ~6x speedup!**

## Cache Levels

### Level 1: Participant Cache

**Location:** `cache/participants/`

**Format:**
```
<label>_clean.parquet    # Preprocessed sequences (CLEAN stage)
<label>_stats.json       # Preprocessing statistics
cache_info.json          # Cache metadata
```

**Contains:**
- All sequences for a participant (may include multiple specimens)
- After Stage 1 preprocessing (CLEAN):
  - Productive sequences only
  - V-score filtered
  - Gene names corrected
  - CDR/FR sequences cleaned
  - Deduplicated

**When created:**
- Automatically on first `load_participant_data()` call
- Subsequent loads use cached version

**When invalidated:**
- When `preprocess_clean()` logic changes
- When raw data files are updated
- When gene reference is updated

### Level 2: Fold Cache

**Location:** `cache/` (fold files live directly in the cache root, alongside participant cache)

**Format:**
```
fold_<id>_<label>_downsampled_sequences.parquet
fold_<id>_<label>_downsampled_metadata.csv
cache_info.json
```

**Reports** (data quality reports, logs) are stored separately in `cache/reports/`.

**Note:** Metadata is stored as CSV instead of Parquet to avoid type inference issues with mixed-type columns (e.g., participant_alt_label containing both integers and strings like "HHC 4").

**Contains:**
- Sequences for a specific fold (train or test split)
- After Stage 2 preprocessing (DOWNSAMPLED):
  - CDR3 length filtered
  - Clone threshold applied
  - Sequence threshold applied
  - Downsampled to 1 seq per clone

**When created:**
- By `cache_and_report_all_data.py` script
- Or on first `cache_fold()` call

**When invalidated:**
- When `preprocess_downsample()` logic changes
- When fold assignments change

## Data Flow

### First Run (No Cache)

```
1. load_participant_data("P001")
   ├─ Load raw CSV
   ├─ Run preprocess_clean() → [CLEAN data]
   ├─ Cache to cache/participants/P001_clean.parquet
   └─ Return [CLEAN data]

2. For each specimen in participant:
   ├─ Run preprocess_downsample() → [DOWNSAMPLED data]
   └─ Accumulate statistics

3. cache_fold(0, "train")
   ├─ Already loaded from steps above
   └─ Cache to cache/fold_0_train_*.parquet
```

### Subsequent Runs (With Cache)

```
1. load_participant_data("P001")
   ├─ Check cache/participants/P001_clean.parquet
   ├─ Load from cache → [CLEAN data]
   └─ Skip preprocessing! (fast!)

2. For each specimen in participant:
   ├─ Run preprocess_downsample() → [DOWNSAMPLED data]
   └─ Accumulate statistics

3. load_cached_fold(0, "train")
   ├─ Check cache/fold_0_train_*.parquet
   ├─ Load from cache → [DOWNSAMPLED data]
   └─ Skip everything! (fastest!)
```

## Cache Metadata

Each cache directory contains a `cache_info.json` file:

```json
{
  "created_at": "2026-03-03T10:30:00",
  "malid_version": "0.1.0",
  "cache_type": "participants",
  "data_dir": "/path/to/data",
  "metadata_path": "/path/to/metadata.tsv",
  "gene_locus": "TCR",
  "preprocessing_stage": "CLEAN"
}
```

**Purpose:**
- Track when cache was created
- Identify Mal-ID version used
- Verify cache compatibility
- Debug stale cache issues

## Cache Management

### View Cache Info

```bash
python scripts/data/manage_cache.py info
```

Shows:
- Cache directory location
- Number of cached participants/folds
- Cache size
- Creation timestamp
- Mal-ID version

### Clear Caches

```bash
# Clear participant cache only
python scripts/data/manage_cache.py clear-participants

# Clear fold cache only
python scripts/data/manage_cache.py clear-folds

# Clear everything
python scripts/data/manage_cache.py clear-all
```

### Rebuild Strategy

**Scenario 1: Changed CLEAN preprocessing**
```bash
# Clear participant cache (keeps fold cache)
python scripts/data/manage_cache.py clear-participants

# Rebuild everything
python scripts/data/cache_and_report_all_data.py
```

**Scenario 2: Changed DOWNSAMPLED preprocessing**
```bash
# Clear fold cache only (keeps participant cache)
python scripts/data/manage_cache.py clear-folds

# Rebuild folds (fast - uses participant cache!)
python scripts/data/cache_and_report_all_data.py
```

**Scenario 3: Major changes or debugging**
```bash
# Clear everything
python scripts/data/manage_cache.py clear-all

# Full rebuild
python scripts/data/cache_and_report_all_data.py
```

## Performance

### Timing Estimates (542 participants)

| Operation | Time | Notes |
|-----------|------|-------|
| First preprocessing (no cache) | ~2-3 hours | Creates participant cache |
| Rebuild folds (participant cache exists) | ~20-30 min | 6x faster than first run |
| Load fold (fold cache exists) | <5 min | Just loads parquet files |
| Load single participant (cached) | <1 sec | Instant |

### Storage Requirements

- Participant cache: ~5.3 GB (542 participants × 2 files: .parquet + .json)
- Fold cache: ~13 GB (12 files: 6 × sequences.parquet + 6 × metadata.csv)
- Reports: ~600 KB (CSV/JSON summary files + log files)
- Total: ~18 GB

## Implementation Details

### Type Conversion for Parquet

**Challenge:** Mixed-type columns cause Parquet type inference issues.

**Example:** The `participant_alt_label` column contains:
- Numeric IDs: `700010698`, `800012345`
- String IDs: `"HHC 4"`, `"HHC 2"`, `"HHC 9"`

When concatenating participants into folds, pandas stores this as `object` dtype. Parquet then attempts to infer the "best" type, sees numbers, and tries to convert to `int64`, which fails on string values.

**Solution:**
```python
# Before saving sequences to Parquet, convert mixed-type columns
for col in sequences_df.columns:
    if sequences_df[col].dtype == 'object' or str(sequences_df[col].dtype).startswith('string'):
        # Force plain object dtype with string values
        sequences_df[col] = sequences_df[col].astype(str).astype('object')
```

This prevents Parquet from attempting automatic type conversion. Metadata files use CSV format entirely to avoid this issue, since metadata is small (~20-40 KB per fold) and doesn't benefit much from Parquet compression.

### Key Methods

**BaseDataLoader:**
```python
# Participant-level caching
cache_participant(participant_label, df, preprocessing_stats)
load_cached_participant(participant_label) → (df, stats)

# Fold-level caching
cache_fold(fold_id, fold_label, stage)
load_cached_fold(fold_id, fold_label, stage) → (sequences_df, metadata_df)

# Cache management
clear_participant_cache(participant_label=None)
clear_fold_cache(fold_id=None, fold_label=None)
clear_all_caches()
get_cache_info() → dict
```

**MalIDPublishedDataLoader:**
```python
def load_participant_data(self, participant_label, stage):
    # Check participant cache
    cached = self.load_cached_participant(participant_label)
    if cached:
        df, etl_stats = cached
        # Use cached data
    else:
        # Load raw file
        # Run preprocess_clean()
        # Cache result

    # Apply DOWNSAMPLED if needed
    # Return data
```

## Best Practices

1. **Always use the caching script for initial setup:**
   ```bash
   python scripts/data/cache_and_report_all_data.py
   ```

2. **Check cache info before training:**
   ```bash
   python scripts/data/manage_cache.py info
   ```

3. **Clear selectively, not wholesale:**
   - Changed CLEAN logic? → Clear participants
   - Changed DOWNSAMPLED logic? → Clear folds
   - Debugging? → Clear all

4. **Version control:**
   - Cache metadata tracks Mal-ID version
   - Check version compatibility when loading cache
   - Rebuild if versions mismatch

5. **Disk space:**
   - Monitor cache size (2-3 GB typical)
   - Clear old caches when not needed
   - Consider .gitignore for cache directories

## Future Enhancements

Potential improvements:

1. **Automatic cache invalidation:**
   - Hash preprocessing functions
   - Detect code changes
   - Auto-rebuild stale caches

2. **Incremental caching:**
   - Only rebuild changed participants
   - Track data file modification times

3. **Compressed caching:**
   - Use compression in Parquet
   - Trade CPU for disk space

4. **Cache warming:**
   - Parallel preprocessing
   - Background cache building

5. **Cache sharing:**
   - Team-shared cache directory
   - Download pre-built caches
