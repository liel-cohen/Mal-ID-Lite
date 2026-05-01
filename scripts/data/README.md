# scripts/data Directory

Utility scripts for data processing and caching.

---

## `cache_and_report_all_data.py`

**Purpose:** Process all TCR repertoire data through the complete preprocessing pipeline, cache results (using efficient 2-phase approach), and generate comprehensive quality reports.

**What it does:**
1. ✅ **Phase 1:** Processes ALL participants once (creates participant-level cache)
2. 🚀 **Phase 2:** Builds fold caches from participant caches (~6x faster than old approach)
3. 💾 Caches at two levels: participant-level (CLEAN) and fold-level (DOWNSAMPLED)
4. 📊 Generates detailed preprocessing reports with sequence cleaning statistics
5. 📈 Creates summary statistics about the dataset
6. 🔍 Provides data quality insights

**Efficiency:**
- OLD approach: Each participant preprocessed 6 times (once per fold) = ~3,252 operations
- NEW approach: Each participant preprocessed once + fold assembly = ~542 operations
- **Result: ~6x faster caching!**

**Usage:**
```bash
cd Mal-ID-Lite
python scripts/data/cache_and_report_all_data.py
```

**Runtime:**
- First run (no cache): ~2-3 hours for full dataset (542 participants)
- Rebuilding folds (with participant cache): ~20-30 minutes
- Using existing fold cache: <5 minutes

**Output:**

### Cache Files

**Two-level caching structure:**
Preprocessed data in Parquet/CSV format for instant loading:
```
cache/
├── participants/                                    (Level 1: Per-participant cache)
│   ├── BFI-0000234_clean.parquet                   (~1-5 MB each)
│   ├── BFI-0000234_stats.json                      (preprocessing stats)
│   ├── ... (542 participant files: .parquet + .json)
│   └── cache_info.json                             (metadata: timestamp, version)
│
├── data_folds/                                      (Level 2: Fold cache)
│   ├── fold_0_train_downsampled_sequences.parquet  (2.8 GB)
│   ├── fold_0_train_downsampled_metadata.csv       (39 KB)
│   ├── fold_0_test_downsampled_sequences.parquet   (1.5 GB)
│   ├── fold_0_test_downsampled_metadata.csv        (20 KB)
│   ├── fold_1_train_downsampled_sequences.parquet  (2.9 GB)
│   ├── fold_1_train_downsampled_metadata.csv       (39 KB)
│   ├── fold_1_test_downsampled_sequences.parquet   (1.5 GB)
│   ├── fold_1_test_downsampled_metadata.csv        (20 KB)
│   ├── fold_2_train_downsampled_sequences.parquet  (3.0 GB)
│   ├── fold_2_train_downsampled_metadata.csv       (39 KB)
│   ├── fold_2_test_downsampled_sequences.parquet   (1.4 GB)
│   ├── fold_2_test_downsampled_metadata.csv        (19 KB)
│   └── cache_info.json                             (metadata: timestamp, version)
│
└── reports/                                         (data quality reports)
    ├── preprocessing_report_full_*.csv
    ├── summary_report_*.csv
    ├── filter_effectiveness_*.csv
    ├── sequence_counts_*.csv
    ├── global_stats_*.json
    └── caching_log_*.txt
```

**Note:** Metadata files use CSV format (not Parquet) to avoid type inference issues with mixed-type columns.

**Speedup:**
- Participant cache → fold assembly: ~6x faster than preprocessing each fold separately
- Fold cache → training: ~30x faster than preprocessing from scratch!

**Storage Requirements:**
- Participant cache: ~5.3 GB (542 files × 2: .parquet + .json)
- Fold cache: ~13 GB (12 files: 6 × sequences.parquet + 6 × metadata.csv)
- Total: ~18 GB

### Report Files (`./reports/`)

#### 1. **Summary Report** (`summary_report_YYYYMMDD_HHMMSS.csv`)
High-level overview of the entire dataset:
- Total specimens and sequences
- Disease distribution
- Specimens dropped (by reason)
- Gene name corrections applied
- V genes missing from reference table (rows that will have NaN FR/CDR columns)
- Fold distribution (specimens/sequences per fold)

**Example:**
```csv
metric,value,category
Total Specimens,550,Overview
Total Sequences (downsampled),45000000,Overview
Covid-19,180,Disease Distribution
HIV,150,Disease Distribution
Healthy,120,Disease Distribution
insufficient_clones (<500),15,Dropped Specimens
insufficient_sequences (<1000),8,Dropped Specimens
TRBV6-2*02->TRBV6-2*01,98000,Gene Name Corrections
TRBV99-1,42,V Genes Missing From Reference (rows with NaN FR/CDR)
```

#### 2. **Sequence Count Statistics** (`sequence_counts_YYYYMMDD_HHMMSS.csv`)
Distribution of sequence counts per specimen across preprocessing stages:
```csv
stage,mean,median,min,max
raw,150000,120000,50000,500000
clean,85000,70000,30000,350000
downsampled,60000,50000,20000,200000
```

#### 3. **Filter Effectiveness** (`filter_effectiveness_YYYYMMDD_HHMMSS.csv`)
How many sequences each filter removed:
```csv
filter,total_dropped,specimens_affected,avg_dropped_per_specimen
Productive Filter,12000000,542,22140
V Score Filter,8500000,542,15682
Missing Fields,500000,320,1562
```

#### 4. **Fold × Disease Distribution** (`fold_disease_distribution_YYYYMMDD_HHMMSS.csv`)
Specimens per disease per fold (for checking balance):
```csv
disease,0,1,2,3,4
Covid-19,36,36,36,36,36
HIV,30,30,30,30,30
Healthy,24,24,24,24,24
```

#### 5. **Clone Statistics** (`clone_stats_YYYYMMDD_HHMMSS.csv`)
Clone count distribution across kept specimens:
```csv
metric,value
mean,8500
median,7200
min,500
max,45000
std,5400
```

#### 6. **Full Preprocessing Report** (`preprocessing_report_full_YYYYMMDD_HHMMSS.csv`)
Detailed per-specimen statistics with all filter counts. Columns:
- `participant_label`, `specimen_label`, `fold_id`
- `original_count` - Raw sequences
- `productive_filter` - Sequences dropped by productive filter
- `v_score_filter` - Sequences dropped by V score filter
- `sequences_before_dedup`, `sequences_after_dedup`
- `gene_name_fixes`, `gene_name_fixes_detail`
- `genes_missing_from_reference` - Dict `{gene_name: row_count}` for V genes absent from the reference table (FR/CDR columns are NaN for these rows)
- `n_genes_missing_from_reference` - Count of such V genes
- `missing_fields`, `missing_fields_detail`
- `after_clean` - Sequences after stage 1
- `n_clones` - Number of clones
- `after_downsample` - Sequences after stage 2
- `kept` - Whether specimen passed all filters
- `drop_reason` - Why specimen was dropped (if applicable)

#### 7. **Global Stats JSON** (`global_stats_YYYYMMDD_HHMMSS.json`)
Machine-readable version for programmatic access.

#### 8. **Processing Log** (`caching_log_YYYYMMDD_HHMMSS.txt`)
Complete processing log with timestamps and detailed progress.

---

## `check_raw_data_cdr3_bad_chars.py`

> **Note:** This script operates on the **original Synapse raw data** (internal format, `data/internal_format/TCR/`) used in code version 0.1.0. It does **not** use the cleaned AIRR format data (`data_clean/airr_format_clean/TCR/`) used in later versions.

**Purpose:** Validate CDR3 amino acid sequences in the raw source data by checking for invalid characters outside the 20 standard amino acids.

**What it does:**
1. Scans all participant tables in the raw internal format
2. Identifies sequences with invalid characters in CDR3 AA sequences
3. Counts occurrences of each bad character
4. Generates summary reports

**Usage:**
```bash
cd Mal-ID-Lite
python scripts/data/check_raw_data_cdr3_bad_chars.py
```

**Output:**
```
scripts/data/output/check_raw_data_cdr3_bad_chars/
├── bad_characters_summary_YYYYMMDD_HHMMSS.csv
│   Columns: participant_label, status, cdr3_column, total_sequences,
│            sequences_with_bad_chars, bad_char_percentage,
│            unique_bad_chars, bad_char_details
└── bad_characters_report_YYYYMMDD_HHMMSS.txt
    Human-readable summary of findings
```

**Valid amino acids checked:**
- 20 standard amino acids: G, A, L, M, F, W, K, Q, E, S, P, V, I, C, Y, H, R, N, D, T
- Spaces are removed before checking (not considered invalid)

**When to use:**
- To characterise the raw source data before any cleaning pipeline is applied
- When investigating why certain sequences were dropped during preprocessing

---

## `compare_raw_TCR_airr_and_internal.py`

> **Note:** This script operates on the **original Synapse raw data** (both `data/airr_format/TCR/` and `data/internal_format/TCR/`) used in code version 0.1.0. It does **not** use the cleaned AIRR format data (`data_clean/airr_format_clean/TCR/`) used in later versions.

**Purpose:** Compare TCR sequence counts between the raw AIRR format and raw internal format files to verify they contain the same sequences.

**What it does:**
1. Loads data from both raw formats for each participant
2. Compares sequence counts between formats
3. Identifies discrepancies
4. Generates comparison reports

**Usage:**
```bash
cd Mal-ID-Lite
python scripts/data/compare_raw_TCR_airr_and_internal.py
```

**Output:**
```
scripts/data/output/compare_raw_TCR_airr_and_internal/
├── comparison_summary_YYYYMMDD_HHMMSS.csv
│   Columns: participant_label, airr_count, internal_count, match, difference, status
└── comparison_report_YYYYMMDD_HHMMSS.txt
    Detailed report with statistics and discrepancies
```

**When to use:**
- To verify that the raw AIRR and internal format files are consistent with each other
- When debugging discrepancies in raw source data

---

## `create_subset_cache.py`

**Purpose:** Create a subset dataset cache from an existing (reference) cache. Useful for running experiments on a smaller set of participants without re-processing raw data or re-computing ESM-2 embeddings.

**How it works:**

The `participants/` and `embeddings/` directories in a cache are per-participant and self-contained. All other cache artifacts (`data_folds/`, `splits/`, `metadata_processed.tsv`) are auto-generated by the pipeline on first access. This script copies (or symlinks) the per-participant files for your chosen participants into a new cache directory and saves the subset metadata. The pipeline then rebuilds everything else automatically at training time.

**Requirements:**

You need:
1. A **subset metadata TSV** with the same format as the original metadata (must have columns: `participant_label`, `specimen_label`, `disease`, `CV_fold`).
2. A **reference cache** that already has `participants/` and `embeddings/` for all participants in the subset.

**Usage:**

```bash
# Using a reference cache directory path:
python scripts/data/create_subset_cache.py \
    --metadata-subset path/to/subset_metadata.tsv \
    --dataset-name "my-subset" \
    --ref-cache-dir path/to/reference/cache

# Using a reference dataset name (resolves to cache/<name>/ under project root):
python scripts/data/create_subset_cache.py \
    --metadata-subset path/to/subset_metadata.tsv \
    --dataset-name "my-subset" \
    --ref-dataset-name "mal-id-orig-data"

# With symlinks instead of copies (saves disk space, but subset depends on
# the reference cache staying in place):
python scripts/data/create_subset_cache.py \
    --metadata-subset path/to/subset_metadata.tsv \
    --dataset-name "my-subset" \
    --ref-cache-dir path/to/reference/cache \
    --symlink

# Force overwrite if the output cache directory already exists:
python scripts/data/create_subset_cache.py \
    --metadata-subset path/to/subset_metadata.tsv \
    --dataset-name "my-subset" \
    --ref-cache-dir path/to/reference/cache \
    --force
```

**Arguments:**

| Argument              | Required | Description                                                                                             |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `--metadata-subset`   | Yes      | Path to the subset metadata TSV file                                                                    |
| `--dataset-name`      | Yes      | Name for the new subset dataset (used as output subdirectory under `cache/`)                            |
| `--ref-cache-dir`     | One of   | Path to the reference cache directory (mutually exclusive with `--ref-dataset-name`)                    |
| `--ref-dataset-name`  | these    | Name of the reference dataset, resolves to `cache/<name>/` under project root                          |
| `--output-cache-dir`  | No       | Explicit output path. Default: `cache/<dataset-name>/` under project root                              |
| `--symlink`           | No       | Create symbolic links instead of copying files                                                          |
| `--force`             | No       | Delete and recreate the output directory if it already exists                                           |

**What it does (step by step):**

1. Validates subset metadata (required columns, non-empty)
2. Checks all subset participants exist in the reference metadata
3. Checks all required files exist in the reference cache (2 per participant in `participants/`, 3 in `embeddings/`)
4. Copies (or symlinks) files into the new cache's `participants/` and `embeddings/` directories
5. Saves metadata as `metadata.tsv` and `metadata_processed.tsv` (atomic writes)
6. Validates all output files are present (and symlinks are not broken)
7. Prints a per-fold summary of participants and disease distribution

**Output:**

```
cache/<dataset-name>/
├── metadata.tsv                    # subset metadata (reference copy)
├── metadata_processed.tsv          # subset metadata (used by pipeline)
├── participants/                   # copied/linked from reference
│   ├── <label>_clean.parquet
│   └── <label>_stats.json
└── embeddings/                     # copied/linked from reference
    ├── <label>_embeddings.npy
    ├── <label>_downsampled.parquet
    └── <label>_stats.json
```

The `data_folds/` and `splits/` directories are **not** created by this script — the pipeline generates them automatically on first training run.

**Training on the subset:**

```bash
python malid_lite/training/train_ensemble.py \
    --metadata-path cache/<dataset-name>/metadata_processed.tsv \
    --cache-dir cache/<dataset-name> \
    --dataset-name <dataset-name> \
    --classification-mode multiclass
```

No `--data-dir` is needed — everything is loaded from the cache.

---

## `manage_cache.py`

**Purpose:** Manage cached data - view cache information and clear caches selectively.

**Usage:**
```bash
# View cache information (size, creation time, version)
python scripts/data/manage_cache.py info

# Use a different dataset (resolves to cache/<dataset-name>/)
python scripts/data/manage_cache.py info --dataset-name my-dataset

# Or specify cache directory directly
python scripts/data/manage_cache.py info --cache-dir /path/to/cache/my-dataset

# Clear participant cache only (keeps fold cache)
python scripts/data/manage_cache.py clear-participants

# Clear fold cache only (keeps participant cache)
python scripts/data/manage_cache.py clear-folds

# Clear all caches
python scripts/data/manage_cache.py clear-all
```

**When to clear caches:**

1. **Participant cache** - clear when:
   - Preprocessing logic changes in `preprocess_clean()`
   - Raw data files are updated
   - Gene reference file is updated
   - You want to rebuild participant-level data

2. **Fold cache** - clear when:
   - Preprocessing logic changes in `preprocess_downsample()`
   - Fold assignments change
   - You want to rebuild fold data (but keep participant cache)

3. **All caches** - clear when:
   - Clone_id parameters changed (also delete model artifacts in `trained_models/`)
   - Major changes to preprocessing pipeline
   - Switching between different datasets
   - Debugging cache-related issues

**Cache metadata:**
Each cache directory contains a `cache_info.json` file with:
- `created_at`: Timestamp when cache was created
- `malid_version`: Version of Mal-ID-Lite used
- `data_dir`: Path to source data
- `preprocessing_stage`: Stage of preprocessing cached

---

## When to Run These Scripts

**Initial setup:**
```bash
# First time - process and cache everything
python scripts/data/cache_and_report_all_data.py
```

**After data updates:**
- Delete `./cache/` directory
- Re-run script to rebuild cache

**After preprocessing changes:**
- Delete `./cache/` directory
- Update preprocessing logic in `malid_lite/dataloader/mal_id_published.py`
- Re-run script

**For data exploration:**
- Run once to generate reports
- Inspect CSV files in `./reports/` to understand dataset

---

## Example Workflow

```bash
# 1. Cache all data and generate reports
python scripts/data/cache_and_report_all_data.py

# 2. Inspect reports
cd reports
ls -lh  # See all generated reports
head summary_report_*.csv
head preprocessing_report_full_*.csv

# 3. Use cached data for fast training
python malid_lite/training/train_model1.py  # Will load from cache automatically
```

---

## Customization

Edit the script to:
- Change cache directory: `cache_dir=Path("./my_cache")`
- Change report directory: `report_dir=Path("./my_reports")`
- Process specific folds only: Modify the `all_folds` list
- Cache different preprocessing stages: Change `PreprocessingStage.DOWNSAMPLED`
- Add custom statistics: Extend `DataReportGenerator.generate_detailed_stats()`

---

## Troubleshooting

**Out of memory:**
- Script uses iterator (processes one participant at a time)
- Should work even on 8GB RAM
- If issues persist, process fewer folds at once

**Slow processing:**
- Expected! Processing 542 participants with full preprocessing takes time
- First run: 2-3 hours
- Subsequent runs with cache: <5 minutes

**Missing dependencies:**
- Requires `pyarrow` for Parquet support: `pip install pyarrow`

**Stale cache:**
- Use `python scripts/data/manage_cache.py` to manage caches (see below)
- Delete specific caches or rebuild everything
- Cache includes version metadata for tracking
