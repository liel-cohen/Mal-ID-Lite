# Test Data

Small subsampled dataset for integration tests. Created from real Mal-ID data
by `scripts/data/create_test_data.py`.

## Contents

- `metadata.tsv` — specimen metadata (76 specimens, 72 participants)
- `raw/` — subsampled AIRR-format .tsv.gz files (2000 sequences/specimen, 19 columns)

## Summary

- Diseases: HIV, Covid19, T1D, Healthy/Background
- Participants: 72 (4 with multiple specimens)
- Specimens: 76
- Sequences per specimen: ~2000
- Columns: 19 (stripped from 121 in full data)
- Total compressed size: 11.3 MB

### Per-disease breakdown
  HIV: 18 participants, 20 specimens, 9988 convergent CDR3s
  Covid19: 18 participants, 18 specimens, 2460 convergent CDR3s
  T1D: 18 participants, 20 specimens, 6970 convergent CDR3s
  Healthy/Background: 18 participants, 18 specimens, 20075 convergent CDR3s

## Usage

```python
from malid_lite.dataloader import MalIDPublishedDataLoader

loader = MalIDPublishedDataLoader(
    data_dir=test_data_dir / "raw",
    metadata_path=None,           # loads from cache_dir/metadata.tsv
    gene_reference_path=None,     # FR/CDR extraction skipped
    cache_dir=test_data_dir,      # fold cache built here on first run
    verbose=0,
)
```

## Generation

Created: 2026-04-26 01:19
Seed: 42
Script: scripts/data/create_test_data.py
