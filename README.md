# Mal-ID-Lite

A streamlined reimplementation of [Mal-ID](https://github.com/maximz/malid) [Zaslavsky et al., Science, 2025](https://www.science.org/doi/10.1126/science.adp2407), a machine learning framework for disease diagnostics using adaptive immune receptor repertoire sequencing data.

This reimplementation was developed for the benchmarking study:

> **BenchRep-T: A Systematic Evaluation of T-Cell Repertoire-Based Disease Diagnostics**
>
> bioRxiv preprint - coming soon!

Mal-ID is a multiclass disease diagnostic framework that classifies patients from their immune receptor repertoires - B cell receptores (BCRs) and T cell receptors (TCRs) - by combining three complementary models: (1) V-J gene usage frequencies, (2) convergent CDR3 cluster identification across patients, and (3) sequence-level classification using protein language model embeddings. The three BCR and three TCR base models are combined by a logistic-regression meta-model to predict immune status, with the strongest performance obtained by combining both receptor types and all three model components (Zaslavsky et al., 2025). 

To integrate Mal-ID into the BenchRep-T benchmark, we implemented a streamlined reimplementation of the published pipeline, which preserves the model of the original method while making it easier to run consistently across benchmark tasks, folds, and future reuse settings. Two adjustments were introduced for the benchmark setting: (1) replacing Model 3's fixed entropy cutoff with a training-set-based percentile threshold to avoid degenerate filtering in some folds, and (2) filling Model 2 abstentions in the ensemble with the mean of Model 1 and Model 3 predictions, ensuring all specimens receive a score. Refer to the BenchRep-T paper for full details.

> [!NOTE]
> This package is under active development. The current release supports TCR repertoires only (BCR integration coming soon), and documentation and features are being expanded. Contributions and feedback are welcome!

---

## Model Architecture

Mal-ID-Lite implements three complementary models that capture disease-associated signals at different levels of the immune repertoire, plus an ensemble meta-learner that combines their predictions:

| Model | Approach | Input | Key Algorithm |
|-------|----------|-------|---------------|
| **Model 1** | Repertoire-level gene usage | V-J gene pair frequencies | Elastic net logistic regression |
| **Model 2** | Convergent CDR3 clusters | CDR3 amino acid sequences | Hierarchical clustering + Fisher exact test + GLM |
| **Model 3** | Sequence-level embeddings | CDR3 sequences via ESM-2 | Per-V-gene ridge classifiers + specimen-level aggregation |
| **Ensemble** | Meta-learner | Base model probability outputs | Ridge logistic regression |

**Model 1** classifies specimens by the relative frequencies of V-J gene pair usage in their repertoire, applying PCA dimensionality reduction followed by elastic net logistic regression.

**Model 2** identifies CDR3 sequences that are convergently selected across patients with the same disease. It clusters CDR3 sequences by sequence similarity, tests each cluster for disease enrichment via Fisher's exact test, and trains a GLM classifier on the resulting cluster-hit features. Model 2 includes an explicit abstention mechanism - specimens with no significant cluster matches produce no prediction.

**Model 3** extracts 640-dimensional embeddings from CDR3 sequences using [ESM-2](https://github.com/facebookresearch/esm) (a pre-trained protein language model), trains per-V-gene-group classifiers on these embeddings, then aggregates sequence-level predictions to the specimen level using an entropy-based filtering strategy.

**Ensemble** combines probability outputs from all three base models using a logistic regression meta-learner, trained on a held-out validation split.

For detailed algorithmic descriptions of each model, see [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md).

---

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for the full guide.

## Installation


```bash
conda create -n mal_id_lite python=3.12
conda activate mal_id_lite

# Core stack
conda install -c conda-forge pandas numpy pyarrow scikit-learn scipy psutil pytest

# glmnet
pip install python-glmnet

# PyTorch - pick one:
# For CUDA GPU
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia   
# For CPU only
conda install pytorch cpuonly -c pytorch                        

# ESM-2 (pip only)
pip install fair-esm
```


---

## Data Requirements

You need two things: a **metadata file** (TSV) and a directory of **participant sequence files** (AIRR format).

### Metadata (one row per specimen)

A participant may have multiple specimens (e.g., different time points or tissue sites). CV splits are performed at the participant level to prevent data leakage.

| Column | Description |
|--------|-------------|
| `participant_label` | Unique participant ID (may have multiple specimens) |
| `specimen_label` | Unique specimen ID (must match `repertoire_id` in sequence files) |
| `disease` | Disease class label |
| `CV_fold` | Cross-validation fold assignment (integer) |

### Sequence files (one per participant)

Named `part_table_{participant_label}.tsv.gz`, placed in a single directory.

**Required columns:** `repertoire_id`, `v_call`, `j_call`, `cdr3_aa`

**Auto-computed if missing:** `clone_id` (via CDR3 hierarchical clustering)

See the [Pipeline Guide](PIPELINE_GUIDE.md) for the full column reference and optional columns.

---

## Quick Start

### 1. Set up paths

```bash
export MALID_CODE="$HOME/mal-id-lite"                       # the cloned repo
export DATA_DIR="$HOME/project/data/TCR"                        # folder with part_table_* files
export METADATA="$HOME/project/data/metadata.tsv"
export DATASET_NAME="my-dataset"
export CACHE_DIR="$HOME/project/data_cache/$DATASET_NAME"                       # cache output directory

cd "$MALID_CODE"
```

### 2. Run tests

The repo includes a built-in mock dataset (`tests/test_data/`) - no external data needed.
New users should run the quick **validity check** to confirm the install works:

```bash
# Validity check (new users): small end-to-end run, ~4-6 min (add --skip-slow for ~1-2 min)
python tests/run_all_tests.py --validity

# Full suite (CI / pre-release): everything incl. ESM-2 / Model 3 (use --n-jobs to speed up)
python tests/run_all_tests.py --n-jobs 8
```

See [PIPELINE_GUIDE.md, Section 3.4](PIPELINE_GUIDE.md#34-run-the-test-suite) for all test tiers (validity / unit / thorough / full) and the `--n-jobs` option.

### 3. Train the full pipeline (ensemble)

```bash
python malid_lite/training/train_ensemble.py \
    --data-dir "$DATA_DIR" \
    --metadata-path "$METADATA" \
    --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" \
    --classification-mode multiclass \
    --n-jobs 8
```

On first run, provide `--data-dir` so the data cache and ESM-2 embeddings can be built. Subsequent runs can omit it - the pipeline loads from cache.

Results are written to `trained_models/<dataset>/cv_ensemble/`:

```bash
cat trained_models/$DATASET_NAME/cv_ensemble/ensemble/TCR/multiclass/RESULTS_*.md
```

### 4. Train individual models

```bash
# Model 1 - Repertoire-level gene usage (fast, ~2-5 min):
python malid_lite/training/train_model1.py \
    --metadata-path "$METADATA" --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" --classification-mode multiclass

# Model 2 - Convergent CDR3 clusters (~1-3 hours):
python malid_lite/training/train_model2.py \
    --metadata-path "$METADATA" --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" --classification-mode multiclass --n-jobs 8

# Model 3 - Sequence-level ESM-2 classifier (hours to days - depending on CPU vs GPU usage):
python malid_lite/training/train_model3.py \
    --metadata-path "$METADATA" --cache-dir "$CACHE_DIR" \
    --dataset-name "$DATASET_NAME" --classification-mode multiclass --n-jobs 8
```

`--n-jobs` controls parallel workers. Use as many cores as available for significantly faster training (especially Models 2 and 3).

For the full CLI reference, resume logic, cache management, and troubleshooting, see the [Pipeline Guide](PIPELINE_GUIDE.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Installation and first run |
| [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) | Full pipeline reference: all CLI arguments, resume logic, cache management, cross-dataset training & external evaluation (Section 10), troubleshooting |
| [malid_lite/evaluation/README.md](malid_lite/evaluation/README.md) | External evaluation reference: score a trained model on a separate dataset (all `evaluate_external` flags) |
| [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md) | Detailed algorithmic description of each model |

---

## Citation

If you use Mal-ID-Lite, please cite both the benchmarking study and the original Mal-ID paper:

BenchRep-T citation - coming soon!

```bibtex
@article{zaslavsky2025disease,
  title={Disease diagnostics using machine learning of B cell and T cell receptor sequences},
  author={Zaslavsky, Maxim E and Craig, Erin and Michuda, Jackson K and Sehgal, Nidhi and Ram-Mohan, Nikhil and Lee, Ji-Yeun and Nguyen, Khoa D and Hoh, Ramona A and Pham, Tho D and R{\"o}ltgen, Katharina and others},
  journal={Science},
  volume={387},
  number={6736},
  pages={eadp2407},
  year={2025},
  doi={10.1126/science.adp2407}
}
```

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](LICENSE).

You are free to use, share, and adapt this software for **non-commercial purposes only**, with appropriate attribution and under the same license terms.

**Commercial use:** For commercial licensing inquiries, please contact us.

---

## Acknowledgments

- **Original Mal-ID:** [Zaslavsky et al., *Science* 2025](https://www.science.org/doi/10.1126/science.adp2407) ([code](https://github.com/maximz/malid))
- **ESM-2 protein language model:** [Meta AI / FAIR](https://github.com/facebookresearch/esm)
- **Inlined utilities** by Maxim Zaslavsky: [wrap-glmnet](https://github.com/maximz/wrap-glmnet), [genetools](https://github.com/maximz/genetools), [multiclass_metrics](https://github.com/maximz/multiclass-metrics)

---

## Contact

*Currently anonymized
