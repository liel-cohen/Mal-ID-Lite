# External Evaluation — `evaluate_external.py`

Apply models trained on the WHOLE of one dataset (a `train_all_ensemble` ensemble, or
standalone `train_all` base models) — or a specific CV fold's model — to a **separate**,
labeled dataset, and report comprehensive metrics. Nothing is trained; models are loaded
and applied. Mechanically it is the "test side" of the CV ensemble replayed on the whole
external dataset (predict base models → build the metamodel feature matrix → apply the
metamodel → score), reusing the same tested functions from `malid_lite.training`.

## Specifying the model (choose exactly ONE way — mixing them errors)

1. **Ensemble, explicit folder:** `--ensemble-dir DIR`. Its base-model directories are
   resolved automatically from the ensemble summary's `base_model_paths` — you never
   re-specify them.
2. **Standalone base models, explicit folders:** one or more of
   `--model1-dir` / `--model2-dir` / `--model3-dir` (a `train_all` run with no metamodel —
   evaluates those base models only, no ensemble).
3. **Ensemble, by convention:** `--train-dataset-name NAME` (+ `--gene-locus`,
   `--classification-mode`, `--output-suffix`) — resolves the canonical
   `trained_models/<NAME>/train_all_ensemble/ensemble/<locus>/<mode>` path.

All base models plus the ensemble are always evaluated and reported (no model-selection
flag). The label space for all metrics is the model's **training** classes.

## Test data

- `--test-cache-dir` is required. If the cache is built, it is used and `--test-data-dir`
  (raw AIRR data) is NOT needed; `--test-data-dir` only builds the cache when absent.
- `--test-metadata-path` is optional (defaults to the cache's `metadata_processed.tsv`);
  if given, it must be consistent with the cache. The test metadata needs a `disease`
  column (ground truth).
- Model 3: `--test-embedding-dir` (default `<test-cache-dir>/embeddings`).

## Evaluating a CV fold's model, and test-fold filtering

- `--model-fold-id i` evaluates a specific CV fold's model (resolves `cv_ensemble`
  artifacts). Fold `i`'s model was **trained on all folds except `i`** (`i` was its
  held-out test fold), so for a clean same-dataset held-out evaluation, pair it with
  `--test-on-folds i`. A warning fires if you test on other folds (they were training data).
- `--test-on-folds N [N ...]` restricts the test set to specimens with those `CV_fold`
  values (independent of the model source; requires a `CV_fold` column).

## Consistency checks

- `--gene-locus` must match the trained model (hard error — a TCR model can't score BCR).
- The clone_id clustering definition is compared and a mismatch **warns** only (cross
  nt/aa is a legitimate experiment); the check is skipped when clone_id was pre-existing
  in the source data on either side.
- `classification_mode` / `reference_class` / `diseases` are read from the model summary.
- If the test dataset has classes the model never saw → error (listing both class sets),
  unless `--allow-unknown-test-classes` (then those specimens are dropped, count logged).

## Output

Flat per `(locus, mode[, pair])` under
`trained_models/<train_dataset>/<context>/evaluated_on/<test_dataset>/<locus>/<mode>/[<pair>/]`:
- `results_<ts>.json` — dataset counts (train + test), per-model + ensemble metrics
  (accuracy, balanced accuracy, AUROC ovo/ovr, AUPRC, MCC, log-loss, abstention),
  confusion matrices (counts + normalized), per-class precision/recall/F1, top confusions,
  per-class ROC/PR AUCs, and (binary) sensitivity/specificity operating points.
- `predictions_<ts>.csv` — per-specimen: true label + each base model's and the ensemble's
  predicted class + per-class probabilities.
- `curves/` — ROC/PR arrays (per model) for re-plotting.
- `figures/` — ROC / PR / confusion-matrix PNGs per model + ensemble + a comparison chart (600 DPI).
- `RESULTS_<ts>.md` — human-readable report.

```bash
# Train-all ensemble (dataset A) evaluated on a separate dataset B
python -m malid_lite.evaluation.evaluate_external \
    --ensemble-dir trained_models/dataset-A/train_all_ensemble/ensemble/TCR/multiclass \
    --test-cache-dir cache/dataset-B --test-dataset-name dataset-B

# Same-dataset held-out: fold-2 CV model tested on fold 2
python -m malid_lite.evaluation.evaluate_external \
    --ensemble-dir trained_models/dataset-A/cv_ensemble/ensemble/TCR/multiclass \
    --model-fold-id 2 --test-on-folds 2 --test-cache-dir cache/dataset-A
```

## Label-free inference and on-the-fly embeddings

- `--inference-only` — run prediction WITHOUT ground-truth labels: writes per-specimen
  predictions (each base model + the ensemble) and skips all metrics/figures. Allows a
  test dataset with no `disease` column (which otherwise errors, to guard against an
  accidental column-name mismatch silently skipping evaluation). Skips class alignment
  and pair-disease filtering (no labels to filter on).
- `--inline-embeddings` — compute any missing Model 3 test embeddings on the fly (into
  `--test-embedding-dir`, default `<test-cache-dir>/embeddings`; uses `--device` /
  `--embedding-batch-size`) instead of requiring them pre-computed.

## Report number formatting

Metrics in `RESULTS_<ts>.md` and the figure AUC/AP legends are displayed to 4 decimal
places; `results_<ts>.json` stores full-precision (unrounded) values.
