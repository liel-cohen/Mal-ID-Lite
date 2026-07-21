"""External evaluation for Mal-ID-Lite (Phase 6).

Apply models trained on the WHOLE of one dataset (a ``train_all`` /
``train_all_ensemble`` model — see ``train_ensemble.py --training-context train_all``)
to a SEPARATE, fully-labeled dataset and report held-out generalization metrics.
Nothing is trained here: models are loaded and applied.

What this is, mechanically
--------------------------
External evaluation is the "test side" of the cross-validation ensemble replayed on
the whole external dataset: get each base model's predictions on the target specimens
-> build the metamodel feature matrix -> apply the metamodel -> compute metrics. The
same tested functions are reused (``predict_model1/2/3`` with ``fold_id=None``,
``build_feature_matrix``, ``evaluate_predictions``).

Label space
-----------
The model predicts over a FIXED label space = its TRAINING classes (read from the
model's summary). All metrics use that label space. If the external dataset contains
classes the model never saw, evaluation ERRORS unless ``--allow-unknown-test-classes``
is given (then those specimens are dropped, with the dropped count logged).

Specifying the model (exactly ONE of these; mixing them is an error)
--------------------------------------------------------------------
1. Ensemble, explicit folder:   ``--ensemble-dir DIR``
2. Standalone base models, explicit folders: one or more of
   ``--model1-dir DIR`` / ``--model2-dir DIR`` / ``--model3-dir DIR`` (a ``train_all``
   run with NO metamodel — evaluates those base models only)
3. Ensemble, by convention:     ``--train-dataset-name NAME`` (+ ``--gene-locus``,
   ``--classification-mode``, ``--output-suffix``) — resolves the canonical
   ``trained_models/<NAME>/train_all_ensemble/ensemble/<locus>/<mode>`` path.
For an ensemble you give ONLY the ensemble location; its base-model directories are
resolved automatically from the ensemble summary's ``base_model_paths`` (you evaluate
exactly the base models the metamodel was trained against). All base models plus the
ensemble are always evaluated and reported (there is no model-selection flag).

Test data
---------
- ``--test-cache-dir`` is required. If its cache is already built, that is used and
  ``--test-data-dir`` (raw AIRR data) is NOT needed; ``--test-data-dir`` is used ONLY to
  build the cache when it does not exist yet.
- ``--test-metadata-path`` is OPTIONAL: if omitted, the cache's ``metadata_processed.tsv``
  is used. If given, it must be consistent with the cache (it must either BE the cache's
  processed metadata or MATCH the cached raw ``metadata.tsv``), else the loader errors
  ("cache may be stale"). The test metadata must have a ``disease`` column (ground truth).
- Model 3 embeddings: ``--test-embedding-dir`` points at the TEST dataset's pre-computed
  ESM-2 embeddings (default: ``<test-cache-dir>/embeddings``).

Consistency
-----------
``gene_locus`` MUST match the trained model (hard error). The clone_id clustering
definition is compared and a mismatch WARNS only (cross nt/aa is a legitimate experiment;
also skipped when clone_id was pre-existing in the source data on either side).
``classification_mode`` / ``reference_class`` / ``diseases`` are read from the model
summary (not the CLI).

Output
------
``trained_models/<train_dataset>/<train_context>/evaluated_on/<test_dataset>/<locus>/<mode>/[<pair>/]``
with ``results_<ts>.json`` (comprehensive: dataset counts, per-model + ensemble metrics,
confusion matrices, per-class precision/recall/F1, top confusions, per-class ROC/PR AUCs,
binary operating points), ``predictions_<ts>.csv`` (per-specimen, all models + ensemble),
``curves/`` (ROC/PR arrays for re-plotting), ``figures/`` (ROC/PR/confusion PNGs @ 600 DPI +
a model-comparison bar chart), and ``RESULTS_<ts>.md``.

Evaluating a CV fold's model
----------------------------
``--model-fold-id i`` evaluates a specific cross-validation fold's model instead of a
train-all model (resolves ``cv_ensemble`` artifacts). Fold ``i``'s model was TRAINED on
all folds EXCEPT ``i`` (``i`` was its held-out test fold), so for a clean held-out
evaluation on the SAME dataset, pair it with ``--test-on-folds i``. ``--test-on-folds``
restricts the test set to specimens with the given ``CV_fold`` value(s) and works
independently of the model source.

Usage
-----
    # Train-all ensemble (dataset A) evaluated on a separate dataset B
    python -m malid_lite.evaluation.evaluate_external \\
        --ensemble-dir trained_models/dataset-A/train_all_ensemble/ensemble/TCR/multiclass \\
        --test-cache-dir cache/dataset-B --test-dataset-name dataset-B

    # Same-dataset held-out eval: fold-2 CV model tested on fold 2
    python -m malid_lite.evaluation.evaluate_external \\
        --ensemble-dir trained_models/dataset-A/cv_ensemble/ensemble/TCR/multiclass \\
        --model-fold-id 2 --test-on-folds 2 \\
        --test-cache-dir cache/dataset-A

    # Label-free inference (no 'disease' column), computing Model 3 embeddings on the fly
    python -m malid_lite.evaluation.evaluate_external \\
        --ensemble-dir trained_models/dataset-A/train_all_ensemble/ensemble/TCR/multiclass \\
        --test-cache-dir cache/dataset-B --test-data-dir raw/dataset-B \\
        --inference-only --inline-embeddings

``--inline-embeddings`` computes any missing Model 3 test embeddings on the fly (into
``--test-embedding-dir``; uses ``--device`` / ``--embedding-batch-size``). ``--inference-only``
runs prediction WITHOUT ground-truth labels (no metrics) and allows a test dataset with no
``disease`` column — without it, a missing ``disease`` column is a hard error (guards against
an accidental column-name mismatch silently skipping evaluation).
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.dataloader import (
    MalIDPublishedDataLoader,
    add_clone_id_args,
    get_clone_id_kwargs,
)
from malid_lite.dataloader.base import PreprocessingStage
from malid_lite.evaluation.evaluate_external_reporting import (
    compact_curves_for_json,
    compute_rich_metrics,
    save_comparison_figure,
    save_curve_csvs,
    save_model_figures,
    write_results_md,
)
from malid_lite.training.train_ensemble import (
    MODEL2_ABSTENTION_STRATEGIES,
    ModelPredictions,
    build_feature_matrix,
    predict_model1,
    predict_model2,
    predict_model3,
)
from malid_lite.training.training_utils import (
    DISEASE_COL,
    FOLD_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    filter_to_binary_pair,
    get_ensemble_output_dir,
    get_metadata_class_counts,
    make_pair_name,
    read_model_summary,
)

logger = logging.getLogger(__name__)

METAMODEL_JOBLIB = "ridge_cv_metamodel.joblib"

# Sentinel disease label attached to the test metadata in --inference-only mode.
# The base-model predict functions (Model 2/3) look up a `disease` column ONLY to
# populate ground-truth *label* fields (abstained-sample labels / alignment) — never
# for features, probabilities, or the abstention decision. Inference-only test data
# has no labels, so we attach this clearly-marked placeholder to satisfy that lookup.
# It is never surfaced: inference-only computes no metrics and omits `true_disease`
# from the predictions output. The double-underscore markers make it obvious in any
# stray output that this is not a real class.
INFERENCE_ONLY_PLACEHOLDER_LABEL = "__PLACEHOLDER_NO_LABEL__"


# ===========================================================================
# Model loading + readiness / leakage checks
# ===========================================================================


def load_trained_model(
    model_dir: Path,
    *,
    is_ensemble: bool,
    fold_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a completed model's summary/config and verify it is ready for inference.

    Enforces the Phase-5 readiness contract (``training_complete: True``, written
    LAST by the trainer). For an ensemble, returns the inference config. For a
    train-all ensemble (fold_id is None) it asserts the base models were trained under
    ``train_all_ensemble`` (not the leaky ``train_all``). For a CV ensemble (fold_id
    given) the metamodel config lives in ``fold_<id>_metamodel_config.json`` and the
    base-model paths in ``run_config.json``; its base models are ``cv_ensemble`` by
    construction (trained on the fold's train split, metamodel on the fold's validation
    split), so no leakage flag is asserted.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Check the path / the "
            f"--train-dataset-name + descriptor args."
        )
    try:
        summary = read_model_summary(model_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No summary_*.json in {model_dir} — this is not a completed model. Train "
            f"it first (train_ensemble.py / train_modelN.py)."
        )
    if not summary.get("training_complete"):
        raise ValueError(
            f"Model at {model_dir} is not a COMPLETED model: its summary_*.json has no "
            f"'training_complete: true' marker (a crashed or partial run). Re-train it "
            f"before evaluating."
        )

    result: Dict[str, Any] = {
        "summary": summary,
        "model_dir": model_dir,
        "gene_locus": summary.get("gene_locus"),
        "classification_mode": summary.get("classification_mode"),
        "reference_class": summary.get("reference_class"),
        "diseases": summary.get("diseases"),
        "clone_id_params": summary.get("clone_id_params"),
        "dataset_counts": summary.get("dataset_counts"),
    }
    if is_ensemble:
        if fold_id is None:
            # Train-all ensemble: inference config is in the summary itself.
            mm_config = summary.get("metamodel_config")
            if not mm_config or "feature_columns" not in mm_config:
                raise ValueError(
                    f"Ensemble summary at {model_dir} has no usable 'metamodel_config' "
                    f"(feature_columns). Is this a train-all ensemble directory?"
                )
            base_ctx = summary.get("base_model_training_context")
            if base_ctx != "train_all_ensemble":
                raise ValueError(
                    f"Ensemble at {model_dir} records base_model_training_context="
                    f"{base_ctx!r}, expected 'train_all_ensemble'. Its base models were "
                    f"NOT trained with the validation set held out (leakage) — not valid "
                    f"for evaluation."
                )
            base_model_paths = summary.get("base_model_paths") or {}
            m2_strategy = summary.get("model2_abstention_strategy")
            metamodel_joblib = model_dir / METAMODEL_JOBLIB
        else:
            # CV ensemble: per-fold metamodel config + run_config for base paths.
            mm_path = model_dir / f"fold_{fold_id}_metamodel_config.json"
            if not mm_path.exists():
                raise FileNotFoundError(
                    f"CV metamodel config not found: {mm_path}. Is fold {fold_id} present "
                    f"in this cv_ensemble directory?"
                )
            with open(mm_path) as f:
                mm_config = json.load(f)
            rc_path = model_dir / "run_config.json"
            if not rc_path.exists():
                raise FileNotFoundError(
                    f"run_config.json not found in {model_dir} — needed to resolve the CV "
                    f"ensemble's base-model paths."
                )
            with open(rc_path) as f:
                run_config = json.load(f)
            base_model_paths = run_config.get("base_model_paths") or {}
            m2_strategy = run_config.get("model2_abstention_strategy")
            metamodel_joblib = model_dir / f"fold_{fold_id}_ridge_cv_metamodel.joblib"
            # CV base models are cv_ensemble by construction (out-of-sample validation),
            # so there is no leakage flag to assert here.
        if not metamodel_joblib.exists():
            raise FileNotFoundError(f"Metamodel artifact not found: {metamodel_joblib}.")
        result.update(
            metamodel_config=mm_config,
            base_model_paths=base_model_paths,
            model2_abstention_strategy=m2_strategy,
            metamodel_joblib=metamodel_joblib,
        )
    return result


# ===========================================================================
# Consistency + class-alignment checks
# ===========================================================================


def check_preprocessing_consistency(test_loader, model_gene_locus: str,
                                     model_clone_id_params: Optional[Dict]) -> None:
    """gene_locus mismatch -> error; clone_id clustering mismatch -> warn (Phase 6.E).

    The clone_id comparison is provenance-aware: it only compares clustering params
    when clone_id was actually COMPUTED (not pre-existing in the source data) on both
    sides; otherwise it is skipped (info log, no spurious warning).
    """
    if model_gene_locus is not None and test_loader.gene_locus != model_gene_locus:
        raise ValueError(
            f"gene_locus mismatch: the model was trained on {model_gene_locus!r} but "
            f"the test dataset loader is {test_loader.gene_locus!r}. A model cannot "
            f"score data from a different locus. Re-run with --gene-locus "
            f"{model_gene_locus}."
        )
    model_params = dict(model_clone_id_params or {})
    test_params = dict(test_loader.clone_id_params)
    model_computed = model_params.get("clone_id_computed")
    test_computed = test_params.get("clone_id_computed")

    if model_computed is False or test_computed is False:
        sides = (["training"] if model_computed is False else []) + \
                (["test"] if test_computed is False else [])
        logger.info(
            f"clone_id was PRE-EXISTING in the source data (not computed by us) on the "
            f"{'/'.join(sides)} dataset(s); the clone definition comes from the input "
            f"data, so there are no clustering params to compare. Skipping the clone_id "
            f"consistency check."
        )
        return
    if model_computed is None or test_computed is None:
        logger.info(
            "clone_id provenance could not be determined for the training and/or test "
            "dataset (no cache stats / old cache format). Skipping the clone_id "
            "consistency check."
        )
        return
    keys = (set(model_params) | set(test_params)) - {"clone_id_computed"}
    diffs = {k: (model_params.get(k), test_params.get(k)) for k in keys
             if model_params.get(k) != test_params.get(k)}
    if diffs:
        logger.warning(
            "clone_id clustering definition differs between the trained model and the "
            "test dataset (train vs test): "
            + "; ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in sorted(diffs.items()))
            + ". This is allowed (e.g. nucleotide vs amino-acid CDR3), but the clone "
            "definitions — and thus downsampling — differ; interpret the cross-dataset "
            "metrics with that in mind."
        )


def align_test_classes(
    meta: pd.DataFrame, training_classes: List[str], *,
    allow_unknown_test_classes: bool, pair_label: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align the test dataset's disease labels to the model's training label space (6.D)."""
    training_set = set(training_classes)
    test_classes = sorted(meta[DISEASE_COL].dropna().unique().tolist())
    test_set = set(test_classes)

    if not (training_set & test_set):
        raise ValueError(
            f"[{pair_label}] No overlap between the model's training classes "
            f"{sorted(training_set)} and the test dataset classes {test_classes}. "
            f"Nothing to evaluate — is this the right test dataset / model?"
        )
    extra = test_set - training_set
    if extra:
        if not allow_unknown_test_classes:
            raise ValueError(
                f"[{pair_label}] The test dataset contains classes the model never saw: "
                f"{sorted(extra)}.\n"
                f"  Model training classes: {sorted(training_set)}\n"
                f"  Test dataset classes:   {test_classes}\n"
                f"The model can only predict its training classes. To evaluate on the "
                f"shared classes anyway (dropping specimens whose true label is not a "
                f"training class), pass --allow-unknown-test-classes."
            )
        n_before = len(meta)
        meta = meta[meta[DISEASE_COL].isin(training_set)].copy()
        logger.warning(
            f"[{pair_label}] --allow-unknown-test-classes: dropped "
            f"{n_before - len(meta)}/{n_before} test specimens whose true label is not a "
            f"training class ({sorted(extra)}); evaluating on {len(meta)} specimens."
        )
    missing = training_set - test_set
    if missing:
        logger.info(
            f"[{pair_label}] Training classes with NO test support (undefined per-class "
            f"metrics): {sorted(missing)}"
        )
    return meta, {
        "training_classes": sorted(training_set),
        "test_classes": test_classes,
        "extra_test_classes": sorted(extra),
        "missing_training_classes": sorted(missing),
    }


# ===========================================================================
# Prediction helpers
# ===========================================================================


def _predict_base_model(
    num: int, model_dir: Path, seqs: pd.DataFrame, meta: pd.DataFrame,
    target_specimens: set, *, gene_locus: str,
    disease_filter: Optional[Tuple[str, str]], embedding_dir: Optional[Path], n_jobs: int,
    fold_id: Optional[int] = None, summary: Optional[dict] = None,
) -> ModelPredictions:
    """Predict one base model. ``fold_id`` selects fold-prefixed CV artifacts
    (``--model-fold-id``); None uses the prefix-less train-all artifacts. ``summary`` is
    the model's own summary — passed through for mode validation and (Model 1) to resolve
    a non-default ``--model1-model-name`` from ``model_names``."""
    if num == 1:
        return predict_model1(model_dir, fold_id, seqs, meta, target_specimens,
                              disease_filter=disease_filter, summary=summary)
    if num == 2:
        return predict_model2(model_dir, fold_id, seqs, meta, target_specimens,
                              gene_locus=gene_locus, disease_filter=disease_filter,
                              n_jobs=n_jobs, summary=summary)
    if num == 3:
        if embedding_dir is None:
            raise ValueError("Model 3 requires --test-embedding-dir with pre-computed embeddings.")
        return predict_model3(model_dir, fold_id, seqs, meta, target_specimens,
                              embedding_dir=embedding_dir, gene_locus=gene_locus,
                              disease_filter=disease_filter, n_jobs=n_jobs, summary=summary)
    raise ValueError(f"Unknown model number: {num}")


def _model_metrics(preds: ModelPredictions, meta_indexed: pd.DataFrame,
                   model_label: str, reference_class: Optional[str],
                   training_classes: List[str]) -> Optional[Dict]:
    """Rich metrics for one base model's predictions, or None if it scored nothing.

    Metrics are computed over the FULL ``training_classes`` label space (not just the
    classes the model happened to output). A model that predicts only a SUBSET of the
    label space (e.g. Model 2 with heavy abstention) has the missing classes padded with
    probability 0 (never argmax-selected), so the confusion matrix / per-class report
    include ALL scored specimens and their true labels — no specimen is silently
    excluded, and the denominators are consistent with the ensemble's.
    """
    proba_df = preds.probabilities
    if len(proba_df) == 0:
        logger.warning(f"    {model_label}: scored 0 specimens — skipping metrics")
        return None
    missing = [c for c in training_classes if c not in proba_df.columns]
    if missing:
        logger.info(
            f"    {model_label}: does not output class(es) {missing}; padding with "
            f"probability 0 so metrics cover the full training label space."
        )
    # reindex to the full label space, in training-class order (drops any stray
    # out-of-label-space column, fills missing classes with 0).
    proba_df = proba_df.reindex(columns=list(training_classes), fill_value=0.0)
    classes = list(training_classes)
    proba = proba_df.values
    y_true = meta_indexed.loc[proba_df.index, DISEASE_COL].values
    y_pred = np.array(classes)[np.argmax(proba, axis=1)]
    return compute_rich_metrics(
        y_true, y_pred, proba, classes, reference_class=reference_class,
        model_label=model_label, n_scored=len(proba_df), n_abstained=preds.n_abstained,
    )


def _assemble_predictions(
    meta_indexed: pd.DataFrame, training_classes: List[str],
    preds_by_model: Dict[int, ModelPredictions],
    ensemble_df: Optional[pd.DataFrame],
    inference_only: bool = False,
) -> List[Dict]:
    """Per-specimen wide predictions: (true label unless inference_only) + each model's +
    the ensemble's output."""
    rows = []
    for specimen in meta_indexed.index:
        row = {
            "specimen_label": specimen,
            "participant_label": meta_indexed.loc[specimen, PARTICIPANT_COL],
        }
        if not inference_only:
            row["true_disease"] = meta_indexed.loc[specimen, DISEASE_COL]
        for num, preds in preds_by_model.items():
            pdf = preds.probabilities
            if specimen in pdf.index:
                probs = pdf.loc[specimen]
                row[f"model{num}_predicted"] = probs.idxmax()
                for cls in training_classes:
                    row[f"model{num}_P({cls})"] = float(probs[cls]) if cls in probs.index else None
            else:
                row[f"model{num}_predicted"] = "ABSTAINED"
                for cls in training_classes:
                    row[f"model{num}_P({cls})"] = None
        if ensemble_df is not None:
            if specimen in ensemble_df.index:
                erow = ensemble_df.loc[specimen]
                row["ensemble_predicted"] = erow["ensemble_predicted"]
                for cls in training_classes:
                    col = f"ensemble_P({cls})"
                    row[col] = float(erow[col]) if col in erow.index else None
            else:
                row["ensemble_predicted"] = "ABSTAINED"
                for cls in training_classes:
                    row[f"ensemble_P({cls})"] = None
        rows.append(row)
    return rows


# ===========================================================================
# Core per-pair evaluation
# ===========================================================================


def evaluate_pair(
    test_loader, *,
    ensemble_dir: Optional[Path],
    standalone_base_dirs: Optional[Dict[int, Path]],
    disease_filter: Optional[Tuple[str, str]],
    gene_locus: str,
    embedding_dir: Optional[Path],
    model2_abstention_strategy: Optional[str],
    allow_unknown_test_classes: bool,
    n_jobs: int,
    test_on_folds: Optional[List[int]] = None,
    model_fold_id: Optional[int] = None,
    inference_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate one pair: base models (+ the ensemble, when an ensemble dir is given).

    model_fold_id : if set, load the CV fold's model (fold-prefixed artifacts); None uses
        the train-all (prefix-less) artifacts.
    inference_only : if True, the test dataset has no ground-truth labels — produce
        per-specimen predictions only (no metrics, no class alignment, no pair-disease
        filtering).
    """
    pair_key = make_pair_name(*disease_filter) if disease_filter else None
    plabel = pair_key or "multiclass"
    reference_class = disease_filter[1] if disease_filter else None
    is_ensemble = ensemble_dir is not None

    # --- Resolve + verify the model(s) and the training label space ---
    if is_ensemble:
        ens = load_trained_model(_pair_dir(ensemble_dir, pair_key), is_ensemble=True,
                                 fold_id=model_fold_id)
        mm_config = ens["metamodel_config"]
        training_classes = list(mm_config["classes"])
        models_included = list(mm_config["models_included"])
        # The ensemble records where its base models live; fail loudly (not with a bare
        # KeyError) if the summary is missing a path for a model the metamodel needs.
        base_model_paths = ens["base_model_paths"]
        missing_paths = [n for n in models_included if not base_model_paths.get(f"model{n}")]
        if missing_paths:
            raise ValueError(
                f"[{plabel}] The metamodel was trained on models {models_included}, but the "
                f"ensemble summary's base_model_paths is missing path(s) for model(s) "
                f"{missing_paths}. The summary is incomplete/corrupt — cannot locate the base "
                f"models to evaluate."
            )
        # base_model_paths already point to the pair-specific dir (binary/mb) — use as-is
        base_dirs = {n: Path(base_model_paths[f"model{n}"]) for n in models_included}
        train_dataset_counts = ens.get("dataset_counts")
        eff_strategy = model2_abstention_strategy or ens.get("model2_abstention_strategy") \
            or "ensemble_abstain"
        if model2_abstention_strategy and ens.get("model2_abstention_strategy") and \
                model2_abstention_strategy != ens["model2_abstention_strategy"]:
            logger.warning(
                f"[{plabel}] --model2-abstention-strategy {model2_abstention_strategy!r} "
                f"differs from the metamodel's trained strategy "
                f"({ens['model2_abstention_strategy']!r}); metrics no longer match the "
                f"trained configuration."
            )
    else:
        models_included = sorted(standalone_base_dirs.keys())
        base_dirs = {n: _pair_dir(standalone_base_dirs[n], pair_key) for n in models_included}
        training_classes = None  # set after loading + cross-checking the base models
        train_dataset_counts = None
        eff_strategy = None

    # Verify each base model dir + preprocessing consistency
    base_infos = {}
    for n in models_included:
        info = load_trained_model(base_dirs[n], is_ensemble=False, fold_id=model_fold_id)
        check_preprocessing_consistency(test_loader, info["gene_locus"], info["clone_id_params"])
        base_infos[n] = info

    if not is_ensemble:
        # All standalone base models MUST share the same training label space — they are
        # evaluated together against the same ground truth, and the prediction CSV /
        # class alignment derive from one label space. Fail loudly on a mismatch rather
        # than silently using the first model's classes.
        per_model_classes = {n: sorted(_infer_classes(base_infos[n])) for n in models_included}
        if len({tuple(c) for c in per_model_classes.values()}) > 1:
            raise ValueError(
                f"[{plabel}] The standalone base models were trained on DIFFERENT class "
                f"sets: "
                + "; ".join(f"model{n}={c}" for n, c in per_model_classes.items())
                + ". They must share the same label space to be evaluated together — "
                f"evaluate them separately, or use models trained on the same dataset/mode."
            )
        training_classes = per_model_classes[models_included[0]]
        train_dataset_counts = base_infos[models_included[0]].get("dataset_counts")

    # --- Load the external test dataset ---
    logger.info(f"[{plabel}] Loading external test dataset...")
    seqs, meta = test_loader.get_all_data(PreprocessingStage.DOWNSAMPLED)
    if test_on_folds is not None:
        seqs, meta = _restrict_to_test_folds(seqs, meta, test_loader, test_on_folds, plabel)

    # Label-dependent steps (pair-disease filtering, class alignment, per-class counts)
    # are skipped in inference-only mode (no ground-truth labels).
    if inference_only:
        alignment = None
        test_dataset_counts = None
        # The base-model predict functions reference a `disease` column to build
        # ground-truth label fields. Inference-only test data has no labels, so
        # attach a clearly-marked placeholder (see INFERENCE_ONLY_PLACEHOLDER_LABEL).
        # It never surfaces: no metrics are computed and `true_disease` is omitted
        # from the predictions output when inference_only=True.
        if DISEASE_COL not in meta.columns:
            meta = meta.copy()
            meta[DISEASE_COL] = INFERENCE_ONLY_PLACEHOLDER_LABEL
    else:
        if disease_filter:
            seqs, meta = filter_to_binary_pair(seqs, meta, disease_filter[0], disease_filter[1])
        meta, alignment = align_test_classes(
            meta, training_classes, allow_unknown_test_classes=allow_unknown_test_classes,
            pair_label=plabel,
        )
        if len(meta) == 0:
            raise ValueError(f"[{plabel}] No test specimens remain after class alignment.")
        test_dataset_counts = get_metadata_class_counts(meta)

    target_specimens = set(meta[SPECIMEN_COL])
    # `raise` (not assert) — a duplicate specimen label would break the .loc-based
    # alignment of predictions to ground truth; must fail loudly even under `python -O`.
    if not meta[SPECIMEN_COL].is_unique:
        raise ValueError(
            f"[{plabel}] Duplicate specimen labels in the test metadata — cannot align "
            f"predictions to specimens. Check the test dataset's metadata."
        )
    meta_indexed = meta.set_index(SPECIMEN_COL)
    logger.info(
        f"[{plabel}] {'Predicting on' if inference_only else 'Evaluating on'} "
        f"{len(target_specimens)} test specimens ({len(seqs):,} sequences)."
    )

    # --- Base-model predictions (+ per-model metrics unless inference-only) ---
    preds_by_model: Dict[int, ModelPredictions] = {}
    base_model_metrics: Dict[int, Dict] = {}
    for n in models_included:
        logger.info(f"  Predicting Model {n}...")
        preds = _predict_base_model(
            n, base_dirs[n], seqs, meta, target_specimens,
            gene_locus=gene_locus, disease_filter=(None if inference_only else disease_filter),
            embedding_dir=embedding_dir, n_jobs=n_jobs, fold_id=model_fold_id,
            summary=base_infos[n]["summary"],
        )
        preds_by_model[n] = preds
        if not inference_only:
            m = _model_metrics(preds, meta_indexed, f"model{n}", reference_class, training_classes)
            if m is not None:
                base_model_metrics[n] = m

    # --- Ensemble (only when an ensemble dir was given) ---
    ensemble_metrics = None
    ensemble_df = None
    if is_ensemble:
        ensemble_metrics, ensemble_df = _evaluate_ensemble(
            ens, preds_by_model, meta_indexed, gene_locus, reference_class,
            eff_strategy, training_classes, target_specimens, plabel,
            inference_only=inference_only,
        )

    predictions_rows = _assemble_predictions(
        meta_indexed, training_classes, preds_by_model, ensemble_df,
        inference_only=inference_only,
    )

    return {
        "pair": plabel,
        "disease_filter": list(disease_filter) if disease_filter else None,
        "reference_class": reference_class,
        "classification_mode": ("multiclass" if disease_filter is None else "binary/multi-binary"),
        "gene_locus": gene_locus,
        "training_classes": training_classes,
        "models_included": models_included,
        "model2_abstention_strategy": eff_strategy,
        "inference_only": inference_only,
        "class_alignment": alignment,
        "training_data_counts": train_dataset_counts,
        "test_data_counts": test_dataset_counts,
        "n_test_specimens": len(target_specimens),
        "base_models": {f"model{n}": m for n, m in base_model_metrics.items()},
        "ensemble": ensemble_metrics,
        "predictions_rows": predictions_rows,
    }


def _evaluate_ensemble(ens, preds_by_model, meta_indexed, gene_locus, reference_class,
                       eff_strategy, training_classes, target_specimens, plabel,
                       inference_only=False):
    """Build the metamodel feature matrix, apply it, and (unless inference_only) compute
    ensemble metrics. Returns (metrics_or_None, per_specimen_predictions_df)."""
    mm_config = ens["metamodel_config"]
    X, _abst_labels, _abst_dis, _fill = build_feature_matrix(
        preds_by_model, gene_locus, reference_class, model2_abstention_strategy=eff_strategy,
    )
    nan_mask = X.isna().any(axis=1)
    if nan_mask.any():
        logger.warning(f"[{plabel}] Dropping {int(nan_mask.sum())}/{len(X)} test specimens "
                       f"with NaN ensemble features (counted as abstentions).")
        X = X[~nan_mask]
    if X.shape[0] == 0:
        # In inference-only mode the point of the run is the per-specimen base-model
        # predictions; don't discard them just because the ensemble can't score. Warn and
        # fall back to base-only (the caller writes predictions with no ensemble columns).
        if inference_only:
            logger.warning(
                f"[{plabel}] All test specimens abstained / had NaN ensemble features — the "
                f"ensemble cannot score any specimen. Writing base-model predictions only."
            )
            return None, None
        raise ValueError(f"[{plabel}] All test specimens abstained / had NaN features — the "
                         f"ensemble cannot score any specimen.")
    expected_cols = list(mm_config["feature_columns"])
    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        if inference_only:
            logger.warning(
                f"[{plabel}] A base model fully abstained on the test data, so metamodel "
                f"feature column(s) {missing[:10]}{'...' if len(missing) > 10 else ''} are not "
                f"all present — cannot apply the ensemble. Writing base-model predictions only."
            )
            return None, None
        raise ValueError(
            f"[{plabel}] Cannot reconstruct the metamodel feature matrix: missing columns "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. A base model likely fully "
            f"abstained on the test data, so the columns the metamodel was trained on are "
            f"not all present. Cannot evaluate the ensemble."
        )
    X = X[expected_cols]

    pipeline = joblib.load(ens["metamodel_joblib"])
    classes = [str(c) for c in pipeline.classes_]
    y_proba = pipeline.predict_proba(X.values)
    y_pred = pipeline.predict(X.values)
    n_abstained = len(target_specimens) - X.shape[0]

    metrics = None
    if not inference_only:
        y_true = meta_indexed.loc[X.index, DISEASE_COL].values
        metrics = compute_rich_metrics(
            y_true, y_pred, y_proba, classes, reference_class=reference_class,
            model_label="ensemble", n_scored=X.shape[0], n_abstained=n_abstained,
        )

    # Per-specimen ensemble predictions frame (for the wide predictions CSV)
    edf = pd.DataFrame(index=X.index)
    edf["ensemble_predicted"] = y_pred
    for j, cls in enumerate(classes):
        edf[f"ensemble_P({cls})"] = y_proba[:, j]
    return metrics, edf


def _restrict_to_test_folds(seqs, meta, test_loader, test_on_folds, plabel):
    """Restrict the test data to specimens whose CV_fold is in ``test_on_folds``.

    Enables evaluating on a specific held-out portion of a test dataset that has CV
    folds defined (e.g. "test on the held-out third"). Independent of how the model was
    trained. Requires a ``CV_fold`` column in the test metadata.
    """
    md = test_loader.metadata
    if FOLD_COL not in md.columns:
        raise ValueError(
            f"[{plabel}] --test-on-folds was given but the test dataset has no "
            f"'{FOLD_COL}' column. Remove --test-on-folds, or use a test dataset that "
            f"defines CV folds."
        )
    available = set(md[FOLD_COL].dropna().astype(int).unique().tolist())
    requested = set(test_on_folds)
    invalid = requested - available
    if invalid:
        raise ValueError(
            f"[{plabel}] --test-on-folds {sorted(requested)} includes fold(s) "
            f"{sorted(invalid)} not present in the test dataset (available: "
            f"{sorted(available)})."
        )
    fold_specimens = set(
        md[md[FOLD_COL].isin(requested)][SPECIMEN_COL]
    )
    meta = meta[meta[SPECIMEN_COL].isin(fold_specimens)].copy()
    seqs = seqs[seqs[SPECIMEN_COL].isin(fold_specimens)].copy()
    logger.info(f"[{plabel}] --test-on-folds {sorted(requested)}: restricted to "
                f"{len(meta)} test specimens.")
    return seqs, meta


def _pair_dir(base: Path, pair_key: Optional[str]) -> Path:
    return base / pair_key if pair_key else base


def _infer_classes(base_info: Dict) -> List[str]:
    """Training classes for a standalone base model, from its summary."""
    summary = base_info["summary"]
    for key in ("model_classes", "classes"):
        if summary.get(key):
            return list(summary[key])
    raise ValueError(
        f"Cannot determine training classes for standalone base model at "
        f"{base_info['model_dir']} (summary has neither 'model_classes' nor 'classes')."
    )


# ===========================================================================
# Pair discovery + top-level evaluation
# ===========================================================================


def discover_pairs(model_dir: Path) -> List[Optional[Tuple[str, str]]]:
    """Discover the disease pairs a model directory holds.

    ``[None]`` for multiclass (a summary lives directly in ``model_dir``), or one
    ``(disease, reference)`` per ``<disease>_vs_<reference>/`` subdir (binary/multi-binary),
    read from each subdir's summary's ``disease_filter``.
    """
    if list(model_dir.glob("summary_*.json")):
        return [None]
    pair_subdirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and "_vs_" in d.name and list(d.glob("summary_*.json"))
    )
    if not pair_subdirs:
        raise FileNotFoundError(
            f"No trained model found in {model_dir}: neither a summary_*.json (multiclass) "
            f"nor any <disease>_vs_<reference>/ pair subdirectory with a summary. Is this a "
            f"completed train-all model directory?"
        )
    pairs = []
    for d in pair_subdirs:
        df = read_model_summary(d).get("disease_filter")
        if not df or len(df) != 2:
            raise ValueError(f"Pair subdirectory {d} has no valid 'disease_filter' in its summary.")
        pairs.append((df[0], df[1]))
    return pairs


def evaluate_external(
    test_loader, *,
    ensemble_dir: Optional[Path],
    standalone_base_dirs: Optional[Dict[int, Path]],
    train_dataset_name: str,
    train_context: str,
    test_dataset_name: str,
    gene_locus: str,
    embedding_dir: Optional[Path],
    model2_abstention_strategy: Optional[str],
    allow_unknown_test_classes: bool,
    n_jobs: int,
    test_on_folds: Optional[List[int]] = None,
    model_fold_id: Optional[int] = None,
    inference_only: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a trained ensemble OR standalone base models on an external dataset.

    Discovers the mode/pairs, evaluates each, writes outputs, and returns a dict keyed
    by pair. Reused directly by Phase 7 (combined train+test).

    test_on_folds : if given, restrict the test dataset to specimens whose ``CV_fold`` is
        in this list (requires a ``CV_fold`` column in the test metadata) — e.g. evaluate
        on the held-out third of a dataset that has CV folds defined.
    """
    discovery_dir = ensemble_dir if ensemble_dir is not None else \
        standalone_base_dirs[sorted(standalone_base_dirs)[0]]
    pairs = discover_pairs(discovery_dir)
    mode = "multiclass" if pairs == [None] else ("binary" if len(pairs) == 1 else "multi-binary")
    kind = "ensemble" if ensemble_dir is not None else "standalone base models"

    # Inference-only is multiclass-only: a binary/multi-binary model is defined PER
    # disease-pair, and applying it requires restricting specimens to each pair's two
    # diseases (a label-based filter) — impossible without ground-truth labels. (The base
    # models also reject a null disease_filter for a binary/multi-binary artifact.)
    if inference_only and mode != "multiclass":
        raise ValueError(
            f"--inference-only is only supported for MULTICLASS models, but this is a "
            f"{mode} model. Binary/multi-binary models are defined per disease-pair and "
            f"need ground-truth labels to restrict specimens to each pair. Evaluate this "
            f"model WITH labels (drop --inference-only), or use a multiclass model for "
            f"label-free inference."
        )

    logger.info(f"External evaluation ({kind}): mode={mode}, {len(pairs)} pair(s), "
                f"test dataset='{test_dataset_name}'.")

    if output_dir is None:
        output_dir = (PROJECT_ROOT / "trained_models" / train_dataset_name / train_context
                      / "evaluated_on" / test_dataset_name / gene_locus / mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_pair = {}
    for pair in pairs:
        result = evaluate_pair(
            test_loader, ensemble_dir=ensemble_dir, standalone_base_dirs=standalone_base_dirs,
            disease_filter=pair, gene_locus=gene_locus, embedding_dir=embedding_dir,
            model2_abstention_strategy=model2_abstention_strategy,
            allow_unknown_test_classes=allow_unknown_test_classes, n_jobs=n_jobs,
            test_on_folds=test_on_folds, model_fold_id=model_fold_id,
            inference_only=inference_only,
        )
        pair_out = output_dir / result["pair"] if pair is not None else output_dir
        _write_pair_outputs(pair_out, result, train_dataset_name, test_dataset_name)
        results_by_pair[result["pair"]] = result

    logger.info(f"\nExternal evaluation complete. Output: {output_dir}")
    return {"mode": mode, "output_dir": str(output_dir), "results_by_pair": results_by_pair}


# ===========================================================================
# Output writing
# ===========================================================================


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return str(x)


def _write_pair_outputs(out_dir: Path, result: Dict[str, Any],
                        train_dataset_name: str, test_dataset_name: str) -> None:
    """Write results.json, predictions.csv, curves/, figures/, and RESULTS.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_rows = result.pop("predictions_rows")

    all_metrics = {**result.get("base_models", {})}
    if result.get("ensemble"):
        all_metrics["ensemble"] = result["ensemble"]

    # --- 1) Write the ESSENTIAL data first (never lose metrics to a plotting hiccup) ---
    # results.json (curve arrays replaced by compact scalar AUC/AP for readability)
    results_json = {
        "timestamp": timestamp,
        "train_dataset": train_dataset_name,
        "test_dataset": test_dataset_name,
        **{k: v for k, v in result.items() if k not in ("base_models", "ensemble")},
        "base_models": {n: compact_curves_for_json(m) for n, m in result.get("base_models", {}).items()},
        "ensemble": compact_curves_for_json(result["ensemble"]) if result.get("ensemble") else None,
    }
    with open(out_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results_json, f, indent=2, default=_json_default)
    pd.DataFrame(predictions_rows).to_csv(out_dir / f"predictions_{timestamp}.csv", index=False)

    # Inference-only: no ground-truth labels → no metrics/curves/figures, just predictions.
    if result.get("inference_only"):
        # Whether an ensemble was run is reflected by the presence of ensemble columns
        # in the per-specimen predictions (result["ensemble"] is always None here).
        has_ensemble = bool(predictions_rows) and "ensemble_predicted" in predictions_rows[0]
        scored = "each base model" + (" + the ensemble" if has_ensemble else "")
        (out_dir / f"RESULTS_{timestamp}.md").write_text(
            f"# External inference (no labels) — {result['pair']}\n\n"
            f"- Train dataset: `{train_dataset_name}`  |  Test dataset: `{test_dataset_name}`\n"
            f"- Gene locus: {result['gene_locus']}  |  Models: {result['models_included']}"
            f"{' + ensemble' if has_ensemble else ''}\n"
            f"- Specimens predicted: {result['n_test_specimens']}\n\n"
            f"Inference-only run: the test dataset has no ground-truth `disease` column, so no "
            f"metrics were computed. Per-specimen predictions ({scored}) "
            f"are in `predictions_{timestamp}.csv`.\n"
        )
        logger.info(f"  Wrote predictions (inference-only) to {out_dir}")
        return

    write_results_md(out_dir / f"RESULTS_{timestamp}.md", results_json)

    # --- 2) Curve CSVs (numbers for re-plotting; non-essential → must not abort the run) ---
    curves_dir = out_dir / "curves"
    for label, metrics in all_metrics.items():
        try:
            save_curve_csvs(curves_dir, label, metrics)
        except Exception as e:  # noqa: BLE001 — curve CSVs are non-essential
            logger.warning(f"  Could not write curve CSVs for {label}: {e}")

    # --- 3) Figures (nice-to-have; a plotting error must not abort the run) ---
    fig_dir = out_dir / "figures"
    for label, metrics in all_metrics.items():
        try:
            save_model_figures(fig_dir, label, metrics, result["training_classes"])
        except Exception as e:  # noqa: BLE001 — figures are non-essential
            logger.warning(f"  Could not render figures for {label}: {e}")
    if all_metrics:
        try:
            save_comparison_figure(fig_dir, all_metrics, result["reference_class"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  Could not render the comparison figure: {e}")
    logger.info(f"  Wrote results to {out_dir}")


# ===========================================================================
# CLI
# ===========================================================================


def _infer_dataset_and_context(model_path: Path, *, is_cv: bool = False) -> Tuple[str, str]:
    """Extract (train_dataset_name, train_context) from a canonical artifact path
    trained_models/<dataset>/<context>/(ensemble|base_models)/...

    These values are used ONLY to name the output directory — the model itself loads
    from the given path regardless. For a non-canonical path (not under
    ``trained_models/``) the segments can't be parsed, so we fall back to
    ``unknown-train-dataset`` and a context inferred from ``is_cv`` (whether
    ``--model-fold-id`` was supplied): ``cv_ensemble`` for a CV model, else
    ``train_all_ensemble``. This keeps the output-dir context label correct even when
    the path can't be parsed.
    """
    parts = model_path.resolve().parts
    if "trained_models" in parts:
        i = parts.index("trained_models")
        if len(parts) > i + 2:
            return parts[i + 1], parts[i + 2]
    return "unknown-train-dataset", ("cv_ensemble" if is_cv else "train_all_ensemble")


def _evaluated_models_included(source: Dict[str, Any], fold_id: Optional[int]) -> set:
    """The set of base-model numbers the evaluated model comprises.

    Standalone: the ``--modelN-dir`` set. Ensemble: ``models_included`` from the ensemble
    summary (identical across pairs, so one summary suffices). Cheap peek used for
    up-front argument validation (--inline-embeddings needs Model 3;
    --model2-abstention-strategy needs Model 2).
    """
    if source["standalone_base_dirs"] is not None:
        return set(source["standalone_base_dirs"])
    ens_dir = source["ensemble_dir"]
    pairs = discover_pairs(ens_dir)
    summary_dir = ens_dir if pairs == [None] else _pair_dir(ens_dir, make_pair_name(*pairs[0]))
    info = load_trained_model(summary_dir, is_ensemble=True, fold_id=fold_id)
    return set(info["metamodel_config"]["models_included"])


def _evaluated_mode(source: Dict[str, Any]) -> str:
    """Classification mode of the evaluated model: 'multiclass' | 'binary' | 'multi-binary'
    (from the pair layout). Cheap peek used to fail fast on --inference-only + non-multiclass."""
    discovery_dir = source["ensemble_dir"] if source["ensemble_dir"] is not None else \
        source["standalone_base_dirs"][sorted(source["standalone_base_dirs"])[0]]
    pairs = discover_pairs(discovery_dir)
    return "multiclass" if pairs == [None] else ("binary" if len(pairs) == 1 else "multi-binary")


def _resolve_model_source(args, parser) -> Dict[str, Any]:
    """Validate the (mutually-exclusive) model-source args and resolve them.

    Exactly one of: --ensemble-dir | --modelN-dir(s) | --train-dataset-name.
    Returns a spec dict for evaluate_external.
    """
    explicit_base = {n: getattr(args, f"model{n}_dir")
                     for n in (1, 2, 3) if getattr(args, f"model{n}_dir") is not None}
    has_ensemble = args.ensemble_dir is not None
    has_base = len(explicit_base) > 0
    has_convention = args.train_dataset_name is not None

    if has_ensemble and has_base:
        parser.error(
            "--ensemble-dir and --modelN-dir are mutually exclusive: evaluate EITHER an "
            "ensemble (--ensemble-dir) OR standalone base models (--modelN-dir), not both."
        )
    if (has_ensemble or has_base) and has_convention:
        parser.error(
            "Explicit artifact dirs (--ensemble-dir / --modelN-dir) and --train-dataset-name "
            "are mutually exclusive: specify the model EITHER by explicit path OR by "
            "convention (--train-dataset-name + descriptors), not both."
        )
    if not (has_ensemble or has_base or has_convention):
        parser.error(
            "No model specified. Provide one of: --ensemble-dir DIR (an ensemble), "
            "--modelN-dir DIR (standalone base models), or --train-dataset-name NAME "
            "(resolve the ensemble by convention, with --gene-locus / --classification-mode "
            "/ --output-suffix)."
        )
    # --model-fold-id selects a CV-trained model (cv_ensemble). Without it, train-all.
    is_cv = args.model_fold_id is not None
    convention_context = "cv_ensemble" if is_cv else "train_all_ensemble"

    if has_convention:
        ensemble_dir = get_ensemble_output_dir(
            dataset_name=args.train_dataset_name, classification_mode=args.classification_mode,
            gene_locus=args.gene_locus, output_suffix=args.output_suffix,
            training_context=convention_context,
        )
        return {"ensemble_dir": ensemble_dir, "standalone_base_dirs": None,
                "train_dataset_name": args.train_dataset_name,
                "train_context": convention_context, "model_fold_id": args.model_fold_id}
    if has_ensemble:
        ds, ctx = _infer_dataset_and_context(args.ensemble_dir, is_cv=is_cv)
        return {"ensemble_dir": args.ensemble_dir, "standalone_base_dirs": None,
                "train_dataset_name": ds, "train_context": ctx,
                "model_fold_id": args.model_fold_id}
    # standalone base models
    if args.model2_abstention_strategy is not None:
        parser.error(
            "--model2-abstention-strategy is only meaningful for an ensemble (it controls "
            "how Model 2 abstentions are handled in the metamodel feature matrix). It does "
            "nothing for standalone base-model evaluation — remove it."
        )
    ds, ctx = _infer_dataset_and_context(explicit_base[sorted(explicit_base)[0]], is_cv=is_cv)
    return {"ensemble_dir": None, "standalone_base_dirs": explicit_base,
            "train_dataset_name": ds, "train_context": ctx,
            "model_fold_id": args.model_fold_id}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Mal-ID-Lite models on an external dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    # --- Test dataset ---
    g = parser.add_argument_group("test dataset")
    g.add_argument("--test-cache-dir", type=Path, required=True,
                   help="Cache directory for the test dataset. If already built, it is used "
                        "and --test-data-dir is NOT needed.")
    g.add_argument("--test-metadata-path", type=Path, default=None,
                   help="Test metadata TSV (needs a 'disease' column). OPTIONAL: if omitted, "
                        "the cache's metadata_processed.tsv is used. If given, it must be "
                        "consistent with the cache (else the loader errors).")
    g.add_argument("--test-data-dir", type=Path, default=None,
                   help="Raw AIRR data dir for the test dataset. Used ONLY to build the cache "
                        "if it does not exist yet; ignored when the cache is present.")
    g.add_argument("--test-dataset-name", type=str, default=None,
                   help="Identifier for the test dataset in output paths (default: cache dir name).")
    g.add_argument("--test-embedding-dir", type=Path, default=None,
                   help="TEST-dataset ESM-2 embeddings for Model 3 "
                        "(default: <test-cache-dir>/embeddings). Must be pre-computed "
                        "unless --inline-embeddings is given.")
    g.add_argument("--inline-embeddings", action="store_true",
                   help="Compute any missing Model 3 test embeddings on the fly (into "
                        "--test-embedding-dir) instead of requiring them pre-computed. "
                        "Uses --device / --embedding-batch-size. Only valid when the "
                        "evaluated model includes Model 3 (errors otherwise).")
    g.add_argument("--device", type=str, default=None,
                   help="Device for --inline-embeddings ESM-2 computation "
                        "('cuda' / 'mps' / 'cpu'; default: auto-detect).")
    g.add_argument("--embedding-batch-size", type=int, default=None,
                   help="Batch size for --inline-embeddings ESM-2 computation "
                        "(default: auto per device).")

    # --- Model source (EXACTLY ONE of the three ways; mixing them is an error) ---
    g = parser.add_argument_group(
        "model source (choose ONE way; see below)",
        "Specify the trained model in exactly one way: (1) --ensemble-dir DIR; "
        "(2) one or more --modelN-dir DIR (standalone base models, no metamodel); or "
        "(3) --train-dataset-name NAME (resolve the ensemble by convention). "
        "--ensemble-dir and --modelN-dir are mutually exclusive; explicit dirs and "
        "--train-dataset-name are mutually exclusive.")
    g.add_argument("--ensemble-dir", type=Path, default=None,
                   help="Explicit path to a trained ENSEMBLE directory (base models are "
                        "resolved automatically from its summary).")
    g.add_argument("--model1-dir", type=Path, default=None,
                   help="Explicit Model 1 directory (standalone base-model evaluation).")
    g.add_argument("--model2-dir", type=Path, default=None,
                   help="Explicit Model 2 directory (standalone base-model evaluation).")
    g.add_argument("--model3-dir", type=Path, default=None,
                   help="Explicit Model 3 directory (standalone base-model evaluation).")
    g.add_argument("--train-dataset-name", type=str, default=None,
                   help="Resolve the ENSEMBLE path by convention (with --gene-locus, "
                        "--classification-mode, --output-suffix).")
    g.add_argument("--output-suffix", type=str, default=None,
                   help="Ensemble output suffix used at training time (convention path only).")
    g.add_argument("--classification-mode", type=str, default="multiclass",
                   choices=["multiclass", "binary", "multi-binary"],
                   help="Only used to resolve the convention path; the actual mode is "
                        "inferred from the model directory.")
    g.add_argument("--gene-locus", type=str, default="TCR", choices=["TCR", "BCR"])

    # --- Evaluation behavior ---
    g = parser.add_argument_group("evaluation behavior")
    g.add_argument("--allow-unknown-test-classes", action="store_true",
                   help="If the test dataset has classes the model never saw, drop those "
                        "specimens and evaluate on the shared classes (default: error).")
    g.add_argument("--model2-abstention-strategy", type=str, default=None,
                   choices=list(MODEL2_ABSTENTION_STRATEGIES),
                   help="Override the Model 2 abstention strategy for the ENSEMBLE (default: "
                        "the strategy the metamodel was trained with). Error if used without "
                        "an ensemble.")

    # --- Test-set / model selection ---
    g = parser.add_argument_group("test-set / model selection")
    g.add_argument("--test-on-folds", nargs="+", type=int, default=None,
                   help="Restrict the test set to specimens with these CV_fold values "
                        "(requires a CV_fold column in the test metadata). E.g. evaluate on "
                        "the held-out third of a dataset that has CV folds defined.")
    g.add_argument("--model-fold-id", type=int, default=None,
                   help="Evaluate a specific CV fold's model instead of a train-all model. "
                        "Fold i's model was TRAINED on all folds EXCEPT i (i was its held-out "
                        "test fold), so for a clean held-out evaluation on the same dataset "
                        "pair it with --test-on-folds i. Resolves cv_ensemble artifacts.")
    g.add_argument("--inference-only", action="store_true",
                   help="Predict WITHOUT ground-truth labels — write per-specimen predictions "
                        "only, no metrics. Allows a test dataset with no 'disease' column. "
                        "Without this flag, a missing 'disease' column is an error (guards "
                        "against an accidental column-name mismatch silently skipping eval).")

    # --- Other ---
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the output directory.")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    add_clone_id_args(parser)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.n_jobs < 1:
        parser.error(f"--n-jobs must be >= 1, got {args.n_jobs}.")
    if args.inference_only and args.allow_unknown_test_classes:
        parser.error(
            "--inference-only and --allow-unknown-test-classes are contradictory: "
            "inference-only has no ground-truth labels, so there is nothing to align or "
            "filter by class. Drop one of them."
        )

    # Leakage warning (6.G.1): a fold-i model was trained on all folds except i, so
    # scoring folds other than i is testing on its training data (only matters when the
    # test dataset IS the dataset the model was trained on — we can't detect that, so warn).
    if args.model_fold_id is not None and args.test_on_folds is not None:
        other = [f for f in args.test_on_folds if f != args.model_fold_id]
        if other:
            logger.warning(
                f"--model-fold-id {args.model_fold_id} loads a model TRAINED on all folds "
                f"except {args.model_fold_id}; --test-on-folds includes {other}, which that "
                f"model was trained on. If this test dataset is the same one the model was "
                f"trained on, those folds are training data (leakage). For a clean held-out "
                f"evaluation, use --test-on-folds {args.model_fold_id}."
            )

    source = _resolve_model_source(args, parser)

    # Cheap up-front peek at which base models the evaluated model comprises (no disk I/O
    # for standalone), so the arg-combination guards below fail fast — BEFORE loading the
    # test dataset, clone-id clustering, or (worst case) computing ESM-2 embeddings.
    _models_included = _evaluated_models_included(source, args.model_fold_id)

    # --inference-only is multiclass-only: a binary/multi-binary model is defined per
    # disease-pair and needs labels to restrict specimens to each pair. Fail fast here
    # (a deeper guard exists in evaluate_external, but it runs only after the expensive
    # data-load / embedding steps below). Compute the mode lazily (it reads the model
    # dir) only when inference-only is requested, so unrelated guards don't pay for it.
    if args.inference_only:
        _mode = _evaluated_mode(source)
        if _mode != "multiclass":
            parser.error(
                f"--inference-only is only supported for MULTICLASS models, but the selected "
                f"model is {_mode}. Binary/multi-binary models are defined per disease-pair "
                f"and need ground-truth labels to restrict specimens to each pair. Evaluate "
                f"WITH labels (drop --inference-only), or use a multiclass model."
            )

    # --inline-embeddings only makes sense when Model 3 is part of the evaluated model
    # (embeddings are used ONLY by Model 3) — else computing them here is wasted work.
    if args.inline_embeddings and 3 not in _models_included:
        parser.error(
            "--inline-embeddings was given, but the model being evaluated does NOT include "
            "Model 3 — embeddings are used only by Model 3, so computing them here would be "
            "wasted work. Remove --inline-embeddings, or evaluate a model that includes Model 3."
        )

    # --model2-abstention-strategy only does something when Model 2 is in the ensemble
    # (it controls how M2 abstentions are filled in the metamodel feature matrix). The
    # standalone case is already rejected in _resolve_model_source; here catch a Model-2-
    # less ensemble so the flag is never a silent no-op.
    if (args.model2_abstention_strategy is not None
            and source["ensemble_dir"] is not None and 2 not in _models_included):
        parser.error(
            "--model2-abstention-strategy was given, but this ensemble does NOT include "
            f"Model 2 (models_included={sorted(_models_included)}); the flag would do "
            "nothing. Remove it, or evaluate an ensemble that includes Model 2."
        )

    # --device / --embedding-batch-size only apply to on-the-fly embedding computation,
    # which requires --inline-embeddings; flag them rather than silently ignore.
    if not args.inline_embeddings:
        _stray = [f for f, v in (("--device", args.device),
                                 ("--embedding-batch-size", args.embedding_batch_size))
                  if v is not None]
        if _stray:
            parser.error(
                f"{' and '.join(_stray)} only appl{'y' if len(_stray) > 1 else 'ies'} to "
                f"on-the-fly embedding computation, which requires --inline-embeddings. Add "
                f"--inline-embeddings, or remove {' / '.join(_stray)}."
            )

    test_dataset_name = args.test_dataset_name or args.test_cache_dir.name

    clone_id_kwargs = get_clone_id_kwargs(args)
    # --inference-only allows a test dataset with no ground-truth 'disease' column
    # (label-free prediction). Without it, a missing 'disease' column errors at metadata
    # load (guards against an accidental column-name mismatch silently skipping eval).
    test_loader = MalIDPublishedDataLoader(
        data_dir=args.test_data_dir, metadata_path=args.test_metadata_path,
        gene_locus=args.gene_locus, verbose=args.verbose, cache_dir=args.test_cache_dir,
        require_disease=not args.inference_only,
        **(clone_id_kwargs or {}),
    )
    if test_loader.cache_dir is not None:
        test_loader.precompute_clone_ids(n_jobs=args.n_jobs)
    # Trigger metadata validation up front (raises with a clear message — including the
    # inference-only hint — if a required column such as 'disease' is missing).
    _ = test_loader.metadata

    # F5: validate --test-on-folds against the loaded metadata up front, before
    # evaluate_external loads + downsamples the whole test set (the deeper check in
    # _restrict_to_test_folds runs only after that expensive load).
    if args.test_on_folds is not None:
        if FOLD_COL not in test_loader.metadata.columns:
            parser.error(
                f"--test-on-folds was given, but the test metadata has no '{FOLD_COL}' "
                f"column. Remove --test-on-folds, or use a test dataset with CV folds defined."
            )
        _available_folds = set(
            int(f) for f in test_loader.metadata[FOLD_COL].dropna().unique()
        )
        _missing_folds = sorted(set(args.test_on_folds) - _available_folds)
        if _missing_folds:
            parser.error(
                f"--test-on-folds {_missing_folds} not present in the test metadata's "
                f"'{FOLD_COL}' values {sorted(_available_folds)}."
            )

    embedding_dir = args.test_embedding_dir or (args.test_cache_dir / "embeddings")

    # --inline-embeddings: compute any missing Model 3 test embeddings up front (resume
    # skips already-present participants). Requires the raw data or a built cache.
    if args.inline_embeddings:
        from malid_lite.training.compute_model3_embeddings import compute_all_embeddings
        logger.info("--inline-embeddings: computing/refreshing test-dataset ESM-2 embeddings...")
        compute_all_embeddings(
            metadata_path=test_loader.metadata_path, cache_dir=args.test_cache_dir,
            data_dir=args.test_data_dir, device=args.device,
            batch_size=args.embedding_batch_size, verbose=args.verbose,
            gene_locus=args.gene_locus, clone_id_kwargs=clone_id_kwargs,
            output_embedding_dir=embedding_dir, require_disease=not args.inference_only,
        )

    evaluate_external(
        test_loader, ensemble_dir=source["ensemble_dir"],
        standalone_base_dirs=source["standalone_base_dirs"],
        train_dataset_name=source["train_dataset_name"], train_context=source["train_context"],
        test_dataset_name=test_dataset_name, gene_locus=args.gene_locus,
        embedding_dir=embedding_dir, model2_abstention_strategy=args.model2_abstention_strategy,
        allow_unknown_test_classes=args.allow_unknown_test_classes, n_jobs=args.n_jobs,
        test_on_folds=args.test_on_folds, model_fold_id=source["model_fold_id"],
        inference_only=args.inference_only, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
