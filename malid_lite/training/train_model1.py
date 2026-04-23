#!/usr/bin/env python
"""Train and evaluate Model 1 (Repertoire Classifier).

Classification modes
--------------------
multiclass
    A single N-class classifier trained on all disease classes. Default.
    If the data has exactly 2 classes, proceeds normally (binary multiclass).

binary
    One binary classifier for a single disease-vs-reference pair.
    Default (no --diseases): requires exactly 2 disease classes in the data.
    With --diseases <disease>: pick one specific disease from any N-class dataset.
    Use --reference-class to specify which class is the reference/negative.
    If --reference-class is omitted with 2-class data, alphabetical order is used.
    If --reference-class is omitted with N-class data and --diseases is given, it is required.

multi-binary
    One independent binary classifier per disease vs. the reference class.
    Default (no --diseases): trains all N-1 non-reference diseases.
    With --diseases <d1> <d2> ...: trains only the specified subset.
    Requires --reference-class when the data has more than 2 classes.

Output directory structure
--------------------------
cv_single_model (default):
  multiclass:   trained_models/<dataset>/cv_single_model/model1/multiclass/<locus>/
  binary:       trained_models/<dataset>/cv_single_model/model1/binary/<locus>/<pair>/
  multi-binary: trained_models/<dataset>/cv_single_model/model1/binary/<locus>/<pair1>/
                                                                                <pair2>/...

cv_ensemble:
  multiclass:   trained_models/<dataset>/cv_ensemble/base_models/<locus>/model1/multiclass/
  binary:       trained_models/<dataset>/cv_ensemble/base_models/<locus>/model1/binary/<pair>/

With --output-suffix <suffix>, the mode directory gets "__<suffix>" appended:
    trained_models/<dataset>/cv_single_model/model1/multiclass__<suffix>/<locus>/

Both binary and multi-binary write to the same binary/<gene_locus>/ subtree, so artifacts
for the same pair are identical regardless of which mode produced them.

Artifacts per fold
------------------
    fold_<id>_<model_name>_model.pkl       — fitted RepertoireClassifier
    fold_<id>_<model_name>_v_genes.json    — V genes kept after frequency filtering
    fold_<id>_<model_name>_results.json    — per-fold evaluation metrics
    summary_<timestamp>.json              — full run summary (all folds aggregated)
    training_<timestamp>.log              — mirrored log

A predictions CSV is written per run alongside other artifacts:
    multiclass:   <model_name>_multiclass_predictions.csv
    binary:       <disease>_vs_<reference>/<model_name>_binary_predictions.csv

Multiclass columns: participant_label, specimen_label, true_disease, predicted_disease,
    score_<class1>, score_<class2>, ..., malid_cross_validation_fold_id_when_in_test_set

Binary columns: participant_label, specimen_label, disease_label (0/1), disease_label_str,
    disease_model, model_score (P(disease)), malid_cross_validation_fold_id_when_in_test_set

Usage examples
--------------
    # Multiclass (default)
    python malid_lite/training/train_model1.py

    # Binary (2-class data)
    python malid_lite/training/train_model1.py \\
        --classification-mode binary --reference-class "Healthy/Background"

    # Multi-binary (N-class data, one model per disease vs Healthy/Background)
    python malid_lite/training/train_model1.py \\
        --classification-mode multi-binary --reference-class "Healthy/Background"

    # Binary for a single disease from N-class data
    python malid_lite/training/train_model1.py \\
        --classification-mode binary --reference-class "Healthy/Background" --diseases Covid19

    # Train only fold 0
    python malid_lite/training/train_model1.py --fold-ids 0

    # Specify model variant and n_pcs
    python malid_lite/training/train_model1.py --model-name lasso_cv --n-pcs 15

    # Run with a suffix (saves to multiclass__no_pca/ instead of multiclass/)
    python malid_lite/training/train_model1.py --output-suffix no_pca --n-pcs 0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

# Add project root to path (malid_lite/training/ → malid_lite/ → project root)
# Must come before any malid_lite imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Custom multiclass metrics that handle unnormalized probabilities and missing
# labels gracefully. Matches the original Mal-ID paper's evaluation methodology.
from malid_lite.utils import multiclass_metrics

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite.models.model1_repertoire import RepertoireClassifier, V_GENE_COL
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    VALID_TRAINING_CONTEXTS,
    aggregate_fold_results,
    filter_to_binary_pair,
    generate_results_md,
    get_dataset_disease_classes,
    get_dataset_fold_ids,
    get_model_output_dir,
    make_pair_name,
    run_training_orchestration,
    save_per_pair_results,
    validate_mode_and_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent



# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def filter_rare_v_genes(sequences: pd.DataFrame, threshold_quantile: float = 0.5) -> List[str]:
    """Return V genes above the frequency median (bottom 50% removed)."""
    freq = sequences[V_GENE_COL].value_counts(normalize=True)
    threshold = freq.quantile(threshold_quantile)
    kept = freq[freq >= threshold].index.tolist()
    removed = freq[freq < threshold].index.tolist()
    logger.info(
        f"  V gene filtering: keeping {len(kept)}/{len(freq)} "
        f"(removed {len(removed)} below freq {threshold:.4f})"
    )
    return kept


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    fold_id: int,
    model_name: str,
    n_train: int,
    n_test: int,
    n_features: int,
    reference_class: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Compute evaluation metrics for one fold.

    Model 1 never abstains, so n_abstained=0 and n_scored=n_test always.

    Parameters
    ----------
    y_true       : String class labels for test specimens.
    reference_class : Reference/negative class. When provided and data has exactly 2 classes,
        also computes auroc_binary and auprc_binary with disease as positive class.

    Returns
    -------
    (metrics, raw_preds)
        metrics   : JSON-serializable dict of evaluation metrics.
        raw_preds : {"y_true", "y_pred", "y_proba", "classes"} for cross-fold aggregation.
    """
    n_correct = int(accuracy_score(y_true, y_pred, normalize=False))
    results = {
        "fold_id": fold_id,
        "model_name": model_name,
        "n_scored": n_test,
        "n_abstained": 0,
        "abstention_rate": 0.0,
        "n_train": n_train,
        "n_features": n_features,
        "accuracy": n_correct / n_test if n_test > 0 else 0.0,
    }

    # Multiclass metrics: only meaningful for 3+ classes.
    # For binary (2-class), these are left as None and the binary-specific
    # auroc_binary / auprc_binary below are used instead.
    # Uses custom multiclass_metrics (from the original Mal-ID paper) which
    # handle unnormalized probabilities and missing labels gracefully.
    if len(classes) >= 3:
        try:
            results["auroc_ovo_weighted"] = float(multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average="weighted",
                multi_class="ovo",
                labels=classes,
            ))
        except ValueError as e:
            logger.warning(f"  AUROC OvO failed: {e}")
            results["auroc_ovo_weighted"] = None

        try:
            results["auprc_ovo_weighted"] = float(multiclass_metrics.auprc(
                y_true, y_proba,
                average="weighted",
                multi_class="ovo",
                labels=classes,
            ))
        except ValueError as e:
            logger.warning(f"  AUPRC OvO failed: {e}")
            results["auprc_ovo_weighted"] = None

        # Per-class AUROC OvR
        auroc_ovr_per_class = {}
        try:
            per_class_scores = multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average=None,
                multi_class="ovr",
                labels=classes,
            )
            for cls, score in zip(classes, per_class_scores):
                auroc_ovr_per_class[str(cls)] = float(score)
        except ValueError as e:
            logger.warning(f"  Per-class AUROC OvR failed: {e}")
            for cls in classes:
                auroc_ovr_per_class[str(cls)] = None
        results["auroc_ovr_per_class"] = auroc_ovr_per_class
    else:
        results["auroc_ovo_weighted"] = None
        results["auprc_ovo_weighted"] = None
        results["auroc_ovr_per_class"] = None

    try:
        results["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    except ValueError as e:
        logger.warning(f"  Log loss failed: {e}")
        results["log_loss"] = None

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    results["classes"] = [str(c) for c in classes]
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # Binary AUROC/AUPRC when exactly 2 classes and reference_class is known.
    # Matches model 2 binary evaluate_on_test methodology.
    if len(classes) == 2 and reference_class is not None:
        str_classes = [str(c) for c in classes]
        if str(reference_class) in str_classes:
            disease_class = next(c for c in str_classes if c != str(reference_class))
            disease_idx = str_classes.index(disease_class)
            y_score = y_proba[:, disease_idx]
            y_binary = (y_true == disease_class).astype(int)
            try:
                results["auroc_binary"] = float(roc_auc_score(y_binary, y_score))
            except ValueError as e:
                logger.warning(f"  Binary AUROC failed: {e}")
                results["auroc_binary"] = None
            try:
                results["auprc_binary"] = float(average_precision_score(y_binary, y_score))
            except ValueError as e:
                logger.warning(f"  Binary AUPRC failed: {e}")
                results["auprc_binary"] = None

    raw_preds = {
        "y_true": y_true if isinstance(y_true, np.ndarray) else np.asarray(y_true),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
    }
    return results, raw_preds


# ---------------------------------------------------------------------------
# Cross-fold aggregation — imported from training_utils
# ---------------------------------------------------------------------------
# _mean_std_per_fold and aggregate_fold_results are imported above.


# ---------------------------------------------------------------------------
# Fold loop (shared by all classification modes)
# ---------------------------------------------------------------------------

def _run_fold_loop(
    loader: MalIDPublishedDataLoader,
    fold_ids: List[int],
    output_dir: Path,
    model_name: str,
    model_params: Dict,
    verbose: int,
    disease_filter: Optional[Tuple[str, str]] = None,
    training_context: str = "cv_single_model",
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Run training + evaluation for all specified folds.

    Parameters
    ----------
    disease_filter : Optional (disease, reference_class) tuple. If provided,
        sequences and metadata are filtered to {disease, reference_class} specimens
        before training. Used for binary and multi-binary modes.

    Returns
    -------
    (all_eval_results, aggregated_by_model)
        all_eval_results    : List of per-fold metric dicts.
        aggregated_by_model : {model_name: aggregated_metrics_dict}
    """
    all_eval_results: List[Dict] = []
    raw_preds_list: List[Optional[Dict]] = []
    predictions_rows: List[Dict] = []  # for binary or multiclass predictions CSV

    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_id in fold_ids:
        pair_tag = (
            f" [{make_pair_name(disease_filter[0], disease_filter[1])}]"
            if disease_filter else ""
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_id}{pair_tag}")
        logger.info(f"{'='*60}")

        # ------------------------------------------------------------------
        # Load + optionally filter training data
        # ------------------------------------------------------------------
        logger.info("Loading training data...")
        train_data, train_meta = loader.get_fold_data(
            fold_id=fold_id,
            fold_label="train",
            preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
        )

        # Filter to the participants assigned to this training context.
        # cv_single_model: uses ts1 + ts2 (all train participants)
        # cv_ensemble: uses ts1 + ts2 (excludes validation participants)
        train_participants = set(loader.get_split_participants(
            fold_id, training_context, ["train_smaller1", "train_smaller2"]
        ))
        mask = train_data["participant_label"].isin(train_participants)
        train_data = train_data[mask].copy()
        train_meta = train_meta[
            train_meta["participant_label"].isin(train_participants)
        ].copy()

        # Assertions: split filtering must produce non-empty data with expected
        # participant counts. Empty data indicates a bug in split generation or
        # a mismatch between fold data and split files.
        actual_participants = train_data["participant_label"].nunique()
        assert len(train_data) > 0, (
            f"Training data is empty after split filtering (fold {fold_id}, "
            f"context={training_context}). Expected {len(train_participants)} participants."
        )
        assert actual_participants == len(train_participants), (
            f"Training participant count mismatch: got {actual_participants}, "
            f"expected {len(train_participants)} (fold {fold_id}, context={training_context})"
        )

        if disease_filter:
            train_data, train_meta = filter_to_binary_pair(
                train_data, train_meta, disease_filter[0], disease_filter[1]
            )
        logger.info(
            f"  Train fold: {len(train_meta)} specimens, "
            f"{len(train_data):,} sequences"
        )

        # ------------------------------------------------------------------
        # Filter rare V genes (on training data only)
        # ------------------------------------------------------------------
        kept_v_genes = filter_rare_v_genes(train_data)
        train_data = train_data[train_data[V_GENE_COL].isin(kept_v_genes)].copy()

        # ------------------------------------------------------------------
        # Extract features and train
        # ------------------------------------------------------------------
        model = RepertoireClassifier(verbose=verbose, **model_params)
        start_time = datetime.now()

        X_train = model.extract_features(sequences=train_data, metadata=train_meta)
        train_meta_aligned = train_meta.set_index(SPECIMEN_COL).loc[X_train.index]
        y_train = train_meta_aligned[DISEASE_COL]
        groups_train = train_meta_aligned[PARTICIPANT_COL]

        logger.info(
            f"  Features: {X_train.shape[0]} specimens x {X_train.shape[1]} features"
        )
        logger.info(f"  Training {model_name}...")
        model.fit(X_train, y_train, groups=groups_train)
        train_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"  Training done in {int(train_time)}s")

        # ------------------------------------------------------------------
        # Save model artifact + V gene list
        # ------------------------------------------------------------------
        model_file = output_dir / f"fold_{fold_id}_{model_name}_model.pkl"
        model.save(model_file)

        v_genes_file = output_dir / f"fold_{fold_id}_{model_name}_v_genes.json"
        with open(v_genes_file, "w") as f:
            json.dump(kept_v_genes, f, indent=2)

        # ------------------------------------------------------------------
        # Load + optionally filter test data
        # ------------------------------------------------------------------
        logger.info("Loading test data...")
        test_data, test_meta = loader.get_fold_data(
            fold_id=fold_id,
            fold_label="test",
            preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
        )
        if disease_filter:
            test_data, test_meta = filter_to_binary_pair(
                test_data, test_meta, disease_filter[0], disease_filter[1]
            )
        # Align V genes to training
        test_data = test_data[test_data[V_GENE_COL].isin(kept_v_genes)].copy()

        logger.info(
            f"  Test fold: {len(test_meta)} specimens, "
            f"{len(test_data):,} sequences"
        )

        # ------------------------------------------------------------------
        # Extract test features and predict
        # ------------------------------------------------------------------
        X_test = model.extract_features(
            sequences=test_data,
            metadata=test_meta,
            train_vj_columns=model.train_vj_columns_,
        )
        test_meta_aligned = test_meta.set_index(SPECIMEN_COL).loc[X_test.index]
        y_test = test_meta_aligned[DISEASE_COL].values  # string labels

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        ref_class = disease_filter[1] if disease_filter else None
        eval_result, raw_preds = evaluate_on_test(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=model.classes_,
            fold_id=fold_id,
            model_name=model_name,
            n_train=X_train.shape[0],
            n_test=X_test.shape[0],
            n_features=X_train.shape[1],
            reference_class=ref_class,
        )

        if disease_filter:
            eval_result["disease"] = disease_filter[0]
            eval_result["reference_class"] = disease_filter[1]

        # Log per-fold summary — use binary AUROC for 2-class, OvO for multiclass
        auroc_val = eval_result.get("auroc_binary") or eval_result.get("auroc_ovo_weighted")
        auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A"
        mcc_str = f"{eval_result['mcc']:.4f}" if eval_result.get("mcc") is not None else "N/A"
        logger.info(
            f"  {model_name}: accuracy={eval_result['accuracy']:.4f} "
            f"AUROC={auroc_str} MCC={mcc_str}"
        )

        # Save per-fold results JSON
        results_file = output_dir / f"fold_{fold_id}_{model_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(
                eval_result, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )

        all_eval_results.append(eval_result)
        raw_preds_list.append(raw_preds)

        # Collect for predictions CSV (one row per test specimen)
        if disease_filter:
            # Binary: one score column (P(disease))
            disease = disease_filter[0]
            classes_list = [str(c) for c in model.classes_]
            disease_idx = classes_list.index(str(disease))
            participant_labels = test_meta_aligned[PARTICIPANT_COL].values
            for specimen, participant, true_disease, score in zip(
                X_test.index,
                participant_labels,
                y_test,
                y_proba[:, disease_idx],
            ):
                predictions_rows.append({
                    "participant_label": participant,
                    "specimen_label": specimen,
                    "disease_label": int(true_disease == disease),
                    "disease_label_str": true_disease,
                    "disease_model": disease,
                    "model_score": float(score),
                    "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                })
        else:
            # Multiclass: one score column per class
            class_names = [str(c) for c in model.classes_]
            participant_labels = test_meta_aligned[PARTICIPANT_COL].values
            for specimen, participant, true_d, pred_d, proba_row in zip(
                X_test.index,
                participant_labels,
                y_test,
                y_pred,
                y_proba,
            ):
                row = {
                    "participant_label": participant,
                    "specimen_label": specimen,
                    "true_disease": true_d,
                    "predicted_disease": pred_d,
                    "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                }
                for cls, score in zip(class_names, proba_row):
                    row[f"score_{cls}"] = float(score)
                predictions_rows.append(row)

    # ------------------------------------------------------------------
    # Save predictions CSV (all folds combined)
    # ------------------------------------------------------------------
    if disease_filter and predictions_rows:
        predictions_df = pd.DataFrame(predictions_rows, columns=[
            "participant_label", "specimen_label", "disease_label", "disease_label_str",
            "disease_model", "model_score",
            "malid_cross_validation_fold_id_when_in_test_set",
        ])
        predictions_file = output_dir / f"{model_name}_binary_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(
            f"  Binary predictions saved: {predictions_file.name} "
            f"({len(predictions_df)} rows)"
        )
    elif not disease_filter and predictions_rows:
        score_cols = sorted(k for k in predictions_rows[0] if k.startswith("score_"))
        fixed_cols = [
            "participant_label", "specimen_label", "true_disease", "predicted_disease",
            "malid_cross_validation_fold_id_when_in_test_set",
        ]
        predictions_df = pd.DataFrame(predictions_rows, columns=fixed_cols + score_cols)
        predictions_file = output_dir / f"{model_name}_multiclass_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(
            f"  Multiclass predictions saved: {predictions_file.name} "
            f"({len(predictions_df)} rows)"
        )

    # ------------------------------------------------------------------
    # Aggregate across folds
    # ------------------------------------------------------------------
    aggregated = aggregate_fold_results(
        all_eval_results, raw_preds_list, disease_filter=disease_filter
    )
    aggregated_by_model = {model_name: aggregated}

    return all_eval_results, aggregated_by_model


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train_all_folds(
    fold_ids: Optional[List[int]],
    metadata_path: Path,
    output_dir: Optional[Path] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    classification_mode: str = "multiclass",
    reference_class: Optional[str] = None,
    diseases: Optional[List[str]] = None,
    model_name: str = "lasso_cv",
    gene_locus: str = "TCR",
    l1_ratio: Optional[float] = None,
    n_pcs: int = 15,
    verbose: int = 1,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    gene_reference_path: Optional[Path] = None,
    output_suffix: Optional[str] = None,
    training_context: str = "cv_single_model",
) -> Dict[str, Dict]:
    """Train Model 1 on all specified folds.

    Parameters
    ----------
    fold_ids            : List of fold IDs to train, or None for all folds in metadata.
    metadata_path       : Path to the metadata TSV file.
    output_dir          : Base output directory. If None, defaults to the canonical path.
    dataset_name        : Dataset identifier used in the output path.
    classification_mode : "multiclass" | "binary" | "multi-binary".
    reference_class     : Reference/negative class for binary and multi-binary modes.
    diseases            : Explicit subset of disease classes to train.
    model_name          : Model variant label (e.g., "lasso_cv").
    gene_locus          : "TCR" or "BCR".
    l1_ratio            : Elastic net L1/L2 ratio. None = use model default for gene_locus.
    n_pcs               : Number of PCA components.
    verbose             : Verbosity level.
    data_dir            : Path to raw data directory. Required if cache is missing.
    cache_dir           : Path to cache directory. None disables caching.
    gene_reference_path : Path to gene reference file (V-gene CDR sequences).
    output_suffix       : Suffix appended to the mode directory name (e.g. "no_pca"
                          produces "multiclass__no_pca"). Ignored when output_dir is set.
    training_context    : Training context controlling data splits and output paths.
                          "cv_single_model" (default) or "cv_ensemble".

    Returns
    -------
    Dict mapping pair/mode key → {"fold_results": List[Dict], "aggregated_by_model": Dict[str, Dict]}.
    - multiclass:   {"multiclass": {...}}
    - binary:       {"<disease>_vs_<reference>": {...}}
    - multi-binary: {"<d1>_vs_<ref>": {...}, "<d2>_vs_<ref>": {...}, ...}
    """
    # Build model params
    model_params: Dict = {"gene_locus": gene_locus, "n_pcs": n_pcs}
    if l1_ratio is not None:
        model_params["l1_ratio"] = l1_ratio

    # Initialize data loader
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir or Path("."),  # placeholder if cache covers all reads
        metadata_path=metadata_path,
        gene_reference_path=gene_reference_path,
        gene_locus=gene_locus,
        cache_dir=cache_dir,
        verbose=0,
    )

    # Resolve fold IDs: None = all folds found in metadata
    if fold_ids is None:
        fold_ids = sorted(
            loader.metadata["malid_cross_validation_fold_id_when_in_test_set"]
            .dropna().unique().astype(int).tolist()
        )
        logger.info(f"  Auto-detected fold IDs from metadata: {fold_ids}")

    # Validate mode against available disease classes
    disease_classes = get_dataset_disease_classes(metadata_path)
    reference_class = validate_mode_and_classes(
        classification_mode, disease_classes, reference_class, diseases=diseases
    )

    # Base output dir (parent of pair subdirs for binary/multi-binary)
    base_dir = output_dir or get_model_output_dir(
        "model1", dataset_name, classification_mode, gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )

    loop_kwargs = dict(
        loader=loader,
        fold_ids=fold_ids,
        model_name=model_name,
        model_params=model_params,
        verbose=verbose,
        training_context=training_context,
    )

    return run_training_orchestration(
        base_dir=base_dir,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_fold_loop,
        loop_kwargs=loop_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Model 1 (Repertoire Classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --- Data and cache paths ---
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to the metadata TSV file (e.g., data/metadata.tsv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Path to raw data directory (AIRR-format files). "
            "Required if the cache does not exist yet. "
            "Not needed when a complete cache is available."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for preprocessed data. "
            "Default: cache/<dataset-name>/ under the project root. "
            "Ignored when --dont-use-cache is set."
        ),
    )
    parser.add_argument(
        "--gene-reference-path",
        type=Path,
        default=None,
        # TODO: Audit whether CDR1/CDR2/FR columns from this file are actually used
        # by any model. If not, this argument can be removed (see TODO_for_release.md #4).
        help="Path to V-gene CDR reference file (e.g., tcrb_v_gene_cdrs.generated.tsv).",
    )
    parser.add_argument(
        "--dont-use-cache",
        action="store_true",
        help=(
            "Disable caching entirely. All data will be loaded and preprocessed "
            "from raw files on every run. Requires --data-dir."
        ),
    )

    # --- Dataset and mode ---
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=(
            f"Dataset identifier used in the output path: "
            f"trained_models/<dataset_name>/model1/... (default: {DEFAULT_DATASET_NAME})"
        ),
    )
    parser.add_argument(
        "--training-context",
        default="cv_single_model",
        choices=list(VALID_TRAINING_CONTEXTS),
        help=(
            "Training context controlling data splits and output directory structure. "
            "'cv_single_model' (default): each model independently CV-evaluated; "
            "trains on ts1+ts2 (all non-test participants). "
            "'cv_ensemble': base model training for the ensemble; "
            "trains on ts1+ts2 (excludes validation participants)."
        ),
    )
    parser.add_argument(
        "--classification-mode",
        default="multiclass",
        choices=["multiclass", "binary", "multi-binary"],
        help=(
            "Classification mode (default: multiclass). "
            "'multiclass': single N-class model. "
            "'binary': one model for exactly 2 classes. "
            "'multi-binary': one independent model per disease vs. reference class."
        ),
    )
    parser.add_argument(
        "--reference-class",
        default=None,
        help=(
            "Reference/negative class for binary and multi-binary modes "
            "(e.g. 'Healthy/Background'). "
            "For multi-binary: required when data has >2 classes. "
            "For binary (2-class data): optional; if omitted, alphabetical order is used. "
            "Ignored for multiclass."
        ),
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=None,
        metavar="DISEASE",
        help=(
            "Explicit subset of disease classes to train (binary and multi-binary only). "
            "binary: provide exactly one disease name. "
            "multi-binary: provide one or more disease names."
        ),
    )
    parser.add_argument(
        "--fold-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Fold IDs to train (default: all folds found in metadata). "
            "Example: --fold-ids 0 1 2"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lasso_cv",
        help=(
            "Label for this model variant, used in output filenames. "
            "Conventional values: lasso_cv (pure L1, TCR default), "
            "ridge_cv (pure L2), elasticnet_cv (mixed L1+L2, set ratio with --l1-ratio). "
            "Default: lasso_cv"
        ),
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help=(
            "Elastic net L1/L2 ratio. 1.0=lasso (pure L1), 0.0=ridge (pure L2). "
            "Default: None = 1.0 for TCR, 0.25 for BCR"
        ),
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=15,
        help="Number of PCA components (default: 15)",
    )
    parser.add_argument(
        "--gene-locus",
        default="TCR",
        choices=["TCR"],
        help="Gene locus (default: TCR). Only TCR is supported at the moment.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Suffix appended to the classification mode directory name. "
            "E.g. --output-suffix no_pca produces "
            "'multiclass__no_pca' instead of 'multiclass'. "
            "Useful for running multiple experiments with different "
            "parameters without overwriting each other. "
            "Mutually exclusive with --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. If not provided, defaults to "
            "trained_models/<dataset_name>/model1/<mode>/<gene_locus>/ under the project root. "
            "For binary/multi-binary, each pair saves to a subdirectory of this base. "
            "Mutually exclusive with --output-suffix."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=basic, 2=detailed (default: 1)",
    )

    args = parser.parse_args()

    # --- Validate --output-dir / --output-suffix mutual exclusion ---
    if args.output_dir is not None and args.output_suffix is not None:
        parser.error(
            "--output-dir and --output-suffix are mutually exclusive. "
            "Use --output-dir for a fully custom path, or --output-suffix "
            "to append to the canonical directory name."
        )

    # Sanitize --output-suffix: only allow alphanumeric, underscore, hyphen, dot.
    if args.output_suffix is not None:
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "_", args.output_suffix)
        if sanitized != args.output_suffix:
            logger.warning(
                f"--output-suffix sanitized: '{args.output_suffix}' -> '{sanitized}' "
                f"(only alphanumeric, underscore, hyphen, and dot are allowed)"
            )
            args.output_suffix = sanitized
        if not sanitized:
            parser.error("--output-suffix must not be empty after sanitization.")

    # --- Resolve cache and data paths ---
    if args.dont_use_cache:
        cache_dir = None
        if args.data_dir is None:
            parser.error("--data-dir is required when --dont-use-cache is set.")
    else:
        cache_dir = args.cache_dir or (PROJECT_ROOT / "cache" / args.dataset_name)
        # Check if cache already exists (participant cache has *_clean.parquet files)
        participants_cache = cache_dir / "participants"
        cache_exists = (
            participants_cache.exists()
            and any(participants_cache.glob("*_clean.parquet"))
        )
        if not cache_exists and args.data_dir is None:
            parser.error(
                f"No existing cache found at {cache_dir}. "
                "Provide --data-dir so the cache can be built, or use --dont-use-cache "
                "to run without caching."
            )

    if args.data_dir is not None and not args.data_dir.exists():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if not args.metadata_path.exists():
        parser.error(f"--metadata-path does not exist: {args.metadata_path}")
    if args.gene_reference_path is not None and not args.gene_reference_path.exists():
        parser.error(f"--gene-reference-path does not exist: {args.gene_reference_path}")

    # Resolve base output dir before logging so the log file can be written from the start
    base_dir = args.output_dir or get_model_output_dir(
        "model1", args.dataset_name, args.classification_mode, args.gene_locus,
        training_context=args.training_context,
        output_suffix=args.output_suffix,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Mirror all logging to a file in the output directory
    log_path = base_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Resolve fold IDs early so logging and summary JSON show the actual values
    fold_ids = args.fold_ids
    if fold_ids is None:
        fold_ids = get_dataset_fold_ids(args.metadata_path)
        logger.info(f"Auto-detected fold IDs from metadata: {fold_ids}")

    logger.info(f"Starting Model 1 training — {timestamp}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Training context:    {args.training_context}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Reference class:     {args.reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {args.diseases or '(all)'}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Model name:          {args.model_name}")
    _eff_l1_ratio = args.l1_ratio if args.l1_ratio is not None else (
        RepertoireClassifier.DEFAULT_L1_RATIOS.get(args.gene_locus, 1.0)
    )
    logger.info(f"  L1 ratio:            {_eff_l1_ratio}")
    logger.info(f"  n_pcs:               {args.n_pcs}")
    logger.info(f"  Base output dir:     {base_dir}")
    if args.output_suffix:
        logger.info(f"  Output suffix:       {args.output_suffix}")
    logger.info(f"  Data dir:            {args.data_dir or '(not provided, using cache)'}")
    logger.info(f"  Cache dir:           {cache_dir or '(caching disabled)'}")
    logger.info(f"  Metadata:            {args.metadata_path}")
    logger.info(f"  Gene reference:      {args.gene_reference_path or '(not provided)'}")

    all_results = train_all_folds(
        fold_ids=fold_ids,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        classification_mode=args.classification_mode,
        reference_class=args.reference_class,
        diseases=args.diseases,
        model_name=args.model_name,
        gene_locus=args.gene_locus,
        l1_ratio=args.l1_ratio,
        n_pcs=args.n_pcs,
        verbose=args.verbose,
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        gene_reference_path=args.gene_reference_path,
        output_suffix=args.output_suffix,
        training_context=args.training_context,
    )

    # ------------------------------------------------------------------
    # Save summary JSON (at base_dir level, covering all pairs/modes)
    # ------------------------------------------------------------------
    summary_path = base_dir / f"summary_{timestamp}.json"
    all_eval_flat = [r for pair_data in all_results.values() for r in pair_data["fold_results"]]

    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset_name": args.dataset_name,
                "training_context": args.training_context,
                "classification_mode": args.classification_mode,
                "reference_class": args.reference_class,
                "diseases": args.diseases,
                "gene_locus": args.gene_locus,
                "output_suffix": args.output_suffix,
                "fold_ids": fold_ids,
                "model_names": [args.model_name],
                "l1_ratio": _eff_l1_ratio,
                "n_pcs": args.n_pcs,
                "results_by_pair": {
                    key: val["fold_results"] for key, val in all_results.items()
                },
                "aggregated_by_pair": {
                    key: val["aggregated_by_model"] for key, val in all_results.items()
                },
            },
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )

    logger.info(f"\nSummary saved to {summary_path}")

    # ------------------------------------------------------------------
    # Save results Markdown
    # ------------------------------------------------------------------
    run_info: Dict = {
        "Dataset": args.dataset_name,
        "Training context": args.training_context,
        "Classification mode": args.classification_mode,
        "Gene locus": args.gene_locus,
        "Folds": ", ".join(str(f) for f in fold_ids),
        "Model variant": args.model_name,
        "L1 ratio (alpha)": _eff_l1_ratio,
        "N PCs": args.n_pcs,
        "Output suffix": args.output_suffix or "(none)",
    }
    if args.classification_mode != "multiclass" and args.reference_class:
        run_info["Reference class"] = args.reference_class
    if args.diseases:
        run_info["Diseases"] = ", ".join(args.diseases)

    md_content = generate_results_md(
        all_results=all_results,
        classification_mode=args.classification_mode,
        timestamp=timestamp,
        model_label="Model 1",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=[args.model_name],
        has_abstention=False,
    )
    md_path = base_dir / f"RESULTS_{timestamp}.md"
    md_path.write_text(md_content)
    logger.info(f"Results MD saved to {md_path}")

    # ------------------------------------------------------------------
    # Per-pair results (binary / multi-binary only)
    # ------------------------------------------------------------------
    save_per_pair_results(
        base_dir=base_dir,
        all_results=all_results,
        classification_mode=args.classification_mode,
        timestamp=timestamp,
        model_label="Model 1",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=[args.model_name],
        has_abstention=False,
    )

    # ------------------------------------------------------------------
    # Print per-fold table
    # ------------------------------------------------------------------
    logger.info("\n--- Summary ---")
    for r in all_eval_flat:
        pair_str = (
            f"{r['disease']}_vs_{r['reference_class']} "
            if "disease" in r else ""
        )
        auroc_val = r.get("auroc_binary") or r.get("auroc_ovo_weighted")
        auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A  "
        mcc_str = f"{r['mcc']:.4f}" if r.get("mcc") is not None else "N/A  "
        logger.info(
            f"  fold={r['fold_id']} {pair_str}{r['model_name']:20s} "
            f"accuracy={r['accuracy']:.4f} AUROC={auroc_str} MCC={mcc_str}"
        )

    # ------------------------------------------------------------------
    # Print aggregated summary
    # ------------------------------------------------------------------
    logger.info("\n--- Aggregated Results ---")
    for pair_key, pair_data in all_results.items():
        for mn, agg in pair_data["aggregated_by_model"].items():
            logger.info(f"  {pair_key} / {mn}:")
            acc_global = agg.get("accuracy_global")
            acc_str = f"{acc_global:.4f}" if acc_global is not None else "N/A"
            mcc_agg = agg.get("mcc", {})
            mcc_mean = mcc_agg.get("mean") if isinstance(mcc_agg, dict) else None
            mcc_str2 = f"{mcc_mean:.4f}" if mcc_mean is not None else "N/A"
            if args.classification_mode == "multiclass":
                auroc_agg = agg.get("auroc_ovo_weighted", {})
                auroc_mean = auroc_agg.get("mean")
                auroc_str_ovo = f"{auroc_mean:.4f}" if auroc_mean is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_OvO={auroc_str_ovo} MCC={mcc_str2}"
                )
                ll_agg = agg.get("log_loss", {})
                ll_mean = ll_agg.get("mean")
                if ll_mean is not None:
                    logger.info(f"    LogLoss={ll_mean:.4f} (std={ll_agg.get('std', 0):.4f})")
            else:
                auroc_p = agg.get("auroc_pooled")
                auprc_p = agg.get("auprc_pooled")
                auroc_str2 = f"{auroc_p:.4f}" if auroc_p is not None else "N/A"
                auprc_str2 = f"{auprc_p:.4f}" if auprc_p is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_pooled={auroc_str2} "
                    f"AUPRC_pooled={auprc_str2} MCC={mcc_str2}"
                )

    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Clean up file handler to flush and release the log file
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
