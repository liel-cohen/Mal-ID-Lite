#!/usr/bin/env python
"""Train and evaluate Model 2 (Convergent Cluster Classifier) for disease classification.

Trains Model 2 on all folds using the two-level cache for fast data loading.
Saves cluster centroids, best p-value thresholds, and fitted classifiers to disk.
Evaluates on the test fold (AUROC, AUPRC, accuracy, abstention rate).

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
    With --diseases <d1> <d2> ...: trains only the specified subset of diseases.
    Requires --reference-class when the data has more than 2 classes.
    If the data has exactly 2 classes (and no --diseases), behaves identically to binary.
    Each binary pair is fully independent: separate clustering, Fisher test, and GLM.

Output directory structure
---------------------------
cv_single_model (default):
  multiclass:   trained_models/<dataset>/cv_single_model/model2/multiclass/<locus>/
  binary:       trained_models/<dataset>/cv_single_model/model2/binary/<locus>/<pair>/
  multi-binary: trained_models/<dataset>/cv_single_model/model2/binary/<locus>/<pair1>/
                                                                                <pair2>/...

cv_ensemble:
  multiclass:   trained_models/<dataset>/cv_ensemble/multiclass/<locus>/base_models/model2/
  binary:       trained_models/<dataset>/cv_ensemble/binary/<locus>/base_models/model2/<pair>/

With --output-suffix <suffix>, the mode directory gets "__<suffix>" appended:
    trained_models/<dataset>/cv_single_model/model2/multiclass__<suffix>/<locus>/

Both binary and multi-binary write to the same binary/<gene_locus>/ subtree, so artifacts
for the same pair are identical regardless of which mode produced them.

A predictions CSV is written per model_name alongside other artifacts:
    multiclass:   <model_name>_multiclass_predictions.csv
    binary:       <disease>_vs_<reference>/<model_name>_binary_predictions.csv

Multiclass columns: participant_label, specimen_label, true_disease, predicted_disease,
    abstained (True/False), score_<class1>, score_<class2>, ...,
    malid_cross_validation_fold_id_when_in_test_set
    Abstained specimens are included with None for predicted_disease and score_* columns.

Binary columns: participant_label, specimen_label, disease_label (0/1), disease_label_str,
    disease_model, model_score (P(disease)), malid_cross_validation_fold_id_when_in_test_set
    Only scored (non-abstained) specimens are included.

Usage examples
--------------
    # Multiclass (default)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv

    # Binary (2-class data)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode binary --reference-class Healthy

    # Multi-binary (N-class data, one model per disease vs Healthy)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy

    # Binary for a single disease from N-class data
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode binary --reference-class Healthy --diseases COVID-19

    # Multi-binary for a specific subset of diseases
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy \\
        --diseases COVID-19 Lupus

    # Train only fold 0, all 5 alpha variants, 8 parallel workers
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --fold-ids 0 --model-names lasso_cv elasticnet_cv0.75 elasticnet_cv elasticnet_cv0.25 ridge_cv \\
        --n-jobs 8

    # Retrain final GLM on train_smaller1+train_smaller2 combined (opt-in improvement)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv --retrain-full

Performance note
----------------
    --n-jobs controls parallelism for Phase 1 (clustering), which is the dominant cost.
    Each (V gene, J gene, CDR3 length) supergroup is processed independently.
    Default is 4 workers — safe for most workstations. On machines with 16+ cores and
    >=32 GB RAM, try --n-jobs 8 or higher for faster training. Memory scales with n_jobs
    because each worker holds its own pairwise distance matrix.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# Add project root to path (malid/training/ → malid/ → project root)
# Must come before any malid_lite imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Custom multiclass metrics that handle unnormalized probabilities and missing
# labels gracefully. Matches the original Mal-ID paper's evaluation methodology.
from malid_lite.utils import multiclass_metrics

from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.models.model2_convergent_clusters import (
    BEST_MODEL_FOR_METAMODEL,
    DEFAULT_P_VALUES,
    FeaturizedData,
    featurize,
    get_artifact_paths,
    train_convergent_cluster_classifier,
)
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
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare_fold(
    loader: MalIDPublishedDataLoader,
    fold_id: int,
    fold_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fold sequences and join with disease metadata.

    Returns
    -------
    (sequences_df, metadata_df)
    sequences_df has all sequence columns plus disease (from metadata join).
    specimen_label column is the specimen identifier.
    """
    sequences_df, metadata_df = loader.get_fold_data(fold_id, fold_label)

    if sequences_df.empty:
        raise ValueError(f"No sequences found for fold {fold_id} {fold_label}")

    # Join disease from metadata (keyed by specimen_label)
    disease_map = metadata_df.set_index("specimen_label")["disease"]
    sequences_df = sequences_df.copy()
    sequences_df[DISEASE_COL] = sequences_df["specimen_label"].map(disease_map)

    # Drop rows where disease is unknown (shouldn't happen, but be safe)
    n_before = len(sequences_df)
    sequences_df = sequences_df.dropna(subset=[DISEASE_COL])
    if len(sequences_df) < n_before:
        logger.warning(
            f"  Dropped {n_before - len(sequences_df)} rows with unknown disease"
        )

    return sequences_df, metadata_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    featurized: FeaturizedData,
    pipeline,
    classes: np.ndarray,
    fold_id: int,
    model_name: str,
    reference_class: Optional[str] = None,
) -> Tuple[Dict, Optional[Dict]]:
    """Compute evaluation metrics on the test fold.

    Parameters
    ----------
    reference_class : Reference/negative class. When provided and data has exactly
        2 classes, also computes auroc_binary and auprc_binary with the non-reference
        class as positive — matching model 1 binary methodology exactly.

    Returns
    -------
    (metrics, raw_preds)
        metrics   : JSON-serializable dict of evaluation metrics.
        raw_preds : {"y_true", "y_pred", "y_proba", "classes"} numpy arrays for
                    cross-fold aggregation (pooled metrics). None if all abstained.
    """
    results = {
        "fold_id": fold_id,
        "model_name": model_name,
        "n_scored": featurized.n_scored,
        "n_abstained": featurized.n_abstained,
        "abstention_rate": featurized.abstention_rate,
        "p_value_threshold": featurized.p_value_threshold,
    }

    if featurized.n_scored == 0:
        logger.warning(f"  fold {fold_id} {model_name}: all specimens abstained on test fold")
        # Accuracy = 0: all specimens are abstentions, all count as wrong.
        results["accuracy"] = 0.0
        return results, None

    y_true = featurized.y
    y_pred = pipeline.predict(featurized.X)
    y_proba = pipeline.predict_proba(featurized.X)

    # Accuracy: abstentions count as wrong (matching original Mal-ID / crosseval behavior).
    # Equivalent to appending y_pred="Unknown" for each abstained specimen, then computing
    # accuracy over scored+abstained. Since "Unknown" never matches any true class:
    #   accuracy = n_correct_scored / (n_scored + n_abstained)
    # AUROC and AUPRC are computed on scored specimens only (no probabilities for abstentions).
    n_correct = int(accuracy_score(y_true, y_pred, normalize=False))
    n_total = featurized.n_scored + featurized.n_abstained
    results["accuracy"] = n_correct / n_total

    if featurized.n_abstained > 0:
        logger.warning(
            f"  fold {fold_id} {model_name}: {featurized.n_abstained}/{n_total} "
            f"({featurized.abstention_rate:.1%}) specimens abstained. "
            f"AUROC and AUPRC are computed on the {featurized.n_scored} scored "
            f"specimens only and do not reflect the missing predictions."
        )
        results["auroc_auprc_note"] = (
            f"Computed on {featurized.n_scored}/{n_total} scored specimens only. "
            f"{featurized.n_abstained} ({featurized.abstention_rate:.1%}) abstained "
            f"specimens are excluded from AUROC/AUPRC."
        )

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

    # Log loss
    try:
        results["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    except ValueError as e:
        logger.warning(f"  Log loss failed: {e}")
        results["log_loss"] = None

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=classes
    ).tolist()
    results["classes"] = [str(c) for c in classes]

    # Binary AUROC/AUPRC with disease as positive (matches model 1 binary exactly).
    # Only meaningful when exactly 2 classes and reference_class is known.
    if len(classes) == 2 and reference_class is not None:
        str_classes = [str(c) for c in classes]
        if str(reference_class) in str_classes:
            disease_class = next(c for c in str_classes if c != str(reference_class))
            disease_idx = str_classes.index(disease_class)
            y_score = y_proba[:, disease_idx]
            y_binary = (y_true.values == disease_class).astype(int)
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
        "y_true": y_true.values,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
    }
    return results, raw_preds


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_fold_artifacts(
    output_dir: Path,
    fold_id: int,
    train_result: Dict,
    retrain_on_full_train: bool = False,
    disease_filter: Optional[Tuple[str, str]] = None,
) -> None:
    """Save all artifacts for one fold to disk.

    Artifact filenames are determined by get_artifact_paths() (single source of truth).

    Saves:
    - fold_{id}_clusters.joblib               : centroids + Fisher scores + disease_classes (shared)
    - fold_{id}_{model}_p_value.joblib         : best p-value (float)
    - fold_{id}_{model}_model_{suffix}.joblib  : fitted sklearn Pipeline
    - fold_{id}_{model}_results_{suffix}.json  : per-p-value metrics
    where suffix is "split1" (retrain_on_full_train=False, default) or "full" (True).

    Parameters
    ----------
    disease_filter : (disease, reference_class) for binary/multi-binary models, None for
        multiclass. Saved into the clusters artifact so that ConvergentClusterClassifier
        can recover the positive/negative class assignment at inference time and always
        return predict_proba columns in [P(reference), P(disease)] order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared cluster artifact (same regardless of model_name or retrain_on_full_train)
    clusters_path = output_dir / f"fold_{fold_id}_clusters.joblib"
    binary_pair = (
        {"disease": disease_filter[0], "reference_class": disease_filter[1]}
        if disease_filter is not None
        else None
    )
    joblib.dump(
        {
            "centroids_with_scores": train_result["centroids_with_scores"],
            "disease_classes": train_result["disease_classes"],
            "binary_pair": binary_pair,
        },
        clusters_path,
    )
    logger.info(f"  Saved clusters: {clusters_path}")

    # Per-model artifacts
    for model_name, model_result in train_result["results"].items():
        if model_result["best_p_value"] is None:
            logger.warning(f"  Skipping save for {model_name} (no valid run)")
            continue

        paths = get_artifact_paths(output_dir, fold_id, model_name, retrain_on_full_train)

        joblib.dump(model_result["best_p_value"], paths["p_value"])
        joblib.dump(model_result["pipeline"], paths["pipeline"])

        with open(paths["metrics"], "w") as f:
            json.dump(
                {
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "retrain_on_full_train": retrain_on_full_train,
                    "best_p_value": model_result["best_p_value"],
                    "all_p_value_metrics": model_result["all_p_value_metrics"],
                },
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )

        logger.info(f"  Saved {model_name}: p_value={model_result['best_p_value']}")


# ---------------------------------------------------------------------------
# Cross-fold aggregation — imported from training_utils
# ---------------------------------------------------------------------------
# aggregate_fold_results is imported above.


# ---------------------------------------------------------------------------
# Fold loop (shared by all classification modes)
# ---------------------------------------------------------------------------

def _run_fold_loop(
    loader: MalIDPublishedDataLoader,
    fold_ids: List[int],
    output_dir: Path,
    model_names: List[str],
    sequence_identity_threshold: float,
    p_values: List[float],
    retrain_on_full_train: bool,
    n_jobs: int,
    verbose: int,
    disease_filter: Optional[Tuple[str, str]] = None,
    training_context: str = "cv_single_model",
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Run training + evaluation for all specified folds.

    Parameters
    ----------
    loader              : Data loader with fold cache and split persistence.
    fold_ids            : Which folds to train/evaluate.
    output_dir          : Directory for artifacts (models, predictions, results).
    model_names         : GLM regularization variants to train (e.g. "lasso_cv").
    sequence_identity_threshold : CDR3 clustering identity threshold (0-1).
    p_values            : Fisher p-value thresholds to evaluate for cluster selection.
    retrain_on_full_train : If True, retrain final GLM on ts1+ts2 combined after
        p-value selection on ts2 alone.
    n_jobs              : Parallel workers for clustering phase.
    verbose             : Logging verbosity (0=quiet, 1=normal, 2=debug).
    disease_filter      : Optional (disease, reference_class) tuple. If provided,
        sequences and metadata are filtered to participants in
        {disease, reference_class} before training. Used for binary and
        multi-binary modes. If None, all participants are used (multiclass).
    training_context    : Controls which participants are used for training via
        centralized split persistence. "cv_single_model" uses all non-test
        participants; "cv_ensemble" excludes validation participants.

    Returns
    -------
    (all_eval_results, aggregated_by_model)
        all_eval_results : List of per-fold metric dicts, one per (fold, model_name).
        aggregated_by_model : Dict mapping model_name → aggregated metrics dict.
    """
    all_eval_results = []
    # raw_preds collected per model_name for cross-fold aggregation
    raw_preds_by_model: Dict[str, List[Optional[Dict]]] = {mn: [] for mn in model_names}
    # per-specimen rows for binary predictions CSV (only populated when disease_filter is set)
    predictions_rows_by_model: Dict[str, List[Dict]] = {mn: [] for mn in model_names}

    for fold_id in fold_ids:
        pair_tag = (
            f" [{make_pair_name(disease_filter[0], disease_filter[1])}]"
            if disease_filter else ""
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_id}{pair_tag}")
        logger.info(f"{'='*60}")

        # ------------------------------------------------------------------
        # Load + filter training data
        # ------------------------------------------------------------------
        logger.info("Loading training data...")
        train_sequences_df, train_metadata_df = load_and_prepare_fold(
            loader, fold_id, "train"
        )
        if disease_filter:
            disease, reference_class = disease_filter
            train_sequences_df, train_metadata_df = filter_to_binary_pair(
                train_sequences_df, train_metadata_df, disease, reference_class
            )

        # Split into train_smaller1 and train_smaller2 using centralized splits.
        # cv_single_model: ts1+ts2 = all train participants
        # cv_ensemble: ts1+ts2 = train participants minus validation
        ts1_participants = set(loader.get_split_participants(
            fold_id, training_context, ["train_smaller1"]
        ))
        ts2_participants = set(loader.get_split_participants(
            fold_id, training_context, ["train_smaller2"]
        ))

        train_smaller1_df = train_sequences_df[
            train_sequences_df[PARTICIPANT_COL].isin(ts1_participants)
        ].copy()
        train_smaller2_df = train_sequences_df[
            train_sequences_df[PARTICIPANT_COL].isin(ts2_participants)
        ].copy()

        # Assertions: split filtering must produce non-empty data with expected
        # participant counts. Empty splits indicate a bug in split generation or
        # a mismatch between fold data and split files.
        # In binary mode, disease_filter was applied above, so the data only has
        # 2 diseases — participant counts will be a subset of the full split.
        # In multiclass mode, counts should match exactly.
        ts1_actual = train_smaller1_df[PARTICIPANT_COL].nunique()
        ts2_actual = train_smaller2_df[PARTICIPANT_COL].nunique()
        assert len(train_smaller1_df) > 0, (
            f"train_smaller1 is empty after split filtering (fold {fold_id}, "
            f"context={training_context}). Expected {len(ts1_participants)} participants."
        )
        assert len(train_smaller2_df) > 0, (
            f"train_smaller2 is empty after split filtering (fold {fold_id}, "
            f"context={training_context}). Expected {len(ts2_participants)} participants."
        )
        if not disease_filter:
            # Multiclass: all split participants should be present in the data
            assert ts1_actual == len(ts1_participants), (
                f"train_smaller1 participant count mismatch: got {ts1_actual}, "
                f"expected {len(ts1_participants)} (fold {fold_id}, context={training_context})"
            )
            assert ts2_actual == len(ts2_participants), (
                f"train_smaller2 participant count mismatch: got {ts2_actual}, "
                f"expected {len(ts2_participants)} (fold {fold_id}, context={training_context})"
            )
        else:
            # Binary: data was filtered to 2 diseases, so only a subset of
            # split participants will be present. Just verify subset relationship.
            assert ts1_actual <= len(ts1_participants), (
                f"train_smaller1 has MORE participants ({ts1_actual}) than split "
                f"({len(ts1_participants)}) — impossible (fold {fold_id})"
            )
            assert ts2_actual <= len(ts2_participants), (
                f"train_smaller2 has MORE participants ({ts2_actual}) than split "
                f"({len(ts2_participants)}) — impossible (fold {fold_id})"
            )

        # Log counts AFTER split filtering so totals reflect actual training data,
        # not the full fold (which includes validation participants in cv_ensemble).
        logger.info(
            f"  Train fold (used): {len(train_smaller1_df) + len(train_smaller2_df):,} sequences, "
            f"{ts1_actual + ts2_actual} participants"
        )
        logger.info(
            f"  train_smaller1: {len(train_smaller1_df):,} sequences, "
            f"{ts1_actual} participants"
        )
        logger.info(
            f"  train_smaller2: {len(train_smaller2_df):,} sequences, "
            f"{ts2_actual} participants"
        )

        # ------------------------------------------------------------------
        # Load + filter test data
        # ------------------------------------------------------------------
        logger.info("Loading test data...")
        test_sequences_df, test_metadata_df = load_and_prepare_fold(loader, fold_id, "test")
        if disease_filter:
            test_sequences_df, test_metadata_df = filter_to_binary_pair(
                test_sequences_df, test_metadata_df, disease, reference_class
            )
        logger.info(
            f"  Test fold: {len(test_sequences_df):,} sequences, "
            f"{test_sequences_df[PARTICIPANT_COL].nunique()} participants"
        )

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        logger.info("Training...")
        train_result = train_convergent_cluster_classifier(
            train_smaller1_df=train_smaller1_df,
            train_smaller2_df=train_smaller2_df,
            sequence_identity_threshold=sequence_identity_threshold,
            model_names=model_names,
            p_values=p_values,
            disease_col=DISEASE_COL,
            retrain_on_full_train=retrain_on_full_train,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # ------------------------------------------------------------------
        # Save artifacts
        # ------------------------------------------------------------------
        logger.info("Saving artifacts...")
        save_fold_artifacts(output_dir, fold_id, train_result, retrain_on_full_train, disease_filter=disease_filter)

        # ------------------------------------------------------------------
        # Evaluate on test fold
        # ------------------------------------------------------------------
        logger.info("Evaluating on test fold...")
        disease_classes = train_result["disease_classes"]

        for model_name, model_result in train_result["results"].items():
            if model_result["best_p_value"] is None:
                logger.warning(f"  Skipping {model_name} evaluation (no valid model)")
                continue

            fd_test = featurize(
                test_sequences_df,
                p_value_threshold=model_result["best_p_value"],
                centroids_with_scores=train_result["centroids_with_scores"],
                sequence_identity_threshold=sequence_identity_threshold,
                disease_classes=disease_classes,
                disease_col=DISEASE_COL,
            )

            eval_result, raw_preds = evaluate_on_test(
                featurized=fd_test,
                pipeline=model_result["pipeline"],
                classes=np.array(disease_classes),
                fold_id=fold_id,
                model_name=model_name,
                reference_class=disease_filter[1] if disease_filter else None,
            )

            # Tag binary pair results for identification in summary
            if disease_filter:
                eval_result["disease"] = disease_filter[0]
                eval_result["reference_class"] = disease_filter[1]

            all_eval_results.append(eval_result)
            raw_preds_by_model[model_name].append(raw_preds)

            # Collect per-specimen rows for predictions CSV
            if disease_filter and raw_preds is not None:
                # Binary: one score column (P(disease)), scored specimens only
                str_classes = [str(c) for c in disease_classes]
                disease_class = next(c for c in str_classes if c != str(disease_filter[1]))
                disease_idx = str_classes.index(disease_class)
                for specimen, participant, true_disease, score in zip(
                    fd_test.y.index,
                    fd_test.participant_labels,
                    fd_test.y,
                    raw_preds["y_proba"][:, disease_idx],
                ):
                    predictions_rows_by_model[model_name].append({
                        "participant_label": participant,
                        "specimen_label": specimen,
                        "disease_label": int(true_disease == disease_class),
                        "disease_label_str": str(true_disease),
                        "disease_model": disease_class,
                        "model_score": float(score),
                        "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                    })
            elif not disease_filter:
                # Multiclass: one score column per class; abstained specimens included with NaN
                str_classes = [str(c) for c in disease_classes]
                spec_to_part = test_metadata_df.set_index(SPECIMEN_COL)[PARTICIPANT_COL]
                if raw_preds is not None:
                    for specimen, participant, true_d, pred_d, proba_row in zip(
                        fd_test.y.index,
                        fd_test.participant_labels,
                        fd_test.y,
                        raw_preds["y_pred"],
                        raw_preds["y_proba"],
                    ):
                        row = {
                            "participant_label": participant,
                            "specimen_label": specimen,
                            "true_disease": str(true_d),
                            "predicted_disease": str(pred_d),
                            "abstained": False,
                            "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                        }
                        for cls, score in zip(str_classes, proba_row):
                            row[f"score_{cls}"] = float(score)
                        predictions_rows_by_model[model_name].append(row)
                for specimen, true_d in fd_test.abstained_sample_y.items():
                    row = {
                        "participant_label": spec_to_part.get(specimen),
                        "specimen_label": specimen,
                        "true_disease": str(true_d),
                        "predicted_disease": None,
                        "abstained": True,
                        "malid_cross_validation_fold_id_when_in_test_set": fold_id,
                    }
                    for cls in str_classes:
                        row[f"score_{cls}"] = None
                    predictions_rows_by_model[model_name].append(row)

            auroc_str = (
                f"{eval_result['auroc_ovo_weighted']:.4f}"
                if eval_result.get("auroc_ovo_weighted") is not None
                else "N/A"
            )
            logloss_str = (
                f"{eval_result['log_loss']:.4f}"
                if eval_result.get("log_loss") is not None
                else "N/A"
            )
            logger.info(
                f"  {model_name}: AUROC={auroc_str} "
                f"LogLoss={logloss_str} "
                f"abstention={eval_result['abstention_rate']:.1%} "
                f"({eval_result['n_scored']}/{eval_result['n_scored'] + eval_result['n_abstained']} scored)"
            )

    # Aggregate per model_name across folds
    aggregated_by_model: Dict[str, Dict] = {}
    for mn in model_names:
        mn_metrics = [r for r in all_eval_results if r.get("model_name") == mn]
        mn_raw_preds = raw_preds_by_model.get(mn, [])
        assert len(mn_metrics) == len(mn_raw_preds), (
            f"BUG: fold_metrics and fold_raw_preds lists are misaligned for model '{mn}': "
            f"{len(mn_metrics)} metrics vs {len(mn_raw_preds)} raw_preds"
        )
        if mn_metrics:
            aggregated_by_model[mn] = aggregate_fold_results(
                mn_metrics, mn_raw_preds, disease_filter=disease_filter
            )

    # Save predictions CSV (one per model_name)
    if disease_filter:
        for mn in model_names:
            rows = predictions_rows_by_model[mn]
            if rows:
                predictions_df = pd.DataFrame(rows, columns=[
                    "participant_label", "specimen_label", "disease_label", "disease_label_str",
                    "disease_model", "model_score",
                    "malid_cross_validation_fold_id_when_in_test_set",
                ])
                predictions_file = output_dir / f"{mn}_binary_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(
                    f"  Binary predictions saved: {predictions_file.name} "
                    f"({len(predictions_df)} rows across {len(fold_ids)} fold(s))"
                )
    else:
        for mn in model_names:
            rows = predictions_rows_by_model[mn]
            if rows:
                score_cols = sorted(k for k in rows[0] if k.startswith("score_"))
                fixed_cols = [
                    "participant_label", "specimen_label", "true_disease", "predicted_disease",
                    "abstained", "malid_cross_validation_fold_id_when_in_test_set",
                ]
                predictions_df = pd.DataFrame(rows, columns=fixed_cols + score_cols)
                predictions_file = output_dir / f"{mn}_multiclass_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(
                    f"  Multiclass predictions saved: {predictions_file.name} "
                    f"({len(predictions_df)} rows across {len(fold_ids)} fold(s))"
                )

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
    model_names: Optional[List[str]] = None,
    gene_locus: str = "TCR",
    p_values: Optional[List[float]] = None,
    sequence_identity_threshold: Optional[float] = None,
    retrain_on_full_train: bool = False,
    n_jobs: int = 4,
    verbose: int = 1,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    gene_reference_path: Optional[Path] = None,
    output_suffix: Optional[str] = None,
    training_context: str = "cv_single_model",
) -> Dict[str, Dict]:
    """Train Model 2 on all specified folds.

    Parameters
    ----------
    fold_ids : List of fold IDs to train, or None for all folds in metadata.
    metadata_path : Path to the metadata TSV file.
    output_dir : Base output directory. If None, defaults to the canonical path
        under the project root (see get_model_output_dir). For binary/multi-binary,
        this is the parent of the per-pair subdirectories.
    dataset_name : Dataset identifier used in the output path.
    classification_mode : "multiclass" | "binary" | "multi-binary". See module
        docstring for details.
    reference_class : Reference/negative class for binary and multi-binary modes.
        Required for multi-binary when data has >2 classes. Optional for binary
        (alphabetical order used if omitted with 2-class data). Ignored for multiclass.
    diseases : Explicit subset of disease classes to train.
        binary: must be a single disease name. Allows selecting one disease from an
            N-class dataset without requiring exactly 2 classes. If omitted, the data
            must have exactly 2 classes (current default behavior).
        multi-binary: list of disease names to train. If omitted, all non-reference
            diseases are trained.
        multiclass: ignored.
    model_names : Classifier names to include. Defaults to
        [BEST_MODEL_FOR_METAMODEL[gene_locus]] (lasso_cv for TCR, ridge_cv for BCR).
    gene_locus : "TCR" or "BCR".
    p_values : P-value candidates for the threshold grid search. Defaults to
        DEFAULT_P_VALUES = [0.0005, 0.001, 0.005, 0.01, 0.05].
    sequence_identity_threshold : Override default threshold (SEQUENCE_IDENTITY_THRESHOLDS).
    retrain_on_full_train : If False (default), final GLM trained on train_smaller1 only,
        matching original Mal-ID. If True, retrain GLM on train_smaller1+2 combined
        (clusters always frozen from train_smaller1 regardless).
    n_jobs : Parallel workers for Phase 1 (clustering) only.
    verbose : Verbosity level.
    data_dir : Path to raw data directory. Required if cache is missing.
    cache_dir : Path to cache directory. None disables caching.
    gene_reference_path : Path to gene reference file (V-gene CDR sequences).
    output_suffix : Suffix appended to the mode directory name (e.g. "strict_pval"
        produces "multiclass__strict_pval"). Ignored when output_dir is set.
    training_context : Training context controlling data splits and output paths.
        "cv_single_model" (default) or "cv_ensemble".

    Returns
    -------
    Dict mapping pair/mode key → {"fold_results": List[Dict], "aggregated_by_model": Dict[str, Dict]}.
    - multiclass:   {"multiclass": {...}}
    - binary:       {"<disease>_vs_<reference>": {...}}
    - multi-binary: {"<d1>_vs_<ref>": {...}, "<d2>_vs_<ref>": {...}, ...}
    """
    from malid_lite.models.model2_convergent_clusters import SEQUENCE_IDENTITY_THRESHOLDS

    if model_names is None:
        model_names = [BEST_MODEL_FOR_METAMODEL[gene_locus]]

    if sequence_identity_threshold is None:
        sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]

    if p_values is None:
        p_values = DEFAULT_P_VALUES

    # Initialize data loader
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir or Path("."),  # placeholder if cache covers all reads
        metadata_path=metadata_path,
        gene_locus=gene_locus,
        cache_dir=cache_dir,
        gene_reference_path=gene_reference_path,
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
        "model2", dataset_name, classification_mode, gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )

    loop_kwargs = dict(
        loader=loader,
        fold_ids=fold_ids,
        model_names=model_names,
        sequence_identity_threshold=sequence_identity_threshold,
        p_values=p_values,
        retrain_on_full_train=retrain_on_full_train,
        n_jobs=n_jobs,
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
        description="Train Model 2 (Convergent Cluster Classifier)",
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
            "Name of the dataset being trained on. Used as the top-level subdirectory "
            f"in the output path: trained_models/<dataset_name>/model2/... (default: {DEFAULT_DATASET_NAME}). "
            "Change this when training on a different dataset to keep results separate."
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
            "'binary': one model for exactly 2 classes (use --reference-class to name them). "
            "'multi-binary': one independent model per disease vs. reference class "
            "(requires --reference-class when data has >2 classes)."
        ),
    )
    parser.add_argument(
        "--reference-class",
        default=None,
        help=(
            "Reference/negative class for binary and multi-binary modes "
            "(e.g. 'Healthy', 'HC', 'control'). "
            "For multi-binary: required when data has >2 classes. "
            "For binary (2-class data): optional; if omitted, alphabetical order is used. "
            "For binary with --diseases on N-class data: required when data has >2 classes. "
            "Ignored for multiclass. "
            "Output subdirectory is named <disease>_vs_<reference>."
        ),
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=None,
        metavar="DISEASE",
        help=(
            "Explicit subset of disease classes to train (binary and multi-binary only). "
            "binary: provide exactly one disease name — allows targeting a single disease "
            "from an N-class dataset without requiring exactly 2 classes in the data. "
            "multi-binary: provide one or more disease names — trains only the specified "
            "diseases vs. --reference-class instead of all non-reference diseases. "
            "All names must match disease labels in the metadata exactly. "
            "Ignored for multiclass."
        ),
    )
    parser.add_argument(
        "--fold-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Fold IDs to train (default: all folds found in metadata). "
            "Example: --fold-ids 0 1 2"
        ),
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help=(
            "Classifier names to train. Default: only the best model for the selected gene locus "
            "(lasso_cv for TCR, ridge_cv for BCR), matching BEST_MODEL_FOR_METAMODEL. "
            "Pass all 5 to replicate original Mal-ID: "
            "lasso_cv elasticnet_cv0.75 elasticnet_cv elasticnet_cv0.25 ridge_cv"
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Suffix appended to the classification mode directory name. "
            "E.g. --output-suffix strict_pval produces "
            "'multiclass__strict_pval' instead of 'multiclass'. "
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
            "trained_models/<dataset_name>/model2/<mode>/<gene_locus>/ under the project root. "
            "For binary/multi-binary, each pair saves to a subdirectory of this base. "
            "Mutually exclusive with --output-suffix."
        ),
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=DEFAULT_P_VALUES,
        help=f"P-value candidates (default: {DEFAULT_P_VALUES})",
    )
    parser.add_argument(
        "--gene-locus",
        default="TCR",
        choices=["TCR"],
        help="Gene locus (default: TCR). Only TCR is supported at the moment.",
    )
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        default=False,
        help=(
            "If set, re-train final GLM on train_smaller1 + train_smaller2 combined "
            "after p-value selection (clusters remain fixed from train_smaller1). "
            "Default: train GLM on train_smaller1 only, matching original Mal-ID behavior."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Number of parallel workers for the clustering phase (Phase 1) only. "
            "Each (v_gene, j_gene, cdr3_len) supergroup is processed independently in a "
            "separate thread. Set to 1 to disable parallelism. "
            "Higher values reduce runtime but increase peak memory usage. "
            "Recommended: 2-8 on most workstations; 8-16 on machines with >=32 GB RAM. "
            "Avoid -1 (all cores) as it can freeze the machine. "
            "(default: 4)"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress, 2=detailed).",
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

    # Resolve model_names before logging so the log shows actual values
    model_names = args.model_names or [BEST_MODEL_FOR_METAMODEL[args.gene_locus]]
    base_dir = args.output_dir or get_model_output_dir(
        "model2", args.dataset_name, args.classification_mode, args.gene_locus,
        training_context=args.training_context,
        output_suffix=args.output_suffix,
    )

    # Create output directory early so the log file can be written from the start.
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Mirror all logging to a file in the output directory.
    log_path = base_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Resolve fold IDs early so logging and summary JSON show the actual values
    fold_ids = args.fold_ids
    if fold_ids is None:
        fold_ids = get_dataset_fold_ids(args.metadata_path)
        logger.info(f"Auto-detected fold IDs from metadata: {fold_ids}")

    logger.info(f"Starting Model 2 training — {timestamp}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Training context:    {args.training_context}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Reference class:     {args.reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {args.diseases or '(all)'}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Models:              {model_names}")
    logger.info(f"  P-values:            {args.p_values}")
    logger.info(f"  Retrain GLM on A+B:  {args.retrain_full}")
    logger.info(f"  Clustering n_jobs:   {args.n_jobs}")
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
        model_names=model_names,
        gene_locus=args.gene_locus,
        p_values=args.p_values,
        retrain_on_full_train=args.retrain_full,
        n_jobs=args.n_jobs,
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

    # Flatten per-fold eval results for logging; keep structured in JSON
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
                "model_names": model_names,
                "p_values": args.p_values,
                "retrain_on_full_train": args.retrain_full,
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
        "Models": ", ".join(model_names),
        "Gene locus": args.gene_locus,
        "P-values": str(args.p_values or DEFAULT_P_VALUES),
        "Folds": str(fold_ids),
        "Retrain GLM on A+B": str(args.retrain_full),
    }
    if args.classification_mode != "multiclass" and args.reference_class:
        run_info["Reference class"] = args.reference_class

    md_content = generate_results_md(
        all_results=all_results,
        classification_mode=args.classification_mode,
        timestamp=timestamp,
        model_label="Model 2",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=model_names,
        has_abstention=True,
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
        model_label="Model 2",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=model_names,
        has_abstention=True,
    )

    # ------------------------------------------------------------------
    # Print final table
    # ------------------------------------------------------------------
    logger.info("\n--- Summary ---")
    for r in all_eval_flat:
        pair_str = (
            f"{r['disease']}_vs_{r['reference_class']} "
            if "disease" in r else ""
        )
        auroc_str = (
            f"{r['auroc_ovo_weighted']:.4f}"
            if r.get("auroc_ovo_weighted") is not None
            else "N/A  "
        )
        logloss_str = (
            f"{r['log_loss']:.4f}"
            if r.get("log_loss") is not None
            else "N/A  "
        )
        logger.info(
            f"  fold={r['fold_id']} {pair_str}{r['model_name']:20s} "
            f"AUROC={auroc_str} "
            f"LogLoss={logloss_str} "
            f"abstention={r['abstention_rate']:.1%}"
        )

    # Print aggregated summary
    logger.info("\n--- Aggregated Results ---")
    for pair_key, pair_data in all_results.items():
        for model_name, agg in pair_data["aggregated_by_model"].items():
            logger.info(f"  {pair_key} / {model_name}:")
            acc_global = agg.get("accuracy_global")
            acc_str = f"{acc_global:.4f}" if acc_global is not None else "N/A"
            if args.classification_mode == "multiclass":
                auroc_agg = agg.get("auroc_ovo_weighted", {})
                ll_agg = agg.get("log_loss", {})
                auroc_mean = auroc_agg.get("mean")
                ll_mean = ll_agg.get("mean")
                auroc_str_ovo = f"{auroc_mean:.4f}" if auroc_mean is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_OvO={auroc_str_ovo} "
                )
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
                    f"AUPRC_pooled={auprc_str2}"
                )

    # Clean up file handler to flush and release the log file
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
