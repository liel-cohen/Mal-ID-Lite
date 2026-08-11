#!/usr/bin/env python
"""Train and evaluate Model 2 (Convergent Cluster Classifier) for disease classification.

What Model 2 does (for readers new to the package): it groups similar CDR3 sequences
into "clusters", uses Fisher's exact test to find clusters that are enriched in a
disease, then classifies each specimen by how many disease-associated clusters it
contains (a logistic-regression "GLM" on those cluster-hit counts). A specimen that
matches no significant cluster ABSTAINS — it gets no prediction (counted as wrong in
accuracy, and excluded from AUROC/AUPRC).

Two ways to run it (see "Training contexts" below):
  - Cross-validation on one dataset (default): trains and evaluates via held-out folds.
  - Train-all: fit on the WHOLE dataset with NO evaluation, producing reusable
    artifacts to score later on a SEPARATE dataset.

Uses the two-level cache for fast data loading. Saves cluster centroids, the selected
p-value threshold(s), and the fitted classifier(s) to disk.

Classification modes
--------------------
multiclass
    A single N-class classifier trained on all disease classes. Default.
    If the data has exactly 2 classes, proceeds normally (binary multiclass).

binary
    One binary classifier for a single disease-vs-reference pair.
    Requires --reference-class. For 2-class datasets, the non-reference
    disease is auto-detected. For N-class datasets, use --diseases <disease>
    to pick one.

multi-binary
    One independent binary classifier per disease vs. the reference class.
    --reference-class is always required.
    Default (no --diseases): trains all N-1 non-reference diseases.
    With --diseases <d1> <d2> ...: trains only the specified subset of diseases.
    If the data has exactly 2 classes (and no --diseases), behaves identically to binary.
    Each binary pair is fully independent: separate clustering, Fisher test, and GLM.

The train_smaller1 / train_smaller2 split ("ts1" / "ts2")
---------------------------------------------------------
Model 2 always divides its training participants into two disjoint subsets
(stratified by disease, deterministic):
  - train_smaller1 ("ts1", ~2/3): builds the clusters + Fisher scores AND trains
    the final classifier.
  - train_smaller2 ("ts2", ~1/3): used ONLY to pick the best Fisher p-value
    threshold (a hyperparameter), scored by MCC. This is model selection — still
    part of training, NOT held-out test evaluation.
This split is what makes Model 2 different from Model 1 (which just uses ts1+ts2
together). The `--retrain-full` flag refits the final classifier on ts1+ts2 combined
after the threshold is chosen (clusters/threshold are unaffected).

Training contexts (--training-context)
--------------------------------------
Cross-validation — evaluate on THIS dataset via held-out folds (requires a CV_fold
column in the metadata):
  cv_single_model (default) — standalone Model 2 (not used inside the ensemble).
      For each test fold, ts1+ts2 = all non-test participants; the fitted model is
      then evaluated on the held-out test fold. (Fold 0 as the test set means the
      remaining folds are pooled as training data, etc.)
  cv_ensemble               — base model for the ensemble. For each test fold, first
      hold out a third of the non-test participants as the ensemble's validation set,
      then ts1+ts2 = the remaining two-thirds.

Train-all — train on the WHOLE dataset with NO held-out test fold and NO evaluation
metrics; produces reusable artifacts to score later on a SEPARATE dataset. No CV_fold
column is required, and --fold-ids is not allowed:
  train_all                 — ts1+ts2 = every participant.
  train_all_ensemble        — base model for a train-all ensemble: hold out a third
      of all participants as validation, then ts1+ts2 = the other two-thirds
      (mirrors cv_ensemble).

Output directory structure
---------------------------
cv_single_model (default):
  multiclass:   trained_models/<dataset>/cv_single_model/model2/multiclass/<locus>/
  binary:       trained_models/<dataset>/cv_single_model/model2/binary/<locus>/<pair>/
  multi-binary: trained_models/<dataset>/cv_single_model/model2/binary/<locus>/<pair1>/
                                                                                <pair2>/...

cv_ensemble:
  multiclass:   trained_models/<dataset>/cv_ensemble/base_models/<locus>/model2/multiclass/
  binary:       trained_models/<dataset>/cv_ensemble/base_models/<locus>/model2/binary/<pair>/

train_all (parallels cv_single_model):
  multiclass:   trained_models/<dataset>/train_all_single_model/model2/multiclass/<locus>/
  binary:       trained_models/<dataset>/train_all_single_model/model2/binary/<locus>/<pair>/

train_all_ensemble (parallels cv_ensemble):
  multiclass:   trained_models/<dataset>/train_all_ensemble/base_models/<locus>/model2/multiclass/

With --output-suffix <suffix>, the mode directory gets "__<suffix>" appended:
    trained_models/<dataset>/cv_single_model/model2/multiclass__<suffix>/<locus>/

Both binary and multi-binary write to the same binary/<gene_locus>/ subtree, so artifacts
for the same pair are identical regardless of which mode produced them.

Artifacts (CV contexts, per fold)
---------------------------------
    fold_<id>_clusters.joblib                        — centroids + Fisher scores (shared)
    fold_<id>_<model>_p_value.joblib                 — best p-value threshold
    fold_<id>_<model>_model_<suffix>.joblib           — fitted sklearn Pipeline
    fold_<id>_<model>_results_<suffix>.json           — per-p-value training metrics
    fold_<id>_predictions.pkl                        — per-fold predictions + _meta for resume
    summary_<timestamp>.json                         — full run summary (all folds)
    training_<timestamp>.log                         — mirrored log

Where <suffix> records which data the final classifier was fit on: "split1" (default —
train_smaller1 only) or "full" (train_smaller1 + train_smaller2, set by --retrain-full).

Artifacts (train-all contexts)
------------------------------
Train-all does no evaluation, so artifacts have NO fold prefix and there are no
results-per-fold / predictions files:
    clusters.joblib                 — centroids + Fisher scores (shared)
    <model>_p_value.joblib          — best p-value threshold (per variant)
    <model>_model_<suffix>.joblib   — fitted sklearn Pipeline (per variant)
    <model>_results_<suffix>.json   — per-p-value training metrics (per variant)
    <model>_NO_VALID_CLUSTERS.txt   — written instead when a variant found no clusters
    meta.json                       — run params + expected_artifacts (for --resume)
    summary_<timestamp>.json        — training summary (no metrics; training_only=true)
    RESULTS_<timestamp>.md           — human-readable training summary (no metrics)
    training_<timestamp>.log         — mirrored log

A predictions CSV is written per model_name alongside other artifacts:
    multiclass:   <model_name>_multiclass_predictions.csv
    binary:       <disease>_vs_<reference>/<model_name>_binary_predictions.csv

Multiclass columns: participant_label, specimen_label, true_disease, predicted_disease,
    abstained (True/False), score_<class1>, score_<class2>, ..., CV_fold
    Abstained specimens are included with None for predicted_disease and score_* columns.

Binary columns: participant_label, specimen_label, disease_label (0/1), disease_label_str,
    disease_model, model_score (P(disease)), CV_fold
    Only scored (non-abstained) specimens are included.

Resume (--resume)
-----------------
When --resume is passed, completed folds are skipped and their results are loaded from
saved artifacts. predictions.pkl is the last file written in fold processing. Its _meta
records an expected_artifacts list — the filenames that save_fold_artifacts wrote to disk.
A fold is considered complete when predictions.pkl exists (>= 1KB) and every file in
expected_artifacts is present on disk (with size checks for pkl/joblib files). If any
expected artifact is missing, this indicates corruption and the fold is retrained.

When a model found no valid p-value threshold (best_p_value=None), its per-model
artifacts (p_value, pipeline, metrics) are not saved. Instead, a human-readable notice
file (fold_{id}_{model}_NO_VALID_CLUSTERS.txt) is written, and expected_artifacts lists
only the files that were actually saved (clusters.joblib + the notice). The fold is still
complete — it contributes nothing to cross-fold aggregation (matching non-resume behavior).

On resume, the saved _meta is validated against the current run parameters — a mismatch
raises ValueError so the user doesn't accidentally mix results from different
configurations. Incomplete folds have their partial artifacts deleted before retraining.

For train-all contexts (single run, no folds), the completeness sentinel is meta.json
(written last), holding params + expected_artifacts. --resume skips the run when
meta.json exists, all expected_artifacts are present, and params match; a param
mismatch raises; a corrupt/incomplete meta.json triggers a retrain.

Usage examples
--------------
    # Multiclass (default): a single N-class model, cross-validated across all folds
    # in the metadata (each fold is the test set once; that fold's model is trained
    # on the other folds).
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv

    # Binary (2-class data): one disease-vs-reference model; the non-reference disease
    # is auto-detected. Cross-validated over all folds.
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode binary --reference-class Healthy

    # Multi-binary (N-class data): one independent model per disease vs. the reference
    # class (Healthy), each cross-validated over all folds.
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy

    # Binary (N-class data): pick one disease explicitly for the disease-vs-reference
    # model (needed when the data has more than 2 classes).
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode binary --reference-class Healthy --diseases COVID-19

    # Multi-binary for a specific subset of diseases (each vs. the reference class)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --classification-mode multi-binary --reference-class Healthy \\
        --diseases COVID-19 Lupus

    # Restrict cross-validation to fold 0 only: fold 0 is the TEST set and the model
    # is trained on the remaining folds (e.g. folds 1 + 2). Other folds are NOT
    # evaluated. Also train all 5 GLM alpha variants with 8 parallel workers.
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv \\
        --fold-ids 0 --model-names lasso_cv elasticnet_cv0.75 elasticnet_cv elasticnet_cv0.25 ridge_cv \\
        --n-jobs 8

    # Refit the final GLM on train_smaller1 + train_smaller2 combined after the p-value
    # threshold is chosen on train_smaller2 (opt-in; default fits on train_smaller1 only).
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv --retrain-full

    # Train-all (standalone): train ONE model on the WHOLE dataset (every participant),
    # no held-out test fold and NO evaluation. Produces reusable artifacts to score
    # later on a SEPARATE dataset. --retrain-full is recommended so the final GLM uses
    # all the data (otherwise it fits on train_smaller1 only).
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv --training-context train_all --retrain-full

    # Train-all (ensemble base model): like train_all but first holds out a third of the
    # dataset as the ensemble's validation set, training on the other two-thirds
    # (mirrors cv_ensemble). Used when building a train-all ensemble.
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv --training-context train_all_ensemble

    # Resume a partially-completed run (CV: skips folds that already finished;
    # train-all: skips the run if its artifacts already exist and params match)
    python malid_lite/training/train_model2.py --dataset-name mal-id-orig \\
        --metadata-path data/metadata.tsv --resume

Performance note
----------------
    --n-jobs controls parallelism for clustering (Phase 1) and cluster assignment
    during featurization (training grid search + test evaluation).
    Each (V gene, J gene, CDR3 length) supergroup is processed independently.
    Default is 4 workers — safe for most workstations. On machines with 16+ cores and
    >=32 GB RAM, try --n-jobs 8 or higher for faster training. Memory scales with n_jobs
    because each worker holds its own pairwise distance matrix.

Clone ID parameters
-------------------
All training scripts accept clone_id flags (--force-clone-id, --clone-id-use-aa,
--clone-id-identity-threshold, --clone-id-linkage-method). These only need to be
specified when building the cache for the first time. On subsequent runs, omitting
them is fine -- the cached values are accepted as-is. If you explicitly specify a
value that conflicts with the cache, the run fails immediately with a clear error.
See PIPELINE_GUIDE.md > Clone ID Computation for details.
"""

import argparse
import json
import logging
import pickle
import sys
import time
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
    matthews_corrcoef,
    roc_auc_score,
)

# Add project root to path (malid/training/ → malid/ → project root)
# Must come before any malid_lite imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Custom multiclass metrics that handle unnormalized probabilities and missing
# labels gracefully. Matches the original Mal-ID paper's evaluation methodology.
from malid_lite.utils import multiclass_metrics

from malid_lite.dataloader import MalIDPublishedDataLoader, add_clone_id_args, get_clone_id_kwargs
from malid_lite.models.model2_convergent_clusters import (
    BEST_MODEL_FOR_METAMODEL,
    DEFAULT_P_VALUES,
    SEQUENCE_IDENTITY_THRESHOLDS,
    _GLMNET_CV_N_SPLITS,
    FeaturizedData,
    featurize,
    get_artifact_paths,
    get_clusters_path,
    get_no_valid_clusters_path,
    train_convergent_cluster_classifier,
)
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    DISEASE_COL,
    FOLD_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    TRAIN_ALL_TRAINING_CONTEXTS,
    VALID_TRAINING_CONTEXTS,
    aggregate_fold_results,
    check_train_all_split,
    delete_stale_summaries,
    filter_to_binary_pair,
    generate_results_md,
    get_dataset_disease_classes,
    get_metadata_class_counts,
    get_model_classes,
    get_model_output_dir,
    make_pair_name,
    run_training_orchestration,
    save_per_pair_results,
    write_train_all_outputs,
    validate_mode_and_classes,
    train_all_artifacts_complete,
    validate_train_all_meta,
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
    fold_id: Optional[int],
    fold_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fold (or whole-dataset) sequences and join with disease metadata.

    fold_label is "train"/"test" (CV) or "all" (train-all: the whole dataset, no
    fold — fold_id is ignored, pass None).

    Returns
    -------
    (sequences_df, metadata_df)
    sequences_df has all sequence columns plus disease (from metadata join).
    specimen_label column is the specimen identifier.
    """
    if fold_label == "all":
        sequences_df, metadata_df = loader.get_all_data()
    else:
        sequences_df, metadata_df = loader.get_fold_data(fold_id, fold_label)

    if sequences_df.empty:
        raise ValueError(
            f"No sequences found for {'all data' if fold_label == 'all' else f'fold {fold_id} {fold_label}'}"
        )

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
    featurized : FeaturizedData from featurize() on the test fold.
    pipeline : Fitted sklearn Pipeline (StandardScaler + GlmnetLogitNetWrapper).
    classes : Array of disease class names from training (train_smaller1). Used as
        ``labels=`` for confusion_matrix, log_loss, and AUROC/AUPRC. If the test
        fold contains classes not in this array (small-data edge case), a warning
        is logged — those specimens are predicted into known classes and count as
        misclassifications in accuracy/MCC, but are excluded from the confusion
        matrix.
    fold_id : Fold identifier (for logging).
    model_name : Classifier variant name (for logging).
    reference_class : Reference/negative class. When provided and data has exactly
        2 classes, also computes auroc_binary and auprc_binary with the non-reference
        class as positive — matching model 1 binary methodology exactly.

    Returns
    -------
    (metrics, raw_preds)
        metrics   : JSON-serializable dict of evaluation metrics. Includes
            "unseen_test_classes" and "n_unseen_test_specimens" when the test fold
            has classes absent from training.
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

    # Class-mismatch check: warn if test fold has classes the model wasn't
    # trained on (e.g., small data caused a class to land entirely outside
    # train_smaller1). These specimens are predicted into known classes and
    # contribute to MCC/accuracy as misclassifications, but are silently
    # excluded from confusion_matrix (which only counts labels in `classes`).
    test_classes = set(y_true.unique())
    train_classes = set(classes)
    unseen_in_test = test_classes - train_classes
    if unseen_in_test:
        n_unseen_specimens = int(y_true.isin(unseen_in_test).sum())
        logger.warning(
            f"  fold {fold_id} {model_name}: test fold has {n_unseen_specimens} "
            f"specimen(s) from class(es) {sorted(unseen_in_test)} that the model "
            f"was not trained on. These specimens are predicted as one of the "
            f"known classes ({sorted(train_classes)}) and count as misclassifications "
            f"in accuracy/MCC. They are excluded from the confusion matrix. "
            f"This typically happens with small datasets — consider increasing "
            f"training data."
        )
        results["unseen_test_classes"] = sorted(unseen_in_test)
        results["n_unseen_test_specimens"] = n_unseen_specimens

    # Also check the reverse: training classes absent from the test fold.
    # Not a bug, but worth noting — confusion matrix will have empty rows.
    missing_from_test = train_classes - test_classes
    if missing_from_test:
        logger.info(
            f"  fold {fold_id} {model_name}: training class(es) "
            f"{sorted(missing_from_test)} have no scored specimens in the test "
            f"fold. Their confusion matrix rows/columns will be empty."
        )

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
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

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
    fold_id: Optional[int],
    train_result: Dict,
    retrain_on_full_train: bool = False,
    disease_filter: Optional[Tuple[str, str]] = None,
) -> List[str]:
    """Save all artifacts for one fold (or the whole-dataset train-all run) to disk.

    Artifact filenames are determined by get_artifact_paths() (single source of truth).
    fold_id is an int for CV; ``None`` for train-all (filenames omit the ``fold_<id>_``
    prefix — e.g. ``clusters.joblib``, ``<model>_p_value.joblib``).

    Saves (``<pfx>`` = ``fold_<id>_`` for CV, empty for train-all):
    - <pfx>clusters.joblib               : centroids + Fisher scores + disease_classes (shared)
    - <pfx>{model}_p_value.joblib         : best p-value (float)
    - <pfx>{model}_model_{suffix}.joblib  : fitted sklearn Pipeline
    - <pfx>{model}_results_{suffix}.json  : per-p-value metrics
    where suffix is "split1" (retrain_on_full_train=False, default) or "full" (True).

    When a model found no valid p-value (best_p_value=None), its per-model artifacts
    are not saved, and a human-readable notice file is written instead:
    - <pfx>{model}_NO_VALID_CLUSTERS.txt

    Parameters
    ----------
    disease_filter : (disease, reference_class) for binary/multi-binary models, None for
        multiclass. Saved into the clusters artifact so that ConvergentClusterClassifier
        can recover the positive/negative class assignment at inference time and always
        return predict_proba columns in [P(reference), P(disease)] order.

    Returns
    -------
    List[str]
        Filenames of all artifacts saved to disk (excluding predictions.pkl,
        which is saved separately). Used by _save_fold_predictions to record
        expected_artifacts in _meta for corruption detection on resume.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_filenames: List[str] = []

    # Shared cluster artifact (same regardless of model_name or retrain_on_full_train)
    clusters_path = get_clusters_path(output_dir, fold_id)
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
    saved_filenames.append(clusters_path.name)

    # Per-model artifacts
    min_cluster_pvalue = train_result.get("min_cluster_pvalue")

    for model_name, model_result in train_result["results"].items():
        if model_result["best_p_value"] is None:
            # Build diagnostic message with p-value suggestion if clusters exist
            pval_suggestion = ""
            if min_cluster_pvalue is not None and np.isfinite(min_cluster_pvalue):
                pval_suggestion = (
                    f"\nDiagnostic info:\n"
                    f"  Minimum p-value across all clusters: {min_cluster_pvalue:.2e}\n"
                    f"  Consider using a wider p-value range that includes values\n"
                    f"  >= {min_cluster_pvalue:.2e}.\n"
                )

            notice_path = get_no_valid_clusters_path(output_dir, fold_id, model_name)
            if fold_id is None:
                # Train-all: no folds, no cross-fold aggregation. A variant that
                # abstains here has NO usable classifier for the whole dataset.
                notice_body = (
                    f"train-all, model '{model_name}': no valid p-value threshold found.\n"
                    f"\n"
                    f"None of the tested p-values produced enough significant convergent\n"
                    f"clusters to train a classifier on the whole dataset. No per-model\n"
                    f"artifacts (p_value, pipeline, metrics) were saved — this variant has\n"
                    f"NO trained model. Downstream scoring (ensemble/external eval) will\n"
                    f"treat every specimen as abstained for this variant.\n"
                    f"{pval_suggestion}\n"
                    f"To get a usable model, widen --p-values (see diagnostic above) or\n"
                    f"train a different --model-names variant. On --resume this run is\n"
                    f"recognized as complete and will not be retrained.\n"
                )
            else:
                notice_body = (
                    f"Fold {fold_id}, model '{model_name}': no valid p-value threshold found.\n"
                    f"\n"
                    f"None of the tested p-values produced enough significant convergent\n"
                    f"clusters to train a classifier. This fold contributes nothing to\n"
                    f"cross-fold aggregation for this model (the other folds carry the\n"
                    f"results). No per-model artifacts (p_value, pipeline, metrics) were\n"
                    f"saved.\n"
                    f"{pval_suggestion}\n"
                    f"This is expected in some folds — it does not indicate an error.\n"
                    f"On resume (--resume), this fold will be correctly recognized as\n"
                    f"complete and will not be retrained.\n"
                )
            notice_path.write_text(notice_body)
            # Mutual exclusion: a prior run may have written a valid pipeline for this
            # model. Remove it so the abstain marker never coexists with a trained
            # pipeline — the ensemble checks the marker FIRST and forces full abstention,
            # so a stale pipeline would be silently ignored (and, symmetrically, a stale
            # marker left over a valid pipeline would silently suppress that model). Cover
            # both retrain suffixes since retrain_on_full_train may have changed between
            # runs. (Leave the shared clusters.joblib — other model variants may still be
            # valid this run.)
            for _rf in (False, True):
                _stale = get_artifact_paths(output_dir, fold_id, model_name, _rf)
                for _key in ("p_value", "pipeline", "metrics"):
                    if _stale[_key].exists():
                        _stale[_key].unlink()
            saved_filenames.append(notice_path.name)
            logger.warning(
                f"  {model_name}: no valid p-value threshold found — "
                f"per-model artifacts not saved (see {notice_path.name})"
            )
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

        saved_filenames.append(paths["p_value"].name)
        saved_filenames.append(paths["pipeline"].name)
        saved_filenames.append(paths["metrics"].name)
        logger.info(f"  Saved {model_name}: p_value={model_result['best_p_value']}")

        # Mutual exclusion (see the abstain branch above): remove any stale abstain
        # marker for this model from a prior run that found no valid clusters, so the
        # ensemble doesn't check the marker first and silently ignore this freshly-
        # trained pipeline. Also remove the other-retrain-suffix pipeline/metrics left
        # by a prior run with the opposite retrain_on_full_train (p_value/clusters are
        # suffix-independent and were just (re)written, so leave them).
        stale_marker = get_no_valid_clusters_path(output_dir, fold_id, model_name)
        if stale_marker.exists():
            stale_marker.unlink()
        _other = get_artifact_paths(
            output_dir, fold_id, model_name, not retrain_on_full_train
        )
        for _key in ("pipeline", "metrics"):
            if _other[_key].exists():
                _other[_key].unlink()

    return saved_filenames


# ---------------------------------------------------------------------------
# Resume support: per-fold artifact check, save, load, and validation
# ---------------------------------------------------------------------------

_MIN_PKL_BYTES = 1024  # guard against truncated pickles from a crash


def _get_fold_artifact_paths(
    output_dir: Path,
    fold_id: Optional[int],
    model_names: List[str],
    retrain_on_full_train: bool,
) -> List[Path]:
    """Return all artifact paths that could exist for this run (CV fold or train-all).

    Used for cleanup when an incomplete run is retrained: every path in the
    returned list that exists on disk is deleted before retraining.

    Includes the shared clusters artifact, per-model artifacts (p_value, pipeline,
    metrics) via get_artifact_paths(), per-model NO_VALID_CLUSTERS notice files, and
    the completeness sentinel — ``fold_<id>_predictions.pkl`` for a CV fold, or
    ``meta.json`` for train-all (``fold_id is None``). All names are fold-optional
    via the shared path builders.
    """
    paths = [get_clusters_path(output_dir, fold_id)]
    for model_name in model_names:
        ap = get_artifact_paths(output_dir, fold_id, model_name, retrain_on_full_train)
        paths.extend([ap["p_value"], ap["pipeline"], ap["metrics"]])
        paths.append(get_no_valid_clusters_path(output_dir, fold_id, model_name))
    # Completeness sentinel (written last): predictions.pkl for CV, meta.json for train-all.
    if fold_id is None:
        paths.append(output_dir / "meta.json")
    else:
        paths.append(output_dir / f"fold_{fold_id}_predictions.pkl")
    return paths


def _check_fold_complete(
    output_dir: Path,
    fold_id: int,
) -> Optional[Dict]:
    """Check whether a fold's training and evaluation completed successfully.

    A fold is complete when:
      1. predictions.pkl exists and can be unpickled
      2. predictions.pkl contains _meta with an expected_artifacts list
      3. Every file in expected_artifacts exists on disk
      4. Pipeline model artifacts (``_model_`` in name) are >= _MIN_PKL_BYTES
         (guards against truncated writes from a crash mid-save)

    predictions.pkl is the LAST file written in fold processing. Its _meta
    records exactly which artifacts were saved by save_fold_artifacts.
    When a model found no valid p-value (best_p_value=None), its per-model
    artifacts are absent from expected_artifacts (only clusters.joblib and
    a NO_VALID_CLUSTERS.txt notice are listed). The fold is still complete —
    it just contributes nothing to cross-fold aggregation.

    Size checks are applied only to pipeline model files (which contain
    ``_model_`` in the name, e.g. ``fold_0_lasso_cv_model_split1.joblib``).
    Other binary artifacts can be legitimately small:
      - predictions.pkl: ~350 bytes when by_model is empty (best_p_value=None)
      - clusters.joblib: ~660 bytes with few centroids
      - _p_value.joblib: ~21 bytes (single float)
    Corruption of predictions.pkl is caught by the pickle.load try/except
    and by structural validation (_meta, expected_artifacts).

    If any file in expected_artifacts is missing on disk, this indicates
    corruption (not a "no clusters" scenario) and the fold needs retraining.

    Returns None if predictions.pkl has no expected_artifacts key — this means
    it was saved by an older version of the code before this field was added.
    The fold must be retrained.

    Returns
    -------
    Optional[Dict]
        The loaded predictions.pkl data if the fold is complete, or None if
        incomplete/corrupt. Returning the loaded data avoids redundant pickle
        loads in the calling code (which would otherwise re-load for
        _validate_fold_meta and result restoration).
    """
    preds_path = output_dir / f"fold_{fold_id}_predictions.pkl"

    if not preds_path.exists():
        return None

    # Load and validate structure — catches truncated/corrupt pickles
    try:
        with open(preds_path, "rb") as f:
            preds_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        return None

    meta = preds_data.get("_meta")
    if meta is None:
        return None

    expected = meta.get("expected_artifacts")
    if expected is None:
        logger.warning(
            f"  Fold {fold_id}: predictions.pkl has no expected_artifacts "
            f"(saved before artifact tracking was added). Will retrain."
        )
        return None

    # Verify every expected artifact exists on disk
    for artifact_name in expected:
        artifact_path = output_dir / artifact_name
        if not artifact_path.exists():
            logger.warning(
                f"  Fold {fold_id}: expected artifact '{artifact_name}' missing "
                f"from disk (possible corruption). Will retrain."
            )
            return None

        # Size-check pipeline model files only — these are large serialized
        # sklearn pipelines that would be truncated by a mid-save crash.
        # Other binary artifacts (clusters.joblib, _p_value.joblib,
        # predictions.pkl) can be legitimately small.
        if "_model_" in artifact_name and artifact_path.stat().st_size < _MIN_PKL_BYTES:
            logger.warning(
                f"  Fold {fold_id}: artifact '{artifact_name}' is truncated "
                f"({artifact_path.stat().st_size} bytes < {_MIN_PKL_BYTES}). "
                f"Will retrain."
            )
            return None

    return preds_data


def _save_fold_predictions(
    output_dir: Path,
    fold_id: int,
    model_names: List[str],
    per_model_data: Dict[str, Dict],
    model_params: Dict,
    training_context: str,
    expected_artifacts: List[str],
) -> Path:
    """Save per-fold predictions + metadata for resume support.

    The pickle contains:
      - by_model: {model_name: {eval_result, raw_preds, predictions_rows}}
        Only models with valid results (best_p_value != None) are included.
      - _meta: model parameters, run settings, and expected_artifacts for
        validation on resume.

    Parameters
    ----------
    expected_artifacts : List of filenames (relative to output_dir) that were
        saved to disk for this fold by save_fold_artifacts. On resume,
        _check_fold_complete verifies that every listed file still exists —
        missing files indicate corruption and trigger retraining.
    """
    preds_path = output_dir / f"fold_{fold_id}_predictions.pkl"
    data = {
        "by_model": per_model_data,
        "_meta": {
            "model_names": sorted(model_names),
            "model_params": model_params,
            "training_context": training_context,
            "fold_id": fold_id,
            "expected_artifacts": sorted(expected_artifacts),
        },
    }
    with open(preds_path, "wb") as f:
        pickle.dump(data, f)
    return preds_path


def _load_fold_results(
    output_dir: Path, fold_id: int,
) -> Dict:
    """Load saved fold predictions for resume.

    Returns the full predictions.pkl dict with "by_model" and "_meta" keys.
    Each entry in by_model[model_name] contains:
      - eval_result: per-fold metric dict
      - raw_preds: {y_true, y_pred, y_proba, classes} or None (all abstained)
      - predictions_rows: list of per-specimen dicts for the predictions CSV
    """
    preds_path = output_dir / f"fold_{fold_id}_predictions.pkl"
    with open(preds_path, "rb") as f:
        return pickle.load(f)


def _validate_fold_meta(
    preds_data: Dict,
    fold_id: int,
    model_names: List[str],
    current_model_params: Dict,
    current_training_context: str,
) -> None:
    """Validate that a resumed fold's saved metadata matches current run parameters.

    Raises ValueError if fold_id, model_names, training_context, or any key in
    model_params differs between the saved artifact and the current run.

    model_params is expected to contain both model hyperparameters
    (sequence_identity_threshold, p_values, retrain_on_full_train) and run-level
    settings (classification_mode, diseases, dataset_name, reference_class,
    disease_filter) that affect training outcomes.

    Parameters
    ----------
    preds_data : The already-loaded predictions.pkl dict (from _check_fold_complete
        or _load_fold_results). Must contain "_meta" key.
    """
    meta = preds_data.get("_meta")
    if meta is None:
        raise ValueError(
            f"Fold {fold_id}: predictions.pkl has no _meta (saved before resume "
            f"metadata was added). Delete fold_{fold_id}_predictions.pkl and "
            f"re-run to retrain this fold."
        )

    # Validate fold_id (defensive: filename encodes fold_id, but catch renamed files)
    saved_fold = meta.get("fold_id")
    if saved_fold is not None and saved_fold != fold_id:
        raise ValueError(
            f"Fold {fold_id}: fold_id mismatch. "
            f"Saved: {saved_fold!r}, current: {fold_id!r}. "
            f"Wrong artifact file?"
        )

    # Validate model_names (sorted comparison)
    saved_names = meta.get("model_names")
    if saved_names is not None and sorted(saved_names) != sorted(model_names):
        raise ValueError(
            f"Fold {fold_id}: model_names mismatch. "
            f"Saved: {sorted(saved_names)!r}, current: {sorted(model_names)!r}. "
            f"Delete fold artifacts and re-run with matching --model-names."
        )

    # Validate training_context
    saved_ctx = meta.get("training_context")
    if saved_ctx != current_training_context:
        raise ValueError(
            f"Fold {fold_id}: training_context mismatch. "
            f"Saved: {saved_ctx!r}, current: {current_training_context!r}. "
            f"Delete fold artifacts and re-run, or use matching --training-context."
        )

    # Validate model_params (model hyperparams + run-level settings).
    # Only compare keys present in the saved artifact — older artifacts may
    # not have keys added later (e.g., classification_mode, disease_filter).
    # If a key IS present in saved (even with value None), it must match.
    saved_params = meta.get("model_params", {})
    for key in sorted(current_model_params):
        if key not in saved_params:
            continue
        saved_val = saved_params[key]
        current_val = current_model_params[key]
        if saved_val != current_val:
            raise ValueError(
                f"Fold {fold_id}: model parameter '{key}' mismatch. "
                f"Saved: {saved_val!r}, current: {current_val!r}. "
                f"Delete fold artifacts and re-run with matching parameters, "
                f"or remove the conflicting CLI argument."
            )


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
    resume: bool = False,
    run_params: Optional[Dict] = None,
    glmnet_cv_n_splits: int = _GLMNET_CV_N_SPLITS,
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
    n_jobs              : Parallel workers for clustering and cluster assignment.
    verbose             : Logging verbosity (0=quiet, 1=normal, 2=debug).
    disease_filter      : Optional (disease, reference_class) tuple. If provided,
        sequences and metadata are filtered to participants in
        {disease, reference_class} before training. Used for binary and
        multi-binary modes. If None, all participants are used (multiclass).
    training_context    : Controls which participants are used for training via
        centralized split persistence. "cv_single_model" uses all non-test
        participants; "cv_ensemble" excludes validation participants.
    resume : If True, skip folds whose artifacts already exist on disk and
        reload their saved results for aggregation. Folds with incomplete
        artifacts are retrained normally.
    run_params : Optional dict with classification_mode, diseases, dataset_name,
        reference_class. Saved in artifact _meta and validated on resume to
        prevent mixing results from different run configurations.

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

    # Build full model params dict for _meta: model hyperparams + run settings
    # + disease_filter. Saved in predictions.pkl and compared on resume.
    meta_model_params = {
        "sequence_identity_threshold": sequence_identity_threshold,
        "p_values": sorted(p_values),
        "retrain_on_full_train": retrain_on_full_train,
        **(run_params or {}),
        "disease_filter": disease_filter,
    }

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
        # Resume: skip folds with complete artifacts on disk
        # ------------------------------------------------------------------
        if resume:
            # _check_fold_complete returns the loaded preds_data if complete,
            # None if incomplete/corrupt. This avoids redundant pickle loads.
            preds_data = _check_fold_complete(output_dir, fold_id)
            if preds_data is not None:
                expected = preds_data["_meta"].get("expected_artifacts", [])
                found_list = ", ".join(
                    sorted([*expected, f"fold_{fold_id}_predictions.pkl"])
                )
                logger.info(
                    f"  Skipped (all artifacts verified on disk)\n"
                    f"  Found: {found_list}\n"
                    f"  Will do: load existing results (no training or evaluation)"
                )

                # Validate saved metadata against current run params
                _validate_fold_meta(
                    preds_data, fold_id, model_names,
                    current_model_params=meta_model_params,
                    current_training_context=training_context,
                )

                # Merge saved results into cross-fold accumulators.
                # Models with best_p_value=None were not saved in by_model —
                # nothing to restore, matching the original behavior where
                # skipped models don't contribute to aggregation.
                for mn in model_names:
                    if mn in preds_data["by_model"]:
                        mn_data = preds_data["by_model"][mn]
                        all_eval_results.append(mn_data["eval_result"])
                        raw_preds_by_model[mn].append(mn_data["raw_preds"])
                        predictions_rows_by_model[mn].extend(
                            mn_data["predictions_rows"]
                        )
                continue

            # Delete any incomplete artifacts before retraining to prevent
            # mixing old and new files (e.g., crash between saving model and
            # saving predictions.pkl would leave a new model with old results).
            for artifact in _get_fold_artifact_paths(
                output_dir, fold_id, model_names, retrain_on_full_train
            ):
                if artifact.exists():
                    logger.info(f"  Deleting incomplete artifact: {artifact.name}")
                    artifact.unlink()

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
        if len(train_smaller1_df) == 0:
            raise ValueError(
                f"train_smaller1 is empty after split filtering (fold {fold_id}, "
                f"context={training_context}). Expected {len(ts1_participants)} participants."
            )
        if len(train_smaller2_df) == 0:
            raise ValueError(
                f"train_smaller2 is empty after split filtering (fold {fold_id}, "
                f"context={training_context}). Expected {len(ts2_participants)} participants."
            )
        if not disease_filter:
            # Multiclass: split participants are a superset of the data only when some
            # were dropped by downsampling QC (all sequences removed) — a LEGITIMATE
            # state get_fold_data already reported. Warn, don't abort (mirrors the
            # lenient train-all handling in check_train_all_split); a hard equality
            # check here would crash CV training on a normal QC state. `> split` is still
            # impossible (data is masked to the split) and is caught by the else branch's
            # logic conceptually; here fewer-than-split is the only reachable mismatch.
            if ts1_actual < len(ts1_participants):
                logger.warning(
                    f"train_smaller1: {len(ts1_participants) - ts1_actual} of "
                    f"{len(ts1_participants)} split participant(s) absent (dropped by "
                    f"downsampling QC) — fold {fold_id}, context={training_context}."
                )
            if ts2_actual < len(ts2_participants):
                logger.warning(
                    f"train_smaller2: {len(ts2_participants) - ts2_actual} of "
                    f"{len(ts2_participants)} split participant(s) absent (dropped by "
                    f"downsampling QC) — fold {fold_id}, context={training_context}."
                )
        else:
            # Binary: data was filtered to 2 diseases, so only a subset of
            # split participants will be present. Just verify subset relationship.
            if ts1_actual > len(ts1_participants):
                raise ValueError(
                    f"train_smaller1 has MORE participants ({ts1_actual}) than split "
                    f"({len(ts1_participants)}) — impossible (fold {fold_id})"
                )
            if ts2_actual > len(ts2_participants):
                raise ValueError(
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
        # specimen → participant mapping for abstention details and predictions CSV
        spec_to_part = test_metadata_df.set_index(SPECIMEN_COL)[PARTICIPANT_COL]

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
            glmnet_cv_n_splits=glmnet_cv_n_splits,
        )

        # ------------------------------------------------------------------
        # Save artifacts
        # ------------------------------------------------------------------
        logger.info("Saving artifacts...")
        saved_artifact_names = save_fold_artifacts(
            output_dir, fold_id, train_result, retrain_on_full_train,
            disease_filter=disease_filter,
        )

        # ------------------------------------------------------------------
        # Evaluate on test fold
        # ------------------------------------------------------------------
        logger.info("Evaluating on test fold...")
        disease_classes = train_result["disease_classes"]

        # Collect per-fold data for predictions.pkl before merging into
        # cross-fold accumulators. Only models with valid results are included.
        fold_per_model_data: Dict[str, Dict] = {}

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
                n_jobs=n_jobs,
            )

            eval_result, raw_preds = evaluate_on_test(
                featurized=fd_test,
                pipeline=model_result["pipeline"],
                classes=np.array(disease_classes),
                fold_id=fold_id,
                model_name=model_name,
                reference_class=disease_filter[1] if disease_filter else None,
            )

            # Abstained specimen details for markdown reporting
            if fd_test.n_abstained > 0:
                eval_result["test_abstained_details"] = [
                    {
                        "specimen_label": str(specimen),
                        "participant_label": str(spec_to_part.get(specimen, "unknown")),
                        "disease": str(disease_label),
                    }
                    for specimen, disease_label in fd_test.abstained_sample_y.items()
                ]
            else:
                eval_result["test_abstained_details"] = []

            # Tag binary pair results for identification in summary
            if disease_filter:
                eval_result["disease"] = disease_filter[0]
                eval_result["reference_class"] = disease_filter[1]

            # Collect per-specimen rows for this fold (predictions CSV + resume)
            fold_pred_rows: List[Dict] = []
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
                    fold_pred_rows.append({
                        "participant_label": participant,
                        "specimen_label": specimen,
                        "disease_label": int(true_disease == disease_class),
                        "disease_label_str": str(true_disease),
                        "disease_model": disease_class,
                        "model_score": float(score),
                        FOLD_COL: fold_id,
                    })
            elif not disease_filter:
                # Multiclass: one score column per class; abstained specimens included with NaN
                str_classes = [str(c) for c in disease_classes]
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
                            FOLD_COL: fold_id,
                        }
                        for cls, score in zip(str_classes, proba_row):
                            row[f"score_{cls}"] = float(score)
                        fold_pred_rows.append(row)
                for specimen, true_d in fd_test.abstained_sample_y.items():
                    row = {
                        "participant_label": spec_to_part.get(specimen),
                        "specimen_label": specimen,
                        "true_disease": str(true_d),
                        "predicted_disease": None,
                        "abstained": True,
                        FOLD_COL: fold_id,
                    }
                    for cls in str_classes:
                        row[f"score_{cls}"] = None
                    fold_pred_rows.append(row)

            # Store in per-fold dict (for predictions.pkl) and extend cross-fold
            fold_per_model_data[model_name] = {
                "eval_result": eval_result,
                "raw_preds": raw_preds,
                "predictions_rows": fold_pred_rows,
            }
            all_eval_results.append(eval_result)
            raw_preds_by_model[model_name].append(raw_preds)
            predictions_rows_by_model[model_name].extend(fold_pred_rows)

            auroc_val = next((eval_result.get(k) for k in ("auroc_binary", "auroc_ovo_weighted") if eval_result.get(k) is not None), None)
            auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A"
            logloss_str = (
                f"{eval_result['log_loss']:.4f}"
                if eval_result.get("log_loss") is not None
                else "N/A"
            )
            mcc_str = f"{eval_result['mcc']:.4f}" if eval_result.get("mcc") is not None else "N/A"
            logger.info(
                f"  {model_name}: AUROC={auroc_str} MCC={mcc_str} "
                f"LogLoss={logloss_str} "
                f"abstention={eval_result['abstention_rate']:.1%} "
                f"({eval_result['n_scored']}/{eval_result['n_scored'] + eval_result['n_abstained']} scored)"
            )

        # Save per-fold predictions for resume support.
        # expected_artifacts records which files were saved by save_fold_artifacts
        # so that _check_fold_complete can detect corruption (missing files) on resume.
        _save_fold_predictions(
            output_dir, fold_id, model_names,
            per_model_data=fold_per_model_data,
            model_params=meta_model_params,
            training_context=training_context,
            expected_artifacts=saved_artifact_names,
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
                    FOLD_COL,
                ])
                predictions_file = output_dir / f"{mn}_binary_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(
                    f"  Binary predictions saved: {predictions_file.name} "
                    f"({len(predictions_df)} rows across {len(fold_ids)} fold(s))"
                )
            else:
                logger.warning(
                    f"  No prediction rows for {mn} — binary predictions CSV not written. "
                    f"This means {mn} had no scored specimens across all {len(fold_ids)} fold(s) "
                    f"(all specimens abstained or no valid model was found)."
                )
    else:
        for mn in model_names:
            rows = predictions_rows_by_model[mn]
            if rows:
                score_cols = sorted({k for _row in rows for k in _row if k.startswith("score_")})
                fixed_cols = [
                    "participant_label", "specimen_label", "true_disease", "predicted_disease",
                    "abstained", FOLD_COL,
                ]
                predictions_df = pd.DataFrame(rows, columns=fixed_cols + score_cols)
                predictions_file = output_dir / f"{mn}_multiclass_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(
                    f"  Multiclass predictions saved: {predictions_file.name} "
                    f"({len(predictions_df)} rows across {len(fold_ids)} fold(s))"
                )
            else:
                logger.warning(
                    f"  No prediction rows for {mn} — multiclass predictions CSV not written. "
                    f"This means {mn} had no valid model across all {len(fold_ids)} fold(s)."
                )

    return all_eval_results, aggregated_by_model


# ---------------------------------------------------------------------------
# Train-all: single-pass training on the whole dataset (no evaluation)
# ---------------------------------------------------------------------------

def _run_train_all(
    loader: MalIDPublishedDataLoader,
    output_dir: Path,
    model_names: List[str],
    sequence_identity_threshold: float,
    p_values: List[float],
    retrain_on_full_train: bool,
    n_jobs: int,
    verbose: int,
    disease_filter: Optional[Tuple[str, str]] = None,
    training_context: str = "train_all",
    run_params: Optional[Dict] = None,
    glmnet_cv_n_splits: int = _GLMNET_CV_N_SPLITS,
    resume: bool = False,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Train Model 2 once on the whole dataset (train-all); no evaluation.

    Single-pass counterpart of ``_run_fold_loop`` for train-all contexts. Model 2
    genuinely uses ts1 and ts2 SEPARATELY — cluster + Fisher on ts1, pick the
    p-value threshold by MCC on ts2 (a training-time hyperparameter search), and
    (if ``retrain_on_full_train``) refit the final GLM on ts1+ts2. There is no test
    set and no evaluation. For ``train_all``, ts1+ts2 = all participants; for
    ``train_all_ensemble``, ts1+ts2 = the 2/3 that excludes the validation third.

    Artifacts are written WITHOUT a fold prefix (``clusters.joblib``,
    ``<model>_p_value.joblib``, ``<model>_model_<suffix>.joblib``,
    ``<model>_results_<suffix>.json``, or ``<model>_NO_VALID_CLUSTERS.txt``), plus a
    ``meta.json`` (resume sentinel). Returns ``([training_info], {})``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_tag = (
        f" [{make_pair_name(disease_filter[0], disease_filter[1])}]"
        if disease_filter else ""
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"Train-all ({training_context}){pair_tag}")
    logger.info(f"{'='*60}")

    meta_file = output_dir / "meta.json"
    # Model params saved in meta.json and compared on resume.
    model_params = {
        "sequence_identity_threshold": sequence_identity_threshold,
        "p_values": sorted(p_values),
        "retrain_on_full_train": retrain_on_full_train,
        "glmnet_cv_n_splits": glmnet_cv_n_splits,
    }
    meta_expected = {
        "model_names": sorted(model_names),
        "model_params": model_params,
        "run_params": run_params or {},
        "training_context": training_context,
        "disease_filter": list(disease_filter) if disease_filter else None,
    }

    # --- Resume: skip if complete + params match; else clear partial artifacts ---
    if resume:
        saved_meta = None
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    saved_meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"  Corrupt meta.json ({e}); treating as incomplete and retraining."
                )
                saved_meta = None
        expected_artifacts = saved_meta.get("expected_artifacts", []) if saved_meta else []
        # A complete run ALWAYS records at least one expected artifact (see the
        # meta.json write below). An empty/missing expected_artifacts list means the
        # meta.json is truncated or predates artifact tracking — treat as incomplete
        # and retrain, rather than letting all([]) == True mark it complete.
        complete = (
            saved_meta is not None
            and "training_info" in saved_meta
            and len(expected_artifacts) > 0
            and train_all_artifacts_complete(
                output_dir / name for name in expected_artifacts
            )
        )
        if complete:
            validate_train_all_meta(
                saved_meta, meta_expected, output_dir,
                match_keys=["model_names", "training_context", "disease_filter"],
            )
            logger.info(
                "  Skipped (all artifacts present and params match); "
                "reloading saved training info."
            )
            return [saved_meta["training_info"]], {}
        # Incomplete/corrupt: delete partial artifacts before retraining. The
        # cleanup list includes meta.json (the train-all sentinel) for fold_id=None.
        for p in _get_fold_artifact_paths(output_dir, None, model_names, retrain_on_full_train):
            if p.exists():
                logger.info(f"  Deleting incomplete artifact: {p.name}")
                p.unlink()
    else:
        # Fresh run (no resume): clear any prior train-all artifacts in this dir so a
        # stale file from a previous run/config can't linger — e.g. a
        # <model>_NO_VALID_CLUSTERS.txt marker left by an earlier narrow-p-value run
        # that the ensemble would otherwise check first and use to force abstention.
        # (save_fold_artifacts also enforces marker/pipeline mutual exclusion per model;
        # this additionally clears the sentinel/clusters for a clean slate.)
        for p in _get_fold_artifact_paths(output_dir, None, model_names, retrain_on_full_train):
            if p.exists():
                logger.info(f"  Removing prior artifact (fresh run): {p.name}")
                p.unlink()

    # --- Load the whole dataset + optional binary filter ---
    logger.info("Loading full dataset (train-all)...")
    seqs_df, meta_df = load_and_prepare_fold(loader, None, "all")
    if disease_filter:
        disease, reference_class = disease_filter
        seqs_df, meta_df = filter_to_binary_pair(seqs_df, meta_df, disease, reference_class)

    # --- Split into ts1 (cluster+Fisher) and ts2 (p-value selection) ---
    ts1_participants = set(loader.get_split_participants(
        None, training_context, ["train_smaller1"]
    ))
    ts2_participants = set(loader.get_split_participants(
        None, training_context, ["train_smaller2"]
    ))
    ts1_df = seqs_df[seqs_df[PARTICIPANT_COL].isin(ts1_participants)].copy()
    ts2_df = seqs_df[seqs_df[PARTICIPANT_COL].isin(ts2_participants)].copy()

    # --- Integrity checks on each subset (Decision 2.C / 3.E) ---
    # `raise` (not `assert`) so these data-state checks survive `python -O`.
    if len(ts1_df) == 0:
        raise RuntimeError(
            f"train_smaller1 is empty after split filtering "
            f"(context={training_context}, pair={disease_filter})."
        )
    if len(ts2_df) == 0:
        raise RuntimeError(
            f"train_smaller2 is empty after split filtering "
            f"(context={training_context}, pair={disease_filter})."
        )
    check_train_all_split(
        loader, set(ts1_df[PARTICIPANT_COL].unique()), ts1_participants,
        training_context, disease_filter, role_label="train_smaller1",
    )
    check_train_all_split(
        loader, set(ts2_df[PARTICIPANT_COL].unique()), ts2_participants,
        training_context, disease_filter, role_label="train_smaller2",
    )
    logger.info(
        f"  train_smaller1: {ts1_df[PARTICIPANT_COL].nunique()} participants, "
        f"{len(ts1_df):,} sequences; "
        f"train_smaller2: {ts2_df[PARTICIPANT_COL].nunique()} participants, "
        f"{len(ts2_df):,} sequences"
    )

    # --- Train (reused unchanged) ---
    logger.info("Training convergent-cluster classifier(s)...")
    train_result = train_convergent_cluster_classifier(
        train_smaller1_df=ts1_df,
        train_smaller2_df=ts2_df,
        sequence_identity_threshold=sequence_identity_threshold,
        model_names=model_names,
        p_values=p_values,
        disease_col=DISEASE_COL,
        retrain_on_full_train=retrain_on_full_train,
        n_jobs=n_jobs,
        verbose=verbose,
        glmnet_cv_n_splits=glmnet_cv_n_splits,
    )

    # --- Save artifacts (no fold prefix; reused save_fold_artifacts) ---
    saved_artifacts = save_fold_artifacts(
        output_dir, None, train_result, retrain_on_full_train,
        disease_filter=disease_filter,
    )

    # --- Build training_info (no metrics) ---
    classes = train_result["disease_classes"]
    per_model = {}
    for model_name, model_result in train_result["results"].items():
        per_model[model_name] = {
            "best_p_value": model_result["best_p_value"],
            "abstained": model_result["best_p_value"] is None,
        }
    training_info = {
        "training_context": training_context,
        "classes": [str(c) for c in classes],
        "n_train_ts1_participants": int(ts1_df[PARTICIPANT_COL].nunique()),
        "n_train_ts1_specimens": int(ts1_df[SPECIMEN_COL].nunique()),
        "n_train_ts1_sequences": int(len(ts1_df)),
        "n_train_ts2_participants": int(ts2_df[PARTICIPANT_COL].nunique()),
        "n_train_ts2_specimens": int(ts2_df[SPECIMEN_COL].nunique()),
        "n_train_ts2_sequences": int(len(ts2_df)),
        "retrain_on_full_train": retrain_on_full_train,
        "min_cluster_pvalue": train_result.get("min_cluster_pvalue"),
        "models": per_model,
    }
    if disease_filter:
        training_info["disease"] = disease_filter[0]
        training_info["reference_class"] = disease_filter[1]

    # --- Write meta.json LAST (resume sentinel + provenance) ---
    meta_out = dict(meta_expected)
    meta_out["expected_artifacts"] = sorted(saved_artifacts)
    meta_out["training_info"] = training_info
    with open(meta_file, "w") as f:
        json.dump(
            meta_out, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )
    logger.info(f"  Saved {len(saved_artifacts)} artifact(s) + meta.json")

    return [training_info], {}


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_training_params(
    p_values: Optional[List[float]] = None,
    sequence_identity_threshold: Optional[float] = None,
    glmnet_cv_n_splits: Optional[int] = None,
    **_kwargs,
) -> None:
    """Validate Model 2 training parameter ranges.

    Called by both the standalone main() and ensemble auto-training dispatch.
    Only non-None values are checked (None means "use model default").

    Raises ValueError with a clear message for any out-of-range value.
    """
    if p_values is not None:
        for pv in p_values:
            if not (0.0 < pv < 1.0):
                raise ValueError(
                    f"Model 2: p_values must all be in (0, 1), got {pv}. "
                    f"Example valid values: [0.0005, 0.001, 0.005, 0.01, 0.05]."
                )
    if sequence_identity_threshold is not None:
        if not (0.0 < sequence_identity_threshold <= 1.0):
            raise ValueError(
                f"Model 2: sequence_identity_threshold must be in (0, 1], "
                f"got {sequence_identity_threshold}."
            )
    if glmnet_cv_n_splits is not None:
        if not isinstance(glmnet_cv_n_splits, int) or glmnet_cv_n_splits < 2:
            raise ValueError(
                f"Model 2: glmnet_cv_n_splits must be an integer >= 2, "
                f"got {glmnet_cv_n_splits}."
            )


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
    resume: bool = False,
    glmnet_cv_n_splits: Optional[int] = None,
    clone_id_kwargs: Optional[Dict] = None,
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
        Required for binary and multi-binary modes. Ignored for multiclass.
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
    n_jobs : Parallel workers for clustering and cluster assignment.
    verbose : Verbosity level.
    data_dir : Path to raw data directory. Required if cache is missing.
    cache_dir : Path to cache directory. None disables caching.
    gene_reference_path : Path to gene reference file (V-gene CDR sequences).
    output_suffix : Suffix appended to the mode directory name (e.g. "strict_pval"
        produces "multiclass__strict_pval"). Ignored when output_dir is set.
    training_context : Training context controlling data splits and output paths.
        "cv_single_model" (default) or "cv_ensemble".
    resume : If True, skip folds whose artifacts already exist on disk and
        reload their saved results. Incomplete folds are retrained normally.
    glmnet_cv_n_splits : Number of inner CV folds for GLM's StratifiedGroupKFold.
        Default None → 5 (matching original Mal-ID). Automatically capped at
        runtime if the data has fewer unique participants per class than requested
        (see cap_cv_splits_for_data). Use 2-3 for small datasets where some
        classes have fewer than 5 participants in the training split.
    clone_id_kwargs : Dict of clone_id parameters for the data loader
        (from get_clone_id_kwargs). None means all params unspecified —
        cached values accepted as-is.

    Returns
    -------
    Dict mapping pair/mode key → {"fold_results": List[Dict], "aggregated_by_model": Dict[str, Dict]}.
    - multiclass:   {"multiclass": {...}}
    - binary:       {"<disease>_vs_<reference>": {...}}
    - multi-binary: {"<d1>_vs_<ref>": {...}, "<d2>_vs_<ref>": {...}, ...}
    """
    if model_names is None:
        model_names = [BEST_MODEL_FOR_METAMODEL[gene_locus]]

    if sequence_identity_threshold is None:
        sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]

    if p_values is None:
        p_values = DEFAULT_P_VALUES

    if glmnet_cv_n_splits is None:
        glmnet_cv_n_splits = _GLMNET_CV_N_SPLITS

    # Initialize data loader
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=gene_locus,
        cache_dir=cache_dir,
        gene_reference_path=gene_reference_path,
        verbose=0,
        **(clone_id_kwargs or {}),
    )
    # Precompute clone IDs in parallel (no-op if all participants cached)
    if loader.cache_dir is not None:
        loader.precompute_clone_ids(n_jobs=n_jobs)

    # Resolve fold IDs: None = all folds found in metadata
    if fold_ids is None:
        fold_ids = sorted(
            loader.metadata[FOLD_COL]
            .dropna().unique().astype(int).tolist()
        )
        logger.info(f"  Auto-detected fold IDs from metadata: {fold_ids}")

    # Validate mode against available disease classes
    disease_classes = get_dataset_disease_classes(loader.metadata)
    reference_class = validate_mode_and_classes(
        classification_mode, disease_classes, reference_class, diseases=diseases
    )

    # Base output dir (parent of pair subdirs for binary/multi-binary)
    base_dir = output_dir or get_model_output_dir(
        "model2", dataset_name, classification_mode, gene_locus,
        training_context=training_context,
        output_suffix=output_suffix,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.monotonic()

    # run_params captures settings that affect training outcomes beyond model
    # hyperparams. Saved in predictions.pkl _meta and validated on resume.
    # gene_locus is included so a resume with a mismatched locus is caught even
    # when an explicit --output-dir bypasses the locus-segregated default path
    # (Model 2 resolves gene_locus into model_names/threshold, so it is not
    # otherwise recorded in the resume identity — unlike Model 1's model_params).
    run_params = {
        "classification_mode": classification_mode,
        "diseases": sorted(diseases) if diseases else None,
        "dataset_name": dataset_name,
        "reference_class": reference_class,
        "gene_locus": gene_locus,
    }

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
        resume=resume,
        run_params=run_params,
        glmnet_cv_n_splits=glmnet_cv_n_splits,
    )

    # Delete old summary/results files BEFORE training so stale files from a prior
    # run don't persist if this run fails partway through.
    delete_stale_summaries(base_dir)

    all_results = run_training_orchestration(
        base_dir=base_dir,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_fold_loop,
        loop_kwargs=loop_kwargs,
    )

    # ------------------------------------------------------------------
    # Write summary JSON, Markdown results, and per-pair results
    # ------------------------------------------------------------------

    # Dataset counts (participants and specimens per disease class)
    dataset_counts = get_metadata_class_counts(loader.metadata)
    metadata_filter_info = loader.metadata_filter_info

    # Summary JSON
    summary_path = base_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset_name": dataset_name,
                "training_context": training_context,
                # Uniform "training complete; ready for inference" marker (5.H).
                "training_complete": True,
                "classification_mode": classification_mode,
                "reference_class": reference_class,
                "diseases": diseases,
                "model_classes": get_model_classes(
                    classification_mode, disease_classes, diseases, reference_class,
                ),
                "gene_locus": gene_locus,
                "output_suffix": output_suffix,
                # Resolved clone_id clustering definition (Phase 6.E cross-dataset check).
                "clone_id_params": loader.clone_id_params,
                "fold_ids": fold_ids,
                "model_names": model_names,
                "p_values": p_values,
                "retrain_on_full_train": retrain_on_full_train,
                "resume": resume,
                "sequence_identity_threshold": sequence_identity_threshold,
                "n_jobs": n_jobs,
                "dataset_counts": dataset_counts,
                "metadata_filter_info": metadata_filter_info,
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

    # Results Markdown
    run_info: Dict = {
        "Dataset": dataset_name,
        "Training context": training_context,
        "Classification mode": classification_mode,
        "Gene locus": gene_locus,
        "Folds": ", ".join(str(f) for f in fold_ids),
        "Model variants": ", ".join(model_names),
        "P-value candidates": str(p_values),
        "Sequence identity threshold": sequence_identity_threshold,
        "Retrain GLM on A+B": str(retrain_on_full_train),
        "Resume": str(resume),
        "n_jobs": n_jobs,
        "Output suffix": output_suffix or "(none)",
        "Total participants": dataset_counts["total_participants"],
        "Total specimens": dataset_counts["total_specimens"],
        "Participants per class": ", ".join(
            f"{k}: {v}" for k, v in dataset_counts["participants_per_class"].items()
        ),
        "Specimens per class": ", ".join(
            f"{k}: {v}" for k, v in dataset_counts["specimens_per_class"].items()
        ),
    }
    if metadata_filter_info and metadata_filter_info["n_filtered_out"] > 0:
        run_info["Metadata filtering"] = (
            f"{metadata_filter_info['n_filtered_out']} participants excluded "
            f"(no raw data files); {metadata_filter_info['n_retained']} retained "
            f"out of {metadata_filter_info['n_original']} in metadata file"
        )
    if classification_mode != "multiclass" and reference_class:
        run_info["Reference class"] = reference_class
    if diseases:
        run_info["Diseases"] = ", ".join(diseases)

    md_content = generate_results_md(
        all_results=all_results,
        classification_mode=classification_mode,
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

    # Per-pair results (binary / multi-binary only)
    save_per_pair_results(
        base_dir=base_dir,
        all_results=all_results,
        classification_mode=classification_mode,
        timestamp=timestamp,
        model_label="Model 2",
        run_info=run_info,
        fold_ids=fold_ids,
        model_names=model_names,
        has_abstention=True,
        gene_locus=gene_locus,
        clone_id_params=loader.clone_id_params,
    )

    # Summary table
    all_eval_flat = [r for pair_data in all_results.values() for r in pair_data["fold_results"]]
    logger.info("\n--- Summary ---")
    for r in all_eval_flat:
        pair_str = (
            f"{r['disease']}_vs_{r['reference_class']} "
            if "disease" in r else ""
        )
        auroc_val = next((r.get(k) for k in ("auroc_binary", "auroc_ovo_weighted") if r.get(k) is not None), None)
        auroc_str = f"{auroc_val:.4f}" if auroc_val is not None else "N/A  "
        logloss_str = (
            f"{r['log_loss']:.4f}"
            if r.get("log_loss") is not None
            else "N/A  "
        )
        mcc_str = f"{r['mcc']:.4f}" if r.get("mcc") is not None else "N/A  "
        logger.info(
            f"  fold={r['fold_id']} {pair_str}{r['model_name']:20s} "
            f"AUROC={auroc_str} MCC={mcc_str} "
            f"LogLoss={logloss_str} "
            f"abstention={r['abstention_rate']:.1%}"
        )

    # Aggregated results
    logger.info("\n--- Aggregated Results ---")
    for pair_key, pair_data in all_results.items():
        for mn, agg in pair_data["aggregated_by_model"].items():
            logger.info(f"  {pair_key} / {mn}:")
            acc_global = agg.get("accuracy_global")
            acc_str = f"{acc_global:.4f}" if acc_global is not None else "N/A"
            mcc_agg = agg.get("mcc", {})
            mcc_mean = mcc_agg.get("mean") if isinstance(mcc_agg, dict) else None
            mcc_str2 = f"{mcc_mean:.4f}" if mcc_mean is not None else "N/A"
            if classification_mode == "multiclass":
                auroc_agg = agg.get("auroc_ovo_weighted", {})
                ll_agg = agg.get("log_loss", {})
                auroc_mean = auroc_agg.get("mean")
                ll_mean = ll_agg.get("mean")
                auroc_str_ovo = f"{auroc_mean:.4f}" if auroc_mean is not None else "N/A"
                logger.info(
                    f"    accuracy_global={acc_str} "
                    f"AUROC_OvO={auroc_str_ovo} MCC={mcc_str2}"
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
                    f"AUPRC_pooled={auprc_str2} MCC={mcc_str2}"
                )

    elapsed = time.monotonic() - t_start
    logger.info(f"train_all_folds completed in {elapsed:.1f}s")

    return all_results


def train_full_dataset(
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
    training_context: str = "train_all",
    resume: bool = False,
    glmnet_cv_n_splits: Optional[int] = None,
    clone_id_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Train Model 2 on the WHOLE dataset (train-all); no evaluation.

    The train-all counterpart of ``train_all_folds()``: no fold loop, no test/eval.
    Produces reusable per-pair artifacts (``clusters.joblib`` +
    ``<model>_p_value.joblib`` / ``_model_<suffix>.joblib`` / ``_results_<suffix>.json``
    or ``<model>_NO_VALID_CLUSTERS.txt``, plus ``meta.json``) and a no-metrics
    summary, to be scored later on a separate dataset (external eval).

    ``training_context`` must be a train-all context; use ``train_all_folds()`` for
    CV. Reuses ``run_training_orchestration`` for pair dispatch, calling
    ``_run_train_all`` per pair. The summary JSON deliberately carries the config
    keys that ``predict_model2`` reads when loading these artifacts later
    (``classification_mode``, ``retrain_on_full_train``, ``model_names``).
    """
    if training_context not in TRAIN_ALL_TRAINING_CONTEXTS:
        raise ValueError(
            f"train_full_dataset requires a train-all context "
            f"{TRAIN_ALL_TRAINING_CONTEXTS}, got {training_context!r}. "
            f"Use train_all_folds() for CV contexts."
        )

    if model_names is None:
        model_names = [BEST_MODEL_FOR_METAMODEL[gene_locus]]
    if sequence_identity_threshold is None:
        sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]
    if p_values is None:
        p_values = DEFAULT_P_VALUES
    if glmnet_cv_n_splits is None:
        glmnet_cv_n_splits = _GLMNET_CV_N_SPLITS

    t_start = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=gene_locus,
        cache_dir=cache_dir,
        gene_reference_path=gene_reference_path,
        verbose=0,
        **(clone_id_kwargs or {}),
    )
    if loader.cache_dir is not None:
        loader.precompute_clone_ids(n_jobs=n_jobs)

    # No fold auto-detection — train-all ignores CV_fold.
    disease_classes = get_dataset_disease_classes(loader.metadata)
    reference_class = validate_mode_and_classes(
        classification_mode, disease_classes, reference_class, diseases=diseases
    )

    base_dir = output_dir or get_model_output_dir(
        "model2", dataset_name, classification_mode, gene_locus,
        training_context=training_context, output_suffix=output_suffix,
    )

    # run_params captures settings that affect training outcomes beyond model
    # hyperparams. Saved in meta.json and validated on resume by
    # validate_train_all_meta. gene_locus is included so a resume with a
    # mismatched locus is caught even when an explicit --output-dir bypasses the
    # locus-segregated default path (Model 2 resolves gene_locus into
    # model_names/threshold, so it is not otherwise recorded in the resume
    # identity — unlike Model 1's model_params).
    run_params = {
        "classification_mode": classification_mode,
        "diseases": sorted(diseases) if diseases else None,
        "dataset_name": dataset_name,
        "reference_class": reference_class,
        "gene_locus": gene_locus,
    }

    loop_kwargs = dict(
        loader=loader,
        model_names=model_names,
        sequence_identity_threshold=sequence_identity_threshold,
        p_values=p_values,
        retrain_on_full_train=retrain_on_full_train,
        n_jobs=n_jobs,
        verbose=verbose,
        training_context=training_context,
        run_params=run_params,
        glmnet_cv_n_splits=glmnet_cv_n_splits,
        resume=resume,
    )

    delete_stale_summaries(base_dir)

    all_results = run_training_orchestration(
        base_dir=base_dir,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        fold_loop_fn=_run_train_all,
        loop_kwargs=loop_kwargs,
    )

    # --- Shared no-metrics outputs: summary JSON + RESULTS.md + per-pair ---
    # summary_extra carries the config keys predict_model2 reads when loading these
    # artifacts later (classification_mode/model_names/reference_class are already in
    # the shared envelope).
    write_train_all_outputs(
        base_dir=base_dir,
        all_results=all_results,
        loader=loader,
        timestamp=timestamp,
        dataset_name=dataset_name,
        training_context=training_context,
        classification_mode=classification_mode,
        reference_class=reference_class,
        diseases=diseases,
        disease_classes=disease_classes,
        gene_locus=gene_locus,
        output_suffix=output_suffix,
        model_names=model_names,
        model_label="Model 2",
        summary_extra={
            "p_values": p_values,
            "retrain_on_full_train": retrain_on_full_train,
            "sequence_identity_threshold": sequence_identity_threshold,
            "n_jobs": n_jobs,
        },
        run_info_extra={
            "Model variants": ", ".join(model_names),
            "P-value candidates": str(p_values),
            "Sequence identity threshold": sequence_identity_threshold,
            "Retrain GLM on A+B": str(retrain_on_full_train),
        },
        # Per-pair (binary/multi-binary) summaries don't get the shared envelope, so
        # carry the config keys predict_model2 reads: classification_mode (mode guard)
        # and retrain_on_full_train (selects the _full vs _split1 pipeline file), plus
        # identity — so a single pair can be reloaded on its own without the base-dir
        # summary.
        per_pair_summary_extra={
            "dataset_name": dataset_name,
            "classification_mode": classification_mode,
            "reference_class": reference_class,
            "gene_locus": gene_locus,
            "model_names": model_names,
            "retrain_on_full_train": retrain_on_full_train,
            "sequence_identity_threshold": sequence_identity_threshold,
            "p_values": p_values,
        },
    )

    elapsed = time.monotonic() - t_start
    logger.info(f"train_full_dataset completed in {elapsed:.1f}s")

    return all_results


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
    add_clone_id_args(parser)

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
            "CV contexts (cross-validate on this dataset): 'cv_single_model' (default) "
            "trains on ts1+ts2 (all non-test participants); 'cv_ensemble' is base-model "
            "training for the ensemble (excludes validation). "
            "Train-all contexts (train on the WHOLE dataset, no test fold, NO evaluation; "
            "artifacts scored later on a separate dataset): 'train_all' clusters on ts1 "
            "and selects the p-value on ts2 over all participants; 'train_all_ensemble' "
            "does the same but excludes the validation third (base model for a train-all "
            "ensemble). Train-all ignores --fold-ids and needs no CV_fold column."
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
            "Required for binary and multi-binary modes. "
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
            "binary: one disease name (optional for 2-class datasets — the non-reference "
            "class is auto-detected; required for N-class datasets to pick one disease). "
            "multi-binary: one or more disease names — trains only the specified "
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
            "Fold(s) to hold out as the test set (default: all folds found in "
            "metadata). For each fold listed, the model is trained from scratch "
            "on all other folds pooled together, then evaluated on that held-out "
            "fold. Example: --fold-ids 0 1 2"
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
        "--resume",
        action="store_true",
        default=False,
        help=(
            "If set, skip completed folds (all artifacts present) and reload their "
            "saved results. Incomplete folds are retrained from scratch. "
            "Saved metadata is validated against current parameters — a mismatch "
            "raises an error to prevent mixing results from different configurations."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Number of parallel workers for clustering (Phase 1) and cluster assignment "
            "during featurization (training grid search + test evaluation). "
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

    # --- Train-all context + --fold-ids is contradictory (fail fast) ---
    if args.training_context in TRAIN_ALL_TRAINING_CONTEXTS and args.fold_ids is not None:
        parser.error(
            f"--fold-ids is not valid with --training-context {args.training_context} "
            f"(train-all trains on the whole dataset; there are no folds). "
            f"Remove --fold-ids, or use a cv_* context for cross-validation."
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

    # --- Validate training parameter ranges ---
    validate_training_params(p_values=args.p_values)

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

    is_train_all = args.training_context in TRAIN_ALL_TRAINING_CONTEXTS

    # Fold IDs: pass through from CLI (None = auto-detect inside train_all_folds
    # from the loader's filtered metadata). Unused in train-all (validated above).
    fold_ids = args.fold_ids

    logger.info(f"Starting Model 2 training — {timestamp}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Training context:    {args.training_context}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Reference class:     {args.reference_class or '(not set)'}")
    logger.info(f"  Diseases filter:     {args.diseases or '(all)'}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(
        f"  Folds:               "
        f"{'(train-all: whole dataset, no folds)' if is_train_all else (fold_ids or '(all, auto-detect)')}"
    )
    logger.info(f"  Models:              {model_names}")
    logger.info(f"  P-values:            {args.p_values}")
    logger.info(f"  Seq identity thresh: {SEQUENCE_IDENTITY_THRESHOLDS[args.gene_locus]}")
    logger.info(f"  Retrain GLM on A+B:  {args.retrain_full}")
    logger.info(f"  Resume:              {args.resume}")
    logger.info(f"  n_jobs:              {args.n_jobs}")
    logger.info(f"  Base output dir:     {base_dir}")
    if args.output_suffix:
        logger.info(f"  Output suffix:       {args.output_suffix}")
    logger.info(f"  Data dir:            {args.data_dir or '(not provided, using cache)'}")
    logger.info(f"  Cache dir:           {cache_dir or '(caching disabled)'}")
    logger.info(f"  Metadata:            {args.metadata_path}")
    logger.info(f"  Gene reference:      {args.gene_reference_path or '(not provided)'}")

    # Dispatch: train-all context → train_full_dataset (whole dataset, no eval);
    # CV → train_all_folds (fold loop + evaluation).
    common_kwargs = dict(
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
        resume=args.resume,
        clone_id_kwargs=get_clone_id_kwargs(args),
    )
    if is_train_all:
        all_results = train_full_dataset(**common_kwargs)
        # Train-all produces no evaluation metrics — log what was trained per
        # pair/model so the run's console output summarizes the outcome (mirrors
        # the CV summary block and Model 1's train-all summary). Model 2 trains on
        # two subsets: train_smaller1 (ts1, clustering + Fisher) and train_smaller2
        # (ts2, p-value threshold selection); "abstained" means no cluster passed
        # the p-value filter so the model makes no prediction for that pair.
        logger.info("\n--- Train-all summary (no evaluation) ---")
        for pair_key, pair_data in all_results.items():
            for info in pair_data["fold_results"]:
                logger.info(
                    f"  {pair_key}: "
                    f"ts1={info['n_train_ts1_participants']} participants / "
                    f"{info['n_train_ts1_specimens']} specimens / "
                    f"{info['n_train_ts1_sequences']:,} sequences; "
                    f"ts2={info['n_train_ts2_participants']} participants / "
                    f"{info['n_train_ts2_specimens']} specimens / "
                    f"{info['n_train_ts2_sequences']:,} sequences; "
                    f"classes={info['classes']}"
                )
                for model_name, model_info in info["models"].items():
                    if model_info["abstained"]:
                        status = "ABSTAINED (no cluster passed the p-value filter)"
                    else:
                        status = f"best_p_value={model_info['best_p_value']}"
                    logger.info(f"    {model_name}: {status}")
    else:
        train_all_folds(fold_ids=fold_ids, **common_kwargs)

    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Clean up file handler to flush and release the log file
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
