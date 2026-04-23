"""Ensemble (metamodel) training for Mal-ID-Lite.

Trains a ridge-regularized logistic regression metamodel on the predictions of
three base models (Model 1: repertoire stats, Model 2: convergent clusters,
Model 3: sequence-level). The metamodel learns how to combine base model outputs
for the final disease classification.

Architecture
------------
For each outer CV fold (0, 1, 2):
  1. Load the cv_ensemble split: test / validation / train_smaller1 / train_smaller2
  2. Load pre-trained base model artifacts (trained on train_smaller)
  3. Get base model predictions on validation specimens
  4. Build metamodel feature matrix from those predictions
  5. Train ridge meta-learner (GlmnetLogitNetWrapper, alpha=0.0, MCC scoring)
  6. Get base model predictions on test specimens
  7. Predict with metamodel, evaluate ensemble + all base models

Artifacts per fold
------------------
  metamodel/fold_<id>_ridge_cv_metamodel.joblib   — fitted Pipeline
  metamodel/fold_<id>_metamodel_config.json        — feature columns, classes, config

Classification modes
--------------------
multiclass
    A single N-class ensemble. Default.

binary
    One ensemble for a single disease-vs-reference pair.
    Requires --diseases <disease> and --reference-class.

multi-binary
    One independent binary ensemble per disease vs. the reference class.
    Default (no --diseases): trains all N-1 non-reference diseases.
    With --diseases <d1> <d2> ...: trains only the specified subset.
    Requires --reference-class when data has more than 2 classes.

Usage
-----
    # Default: all 3 models, multiclass, all folds
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data

    # Only Models 1 and 3
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --models 1 3

    # Binary: single disease vs reference
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --classification-mode binary --diseases Covid19 \\
        --reference-class "Healthy/Background"

    # Multi-binary: one ensemble per disease vs reference
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --classification-mode multi-binary \\
        --reference-class "Healthy/Background"

    # With custom model suffixes (for different training variants)
    python malid_lite/training/train_ensemble.py \\
        --metadata-path cache/mal-id-orig-data/metadata.tsv \\
        --cache-dir cache/mal-id-orig-data \\
        --model3-suffix pct_0-1

Pre-requisites
--------------
Base models must be trained with --training-context cv_ensemble before running
this script. For binary/multi-binary ensemble, base models must also be trained
in binary mode for each disease pair. See each model's training script for details.
"""

import argparse
import dataclasses
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
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.dataloader.base import PreprocessingStage
from malid_lite.models.model1_repertoire import (
    V_GENE_COL,
    RepertoireClassifier,
)
from malid_lite.models.model2_convergent_clusters import (
    BEST_MODEL_FOR_METAMODEL,
    SEQUENCE_IDENTITY_THRESHOLDS,
    featurize,
    get_artifact_paths,
)
from malid_lite.training.training_utils import (
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    aggregate_fold_results,
    filter_to_binary_pair,
    get_dataset_disease_classes,
    get_dataset_fold_ids,
    get_ensemble_output_dir,
    get_model_output_dir,
    make_pair_name,
    validate_mode_and_classes,
)
from malid_lite.utils import multiclass_metrics
from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper

logger = logging.getLogger(__name__)

TRAINING_CONTEXT = "cv_ensemble"

# Display names for feature column prefixes, matching original Mal-ID config
MODEL_DISPLAY_NAMES = {
    1: "repertoire_stats",
    2: "convergent_cluster_model",
    3: "sequence_model",
}


# ============================================================================
# ModelPredictions dataclass
# ============================================================================

@dataclasses.dataclass
class ModelPredictions:
    """Standardized prediction output for ensemble consumption.

    All base models produce this format. Models 1 and 3 return empty abstention
    lists. Model 2 may abstain on specimens with zero cluster matches.

    Attributes
    ----------
    probabilities : DataFrame indexed by specimen_label, columns = disease classes.
        Shape (n_scored_specimens, n_classes).
    abstained_specimen_labels : Specimen labels with no prediction.
    abstained_specimen_diseases : Ground-truth disease labels of abstained specimens.
    """

    probabilities: pd.DataFrame
    abstained_specimen_labels: list
    abstained_specimen_diseases: list

    @property
    def n_scored(self) -> int:
        return len(self.probabilities)

    @property
    def n_abstained(self) -> int:
        return len(self.abstained_specimen_labels)


# ============================================================================
# Base model prediction functions
# ============================================================================

def predict_model1(
    model_dir: Path,
    fold_id: int,
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    model_name: str = "lasso_cv",
    disease_filter: Optional[Tuple[str, str]] = None,
) -> ModelPredictions:
    """Load Model 1 artifacts and predict on target specimens.

    Parameters
    ----------
    model_dir : Directory containing fold_<id>_<model_name>_model.pkl and v_genes.json.
    target_specimens : Set of specimen_labels to predict on.
    disease_filter : (disease, reference_class) for binary mode, or None.

    Returns
    -------
    ModelPredictions with probabilities indexed by specimen_label.
    """
    # --- Load artifacts ---
    model_path = model_dir / f"fold_{fold_id}_{model_name}_model.pkl"
    v_genes_path = model_dir / f"fold_{fold_id}_{model_name}_v_genes.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model 1 artifact not found: {model_path}. "
            f"Train Model 1 with --training-context cv_ensemble first."
        )
    if not v_genes_path.exists():
        raise FileNotFoundError(
            f"Model 1 V-gene list not found: {v_genes_path}. "
            f"Train Model 1 with --training-context cv_ensemble first."
        )

    model = RepertoireClassifier.load(model_path)
    with open(v_genes_path) as f:
        kept_v_genes = json.load(f)

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Filter to training V-genes
    seq = seq[seq[V_GENE_COL].isin(kept_v_genes)].copy()

    # --- Extract features and predict ---
    X = model.extract_features(
        sequences=seq,
        metadata=meta,
        train_vj_columns=model.train_vj_columns_,
    )
    y_proba = model.predict_proba(X)

    proba_df = pd.DataFrame(
        y_proba,
        index=X.index,  # specimen_label
        columns=model.classes_,
    )

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=[],
        abstained_specimen_diseases=[],
    )


def predict_model2(
    model_dir: Path,
    fold_id: int,
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    gene_locus: str = "TCR",
    model_name: Optional[str] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
) -> ModelPredictions:
    """Load Model 2 artifacts and predict on target specimens.

    Parameters
    ----------
    model_dir : Directory containing fold_<id>_clusters.joblib, etc.
    target_specimens : Set of specimen_labels to predict on.
    model_name : GLM variant name (default: BEST_MODEL_FOR_METAMODEL[gene_locus]).
    disease_filter : (disease, reference_class) for binary mode, or None.

    Returns
    -------
    ModelPredictions with probabilities for scored specimens; abstention info
    for specimens with zero cluster matches.
    """
    if model_name is None:
        model_name = BEST_MODEL_FOR_METAMODEL[gene_locus]

    # --- Load artifacts ---
    paths = get_artifact_paths(model_dir, fold_id, model_name, retrain_on_full_train=False)
    for key, path in paths.items():
        if key == "metrics":
            continue  # metrics file not needed for prediction
        if not path.exists():
            raise FileNotFoundError(
                f"Model 2 artifact not found: {path} (key={key}). "
                f"Train Model 2 with --training-context cv_ensemble first."
            )

    clusters_data = joblib.load(paths["clusters"])
    best_p_value = joblib.load(paths["p_value"])
    pipeline = joblib.load(paths["pipeline"])

    centroids_with_scores = clusters_data["centroids_with_scores"]
    disease_classes = clusters_data["disease_classes"]
    sequence_identity_threshold = SEQUENCE_IDENTITY_THRESHOLDS[gene_locus]

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Model 2 featurize() requires a disease column on sequences
    if DISEASE_COL not in seq.columns:
        disease_map = meta.set_index(SPECIMEN_COL)[DISEASE_COL]
        seq[DISEASE_COL] = seq[SPECIMEN_COL].map(disease_map)

    # --- Featurize and predict ---
    fd = featurize(
        seq,
        p_value_threshold=best_p_value,
        centroids_with_scores=centroids_with_scores,
        sequence_identity_threshold=sequence_identity_threshold,
        disease_classes=disease_classes,
        disease_col=DISEASE_COL,
    )

    if fd.n_scored == 0:
        # All specimens abstained
        return ModelPredictions(
            probabilities=pd.DataFrame(columns=disease_classes),
            abstained_specimen_labels=list(fd.abstained_sample_names),
            abstained_specimen_diseases=list(fd.abstained_sample_y),
        )

    y_proba = pipeline.predict_proba(fd.X)

    proba_df = pd.DataFrame(
        y_proba,
        index=fd.sample_names,
        columns=pipeline.classes_,
    )

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=list(fd.abstained_sample_names),
        abstained_specimen_diseases=list(fd.abstained_sample_y),
    )


def predict_model3(
    model_dir: Path,
    fold_id: int,
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    embedding_dir: Path,
    gene_locus: str = "TCR",
    disease_filter: Optional[Tuple[str, str]] = None,
) -> ModelPredictions:
    """Load Model 3 artifacts and predict on target specimens.

    Parameters
    ----------
    model_dir : Directory containing fold_<id>_stage1.pkl and fold_<id>_stage2.pkl.
    target_specimens : Set of specimen_labels to predict on.
    embedding_dir : Directory with pre-computed ESM-2 embeddings.
    disease_filter : (disease, reference_class) for binary mode, or None.

    Returns
    -------
    ModelPredictions with probabilities indexed by specimen_label. Never abstains.
    """
    from malid_lite.models.model3_sequence_level import (
        DISEASE_COL as M3_DISEASE_COL,
        PARTICIPANT_COL as M3_PARTICIPANT_COL,
        SPECIMEN_COL as M3_SPECIMEN_COL,
        make_tcr_model,
        make_bcr_model,
    )
    from malid_lite.training.train_model3 import load_precomputed_embeddings

    # --- Load artifacts ---
    stage1_path = model_dir / f"fold_{fold_id}_stage1.pkl"
    stage2_path = model_dir / f"fold_{fold_id}_stage2.pkl"
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Model 3 Stage 1 artifact not found: {stage1_path}. "
            f"Train Model 3 with --training-context cv_ensemble first."
        )
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Model 3 Stage 2 artifact not found: {stage2_path}. "
            f"Train Model 3 with --training-context cv_ensemble first."
        )

    # Create model with paper-best config, then load artifacts
    make_model = make_tcr_model if gene_locus == "TCR" else make_bcr_model
    model = make_model()

    with open(stage1_path, "rb") as f:
        s1_data = pickle.load(f)
    model.load_stage1_artifacts(s1_data)

    with open(stage2_path, "rb") as f:
        s2_data = pickle.load(f)
    model.load_stage2_artifacts(s2_data)

    # --- Filter to target specimens ---
    seq = sequences_df[sequences_df[M3_SPECIMEN_COL].isin(target_specimens)].copy()
    meta = metadata_df[metadata_df[M3_SPECIMEN_COL].isin(target_specimens)].copy()
    if disease_filter:
        seq, meta = filter_to_binary_pair(seq, meta, disease_filter[0], disease_filter[1])

    # Ensure disease column is present (needed for ground-truth alignment)
    if M3_DISEASE_COL not in seq.columns:
        disease_map = meta.set_index(M3_SPECIMEN_COL)[DISEASE_COL]
        seq[M3_DISEASE_COL] = seq[M3_SPECIMEN_COL].map(disease_map)

    # --- Load embeddings and predict ---
    seq = seq.reset_index(drop=True)
    embeddings = load_precomputed_embeddings(seq, embedding_dir)
    proba_df = model.predict_proba(seq, embeddings)

    return ModelPredictions(
        probabilities=proba_df,
        abstained_specimen_labels=[],
        abstained_specimen_diseases=[],
    )


# ============================================================================
# Feature matrix construction
# ============================================================================

def build_feature_matrix(
    predictions_by_model: Dict[int, ModelPredictions],
    gene_locus: str,
    reference_class: Optional[str] = None,
) -> Tuple[pd.DataFrame, list, list]:
    """Build the metamodel feature matrix from base model predictions.

    Steps:
    1. For binary models (2 columns), keep only the non-reference class column.
    2. Rename columns: {locus}:{model_display_name}:{class_name}.
    3. Harmonize abstentions: keep only specimens scored by ALL models.
    4. Concatenate horizontally, sort columns for determinism.

    Parameters
    ----------
    predictions_by_model : {model_number: ModelPredictions}.
    gene_locus : "TCR" or "BCR".
    reference_class : For binary mode, the reference/negative class.

    Returns
    -------
    (X, abstained_labels, abstained_diseases)
        X : DataFrame (n_common_specimens, n_features), index=specimen_label.
        abstained_labels : Specimen labels excluded (union of all abstentions).
        abstained_diseases : Ground-truth diseases of excluded specimens.
    """
    # --- Step 1: Binary column selection + column renaming ---
    renamed_dfs = {}
    for model_num, preds in sorted(predictions_by_model.items()):
        proba = preds.probabilities.copy()
        display_name = MODEL_DISPLAY_NAMES[model_num]

        # For binary classifiers, keep only the non-reference class column
        if proba.shape[1] == 2 and reference_class is not None:
            non_ref_cols = [c for c in proba.columns if str(c) != str(reference_class)]
            assert len(non_ref_cols) == 1, (
                f"Model {model_num}: expected 1 non-reference class, "
                f"got {non_ref_cols} (reference={reference_class})"
            )
            proba = proba[non_ref_cols]

        # Rename columns: {locus}:{model_name}:{class_name}
        proba.columns = [
            f"{gene_locus}:{display_name}:{cls}" for cls in proba.columns
        ]
        renamed_dfs[model_num] = proba

    # --- Step 2: Harmonize abstentions ---
    # Common scored specimens = intersection of all models' scored sets
    scored_sets = [set(df.index) for df in renamed_dfs.values()]
    common_scored = scored_sets[0]
    for s in scored_sets[1:]:
        common_scored &= s

    # Collect all abstention info
    all_abstained_labels = []
    all_abstained_diseases = []
    for preds in predictions_by_model.values():
        all_abstained_labels.extend(preds.abstained_specimen_labels)
        all_abstained_diseases.extend(preds.abstained_specimen_diseases)

    # Specimens scored by some models but not all are also effectively abstained
    all_scored = set()
    for s in scored_sets:
        all_scored |= s
    partially_scored = all_scored - common_scored
    # We don't have disease labels for partially-scored specimens here;
    # the caller will handle them via metadata lookup if needed.

    if partially_scored:
        logger.info(
            f"  {len(partially_scored)} specimens scored by some models but not all "
            f"— excluded from ensemble"
        )

    # --- Step 3: Filter to common set and concatenate ---
    common_sorted = sorted(common_scored)
    filtered_dfs = [df.loc[common_sorted] for df in renamed_dfs.values()]
    X = pd.concat(filtered_dfs, axis=1)

    # Sort columns for deterministic ordering
    X = X[sorted(X.columns)]

    return X, all_abstained_labels, all_abstained_diseases


# ============================================================================
# Meta-learner training
# ============================================================================

def train_metamodel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> Pipeline:
    """Train the ridge meta-learner on validation predictions.

    Pipeline: StandardScaler -> GlmnetLogitNetWrapper(alpha=0.0, MCC scoring).
    Internal CV: 5-fold StratifiedGroupKFold, grouped by participant.

    Parameters
    ----------
    X_train : Feature matrix (n_validation_specimens, n_features).
    y_train : Disease labels, aligned with X_train.
    groups_train : Participant labels for group-aware CV, aligned with X_train.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    import glmnet.scorer

    # TODO: write a custom MCC scorer with NaN guard (like deviance_scorer/rocauc_scorer
    # in glmnet_wrapper.py) to handle degenerate lambdas explicitly. Currently uses
    # glmnet's built-in make_scorer, which goes through predict() → argmax(predict_proba()).
    # If predict_proba returns NaN, argmax silently picks a wrong class instead of failing.
    # Low risk for the metamodel's small dense feature matrix, and matches original Mal-ID.
    mcc_scorer = glmnet.scorer.make_scorer(matthews_corrcoef)
    cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GlmnetLogitNetWrapper(
            alpha=0.0,
            scoring=mcc_scorer,
            n_lambda=100,
            standardize=False,
            random_state=0,
            class_weight="balanced",
            internal_cv=cv_strategy,
            require_cv_group_labels=True,
            use_lambda_1se=False,
        )),
    ])

    pipeline.fit(
        X_train.values,
        y_train.values,
        classifier__groups=groups_train.values,
    )

    return pipeline


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    fold_id: int,
    model_label: str,
    n_scored: int,
    n_abstained: int,
    reference_class: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Compute evaluation metrics for one model on one fold.

    Shared by both the ensemble and individual base model evaluations.
    Accuracy includes abstention penalty: n_correct / (n_scored + n_abstained).

    Returns
    -------
    (metrics_dict, raw_preds_dict)
    """
    n_total = n_scored + n_abstained
    n_correct = int(accuracy_score(y_true, y_pred, normalize=False))

    results = {
        "fold_id": fold_id,
        "model_name": model_label,
        "n_scored": n_scored,
        "n_abstained": n_abstained,
        "abstention_rate": n_abstained / n_total if n_total > 0 else 0.0,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }

    # Multiclass metrics (3+ classes)
    # Matches original crosseval defaults: weighted + macro OvO for both AUROC and AUPRC.
    # Uses our multiclass_metrics module (verbatim copy of original Maxim-multiclass-metrics-repo).
    if len(classes) >= 3:
        for avg_method in ["weighted", "macro"]:
            key_auroc = f"auroc_ovo_{avg_method}"
            key_auprc = f"auprc_ovo_{avg_method}"
            try:
                results[key_auroc] = float(multiclass_metrics.roc_auc_score(
                    y_true, y_proba,
                    average=avg_method, multi_class="ovo", labels=classes,
                ))
            except ValueError as e:
                logger.warning(f"  {key_auroc} failed for {model_label}: {e}")
                results[key_auroc] = None

            try:
                results[key_auprc] = float(multiclass_metrics.auprc(
                    y_true, y_proba,
                    average=avg_method, multi_class="ovo", labels=classes,
                ))
            except ValueError as e:
                logger.warning(f"  {key_auprc} failed for {model_label}: {e}")
                results[key_auprc] = None

        # Per-class AUROC OvR (Lite addition, not in original crosseval defaults)
        auroc_ovr_per_class = {}
        try:
            per_class_scores = multiclass_metrics.roc_auc_score(
                y_true, y_proba,
                average=None, multi_class="ovr", labels=classes,
            )
            for cls, score in zip(classes, per_class_scores):
                auroc_ovr_per_class[str(cls)] = float(score)
        except ValueError as e:
            logger.warning(f"  Per-class AUROC OvR failed for {model_label}: {e}")
            for cls in classes:
                auroc_ovr_per_class[str(cls)] = None
        results["auroc_ovr_per_class"] = auroc_ovr_per_class
    else:
        results["auroc_ovo_weighted"] = None
        results["auroc_ovo_macro"] = None
        results["auprc_ovo_weighted"] = None
        results["auprc_ovo_macro"] = None
        results["auroc_ovr_per_class"] = None

    # Log loss (normalize for Model 3's OvR probabilities that don't sum to 1)
    from sklearn.metrics import log_loss as sklearn_log_loss
    if len(classes) >= 3:
        row_sums = y_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_proba_for_loss = y_proba / row_sums
    else:
        y_proba_for_loss = y_proba
    try:
        results["log_loss"] = float(sklearn_log_loss(y_true, y_proba_for_loss, labels=classes))
    except ValueError as e:
        logger.warning(f"  Log loss failed for {model_label}: {e}")
        results["log_loss"] = None

    # Binary metrics: AUROC and AUPRC with disease as positive class
    if len(classes) == 2 and reference_class is not None:
        str_classes = [str(c) for c in classes]
        disease_class = next(c for c in str_classes if c != str(reference_class))
        disease_idx = str_classes.index(disease_class)

        from sklearn.metrics import roc_auc_score, average_precision_score
        y_true_binary = (np.array([str(c) for c in y_true]) == disease_class).astype(int)
        y_score = y_proba[:, disease_idx]

        try:
            results["auroc_binary"] = float(roc_auc_score(y_true_binary, y_score))
        except ValueError as e:
            logger.warning(f"  Binary AUROC failed for {model_label}: {e}")
            results["auroc_binary"] = None
        try:
            results["auprc_binary"] = float(average_precision_score(y_true_binary, y_score))
        except ValueError as e:
            logger.warning(f"  Binary AUPRC failed for {model_label}: {e}")
            results["auprc_binary"] = None

    # MCC on predicted labels
    try:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError as e:
        logger.warning(f"  MCC failed for {model_label}: {e}")
        results["mcc"] = None

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_labels"] = [str(c) for c in classes]
    except ValueError as e:
        logger.warning(f"  Confusion matrix failed for {model_label}: {e}")
        results["confusion_matrix"] = None
        results["confusion_matrix_labels"] = None

    raw_preds = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
    }

    return results, raw_preds


# ============================================================================
# Fold loop: the core orchestration
# ============================================================================

def run_ensemble_fold(
    loader: MalIDPublishedDataLoader,
    fold_id: int,
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]] = None,
    reference_class: Optional[str] = None,
    verbose: int = 1,
) -> Dict:
    """Run the full ensemble pipeline for one fold.

    Returns a dict with keys: fold_id, ensemble_metrics, ensemble_raw_preds,
    base_model_metrics, base_model_raw_preds, pipeline, metamodel_config,
    predictions_rows.
    """
    t_fold_start = time.monotonic()
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_id}")
    logger.info(f"{'='*70}")

    # --- Step 1: Load split participants ---
    validation_participants = set(
        loader.get_split_participants(fold_id, TRAINING_CONTEXT, ["validation"])
    )
    # Model 1 trains on ts1+ts2 (all of train_smaller); Models 2, 3 split internally
    logger.info(f"  Validation participants: {len(validation_participants)}")

    # --- Step 2: Load fold data ---
    logger.info("  Loading train fold data...")
    t0 = time.monotonic()
    train_seq, train_meta = loader.get_fold_data(
        fold_id=fold_id, fold_label="train",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
    )
    logger.info(
        f"  Train fold: {len(train_meta)} specimens, "
        f"{len(train_seq):,} sequences [{time.monotonic()-t0:.1f}s]"
    )

    logger.info("  Loading test fold data...")
    t0 = time.monotonic()
    test_seq, test_meta = loader.get_fold_data(
        fold_id=fold_id, fold_label="test",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
    )
    logger.info(
        f"  Test fold: {len(test_meta)} specimens, "
        f"{len(test_seq):,} sequences [{time.monotonic()-t0:.1f}s]"
    )

    # Identify validation and test specimen sets.
    # In binary mode, restrict to specimens from the two target diseases only —
    # specimens from other diseases are not "abstained", they are simply outside
    # the classification scope.
    validation_specimens = set(
        train_meta[train_meta[PARTICIPANT_COL].isin(validation_participants)][SPECIMEN_COL]
    )
    test_specimens = set(test_meta[SPECIMEN_COL])

    if disease_filter:
        disease, ref = disease_filter
        target_diseases = {disease, ref}
        validation_specimens = set(
            train_meta[
                train_meta[PARTICIPANT_COL].isin(validation_participants)
                & train_meta[DISEASE_COL].isin(target_diseases)
            ][SPECIMEN_COL]
        )
        test_specimens = set(
            test_meta[test_meta[DISEASE_COL].isin(target_diseases)][SPECIMEN_COL]
        )

    logger.info(f"  Validation specimens: {len(validation_specimens)}")
    logger.info(f"  Test specimens: {len(test_specimens)}")

    # --- Step 3: Get base model predictions on validation ---
    logger.info("\n  Collecting validation predictions...")
    val_predictions: Dict[int, ModelPredictions] = {}
    for model_num in model_nums:
        t0 = time.monotonic()
        preds = _get_model_predictions(
            model_num, model_dirs[model_num], fold_id,
            train_seq, train_meta, validation_specimens,
            gene_locus, embedding_dir, disease_filter,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            f"    Model {model_num}: {preds.n_scored} scored, "
            f"{preds.n_abstained} abstained [{elapsed:.1f}s]"
        )
        val_predictions[model_num] = preds

    # --- Step 4: Build validation feature matrix ---
    X_val, val_abstained_labels, val_abstained_diseases = build_feature_matrix(
        val_predictions, gene_locus, reference_class,
    )
    logger.info(
        f"  Validation feature matrix: {X_val.shape[0]} specimens x {X_val.shape[1]} features"
    )
    if val_abstained_labels:
        logger.info(f"  Validation abstentions: {len(val_abstained_labels)}")

    if X_val.shape[0] == 0:
        raise ValueError(
            f"Fold {fold_id}: all validation specimens were abstained by at least one "
            f"base model — 0 specimens with complete predictions. Cannot train metamodel."
        )

    # Get validation labels and groups, aligned to the feature matrix index
    val_meta_aligned = train_meta.set_index(SPECIMEN_COL).loc[X_val.index]
    y_val = val_meta_aligned[DISEASE_COL]
    groups_val = val_meta_aligned[PARTICIPANT_COL]

    # --- Step 5: Train metamodel ---
    logger.info("  Training metamodel...")
    t0 = time.monotonic()
    pipeline = train_metamodel(X_val, y_val, groups_val)
    train_time = time.monotonic() - t0
    logger.info(f"  Metamodel training done [{train_time:.1f}s]")

    # Log selected lambda
    clf = pipeline.named_steps["classifier"]
    logger.info(f"  Selected lambda: {clf.lambda_best_:.6f}")

    # --- Step 6: Get base model predictions on test ---
    logger.info("\n  Collecting test predictions...")
    test_predictions: Dict[int, ModelPredictions] = {}
    for model_num in model_nums:
        t0 = time.monotonic()
        preds = _get_model_predictions(
            model_num, model_dirs[model_num], fold_id,
            test_seq, test_meta, test_specimens,
            gene_locus, embedding_dir, disease_filter,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            f"    Model {model_num}: {preds.n_scored} scored, "
            f"{preds.n_abstained} abstained [{elapsed:.1f}s]"
        )
        test_predictions[model_num] = preds

    # --- Step 7: Build test feature matrix (same column order as validation) ---
    X_test, test_abstained_labels, test_abstained_diseases = build_feature_matrix(
        test_predictions, gene_locus, reference_class,
    )

    # Enforce same column order as validation
    missing_cols = set(X_val.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_val.columns)
    if missing_cols:
        raise ValueError(
            f"Test feature matrix is missing columns present in validation: {missing_cols}. "
            f"This indicates a class mismatch between validation and test."
        )
    if extra_cols:
        logger.warning(
            f"Test has {len(extra_cols)} extra columns not in validation — dropping: {extra_cols}"
        )
    X_test = X_test[X_val.columns]
    assert list(X_test.columns) == list(X_val.columns), (
        f"Column mismatch after reindexing: "
        f"X_test has {list(X_test.columns)[:5]}... vs X_val {list(X_val.columns)[:5]}..."
    )

    logger.info(
        f"  Test feature matrix: {X_test.shape[0]} specimens x {X_test.shape[1]} features"
    )
    n_test_abstained = len(test_specimens) - X_test.shape[0]
    if n_test_abstained > 0:
        logger.info(f"  Test abstentions: {n_test_abstained}")

    if X_test.shape[0] == 0:
        raise ValueError(
            f"Fold {fold_id}: all test specimens were abstained by at least one "
            f"base model — 0 specimens with complete predictions. Cannot evaluate."
        )

    # --- Step 8: Predict with metamodel ---
    y_pred = pipeline.predict(X_test.values)
    y_proba = pipeline.predict_proba(X_test.values)
    classes = pipeline.classes_

    # Align ground-truth labels to the test feature matrix index
    test_meta_aligned = test_meta.set_index(SPECIMEN_COL).loc[X_test.index]
    y_true = test_meta_aligned[DISEASE_COL].values

    # --- Step 9: Evaluate ensemble ---
    ensemble_metrics, ensemble_raw_preds = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        fold_id=fold_id,
        model_label="ensemble",
        n_scored=X_test.shape[0],
        n_abstained=n_test_abstained,
        reference_class=reference_class,
    )
    logger.info(
        f"\n  Ensemble: accuracy={ensemble_metrics['accuracy']:.4f}, "
        f"MCC={ensemble_metrics.get('mcc', 'N/A')}"
    )

    # --- Step 10: Evaluate each base model on same test specimens ---
    # All models are evaluated on the same common specimen set (intersection of all models'
    # scored specimens). n_abstained is the ensemble-level count (specimens any model
    # abstained on), applied equally to all models so accuracy is directly comparable.
    # This means Models 1/3 (which never abstain) get penalized for Model 2's abstentions.
    # This is intentional — matches original Mal-ID's apples-to-apples comparison design.
    # Standalone base model performance (without this penalty) is reported by each model's
    # own training script.
    base_model_metrics = {}
    base_model_raw_preds = {}
    for model_num in model_nums:
        preds = test_predictions[model_num]

        # Filter base model probabilities to the common scored set
        common_specimens = X_test.index
        proba_common = preds.probabilities.loc[
            preds.probabilities.index.isin(common_specimens)
        ].loc[common_specimens]  # enforce same order

        bm_classes = np.array(sorted(preds.probabilities.columns))
        bm_proba = proba_common[bm_classes].values
        bm_y_true = test_meta_aligned.loc[common_specimens, DISEASE_COL].values
        bm_y_pred = bm_classes[np.argmax(bm_proba, axis=1)]

        bm_metrics, bm_raw = evaluate_predictions(
            y_true=bm_y_true,
            y_pred=bm_y_pred,
            y_proba=bm_proba,
            classes=bm_classes,
            fold_id=fold_id,
            model_label=f"model{model_num}",
            n_scored=len(common_specimens),
            n_abstained=n_test_abstained,
            reference_class=reference_class,
        )
        base_model_metrics[model_num] = bm_metrics
        base_model_raw_preds[model_num] = bm_raw

        logger.info(
            f"  Model {model_num}: accuracy={bm_metrics['accuracy']:.4f}, "
            f"MCC={bm_metrics.get('mcc', 'N/A')}"
        )

    # --- Build per-specimen prediction rows for CSV ---
    predictions_rows = []
    for i, specimen in enumerate(X_test.index):
        row = {
            "fold_id": fold_id,
            "specimen_label": specimen,
            "participant_label": test_meta_aligned.loc[specimen, PARTICIPANT_COL],
            "true_disease": y_true[i],
            "ensemble_predicted": y_pred[i],
        }
        for j, cls in enumerate(classes):
            row[f"ensemble_P({cls})"] = float(y_proba[i, j])
        predictions_rows.append(row)

    # --- Metamodel config for saving ---
    metamodel_config = {
        "feature_columns": list(X_val.columns),
        "classes": [str(c) for c in classes],
        "gene_locus": gene_locus,
        "models_included": model_nums,
        "n_features": X_val.shape[1],
        "n_validation_specimens": X_val.shape[0],
        "n_test_specimens": X_test.shape[0],
        "n_test_abstained": n_test_abstained,
        "lambda_best": float(clf.lambda_best_),
    }

    elapsed = time.monotonic() - t_fold_start
    logger.info(f"\n  Fold {fold_id} complete [{elapsed:.1f}s]")

    return {
        "fold_id": fold_id,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_raw_preds": ensemble_raw_preds,
        "base_model_metrics": base_model_metrics,
        "base_model_raw_preds": base_model_raw_preds,
        "pipeline": pipeline,
        "metamodel_config": metamodel_config,
        "predictions_rows": predictions_rows,
    }


def _get_model_predictions(
    model_num: int,
    model_dir: Path,
    fold_id: int,
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_specimens: set,
    gene_locus: str,
    embedding_dir: Optional[Path],
    disease_filter: Optional[Tuple[str, str]],
) -> ModelPredictions:
    """Dispatch to the appropriate model's prediction function."""
    if model_num == 1:
        return predict_model1(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, disease_filter=disease_filter,
        )
    elif model_num == 2:
        return predict_model2(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, gene_locus=gene_locus,
            disease_filter=disease_filter,
        )
    elif model_num == 3:
        if embedding_dir is None:
            raise ValueError(
                "Model 3 requires --model3-embedding-dir for loading pre-computed embeddings."
            )
        return predict_model3(
            model_dir, fold_id, sequences_df, metadata_df,
            target_specimens, embedding_dir=embedding_dir,
            gene_locus=gene_locus, disease_filter=disease_filter,
        )
    else:
        raise ValueError(f"Unknown model number: {model_num}")


# ============================================================================
# Artifact saving
# ============================================================================

def save_fold_artifacts(
    output_dir: Path,
    fold_result: Dict,
):
    """Save metamodel pipeline + config for one fold."""
    metamodel_dir = output_dir / "metamodel"
    metamodel_dir.mkdir(parents=True, exist_ok=True)

    fold_id = fold_result["fold_id"]

    # Save fitted pipeline
    pipeline_path = metamodel_dir / f"fold_{fold_id}_ridge_cv_metamodel.joblib"
    joblib.dump(fold_result["pipeline"], pipeline_path)

    # Save config
    config_path = metamodel_dir / f"fold_{fold_id}_metamodel_config.json"
    with open(config_path, "w") as f:
        json.dump(fold_result["metamodel_config"], f, indent=2)

    logger.info(f"  Saved: {pipeline_path.name}, {config_path.name}")


# ============================================================================
# Main training orchestrator
# ============================================================================

def train_ensemble(
    loader: MalIDPublishedDataLoader,
    fold_ids: List[int],
    model_nums: List[int],
    model_dirs: Dict[int, Path],
    gene_locus: str,
    output_dir: Path,
    embedding_dir: Optional[Path] = None,
    disease_filter: Optional[Tuple[str, str]] = None,
    reference_class: Optional[str] = None,
    run_config: Optional[Dict] = None,
    verbose: int = 1,
) -> Tuple[List[Dict], Dict]:
    """Train the ensemble across all folds.

    Returns
    -------
    (all_fold_results, aggregated_metrics)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration (base model paths, args, and settings)
    if run_config is not None:
        run_config_path = output_dir / "run_config.json"
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        logger.info(f"Saved run config: {run_config_path}")

    all_fold_results = []
    all_ensemble_metrics = []
    all_ensemble_raw_preds = []
    all_predictions_rows = []

    for fold_id in fold_ids:
        fold_result = run_ensemble_fold(
            loader=loader,
            fold_id=fold_id,
            model_nums=model_nums,
            model_dirs=model_dirs,
            gene_locus=gene_locus,
            embedding_dir=embedding_dir,
            disease_filter=disease_filter,
            reference_class=reference_class,
            verbose=verbose,
        )

        save_fold_artifacts(output_dir, fold_result)

        all_fold_results.append(fold_result)
        all_ensemble_metrics.append(fold_result["ensemble_metrics"])
        all_ensemble_raw_preds.append(fold_result["ensemble_raw_preds"])
        all_predictions_rows.extend(fold_result["predictions_rows"])

    # --- Save predictions CSV ---
    predictions_df = pd.DataFrame(all_predictions_rows)
    predictions_path = output_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"\nSaved predictions: {predictions_path}")

    # --- Aggregate ensemble metrics across folds ---
    aggregated = aggregate_fold_results(
        all_ensemble_metrics,
        all_ensemble_raw_preds,
        disease_filter=disease_filter,
    )

    # --- Aggregate base model metrics ---
    base_model_aggregated = {}
    for model_num in model_nums:
        bm_metrics = [fr["base_model_metrics"][model_num] for fr in all_fold_results]
        bm_raw = [fr["base_model_raw_preds"][model_num] for fr in all_fold_results]
        base_model_aggregated[model_num] = aggregate_fold_results(
            bm_metrics, bm_raw, disease_filter=disease_filter,
        )

    # --- Save summary JSON ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "models_included": model_nums,
        "gene_locus": gene_locus,
        "fold_ids": fold_ids,
        "disease_filter": list(disease_filter) if disease_filter else None,
        "ensemble": aggregated,
        "base_models": {
            f"model{num}": agg for num, agg in base_model_aggregated.items()
        },
    }
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary: {summary_path}")

    # --- Generate results MD ---
    md_content = _generate_ensemble_results_md(
        run_config=run_config,
        ensemble_agg=aggregated,
        base_model_agg=base_model_aggregated,
        model_nums=model_nums,
        all_fold_results=all_fold_results,
        timestamp=timestamp,
    )
    md_path = output_dir / f"RESULTS_{timestamp}.md"
    md_path.write_text(md_content)
    logger.info(f"Saved results MD: {md_path}")

    # --- Print comparison table ---
    _log_comparison_table(aggregated, base_model_aggregated, model_nums)

    return all_fold_results, summary


def _generate_ensemble_results_md(
    run_config: Optional[Dict],
    ensemble_agg: Dict,
    base_model_agg: Dict[int, Dict],
    model_nums: List[int],
    all_fold_results: List[Dict],
    timestamp: str,
) -> str:
    """Generate a Markdown summary of ensemble training results."""

    def _fv(val, fmt=".4f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    def _ms(agg, key):
        d = agg.get(key, {})
        if not d or not isinstance(d, dict):
            return "N/A"
        return f"{_fv(d.get('mean'))} +/- {_fv(d.get('std'))}"

    lines: List[str] = []
    lines += ["# Ensemble Training Results", ""]
    lines.append(f"**Timestamp**: {timestamp}")

    if run_config:
        lines += ["", "## Run Configuration", ""]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in run_config.items():
            if k == "metamodel_config" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| metamodel.{mk} | {mv} |")
            elif k == "base_model_paths" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| base_model_path.{mk} | `{mv}` |")
            elif k == "base_model_suffixes" and isinstance(v, dict):
                for mk, mv in v.items():
                    lines.append(f"| base_model_suffix.{mk} | {mv or '(none)'} |")
            else:
                lines.append(f"| {k} | {v} |")

    lines += ["", "---", ""]

    # --- Comparison table (binary vs multiclass columns) ---
    is_binary = "auroc_pooled" in ensemble_agg
    lines += ["## Model Comparison", ""]
    if is_binary:
        lines.append("| Model | Accuracy (global) | AUROC (pooled) | AUPRC (pooled) | MCC |")
        lines.append("|-------|-------------------|----------------|----------------|-----|")
    else:
        lines.append("| Model | Accuracy (global) | AUROC OvO weighted | AUROC OvO macro | MCC |")
        lines.append("|-------|-------------------|--------------------|-----------------|-----|")

    for label, agg in [("**Ensemble**", ensemble_agg)] + [
        (f"Model {num}", base_model_agg[num]) for num in model_nums
    ]:
        acc = _fv(agg.get("accuracy_global"))
        mcc_d = agg.get("mcc", {})
        mcc = _fv(mcc_d.get("mean")) if isinstance(mcc_d, dict) else "N/A"
        if is_binary:
            auroc = _fv(agg.get("auroc_pooled"))
            auprc = _fv(agg.get("auprc_pooled"))
            lines.append(f"| {label} | {acc} | {auroc} | {auprc} | {mcc} |")
        else:
            auroc_w = _ms(agg, "auroc_ovo_weighted")
            auroc_m = _ms(agg, "auroc_ovo_macro")
            lines.append(f"| {label} | {acc} | {auroc_w} | {auroc_m} | {mcc} |")
    lines += [""]

    # --- Per-fold ensemble results ---
    auroc_col = "AUROC" if is_binary else "AUROC OvO weighted"
    auroc_key = "auroc_binary" if is_binary else "auroc_ovo_weighted"
    lines += ["## Per-Fold Ensemble Results", ""]
    lines.append(f"| Fold | Accuracy | {auroc_col} | MCC | N scored | N abstained |")
    lines.append("|------|----------|" + "-" * (len(auroc_col) + 2) + "|-----|----------|-------------|")
    for fr in all_fold_results:
        em = fr["ensemble_metrics"]
        lines.append(
            f"| {em['fold_id']} | {_fv(em.get('accuracy'))} | "
            f"{_fv(em.get(auroc_key))} | "
            f"{_fv(em.get('mcc'))} | "
            f"{em.get('n_scored', 'N/A')} | {em.get('n_abstained', 0)} |"
        )
    lines += [""]

    # --- Per-fold base model results ---
    for num in model_nums:
        lines += [f"## Per-Fold Model {num} Results", ""]
        lines.append(f"| Fold | Accuracy | {auroc_col} | MCC |")
        lines.append("|------|----------|" + "-" * (len(auroc_col) + 2) + "|-----|")
        for fr in all_fold_results:
            bm = fr["base_model_metrics"][num]
            lines.append(
                f"| {bm['fold_id']} | {_fv(bm.get('accuracy'))} | "
                f"{_fv(bm.get(auroc_key))} | "
                f"{_fv(bm.get('mcc'))} |"
            )
        lines += [""]

    # --- Confusion matrix ---
    cm = ensemble_agg.get("confusion_matrix_aggregated")
    classes = ensemble_agg.get("classes", [])
    if cm and classes:
        lines += ["## Aggregated Confusion Matrix (Ensemble, All Folds)", ""]
        lines.append("| | " + " | ".join(str(c) for c in classes) + " |")
        lines.append("|-" + "-|-".join("---" for _ in classes) + "-|")
        for i, cls in enumerate(classes):
            row_vals = " | ".join(str(cm[i][j]) for j in range(len(classes)))
            lines.append(f"| **{cls}** | {row_vals} |")
        lines += [""]

    return "\n".join(lines)


def _log_comparison_table(
    ensemble_agg: Dict,
    base_model_agg: Dict[int, Dict],
    model_nums: List[int],
):
    """Log a comparison table of ensemble vs base model performance."""
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*70}")

    header = f"{'Model':<25} {'Accuracy':>10} {'AUROC':>10} {'MCC':>10}"
    logger.info(header)
    logger.info("-" * 55)

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    def _get_metric_mean(agg, key):
        d = agg.get(key, {})
        return d.get("mean") if isinstance(d, dict) else None

    is_binary = "auroc_pooled" in ensemble_agg

    def _get_auroc(agg):
        if is_binary:
            return agg.get("auroc_pooled")
        return _get_metric_mean(agg, "auroc_ovo_weighted")

    # Ensemble
    acc = ensemble_agg.get("accuracy_global", _get_metric_mean(ensemble_agg, "accuracy_per_fold"))
    auroc = _get_auroc(ensemble_agg)
    mcc_mean = _get_metric_mean(ensemble_agg, "mcc")
    logger.info(f"{'Ensemble':<25} {_fmt(acc):>10} {_fmt(auroc):>10} {_fmt(mcc_mean):>10}")

    # Base models
    for num in model_nums:
        agg = base_model_agg[num]
        acc = agg.get("accuracy_global", _get_metric_mean(agg, "accuracy_per_fold"))
        auroc = _get_auroc(agg)
        mcc_mean = _get_metric_mean(agg, "mcc")
        logger.info(f"{'Model ' + str(num):<25} {_fmt(acc):>10} {_fmt(auroc):>10} {_fmt(mcc_mean):>10}")


# ============================================================================
# CLI
# ============================================================================

def _save_multi_binary_summary(
    base_output_dir: Path,
    all_pair_summaries: Dict[str, Dict],
    pairs_to_train: List[Tuple[str, str]],
    reference_class: str,
) -> None:
    """Save a cross-pair comparison summary for multi-binary ensemble training.

    Writes both a Markdown comparison table and a JSON summary to the base
    binary output directory (parent of all pair subdirectories).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    def _fv(val, fmt=".4f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    # --- Cross-pair MD ---
    lines = ["# Multi-Binary Ensemble Summary", ""]
    lines.append(f"**Timestamp**: {timestamp}")
    lines.append(f"**Reference class**: {reference_class}")
    lines.append(f"**Pairs trained**: {len(pairs_to_train)}")
    lines += ["", "## Cross-Pair Comparison (Ensemble)", ""]
    lines.append("| Pair | Accuracy | AUROC (pooled) | AUPRC (pooled) | MCC |")
    lines.append("|------|----------|----------------|----------------|-----|")

    for pair_key, summary in all_pair_summaries.items():
        ens = summary.get("ensemble", {})
        acc = _fv(ens.get("accuracy_global"))
        auroc = _fv(ens.get("auroc_pooled"))
        auprc = _fv(ens.get("auprc_pooled"))
        mcc_d = ens.get("mcc", {})
        mcc = _fv(mcc_d.get("mean")) if isinstance(mcc_d, dict) else "N/A"
        lines.append(f"| {pair_key} | {acc} | {auroc} | {auprc} | {mcc} |")
    lines += [""]

    md_path = base_output_dir / f"MULTI_BINARY_SUMMARY_{timestamp}.md"
    md_path.write_text("\n".join(lines))
    logger.info(f"\nSaved multi-binary summary: {md_path}")

    # --- Cross-pair JSON ---
    cross_summary = {
        "timestamp": timestamp,
        "reference_class": reference_class,
        "n_pairs": len(pairs_to_train),
        "pairs": {},
    }
    for pair_key, summary in all_pair_summaries.items():
        ens = summary.get("ensemble", {})
        cross_summary["pairs"][pair_key] = {
            "accuracy_global": ens.get("accuracy_global"),
            "auroc_pooled": ens.get("auroc_pooled"),
            "auprc_pooled": ens.get("auprc_pooled"),
            "mcc_mean": ens.get("mcc", {}).get("mean") if isinstance(ens.get("mcc"), dict) else None,
        }

    json_path = base_output_dir / f"multi_binary_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(cross_summary, f, indent=2)
    logger.info(f"Saved multi-binary summary JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble (metamodel) for Mal-ID-Lite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Data paths ---
    parser.add_argument(
        "--metadata-path", type=Path, default=None,
        help="Path to metadata.tsv (optional if cache has a copy).",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=PROJECT_ROOT / "cache" / "mal-id-orig-data",
        help="Cache directory with preprocessed data.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to AIRR data directory (optional when using cache).",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="mal-id-orig-data",
        help="Dataset name for output directory structure.",
    )

    # --- Classification mode ---
    parser.add_argument(
        "--classification-mode", type=str, default="multiclass",
        choices=["multiclass", "binary", "multi-binary"],
        help="Classification mode. multi-binary trains one ensemble per disease vs reference.",
    )
    parser.add_argument(
        "--reference-class", type=str, default="Healthy/Background",
        help="Reference class for binary mode.",
    )
    parser.add_argument(
        "--diseases", nargs="+", type=str, default=None,
        help="Disease classes to include (default: all from metadata).",
    )

    # --- Model selection ---
    parser.add_argument(
        "--models", nargs="+", type=int, default=[1, 2, 3],
        help="Which base models to include (default: 1 2 3).",
    )
    parser.add_argument(
        "--gene-locus", type=str, default="TCR", choices=["TCR", "BCR"],
        help="Gene locus (default: TCR).",
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, default=None,
        help="Fold IDs to process (default: all folds).",
    )

    # --- Model-specific suffixes ---
    parser.add_argument(
        "--model1-suffix", type=str, default=None,
        help="Output suffix for Model 1 artifacts.",
    )
    parser.add_argument(
        "--model2-suffix", type=str, default=None,
        help="Output suffix for Model 2 artifacts.",
    )
    parser.add_argument(
        "--model3-suffix", type=str, default=None,
        help="Output suffix for Model 3 artifacts.",
    )

    # --- Model 3 specific ---
    parser.add_argument(
        "--model3-embedding-dir", type=Path, default=None,
        help="Directory with pre-computed ESM-2 embeddings. "
             "Defaults to cache_dir/embeddings.",
    )

    # --- Ensemble output ---
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help="Suffix for ensemble output directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (ignores canonical path).",
    )

    # --- Runtime ---
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- Resolve paths ---
    data_dir = args.data_dir or Path(".")
    metadata_path = args.metadata_path

    # --- Initialize data loader ---
    loader = MalIDPublishedDataLoader(
        data_dir=data_dir,
        metadata_path=metadata_path,
        gene_locus=args.gene_locus,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
    )

    # --- Resolve fold IDs ---
    if args.fold_ids is not None:
        fold_ids = args.fold_ids
    else:
        fold_ids = get_dataset_fold_ids(loader.metadata_path)
    logger.info(f"Fold IDs: {fold_ids}")

    # --- Resolve base model artifact directories ---
    # For binary/multi-binary these point to .../binary/ — pair subdirectory is
    # appended per pair below. For multiclass these are the final directories.
    model_suffixes = {
        1: args.model1_suffix,
        2: args.model2_suffix,
        3: args.model3_suffix,
    }
    base_model_dirs = {}
    for num in args.models:
        base_model_dirs[num] = get_model_output_dir(
            model_name=f"model{num}",
            dataset_name=args.dataset_name,
            classification_mode=args.classification_mode,
            gene_locus=args.gene_locus,
            training_context=TRAINING_CONTEXT,
            output_suffix=model_suffixes.get(num),
        )

    # --- Resolve embedding directory ---
    embedding_dir = args.model3_embedding_dir
    if embedding_dir is None and 3 in args.models:
        embedding_dir = args.cache_dir / "embeddings"
    if 3 in args.models and (embedding_dir is None or not embedding_dir.exists()):
        logger.error(
            f"Model 3 embedding directory not found: {embedding_dir}\n"
            f"Compute embeddings first with compute_model3_embeddings.py."
        )
        sys.exit(1)

    # --- Resolve base output directory ---
    # For multi-binary, each pair gets a subdirectory under this base.
    if args.output_dir is not None:
        base_output_dir = args.output_dir
    else:
        base_output_dir = get_ensemble_output_dir(
            dataset_name=args.dataset_name,
            classification_mode=args.classification_mode,
            gene_locus=args.gene_locus,
            output_suffix=args.output_suffix,
        )

    # --- Resolve disease pairs to train ---
    reference_class = None
    if args.classification_mode == "multiclass":
        pairs_to_train = [None]

    elif args.classification_mode == "binary":
        if not args.diseases or len(args.diseases) != 1:
            logger.error(
                "Binary mode requires exactly one disease via --diseases <disease>.\n"
                "Example: --classification-mode binary --diseases Covid19 "
                "--reference-class Healthy/Background"
            )
            sys.exit(1)
        disease_classes = get_dataset_disease_classes(loader.metadata_path)
        reference_class = validate_mode_and_classes(
            "binary", disease_classes, args.reference_class, args.diseases,
        )
        pairs_to_train = [(args.diseases[0], reference_class)]

    elif args.classification_mode == "multi-binary":
        disease_classes = get_dataset_disease_classes(loader.metadata_path)
        reference_class = validate_mode_and_classes(
            "multi-binary", disease_classes, args.reference_class, args.diseases,
        )
        if args.diseases is not None:
            invalid = [d for d in args.diseases if d not in disease_classes]
            if invalid:
                raise ValueError(
                    f"--diseases {invalid} not found in data: {disease_classes}"
                )
            as_ref = [d for d in args.diseases if d == reference_class]
            if as_ref:
                raise ValueError(
                    f"--diseases includes reference class '{reference_class}'. "
                    f"Remove it from --diseases or change --reference-class."
                )
            diseases_to_train = list(args.diseases)
        else:
            diseases_to_train = [c for c in disease_classes if c != reference_class]
        pairs_to_train = [(d, reference_class) for d in diseases_to_train]

    # --- Log configuration ---
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"  Dataset:             {args.dataset_name}")
    logger.info(f"  Classification mode: {args.classification_mode}")
    logger.info(f"  Gene locus:          {args.gene_locus}")
    logger.info(f"  Models:              {args.models}")
    logger.info(f"  Folds:               {fold_ids}")
    logger.info(f"  Output:              {base_output_dir}")
    if args.classification_mode == "multi-binary":
        logger.info(f"  Reference class:     {reference_class}")
        logger.info(f"  Pairs to train:      {len(pairs_to_train)}")
        for d, r in pairs_to_train:
            logger.info(f"    {make_pair_name(d, r)}")
    elif args.classification_mode == "binary":
        logger.info(f"  Disease filter:      {pairs_to_train[0]}")

    # --- Train each pair (single iteration for multiclass/binary, N for multi-binary) ---
    all_pair_summaries = {}
    for pair in pairs_to_train:
        if pair is None:
            # Multiclass: base_model_dirs already point to .../multiclass/
            model_dirs = dict(base_model_dirs)
            output_dir = base_output_dir
            disease_filter = None
            ref_class = None
            pair_key = "multiclass"
        else:
            # Binary pair: append pair subdirectory to each base model dir
            disease, ref = pair
            pair_key = make_pair_name(disease, ref)
            model_dirs = {num: d / pair_key for num, d in base_model_dirs.items()}
            output_dir = base_output_dir / pair_key
            disease_filter = pair
            ref_class = ref

        if args.classification_mode == "multi-binary":
            logger.info(f"\n{'*'*60}")
            logger.info(f"Binary pair: {pair_key}")
            logger.info(f"{'*'*60}")

        # Validate artifact directories exist
        for num, d in model_dirs.items():
            if not d.exists():
                logger.error(
                    f"Model {num} artifact directory not found: {d}\n"
                    f"Train Model {num} with --training-context cv_ensemble first."
                )
                sys.exit(1)
            logger.info(f"  Model {num} artifacts: {d}")

        # Build run config for this pair
        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset_name": args.dataset_name,
            "classification_mode": args.classification_mode,
            "gene_locus": args.gene_locus,
            "models_included": args.models,
            "fold_ids": fold_ids,
            "reference_class": ref_class,
            "diseases": args.diseases,
            "disease_filter": list(disease_filter) if disease_filter else None,
            "output_suffix": args.output_suffix,
            "base_model_paths": {
                f"model{num}": str(d) for num, d in model_dirs.items()
            },
            "base_model_suffixes": {
                f"model{num}": model_suffixes.get(num) for num in args.models
            },
            "embedding_dir": str(embedding_dir) if embedding_dir else None,
            "metamodel_config": {
                "algorithm": "ridge_cv",
                "alpha": 0.0,
                "n_lambda": 100,
                "scoring": "MCC",
                "internal_cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)",
                "class_weight": "balanced",
                "use_lambda_1se": False,
            },
        }

        fold_results, summary = train_ensemble(
            loader=loader,
            fold_ids=fold_ids,
            model_nums=args.models,
            model_dirs=model_dirs,
            gene_locus=args.gene_locus,
            output_dir=output_dir,
            embedding_dir=embedding_dir,
            disease_filter=disease_filter,
            reference_class=ref_class,
            run_config=run_config,
            verbose=args.verbose,
        )
        all_pair_summaries[pair_key] = summary

    # --- Multi-binary cross-pair summary ---
    if args.classification_mode == "multi-binary" and len(all_pair_summaries) > 1:
        _save_multi_binary_summary(
            base_output_dir, all_pair_summaries, pairs_to_train, reference_class,
        )

    logger.info(f"\nDone. Output: {base_output_dir}")


if __name__ == "__main__":
    main()
