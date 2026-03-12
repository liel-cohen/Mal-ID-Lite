#!/usr/bin/env python
"""Train and evaluate Model 1 (Repertoire Classifier) with cross-validation.

This script trains Model 1 on all folds using the two-level cache for fast loading.
It computes multiclass one-vs-one AUROC scores and saves models and results.

Usage:
    python scripts/train_model1.py [--fold-ids 0 1 2] [--model-name lasso_cv] [--output-dir models]

Example:
    # Train on all folds with default elastic net
    python scripts/train_model1.py

    # Train only fold 0
    python scripts/train_model1.py --fold-ids 0

    # Use pure lasso instead of elastic net
    python scripts/train_model1.py --model-name lasso_cv --l1-ratio 1.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from malid.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid.models import RepertoireClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FoldResults:
    """Results for a single fold."""

    def __init__(
        self,
        fold_id: int,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        classes: np.ndarray,
        train_time: float,
        n_train: int,
        n_test: int,
        n_features: int,
    ):
        self.fold_id = fold_id
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.classes = classes
        self.train_time = train_time
        self.n_train = n_train
        self.n_test = n_test
        self.n_features = n_features

        # Compute metrics
        self.accuracy = accuracy_score(y_true, y_pred)

        # AUROC (one-vs-one, weighted) - primary metric
        self.auroc_ovo = roc_auc_score(
            y_true, y_proba,
            average="weighted",
            multi_class="ovo",
            labels=classes
        )

        # AUPRC (one-vs-rest, weighted) - average_precision_score does not support OVO
        self.auprc_ovr = average_precision_score(
            y_true, y_proba,
            average="weighted"
        )

        # Per-class AUROC (one-vs-rest) - for individual disease performance
        self.auroc_ovr_per_class = {}
        try:
            auroc_ovr_scores = roc_auc_score(
                y_true, y_proba,
                average=None,  # Return per-class scores
                multi_class="ovr",
                labels=classes
            )
            for i, cls in enumerate(classes):
                self.auroc_ovr_per_class[cls] = float(auroc_ovr_scores[i])
        except ValueError as e:
            # Handle case where some classes might not have enough samples
            for cls in classes:
                self.auroc_ovr_per_class[cls] = None

        self.log_loss = log_loss(y_true, y_proba, labels=classes)
        self.confusion = confusion_matrix(y_true, y_pred, labels=classes)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "fold_id": int(self.fold_id),
            "model_name": self.model_name,
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "n_features": int(self.n_features),
            "train_time": float(self.train_time),
            "accuracy": float(self.accuracy),
            "auroc_ovo_weighted": float(self.auroc_ovo),
            "auprc_ovr_weighted": float(self.auprc_ovr),
            "auroc_ovr_per_class": self.auroc_ovr_per_class,
            "log_loss": float(self.log_loss),
            "classes": self.classes.tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"FoldResults(fold={self.fold_id}, "
            f"accuracy={self.accuracy:.3f}, "
            f"auroc_ovo={self.auroc_ovo:.3f})"
        )


def filter_rare_v_genes(
    sequences: pd.DataFrame,
    threshold_quantile: float = 0.5,
    verbose: bool = True
) -> List[str]:
    """Filter out bottom 50% of V genes by frequency.

    Args:
        sequences: DataFrame with v_gene column
        threshold_quantile: Quantile threshold (0.5 = median)
        verbose: Whether to print filtering info

    Returns:
        List of V genes to keep
    """
    # Compute V gene frequencies
    v_gene_freq = sequences['v_gene'].value_counts(normalize=True)

    # Get threshold (median frequency)
    threshold = v_gene_freq.quantile(threshold_quantile)

    # Filter
    kept_v_genes = v_gene_freq[v_gene_freq >= threshold].index.tolist()
    removed_v_genes = v_gene_freq[v_gene_freq < threshold].index.tolist()

    if verbose:
        logger.info(
            f"V gene filtering: keeping {len(kept_v_genes)}/{len(v_gene_freq)} V genes "
            f"(removed {len(removed_v_genes)} with freq < {threshold:.4f})"
        )
        logger.info(f"  Removed: {removed_v_genes[:10]}{'...' if len(removed_v_genes) > 10 else ''}")

    return kept_v_genes


def train_and_evaluate_fold(
    loader: MalIDPublishedDataLoader,
    fold_id: int,
    model_name: str,
    model_params: Dict,
    output_dir: Path,
    verbose: int = 1,
) -> FoldResults:
    """Train and evaluate model on a single fold.

    Args:
        loader: Data loader
        fold_id: Fold ID (0, 1, or 2)
        model_name: Model variant name
        model_params: Model hyperparameters
        output_dir: Directory to save model and results
        verbose: Verbosity level

    Returns:
        FoldResults with metrics and predictions
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_id}")
    logger.info(f"{'='*60}\n")

    # Load training data
    logger.info(f"Loading training data (fold {fold_id})...")
    train_data, train_metadata = loader.get_fold_data(
        fold_id=fold_id,
        fold_label="train",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED
    )

    logger.info(
        f"  Loaded: {len(train_metadata)} specimens, "
        f"{len(train_data):,} sequences"
    )

    # Filter rare V genes (training only)
    logger.info("Filtering rare V genes...")
    kept_v_genes = filter_rare_v_genes(train_data, verbose=True)
    train_data_filtered = train_data[train_data['v_gene'].isin(kept_v_genes)].copy()

    logger.info(
        f"  After filtering: {len(train_data_filtered):,} sequences "
        f"({len(train_data_filtered)/len(train_data)*100:.1f}%)"
    )

    # Initialize model
    model = RepertoireClassifier(verbose=verbose, **model_params)

    # Extract training features
    logger.info("Extracting training features...")
    start_time = datetime.now()

    X_train = model.extract_features(
        sequences=train_data_filtered,
        metadata=train_metadata,
    )

    # Prepare labels and groups
    train_metadata_aligned = train_metadata.set_index("specimen_label").loc[X_train.index]
    y_train = train_metadata_aligned["disease"]
    groups_train = train_metadata_aligned["participant_label"]

    logger.info(
        f"  Features: {X_train.shape[0]} specimens × {X_train.shape[1]} features"
    )
    logger.info(f"  Classes: {y_train.nunique()} ({y_train.unique().tolist()})")

    # Train model
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train, groups=groups_train)

    train_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"  Training completed in {int(train_time)} seconds")

    # Load test data
    logger.info(f"Loading test data (fold {fold_id})...")
    test_data, test_metadata = loader.get_fold_data(
        fold_id=fold_id,
        fold_label="test",
        preprocessing_stage=PreprocessingStage.DOWNSAMPLED
    )

    logger.info(
        f"  Loaded: {len(test_metadata)} specimens, "
        f"{len(test_data):,} sequences"
    )

    # Filter to same V genes as training
    test_data_filtered = test_data[test_data['v_gene'].isin(kept_v_genes)].copy()

    logger.info(
        f"  After V gene alignment: {len(test_data_filtered):,} sequences "
        f"({len(test_data_filtered)/len(test_data)*100:.1f}%)"
    )

    # Extract test features (aligned to training columns)
    logger.info("Extracting test features...")
    X_test = model.extract_features(
        sequences=test_data_filtered,
        metadata=test_metadata,
        train_vj_columns=model.train_vj_columns_,  # Align to training
    )

    # Prepare test labels
    test_metadata_aligned = test_metadata.set_index("specimen_label").loc[X_test.index]
    y_test = test_metadata_aligned["disease"]

    logger.info(f"  Test features: {X_test.shape[0]} specimens × {X_test.shape[1]} features")

    # Predict
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Create results
    results = FoldResults(
        fold_id=fold_id,
        model_name=model_name,
        y_true=y_test.values,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=model.classes_,
        train_time=train_time,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        n_features=X_train.shape[1],
    )

    logger.info(f"\n  Results:")
    logger.info(f"    Accuracy: {results.accuracy:.3f}")
    logger.info(f"    AUROC (OvO, weighted): {results.auroc_ovo:.3f}")
    logger.info(f"    AUPRC (OvR, weighted): {results.auprc_ovr:.3f}")
    logger.info(f"    Log loss: {results.log_loss:.3f}")
    logger.info(f"  Per-class AUROC (OvR):")
    for cls in results.classes:
        score = results.auroc_ovr_per_class.get(cls)
        if score is not None:
            logger.info(f"    {cls}: {score:.3f}")
        else:
            logger.info(f"    {cls}: N/A")

    # Save model
    model_file = output_dir / f"{model_name}.fold{fold_id}.pkl"
    model.save(model_file)
    logger.info(f"  Model saved: {model_file}")

    # Save predictions
    predictions_file = output_dir / f"{model_name}.fold{fold_id}.predictions.csv"
    predictions_df = pd.DataFrame({
        "repertoire_id": X_test.index,
        "y_true": y_test.values,
        "y_pred": y_pred,
        **{f"prob_{cls}": y_proba[:, i] for i, cls in enumerate(model.classes_)}
    })
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"  Predictions saved: {predictions_file}")

    # Save V gene list
    v_genes_file = output_dir / f"{model_name}.fold{fold_id}.v_genes.json"
    with open(v_genes_file, 'w') as f:
        json.dump(kept_v_genes, f, indent=2)
    logger.info(f"  V genes saved: {v_genes_file}")

    return results


def format_confusion_matrix(confusion: np.ndarray, classes: List[str]) -> str:
    """Format confusion matrix as a nice table.

    Args:
        confusion: Confusion matrix
        classes: Class names

    Returns:
        Formatted string representation
    """
    # Calculate column widths
    max_class_len = max(len(cls) for cls in classes)
    col_width = max(max_class_len, 6)

    # Header
    lines = []
    lines.append("True Label → Predicted Label\n")

    # Column headers
    header = " " * (max_class_len + 2)
    for cls in classes:
        header += f"{cls:>{col_width}}  "
    header += "│ Total"
    lines.append(header)

    # Separator
    sep_len = len(header)
    lines.append("─" * (sep_len - 8) + "┼──────")

    # Rows
    for i, true_cls in enumerate(classes):
        row = f"{true_cls:<{max_class_len}}  "
        for j in range(len(classes)):
            row += f"{confusion[i, j]:>{col_width}}  "
        row_total = confusion[i, :].sum()
        row += f"│ {row_total:>4}"
        lines.append(row)

    # Separator
    lines.append("─" * (sep_len - 8) + "┼──────")

    # Column totals
    totals_row = f"{'Total':<{max_class_len}}  "
    for j in range(len(classes)):
        totals_row += f"{confusion[:, j].sum():>{col_width}}  "
    grand_total = confusion.sum()
    totals_row += f"│ {grand_total:>4}"
    lines.append(totals_row)

    # Diagonal accuracy
    diagonal_correct = np.trace(confusion)
    accuracy = diagonal_correct / grand_total if grand_total > 0 else 0
    lines.append("")
    lines.append(f"Diagonal (correct): {int(diagonal_correct)} / {int(grand_total)} = {accuracy:.1%} accuracy")

    return "\n".join(lines)


def aggregate_results(fold_results: List[FoldResults]) -> Dict:
    """Aggregate results across folds.

    Following paper methodology:
    - Accuracy: Computed globally by concatenating all test predictions
    - AUROC/AUPRC: Computed per-fold then averaged (probabilities on different scales)
    - Per-class AUROC: Computed per-fold then averaged for each class

    Args:
        fold_results: List of FoldResults

    Returns:
        Dictionary with aggregated metrics
    """
    # Per-fold metrics (then averaged)
    accuracies_per_fold = [r.accuracy for r in fold_results]
    aurocs = [r.auroc_ovo for r in fold_results]
    auprcs = [r.auprc_ovr for r in fold_results]
    log_losses = [r.log_loss for r in fold_results]

    # Global accuracy: concatenate all predictions
    all_y_true = np.concatenate([r.y_true for r in fold_results])
    all_y_pred = np.concatenate([r.y_pred for r in fold_results])
    global_accuracy = accuracy_score(all_y_true, all_y_pred)

    # Per-class AUROC (one-vs-rest): average across folds for each class
    all_classes = fold_results[0].classes
    auroc_ovr_per_class_aggregated = {}
    for cls in all_classes:
        class_scores = [r.auroc_ovr_per_class[cls] for r in fold_results
                       if r.auroc_ovr_per_class[cls] is not None]
        if class_scores:
            auroc_ovr_per_class_aggregated[cls] = {
                "mean": float(np.mean(class_scores)),
                "std": float(np.std(class_scores, ddof=1)),
                "per_fold": class_scores,
            }
        else:
            auroc_ovr_per_class_aggregated[cls] = None

    return {
        "n_folds": len(fold_results),
        "fold_ids": [r.fold_id for r in fold_results],
        "accuracy_per_fold": {
            "mean": float(np.mean(accuracies_per_fold)),
            "std": float(np.std(accuracies_per_fold, ddof=1)),
            "per_fold": accuracies_per_fold,
        },
        "accuracy_global": float(global_accuracy),
        "auroc_ovo_weighted": {
            "mean": float(np.mean(aurocs)),
            "std": float(np.std(aurocs, ddof=1)),
            "per_fold": aurocs,
        },
        "auprc_ovr_weighted": {
            "mean": float(np.mean(auprcs)),
            "std": float(np.std(auprcs, ddof=1)),
            "per_fold": auprcs,
        },
        "auroc_ovr_per_class": auroc_ovr_per_class_aggregated,
        "log_loss": {
            "mean": float(np.mean(log_losses)),
            "std": float(np.std(log_losses, ddof=1)),
            "per_fold": log_losses,
        },
    }


def write_results_markdown(
    output_file: Path,
    fold_results: List[FoldResults],
    aggregated: Dict,
    model_params: Dict,
    timestamp: str,
) -> None:
    """Write formatted results to a markdown file.

    Args:
        output_file: Path to write the markdown file
        fold_results: List of FoldResults
        aggregated: Aggregated metrics dictionary
        model_params: Model parameters
        timestamp: Timestamp string
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("# Model 1 Training Results\n\n")
        f.write(f"**Timestamp**: {timestamp}\n")
        f.write(f"**Model**: {fold_results[0].model_name}\n")
        f.write(f"**Parameters**: {model_params}\n\n")
        f.write("---\n\n")

        # Global metrics
        f.write("## Overall Performance\n\n")
        f.write("### Global Metrics (Following Paper Methodology)\n\n")
        f.write(f"**Accuracy**:\n")
        f.write(f"- **Global (concatenated predictions)**: {aggregated['accuracy_global']:.3f}\n")
        f.write(f"- Per-fold average: {aggregated['accuracy_per_fold']['mean']:.3f} ± {aggregated['accuracy_per_fold']['std']:.3f}\n\n")

        f.write("### Probability-Based Metrics (Per-Fold Then Averaged)\n\n")
        f.write(f"**Primary Metrics**:\n")
        f.write(f"- **AUROC (OvO, weighted)**: **{aggregated['auroc_ovo_weighted']['mean']:.3f} ± {aggregated['auroc_ovo_weighted']['std']:.3f}**\n")
        f.write(f"- **AUPRC (OvR, weighted)**: **{aggregated['auprc_ovr_weighted']['mean']:.3f} ± {aggregated['auprc_ovr_weighted']['std']:.3f}**\n\n")
        f.write(f"**Other**:\n")
        f.write(f"- Log loss: {aggregated['log_loss']['mean']:.3f} ± {aggregated['log_loss']['std']:.3f}\n\n")

        # Per-class AUROC
        f.write("### Per-Class AUROC (OvR) - Individual Disease Performance\n\n")
        f.write("| Disease | AUROC (OvR) | Std Dev | Performance |\n")
        f.write("|---------|-------------|---------|-------------|\n")

        # Sort classes alphabetically for display
        classes = sorted(aggregated['auroc_ovr_per_class'].keys())
        for cls in classes:
            scores = aggregated['auroc_ovr_per_class'][cls]
            if scores is not None:
                mean_score = scores['mean']
                std_score = scores['std']
                # Performance label
                if mean_score >= 0.95:
                    perf = "Excellent"
                elif mean_score >= 0.90:
                    perf = "Very Good"
                elif mean_score >= 0.80:
                    perf = "Good"
                else:
                    perf = "Fair"
                f.write(f"| {cls} | {mean_score:.3f} | ±{std_score:.3f} | {perf} |\n")
            else:
                f.write(f"| {cls} | N/A | N/A | N/A |\n")
        f.write("\n")

        # Per-fold results
        f.write("### Per-Fold Results\n\n")
        f.write("| Fold | Accuracy | AUROC (OvO) | AUPRC (OvR) | Log Loss |\n")
        f.write("|------|----------|-------------|-------------|----------|\n")
        for result in fold_results:
            f.write(f"| {result.fold_id} | {result.accuracy:.3f} | {result.auroc_ovo:.3f} | {result.auprc_ovr:.3f} | {result.log_loss:.3f} |\n")
        f.write("\n")

        # Aggregated confusion matrix
        f.write("## Aggregated Confusion Matrix (All Folds)\n\n")
        # Combine confusion matrices
        classes_list = fold_results[0].classes.tolist()
        combined_confusion = sum(r.confusion for r in fold_results)

        # Sort classes alphabetically and reorder confusion matrix
        sorted_indices = np.argsort(classes_list)
        sorted_classes = [classes_list[i] for i in sorted_indices]
        sorted_confusion = combined_confusion[sorted_indices, :][:, sorted_indices]

        f.write("```\n")
        f.write(format_confusion_matrix(sorted_confusion, sorted_classes))
        f.write("\n```\n\n")

        # Per-class accuracy
        f.write("### Per-Class Accuracy\n\n")
        f.write("| Disease | Correct | Total | Accuracy |\n")
        f.write("|---------|---------|-------|----------|\n")
        for i, cls in enumerate(sorted_classes):
            correct = sorted_confusion[i, i]
            total = sorted_confusion[i, :].sum()
            acc = correct / total if total > 0 else 0
            f.write(f"| {cls} | {int(correct)} | {int(total)} | {acc:.1%} |\n")
        f.write("\n")

        # Per-fold confusion matrices
        f.write("## Individual Fold Confusion Matrices\n\n")
        for result in fold_results:
            f.write(f"### Fold {result.fold_id}\n\n")

            # Sort for this fold too
            fold_classes_list = result.classes.tolist()
            fold_sorted_indices = np.argsort(fold_classes_list)
            fold_sorted_classes = [fold_classes_list[i] for i in fold_sorted_indices]
            fold_sorted_confusion = result.confusion[fold_sorted_indices, :][:, fold_sorted_indices]

            f.write("```\n")
            f.write(format_confusion_matrix(fold_sorted_confusion, fold_sorted_classes))
            f.write("\n```\n\n")

            # Per-class AUROC for this fold
            f.write(f"**Per-class AUROC (OvR) - Fold {result.fold_id}**:\n\n")
            for cls in fold_sorted_classes:
                score = result.auroc_ovr_per_class.get(cls)
                if score is not None:
                    f.write(f"- {cls}: {score:.3f}\n")
                else:
                    f.write(f"- {cls}: N/A\n")
            f.write("\n")

        # Footer
        f.write("---\n\n")
        f.write("*Generated by Model 1 training script*\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Model 1 (Repertoire Classifier)"
    )
    parser.add_argument(
        "--fold-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Fold IDs to train on (default: 0 1 2)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lasso_cv",
        help="Model variant name (default: lasso_cv for TCR)"
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="Elastic net L1/L2 ratio (default: None = use 1.0 for TCR, 0.25 for BCR as per paper)"
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=15,
        help="Number of PCA components per isotype (default: 15)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/model1"),
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Cache directory with preprocessed data"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=basic, 2=detailed)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.output_dir / f"training_{timestamp}.log"

    # Add file handler to logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("="*60)
    logger.info("MODEL 1 TRAINING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    logger.info(f"Model name: {args.model_name}")
    if args.l1_ratio is not None:
        logger.info(f"L1 ratio: {args.l1_ratio} (specified)")
    else:
        logger.info(f"L1 ratio: default (1.0 for TCR, 0.25 for BCR)")
    logger.info(f"n_pcs: {args.n_pcs}")
    logger.info(f"Folds: {args.fold_ids}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Cache: {args.cache_dir}")
    logger.info("")

    # Initialize data loader
    logger.info("Initializing data loader...")
    project_root = Path(__file__).parent.parent

    loader = MalIDPublishedDataLoader(
        data_dir=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data_clean/airr_format_clean/TCR/"),
        metadata_path=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv"),
        gene_reference_path=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/tcrb_v_gene_cdrs.generated.tsv"),
        gene_locus="TCR",
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )

    # Check cache
    if args.cache_dir.exists():
        n_participant_cache = len(list((args.cache_dir / "participants").glob("*.parquet")))
        n_fold_cache = len(list(args.cache_dir.glob("fold_*.parquet")))
        logger.info(f"Cache status:")
        logger.info(f"  Participant cache: {n_participant_cache} files")
        logger.info(f"  Fold cache: {n_fold_cache} files")
    else:
        logger.warning(f"Cache not found at {args.cache_dir}")

    # Model parameters
    model_params = {
        "gene_locus": "TCR",
        "n_pcs": args.n_pcs,
    }
    # Only add l1_ratio if specified (otherwise use model's default for gene_locus)
    if args.l1_ratio is not None:
        model_params["l1_ratio"] = args.l1_ratio

    # Train on each fold
    fold_results = []
    for fold_id in args.fold_ids:
        try:
            results = train_and_evaluate_fold(
                loader=loader,
                fold_id=fold_id,
                model_name=args.model_name,
                model_params=model_params,
                output_dir=args.output_dir,
                verbose=args.verbose,
            )
            fold_results.append(results)

        except Exception as e:
            logger.error(f"Fold {fold_id} failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Aggregate results
    if len(fold_results) == 0:
        logger.error("All folds failed!")
        return 1

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}\n")

    aggregated = aggregate_results(fold_results)

    logger.info(f"Folds completed: {aggregated['n_folds']}/{len(args.fold_ids)}")

    logger.info(f"\nGlobal metrics (following paper methodology):")
    logger.info(f"  Accuracy (concatenated predictions): {aggregated['accuracy_global']:.3f}")
    logger.info(f"  Accuracy (per-fold average): {aggregated['accuracy_per_fold']['mean']:.3f} ± {aggregated['accuracy_per_fold']['std']:.3f}")

    logger.info(f"\nProbability-based metrics (per-fold then averaged):")
    logger.info(f"  AUROC (OvO, weighted): {aggregated['auroc_ovo_weighted']['mean']:.3f} ± {aggregated['auroc_ovo_weighted']['std']:.3f}")
    logger.info(f"  AUPRC (OvR, weighted): {aggregated['auprc_ovr_weighted']['mean']:.3f} ± {aggregated['auprc_ovr_weighted']['std']:.3f}")
    logger.info(f"  Log loss: {aggregated['log_loss']['mean']:.3f} ± {aggregated['log_loss']['std']:.3f}")

    logger.info(f"\nPer-class AUROC (OvR, per-fold then averaged):")
    # Sort classes alphabetically for display
    for cls in sorted(aggregated['auroc_ovr_per_class'].keys()):
        scores = aggregated['auroc_ovr_per_class'][cls]
        if scores is not None:
            logger.info(f"  {cls}: {scores['mean']:.3f} ± {scores['std']:.3f}")
        else:
            logger.info(f"  {cls}: N/A")

    logger.info(f"\nPer-fold results:")
    for result in fold_results:
        logger.info(
            f"  Fold {result.fold_id}: "
            f"Acc={result.accuracy:.3f}, "
            f"AUROC={result.auroc_ovo:.3f}, "
            f"AUPRC={result.auprc_ovr:.3f}"
        )

    # Save aggregated results
    results_file = args.output_dir / f"{args.model_name}.results_{timestamp}.json"
    results_data = {
        "model_name": args.model_name,
        "timestamp": timestamp,
        "parameters": model_params,
        "aggregated_metrics": aggregated,
        "fold_results": [r.to_dict() for r in fold_results],
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Write formatted markdown results
    markdown_file = args.output_dir / f"{args.model_name}.RESULTS_{timestamp}.md"
    write_results_markdown(
        output_file=markdown_file,
        fold_results=fold_results,
        aggregated=aggregated,
        model_params=model_params,
        timestamp=timestamp,
    )

    logger.info(f"\nResults saved to:")
    logger.info(f"  JSON: {results_file}")
    logger.info(f"  Markdown: {markdown_file}")
    logger.info(f"  Log: {log_file}")

    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
