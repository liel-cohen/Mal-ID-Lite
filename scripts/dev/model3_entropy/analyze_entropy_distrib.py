"""Analyze per-sequence entropy distributions by specimen and disease.

Loads a trained Stage 1 model, generates sequence-level predictions on
train_smaller2, computes Shannon entropy for each sequence, and visualizes
the entropy distribution per specimen grouped by disease.

Usage:
    # First run (computes entropy values):
    python scripts/dev/model3_entropy/analyze_entropy_distrib.py \
        --exp-name multiclass_fold0 \
        --model-dir trained_models/mal-id-orig-data/model3/multiclass/TCR \
        --fold-id 0 \
        --metadata-path /path/to/metadata.tsv \
        --embedding-dir cache/mal-id-orig-data/embeddings

    # Subsequent runs (reuse cached entropy values):
    python scripts/dev/model3_entropy/analyze_entropy_distrib.py \
        --exp-name multiclass_fold0 \
        --model-dir trained_models/mal-id-orig-data/model3/multiclass/TCR \
        --fold-id 0 \
        --metadata-path /path/to/metadata.tsv

Expected runtime:
    - Entropy computation: ~5-15 min (depends on fold size)
    - Plot generation: ~30 sec
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

# Monkey-patch pandas flatten_axes to return a list instead of a generator.
# Newer pandas (>=2.x) returns a generator from flatten_axes, which breaks
# joypy's internal indexing (_axes[i]).  This must run before importing joypy.
try:
    import pandas.plotting._matplotlib.tools as _pmt
    _orig_flatten = _pmt.flatten_axes
    _pmt.flatten_axes = lambda axes: list(_orig_flatten(axes))
except AttributeError:
    pass  # older pandas without flatten_axes; joypy uses _flatten instead

import joypy
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from malid_lite.dataloader import MalIDPublishedDataLoader
from malid_lite.models.model3_sequence_level import SequenceLevelClassifier
from malid_lite.training.train_model3 import (
    load_and_prepare_fold,
    load_precomputed_embeddings,
)
from malid_lite.training.training_utils import (
    filter_to_binary_pair,
    split_train_smaller,
)


# ---------------------------------------------------------------------------
# Joyplot helper (adapted from LielTools.PlotTools.plot_joyplot)
# ---------------------------------------------------------------------------

def plot_joyplot(
    df, vals_col, class_col, class_order=None, figsize=(10, 15),
    add_all_distribution=True, class_color_dict=None, all_color='#1a1a1a',
    xlabel='', ylabel='Class', fill=False, overlap=0.6, alpha=0.75,
    cut_kde_to_data_limits=True, class_label_y_pos=-0.05,
    ticklabels_fontsize=15, labels_fontsize=18,
    add_median=False, median_num_digits=1, classes_text_bold=False,
    save_path=None, xlim=None, x_ticks=None,
    vertical_line_x=None, vertical_line_color='#bfbfbf', vertical_line_alpha=0.3,
):
    """Plot distribution of values per class using a ridgeplot/joyplot (joypy).

    Adapted from LielTools.PlotTools.plot_joyplot.
    """
    df_copy = df.copy()
    df_copy = df_copy.loc[df_copy[vals_col].notna()]

    if class_order is None:
        class_order = list(df_copy[class_col].unique())

    if add_all_distribution:
        all_vals = df_copy[vals_col].values
        df_copy = pd.concat(
            [df_copy, pd.DataFrame({vals_col: all_vals, class_col: 'All'})],
            ignore_index=True,
        )
        class_order = class_order.copy()
        class_order += ['All']
        class_color_dict = class_color_dict.copy()
        class_color_dict['All'] = all_color

    # Order classes by given order
    df_copy = df_copy.reset_index(drop=True)
    df_copy[vals_col] = df_copy[vals_col].astype(float)
    df_copy[class_col] = df_copy[class_col].astype('category')
    df_copy[class_col] = df_copy[class_col].cat.set_categories(class_order, ordered=True)

    # Build colormap from the ordered color list
    colors = [class_color_dict[cla] for cla in class_order]
    cmap = matplotlib.colors.ListedColormap(colors, name='Custom cmap', N=len(colors))

    fig, axes = joypy.joyplot(
        df_copy, column=vals_col, by=class_col, fill=fill,
        colormap=cmap, figsize=figsize, overlap=overlap,
        alpha=alpha, range_style='own',
        tails=0 if cut_kde_to_data_limits else 0.2, x_range=xlim,
    )
    # Some joypy versions return a generator instead of a list for axes
    if not isinstance(axes, list):
        axes = list(axes)

    if add_median:
        for ax in axes:
            ylab = ax.get_yticklabels()[0].get_text()
            if ylab != '':
                if ylab == 'All':
                    median_val = df_copy[vals_col].median()
                else:
                    median_val = df_copy.loc[
                        df_copy[class_col].astype(str) == ylab, vals_col
                    ].median()
                ax.set_yticklabels(
                    [r'$\bf{' + ylab + r'}$ ('
                     + format(median_val, "." + str(median_num_digits) + "f") + ')'],
                    fontsize=ticklabels_fontsize,
                )
    else:
        for ax in axes:
            ylab = ax.get_yticklabels()[0].get_text()
            if classes_text_bold:
                ax.set_yticklabels(
                    [r'$\bf{' + ylab + r'}$'], fontsize=ticklabels_fontsize,
                )
            else:
                ax.set_yticklabels([ylab], fontsize=ticklabels_fontsize)

    # Set xticklabels fontsize
    for ax in axes:
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, fontsize=ticklabels_fontsize)
        else:
            ax.tick_params(axis='x', labelsize=ticklabels_fontsize)

    plt.xlabel(xlabel, fontsize=labels_fontsize)
    fig.text(class_label_y_pos, 0.5, ylabel, va='center', rotation='vertical',
             fontsize=labels_fontsize)

    if vertical_line_x is not None:
        for ax in axes:
            ax.axvline(x=vertical_line_x, color=vertical_line_color,
                       alpha=vertical_line_alpha)

    if save_path is not None:
        fig.savefig(save_path, dpi=500, bbox_inches='tight')

# Disease color palette (consistent across plots)
DISEASE_COLORS = {
    "Covid19": "#E64B35",
    "HIV": "#4DBBD5",
    "Healthy/Background": "#91D1C2",
    "Influenza": "#F39B7F",
    "Lupus": "#8491B4",
    "T1D": "#B09C85",
}


def load_stage1_model(
    model_dir: Path,
    fold_id: int,
    locus: str = "TCR",
    n_jobs: int = 1,
) -> SequenceLevelClassifier:
    """Load a trained Stage 1 model from disk.

    Parameters
    ----------
    model_dir : Directory containing fold_X_stage1.pkl artifacts.
    fold_id : Which fold's model to load.
    locus : "TCR" or "BCR".
    n_jobs : Parallel workers for generate_sequence_predictions.

    Returns
    -------
    SequenceLevelClassifier with Stage 1 fitted state restored.
    """
    artifact_path = model_dir / f"fold_{fold_id}_stage1.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Stage 1 artifact not found: {artifact_path}\n"
            f"Available files: {sorted(p.name for p in model_dir.glob('fold_*'))}"
        )

    with open(artifact_path, "rb") as f:
        data = pickle.load(f)

    model = SequenceLevelClassifier(locus=locus, n_jobs=n_jobs, verbose=0)
    model.load_stage1_artifacts(data)

    meta = data.get("_meta", {})
    n_groups = meta.get("n_groups", len(model.group_models_))
    classes = list(model.classes_)
    print(f"  Loaded Stage 1: fold={fold_id}, {n_groups} groups, classes={classes}")

    return model


def compute_sequence_entropies(
    model: SequenceLevelClassifier,
    sequences_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Generate Stage 1 predictions and compute per-sequence entropy.

    Only sequences whose disease is in the model's classes are included.
    For multiclass models this keeps all sequences; for binary models it
    keeps only the two relevant diseases (matching the training setup).

    Returns
    -------
    DataFrame with columns: specimen_label, disease, v_gene, entropy,
                            max_prob, prob_sum, n_model_classes
    """
    # Filter to only diseases the model was trained on
    model_classes = set(model.classes_)
    mask = sequences_df["disease"].isin(model_classes)
    n_before = len(sequences_df)
    sequences_df = sequences_df[mask].copy().reset_index(drop=True)
    embeddings = embeddings[mask.values]
    print(
        f"  Filtered to model classes {sorted(model_classes)}: "
        f"{len(sequences_df):,} / {n_before:,} sequences"
    )

    print("  Generating Stage 1 predictions...")
    t0 = time.monotonic()
    seq_preds = model.generate_sequence_predictions(sequences_df, embeddings)
    print(f"  Predictions done [{time.monotonic() - t0:.1f}s]")

    # Extract probability columns
    prob_cols = [f"prob_{c}" for c in model.classes_]
    probs = seq_preds[prob_cols].values

    # Keep only sequences that have predictions (not NaN)
    has_pred = seq_preds["has_prediction"].values
    print(
        f"  Sequences with predictions: {has_pred.sum():,} / {len(has_pred):,} "
        f"({has_pred.sum() / len(has_pred) * 100:.1f}%)"
    )

    # Compute entropy using the same method as the production filter in
    # _entropy_threshold_aggregate: scipy.stats.entropy(probs.T).
    # scipy normalizes each distribution to sum to 1 internally, which is
    # required for OvR classifiers whose per-class probs don't sum to 1.
    print("  Computing entropy values...")
    # scipy.stats.entropy treats columns as distributions; .T makes it per-row
    entropy_values = scipy_entropy(probs.T)

    # n_model_classes: the number of classes the model was trained on (= number
    # of probability columns). This can differ from the number of unique diseases
    # in the data when running a binary model on full fold data.
    n_model_classes = len(model.classes_)

    result = pd.DataFrame(
        {
            "specimen_label": seq_preds["specimen_label"].values,
            "disease": sequences_df["disease"].values,
            "v_gene": sequences_df["v_gene"].values,
            "entropy": entropy_values,
            "max_prob": np.nanmax(probs, axis=1),
            "prob_sum": np.nansum(probs, axis=1),
            "has_prediction": has_pred,
            "n_model_classes": n_model_classes,
        }
    )

    # Filter to only sequences with predictions for analysis
    result = result[result["has_prediction"]].copy()
    result = result.drop(columns=["has_prediction"])

    return result


def _compute_percentile_thresholds(
    entropy_values: np.ndarray,
) -> dict:
    """Compute entropy thresholds at various bottom-N percentiles.

    Returns dict mapping label -> threshold value.
    "Bottom 5%" means the entropy value at the 5th percentile
    (i.e., the 5% of sequences with the lowest entropy = most confident).
    """
    return {
        "bottom 5%": np.percentile(entropy_values, 5),
        "bottom 2%": np.percentile(entropy_values, 2),
        "bottom 1%": np.percentile(entropy_values, 1),
        "bottom 0.5%": np.percentile(entropy_values, 0.5),
        "bottom 0.1%": np.percentile(entropy_values, 0.1),
        "bottom 0.01%": np.percentile(entropy_values, 0.01),
        "bottom 0.001%": np.percentile(entropy_values, 0.001),
        "bottom 0.0001%": np.percentile(entropy_values, 0.0001),
    }


def plot_combined_joyplot(
    entropy_df: pd.DataFrame,
    n_classes: int,
    output_dir: Path,
    entropy_col: str = "entropy",
) -> None:
    """Create a single joyplot of entropy distributions for all specimens, grouped by disease.

    Each row = one specimen. Specimens are sorted by disease (alphabetical),
    then by median entropy within each disease. Each disease gets a distinct
    color. Vertical dashed lines show bottom-percentile thresholds, solid lines
    show entropy cutoffs (5%, 10%, 20%).

    Parameters
    ----------
    entropy_df : DataFrame with columns: specimen_label, disease, entropy.
    n_classes : Number of disease classes (for max_entropy calculation).
    output_dir : Where to save the figure.
    entropy_col : Which entropy column to use.
    """
    from matplotlib.lines import Line2D

    max_entropy = np.log(n_classes)  # ln(n_classes) in nats
    diseases = sorted(entropy_df["disease"].unique())
    n_specimens = entropy_df["specimen_label"].nunique()

    if n_specimens == 0:
        print("  Skipping combined plot: no specimens")
        return

    # --- Build specimen order: grouped by disease, sorted by median within ---
    # Within each disease, sort by median entropy descending (most confident
    # = lowest median at the bottom of the plot).
    specimen_order = []
    color_dict = {}
    for disease in diseases:
        d_df = entropy_df[entropy_df["disease"] == disease]
        medians = (
            d_df.groupby("specimen_label")[entropy_col]
            .median()
            .sort_values(ascending=False)
        )
        color = DISEASE_COLORS.get(disease, "#666666")
        for spec in medians.index:
            specimen_order.append(spec)
            color_dict[spec] = color

    # Compute percentile thresholds across ALL sequences
    thresholds = _compute_percentile_thresholds(entropy_df[entropy_col].values)

    # x-axis: start at the minimum of the data min and the lowest cutoff line
    # (20% cutoff = 0.80 * max_entropy), so all vertical lines are visible.
    # Extend slightly beyond data max so KDE peaks at edges aren't clipped.
    data_min = entropy_df[entropy_col].min()
    data_max = entropy_df[entropy_col].max()
    cutoff_20pct = 0.80 * max_entropy
    xmin = min(data_min, cutoff_20pct)
    padding = (data_max - xmin) * 0.02
    xlim = [xmin - padding, data_max + padding]

    fig_height = max(6, n_specimens * 0.2)

    plot_joyplot(
        df=entropy_df,
        vals_col=entropy_col,
        class_col="specimen_label",
        class_order=specimen_order,
        figsize=(14, fig_height),
        add_all_distribution=False,
        class_color_dict=color_dict,
        xlabel="Entropy (nats)",
        ylabel="Specimen",
        fill=True,
        overlap=0.7,
        alpha=0.6,
        cut_kde_to_data_limits=True,
        xlim=xlim,
        ticklabels_fontsize=6,
        labels_fontsize=14,
        add_median=True,
        median_num_digits=3,
    )

    fig = plt.gcf()

    # Title with max theoretical entropy
    fig.suptitle(
        f"Entropy distribution by specimen  "
        f"(n = {n_specimens} specimens, {len(diseases)} diseases, "
        f"max entropy = {max_entropy:.4f} nats = ln({n_classes}))",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # --- Vertical lines: percentile thresholds (dashed) ---
    threshold_colors = {
        "bottom 5%": "#2ca02c",
        "bottom 2%": "#ff7f0e",
        "bottom 1%": "#d62728",
        "bottom 0.5%": "#9467bd",
        "bottom 0.1%": "#17becf",
        "bottom 0.01%": "#e377c2",
        "bottom 0.001%": "#bcbd22",
        "bottom 0.0001%": "#7f7f7f",
    }
    for ax in fig.axes:
        for label, thresh_val in thresholds.items():
            ax.axvline(
                x=thresh_val,
                color=threshold_colors[label],
                alpha=0.5,
                linestyle="--",
                linewidth=0.8,
            )

    # --- Vertical lines: entropy cutoffs (solid) ---
    threshold_20pct = 0.80 * max_entropy
    threshold_10pct = 0.90 * max_entropy
    threshold_5pct = 0.95 * max_entropy
    cutoff_lines = [
        (threshold_20pct, "20% cutoff", "#000000"),
        (threshold_10pct, "10% cutoff", "#555555"),
        (threshold_5pct, "5% cutoff", "#888888"),
    ]
    for thresh_val, _, thresh_color in cutoff_lines:
        for ax in fig.axes:
            ax.axvline(
                x=thresh_val,
                color=thresh_color,
                alpha=0.7,
                linestyle="-",
                linewidth=1.2,
            )

    # --- Build legend ---
    legend_lines = []
    legend_labels = []

    # Disease colors
    for disease in diseases:
        color = DISEASE_COLORS.get(disease, "#666666")
        legend_lines.append(
            Line2D([0], [0], color=color, linewidth=4, alpha=0.6)
        )
        legend_labels.append(disease)

    # Separator
    legend_lines.append(Line2D([0], [0], color="none"))
    legend_labels.append("")

    # Percentile thresholds
    for label, thresh_val in thresholds.items():
        legend_lines.append(
            Line2D([0], [0], color=threshold_colors[label],
                   linestyle="--", linewidth=1.5)
        )
        legend_labels.append(f"{label}: {thresh_val:.4f}")

    # Cutoff lines
    for thresh_val, thresh_label, thresh_color in cutoff_lines:
        legend_lines.append(
            Line2D([0], [0], color=thresh_color, linestyle="-", linewidth=1.5)
        )
        legend_labels.append(f"{thresh_label}: {thresh_val:.4f}")

    fig.legend(
        legend_lines,
        legend_labels,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        ncol=1,
        bbox_to_anchor=(1.01, 1.0),
    )

    # Save
    save_path = output_dir / "entropy_distrib_combined.png"
    fig.savefig(save_path, dpi=3000, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name} ({n_specimens} specimens, {len(diseases)} diseases)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-sequence entropy distributions by disease.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (used as output subfolder).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model artifacts folder (containing fold_X_stage1.pkl).",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        required=True,
        help="Fold ID to analyze.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to metadata TSV file.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help=(
            "Path to pre-computed embeddings directory. "
            "Required for first run (entropy computation). "
            "Default: cache/<dataset>/embeddings/ under project root."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for preprocessed data. Default: cache/mal-id-orig-data/.",
    )
    parser.add_argument(
        "--locus",
        type=str,
        default="TCR",
        choices=["TCR", "BCR"],
        help="Gene locus. Default: TCR.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for Stage 1 prediction (per-V-gene groups). Default: 1.",
    )
    parser.add_argument(
        "--recalc-entropy",
        action="store_true",
        help="Force recalculation of entropy values even if cached artifact exists.",
    )
    args = parser.parse_args()

    # --- Output directory ---
    output_dir = Path(__file__).parent / "output" / "analyze_entropy_distrib" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Entropy values artifact ---
    entropy_artifact_path = output_dir / f"sequence_entropies_fold{args.fold_id}.parquet"

    if entropy_artifact_path.exists() and not args.recalc_entropy:
        # Load cached entropy values
        print(f"Loading cached entropy values: {entropy_artifact_path.name}")
        entropy_df = pd.read_parquet(entropy_artifact_path)
        print(f"  {len(entropy_df):,} sequences loaded")
    else:
        # Compute entropy values from scratch
        if args.recalc_entropy and entropy_artifact_path.exists():
            print("Recalculating entropy values (--recalc-entropy specified)")
        else:
            print("Computing entropy values (no cached artifact found)")

        # --- Load model FIRST (needed to know which classes to filter to) ---
        print(f"Loading Stage 1 model from: {args.model_dir}")
        model = load_stage1_model(args.model_dir, args.fold_id, args.locus, args.n_jobs)

        # --- Load data ---
        cache_dir = args.cache_dir
        if cache_dir is None:
            cache_dir = project_root / "cache" / "mal-id-orig-data"

        print(f"Loading data loader (cache: {cache_dir})...")
        loader = MalIDPublishedDataLoader(
            data_dir=Path("."),  # placeholder, cache covers all reads
            metadata_path=args.metadata_path,
            cache_dir=cache_dir,
            verbose=0,
        )

        print(f"Loading fold {args.fold_id} training data...")
        t0 = time.monotonic()
        train_seq, train_meta = load_and_prepare_fold(loader, args.fold_id, "train")
        print(
            f"  Train fold: {len(train_seq):,} sequences, "
            f"{train_seq['participant_label'].nunique()} participants "
            f"[{time.monotonic() - t0:.1f}s]"
        )

        # For binary models, filter to the model's two classes BEFORE splitting
        # into ts1/ts2.  This matches the training pipeline in train_model3.py
        # (lines 1527-1530) where filter_to_binary_pair() runs before
        # split_train_smaller().  Without this, the stratified split produces
        # different participant assignments because the disease distribution
        # changes.
        model_classes = set(model.classes_)
        n_data_diseases = train_meta["disease"].nunique()
        if len(model_classes) < n_data_diseases:
            # Binary model: identify disease and reference class from model.classes_
            # (order doesn't matter for filter_to_binary_pair, it keeps both)
            cls_list = sorted(model_classes)
            print(
                f"  Binary model detected ({cls_list}): filtering to model classes "
                f"before split (matching training pipeline)..."
            )
            train_seq, train_meta = filter_to_binary_pair(
                train_seq, train_meta, cls_list[0], cls_list[1]
            )
            print(
                f"  After filtering: {len(train_seq):,} sequences, "
                f"{train_seq['participant_label'].nunique()} participants"
            )

        # Split into ts1/ts2 (same split as training: stratified, random_state=0)
        ts1, ts2 = split_train_smaller(train_seq, train_meta)
        print(
            f"  train_smaller2: {len(ts2):,} sequences, "
            f"{ts2['participant_label'].nunique()} participants"
        )

        # --- Load embeddings ---
        embedding_dir = args.embedding_dir
        if embedding_dir is None:
            embedding_dir = cache_dir / "embeddings"

        print(f"Loading ts2 embeddings from: {embedding_dir}")
        t0 = time.monotonic()
        emb_ts2 = load_precomputed_embeddings(ts2, embedding_dir)
        print(f"  Loaded {len(emb_ts2):,} embeddings [{time.monotonic() - t0:.1f}s]")

        # --- Compute entropy ---
        entropy_df = compute_sequence_entropies(model, ts2, emb_ts2)

        # --- Save artifact ---
        entropy_df.to_parquet(entropy_artifact_path, index=False)
        print(f"Saved entropy artifact: {entropy_artifact_path.name}")
        print(f"  {len(entropy_df):,} sequences with entropy values")

        # Free memory
        del emb_ts2, ts1, ts2, train_seq, train_meta

    # --- Analysis and plotting ---
    if entropy_df.empty:
        print("ERROR: No sequences received predictions (all in rare V-gene groups "
              "without trained models). Check model/data alignment.")
        return 1

    # n_model_classes = number of classes the model outputs (2 for binary, 6 for
    # multiclass). This determines max_entropy and the cutoff thresholds.
    n_classes = int(entropy_df["n_model_classes"].iloc[0])
    max_entropy = np.log(n_classes)
    diseases = sorted(entropy_df["disease"].unique())

    print(f"\n{'=' * 70}")
    print(f"ENTROPY DISTRIBUTION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Classes: {n_classes} ({', '.join(diseases)})")
    print(f"  Max theoretical entropy: {max_entropy:.4f} nats (ln({n_classes}))")
    print(f"  Total sequences: {len(entropy_df):,}")

    # --- Per-disease summary stats ---
    print(f"\n{'=' * 70}")
    print("PER-DISEASE SUMMARY")
    print(f"{'=' * 70}")

    summary_rows = []
    for disease in diseases:
        d_df = entropy_df[entropy_df["disease"] == disease]
        e_vals = d_df["entropy"]
        n_specimens = d_df["specimen_label"].nunique()
        n_seqs = len(d_df)

        # Thresholds: keep sequences with entropy < (1 - fraction) * max_entropy
        thresh_20pct = 0.80 * max_entropy  # 20% cutoff
        thresh_10pct = 0.90 * max_entropy  # 10% cutoff
        thresh_5pct = 0.95 * max_entropy   # 5% cutoff
        n_pass_20 = (e_vals < thresh_20pct).sum()
        n_pass_10 = (e_vals < thresh_10pct).sum()
        n_pass_5 = (e_vals < thresh_5pct).sum()

        summary_rows.append(
            {
                "disease": disease,
                "n_specimens": n_specimens,
                "n_sequences": n_seqs,
                "entropy_mean": e_vals.mean(),
                "entropy_std": e_vals.std(),
                "entropy_median": e_vals.median(),
                "entropy_min": e_vals.min(),
                "entropy_max": e_vals.max(),
                "pct5": np.percentile(e_vals, 5),
                "pct2": np.percentile(e_vals, 2),
                "pct1": np.percentile(e_vals, 1),
                "pct05": np.percentile(e_vals, 0.5),
                "pct01": np.percentile(e_vals, 0.1),
                "pct001": np.percentile(e_vals, 0.01),
                "pct0001": np.percentile(e_vals, 0.001),
                "pct00001": np.percentile(e_vals, 0.0001),
                "n_pass_20pct_filter": n_pass_20,
                "pct_pass_20pct_filter": n_pass_20 / n_seqs * 100,
                "n_pass_10pct_filter": n_pass_10,
                "pct_pass_10pct_filter": n_pass_10 / n_seqs * 100,
                "n_pass_5pct_filter": n_pass_5,
                "pct_pass_5pct_filter": n_pass_5 / n_seqs * 100,
            }
        )

        print(
            f"  {disease:25s}: {n_specimens:3d} specimens, {n_seqs:>9,} seqs, "
            f"median={e_vals.median():.4f}, "
            f"pass 20%: {n_pass_20:>6,} ({n_pass_20 / n_seqs * 100:.4f}%), "
            f"pass 10%: {n_pass_10:>6,} ({n_pass_10 / n_seqs * 100:.4f}%), "
            f"pass 5%: {n_pass_5:>6,} ({n_pass_5 / n_seqs * 100:.4f}%)"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"entropy_summary_fold{args.fold_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path.name}")

    # --- Percentile sequence counts (MD summary) ---
    percentile_levels = [5, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]
    e_all = entropy_df["entropy"].values
    n_total = len(e_all)

    md_lines = [
        f"# Entropy Percentile Summary - Fold {args.fold_id}",
        "",
        f"- **Total sequences**: {n_total:,}",
        f"- **n_classes**: {n_classes}",
        f"- **Max entropy**: {max_entropy:.4f} nats = ln({n_classes})",
        "",
        "## Overall (all diseases)",
        "",
        "| Percentile | Entropy threshold | # sequences at or below | % of total |",
        "|---|---|---|---|",
    ]
    for pct in percentile_levels:
        thresh = np.percentile(e_all, pct)
        n_below = int((e_all <= thresh).sum())
        md_lines.append(
            f"| bottom {pct}% | {thresh:.6f} | {n_below:,} | {n_below / n_total * 100:.4f}% |"
        )

    # Per-disease breakdown
    for disease in diseases:
        d_vals = entropy_df.loc[entropy_df["disease"] == disease, "entropy"].values
        n_d = len(d_vals)
        md_lines.append("")
        md_lines.append(f"## {disease} ({n_d:,} sequences)")
        md_lines.append("")
        md_lines.append(
            "| Percentile | Entropy threshold | # sequences at or below | % of disease total |"
        )
        md_lines.append("|---|---|---|---|")
        for pct in percentile_levels:
            thresh = np.percentile(d_vals, pct)
            n_below = int((d_vals <= thresh).sum())
            md_lines.append(
                f"| bottom {pct}% | {thresh:.6f} | {n_below:,} | {n_below / n_d * 100:.4f}% |"
            )

    md_path = output_dir / f"entropy_percentiles_fold{args.fold_id}.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved percentile summary: {md_path.name}")

    # --- Generate combined joyplot ---
    print(f"\n{'=' * 70}")
    print(f"GENERATING JOYPLOT")
    print(f"{'=' * 70}")

    plot_combined_joyplot(
        entropy_df=entropy_df,
        n_classes=n_classes,
        output_dir=output_dir,
        entropy_col="entropy",
    )

    print(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
