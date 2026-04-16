"""Test multiclass Model 3 with mean aggregation instead of entropy_twenty_percent_cutoff.

Hypothesis: The entropy filter kills all signal for multiclass OvR because 6 independent
~0.5 probabilities look uniform after scipy.stats.entropy's internal normalization.
Mean aggregation should preserve the signal that binary models successfully exploit.

Approach:
  1. Load existing Stage 1 models (identical for any aggregation strategy)
  2. On ts2: generate sequence predictions, aggregate with mean, train Stage 2
  3. On test: generate sequence predictions, aggregate with mean, predict and evaluate
  4. Compare results with the original entropy-filtered results

For each fold: reuses the same Stage 1 models, only changes the aggregation strategy.

Output: scripts/dev/output/test_mean_aggregation/
  fold_{id}_results.json   — per-fold evaluation metrics
  fold_{id}_predictions.pkl — raw predictions
  summary.json             — cross-fold summary

Run: python scripts/dev/test_mean_aggregation.py
     python scripts/dev/test_mean_aggregation.py --fold-ids 0  # single fold
"""
import sys, gc, pickle, json, time, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from functools import partial
import pyarrow.parquet as pq
print = partial(print, flush=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from malid_lite.models.model3_sequence_level import (
    SequenceLevelClassifier,
    AggregationStrategy,
)
from malid_lite.training.training_utils import split_train_smaller

# --- Paths (all relative to project root) ---
EMBEDDING_DIR = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "embeddings"
MODEL_DIR = PROJECT_ROOT / "trained_models" / "mal-id-orig-data" / "model3" / "multiclass" / "TCR"
FOLD_DIR = PROJECT_ROOT / "cache" / "mal-id-orig-data" / "data_folds"
OUTPUT_DIR = Path(__file__).parent / "output" / "test_mean_aggregation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEEDED_COLS = [
    "participant_label", "repertoire_id", "v_gene",
    "igh_or_tcrb_clone_id", "isotype_supergroup",
]
SPECIMEN_COL = "repertoire_id"
DISEASE_COL = "disease"


# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Test multiclass Model 3 with mean aggregation")
parser.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2],
                    help="Which folds to process (default: 0 1 2)")
args = parser.parse_args()


def load_fold_data(fold_id, fold_label):
    """Load fold sequence data + metadata.

    Returns (sequences_df, metadata_df). Only loads columns needed for
    embedding alignment and Stage 1/2 processing.
    """
    seq_path = FOLD_DIR / f"fold_{fold_id}_{fold_label}_downsampled_sequences.parquet"
    meta_path = FOLD_DIR / f"fold_{fold_id}_{fold_label}_downsampled_metadata.csv"
    all_cols = pq.ParquetFile(seq_path).schema.names
    cols_to_load = [c for c in NEEDED_COLS if c in all_cols]
    if "amplification_label" in all_cols:
        cols_to_load.append("amplification_label")
    seq = pd.read_parquet(seq_path, columns=cols_to_load)
    meta = pd.read_csv(meta_path)
    disease_map = meta.set_index("specimen_label")["disease"]
    seq["disease"] = seq["repertoire_id"].map(disease_map)
    seq = seq.dropna(subset=["disease"])
    if "specimen_label" not in seq.columns:
        seq["specimen_label"] = seq["repertoire_id"]
    return seq, meta


def load_embeddings_for_split(split_df):
    """Load and align pre-computed embeddings for all participants in the split.

    Returns (embeddings_array, split_df_aligned) where rows are aligned 1:1.
    """
    participants = split_df["participant_label"].unique()
    emb_list = []
    idx_list = []  # row indices in split_df that we successfully matched

    for pi, participant in enumerate(participants, 1):
        if pi % 50 == 0 or pi == 1:
            print(f"    Loading embeddings: participant {pi}/{len(participants)}")

        p_mask = split_df["participant_label"].values == participant
        p_split_df = split_df.loc[p_mask]

        p_emb = np.load(str(EMBEDDING_DIR / f"{participant}_embeddings.npy")).astype(np.float32)
        p_parquet = pd.read_parquet(EMBEDDING_DIR / f"{participant}_downsampled.parquet")

        key_cols = ["repertoire_id", "igh_or_tcrb_clone_id", "isotype_supergroup"]
        if "amplification_label" in p_parquet.columns:
            key_cols.append("amplification_label")

        def resolve(df, col):
            if col in df.columns: return col
            if col == "specimen_label" and "repertoire_id" in df.columns: return "repertoire_id"
            if col == "repertoire_id" and "specimen_label" in df.columns: return "specimen_label"
            raise KeyError(f"{col} not in {list(df.columns)[:10]}")

        # Build lookup from pre-computed parquet -> embedding index
        pre_keys = [
            tuple(str(x) for x in t)
            for t in zip(*(p_parquet[resolve(p_parquet, c)].values for c in key_cols))
        ]
        key_to_idx = {k: i for i, k in enumerate(pre_keys)}

        # Match split_df rows to embedding indices
        fold_keys = [
            tuple(str(x) for x in t)
            for t in zip(*(p_split_df[resolve(p_split_df, c)].values for c in key_cols))
        ]
        indices = [key_to_idx[k] for k in fold_keys]
        emb_list.append(p_emb[indices])
        idx_list.extend(p_split_df.index.tolist())

        del p_emb, p_parquet

    embeddings = np.concatenate(emb_list, axis=0)
    del emb_list
    # Reorder split_df to match the embedding order
    split_df_aligned = split_df.loc[idx_list].reset_index(drop=True)
    assert len(split_df_aligned) == len(embeddings), (
        f"Alignment mismatch: {len(split_df_aligned)} rows vs {len(embeddings)} embeddings"
    )
    return embeddings, split_df_aligned


def build_model_from_stage1(stage1_data, aggregation_strategy):
    """Build a SequenceLevelClassifier and inject the loaded Stage 1 models.

    This creates a model object with the desired aggregation strategy,
    then sets Stage 1 attributes from the saved artifact so we can
    proceed directly to Stage 2 training.
    """
    group_models = stage1_data["group_models"]
    meta = stage1_data.get("_meta", {})
    classes = np.array(meta.get("classes", list(next(iter(group_models.values())).classes_)))

    model = SequenceLevelClassifier(
        locus="TCR",
        aggregation_strategy=aggregation_strategy,
        exclude_rare_v_genes=True,
        reweigh_by_subset_frequencies=True,
        n_estimators_stage2=100,
        n_jobs=4,
        verbose=1,
    )

    # Inject Stage 1 state
    model.group_models_ = group_models
    model.classes_ = classes

    # Reconstruct non_rare_v_genes_ from group_models keys
    model.non_rare_v_genes_ = sorted(set(gk[0] for gk in group_models.keys()))

    return model


def evaluate_predictions(y_true, y_pred, y_proba, classes):
    """Compute evaluation metrics: accuracy, AUROC, AUPRC, log loss, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, log_loss, confusion_matrix,
        roc_auc_score, average_precision_score,
    )

    results = {
        "n_scored": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classes": list(classes),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    results["confusion_matrix"] = cm.tolist()

    # Log loss
    try:
        results["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    except Exception as e:
        print(f"    Warning: log_loss failed: {e}")
        results["log_loss"] = None

    # Per-class AUROC (OvR)
    auroc_per_class = {}
    for i, cls in enumerate(classes):
        y_bin = (np.array(y_true) == cls).astype(int)
        if len(np.unique(y_bin)) < 2:
            auroc_per_class[cls] = None
            continue
        try:
            auroc_per_class[cls] = float(roc_auc_score(y_bin, y_proba[:, i]))
        except Exception:
            auroc_per_class[cls] = None
    results["auroc_ovr_per_class"] = auroc_per_class

    # Weighted AUROC (macro average over classes with valid AUROC)
    valid_aurocs = [v for v in auroc_per_class.values() if v is not None]
    results["auroc_ovr_weighted"] = float(np.mean(valid_aurocs)) if valid_aurocs else None

    # Per-class AUPRC
    auprc_per_class = {}
    for i, cls in enumerate(classes):
        y_bin = (np.array(y_true) == cls).astype(int)
        if y_bin.sum() == 0:
            auprc_per_class[cls] = None
            continue
        try:
            auprc_per_class[cls] = float(average_precision_score(y_bin, y_proba[:, i]))
        except Exception:
            auprc_per_class[cls] = None
    results["auprc_ovr_per_class"] = auprc_per_class

    valid_auprcs = [v for v in auprc_per_class.values() if v is not None]
    results["auprc_ovr_weighted"] = float(np.mean(valid_auprcs)) if valid_auprcs else None

    return results


# --- Main ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
all_results = []

for fold_id in args.fold_ids:
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id}")
    print(f"{'='*60}")
    t_fold = time.monotonic()

    # --- Load Stage 1 model ---
    print(f"  Loading Stage 1 model...")
    stage1_path = MODEL_DIR / f"fold_{fold_id}_stage1.pkl"
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found: {stage1_path}\n"
            f"Run the full training pipeline first to generate Stage 1 models."
        )
    with open(stage1_path, "rb") as f:
        stage1_data = pickle.load(f)

    # Build model with MEAN aggregation (the key change)
    model = build_model_from_stage1(stage1_data, AggregationStrategy.mean)
    classes = list(model.classes_)
    print(f"  Classes: {classes}")
    print(f"  V-gene groups: {len(model.group_models_)}")
    del stage1_data; gc.collect()

    # --- Load ts2 (Stage 2 training data) ---
    print(f"  Loading train data for ts1/ts2 split...")
    train_seq, train_meta = load_fold_data(fold_id, "train")
    ts1_df, ts2_df = split_train_smaller(train_seq, train_meta)
    del train_seq, ts1_df, train_meta; gc.collect()
    print(f"  ts2: {len(ts2_df):,} seqs, {ts2_df['participant_label'].nunique()} participants, "
          f"{ts2_df[SPECIMEN_COL].nunique()} specimens")

    # --- Load ts2 embeddings ---
    print(f"  Loading ts2 embeddings...")
    t0 = time.monotonic()
    ts2_emb, ts2_df_aligned = load_embeddings_for_split(ts2_df)
    del ts2_df; gc.collect()
    print(f"  ts2 embeddings: {ts2_emb.shape} [{time.monotonic() - t0:.1f}s]")

    # --- Train Stage 2 with mean aggregation ---
    print(f"  Training Stage 2 (mean aggregation)...")
    t_s2 = time.monotonic()
    model.fit_stage2(ts2_df_aligned, ts2_emb)
    print(f"  Stage 2 trained in {time.monotonic() - t_s2:.1f}s")
    del ts2_emb, ts2_df_aligned; gc.collect()

    # --- Load test data ---
    print(f"  Loading test data...")
    test_seq, test_meta = load_fold_data(fold_id, "test")
    print(f"  test: {len(test_seq):,} seqs, {test_seq['participant_label'].nunique()} participants, "
          f"{test_seq[SPECIMEN_COL].nunique()} specimens")

    # --- Load test embeddings ---
    print(f"  Loading test embeddings...")
    t0 = time.monotonic()
    test_emb, test_df_aligned = load_embeddings_for_split(test_seq)
    del test_seq; gc.collect()
    print(f"  test embeddings: {test_emb.shape} [{time.monotonic() - t0:.1f}s]")

    # --- Predict on test: get specimen-level probabilities ---
    print(f"  Predicting on test set...")
    t0 = time.monotonic()
    proba_df = model.predict_proba(test_df_aligned, test_emb)
    del test_emb, test_df_aligned; gc.collect()
    print(f"  predict_proba done [{time.monotonic() - t0:.1f}s]")

    # --- Derive y_true from metadata, aligned to proba_df specimen order ---
    specimen_disease = (
        test_meta.drop_duplicates("specimen_label")
        .set_index("specimen_label")["disease"]
    )
    missing = [s for s in proba_df.index if s not in specimen_disease.index]
    if missing:
        raise ValueError(
            f"Test specimens missing disease labels: {missing[:10]}. "
            f"This indicates a data or metadata alignment bug."
        )
    y_true = np.array([specimen_disease[s] for s in proba_df.index])
    y_proba = proba_df.values
    y_pred = np.array(classes)[np.argmax(y_proba, axis=1)]
    scored_specimens = list(proba_df.index)

    print(f"  Predictions: {len(y_pred)} specimens scored")
    print(f"  y_proba range per class:")
    for i, cls in enumerate(classes):
        col = y_proba[:, i]
        print(f"    {cls}: [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}")

    # --- Evaluate ---
    eval_results = evaluate_predictions(y_true, y_pred, y_proba, classes)
    eval_results["fold_id"] = fold_id

    print(f"\n  === Results (fold {fold_id}, mean aggregation) ===")
    print(f"  Accuracy:          {eval_results['accuracy']:.4f}")
    if eval_results['auroc_ovr_weighted'] is not None:
        print(f"  AUROC (weighted):  {eval_results['auroc_ovr_weighted']:.4f}")
    else:
        print(f"  AUROC (weighted):  None (constant predictions)")
    if eval_results['auprc_ovr_weighted'] is not None:
        print(f"  AUPRC (weighted):  {eval_results['auprc_ovr_weighted']:.4f}")
    else:
        print(f"  AUPRC (weighted):  None")
    if eval_results['log_loss'] is not None:
        print(f"  Log loss:          {eval_results['log_loss']:.4f}")
    print(f"  Per-class AUROC:")
    for cls, auc in eval_results["auroc_ovr_per_class"].items():
        print(f"    {cls}: {auc:.4f}" if auc is not None else f"    {cls}: None")
    print(f"  Confusion matrix:")
    cm = np.array(eval_results["confusion_matrix"])
    # Header row: predicted class labels
    print(f"    {'Predicted ->':>22s}  " + "  ".join(f"{c[:8]:>8s}" for c in classes))
    for i, cls in enumerate(classes):
        print(f"    {cls:>22s}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(len(classes))))

    # --- Save fold results ---
    results_path = OUTPUT_DIR / f"fold_{fold_id}_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Saved: {results_path}")

    preds_path = OUTPUT_DIR / f"fold_{fold_id}_predictions.pkl"
    with open(preds_path, "wb") as f:
        pickle.dump({
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_true": y_true,
            "scored_specimens": scored_specimens,
            "classes": classes,
        }, f)
    print(f"  Saved: {preds_path}")

    all_results.append(eval_results)
    elapsed = time.monotonic() - t_fold
    print(f"\n  Fold {fold_id} completed in {elapsed:.1f}s")

    del model, test_meta, proba_df; gc.collect()


# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY: Mean aggregation vs Entropy filter")
print(f"{'='*60}")

summary = {
    "timestamp": timestamp,
    "aggregation_strategy": "mean",
    "fold_results": all_results,
}

# Compute cross-fold averages
accs = [r["accuracy"] for r in all_results]
aurocs = [r["auroc_ovr_weighted"] for r in all_results if r["auroc_ovr_weighted"] is not None]
auprcs = [r["auprc_ovr_weighted"] for r in all_results if r["auprc_ovr_weighted"] is not None]

print(f"\nMean aggregation results:")
print(f"  Per-fold accuracy:  {[f'{a:.4f}' for a in accs]}")
print(f"  Mean accuracy:      {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
if aurocs:
    print(f"  Per-fold AUROC:     {[f'{a:.4f}' for a in aurocs]}")
    print(f"  Mean AUROC:         {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
else:
    print(f"  AUROC: None (all folds had constant predictions)")
if auprcs:
    print(f"  Per-fold AUPRC:     {[f'{a:.4f}' for a in auprcs]}")
    print(f"  Mean AUPRC:         {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")

summary["accuracy_mean"] = float(np.mean(accs))
summary["accuracy_std"] = float(np.std(accs))
summary["auroc_mean"] = float(np.mean(aurocs)) if aurocs else None
summary["auprc_mean"] = float(np.mean(auprcs)) if auprcs else None

# Compare with original entropy-filtered results (find the summary file dynamically)
orig_summaries = sorted(MODEL_DIR.glob("summary_*.json"))
if orig_summaries:
    original_path = orig_summaries[-1]  # most recent
    print(f"\nComparison with original ({original_path.name}):")
    with open(original_path) as f:
        orig = json.load(f)
    orig_accs = orig["aggregated_by_pair"]["multiclass"]["model3"]["accuracy_per_fold"]["per_fold"]
    print(f"  Original (entropy filter) per-fold accuracy: {[f'{a:.4f}' for a in orig_accs]}")
    print(f"  Original mean accuracy: {np.mean(orig_accs):.4f}")
    summary["original_accuracy_per_fold"] = orig_accs
    summary["original_summary_file"] = original_path.name
else:
    print(f"\n  No original summary found in {MODEL_DIR} for comparison.")

summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {OUTPUT_DIR}/")
print("Done.")
