"""Metrics, curves, figures, and report writing for external evaluation (Phase 6).

Kept separate from the orchestration in ``evaluate_external.py`` so the (large)
reporting surface — comprehensive per-model metrics, ROC/PR curve data + figures,
confusion matrices, and the human-readable RESULTS.md — is self-contained.

Everything downstream is driven by the standard prediction arrays
(``y_true``, ``y_pred``, ``y_proba``, ``classes``) that the orchestration already
computes for each base model and the ensemble.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from malid_lite.training.train_ensemble import evaluate_predictions

logger = logging.getLogger(__name__)

FIGURE_DPI = 600  # per project convention for saved figures


# ===========================================================================
# Metrics
# ===========================================================================


def compute_rich_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    *,
    reference_class: Optional[str],
    model_label: str,
    n_scored: int,
    n_abstained: int,
) -> Dict[str, Any]:
    """Compute a comprehensive metrics dict for one model's predictions.

    Includes the standard metrics (via the shared ``evaluate_predictions``:
    accuracy, AUROC ovo/ovr, AUPRC, MCC, log-loss, abstention) plus: a per-class
    precision/recall/F1/support report, the confusion matrix (counts +
    row-normalized), the top confusions, per-class OvR ROC + PR curve arrays
    (with AUC/AP), and — for binary — a sensitivity/specificity operating point.

    The curve ARRAYS are included here (for figures + CSVs); a compact scalar-only
    view is written to results.json via ``compact_curves_for_json``.
    """
    classes = [str(c) for c in classes]
    y_true = np.array([str(v) for v in y_true])
    y_pred = np.array([str(v) for v in y_pred])

    metrics, _ = evaluate_predictions(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba, classes=np.array(classes),
        fold_id=0, model_label=model_label, n_scored=n_scored,
        n_abstained=n_abstained, reference_class=reference_class,
    )

    # Balanced accuracy (mean per-class recall) — not in evaluate_predictions
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # Per-class precision / recall / F1 / support
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0,
    )

    # Confusion matrix (counts already in `metrics`; add row-normalized + top confusions)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with np.errstate(all="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()
    metrics["top_confusions"] = _top_confusions(cm, classes)

    # Per-class OvR ROC + PR curves (only where both a positive and a negative exist)
    roc_curves: Dict[str, Any] = {}
    pr_curves: Dict[str, Any] = {}
    for col, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(y_bin.sum())
        if n_pos == 0 or n_pos == len(y_bin):
            continue  # AUROC/AP undefined without both classes present
        score = y_proba[:, col]
        fpr, tpr, roc_thr = roc_curve(y_bin, score)
        prec, rec, pr_thr = precision_recall_curve(y_bin, score)
        roc_curves[c] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thr.tolist(),
            "auc": float(roc_auc_score(y_bin, score)),
        }
        pr_curves[c] = {
            "precision": prec.tolist(), "recall": rec.tolist(),
            "thresholds": pr_thr.tolist(), "ap": float(average_precision_score(y_bin, score)),
        }
    metrics["roc_curves"] = roc_curves
    metrics["pr_curves"] = pr_curves

    # Binary operating point (sensitivity / specificity / Youden's J / threshold)
    if reference_class is not None:
        op = _binary_operating_point(y_true, y_proba, classes, reference_class)
        if op is not None:
            metrics["binary_operating_point"] = op

    return metrics


def _top_confusions(cm: np.ndarray, classes: List[str], top_n: int = 5) -> List[Dict]:
    """Most-frequent off-diagonal (true -> predicted) confusions, descending."""
    pairs = [
        {"true": classes[i], "predicted": classes[j], "count": int(cm[i, j])}
        for i in range(len(classes)) for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_n]


def _binary_operating_point(
    y_true: np.ndarray, y_proba: np.ndarray, classes: List[str], reference_class: str,
) -> Optional[Dict]:
    """Sensitivity/specificity at the Youden-optimal threshold and at 0.5.

    The positive class is the single non-reference disease. Returns None if the
    positive class has no support (can't define an operating point).
    """
    positives = [c for c in classes if c != reference_class]
    if len(positives) != 1:
        return None
    disease = positives[0]
    col = classes.index(disease)
    y_bin = (y_true == disease).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return None
    score = y_proba[:, col]
    fpr, tpr, thr = roc_curve(y_bin, score)
    youden = tpr - fpr
    best = int(np.argmax(youden))

    def _sens_spec_at(threshold):
        pred = (score >= threshold).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        tn = int(((pred == 0) & (y_bin == 0)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        return (
            tp / (tp + fn) if (tp + fn) else None,
            tn / (tn + fp) if (tn + fp) else None,
        )

    sens_05, spec_05 = _sens_spec_at(0.5)
    return {
        "positive_class": disease,
        "reference_class": reference_class,
        "youden_optimal": {
            "threshold": float(thr[best]),
            "sensitivity": float(tpr[best]),
            "specificity": float(1.0 - fpr[best]),
            "youden_j": float(youden[best]),
        },
        "at_threshold_0.5": {"sensitivity": sens_05, "specificity": spec_05},
    }


def compact_curves_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a metrics dict with curve ARRAYS replaced by scalar AUC/AP.

    The full curve arrays go to the per-model CSVs (for re-plotting); results.json
    keeps only the per-class AUC/AP so it stays readable.
    """
    out = dict(metrics)
    if "roc_curves" in out:
        out["roc_auc_per_class"] = {c: d["auc"] for c, d in out.pop("roc_curves").items()}
    if "pr_curves" in out:
        out["pr_ap_per_class"] = {c: d["ap"] for c, d in out.pop("pr_curves").items()}
    return out


# ===========================================================================
# Curve CSVs (numbers for figure reproducibility)
# ===========================================================================


def save_curve_csvs(curves_dir: Path, model_label: str, metrics: Dict[str, Any]) -> None:
    """Write the ROC and PR curve arrays for one model as tidy CSVs.

    roc_<model>.csv: class, fpr, tpr, threshold.  pr_<model>.csv: class, recall,
    precision, threshold. (PR's precision/recall arrays are length N+1 while its
    thresholds are length N — the final point has no threshold, filled with NaN.)
    """
    curves_dir.mkdir(parents=True, exist_ok=True)
    roc = metrics.get("roc_curves", {})
    if roc:
        rows = []
        for c, d in roc.items():
            for fpr, tpr, thr in zip(d["fpr"], d["tpr"], d["thresholds"]):
                rows.append({"class": c, "fpr": fpr, "tpr": tpr, "threshold": thr})
        pd.DataFrame(rows).to_csv(curves_dir / f"roc_{model_label}.csv", index=False)
    pr = metrics.get("pr_curves", {})
    if pr:
        rows = []
        for c, d in pr.items():
            thr = list(d["thresholds"]) + [float("nan")]  # precision/recall have one extra point
            for prec, rec, t in zip(d["precision"], d["recall"], thr):
                rows.append({"class": c, "recall": rec, "precision": prec, "threshold": t})
        pd.DataFrame(rows).to_csv(curves_dir / f"pr_{model_label}.csv", index=False)


# ===========================================================================
# Figures (PNG, DPI 600)
# ===========================================================================


def save_model_figures(
    fig_dir: Path, model_label: str, metrics: Dict[str, Any], classes: List[str],
) -> None:
    """Save ROC, PR, and confusion-matrix figures for one model."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    roc = metrics.get("roc_curves", {})
    if roc:
        fig, ax = plt.subplots(figsize=(5, 5))
        for c, d in roc.items():
            ax.plot(d["fpr"], d["tpr"], label=f"{c} (AUC={d['auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC — {model_label}", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"roc_{model_label}.png", dpi=FIGURE_DPI)
        plt.close(fig)

    pr = metrics.get("pr_curves", {})
    if pr:
        fig, ax = plt.subplots(figsize=(5, 5))
        for c, d in pr.items():
            ax.plot(d["recall"], d["precision"], label=f"{c} (AP={d['ap']:.4f})")
        ax.set(xlabel="Recall", ylabel="Precision",
               title=f"Precision–Recall — {model_label}", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"pr_{model_label}.png", dpi=FIGURE_DPI)
        plt.close(fig)

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        cm = np.array(cm)
        cm_norm = np.array(metrics.get("confusion_matrix_normalized", cm))
        # Use THIS model's own confusion-matrix labels — a base model may predict over
        # a subset of the global training classes (e.g. Model 2 with heavy abstention),
        # so its matrix can be smaller than the full label space.
        cm_labels = metrics.get("confusion_matrix_labels") or classes
        n = cm.shape[0]
        cm_labels = list(cm_labels)[:n]
        fig, ax = plt.subplots(figsize=(1.5 + 0.9 * n, 1.5 + 0.9 * n))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set(xticks=range(n), yticks=range(n),
               xlabel="Predicted", ylabel="True",
               title=f"Confusion matrix — {model_label}")
        ax.set_xticklabels(cm_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cm_labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{int(cm[i, j])}\n{cm_norm[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_matrix_{model_label}.png", dpi=FIGURE_DPI)
        plt.close(fig)


def save_comparison_figure(
    fig_dir: Path, per_model_metrics: Dict[str, Dict], reference_class: Optional[str],
) -> None:
    """Save a bar chart comparing key metrics across models + the ensemble."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    auroc_key = "auroc_binary" if reference_class is not None else "auroc_ovo_weighted"
    labels = list(per_model_metrics.keys())
    metric_keys = [("accuracy", "Accuracy"), (auroc_key, "AUROC"), ("mcc", "MCC")]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(labels), 4))
    for i, (mk, mlabel) in enumerate(metric_keys):
        # Missing metrics -> NaN (matplotlib draws no bar), so a "not computed" metric
        # reads as an empty gap rather than a misleading genuine 0.0.
        vals = [per_model_metrics[m].get(mk) if per_model_metrics[m].get(mk) is not None
                else np.nan for m in labels]
        ax.bar(x + (i - 1) * width, vals, width, label=mlabel)
    ax.set(xticks=x, ylim=(0, 1.02), ylabel="Score", title="Model comparison")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_comparison.png", dpi=FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# RESULTS.md
# ===========================================================================


def write_results_md(path: Path, results: Dict[str, Any]) -> None:
    """Human-readable report: dataset tables, per-model metrics, confusions, figures."""
    ref = results["reference_class"]
    auroc_key = "auroc_binary" if ref is not None else "auroc_ovo_weighted"
    auprc_key = "auprc_binary" if ref is not None else "auprc_ovo_weighted"

    def _f(m, key):
        v = m.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"

    L = [
        f"# External evaluation — {results['pair']}",
        "",
        f"- **Train dataset:** `{results['train_dataset']}`  |  "
        f"**Test dataset:** `{results['test_dataset']}`",
        f"- **Gene locus:** {results['gene_locus']}  |  **Mode:** {results['classification_mode']}",
        f"- **Training label space:** {results['training_classes']}",
        f"- **Model 2 abstention strategy:** {results['model2_abstention_strategy']}",
        "",
        "## Datasets",
        "",
        "### Training data (per class: participants / specimens)",
        _counts_table(results.get("training_data_counts")),
        "",
        "### Test data (per class: participants / specimens)",
        _counts_table(results.get("test_data_counts")),
        "",
        "### Class alignment",
        f"- Test classes: {results['class_alignment']['test_classes']}",
        f"- Extra test classes (not in the model; dropped if --allow-unknown-test-classes): "
        f"{results['class_alignment']['extra_test_classes']}",
        f"- Training classes with no test support: "
        f"{results['class_alignment']['missing_training_classes']}",
        f"- Test specimens evaluated: {results['n_test_specimens']}",
        "",
        "## Metrics",
        "",
        "| Model | Accuracy | Bal. acc | AUROC | AUPRC | MCC | Log-loss | Scored | Abstained |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    model_order = [f"model{n}" for n in results["models_included"]]
    if "ensemble" in results and results["ensemble"] is not None:
        model_order.append("ensemble")
    all_metrics = {**results.get("base_models", {})}
    if results.get("ensemble"):
        all_metrics["ensemble"] = results["ensemble"]

    for name in model_order:
        m = all_metrics.get(name)
        disp = "**Ensemble**" if name == "ensemble" else f"Model {name[-1]}"
        if m is None:
            L.append(f"| {disp} | (fully abstained / not evaluated) | | | | | | | |")
            continue
        L.append(
            f"| {disp} | {_f(m,'accuracy')} | {_f(m,'balanced_accuracy')} | {_f(m,auroc_key)} "
            f"| {_f(m,auprc_key)} | {_f(m,'mcc')} | {_f(m,'log_loss')} "
            f"| {m.get('n_scored','?')} | {m.get('n_abstained','?')} |"
        )

    # Per-model detail: classification report, top confusions, binary operating point
    for name in model_order:
        m = all_metrics.get(name)
        if m is None:
            continue
        disp = "Ensemble" if name == "ensemble" else f"Model {name[-1]}"
        L += ["", f"## {disp} — detail", "",
              "### Per-class precision / recall / F1 / support", ""]
        L += _classification_report_table(m.get("classification_report", {}), results["training_classes"])
        if m.get("top_confusions"):
            L += ["", "### Top confusions (true → predicted)", ""]
            L += [f"- {c['true']} → {c['predicted']}: {c['count']}" for c in m["top_confusions"]]
        if m.get("binary_operating_point"):
            op = m["binary_operating_point"]
            yo = op["youden_optimal"]
            L += ["", "### Binary operating point",
                  f"- Positive class: **{op['positive_class']}** (vs {op['reference_class']})",
                  f"- Youden-optimal: threshold={yo['threshold']:.4f}, "
                  f"sensitivity={yo['sensitivity']:.4f}, specificity={yo['specificity']:.4f} "
                  f"(J={yo['youden_j']:.4f})",
                  f"- At threshold 0.5: sensitivity="
                  f"{_fmt(op['at_threshold_0.5']['sensitivity'])}, specificity="
                  f"{_fmt(op['at_threshold_0.5']['specificity'])}"]

    L += ["", "## Figures",
          "Per-model ROC / PR / confusion-matrix PNGs are in `figures/`; the raw curve "
          "numbers are in `curves/` (for re-plotting). Model comparison: "
          "`figures/metrics_comparison.png`.", ""]
    path.write_text("\n".join(L))


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"


def _counts_table(counts: Optional[Dict]) -> str:
    """Render a per-class participants/specimens table from get_metadata_class_counts.

    ``get_metadata_class_counts`` returns
    ``{participants_per_class, specimens_per_class, total_participants, total_specimens}``.
    """
    if not counts:
        return "_(not available)_"
    ppc = counts.get("participants_per_class", {})
    spc = counts.get("specimens_per_class", {})
    lines = ["| Class | Participants | Specimens |", "|---|---|---|"]
    for cls in sorted(set(ppc) | set(spc)):
        lines.append(f"| {cls} | {ppc.get(cls, '?')} | {spc.get(cls, '?')} |")
    lines.append(
        f"| **Total** | {counts.get('total_participants', '?')} "
        f"| {counts.get('total_specimens', '?')} |"
    )
    return "\n".join(lines)


def _classification_report_table(report: Dict, classes: List[str]) -> List[str]:
    if not report:
        return ["_(not available)_"]
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls in classes:
        r = report.get(cls)
        if not isinstance(r, dict):
            continue
        lines.append(
            f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} "
            f"| {int(r['support'])} |"
        )
    for agg in ("macro avg", "weighted avg"):
        r = report.get(agg)
        if isinstance(r, dict):
            lines.append(
                f"| _{agg}_ | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} "
                f"| {int(r['support'])} |"
            )
    return lines
