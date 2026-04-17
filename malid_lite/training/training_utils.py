"""Shared utilities for Model 1 and Model 2 training scripts.

Constants, helper functions, and the classification-mode dispatch logic used
by both train_model1.py and train_model2.py.  Model-specific code (fold
loops, feature extraction, artifact saving) stays in each training script.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

DISEASE_COL = "disease"
SPECIMEN_COL = "specimen_label"      # specimen identifier in metadata DataFrames
PARTICIPANT_COL = "participant_label"
DEFAULT_DATASET_NAME = "mal-id-orig-data"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent


def make_pair_name(disease: str, reference: str) -> str:
    """Create a filesystem-safe directory name: '<disease>_vs_<reference>'.

    Spaces and slashes are replaced with underscores to ensure valid directory names.
    """
    def _safe(s: str) -> str:
        return s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{_safe(disease)}_vs_{_safe(reference)}"


def get_model_output_dir(
    model_name: str,
    dataset_name: str,
    classification_mode: str,
    gene_locus: str,
) -> Path:
    """Return the canonical base output directory for a trained model.

    Path pattern:
      trained_models/<dataset_name>/<model_name>/<mode_dir>/<gene_locus>/

    where <mode_dir> is "binary" for both binary and multi-binary modes, and
    equals <classification_mode> for all other modes (e.g. "multiclass").

    Individual pair artifacts for binary/multi-binary live one level deeper:
      <base>/<disease>_vs_<reference>/   (created by the training script)
    """
    mode_dir = "binary" if classification_mode in ("binary", "multi-binary") else classification_mode
    return PROJECT_ROOT / "trained_models" / dataset_name / model_name / mode_dir / gene_locus


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"


def get_dataset_disease_classes(metadata_path: Path) -> List[str]:
    """Return sorted list of all disease classes found in the metadata file."""
    meta = pd.read_csv(metadata_path, sep="\t", usecols=[DISEASE_COL])
    return sorted(meta[DISEASE_COL].dropna().unique().tolist())


def get_dataset_fold_ids(metadata_path: Path) -> List[int]:
    """Return sorted list of all fold IDs found in the metadata file."""
    meta = pd.read_csv(metadata_path, sep="\t", usecols=[FOLD_COL])
    return sorted(meta[FOLD_COL].dropna().unique().astype(int).tolist())


def validate_mode_and_classes(
    classification_mode: str,
    disease_classes: List[str],
    reference_class: Optional[str],
    diseases: Optional[List[str]] = None,
) -> Optional[str]:
    """Validate classification mode against available disease classes.

    Parameters
    ----------
    diseases : Explicit subset of disease classes. For binary mode, the 2-class
        data requirement is relaxed when diseases is provided. For multi-binary,
        only those diseases are trained. Ignored for multiclass.

    Returns the validated (or inferred) reference_class.
    Raises ValueError with a helpful message for incompatible combinations.
    """
    n = len(disease_classes)

    if classification_mode == "multiclass":
        if reference_class is not None:
            logger.warning(
                f"--reference-class '{reference_class}' is ignored in multiclass mode."
            )
        if n == 2:
            logger.info(
                f"multiclass mode with 2 classes {disease_classes} — "
                f"proceeding with binary multiclass training."
            )
        return reference_class

    elif classification_mode == "binary":
        if diseases is None and n != 2:
            raise ValueError(
                f"--classification-mode binary requires exactly 2 disease classes in the data, "
                f"but found {n}: {disease_classes}.\n\n"
                f"Options:\n"
                f"  --diseases <disease>   Pick one disease from this dataset explicitly.\n"
                f"  --classification-mode multi-binary --reference-class <ref>\n"
                f"      Trains one independent binary model per disease vs. <ref>.\n"
                f"  --classification-mode multiclass\n"
                f"      Trains a single {n}-class model over all disease classes."
            )
        if reference_class is not None and reference_class not in disease_classes:
            raise ValueError(
                f"--reference-class '{reference_class}' not found in data classes: {disease_classes}"
            )
        return reference_class

    elif classification_mode == "multi-binary":
        if reference_class is None:
            if n == 2:
                reference_class = sorted(disease_classes)[1]
                logger.warning(
                    f"multi-binary mode with 2 classes {disease_classes}: "
                    f"no --reference-class provided. "
                    f"Inferring reference as '{reference_class}' (alphabetical order). "
                    f"Use --reference-class to specify explicitly."
                )
            else:
                raise ValueError(
                    f"--reference-class is required for multi-binary mode "
                    f"with {n} disease classes.\n"
                    f"Data classes: {disease_classes}\n"
                    f"Example: --reference-class '<reference class name>'"
                )
        elif reference_class not in disease_classes:
            raise ValueError(
                f"--reference-class '{reference_class}' not found in data classes: {disease_classes}"
            )

        if diseases is None:
            diseases_to_train = [c for c in disease_classes if c != reference_class]
            n_models = len(diseases_to_train)
            if n_models == 1:
                logger.info(
                    f"multi-binary mode with 2 classes — will train one binary model: "
                    f"{make_pair_name(diseases_to_train[0], reference_class)}"
                )
            else:
                logger.info(
                    f"multi-binary mode: will train {n_models} binary models:\n"
                    + "\n".join(f"  {make_pair_name(d, reference_class)}" for d in diseases_to_train)
                )
        return reference_class

    else:
        raise ValueError(f"Unknown classification_mode: '{classification_mode}'")


def filter_to_binary_pair(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    disease: str,
    reference_class: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter data to participants in {disease, reference_class} only.

    Removes all other disease classes from both DataFrames. Filtering is done
    at the participant level — all sequences and metadata rows belonging to
    participants outside the two-class pool are removed.

    Under the one-disease-per-participant constraint enforced at data load time,
    participant-level and specimen-level filtering are equivalent.

    Parameters
    ----------
    sequences_df    : Sequences DataFrame with PARTICIPANT_COL.
    metadata_df     : Metadata DataFrame with PARTICIPANT_COL and DISEASE_COL.
    disease         : The target disease class (positive class).
    reference_class : The reference/negative class.

    Returns
    -------
    (filtered_sequences_df, filtered_metadata_df)
    """
    keep_participants = set(
        metadata_df.loc[
            metadata_df[DISEASE_COL].isin({disease, reference_class}), PARTICIPANT_COL
        ].unique()
    )
    return (
        sequences_df[sequences_df[PARTICIPANT_COL].isin(keep_participants)].copy(),
        metadata_df[metadata_df[PARTICIPANT_COL].isin(keep_participants)].copy(),
    )


# ---------------------------------------------------------------------------
# Train_smaller split
# ---------------------------------------------------------------------------

def split_train_smaller(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training fold into train_smaller1 (2/3) and train_smaller2 (1/3).

    Split is at the participant level, stratified by disease.
    random_state=0 ensures reproducibility (matches original Mal-ID design).

    Returns
    -------
    (train_smaller1_df, train_smaller2_df): each is a sequences DataFrame.
    """
    participant_disease = (
        metadata_df
        .drop_duplicates(subset=[PARTICIPANT_COL])
        [[PARTICIPANT_COL, DISEASE_COL]]
        .set_index(PARTICIPANT_COL)[DISEASE_COL]
    )

    train_participants = sequences_df[PARTICIPANT_COL].unique()
    participant_disease = participant_disease.reindex(train_participants).dropna()

    ts1_participants, ts2_participants = train_test_split(
        participant_disease.index.tolist(),
        test_size=1 / 3,
        stratify=participant_disease.values.tolist(),
        random_state=0,
    )

    ts1_set = set(ts1_participants)
    ts2_set = set(ts2_participants)

    train_smaller1 = sequences_df[sequences_df[PARTICIPANT_COL].isin(ts1_set)].copy()
    train_smaller2 = sequences_df[sequences_df[PARTICIPANT_COL].isin(ts2_set)].copy()

    return train_smaller1, train_smaller2


# ---------------------------------------------------------------------------
# Cross-fold aggregation
# ---------------------------------------------------------------------------

def _mean_std_per_fold(values: List, label: str) -> Dict:
    """Build a {mean, std, per_fold, n_folds_valid} dict over a list that may contain None.

    per_fold is always returned as-is (aligned with fold_ids, may contain None).
    mean/std are computed over non-None values only.
    A WARNING is logged if any values are None (either all-abstained fold or failed
    computation), so failures are never silent.
    """
    valid = [v for v in values if v is not None]
    n_missing = len(values) - len(valid)
    if n_missing > 0:
        logger.warning(
            f"  {label}: {n_missing}/{len(values)} fold(s) missing a value "
            f"(all specimens abstained or metric computation failed). "
            f"Mean/std computed over {len(valid)} fold(s) only."
        )
    return {
        "mean": float(np.mean(valid)) if valid else None,
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else (0.0 if len(valid) == 1 else None),
        "per_fold": values,
        "n_folds_valid": len(valid),
    }


def aggregate_fold_results(
    fold_metrics: List[Dict],
    fold_raw_preds: List[Optional[Dict]],
    disease_filter: Optional[Tuple[str, str]] = None,
) -> Dict:
    """Aggregate per-fold evaluation metrics across folds.

    Abstention handling (matching original Mal-ID / crosseval with_abstention=True):
    - Accuracy: abstentions count as wrong. Global accuracy = n_correct / (n_scored + n_abstained).
      Per-fold accuracy is already stored penalized in fold_metrics (including 0.0 for
      all-abstained folds). Matches crosseval: appending y_pred="Unknown" for each abstained
      specimen, so all abstentions are misclassifications.
    - AUROC / AUPRC: computed on scored specimens only (no predicted probability for abstentions).

    Per_fold alignment:
    - All per_fold lists have length == n_folds, aligned with fold_ids.
    - None entries indicate either all-abstained or failed metric computation.
    - A WARNING is logged whenever any per_fold value is None, so no failure is silent.

    For multiclass (disease_filter=None):
        - accuracy_global: n_correct / (n_scored + n_abstained), pooled across all folds
        - accuracy_per_fold, auroc_ovo_weighted, auprc_ovo_weighted, log_loss:
          each a {mean, std, per_fold, n_folds_valid} dict, per_fold aligned to all folds
        - auroc_ovr_per_class: per-class same structure
        - confusion_matrix_aggregated: summed over scored folds (no class for abstained)

    For binary (disease_filter provided):
        - auroc_pooled, auprc_pooled: disease-as-positive, all scored folds pooled
        - auroc_per_fold, auprc_per_fold: aligned to all folds, None for all-abstained
        - accuracy_global: penalized as above
        - confusion_matrix_aggregated: summed over scored folds

    Parameters
    ----------
    fold_metrics    : List of per-fold metric dicts from evaluate_on_test.
    fold_raw_preds  : Matching list; None entry for any fold that had zero scored specimens.
    disease_filter  : (disease, reference_class) if binary/multi-binary; None if multiclass.
    """
    n_folds = len(fold_metrics)

    scored_pairs = [
        (m, r) for m, r in zip(fold_metrics, fold_raw_preds) if r is not None
    ]
    if not scored_pairs:
        return {
            "n_folds": n_folds,
            "n_folds_scored": 0,
            "fold_ids": [m["fold_id"] for m in fold_metrics],
            "warning": "all folds had zero scored specimens",
        }

    scored_metrics, scored_raws = zip(*scored_pairs)

    n_correct_total = int(sum(
        (r["y_true"] == r["y_pred"]).sum() for r in scored_raws
    ))
    n_total = int(sum(m["n_scored"] + m["n_abstained"] for m in fold_metrics))
    accuracy_global = n_correct_total / n_total if n_total > 0 else 0.0

    classes = scored_raws[0]["classes"]
    cm_sum = sum(
        confusion_matrix(r["y_true"], r["y_pred"], labels=classes)
        for r in scored_raws
    )

    result = {
        "n_folds": n_folds,
        "n_folds_scored": len(scored_pairs),
        "fold_ids": [m["fold_id"] for m in fold_metrics],
        "accuracy_global": float(accuracy_global),
        "confusion_matrix_aggregated": cm_sum.tolist(),
        "classes": [str(c) for c in classes],
    }

    accuracy_per_fold = [m["accuracy"] for m in fold_metrics]

    if disease_filter is None:
        # --- Multiclass aggregation ---
        auroc_per_fold = [m.get("auroc_ovo_weighted") for m in fold_metrics]
        auprc_per_fold = [m.get("auprc_ovo_weighted") for m in fold_metrics]
        ll_per_fold    = [m.get("log_loss")            for m in fold_metrics]

        result["accuracy_per_fold"] = {
            "mean": float(np.mean(accuracy_per_fold)),
            "std": float(np.std(accuracy_per_fold, ddof=1)) if n_folds > 1 else 0.0,
            "per_fold": accuracy_per_fold,
            "n_folds_valid": n_folds,
        }
        result["auroc_ovo_weighted"] = _mean_std_per_fold(auroc_per_fold, "auroc_ovo_weighted")
        result["auprc_ovo_weighted"] = _mean_std_per_fold(auprc_per_fold, "auprc_ovo_weighted")
        result["log_loss"]           = _mean_std_per_fold(ll_per_fold,    "log_loss")

        str_classes = [str(c) for c in classes]
        auroc_ovr_per_class_aggregated = {}
        for cls in str_classes:
            per_fold_cls = [
                m["auroc_ovr_per_class"].get(cls) if m.get("auroc_ovr_per_class") else None
                for m in fold_metrics
            ]
            auroc_ovr_per_class_aggregated[cls] = _mean_std_per_fold(
                per_fold_cls, f"auroc_ovr_per_class[{cls}]"
            )
        result["auroc_ovr_per_class"] = auroc_ovr_per_class_aggregated

    else:
        # --- Binary aggregation ---
        disease, reference_class = disease_filter
        str_classes = [str(c) for c in classes]
        result["auroc_pooled"] = None
        result["auprc_pooled"] = None
        if str(reference_class) in str_classes:
            disease_idx = next(
                i for i, c in enumerate(str_classes) if c != str(reference_class)
            )
            y_binary_all = np.concatenate([
                (r["y_true"] == str_classes[disease_idx]).astype(int)
                for r in scored_raws
            ])
            y_score_all = np.concatenate([r["y_proba"][:, disease_idx] for r in scored_raws])
            try:
                result["auroc_pooled"] = float(roc_auc_score(y_binary_all, y_score_all))
            except ValueError as e:
                logger.warning(f"  Pooled AUROC failed: {e}")
            try:
                result["auprc_pooled"] = float(average_precision_score(y_binary_all, y_score_all))
            except ValueError as e:
                logger.warning(f"  Pooled AUPRC failed: {e}")
        else:
            logger.warning(
                f"  Binary aggregate: reference_class '{reference_class}' not found in "
                f"model classes {str_classes}. auroc_pooled and auprc_pooled set to None."
            )

        result["auroc_per_fold"] = [m.get("auroc_binary") for m in fold_metrics]
        result["auprc_per_fold"] = [m.get("auprc_binary") for m in fold_metrics]
        result["accuracy_per_fold"] = accuracy_per_fold
        result["disease"] = disease
        result["reference_class"] = reference_class

    return result


# ---------------------------------------------------------------------------
# Training orchestration (shared dispatch logic)
# ---------------------------------------------------------------------------

def run_training_orchestration(
    base_dir: Path,
    classification_mode: str,
    reference_class: Optional[str],
    diseases: Optional[List[str]],
    disease_classes: List[str],
    fold_loop_fn: Callable,
    loop_kwargs: Dict,
) -> Dict[str, Dict]:
    """Dispatch training across classification modes.

    Calls fold_loop_fn(output_dir=..., disease_filter=..., **loop_kwargs) for each
    pair or mode. Handles all three classification modes: multiclass, binary, and
    multi-binary.

    Parameters
    ----------
    base_dir            : Base output directory. Pair subdirectories are created
                          within it for binary and multi-binary modes.
    classification_mode : "multiclass" | "binary" | "multi-binary".
    reference_class     : Validated reference/negative class (returned by
                          validate_mode_and_classes). May be None for multiclass.
    diseases            : Explicit disease subset from CLI, or None.
    disease_classes     : All disease classes in the dataset (sorted).
    fold_loop_fn        : Model-specific fold loop function. Must accept
                          (output_dir, disease_filter, **loop_kwargs) and return
                          (fold_results, aggregated_by_model).
    loop_kwargs         : Keyword arguments forwarded verbatim to fold_loop_fn.
                          Typically includes loader, fold_ids, and model parameters.

    Returns
    -------
    Dict mapping pair key → {"fold_results": List[Dict], "aggregated_by_model": Dict[str, Dict]}.
    Keys: "multiclass" for multiclass; "<disease>_vs_<reference>" for each binary pair.
    """
    all_results: Dict[str, Dict] = {}

    def _store(key: str, fold_results: List[Dict], aggregated: Dict[str, Dict]) -> None:
        all_results[key] = {"fold_results": fold_results, "aggregated_by_model": aggregated}

    if classification_mode == "multiclass":
        fold_results, aggregated = fold_loop_fn(
            output_dir=base_dir,
            disease_filter=None,
            **loop_kwargs,
        )
        _store("multiclass", fold_results, aggregated)

    elif classification_mode == "binary":
        # Resolve which disease and reference to use.
        if diseases is not None:
            if len(diseases) != 1:
                raise ValueError(
                    f"binary mode requires exactly one entry in --diseases, "
                    f"got {len(diseases)}: {diseases}. "
                    f"For multiple diseases use --classification-mode multi-binary."
                )
            disease = diseases[0]
            if disease not in disease_classes:
                raise ValueError(
                    f"--diseases '{disease}' not found in data: {disease_classes}"
                )
            if reference_class is not None:
                if disease == reference_class:
                    raise ValueError(
                        f"--diseases '{disease}' is the same as --reference-class '{reference_class}'"
                    )
                ref = reference_class
            else:
                others = [c for c in disease_classes if c != disease]
                if len(others) > 1:
                    raise ValueError(
                        f"--reference-class is required when using --diseases with N-class data. "
                        f"Data classes (excluding '{disease}'): {others}"
                    )
                ref = others[0]
                logger.info(
                    f"binary mode: no --reference-class provided. "
                    f"Inferred reference as '{ref}' (only other class). "
                    f"Use --reference-class to specify explicitly."
                )
        else:
            # No explicit disease: data must have exactly 2 classes (validated above).
            if reference_class is not None:
                disease = next(c for c in disease_classes if c != reference_class)
                ref = reference_class
            else:
                disease, ref = sorted(disease_classes)
                logger.info(
                    f"binary mode: no --reference-class provided. "
                    f"Using '{disease}' as disease and '{ref}' as reference (alphabetical). "
                    f"Use --reference-class to specify explicitly."
                )

        pair_name = make_pair_name(disease, ref)
        fold_results, aggregated = fold_loop_fn(
            output_dir=base_dir / pair_name,
            disease_filter=(disease, ref),
            **loop_kwargs,
        )
        _store(pair_name, fold_results, aggregated)

    elif classification_mode == "multi-binary":
        # Resolve which diseases to train.
        if diseases is not None:
            invalid = [d for d in diseases if d not in disease_classes]
            if invalid:
                raise ValueError(
                    f"--diseases {invalid} not found in data: {disease_classes}"
                )
            if reference_class is not None:
                as_ref = [d for d in diseases if d == reference_class]
                if as_ref:
                    raise ValueError(
                        f"--diseases includes reference class '{reference_class}'. "
                        f"Remove it from --diseases or change --reference-class."
                    )
            diseases_to_train = list(diseases)
            logger.info(
                f"multi-binary mode: training {len(diseases_to_train)} specified disease(s):\n"
                + "\n".join(f"  {make_pair_name(d, reference_class)}" for d in diseases_to_train)
            )
        else:
            diseases_to_train = [c for c in disease_classes if c != reference_class]

        for disease in diseases_to_train:
            pair_name = make_pair_name(disease, reference_class)
            logger.info(f"\n{'*'*60}")
            logger.info(f"Binary pair: {pair_name}")
            logger.info(f"{'*'*60}")
            fold_results, aggregated = fold_loop_fn(
                output_dir=base_dir / pair_name,
                disease_filter=(disease, reference_class),
                **loop_kwargs,
            )
            _store(pair_name, fold_results, aggregated)

    return all_results


# ---------------------------------------------------------------------------
# Per-pair results (binary / multi-binary)
# ---------------------------------------------------------------------------

def save_per_pair_results(
    base_dir: Path,
    all_results: Dict[str, Dict],
    classification_mode: str,
    timestamp: str,
    model_label: str,
    run_info: Dict,
    fold_ids: List[int],
    model_names: List[str],
    has_abstention: bool,
    summary_json_extra: Optional[Dict] = None,
) -> None:
    """Save per-pair summary JSON and results MD inside each pair subdirectory.

    Only applies to binary and multi-binary modes.  For multiclass this is a
    no-op (no pair subdirectories exist).

    Each pair directory receives:
      - summary_<timestamp>.json  — same structure as the top-level summary,
        scoped to this pair only.
      - RESULTS_<timestamp>.md    — human-readable results for this pair.

    Parameters
    ----------
    base_dir            : Top-level output directory (parent of pair subdirs).
    all_results         : Output of run_training_orchestration.
    classification_mode : "multiclass" | "binary" | "multi-binary".
    timestamp           : Run timestamp string.
    model_label         : Display name ("Model 1", "Model 2", "Model 3").
    run_info            : Ordered dict of key-value pairs for the MD header.
    fold_ids            : List of fold IDs that were trained.
    model_names         : Model variant names.
    has_abstention      : Whether to include abstention columns.
    summary_json_extra  : Model-specific fields to include in the JSON envelope
                          (e.g. aggregation_strategy for Model 3).
    """
    if classification_mode == "multiclass":
        return

    for pair_key, pair_data in all_results.items():
        pair_dir = base_dir / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-pair summary JSON ---
        pair_json = {
            "timestamp": timestamp,
            "pair": pair_key,
            "fold_ids": fold_ids,
            "model_names": model_names,
            "fold_results": pair_data["fold_results"],
            "aggregated_by_model": pair_data["aggregated_by_model"],
        }
        if summary_json_extra:
            pair_json.update(summary_json_extra)

        pair_summary_path = pair_dir / f"summary_{timestamp}.json"
        with open(pair_summary_path, "w") as f:
            json.dump(
                pair_json, f, indent=2,
                default=lambda x: (
                    x.tolist() if isinstance(x, np.ndarray)
                    else float(x) if isinstance(x, (np.floating, np.integer))
                    else x
                ),
            )

        # --- Per-pair results MD ---
        # Build a single-pair results dict and render as "binary" mode
        # so generate_results_md produces a clean single-pair report
        # (no multi-binary summary table).
        single_pair_results = {pair_key: pair_data}
        pair_run_info = dict(run_info)
        pair_run_info["Pair"] = pair_key.replace("_", " ")

        pair_md = generate_results_md(
            all_results=single_pair_results,
            classification_mode="binary",
            timestamp=timestamp,
            model_label=model_label,
            run_info=pair_run_info,
            fold_ids=fold_ids,
            model_names=model_names,
            has_abstention=has_abstention,
        )
        pair_md_path = pair_dir / f"RESULTS_{timestamp}.md"
        with open(pair_md_path, "w") as f:
            f.write(pair_md)

        logger.info(f"  Per-pair results saved to {pair_dir.name}/")


# ---------------------------------------------------------------------------
# Results Markdown generation
# ---------------------------------------------------------------------------

def _auroc_rating(auroc: Optional[float]) -> str:
    """Map AUROC to a qualitative rating."""
    if auroc is None:
        return "N/A"
    if auroc >= 0.960:
        return "Excellent"
    if auroc >= 0.920:
        return "Very Good"
    if auroc >= 0.880:
        return "Good"
    if auroc >= 0.800:
        return "Fair"
    return "Poor"


def _fv(val: Optional[float], fmt: str = ".3f") -> str:
    """Format a float or return 'N/A'."""
    return "N/A" if val is None else format(val, fmt)


def _fmt_conf_matrix(cm_list: List[List[int]], classes: List[str]) -> str:
    """Render a confusion matrix as a fenced code block."""
    n = len(classes)
    col_w = max(len(c) for c in classes) + 5
    row_label_w = max(len(c) for c in classes) + 2

    hdr = (
        " " * row_label_w
        + "  "
        + "  ".join(f"{c:>{col_w}}" for c in classes)
        + "  │ Total"
    )
    bar_len = row_label_w + 2 + n * col_w + 2 * n
    sep = "─" * bar_len + "┼──────"

    totals_col = [sum(cm_list[r][c] for r in range(n)) for c in range(n)]
    row_totals = [sum(cm_list[r]) for r in range(n)]
    grand_total = sum(row_totals)
    n_correct = sum(cm_list[i][i] for i in range(n))

    parts = ["```", "True Label → Predicted Label", "", hdr, sep]
    for i, cls in enumerate(classes):
        cells = "  ".join(f"{cm_list[i][j]:>{col_w}}" for j in range(n))
        parts.append(f"{cls:<{row_label_w}}  {cells}  │ {row_totals[i]:>5}")
    parts.append(sep)
    col_totals_str = "  ".join(f"{totals_col[c]:>{col_w}}" for c in range(n))
    parts.append(f"{'Total':<{row_label_w}}  {col_totals_str}  │ {grand_total:>5}")
    parts.append("")
    if grand_total > 0:
        parts.append(
            f"Diagonal (correct): {n_correct} / {grand_total} = "
            f"{n_correct / grand_total:.1%} accuracy"
        )
    parts.append("```")
    return "\n".join(parts)


def generate_results_md(
    all_results: Dict[str, Dict],
    classification_mode: str,
    timestamp: str,
    model_label: str,
    run_info: Dict,
    fold_ids: List[int],
    model_names: List[str],
    has_abstention: bool = False,
) -> str:
    """Generate a Markdown summary of training results.

    Produces a file analogous to the legacy RESULTS .md files, with structure
    adapted to the classification mode.

    Parameters
    ----------
    all_results       : Output of run_training_orchestration.
    classification_mode : "multiclass" | "binary" | "multi-binary".
    timestamp         : Run timestamp string (e.g. "20260312_100308").
    model_label       : Display name: "Model 1" or "Model 2".
    run_info          : Ordered dict of key-value pairs displayed in the header.
    fold_ids          : List of fold IDs that were trained.
    model_names       : Model variant names. One entry for Model 1; potentially
                        several for Model 2.
    has_abstention    : If True, include abstention columns (Model 2 only).
    """
    lines: List[str] = []
    multi_model = len(model_names) > 1

    mode_title = {
        "multiclass": "Multiclass",
        "binary": "Binary",
        "multi-binary": "Multi-Binary",
    }.get(classification_mode, classification_mode.title())

    lines += [f"# {model_label} Training Results — {mode_title}", ""]
    lines.append(f"**Timestamp**: {timestamp}")
    for k, v in run_info.items():
        lines.append(f"**{k}**: {v}")
    lines += ["", "---", ""]

    # Heading-level helpers depending on whether there are multiple model_names.
    # For single-model runs the top result section sits at ##.
    # For multi-model runs each model gets its own ## block and subsections step down.
    if multi_model:
        h2, h3, h4 = "###", "####", "#####"
    else:
        h2, h3, h4 = "##", "###", "####"

    # ------------------------------------------------------------------
    # MULTICLASS
    # ------------------------------------------------------------------
    if classification_mode == "multiclass":
        pair_data = all_results.get("multiclass", {})
        fold_results = pair_data.get("fold_results", [])
        agg_by_model = pair_data.get("aggregated_by_model", {})

        # Cross-model comparison table (only when > 1 model)
        if multi_model:
            abs_hdr = " | Abstention" if has_abstention else ""
            abs_sep = " | ----------" if has_abstention else ""
            lines += ["## Model Comparison", ""]
            lines.append(
                f"| Model | Accuracy (global) | AUROC OvO | AUPRC OvO | Log Loss{abs_hdr} |"
            )
            lines.append(
                f"|-------|-------------------|-----------|-----------|----------{abs_sep}|"
            )
            for mn in model_names:
                agg = agg_by_model.get(mn, {})

                def _ms(key: str) -> str:
                    d = agg.get(key, {})
                    if not d:
                        return "N/A"
                    return f"{_fv(d.get('mean'), '.3f')} ± {_fv(d.get('std'), '.3f')}"

                abs_val = ""
                if has_abstention:
                    ar = [
                        r.get("abstention_rate")
                        for r in fold_results
                        if r.get("model_name") == mn and r.get("abstention_rate") is not None
                    ]
                    abs_val = f" | {float(np.mean(ar)):.1%}" if ar else " | N/A"
                lines.append(
                    f"| {mn} | {_fv(agg.get('accuracy_global'), '.3f')} | "
                    f"{_ms('auroc_ovo_weighted')} | {_ms('auprc_ovo_weighted')} | "
                    f"{_ms('log_loss')}{abs_val} |"
                )
            lines += [""]

        for mn in model_names:
            agg = agg_by_model.get(mn, {})
            if not agg:
                continue

            if multi_model:
                lines += [f"## Model: {mn}", ""]

            # Overall performance
            acc_global = agg.get("accuracy_global")
            acc_pf = agg.get("accuracy_per_fold", {})
            auroc_d = agg.get("auroc_ovo_weighted", {})
            auprc_d = agg.get("auprc_ovo_weighted", {})
            ll_d = agg.get("log_loss", {})

            lines += [f"{h2} Overall Performance", ""]
            lines += [
                f"{h3} Global Metrics (Following Paper Methodology)",
                "",
                "**Accuracy**:",
                f"- **Global (concatenated predictions)**: {_fv(acc_global, '.3f')}",
                f"- Per-fold average: {_fv(acc_pf.get('mean'), '.3f')} ± {_fv(acc_pf.get('std'), '.3f')}",
                "",
                f"{h3} Probability-Based Metrics (Per-Fold Then Averaged)",
                "",
                "**Primary Metrics**:",
                f"- **AUROC (OvO, weighted)**: **{_fv(auroc_d.get('mean'), '.3f')} ± {_fv(auroc_d.get('std'), '.3f')}**",
                f"- **AUPRC (OvO, weighted)**: **{_fv(auprc_d.get('mean'), '.3f')} ± {_fv(auprc_d.get('std'), '.3f')}**",
                "",
                "**Other**:",
                f"- Log loss: {_fv(ll_d.get('mean'), '.3f')} ± {_fv(ll_d.get('std'), '.3f')}",
            ]
            if has_abstention:
                ar = [
                    r.get("abstention_rate")
                    for r in fold_results
                    if r.get("model_name") == mn and r.get("abstention_rate") is not None
                ]
                if ar:
                    am = float(np.mean(ar))
                    as_ = float(np.std(ar, ddof=1)) if len(ar) > 1 else 0.0
                    lines.append(f"- Abstention rate: {am:.1%} ± {as_:.1%}")
                    # Compute average n_scored and n_total across folds for the note
                    mn_folds_for_note = [
                        r for r in fold_results
                        if r.get("model_name") == mn and r.get("n_scored") is not None
                    ]
                    avg_scored = np.mean([r["n_scored"] for r in mn_folds_for_note])
                    avg_abstained = np.mean([r.get("n_abstained", 0) for r in mn_folds_for_note])
                    avg_total = avg_scored + avg_abstained
                    lines += [
                        "",
                        f"> **Note:** AUROC and AUPRC are computed on scored specimens only "
                        f"(avg {avg_scored:.0f}/{avg_total:.0f} per fold). "
                        f"The {avg_abstained:.0f} abstained specimens per fold "
                        f"({am:.1%}) are excluded from these metrics.",
                    ]
            lines += [""]

            # Per-class AUROC OvR
            per_class = agg.get("auroc_ovr_per_class", {})
            if per_class:
                lines += [f"{h3} Per-Class AUROC (OvR) — Individual Disease Performance", ""]
                lines.append("| Disease | AUROC (OvR) | Std Dev | Performance |")
                lines.append("|---------|-------------|---------|-------------|")
                for cls, stats in sorted(per_class.items()):
                    m_val = stats.get("mean") if stats else None
                    s_val = stats.get("std") if stats else None
                    lines.append(
                        f"| {cls} | {_fv(m_val, '.3f')} | ±{_fv(s_val, '.3f')} "
                        f"| {_auroc_rating(m_val)} |"
                    )
                lines += [""]

            # Per-fold summary table
            mn_folds = [r for r in fold_results if r.get("model_name") == mn]
            if mn_folds:
                abs_hdr = " | Abstained" if has_abstention else ""
                lines += [f"{h3} Per-Fold Results", ""]
                lines.append(
                    f"| Fold | Accuracy | AUROC (OvO) | AUPRC (OvO) | Log Loss{abs_hdr} |"
                )
                lines.append(
                    "|------|----------|-------------|-------------|----------|---------| "
                    if has_abstention else
                    "|------|----------|-------------|-------------|----------|"
                )
                for r in mn_folds:
                    abs_cell = f" | {r.get('n_abstained', 0)}" if has_abstention else ""
                    lines.append(
                        f"| {r['fold_id']} | {_fv(r.get('accuracy'), '.3f')} | "
                        f"{_fv(r.get('auroc_ovo_weighted'), '.3f')} | "
                        f"{_fv(r.get('auprc_ovo_weighted'), '.3f')} | "
                        f"{_fv(r.get('log_loss'), '.3f')}{abs_cell} |"
                    )
                lines += [""]

            # Aggregated confusion matrix + per-class accuracy
            cm_agg = agg.get("confusion_matrix_aggregated")
            classes = agg.get("classes", [])
            if cm_agg and classes:
                lines += [f"{h2} Aggregated Confusion Matrix (All Folds)", ""]
                lines.append(_fmt_conf_matrix(cm_agg, classes))
                lines += [""]

                lines += [f"{h3} Per-Class Accuracy", ""]
                lines.append("| Disease | Correct | Total | Accuracy |")
                lines.append("|---------|---------|-------|----------|")
                for i, cls in enumerate(classes):
                    correct = cm_agg[i][i]
                    total = sum(cm_agg[i])
                    acc_cls = correct / total if total > 0 else 0.0
                    lines.append(f"| {cls} | {correct} | {total} | {acc_cls:.1%} |")
                lines += [""]

            # Individual fold confusion matrices
            if mn_folds:
                lines += [f"{h2} Individual Fold Confusion Matrices", ""]
                for r in mn_folds:
                    lines += [f"{h3} Fold {r['fold_id']}", ""]
                    if "confusion_matrix" in r and "classes" in r:
                        lines.append(_fmt_conf_matrix(r["confusion_matrix"], r["classes"]))
                        lines += [""]
                    per_cls = r.get("auroc_ovr_per_class", {})
                    if per_cls:
                        lines += [f"**Per-class AUROC (OvR) — Fold {r['fold_id']}**:", ""]
                        for cls, score in sorted(per_cls.items()):
                            lines.append(f"- {cls}: {_fv(score, '.3f')}")
                        lines += [""]

    # ------------------------------------------------------------------
    # BINARY / MULTI-BINARY
    # ------------------------------------------------------------------
    else:
        pairs = list(all_results.items())
        is_multi_pair = len(pairs) > 1
        main_model = model_names[0]

        # Summary table across pairs (multi-binary only)
        if is_multi_pair:
            # Get reference class from the first pair's aggregated data for the caption
            first_agg = (pairs[0][1].get("aggregated_by_model") or {}).get(main_model, {})
            ref_display = first_agg.get("reference_class", "reference class")
            abs_hdr = " | Abstention" if has_abstention else ""
            abs_sep = " | ----------" if has_abstention else ""
            lines += ["## Summary: Per-Disease Binary Classification", ""]
            lines.append(
                f"Each model trained as: *disease* vs *{ref_display}*. "
                "AUROC and AUPRC are computed on all fold predictions pooled together."
            )
            lines += [""]
            lines.append(
                f"| Disease | AUROC (pooled) | AUPRC (pooled) | Test specimens{abs_hdr} |"
            )
            lines.append(
                f"|---------|----------------|----------------|---------------{abs_sep}|"
            )
            for pair_key, pair_data in pairs:
                agg = (pair_data.get("aggregated_by_model") or {}).get(main_model, {})
                disease = agg.get(
                    "disease",
                    pair_key.split("_vs_")[0] if "_vs_" in pair_key else pair_key,
                )
                cm_agg = agg.get("confusion_matrix_aggregated")
                test_total: object = (
                    sum(sum(row) for row in cm_agg) if cm_agg else "?"
                )
                abs_val = ""
                if has_abstention:
                    fr = pair_data.get("fold_results", [])
                    ar = [
                        r.get("abstention_rate")
                        for r in fr
                        if r.get("model_name") == main_model
                        and r.get("abstention_rate") is not None
                    ]
                    abs_val = f" | {float(np.mean(ar)):.1%}" if ar else " | N/A"
                lines.append(
                    f"| {disease} | {_fv(agg.get('auroc_pooled'), '.3f')} | "
                    f"{_fv(agg.get('auprc_pooled'), '.3f')} | {test_total}{abs_val} |"
                )
            lines += [""]
            if has_abstention:
                # Check if any pair has abstentions
                any_abstention = False
                for _, pd_ in pairs:
                    for r in pd_.get("fold_results", []):
                        if r.get("model_name") == main_model and r.get("n_abstained", 0) > 0:
                            any_abstention = True
                            break
                    if any_abstention:
                        break
                if any_abstention:
                    lines += [
                        "> **Note:** AUROC and AUPRC are computed on scored specimens only. "
                        "Abstained specimens are excluded from these metrics. "
                        "See per-disease detail sections below for specimen counts.",
                        "",
                    ]
            lines += ["## Per-Disease Detail", ""]

        for pair_key, pair_data in pairs:
            fold_results = pair_data.get("fold_results", [])
            agg_by_model = pair_data.get("aggregated_by_model", {})
            first_agg = agg_by_model.get(main_model, {})
            disease = first_agg.get("disease", "")
            reference_class = first_agg.get("reference_class", "")
            pair_title = (
                f"{disease} vs {reference_class}"
                if disease and reference_class
                else pair_key
            )

            heading_level = "###" if is_multi_pair else "##"
            lines += [f"{heading_level} {pair_title}", ""]

            # Cross-model AUROC/AUPRC table (multi-model only)
            if multi_model:
                abs_hdr = " | Abstention" if has_abstention else ""
                abs_sep = " | ----------" if has_abstention else ""
                lines.append(f"| Model | AUROC (pooled) | AUPRC (pooled){abs_hdr} |")
                lines.append(f"|-------|----------------|---------------{abs_sep}|")
                for mn in model_names:
                    agg = agg_by_model.get(mn, {})
                    abs_val = ""
                    if has_abstention:
                        ar = [
                            r.get("abstention_rate")
                            for r in fold_results
                            if r.get("model_name") == mn
                            and r.get("abstention_rate") is not None
                        ]
                        abs_val = f" | {float(np.mean(ar)):.1%}" if ar else " | N/A"
                    lines.append(
                        f"| {mn} | {_fv(agg.get('auroc_pooled'), '.3f')} | "
                        f"{_fv(agg.get('auprc_pooled'), '.3f')}{abs_val} |"
                    )
                lines += [""]

            for mn in model_names:
                agg = agg_by_model.get(mn, {})
                if not agg:
                    continue

                if multi_model:
                    sub_h = "####" if is_multi_pair else "###"
                    lines += [f"{sub_h} Model: {mn}", ""]

                lines += [
                    f"**AUROC (pooled)**: {_fv(agg.get('auroc_pooled'), '.3f')}",
                    f"**AUPRC (pooled)**: {_fv(agg.get('auprc_pooled'), '.3f')}",
                    "",
                ]

                if has_abstention:
                    mn_folds_note = [
                        r for r in fold_results
                        if r.get("model_name") == mn and r.get("n_scored") is not None
                    ]
                    ar_vals = [r.get("abstention_rate", 0) for r in mn_folds_note]
                    if ar_vals and float(np.mean(ar_vals)) > 0:
                        avg_s = np.mean([r["n_scored"] for r in mn_folds_note])
                        avg_a = np.mean([r.get("n_abstained", 0) for r in mn_folds_note])
                        avg_t = avg_s + avg_a
                        am = float(np.mean(ar_vals))
                        lines += [
                            f"> **Note:** AUROC and AUPRC are computed on scored specimens only "
                            f"(avg {avg_s:.0f}/{avg_t:.0f} per fold). "
                            f"The {avg_a:.0f} abstained specimens per fold "
                            f"({am:.1%}) are excluded from these metrics.",
                            "",
                        ]

                mn_folds = [r for r in fold_results if r.get("model_name") == mn]
                has_n_train = any("n_train" in r for r in mn_folds)
                abs_hdr = " | Abstained" if has_abstention else ""

                if has_n_train:
                    lines.append(
                        f"| Fold | AUROC | AUPRC | Train specimens | "
                        f"Test disease | Test reference | Test total{abs_hdr} |"
                    )
                    lines.append(
                        "|------|-------|-------|-----------------|-------------|----------------|----------|---------| "
                        if has_abstention else
                        "|------|-------|-------|-----------------|-------------|----------------|----------|"
                    )
                else:
                    lines.append(
                        f"| Fold | AUROC | AUPRC | Test disease | Test reference | Test total{abs_hdr} |"
                    )
                    lines.append(
                        "|------|-------|-------|--------------|----------------|----------|---------| "
                        if has_abstention else
                        "|------|-------|-------|--------------|----------------|----------|"
                    )

                for r in mn_folds:
                    n_scored = r.get("n_scored", "?")
                    abs_cell = f" | {r.get('n_abstained', 0)}" if has_abstention else ""
                    # Per-class test counts from the fold confusion matrix
                    test_d: object = "?"
                    test_r: object = "?"
                    if "confusion_matrix" in r and "classes" in r:
                        cm = r["confusion_matrix"]
                        cls_list = r["classes"]
                        d_label = r.get("disease", disease)
                        r_label = r.get("reference_class", reference_class)
                        if d_label in cls_list:
                            test_d = sum(cm[cls_list.index(d_label)])
                        if r_label in cls_list:
                            test_r = sum(cm[cls_list.index(r_label)])
                    if has_n_train:
                        lines.append(
                            f"| {r['fold_id']} | {_fv(r.get('auroc_binary'), '.3f')} | "
                            f"{_fv(r.get('auprc_binary'), '.3f')} | {r.get('n_train', '?')} | "
                            f"{test_d} | {test_r} | {n_scored}{abs_cell} |"
                        )
                    else:
                        lines.append(
                            f"| {r['fold_id']} | {_fv(r.get('auroc_binary'), '.3f')} | "
                            f"{_fv(r.get('auprc_binary'), '.3f')} | "
                            f"{test_d} | {test_r} | {n_scored}{abs_cell} |"
                        )
                lines += [""]

    lines += ["---", "", f"*Generated by {model_label} training script*", ""]
    return "\n".join(lines)
