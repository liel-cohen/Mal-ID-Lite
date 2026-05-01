"""Shared utilities for model training scripts (Models 1, 2, and 3).

Constants, helper functions, and the classification-mode dispatch logic used
by train_model1.py, train_model2.py, and train_model3.py.  Model-specific
code (fold loops, feature extraction, artifact saving) stays in each
training script.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from malid_lite.utils.markdown import pad_md_tables

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


VALID_TRAINING_CONTEXTS = ("cv_single_model", "cv_ensemble")


def get_model_output_dir(
    model_name: str,
    dataset_name: str,
    classification_mode: str,
    gene_locus: str,
    training_context: str = "cv_single_model",
    output_suffix: Optional[str] = None,
) -> Path:
    """Return the canonical base output directory for a trained model.

    Path pattern depends on training_context:

      cv_single_model (default):
        trained_models/<dataset>/cv_single_model/<model_name>/<mode_dir>/<locus>/

      cv_ensemble (base models):
        trained_models/<dataset>/cv_ensemble/base_models/<locus>/<model_name>/<mode_dir>/

    where <mode_dir> is "binary" for both binary and multi-binary modes, and
    equals <classification_mode> for all other modes (e.g. "multiclass").

    If output_suffix is provided, it is appended to the mode directory:
      <mode_dir>__<suffix>   (per-model, not global)

    Individual pair artifacts for binary/multi-binary live one level deeper:
      <base>/<disease>_vs_<reference>/   (created by the training script)
    """
    if training_context not in VALID_TRAINING_CONTEXTS:
        raise ValueError(
            f"training_context must be one of {VALID_TRAINING_CONTEXTS}, "
            f"got: {training_context!r}"
        )

    mode_dir = "binary" if classification_mode in ("binary", "multi-binary") else classification_mode
    if output_suffix:
        mode_dir = f"{mode_dir}__{output_suffix}"

    base = PROJECT_ROOT / "trained_models" / dataset_name

    if training_context == "cv_ensemble":
        # base_models/<locus>/<model_name>/<mode_dir>/
        return base / "cv_ensemble" / "base_models" / gene_locus / model_name / mode_dir
    else:
        # cv_single_model: context/model/mode/locus
        return base / training_context / model_name / mode_dir / gene_locus


def get_ensemble_output_dir(
    dataset_name: str,
    classification_mode: str,
    gene_locus: str,
    output_suffix: Optional[str] = None,
) -> Path:
    """Return the canonical output directory for a trained ensemble.

    Path pattern:
      trained_models/<dataset>/cv_ensemble/ensemble/<locus>/<mode_dir>/

    If output_suffix is provided, it is appended to the mode directory:
      <mode_dir>__<suffix>
    """
    mode_dir = "binary" if classification_mode in ("binary", "multi-binary") else classification_mode
    if output_suffix:
        mode_dir = f"{mode_dir}__{output_suffix}"

    return PROJECT_ROOT / "trained_models" / dataset_name / "cv_ensemble" / "ensemble" / gene_locus / mode_dir


# ---------------------------------------------------------------------------
# Model artifact discovery
# ---------------------------------------------------------------------------


def read_model_summary(model_dir: Path) -> dict:
    """Find and read the single summary_*.json in a model artifact directory.

    Every training script writes exactly one summary JSON per output directory.
    This function locates it and returns the parsed dict.

    Raises FileNotFoundError if none found, ValueError if multiple found.
    """
    summaries = sorted(model_dir.glob("summary_*.json"))
    if len(summaries) == 0:
        raise FileNotFoundError(
            f"No summary_*.json found in {model_dir}. "
            f"Train the model first to generate artifacts."
        )
    if len(summaries) > 1:
        raise ValueError(
            f"Multiple summary_*.json files found in {model_dir}: "
            f"{[s.name for s in summaries]}. Expected exactly one."
        )
    with open(summaries[0]) as f:
        return json.load(f)


def validate_model_summary(
    summary: dict,
    expected: Dict[str, Any],
    model_label: str,
) -> None:
    """Validate that a model's summary config matches expected values.

    Parameters
    ----------
    summary : Loaded summary dict from read_model_summary().
    expected : Key-value pairs to check. Each key must exist in summary
        and its value must match exactly.
    model_label : Human-readable label for error messages
        (e.g. "Model 1 multiclass").
    """
    for key, expected_val in expected.items():
        actual_val = summary.get(key)
        if actual_val != expected_val:
            raise ValueError(
                f"{model_label}: summary config mismatch for '{key}': "
                f"expected {expected_val!r}, got {actual_val!r}. "
                f"Summary timestamp: {summary.get('timestamp', 'unknown')}"
            )


def resolve_model_artifact_dir(
    model_name: str,
    dataset_name: str,
    classification_mode: str,
    gene_locus: str,
    training_context: str = "cv_ensemble",
    output_suffix: Optional[str] = None,
) -> Tuple[Path, Optional[str]]:
    """Resolve the artifact directory for a trained model, with auto-detection.

    Returns (resolved_path, detected_suffix). detected_suffix is None when the
    default (unsuffixed) directory was used, or the suffix string if a suffixed
    directory was auto-detected.

    Resolution order:
    1. If output_suffix is given, use exact path. Error if it doesn't exist.
    2. Try the default path (no suffix). Use if it exists and has summary_*.json.
    3. Scan the parent directory for <mode>__* directories that contain
       summary_*.json. Exactly one candidate → use it. Multiple → error
       listing candidates so the user can specify --modelN-suffix.
    """
    # --- Case 1: explicit suffix → exact path, no fallback ---
    if output_suffix is not None:
        exact_dir = get_model_output_dir(
            model_name=model_name,
            dataset_name=dataset_name,
            classification_mode=classification_mode,
            gene_locus=gene_locus,
            training_context=training_context,
            output_suffix=output_suffix,
        )
        if not exact_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {exact_dir}. "
                f"Check --{model_name}-suffix value."
            )
        return exact_dir, output_suffix

    # --- Case 2: try default (unsuffixed) path ---
    default_dir = get_model_output_dir(
        model_name=model_name,
        dataset_name=dataset_name,
        classification_mode=classification_mode,
        gene_locus=gene_locus,
        training_context=training_context,
        output_suffix=None,
    )
    if default_dir.exists() and list(default_dir.glob("summary_*.json")):
        return default_dir, None

    # --- Case 3: scan parent for <mode>__* directories ---
    parent_dir = default_dir.parent
    mode_base = "binary" if classification_mode in ("binary", "multi-binary") else classification_mode

    if not parent_dir.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {parent_dir}. "
            f"Train {model_name} with --training-context {training_context} first."
        )

    candidates = []
    for d in sorted(parent_dir.iterdir()):
        if not d.is_dir():
            continue
        dir_name = d.name
        # Match <mode>__<suffix> pattern
        if dir_name.startswith(f"{mode_base}__") and list(d.glob("summary_*.json")):
            suffix = dir_name[len(f"{mode_base}__"):]
            candidates.append((d, suffix))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No artifact directory found for {model_name} "
            f"({classification_mode}, {gene_locus}). "
            f"Looked in: {parent_dir}. "
            f"Train {model_name} with --training-context {training_context} first."
        )
    if len(candidates) == 1:
        resolved_dir, suffix = candidates[0]
        logger.info(
            f"Auto-detected {model_name} artifacts: {resolved_dir.name} "
            f"(suffix={suffix!r})"
        )
        return resolved_dir, suffix

    # Multiple candidates — user must disambiguate
    candidate_names = [d.name for d, _ in candidates]
    suffixes = [s for _, s in candidates]
    raise ValueError(
        f"Multiple artifact directories found for {model_name} "
        f"({classification_mode}, {gene_locus}): {candidate_names}. "
        f"Specify which to use with "
        f"--{model_name}-suffix <suffix>. "
        f"Available suffixes: {suffixes}"
    )


def preflight_check_fold_artifacts(
    model_dirs: Dict[int, Path],
    fold_ids: List[int],
    classification_mode: str,
    disease_pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Verify that all required fold artifacts exist before starting computation.

    Checks that each model directory contains the expected per-fold files
    for all requested folds. Raises FileNotFoundError with a comprehensive
    report of all missing artifacts (not just the first one found).

    Parameters
    ----------
    model_dirs : {model_number: artifact_directory} mapping.
    fold_ids : List of fold IDs the ensemble will process.
    classification_mode : "multiclass", "binary", or "multi-binary".
    disease_pairs : For binary/multi-binary, list of (disease, reference) pairs
        whose subdirectories should also be checked. None for multiclass.
    """
    # Per-model expected file patterns (at least one must exist per fold)
    model_artifact_patterns = {
        1: [
            "fold_{fold_id}_*_model.pkl",
            "fold_{fold_id}_*_v_genes.json",
        ],
        2: [
            "fold_{fold_id}_clusters.joblib",
            "fold_{fold_id}_*_model_*.joblib",
        ],
        3: [
            "fold_{fold_id}_stage2.pkl",
        ],
    }

    missing = []

    for model_num, model_dir in sorted(model_dirs.items()):
        patterns = model_artifact_patterns.get(model_num, [])
        dirs_to_check = [model_dir]

        # For binary/multi-binary, also check each pair subdirectory
        if disease_pairs and classification_mode in ("binary", "multi-binary"):
            dirs_to_check = [
                model_dir / make_pair_name(disease, ref)
                for disease, ref in disease_pairs
            ]

        for check_dir in dirs_to_check:
            if not check_dir.exists():
                missing.append(
                    f"  Model {model_num}: directory not found: {check_dir}"
                )
                continue

            for pattern_template in patterns:
                for fold_id in fold_ids:
                    pattern = pattern_template.format(fold_id=fold_id)
                    matches = list(check_dir.glob(pattern))
                    if not matches:
                        missing.append(
                            f"  Model {model_num}, fold {fold_id}: "
                            f"no files matching '{pattern}' in {check_dir}"
                        )

    if missing:
        raise FileNotFoundError(
            f"Pre-flight check failed — missing artifacts:\n"
            + "\n".join(missing)
            + f"\n\nTrain the base models with --training-context cv_ensemble "
            f"before running the ensemble."
        )


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

FOLD_COL = "CV_fold"


def get_dataset_disease_classes(metadata: pd.DataFrame) -> List[str]:
    """Return sorted list of all disease classes in the metadata DataFrame.

    Parameters
    ----------
    metadata : pd.DataFrame
        Loader metadata (already filtered to participants with raw data
        and the active gene locus).
    """
    return sorted(metadata[DISEASE_COL].dropna().unique().tolist())


def get_dataset_fold_ids(metadata: pd.DataFrame) -> List[int]:
    """Return sorted list of all fold IDs in the metadata DataFrame.

    Parameters
    ----------
    metadata : pd.DataFrame
        Loader metadata (already filtered to participants with raw data
        and the active gene locus).
    """
    return sorted(metadata[FOLD_COL].dropna().unique().astype(int).tolist())


def get_metadata_class_counts(metadata: pd.DataFrame) -> Dict:
    """Compute participant and specimen counts per disease class from metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        Loader metadata (already filtered to participants with raw data).

    Returns
    -------
    dict with keys:
        participants_per_class : dict mapping disease → participant count
        specimens_per_class   : dict mapping disease → specimen (row) count
        total_participants    : int
        total_specimens       : int
    """
    # Participant counts: deduplicate to one row per participant
    participants_per_class = (
        metadata.drop_duplicates(subset=[PARTICIPANT_COL])
        .groupby(DISEASE_COL)[PARTICIPANT_COL].count()
        .sort_index()
        .to_dict()
    )
    # Specimen counts: each row in metadata is one specimen
    specimens_per_class = (
        metadata.groupby(DISEASE_COL)[PARTICIPANT_COL].count()
        .sort_index()
        .to_dict()
    )
    return {
        "participants_per_class": participants_per_class,
        "specimens_per_class": specimens_per_class,
        "total_participants": sum(participants_per_class.values()),
        "total_specimens": sum(specimens_per_class.values()),
    }


def validate_mode_and_classes(
    classification_mode: str,
    disease_classes: List[str],
    reference_class: Optional[str],
    diseases: Optional[List[str]] = None,
) -> Optional[str]:
    """Validate classification mode against available disease classes.

    Parameters
    ----------
    classification_mode : "multiclass", "binary", or "multi-binary".
    disease_classes : All disease classes in the dataset (sorted).
    reference_class : Reference/negative class. Required for binary and
        multi-binary modes (raises ValueError if None).
    diseases : Explicit subset of disease classes. For binary mode, the 2-class
        data requirement is relaxed when diseases is provided. For multi-binary,
        only those diseases are trained. Ignored for multiclass.

    Returns the validated reference_class (unchanged for binary/multi-binary,
    passed through for multiclass).
    Raises ValueError with a helpful message for incompatible combinations.
    """
    n = len(disease_classes)

    if classification_mode == "multiclass":
        if diseases is not None:
            raise ValueError(
                "--diseases is not supported in multiclass mode (all disease classes "
                "from the data are used). To train on a subset of diseases, use "
                "--classification-mode binary or --classification-mode multi-binary."
            )
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
        if reference_class is None:
            raise ValueError(
                "--reference-class is required for binary mode.\n"
                f"Data classes: {disease_classes}\n"
                f"Example: --reference-class '<reference class name>'"
            )
        if reference_class not in disease_classes:
            raise ValueError(
                f"--reference-class '{reference_class}' not found in data classes: {disease_classes}"
            )
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
        return reference_class

    elif classification_mode == "multi-binary":
        if reference_class is None:
            raise ValueError(
                "--reference-class is required for multi-binary mode.\n"
                f"Data classes: {disease_classes}\n"
                f"Example: --reference-class '<reference class name>'"
            )
        if reference_class not in disease_classes:
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


def get_model_classes(
    classification_mode: str,
    disease_classes: List[str],
    diseases: Optional[List[str]],
    reference_class: Optional[str],
) -> List[str]:
    """Return the sorted list of classes this model trains on.

    Parameters
    ----------
    classification_mode : "multiclass", "binary", or "multi-binary".
    disease_classes : All disease classes in the dataset (sorted).
    diseases : User's --diseases subset, or None.
    reference_class : Validated reference class (required for binary/multi-binary).

    Returns
    -------
    Sorted list of effective training classes.

    For multiclass: all disease classes in the dataset.
    For binary: [disease, reference_class] (the single pair).
    For multi-binary with --diseases: diseases + [reference_class].
    For multi-binary without --diseases: all disease classes (all non-ref + ref = all).
    """
    if classification_mode == "multiclass":
        return sorted(disease_classes)
    elif classification_mode == "binary":
        if diseases is not None:
            return sorted([diseases[0], reference_class])
        else:
            # 2-class data, no --diseases: both classes used
            return sorted(disease_classes)
    elif classification_mode == "multi-binary":
        if diseases is not None:
            return sorted(set(diseases) | {reference_class})
        else:
            return sorted(disease_classes)
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
# CV split safety check
# ---------------------------------------------------------------------------


def cap_cv_splits_for_data(
    requested_n_splits: int,
    y: np.ndarray,
    groups: np.ndarray,
    context: str = "CV",
) -> int:
    """Cap StratifiedGroupKFold n_splits based on groups (participants) per class.

    StratifiedGroupKFold requires at least n_splits groups per class (whole
    groups go into the same fold). This function checks the data and reduces
    n_splits if necessary, with clear warnings.

    The auto-cap minimum is 3 (for reliable CV). If the caller explicitly
    requests fewer splits (e.g. n_splits=2), that lower value is honored as
    the minimum — this allows callers who knowingly accept reduced CV quality
    to proceed. The absolute floor is 2 (StratifiedGroupKFold requirement).

    Parameters
    ----------
    requested_n_splits : The desired number of CV folds.
    y : Class labels array (one per sample).
    groups : Group labels array (e.g. participant_label, one per sample).
    context : Descriptive label for warning messages
        (e.g. "metamodel CV", "Model 2 GLM CV").

    Returns
    -------
    int : The effective n_splits to use (may be lower than requested).

    Raises
    ------
    ValueError
        If the data is empty, or if the smallest class has fewer groups than
        the allowed minimum (3 by default, or requested_n_splits if < 3).
    """
    if requested_n_splits < 2:
        raise ValueError(
            f"{context}: requested_n_splits must be >= 2, got {requested_n_splits}. "
            f"StratifiedGroupKFold requires at least 2 folds."
        )

    if len(y) == 0:
        raise ValueError(
            f"{context}: received empty training data (0 samples). "
            f"Cannot perform cross-validation."
        )

    # Count unique groups per class
    class_group_df = pd.DataFrame({"class": y, "group": groups})
    groups_per_class = class_group_df.groupby("class")["group"].nunique()
    min_groups = int(groups_per_class.min())

    if min_groups >= requested_n_splits:
        return requested_n_splits

    # Auto-cap floor is 3 for reliable CV. If the caller explicitly requested
    # fewer (e.g. 2), honor that as the floor. Absolute minimum is 2.
    min_allowed = max(2, min(3, requested_n_splits))

    if min_groups >= min_allowed:
        # Enough groups to reduce — warn and cap
        logger.warning(
            f"  {context}: smallest class has only {min_groups} group(s) "
            f"(participants) — reducing n_splits from {requested_n_splits} "
            f"to {min_groups}. Consider increasing training data. "
            f"Groups per class: {dict(groups_per_class)}"
        )
        return min_groups

    # Not enough groups even for the allowed minimum
    if min_groups >= 2:
        # Data could technically support 2-fold, but the auto floor is 3.
        # Tell the user they can explicitly request 2 if they accept the risk.
        raise ValueError(
            f"{context}: smallest class has only {min_groups} group(s) "
            f"(participants), but at least {min_allowed} are required for "
            f"reliable CV. Groups per class: {dict(groups_per_class)}. "
            f"Pass n_splits=2 explicitly to proceed with 2-fold CV "
            f"(results may be less reliable), or increase training data."
        )

    # min_groups < 2 — cannot do any CV
    raise ValueError(
        f"{context}: smallest class has only {min_groups} group(s) "
        f"(participants) — cannot perform cross-validation (need at least 2). "
        f"Groups per class: {dict(groups_per_class)}. "
        f"Increase training data."
    )


# ---------------------------------------------------------------------------
# Train_smaller split
# ---------------------------------------------------------------------------

def split_train_smaller(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training fold into train_smaller1 (2/3) and train_smaller2 (1/3).

    .. deprecated::
        This function is deprecated and kept only for backward compatibility
        in tests and ad-hoc analysis scripts. Production training code should
        use ``loader.get_split_participants(fold_id, training_context, roles)``
        which provides centralized, persistent, deterministic splits that are
        consistent across all models and training contexts.

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
        - accuracy_per_fold, auroc_ovo_weighted, auroc_ovo_macro,
          auprc_ovo_weighted, auprc_ovo_macro, log_loss:
          each a {mean, std, per_fold, n_folds_valid} dict, per_fold aligned to all folds
        - auroc_ovr_per_class: per-class same structure
        - confusion_matrix_aggregated: summed over scored folds (no class for abstained)

    For binary (disease_filter provided):
        - auroc_pooled, auprc_pooled: disease-as-positive, all scored folds pooled
        - auroc_per_fold, auprc_per_fold: aligned to all folds, None for all-abstained
        - accuracy_global: penalized as above
        - accuracy_per_fold: {mean, std, per_fold, n_folds_valid} dict (same structure
          as multiclass)
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
        ll_per_fold = [m.get("log_loss") for m in fold_metrics]

        result["accuracy_per_fold"] = {
            "mean": float(np.mean(accuracy_per_fold)),
            "std": float(np.std(accuracy_per_fold, ddof=1)) if n_folds > 1 else 0.0,
            "per_fold": accuracy_per_fold,
            "n_folds_valid": n_folds,
        }

        for metric_key in [
            "auroc_ovo_weighted", "auroc_ovo_macro",
            "auprc_ovo_weighted", "auprc_ovo_macro",
        ]:
            # Only aggregate keys that exist in the fold metrics (e.g., macro
            # variants are produced by the ensemble but not by base model scripts)
            if any(metric_key in m for m in fold_metrics):
                per_fold = [m.get(metric_key) for m in fold_metrics]
                result[metric_key] = _mean_std_per_fold(per_fold, metric_key)

        result["log_loss"] = _mean_std_per_fold(ll_per_fold, "log_loss")

        mcc_per_fold = [m.get("mcc") for m in fold_metrics]
        if any(v is not None for v in mcc_per_fold):
            result["mcc"] = _mean_std_per_fold(mcc_per_fold, "mcc")

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

        result["auroc_per_fold"] = _mean_std_per_fold(
            [m.get("auroc_binary") for m in fold_metrics], "auroc_binary"
        )
        result["auprc_per_fold"] = _mean_std_per_fold(
            [m.get("auprc_binary") for m in fold_metrics], "auprc_binary"
        )
        result["accuracy_per_fold"] = _mean_std_per_fold(
            accuracy_per_fold, "accuracy"
        )

        mcc_per_fold = [m.get("mcc") for m in fold_metrics]
        if any(v is not None for v in mcc_per_fold):
            result["mcc"] = _mean_std_per_fold(mcc_per_fold, "mcc")

        ll_per_fold = [m.get("log_loss") for m in fold_metrics]
        result["log_loss"] = _mean_std_per_fold(ll_per_fold, "log_loss")

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
    stage1_base_dir: Optional[Path] = None,
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
    stage1_base_dir     : If provided, passed as stage1_source_dir to fold_loop_fn.
                          For binary/multi-binary, pair subdirectories are appended
                          automatically (same as for output_dir).

    Returns
    -------
    Dict mapping pair key → {"fold_results": List[Dict], "aggregated_by_model": Dict[str, Dict]}.
    Keys: "multiclass" for multiclass; "<disease>_vs_<reference>" for each binary pair.
    """
    all_results: Dict[str, Dict] = {}

    def _store(key: str, fold_results: List[Dict], aggregated: Dict[str, Dict]) -> None:
        all_results[key] = {"fold_results": fold_results, "aggregated_by_model": aggregated}

    # Only pass stage1_source_dir when explicitly set -- Model 1 and Model 2
    # fold_loop_fn signatures don't accept it, so we must not send it as None.
    _s1_kwargs: Dict[str, Any] = {}
    if stage1_base_dir is not None:
        _s1_kwargs["stage1_source_dir"] = stage1_base_dir

    if classification_mode == "multiclass":
        fold_results, aggregated = fold_loop_fn(
            output_dir=base_dir,
            disease_filter=None,
            **_s1_kwargs,
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
            if disease == reference_class:
                raise ValueError(
                    f"--diseases '{disease}' is the same as --reference-class '{reference_class}'"
                )
            ref = reference_class
        else:
            # No explicit disease: data must have exactly 2 classes (validated above).
            # reference_class is guaranteed by validate_mode_and_classes.
            disease = next(c for c in disease_classes if c != reference_class)
            ref = reference_class

        pair_name = make_pair_name(disease, ref)
        _s1_kw_bin: Dict[str, Any] = {}
        if stage1_base_dir is not None:
            _s1_kw_bin["stage1_source_dir"] = stage1_base_dir / pair_name
        fold_results, aggregated = fold_loop_fn(
            output_dir=base_dir / pair_name,
            disease_filter=(disease, ref),
            **_s1_kw_bin,
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
            _s1_kw_mb: Dict[str, Any] = {}
            if stage1_base_dir is not None:
                _s1_kw_mb["stage1_source_dir"] = stage1_base_dir / pair_name
            fold_results, aggregated = fold_loop_fn(
                output_dir=base_dir / pair_name,
                disease_filter=(disease, reference_class),
                **_s1_kw_mb,
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


def _fv(val: Optional[float], fmt: str = ".4f") -> str:
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
    model_label       : Display name: "Model 1", "Model 2", or "Model 3".
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
                    return f"{_fv(d.get('mean'), '.4f')} ± {_fv(d.get('std'), '.4f')}"

                abs_val = ""
                if has_abstention:
                    ar = [
                        r.get("abstention_rate")
                        for r in fold_results
                        if r.get("model_name") == mn and r.get("abstention_rate") is not None
                    ]
                    abs_val = f" | {float(np.mean(ar)):.1%}" if ar else " | N/A"
                lines.append(
                    f"| {mn} | {_fv(agg.get('accuracy_global'), '.4f')} | "
                    f"{_ms('auroc_ovo_weighted')} | {_ms('auprc_ovo_weighted')} | "
                    f"{_ms('log_loss')}{abs_val} |"
                )
            lines += [""]

        deferred_abstention_blocks: List[tuple] = []

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
                f"- **Global (concatenated predictions)**: {_fv(acc_global, '.4f')}",
                f"- Per-fold average: {_fv(acc_pf.get('mean'), '.4f')} ± {_fv(acc_pf.get('std'), '.4f')}",
                "",
                f"{h3} Probability-Based Metrics (Per-Fold Then Averaged)",
                "",
                "**Primary Metrics**:",
                f"- **AUROC (OvO, weighted)**: **{_fv(auroc_d.get('mean'), '.4f')} ± {_fv(auroc_d.get('std'), '.4f')}**",
                f"- **AUPRC (OvO, weighted)**: **{_fv(auprc_d.get('mean'), '.4f')} ± {_fv(auprc_d.get('std'), '.4f')}**",
                "",
                "**Other**:",
                f"- Log loss: {_fv(ll_d.get('mean'), '.4f')} ± {_fv(ll_d.get('std'), '.4f')}",
            ]
            mcc_d = agg.get("mcc", {})
            if isinstance(mcc_d, dict) and mcc_d.get("mean") is not None:
                lines.append(
                    f"- MCC: {_fv(mcc_d['mean'], '.4f')} ± {_fv(mcc_d.get('std'), '.4f')}"
                )
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
                        f"| {cls} | {_fv(m_val, '.4f')} | ±{_fv(s_val, '.4f')} "
                        f"| {_auroc_rating(m_val)} |"
                    )
                lines += [""]

            # Per-fold summary table
            mn_folds = [r for r in fold_results if r.get("model_name") == mn]
            if mn_folds:
                abs_hdr = " | Abstained" if has_abstention else ""
                lines += [f"{h3} Per-Fold Results", ""]
                lines.append(
                    f"| Fold | Accuracy | AUROC (OvO) | AUPRC (OvO) | MCC | Log Loss{abs_hdr} |"
                )
                lines.append(
                    "|------|----------|-------------|-------------|-----|----------|---------| "
                    if has_abstention else
                    "|------|----------|-------------|-------------|-----|----------|"
                )
                for r in mn_folds:
                    abs_cell = f" | {r.get('n_abstained', 0)}" if has_abstention else ""
                    lines.append(
                        f"| {r['fold_id']} | {_fv(r.get('accuracy'), '.4f')} | "
                        f"{_fv(r.get('auroc_ovo_weighted'), '.4f')} | "
                        f"{_fv(r.get('auprc_ovo_weighted'), '.4f')} | "
                        f"{_fv(r.get('mcc'), '.4f')} | "
                        f"{_fv(r.get('log_loss'), '.4f')}{abs_cell} |"
                    )
                lines += [""]

            # Collect abstained specimen details (appended at end of report)
            if has_abstention and mn_folds:
                mn_abstained = [
                    (r["fold_id"], detail)
                    for r in mn_folds
                    for detail in r.get("test_abstained_details", [])
                ]
                if mn_abstained:
                    deferred_abstention_blocks.append(
                        (mn, mn_abstained)
                    )
                    lines.append(
                        f"*{len(mn_abstained)} abstained specimen(s) — "
                        f"see end of report for details.*"
                    )
                    lines.append("")

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
                            lines.append(f"- {cls}: {_fv(score, '.4f')}")
                        lines += [""]

        # Deferred abstained specimen details (multiclass, at end)
        if deferred_abstention_blocks:
            lines += ["---", ""]
            for mn, mn_abstained in deferred_abstention_blocks:
                lines += [f"## Abstained Specimens — {mn}", ""]
                lines.append("| Fold | Specimen | Participant | Disease |")
                lines.append("|------|----------|-------------|---------|")
                for fold_id_val, detail in mn_abstained:
                    lines.append(
                        f"| {fold_id_val} | {detail['specimen_label']} | "
                        f"{detail['participant_label']} | {detail['disease']} |"
                    )
                lines += [""]

    # ------------------------------------------------------------------
    # BINARY / MULTI-BINARY
    # ------------------------------------------------------------------
    else:
        pairs = list(all_results.items())
        is_multi_pair = len(pairs) > 1
        main_model = model_names[0]
        binary_abstention_blocks: List[str] = []
        binary_per_pair_abstentions: List[tuple] = []

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
                    f"| {disease} | {_fv(agg.get('auroc_pooled'), '.4f')} | "
                    f"{_fv(agg.get('auprc_pooled'), '.4f')} | {test_total}{abs_val} |"
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
                    # Build per-disease abstention blocks (appended at end of report)
                    for pk, pd_ in pairs:
                        pair_abstained = [
                            (r["fold_id"], detail)
                            for r in pd_.get("fold_results", [])
                            if r.get("model_name") == main_model
                            for detail in r.get("test_abstained_details", [])
                        ]
                        if not pair_abstained:
                            continue
                        pair_agg = (pd_.get("aggregated_by_model") or {}).get(main_model, {})
                        pair_disease = pair_agg.get(
                            "disease",
                            pk.split("_vs_")[0] if "_vs_" in pk else pk,
                        )
                        binary_abstention_blocks.append(
                            f"**{pair_disease}** ({len(pair_abstained)} abstained):"
                        )
                        binary_abstention_blocks.append("")
                        binary_abstention_blocks.append("| Fold | Specimen | Participant | Disease |")
                        binary_abstention_blocks.append("|------|----------|-------------|---------|")
                        for fold_id_val, detail in pair_abstained:
                            binary_abstention_blocks.append(
                                f"| {fold_id_val} | {detail['specimen_label']} | "
                                f"{detail['participant_label']} | {detail['disease']} |"
                            )
                        binary_abstention_blocks.append("")
                    if binary_abstention_blocks:
                        lines.append(
                            "See [Abstained Specimens by Disease Model]"
                            "(#abstained-specimens-by-disease-model) at the "
                            "end of this report for the full specimen list."
                        )
                        lines.append("")
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
                        f"| {mn} | {_fv(agg.get('auroc_pooled'), '.4f')} | "
                        f"{_fv(agg.get('auprc_pooled'), '.4f')}{abs_val} |"
                    )
                lines += [""]

            for mn in model_names:
                agg = agg_by_model.get(mn, {})
                if not agg:
                    continue

                if multi_model:
                    sub_h = "####" if is_multi_pair else "###"
                    lines += [f"{sub_h} Model: {mn}", ""]

                # Summary metrics
                lines += [
                    f"**AUROC (pooled)**: {_fv(agg.get('auroc_pooled'), '.4f')}",
                    f"**AUPRC (pooled)**: {_fv(agg.get('auprc_pooled'), '.4f')}",
                ]
                acc_d = agg.get("accuracy_per_fold", {})
                if isinstance(acc_d, dict) and acc_d.get("mean") is not None:
                    lines.append(
                        f"**Accuracy**: {_fv(acc_d['mean'], '.4f')} "
                        f"+/- {_fv(acc_d.get('std'), '.4f')}"
                    )
                mcc_d = agg.get("mcc", {})
                if isinstance(mcc_d, dict) and mcc_d.get("mean") is not None:
                    lines.append(
                        f"**MCC**: {_fv(mcc_d['mean'], '.4f')} "
                        f"+/- {_fv(mcc_d.get('std'), '.4f')}"
                    )
                ll_d = agg.get("log_loss", {})
                if isinstance(ll_d, dict) and ll_d.get("mean") is not None:
                    lines.append(
                        f"**Log loss**: {_fv(ll_d['mean'], '.4f')} "
                        f"+/- {_fv(ll_d.get('std'), '.4f')}"
                    )
                lines.append("")

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

                # Per-fold table
                mn_folds = [r for r in fold_results if r.get("model_name") == mn]
                has_n_train = any("n_train" in r for r in mn_folds)
                abs_hdr = " | Abstained" if has_abstention else ""

                if has_n_train:
                    lines.append(
                        f"| Fold | AUROC | AUPRC | Accuracy | MCC | Train specimens | "
                        f"Test disease | Test reference | Test total{abs_hdr} |"
                    )
                    lines.append(
                        "|------|-------|-------|----------|-----|-----------------|"
                        "-------------|----------------|----------|---------| "
                        if has_abstention else
                        "|------|-------|-------|----------|-----|-----------------|"
                        "-------------|----------------|----------|"
                    )
                else:
                    lines.append(
                        f"| Fold | AUROC | AUPRC | Accuracy | MCC | "
                        f"Test disease | Test reference | Test total{abs_hdr} |"
                    )
                    lines.append(
                        "|------|-------|-------|----------|-----|"
                        "--------------|----------------|----------|---------| "
                        if has_abstention else
                        "|------|-------|-------|----------|-----|"
                        "--------------|----------------|----------|"
                    )

                for r in mn_folds:
                    n_scored = r.get("n_scored", "?")
                    abs_cell = f" | {r.get('n_abstained', 0)}" if has_abstention else ""
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
                            f"| {r['fold_id']} | {_fv(r.get('auroc_binary'), '.4f')} | "
                            f"{_fv(r.get('auprc_binary'), '.4f')} | "
                            f"{_fv(r.get('accuracy'), '.4f')} | "
                            f"{_fv(r.get('mcc'), '.4f')} | {r.get('n_train', '?')} | "
                            f"{test_d} | {test_r} | {n_scored}{abs_cell} |"
                        )
                    else:
                        lines.append(
                            f"| {r['fold_id']} | {_fv(r.get('auroc_binary'), '.4f')} | "
                            f"{_fv(r.get('auprc_binary'), '.4f')} | "
                            f"{_fv(r.get('accuracy'), '.4f')} | "
                            f"{_fv(r.get('mcc'), '.4f')} | "
                            f"{test_d} | {test_r} | {n_scored}{abs_cell} |"
                        )
                lines += [""]

                # Collect abstained specimen details (appended at end of report)
                if has_abstention:
                    mn_abstained = [
                        (r["fold_id"], detail)
                        for r in mn_folds
                        for detail in r.get("test_abstained_details", [])
                    ]
                    if mn_abstained:
                        binary_per_pair_abstentions.append(
                            (pair_title, mn, mn_abstained)
                        )
                        lines.append(
                            f"*{len(mn_abstained)} abstained specimen(s) — "
                            f"see end of report for details.*"
                        )
                        lines.append("")

                # Confusion matrix (aggregated across folds)
                cm_agg = agg.get("confusion_matrix_aggregated")
                cm_classes = agg.get("classes", [])
                if cm_agg and cm_classes:
                    lines += [f"**Confusion Matrix** (aggregated across folds):", ""]
                    lines.append("| | " + " | ".join(str(c) for c in cm_classes) + " |")
                    lines.append("|-" + "-|-".join("---" for _ in cm_classes) + "-|")
                    for i, cls in enumerate(cm_classes):
                        row_vals = " | ".join(str(cm_agg[i][j]) for j in range(len(cm_classes)))
                        lines.append(f"| **{cls}** | {row_vals} |")
                    lines += [""]

        # Deferred abstained specimen details (binary/multi-binary, at end)
        all_abstention_items = []
        # From cross-pair summary (multi-binary)
        if binary_abstention_blocks:
            all_abstention_items.extend(binary_abstention_blocks)
        # From per-pair per-model sections
        for pair_title_val, mn_val, mn_abstained_val in binary_per_pair_abstentions:
            if is_multi_pair:
                all_abstention_items.append(
                    f"**{pair_title_val} — {mn_val}** "
                    f"({len(mn_abstained_val)} abstained):"
                )
            else:
                all_abstention_items.append(
                    f"**{mn_val}** ({len(mn_abstained_val)} abstained):"
                )
            all_abstention_items.append("")
            all_abstention_items.append("| Fold | Specimen | Participant | Disease |")
            all_abstention_items.append("|------|----------|-------------|---------|")
            for fold_id_val, detail in mn_abstained_val:
                all_abstention_items.append(
                    f"| {fold_id_val} | {detail['specimen_label']} | "
                    f"{detail['participant_label']} | {detail['disease']} |"
                )
            all_abstention_items.append("")
        if all_abstention_items:
            heading = (
                "## Abstained Specimens by Disease Model"
                if is_multi_pair
                else "## Abstained Specimens"
            )
            lines += ["---", ""]
            lines += [heading, ""]
            lines += all_abstention_items

    lines += ["---", "", f"*Generated by {model_label} training script*", ""]
    return pad_md_tables("\n".join(lines))
