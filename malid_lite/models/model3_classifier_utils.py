"""OvR classifier utilities for Model 3 (sequence-level classifier).

Contains two custom One-vs-Rest classifier implementations used internally
by SequenceLevelClassifier (in model3_sequence_level.py):

1. CustomOneVsRestClassifier — Stage 1 TCR: standard OvR with groups
   passthrough (for participant-level CV in glmnet) and failure tolerance.
2. BinaryOvRClassifierWithFeatureSubsettingByClass — Stage 2 (both loci):
   OvR where each binary classifier uses only its own class's feature columns.

These are helper classes, not standalone models. They are instantiated and
managed by SequenceLevelClassifier.

References (relative to Maxim-malid-release-202408/):
  malid/train/one_vs_rest_except_negative_class_classifier.py:57-400
  malid/train/vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr.py
  malid/train/train_vj_gene_specific_sequence_model_rollup.py:428-479
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.base
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom One-vs-Rest classifier (Stage 1 TCR)
# ---------------------------------------------------------------------------

@dataclass
class _InnerEstimator:
    """Storage for one fitted binary classifier within the OvR wrapper."""
    clf: BaseEstimator
    positive_class: str
    negative_class: str


class CustomOneVsRestClassifier:
    """Custom One-vs-Rest classifier with groups passthrough and failure tolerance.

    Replaces sklearn's OneVsRestClassifier for Stage 1 TCR, adding features that
    sklearn's implementation lacks:
      1. Passes `groups` to inner classifiers (for participant-level CV in glmnet)
      2. Optionally allows some classes to fail training gracefully
      3. Does NOT normalize probabilities by default (each binary clf is independent)

    Matches the algorithmic behavior of the original Mal-ID's custom OneVsRestClassifier
    (malid/train/one_vs_rest_except_negative_class_classifier.py:57-400).

    Parameters
    ----------
    estimator : BaseEstimator
        Base classifier to clone for each binary sub-problem.
    normalize_predicted_probabilities : bool, default False
        If True, normalize predicted probabilities to sum to 1 across classes.
        Discouraged — the probabilities come from independent binary models with
        different calibrations, so normalizing is not theoretically justified.
    n_jobs : int or None, default 4
        Number of parallel jobs for fitting binary classifiers (via joblib).
        None or 1 = sequential.
    allow_some_classes_to_fail_to_train : bool, default False
        If True, classes that fail to train (e.g. insufficient data for internal CV)
        are skipped with a warning. An error is still raised if ALL classes fail.
        Has no effect for binary problems (2 classes = 1 inner clf; failure always raises).

    Attributes (after fit)
    ----------------------
    classes_ : np.ndarray
        Sorted array of class labels (may exclude classes that failed to train).
    estimators_ : list of _InnerEstimator
        Fitted binary classifiers, one per class in classes_ (multiclass) or one
        for binary problems.
    n_features_in_ : int (if inner clf provides it)
    feature_names_in_ : np.ndarray (if inner clf provides it)

    Reference
    ---------
    Original: malid/train/one_vs_rest_except_negative_class_classifier.py:57-400
    Design doc: MODEL3_ARCHITECTURE.md section A5b
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        normalize_predicted_probabilities: bool = False,
        n_jobs: Optional[int] = 4,
        allow_some_classes_to_fail_to_train: bool = False,
    ):
        self.estimator = estimator
        self.normalize_predicted_probabilities = normalize_predicted_probabilities
        self.n_jobs = n_jobs
        self.allow_some_classes_to_fail_to_train = allow_some_classes_to_fail_to_train

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def _fit_binary(
        self,
        clf: BaseEstimator,
        X: np.ndarray,
        y_binary: np.ndarray,
        positive_class: str,
        negative_class: str,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Optional[_InnerEstimator]:
        """Fit one binary classifier (positive_class vs rest).

        Parameters
        ----------
        clf : Cloned (unfitted) estimator.
        X : Full feature matrix (all samples).
        y_binary : Binary labels (1 = positive_class, 0 = rest).
        positive_class, negative_class : Class label strings for logging.
        sample_weight : Per-sample weights, or None.
        groups : Per-sample group labels (e.g. participant_label), or None.

        Returns
        -------
        _InnerEstimator on success, None if training failed and failure is tolerated.
        """
        try:
            # Build fit kwargs: only include non-None params.
            # If the inner clf doesn't accept them, it will raise TypeError —
            # that's the correct behavior (fail loudly on configuration errors).
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            if groups is not None:
                fit_kwargs["groups"] = groups

            clf.fit(X, y_binary, **fit_kwargs)
            return _InnerEstimator(
                clf=clf,
                positive_class=positive_class,
                negative_class=negative_class,
            )
        except Exception as e:
            msg = (
                f"_fit_binary failed for positive_class='{positive_class}', "
                f"negative_class='{negative_class}': {e}"
            )
            if self.allow_some_classes_to_fail_to_train:
                logger.warning(f"  OvR: {msg}")
                return None
            raise RuntimeError(msg) from e

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "CustomOneVsRestClassifier":
        """Fit one binary classifier per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — class labels (not pre-binarized)
        sample_weight : Per-sample weights, passed to inner clf.fit() if not None.
        groups : Per-sample group labels (e.g. participant_label for CV grouping),
                 passed to inner clf.fit() if not None.

        Returns
        -------
        self
        """
        # --- Step 1: Binarize labels ---
        # LabelBinarizer sorts classes alphabetically and creates a binary indicator
        # matrix. For K classes → (n_samples, K) columns. For 2 classes → (n_samples, 1).
        label_binarizer = LabelBinarizer(sparse_output=True)
        Y = label_binarizer.fit_transform(y)
        Y = Y.tocsc()

        self.classes_ = label_binarizer.classes_
        is_binary = len(label_binarizer.classes_) == 2

        if len(self.classes_) < 2:
            raise ValueError(
                f"Only {len(self.classes_)} class(es) in data ({self.classes_}); "
                f"need at least 2 to train OvR."
            )

        # --- Step 2: Generate binary sub-problems ---
        # Each column of Y is a binary indicator for one class.
        # Multiclass (K≥3): Y has shape (n, K) — column i is indicator for classes_[i].
        # Binary (K=2): Y has shape (n, 1) — the single column is indicator for
        #   classes_[1] (sklearn convention: "positive label = second in sorted order").
        columns = [col.toarray().ravel() for col in Y.T]

        if is_binary:
            # Single column: 1 = classes_[1], 0 = classes_[0]
            jobs = [(columns[0], self.classes_[1], self.classes_[0])]
        else:
            # Column i: 1 = classes_[i], 0 = rest
            jobs = [
                (columns[i], self.classes_[i], f"not {self.classes_[i]}")
                for i in range(len(columns))
            ]

        # --- Step 3: Fit binary classifiers in parallel ---
        results: List[Optional[_InnerEstimator]] = Parallel(
            n_jobs=self.n_jobs, backend="loky"
        )(
            delayed(self._fit_binary)(
                clf=sklearn.base.clone(self.estimator),
                X=X,
                y_binary=y_bin,
                positive_class=pos_cls,
                negative_class=neg_cls,
                sample_weight=sample_weight,
                groups=groups,
            )
            for y_bin, pos_cls, neg_cls in jobs
        )

        # --- Step 4: Handle failures ---
        # Remove None entries (classes that failed to train)
        self.estimators_: List[_InnerEstimator] = [
            est for est in results if est is not None
        ]

        if len(self.estimators_) == 0:
            raise ValueError(
                "Failed to train any classes: all _fit_binary calls failed."
            )

        if is_binary:
            # Binary: keep both classes regardless. The single inner classifier
            # handles both. Failure tolerance does not apply — we already checked
            # that estimators_ is non-empty above.
            self.classes_ = label_binarizer.classes_
        else:
            # Multiclass: only keep classes that trained successfully.
            # Sort to ensure deterministic ordering (original uses parallel execution
            # order which may vary; we sort for reproducibility).
            trained_classes = sorted([est.positive_class for est in self.estimators_])
            self.classes_ = np.array(trained_classes)
            # Re-order estimators_ to match sorted classes_
            est_by_class = {est.positive_class: est for est in self.estimators_}
            self.estimators_ = [est_by_class[cls] for cls in self.classes_]

        # --- Step 5: Copy metadata from first estimator (for sklearn compat) ---
        # If the first estimator has these attributes, copy them. Otherwise, remove
        # stale values from a prior fit() call (matches original's delattr_if_exists).
        first_clf = self.estimators_[0].clf
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        elif hasattr(self, "n_features_in_"):
            del self.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities from the independent binary classifiers.

        Each binary clf outputs P(positive_class | X). These are collected into
        a matrix of shape (n_samples, n_classes). Probabilities do NOT sum to 1
        unless normalize_predicted_probabilities=True.

        For binary problems (1 estimator): returns [1-p, p] for the two classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        if not hasattr(self, "estimators_"):
            raise RuntimeError("CustomOneVsRestClassifier has not been fitted yet.")

        # Collect P(positive_class) from each binary classifier.
        # est.clf.predict_proba returns shape (n_samples, 2); column 1 = P(positive).
        Y = np.array([
            est.clf.predict_proba(X)[:, 1]
            for est in self.estimators_
        ]).T  # shape: (n_samples, n_estimators)

        if len(self.estimators_) == 1:
            # Binary problem: one estimator predicts P(class_1). Reconstruct both
            # columns: [P(class_0), P(class_1)] = [1-p, p].
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if self.normalize_predicted_probabilities:
            row_sums = Y.sum(axis=1, keepdims=True)
            # Avoid division by zero (all-zero rows stay zero)
            row_sums[row_sums == 0] = 1.0
            Y = Y / row_sums

        return Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (highest probability class).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    @property
    def n_classes_(self) -> int:
        """Number of classes."""
        return len(self.classes_)


# ---------------------------------------------------------------------------
# Binary OvR with feature subsetting (Stage 2, both loci)
# ---------------------------------------------------------------------------

def _fit_one_binary_clf(
    cls_str: str,
    subset_cols: List[str],
    X_sub_values: np.ndarray,
    y_binary: np.ndarray,
    clf,
    sample_weight: Optional[np.ndarray],
) -> Tuple[str, List[str], object]:
    """Train one binary classifier. Module-level for joblib loky pickling.

    Returns (cls_str, subset_cols, fitted_clf).
    """
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(X_sub_values, y_binary, **fit_kwargs)
    return (cls_str, subset_cols, clf)


class BinaryOvRClassifierWithFeatureSubsettingByClass:
    """Train N independent binary classifiers, one per disease class.

    Each binary classifier uses only the feature columns corresponding to its
    own disease class (those whose name starts with "{class_name}_"). This
    prevents cross-class feature leakage and reduces dimensionality per
    classifier.

    Multiclass (3+ classes): trains N classifiers, one per class. Output
    probabilities do NOT sum to 1 (each classifier is independent).

    Binary (exactly 2 classes): trains only 1 classifier and derives the
    other class's probability as 1 - p. Output probabilities sum to 1.
    Which class's features are used:
      - If reference_class is set (binary/multi-binary mode): uses the disease
        (non-reference) class's features. This is intentional — ensures the
        classifier learns from disease-specific patterns.
      - If reference_class is None: uses classes_[0]'s features, matching the
        original Mal-ID behavior (which selects features based on alphabetical
        order via LabelBinarizer, not by disease/reference semantics).

    Reference:
      malid/train/vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr.py
      malid/train/one_vs_rest_except_negative_class_classifier.py:57-400
      malid/train/train_vj_gene_specific_sequence_model_rollup.py:428-479
    """

    def __init__(
        self,
        base_clf_factory,
        classes: np.ndarray,
        n_jobs: int = 1,
        reference_class: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        base_clf_factory : Callable returning a new unfitted classifier.
        classes          : Array of class labels (strings).
        n_jobs           : Parallel workers for training binary classifiers.
                           Original uses n_jobs=n_jobs in the OvR wrapper.
        reference_class  : For binary/multi-binary mode: the reference (negative)
                           class (e.g. "Healthy"). In binary mode, the classifier
                           uses the non-reference (disease) class's features.
                           None for multiclass mode (original behavior).
        """
        self.base_clf_factory = base_clf_factory
        self.classes_ = classes
        self.n_jobs = n_jobs
        self.reference_class = reference_class
        self.classifiers_: Dict[str, object] = {}
        self.feature_subsets_: Dict[str, List[str]] = {}
        # True when len(classes_) == 2: only 1 classifier is trained, and
        # predict_proba returns [1-p, p] (matching original sklearn OvR binary
        # behavior). See one_vs_rest_except_negative_class_classifier.py:354-356.
        self._is_binary: bool = False
        # Set in fit() for binary case: which class provides features, and which
        # class has label=1 in the binary encoding.
        self._binary_feature_cls: Optional[str] = None
        self._binary_label1_cls: Optional[str] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BinaryOvRClassifierWithFeatureSubsettingByClass":
        """Train binary classifiers using per-class feature subsetting.

        Binary (2 classes): trains 1 classifier, derives other class as 1-p.
        Multiclass (3+): trains N classifiers in parallel, one per class.

        See class docstring for which features are used in binary mode.

        Parameters
        ----------
        X : DataFrame with feature columns named "{class_name}_{group_key}".
        y : String class labels, one per specimen.
        sample_weight : Per-specimen weights or None.
        """
        all_cols = list(X.columns)
        self._is_binary = (len(self.classes_) == 2)

        if self._is_binary:
            return self._fit_binary(X, y, all_cols, sample_weight)

        return self._fit_multiclass(X, y, all_cols, sample_weight)

    def _fit_binary(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        all_cols: List[str],
        sample_weight: Optional[np.ndarray],
    ) -> "BinaryOvRClassifierWithFeatureSubsettingByClass":
        """Binary case: train 1 classifier, derive other class as 1-p.

        Matches original Mal-ID binary OvR behavior
        (one_vs_rest_except_negative_class_classifier.py:354-356).
        """
        # --- Step 1: Determine which class provides features (feature_cls)
        #     and which class is encoded as label=1 (label1_cls) ---
        #
        # With reference_class (binary/multi-binary mode):
        #   feature_cls = disease (non-reference), label1_cls = reference
        #   → classifier sees disease-specific features
        #
        # Without reference_class (original behavior):
        #   feature_cls = classes_[0] (first alphabetically), label1_cls = classes_[1]
        #   → matches sklearn LabelBinarizer convention
        #
        # Note: the label assignment (which class is 0 vs 1) does not affect the
        # RF decision boundary — only the output probability mapping. The mapping
        # is handled correctly in predict_proba via _binary_label1_cls.
        if self.reference_class is not None:
            ref_str = str(self.reference_class)
            non_ref = [str(c) for c in self.classes_ if str(c) != ref_str]
            if len(non_ref) != 1:
                raise ValueError(
                    f"reference_class='{self.reference_class}' but classes_ "
                    f"has {len(non_ref)} non-reference classes: {non_ref}. "
                    f"Expected exactly 1 for binary mode."
                )
            self._binary_feature_cls = non_ref[0]
            self._binary_label1_cls = ref_str
        else:
            self._binary_feature_cls = str(self.classes_[0])
            self._binary_label1_cls = str(self.classes_[1])

        # --- Step 2: Select feature columns for the chosen class ---
        # e.g., feature_cls="Lupus" → columns starting with "Lupus_"
        prefix = f"{self._binary_feature_cls}_"
        subset_cols = [c for c in all_cols if c.startswith(prefix)]
        if not subset_cols:
            raise ValueError(
                f"No feature columns found for class '{self._binary_feature_cls}' "
                f"(prefix '{prefix}'). This indicates a bug in the feature "
                f"matrix construction — featurize_specimens() should create "
                f"columns for every class. Available columns (first 20): "
                f"{all_cols[:20]}"
            )
        self.feature_subsets_[self._binary_feature_cls] = subset_cols
        X_sub = X[subset_cols].values

        # --- Step 3: Create binary labels and validate ---
        # label1_cls gets encoded as 1, the other class as 0
        y_binary = (y == self._binary_label1_cls).astype(int)
        if y_binary.sum() == 0 or (1 - y_binary).sum() == 0:
            raise ValueError(
                f"Only one class present in training data for binary problem "
                f"(classes: {list(self.classes_)}). Both classes must have at "
                f"least one specimen."
            )

        # --- Step 4: Train single classifier ---
        clf = self.base_clf_factory()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        clf.fit(X_sub, y_binary, **fit_kwargs)
        self.classifiers_[self._binary_feature_cls] = clf
        return self

    def _fit_multiclass(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        all_cols: List[str],
        sample_weight: Optional[np.ndarray],
    ) -> "BinaryOvRClassifierWithFeatureSubsettingByClass":
        """Multiclass case: train N binary classifiers in parallel.

        Each class gets its own classifier trained on its own feature subset.
        Reference: one_vs_rest_except_negative_class_classifier.py:233-253
        """
        # --- Step 1: Prepare per-class job arguments ---
        # For each class: select its feature columns, create binary labels,
        # validate, and instantiate a fresh classifier.
        jobs = []
        for cls in self.classes_:
            cls_str = str(cls)

            # Select only this class's feature columns (e.g., "COVID-19_TRBV5-1")
            prefix = f"{cls_str}_"
            subset_cols = [c for c in all_cols if c.startswith(prefix)]
            if not subset_cols:
                raise ValueError(
                    f"No feature columns found for class '{cls_str}' "
                    f"(prefix '{prefix}'). This indicates a bug in the feature "
                    f"matrix construction — featurize_specimens() should create "
                    f"columns for every class. Available columns (first 20): "
                    f"{all_cols[:20]}"
                )

            # Extract numpy values for the subset (passed to workers)
            X_sub = X[subset_cols].values

            # Binary labels: 1 = this class, 0 = everything else
            y_binary = (y == cls_str).astype(int)
            if y_binary.sum() == 0 or (1 - y_binary).sum() == 0:
                # Build a disease distribution summary for the error message
                from collections import Counter
                class_counts = Counter(y)
                dist_str = ", ".join(
                    f"'{c}': {class_counts[c]}" for c in sorted(class_counts)
                )
                raise ValueError(
                    f"Stage 2 OvR: class '{cls_str}' has {y_binary.sum()} "
                    f"positive and {int((1 - y_binary).sum())} negative "
                    f"specimen(s) — both must be >= 1. Training data has "
                    f"{len(y)} specimens total ({dist_str}). "
                    f"Expected classes (from Stage 1): {list(self.classes_)}. "
                    f"This typically means the dataset is too small: after the "
                    f"ensemble's train/validation/train_smaller1/train_smaller2 "
                    f"splits and Stage 2 featurization dropout, too few "
                    f"specimens survived for class '{cls_str}'. "
                    f"Fix: increase the number of participants per disease, or "
                    f"use fewer cross-validation folds."
                )

            # Fresh classifier instance (created in main process, picklable)
            jobs.append((cls_str, subset_cols, X_sub, y_binary, self.base_clf_factory()))

        # --- Step 2: Train all binary classifiers in parallel ---
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fit_one_binary_clf)(
                cls_str, subset_cols, X_sub, y_binary, clf, sample_weight,
            )
            for cls_str, subset_cols, X_sub, y_binary, clf in jobs
        )

        # --- Step 3: Collect fitted classifiers ---
        for cls_str, subset_cols, fitted_clf in results:
            self.feature_subsets_[cls_str] = subset_cols
            self.classifiers_[cls_str] = fitted_clf

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-class probabilities, one column per class in classes_ order.

        Output shape: (n_specimens, n_classes).
        Binary: sums to 1 per row (complementary probabilities from 1 classifier).
        Multiclass: does NOT sum to 1 (each classifier is independent).

        Parameters
        ----------
        X : DataFrame with the same feature columns as used during fit().
        """
        n = len(X)

        if self._is_binary:
            # --- Binary: 1 classifier → reconstruct both class probabilities ---
            # The single classifier was trained on _binary_feature_cls's columns,
            # with label 1 = _binary_label1_cls. Its predict_proba[:, 1] gives
            # P(_binary_label1_cls). The other class gets 1 - p.
            # Reference: one_vs_rest_except_negative_class_classifier.py:354-356
            subset_cols = self.feature_subsets_[self._binary_feature_cls]
            X_sub = X[subset_cols].values
            clf = self.classifiers_[self._binary_feature_cls]

            # P(label1_cls) from the single estimator's positive-class column
            p_label1 = clf.predict_proba(X_sub)[:, 1]

            # Map to classes_ order: label1_cls column gets p, the other gets 1-p.
            # Example: classes_=["Healthy", "Lupus"], label1_cls="Healthy"
            #   → column 0 (Healthy) = p_label1, column 1 (Lupus) = 1-p_label1
            proba = np.empty((n, 2))
            for j, cls in enumerate(self.classes_):
                if str(cls) == self._binary_label1_cls:
                    proba[:, j] = p_label1
                else:
                    proba[:, j] = 1.0 - p_label1
            return proba

        # --- Multiclass: one classifier per class ---
        # Initialize with NaN as safety net — all columns are guaranteed to be
        # overwritten since fit() validates every class. NaN would surface any
        # bug that leaves a column unset (rather than silently producing zeros).
        n_classes = len(self.classes_)
        proba = np.full((n, n_classes), fill_value=np.nan)

        for j, cls in enumerate(self.classes_):
            cls_str = str(cls)
            # Each classifier was trained on its own feature subset, predicting
            # 1 = this class vs 0 = rest. predict_proba[:, 1] = P(this class).
            subset_cols = self.feature_subsets_[cls_str]
            X_sub = X[subset_cols].values
            clf = self.classifiers_[cls_str]
            proba[:, j] = clf.predict_proba(X_sub)[:, 1]

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class with the highest probability for each specimen."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
