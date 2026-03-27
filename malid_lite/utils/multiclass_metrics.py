"""Multiclass metrics with missing-label support.

Inlined from multiclass_metrics by Maxim Zaslavsky:
    Version: 0.0.1
    GitHub:  https://github.com/maximz/multiclass-metrics
    Source:  multiclass_metrics/__init__.py

All code is reproduced verbatim from the original source.
"""

from inspect import signature
from typing import Callable, Optional, Tuple, List
import logging
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics._base import _average_multiclass_ovo_score, _average_binary_score

# Set default logging handler to avoid "No handler found" warnings.
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

logger = logging.getLogger(__name__)


def _inject_missing_labels(
    y_true: np.ndarray, y_score: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns: new probability matrix, class names in order for that matrix

    (for binary classification, converts to and returns probability martix, rather than single vector corresponding to labels[1])

    Arguments:

    - y_true: all ground truth labels
    - y_score: predicted probabilities matrix values
    - labels: column names for y_score (i.e. class names)
    """
    y_score = np.array(y_score)  # defensive cast
    if y_score.ndim == 1 and len(labels) == 2:
        # if y_score has already been converted to a single vector for binary case (for positive class),
        # reshape it to a 2d array with each row being (1-p, p)
        y_score = np.c_[1 - y_score, y_score]

    y_true_classes = set(y_true)
    y_score_classes = set(labels)

    if y_true_classes != y_score_classes:
        y_score = pd.DataFrame(y_score, columns=labels)

        # Are any classes from y_true missing from y_score?
        missing_classes = y_true_classes - y_score_classes
        if len(missing_classes) > 0:
            # Add missing classes to probability matrix
            for c in missing_classes:
                # add each missing class to y_score, as a column of 0s
                logger.warning(f"Inserting phantom class: {c}")
                y_score[c] = 0.0

            # get new labels list, which sklearn's roc_auc_score insists must be in sorted order
            labels = sorted(np.unique(y_score.columns))

            # arrange columns in that sort order
            y_score = y_score[labels]

        # Are any classes from y_score missing from y_true?
        missing_classes_reverse = y_score_classes - y_true_classes
        if len(missing_classes_reverse) > 0:
            # Remove these classes from probability matrix.
            for c in missing_classes_reverse:
                logger.warning(f"Removing class absent from y_true: {c}")
                y_score.drop(c, axis=1, inplace=True)

            # get new labels list, which sklearn's roc_auc_score insists must be in sorted order
            labels = sorted(np.unique(y_score.columns))
            # arrange columns in that sort order
            y_score = y_score[labels]

        # convert back to numpy array
        y_score = np.array(y_score)

    return y_score, labels


def _multiclass_score(
    binary_score_func: Callable,
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    multi_class: str = "ovo",
) -> float:
    """
    Multiclass probabilistic score with missing labels:

    For multiclass situations where classifier does not learn to predict a certain class, but that class exists in test data: allow running ROC-AUC (or other probabilistic score like auPRC) by inserting missing labels into predict_proba output with probability 0.

    Also handles `y_true` not including all the classes listed in `labels`.

    Falls back to standard score if all labels are present: follows the sklearn implementation but not restricted to ROC AUC and does not require probabilities to sum to 1 (https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_ranking.py#L588)

    Parameters:

    - ``y_true``: true multiclass labels as a 1d array.
    - ``y_score``: predicted probabilities for each class as a 2d array. (In binary case, you can instead provide 1d array, like what sklearn metric functions expect.)
    - ``labels``: y_score column names. ``labels`` defaults to ``np.unique(y_true)``.
    - ``average``: can be ``macro`` or ``weighted``.
    - ``multiclass``: can be ``ovo`` or ``ovr``.
    """
    # TODO: Docs for "Also handles `y_true` not including all the classes listed in `labels`"
    y_score = np.array(y_score)  # defensive cast to numpy array

    if labels is None:
        labels = np.unique(y_true)
        # Sanity check:
        if (
            y_score.ndim == 2
            and (y_score.shape[1] > 2 or len(labels) > 2)
            and len(labels) != y_score.shape[1]
        ):
            # Multiclass classification, labels weren't provided, and y_score has a different number of (e.g. more) columns than y_true has unique values.
            raise ValueError(
                "Labels must be provided for this multiclass problem: y_true has fewer uniques than y_score number of columns, so can't infer labels."
            )
    elif len(labels) != len(np.unique(y_true)):
        # For probabilistic scores, y_true's number of unique entries must match the length of labels. That's literally the error text otherwise.
        y_score, labels = _inject_missing_labels(
            y_true=y_true, y_score=y_score, labels=labels
        )

    if len(labels) == 1:
        # Perhaps only one label left after removing phantom classes.
        # Throw same error as roc_auc_score does.
        raise ValueError(
            "Only one class present in y_true. Probability-based score is not defined in that case."
        )

    if (y_score.ndim == 2 and y_score.shape[1] == 2) or y_score.ndim == 1:
        # Binary classification case. len(labels) should be 2 as well.

        if y_score.ndim == 2 and y_score.shape[1] == 2:
            # Cast to 1d array if not already 1d.
            y_score = y_score[:, 1]

        # average_precision_score expects pos_label
        requires_pos_label = "pos_label" in signature(binary_score_func).parameters
        if requires_pos_label and len(labels) != 2:
            # Binary classification (y_score has 2 columns or is 1d),
            # and a choice of positive label is required,
            # but there's a different number of labels (either supplied by user or autogenerated from y_true's unique values).
            # We already checked len(labels) != 1 above. So len(labels) must be > 2.
            raise ValueError(
                "Can't infer positive label for this binary problem: more than 2 labels were provided or were generated from y_true's unique values."
            )
        # Set second class as the positive class.
        kwargs = {"pos_label": labels[1]} if requires_pos_label else {}

        return binary_score_func(y_true, y_score, sample_weight=sample_weight, **kwargs)

    if multi_class == "ovo":
        if sample_weight is not None:
            logger.warning("sample_weight is ignored for OvO multiclass_score")

        # Encode y_true as [0, ..., n_classes - 1] in labels order,
        # because y_true_encoded class "N" values will be matched to column number N of y_score
        y_true_cat = pd.Categorical(y_true, categories=labels)
        y_true_encoded = y_true_cat.codes

        # average_precision_score expects pos_label, but the default pos_label=1 is appropriate,
        # because _average_multiclass_ovo_score's a_true will be 1 when a and 0 when b (all other rows are removed),
        # and vice-versa for b_true, for each pair of classes a and b.
        return _average_multiclass_ovo_score(
            binary_metric=binary_score_func,
            y_true=y_true_encoded,
            y_score=y_score,
            average=average,
        )
    elif multi_class == "ovr":
        return _average_binary_score(
            binary_metric=binary_score_func,
            y_true=sklearn.preprocessing.label_binarize(y_true, classes=labels),
            y_score=y_score,
            average=average,
            sample_weight=sample_weight,
        )
    else:
        raise ValueError("Only OvO and OvR multiclass are supported.")


def roc_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    multi_class: str = "ovo",
) -> float:
    """
    Area under ROC curve, binary or multiclass OvO.

    Multiclass OvO ROC-AUC is calculated for each pair of classes, then macro or weighted averaged.
    (For each pair of classes, the calculated AUC is the average of the AUCs when each class gets a turn to be the positive class.)

    If binary: falls back to standard AUC as implemented in sklearn.

    Differences from sklearn implementation:

    - Accommodates missing classes in y_true or y_score. (see _multiclass_score docs)
    - Does not require y_score rows to sum to 1.

    Parameters:

    - ``y_true``: true multiclass labels as a 1d array.
    - ``y_score``: predicted probabilities for each class as a 2d array. (In binary case, you can instead provide 1d array, like what sklearn metric functions expect.)
    - ``labels``: y_score columns must be in ``labels`` parameter order. ``labels`` defaults to ``np.unique(y_true)``.
    - ``average``: can be ``macro`` or ``weighted``
    - ``multiclass='ovo'``: only OvO is supported.
    """
    return _multiclass_score(
        binary_score_func=sklearn.metrics.roc_auc_score,
        y_true=y_true,
        y_score=y_score,
        average=average,
        labels=labels,
        multi_class=multi_class,
        sample_weight=sample_weight,
    )


def auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    multi_class: str = "ovo",
) -> float:
    """
    Area under precision-recall curve.

    Just like multiclass OvO ROC-AUC, the auPRC is calculated for each pair of classes, then macro or weighted averaged.
    (For each pair of classes, the calculated auPRC is the average of the auPRCs when each class gets a turn to be the positive class.)

    If binary: falls back to standard auPRC as implemented in sklearn.

    Parameters:

    - ``y_true``: true multiclass labels as a 1d array.
    - ``y_score``: predicted probabilities for each class as a 2d array. (In binary case, you can instead provide 1d array, like what sklearn metric functions expect.)
    - ``labels``: y_score columns must be in ``labels`` parameter order. ``labels`` defaults to ``np.unique(y_true)``.
    - ``average``: can be ``macro`` or ``weighted``
    - ``multiclass='ovo'``: only OvO is supported.
    """
    return _multiclass_score(
        binary_score_func=sklearn.metrics.average_precision_score,
        y_true=y_true,
        y_score=y_score,
        average=average,
        labels=labels,
        multi_class=multi_class,
        sample_weight=sample_weight,
    )
