"""Abstract base class for all Mal-ID models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pickle
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class defining the interface for all Mal-ID models.

    All models must implement:
    - fit(): Train the model on feature matrix X and labels y
    - predict(): Return class labels for samples
    - predict_proba(): Return class probabilities for samples
    - extract_features(): Convert raw sequences to feature matrix

    Models can optionally override:
    - save() / load(): Custom serialization
    - get_params() / set_params(): For sklearn compatibility
    """

    def __init__(self, verbose: int = 0):
        """Initialize base model.

        Parameters
        ----------
        verbose : int, default=0
            Verbosity level:
            - 0: Silent
            - 1: Basic progress
            - 2: Detailed logging
        """
        self.verbose = verbose
        self.is_fitted_ = False
        self.classes_ = None
        self.n_features_in_ = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """Train the model.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix (one row per specimen)
        y : pd.Series, shape (n_samples,)
            Target labels
        **kwargs : dict
            Additional parameters (e.g., sample_weight, groups for CV)

        Returns
        -------
        self : BaseModel
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        pass

    @abstractmethod
    def extract_features(
        self,
        sequences: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract features from raw sequences.

        This is model-specific feature engineering. For Model 1, this computes
        V-J gene pair frequencies. For Models 2-3, this may compute different features.

        Parameters
        ----------
        sequences : pd.DataFrame
            Raw sequence data (after preprocessing)
            Expected columns depend on model type
        metadata : pd.DataFrame, optional
            Specimen-level metadata (for joining additional info)
        **kwargs : dict
            Model-specific parameters

        Returns
        -------
        features : pd.DataFrame, shape (n_specimens, n_features)
            Feature matrix with one row per specimen
            Index should be specimen_label
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Default implementation uses pickle. Override for custom serialization.

        Parameters
        ----------
        path : str or Path
            Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose >= 1:
            logger.info(f"Saving model to {path}")

        with open(path, "wb") as f:
            pickle.dump(self, f)

        if self.verbose >= 1:
            logger.info(f"Model saved successfully")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load model from disk.

        Default implementation uses pickle. Override for custom deserialization.

        Parameters
        ----------
        path : str or Path
            Path to saved model

        Returns
        -------
        model : BaseModel
            Loaded model
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")

        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        For sklearn compatibility.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for sub-objects

        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        return {"verbose": self.verbose}

    def set_params(self, **params) -> "BaseModel":
        """Set parameters for this estimator.

        For sklearn compatibility.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : BaseModel
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted.

        Raises
        ------
        RuntimeError
            If model has not been fitted
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call .fit() first."
            )

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
