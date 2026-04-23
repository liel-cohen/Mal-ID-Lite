"""Model 1: Repertoire Classifier

Extracts repertoire-level statistics (V-J gene pair frequencies) and trains
an elastic net logistic regression model.

Based on original Mal-ID implementation:
- malid/trained_model_wrappers/repertoire_classifier.py (_create_repertoire_stats)
- malid/train/train_repertoire_stats_model.py (make_column_transformer)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from malid_lite.utils.glmnet_wrapper import GlmnetLogitNetWrapper

from .base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
V_GENE_COL = "v_gene"
J_GENE_COL = "j_gene"


class RepertoireClassifier(BaseModel):
    """Model 1: Repertoire-level V-J gene pair frequency classifier.

    This model computes V-J gene pair frequencies for each specimen and isotype,
    applies dimensionality reduction (PCA), and trains an elastic net classifier.

    Pipeline:
    1. Extract V-J gene pair frequencies (per specimen, per isotype)
    2. ColumnTransformer (per isotype): log1p → StandardScaler → PCA(15)
    3. StandardScaler on all features
    4. Elastic net logistic regression (glmnet)

    For TCR data, only one isotype ("TCRB") is used. For BCR data (future),
    three isotypes are used ("IGHG", "IGHA", "IGHD-M") along with mutation features.

    Parameters
    ----------
    gene_locus : {"TCR", "BCR"}
        Gene locus to use. Currently only "TCR" is supported.
    n_pcs : int, default=15
        Number of principal components to extract per isotype
    l1_ratio : float, optional
        Elastic net mixing parameter (0=ridge, 1=lasso).
        If None (default), uses best values from paper:
        - TCR: 1.0 (pure lasso)
        - BCR: 0.25 (elastic net with 25% L1, 75% L2)
    n_lambda : int, default=100
        Number of lambda values to try in internal CV
    cv_folds : int, default=5
        Number of folds for internal cross-validation (lambda tuning)
    use_lambda_1se : bool, default=False
        If True, use lambda within 1 SE of best. If False, use best lambda.
    random_state : int, default=0
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed)

    Attributes
    ----------
    pipeline_ : Pipeline
        Fitted sklearn pipeline (ColumnTransformer → StandardScaler → Classifier)
    train_vj_columns_ : Dict[str, pd.Index]
        Column names for each isotype's V-J count matrix (from training)
        Used to align test set columns to match training
    isotype_groups_ : List[str]
        Isotype groups for this gene locus (["TCRB"] for TCR)
    """

    DEFAULT_L1_RATIOS: Dict[str, float] = {
        "TCR": 1.0,    # Pure lasso — best for TCR per paper
        "BCR": 0.25,   # Elastic net 0.25 — best for BCR per paper
    }

    # Class-level constants
    n_pcs = 15  # Number of PCs per isotype
    _isotype_groups = {
        "TCR": ["TCRB"],
        "BCR": ["IGHG", "IGHA", "IGHD-M"],  # TODO: BCR support
    }
    _features_from_obs = {
        "TCR": [],  # No extra features for TCR
        "BCR": [  # TODO: BCR support
            "v_mut_median_per_specimen:IGHG",
            "v_sequence_is_mutated:IGHG",
            "v_mut_median_per_specimen:IGHA",
            "v_sequence_is_mutated:IGHA",
            "v_mut_median_per_specimen:IGHD-M",
            "v_sequence_is_mutated:IGHD-M",
        ],
    }

    def __init__(
        self,
        gene_locus: str = "TCR",
        n_pcs: int = 15,
        l1_ratio: float = None,  # Will be set based on gene_locus if not provided
        n_lambda: int = 100,
        cv_folds: int = 5,
        use_lambda_1se: bool = False,
        random_state: int = 0,
        verbose: int = 0,
    ):
        """Initialize repertoire classifier."""
        super().__init__(verbose=verbose)

        # Validate gene locus
        if gene_locus not in ["TCR", "BCR"]:
            raise ValueError(f"gene_locus must be 'TCR' or 'BCR', got {gene_locus}")
        if gene_locus == "BCR":
            raise NotImplementedError("BCR support not yet implemented")

        self.gene_locus = gene_locus
        self.n_pcs = n_pcs

        if l1_ratio is None:
            l1_ratio = self.DEFAULT_L1_RATIOS.get(gene_locus, 1.0)

        self.l1_ratio = l1_ratio
        self.n_lambda = n_lambda
        self.cv_folds = cv_folds
        self.use_lambda_1se = use_lambda_1se
        self.random_state = random_state

        # Will be set during fit
        self.pipeline_ = None
        self.train_vj_columns_ = None
        self.isotype_groups_ = self._isotype_groups[gene_locus]

    def extract_features(
        self,
        sequences: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        train_vj_columns: Optional[Dict[str, pd.Index]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract V-J gene pair frequency features from sequences.

        Creates a feature matrix with one row per specimen. For each isotype,
        computes V-J gene pair frequencies and creates count matrix.

        Parameters
        ----------
        sequences : pd.DataFrame
            Preprocessed sequences (DOWNSAMPLED stage).
            Required columns: specimen_label, v_gene, j_gene, isotype_supergroup
        metadata : pd.DataFrame, optional
            Specimen metadata (not used in Model 1)
        train_vj_columns : Dict[str, pd.Index], optional
            Column structure from training set. If provided (for test set),
            aligns this matrix's columns to match training.

        Returns
        -------
        features : pd.DataFrame, shape (n_specimens, n_features)
            Feature matrix with index = specimen_label
            Columns named: "{vj_pair}:{isotype}"
        """
        if self.verbose >= 1:
            logger.info(f"Extracting features from {len(sequences)} sequences")

        # Create V-J gene pair column
        sequences = sequences.copy()
        sequences["vgene_jgene"] = (
            sequences[V_GENE_COL].astype(str) + "|" + sequences[J_GENE_COL].astype(str)
        )

        # Compute V-J pair frequencies per specimen per isotype
        specimen_vj_counts = (
            sequences.groupby(["specimen_label", "isotype_supergroup"], observed=True)[
                "vgene_jgene"
            ]
            .value_counts(normalize=True)  # Convert to frequencies
            .rename("frequency")
            .reset_index()
        )

        # Build count matrices per isotype
        vj_count_matrices_by_isotype = {}

        for isotype in self.isotype_groups_:
            if self.verbose >= 2:
                logger.info(f"Processing isotype: {isotype}")

            # Filter to this isotype
            grp = specimen_vj_counts[
                specimen_vj_counts["isotype_supergroup"] == isotype
            ]

            # Pivot to create counts matrix
            count_matrix = pd.pivot_table(
                grp,
                index="specimen_label",
                columns="vgene_jgene",
                values="frequency",
            ).fillna(0)

            # Rename columns to include isotype
            count_matrix = count_matrix.rename(
                columns=lambda col: f"{col}:{isotype}"
            )

            # Align columns to training set if provided (for test set)
            if train_vj_columns is not None:
                train_cols = train_vj_columns[isotype]

                if self.verbose >= 2:
                    logger.info(
                        f"{isotype}: Aligning {len(count_matrix.columns)} test columns "
                        f"to {len(train_cols)} train columns"
                    )

                # First, downselect to intersection of columns
                count_matrix = count_matrix[
                    train_cols.intersection(count_matrix.columns)
                ]

                # Then reindex to full train column list (adds missing cols with NaN)
                count_matrix = count_matrix.reindex(columns=train_cols).fillna(0)

            # Reindex to include all specimens in sequences (even if no data for this isotype)
            all_specimens = sequences["specimen_label"].unique()
            count_matrix = count_matrix.reindex(index=all_specimens).fillna(0)

            # Normalize rows to sum to 1 (adjust for sampling depth)
            # This is important after potentially dropping columns
            row_sums = count_matrix.sum(axis=1)
            count_matrix = count_matrix.div(row_sums, axis=0)

            # If a row was all 0s, normalize will create NaNs → fill with 0
            count_matrix = count_matrix.fillna(0)

            if self.verbose >= 2:
                logger.info(
                    f"{isotype}: Final matrix shape = {count_matrix.shape} "
                    f"({count_matrix.shape[0]} specimens × {count_matrix.shape[1]} V-J pairs)"
                )

            vj_count_matrices_by_isotype[isotype] = count_matrix

        # Concatenate isotype matrices horizontally
        # Column names are already unique because we added isotype suffix
        features = pd.concat(vj_count_matrices_by_isotype.values(), axis=1)

        # Warn about specimens with all-zero features (no V-J pairs after filtering).
        # This indicates a data quality issue — the specimen has no usable sequences.
        # The model will still produce a prediction, but it will be uninformative.
        zero_rows = (features == 0).all(axis=1)
        if zero_rows.any():
            n_zero = int(zero_rows.sum())
            n_total = len(features)
            pct = 100.0 * n_zero / n_total if n_total > 0 else 0.0
            zero_specimens = list(features.index[zero_rows])
            logger.warning(
                f"  {n_zero}/{n_total} ({pct:.1f}%) specimen(s) have all-zero features "
                f"(no V-J gene pairs after filtering). This may indicate a data "
                f"quality issue — these specimens have no usable sequences and "
                f"their predictions will be uninformative. "
                f"Specimens: {zero_specimens[:10]}"
                + (f" (and {n_zero - 10} more)" if n_zero > 10 else "")
            )

        if self.verbose >= 1:
            logger.info(
                f"Extracted features: {features.shape[0]} specimens × {features.shape[1]} features"
            )

        # TODO: For BCR, add mutation features here

        return features

    def _make_column_transformer(self, n_pcs: int) -> ColumnTransformer:
        """Create column transformer for preprocessing.

        Applies log1p → StandardScaler → PCA to each isotype's V-J count columns.

        Parameters
        ----------
        n_pcs : int
            Number of principal components per isotype

        Returns
        -------
        column_transformer : ColumnTransformer
            Transformer with one pipeline per isotype
        """
        transformers = []

        for isotype in self.isotype_groups_:
            # Pipeline: log1p → scale → PCA
            pipeline = Pipeline(
                steps=[
                    (
                        "log1p",
                        FunctionTransformer(
                            np.log1p,
                            validate=True,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_pcs, random_state=self.random_state)),
                ]
            )

            # This transformer applies to columns containing ":{isotype}"
            # For TCR, this matches columns like "TRBV7-8|TRBJ2-1:TCRB"
            transformers.append(
                (
                    f"log1p-scale-PCA_{isotype}",
                    pipeline,
                    make_column_selector(pattern=f":{isotype}"),
                )
            )

        # Passthrough any remaining columns (e.g., BCR mutation features)
        return ColumnTransformer(transformers, remainder="passthrough")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        **kwargs
    ) -> "RepertoireClassifier":
        """Train the model.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix from extract_features()
            Index should be specimen_label
        y : pd.Series, shape (n_samples,)
            Disease labels
        groups : pd.Series, optional, shape (n_samples,)
            Participant labels for grouped CV (keeps specimens from same
            participant together during internal CV for lambda tuning)
        **kwargs : dict
            Additional parameters (e.g., sample_weight)

        Returns
        -------
        self : RepertoireClassifier
            Fitted model
        """
        if self.verbose >= 1:
            logger.info(f"Training {self.__class__.__name__}")
            logger.info(f"  Training samples: {X.shape[0]}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  Classes: {y.unique().tolist()}")

        # Store training column structure for test set alignment
        self.train_vj_columns_ = {}
        for isotype in self.isotype_groups_:
            cols = X.columns[X.columns.str.endswith(f":{isotype}")]
            self.train_vj_columns_[isotype] = cols
            if self.verbose >= 2:
                logger.info(f"  {isotype}: {len(cols)} V-J pairs")

        # Determine effective n_pcs (can't exceed n_samples)
        n_samples = X.shape[0]
        n_pcs_effective = min(n_samples, self.n_pcs)
        if n_pcs_effective != self.n_pcs:
            logger.warning(
                f"Using {n_pcs_effective} PCs instead of {self.n_pcs} "
                f"because n_samples={n_samples}"
            )

        # Build pipeline: ColumnTransformer → StandardScaler → Classifier
        column_transformer = self._make_column_transformer(n_pcs=n_pcs_effective)

        # Elastic net classifier with internal CV
        classifier = self._make_glmnet_classifier()

        self.pipeline_ = Pipeline(
            steps=[
                ("columntransformer", column_transformer),
                ("scaler", StandardScaler()),  # Scale all features after PCA
                ("classifier", classifier),
            ]
        )

        # Fit the pipeline
        fit_params = {}
        if groups is not None:
            # Pass groups to classifier for grouped CV
            fit_params["classifier__groups"] = groups.values

        if "sample_weight" in kwargs:
            fit_params["classifier__sample_weight"] = kwargs["sample_weight"]

        self.pipeline_.fit(X, y, **fit_params)

        # Store fitted attributes
        self.is_fitted_ = True
        self.classes_ = self.pipeline_.named_steps["classifier"].classes_
        self.n_features_in_ = X.shape[1]

        if self.verbose >= 1:
            logger.info(f"Model fitted successfully")
            logger.info(f"  Classes: {self.classes_}")

        return self

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
        self._check_is_fitted()
        return self.pipeline_.predict(X)

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
        self._check_is_fitted()
        return self.pipeline_.predict_proba(X)

    def _make_glmnet_classifier(self) -> GlmnetLogitNetWrapper:
        """Create glmnet elastic net classifier with internal CV.

        Returns
        -------
        classifier : GlmnetLogitNetWrapper
            Configured elastic net classifier
        """
        from sklearn.model_selection import StratifiedGroupKFold

        # Use StratifiedGroupKFold for internal CV (patient-aware, matching original Mal-ID).
        # All specimens from the same participant are always in the same internal CV fold,
        # preventing within-patient data leakage during lambda selection.
        # Groups (participant_label) must be passed to fit() via classifier__groups.
        cv_strategy = StratifiedGroupKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        return GlmnetLogitNetWrapper(
            alpha=self.l1_ratio,  # 0=ridge, 1=lasso
            n_lambda=self.n_lambda,
            internal_cv=cv_strategy,  # Internal cross-validation strategy
            scoring=GlmnetLogitNetWrapper.deviance_scorer,  # Deviance (default for glmnet)
            random_state=self.random_state,
            use_lambda_1se=self.use_lambda_1se,
            require_cv_group_labels=True,  # Ensure groups are passed
            class_weight="balanced",  # Handle class imbalance
            standardize=False,  # We standardize in sklearn pipeline
        )

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"gene_locus={self.gene_locus}, "
            f"n_pcs={self.n_pcs}, "
            f"l1_ratio={self.l1_ratio}, "
            f"{fitted_str})"
        )
