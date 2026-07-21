# Model Description

A thorough, step-by-step algorithmic description of each model in Mal-ID-Lite,
from input data to final predictions.

---

## Model 1: Repertoire-Level V-J Gene Frequency Classifier

Model 1 classifies specimens by their **repertoire composition** — the relative
frequencies of V-J gene pair usage across TCR (or BCR) sequences. The
biological hypothesis is that different diseases induce distinct biases in V-gene
and J-gene usage, reflecting antigen-driven selection of specific receptor
types.

Unlike Models 2 and 3, which analyze individual CDR3 sequences, Model 1 treats
each specimen's repertoire as a single high-dimensional vector of V-J pair
frequencies. This makes it the simplest and fastest of the three models.

Disease is a participant-level attribute (each participant has one disease), but
a single participant may contribute multiple specimens. The pipeline operates at
the specimen level throughout: features are computed per specimen, and
predictions are per specimen.

### Step 0: Training Data

For each cross-validation fold, the training data is selected by combining
**train_smaller1 + train_smaller2** participants (the same split used by Models
2 and 3). Unlike Models 2 and 3, which use these sub-splits separately (e.g.,
train_smaller1 for Stage 1, train_smaller2 for Stage 2 / threshold selection),
Model 1 **merges them into a single training set**. For `cv_single_model`, this
includes all non-test participants. For `cv_ensemble`, this excludes the
validation holdout (validation participants are used only by the ensemble
meta-learner). The train-all contexts apply the identical ts1+ts2-union logic
over the whole dataset (no CV fold): `train_all` merges every participant;
`train_all_ensemble` merges everyone except the validation holdout.

**Input**: Sequences at the DOWNSAMPLED preprocessing stage (after Stage 1 and
Stage 2 preprocessing: productive, V-score filtered, deduplicated, cleaned,
clone/sequence thresholds applied, 1 sequence per clone).

### Step 1: V-Gene Frequency Filtering

**Goal**: Remove rare V-genes whose minute frequency differences contribute
noise rather than signal.

#### 1a. Global V-Gene Frequencies

Compute the relative frequency of each V-gene across all training sequences
(all diseases pooled): `v_gene.value_counts(normalize=True)`. This produces one
frequency value per V-gene.

#### 1b. Median-Based Threshold

`threshold = frequencies.quantile(0.5)` — the median frequency across all
V-genes. V-genes with frequency >= threshold are kept; those below are removed.
This removes the bottom ~50% of V-genes by rank.

Example: if 50 V-genes are observed and the median frequency is 0.012, V-genes
with frequency < 0.012 are discarded. Typically ~25 V-genes survive for TCR.

#### 1c. Apply Filter

Sequences with a removed V-gene are dropped from the training data. The kept
V-gene list is saved and applied identically to the test set (Step 6).

**Difference**: None. Both codebases use global `value_counts(normalize=True)`
with `quantile(0.5)` on training data. (The original has a TODO to switch to a
per-disease max-frequency variant in `helpers.find_non_rare_v_genes()`, but
this is not used — the comment notes "Results are the same for default sample
weight strategy.")

### Step 2: V-J Gene Pair Frequency Matrix

**Goal**: Convert each specimen's sequences into a frequency vector over V-J
gene pairs, separately per isotype.

#### 2a. V-J Pair Construction

Each sequence's V-gene and J-gene are cast to string (`astype(str)`) and
concatenated into a single string: `v_gene + "|" + j_gene` (e.g.,
`"TRBV5-1|TRBJ2-1"`). The explicit `astype(str)` ensures NaN gene values
(if any) are handled gracefully (converted to the string `"nan"`).

#### 2b. Per-Isotype Frequency Computation

For each isotype group (TCR: only `"TCRB"`; BCR: `"IGHG"`, `"IGHA"`,
`"IGHD-M"`):

1. Filter sequences to this isotype.
2. Compute `groupby(["specimen_label", "isotype_supergroup"], observed=True)
   ["vgene_jgene"].value_counts(normalize=True)`. The `observed=True` excludes
   unused categorical levels from the groupby (prevents spurious empty groups).
   The result is the relative frequency of each V-J pair within each specimen
   for this isotype.
3. Pivot into a matrix via `pd.pivot_table(index="specimen_label",
   columns="vgene_jgene", values="frequency")`. Missing (specimen, V-J pair)
   cells are filled with 0 via `.fillna(0)`.

**Result per isotype**: A specimen x V-J-pair matrix where each row sums to 1.0
(frequencies within that isotype for that specimen).

#### 2c. Column Naming

Column names include the isotype suffix for uniqueness across isotypes:
- **Lite**: `"{v_gene}|{j_gene}:{isotype}"` (e.g., `"TRBV5-1|TRBJ2-1:TCRB"`).
  The ColumnTransformer selects these via `pattern=f":{isotype}"`.
- **Original**: `"{v_gene}|{j_gene}:{isotype}"` initially, then renamed to
  `"{isotype}:pca_{v_gene}|{j_gene}:{isotype}"` before the pipeline. The
  ColumnTransformer selects these via `pattern=f"{isotype_group}:pca"`.

The naming convention differs but both patterns correctly match the intended
columns. The mathematical content (frequencies) is identical.

#### 2d. Specimen Reindexing

The matrix is reindexed to include all specimens in the input — even those with
no sequences for this isotype (e.g., a TCR specimen missing data). Missing
specimens receive an all-zero row.

#### 2e. Row Renormalization

Rows are renormalized to sum to 1: `row / row.sum()`. This step runs for
**both training and test data** — not only after column subsetting at test time.
During training, it is needed because specimen reindexing (Step 2d) may have
added all-zero rows. At test time, it is additionally needed because column
alignment (Step 6b) may have dropped V-J pairs, changing the row sums.

If a row was all zeros (specimen has no sequences for this isotype), division
produces NaN, which is filled back to 0.

The original uses `genetools.stats.normalize_rows()`, which internally does the
same `df.div(df.sum(axis=1), axis=0)`, followed by `fillna(0)`.

#### 2f. BCR Mutation Features (BCR Only, Not Yet Implemented in Lite)

For BCR, two additional per-isotype features are appended to the feature matrix:
- `v_mut_median_per_specimen:{isotype}` — median somatic hypermutation rate
- `v_sequence_is_mutated:{isotype}` — proportion of sequences with
  `v_mut >= 0.01`

These are not used for TCR.

#### 2g. Concatenation

The per-isotype matrices are concatenated horizontally. For TCR, this is a
single matrix (one isotype). For BCR, three matrices plus mutation features.

**Output**: A specimen x feature DataFrame. For TCR with ~25 V-genes and
~13 J-genes, the number of observed V-J pairs is typically ~200-400. Each row
sums to 1.0 (for TCR's single isotype).

**Difference**: Column naming convention differs (see 2c). The original wraps
the matrix in an AnnData object and extracts BCR mutation features from `.obs`.
Lite works with plain DataFrames. Identical mathematical result.

### Step 3: ColumnTransformer — Per-Isotype PCA Pipeline

**Goal**: Apply dimensionality reduction to each isotype's V-J frequency
columns independently, reducing hundreds of sparse frequency features to a
compact set of principal components.

#### 3a. Per-Isotype Pipeline

For each isotype group (TCR: one pipeline for `"TCRB"`), a three-step
sub-pipeline is applied:

1. **`log1p`** (`np.log1p`): `x -> log(1 + x)`. Compresses the dynamic range
   of frequencies. Since frequencies are in [0, 1], `log1p` maps them to
   [0, 0.693]. This is important because V-J pair frequencies are typically
   highly skewed (a few dominant pairs with frequency ~0.05-0.10, many rare
   pairs with frequency ~0.001).

2. **`StandardScaler`**: Centers each V-J pair column to mean 0 and scales to
   unit variance. Fitted on training data only.

3. **`PCA(n_components=15, random_state=0)`**: Reduces the ~200-400 V-J pair
   columns to 15 principal components. Fitted on training data only. If
   `n_samples < 15`, the effective number of PCs is clamped to `n_samples`.

#### 3b. Column Selection

Each pipeline is applied only to columns matching its isotype, selected via
`make_column_selector(pattern=...)`:
- **Lite**: pattern = `":{isotype}"` (e.g., `":TCRB"`)
- **Original**: pattern = `"{isotype_group}:pca"` (e.g., `"TCRB:pca"`)

Both match the correct columns within their respective naming conventions.

#### 3c. Remainder Handling

`remainder="passthrough"` — any columns not matched by an isotype pipeline
(e.g., BCR mutation features) are passed through unchanged. For TCR, there are
no such columns.

**Output**: For TCR: 15 PCA components per specimen. For BCR: 15 PCs x
3 isotypes + 6 mutation features = 51 features.

**Difference**: The original uses `StandardScalerThatPreservesInputType`
(a thin wrapper preserving DataFrame types) instead of sklearn's
`StandardScaler`. Same math.

### Step 4: Post-PCA StandardScaler

A second `StandardScaler` is applied to **all features** after the
ColumnTransformer output. This re-centers and re-scales the PCA components
(and any passthrough features) to mean 0, unit variance before the classifier.

Without this scaler, PCA components with larger eigenvalues would dominate the
classifier's loss function. Standardizing ensures all components contribute on
comparable scales.

**Difference**: The original prepends this scaler via
`prepend_scaler_if_not_present()` (which uses
`StandardScalerThatPreservesInputType`). Lite adds it as an explicit pipeline
step. Same math.

### Step 5: Classifier — Elastic Net Logistic Regression (Glmnet)

**Goal**: Given the 15-dimensional (TCR) or 51-dimensional (BCR) feature vector
per specimen, predict the specimen's disease label.

#### 5a. Classifier Type

`GlmnetLogitNetWrapper` — a logistic regression classifier using the glmnet
coordinate descent algorithm (via the `python-glmnet` / `civis-python-glmnet`
package). Glmnet efficiently fits the entire regularization path (all lambda
values) in a single pass.

For K >= 3 classes, glmnet fits a **multinomial (softmax) logistic regression**
— a single model with K coefficient vectors, optimized jointly. This is NOT
One-vs-Rest; the K classes are coupled through the softmax normalization.

For K = 2 classes, glmnet fits a standard **binary logistic regression** with a
single coefficient vector.

#### 5b. Configuration

| Parameter                 | Value                                                            | Meaning                                                                                                                                       |
| ------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`                   | TCR: 1.0 (lasso). BCR: 0.25 (elastic net).                       | L1/L2 penalty ratio. 1.0 = pure L1 (lasso, drives coefficients to zero for feature selection). 0.0 = pure L2 (ridge). 0.25 = 75% L2 + 25% L1. |
| `n_lambda`                | 100                                                              | Number of lambda (regularization strength) values along the path                                                                              |
| `standardize`             | False                                                            | Handled by the pipeline's StandardScaler steps                                                                                                |
| `use_lambda_1se`          | False                                                            | Use lambda minimizing CV deviance (not the 1-SE rule)                                                                                         |
| `class_weight`            | "balanced"                                                       | Upweight minority classes inversely proportional to class frequency                                                                           |
| `internal_cv`             | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | 5-fold stratified CV for lambda selection                                                                                                     |
| `scoring`                 | Deviance (log-loss)                                              | CV scoring metric for lambda selection                                                                                                        |
| `require_cv_group_labels` | True                                                             | Enforces that `groups` (participant labels) must be passed to `fit()`                                                                         |
| `random_state`            | 0                                                                | Seed for CV fold generation                                                                                                                   |

#### 5c. Internal Cross-Validation for Lambda Selection

Lambda (regularization strength) is tuned via 5-fold internal CV:

1. Glmnet computes a path of 100 lambda values, from lambda_max (where all
   coefficients are zero) down to a small lambda (minimal regularization).
2. For each lambda, the model is fit on 4/5 of the training data and evaluated
   on the held-out 1/5 using deviance (log-loss).
3. The CV is grouped by `participant_label` via `StratifiedGroupKFold` — all
   specimens from the same participant are always in the same inner fold,
   preventing within-participant data leakage during lambda tuning.
4. The lambda with the lowest mean CV deviance is selected (`use_lambda_1se =
   False`).

#### 5d. `predict_proba` Output

`GlmnetLogitNetWrapper.predict_proba()` converts logits to probabilities:
- **Binary (K=2)**: Sigmoid — `P(positive) = 1 / (1 + exp(-logit))`, output is
  `[1-P, P]` with shape (n_samples, 2).
- **Multiclass (K>=3)**: Softmax — normalizes the K-dimensional logit vector to
  a proper probability distribution summing to 1.

Probabilities are ordered according to `classifier.classes_`, which is set via
`np.unique(y)` during `fit()` — guaranteed sorted (alphabetical for string
labels).

#### 5e. Full Pipeline

```
Pipeline([
    ("columntransformer", ColumnTransformer([
        ("log1p-scale-PCA_TCRB", Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
            ("pca", PCA(15)),
        ]), make_column_selector(pattern=":TCRB")),
    ], remainder="passthrough")),
    ("scaler", StandardScaler()),
    ("classifier", GlmnetLogitNetWrapper(alpha=1.0, ...)),
])
```

All transformers (log1p is stateless; StandardScaler, PCA) are fitted on
training data only and applied to both train and test.

**Difference**: None in configuration. Both use `GlmnetLogitNetWrapper` with
identical parameters. The original trains multiple model variants (lasso_cv,
ridge_cv, elasticnet_cv, etc.) and picks the best per-locus; Lite trains only
the pre-selected best variant by default (`lasso_cv` for TCR, `elasticnet_cv`
with `l1_ratio=0.25` for BCR).

### Step 6: Prediction at Test Time

Given a test specimen's sequences:

#### 6a. V-Gene Filtering

Filter test sequences to the V-genes kept during training (Step 1). Sequences
with V-genes not in the training set are dropped.

#### 6b. Feature Extraction with Column Alignment

1. Compute V-J pair frequencies per isotype, as in Step 2.
2. **Column alignment**: The test feature matrix must match the training matrix's
   column structure exactly. Three operations:
   a. **Intersection**: Retain only V-J pairs that exist in both train and test
      column sets.
   b. **Reindex to training columns**: Add columns for V-J pairs present in
      training but absent in test, filled with 0.
   c. **Column ordering**: Reorder test columns to match training order.

   This ensures the test matrix has identical shape and column semantics to the
   training matrix.

3. **Specimen reindexing**: Include all test specimens, even those with no
   sequences remaining after V-gene filtering. Missing specimens get all-zero
   rows.

4. **Row renormalization**: Rows are renormalized to sum to 1 after column
   subsetting (Step 2e). All-zero rows produce NaN, filled to 0.

#### 6c. Pipeline Transform and Predict

Pass the aligned test feature matrix through the fitted pipeline:
1. ColumnTransformer: `log1p` -> `StandardScaler.transform()` ->
   `PCA.transform()` (all training-fitted).
2. Post-PCA `StandardScaler.transform()` (training-fitted).
3. `GlmnetLogitNetWrapper.predict_proba()` -> probability vector per specimen.

**Output**: One K-dimensional probability vector per specimen (summing to 1 for
multiclass via softmax, summing to 1 for binary via sigmoid).

### Step 7: Classification Modes

Model 1 supports three classification modes:

#### 7a. Multiclass (Default)

A single K-class classifier trained on all disease classes simultaneously using
multinomial (softmax) logistic regression.

- **Input**: All specimens from all disease classes.
- **Output**: K-dimensional probability vector per specimen.

#### 7b. Binary

A single 2-class classifier for one specific disease vs. a reference class.

- **Input**: Specimens filtered to {disease, reference_class} only.
- **Output**: 2-dimensional probability vector `[P(reference), P(disease)]`.
- The reference class is either specified explicitly or determined
  alphabetically for 2-class datasets.

#### 7c. Multi-Binary

One independent binary classifier per disease, each trained as disease vs.
reference class.

- **Input per classifier**: Specimens filtered to {disease_k, reference_class}.
- **Output per classifier**: P(disease_k) for each specimen.
- Each classifier is fully independent — separate V-gene filtering, separate
  feature matrices, separate pipeline fitting.

### Hyperparameter Summary

| Hyperparameter                       | How Tuned                                         | Selection Criterion                           |
| ------------------------------------ | ------------------------------------------------- | --------------------------------------------- |
| **Lambda** (regularization strength) | Automatically by glmnet's internal 5-fold CV      | Deviance (log-loss)                           |
| **Alpha** (L1/L2 ratio)              | Fixed per locus (pre-selected in original Mal-ID) | N/A (lasso for TCR, elastic net 0.25 for BCR) |
| **n_pcs** (PCA components)           | Fixed at 15                                       | N/A (paper-best value)                        |
| **V-gene frequency threshold**       | Fixed at 50th percentile (median)                 | N/A                                           |

### Handling of Specimens with Insufficient Data

Model 1 always produces a prediction for every specimen. There is no abstention
mechanism.

If a specimen has no sequences belonging to any non-rare V-gene (after V-gene
filtering to the training set), its V-J frequency matrix is all zeros. After
row renormalization (frequencies sum to 1), an all-zero row produces NaN, which
is filled back to 0. The all-zero feature vector flows through log1p(0) = 0,
StandardScaler, and PCA, resulting in a feature vector that carries no
specimen-specific information. The classifier still produces a probability
vector, but it is uninformative (driven by the model's bias/intercept rather
than any data from this specimen).

A warning is logged with the count and percentage of affected specimens. This
situation is extremely unlikely in practice (would require a specimen with zero
sequences from any V-gene seen during training) and indicates a data quality
issue if it occurs.

### Summary of Differences (Original Mal-ID vs Mal-ID-Lite)

| #   | Aspect                 | Original                                                                                                                | Lite                                                | Impact                                                                   |
| --- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| 1   | Column naming          | `"{isotype}:pca_{vj_pair}:{isotype}"`                                                                                   | `"{vj_pair}:{isotype}"`                             | Naming convention only. Both correctly match via `make_column_selector`. |
| 2   | StandardScaler variant | `StandardScalerThatPreservesInputType`                                                                                  | sklearn `StandardScaler`                            | Same math. Original preserves DataFrame type.                            |
| 3   | Data structure         | AnnData object with `.obs` and `.var`                                                                                   | Plain pandas DataFrames                             | Same mathematical result. Lite avoids the AnnData dependency.            |
| 4   | Model variants trained | All 5 alpha variants (+ OvR, RF, XGBoost, etc.) trained per fold; best hardcoded per locus                              | Only the pre-selected best model trained by default | Lite skips redundant work. Same final model.                             |
| 5   | BCR mutation features  | Computed from `v_mut` column in `.obs`                                                                                  | Not yet implemented (BCR not supported)             | No impact for TCR.                                                       |
| 6   | Serialization          | `joblib.dump` / `joblib.load`                                                                                           | `pickle.dump` / `pickle.load`                       | Format difference only.                                                  |
| 7   | Clone-size weighting   | Supports `SampleWeightStrategy.CLONE_SIZE` for V-gene filtering and V-J frequency computation (weighted `value_counts`) | Not implemented — all sequences weighted equally    | No impact for default sample weight strategy (non-CLONE_SIZE).           |
| 8   | Glmnet `verbose`       | `verbose=True` (logs internal CV progress)                                                                              | Not passed (default `False`)                        | Logging verbosity only. No impact on results.                            |
| 9   | Glmnet `n_jobs`        | `n_jobs=n_jobs` (parallelizes internal CV scoring)                                                                      | Not passed (default `1` — serial)                   | Execution speed only. No impact on results.                              |

For the **paper-best TCR configuration** (`lasso_cv`, `n_pcs=15`, default
sample weight strategy), both codebases produce equivalent results. The V-gene
filtering criterion, V-J frequency computation, log1p -> scale -> PCA pipeline,
post-PCA scaling, and glmnet classifier configuration are identical.

---

## Model 2: Convergent Cluster Classifier

Model 2 identifies disease-associated CDR3 sequence clusters that are shared
across multiple patients ("convergent" clusters), then classifies specimens by
counting how many such clusters their sequences match.

The biological hypothesis is that certain CDR3 sequences are convergently
selected by the immune system in response to specific diseases. If multiple
unrelated patients with the same disease independently produce similar CDR3
sequences, those sequences are likely targeting disease-specific antigens. By
finding such convergent clusters and counting how many a new specimen matches,
the model can infer which disease the specimen is associated with.

Unlike Models 1 and 3, Model 2 has an explicit **abstention mechanism**: if a
specimen's sequences do not match any disease-enriched cluster, it produces no
prediction rather than an uninformative one.

### Step 0: Training Data Split

Identical to Model 3. For each fold, the training data is split at the
**participant level** (stratified by disease, `random_state=0`):

- **train_smaller1** (2/3 of training participants) — used for clustering,
  Fisher's test, and final classifier training.
- **train_smaller2** (1/3 of training participants) — used for p-value threshold
  selection via MCC grid search.

Clusters and Fisher p-values are always derived from train_smaller1 and are
never re-computed. train_smaller2 is used only to evaluate which p-value
threshold produces the best classification performance.

The train-all contexts apply this same ts1/ts2 split over the whole dataset (no
CV fold, no evaluation): `train_all` uses every participant; `train_all_ensemble`
excludes the validation third. Both are training-time uses of the split — p-value
selection on ts2 is hyperparameter tuning, not held-out evaluation.

Reference: `base.py:560-573`

### Step 1: Clustering CDR3 Sequences

**Goal**: Group CDR3 sequences that are highly similar (likely targeting the same
antigen) into clusters. Clustering is performed on **all sequences in
train_smaller1** across all participants and all diseases simultaneously — not
per-disease.

#### 1a. Supergroup Formation

Before clustering, sequences are partitioned into **supergroups** by
`(v_gene, j_gene, cdr3_aa_sequence_trim_len)`. Two sequences can only be
clustered together if they share the same V-gene, J-gene, and CDR3 length.
This is both biologically motivated (these properties constrain CDR3 structure)
and computationally necessary (Hamming distance requires equal-length strings).

#### 1b. Distance Computation

Within each supergroup, CDR3 amino acid strings are converted to integer ordinal
vectors (one uint8 value per amino acid position) via `strings_to_numeric_vectors()`.
Pairwise **normalized Hamming distances** are computed using
`scipy.spatial.distance.pdist(vectors, metric="hamming")`. Normalized Hamming
distance = (number of differing positions) / (total positions). For two
13-residue CDR3 sequences differing at 1 position: distance = 1/13 = 0.077.

#### 1c. Single-Linkage Hierarchical Clustering

`scipy.cluster.hierarchy.linkage(dist_condensed, method="single")` constructs
a hierarchical clustering tree. Single linkage merges the two nearest clusters
at each step (i.e., the minimum inter-cluster distance).

The dendrogram is cut at distance = `1 - sequence_identity_threshold`:
- **TCR**: threshold = 0.90 -> cut at distance 0.10 (sequences within 10%
  Hamming distance are co-clustered)
- **BCR**: threshold = 0.85 -> cut at distance 0.15

`scipy.cluster.hierarchy.fcluster(Z, t=cut_distance, criterion="distance")`
returns 1-indexed cluster labels.

#### 1d. Degenerate Groups

If a supergroup contains only one unique CDR3 sequence (possibly appearing
multiple times across patients), all sequences are assigned to cluster ID 0.
This avoids the `ValueError` that scipy raises on an empty distance matrix.

#### 1e. Global Cluster ID

Cluster IDs from `fcluster` are only unique within their supergroup. A globally
unique identifier is constructed as a tuple:
`(v_gene, j_gene, cdr3_len, local_cluster_id)`.

#### 1f. Clone Member Counts

Each sequence carries a `num_clone_members` field indicating how many unique
VDJ sequences the clone contains. Missing values are filled with 1. This is used
later for centroid weighting.

**Output**: The input DataFrame with two new columns: `cluster_id_within_clustering_group`
and `global_resulting_cluster_ID`.

**Difference**: The original uses `cdr3_seq_aa_q_trim` as the sequence column
and the `fisher` package's `pvalue_npy` for the statistical test (Step 2). Lite
uses `cdr3_aa` and `scipy.stats.hypergeom.sf`. Both produce identical results.
The clustering algorithm (single-linkage on Hamming distance) is identical.

### Step 2: Fisher's Exact Test (Cluster Enrichment)

**Goal**: For each cluster, test whether it is significantly enriched for a
specific disease — i.e., does it contain more patients of that disease than
expected by chance?

#### 2a. Counting Unit

The counting unit is **unique participants**, not sequences or specimens. If one
participant has 10 sequences in a cluster, they count as 1. This prevents
high-depth participants from dominating the enrichment signal.

#### 2b. Contingency Table

For each (cluster, disease_class) pair, a one-vs-rest 2x2 contingency table is
constructed:

```
                     | NOT in cluster | In cluster
   This disease      |       c        |     a
   All other diseases|       d        |     b
```

Where a, b, c, d are counts of unique participants:
- `a` = participants with this disease who are in this cluster
- `b` = participants with other diseases who are in this cluster
- `c` = participants with this disease who are NOT in this cluster
- `d` = participants with other diseases who are NOT in this cluster

#### 2c. Statistical Test

**Right-tail hypergeometric test** (equivalent to one-sided Fisher's exact test):

`p = hypergeom.sf(a - 1, M=total_participants, n=cluster_size, N=n_disease)`
`  = P(X >= a)`, where `X ~ Hypergeom(M, n, N)`

This tests: "given that the cluster contains `a+b` participants total and there
are `a+c` participants with this disease overall, what is the probability of
seeing `a` or more disease participants in the cluster by chance?"

When `a = 0`: `sf(-1) = 1.0` (no enrichment).

The original uses the `fisher` package's `pvalue_npy` function for vectorized
Fisher exact test computation. Lite uses `scipy.stats.hypergeom.sf`. Both
compute the same right-tail p-value.

#### 2d. No Multiple Testing Correction

No Bonferroni, FDR, or other multiple testing correction is applied. Instead,
the p-value threshold is treated as a hyperparameter and selected via grid
search on train_smaller2 (Step 5).

**Output**: A DataFrame indexed by `(v_gene, j_gene, cdr3_len, cluster_id)`
with one column per disease class, containing right-tail p-values.

### Step 3: Pre-Filtering and Centroid Computation

#### 3a. Pre-Filtering

Before computing centroids (which is expensive), clusters that are not
significant for any disease at any candidate p-value threshold are discarded.
The criterion: keep clusters where `min(p-values across diseases) <= max(p_values_candidates)`.
With default candidates `[0.0005, 0.001, 0.005, 0.01, 0.05]`, this means
keeping clusters with p <= 0.05 for at least one disease.

This is a major optimization: typically ~1M clusters are created but only ~10K
survive this filter, making centroid computation ~100x faster.

#### 3b. Centroid Computation

**Goal**: For each surviving cluster, compute a single representative
(consensus) CDR3 sequence that serves as the cluster's "centroid."

**Algorithm**: Weighted majority-vote consensus at each amino acid position.

1. Within each cluster, deduplicate CDR3 sequences, counting occurrences.
2. Each unique CDR3's weight = `occurrence_count * num_clone_members`.
3. For each position in the CDR3, the consensus amino acid is the one with the
   highest total weight across all sequences in the cluster (`weighted_mode`).
4. The consensus amino acids are concatenated into the centroid string.

Example: cluster with sequences ["CASSLG", "CASSLG", "CASSLA"] (clone sizes
[1, 2, 1]). Weights: CASSLG has 1*1 + 1*2 = 3, CASSLA has 1*1 = 1. At position
5: 'G' has weight 3, 'A' has weight 1. Centroid position 5 = 'G'. Full centroid:
"CASSLG".

#### 3c. Merge

Centroids are merged with their per-disease Fisher p-values into a single
`centroids_with_scores` DataFrame. This artifact is saved and used for all
downstream featurization (both during p-value grid search and at test time).

**Difference**: None. Both codebases use the same `make_consensus_sequence`
implementation from `genetools.arrays` (original) / `malid_lite.utils.arrays`
(Lite).

### Step 4: Featurization — Cluster Hits to Feature Matrix

**Goal**: Given a set of sequences (from train_smaller1, train_smaller2, or
test), produce a specimen-level feature matrix where each cell counts how many
disease-associated clusters that specimen's sequences match.

#### 4a. Filter Predictive Clusters

From the pre-filtered `centroids_with_scores`, keep only clusters whose minimum
p-value across diseases is <= the current `p_value_threshold`. This is a tighter
filter than the pre-filtering in Step 3a (which used the loosest candidate
threshold).

If no clusters pass: all specimens abstain.

#### 4b. Assign Sequences to Nearest Centroids

For each input sequence, find the nearest cluster centroid within the same
`(v_gene, j_gene, cdr3_len)` supergroup:

1. Sequences are deduplicated on `(v_gene, j_gene, cdr3_len, cdr3_aa)` before
   distance computation (optimization — one distance computation per unique
   combination, then assignments are merged back to all rows).
2. Pairwise Hamming distances are computed between each input sequence and all
   centroids in its supergroup: `cdist(test_vecs, centroid_vecs, metric="hamming")`.
3. Distances exceeding `1 - sequence_identity_threshold` are masked.
4. Each sequence is assigned to the nearest unmasked centroid via `masked_argmin`.
5. Sequences with no centroid within threshold receive `cluster_id = NaN`
   (unassigned).
6. Sequences in supergroups with no training centroids receive `cluster_id = NaN`.

Assignments are merged back to the full (non-deduplicated) DataFrame.

#### 4c. Build Cluster-Disease Association Table

The `centroids_filtered` DataFrame (one row per cluster, one column per disease
class with p-values) is melted to long format: one row per (cluster,
disease_class) pair. Pairs where `p > p_value_threshold` are dropped. The result
is a lookup table of which disease(s) each cluster predicts.

A single cluster can be associated with multiple diseases (rare at tight
thresholds like 0.0005, more common at permissive ones like 0.05).

#### 4d. Collapse to Unique (Specimen, Cluster) Hits

1. Drop sequences with NaN `cluster_id` (unassigned).
2. Deduplicate to unique `(specimen, cluster)` pairs — a specimen matching the
   same cluster with 10 sequences still counts as 1 hit.
3. Inner-join with the cluster-disease association table to annotate each
   (specimen, cluster) hit with the disease class(es) it predicts.

#### 4e. Scoring: Count Distinct Clusters Per Disease

For each `(specimen, disease_class)` pair, count the number of **distinct global
cluster IDs** matched. This is the feature value.

**Scoring unit**: unique clusters, not sequences or clone members. A specimen
matching 3 different COVID-associated clusters scores 3, regardless of how many
individual sequences fell into each cluster.

#### 4f. Pivot to Feature Matrix

Convert from long format to wide:
- **Rows**: specimens
- **Columns**: disease classes (alphabetically sorted)
- **Values**: integer cluster-hit counts (>= 0)

Missing `(specimen, disease_class)` pairs are filled with 0. All disease class
columns are present even if no specimen scored for a particular class.

**Invariant**: Every row in the feature matrix must have at least one non-zero
cell. All-zero rows indicate a bug (the specimen should have been classified as
an abstention).

#### 4g. Abstention

Specimens present in the input but absent from the feature matrix had zero
sequences match any predictive cluster. They are separated into
`abstained_sample_names` and `abstained_sample_y` in the `FeaturizedData`
output.

**Difference**: None. Both codebases implement the same scoring logic (unique
cluster IDs per specimen per disease class). The original additionally tracks
`total_num_clone_members` and `total_num_clones` per (specimen, cluster) pair
in the intermediate DataFrame, but these are not used in the final score — only
`nunique(global_resulting_cluster_ID)` matters.

### Step 5: P-Value Grid Search and Threshold Selection

**Goal**: Select the p-value threshold that produces the best classification
performance, balancing the tradeoff between strictness (fewer clusters, more
abstentions, higher confidence) and permissiveness (more clusters, fewer
abstentions, more noise).

#### 5a. Candidate Grid

Default p-value candidates: `[0.0005, 0.001, 0.005, 0.01, 0.05]`.

For each candidate p-value, and for each model type (default: one per locus):

1. **Featurize train_smaller1** at this p-value -> `fd_train`
2. **Featurize train_smaller2** at this p-value -> `fd_val`
3. Skip if either produces zero scored specimens or only one disease class.
4. **Build pipeline**: `StandardScaler` -> `GlmnetLogitNetWrapper`
5. **Fit** on `fd_train.X`, `fd_train.y` with participant-level CV grouping.
6. **Predict** on `fd_val.X`.
7. **Compute MCC-with-abstention** on train_smaller2.

#### 5b. MCC-with-Abstention

Abstained specimens are treated as misclassifications:
- Append abstained specimens to `y_true` with their real disease labels.
- Append a synthetic "UNKNOWN99" label to `y_pred` for each abstained specimen.
- Compute sklearn's `matthews_corrcoef` on the expanded arrays.

This penalizes high abstention rates: a model that abstains on all specimens
gets MCC = 0. A model with perfect accuracy but 50% abstention still loses MCC
for the abstained half.

This matches the original Mal-ID's `crosseval` approach.

#### 5c. Best P-Value Selection

For each model type, select the p-value with the **highest MCC-with-abstention**
on train_smaller2.

### Step 6: Classifier Training

**Goal**: Train the final classifier at the best p-value threshold.

#### 6a. Training Data

By default (`retrain_on_full_train=False`, matching original Mal-ID): the final
classifier is trained on **train_smaller1 only**, featurized at the best
p-value.

Optionally (`retrain_on_full_train=True`): the final classifier is trained on
**train_smaller1 + train_smaller2 combined**. Clusters and Fisher p-values
remain frozen from train_smaller1 — only the regression training pool is
expanded. This option is not used in the original Mal-ID.

#### 6b. Classifier Pipeline

```
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GlmnetLogitNetWrapper(...)),
])
```

**Input features**: The feature matrix from Step 4 — one row per scored
specimen, one column per disease class, values are integer cluster-hit counts.

**Target variable**: The specimen's disease label.

#### 6c. GlmnetLogitNetWrapper Configuration

| Parameter        | Value                                                            | Meaning                                                        |
| ---------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `alpha`          | Depends on model type: lasso_cv=1.0, ridge_cv=0.0, etc.          | L1/L2 ratio. Default: lasso for TCR, ridge for BCR.            |
| `n_lambda`       | 100                                                              | Lambda path size                                               |
| `internal_cv`    | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | 5-fold CV for lambda selection, grouped by `participant_label` |
| `scoring`        | Deviance (log-loss)                                              | CV scoring metric                                              |
| `standardize`    | False                                                            | `StandardScaler` in pipeline handles this                      |
| `use_lambda_1se` | False                                                            | Use lambda minimizing CV deviance (not 1-SE rule)              |
| `class_weight`   | "balanced"                                                       | Upweight minority classes                                      |

**Model type selection**: In the original Mal-ID, all 5 alpha variants are
trained, evaluated on a validation set, and the best is manually hardcoded per
locus (`lasso_cv` for TCR, `ridge_cv` for BCR). Lite skips this step and uses
the pre-made result via `BEST_MODEL_FOR_METAMODEL`, training only that one model
by default.

#### 6d. Hyperparameter Summary

Three hyperparameters are tuned at different levels:

| Hyperparameter                       | How tuned                                                      | Selection criterion                |
| ------------------------------------ | -------------------------------------------------------------- | ---------------------------------- |
| **Lambda** (regularization strength) | Automatically by glmnet's internal 5-fold CV on train_smaller1 | Deviance (log-loss)                |
| **P-value threshold**                | Grid search over 5 candidates, evaluated on train_smaller2     | MCC-with-abstention                |
| **Alpha** (L1/L2 ratio)              | Fixed per locus (pre-selected in original Mal-ID)              | N/A (lasso for TCR, ridge for BCR) |

### Step 7: Prediction at Test Time

Given a test specimen's sequences:

1. **Assign sequences to training clusters** (Step 4b): For each sequence, find
   the nearest centroid within the same (v_gene, j_gene, cdr3_len) supergroup.
   Sequences beyond the distance threshold are unassigned.

2. **Featurize** (Steps 4c-4f): Count unique cluster hits per disease class ->
   feature matrix. Specimens with zero hits abstain.

3. **Scale**: `StandardScaler.transform()` on the feature matrix.

4. **Predict**: `GlmnetLogitNetWrapper.predict_proba()` converts logits to
   probabilities via **sigmoid** (binary: 2 classes) or **softmax** (multiclass:
   3+ classes). Binary sigmoid produces `[1-P, P]`; multiclass softmax normalizes
   the K-dimensional logit vector to a proper probability distribution summing to
   1.

**Output**:
- **Scored specimens**: class probability vector (multiclass: columns in
  alphabetical order; binary: [P(reference), P(disease)]).
- **Abstained specimens**: no prediction. Tracked separately in `FeaturizedData`.

### Handling of Specimens with Insufficient Data

Model 2 has an explicit **abstention mechanism**. A specimen abstains (receives
no prediction) if none of its CDR3 sequences match any disease-significant
cluster. This can happen when:

1. **No significant clusters exist** at the chosen p-value threshold (Fisher's
   test produced no enriched clusters). All specimens abstain.
2. **No sequence matches any centroid** within the Hamming distance threshold.
   The specimen's sequences are all in supergroups with no training centroids, or
   all distances exceed `1 - sequence_identity_threshold`.
3. **Sequences match centroids, but none of the matched clusters are
   significant** at the chosen p-value threshold. (This is handled implicitly:
   Step 4a filters centroids before assignment, so only significant centroids are
   considered.)

Abstention affects evaluation metrics:

- **MCC**: abstentions count as misclassifications (appended with a synthetic
  "UNKNOWN99" label). This penalizes models with high abstention rates.
- **AUROC and AUPRC**: computed on scored specimens only. Abstained specimens are
  excluded because no probability vector exists. These metrics therefore do not
  reflect the model's inability to predict for abstained specimens.

Abstention rates typically range from ~5% to ~30% depending on the p-value
threshold and disease. This is by design: Model 2 trades coverage for precision.

### Summary of Differences (Original Mal-ID vs Mal-ID-Lite)

| #   | Aspect                      | Original                                               | Lite                                                        | Impact                                                       |
| --- | --------------------------- | ------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 1   | Fisher test implementation  | `fisher` package `pvalue_npy` (vectorized C)           | `scipy.stats.hypergeom.sf` (vectorized Python)              | Same result. Lite avoids an extra dependency.                |
| 2   | CDR3 column name            | `cdr3_seq_aa_q_trim`                                   | `cdr3_aa`                                                   | Column rename only.                                          |
| 3   | Model type selection        | All 5 alpha variants trained; best hardcoded per locus | Only the pre-selected best model trained by default         | Lite skips redundant work. Same final model.                 |
| 4   | Retrain on full train       | Not implemented                                        | Optional (`retrain_on_full_train=True`)                     | Lite offers an additional option. Default matches original.  |
| 5   | Validation split            | Uses a separate validation set (~2/9 of N)             | Not implemented; train_smaller1/2 are proportionally larger | Lite's train splits are larger due to no validation holdout. |
| 6   | Featurization intermediates | Tracks `total_num_clone_members`, `total_num_clones`   | Does not track these (unused in final score)                | No impact on results.                                        |

For the paper-best configuration, both codebases produce equivalent results.
The clustering algorithm, Fisher test logic, centroid computation, scoring
(unique cluster IDs per specimen per disease), and classifier pipeline are
identical.

---

## Model 3: Sequence-Level Classifier

Model 3 is a sequence-level disease classifier for immune repertoire data. It
predicts the disease associated with a specimen (a single blood sample /
repertoire) by analyzing the CDR3 regions of its TCR (or BCR) sequences.

The model has two stages:

- **Stage 1** operates at the **individual sequence level**, within V-gene
  groups. For each V-gene group, a classifier is trained to predict disease from
  single-sequence ESM-2 embeddings. Each sequence receives a disease probability
  vector.

- **Stage 2** operates at the **specimen level**. It aggregates all per-sequence
  predictions for a specimen into a fixed-length feature vector, then trains a
  RandomForest-based classifier to predict the specimen's disease.

Disease is a participant-level attribute (each participant has one disease), but
a single participant may contribute multiple specimens (e.g., different
timepoints or sample types). All sequences within a specimen share the same
disease label. The pipeline operates at the specimen level throughout: Stage 1
labels are derived from each sequence's specimen, and Stage 2 predictions are
per-specimen.

### Step 0: Training Data Split

For each cross-validation fold, the training data is split into two
non-overlapping subsets at the **participant level** (all specimens from the same
participant go to the same subset):

- **train_smaller1** (2/3 of training participants) — used to train Stage 1
- **train_smaller2** (1/3 of training participants) — used to train Stage 2

The split is stratified by disease to maintain class balance in both subsets, and
uses a fixed random seed (`random_state=0`) for reproducibility.

This separation prevents data leakage: Stage 2 trains on Stage 1's predictions
for specimens it has never seen, giving it a realistic estimate of Stage 1's
out-of-sample performance. If Stage 2 trained on specimens Stage 1 had already
memorized, it would learn to over-trust Stage 1's outputs.

Reference: `base.py:560-573` (`train_test_split(..., test_size=1/3,
stratify=diseases)`)

The **train-all** contexts apply this same ts1/ts2 split over the whole dataset
(no CV fold, no held-out test, no evaluation): `train_all` uses every participant
(ts1 = 2/3, ts2 = 1/3); `train_all_ensemble` first holds out ~1/3 of participants
as the ensemble metamodel's validation set, then splits the remaining ~2/3 into
ts1/ts2. Stage 1 still trains on ts1 and Stage 2 on ts2 exactly as in CV — the
leakage-prevention rationale above is identical. Train-all produces the reusable
`stage1.pkl` / `stage2.pkl` (no fold prefix), to be scored later on a separate
dataset. See `train_model3.py::train_full_dataset` / `_run_train_all`.

### Step 1: ESM-2 Embedding Extraction

**Goal**: Represent each CDR3 amino acid sequence as a fixed-length numeric
vector capturing its biochemical and structural properties.

**Model**: `esm2_t30_150M_UR50D` — a pre-trained protein language model (30
transformer layers, 150M parameters, 640-dimensional hidden representations).
Used off-the-shelf, not fine-tuned.

**Process** (identical in both codebases):

1. **Tokenization**: CDR3 amino acid strings are tokenized via ESM-2's
   `alphabet.get_batch_converter()`. Each amino acid becomes one token, with BOS
   (beginning-of-sequence) and EOS (end-of-sequence) special tokens added
   automatically.

2. **Forward pass**: Representations are extracted from **layer 30** (the final
   layer). For a CDR3 of length L, the output tensor has shape `(L+2, 640)` — L
   amino acid positions plus 2 special tokens.

3. **Special token removal**: Positions 0 (BOS) and L+1 (EOS) are stripped,
   retaining only the L amino acid positions: `representations[1 : L+1]`.

4. **Mean pooling**: The L position-level vectors are averaged to produce a
   single 640-dimensional embedding per sequence:
   `representations[1:L+1].mean(dim=0)`.

5. **Storage**: Embeddings are stored as float16.

**Output**: One 640-dim vector per CDR3 sequence.

**Difference**: None.

### Step 2: V-Gene Group Formation

**Goal**: Partition sequences by V-gene (and isotype for BCR) so that Stage 1
trains specialized classifiers per group. The biological rationale is that V-gene
usage shapes CDR3 properties (length distribution, amino acid composition), so a
classifier conditioned on V-gene context can learn more specific
disease-associated patterns.

**Grouping**:

- **TCR**: group by `v_gene` — e.g., `(TRBV5-1,)`
- **BCR**: group by `(v_gene, isotype_supergroup)` — e.g., `(IGHV3-23, IGHG)`

**Filtering**:

- **Rare V-gene filtering**: For each V-gene, compute its maximum relative
  frequency across disease classes. V-genes whose maximum frequency falls below
  the median of all V-genes' maximum frequencies are excluded. For TCR, typically
  ~28 V-genes survive out of ~50+ total.

- **Minimum group size**: Groups with fewer than 10 sequences are skipped.

**Column sanitization**: Underscores in V-gene and isotype values are replaced
with hyphens (`_` -> `-`) to avoid ambiguity in feature column names, which use
`_` as the class/group delimiter.

**Difference**: None.

### Step 3: Stage 1 — Per-V-Gene Classifier Training

**Goal**: For each V-gene group, train a classifier whose input is a single CDR3
embedding and whose target is the disease label of the specimen that produced
that sequence. The classifier learns: "given the ESM-2 representation of this
CDR3 (from a specific V-gene group), which disease is the originating specimen
most likely associated with?"

#### 3a. Input Features

Per sequence in the group:

- **TCR**: the 640-dim ESM-2 embedding (float32 at computation time)
- **BCR**: 641 features = 640-dim ESM-2 embedding + somatic hypermutation rate
  (`v_mut`). Missing `v_mut` values are filled with 0.

#### 3b. Target Variable

The disease label of each sequence's specimen (categorical — e.g., one of
{COVID-19, HIV, Healthy, Influenza, Lupus, T1D} for the 6-class multiclass
setting).

Reference: `model3_sequence_level.py:1202` —
`y = sequences_df[DISEASE_COL].values`

#### 3c. Per-Group StandardScaler

A `StandardScaler` is fitted **independently per V-gene group** on that group's
training sequences. Each feature dimension is centered (subtract mean) and
scaled (divide by std) based on the within-group distribution, ensuring all 640
embedding dimensions are on comparable scales and accounting for V-gene-specific
distributional differences. The fitted scaler is saved per group for reuse at
prediction time.

#### 3d. Single-Class Group Handling

If all training sequences in a group come from a single disease class, no
classifier can be meaningfully trained:

| Scenario | Original TCR                                                                                                               | Original BCR                                                                                                                                                                                 | Lite (both loci)                                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Behavior | OvR raises `ValueError("Only one class in data")`. Group skipped. Sequences receive NaN and are excluded from aggregation. | RF trains: `classes_ = [single_class]`, `predict_proba` -> `[[1.0]]`. After alignment to global classes (zero-fill): `[1, 0, ..., 0]` with entropy = 0 — always survives the entropy filter. | Pre-filter: `len(unique_labels) < 2` -> skip. Sequences receive NaN and are excluded, same as original TCR. |

**Key difference**: For **BCR only**, the original trains single-class groups,
producing `[1, 0, ..., 0]` predictions that inject V-gene frequency signal into
Stage 2. Lite skips them. For TCR, behavior is equivalent.

#### 3e. Classifier Configuration

**TCR — One-vs-Rest (OvR) Ridge Regression**

For a K-class problem, the OvR wrapper (`CustomOneVsRestClassifier`) creates K
independent binary classifiers. Each is trained on a binarized version of the
labels: "is this sequence from disease X?" (positive) vs. "any other disease?"
(negative).

Each binary classifier is a `GlmnetLogitNetWrapper` (logistic regression via the
glmnet coordinate descent algorithm):

| Parameter        | Value                                                            | Meaning                                                                                                                                                                 |
| ---------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`          | 0.0                                                              | Pure L2 (ridge) penalty — no L1/lasso                                                                                                                                   |
| `n_lambda`       | 100                                                              | Size of the regularization path                                                                                                                                         |
| `standardize`    | False                                                            | Scaling handled externally by the per-group scaler                                                                                                                      |
| `use_lambda_1se` | False                                                            | Select lambda minimizing CV deviance (not the 1-SE rule)                                                                                                                |
| `class_weight`   | "balanced"                                                       | Upweight the minority class in each binary subproblem                                                                                                                   |
| `internal_cv`    | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | 5-fold stratified CV for lambda selection, grouped by `participant_label` to prevent same-participant sequences from appearing in both inner train and inner validation |

OvR-level settings:

| Parameter                             | Value | Meaning                                                                                                                   |
| ------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------- |
| `normalize_predicted_probabilities`   | False | The K binary outputs are **not** normalized to sum to 1. Each P(disease_k) is independent; typical row sums are ~2.0-3.5. |
| `allow_some_classes_to_fail_to_train` | True  | If a binary sub-classifier fails (e.g., insufficient samples for a class), skip it rather than abort the entire group.    |

**BCR — RandomForestClassifier (multiclass, not OvR)**

| Parameter      | Value                |
| -------------- | -------------------- |
| `n_estimators` | 100                  |
| `class_weight` | "balanced_subsample" |
| `random_state` | 0                    |
| `n_jobs`       | 1                    |

Wrapped in a `GroupSequenceClassifier` that handles per-group scaling and class
alignment.

#### 3f. Parallelization

- **Lite TCR (flattened OvR)**: All (group x class) binary jobs (~28 x 6 = ~168)
  are flattened into a single `Parallel(n_jobs=4)` call for better load
  balancing.
- **Lite BCR / Original (both)**: Parallelized at the group level.
- Results are mathematically identical regardless of parallelization strategy.

### Step 4: Stage 1 — Prediction (`predict_proba`)

**Goal**: Given a set of sequences (train_smaller2 during training, or test data
during evaluation), produce a per-sequence disease probability vector using the
trained per-group classifiers.

#### 4a. Per-Group Prediction

For each V-gene group with a trained model:

1. Select sequences in that group (boolean mask on the V-gene column).
2. Build features: 640-dim embeddings (TCR) or 641-dim (BCR).
3. Apply the group's saved StandardScaler (fitted in Step 3c).
4. Call `classifier.predict_proba()` on the scaled features.

#### 4b. Probability Assembly

**TCR (OvR ridge)**: Each of the K binary classifiers outputs P(positive class)
via `predict_proba()[:, 1]`. These K values are stacked into a K-dimensional
vector per sequence. With `normalize_predicted_probabilities = False`, these are
K independent probabilities that do **not** sum to 1.

Example for one sequence (K=6): `[0.7, 0.3, 0.2, 0.1, 0.4, 0.2]` — sum = 1.9.

For the **binary case** (K=2): a single estimator predicts P(class_1); the
output is `[1 - P(class_1), P(class_1)]`, which sums to exactly 1.0.

**BCR (RF multiclass)**: RF outputs a proper probability distribution over
classes it saw during training (sums to ~1.0 within those classes).

#### 4c. Class Alignment (Zero-Filling for Unseen Classes)

A group's classifier may have seen only a subset of the global K classes during
training. Unseen classes are padded with probability **0**:

- **Original**: `DataFrame.reindex(columns=global_classes).fillna(0)`
- **Lite**: Pre-allocate a zeros array of shape (n_samples, K), scatter
  known-class probabilities into the correct columns.

Example: group trained on [COVID-19, Healthy] predicts `[0.6, 0.4]`. After
alignment to global order: `[0.6, 0, 0.4, 0, 0, 0]`. The zero-padded entries
produce artificially low entropy, which is relevant for the entropy filter in
Step 5b.

#### 4d. Sequences Without Trained Models

Sequences whose V-gene group has no trained model receive NaN probabilities,
are flagged `has_prediction = False`, and are **excluded from all downstream
aggregation**.

#### 4e. Output

A DataFrame with one row per input sequence:

- `prob_<class>` columns — float32 in Lite, float64 in original (NaN if no model)
- `specimen_label`, `participant_label`, V-gene group columns
- `weight` — sample weight (1.0 for TCR; isotype-rebalancing weight for BCR)
- `has_prediction` — boolean flag

**Minor difference**: Lite uses float32 for probabilities (saves ~12 MB for
large datasets, no measurable precision impact). Original uses float64.

### Step 5: Aggregation — Sequence-Level to Specimen-Level Features

**Goal**: Compress all per-sequence probability vectors for a given (specimen,
V-gene group) combination into a single summary vector. The result is a
fixed-size feature matrix suitable for Stage 2.

#### 5a. Filtering

Only sequences with `has_prediction = True` participate. Sequences from V-gene
groups without trained models are excluded.

#### 5b. Per-(Specimen, V-Gene Group) Aggregation

For each (specimen, V-gene group) pair, the sequences' probability vectors are
aggregated using one of the following strategies:

**Mean** (paper-best for BCR):

Weighted average of all sequence probability vectors:
`agg[k] = sum(prob_k_i * w_i) / sum(w_i)` for each class k. For TCR (uniform
weights), this is a simple arithmetic mean.

**Entropy-based filtering** (paper-best for TCR:
`entropy_twenty_percent_cutoff`):

Retains only sequences whose probability vector has low Shannon entropy
(concentrated on few classes), discarding uncertain sequences.

1. **Maximum entropy**: `H_max = ln(K)` nats, the entropy of a uniform
   distribution over K classes. For K=6: H_max = ln(6) = 1.792 nats.

2. **Threshold**: `threshold = max_fraction * H_max`. For twenty_percent_cutoff:
   `threshold = 0.80 * 1.792 = 1.433 nats`.

3. **Per-sequence entropy**: `scipy.stats.entropy()` is applied to each
   sequence's probability vector. **Important**: scipy internally normalizes the
   input to sum to 1 before computing entropy. For unnormalized OvR vectors
   (which don't sum to 1), this normalization materially changes the
   distribution shape.

4. **Filter**: Retain sequences where `entropy < threshold` (strict `<`).

5. **Aggregate**: If some sequences survive, take their weighted mean. If **no
   sequences survive**, return the uniform prior `[1/K, ..., 1/K]`.

**The multiclass entropy filter problem**: With K=6 and unnormalized OvR
outputs, even strongly "opinionated" vectors produce high entropy after scipy's
internal normalization. Example: `[0.9, 0.2, 0.2, 0.2, 0.2, 0.2]` normalizes to
`[0.47, 0.11, ...]` with entropy = 1.54 nats, exceeding the threshold of 1.43.
Nearly all sequences are filtered out, and aggregation falls back to the
uninformative uniform prior for most (specimen, group) pairs. In the binary case
(K=2), OvR outputs sum to exactly 1.0 and the filter works as intended. See
`dev_MDs/MODEL3_MULTICLASS_ISSUE.md` for detailed analysis.

#### 5c. Cases That Produce Uniform 1/K Values

A (specimen, V-gene group) cell in the feature matrix receives the uniform prior
`[1/K, ..., 1/K]` in three cases:

1. **No sequences for that (specimen, group) pair** — either the specimen has no
   sequences in that V-gene group, or all its sequences in that group lacked a
   trained model. The pre-allocated output array is filled with `1/K`, and these
   cells are never overwritten.

2. **Entropy filter removes all sequences** — all sequences in the pair have
   `entropy >= threshold`. The function returns `np.ones(K) / K`.

3. **Empty probs array** — zero-length array passed to `aggregate_group` (edge
   case). Returns `np.ones(K) / K`.

#### 5d. Weight Handling

- **TCR**: Uniform weights (1.0 for all sequences).
- **BCR**: Sequences may carry isotype-rebalancing weights
  (`sample_weight_isotype_rebalance`). If all weights are NaN, treated as
  uniform 1.0.

#### 5e. Feature Matrix Structure

After aggregation, the output is a wide DataFrame:

- **Rows**: specimens (one per specimen in the input).
- **Columns**: one per (disease class, V-gene group) pair, named
  `"{class}_{group_key}"`.
  - Example: `"COVID-19_TRBV5-1"`, `"HIV_TRBV5-1"`, ..., `"T1D_TRBV7-2"`.
  - For TCR with 6 classes and 28 V-genes: 6 x 28 = 168 features per specimen.
- **Values**: aggregated probabilities (or uniform 1/K for missing/filtered
  cells).

#### 5f. Specimens With Zero Valid Predictions

If ALL of a specimen's sequences belong to V-gene groups without trained models:

- **Original**: The specimen is absent from the feature matrix — the groupby
  produces no rows for it, and Stage 2 never sees it.
- **Lite**: The specimen is included with all-uniform features (from
  pre-allocation). After scaling and frequency multiplication, its features
  become all zeros. Stage 2 produces an uninformative prediction.

Extremely rare in practice.

#### 5g. Difference: Entropy Zero-Weight Edge Case

When the entropy filter retains some sequences but all surviving sequences have
zero weight:

- **Original**: Returns 0 for each class.
- **Lite**: Returns the unweighted mean of surviving sequences.

Practically impossible (TCR has uniform weights; BCR isotype weights are always
> 0 for valid sequences).

### Step 6: Pre-Aggregation StandardScaler

**Goal**: Standardize the aggregated features before V-gene frequency reweighing
(Step 7), so all feature columns are on comparable scales prior to
multiplication.

A `StandardScaler` is fitted on the aggregated feature matrix from Step 5. For
each column: `(x - mean) / std`. Saved as `preagg_scaler_`.

The original uses `StandardScalerThatPreservesInputType` (a thin wrapper
preserving DataFrame types). Same math as sklearn's `StandardScaler`.

**Difference**: None.

### Step 7: V-Gene Frequency Reweighing

**Goal**: Modulate each feature by the relative frequency of its V-gene group
within each specimen. This downweights features from rare V-gene groups and
upweights features from dominant ones, encoding V-gene usage patterns as
additional signal for Stage 2.

#### 7a. Counting Sequences Per (Specimen, Group)

For each specimen, count how many sequences belong to each V-gene group. Only
sequences with `has_prediction = True` are counted (i.e., sequences in groups
with trained models).

**Equivalence note**: The original counts all sequences first, then drops groups
without trained models before normalization (`normalize_after_subsetting =
True`). Lite counts only sequences with predictions directly. Both produce
identical frequencies.

#### 7b. Normalization to Frequencies

- **TCR**: Row-normalize so frequencies sum to 1 per specimen. Example: groups
  [TRBV5-1, TRBV7-2] with counts [30, 12] -> frequencies [0.714, 0.286].

- **BCR**: Normalize **within each isotype separately** so each isotype's V-gene
  frequencies sum to 1 per specimen. Rationale: isotype proportions (IgG/IgA/IgM
  split) are technical artifacts of sample preparation; V-gene usage within an
  isotype is biological signal.

Division by zero (specimen has no sequences in a group or isotype) produces NaN,
filled to 0.0.

#### 7c. Frequency Replication

The frequency for a V-gene group is replicated across all class columns for that
group. Both `"COVID-19_TRBV5-1"` and `"Healthy_TRBV5-1"` receive the same
frequency value (freq(TRBV5-1) for that specimen).

#### 7d. Element-Wise Multiplication

`reweighed_features = scaled_features * frequencies`

Each feature value is multiplied by how prevalent its V-gene group is in that
specimen. Rare V-gene groups contribute proportionally less; dominant groups
contribute more.

**Difference**: None.

### Step 8: Stage 2 StandardScaler

A second `StandardScaler` is fitted on the frequency-reweighed features,
re-centering and re-scaling all columns (mean=0, std=1) before the Stage 2
classifier. Saved as `stage2_scaler_`.

In the original, this scaler is prepended to the model pipeline via
`prepend_scaler_if_not_present()`. In Lite, it is applied as an explicit
separate step. Same math.

**Difference**: None.

### Step 9: Stage 2 — Specimen-Level Classifier

**Goal**: Given the per-specimen feature matrix (e.g., 168 features for 6-class
TCR: the aggregated, scaled, frequency-reweighed disease probabilities across
V-gene groups), predict each specimen's disease label.

#### 9a. BinaryOvRClassifierWithFeatureSubsettingByClass

Stage 2 uses a One-vs-Rest structure with a key design choice: **each binary
classifier receives only its own class's feature columns**.

For a K-class problem, K independent binary RandomForest classifiers are
created:

1. **"COVID-19 vs rest" classifier**:
   - **Input features**: only columns prefixed `"COVID-19_"` — the aggregated
     Stage 1 COVID-19 probability across all V-gene groups. For 28 V-genes: 28
     features.
   - **Target**: binary — 1 if the specimen's disease is COVID-19, 0 otherwise.
   - **What it learns**: which pattern of COVID-19 Stage 1 probabilities across
     V-gene groups (weighted by V-gene frequency) is characteristic of actual
     COVID specimens vs. non-COVID specimens.
   - **Output**: P(specimen is COVID-19).

2. **"HIV vs rest"**: same structure, using only `"HIV_..."` columns.

3. ... and so on for each disease class.

The feature subsetting is biologically motivated: the Stage 1 COVID-19
probability pattern across V-gene groups is the most informative signal for
predicting COVID-19.

#### 9b. Binary Case (K=2)

Only one classifier is trained, using the non-reference (disease) class's
features. The reference class probability is derived as `1 - P(disease)`.

#### 9c. RandomForest Configuration (Stage 2)

| Parameter      | Value                                                |
| -------------- | ---------------------------------------------------- |
| `n_estimators` | 100                                                  |
| `class_weight` | "balanced_subsample"                                 |
| `random_state` | 0                                                    |
| `n_jobs`       | 1 (inner parallelism at the OvR level, not per-tree) |

#### 9d. Output

Each binary classifier independently outputs P(its class). The K values are
assembled into a probability vector. Since they are independent binary outputs,
they do **not** sum to 1.

**Difference**: None.

### Step 10: Full Prediction Pipeline at Test Time

Given a test specimen's sequences and pre-computed ESM-2 embeddings:

1. **Stage 1 predict** (Step 4): Per V-gene group, apply saved scaler, run
   `predict_proba()`, zero-fill unseen classes. Result: per-sequence probability
   vectors.

2. **Aggregate** (Step 5): Per (specimen, V-gene group), apply the trained
   aggregation strategy (entropy filter or mean). Result: specimen-level feature
   matrix.

3. **Pre-agg scaling** (Step 6): `preagg_scaler_.transform()` (training-fitted).

4. **V-gene frequency reweighing** (Step 7): Compute frequencies from test data,
   normalize, multiply with scaled features.

5. **Final scaling** (Step 8): `stage2_scaler_.transform()` (training-fitted).

6. **Stage 2 predict** (Step 9): BinaryOvR RF with feature subsetting produces
   the final disease probability vector.

**Output**: One K-dimensional probability vector per specimen.

### Summary of Differences (Original Mal-ID vs Mal-ID-Lite)

| #   | Aspect                                 | Original                                                                              | Lite                                         | Impact                                                   |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------- |
| 1   | **BCR single-class V-gene groups**     | RF trains -> predictions `[1,0,...,0]` with entropy=0, always survives entropy filter | Skipped via pre-filter                       | Significant for BCR with entropy filtering. Fix planned. |
| 2   | `trim_bottom_five_percent` aggregation | Per-column: sorts each class's probs independently, trims bottom 5% of each           | Removed                                      | Not used in any paper-best config.                       |
| 3   | Specimens with zero valid predictions  | Absent from Stage 2 matrix                                                            | Included with uniform -> zeros after scaling | Extremely rare. Lite more robust.                        |
| 4   | Sequence probability dtype             | float64                                                                               | float32                                      | Intentional optimization. No measurable impact.          |
| 5   | Zero-weight entropy fallback           | Returns 0 per class                                                                   | Returns unweighted mean                      | Practically impossible scenario.                         |
| 6   | TCR parallelization                    | Group-level                                                                           | Flattened (group x class)                    | Identical results, better load balancing.                |

For the **paper-best TCR configuration** (`entropy_twenty_percent_cutoff`), both
codebases behave identically. The entropy filter problem with multiclass OvR
probabilities is inherent to the algorithm design, not a reimplementation bug.
See `dev_MDs/MODEL3_MULTICLASS_ISSUE.md` for detailed analysis.

### Handling of specimens with insufficient data

Model 3 always produces a prediction for every specimen. There is no abstention
mechanism.

If a specimen's sequences all belong to V-gene groups without a trained model
(e.g., all in rare V-genes that were filtered during training), those sequences
receive NaN probabilities in Stage 1 and are excluded by the `has_prediction`
filter during aggregation. For such a specimen:

- Every V-gene group gets the uniform prior fill value of 1/n_classes (since no
  sequences contributed to any group for this specimen).
- During frequency reweighing (when enabled, as in the paper-best TCR config),
  the specimen's V-gene group frequencies are all 0 (no valid sequences to
  count). The element-wise multiplication of scaled features by zero frequencies
  produces an all-zero feature vector.
- After the final StandardScaler, this becomes a constant vector. The Stage 2
  RF classifier still produces a prediction, but it is uninformative.

A warning is logged with the count and percentage of affected specimens. This
situation is extremely unlikely in practice (would require every sequence in a
specimen to belong to a V-gene group that was either too rare or had only one
disease class during Stage 1 training) and indicates a data quality issue if it
occurs.

---

## Ensemble (Meta-Learner)

The ensemble is a **stacked meta-learner**: a ridge-regularized (L2) logistic
regression (`GlmnetLogitNetWrapper`, `alpha=0.0`, MCC scoring, balanced class
weights) trained on the three base models' per-specimen disease **probabilities**
as features.

### Architecture and the anti-leakage split

The subtle part is producing the metamodel's training data without leakage. Each
`*_ensemble` split reserves a **validation** third that the base models do NOT
train on:

- Base models are trained on `train_smaller1 + train_smaller2` (~2/3), with the
  **validation** third (~1/3) held out (the `cv_ensemble` / `train_all_ensemble`
  contexts — see each model's "Step 0").
- The metamodel is trained on the base models' predictions over the **validation**
  third. Because those are base-model *out-of-sample* predictions, the stacked
  features reflect generalization, not memorization.

Feature columns are named `<locus>:<model_display_name>:<class>`; per-model
contribution is therefore derivable from the column names.

### Cross-validation vs. train-all (`--training-context`)

- **`cv` (`cv_ensemble`)** — per fold: train the metamodel on the validation
  third → get base-model predictions on the held-out **test** fold → evaluate.
  Reports test metrics, aggregated across folds.
- **`train_all` (`train_all_ensemble`)** — a single whole-dataset pass: the
  metamodel trains on base-model predictions over the validation third, and there
  is **no test set → no metrics**. The trained metamodel (+ config) is saved for
  evaluation later on a *separate* dataset. Base models must be the
  `train_all_ensemble` variant (validation excluded); a leakage guard rejects base
  models trained as the leaky `train_all` (which saw the validation set).

### Handling of specimens with insufficient data

Model 2 may **abstain** on specimens with zero cluster matches (no probability
vector). The `--model2-abstention-strategy` controls this in the feature matrix:

- `ensemble_abstain` (default): the specimen is dropped from the metamodel
  (original Mal-ID behavior).
- `fill_0.5`: Model 2's features are filled with 0.5 (uninformative prior), so the
  specimen is kept and scored by the other models.
- `fill_models13_mean`: Model 2's features are filled with the mean of Models 1
  and 3's per-class predictions (requires both in `--models`).

Fill/abstention counts are recorded in `val_fill_info` (never silently dropped).
A base model that fully abstains (e.g. Model 2 with no valid clusters anywhere) is
excluded from the feature matrix, and this is logged.

### Artifact separation

Ensemble-trained base models live under `<context>/base_models/<locus>/<model>/`
and are distinct from standalone base models (`*_single_model/`). The metamodel
lives under `<context>/ensemble/<locus>/<mode>/`. See `train_ensemble.py` and
`malid_lite/training/README.md` (Ensemble Meta-Learner) for the exact layout and
the `training_complete` readiness marker used by downstream external evaluation.

### Cross-dataset (external) evaluation

A model trained on the whole of one dataset (`train_all` / `train_all_ensemble`) — or a
specific CV fold's model — can be scored on a **separate** labeled dataset via
`evaluate_external.py`. This is the "test side" of the ensemble applied to the external
data: each base model predicts on the external specimens, the metamodel feature matrix is
rebuilt (in the exact feature-column order the metamodel was trained with), and the
metamodel produces the final prediction. **Metrics use the model's TRAINING label space**
— the model can only predict classes it was trained on, so external specimens whose true
label is outside that space are rejected (or explicitly dropped with a flag). `gene_locus`
must match; the clone_id clustering definition is compared and only warned on. See
`malid_lite/training/README.md` (External Evaluation).

---

## Appendix: Complete Parameter Reference (Supplementary Materials)

A consolidated reference of all algorithmic parameters and hyperparameters across
the Mal-ID-Lite pipeline. Infrastructure parameters (file paths, device, n-jobs,
batch size) are excluded.

### A1. Data Preprocessing

#### A1.1 Stage 1 — CLEAN (Per Participant)

| Step | Parameter | Value | Description |
|------|-----------|-------|-------------|
| Productive filter | condition | `productive == "T"` | Keep only productive rearrangements |
| V-score filter | `V_SCORE_THRESHOLD` | TCR: 80, BCR: 200 | Minimum V-gene alignment quality score |
| Non-standard AA filter | `VALID_AMINO_ACIDS` | 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY) | Drop sequences containing non-standard characters (*, X, ., -) |
| Deduplication | groupby key | `(sequence, replicate_label, [extracted_isotype])` | Sum `num_reads` for duplicate sequences |
| Gene allele corrections | `GENE_ALLELE_FIXES` | `TRBV6-2*02` -> `TRBV6-2*01` | Known allele annotation errors |
| Gene-level corrections | `GENE_FIXES` | `TRBV12-4` -> `TRBV12-3`, `TRBV6-3` -> `TRBV6-2` | Known gene annotation errors |
| Pseudogene removal | `GENES_TO_REMOVE` | `{TRBV25/OR9-2*01}` | No reference data available |
| Drop missing fields | required columns | `v_gene`, `j_gene`, `cdr3_aa` | Drop sequences missing any critical field |

#### A1.2 Clone ID Assignment (if missing from input data)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clone_id_identity_threshold` | TCR-NT: 0.95, TCR-AA: 0.90, BCR-NT: 0.90, BCR-AA: 0.85 | CDR3 sequence identity threshold for hierarchical clustering |
| `clone_id_linkage_method` | `"single"` | Hierarchical clustering linkage method (single/complete/average) |
| `clone_id_use_aa` | `False` | Use amino acid CDR3 (True) or nucleotide CDR3 (False) |
| Distance metric | Normalized Hamming distance | Fraction of mismatching positions between equal-length CDR3s |
| Clustering unit | Sequences sharing same (V gene, J gene, CDR3 length) | Supergroup definition for clustering |

#### A1.3 Stage 2 — DOWNSAMPLED (Per Specimen)

| Step | Parameter | Value | Description |
|------|-----------|-------|-------------|
| CDR3 length filter | minimum | 8 amino acids | Drop CDR3s shorter than 8 residues |
| Clone threshold | `MIN_CLONES["TCRB"]` | 500 | Drop specimen if fewer than 500 unique clones |
| Sequence threshold | `MIN_SEQUENCES` | 1000 | Drop specimen if fewer than 1000 sequences |
| Downsampling | unit | 1 sequence per (specimen, clone, isotype, amplification) | Select the sequence with maximum `num_reads` per unit |

### A2. Cross-Validation Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| Fold assignment | Pre-assigned in metadata (`CV_fold` column) | Folds are not computed dynamically |
| Fold usage | Test = fold_id; Train = all other folds | Standard K-fold cross-validation |
| Training sub-splits | train_smaller1 (~2/3), train_smaller2 (~1/3) | Stratified by disease, `random_state=0`, participant-level split |
| Sub-split usage (M2, M3) | train_smaller1: primary training; train_smaller2: threshold selection / Stage 2 training | Prevents data leakage between stages |
| Sub-split usage (M1) | train_smaller1 + train_smaller2 merged | Single-stage model uses all training data |
| Sub-split usage (ensemble) | Validation set held out from training | Base model predictions on validation set serve as meta-learner features |

### A3. Model 1 — Gene Usage Classifier

#### A3.1 Feature Extraction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Feature unit | V-J gene pair relative frequencies per specimen | One frequency vector per specimen per isotype |
| Isotypes (TCR) | `["TCRB"]` | |
| V-gene frequency filter | `quantile(0.5)` (median) on global training frequencies | Remove bottom ~50% of V-genes by rank |
| Row renormalization | Frequencies re-summed to 1 after filtering | Applied to both train and test |

#### A3.2 Feature Pipeline

| Step | Description |
|------|-------------|
| 1. `log1p` | `x` -> `log(1 + x)` per feature |
| 2. `StandardScaler` | Per-feature centering and scaling (fitted on train) |
| 3. `PCA(n_components=15)` | Per-isotype dimensionality reduction (`random_state=0`) |
| 4. `StandardScaler` | Post-PCA re-scaling of all features (fitted on train) |

#### A3.3 Classifier: Elastic Net Logistic Regression (Glmnet)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` (L1 ratio) | TCR: 1.0 (pure lasso), BCR: 0.25 (elastic net) | Pre-selected per locus from original Mal-ID |
| `n_lambda` | 100 | Regularization path size |
| Internal CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | Participant-grouped for lambda selection |
| CV scoring | Deviance (log-loss) | glmnet default |
| `use_lambda_1se` | `False` | Use lambda minimizing CV deviance |
| `class_weight` | `"balanced"` | Inverse frequency weighting |
| `standardize` | `False` | Handled by pipeline |
| Multiclass mode | Multinomial (softmax) for K >= 3; binary logistic for K = 2 | Joint optimization, not One-vs-Rest |

### A4. Model 2 — Convergent Cluster Classifier

#### A4.1 CDR3 Clustering

| Parameter | Value | Description |
|-----------|-------|-------------|
| Clustering unit | Unique CDR3 sequences within each (V gene, J gene, CDR3 length) supergroup | Sequences can only cluster within same supergroup |
| Distance metric | Normalized Hamming distance | Fraction of differing positions |
| Linkage method | Single | Minimum inter-cluster distance |
| `sequence_identity_threshold` | TCR: 0.90 (cut at 10% distance), BCR: 0.85 (cut at 15%) | Dendrogram cut distance = 1 - threshold |

#### A4.2 Fisher's Exact Test (Cluster Enrichment)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Counting unit | Unique participants per (cluster, disease class) | Prevents high-depth participants from dominating |
| Statistical test | Right-tail hypergeometric (`hypergeom.sf`) | Equivalent to one-sided Fisher's exact test |
| Multiple testing correction | None | P-value threshold tuned via grid search instead |

#### A4.3 Centroid Computation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Method | Weighted majority-vote consensus per amino acid position | Weight = occurrence_count x num_clone_members |

#### A4.4 P-Value Grid Search

| Parameter | Value | Description |
|-----------|-------|-------------|
| `p_values` candidates | [0.0005, 0.001, 0.005, 0.01, 0.05] | Default grid |
| Evaluation set | train_smaller2 | Independent from clustering/training data |
| Selection criterion | MCC-with-abstention | Abstained specimens treated as misclassifications (label "UNKNOWN99") |

#### A4.5 Final GLM Classifier

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model type (TCR) | `lasso_cv` (`alpha=1.0`, pure L1) | Pre-selected from original Mal-ID |
| Model type (BCR) | `ridge_cv` (`alpha=0.0`, pure L2) | Pre-selected from original Mal-ID |
| `n_lambda` | 100 | Regularization path size |
| Internal CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | Participant-grouped |
| CV scoring | Deviance (log-loss) | glmnet default |
| `use_lambda_1se` | `False` | Use lambda minimizing CV deviance |
| `class_weight` | `"balanced"` | Inverse frequency weighting |
| Training data | train_smaller1 (default) | Optionally expanded to train_smaller1 + train_smaller2 with `retrain_on_full_train` |
| Input features | Unique cluster hits per disease class per specimen | Integer counts |

Available GLM variants (alpha values): `lasso_cv` (1.0), `elasticnet_cv0.75` (0.75), `elasticnet_cv` (0.5), `elasticnet_cv0.25` (0.25), `ridge_cv` (0.0).

### A5. Model 3 — Sequence-Level Classifier

#### A5.1 ESM-2 Embedding Extraction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `esm2_t30_150M_UR50D` | 30-layer, 150M-parameter protein language model |
| Embedding dimension | 640 | Per-sequence representation |
| Representation layer | 30 (final layer) | Layer used for extraction |
| Pooling | Mean over amino acid positions | Excluding BOS/EOS special tokens |
| Input | CDR3 amino acid sequence (`cdr3_aa`) | |
| Storage precision | float16 | Half-precision |

#### A5.2 V-Gene Group Formation

| Parameter | Value | Description |
|-----------|-------|-------------|
| TCR grouping | `(v_gene,)` | One group per V-gene |
| BCR grouping | `(v_gene, isotype_supergroup)` | One group per V-gene x isotype |
| Rare V-gene filter | Median of per-V-gene maximum disease frequencies | Same logic as Model 1 V-gene filtering |
| `min_sequences_per_group` | 10 | Skip groups with fewer training sequences |

#### A5.3 Stage 1 — Per-V-Gene Sequence Classifier

**TCR: One-vs-Rest Ridge Regression (Glmnet)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.0 (pure L2 / ridge) | No L1 penalty |
| `n_lambda` | 100 | Regularization path size |
| Internal CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | Participant-grouped |
| `use_lambda_1se` | `False` | Use lambda minimizing CV deviance |
| `class_weight` | `"balanced"` | Per binary sub-problem |
| `standardize` | `False` | Per-group StandardScaler applied externally |
| OvR probability normalization | `False` | K binary outputs are independent, do not sum to 1 |
| OvR allow class failure | `True` | Skip failed binary sub-classifiers |

**BCR: RandomForest (multiclass)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 100 | Number of trees |
| `class_weight` | `"balanced_subsample"` | Per-tree balanced weighting |
| `random_state` | 0 | |

#### A5.4 Aggregation (Sequence -> Specimen)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aggregation_strategy` | `entropy_percentile_cutoff` | How to aggregate per-sequence predictions to specimen level |
| `entropy_bottom_percentile` | 0.01 | For `entropy_percentile_cutoff`: keep sequences in bottom 1% of training entropy distribution |
| `entropy_max_fraction` | 0.80 | For `entropy_cutoff`: threshold = fraction x ln(n_classes) |
| Fallback for empty groups | Uniform prior `[1/K, ..., 1/K]` | When no sequences survive filtering |

Available aggregation strategies: `mean`, `median`, `entropy_cutoff`, `entropy_percentile_cutoff` (default), `entropy_ten_percent_cutoff`, `entropy_twenty_percent_cutoff`, `auto_tuned`, `paper_best` (TCR = `entropy_cutoff` with fraction 0.80, BCR = `mean`).

#### A5.5 V-Gene Frequency Reweighing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Enabled | `True` (default) | Modulate features by V-gene group frequency within each specimen |
| TCR normalization | Row-normalize frequencies to sum to 1 per specimen | |
| BCR normalization | Normalize within each isotype separately | Removes isotype proportion artifacts |

#### A5.6 Stage 2 — Specimen-Level Classifier

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | `BinaryOvRClassifierWithFeatureSubsettingByClass` | Each binary classifier receives only its own class's feature columns |
| Base classifier | `RandomForestClassifier` | Both TCR and BCR |
| `n_estimators` | 100 | Number of trees |
| `class_weight` | `"balanced_subsample"` | Per-tree balanced weighting |
| `random_state` | 0 | |
| Input features | Aggregated, scaled, V-gene-frequency-reweighed class probabilities | One column per (disease class, V-gene group) |

#### A5.7 Feature Pipeline Summary (Stage 1 output -> Stage 2 input)

| Step | Description |
|------|-------------|
| 1. Aggregate | Per-(specimen, V-gene group) aggregation of sequence predictions |
| 2. `StandardScaler` | Pre-aggregation scaling (fitted on train) |
| 3. V-gene frequency reweighing | Element-wise multiplication by per-specimen V-gene group frequencies |
| 4. `StandardScaler` | Post-reweighing scaling (fitted on train) |
| 5. Feature subsetting | Each OvR binary classifier receives only its class's columns |

### A6. Ensemble Meta-Learner

| Parameter | Value | Description |
|-----------|-------|-------------|
| Meta-learner | Ridge logistic regression (Glmnet, `alpha=0.0`) | Pure L2 regularization |
| Pipeline | `StandardScaler` -> `GlmnetLogitNetWrapper` | |
| `n_lambda` | 100 | Regularization path size |
| Internal CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` | Participant-grouped; auto-capped if insufficient groups |
| CV scoring | MCC (Matthews Correlation Coefficient) | Unlike base models which use deviance |
| `use_lambda_1se` | `False` | Use lambda minimizing CV MCC |
| `class_weight` | `"balanced"` | Inverse frequency weighting |
| `standardize` | `False` | Handled by pipeline scaler |
| `random_state` | 0 | |
| Input features | Base model predicted probabilities per specimen | One column per (model, disease class); binary mode keeps only non-reference class |
| `model2_abstention_strategy` | `"ensemble_abstain"` (default) | How to handle specimens where Model 2 abstained |

Model 2 abstention strategies:

| Strategy | Description |
|----------|-------------|
| `ensemble_abstain` | Drop specimens where Model 2 abstained (no ensemble prediction) |
| `fill_0.5` | Fill Model 2 predictions with 0.5 (uninformative prior) |
| `fill_models13_mean` | Fill Model 2 predictions with the mean of Models 1 and 3 predictions |

### A7. Hyperparameter Tuning Summary

| Hyperparameter | Model(s) | How Tuned | Selection Criterion |
|----------------|----------|-----------|---------------------|
| Lambda (regularization strength) | M1, M2, M3-Stage1, Ensemble | Automatically by glmnet internal 5-fold CV | M1/M2/M3: deviance; Ensemble: MCC |
| Alpha (L1/L2 ratio) | M1, M2 | Fixed per locus (pre-selected in original Mal-ID) | N/A |
| PCA components (`n_pcs`) | M1 | Fixed at 15 | N/A |
| V-gene frequency threshold | M1, M3 | Fixed at 50th percentile (median) | N/A |
| P-value threshold | M2 | Grid search over 5 candidates on train_smaller2 | MCC-with-abstention |
| Aggregation strategy | M3 | Fixed (default: `entropy_percentile_cutoff`) | N/A (optionally tunable via `auto_tuned`) |
| Entropy percentile | M3 | Fixed at 0.01 (default) | N/A (optionally tunable via `auto_tuned`) |
| CDR3 clustering identity threshold | M2 | Fixed per locus | N/A |
| Clone ID clustering threshold | Preprocessing | Fixed per locus and CDR3 type | N/A |
