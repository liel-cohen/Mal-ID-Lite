# Model Description

A thorough, step-by-step algorithmic description of each model in Mal-ID-Lite,
from input data to final predictions.

---

## Model 1: Repertoire-Level V-J Gene Frequency Classifier

<!-- TODO: Full algorithmic description -->
<!-- Topics to cover:
- Input: per-specimen sequences (DOWNSAMPLED stage)
- Feature extraction: V-J gene pair frequencies per isotype
- Pipeline: log1p -> StandardScaler -> PCA(15) per isotype -> StandardScaler -> glmnet elastic net
- Internal CV: StratifiedGroupKFold (participant-level) for lambda tuning
- Classification modes: multiclass, binary, multi-binary
- Test-time feature alignment: reindex to training V-J columns, fill missing with 0
- Output: class probabilities per specimen
-->

### Handling of specimens with insufficient data

Model 1 always produces a prediction for every specimen. There is no abstention
mechanism.

If a specimen has no sequences belonging to any non-rare V-gene (after V-gene
filtering to the training set), its V-J frequency matrix is all zeros. After
normalization (frequencies sum to 1), an all-zero row produces NaN, which is
filled back to 0. The all-zero feature vector flows through log1p(0)=0,
StandardScaler, and PCA, resulting in a feature vector that carries no
specimen-specific information. The classifier still produces a probability
vector, but it is uninformative (driven by the model's bias/intercept rather
than any data from this specimen).

A warning is logged with the count and percentage of affected specimens. This
situation is extremely unlikely in practice (would require a specimen with zero
sequences from any V-gene seen during training) and indicates a data quality
issue if it occurs.

---

## Model 2: Convergent Cluster Classifier

<!-- TODO: Full algorithmic description -->
<!-- Topics to cover:
- Phase 1: Clustering (CDR3 sequences -> centroids via greedy clustering)
- Phase 2: Fisher's exact test per cluster per disease (statistical significance)
- Phase 3: Featurization (count hits to significant clusters per specimen per disease)
- Phase 4: Classifier training (glmnet on hit-count features)
- P-value threshold selection via internal CV (train_smaller1 -> train_smaller2 validation)
- Abstention mechanism: specimens with zero hits to any significant cluster
- Classification modes: multiclass, binary, multi-binary
- Output: class probabilities for scored specimens; abstention flag for others
-->

### Handling of specimens with insufficient data

Model 2 has an explicit **abstention mechanism**. A specimen abstains (receives
no prediction) if none of its CDR3 sequences match any disease-significant
cluster. This can happen when:

1. No statistically significant clusters exist at all (e.g., the Fisher's exact
   test finds no cluster enriched in any disease at the chosen p-value
   threshold). In this case, ALL specimens abstain.
2. The specimen's CDR3 sequences do not match any of the disease-enriched
   clusters. The specimen's row in the cluster-hit count matrix is all zeros,
   and it is removed from the scored set.

Abstention affects evaluation metrics differently:

- **Accuracy and MCC**: abstentions count as misclassifications. Accuracy =
  n_correct_scored / (n_scored + n_abstained). MCC appends a synthetic
  "Unknown" label for each abstained specimen.
- **AUROC and AUPRC**: computed on scored specimens only. Abstained specimens
  are completely excluded because no probability vector exists for them. This
  means these metrics do not reflect the model's inability to predict for
  abstained specimens. When abstention rate is > 0, a note is included in the
  results report stating how many specimens were excluded.

Abstention rates typically range from ~5% to ~30% depending on the p-value
threshold and the disease. This is by design: Model 2 trades coverage for
precision, only making predictions when it has cluster-based evidence.

---

## Model 3: Sequence-Level Classifier

<!-- TODO: Full algorithmic description -->
<!-- Topics to cover:
- Two-stage architecture: Stage 1 (per-V-gene-group classifiers) + Stage 2 (specimen-level rollup)
- Data split: train_smaller1 (2/3) for Stage 1, train_smaller2 (1/3) for Stage 2
- Stage 1: ESM-2 embeddings -> StandardScaler -> per-group classifier (TCR: ridge OvR via glmnet, BCR: RF)
  - Rare V-gene filtering
  - Participant-level CV grouping in glmnet
  - CustomOneVsRestClassifier with failure tolerance
- Sequence prediction: per-sequence probability vectors from group models
- Aggregation: sequence predictions -> specimen-level features
  - Aggregation strategies: mean, median, entropy thresholding
  - Feature column naming: "{class}_{v_gene}" for TCR
  - Missing groups filled with 1/n_classes uniform prior
- Optional frequency reweighing: scale features by V-gene group prevalence
- Stage 2: StandardScaler -> BinaryOvRClassifierWithFeatureSubsettingByClass (RF)
  - Each binary classifier uses only its own class's feature columns
  - Binary special case: 1 classifier, [1-p, p] reconstruction
- Classification modes: multiclass, binary, multi-binary (via reference_class)
- Output: class probabilities per specimen (do NOT sum to 1 in multiclass)
-->

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

<!-- TODO: Full algorithmic description -->
<!-- Topics to cover:
- Architecture: train base models on train_split, collect predictions on validation_split, train meta-learner
- Data splits: train_smaller (2/3 of train) -> train_split (2/3) + validation_split (1/3)
- Base model predictions on validation_split as features for meta-learner
- Meta-learner training (classifier type TBD)
- Handling of Model 2 abstentions in the ensemble feature vector
- Artifact separation: ensemble-trained base models vs standalone base models
- Reference: TODO_for_release.md "Ensemble training architecture" section
-->

### Handling of specimens with insufficient data

<!-- TODO: Document how the ensemble handles:
- Model 2 abstentions (no probability vector for some specimens)
- Model 1 / Model 3 uninformative predictions (all-zero / uniform features)
- Whether the ensemble should abstain if Model 2 abstains, or use the other
  models' predictions as fallback
-->

Not yet implemented. See `TODO_for_release.md` for the planned architecture.
