"""Quick smoke test for Model 2 (Convergent Cluster Classifier).

Tests end-to-end on fold 0 using a participant subset (~60 participants)
for speed. Verifies each phase independently before running the full
training pipeline.

Tests
-----
1. Data loading and disease join (load_and_prepare_fold)
   - Fold 0 train data loaded; disease column present and non-null
   - Required columns exist (repertoire_id, cdr3_seq_aa_q_trim, etc.)

2. Train_smaller1 / train_smaller2 split (split_train_smaller)
   - Participant-level stratified split with no overlap
   - Split ratio approximately 2/3 : 1/3

3. Clustering (cluster_training_set)
   - Clusters assigned to every sequence; global_resulting_cluster_ID column present
   - No NaN cluster IDs

4. Centroid computation (get_cluster_centroids)
   - One centroid per cluster; centroid_sequence non-empty

5. Fisher's exact test (compute_fisher_scores)
   - P-values in [0, 1] for all disease classes
   - Reports significant cluster counts at p=0.01 and p=0.05

6. Featurization (featurize on train_smaller2)
   - FeaturizedData returned with correct columns (one per disease class)
   - n_scored + n_abstained equals total specimens

7. Full pipeline (train_convergent_cluster_classifier)
   - Multiclass on subset: best_p_value selected, pipeline fitted
   - All expected keys present in train_result

8. Inference (save_fold_artifacts + ConvergentClusterClassifier.load_artifacts)
   - Artifacts saved and loaded; predict + predict_proba produce valid output
   - Probabilities sum to 1.0

9. validate_mode_and_classes (unit test, 11 cases)
   - Multiclass with 2/N classes, with/without reference_class
   - Binary with 2/>2 classes, valid/unknown reference_class
   - Multi-binary with/without reference_class
   - Unknown mode raises ValueError

10. filter_to_binary_pair
    - Only target + reference diseases remain; no other classes present
    - Participant sets consistent between sequences and metadata

11. Full binary pipeline
    - 2-class filtered subset: clustering, Fisher, featurization, training, inference
    - Binary pair saved in artifacts and loaded into classifier attributes
    - predict_proba column order: col0=P(reference), col1=P(disease)
    - predict returns only valid class labels

12. cv_ensemble split verification and pipeline
    - cv_ensemble training participants are a strict subset of cv_single_model
    - No overlap between validation and training; partition completeness
    - No overlap between ts1 and ts2
    - Full pipeline on cv_ensemble subset produces valid results

Design notes
------------
- Uses ~60 participants (subset) to keep clustering O(n^2) manageable.
- Shared functions (filter_to_binary_pair, validate_mode_and_classes,
  split_train_smaller) imported from malid_lite.training.training_utils.
- Model-specific functions (load_and_prepare_fold, save_fold_artifacts) imported
  from malid_lite.training.train_model2.

Requirements
------------
- Fold cache built: cache/mal-id-orig-data/data_folds/fold_*.parquet
- All dependencies from requirements.txt (scipy, scikit-learn, joblib, etc.)

Expected runtime
----------------
- With cache: ~5-20 minutes (clustering is the bottleneck)
- Without cache: ~25-40 minutes

Output files
------------
All outputs saved to tests/test_outputs/test_model2_quick/:
- test_log_YYYYMMDD_HHMMSS.txt              - Full log
- test_log_YYYYMMDD_HHMMSS.json             - Structured results
- test_artifacts/                            - Multiclass pipeline artifacts
- test_binary_artifacts/                     - Binary pipeline artifacts
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite.models.model2_convergent_clusters import (
    CDR3_COL,
    CLUSTER_ID_COL,
    DEFAULT_P_VALUES,
    DISEASE_COL,
    PARTICIPANT_COL,
    SPECIMEN_COL,
    ConvergentClusterClassifier,
    FeaturizedData,
    cluster_training_set,
    compute_fisher_scores,
    featurize,
    get_cluster_centroids,
    merge_centroids_with_scores,
    train_convergent_cluster_classifier,
)
from malid_lite.training.training_utils import (
    DEFAULT_DATASET_NAME,
    filter_to_binary_pair,
    split_train_smaller,
    validate_mode_and_classes,
)
from malid_lite.training.train_model2 import (
    load_and_prepare_fold,
    save_fold_artifacts,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data_clean/airr_format_clean/TCR")
METADATA_PATH = Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv")
GENE_REFERENCE_PATH = Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/tcrb_v_gene_cdrs.generated.tsv")
CACHE_DIR = PROJECT_ROOT / "cache" / "mal-id-orig-data"


class TestLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, "a")
        self.results = {"tests": [], "start_time": datetime.now().isoformat()}

    def log(self, message, to_file_only=False):
        self.file.write(message + "\n")
        self.file.flush()
        if not to_file_only:
            print(message)

    def add_result(self, test_name, status, details=None):
        self.results["tests"].append({
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def close(self):
        self.results["end_time"] = datetime.now().isoformat()
        self.file.close()
        results_file = self.log_file.with_suffix(".json")
        with open(results_file, "w") as f:
            json.dump(
                self.results, f, indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
            )
        return results_file


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).stem

    output_dir = Path(__file__).parent / "test_outputs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tlog = TestLogger(output_dir / f"test_log_{timestamp}.txt")
    tlog.log(f"Model 2 quick smoke test — {timestamp}")
    tlog.log("=" * 60)

    # Use a subset of participants to keep runtime short
    SUBSET_PARTICIPANTS = 60   # roughly 1/9 of dataset — enough for meaningful clustering
    FOLD_ID = 0
    MODEL_NAME = "lasso_cv"

    # -----------------------------------------------------------------------
    # Setup: initialize loader
    # -----------------------------------------------------------------------
    loader = MalIDPublishedDataLoader(
        data_dir=DATA_DIR,
        metadata_path=METADATA_PATH,
        gene_locus="TCR",
        cache_dir=CACHE_DIR,
        gene_reference_path=GENE_REFERENCE_PATH,
        verbose=0,
    )

    # -----------------------------------------------------------------------
    # Test 1: Load and prepare fold data
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 1] Data loading and disease join")
    try:
        train_sequences_df, train_metadata_df = load_and_prepare_fold(
            loader, FOLD_ID, "train"
        )
        assert DISEASE_COL in train_sequences_df.columns, "disease column missing after join"
        assert train_sequences_df[DISEASE_COL].notna().all(), "NaN disease values"
        assert "repertoire_id" in train_sequences_df.columns
        assert CDR3_COL in train_sequences_df.columns

        n_participants = train_sequences_df[PARTICIPANT_COL].nunique()
        n_sequences = len(train_sequences_df)
        diseases = sorted(train_sequences_df[DISEASE_COL].unique().tolist())

        tlog.log(f"  {n_sequences:,} sequences, {n_participants} participants")
        tlog.log(f"  Diseases: {diseases}")
        tlog.add_result("data_loading", "PASS", {
            "n_sequences": n_sequences,
            "n_participants": n_participants,
            "diseases": diseases,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("data_loading", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Subset to SUBSET_PARTICIPANTS for speed
    # -----------------------------------------------------------------------
    tlog.log(f"\nSubsetting to {SUBSET_PARTICIPANTS} participants for speed")
    subset_participants = (
        train_metadata_df
        .drop_duplicates(subset=[PARTICIPANT_COL])
        .groupby(DISEASE_COL)[PARTICIPANT_COL]
        .apply(lambda x: x.head(SUBSET_PARTICIPANTS // 6))
        .reset_index(drop=True)
    )
    subset_set = set(subset_participants)

    train_sequences_sub = train_sequences_df[
        train_sequences_df[PARTICIPANT_COL].isin(subset_set)
    ].copy()
    train_metadata_sub = train_metadata_df[
        train_metadata_df[PARTICIPANT_COL].isin(subset_set)
    ].copy()

    tlog.log(
        f"  Subset: {len(train_sequences_sub):,} sequences, "
        f"{train_sequences_sub[PARTICIPANT_COL].nunique()} participants"
    )

    # -----------------------------------------------------------------------
    # Test 2: Train_smaller split
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 2] Train_smaller1 / train_smaller2 split")
    try:
        ts1, ts2 = split_train_smaller(train_sequences_sub, train_metadata_sub)

        p1 = set(ts1[PARTICIPANT_COL].unique())
        p2 = set(ts2[PARTICIPANT_COL].unique())
        overlap = p1 & p2

        assert len(overlap) == 0, f"Participants overlap between splits: {overlap}"
        assert len(p1) + len(p2) == len(subset_participants), "Split doesn't cover all participants"
        assert abs(len(p2) / (len(p1) + len(p2)) - 1 / 3) < 0.05, "Split ratio far from 1/3"

        tlog.log(f"  train_smaller1: {len(ts1):,} seqs, {len(p1)} participants")
        tlog.log(f"  train_smaller2: {len(ts2):,} seqs, {len(p2)} participants")
        tlog.log(f"  No overlap: confirmed")
        tlog.add_result("train_smaller_split", "PASS", {
            "n_ts1_seqs": len(ts1), "n_ts1_participants": len(p1),
            "n_ts2_seqs": len(ts2), "n_ts2_participants": len(p2),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("train_smaller_split", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 3: Clustering
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 3] Clustering (cluster_training_set) on train_smaller1 subset")
    try:
        threshold = 0.90  # TCR threshold
        clustered_df = cluster_training_set(ts1, sequence_identity_threshold=threshold, n_jobs=1)

        assert CLUSTER_ID_COL in clustered_df.columns
        assert "global_resulting_cluster_ID" in clustered_df.columns
        assert not clustered_df[CLUSTER_ID_COL].isna().any()

        n_clusters = clustered_df["global_resulting_cluster_ID"].nunique()
        n_supergroups = clustered_df.groupby(
            ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len"]
        ).ngroups

        tlog.log(f"  {n_supergroups:,} V-J-len supergroups")
        tlog.log(f"  {n_clusters:,} total clusters")
        tlog.log(f"  Avg clusters per supergroup: {n_clusters / n_supergroups:.1f}")
        tlog.add_result("clustering", "PASS", {
            "n_supergroups": n_supergroups,
            "n_clusters": n_clusters,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("clustering", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 4: Centroids
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 4] Centroid computation (get_cluster_centroids)")
    try:
        centroids = get_cluster_centroids(clustered_df)

        assert "centroid_sequence" in centroids.columns
        assert len(centroids) == n_clusters
        assert centroids["centroid_sequence"].str.len().ge(1).all()

        sample_centroid = centroids.iloc[0]
        tlog.log(f"  {len(centroids):,} centroids computed")
        tlog.log(f"  Sample centroid: {sample_centroid['centroid_sequence']!r} "
                 f"(V={sample_centroid['v_gene']}, J={sample_centroid['j_gene']})")
        tlog.add_result("centroids", "PASS", {"n_centroids": len(centroids)})
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("centroids", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 5: Fisher scores
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 5] Fisher's exact test (compute_fisher_scores)")
    try:
        pvalue_df = compute_fisher_scores(clustered_df, disease_col=DISEASE_COL)

        disease_classes = sorted(pvalue_df.columns.tolist())
        assert len(disease_classes) > 0
        assert (pvalue_df.values >= 0).all() and (pvalue_df.values <= 1.0).all()

        # Check distribution of significant clusters
        for p_thresh in [0.01, 0.05]:
            n_sig = int((pvalue_df.min(axis=1) <= p_thresh).sum())
            tlog.log(f"  Clusters with min(p) <= {p_thresh}: {n_sig:,} "
                     f"({100 * n_sig / len(pvalue_df):.1f}%)")

        tlog.add_result("fisher_scores", "PASS", {
            "disease_classes": disease_classes,
            "n_clusters_tested": len(pvalue_df),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("fisher_scores", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 6: Featurization at a single p-value
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 6] Featurization (featurize) on train_smaller2")
    try:
        centroids_with_scores = merge_centroids_with_scores(centroids, pvalue_df)
        p_val = 0.01

        fd = featurize(
            ts2,
            p_value_threshold=p_val,
            centroids_with_scores=centroids_with_scores,
            sequence_identity_threshold=threshold,
            disease_classes=disease_classes,
        )

        assert isinstance(fd, FeaturizedData)
        assert list(fd.X.columns) == disease_classes
        assert fd.n_scored + fd.n_abstained == ts2["repertoire_id"].nunique()

        tlog.log(f"  p_value={p_val}")
        tlog.log(f"  Scored: {fd.n_scored} specimens, Abstained: {fd.n_abstained}")
        tlog.log(f"  Abstention rate: {fd.abstention_rate:.1%}")
        if fd.n_scored > 0:
            tlog.log(f"  Feature matrix shape: {fd.X.shape}")
            tlog.log(f"  Feature stats (mean per disease):")
            for col in fd.X.columns:
                tlog.log(f"    {col}: {fd.X[col].mean():.2f} clusters/specimen")
        tlog.add_result("featurization", "PASS", {
            "p_value": p_val,
            "n_scored": fd.n_scored,
            "n_abstained": fd.n_abstained,
            "abstention_rate": float(fd.abstention_rate),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("featurization", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 7: Full training pipeline (subset, one model)
    # -----------------------------------------------------------------------
    tlog.log(f"\n[Test 7] Full pipeline: train_convergent_cluster_classifier (fold 0, {MODEL_NAME})")
    try:
        train_result = train_convergent_cluster_classifier(
            train_smaller1_df=ts1,
            train_smaller2_df=ts2,
            sequence_identity_threshold=threshold,
            model_names=[MODEL_NAME],
            p_values=[0.001, 0.01, 0.05],  # Fewer p-values for speed
            disease_col=DISEASE_COL,
            n_jobs=1,  # Small subset — serial is faster (no thread-spawn overhead)
            verbose=1,
        )

        assert "centroids_with_scores" in train_result
        assert "disease_classes" in train_result
        assert "results" in train_result
        assert MODEL_NAME in train_result["results"]

        model_result = train_result["results"][MODEL_NAME]
        assert model_result["best_p_value"] is not None
        assert model_result["pipeline"] is not None

        best_p = model_result["best_p_value"]
        tlog.log(f"  Best p-value: {best_p}")
        tlog.log(f"  Classifier classes: {model_result['pipeline'].classes_}")
        tlog.add_result("full_pipeline", "PASS", {
            "best_p_value": best_p,
            "n_classes": len(model_result["pipeline"].classes_),
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("full_pipeline", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 8: Save artifacts and inference (load + featurize + predict)
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 8] Save artifacts and inference (featurize + predict on test)")
    try:
        artifact_dir = output_dir / "test_artifacts"
        save_fold_artifacts(artifact_dir, fold_id=99, train_result=train_result)

        # Verify files exist
        assert (artifact_dir / "fold_99_clusters.joblib").exists()
        assert (artifact_dir / f"fold_99_{MODEL_NAME}_p_value.joblib").exists()
        assert (artifact_dir / f"fold_99_{MODEL_NAME}_model_split1.joblib").exists()  # default: retrain_on_full_train=False

        # Load a small test subset (use ts2 as proxy for "test" in this unit test)
        clf = ConvergentClusterClassifier(gene_locus="TCR", model_name=MODEL_NAME)
        clf.load_artifacts(artifact_dir, fold_id=99, model_name=MODEL_NAME)
        assert clf._is_loaded

        fd_test = clf.featurize(ts2, disease_col=DISEASE_COL)
        tlog.log(f"  Test featurization: {fd_test.n_scored} scored, {fd_test.n_abstained} abstained")

        if fd_test.n_scored > 0:
            y_pred = clf.predict(fd_test.X)
            y_proba = clf.predict_proba(fd_test.X)
            assert len(y_pred) == fd_test.n_scored
            assert y_proba.shape == (fd_test.n_scored, len(clf.classes_))
            assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)
            tlog.log(f"  Predicted classes: {np.unique(y_pred).tolist()}")
            tlog.log(f"  predict_proba shape: {y_proba.shape} (sums to 1.0: confirmed)")
        else:
            tlog.log("  All specimens abstained (too few significant clusters at this p-value)")

        tlog.add_result("inference", "PASS", {
            "n_scored": fd_test.n_scored,
            "n_abstained": fd_test.n_abstained,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("inference", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 9: validate_mode_and_classes (unit test — no data loading)
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 9] validate_mode_and_classes: mode/data compatibility validation")
    try:
        errors = []

        # multiclass with 2 classes — should warn but not raise
        try:
            validate_mode_and_classes("multiclass", ["A", "B"], None)
        except Exception as e:
            errors.append(f"multiclass/2-class raised unexpectedly: {e}")

        # multiclass with N classes — should not raise
        try:
            validate_mode_and_classes("multiclass", ["A", "B", "C"], None)
        except Exception as e:
            errors.append(f"multiclass/N-class raised unexpectedly: {e}")

        # multiclass with unknown reference_class — should warn but not raise
        try:
            validate_mode_and_classes("multiclass", ["A", "B"], "X")
        except Exception as e:
            errors.append(f"multiclass/ref-class raised unexpectedly: {e}")

        # binary with exactly 2 classes — should succeed
        try:
            validate_mode_and_classes("binary", ["A", "B"], None)
        except Exception as e:
            errors.append(f"binary/2-class raised unexpectedly: {e}")

        # binary with >2 classes — must raise ValueError
        raised = False
        try:
            validate_mode_and_classes("binary", ["A", "B", "C"], None)
        except ValueError:
            raised = True
        if not raised:
            errors.append("binary/>2-class did not raise ValueError")

        # binary with valid reference_class — should succeed
        try:
            validate_mode_and_classes("binary", ["A", "B"], "B")
        except Exception as e:
            errors.append(f"binary/valid-ref raised unexpectedly: {e}")

        # binary with unknown reference_class — must raise ValueError
        raised = False
        try:
            validate_mode_and_classes("binary", ["A", "B"], "Z")
        except ValueError:
            raised = True
        if not raised:
            errors.append("binary/unknown-ref did not raise ValueError")

        # multi-binary with 2 classes, no reference_class — should infer ref
        ref = validate_mode_and_classes("multi-binary", ["A", "B"], None)
        if ref is None:
            errors.append("multi-binary/2-class/no-ref: expected inferred reference, got None")

        # multi-binary with N classes, no reference_class — must raise ValueError
        raised = False
        try:
            validate_mode_and_classes("multi-binary", ["A", "B", "C"], None)
        except ValueError:
            raised = True
        if not raised:
            errors.append("multi-binary/N-class/no-ref did not raise ValueError")

        # multi-binary with N classes, valid reference_class — should succeed
        try:
            ref = validate_mode_and_classes("multi-binary", ["A", "B", "C"], "C")
            assert ref == "C", f"expected ref='C', got {ref!r}"
        except Exception as e:
            errors.append(f"multi-binary/valid-ref raised unexpectedly: {e}")

        # multi-binary with N classes, unknown reference_class — must raise ValueError
        raised = False
        try:
            validate_mode_and_classes("multi-binary", ["A", "B", "C"], "Z")
        except ValueError:
            raised = True
        if not raised:
            errors.append("multi-binary/unknown-ref did not raise ValueError")

        # unknown mode — must raise ValueError
        raised = False
        try:
            validate_mode_and_classes("invalid-mode", ["A", "B"], None)
        except ValueError:
            raised = True
        if not raised:
            errors.append("invalid mode did not raise ValueError")

        if errors:
            raise AssertionError("Validation failures:\n  " + "\n  ".join(errors))

        tlog.log("  All mode/data validation cases passed")
        tlog.add_result("validate_mode_and_classes", "PASS", {"n_cases_checked": 11})
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("validate_mode_and_classes", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 10: filter_to_binary_pair
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 10] filter_to_binary_pair: filter to 2-class pool")
    try:
        all_diseases = sorted(train_sequences_sub[DISEASE_COL].unique().tolist())
        if len(all_diseases) < 2:
            raise RuntimeError(f"Need >=2 diseases for binary filter test, got: {all_diseases}")

        target_disease = all_diseases[0]
        reference_disease = all_diseases[1]

        filtered_seqs, filtered_meta = filter_to_binary_pair(
            train_sequences_sub, train_metadata_sub, target_disease, reference_disease
        )

        # Only the 2 target diseases should remain
        remaining_diseases = sorted(filtered_seqs[DISEASE_COL].unique().tolist())
        assert remaining_diseases == sorted([target_disease, reference_disease]), (
            f"Expected {sorted([target_disease, reference_disease])}, got {remaining_diseases}"
        )

        # Participants in filtered data should only belong to those 2 diseases
        remaining_participants = set(filtered_seqs[PARTICIPANT_COL].unique())
        meta_participants = set(filtered_meta[PARTICIPANT_COL].unique())
        assert remaining_participants == meta_participants, (
            "Participant sets in sequences and metadata differ after filter"
        )

        # No rows from other diseases
        assert not filtered_seqs[DISEASE_COL].isin(
            [d for d in all_diseases if d not in {target_disease, reference_disease}]
        ).any(), "Other diseases still present in filtered sequences"

        n_all = train_sequences_sub[PARTICIPANT_COL].nunique()
        n_filtered = filtered_seqs[PARTICIPANT_COL].nunique()

        tlog.log(f"  Filtered to {target_disease!r} vs {reference_disease!r}")
        tlog.log(f"  Participants: {n_all} → {n_filtered}")
        tlog.log(f"  Sequences: {len(train_sequences_sub):,} → {len(filtered_seqs):,}")
        tlog.log(f"  Diseases remaining: {remaining_diseases}")
        tlog.add_result("filter_to_binary_pair", "PASS", {
            "target_disease": target_disease,
            "reference_disease": reference_disease,
            "n_participants_before": n_all,
            "n_participants_after": n_filtered,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("filter_to_binary_pair", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 11: Full binary pipeline on 2-class filtered subset
    # -----------------------------------------------------------------------
    tlog.log(f"\n[Test 11] Full binary pipeline: {target_disease!r} vs {reference_disease!r}")
    try:
        bin_ts1, bin_ts2 = split_train_smaller(filtered_seqs, filtered_meta)

        binary_result = train_convergent_cluster_classifier(
            train_smaller1_df=bin_ts1,
            train_smaller2_df=bin_ts2,
            sequence_identity_threshold=threshold,
            model_names=[MODEL_NAME],
            p_values=[0.001, 0.01, 0.05],
            disease_col=DISEASE_COL,
            n_jobs=1,
            verbose=1,
        )

        assert "centroids_with_scores" in binary_result
        assert "disease_classes" in binary_result
        assert "results" in binary_result
        assert MODEL_NAME in binary_result["results"]

        binary_disease_classes = binary_result["disease_classes"]
        assert set(binary_disease_classes) == {target_disease, reference_disease}, (
            f"Expected classes {sorted([target_disease, reference_disease])}, "
            f"got {binary_disease_classes}"
        )
        assert len(binary_disease_classes) == 2, (
            f"Binary model should have exactly 2 classes, got {len(binary_disease_classes)}"
        )

        bin_model_result = binary_result["results"][MODEL_NAME]
        assert bin_model_result["best_p_value"] is not None
        assert bin_model_result["pipeline"] is not None

        best_p = bin_model_result["best_p_value"]
        tlog.log(f"  Binary classes: {binary_disease_classes}")
        tlog.log(f"  Best p-value: {best_p}")
        tlog.log(f"  Classifier classes: {bin_model_result['pipeline'].classes_}")

        # Save binary artifacts — pass disease_filter so binary_pair is stored in the artifact
        binary_artifact_dir = output_dir / "test_binary_artifacts"
        save_fold_artifacts(
            binary_artifact_dir,
            fold_id=99,
            train_result=binary_result,
            disease_filter=(target_disease, reference_disease),
        )

        assert (binary_artifact_dir / "fold_99_clusters.joblib").exists()
        assert (binary_artifact_dir / f"fold_99_{MODEL_NAME}_p_value.joblib").exists()
        assert (binary_artifact_dir / f"fold_99_{MODEL_NAME}_model_split1.joblib").exists()

        # Verify binary_pair is saved in the clusters artifact
        import joblib as _joblib
        _clusters_artifact = _joblib.load(binary_artifact_dir / "fold_99_clusters.joblib")
        assert _clusters_artifact.get("binary_pair") == {
            "disease": target_disease,
            "reference_class": reference_disease,
        }, f"binary_pair not saved correctly: {_clusters_artifact.get('binary_pair')}"

        # Inference on binary test set (use bin_ts2 as proxy)
        clf_binary = ConvergentClusterClassifier(gene_locus="TCR", model_name=MODEL_NAME)
        clf_binary.load_artifacts(binary_artifact_dir, fold_id=99, model_name=MODEL_NAME)
        assert clf_binary._is_loaded

        # Verify binary_pair loaded into classifier attributes
        assert clf_binary.disease_class_ == target_disease, (
            f"disease_class_ expected {target_disease!r}, got {clf_binary.disease_class_!r}"
        )
        assert clf_binary.reference_class_ == reference_disease, (
            f"reference_class_ expected {reference_disease!r}, got {clf_binary.reference_class_!r}"
        )

        # Verify classes_ returns [reference, disease] order (negative first, positive second)
        assert list(clf_binary.classes_) == [reference_disease, target_disease], (
            f"classes_ should be [reference, disease] = [{reference_disease!r}, {target_disease!r}], "
            f"got {list(clf_binary.classes_)}"
        )

        fd_binary = clf_binary.featurize(bin_ts2, disease_col=DISEASE_COL)
        tlog.log(f"  Binary inference: {fd_binary.n_scored} scored, {fd_binary.n_abstained} abstained")

        if fd_binary.n_scored > 0:
            y_pred_bin = clf_binary.predict(fd_binary.X)
            y_proba_bin = clf_binary.predict_proba(fd_binary.X)
            assert len(y_pred_bin) == fd_binary.n_scored
            assert y_proba_bin.shape == (fd_binary.n_scored, 2)
            assert np.allclose(y_proba_bin.sum(axis=1), 1.0, atol=1e-6)

            # Verify predict_proba column order: col0=P(reference), col1=P(disease)
            # Both columns must be in [0, 1] and sum to 1 (already checked above).
            # Additionally, col1 should be higher for specimens predicted as the disease class.
            disease_predicted_mask = (y_pred_bin == target_disease)
            if disease_predicted_mask.any():
                assert (y_proba_bin[disease_predicted_mask, 1] > 0.5).all(), (
                    "For disease-predicted specimens, P(disease) = col1 should be > 0.5"
                )
            reference_predicted_mask = (y_pred_bin == reference_disease)
            if reference_predicted_mask.any():
                assert (y_proba_bin[reference_predicted_mask, 0] > 0.5).all(), (
                    "For reference-predicted specimens, P(reference) = col0 should be > 0.5"
                )

            # Verify predict returns only valid class labels
            unique_predicted = set(np.unique(y_pred_bin).tolist())
            assert unique_predicted <= {target_disease, reference_disease}, (
                f"predict returned unexpected labels: {unique_predicted}"
            )

            tlog.log(f"  Predicted classes: {sorted(unique_predicted)}")
            tlog.log(f"  predict_proba shape: {y_proba_bin.shape} (sums to 1.0: confirmed)")
            tlog.log(f"  classes_ order: {list(clf_binary.classes_)} (reference first, disease second)")

        tlog.add_result("binary_pipeline", "PASS", {
            "target_disease": target_disease,
            "reference_disease": reference_disease,
            "best_p_value": best_p,
            "binary_disease_classes": binary_disease_classes,
            "n_scored": fd_binary.n_scored,
            "n_abstained": fd_binary.n_abstained,
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("binary_pipeline", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Test 12: cv_ensemble split verification and pipeline
    # -----------------------------------------------------------------------
    tlog.log("\n[Test 12] cv_ensemble: split verification and pipeline")
    try:
        # Get split participants for both contexts
        sm_ts1 = set(loader.get_split_participants(FOLD_ID, "cv_single_model", ["train_smaller1"]))
        sm_ts2 = set(loader.get_split_participants(FOLD_ID, "cv_single_model", ["train_smaller2"]))
        ens_ts1 = set(loader.get_split_participants(FOLD_ID, "cv_ensemble", ["train_smaller1"]))
        ens_ts2 = set(loader.get_split_participants(FOLD_ID, "cv_ensemble", ["train_smaller2"]))
        ens_val = set(loader.get_split_participants(FOLD_ID, "cv_ensemble", ["validation"]))

        # cv_ensemble training is strictly smaller
        sm_train = sm_ts1 | sm_ts2
        ens_train = ens_ts1 | ens_ts2
        assert len(ens_train) < len(sm_train), (
            f"cv_ensemble train ({len(ens_train)}) should be < "
            f"cv_single_model train ({len(sm_train)})"
        )

        # No overlap between validation and training
        assert not (ens_train & ens_val), (
            f"{len(ens_train & ens_val)} participants in both train and validation"
        )

        # Partition completeness: train + validation = full cv_single_model train
        assert ens_train | ens_val == sm_train, (
            f"cv_ensemble train+val does not equal cv_single_model train: "
            f"{len((ens_train | ens_val) - sm_train)} extra, "
            f"{len(sm_train - (ens_train | ens_val))} missing"
        )

        # No overlap between ts1 and ts2
        assert not (ens_ts1 & ens_ts2), (
            f"{len(ens_ts1 & ens_ts2)} participants in both ts1 and ts2"
        )

        tlog.log(f"  cv_single_model train: {len(sm_train)} participants")
        tlog.log(f"  cv_ensemble train: {len(ens_train)} participants "
                 f"(ts1={len(ens_ts1)}, ts2={len(ens_ts2)})")
        tlog.log(f"  cv_ensemble validation: {len(ens_val)} participants")

        # Filter full fold data to cv_ensemble ts1/ts2, then subset for speed
        ens_ts1_df = train_sequences_df[
            train_sequences_df[PARTICIPANT_COL].isin(ens_ts1)
        ].copy()
        ens_ts2_df = train_sequences_df[
            train_sequences_df[PARTICIPANT_COL].isin(ens_ts2)
        ].copy()

        # Verify validation data is NOT in ts1 or ts2
        ts1_participants_actual = set(ens_ts1_df[PARTICIPANT_COL].unique())
        ts2_participants_actual = set(ens_ts2_df[PARTICIPANT_COL].unique())
        assert not (ts1_participants_actual & ens_val), "Validation participants leaked into ts1"
        assert not (ts2_participants_actual & ens_val), "Validation participants leaked into ts2"

        # Stratified subset for speed (same approach as main test)
        ens_ts1_meta = train_metadata_df[
            train_metadata_df[PARTICIPANT_COL].isin(ens_ts1)
        ]
        ens_ts1_subset = (
            ens_ts1_meta
            .drop_duplicates(subset=[PARTICIPANT_COL])
            .groupby(DISEASE_COL)[PARTICIPANT_COL]
            .apply(lambda x: x.head(SUBSET_PARTICIPANTS // 6))
            .reset_index(drop=True)
        )
        ens_ts2_meta = train_metadata_df[
            train_metadata_df[PARTICIPANT_COL].isin(ens_ts2)
        ]
        ens_ts2_subset = (
            ens_ts2_meta
            .drop_duplicates(subset=[PARTICIPANT_COL])
            .groupby(DISEASE_COL)[PARTICIPANT_COL]
            .apply(lambda x: x.head(SUBSET_PARTICIPANTS // 12))
            .reset_index(drop=True)
        )

        ens_ts1_sub = ens_ts1_df[
            ens_ts1_df[PARTICIPANT_COL].isin(set(ens_ts1_subset))
        ].copy()
        ens_ts2_sub = ens_ts2_df[
            ens_ts2_df[PARTICIPANT_COL].isin(set(ens_ts2_subset))
        ].copy()

        tlog.log(f"  Subsetted ts1: {ens_ts1_sub[PARTICIPANT_COL].nunique()} participants, "
                 f"{len(ens_ts1_sub):,} seqs")
        tlog.log(f"  Subsetted ts2: {ens_ts2_sub[PARTICIPANT_COL].nunique()} participants, "
                 f"{len(ens_ts2_sub):,} seqs")

        # Run pipeline on cv_ensemble subset
        ens_result = train_convergent_cluster_classifier(
            train_smaller1_df=ens_ts1_sub,
            train_smaller2_df=ens_ts2_sub,
            sequence_identity_threshold=threshold,
            model_names=[MODEL_NAME],
            p_values=[0.001, 0.01, 0.05],
            disease_col=DISEASE_COL,
            n_jobs=1,
            verbose=1,
        )

        assert "centroids_with_scores" in ens_result
        assert "disease_classes" in ens_result
        assert MODEL_NAME in ens_result["results"]

        ens_model_result = ens_result["results"][MODEL_NAME]
        assert ens_model_result["best_p_value"] is not None
        assert ens_model_result["pipeline"] is not None

        tlog.log(f"  cv_ensemble pipeline: best_p={ens_model_result['best_p_value']}, "
                 f"classes={ens_result['disease_classes']}")
        tlog.add_result("cv_ensemble_pipeline", "PASS", {
            "ens_train_participants": len(ens_train),
            "sm_train_participants": len(sm_train),
            "val_participants": len(ens_val),
            "best_p_value": ens_model_result["best_p_value"],
        })
    except Exception as e:
        tlog.log(f"  FAIL: {e}")
        tlog.add_result("cv_ensemble_pipeline", "FAIL", {"error": str(e)})
        raise

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    tlog.log("\n" + "=" * 60)
    passed = sum(1 for r in tlog.results["tests"] if r["status"] == "PASS")
    total = len(tlog.results["tests"])
    tlog.log(f"Results: {passed}/{total} tests passed")

    results_file = tlog.close()
    print(f"\nResults saved to: {results_file}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    main()
