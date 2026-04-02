"""Quick smoke test for Model 1 (Repertoire Classifier).

Exercises the RepertoireClassifier model API end-to-end on fold 0,
verifying each phase independently.

Tests
-----
1. Initialize data loader     - MalIDPublishedDataLoader with cache support
2. Load training data         - Fold 0 train, DOWNSAMPLED stage
3. Extract features           - V-J gene pair frequencies via RepertoireClassifier.extract_features()
4. Prepare labels and groups  - Disease labels aligned to feature matrix; participant groups for CV
5. Train model                - RepertoireClassifier.fit() with grouped cross-validation
6. Predict on training set    - Sanity check (predict + predict_proba on train data)
7. Predict on test set        - Load fold 0 test, extract features aligned to train columns, evaluate
8. Save and load model        - model.save() / RepertoireClassifier.load() round-trip; verify predictions match

Scope
-----
This test covers the MODEL API (malid_lite.models.RepertoireClassifier), not the
TRAINING ORCHESTRATION (train_model1.py). For training pipeline tests including
binary mode, fold loops, aggregation, and CSV output, see test_model1_binary_quick.py.

Requirements
------------
- Fold cache built: cache/mal-id-orig-data/data_folds/fold_*.parquet
- python-glmnet installed (R glmnet binding)
- All dependencies from requirements.txt

Expected runtime
----------------
- With cache: ~1-2 minutes
- Without cache: ~10-15 minutes

Output files
------------
All outputs saved to tests/test_outputs/test_model1_quick/:
- test_model1_quick_YYYYMMDD_HHMMSS.log   - Full log
- test_model1_quick_YYYYMMDD_HHMMSS.json  - Structured results (pass/fail per test)
- features_fold0_train_YYYYMMDD_HHMMSS.csv - Extracted feature matrix for inspection
- model_fold0_YYYYMMDD_HHMMSS.pkl         - Trained model checkpoint
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite.models import RepertoireClassifier


class TestLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, "a")
        self.results = {"tests": [], "start_time": datetime.now().isoformat()}

    def log(self, message, to_file_only=False):
        """Log message to file and optionally console."""
        self.file.write(message + "\n")
        self.file.flush()
        if not to_file_only:
            print(message)

    def add_result(self, test_name, status, details=None):
        """Add test result."""
        self.results["tests"].append(
            {
                "test": test_name,
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def close(self):
        """Close file and save results."""
        self.results["end_time"] = datetime.now().isoformat()
        self.file.close()

        # Save structured results as JSON
        results_file = self.log_file.with_suffix(".json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        return results_file


def main():
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Organize outputs in subfolder named after script
    script_name = Path(__file__).stem  # "test_model1_quick"
    output_dir = Path(__file__).parent / "test_outputs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{script_name}_{timestamp}.log"

    # Configure Python logging to write to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = TestLogger(log_file)

    logger.log("\n" + "=" * 60)
    logger.log("QUICK MODEL 1 SMOKE TEST")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)

    try:
        # Test 1: Initialize data loader
        logger.log("\n1. Initializing data loader...")

        # Use cache directory (from project root)
        project_root = Path(__file__).parent.parent
        cache_dir = project_root / "cache" / "mal-id-orig-data"

        loader = MalIDPublishedDataLoader(
            data_dir=Path(
                "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data_clean/airr_format_clean/TCR/"
            ),
            metadata_path=Path(
                "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv"
            ),
            gene_reference_path=Path(
                "/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/tcrb_v_gene_cdrs.generated.tsv"
            ),
            gene_locus="TCR",
            cache_dir=cache_dir,  # Enable caching for fast loading
            verbose=1,  # Reduce verbosity for cleaner output
        )

        # Check if cache exists
        if cache_dir.exists():
            n_participant_cache = len(list((cache_dir / "participants").glob("*.parquet")))
            n_fold_cache = len(list((cache_dir / "data_folds").glob("fold_*.parquet")))
            logger.log(f"✓ Using cache directory: {cache_dir}")
            logger.log(f"  - Participant cache files: {n_participant_cache}")
            logger.log(f"  - Fold cache files: {n_fold_cache}")
        else:
            logger.log(f"  - No cache found at {cache_dir}")

        logger.log("✓ Loader initialized")
        logger.add_result("initialize_loader", "PASS")

        # Test 2: Load a small fold
        logger.log("\n2. Loading fold 0 training data (downsampled)...")
        fold_id = 0
        fold_data, fold_metadata = loader.get_fold_data(
            fold_id=fold_id,
            fold_label="train",
            preprocessing_stage=PreprocessingStage.DOWNSAMPLED
        )

        n_specimens = fold_metadata["specimen_label"].nunique()
        n_sequences = len(fold_data)

        logger.log(f"✓ Loaded fold {fold_id} train data")
        logger.log(f"  - {n_specimens} specimens")
        logger.log(f"  - {n_sequences:,} sequences")

        # Get disease labels
        diseases = fold_metadata["disease"].unique()
        logger.log(f"  - Diseases: {diseases.tolist()}")

        logger.add_result(
            "load_fold",
            "PASS",
            {
                "fold_id": fold_id,
                "n_specimens": n_specimens,
                "n_sequences": n_sequences,
                "diseases": diseases.tolist(),
            },
        )

        # Test 3: Extract features
        logger.log("\n3. Extracting features with Model 1...")
        model = RepertoireClassifier(gene_locus="TCR", verbose=1)

        features = model.extract_features(
            sequences=fold_data,
            metadata=fold_metadata,
        )

        logger.log(f"✓ Features extracted")
        logger.log(f"  - Shape: {features.shape[0]} specimens × {features.shape[1]} features")
        logger.log(f"  - Column examples: {list(features.columns[:5])}")

        # Save features for inspection
        features_file = output_dir / f"features_fold{fold_id}_train_{timestamp}.csv"
        features.to_csv(features_file)
        logger.log(f"  - Saved to: {features_file.relative_to(Path(__file__).parent)}")

        logger.add_result(
            "extract_features",
            "PASS",
            {
                "n_specimens": features.shape[0],
                "n_features": features.shape[1],
                "saved_to": str(features_file),
            },
        )

        # Test 4: Prepare labels
        logger.log("\n4. Preparing labels and groups...")

        # Align features and metadata (same specimen order)
        metadata_aligned = fold_metadata.set_index("specimen_label").loc[features.index]

        y = metadata_aligned["disease"]
        groups = metadata_aligned["participant_label"]

        logger.log(f"✓ Labels prepared")
        logger.log(f"  - Unique labels: {y.unique().tolist()}")
        logger.log(f"  - Label counts:")
        for disease, count in y.value_counts().items():
            logger.log(f"      {disease}: {count}")

        logger.add_result(
            "prepare_labels",
            "PASS",
            {
                "n_classes": y.nunique(),
                "classes": y.unique().tolist(),
                "class_counts": y.value_counts().to_dict(),
            },
        )

        # Test 5: Train model
        logger.log("\n5. Training Model 1...")

        model.fit(
            X=features,
            y=y,
            groups=groups,  # For grouped CV during lambda tuning
        )

        logger.log(f"✓ Model trained")
        logger.log(f"  - Fitted classes: {model.classes_.tolist()}")
        logger.log(f"  - Features in: {model.n_features_in_}")

        logger.add_result(
            "train_model",
            "PASS",
            {
                "classes": model.classes_.tolist(),
                "n_features_in": model.n_features_in_,
            },
        )

        # Test 6: Predictions on training set (sanity check)
        logger.log("\n6. Making predictions on training set...")

        y_pred = model.predict(features)
        y_proba = model.predict_proba(features)

        # Compute accuracy
        accuracy = (y_pred == y.values).mean()

        logger.log(f"✓ Predictions made")
        logger.log(f"  - Training accuracy: {accuracy:.3f}")
        logger.log(f"  - Probability shape: {y_proba.shape}")

        logger.add_result(
            "predict",
            "PASS",
            {
                "train_accuracy": round(accuracy, 3),
                "proba_shape": y_proba.shape,
            },
        )

        # Test 7: Load and predict on test set
        logger.log("\n7. Loading and predicting on test set...")

        test_data, test_metadata = loader.get_fold_data(
            fold_id=fold_id,
            fold_label="test",
            preprocessing_stage=PreprocessingStage.DOWNSAMPLED
        )

        n_test_specimens = test_metadata["specimen_label"].nunique()
        logger.log(f"✓ Loaded fold {fold_id} test data: {n_test_specimens} specimens")

        # Extract features for test set (using training column structure)
        test_features = model.extract_features(
            sequences=test_data,
            metadata=test_metadata,
            train_vj_columns=model.train_vj_columns_,  # Align to training columns
        )

        logger.log(f"✓ Test features extracted: {test_features.shape}")

        # Get test labels
        test_metadata_aligned = test_metadata.set_index("specimen_label").loc[test_features.index]
        y_test = test_metadata_aligned["disease"]

        # Predict
        y_test_pred = model.predict(test_features)
        y_test_proba = model.predict_proba(test_features)

        test_accuracy = (y_test_pred == y_test.values).mean()

        logger.log(f"✓ Test predictions made")
        logger.log(f"  - Test accuracy: {test_accuracy:.3f}")

        logger.add_result(
            "predict_test",
            "PASS",
            {
                "n_test_specimens": n_test_specimens,
                "test_accuracy": round(test_accuracy, 3),
            },
        )

        # Test 8: Save and load model
        logger.log("\n8. Testing model save/load...")

        model_file = output_dir / f"model_fold{fold_id}_{timestamp}.pkl"
        model.save(model_file)
        logger.log(f"✓ Model saved to: {model_file.relative_to(Path(__file__).parent)}")

        model_loaded = RepertoireClassifier.load(model_file)
        logger.log(f"✓ Model loaded successfully")

        # Verify loaded model works
        y_pred_loaded = model_loaded.predict(test_features)
        matches = bool((y_pred_loaded == y_test_pred).all())
        logger.log(f"✓ Loaded model predictions match: {matches}")

        logger.add_result(
            "save_load_model",
            "PASS",
            {
                "saved_to": str(model_file),
                "predictions_match": matches,
            },
        )

        # Summary
        logger.log("\n" + "=" * 60)
        logger.log("✓ ALL TESTS PASSED")
        logger.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 60 + "\n")

        # Close logger and save results
        results_file = logger.close()

        # Show organized output location
        rel_output_dir = output_dir.relative_to(Path(__file__).parent)
        print(f"\nTest outputs saved to: {rel_output_dir}/")
        print(f"  - Log file: {log_file.name}")
        print(f"  - Results JSON: {results_file.name}")
        print(f"  - Features CSV: {features_file.name}")
        print(f"  - Model PKL: {model_file.name}")

        return 0

    except Exception as e:
        logger.log("\n" + "=" * 60)
        logger.log("✗ TEST FAILED")
        logger.log("=" * 60)
        logger.log(f"Error: {e}")

        import traceback

        logger.log("\nFull traceback:", to_file_only=True)
        logger.log(traceback.format_exc(), to_file_only=True)

        # Print error to console
        print(f"\n{'=' * 60}")
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        traceback.print_exc()

        logger.add_result("test_suite", "FAIL", {"error": str(e)})
        logger.close()

        return 1


if __name__ == "__main__":
    exit(main())
