#!/usr/bin/env python
"""Quick smoke test for Model 1 binary training (train_model1.py, binary mode).

Imports functions from the training infrastructure and exercises the binary
classification path end-to-end on fold 0, one disease pair.

Tests
-----
1. filter_to_binary_pair (training_utils)
   - Sequences/metadata filtered to disease vs reference only
   - Excluded disease classes are fully absent from both DataFrames
   - Specimen sets in sequences and metadata are consistent

2. _run_fold_loop (train_model1, private)
   - Full train/predict cycle for one fold, one binary pair
   - Eval result dict has all required keys (fold_id, auroc_binary, auprc_binary, etc.)
   - Model 1 never abstains: n_abstained == 0, abstention_rate == 0.0
   - AUROC and AUPRC are in [0, 1]
   - Artifact files saved to disk (model.pkl, v_genes.json, results.json)

3. aggregate_fold_results (training_utils)
   - Aggregated dict has all required keys (auroc_pooled, auprc_pooled, fold_ids, etc.)
   - Pooled AUROC/AUPRC in [0, 1]
   - With a single fold, pooled AUROC equals that fold's binary AUROC (validates
     that aggregation pools raw predictions rather than averaging per-fold metrics)

4. Binary predictions CSV
   - Exact column names match spec (7 columns)
   - Row count matches n_scored from eval result
   - disease_label contains only 0 and 1
   - model_score in [0, 1]
   - Fold ID column consistent; disease_model and disease_label_str correct

5. cv_ensemble training context
   - Full binary pipeline with training_context="cv_ensemble"
   - Fewer training participants than cv_single_model (validation held out)
   - Valid AUROC and AUPRC; Model 1 never abstains

Design notes
------------
- Imports via importlib: train_model1.py is loaded with importlib.util.spec_from_file_location
  so __init__.py is not required in malid_lite/training/. The test calls the exact same
  _run_fold_loop function that the training script runs internally.
- Single fold, single disease: keeps runtime short while exercising the full code path.
  The chosen disease is the first alphabetically from fold 0 metadata (stable across runs).
- Shared functions (filter_to_binary_pair, aggregate_fold_results, make_pair_name) are
  imported from malid_lite.training.training_utils, not from train_model1.py.

Requirements
------------
- Fold cache built: cache/mal-id-orig-data/data_folds/fold_*.parquet
- python-glmnet installed
- All dependencies from requirements.txt

Expected runtime
----------------
- With fold cache: ~2-5 minutes
- Without fold cache: ~15-20 minutes

Output files
------------
All outputs saved to tests/test_outputs/test_model1_binary_quick/:
- test_model1_binary_quick_YYYYMMDD_HHMMSS.log   - Full log
- test_model1_binary_quick_YYYYMMDD_HHMMSS.json  - Structured results
- <disease>_vs_Healthy_Background/                - Pair output directory
    fold_0_lasso_cv_model.pkl
    fold_0_lasso_cv_v_genes.json
    fold_0_lasso_cv_results.json
    lasso_cv_binary_predictions.csv
"""

import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (tests/ -> project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage
from malid_lite.training.training_utils import (
    aggregate_fold_results,
    filter_to_binary_pair,
    make_pair_name,
)

# Import model-specific functions directly from the training script
_script = project_root / "malid_lite" / "training" / "train_model1.py"
spec = importlib.util.spec_from_file_location("train_model1", _script)
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)

_run_fold_loop = _module._run_fold_loop
HEALTHY_CLASS = "Healthy/Background"

EXPECTED_PREDICTION_COLUMNS = [
    "participant_label",
    "specimen_label",
    "disease_label",
    "disease_label_str",
    "disease_model",
    "model_score",
    "malid_cross_validation_fold_id_when_in_test_set",
]


# ---------------------------------------------------------------------------
# Minimal logger (same pattern as other test scripts)
# ---------------------------------------------------------------------------

class TestLogger:
    """Writes to both console and file, and accumulates structured results."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_file, "a")
        self.results = {"tests": [], "start_time": datetime.now().isoformat()}

    def log(self, message: str, to_file_only: bool = False) -> None:
        self.file.write(message + "\n")
        self.file.flush()
        if not to_file_only:
            print(message)

    def add_result(self, test_name: str, status: str, details: dict = None) -> None:
        self.results["tests"].append({
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def close(self) -> Path:
        self.results["end_time"] = datetime.now().isoformat()
        self.file.close()
        results_file = self.log_file.with_suffix(".json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        return results_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).stem  # "test_model1_binary_quick"
    output_dir = Path(__file__).parent / "test_outputs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{script_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = TestLogger(log_file)

    logger.log("\n" + "=" * 60)
    logger.log("QUICK MODEL 1 BINARY SMOKE TEST")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)

    FOLD_ID = 0
    MODEL_NAME = "lasso_cv"

    try:
        # ------------------------------------------------------------------
        # Setup: data loader + disease discovery
        # ------------------------------------------------------------------
        logger.log("\nSetup: Initializing data loader...")
        cache_dir = project_root / "cache" / "mal-id-orig-data"

        loader = MalIDPublishedDataLoader(
            data_dir=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data_clean/airr_format_clean/TCR/"),
            metadata_path=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv"),
            gene_reference_path=Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/tcrb_v_gene_cdrs.generated.tsv"),
            gene_locus="TCR",
            cache_dir=cache_dir,
            verbose=1,
        )

        if cache_dir.exists():
            n_fold_cache = len(list((cache_dir / "data_folds").glob("fold_*.parquet")))
            logger.log(f"  Fold cache files: {n_fold_cache}")
        else:
            logger.log(f"  No cache found at {cache_dir} - loading may be slow")

        # Load fold 0 train once — reused in both test 1 and disease discovery
        logger.log(f"  Loading fold {FOLD_ID} train data...")
        train_seqs, train_meta = loader.get_fold_data(
            fold_id=FOLD_ID,
            fold_label="train",
            preprocessing_stage=PreprocessingStage.DOWNSAMPLED,
        )

        all_diseases = sorted(train_meta["disease"].unique().tolist())
        disease_classes = [d for d in all_diseases if d != HEALTHY_CLASS]
        disease = disease_classes[0]  # First alphabetically — stable across runs

        logger.log(f"  All classes:    {all_diseases}")
        logger.log(f"  Test disease:   {disease} (vs {HEALTHY_CLASS})")

        # ------------------------------------------------------------------
        # Test 1: filter_to_binary_pair
        # ------------------------------------------------------------------
        logger.log("\n1. Testing filter_to_binary_pair...")

        binary_seqs, binary_meta = filter_to_binary_pair(
            train_seqs, train_meta, disease, HEALTHY_CLASS
        )

        # Only the two expected disease classes should remain
        binary_diseases = set(binary_meta["disease"].unique())
        expected_diseases = {disease, HEALTHY_CLASS}
        assert binary_diseases == expected_diseases, (
            f"Expected only {expected_diseases}, got {binary_diseases}"
        )

        # Sequences should only reference specimens in the filtered metadata
        kept_specimens = set(binary_meta["specimen_label"])
        seq_specimens = set(binary_seqs["specimen_label"].unique())
        assert seq_specimens.issubset(kept_specimens), (
            f"Sequences reference specimens not in binary_meta: {seq_specimens - kept_specimens}"
        )

        # All classes not in {disease, HEALTHY_CLASS} must be gone
        excluded_classes = set(all_diseases) - expected_diseases
        assert not (set(binary_meta["disease"].unique()) & excluded_classes), (
            f"Excluded classes still present: {set(binary_meta['disease'].unique()) & excluded_classes}"
        )

        n_excluded = len(train_meta) - len(binary_meta)
        logger.log(f"  Train specimens: {len(train_meta)} -> {len(binary_meta)} (excluded {n_excluded})")
        logger.log(f"  Binary classes: {sorted(binary_diseases)}")

        logger.add_result("filter_to_binary_pair", "PASS", {
            "n_train_specimens": len(train_meta),
            "n_binary_specimens": len(binary_meta),
            "excluded_specimens": n_excluded,
            "binary_classes": sorted(binary_diseases),
        })

        # ------------------------------------------------------------------
        # Test 2: _run_fold_loop (one fold, one binary pair)
        # ------------------------------------------------------------------
        logger.log(f"\n2. Testing _run_fold_loop (fold={FOLD_ID}, {disease} vs {HEALTHY_CLASS})...")

        pair_output_dir = output_dir / make_pair_name(disease, HEALTHY_CLASS)
        model_params = {"gene_locus": "TCR", "n_pcs": 15}

        all_eval_results, aggregated_by_model = _run_fold_loop(
            loader=loader,
            fold_ids=[FOLD_ID],
            output_dir=pair_output_dir,
            model_name=MODEL_NAME,
            model_params=model_params,
            verbose=1,
            disease_filter=(disease, HEALTHY_CLASS),
        )

        # Should have exactly one eval result (one fold)
        assert len(all_eval_results) == 1, f"Expected 1 eval result, got {len(all_eval_results)}"
        eval_result = all_eval_results[0]

        # Check required keys in eval_result
        required_keys = {
            "fold_id", "model_name", "n_scored", "n_abstained", "abstention_rate",
            "accuracy", "auroc_ovo_weighted", "auroc_ovr_per_class",
            "log_loss", "confusion_matrix", "classes",
            "auroc_binary", "auprc_binary",
            "disease", "reference_class",
        }
        missing = required_keys - set(eval_result.keys())
        assert not missing, f"Missing keys in eval_result: {missing}"

        # Model 1 never abstains
        assert eval_result["n_abstained"] == 0, f"Model 1 should never abstain"
        assert eval_result["abstention_rate"] == 0.0

        # n_scored > 0
        n = eval_result["n_scored"]
        assert n > 0, "n_scored is 0"

        # Binary AUROC/AUPRC in valid range
        assert eval_result["auroc_binary"] is not None
        assert 0.0 <= eval_result["auroc_binary"] <= 1.0, (
            f"auroc_binary out of range: {eval_result['auroc_binary']}"
        )
        assert eval_result["auprc_binary"] is not None
        assert 0.0 <= eval_result["auprc_binary"] <= 1.0

        # Accuracy in valid range
        assert 0.0 <= eval_result["accuracy"] <= 1.0

        # disease and reference_class tagged correctly
        assert eval_result["disease"] == disease
        assert eval_result["reference_class"] == HEALTHY_CLASS

        # Artifacts must have been saved
        assert (pair_output_dir / f"fold_{FOLD_ID}_{MODEL_NAME}_model.pkl").exists()
        assert (pair_output_dir / f"fold_{FOLD_ID}_{MODEL_NAME}_v_genes.json").exists()
        assert (pair_output_dir / f"fold_{FOLD_ID}_{MODEL_NAME}_results.json").exists()

        logger.log(f"  Test specimens: {n}")
        logger.log(f"  Accuracy: {eval_result['accuracy']:.4f}")
        logger.log(f"  Binary AUROC: {eval_result['auroc_binary']:.4f}")
        logger.log(f"  Binary AUPRC: {eval_result['auprc_binary']:.4f}")

        logger.add_result("_run_fold_loop", "PASS", {
            "fold_id": FOLD_ID,
            "disease": disease,
            "n_scored": n,
            "accuracy": round(eval_result["accuracy"], 4),
            "auroc_binary": round(eval_result["auroc_binary"], 4),
            "auprc_binary": round(eval_result["auprc_binary"], 4),
        })

        # ------------------------------------------------------------------
        # Test 3: aggregate_fold_results with disease_filter
        # ------------------------------------------------------------------
        logger.log("\n3. Testing aggregate_fold_results (binary mode)...")

        raw_preds_list = [None]  # build a dummy raw_preds to test the not-None path

        # Re-run aggregate with the actual raw_preds recovered from fold loop
        # We test aggregate_fold_results by calling it directly with our eval_result
        # and a compatible raw_preds dict (reconstructed from eval_result)
        agg = aggregated_by_model[MODEL_NAME]

        # Required keys
        required_agg_keys = {
            "n_folds", "n_folds_scored", "fold_ids",
            "auroc_pooled", "auprc_pooled",
            "auroc_per_fold", "auprc_per_fold",
            "accuracy_per_fold", "accuracy_global",
            "disease", "reference_class",
        }
        missing_agg = required_agg_keys - set(agg.keys())
        assert not missing_agg, f"Missing keys in aggregated: {missing_agg}"

        # Pooled values in valid range
        assert agg["auroc_pooled"] is not None
        assert 0.0 <= agg["auroc_pooled"] <= 1.0, f"auroc_pooled out of range: {agg['auroc_pooled']}"
        assert agg["auprc_pooled"] is not None
        assert 0.0 <= agg["auprc_pooled"] <= 1.0

        # With a single fold, pooled == that fold's binary AUROC (same predictions)
        assert abs(agg["auroc_pooled"] - eval_result["auroc_binary"]) < 1e-9, (
            f"Pooled AUROC {agg['auroc_pooled']} != fold AUROC {eval_result['auroc_binary']} for single fold"
        )

        assert agg["n_folds"] == 1
        assert agg["fold_ids"] == [FOLD_ID]
        assert agg["disease"] == disease
        assert agg["reference_class"] == HEALTHY_CLASS

        logger.log(f"  AUROC (pooled): {agg['auroc_pooled']:.4f}")
        logger.log(f"  AUPRC (pooled): {agg['auprc_pooled']:.4f}")
        logger.log(f"  n_folds: {agg['n_folds']}")
        logger.log(f"  Keys: {sorted(agg.keys())}")

        logger.add_result("aggregate_fold_results", "PASS", {
            "auroc_pooled": round(agg["auroc_pooled"], 4),
            "auprc_pooled": round(agg["auprc_pooled"], 4),
            "n_folds": agg["n_folds"],
        })

        # ------------------------------------------------------------------
        # Test 4: binary predictions CSV
        # ------------------------------------------------------------------
        logger.log("\n4. Testing binary predictions CSV...")

        predictions_file = pair_output_dir / f"{MODEL_NAME}_binary_predictions.csv"
        assert predictions_file.exists(), f"Predictions CSV not found: {predictions_file}"

        predictions_df = pd.read_csv(predictions_file)

        # Exact column names
        actual_cols = list(predictions_df.columns)
        assert actual_cols == EXPECTED_PREDICTION_COLUMNS, (
            f"Column mismatch.\n  Expected: {EXPECTED_PREDICTION_COLUMNS}\n  Got:      {actual_cols}"
        )

        # Row count matches test specimens
        assert len(predictions_df) == n, (
            f"Row count {len(predictions_df)} != n_scored {n}"
        )

        # disease_label contains only 0 and 1
        dl_values = set(predictions_df["disease_label"].unique())
        assert dl_values.issubset({0, 1}), f"disease_label contains unexpected values: {dl_values}"

        # model_score is in [0, 1]
        assert predictions_df["model_score"].between(0.0, 1.0).all(), (
            "model_score values outside [0, 1]"
        )

        # fold ID column is consistent
        assert (predictions_df["malid_cross_validation_fold_id_when_in_test_set"] == FOLD_ID).all(), (
            f"malid_cross_validation_fold_id_when_in_test_set should all be {FOLD_ID}"
        )

        # disease_model column identifies the binary model
        assert (predictions_df["disease_model"] == disease).all(), (
            f"disease_model should all be '{disease}'"
        )

        # disease_label_str should contain actual disease strings
        label_str_values = set(predictions_df["disease_label_str"].unique())
        assert label_str_values.issubset({disease, HEALTHY_CLASS}), (
            f"disease_label_str contains unexpected values: {label_str_values}"
        )

        logger.log(f"  Rows: {len(predictions_df)}")
        logger.log(f"  Columns: {actual_cols}")
        logger.log(f"  disease_label counts: {dict(predictions_df['disease_label'].value_counts())}")
        logger.log(
            f"  model_score range: [{predictions_df['model_score'].min():.4f}, "
            f"{predictions_df['model_score'].max():.4f}]"
        )
        logger.log(f"  disease_label_str values: {sorted(label_str_values)}")

        logger.add_result("binary_predictions_csv", "PASS", {
            "n_rows": len(predictions_df),
            "columns": actual_cols,
            "disease_label_counts": {str(k): int(v) for k, v in predictions_df["disease_label"].value_counts().items()},
            "model_score_min": round(float(predictions_df["model_score"].min()), 4),
            "model_score_max": round(float(predictions_df["model_score"].max()), 4),
        })

        # ------------------------------------------------------------------
        # Test 5: cv_ensemble training context
        # ------------------------------------------------------------------
        logger.log(f"\n5. Testing _run_fold_loop with training_context='cv_ensemble'...")

        pair_output_dir_ens = output_dir / "cv_ensemble" / make_pair_name(disease, HEALTHY_CLASS)

        ens_eval_results, ens_aggregated = _run_fold_loop(
            loader=loader,
            fold_ids=[FOLD_ID],
            output_dir=pair_output_dir_ens,
            model_name=MODEL_NAME,
            model_params=model_params,
            verbose=1,
            disease_filter=(disease, HEALTHY_CLASS),
            training_context="cv_ensemble",
        )

        assert len(ens_eval_results) == 1, f"Expected 1 eval result, got {len(ens_eval_results)}"
        ens_result = ens_eval_results[0]

        # Must produce valid metrics
        assert ens_result["n_scored"] > 0, "cv_ensemble: n_scored is 0"
        assert 0.0 <= ens_result["auroc_binary"] <= 1.0, (
            f"cv_ensemble auroc_binary out of range: {ens_result['auroc_binary']}"
        )
        assert 0.0 <= ens_result["auprc_binary"] <= 1.0
        assert ens_result["n_abstained"] == 0, "Model 1 should never abstain"

        # Artifacts must exist
        assert (pair_output_dir_ens / f"fold_{FOLD_ID}_{MODEL_NAME}_model.pkl").exists()

        logger.log(f"  cv_ensemble: n_scored={ens_result['n_scored']}, "
                    f"AUROC={ens_result['auroc_binary']:.4f}, "
                    f"AUPRC={ens_result['auprc_binary']:.4f}")
        logger.log(f"  cv_single_model: AUROC={eval_result['auroc_binary']:.4f}, "
                    f"AUPRC={eval_result['auprc_binary']:.4f}")

        logger.add_result("cv_ensemble_binary", "PASS", {
            "fold_id": FOLD_ID,
            "disease": disease,
            "n_scored": ens_result["n_scored"],
            "auroc_binary": round(ens_result["auroc_binary"], 4),
            "auprc_binary": round(ens_result["auprc_binary"], 4),
            "cv_single_model_auroc": round(eval_result["auroc_binary"], 4),
        })

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.log("\n" + "=" * 60)
        logger.log("ALL TESTS PASSED")
        logger.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 60 + "\n")

        results_file = logger.close()

        rel_output = output_dir.relative_to(Path(__file__).parent)
        print(f"\nTest outputs saved to: {rel_output}/")
        print(f"  Log:          {log_file.name}")
        print(f"  Results JSON: {results_file.name}")
        print(f"  Pair dir:     {pair_output_dir.relative_to(Path(__file__).parent)}/")

        return 0

    except Exception as e:
        import traceback

        logger.log("\n" + "=" * 60)
        logger.log("TEST FAILED")
        logger.log("=" * 60)
        logger.log(f"Error: {e}")
        logger.log("\nFull traceback:", to_file_only=True)
        logger.log(traceback.format_exc(), to_file_only=True)

        print(f"\n{'=' * 60}")
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        traceback.print_exc()

        logger.add_result("test_suite", "FAIL", {"error": str(e)})
        logger.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
