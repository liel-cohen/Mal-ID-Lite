"""Quick smoke test for data loader - tests basic functionality only."""

import sys
from pathlib import Path
from datetime import datetime
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from malid.dataloader import MalIDPublishedDataLoader, PreprocessingStage


class TestLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode since logging handlers already opened it
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

    # Create dedicated output folder for this test
    test_name = Path(__file__).stem  # "test_dataloader_quick"
    output_dir = Path(__file__).parent / "test_outputs" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"test_dataloader_quick_{timestamp}.log"

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
    logger.log("QUICK DATA LOADER SMOKE TEST")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)

    try:
        # Test 1: Initialize loader
        logger.log("\n1. Initializing data loader...")
        project_root = Path(__file__).parent.parent
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
            cache_dir=project_root / "cache",
            verbose=2,  # Debug level to see all logs
        )
        logger.log("✓ Loader initialized")
        logger.add_result("initialize_loader", "PASS")

        # Test 2: Load metadata
        logger.log("\n2. Loading metadata...")
        metadata = loader.metadata
        n_samples = len(metadata)
        n_participants = metadata["participant_label"].nunique()
        n_specimens = metadata["specimen_label"].nunique()

        logger.log(f"✓ Loaded {n_samples} samples")
        logger.log(f"  - {n_participants} participants")
        logger.log(f"  - {n_specimens} specimens")

        logger.add_result(
            "load_metadata",
            "PASS",
            {
                "n_samples": n_samples,
                "n_participants": n_participants,
                "n_specimens": n_specimens,
            },
        )

        # Test 3: Load participant data
        participant_label = metadata["participant_label"].iloc[0]
        logger.log(f"\n3. Testing with participant: {participant_label}")

        # 3a: RAW
        logger.log("\n  a) Loading RAW data...")
        df_raw = loader.load_participant_data(
            participant_label, PreprocessingStage.RAW
        )
        logger.log(f"     ✓ {len(df_raw)} sequences")

        # Save RAW data for manual inspection (CSV for easy viewing)
        raw_file = output_dir / f"sample_raw_{timestamp}.csv"
        df_raw.to_csv(raw_file, index=False)
        logger.log(f"     ✓ Saved to: {raw_file.name}")

        logger.add_result(
            "load_raw",
            "PASS",
            {"participant": participant_label, "n_sequences": len(df_raw), "saved_to": str(raw_file)},
        )

        # 3b: CLEAN
        logger.log("\n  b) Loading CLEAN data...")
        df_clean = loader.load_participant_data(
            participant_label, PreprocessingStage.CLEAN
        )
        logger.log(f"     ✓ {len(df_clean)} sequences after cleaning")

        # Check key columns
        key_cols = ["v_gene", "v_gene_w_allele", "j_gene", "isotype_supergroup"]
        present = [col for col in key_cols if col in df_clean.columns]
        logger.log(f"     ✓ Key columns present: {present}")

        isotypes = []
        if "isotype_supergroup" in df_clean.columns:
            isotypes = df_clean["isotype_supergroup"].unique().tolist()
            logger.log(f"     ✓ Isotypes: {isotypes}")

        # Show sequence cleaning statistics
        report = loader.get_preprocessing_report()
        if len(report) > 0:
            participant_stats = report[report["participant_label"] == participant_label]
            if len(participant_stats) > 0 and "seq_cleaning_changes" in participant_stats.columns:
                # Get the first row's cleaning changes (stats are per participant)
                cleaning_changes = participant_stats.iloc[0]["seq_cleaning_changes"]
                if cleaning_changes and isinstance(cleaning_changes, dict) and len(cleaning_changes) > 0:
                    logger.log(f"     ✓ Sequence cleaning (bad chars removed):")
                    for col, count in sorted(cleaning_changes.items(), key=lambda x: x[1], reverse=True):
                        logger.log(f"       - {col}: {count:,} sequences cleaned")

        # Save CLEAN data for manual inspection (CSV for easy viewing)
        clean_file = output_dir / f"sample_clean_{timestamp}.csv"
        df_clean.to_csv(clean_file, index=False)
        logger.log(f"     ✓ Saved to: {clean_file.name}")

        logger.add_result(
            "load_clean",
            "PASS",
            {
                "participant": participant_label,
                "n_sequences": len(df_clean),
                "columns_present": present,
                "isotypes": isotypes,
                "saved_to": str(clean_file),
            },
        )

        # 3c: DOWNSAMPLED
        logger.log("\n  c) Loading DOWNSAMPLED data...")
        df_down = loader.load_participant_data(
            participant_label, PreprocessingStage.DOWNSAMPLED
        )
        logger.log(f"     ✓ {len(df_down)} sequences after downsampling")

        specimens = []
        n_clones = 0
        if len(df_down) > 0:
            if "repertoire_id" in df_down.columns:
                specimens = df_down["repertoire_id"].unique().tolist()
                logger.log(f"     ✓ Specimens: {specimens}")
            if "igh_or_tcrb_clone_id" in df_down.columns:
                n_clones = df_down["igh_or_tcrb_clone_id"].nunique()
                logger.log(f"     ✓ Clones: {n_clones}")

        logger.add_result(
            "load_downsampled",
            "PASS",
            {
                "participant": participant_label,
                "n_sequences": len(df_down),
                "specimens": specimens,
                "n_clones": n_clones,
            },
        )

        # Test 4: Preprocessing report
        logger.log("\n4. Checking preprocessing report...")
        report = loader.get_preprocessing_report()

        if len(report) > 0:
            logger.log(f"✓ Report has {len(report)} rows")

            if "kept" in report.columns:
                kept = int(report["kept"].sum())
                dropped = len(report) - kept
                logger.log(f"  - Kept: {kept}, Dropped: {dropped}")

            # Save report
            report_file = output_dir / f"preprocessing_report_{timestamp}.csv"
            loader.save_preprocessing_report(report_file)
            logger.log(f"  - Saved to: {report_file.name}")

            logger.add_result(
                "preprocessing_report",
                "PASS",
                {
                    "n_rows": len(report),
                    "kept": kept if "kept" in report.columns else None,
                    "dropped": dropped if "kept" in report.columns else None,
                    "report_file": str(report_file),
                },
            )
        else:
            logger.log("⚠ No preprocessing statistics accumulated")
            logger.add_result("preprocessing_report", "SKIP", {"reason": "no_stats"})
            report_file = None

        # Test 5: Data flow summary
        logger.log("\n5. Data flow summary:")
        logger.log(f"   RAW:        {len(df_raw):,} sequences")
        clean_pct = f"{len(df_clean)/len(df_raw)*100:.1f}%" if len(df_raw) > 0 else "N/A"
        down_pct = f"{len(df_down)/len(df_clean)*100:.1f}%" if len(df_clean) > 0 else "N/A"
        logger.log(f"   CLEAN:      {len(df_clean):,} sequences ({clean_pct} retained)")
        logger.log(f"   DOWNSAMPLED: {len(df_down):,} sequences ({down_pct} retained)")

        logger.add_result(
            "data_flow",
            "PASS",
            {
                "raw": len(df_raw),
                "clean": len(df_clean),
                "downsampled": len(df_down),
                "clean_retention_pct": round(len(df_clean) / len(df_raw) * 100, 2),
                "downsample_retention_pct": round(len(df_down) / len(df_clean) * 100, 2),
            },
        )

        # Summary
        logger.log("\n" + "=" * 60)
        logger.log("✓ ALL TESTS PASSED")
        logger.log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 60 + "\n")

        # Close logger and save results
        results_file = logger.close()

        print(f"\n📝 Test output saved to tests/test_outputs/:")
        print(f"  - Log file: {log_file.name}")
        print(f"  - Results JSON: {results_file.name}")
        print(f"  - Preprocessing report: {report_file.name if report_file else 'N/A (no stats)'}")
        print(f"  - Raw data sample: sample_raw_{timestamp}.csv")
        print(f"  - Clean data sample: sample_clean_{timestamp}.csv")

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
