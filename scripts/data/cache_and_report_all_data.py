#!/usr/bin/env python
"""
Cache all preprocessed data and generate comprehensive data quality reports.

This script uses a two-phase approach:
1. Phase 1: Process all participants once (creates participant-level cache)
   - Skipped only when ALL participants are already cached
   - Reads stats from existing cache JSON files if available
2. Phase 2: Build fold caches from participant caches (~6x faster)
3. Generate detailed preprocessing reports and summary statistics
4. Provide data quality insights at each preprocessing stage

Usage:
    cd Mal-ID-Lite
    python scripts/data/cache_and_report_all_data.py

    To force reprocessing (ignore existing cache):
    Edit main() call at bottom: main(use_participant_cache=False)

Output:
    - cache/<dataset_name>/participants/  (participant-level cache)
    - cache/<dataset_name>/data_folds/   (fold-level cache)
    - cache/<dataset_name>/reports/      (data quality reports)
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List
import pandas as pd
import numpy as np

# Add project root to path (script is in scripts/data/, go up 2 levels)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader, PreprocessingStage


class DataReportGenerator:
    """Generate comprehensive data quality reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate statistics across all folds
        self.global_stats = {
            "total_participants": 0,
            "total_specimens": 0,
            "specimens_by_stage": {},
            "sequences_by_stage": {},
            "drop_reasons": {},
            "disease_distribution": {},
            "gene_fixes": {},
            "seq_cleaning_changes": {},
            "genes_missing_from_reference": {},
            "fold_distribution": {},
        }

    def update_fold_stats(
        self,
        fold_id: int,
        fold_label: str,
        sequences_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        preprocessing_report: pd.DataFrame
    ):
        """Update global statistics with fold data."""
        fold_key = f"fold_{fold_id}_{fold_label}"

        # Count specimens
        n_specimens = len(metadata_df)
        self.global_stats["total_specimens"] += n_specimens

        # Count sequences
        stage = "downsampled"
        if stage not in self.global_stats["sequences_by_stage"]:
            self.global_stats["sequences_by_stage"][stage] = 0
        self.global_stats["sequences_by_stage"][stage] += len(sequences_df)

        # Fold distribution
        self.global_stats["fold_distribution"][fold_key] = {
            "specimens": n_specimens,
            "sequences": len(sequences_df)
        }

        # Disease distribution
        if "disease" in metadata_df.columns:
            for disease, count in metadata_df["disease"].value_counts().items():
                if disease not in self.global_stats["disease_distribution"]:
                    self.global_stats["disease_distribution"][disease] = 0
                self.global_stats["disease_distribution"][disease] += count

        # Drop reasons from preprocessing report
        if "kept" in preprocessing_report.columns:
            dropped = preprocessing_report[preprocessing_report["kept"] == False]
            if "drop_reason" in dropped.columns:
                for reason in dropped["drop_reason"].dropna():
                    if reason not in self.global_stats["drop_reasons"]:
                        self.global_stats["drop_reasons"][reason] = 0
                    self.global_stats["drop_reasons"][reason] += 1

        # Gene fixes
        if "gene_name_fixes_detail" in preprocessing_report.columns:
            for fixes_dict in preprocessing_report["gene_name_fixes_detail"].dropna():
                if isinstance(fixes_dict, dict):
                    for fix, count in fixes_dict.items():
                        if fix not in self.global_stats["gene_fixes"]:
                            self.global_stats["gene_fixes"][fix] = 0
                        self.global_stats["gene_fixes"][fix] += count

        # Sequence cleaning changes (bad characters removed)
        if "seq_cleaning_changes" in preprocessing_report.columns:
            for cleaning_dict in preprocessing_report["seq_cleaning_changes"].dropna():
                if isinstance(cleaning_dict, dict):
                    for col, count in cleaning_dict.items():
                        if col not in self.global_stats["seq_cleaning_changes"]:
                            self.global_stats["seq_cleaning_changes"][col] = 0
                        self.global_stats["seq_cleaning_changes"][col] += count

        # Genes missing from reference table
        if "genes_missing_from_reference" in preprocessing_report.columns:
            for missing_dict in preprocessing_report["genes_missing_from_reference"].dropna():
                if isinstance(missing_dict, dict):
                    for gene, count in missing_dict.items():
                        if gene not in self.global_stats["genes_missing_from_reference"]:
                            self.global_stats["genes_missing_from_reference"][gene] = 0
                        self.global_stats["genes_missing_from_reference"][gene] += count

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate high-level summary statistics."""
        summary = []

        # Overall counts
        summary.append({
            "metric": "Total Specimens",
            "value": self.global_stats["total_specimens"],
            "category": "Overview"
        })

        # Sequences by stage
        for stage, count in self.global_stats["sequences_by_stage"].items():
            summary.append({
                "metric": f"Total Sequences ({stage})",
                "value": count,
                "category": "Overview"
            })

        # Disease distribution
        for disease, count in sorted(
            self.global_stats["disease_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary.append({
                "metric": disease,
                "value": count,
                "category": "Disease Distribution"
            })

        # Drop reasons
        for reason, count in sorted(
            self.global_stats["drop_reasons"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary.append({
                "metric": reason,
                "value": count,
                "category": "Dropped Specimens"
            })

        # Gene fixes
        for fix, count in sorted(
            self.global_stats["gene_fixes"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary.append({
                "metric": fix,
                "value": count,
                "category": "Gene Name Corrections"
            })

        # Sequence cleaning changes
        for col, count in sorted(
            self.global_stats["seq_cleaning_changes"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary.append({
                "metric": f"{col} sequences cleaned",
                "value": count,
                "category": "Sequence Cleaning"
            })

        # Genes missing from reference table
        for gene, count in sorted(
            self.global_stats["genes_missing_from_reference"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            summary.append({
                "metric": gene,
                "value": count,
                "category": "V Genes Missing From Reference (rows with NaN FR/CDR)"
            })

        # Fold distribution
        for fold_key, stats in sorted(self.global_stats["fold_distribution"].items()):
            summary.append({
                "metric": f"{fold_key} specimens",
                "value": stats["specimens"],
                "category": "Fold Distribution"
            })
            summary.append({
                "metric": f"{fold_key} sequences",
                "value": stats["sequences"],
                "category": "Fold Distribution"
            })

        return pd.DataFrame(summary)

    def generate_detailed_stats(
        self,
        preprocessing_report: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Generate detailed statistics from preprocessing report."""
        reports = {}

        # 1. Sequence count distributions per specimen
        if "original_count" in preprocessing_report.columns:
            reports["sequence_counts"] = pd.DataFrame({
                "stage": ["raw", "clean", "downsampled"],
                "mean": [
                    preprocessing_report["original_count"].mean(),
                    preprocessing_report["after_clean"].mean() if "after_clean" in preprocessing_report.columns else np.nan,
                    preprocessing_report["after_downsample"].mean() if "after_downsample" in preprocessing_report.columns else np.nan,
                ],
                "median": [
                    preprocessing_report["original_count"].median(),
                    preprocessing_report["after_clean"].median() if "after_clean" in preprocessing_report.columns else np.nan,
                    preprocessing_report["after_downsample"].median() if "after_downsample" in preprocessing_report.columns else np.nan,
                ],
                "min": [
                    preprocessing_report["original_count"].min(),
                    preprocessing_report["after_clean"].min() if "after_clean" in preprocessing_report.columns else np.nan,
                    preprocessing_report["after_downsample"].min() if "after_downsample" in preprocessing_report.columns else np.nan,
                ],
                "max": [
                    preprocessing_report["original_count"].max(),
                    preprocessing_report["after_clean"].max() if "after_clean" in preprocessing_report.columns else np.nan,
                    preprocessing_report["after_downsample"].max() if "after_downsample" in preprocessing_report.columns else np.nan,
                ],
            })

        # 2. Filter effectiveness (sequences dropped at each stage)
        filter_stats = []
        filter_cols = [
            ("productive_filter", "Productive Filter"),
            ("v_score_filter", "V Score Filter"),
            ("missing_fields", "Missing Fields"),
        ]
        for col, name in filter_cols:
            if col in preprocessing_report.columns:
                filter_stats.append({
                    "filter": name,
                    "total_dropped": preprocessing_report[col].sum(),
                    "specimens_affected": (preprocessing_report[col] > 0).sum(),
                    "avg_dropped_per_specimen": preprocessing_report[col].mean(),
                })

        if filter_stats:
            reports["filter_effectiveness"] = pd.DataFrame(filter_stats)

        # 3. Specimens by fold and disease
        if "fold_id" in preprocessing_report.columns and "disease" in preprocessing_report.columns:
            kept = preprocessing_report[preprocessing_report["kept"] == True]
            fold_disease = kept.groupby(["fold_id", "disease"]).size().reset_index(name="count")
            reports["fold_disease_distribution"] = fold_disease.pivot(
                index="disease", columns="fold_id", values="count"
            ).fillna(0).astype(int)

        # 4. Clone count statistics (if available)
        if "n_clones" in preprocessing_report.columns:
            kept = preprocessing_report[preprocessing_report["kept"] == True]
            reports["clone_stats"] = pd.DataFrame({
                "metric": ["mean", "median", "min", "max", "std"],
                "value": [
                    kept["n_clones"].mean(),
                    kept["n_clones"].median(),
                    kept["n_clones"].min(),
                    kept["n_clones"].max(),
                    kept["n_clones"].std(),
                ]
            })

        return reports

    def save_reports(
        self,
        summary_df: pd.DataFrame,
        detailed_reports: Dict[str, pd.DataFrame],
        preprocessing_report: pd.DataFrame,
        timestamp: str,
        logger=None
    ):
        """Save all reports to disk."""
        log = logger.info if logger else print

        # Summary report
        summary_file = self.output_dir / f"summary_report_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        log(f"  ✓ Summary report: {summary_file.name}")

        # Detailed statistics
        for report_name, df in detailed_reports.items():
            report_file = self.output_dir / f"{report_name}_{timestamp}.csv"
            df.to_csv(report_file)
            log(f"  ✓ {report_name}: {report_file.name}")

        # Full preprocessing report
        preproc_file = self.output_dir / f"preprocessing_report_full_{timestamp}.csv"
        preprocessing_report.to_csv(preproc_file, index=False)
        log(f"  ✓ Full preprocessing report: {preproc_file.name}")

        # JSON summary for programmatic access
        json_file = self.output_dir / f"global_stats_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.global_stats, f, indent=2, default=str)
        log(f"  ✓ Global stats JSON: {json_file.name}")


def load_existing_participant_stats(
    cache_dir: Path, metadata: pd.DataFrame, logger
) -> pd.DataFrame:
    """Load preprocessing stats from existing participant cache JSON files, enriched with metadata."""
    participants_dir = cache_dir / "participants"

    if not participants_dir.exists():
        logger.info("No existing participant cache found")
        return pd.DataFrame()

    stats_files = list(participants_dir.glob("*_stats.json"))
    if not stats_files:
        logger.info("No stats JSON files found in participant cache")
        return pd.DataFrame()

    logger.info(f"Loading stats from {len(stats_files)} cached participants...")

    all_stats = []
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            # Extract participant label from filename
            participant_label = stats_file.stem.replace("_stats", "")
            stats["participant_label"] = participant_label

            all_stats.append(stats)
        except Exception as e:
            logger.warning(f"Failed to load {stats_file.name}: {e}")

    if not all_stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_stats)
    logger.info(f"✓ Loaded stats for {len(stats_df)} participants from cache")

    # Enrich with disease info from metadata (one disease value per participant)
    if metadata is not None and len(metadata) > 0 and "participant_label" in metadata.columns:
        if "disease" in metadata.columns:
            meta_per_participant = (
                metadata.groupby("participant_label", as_index=False)
                .first()[["participant_label", "disease"]]
            )
            stats_df = stats_df.merge(meta_per_participant, on="participant_label", how="left")

    return stats_df


def main(use_participant_cache: bool = True):
    """
    Cache all preprocessed data and generate comprehensive reports.

    Args:
        use_participant_cache: If True (default), reuse existing participant cache
                              only when ALL participants are already cached.
                              If False, force reprocessing of all participants.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Auto-detect project root (script is in scripts/data/, go up two levels)
    project_root = Path(__file__).parent.parent.parent

    # Output directories - cache/<dataset_name>/ scopes each dataset separately
    cache_root = project_root / "cache"
    dataset_name = "mal-id-orig-data"
    cache_dir = cache_root / dataset_name  # Base cache directory for loader
    report_dir = cache_dir / "reports"

    # Configure logging
    log_file = report_dir / f"caching_log_{timestamp}.txt"
    report_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("DATA CACHING AND REPORTING SCRIPT")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Use existing participant cache: {use_participant_cache}")
    logger.info("\nCache organization:")
    logger.info(f"  {cache_root}/")
    logger.info(f"  └── {dataset_name}/")
    logger.info(f"      ├── participants/              (participant-level cache, CLEAN stage)")
    logger.info(f"      ├── data_folds/fold_*.parquet (fold-level cache, DOWNSAMPLED stage)")
    logger.info(f"      ├── data_folds/fold_*.csv     (fold metadata)")
    logger.info(f"      └── reports/                  (data quality reports)")

    # Initialize data loader with caching enabled
    logger.info("\n1. Initializing data loader...")
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
        verbose=1,
        cache_dir=cache_dir,
    )
    logger.info("✓ Loader initialized with caching enabled")
    logger.info(f"  Cache directory: {cache_dir.absolute()}")
    logger.info(f"  Reports directory: {report_dir.absolute()}")

    # Initialize report generator
    report_gen = DataReportGenerator(report_dir)

    # PHASE 1: Process all participants once (creates participant-level cache)
    logger.info("\n2. Phase 1: Processing all participants...")
    logger.info("   (Creates participant-level cache for efficient fold building)\n")

    # Get all unique participants
    all_participants = loader.metadata["participant_label"].unique()
    total_participants = len(all_participants)

    # Check if participant cache is complete (ALL participants cached, not just some)
    participants_dir = cache_dir / "participants"
    cached_count = len(list(participants_dir.glob("*_clean.parquet"))) if participants_dir.exists() else 0
    existing_cache = (cached_count == total_participants)

    if use_participant_cache and existing_cache:
        logger.info(f"✓ Found complete participant cache ({cached_count}/{total_participants} participants)")
        logger.info(f"   Skipping Phase 1 (reprocessing disabled)")
        logger.info(f"   To force reprocessing, set use_participant_cache=False\n")
    else:
        if not use_participant_cache:
            logger.info(f"   Reprocessing enabled (use_participant_cache=False)")
        elif cached_count > 0:
            logger.info(f"   Incomplete cache found ({cached_count}/{total_participants}). Reprocessing all participants.")
        logger.info(f"Total participants to process: {total_participants}\n")

        for idx, participant_label in enumerate(all_participants, 1):
            if idx % 50 == 0 or idx == 1:
                logger.info(f"Processing participant {idx}/{total_participants}: {participant_label}")

            # Load participant data - this will automatically cache if not already cached
            df = loader.load_participant_data(participant_label, PreprocessingStage.CLEAN)

            if df.empty:
                logger.warning(f"No data for participant: {participant_label}")

        logger.info(f"\n✓ Phase 1 complete: All {total_participants} participants cached")

    # Show cache info
    cache_info = loader.get_cache_info()
    if cache_info["participants"]:
        logger.info(f"  Participant cache: {cache_info['participants']['count']} files")
        if cache_info['participants']['metadata']:
            created_at = cache_info['participants']['metadata'].get('created_at', 'unknown')
            version = cache_info['participants']['metadata'].get('malid_version', 'unknown')
            logger.info(f"  Created: {created_at}")
            logger.info(f"  Mal-ID version: {version}")

    # PHASE 2: Build fold caches from participant caches
    logger.info("\n" + "=" * 70)
    logger.info("3. Phase 2: Building fold-level caches...")
    logger.info("   (Much faster - loads from participant caches)")
    logger.info("=" * 70 + "\n")

    all_folds = []
    for fold_id in range(3):  # Folds 0-2
        for fold_label in ["train", "test"]:
            all_folds.append((fold_id, fold_label))

    total_folds = len(all_folds)

    for idx, (fold_id, fold_label) in enumerate(all_folds, 1):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"Processing Fold {fold_id} - {fold_label.upper()} ({idx}/{total_folds})")
        logger.info(f"{'─' * 70}")

        # Check if already cached
        try:
            cached_data = loader.load_cached_fold(
                fold_id, fold_label, PreprocessingStage.DOWNSAMPLED
            )
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            cached_data = None

        if cached_data is not None:
            sequences_df, metadata_df = cached_data
            logger.info(f"✓ Loaded from fold cache")
            logger.info(f"  - {len(metadata_df)} specimens")
            logger.info(f"  - {len(sequences_df):,} sequences")
        else:
            # Build fold from participant caches (fast!)
            logger.info(f"Building from participant caches...")

            # Get data (loads from participant cache, applies DOWNSAMPLED stage,
            # and populates loader._preprocessing_stats for reports)
            sequences_df, metadata_df = loader.get_fold_data(
                fold_id, fold_label, PreprocessingStage.DOWNSAMPLED
            )

            logger.info(f"✓ Built fold")
            logger.info(f"  - {len(metadata_df)} specimens")
            logger.info(f"  - {len(sequences_df):,} sequences")

            # Cache for future use
            if len(sequences_df) > 0:
                logger.info(f"Caching fold to disk...")
                try:
                    loader.cache_fold(
                        fold_id, fold_label, PreprocessingStage.DOWNSAMPLED
                    )
                    logger.info(f"✓ Cached successfully")
                except Exception as e:
                    logger.error(f"Error caching fold: {e}")
            else:
                logger.info(f"⚠ No data to cache (empty fold)")

        # Show simple fold summary
        if len(sequences_df) > 0:
            diseases = metadata_df["disease"].value_counts() if "disease" in metadata_df.columns else {}
            logger.info(f"\nFold summary:")
            logger.info(f"  Specimens: {len(metadata_df)}")
            logger.info(f"  Sequences: {len(sequences_df):,}")
            if len(diseases) > 0:
                logger.info(f"  Diseases: {dict(diseases)}")

        # Track fold distribution for final summary
        fold_key = f"fold_{fold_id}_{fold_label}"
        report_gen.global_stats["fold_distribution"][fold_key] = {
            "specimens": len(metadata_df),
            "sequences": len(sequences_df)
        }

    # Generate comprehensive reports after Phase 2
    # (Phase 2 populates loader._preprocessing_stats with all specimen-level detail)
    logger.info("\n" + "=" * 70)
    logger.info("4. Generating comprehensive reports from all participants...")
    logger.info("=" * 70 + "\n")

    # Get full preprocessing report from in-memory stats (populated during Phase 2 fold building)
    full_preprocessing_report = loader.get_preprocessing_report()

    if len(full_preprocessing_report) > 0:
        # _preprocessing_stats accumulates one entry per (participant × fold appearance).
        # Each specimen appears in ~3 folds (1 test + 2 train), so deduplicate.
        if "specimen_label" in full_preprocessing_report.columns:
            n_before = len(full_preprocessing_report)
            full_preprocessing_report = full_preprocessing_report.drop_duplicates(
                subset=["specimen_label"]
            )
            n_after = len(full_preprocessing_report)
            if n_before != n_after:
                logger.info(
                    f"Deduplicated preprocessing report: {n_before} entries → {n_after} unique specimens"
                )
    else:
        # Phase 2 loaded from fold cache — _preprocessing_stats not populated.
        # Fall back to loading from participant stats JSON files.
        logger.info("No in-memory preprocessing stats (fold caches were used) - loading from cache files...")
        full_preprocessing_report = load_existing_participant_stats(
            cache_dir, loader.metadata, logger
        )

    if len(full_preprocessing_report) == 0:
        logger.warning("⚠ No preprocessing statistics available")
        logger.warning("   Cannot generate reports without stats")
        logger.warning("   Set use_participant_cache=False to reprocess and collect stats")
    else:
        logger.info(f"Preprocessing report: {len(full_preprocessing_report)} entries")

        # Update report generator with participant-level data.
        # Rows may come from either _preprocessing_stats (has specimen_label, kept, drop_reason)
        # or from stats JSON files (has participant_label, disease, gene fixes, etc.)
        for _, row in full_preprocessing_report.iterrows():
            # Determine which label key is present
            has_specimen = "specimen_label" in row.index and pd.notna(row.get("specimen_label"))
            has_participant = "participant_label" in row.index and pd.notna(row.get("participant_label"))

            if not has_specimen and not has_participant:
                continue

            # Add to disease distribution
            if "disease" in row.index and pd.notna(row.get("disease")):
                disease = row["disease"]
                if disease not in report_gen.global_stats["disease_distribution"]:
                    report_gen.global_stats["disease_distribution"][disease] = 0
                report_gen.global_stats["disease_distribution"][disease] += 1

            # Add to drop reasons (only available from specimen-level _preprocessing_stats)
            if has_specimen:
                if "kept" in row.index and row["kept"] == False and "drop_reason" in row.index:
                    reason = row["drop_reason"]
                    if pd.notna(reason):
                        if reason not in report_gen.global_stats["drop_reasons"]:
                            report_gen.global_stats["drop_reasons"][reason] = 0
                        report_gen.global_stats["drop_reasons"][reason] += 1

                # Count kept specimens (specimen-level only)
                if "kept" in row.index and row["kept"] == True:
                    report_gen.global_stats["total_specimens"] += 1

            # Gene fixes
            if "gene_name_fixes_detail" in row.index and pd.notna(row.get("gene_name_fixes_detail")):
                fixes_dict = row["gene_name_fixes_detail"]
                if isinstance(fixes_dict, dict):
                    for fix, count in fixes_dict.items():
                        if fix not in report_gen.global_stats["gene_fixes"]:
                            report_gen.global_stats["gene_fixes"][fix] = 0
                        report_gen.global_stats["gene_fixes"][fix] += count

            # Sequence cleaning changes
            if "seq_cleaning_changes" in row.index and pd.notna(row.get("seq_cleaning_changes")):
                cleaning_dict = row["seq_cleaning_changes"]
                if isinstance(cleaning_dict, dict):
                    for col, count in cleaning_dict.items():
                        if col not in report_gen.global_stats["seq_cleaning_changes"]:
                            report_gen.global_stats["seq_cleaning_changes"][col] = 0
                        report_gen.global_stats["seq_cleaning_changes"][col] += count

            # Genes missing from reference table
            if "genes_missing_from_reference" in row.index and pd.notna(row.get("genes_missing_from_reference")):
                missing_dict = row["genes_missing_from_reference"]
                if isinstance(missing_dict, dict):
                    for gene, count in missing_dict.items():
                        if gene not in report_gen.global_stats["genes_missing_from_reference"]:
                            report_gen.global_stats["genes_missing_from_reference"][gene] = 0
                        report_gen.global_stats["genes_missing_from_reference"][gene] += count

        # Generate summary
        summary_df = report_gen.generate_summary_report()

        # Generate detailed statistics
        detailed_reports = report_gen.generate_detailed_stats(full_preprocessing_report)

        # Save all reports
        logger.info("\nSaving reports...")
        report_gen.save_reports(
            summary_df, detailed_reports, full_preprocessing_report, timestamp, logger
        )

        logger.info(f"\n✓ Reports generated and saved")
        logger.info(f"  See: {report_dir}/")
        logger.info(f"  - summary_report_{timestamp}.csv")
        logger.info(f"  - preprocessing_report_full_{timestamp}.csv")
        logger.info(f"  - and {len(detailed_reports)} detailed report files")

    # If total_specimens wasn't counted from per-specimen stats (e.g., fold caches used),
    # derive it from the test fold distribution: each specimen appears in exactly one test fold.
    if report_gen.global_stats["total_specimens"] == 0 and report_gen.global_stats["fold_distribution"]:
        report_gen.global_stats["total_specimens"] = sum(
            fold_data["specimens"]
            for fold_key, fold_data in report_gen.global_stats["fold_distribution"].items()
            if "test" in fold_key
        )

    # Print summary to console
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS (All Participants)")
    logger.info("=" * 70)
    logger.info(f"\nTotal specimens processed: {report_gen.global_stats['total_specimens']}")

    # Calculate total sequences from fold distribution (to avoid duplication)
    total_sequences = sum(
        fold_data["sequences"]
        for fold_key, fold_data in report_gen.global_stats["fold_distribution"].items()
        if "test" in fold_key  # Only count test sets to avoid duplication
    )
    if total_sequences > 0:
        logger.info(f"Total sequences in test sets: {total_sequences:,}")

    if report_gen.global_stats['disease_distribution']:
        logger.info("\nDisease distribution:")
        for disease, count in sorted(
            report_gen.global_stats['disease_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {disease}: {count}")

    if report_gen.global_stats['drop_reasons']:
        logger.info("\nSpecimens dropped by reason:")
        for reason, count in sorted(
            report_gen.global_stats['drop_reasons'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {reason}: {count}")

    if report_gen.global_stats['gene_fixes']:
        logger.info("\nGene name corrections applied:")
        for fix, count in sorted(
            report_gen.global_stats['gene_fixes'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {fix}: {count:,} sequences")

    if report_gen.global_stats['seq_cleaning_changes']:
        logger.info("\nSequence cleaning (bad characters removed):")
        for col, count in sorted(
            report_gen.global_stats['seq_cleaning_changes'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {col}: {count:,} sequences")

    if report_gen.global_stats['genes_missing_from_reference']:
        logger.warning("\nV genes missing from reference table (FR/CDR will be NaN for these rows):")
        for gene, count in sorted(
            report_gen.global_stats['genes_missing_from_reference'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.warning(f"  {gene}: {count:,} rows")
    else:
        logger.info("\nAll V genes found in reference table.")

    # Fold distribution summary
    if report_gen.global_stats["fold_distribution"]:
        logger.info("\nFold distribution:")
        for fold_key in sorted(report_gen.global_stats["fold_distribution"].keys()):
            fold_data = report_gen.global_stats["fold_distribution"][fold_key]
            logger.info(f"  {fold_key}: {fold_data['specimens']} specimens, {fold_data['sequences']:,} sequences")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ PROCESSING COMPLETE")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Update cache info after Phase 2
    cache_info = loader.get_cache_info()

    logger.info(f"\nAll outputs in: {cache_dir}/")
    logger.info(f"   Participant cache: {cache_dir / 'participants'}/")
    logger.info(f"      - {cache_info['participants']['count']} participant files (CLEAN stage)")
    logger.info(f"      - Enables efficient fold building (~6x speedup)")
    logger.info(f"   Fold cache: {cache_dir / 'data_folds'}/")
    logger.info(f"      - {cache_info['folds']['count']} fold files (DOWNSAMPLED stage)")
    logger.info(f"      - Ready for model training")
    logger.info(f"   Reports: {report_dir}/")
    logger.info(f"   Log file: {log_file.name}\n")

    return 0


if __name__ == "__main__":
    exit(main())
