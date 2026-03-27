#!/usr/bin/env python
"""
Cache management utility for Mal-ID-Lite.

Provides commands to:
- View cache information
- Clear participant cache
- Clear fold cache
- Clear all caches

Usage:
    python scripts/data/manage_cache.py info              # Show cache info
    python scripts/data/manage_cache.py clear-participants # Clear participant cache
    python scripts/data/manage_cache.py clear-folds        # Clear fold cache
    python scripts/data/manage_cache.py clear-all          # Clear all caches
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path (script is in scripts/data/, go up 2 levels)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from malid_lite.dataloader import MalIDPublishedDataLoader


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def get_dir_size(path):
    """Get total size of directory."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def show_cache_info(loader):
    """Display cache information."""
    print("\n" + "=" * 70)
    print("CACHE INFORMATION")
    print("=" * 70 + "\n")

    cache_info = loader.get_cache_info()

    if not cache_info["cache_dir"]:
        print("❌ No cache directory configured")
        return

    cache_dir = Path(cache_info["cache_dir"])
    if not cache_dir.exists():
        print(f"📂 Cache directory: {cache_dir}")
        print("❌ Cache directory does not exist yet")
        return

    print(f"📂 Cache directory: {cache_dir}\n")

    # Participant cache
    participants_dir = cache_dir / "participants"
    if participants_dir.exists():
        print("👤 PARTICIPANT CACHE (CLEAN stage)")
        print("   " + "─" * 66)
        p_info = cache_info["participants"]
        p_size = get_dir_size(participants_dir)

        print(f"   Files: {p_info['count']} participants")
        print(f"   Size:  {format_size(p_size)}")

        if p_info["metadata"]:
            meta = p_info["metadata"]
            print(f"   Created: {meta.get('created_at', 'unknown')}")
            print(f"   Version: {meta.get('malid_version', 'unknown')}")
            print(f"   Data source: {meta.get('data_dir', 'unknown')}")

        print()
    else:
        print("👤 PARTICIPANT CACHE: None\n")

    # Fold cache
    data_folds_dir = cache_dir / "data_folds"
    fold_files = list(data_folds_dir.glob("fold_*.parquet")) if data_folds_dir.exists() else []
    if fold_files:
        print("📁 FOLD CACHE (DOWNSAMPLED stage)")
        print("   " + "─" * 66)
        f_info = cache_info["folds"]
        f_size = sum(f.stat().st_size for f in fold_files)

        print(f"   Files: {f_info['count']} fold files")
        print(f"   Size:  {format_size(f_size)}")

        if f_info["metadata"]:
            meta = f_info["metadata"]
            print(f"   Created: {meta.get('created_at', 'unknown')}")
            print(f"   Version: {meta.get('malid_version', 'unknown')}")

        # Count folds
        folds_by_id = {}
        for f in fold_files:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                fold_id = parts[1]
                fold_label = parts[2]
                key = f"fold_{fold_id}"
                if key not in folds_by_id:
                    folds_by_id[key] = []
                folds_by_id[key].append(fold_label)

        print(f"   Folds: {len(folds_by_id)} folds")
        for fold_key, labels in sorted(folds_by_id.items()):
            print(f"      - {fold_key}: {', '.join(sorted(set(labels)))}")

        print()
    else:
        print("📁 FOLD CACHE: None\n")

    # Total
    total_size = get_dir_size(cache_dir)
    print(f"💾 Total cache size: {format_size(total_size)}\n")


def clear_participants(loader, confirm=True):
    """Clear participant cache."""
    print("\n🗑️  Clearing participant cache...")
    if confirm:
        response = input("Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    loader.clear_participant_cache(confirm=False)
    print("✓ Participant cache cleared\n")


def clear_folds(loader, confirm=True):
    """Clear fold cache."""
    print("\n🗑️  Clearing fold cache...")
    if confirm:
        response = input("Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    loader.clear_fold_cache(confirm=False)
    print("✓ Fold cache cleared\n")


def clear_all(loader, confirm=True):
    """Clear all caches."""
    print("\n🗑️  Clearing ALL caches...")
    if confirm:
        response = input("⚠️  This will delete ALL cached data. Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    loader.clear_all_caches(confirm=False)
    print("✓ All caches cleared\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1].lower()

    # Auto-detect project root (script is in scripts/data/, go up 2 levels)
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "cache" / "mal-id-orig-data"

    # Initialize loader (minimal setup for cache management)
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

    if command == "info":
        show_cache_info(loader)
    elif command == "clear-participants":
        clear_participants(loader)
    elif command == "clear-folds":
        clear_folds(loader)
    elif command == "clear-all":
        clear_all(loader)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
