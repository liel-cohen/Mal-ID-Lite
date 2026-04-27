#!/usr/bin/env python
"""
Cache management utility for Mal-ID-Lite.

Provides commands to inspect and clear the preprocessing cache.

Usage:
    python scripts/data/manage_cache.py info
    python scripts/data/manage_cache.py info --cache-dir /path/to/cache/mal-id-orig-data
    python scripts/data/manage_cache.py clear-participants --cache-dir /path/to/cache
    python scripts/data/manage_cache.py clear-folds --cache-dir /path/to/cache
    python scripts/data/manage_cache.py clear-embeddings --cache-dir /path/to/cache
    python scripts/data/manage_cache.py clear-all --cache-dir /path/to/cache

If --cache-dir is omitted, defaults to cache/mal-id-orig-data/ under the project root.
"""

import argparse
import shutil
import json
from pathlib import Path

import pandas as pd


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def get_dir_size(path):
    """Get total size of all files in a directory (recursive)."""
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob('*'):
        try:
            if item.is_file():
                total += item.stat().st_size
        except (PermissionError, OSError):
            pass
    return total


def read_cache_info(subdir):
    """Read cache_info.json from a cache subdirectory. Returns None on missing or corrupt file."""
    info_file = subdir / "cache_info.json"
    if info_file.exists():
        try:
            with open(info_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"   Warning: corrupt {info_file.name}: {e}")
            return None
    return None


def show_cache_info(cache_dir):
    """Display cache information."""
    print("\n" + "=" * 70)
    print("CACHE INFORMATION")
    print("=" * 70 + "\n")

    if not cache_dir.exists():
        print(f"Cache directory: {cache_dir}")
        print("Cache directory does not exist yet")
        return

    print(f"Cache directory: {cache_dir}\n")

    # Cached metadata
    cached_metadata = cache_dir / "metadata.tsv"
    cached_metadata_processed = cache_dir / "metadata_processed.tsv"
    if cached_metadata.exists() or cached_metadata_processed.exists():
        print("CACHED METADATA")
        if cached_metadata.exists():
            size = cached_metadata.stat().st_size
            print(f"   Original:  {format_size(size)} — {cached_metadata}")
        if cached_metadata_processed.exists():
            size = cached_metadata_processed.stat().st_size
            print(f"   Processed: {format_size(size)} — {cached_metadata_processed}")
            try:
                meta_proc = pd.read_csv(cached_metadata_processed, sep="\t")
                n_participants = meta_proc["participant_label"].nunique()
                print(f"              {n_participants} participants (filtered to those with raw data)")
            except Exception as e:
                print(f"              Warning: could not parse metadata: {e}")
    else:
        print("CACHED METADATA: None (cache not self-contained)")
    print()

    # Participant cache
    participants_dir = cache_dir / "participants"
    if participants_dir.exists():
        parquet_files = list(participants_dir.glob("*.parquet"))
        p_size = get_dir_size(participants_dir)
        meta = read_cache_info(participants_dir)

        print("PARTICIPANT CACHE (CLEAN stage)")
        print("   " + "-" * 66)
        print(f"   Files: {len(parquet_files)} participants")
        print(f"   Size:  {format_size(p_size)}")

        if meta:
            print(f"   Created: {meta.get('created_at', 'unknown')}")
            print(f"   Version: {meta.get('malid_version', 'unknown')}")
            print(f"   Data source: {meta.get('data_dir', 'unknown')}")
        print()
    else:
        print("PARTICIPANT CACHE: None\n")

    # Fold cache
    data_folds_dir = cache_dir / "data_folds"
    fold_files = list(data_folds_dir.glob("fold_*.parquet")) if data_folds_dir.exists() else []
    if fold_files:
        f_size = sum(f.stat().st_size for f in fold_files)
        meta = read_cache_info(data_folds_dir)

        print("FOLD CACHE (DOWNSAMPLED stage)")
        print("   " + "-" * 66)
        print(f"   Files: {len(fold_files)} fold files")
        print(f"   Size:  {format_size(f_size)}")

        if meta:
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
        print("FOLD CACHE: None\n")

    # Embedding cache
    embeddings_dir = cache_dir / "embeddings"
    if embeddings_dir.exists():
        npy_files = list(embeddings_dir.glob("*.npy"))
        e_size = get_dir_size(embeddings_dir)
        meta = read_cache_info(embeddings_dir)

        print("EMBEDDING CACHE (ESM-2)")
        print("   " + "-" * 66)
        print(f"   Files: {len(npy_files)} participants")
        print(f"   Size:  {format_size(e_size)}")

        if meta:
            print(f"   Created: {meta.get('created_at', 'unknown')}")
            print(f"   Model: {meta.get('model_name', 'unknown')}")
            print(f"   Dim: {meta.get('embedding_dim', 'unknown')}")
            print(f"   Dtype: {meta.get('storage_dtype', 'unknown')}")
            total_seqs = meta.get('total_sequences_embedded')
            if total_seqs:
                print(f"   Total sequences: {total_seqs:,}")
        print()
    else:
        print("EMBEDDING CACHE: None\n")

    # Total
    total_size = get_dir_size(cache_dir)
    print(f"Total cache size: {format_size(total_size)}\n")


def clear_directory(dir_path, label, confirm=True):
    """Clear a cache subdirectory."""
    if not dir_path.exists():
        print(f"No {label} cache to clear")
        return

    n_files = len(list(dir_path.iterdir()))
    if confirm:
        response = input(
            f"Delete {label} cache ({n_files} files in {dir_path})? (y/N): "
        )
        if response.lower() != 'y':
            print("Cancelled.")
            return

    shutil.rmtree(dir_path)
    print(f"Deleted {label} cache ({n_files} files)")


def parse_args():
    """Parse command-line arguments."""
    project_root = Path(__file__).parent.parent.parent
    default_cache = project_root / "cache" / "mal-id-orig-data"

    parser = argparse.ArgumentParser(
        description="Cache management utility for Mal-ID-Lite.",
    )
    parser.add_argument(
        "command",
        choices=["info", "clear-participants", "clear-folds",
                 "clear-embeddings", "clear-all"],
        help="Command to run.",
    )
    parser.add_argument(
        "--cache-dir", default=str(default_cache),
        help=f"Cache directory (default: {default_cache}).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    confirm = not args.yes

    if args.command == "info":
        show_cache_info(cache_dir)
    elif args.command == "clear-participants":
        clear_directory(cache_dir / "participants", "participant", confirm)
    elif args.command == "clear-folds":
        clear_directory(cache_dir / "data_folds", "fold", confirm)
    elif args.command == "clear-embeddings":
        clear_directory(cache_dir / "embeddings", "embedding", confirm)
    elif args.command == "clear-all":
        if confirm:
            response = input(
                f"Delete ALL caches in {cache_dir}? This cannot be undone. (y/N): "
            )
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
        for subdir, label in [
            ("participants", "participant"),
            ("data_folds", "fold"),
            ("embeddings", "embedding"),
        ]:
            path = cache_dir / subdir
            if path.exists():
                n_files = len(list(path.iterdir()))
                shutil.rmtree(path)
                print(f"Deleted {label} cache ({n_files} files)")
        # Delete metadata files
        for meta_name in ("metadata.tsv", "metadata_processed.tsv"):
            meta_path = cache_dir / meta_name
            if meta_path.exists():
                meta_path.unlink()
                print(f"Deleted cached {meta_name}")
        # Delete splits
        splits_dir = cache_dir / "splits"
        if splits_dir.exists():
            n_files = len(list(splits_dir.iterdir()))
            shutil.rmtree(splits_dir)
            print(f"Deleted splits cache ({n_files} files)")
        print("All caches cleared")

    return 0


if __name__ == "__main__":
    exit(main())
