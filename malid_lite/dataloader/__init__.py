"""Data loading and preprocessing for immune repertoire data."""

import argparse
from typing import Dict

from .base import BaseDataLoader, PreprocessingStage, FOLD_COL, normalize_fold_column
from .mal_id_published import MalIDPublishedDataLoader

__all__ = [
    "BaseDataLoader",
    "PreprocessingStage",
    "MalIDPublishedDataLoader",
    "FOLD_COL",
    "normalize_fold_column",
    "add_clone_id_args",
    "get_clone_id_kwargs",
]


def add_clone_id_args(parser: argparse.ArgumentParser) -> None:
    """Add clone ID computation arguments to an argparse parser.

    Adds a "Clone ID computation" argument group with flags for controlling
    how clone_id is assigned — both when missing from the input data (auto-
    computed) and when overriding existing values (--force-clone-id).

    Args:
        parser: ArgumentParser to add the argument group to.
    """
    clone_group = parser.add_argument_group("Clone ID computation")
    clone_group.add_argument(
        "--force-clone-id",
        action="store_true",
        help=(
            "Compute clone_id even when the column already exists in the raw data. "
            "The original clone_id is preserved as clone_id_original. "
            "Only matters at cache build time — once cached, can be omitted on "
            "subsequent runs. If the cache was built without this flag, setting "
            "it later requires clearing the cache first (see manage_cache.py)."
        ),
    )
    clone_group.add_argument(
        "--clone-id-identity-threshold",
        type=float,
        default=None,
        help=(
            "Override the default CDR3 identity threshold for clone assignment. "
            "Defaults: TCR-NT=0.95, BCR-NT=0.90, TCR-AA=0.90, BCR-AA=0.85."
        ),
    )
    clone_group.add_argument(
        "--clone-id-linkage-method",
        default="single",
        choices=["single", "complete", "average"],
        help="Linkage method for hierarchical clustering (default: single).",
    )
    clone_group.add_argument(
        "--clone-id-use-aa",
        action="store_true",
        help=(
            "Use amino acid CDR3 for clone assignment instead of nucleotide. "
            "Uses lower identity thresholds (TCR: 0.90, BCR: 0.85). "
            "Required: explicitly opt in — the pipeline errors if nucleotide "
            "CDR3 is missing and this flag is not set."
        ),
    )


def get_clone_id_kwargs(args: argparse.Namespace) -> Dict:
    """Extract clone ID kwargs from parsed args for MalIDPublishedDataLoader.

    Returns a dict that can be unpacked into the loader constructor::

        loader = MalIDPublishedDataLoader(..., **get_clone_id_kwargs(args))

    Args:
        args: Parsed argparse namespace (must have add_clone_id_args attributes).

    Returns:
        Dict with keys: force_clone_id, clone_id_identity_threshold,
        clone_id_linkage_method, clone_id_use_aa.
    """
    return {
        "force_clone_id": args.force_clone_id,
        "clone_id_identity_threshold": args.clone_id_identity_threshold,
        "clone_id_linkage_method": args.clone_id_linkage_method,
        "clone_id_use_aa": args.clone_id_use_aa,
    }
