"""Data loading and preprocessing for immune repertoire data."""

import argparse
from typing import Dict

from .base import (
    BaseDataLoader,
    PreprocessingStage,
    FOLD_COL,
    VALID_TRAINING_CONTEXTS,
    CV_TRAINING_CONTEXTS,
    TRAIN_ALL_TRAINING_CONTEXTS,
    normalize_fold_column,
    normalize_identifier_columns,
    _IDENTIFIER_COLS,
)
from .mal_id_published import MalIDPublishedDataLoader

__all__ = [
    "BaseDataLoader",
    "PreprocessingStage",
    "MalIDPublishedDataLoader",
    "FOLD_COL",
    "VALID_TRAINING_CONTEXTS",
    "CV_TRAINING_CONTEXTS",
    "TRAIN_ALL_TRAINING_CONTEXTS",
    "normalize_fold_column",
    "normalize_identifier_columns",
    "_IDENTIFIER_COLS",
    "add_clone_id_args",
    "get_clone_id_kwargs",
]


def add_clone_id_args(parser: argparse.ArgumentParser) -> None:
    """Add clone ID computation arguments to an argparse parser.

    Adds a "Clone ID computation" argument group with flags for controlling
    how clone_id is assigned — both when missing from the input data (auto-
    computed) and when overriding existing values (--force-clone-id).

    **Default behavior (no flags):** clone_id parameters default to None,
    meaning "unspecified." When loading from an existing cache, unspecified
    parameters are not validated — the cached values are accepted as-is.
    Only explicitly-provided parameters are validated against the cache.
    This allows the natural workflow: set clone_id parameters once at cache
    build time, then omit them on all subsequent training/embedding runs.

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
            "Defaults: TCR-NT=0.95, BCR-NT=0.90, TCR-AA=0.90, BCR-AA=0.85. "
            "Only needs to be specified at cache build time — subsequent "
            "commands accept the cached value if this flag is omitted."
        ),
    )
    clone_group.add_argument(
        "--clone-id-linkage-method",
        default=None,
        choices=["single", "complete", "average"],
        help=(
            "Linkage method for hierarchical clustering. "
            "Default when building cache: single. "
            "Only needs to be specified at cache build time — subsequent "
            "commands accept the cached value if this flag is omitted."
        ),
    )
    clone_group.add_argument(
        "--clone-id-use-aa",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Use amino acid CDR3 for clone assignment instead of nucleotide. "
            "Uses lower identity thresholds (TCR: 0.90, BCR: 0.85). "
            "Required when nucleotide CDR3 is not available in the data. "
            "Only needs to be specified at cache build time — subsequent "
            "commands accept the cached value if this flag is omitted."
        ),
    )


def get_clone_id_kwargs(args: argparse.Namespace) -> Dict:
    """Extract clone ID kwargs from parsed args for MalIDPublishedDataLoader.

    Returns a dict that can be unpacked into the loader constructor::

        loader = MalIDPublishedDataLoader(..., **get_clone_id_kwargs(args))

    Parameters that the user did not specify on the command line will be None,
    signaling "unspecified" to the loader. The loader resolves None to defaults
    for cache building, and skips validation for None params when loading from
    an existing cache.

    Args:
        args: Parsed argparse namespace (must have add_clone_id_args attributes).

    Returns:
        Dict with keys: force_clone_id, clone_id_identity_threshold,
        clone_id_linkage_method, clone_id_use_aa. Values are None for
        unspecified parameters (except force_clone_id which is bool).
    """
    return {
        "force_clone_id": args.force_clone_id,
        "clone_id_identity_threshold": args.clone_id_identity_threshold,
        "clone_id_linkage_method": args.clone_id_linkage_method,
        "clone_id_use_aa": args.clone_id_use_aa,
    }
