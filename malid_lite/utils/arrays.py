"""Array and string utility functions.

Inlined from the genetools package:
    Author:  Maxim Zaslavsky
    Version: 0.7.5
    GitHub:  https://github.com/maximz/genetools
    Source:  genetools/arrays.py

Only the functions required by Mal-ID-Lite are included. The logic is
unchanged from the original source.
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd


def strings_to_character_arrays(
    strs: Union[np.ndarray, List[str], pd.Series], validate_equal_lengths: bool = True
) -> np.ndarray:
    """Create character matrix by viewing strings as 1-character string arrays, then reshaping.

    Each row is one string; each column is one character position.
    """
    char_matrix = np.array(strs)
    char_matrix = (
        char_matrix.astype("bytes").view("S1").reshape((char_matrix.shape[0], -1))
    )

    if validate_equal_lengths and b"" in char_matrix:
        # Spaces (" ", ASCII 32) won't trigger this — only true empty slots (\x00).
        raise ValueError("Input strings must be of equal lengths.")

    return char_matrix


def strings_to_numeric_vectors(
    strs: Union[np.ndarray, List[str], pd.Series], validate_equal_lengths: bool = True
) -> np.ndarray:
    """Convert strings to numeric vectors (one uint8 entry per character).

    Blank positions (\\x00) are replaced with np.nan (cast to float).
    Used for computing pairwise Hamming distances between CDR3 sequences.
    """
    numeric_arr = strings_to_character_arrays(
        strs, validate_equal_lengths=validate_equal_lengths
    ).view(np.uint8)

    if 0 in numeric_arr:
        numeric_arr = numeric_arr.astype(float)
        np.place(numeric_arr, numeric_arr == 0.0, np.nan)

    return numeric_arr


def weighted_mode(
    arr: Union[list, np.ndarray, pd.Series],
    weights: Union[List[int], np.ndarray, pd.Series],
) -> Any:
    """Return the weighted mode (most common value) of an array.

    Faster than sklearn.utils.extmath.weighted_mode but does not support
    axis vectorization — must be called per column.
    """
    return (
        pd.DataFrame({"key": arr, "value": weights})
        .groupby("key", observed=True, sort=False)["value"]
        .sum()
        .idxmax()
    )


def make_consensus_vector(
    matrix: np.ndarray, frequencies: Union[np.ndarray, List[int], pd.Series]
) -> np.ndarray:
    """Get the weighted mode for each position across a set of equal-length vectors.

    Applies weighted_mode column-wise to a 2D character matrix.
    """
    return np.apply_along_axis(
        lambda col: weighted_mode(col, frequencies), 0, matrix
    )


def make_consensus_sequence(
    sequences: Union[np.ndarray, pd.Series, List[str]],
    frequencies: Union[np.ndarray, pd.Series, List[int]],
) -> str:
    """Get the weighted-mode character at each position across equal-length strings.

    Returns the consensus (centroid) string. Used for computing cluster centroid
    CDR3 sequences in the convergent cluster classifier.
    """
    sequences = np.array(sequences)
    if sequences.shape[0] == 1:
        return sequences.item(0)

    char_matrix = strings_to_character_arrays(sequences, validate_equal_lengths=True)
    consensus_elements = make_consensus_vector(char_matrix, frequencies)
    return "".join(consensus_elements.astype(str))


def _masked_vector_fill_for_argmin_argmax_output(
    masked_arr: np.ma.MaskedArray,
    output: Union[np.ndarray, float, int],
    axis: Optional[int] = None,
) -> Union[np.ndarray, float, int]:
    """Fix argmin/argmax output for all-masked rows/columns.

    numpy returns 0 for all-masked rows/columns; this replaces those with NaN.
    Internal helper for masked_argmin and masked_argmax.
    """
    if np.isscalar(output):
        return np.nan if masked_arr.mask.all(axis=axis) else output

    output = output.astype(float)
    output[masked_arr.mask.all(axis=axis)] = np.nan
    return output


def masked_argmin(
    masked_arr: np.ma.MaskedArray, axis: Optional[int] = None
) -> Union[np.ndarray, float, int]:
    """argmin on a masked array. Returns NaN for all-masked rows/columns.

    Used for assigning sequences to their nearest cluster centroid.
    """
    argmin_output = masked_arr.argmin(
        fill_value=np.ma.minimum_fill_value(masked_arr), axis=axis
    )
    return _masked_vector_fill_for_argmin_argmax_output(
        masked_arr, argmin_output, axis=axis
    )
