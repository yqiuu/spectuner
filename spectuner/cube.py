from typing import Optional

import numpy as np


def to_dense_matrix(arr: np.ndarray,
                    indices: np.ndarray,
                    shape: Optional[tuple]=None):
    """Covnert input array to dense matrix using indices.

    Args:
        arr:  Input array, whose shape can be (N,) or (N, D). If the shape is
            (N, D), the output matrix will have shape (W, H, D).
        indices: Indices (N, 2).
        shape: Shape of the output matrix. If ``None``, the shape will be
            inferred from the maximum indices.

    Returns:
        Dense matrix (W, H) or (W, H, D).
    """
    if shape is None:
        shape = np.max(indices, axis=0) + 1
    if len(arr.shape) > 1:
        shape = (*shape, *arr.shape[1:])

    mat = np.full(shape, np.nan)
    mat[indices[:, 0], indices[:, 1]] = arr
    return mat
