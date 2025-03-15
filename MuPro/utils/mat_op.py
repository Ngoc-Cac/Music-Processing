import numpy as np


from numpy.typing import NDArray


def normalize_column(
    matrix: NDArray,
    threshold: float,
    p_norm: float = 2
):
    """
    Normalize a matrix by column using p-norm.
    """
    unit_vec = np.ones((matrix.shape[0], 1))
    unit_vec /= np.sum(unit_vec) ** (1 / p_norm)
    mags = (np.abs(matrix) ** p_norm).sum(axis=0) ** (1 / p_norm)

    normalized_mat = matrix / mags
    normalized_mat[:, mags < threshold] = unit_vec
    return normalized_mat