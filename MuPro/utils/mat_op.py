import numpy as np
from scipy import signal

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

def smooth_mat(
    matrix: NDArray,
    filter_len: int,
    window: str = 'boxcar'
):
    kernel_func = signal.windows.get_window(window, filter_len)[None, :]
    return signal.convolve(matrix, kernel_func, 'same') / np.sum(kernel_func)

def downsample(
    matrix: NDArray,
    hop_size: int,
    axis: int = 1
):
    if axis == 0:
        return matrix[::hop_size]
    elif axis == 1:
        return matrix[:, ::hop_size]
    else:
        raise ValueError('Weird value for `axis` argument.')