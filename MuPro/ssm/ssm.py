import numpy as np

from numpy.typing import NDArray
from typing import Optional

def generate_ssm(
    chroma_features: NDArray,
    window_length: Optional[int] = None,
    tempo_deviations: Optional[list[float]] = None,
):
    ssm = chroma_features.T @ chroma_features
    if not window_length is None:
        ssm = smooth_ssm(ssm, window_length, tempo_deviations)
        # ssm = np.max([forward_smoothed, backward_smoothed], axis=0)
    return ssm

def smooth_ssm(
    ssm: NDArray, window_length: int,
    tempo_deviations: Optional[list[float]] = None,
):
    dim = ssm.shape[0]
    if tempo_deviations is None:
        tempo_deviations = [1]

    smoothed_ssm = np.zeros((len(tempo_deviations), dim, dim), dtype=np.float64)

    col_pad = int(np.rint(window_length * np.max(tempo_deviations)))
    ssm = np.pad(ssm, ((window_length, 0), (col_pad, 0)))
    for i, tempo_deviation in enumerate(tempo_deviations):
        for row in range(window_length):
            col = int(np.rint(tempo_deviation * row))
            smoothed_ssm[i] += ssm[row:(row + dim), col:(col + dim)]
    return np.max(smoothed_ssm / window_length, axis=0)