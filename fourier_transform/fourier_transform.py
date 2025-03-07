import numpy as np

from fourier_transform.utils import init_dft_mat, init_twiddle_factors

from numpy.typing import NDArray
from typing import Callable

def dft(signal: NDArray[np.complex128]):
    dft_mat = init_dft_mat(signal.shape[0])
    return dft_mat @ signal


def fft(signal: NDArray[np.complex128]):
    log_2N = np.log2(signal.shape[0])
    if not log_2N.is_integer():
        raise ValueError('Fast Fourier Transform can only be applied on signal length of powers of two')
    return _inner_fft(signal)
    
def _inner_fft(signal: NDArray[np.complex128]):
    dim = signal.shape[0]
    if dim == 1:
        return signal[0:1]
    
    A_mat = _inner_fft(signal[:dim-1:2])
    B_mat = _inner_fft(signal[1:dim:2])
    C_mat = init_twiddle_factors(dim) * B_mat

    return np.concatenate([A_mat + C_mat, A_mat-C_mat])

def stft(
    signal: NDArray[np.complex128],
    window: NDArray[np.complex128],
    hop_size: int
):
    if not isinstance(hop_size, int):
        raise TypeError("'hope_size' must be a positive integer")
    elif hop_size < 1:
        raise ValueError("'hop_size' must be positive")
    
    window_length = window.shape[0]
    if np.log2(window_length).is_integer():
        fourier_transform = fft
    else:
        fourier_transform = dft

    coefs = []
    frame_index = -1
    while (frame_index := frame_index + 1) * hop_size + window_length < signal.shape[0]:
        windowed_sig = signal[frame_index * hop_size:window_length + frame_index * hop_size] * np.conjugate(window)
        coefs.append(fourier_transform(windowed_sig))

    return np.array(coefs).T[::-1]