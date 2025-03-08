import numpy as np


from numpy.typing import NDArray


def _init_dft_mat(dimension: int) -> NDArray[np.complex128]:
    xi, yi = np.mgrid[:dimension,:dimension]
    return np.exp(-2j * (np.pi / dimension) * xi * yi)

def _init_twiddle_factors(dimension: int) -> NDArray[np.complex128]:
    # return np.fromiter((np.exp(-2j * (np.pi / dimension) * i) for i in range(dimension // 2)), dtype=np.complex128)
    return np.exp(-2j * (np.pi / dimension) * np.arange(dimension // 2, dtype=np.complex128))


def dft(signal: NDArray[np.complex128]):
    dft_mat = _init_dft_mat(signal.shape[0])
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
    C_mat = _init_twiddle_factors(dim) * B_mat

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
    
    frames = np.floor((signal.shape[0] - window_length) / hop_size) + 1
    window_length = window.shape[0]
    if np.log2(window_length).is_integer():
        fourier_transform = fft
    else:
        fourier_transform = dft

    
    mask_idx = hop_size * np.arange(frames, dtype=np.int64)[:, None] +\
               np.arange(window_length, dtype=np.int64)
    slided_signal = signal[mask_idx].T
    windowed_sig_by_frame = np.conjugate(window.reshape((window_length, 1))) * slided_signal

    return np.apply_along_axis(fourier_transform, 0, windowed_sig_by_frame)