import numpy as np
import matplotlib.pyplot as plt


from numpy.typing import NDArray
from numbers import Number
from matplotlib.axes import Axes
from typing import Optional


def spectrogram(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    show_frequency: bool = True,
    *,
    ax: Optional[Axes] = None
) -> Axes:
    if ax is None:
        ax = plt.subplot(111)

    right = (stft_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else stft_coefs.shape[1]
    upper = stft_coefs.shape[0] * sampling_rate / window_length\
            if show_frequency else stft_coefs.shape[0]
    ax.imshow(np.abs(stft_coefs), origin='lower', cmap='Greys', aspect='auto',
              extent=[0, right, 0, upper])
    
    ax.set_ylabel(f'Frequency ({'Hz' if show_frequency else 'index'})')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    return ax