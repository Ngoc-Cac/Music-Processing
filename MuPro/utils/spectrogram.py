import numpy as np
import matplotlib.pyplot as plt

from MuPro.utils import _NOTE_LABELS

from numpy.typing import NDArray
from numbers import Number
from matplotlib.axes import Axes
from typing import Optional


def spectrogram(
    intensities: NDArray[np.float64],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    show_frequency: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Axes] = None
) -> Axes:    
    if ax is None:
        ax = plt.subplot(111)

    right = (intensities.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else intensities.shape[1]
    upper = intensities.shape[0] * sampling_rate / window_length\
            if show_frequency else intensities.shape[0]
    img = ax.imshow(intensities, origin='lower', aspect='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, upper])
    
    ax.set_ylabel(f'Frequency ({'Hz' if show_frequency else 'index'})')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    ax.get_figure().colorbar(img, orientation='vertical')\
                   .set_label(f"Magnitude")
    return ax

def LF_spectrogram(
    lf_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Axes] = None
) -> tuple[Axes, NDArray]:
    if ax is None:
        ax = plt.subplot(111)

    right = (lf_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else lf_coefs.shape[1]
    img = ax.imshow(lf_coefs, origin='lower', aspect='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, -.5, 127.5])
    
    ax.set_ylabel(f'Frequency (MIDI pitch)')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude')
    return ax

def chromagram(
    chroma_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Axes] = None
) -> tuple[Axes, NDArray]:
    if ax is None:
        ax = plt.subplot(111)
        
    right = (chroma_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else chroma_coefs.shape[1]
    img = ax.imshow(chroma_coefs, origin='lower', aspect='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0.5, 12.5])
    
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    ax.set_ylabel(f'Chroma')
    ax.set_yticks(np.arange(1, 13), _NOTE_LABELS)
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude')
    return ax