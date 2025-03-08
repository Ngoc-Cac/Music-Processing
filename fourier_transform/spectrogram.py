import numpy as np
import matplotlib.pyplot as plt

from fourier_transform.utils import midi_pitch

from numpy.typing import NDArray
from numbers import Number
from matplotlib.axes import Axes
from typing import Optional


def _compute_LF(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    *,
    reference_pitch: Number = 440,
) -> NDArray[np.float64]:
    stft_coefs = np.abs(stft_coefs) ** 2
    lf_coefs = np.zeros((128, stft_coefs.shape[1]), dtype=np.float64)
    all_frequencies = np.arange(stft_coefs.shape[0]) * sampling_rate / window_length

    for i in range(128):
        lower = midi_pitch(i - 0.5, reference_pitch=reference_pitch)
        upper = midi_pitch(i + 0.5, reference_pitch=reference_pitch)
        pitch_i = stft_coefs[(lower <= all_frequencies) & (all_frequencies <= upper)]
        lf_coefs[i] = pitch_i.sum(axis=0)
    return lf_coefs

def spectrogram(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    show_frequency: bool = True,
    show_decibels: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Axes] = None
) -> Axes:
    if ax is None:
        ax = plt.subplot(111)
    intensities = np.abs(stft_coefs) ** 2
    if show_decibels:
        intensities = 10 * (np.log10(intensities + np.finfo(float).eps))

    right = (intensities.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else intensities.shape[1]
    upper = intensities.shape[0] * sampling_rate / window_length\
            if show_frequency else intensities.shape[0]
    img = ax.imshow(intensities, origin='lower', cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, upper])
    
    ax.set_ylabel(f'Frequency ({'Hz' if show_frequency else 'index'})')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    ax.get_figure().colorbar(img, orientation='vertical')\
                   .set_label(f"Magnitude {'(dB)' if show_decibels else ''}")
    return ax, img

def LF_spectrogram(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    show_decibels: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    reference_pitch: Number = 440,
    ax: Optional[Axes] = None
) -> Axes:
    if ax is None:
        ax = plt.gca()

    lf_coefs = _compute_LF(stft_coefs, sampling_rate, window_length,
                           reference_pitch=reference_pitch)
    if show_decibels:
        # + eps to avoid division by zero
        lf_coefs = 10 * (np.log10(lf_coefs + np.finfo(float).eps))

    right = (lf_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else lf_coefs.shape[1]
    img = ax.imshow(lf_coefs, origin='lower', cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, 128])
    
    ax.set_ylabel(f'Frequency (MIDI pitch)')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude {'(dB)' if show_decibels else ''}')
    return ax, img

def chromagram(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    hop_size: int,
    show_realtime: bool = True,
    show_decibels: bool = True,
    *,
    cmap: str = 'gray_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    reference_pitch: Number = 440,
    ax: Optional[Axes] = None
) -> Axes:
    if ax is None:
        ax = plt.gca()

    lf_coefs = _compute_LF(stft_coefs, sampling_rate, window_length,
                           reference_pitch=reference_pitch)
    
    midi_range = np.arange(128)
    chroma_coefs = np.zeros((12, lf_coefs.shape[1]))
    for chroma in range(12):
        chroma_coefs[chroma] = lf_coefs[(midi_range % 12) == chroma].sum(axis=0)

    if show_decibels:
        # + eps to avoid division by zero
        chroma_coefs = 10 * (np.log10(chroma_coefs + np.finfo(float).eps))

    right = (chroma_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else chroma_coefs.shape[1]
    img = ax.imshow(chroma_coefs, origin='lower', cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, 12])
    
    ax.set_ylabel(f'Chroma')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude {'(dB)' if show_decibels else ''}')
    return ax, img