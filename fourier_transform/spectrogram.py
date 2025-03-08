import numpy as np
import matplotlib.pyplot as plt

from fourier_transform.utils import midi_pitch, convert_to_decibels

from numpy.typing import NDArray
from numbers import Number
from matplotlib.axes import Axes
from typing import Optional


_MIDI_NOTE_RANGE = np.arange(128)

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

    lower = midi_pitch(_MIDI_NOTE_RANGE - 0.5, reference_pitch=reference_pitch)
    upper = midi_pitch(_MIDI_NOTE_RANGE + 0.5, reference_pitch=reference_pitch)
    for i in _MIDI_NOTE_RANGE:
        pitch_i = stft_coefs[(lower[i] <= all_frequencies) & (all_frequencies <= upper[i])]
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
    intensities = np.abs(stft_coefs) ** 2
    if show_decibels:
        intensities = convert_to_decibels(intensities)

    
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
                   .set_label(f"Magnitude {'(dB)' if show_decibels else ''}")
    return ax

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
    lf_coefs = _compute_LF(stft_coefs, sampling_rate, window_length,
                           reference_pitch=reference_pitch)
    if show_decibels:
        intensities = convert_to_decibels(intensities)


    if ax is None:
        ax = plt.subplot(111)

    right = (lf_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else lf_coefs.shape[1]
    img = ax.imshow(lf_coefs, origin='lower', aspect='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, 128])
    
    ax.set_ylabel(f'Frequency (MIDI pitch)')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude {'(dB)' if show_decibels else ''}')
    return ax

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
    lf_coefs = _compute_LF(stft_coefs, sampling_rate, window_length,
                           reference_pitch=reference_pitch)
    chroma_coefs = np.zeros((12, lf_coefs.shape[1]))
    for chroma in range(12):
        chroma_coefs[chroma] = lf_coefs[(_MIDI_NOTE_RANGE % 12) == chroma].sum(axis=0)

    if show_decibels:
        intensities = convert_to_decibels(intensities)


    if ax is None:
        ax = plt.subplot(111)
        
    right = (chroma_coefs.shape[1] * hop_size + window_length) / sampling_rate\
            if show_realtime else chroma_coefs.shape[1]
    img = ax.imshow(chroma_coefs, origin='lower', aspect='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, right, 0, 12])
    
    ax.set_ylabel(f'Chroma')
    ax.set_xlabel(f'Time ({'seconds' if show_realtime else 'frames'})')
    cbar = ax.get_figure().colorbar(img, orientation='vertical')
    cbar.set_label(f'Magnitude {'(dB)' if show_decibels else ''}')
    return ax