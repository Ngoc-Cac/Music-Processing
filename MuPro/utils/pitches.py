import numpy as np

from numbers import Number
from numpy.typing import NDArray


_MIDI_NOTE_RANGE = np.arange(128)
_NOTE_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_pitch(
    note_number: Number,
    *,
    reference_pitch: Number = 440,
    reference_note: int = 69
):
    return 2 ** ((note_number - reference_note)/ 12) * reference_pitch


def to_LF(
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

def to_chroma(
    stft_coefs: NDArray[np.complex128],
    sampling_rate: Number,
    window_length: int,
    *,
    reference_pitch: Number = 440,
) -> NDArray[np.float64]:
    lf_coefs = to_LF(stft_coefs, sampling_rate, window_length,
                     reference_pitch=reference_pitch)
    chroma_coefs = np.zeros((12, lf_coefs.shape[1]))
    for chroma in range(12):
        chroma_coefs[chroma] = lf_coefs[(_MIDI_NOTE_RANGE % 12) == chroma].sum(axis=0)
    return chroma_coefs


def convert_to_decibels(intensities: NDArray, *, threshold: float = -12):
    """
    threshold should be powers of 10
    """
    # eps is to avoid division by 0 if intensities = 0
    return 10 * (np.log10(intensities + np.finfo(float).eps) - threshold)

def logarithmic_compress(values: NDArray, gamma: float):
    return np.log(1 + gamma * values)


def tranpose_chromas(chroma_coefs: NDArray, semitones: int):
    return np.roll(chroma_coefs, semitones % 12, axis=0)