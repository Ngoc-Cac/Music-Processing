import numpy as np

from numbers import Number
from numpy.typing import NDArray

def midi_pitch(
    note_number: Number,
    *,
    reference_pitch: Number = 440,
    reference_note: int = 69
):
    return 2 ** ((note_number - reference_note)/ 12) * reference_pitch

def convert_to_decibels(intensities: NDArray, *, threshold: float = -12):
    """
    threshold should be powers of 10
    """
    # eps is to avoid division by 0 if intensities = 0
    return 10 * (np.log10(intensities + np.finfo(float).eps) - threshold)

def logarithmic_compress(values: NDArray, gamma: float):
    return np.log(1 + gamma * values)