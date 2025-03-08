import numpy as np

from numbers import Number

def midi_pitch(
    note_number: Number,
    *,
    reference_pitch: Number = 440,
    reference_note: int = 69
):
    return 2 ** ((note_number - reference_note)/ 12) * reference_pitch

def init_dft_mat(dimension: int):
    if not isinstance(dimension, int):
        raise TypeError("'dimension' must be a positive integer")
    elif dimension < 1:
        raise TypeError("'dimension' must be positive")
    
    dft_matrix = np.ndarray((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            dft_matrix[i, j] = np.exp(-2j * (np.pi / dimension) * i * j)
    return dft_matrix

def init_twiddle_factors(dimension: int):
    return np.fromiter((np.exp(-2j * (np.pi / dimension) * i) for i in range(dimension // 2)), dtype=np.complex128)