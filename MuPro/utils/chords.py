import numpy as np

from copy import deepcopy

from MuPro.utils import tranpose_chromas


from numpy.typing import NDArray


_BINARY_CHROMA_TEMPLATE = [np.zeros((12, 1))]
# Major chords
_BINARY_CHROMA_TEMPLATE[0][[0, 4, 7]] = 1
for _ in range(11):
    _BINARY_CHROMA_TEMPLATE.append(tranpose_chromas(_BINARY_CHROMA_TEMPLATE[-1], 1))
# Minor chords
_BINARY_CHROMA_TEMPLATE.append(np.zeros((12, 1)))
_BINARY_CHROMA_TEMPLATE[-1][[0, 3, 7]] = 1
for _ in range(11):
    _BINARY_CHROMA_TEMPLATE.append(tranpose_chromas(_BINARY_CHROMA_TEMPLATE[-1], 1))


def get_chord_template(
    chroma: bool = True,
    binary: bool = True
) -> list[NDArray[np.float64]]:
    if chroma:
        return deepcopy(_BINARY_CHROMA_TEMPLATE)