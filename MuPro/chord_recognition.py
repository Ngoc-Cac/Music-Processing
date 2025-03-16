import numpy as np

from numpy.typing import NDArray


_ZERO_EPS = np.finfo(float).eps
def template_similarity(
    features: NDArray,
    chord_template: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    r"""
    Compute the cosine similarity between temporal features and a set of chord templates.
    The result is a similarity matrix and a mask matrix representing the chord labels for each frame.
    Both matrices have their rows representing the index in the template set and their columns representing
    the time frame in the features matrix.

    :param numpy.NDArray features: An `(N, M)` matrix of feature vectors.
    :param numpy.NDArray chord_template: An `(N, C)` matrix of template vectors. The template vectors should
    have the same dimension as feature vectors.
    :return: Two `(C, M)` matrices. The first matrix contains similarity values between each template vector
    and feature vector. The second matrix determines the chord label for each feature vector.
    :rtype: tuple[numpy.NDArray, numpy.NDArray]
    """
    norm_ct = np.sqrt(np.sum(chord_template ** 2, axis=0))
    normalized_ct = chord_template / norm_ct[None, :]
    normalized_ct[:, norm_ct < _ZERO_EPS] = 0

    norm_feat = np.sqrt(np.sum(features ** 2, axis=0))
    normalized_feat = features / norm_feat[None, :]
    normalized_feat[:, norm_feat < _ZERO_EPS] = 0

    chord_sim = normalized_ct.T @ normalized_feat
    return chord_sim, get_chord_labels(chord_sim)

def get_chord_labels(
    similarity_mat: NDArray
) -> NDArray[np.bool]:
    """
    Get the chord labels from a chord similarity matrix. The chord label at each column is the row index
    with highest similarity value.

    :param numpy.NDArray similarity_mat: A chord similarity matrix. Each position `(i, j)` in the matrix
    represent the similarity value between the `i`-th chord and `j`-th frame.
    :return: A mask matrix of the same shape as `similarity_mat`.
    :rtype: numpy.NDArray
    """
    max_indices = similarity_mat.argmax(axis=0)
    chord_results = np.zeros(similarity_mat.shape, dtype=np.bool)
    chord_results[max_indices, range(chord_results.shape[1])] = True
    return chord_results