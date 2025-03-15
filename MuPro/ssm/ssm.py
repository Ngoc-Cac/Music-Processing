import numpy as np

from numpy.typing import NDArray
from typing import Optional, Iterable, Literal

def threshold_ssm(
    ssm: NDArray,
    threshold: float,
    col_thres: Optional[float] = None,
    *,
    penalty: float = 0,
    relative_thresholding: bool = False,
    rescale_thresholded: bool = False,
    global_thresholding: bool = True
) -> NDArray:
    """
    Thresholding a Self-Similarity Matrix. The threshold can either be a similarity value
    or a percent (value between 0 and 1).

    :param numpy.NDArray ssm: The SSM to threshold.
    :param float threshold: The threshold below which values are discarded. This can be a
        similarity value or a `float` in the range [0, 1] if relative_thresholding is `True`.
    :param float | None col_threhsold: The threshold on columns. This parameter is only used
        if global_thresholding is `False`.
    :param float penalty: The penalty to apply on discarded values. This defautls to 0, suitable
        values should be negative. This penalty is applied AFTER rescaling is done.
    :param bool relative_thresholding: Whether or relative or absolute threshold is used. Absolute
        threshold applies thresholding on values as raw. Relative threshold uses the threshold as
        percentile where only values above said percentiles is kept.
        
        For example, if `threshold=0.2`, relative thresholding will keep the top 20% highest cells.
        This parameter defaults to `False`.
    :param bool rescale_thresholding: Whether or not to rescale the SSM after thresholding.
        This rescales all thersholded values to the range [0, 1]. This defaults to `False`.
    :param bool global_thresholding: Whether to apply global or local thresholding. Global
        thresholding apply a constant threshold for all cells. Local thresholding use a threshold
        on the rows and another threshold on the columns. If `col_threshold` is not specified,
        `threshold` is used for both row and column.
    :return: the threhsolded matrix of the same dimension
    :rtype: numpy.NDArray
    """
    if relative_thresholding:
        threshold = np.percentile(ssm, 1-threshold, axis=None if global_thresholding else 0)
        if not col_thres is None:
            col_thres = np.percentile(ssm, 1-col_thres, axis=None if global_thresholding else 0)
    thresholded = np.array(ssm)

    if global_thresholding:
        thresholded[ssm <= threshold] = 0
    else:
        row_threshold = np.tile(threshold, (ssm.shape[0], 1))
        col_threshold = row_threshold.T if col_thres is None else np.tile(col_thres, (1, ssm.shape[0]))
        mask = ssm > row_threshold
        mask = (mask & (ssm > col_threshold))
        thresholded[mask] = 0

    if rescale_thresholded:
        minimum = np.min(thresholded[thresholded != 0])
        thresholded = (thresholded - minimum) / (np.max(thresholded) - minimum)
    thresholded[thresholded <= 0] = penalty

    return thresholded


def get_induced_segments(
    paths: Iterable[tuple[int, int]]
) -> NDArray[np.int64]:
    r"""
    Get the induced segments from an Iterable of paths. The induced segment of a path is
    the starting row and ending row position of the path. In the case the paths form a 
    path family, this is equivalent to computing the induced segment family.

    :param Iterable[tuple[int, int]] paths: The paths to compute the induced segments on.
    :return: A `numpy.NDArray` of shape `(N, 2)`. Each row of the array represents `(start_row, end_row)`
    :rtype: NDArray[numpy.int64]
    """
    induced_segment_fam = np.zeros((len(paths), 2), dtype=np.int64)
    for i, path in enumerate(paths):
        induced_segment_fam[i] = path[0][0], path[-1][0]
    return induced_segment_fam

def compute_score(
    segmented_ssm: NDArray,
    *,
    step_sizes: None=None
) -> tuple[NDArray, np.float64]:
    """
    Compute the score of a segment. The score of a segment is the score of its
    optimal path family.

    :param NDArray segmented_ssm: The segment in interest of the SSM
    :param None step_sizes: This currently do nothing, future feature!
    :return: The accumulated score matrix and the score
    :rtype: tuple[NDArray, numpy.float64]
    """
    # if step_sizes is None:
    #     step_sizes = ((1, 1), (1, 2), (2, 1))

    # step_sizes = np.array(step_sizes)
    step_sizes = np.array(((1, 1), (1, 2), (2, 1)))
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]


    dimension = (segmented_ssm.shape[0], segmented_ssm.shape[1] + 1)
    accum_score = -np.ones(dimension, dtype=np.float64) * np.inf
    accum_score[0, 0] = 0
    accum_score[0, 1] = segmented_ssm[0, 0]

    for i in range(1, accum_score.shape[0]):
        accum_score[i, 0] = np.max(accum_score[i - 1, [0, -1]])
        accum_score[i, 1] = accum_score[i, 0] + segmented_ssm[i, 0]
        for j in range(2, accum_score.shape[1]):
            row_indices = i - row_steps
            col_indices = j - col_steps
            mask = (row_indices >= 0) & (col_indices >= 0)
            maximal_prev = np.max(accum_score[row_indices[mask], col_indices[mask]])

            accum_score[i, j] = segmented_ssm[i, j - 1] + maximal_prev

    return accum_score, np.max(accum_score[-1, [0, -1]])

def get_optimal_path(
    accum_score: NDArray,
) -> list[list[tuple[int, int]]]:
    """
    Get the optimal path family with backtracking from an accumulated score matrix.

    :param numpy.NDArray accum_score: The accumulated score matrix
    :return: A `list` of paths, each path is another `list` containing `tuple` reprepresenting
        coordinates of the matrix.
    :rtype: list[list[tuple[int, int]]]
    """
    # Noting this so I don't forget in the future:
    # The backtracking is a bit unintuitive. The step_sizes is for generalization in the future if
    # I decide to actually do it. For now, only step_sizes allowed are as below. Interpretation for the
    # accum_score matrix is also mentioned in the function above.
    # Now, the way the backtracking works is it starts at the phantom row N of the accum_score matrix
    # in the 'elevator' column (0th col). Everytime we are at the elevator, we always move down a level.
    # However, the column at which we arrive is either 0 or (M - 1) depending on which cell is larger.
    # Essentially, we are either choosing to: 1-move down a level (arriving at col 0) or
    # 2-start a new path (arriving at col M - 1). When a new path has been started, we want to create a new
    # list and append it to the family straight away, we can do this because we only reference back to the
    # list to append new coordinates.
    # Now onto how to move along the path, this is exactly like backtracking for path alignment in DTW.
    # However, there is a mechanism for checking the appropriate steps, this is again a generalizing mechanism.
    # When we finally reach the column 1, this is equivalent to the column 0 in the SSM. So we end our path and
    # move to the elevator column.
    max_col = accum_score.shape[1] - 1

    step_sizes = np.array(((1, 1), (1, 2), (2, 1)))
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]

    path_family = []
    cur_path = []
    row, col = accum_score.shape[0], 0
    while row != 0 or col != 0:
        if row == 0 or col == 1:
            col -= 1
        elif col == 0:
            row -= 1
            if accum_score[row, 0] <= accum_score[row, max_col]:
                col = max_col
                
                cur_path.reverse()
                cur_path = [(row, col - 1)]
                path_family.append(cur_path)
        else:
            row_indices = row - row_steps
            col_indices = col - col_steps
            mask = (row_indices >= 0) & (col_indices >= 0)

            step_index = np.argmax(accum_score[row_indices[mask], col_indices[mask]])
            row -= row_steps[step_index]
            col -= col_steps[step_index]

            cur_path.append((row, col - 1))

    cur_path.reverse()
    path_family.reverse() # this is optional, just re-ordering the paths in increasing row order
    return path_family

def compute_fitness(
    ssm: NDArray,
    segment: tuple[int, int]
) -> dict[Literal['score', 'score_matrix', 'normalized_score', 'normalized_coverage', 'fitness']]:
    """
    Compute the fitness for a segment. This also computes and return every other related metrics:
    - The score of the segment
    - Normalized Score of the segment
    - Normalized coverage of the segment

    The score of a segment is essentially the total score for its optimal path family. In terms of
    the SSM, this score determines whether or not the segment in interest very similar to the whole
    matrix itself.

    The normalized score then compensates for self-similarity and the segment's length.

    The normalized coverage also compensates for the segment's length. Note that coverage of a segment
    is actually the total length of the induced segment family from the optimal path family.

    :param numpy.NDArray ssm: The self-similarity matrix to compute on.
    :param tuple[int, int] segment: The segment in iterest, represented in terms of `(start_point, end_point)`.
    :return: A dictionary of keys: `'score'`, `'score_matrix'`, `'normalized_score'`, `'normalized_coverage'`
        and `'fitness'`.
    :rtype: dict
    """
    segmented_ssm = ssm[:, segment[0]:(segment[1] + 1)]
    score_mat, score = compute_score(segmented_ssm)
    path_family = get_optimal_path(score_mat)

    normalized_score = (score - segmented_ssm.shape[1]) / sum(len(path) for path in path_family)

    induced_family = get_induced_segments(path_family)
    coverage = np.sum(induced_family[:, 1] - induced_family[:, 0])
    normalized_cov = (coverage - segmented_ssm.shape[1]) / ssm.shape[0]

    fitness = 2 * normalized_cov * normalized_score / (normalized_score + normalized_cov)
    return {
        'score': score,
        'score_matrix': score_mat,
        'normalized_score': normalized_score,
        'normalized_coverage': normalized_cov,
        'fitness': fitness
    }