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

    ## Parameters:
        ssm: `NDArray`
            The SSM to threshold
        threshold: `float`
            The threshold below which values are discarded. This can be a similarity value or
            a float in the range [0, 1] if relative_thresholding is `True`.
        col_threshold: `float | None`
            The threshold on columns. This parameter is only used if global_thresholding is `False`.
        penalty: `float`
            The penalty to apply on discarded values. This defautls to 0, suitable values should
            be negative. This penalty is applied AFTER rescaling is done.
        relative_thresholding: `bool`
            Whether or relative or absolute threshold is used. Absolute threshold applies thresholding
            on values as raw. Relative threshold uses the threshold as percentile where only values
            above said percentiles is kept. For example, if `threshold=0.5`, relative thresholding
            will keep the top 50% highest cells. This parameter defaults to `False`.
        rescale_thresholded: `bool`
            Whether or not to rescale the SSM after thresholding. This rescales all thersholded values
            to the range [0, 1]. This defaults to `False`.
        global_thresholding: `bool`
            Whether to apply global or local thresholding. Global thresholding apply a constant threshold
            for all cells. Local thresholding use a threshold on the rows and another threshold on the columns.
            If `col_threshold` is not specified, `threshold` is used for both row and column.
    """
    if relative_thresholding:
        threshold = np.percentile(ssm, threshold, axis=None if global_thresholding else 0)
        if not col_thres is None:
            col_thres = np.percentile(ssm, col_thres, axis=None if global_thresholding else 0)
    thresholded = np.array(ssm)

    if global_thresholding:
        thresholded[ssm <= threshold] = 0
    else:
        row_threshold = np.tile(threshold, (ssm.shape[0], 1))
        col_threshold = row_threshold.T if col_thres is None else np.tile(col_thres, (1, ssm.shape[0]))
        mask = ssm > row_threshold
        mask = (mask & (ssm > col_threshold))
        thresholded = 0

    if rescale_thresholded:
        minimum = np.min(thresholded[thresholded != 0])
        thresholded = (thresholded - minimum) / (np.max(thresholded) - minimum)
    thresholded[thresholded <= 0] = penalty

    return thresholded


def get_induced_segments(
    paths: Iterable[tuple[int, int]]
):
    induced_segment_fam = np.zeros((len(paths), 2), dtype=np.int64)
    for i, path in enumerate(paths):
        induced_segment_fam[i] = path[0][0], path[-1][0]
    return induced_segment_fam

def compute_score(
    segmented_ssm: NDArray,
) -> tuple[NDArray, np.number]:
    # if step_sizes is None:
    #     step_sizes = ((1, 1), (1, 2), (2, 1))

    # step_sizes = np.array(step_sizes)
    step_sizes = np.array(((1, 1), (1, 2), (2, 1)))
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]


    accum_score = -np.ones((segmented_ssm.shape[0], segmented_ssm.shape[1] + 1)) * np.inf
    accum_score[0, 0] = 0
    accum_score[0, 1] = segmented_ssm[0, 0]

    for i in range(1, accum_score.shape[0]):
        accum_score[i, 0] = np.max(accum_score[i - 1, [0, -1]])
        accum_score[i, 1] = accum_score[i, 0] + segmented_ssm[i, 0]
        for j in range(2, accum_score.shape[1]):
            accum_score[i, j] = segmented_ssm[i, j - 1] + np.max(accum_score[i - row_steps, j - col_steps])

    return accum_score, np.max(accum_score[-1, [0, -1]])

def get_optimal_path(
    accum_score: NDArray,
) -> list[tuple[int, int]]:
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
    path_family.reverse()
    return path_family

def compute_fitness(
    ssm: NDArray,
    segment: tuple[int, int]
) -> dict[Literal['score', 'score_matrix', 'normalized_score', 'normalized_coverage', 'fitness']]:
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