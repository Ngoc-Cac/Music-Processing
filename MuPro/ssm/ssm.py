import numpy as np

from numpy.typing import NDArray
from typing import Optional, Iterable

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
    thresholded = ssm

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

def compute_score(
    segmented_ssm: NDArray,
    *,
    step_sizes: Optional[Iterable[tuple[int, int]]] = None
):
    # if step_sizes is None:
    #     step_sizes = ((1, 1), (1, 2), (2, 1))

    # step_sizes = np.array(step_sizes)
    step_sizes = np.array(((1, 1), (1, 2), (2, 1)))
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]


    accum_score = np.zeros(segmented_ssm.shape)
    accum_score[0, 1] = segmented_ssm[0, 0]
    accum_score[0, 2:] = -np.inf

    for i in range(1, accum_score.shape[0]):
        accum_score[i, 0] = np.max(accum_score[i - 1, [0, -1]])
        accum_score[i, 1] = accum_score[i, 0] + segmented_ssm[i, 0]
        for j in range(2, accum_score.shape[1]):
            accum_score[i, j] = segmented_ssm[i, j] + np.max(accum_score[i - row_steps, j - col_steps])

    return accum_score, np.max(accum_score[-1, [0, -1]])