import numpy as np
import scipy.spatial.distance as sdist

from numpy.typing import NDArray
from typing import Literal

def _compute_accumulated_cost(
    X: NDArray,
    Y: NDArray,
    metric: Literal['cosine', 'euclidean'] = 'cosine'
) -> NDArray:
    step_sizes = np.array(((1, 1), (0, 1), (1, 0)), dtype=np.int64)
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]

    cost = sdist.cdist(X, Y, metric=metric)
    accum_cost = np.zeros(cost.shape, dtype=np.float64)
    accum_cost[0, 0] = cost[0, 0]

    for i in range(1, cost.shape[0]):
        accum_cost[i, 0] = cost[i, 0] + accum_cost[i-1, 0]
    for i in range(1, cost.shape[1]):
        accum_cost[0, i] = cost[0, i] + accum_cost[0, i-1]


    for i in range(1, cost.shape[0]):
        for j in range(1, cost.shape[1]):
            row_indices = i - row_steps
            col_indices = j - col_steps
            mask = (row_indices >= 0) & (row_indices >= 0)
            prev_minimal = np.min(accum_cost[row_indices[mask], col_indices[mask]])

            accum_cost[i, j] = cost[i, j] + prev_minimal
    return accum_cost

def warp(
    X: NDArray,
    Y: NDArray,
    metric: Literal['cosine', 'euclidean'] = 'cosine',
    *,
    step_sizes: None=None
) -> tuple[NDArray, list[tuple[int, int]]]:
    step_sizes = np.array(((1, 1), (0, 1), (1, 0)), dtype=np.int64)
    row_steps = step_sizes[:, 0]
    col_steps = step_sizes[:, 1]

    accum_cost = _compute_accumulated_cost(X, Y, metric)

    path = []
    row, col = X.shape
    row -= 1
    col -= 1
    while row != 0 or col != 0:
        if row == 0: col -= 1
        elif col == 0: row -= 1
        else:
            row_indices = row - row_steps
            col_indices = col - col_steps
            mask = (row_indices >= 0) & (row_indices >= 0)
            minimal_index = np.argmin(accum_cost[row_indices[mask], col_indices[mask]])

            row -= row_steps[minimal_index]
            col -= col_steps[minimal_index]
        
        path.append((row, col))
    path.reverse()
    return accum_cost, path