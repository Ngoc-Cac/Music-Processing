import numpy as np
import scipy.spatial.distance as sdist

from numpy.typing import NDArray
from typing import Literal

def warp(
    X: NDArray, Y: NDArray,
    metric: Literal['cosine', 'euclidean'] = 'cosine'
) -> tuple[NDArray, NDArray]:
    cost = sdist.cdist(X, Y, metric=metric)
    acum_cost = np.zeros(cost.shape, dtype=np.float64)
    acum_cost[0, 0] = cost[0, 0]

    for i in range(1, cost.shape[0]):
        acum_cost[i, 0] = cost[i, 0] + acum_cost[i-1, 0]
    for i in range(1, cost.shape[1]):
        acum_cost[0, i] = cost[0, i] + acum_cost[0, i-1]

    for i in range(1, cost.shape[0]):
        for j in range(1, cost.shape[1]):
            acum_cost[i, j] = cost[i, j] + min([acum_cost[i - 1, j - 1],
                                                acum_cost[i - 1, j],
                                                acum_cost[i, j - 1]])
            
    path = [(X.shape[0] - 1, Y.shape[0] - 1)]
    while path[-1] != (0, 0):
        if path[-1][0] == 0:
            new_pos = (0, path[-1][1] - 1)
        elif path[-1][1] == 0:
            new_pos = (path[-1][0] - 1, 0)
        else:
            new_pos = min([(acum_cost[path[-1][0] - 1, path[-1][1] - 1], (path[-1][0] - 1, path[-1][1] - 1)),
                           (acum_cost[path[-1][0] - 1, path[-1][1]], (path[-1][0] - 1, path[-1][1])),
                           (acum_cost[path[-1][0], path[-1][1] - 1], (path[-1][0], path[-1][1] - 1))])[1]
        
        path.append(new_pos)
    path.reverse()
    return acum_cost, np.array(path)