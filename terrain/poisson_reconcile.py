"""Keep a local height field's slopes when its border has to match a neighbor.

The discrete system is Poisson image editing on a height field: interior
gradients stay those of the local surface, and cells outside the free mask
keep the shared height. It is a local join, not a substitute for a global
ridge or river.
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def reconcile(source, fixed, free):
    """Solve for `free` cells. Everywhere else the result is `fixed`."""
    source = np.asarray(source, np.float64)
    fixed = np.asarray(fixed, np.float64)
    free = np.asarray(free, bool)
    if source.shape != fixed.shape or source.shape != free.shape:
        raise ValueError("source, fixed, and free must share a shape")
    if source.ndim != 2:
        raise ValueError("reconcile expects a 2D height field")
    rows, cols = source.shape
    sites = np.argwhere(free)
    count = int(sites.shape[0])
    if count == 0:
        return np.asarray(fixed, np.float32)
    index = -np.ones((rows, cols), np.int32)
    index[free] = np.arange(count, dtype=np.int32)
    matrix = lil_matrix((count, count), dtype=np.float64)
    rhs = np.zeros(count, np.float64)
    for k, (y, x) in enumerate(sites):
        diag = 0.0
        guide = 0.0
        shared = 0.0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            diag += 1.0
            guide += source[y, x] - source[ny, nx]
            other = int(index[ny, nx])
            if other >= 0:
                matrix[k, other] = -1.0
            else:
                shared += fixed[ny, nx]
        matrix[k, k] = diag
        rhs[k] = guide + shared
    solved = spsolve(matrix.tocsr(), rhs)
    out = np.array(fixed, np.float64, copy=True)
    out[free] = np.asarray(solved, np.float64).reshape(-1)
    return out.astype(np.float32)
