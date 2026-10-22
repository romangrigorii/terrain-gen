"""Downhill routing that stops at the map edge. Water does not wrap around the world."""
import numpy as np

from ..core.grid import neighbor, neighbor_index

NEIGH8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def route(elevation, cell_m):
    """Steepest descent. Receiver is -1 where nothing is strictly downhill."""
    height = np.asarray(elevation, np.float64)
    rows, cols = height.shape
    best = np.zeros(height.shape, np.float64)
    rec = -np.ones(height.shape, np.int32)
    run_of = np.ones(height.shape, np.float64)
    for dy, dx in NEIGH8:
        run = float(np.hypot(dy, dx) * cell_m)
        drop = (height - neighbor(height, dy, dx)) / run
        drop = np.where(np.isnan(drop), -np.inf, drop)
        take = drop > best
        best = np.where(take, drop, best)
        rec = np.where(take, neighbor_index(height.shape, dy, dx), rec)
        run_of = np.where(take, run, run_of)
    slope = np.maximum(best, 0.0)
    return rec.astype(np.int32), slope


def accumulate(receivers, weights):
    """Each cell keeps its own weight, plus everything that flows into it."""
    rec = np.asarray(receivers, np.int32)
    acc = np.asarray(weights, np.float64).copy().ravel()
    flat = rec.ravel()
    flows = flat >= 0
    indeg = np.bincount(flat[flows], minlength=acc.size).astype(np.int32)
    front = np.flatnonzero(indeg == 0)
    stamp = np.empty(acc.size, np.int64)
    while front.size:
        moving = front[flows[front]]
        if moving.size == 0:
            break
        target = flat[moving]
        np.add.at(acc, target, acc[moving])
        np.subtract.at(indeg, target, 1)
        ready = target[indeg[target] == 0]
        pos = np.arange(ready.size)
        stamp[ready] = pos
        front = ready[stamp[ready] == pos]
    return acc.reshape(rec.shape)
