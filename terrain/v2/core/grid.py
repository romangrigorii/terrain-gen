"""Cell geometry for the layered world. Distinct from `terrain.grid`."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    shape: tuple
    cell_m: float
    origin_m: tuple = (0.0, 0.0)

    def __post_init__(self):
        rows, cols = self.shape
        if rows < 2 or cols < 2:
            raise ValueError("grid must be at least 2x2")
        if not np.isfinite(self.cell_m) or self.cell_m <= 0:
            raise ValueError("cell_m must be positive")

    @property
    def rows(self):
        return int(self.shape[0])

    @property
    def cols(self):
        return int(self.shape[1])

    @property
    def cell_area_m2(self):
        return float(self.cell_m) * float(self.cell_m)

    def coordinates(self):
        """Cell-centre coordinates in metres, as (y, x) arrays."""
        y0, x0 = self.origin_m
        y = y0 + (np.arange(self.rows) + 0.5) * self.cell_m
        x = x0 + (np.arange(self.cols) + 0.5) * self.cell_m
        return np.meshgrid(y, x, indexing="ij")


def neighbor(array, dy, dx):
    """Sample `array[y+dy, x+dx]`. Cells with no such neighbour are NaN."""
    rows, cols = array.shape
    out = np.full(array.shape, np.nan, np.float64)
    dst_y = slice(max(-dy, 0), rows - max(dy, 0))
    dst_x = slice(max(-dx, 0), cols - max(dx, 0))
    src_y = slice(max(dy, 0), rows - max(-dy, 0))
    src_x = slice(max(dx, 0), cols - max(-dx, 0))
    out[dst_y, dst_x] = np.asarray(array, np.float64)[src_y, src_x]
    return out


def neighbor_index(shape, dy, dx):
    """Flat index of each cell's neighbour, or -1 where it would leave the grid."""
    rows, cols = shape
    yy = np.arange(rows)[:, None] + dy
    xx = np.arange(cols)[None, :] + dx
    valid = (yy >= 0) & (yy < rows) & (xx >= 0) & (xx < cols)
    return np.where(valid, yy * cols + xx, -1).astype(np.int32)
