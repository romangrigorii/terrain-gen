"""Waves cut a shore and leave part of the debris as a beach."""
import numpy as np

from ..core.grid import neighbor, neighbor_index
from ..core.pipeline import Stage
from ..core.registry import register

NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def shore_mask(elevation, open_edge=True):
    """Land with sea beside it.

    Cells outside the map count as sea when `open_edge` is set. A stamp on one
    tile leaves the edge closed so an inland border is not treated as ocean.
    """
    height = np.asarray(elevation, np.float64)
    land = height > 0.0
    exposed = np.zeros(height.shape, dtype=bool)
    for dy, dx in NEIGH4:
        other = neighbor(height, dy, dx)
        outside = np.isnan(other)
        sea = (other <= 0.0) & ~outside
        exposed |= (outside | sea) if open_edge else sea
    return land & exposed


def retreat(state, k=2.0, beach_fraction=0.4, where=None, open_edge=True):
    """Lower shore cells. Soft rock retreats faster. Some of the cut becomes sand.

    `where` limits the cut to cells the caller painted. The rest of a removed
    column leaves the map and is counted in `offshore_thickness_sum`.
    """
    height = state.elevation
    shore = shore_mask(height, open_edge=open_edge)
    if where is not None:
        shore &= np.asarray(where, dtype=bool)
    hardness = np.clip(state.get("hardness").astype(np.float64), 0.0, 1.0)
    cut = np.where(shore, float(k) * (1.0 - 0.75 * hardness), 0.0)
    cut = np.minimum(cut, np.maximum(height, 0.0))
    removed = state.peel(cut)
    beach = removed * float(beach_fraction)
    on_cell = beach * 0.5
    state.deposit("sand_m", on_cell)
    give = np.zeros(height.shape, np.float64)
    best = np.full(height.shape, np.inf)
    dest = -np.ones(height.shape, np.int32)
    for dy, dx in NEIGH4:
        other = neighbor(height, dy, dx)
        index = neighbor_index(height.shape, dy, dx)
        take = ~np.isnan(other) & (other < best)
        best = np.where(take, other, best)
        dest = np.where(take, index, dest)
    payload = np.where((dest >= 0) & shore, beach * 0.5, 0.0)
    flows = (dest >= 0).ravel()
    np.add.at(give.ravel(), dest.ravel()[flows], payload.ravel()[flows])
    state.deposit("sand_m", give)
    exported = float((removed - on_cell - payload).sum())
    total = float(state.features.get("offshore_thickness_sum", 0.0)) + exported
    state.features["offshore_thickness_sum"] = total
    return removed


@register
class Coastal(Stage):
    name = "coastal"
    requires = ("bedrock_m", "hardness")
    writes = ("sand_m",)

    def __init__(self, k=2.0):
        self.k = float(k)

    def run(self, world, ctx):
        retreat(world, k=self.k)
