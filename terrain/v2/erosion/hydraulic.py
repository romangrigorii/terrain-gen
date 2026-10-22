"""One storm step of shallow water. Outflow cannot exceed the water in the cell."""
import numpy as np

from ..core.grid import neighbor, neighbor_index
from ..core.pipeline import Stage
from ..core.registry import register

NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _transfers(elevation, water, rain_m):
    """Water depth after rain, and the depth each cell offers to each neighbour.

    A missing neighbour is outside the grid. Outflow is scaled so a cell cannot
    give away more water than it holds.
    """
    depth = np.maximum(np.asarray(water, np.float64), 0.0) + np.maximum(
        np.asarray(rain_m, np.float64), 0.0)
    surface = np.asarray(elevation, np.float64) + depth
    offered = []
    for dy, dx in NEIGH4:
        other = neighbor(surface, dy, dx)
        head = np.where(np.isnan(other), 0.0, surface - other)
        offered.append(np.maximum(head, 0.0) * 0.25)
    stack = np.stack(offered, axis=0)
    demand = stack.sum(axis=0)
    scale = np.ones(depth.shape, np.float64)
    wet = demand > 1e-12
    scale[wet] = np.minimum(1.0, depth[wet] / demand[wet])
    return depth, [offer * scale for offer in offered]


def _apply(depth, takes):
    arrived = np.zeros(depth.shape, np.float64)
    left = np.zeros(depth.shape, np.float64)
    for take, (dy, dx) in zip(takes, NEIGH4):
        dest = neighbor_index(depth.shape, dy, dx)
        flows = (dest >= 0).ravel()
        np.add.at(arrived.ravel(), dest.ravel()[flows], take.ravel()[flows])
        left += take
    return depth - left + arrived


def step_water(elevation, water, rain_m):
    """Redistribute a water column toward lower water surfaces.

    Rain is added first. The sum of water depths then stays equal to the
    previous sum plus the rain, and no cell goes dry below zero.
    """
    depth, takes = _transfers(elevation, water, rain_m)
    return _apply(depth, takes)


def tiled_step(elevation, water, rain_m, tile_h, tile_w):
    """Same redistribution, with each tile reading only a one-cell halo.

    Flux that crosses a tile edge is the offer computed from that halo, then
    applied on the whole grid. The result matches one step on the unsplit map.
    """
    elevation = np.asarray(elevation, np.float64)
    water = np.asarray(water, np.float64)
    rain = np.asarray(rain_m, np.float64)
    if tile_h < 2 or tile_w < 2:
        raise ValueError("tiles must be at least 2 cells on a side")
    rows, cols = elevation.shape
    stitched = [np.zeros((rows, cols), np.float64) for _ in NEIGH4]
    for y0 in range(0, rows, int(tile_h)):
        for x0 in range(0, cols, int(tile_w)):
            y1 = min(rows, y0 + int(tile_h))
            x1 = min(cols, x0 + int(tile_w))
            hy0, hy1 = max(0, y0 - 1), min(rows, y1 + 1)
            hx0, hx1 = max(0, x0 - 1), min(cols, x1 + 1)
            _, local = _transfers(
                elevation[hy0:hy1, hx0:hx1],
                water[hy0:hy1, hx0:hx1],
                rain[hy0:hy1, hx0:hx1],
            )
            oy, ox = y0 - hy0, x0 - hx0
            th, tw = y1 - y0, x1 - x0
            for band, take in zip(stitched, local):
                band[y0:y1, x0:x1] = take[oy:oy + th, ox:ox + tw]
    depth = np.maximum(water, 0.0) + np.maximum(rain, 0.0)
    return _apply(depth, stitched)


@register
class Hydraulic(Stage):
    name = "hydraulic"
    requires = ("rainfall_mm_day", "bedrock_m")
    writes = ("water_m",)

    def __init__(self, storm_hours=6.0):
        self.storm_hours = float(storm_hours)

    def run(self, world, ctx):
        rain_m = (world.get("rainfall_mm_day").astype(np.float64)
                  * (self.storm_hours / 24.0) / 1000.0)
        updated = step_water(world.elevation, world.get("water_m"), rain_m)
        world.set("water_m", updated)
