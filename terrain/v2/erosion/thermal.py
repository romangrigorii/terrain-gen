"""Move loose cover down a slope steeper than the material's angle of repose."""
import numpy as np

from ..core.grid import neighbor, neighbor_index
from ..core.pipeline import Stage
from ..core.registry import register

NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def relax(state, repose, rate):
    """Double-buffered talus step. `repose` is radians, a scalar or a field.

    Only soil, sand, and regolith move. The sum of solid elevation is unchanged.
    """
    height = state.elevation
    cover = (state.get("soil_m").astype(np.float64)
             + state.get("sand_m")
             + state.get("regolith_m"))
    phi = np.asarray(repose, np.float64)
    critical = float(state.grid.cell_m) * np.tan(phi)
    offered = []
    for dy, dx in NEIGH4:
        run = float(np.hypot(dy, dx))
        other = neighbor(height, dy, dx)
        limit = critical * run
        excess = height - other - limit
        excess = np.where(np.isnan(other), 0.0, excess)
        offered.append(np.maximum(excess, 0.0) * float(rate))
    stack = np.stack(offered, axis=0)
    demand = stack.sum(axis=0)
    scale = np.ones(height.shape, np.float64)
    hot = demand > 1e-12
    scale[hot] = np.minimum(1.0, cover[hot] / demand[hot])
    give = np.zeros(height.shape, np.float64)
    moved = 0.0
    for offer, (dy, dx) in zip(stack, NEIGH4):
        take = offer * scale
        dest = neighbor_index(height.shape, dy, dx)
        flows = (dest >= 0).ravel()
        np.add.at(give.ravel(), dest.ravel()[flows], take.ravel()[flows])
        moved += float(take.sum())
    state.peel(stack.sum(axis=0) * scale)
    state.deposit("sand_m", give)
    return moved


@register
class Thermal(Stage):
    name = "thermal"
    requires = ("bedrock_m",)
    writes = ("sand_m",)

    def __init__(self, rate=0.2, repose_deg=32.0, root_deg=12.0):
        self.rate = float(rate)
        self.repose_deg = float(repose_deg)
        self.root_deg = float(root_deg)

    def run(self, world, ctx):
        cohesion = np.clip(world.get("cohesion").astype(np.float64), 0.0, 1.0)
        repose = np.deg2rad(self.repose_deg + cohesion * self.root_deg)
        relax(world, repose, self.rate)
