"""Ice thickens in the cold, flows downhill, and widens the valley it occupies."""
import numpy as np

from ..core.grid import neighbor
from ..core.pipeline import Stage
from ..core.registry import register
from ..hydrology.routing import route
from .hydraulic import step_water

NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _widen(ice):
    total = np.asarray(ice, np.float64).copy()
    count = np.ones(total.shape, np.float64)
    for dy, dx in NEIGH4:
        other = neighbor(total, dy, dx)
        valid = ~np.isnan(other)
        total = total + np.where(valid, other, 0.0)
        count = count + valid
    return total / count


def glaciate(state, k=0.04, snow_m=1.5, flow=0.45):
    """One glacial step. Warm ice melts into a thin till. Cold ice abrades rock.

    Abrasion uses ice averaged with its neighbours, so the valley walls retreat
    with the floor instead of leaving a knife-cut slot.
    """
    temp = state.get("temperature_c").astype(np.float64)
    rain = np.maximum(state.get("rainfall_mm_day").astype(np.float64), 0.0)
    ice = state.get("ice_m").astype(np.float64)
    cold = temp < 0.0
    ice = ice + np.where(cold, float(snow_m) * rain / 4.0, 0.0)
    melt = np.where(~cold, ice * 0.8, 0.0)
    ice = ice - melt
    state.deposit("regolith_m", melt * 0.15)
    moving = ice * float(flow)
    ice = (ice - moving) + step_water(state.elevation, moving, 0.0)
    ice = np.maximum(ice, 0.0)
    widened = _widen(ice)
    _, slope = route(state.elevation, state.grid.cell_m)
    cut = float(k) * widened * (0.08 + slope)
    cut = np.minimum(cut, widened * 0.2)
    state.peel(cut)
    state.set("ice_m", ice)
    return cut


@register
class Glacial(Stage):
    name = "glacial"
    requires = ("temperature_c", "rainfall_mm_day", "bedrock_m")
    writes = ("ice_m", "regolith_m")

    def __init__(self, k=0.04, snow_m=1.5):
        self.k = float(k)
        self.snow_m = float(snow_m)

    def run(self, world, ctx):
        glaciate(world, k=self.k, snow_m=self.snow_m)
