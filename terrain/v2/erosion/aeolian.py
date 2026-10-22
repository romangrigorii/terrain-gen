"""Loose sand moves downwind and stops against plants, building a dune."""
import numpy as np

from ..core.grid import neighbor, neighbor_index
from ..core.pipeline import Stage
from ..core.registry import register


def wind_vector(wind_deg):
    """Snap a compass bearing to a cardinal step. 0 is north, 90 is east."""
    rad = np.deg2rad(float(wind_deg))
    vy = -np.cos(rad)
    vx = np.sin(rad)
    if abs(vx) >= abs(vy):
        return 0, int(np.sign(vx) or 1)
    return int(np.sign(vy) or 1), 0


def blow(state, wind_deg=90.0, rate=0.35, shelter=0.55):
    """Shift mobile sand one cell downwind.

    Plants and wet ground hold the grains. Sand that would leave the map is
    counted in `blown_thickness_sum`. The rest stays on the grid.
    """
    sand = state.get("sand_m").astype(np.float64)
    veg = np.clip(state.get("vegetation").astype(np.float64), 0.0, 1.0)
    wet = np.clip(state.get("moisture").astype(np.float64), 0.0, 1.0)
    mobile = (1.0 - veg) * (1.0 - wet)
    flux = np.minimum(sand, sand * mobile * float(rate))
    dy, dx = wind_vector(wind_deg)
    dest = neighbor_index(sand.shape, dy, dx)
    ahead_veg = neighbor(veg, dy, dx)
    blocked = np.isnan(ahead_veg) | (ahead_veg >= float(shelter))
    send = np.where(blocked, 0.0, flux)
    stay = flux - send
    arrived = np.zeros(sand.shape, np.float64)
    flows = (dest >= 0).ravel()
    np.add.at(arrived.ravel(), dest.ravel()[flows], send.ravel()[flows])
    exported = float(send[dest < 0].sum())
    state.set("sand_m", sand - flux + stay + arrived)
    total = float(state.features.get("blown_thickness_sum", 0.0)) + exported
    state.features["blown_thickness_sum"] = total
    return exported


@register
class Aeolian(Stage):
    name = "aeolian"
    requires = ("sand_m", "vegetation", "moisture")
    writes = ("sand_m",)

    def __init__(self, rate=0.35):
        self.rate = float(rate)

    def run(self, world, ctx):
        blow(world, wind_deg=float(ctx.parameters.get("wind_deg", 90.0)), rate=self.rate)
