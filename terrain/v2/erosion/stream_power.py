"""Incise rock in proportion to drainage area and slope, and pass the debris downstream."""
import numpy as np

from ..core.pipeline import Stage
from ..core.registry import register
from ..hydrology.routing import accumulate, route

REF_RAIN_MM_DAY = 1.0


def incise(state, k, m, n, dt_years, rain):
    """Apply one stream-power step. Cut rock is handed to the downstream cell as sand.

    A cell with no downhill neighbour keeps that sand. The thickness that
    leaves the grid is added to `exported_thickness_sum`.
    """
    cell = float(state.grid.cell_m)
    uplift = np.maximum(state.get("uplift_m_year").astype(np.float64), 0.0) * float(dt_years)
    state.deposit("bedrock_m", uplift)
    height = state.elevation
    receivers, slope = route(height, cell)
    weights = np.ones(height.shape, np.float64)
    area = accumulate(receivers, weights) * (cell * cell)
    runoff = np.maximum(np.asarray(rain, np.float64), 0.0) / REF_RAIN_MM_DAY
    resistance = 1.0 - 0.8 * np.clip(state.get("hardness").astype(np.float64), 0.0, 1.0)
    cut = (float(k) * resistance * runoff
           * np.power(np.maximum(area, 1.0), float(m)) * np.power(slope, float(n)))
    cut = np.maximum(cut, 0.0) * float(dt_years)
    flat = height.ravel()
    dest = receivers.ravel()
    room = np.full(flat.shape, np.inf, np.float64)
    flows = dest >= 0
    room[flows] = np.maximum(flat[flows] - flat[dest[flows]], 0.0) * 0.5
    cut = np.minimum(cut, room.reshape(height.shape))
    removed = state.peel(cut)
    passed = np.zeros(height.shape, np.float64)
    np.add.at(passed.ravel(), dest[flows], removed.ravel()[flows])
    state.deposit("sand_m", passed)
    exported = removed.copy()
    exported.ravel()[flows] = 0.0
    total = float(state.features.get("exported_thickness_sum", 0.0)) + float(exported.sum())
    state.features["exported_thickness_sum"] = total
    state.features["receivers"] = receivers
    state.set("flow_accumulation", area.astype(np.float32))
    return removed


@register
class StreamPower(Stage):
    name = "stream_power"
    requires = ("rainfall_mm_day", "uplift_m_year", "bedrock_m")
    writes = ("flow_accumulation", "sand_m", "bedrock_m")

    def __init__(self, m=0.45, n=1.0):
        self.m = float(m)
        self.n = float(n)

    def run(self, world, ctx):
        incise(world, ctx.parameters["erodibility"], self.m, self.n,
               ctx.dt_years, world.get("rainfall_mm_day"))
