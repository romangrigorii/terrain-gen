"""Groundwater dissolves a cavity under a roof. The roof falls only when it is too thin."""
import numpy as np

from ..core.pipeline import Stage
from ..core.registry import register


def dissolve(state, k=0.0015, roof_m=8.0, dt_years=1.0):
    """Grow `void_m` without lowering the surface, until the roof is `roof_m` thick.

    Dissolved thickness is recorded in `dissolved_thickness_sum`. Anything past
    the roof collapses: the surface drops and the void shrinks by the same amount.
    """
    column = np.maximum(state.elevation, 0.0)
    void = state.get("void_m").astype(np.float64)
    porosity = np.clip(state.get("porosity").astype(np.float64), 0.0, 1.0)
    hardness = np.clip(state.get("hardness").astype(np.float64), 0.0, 1.0)
    rain = np.maximum(state.get("rainfall_mm_day").astype(np.float64), 0.0)
    limit = np.maximum(0.0, column - float(roof_m))
    overload = np.maximum(0.0, void - limit)
    if float(overload.sum()) > 0.0:
        state.peel(overload)
        void = np.maximum(void - overload, 0.0)
        column = np.maximum(state.elevation, 0.0)
        limit = np.maximum(0.0, column - float(roof_m))
    room = np.maximum(0.0, limit - void)
    growth = float(k) * rain * porosity * (1.0 - hardness) * float(dt_years)
    growth = np.minimum(np.maximum(growth, 0.0), room)
    void = void + growth
    state.set("void_m", void)
    dissolved = float(state.features.get("dissolved_thickness_sum", 0.0)) + float(growth.sum())
    state.features["dissolved_thickness_sum"] = dissolved
    return growth


@register
class Karst(Stage):
    name = "karst"
    requires = ("rainfall_mm_day", "porosity", "hardness", "bedrock_m")
    writes = ("void_m",)

    def __init__(self, k=0.0015, roof_m=8.0):
        self.k = float(k)
        self.roof_m = float(roof_m)

    def run(self, world, ctx):
        dissolve(world, k=self.k, roof_m=self.roof_m, dt_years=ctx.dt_years)
