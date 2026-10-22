"""Lay an existing generated world onto the layered state without changing its height."""
import numpy as np

from ..core.pipeline import Stage
from ..core.registry import register
from ..core.state import TerrainState


def import_world(world):
    """Copy a `terrain.world.World` onto material layers.

    Solid elevation matches `world.height` converted to metres. Cover thickness
    stays zero so the derived surface is that same heightmap.
    """
    from terrain.climate import precip_mm
    from terrain.export import METRES_PER_UNIT

    cell = float(world.cfg.world_width_m) / float(max(world.cfg.width, 1))
    state = TerrainState(world.height.shape, cell)
    metres = np.asarray(world.height, np.float64) * METRES_PER_UNIT
    state.set("bedrock_m", metres)
    lakes = np.clip(np.asarray(world.water.lake_depth, np.float64), 0.0, None)
    state.set("water_m", lakes * METRES_PER_UNIT)
    state.set("temperature_c", np.asarray(world.temp, np.float32))
    rain = precip_mm(world.cfg) * np.asarray(world.runoff, np.float64) / 365.0
    state.set("rainfall_mm_day", np.maximum(rain, 0.0))
    state.set("flow_accumulation",
              np.asarray(world.water.flow, np.float64) * state.grid.cell_area_m2)
    cover = 0.0
    if world.trees is not None:
        cover = np.clip(np.asarray(world.trees.cover, np.float64), 0.0, None)
    state.set("vegetation", cover)
    state.features["legacy_world"] = world
    return state


@register
class LegacyImport(Stage):
    name = "legacy_import"
    requires = ()
    writes = ("bedrock_m", "water_m", "temperature_c", "rainfall_mm_day",
              "vegetation", "flow_accumulation")

    def run(self, world, ctx):
        imported = import_world(ctx.parameters["legacy_world"])
        if imported.grid.shape != world.grid.shape or imported.grid.cell_m != world.grid.cell_m:
            raise ValueError("legacy world grid does not match this state")
        for name in imported.fields:
            world.set(name, imported.fields[name])
        world.features["legacy_world"] = imported.features["legacy_world"]
