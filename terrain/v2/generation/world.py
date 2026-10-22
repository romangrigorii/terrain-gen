"""Build the starting relief and run one scenario through the shared loop."""
import numpy as np

from ..core.parameters import NoiseField
from ..core.pipeline import Context, Stage
from ..core.registry import register
from ..core.scheduler import Scheduler
from ..core.state import TerrainState
from ..core.validation import check_state
from ..environment.vegetation import Vegetation
from ..erosion.aeolian import Aeolian
from ..erosion.coastal import Coastal
from ..erosion.glacial import Glacial
from ..erosion.hydraulic import Hydraulic
from ..erosion.karst import Karst
from ..erosion.stream_power import StreamPower
from ..erosion.thermal import Thermal
from .scenarios import parameters_for


@register
class SeedRelief(Stage):
    name = "seed_relief"
    requires = ()
    writes = ("bedrock_m",)

    def run(self, world, ctx):
        if "legacy_world" in world.features:
            return
        yy, xx = world.grid.coordinates()
        cy = world.grid.origin_m[0] + world.grid.rows * world.grid.cell_m * 0.5
        cx = world.grid.origin_m[1] + world.grid.cols * world.grid.cell_m * 0.5
        sigma = 0.22 * min(world.grid.rows, world.grid.cols) * world.grid.cell_m
        hill = 700.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
        grain = NoiseField(1.0 / (6.0 * world.grid.cell_m), 8.0, ctx.seed).sample_grid(world.grid)
        world.set("bedrock_m", hill + grain)


@register
class PaintClimate(Stage):
    name = "climate_fields"
    requires = ()
    writes = ("temperature_c", "rainfall_mm_day", "uplift_m_year", "hardness", "porosity")

    def run(self, world, ctx):
        params = ctx.parameters
        grid = world.grid
        world.set("temperature_c", params["temperature"].sample_grid(grid))
        world.set("rainfall_mm_day", params["rainfall"].sample_grid(grid))
        world.set("uplift_m_year", params["uplift"].sample_grid(grid))
        world.set("hardness", params["hardness"].sample_grid(grid))
        world.set("porosity", params["porosity"].sample_grid(grid))


def loop_stages(processes):
    stages = [Vegetation(), StreamPower(), Thermal(), Hydraulic()]
    if processes.get("glacial"):
        stages.append(Glacial())
    if processes.get("aeolian"):
        stages.append(Aeolian())
    if processes.get("coastal"):
        stages.append(Coastal())
    if processes.get("karst"):
        stages.append(Karst())
    return stages


def run(name="temperate", shape=(36, 48), cell_m=100.0, seed=0, passes=2,
        dt_years=200.0, overrides=None):
    """Generate one landscape from a scenario name and a shared set of stages."""
    state = TerrainState(shape, cell_m)
    params = parameters_for(name, shape, cell_m, seed, overrides)
    ctx = Context(seed=int(seed), dt_years=float(dt_years), parameters=params)
    Scheduler(
        [SeedRelief(), PaintClimate()],
        loop_stages(params["processes"]),
        passes=passes,
    ).run(state, ctx)
    check_state(state)
    state.features["scenario"] = name
    state.features["time_years"] = ctx.time_years
    state.features["summary"] = params["summary"]
    return state
