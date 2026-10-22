"""Plant cover limits itself by the scarcest of heat, water, and light."""
import numpy as np

from ..core.pipeline import Stage
from ..core.registry import register


def suitability(value, preferred, half_width):
    return np.clip(1.0 - np.abs(np.asarray(value, np.float64) - preferred) / half_width, 0.0, 1.0)


def viability(temperature_c, moisture, sunlight, preferred_c, half_width=12.0):
    """The weakest of temperature, moisture, and sunlight sets the cover."""
    heat = suitability(temperature_c, preferred_c, half_width)
    wet = np.clip(np.asarray(moisture, np.float64), 0.0, 1.0)
    light = np.clip(np.asarray(sunlight, np.float64), 0.0, 1.0)
    return np.minimum(np.minimum(heat, wet), light)


@register
class Vegetation(Stage):
    name = "vegetation"
    requires = ("temperature_c", "rainfall_mm_day")
    writes = ("vegetation", "cohesion", "moisture")

    def __init__(self, rain_for_full_cover=8.0, root_cohesion=0.45):
        self.rain_for_full_cover = float(rain_for_full_cover)
        self.root_cohesion = float(root_cohesion)

    def run(self, world, ctx):
        rain = world.get("rainfall_mm_day").astype(np.float64)
        moisture = np.clip(rain / self.rain_for_full_cover, 0.0, 1.0)
        cover = viability(
            world.get("temperature_c"), moisture, 1.0,
            float(ctx.parameters["preferred_c"]))
        base = float(ctx.parameters.get("base_cohesion", 0.05))
        world.set("moisture", moisture)
        world.set("vegetation", cover)
        world.set("cohesion", base + self.root_cohesion * cover)
