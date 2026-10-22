"""Layered landscape. Solid elevation is bedrock plus the cover on top of it."""
import numpy as np

from .grid import Grid

SOLID = ("bedrock_m", "regolith_m", "sand_m", "soil_m")
COVER = ("soil_m", "sand_m", "regolith_m")
FIELDS = SOLID + (
    "hardness",
    "porosity",
    "cohesion",
    "water_m",
    "sediment_kg_m2",
    "moisture",
    "vegetation",
    "temperature_c",
    "rainfall_mm_day",
    "flow_accumulation",
    "uplift_m_year",
    "ice_m",
    "void_m",
)
THICKNESS = COVER + (
    "hardness",
    "porosity",
    "cohesion",
    "water_m",
    "sediment_kg_m2",
    "moisture",
    "vegetation",
    "rainfall_mm_day",
    "ice_m",
    "void_m",
)


class TerrainState:
    def __init__(self, shape, cell_m, origin_m=(0.0, 0.0)):
        self.grid = Grid(tuple(shape), float(cell_m), tuple(origin_m))
        self.fields = {}
        self.features = {}
        zeros = np.zeros(self.grid.shape, np.float32)
        for name in FIELDS:
            self.fields[name] = zeros.copy()

    def get(self, name):
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    def set(self, name, data):
        array = np.array(data, dtype=np.float32, copy=True)
        if array.shape != self.grid.shape:
            raise ValueError(f"{name}: expected {self.grid.shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: invalid values")
        self.fields[name] = array
        return array

    @property
    def elevation(self):
        return (self.get("bedrock_m").astype(np.float64)
                + self.get("regolith_m")
                + self.get("sand_m")
                + self.get("soil_m"))

    def peel(self, metres):
        """Take solid material off the top. Cover stays nonnegative.

        Bedrock is an elevation, so it may fall below zero once the cover is gone.
        Returns the thickness removed at each cell.
        """
        left = np.maximum(np.asarray(metres, np.float64), 0.0)
        if left.shape != self.grid.shape:
            raise ValueError("peel mask does not match the grid")
        removed = np.zeros(self.grid.shape, np.float64)
        for name in COVER:
            layer = self.get(name).astype(np.float64)
            take = np.minimum(layer, left)
            self.set(name, layer - take)
            left = left - take
            removed = removed + take
        self.set("bedrock_m", self.get("bedrock_m").astype(np.float64) - left)
        return removed + left

    def deposit(self, layer, metres):
        if layer not in COVER and layer != "bedrock_m":
            raise ValueError(f"cannot deposit onto {layer}")
        add = np.asarray(metres, np.float64)
        if np.any(add < 0):
            raise ValueError(f"{layer}: deposit thickness must be nonnegative")
        self.set(layer, self.get(layer).astype(np.float64) + add)
