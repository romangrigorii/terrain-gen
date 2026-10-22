"""Stamp glaciers, dunes, shores, and caves onto a generated heightmap."""
import numpy as np

from terrain.elevation import _stroke_mask

from ..erosion.aeolian import blow
from ..erosion.coastal import retreat
from ..erosion.glacial import glaciate
from ..core.state import TerrainState

KINDS = ("glacial", "dune", "coast", "cave")


def placed_strokes(strokes):
    """Strokes the generator can stamp. Mountains and valleys are not included."""
    chosen = []
    for stroke in strokes or ():
        if not isinstance(stroke, dict) or stroke.get("kind") not in KINDS:
            continue
        try:
            metres = float(stroke.get("metres", 0.0))
            radius = float(stroke.get("radius", 0.0))
        except (TypeError, ValueError):
            continue
        points = stroke.get("points") or ()
        if metres > 0 and radius > 0 and len(points) > 0 and np.isfinite(metres):
            chosen.append(stroke)
    return chosen


def _cover(shape, strokes, kind):
    cover = None
    for stroke in strokes:
        if stroke.get("kind") != kind:
            continue
        mask = _stroke_mask(shape, stroke["points"], float(stroke["radius"]))
        cover = mask if cover is None else np.maximum(cover, mask)
    if cover is None:
        return np.zeros(shape, np.float32)
    cover = np.array(cover, np.float32, copy=True)
    cover[0, :] = 0
    cover[-1, :] = 0
    cover[:, 0] = 0
    cover[:, -1] = 0
    return cover


def _metres(strokes, kind):
    return max(float(stroke["metres"]) for stroke in strokes if stroke.get("kind") == kind)


def _field(shape, value, fill):
    if value is None:
        return np.full(shape, fill, np.float32)
    array = np.asarray(value, np.float32)
    if array.shape != shape:
        raise ValueError(f"expected {shape}, got {array.shape}")
    return array


def place_height(height, strokes, cell_m, temp=None, rain=None):
    """Bake placed brushes into a heightmap.

    `height` uses the generator's units. Brush sizes are metres. The shared rim
    is left as it was. With nothing placed, the same height array comes back
    and the ice, sand, and void layers are absent.
    """
    chosen = placed_strokes(strokes)
    if not chosen:
        return height, None, None, None
    from terrain.export import METRES_PER_UNIT

    scale = float(METRES_PER_UNIT)
    base = np.asarray(height, np.float32)
    state = TerrainState(base.shape, float(cell_m))
    state.set("bedrock_m", np.asarray(base, np.float64) * scale)
    state.set("temperature_c", _field(base.shape, temp, 8.0))
    state.set("rainfall_mm_day", np.maximum(_field(base.shape, rain, 1.0), 0.0))

    ice_mask = _cover(base.shape, chosen, "glacial")
    if float(ice_mask.max()) > 0.0:
        cold = np.where(ice_mask > 0.15, -12.0, 8.0)
        state.set("temperature_c", cold)
        state.set("rainfall_mm_day", np.maximum(state.get("rainfall_mm_day"), 3.0))
        state.set("ice_m", _metres(chosen, "glacial") * ice_mask)
        for _ in range(4):
            glaciate(state, k=0.8, snow_m=0.5)

    dune_mask = _cover(base.shape, chosen, "dune")
    if float(dune_mask.max()) > 0.0:
        state.set("vegetation", np.where(dune_mask > 0.2, 0.0, 1.0))
        state.set("moisture", np.zeros(base.shape, np.float32))
        state.deposit("sand_m", _metres(chosen, "dune") * dune_mask)
        for _ in range(4):
            blow(state, wind_deg=90.0, rate=0.45)

    coast_mask = _cover(base.shape, chosen, "coast")
    if float(coast_mask.max()) > 0.0:
        retreat(state, k=_metres(chosen, "coast"), where=coast_mask > 0.35, open_edge=False)

    cave_mask = _cover(base.shape, chosen, "cave")
    if float(cave_mask.max()) > 0.0:
        column = np.maximum(state.elevation, 0.0)
        state.peel(cave_mask * np.minimum(_metres(chosen, "cave") * 0.5, column * 0.45))
        column = np.maximum(state.elevation, 0.0)
        void = np.minimum(_metres(chosen, "cave") * cave_mask, np.maximum(0.0, column - 8.0))
        state.set("void_m", void)

    out = np.array(base, np.float32, copy=True)
    metres = state.elevation
    changed = np.abs(metres - np.asarray(base, np.float64) * scale) > 0.05
    changed[0, :] = False
    changed[-1, :] = False
    changed[:, 0] = False
    changed[:, -1] = False
    out[changed] = (metres[changed] / scale).astype(np.float32)
    layers = []
    for name in ("ice_m", "sand_m", "void_m"):
        layer = state.get(name).astype(np.float32, copy=True)
        layer[0, :] = 0
        layer[-1, :] = 0
        layer[:, 0] = 0
        layer[:, -1] = 0
        layers.append(layer)
    return out, layers[0], layers[1], layers[2]
