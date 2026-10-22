"""Engine-oriented derivative layers, intentionally not part of simulation state."""
import numpy as np

from . import grid, climate


def normals(height, vertical_scale=1.0, horizontal_scale=1.0):
    """(H,W,3) unit normals; x=east, y=image-row down, z=up."""
    if horizontal_scale <= 0:
        raise ValueError("horizontal_scale must be positive")
    dy, dx = grid.gradient(np.asarray(height, dtype=np.float32))
    zscale = vertical_scale / horizontal_scale
    vec = np.stack((-dx * zscale, -dy * zscale, np.ones_like(dx)), axis=-1)
    vec /= np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)
    return vec.astype(np.float32)


def slope(height, vertical_scale=1.0, horizontal_scale=1.0):
    """Slope angle in radians, 0 on flat ground."""
    return np.arccos(np.clip(normals(height, vertical_scale, horizontal_scale)[..., 2], 0, 1))


def curvature(height):
    """Discrete height Laplacian: negative around crests, positive in hollows."""
    h = np.asarray(height, dtype=np.float32)
    return (grid.shift(h, -1, 0) + grid.shift(h, 1, 0) +
            grid.shift(h, 0, -1) + grid.shift(h, 0, 1) - 4*h)


def material_weights(world, vertical_scale=1.0, horizontal_scale=1.0):
    """Procedural PBR-ready splat weights (H,W,4)."""
    h = world.height
    ground = world.land & ~world.lakes & ~world.rivers
    sl = np.clip(slope(h, vertical_scale, horizontal_scale) / np.deg2rad(50.0), 0, 1)
    coastal = np.exp(-grid.edt(world.land) / 2.0)
    dry = np.clip((0.65 - world.runoff) / 0.65, 0, 1)
    is_desert = world.biome == climate.DESERT
    sand = 0.07 + 2.2 * coastal + 2.5 * np.maximum(dry, is_desert.astype(float))
    soil = np.full_like(h, 1.5, dtype=np.float32)
    rock = 0.02 + 4.0 * sl**2 + 1.3 * np.clip(h - 0.20, 0, 1)
    snow = 4.0 * np.clip((-world.temp - 1.0) / 16.0, 0, 1)**1.5
    snow += 10.0 * (world.biome == climate.ICE)
    raw = np.stack((sand, soil, rock, snow), axis=-1).astype(np.float32)
    raw /= np.maximum(raw.sum(axis=-1, keepdims=True), 1e-9)
    raw *= ground[..., None]
    return raw


def wetness(world):
    """Approximate terrain moisture for shader effects (0-1, not groundwater)."""
    rain = np.clip(world.runoff / 3, 0, 1)
    drainage = np.log1p(np.maximum(world.water.flow, 0))
    scale = float(np.percentile(drainage[world.land], 99)) if world.land.any() else 1.0
    drainage = np.clip(drainage / max(scale, 1e-6), 0, 1)
    shore = np.exp(-grid.edt(~(world.rivers | world.lakes)) / 4.0)
    return np.where(world.land, np.clip(0.30*rain + 0.45*drainage + 0.25*shore, 0, 1), 0).astype(np.float32)


def flow_directions(world):
    """Unit (dy,dx) D8 drainage vectors, wrapping x; (0,0) for sinks."""
    H, W = world.height.shape
    yy, xx = np.indices((H, W))
    rec = world.water.receivers.astype(np.int64)
    ry, rx = rec // W, rec % W
    dx = (rx - xx + W//2) % W - W//2
    dy = ry - yy
    norm = np.hypot(dy, dx)
    norm = np.maximum(norm, 1)
    return np.stack((dy / norm, dx / norm), axis=-1).astype(np.float32)
