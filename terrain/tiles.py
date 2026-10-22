"""Hierarchical tiles sampled from one macro world."""
from dataclasses import dataclass

import numpy as np

from . import grid, hydrology, noise
from .export import METRES_PER_UNIT

RELIEF_UNITS = 0.08


@dataclass(frozen=True)
class TileSpec:
    tx: int
    ty: int
    cells: int = 512
    meters_per_cell: float = 2.0
    halo: int = 64


@dataclass
class Tile:
    spec: TileSpec
    height: np.ndarray
    bedrock: np.ndarray
    sediment: np.ndarray
    hardness: np.ndarray
    materials: np.ndarray
    origin_m: tuple
    meters_per_cell: float


def world_extent_m(cfg):
    """`(height_m, width_m)` covered by the macro raster."""
    width_m = float(cfg.world_width_m)
    if not np.isfinite(width_m) or width_m <= 0:
        raise ValueError("world_width_m must be finite and positive")
    height_m = width_m * (float(cfg.height) / float(cfg.width))
    return height_m, width_m


def _check_spec(spec):
    if spec.cells < 2:
        raise ValueError("cells must be at least 2")
    if spec.halo < 1:
        raise ValueError("halo must be at least 1")
    if spec.meters_per_cell <= 0:
        raise ValueError("meters_per_cell must be positive")


def _layered_relief(macro, y_m, x_m, meters_per_cell, seed, world_width_m):
    """Macro elevation plus ridges and gullies that follow its slope."""
    dy, dx = np.gradient(np.asarray(macro, np.float64), float(meters_per_cell))
    grade = np.hypot(dy, dx).astype(np.float32) * np.float32(METRES_PER_UNIT)
    steep = np.clip(grade / np.float32(0.35), 0.0, 1.0)
    ridge = noise.fbm_world(
        y_m, x_m, wavelength=max(meters_per_cell * 8.0, 400.0),
        octaves=4, seed=int(seed) + 11, ridged=True, world_width_m=world_width_m)
    gully = noise.fbm_world(
        y_m, x_m, wavelength=max(meters_per_cell * 4.0, 40.0),
        octaves=3, seed=int(seed) + 29, world_width_m=world_width_m)
    ridge_m = 40.0 + 380.0 * steep
    gully_m = (4.0 + 18.0 * np.clip(grade / np.float32(0.2), 0.0, 1.0))
    relief = (ridge * ridge_m - np.clip(gully - np.float32(0.45), 0.0, None) * gully_m)
    return (macro + relief / np.float32(METRES_PER_UNIT)).astype(np.float32), grade


def _strata(surface, grade, y_m, x_m, seed, world_width_m):
    hardness = noise.fbm_world(
        y_m, x_m, wavelength=4000.0, octaves=3, seed=int(seed) + 47,
        world_width_m=world_width_m)
    hardness = np.clip(0.25 + 0.6 * hardness, 0.05, 0.95).astype(np.float32)
    mantle_m = 2.0 + 30.0 * np.exp(-np.asarray(grade, np.float64) * 8.0)
    sediment = (mantle_m / METRES_PER_UNIT).astype(np.float32)
    bedrock = (surface - sediment).astype(np.float32)
    return bedrock, sediment, hardness


def material_weights(surface, sediment, grade, runoff, temp):
    """Continuous weights: sand, soil, rock, snow. Each pixel sums to one."""
    steep = np.clip(np.asarray(grade, np.float32) / np.float32(0.6), 0.0, 1.0)
    mantle = np.clip(np.asarray(sediment, np.float32) * np.float32(METRES_PER_UNIT) / np.float32(8.0), 0.0, 1.0)
    wet = np.clip(np.asarray(runoff, np.float32), 0.0, 1.0)
    cold = np.clip(-np.asarray(temp, np.float32) / np.float32(12.0), 0.0, 1.0)
    rock = 0.05 + steep * (1.0 - 0.7 * mantle)
    sand = (1.0 - steep) * (1.0 - wet) * mantle
    soil = (1.0 - steep) * (0.35 + 0.65 * wet) * np.maximum(mantle, np.float32(0.25))
    snow = cold * (0.4 + 0.6 * steep)
    raw = np.stack((sand, soil, rock, snow), axis=-1).astype(np.float32)
    raw /= np.maximum(raw.sum(axis=-1, keepdims=True), np.float32(1e-6))
    return raw


def _solve_patch(macro, flow, runoff, temp, y_m, x_m, meters_per_cell, seed,
                 world_width_m, iters):
    from . import geology

    surface, grade = _layered_relief(
        macro, y_m, x_m, meters_per_cell, seed, world_width_m)
    bedrock, sediment, hardness = _strata(
        surface, grade, y_m, x_m, seed, world_width_m)
    bedrock_m = np.asarray(bedrock, np.float64) * METRES_PER_UNIT
    reference_m = (2000.0
                   + np.asarray(y_m, np.float64) * 0.02
                   + np.asarray(x_m, np.float64) * 0.01)
    strata = geology.strata_erodability(
        bedrock_m, reference_m, metres_per_unit=1.0)
    hardness = np.clip(0.7 * hardness + 0.3 * (1.0 - strata), 0.05, 0.95)
    hardness = hardness.astype(np.float32)
    erodability = (1.0 - hardness).astype(np.float32)
    bedrock, sediment, surface = hydrology.erode_multiscale(
        bedrock, sediment, erodability, flow, float(meters_per_cell),
        max_radius=int(iters), wrap_x=False)
    weights = material_weights(surface, sediment, grade, runoff, temp)
    return surface, bedrock, sediment, hardness, weights


def generate_tile(world, spec):
    """Build one tile from `world`, including its shared edge vertices."""
    _check_spec(spec)
    extent_y, extent_x = world_extent_m(world.cfg)
    pad = int(spec.halo)
    rows = spec.cells + 1 + 2 * pad
    cols = spec.cells + 1 + 2 * pad
    origin_y = spec.ty * spec.cells * spec.meters_per_cell
    origin_x = spec.tx * spec.cells * spec.meters_per_cell
    y_m, x_m = grid.vertex_axes(
        origin_y - pad * spec.meters_per_cell,
        origin_x - pad * spec.meters_per_cell,
        rows, cols, spec.meters_per_cell, spec.meters_per_cell)
    sample = lambda field: grid.sample_raster(field, y_m, x_m, extent_y, extent_x, wrap_x=True)
    iters = min(max(int(spec.halo) - 1, 0), 16)
    surface, bedrock, sediment, hardness, weights = _solve_patch(
        sample(world.height), sample(world.water.flow), sample(world.runoff),
        sample(world.temp), y_m, x_m, spec.meters_per_cell, world.cfg.seed,
        extent_x, iters)
    core = slice(pad, pad + spec.cells + 1)
    return Tile(
        spec=spec,
        height=surface[core, core],
        bedrock=bedrock[core, core],
        sediment=sediment[core, core],
        hardness=hardness[core, core],
        materials=weights[core, core],
        origin_m=(float(origin_y), float(origin_x)),
        meters_per_cell=float(spec.meters_per_cell),
    )


def hierarchical_window(macro, y0, x0, height, width, full_h, full_w, seed,
                        world_width_m, world_height_m, halo=8, iters=4,
                        wrap_x=True, meters_per_cell=None):
    """Regenerate one export window, padded, then cropped back to it."""
    if halo <= iters:
        raise ValueError("halo must be wider than the sediment iteration count")
    if world_width_m <= 0 or world_height_m <= 0:
        raise ValueError("world extent must be positive")
    full_h = int(full_h)
    full_w = int(full_w)
    if meters_per_cell is None:
        meters_per_cell = float(world_width_m) / float(full_w)

    def metres(y, x):
        y_m = (np.asarray(y, np.float64) + 0.5) / full_h * float(world_height_m)
        x_m = (np.asarray(x, np.float64) + 0.5) / full_w * float(world_width_m)
        return y_m, x_m

    y_idx = np.arange(int(y0) - int(halo), int(y0) + int(height) + int(halo))
    x_idx = np.arange(int(x0) - int(halo), int(x0) + int(width) + int(halo))
    if not wrap_x:
        x_idx = np.clip(x_idx, 0, full_w - 1)
    y_m, x_m = metres(y_idx[:, None], x_idx[None, :])
    sampled = grid.sample_raster(
        macro, y_m, x_m, world_height_m, world_width_m, wrap_x=wrap_x)
    flow = np.ones(sampled.shape, np.float32)
    runoff = np.full(sampled.shape, np.float32(0.5))
    temp = np.zeros(sampled.shape, np.float32)
    surface, _, _, _, _ = _solve_patch(
        sampled, flow, runoff, temp, y_m, x_m, meters_per_cell, seed,
        world_width_m, iters)
    return np.ascontiguousarray(surface[halo:halo + height, halo:halo + width])
