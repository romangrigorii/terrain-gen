"""Tiles of one planet. A tile does not read its neighbors.

Shared vertices are the same global indices, so tile (5, 3) matches tile (4, 3)
whether or not that neighbor has been generated. Longitude repeats. The two
pole rows are the two ends of the grid; a sample that steps past a pole comes
back on the opposite meridian.

Local steps are detail, smoothing, and normals. Each one only looks a fixed
distance, and the halo is that distance. Stream power, hydraulic erosion,
watersheds, and rivers move water across the planet, so they belong to the
global pass that is stored on the planet before any tile is cut.

Optional patch corrections come from terrain.trend_patches. They are anchored
to the global stride grid and added after the global smooth, so a canyon is
not rebuilt relative to the output tile and is not faded out on its edge.
"""
from dataclasses import dataclass

import numpy as np

from . import grid
from .trend_patches import smootherstep

TILE_CELLS = 256
DEFAULT_CELLS_X = TILE_CELLS * 64   # 16,384
DEFAULT_CELLS_Y = TILE_CELLS * 32   # 8,192
DEFAULT_HALO = 16

LOCAL_OPERATIONS = ("detail", "smooth", "normals", "slope", "curvature")
GLOBAL_OPERATIONS = ("stream_power", "hydraulic", "watershed", "sediment", "rivers")


def operation_scope(name):
    """'local' fits in a halo. 'global' has to be solved for the whole planet."""
    if name in LOCAL_OPERATIONS:
        return "local"
    if name in GLOBAL_OPERATIONS:
        return "global"
    raise ValueError(f"unknown terrain operation: {name}")


def required_halo(smooth_passes):
    """Cells of halo a tile needs for `smooth_passes` plus a normal stencil."""
    if smooth_passes < 0:
        raise ValueError("smooth_passes must be nonnegative")
    return int(smooth_passes) + 1


@dataclass(frozen=True)
class PlanetGrid:
    cells_x: int = DEFAULT_CELLS_X
    cells_y: int = DEFAULT_CELLS_Y
    tile: int = TILE_CELLS
    halo: int = DEFAULT_HALO

    def __post_init__(self):
        if self.tile < 2:
            raise ValueError("tile must be at least 2 cells")
        if self.halo < 1:
            raise ValueError("halo must be at least 1")
        if self.cells_x % self.tile or self.cells_y % self.tile:
            raise ValueError("planet cells must divide into whole tiles")
        if self.cells_x % 2:
            raise ValueError("longitude cell count must be even")
        if self.cells_x < self.tile or self.cells_y < self.tile:
            raise ValueError("planet is smaller than one tile")

    @property
    def tiles_x(self):
        return self.cells_x // self.tile

    @property
    def tiles_y(self):
        return self.cells_y // self.tile

    def tile_origin(self, tx, ty):
        tx, ty = _tile_index(self, tx, ty)
        return ty * self.tile, tx * self.tile


@dataclass
class PlanetState:
    """Global fields. Building a tile only reads this."""

    seed: int
    grid: PlanetGrid
    sea_level: float
    centers: np.ndarray
    weights: np.ndarray
    sigma: np.ndarray
    coarse: np.ndarray = None
    patches: object = None
    features: object = None
    polar_transition: int = 0


@dataclass(frozen=True)
class TileConfig:
    """Output-tile size for one planet. Patches use the same cell counts."""

    world_width: int
    world_height: int
    tile_cells: int = TILE_CELLS
    seed: int = 0
    halo: int = DEFAULT_HALO


@dataclass
class PlanetTile:
    tx: int
    ty: int
    height: np.ndarray
    normal: np.ndarray
    bedrock: np.ndarray
    sea_level: float


def build_planet(seed, cells_x=64, cells_y=32, tile=16, halo=4, coarse=None,
                 sea_level=0.0, features=8, polar_transition=None):
    """One planet. `coarse` is an optional macro height raster, sampled by area."""
    grid = PlanetGrid(int(cells_x), int(cells_y), int(tile), int(halo))
    if coarse is not None:
        coarse = np.asarray(coarse, np.float32)
        if coarse.ndim != 2:
            raise ValueError("coarse height must be a 2D raster")
    if polar_transition is None:
        polar_transition = 8 if coarse is not None else 0
    polar_transition = int(polar_transition)
    if polar_transition < 0:
        raise ValueError("polar transition must be nonnegative")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    centers = rng.normal(size=(int(features), 3))
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return PlanetState(
        seed=int(seed), grid=grid, sea_level=float(sea_level),
        centers=centers, weights=rng.normal(size=int(features)),
        sigma=rng.uniform(0.35, 0.9, size=int(features)),
        coarse=coarse, polar_transition=polar_transition)


def from_world(world, tile, halo=8):
    """Global state from a finished world. Tiles sample it; they do not rerun it."""
    height = np.asarray(world.height, np.float32)
    cells_y, cells_x = height.shape
    return build_planet(int(world.cfg.seed), cells_x, cells_y, tile, halo,
                        coarse=height, sea_level=0.0, features=4)


def _tile_index(grid, tx, ty):
    tx = int(tx) % grid.tiles_x
    ty = int(ty)
    if not 0 <= ty < grid.tiles_y:
        raise ValueError("latitude tile is outside the poles")
    return tx, ty


def canonical_vertices(gy, gx, cells_y, cells_x):
    """Fold vertex indices onto the planet. Past a pole, longitude shifts halfway."""
    gy = np.array(gy, np.int64, copy=True)
    gx = np.array(gx, np.int64, copy=True)
    for _ in range(8):
        low = gy < 0
        high = gy > cells_y
        if not np.any(low) and not np.any(high):
            break
        if np.any(low):
            gy = np.where(low, -gy, gy)
            gx = np.where(low, gx + cells_x // 2, gx)
        if np.any(high):
            gy = np.where(high, 2 * cells_y - gy, gy)
            gx = np.where(high, gx + cells_x // 2, gx)
    return gy, np.mod(gx, cells_x)


def _hash_detail(gy, gx, seed):
    mixed = (gy * np.int64(374761393) + gx * np.int64(668265263)
             + np.int64(seed) * np.int64(1442695041))
    mixed &= np.int64(0xFFFFFFFF)
    mixed ^= mixed >> np.int64(13)
    mixed = (mixed * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    unit = (mixed & np.int64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)
    return unit - 0.5


def sample_global_height(gy, gx, state):
    """Bedrock at vertex indices. The same index always returns the same value."""
    grid = state.grid
    gy, gx = canonical_vertices(gy, gx, grid.cells_y, grid.cells_x)
    at_pole = (gy == 0) | (gy == grid.cells_y)
    gx = np.where(at_pole, np.int64(0), gx)
    phi = gy.astype(np.float64) / grid.cells_y * np.pi - np.pi / 2.0
    lam = gx.astype(np.float64) / grid.cells_x * (2.0 * np.pi)
    cos_phi = np.cos(phi)
    position = (cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi))
    height = np.zeros(np.broadcast_shapes(np.shape(gy), np.shape(gx)), np.float64)
    for center, weight, sigma in zip(state.centers, state.weights, state.sigma):
        dot = sum(part * float(axis) for part, axis in zip(position, center))
        angular = np.arccos(np.clip(dot, -1.0, 1.0))
        height += float(weight) * np.exp(-0.5 * (angular / float(sigma)) ** 2)
    if state.coarse is not None:
        height += sample_coarse_height(gy, gx, state)
    features = getattr(state, "features", None)
    if features is not None:
        height += features.evaluate(gy, gx)
    height += _hash_detail(gy, gx, state.seed) * 0.08
    return height.astype(np.float32)


def sample_coarse_height(gy, gx, state):
    """Macro height. Each polar cap is a quadratic in the tangent plane, not a clamped row."""
    grid = state.grid
    sampled = grid_sample(state.coarse, gy, gx, grid.cells_y, grid.cells_x)
    transition = int(getattr(state, "polar_transition", 0) or 0)
    if transition < 2 or state.coarse is None:
        return np.asarray(sampled, np.float64)
    model = _polar_model(state, transition)
    if model is None:
        return np.asarray(sampled, np.float64)
    return _blend_caps(gy, gx, sampled, model, grid.cells_y, grid.cells_x, transition)


def _fit_cap(height, alpha, lam):
    """Least squares ``a + b x + c y + d x² + e xy + f y²`` on the tangent plane."""
    scale = max(float(np.max(alpha)), 1e-6)
    xn = alpha / scale * np.cos(lam)
    yn = alpha / scale * np.sin(lam)
    basis = np.column_stack((
        np.ones(xn.size), xn, yn, xn * xn, xn * yn, yn * yn,
    ))
    coef, *_ = np.linalg.lstsq(basis, np.asarray(height, np.float64), rcond=None)
    return np.asarray(coef, np.float64), scale


def _eval_cap(coef, scale, alpha, lam):
    xn = np.asarray(alpha, np.float64) / scale * np.cos(lam)
    yn = np.asarray(alpha, np.float64) / scale * np.sin(lam)
    a, b, c, d, e, f = coef
    return a + b * xn + c * yn + d * xn * xn + e * xn * yn + f * yn * yn


def _polar_model(state, transition):
    cached = getattr(state, "_polar_model", None)
    if cached is not None and cached[0] == transition:
        return cached[1]
    grid = state.grid
    span = min(int(transition), max(grid.cells_y // 2 - 1, 0))
    if span < 2:
        state._polar_model = (transition, None)
        return None
    step = max(1, grid.cells_x // 64)
    gx = np.arange(0, grid.cells_x, step, dtype=np.float64)
    rings = np.arange(1, span + 1, dtype=np.float64)

    def fit(gys, south):
        yy = np.repeat(gys, gx.size)
        xx = np.tile(gx, gys.size)
        height = np.asarray(
            grid_sample(state.coarse, yy, xx, grid.cells_y, grid.cells_x), np.float64)
        if south:
            alpha = yy / grid.cells_y * np.pi
        else:
            alpha = (grid.cells_y - yy) / grid.cells_y * np.pi
        lam = xx / grid.cells_x * (2.0 * np.pi)
        return _fit_cap(height, alpha, lam)

    model = (fit(rings, True), fit(grid.cells_y - rings, False))
    state._polar_model = (transition, model)
    return model


def _blend_caps(gy, gx, bilinear, model, cells_y, cells_x, transition):
    """Polynomial at the pole, macro raster at the cap rim, quintic in between."""
    gy, gx = np.broadcast_arrays(np.asarray(gy, np.float64), np.asarray(gx, np.float64))
    out = np.asarray(bilinear, np.float64)
    lam = gx / float(cells_x) * (2.0 * np.pi)
    transition = min(int(transition), cells_y // 2)
    south_coef, south_scale = model[0]
    north_coef, north_scale = model[1]

    def mix(distance, poly):
        fade = smootherstep(np.clip(distance / float(transition), 0.0, 1.0))
        return poly * (1.0 - fade) + out * fade

    south = gy <= transition
    if np.any(south):
        alpha = gy / float(cells_y) * np.pi
        blended = mix(gy, _eval_cap(south_coef, south_scale, alpha, lam))
        out = np.where(south, blended, out)
    north = gy >= cells_y - transition
    if np.any(north):
        alpha = (float(cells_y) - gy) / float(cells_y) * np.pi
        distance = float(cells_y) - gy
        blended = mix(distance, _eval_cap(north_coef, north_scale, alpha, lam))
        out = np.where(north, blended, out)
    return out


def grid_sample(field, gy, gx, cells_y, cells_x):
    """Sample a macro raster at canonical vertex indices."""
    yy = np.asarray(gy, np.float64)
    xx = np.asarray(gx, np.float64)
    return grid.sample_raster(field, yy, xx, float(cells_y), float(cells_x), wrap_x=True)


def _smooth_once(height):
    """One 5-tap average. The outer pixel of `height` is left as it arrived."""
    out = np.array(height, np.float32, copy=True)
    center = height[1:-1, 1:-1]
    acc = np.array(center, np.float32, copy=True)
    acc += height[:-2, 1:-1]
    acc += height[2:, 1:-1]
    acc += height[1:-1, :-2]
    acc += height[1:-1, 2:]
    out[1:-1, 1:-1] = acc * np.float32(0.2)
    return out


def _normals(height):
    """Unit normals from the cell steps. `height` includes the stencil margin."""
    south = height[2:, 1:-1] - height[:-2, 1:-1]
    east = height[1:-1, 2:] - height[1:-1, :-2]
    nx = -east
    ny = np.full(nx.shape, np.float32(2.0), np.float32)
    nz = -south
    scale = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), np.float32(1e-8))
    return np.stack((nx / scale, ny / scale, nz / scale), axis=-1).astype(np.float32)


def generate_planet_tile(state, tx, ty, smooth_passes=1):
    """One tile, including the vertices it shares with the next tile."""
    grid = state.grid
    need = required_halo(smooth_passes)
    if grid.halo < need:
        raise ValueError(
            f"halo {grid.halo} is shorter than {need} for {smooth_passes} smooth passes")
    tx, ty = _tile_index(grid, tx, ty)
    y0, x0 = ty * grid.tile, tx * grid.tile
    gy = np.arange(y0 - grid.halo, y0 + grid.tile + grid.halo + 1)
    gx = np.arange(x0 - grid.halo, x0 + grid.tile + grid.halo + 1)
    work = sample_global_height(gy[:, None], gx[None, :], state)
    bedrock = work[grid.halo:grid.halo + grid.tile + 1,
                   grid.halo:grid.halo + grid.tile + 1].copy()
    for _ in range(int(smooth_passes)):
        work = _smooth_once(work)
    blender = getattr(state, "patches", None)
    if blender is not None:
        cy, cx = canonical_vertices(gy[:, None], gx[None, :], grid.cells_y, grid.cells_x)
        at_pole = (cy == 0) | (cy == grid.cells_y)
        cx = np.where(at_pole, np.int64(0), cx)
        work = np.asarray(work, np.float32) + blender.correction(cy, cx).astype(np.float32)
    core = slice(grid.halo, grid.halo + grid.tile + 1)
    height = np.ascontiguousarray(work[core, core])
    normal = _normals(work[grid.halo - 1:grid.halo + grid.tile + 2,
                           grid.halo - 1:grid.halo + grid.tile + 2])
    return PlanetTile(tx=tx, ty=ty, height=height, normal=normal,
                      bedrock=np.ascontiguousarray(bedrock),
                      sea_level=float(state.sea_level))


class TiledPlanet:
    """One planet cut into output tiles. Patches, when set, are not per-tile windows."""

    def __init__(self, global_world, cfg):
        if hasattr(global_world, "height"):
            coarse = np.asarray(global_world.height, np.float32)
        else:
            coarse = np.asarray(global_world, np.float32)
        if coarse.ndim != 2:
            raise ValueError("global world must be a height raster")
        self.cfg = cfg
        self.state = build_planet(
            int(cfg.seed), int(cfg.world_width), int(cfg.world_height),
            int(cfg.tile_cells), halo=int(cfg.halo), coarse=coarse,
            polar_transition=0)
        from .features import build_features
        grid = self.state.grid
        self.state.features = build_features(
            int(cfg.seed), grid.cells_y, grid.cells_x,
            world_width_m=float(grid.cells_x) * 1000.0)

    def set_features(self, features):
        self.state.features = features

    def set_patch_blender(self, blender):
        grid = self.state.grid
        if blender.world_width != grid.cells_x or blender.world_height != grid.cells_y:
            raise ValueError("patch grid does not match the planet")
        if blender.stride != grid.tile:
            raise ValueError("patch stride must equal the output tile")
        self.state.patches = blender
        self.state.polar_transition = int(blender.polar_transition_cells)
        self.state._polar_model = None

    def tile(self, tx, ty, halo=None, smooth_passes=1):
        grid = self.state.grid
        if halo is not None and int(halo) != grid.halo:
            self.state.grid = PlanetGrid(grid.cells_x, grid.cells_y, grid.tile, int(halo))
        return generate_planet_tile(self.state, tx, ty, smooth_passes=smooth_passes)
