"""Globally anchored corrections, blended so a tile edge is not a terrain edge.

A generation patch lives on the fixed stride grid. Its origin is
``index * stride`` in global cells, so every output tile that overlaps it
reads the same correction, whether or not the neighbor tile exists. Weights
are a quintic window: value, slope, and curvature are all zero where a
patch's influence ends, and the next patch takes over. Nothing is faded to
zero on an output-tile boundary.

Blending independently eroded surfaces does not conserve water or sediment.
A river, glacier, or other connected feature has to exist in the global
field first; the patches only refine it.
"""
from pathlib import Path

import numpy as np


def smootherstep(t):
    """Quintic smootherstep. Value, first, and second derivatives match 0 and 1."""
    u = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def quintic_window(t):
    """Patch weight on ``[0, 1]``. Flat at both ends and at the midpoint."""
    t = np.asarray(t, np.float64)
    rising = smootherstep(np.clip(t * 2.0, 0.0, 1.0))
    falling = smootherstep(np.clip(2.0 - 2.0 * t, 0.0, 1.0))
    window = np.where(t < 0.5, rising, falling)
    return np.where((t > 0.0) & (t < 1.0), window, np.zeros(np.shape(t), np.float64))


def continue_terrain_trend(distance, width, height_start, slope_start,
                           height_end, slope_end):
    """Cubic Hermite join. ``slope_*`` are dH/dd, and ``width`` is the run of the join."""
    width = float(width)
    if width <= 0.0:
        raise ValueError("transition width must be positive")
    t = np.clip(np.asarray(distance, np.float64) / width, 0.0, 1.0)
    h00 = (2.0 * t - 3.0) * t * t + 1.0
    h10 = (t - 2.0) * t * t + t
    h01 = (-2.0 * t + 3.0) * t * t
    h11 = (t - 1.0) * t * t
    return (h00 * height_start + h10 * width * slope_start
            + h01 * height_end + h11 * width * slope_end)


def continue_terrain_trend_quintic(distance, width, height_start, slope_start, curve_start,
                                   height_end, slope_end, curve_end):
    """Quintic Hermite join. Also matches d²H/dd² at each end."""
    width = float(width)
    if width <= 0.0:
        raise ValueError("transition width must be positive")
    t = np.clip(np.asarray(distance, np.float64) / width, 0.0, 1.0)
    h00 = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
    h10 = t - 6.0 * t ** 3 + 8.0 * t ** 4 - 3.0 * t ** 5
    h20 = 0.5 * t ** 2 - 1.5 * t ** 3 + 1.5 * t ** 4 - 0.5 * t ** 5
    h01 = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
    h11 = -4.0 * t ** 3 + 7.0 * t ** 4 - 3.0 * t ** 5
    h21 = 0.5 * t ** 3 - t ** 4 + 0.5 * t ** 5
    span = width * width
    return (h00 * height_start + h10 * width * slope_start + h20 * span * curve_start
            + h01 * height_end + h11 * width * slope_end + h21 * span * curve_end)


def example_patch_sampler(stride, world_width, seed, support=None, amplitude=1.0):
    """A canyon centered in each patch, plus one global wave.

    The canyon center sits on the stride line between output tiles, so the
    feature is deepest on the tile edge instead of dying there. Replace this
    with a sampler that returns corrections from the real erosion solver.
    """
    stride = int(stride)
    support = int(2 * stride if support is None else support)
    world_width = float(world_width)
    amplitude = float(amplitude)
    seed = int(seed)

    def sampler(py, px):
        py, px = int(py), int(px)
        local_y = np.arange(support + 1, dtype=np.float64)[:, None]
        local_x = np.arange(support + 1, dtype=np.float64)[None, :]
        center = support * 0.5
        spread = support * 0.22
        canyon = np.exp(-((local_y - center) / spread) ** 2 - ((local_x - center) / spread) ** 2)
        phase = ((py * 17 + px * 31 + seed) & 1023) / 1023.0
        depth = amplitude * (0.65 + 0.35 * phase)
        trend = 0.15 * amplitude * np.sin(
            (2.0 * np.pi) * (px * stride + local_x) / world_width)
        return (trend - depth * canyon).astype(np.float32)

    return sampler


def _sample_grid(field, y, x):
    """Bilinear sample of a vertex grid. Coordinates are in grid index units."""
    field = np.asarray(field, np.float64)
    y = np.clip(np.asarray(y, np.float64), 0.0, field.shape[0] - 1)
    x = np.clip(np.asarray(x, np.float64), 0.0, field.shape[1] - 1)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, field.shape[0] - 1)
    x1 = np.minimum(x0 + 1, field.shape[1] - 1)
    ty = (y - y0).astype(np.float64)
    tx = (x - x0).astype(np.float64)
    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]
    return v00 + (v10 - v00) * tx + (v01 - v00) * ty + (v00 - v10 - v01 + v11) * tx * ty


class PatchBlender:
    """Partition-of-unity blend of patches anchored at ``index * stride``."""

    def __init__(self, world_width, world_height, stride, sampler,
                 polar_transition_cells=0, support=None, cache_dir=None):
        self.world_width = int(world_width)
        self.world_height = int(world_height)
        self.stride = int(stride)
        self.support = int(2 * self.stride if support is None else support)
        self.sampler = sampler
        self.polar_transition_cells = int(polar_transition_cells)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._memory = {}
        if self.stride < 2:
            raise ValueError("stride must be at least 2")
        if self.support <= self.stride:
            raise ValueError("patch support must be wider than the stride")
        if self.world_width % self.stride or self.world_height % self.stride:
            raise ValueError("world cells must divide into whole patch strides")
        if self.world_width < self.support or self.world_height < 2:
            raise ValueError("world is smaller than one patch")
        if self.polar_transition_cells < 0:
            raise ValueError("polar transition must be nonnegative")

    @property
    def patches_x(self):
        return self.world_width // self.stride

    @property
    def patches_y(self):
        return self.world_height // self.stride

    def patch_ids(self, tx, ty, tile=None):
        """Canonical ``(py, px)`` pairs whose support meets output tile ``(tx, ty)``."""
        tile = self.stride if tile is None else int(tile)
        y0, y1 = ty * tile, ty * tile + tile
        x0, x1 = tx * tile, tx * tile + tile
        pairs = []
        for py in self._py_list(y0, y1):
            for px in self._px_list(x0, x1):
                pairs.append((py, px))
        return tuple(pairs)

    def correction(self, gy, gx):
        """Blended ``ΔH`` at vertex coordinates. Missing weight leaves the correction at 0."""
        gy, gx = np.broadcast_arrays(np.asarray(gy, np.float64), np.asarray(gx, np.float64))
        num = np.zeros(gy.shape, np.float64)
        den = np.zeros(gy.shape, np.float64)
        y0, y1 = float(np.min(gy)), float(np.max(gy))
        x0, x1 = float(np.min(gx)), float(np.max(gx))
        for py in self._py_list(y0, y1):
            origin_y = py * self.stride
            local_y = gy - origin_y
            wy = quintic_window(local_y / self.support)
            if not np.any(wy > 0.0):
                continue
            for px in self._px_list(x0, x1):
                origin_x = px * self.stride
                local_x = np.mod(gx - origin_x, self.world_width)
                wx = quintic_window(local_x / self.support)
                weight = wy * wx
                if not np.any(weight > 1e-12):
                    continue
                field = self._patch(py, px)
                num += weight * _sample_grid(field, local_y, local_x)
                den += weight
        out = np.zeros(gy.shape, np.float64)
        known = den > 1e-12
        out[known] = num[known] / den[known]
        return out

    def apply(self, gy, gx, global_height):
        """``H_global + Σ w ΔH / Σ w``."""
        base = np.asarray(global_height, np.float64)
        return (base + self.correction(gy, gx)).astype(np.float32)

    def weight_sum(self, gy, gx):
        """Sum of the un-normalized windows. Interior points of a covered planet are 1."""
        gy, gx = np.broadcast_arrays(np.asarray(gy, np.float64), np.asarray(gx, np.float64))
        total = np.zeros(gy.shape, np.float64)
        y0, y1 = float(np.min(gy)), float(np.max(gy))
        x0, x1 = float(np.min(gx)), float(np.max(gx))
        for py in self._py_list(y0, y1):
            wy = quintic_window((gy - py * self.stride) / self.support)
            if not np.any(wy > 0.0):
                continue
            for px in self._px_list(x0, x1):
                local_x = np.mod(gx - px * self.stride, self.world_width)
                total += wy * quintic_window(local_x / self.support)
        return total

    def _py_list(self, y0, y1):
        first = int(np.floor((y0 - self.support) / self.stride)) - 1
        last = int(np.floor(y1 / self.stride)) + 1
        found = []
        for py in range(first, last + 1):
            origin = py * self.stride
            if origin + self.support <= 0 or origin >= self.world_height:
                continue
            found.append(py)
        return found

    def _px_list(self, x0, x1):
        first = int(np.floor((x0 - self.support) / self.stride)) - 1
        last = int(np.floor(x1 / self.stride)) + 1
        found = []
        seen = set()
        count = self.patches_x
        for px in range(first, last + 1):
            canon = px % count
            if canon in seen:
                continue
            seen.add(canon)
            found.append(canon)
        return found

    def _patch(self, py, px):
        key = (int(py), int(px) % self.patches_x)
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        path = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"patch_{key[0]}_{key[1]}.npy"
            if path.is_file():
                field = np.load(path)
                self._memory[key] = field
                return field
        field = np.asarray(self.sampler(key[0], key[1]), np.float64)
        expect = (self.support + 1, self.support + 1)
        if field.shape != expect:
            raise ValueError(f"patch {key} has shape {field.shape}, expected {expect}")
        self._memory[key] = field
        if path is not None:
            np.save(path, field)
        return field
