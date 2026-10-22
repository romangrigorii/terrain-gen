"""Perlin/fBm noise on the world grid, tileable in x."""
import zlib

import numpy as np
from scipy.ndimage import map_coordinates
from contextvars import ContextVar
from contextlib import contextmanager

_BACKEND = ContextVar("terrain_noise_backend", default="numpy")
_FAST = None


@contextmanager
def use_backend(backend="numpy"):
    """Select a backend only within the current run/task; no global RNG state."""
    token = _BACKEND.set(backend)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def _accelerated_kernel():
    """Lazy Numba import: default NumPy path needs no JIT installed."""
    global _FAST
    if _FAST is not None:
        return _FAST
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("Install terrain-world-generator-upgrade[accel] for the numba noise backend") from exc

    @njit(cache=True)
    def sample(h, w, py, px, gy, gx):
        arr = np.empty((h, w), dtype=np.float32)
        for y in range(h):
            oy = y * (float(py) / h)
            iy = int(oy)
            fy = oy - iy
            v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
            for x in range(w):
                ox = x * (float(px) / w)
                ix = int(ox)
                fx = ox - ix
                u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
                n00 = gy[iy,ix] * fy + gx[iy,ix] * fx
                n01 = gy[iy,ix+1] * fy + gx[iy,ix+1] * (fx-1)
                n10 = gy[iy+1,ix] * (fy-1) + gx[iy+1,ix] * fx
                n11 = gy[iy+1,ix+1] * (fy-1) + gx[iy+1,ix+1] * (fx-1)
                lo = n00 + (n01 - n00) * u
                hi = n10 + (n11 - n10) * u
                arr[y,x] = (lo + (hi-lo) * v) * 1.42
        return arr
    _FAST = sample
    return sample


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(h, w, py, px, rng):
    """Gradient noise, py by px lattice cells, seamless across the x seam."""
    py, px = max(1, int(py)), max(1, int(px))
    ang = rng.uniform(0, 2 * np.pi, (py + 1, px + 1))
    gy, gx = np.sin(ang, dtype=np.float32), np.cos(ang, dtype=np.float32)
    gy[:, -1] = gy[:, 0]  # seam
    gx[:, -1] = gx[:, 0]
    if _BACKEND.get() == "numba":
        return _accelerated_kernel()(h, w, py, px, gy, gx)
    ys = np.linspace(0, py, h, endpoint=False)
    xs = np.linspace(0, px, w, endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    def dot(dy, dx):
        """Corner gradient dotted with the offset to it, in place."""
        iy, ix = y0 + dy, x0 + dx
        out = gy[iy][:, ix]
        out *= fy - dy
        other = gx[iy][:, ix]
        other *= fx - dx
        out += other
        return out

    u, v = _fade(fx), _fade(fy)
    d00, d01, d10, d11 = dot(0, 0), dot(0, 1), dot(1, 0), dot(1, 1)
    for lo, hi, t in ((d00, d01, u), (d10, d11, u), (d01, d11, v)):
        np.subtract(hi, lo, out=hi)
        np.multiply(hi, t, out=hi)
        np.add(hi, lo, out=hi)          # hi now holds lerp(lo, hi, t)
    return np.multiply(d11, 1.42, out=d11)


def fbm(h, w, rng, periods=4, octaves=6, gain=0.5, ridged=False):
    """Fractal sum of perlin octaves. Ridged variant gives sharp crests."""
    total = np.zeros((h, w), np.float32)
    amp, norm, p = 1.0, 0.0, float(periods)
    aspect = h / w
    for _ in range(octaves):
        n = perlin(h, w, round(p * aspect), round(p), rng)
        if ridged:
            n = 1.0 - np.abs(n)
            n = n * n * 2.0 - 1.0
        total += amp * n
        norm += amp
        amp *= gain
        p *= 2.0
    return total / norm


def stage_rng(seed, stage, tx=0, ty=0):
    """Generator owned by one stage and, when set, one tile."""
    if isinstance(stage, str):
        stage_id = zlib.crc32(stage.encode("utf-8")) & 0xFFFFFFFF
    else:
        stage_id = int(stage) & 0xFFFFFFFF
    sequence = np.random.SeedSequence([
        int(seed) & 0xFFFFFFFF,
        stage_id,
        int(tx) & 0xFFFFFFFF,
        int(ty) & 0xFFFFFFFF,
    ])
    return np.random.default_rng(sequence)


def _hash2(iy, ix, seed):
    iy = np.asarray(iy, np.int64)
    ix = np.asarray(ix, np.int64)
    mixed = (iy * np.int64(374761393) + ix * np.int64(668265263)
             + np.int64(seed) * np.int64(1442695041))
    mixed &= np.int64(0xFFFFFFFF)
    mixed ^= mixed >> np.int64(13)
    mixed = (mixed * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    return (mixed & np.int64(0xFFFFFF)).astype(np.float32) * np.float32(1.0 / 0xFFFFFF)


def value_noise_world(y_m, x_m, wavelength, seed, world_width_m=None):
    """Smooth value noise at absolute metres. Adjacent tiles evaluate the same field."""
    yy, xx = np.broadcast_arrays(np.asarray(y_m, np.float64), np.asarray(x_m, np.float64))
    period = float(max(wavelength, 1e-6))
    y = yy / period
    x = xx / period
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    if world_width_m is not None and world_width_m > 0:
        cells = max(1, int(round(float(world_width_m) / period)))
        x0 = np.mod(x0, cells).astype(np.int64)
        x1 = np.mod(x0 + 1, cells).astype(np.int64)
    else:
        x1 = x0 + 1
    fy = (y - np.floor(y)).astype(np.float32)
    fx = (x - np.floor(x)).astype(np.float32)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)
    n00 = _hash2(y0, x0, seed)
    n10 = _hash2(y0, x1, seed)
    n01 = _hash2(y0 + 1, x0, seed)
    n11 = _hash2(y0 + 1, x1, seed)
    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return (nx0 + (nx1 - nx0) * fy).astype(np.float32)


def fbm_world(y_m, x_m, wavelength, octaves=4, seed=0, gain=0.5, ridged=False,
              world_width_m=None):
    """Fractal noise in metres. Wavelength is a physical length, not a pixel period."""
    total = np.zeros(np.broadcast_shapes(np.shape(y_m), np.shape(x_m)), np.float32)
    amplitude, weight, length = 1.0, 0.0, float(wavelength)
    for octave in range(int(octaves)):
        value = value_noise_world(y_m, x_m, length, int(seed) + octave * 101, world_width_m)
        if ridged:
            value = 1.0 - np.abs(value * 2.0 - 1.0)
        total += np.float32(amplitude) * value
        weight += amplitude
        amplitude *= gain
        length *= 0.5
    return total / np.float32(weight)


def warp(field, dy, dx):
    """Resample `field` at (y+dy, x+dx); x wraps, y clamps."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([np.clip(yy + dy, 0, h - 1), (xx + dx) % w])
    return map_coordinates(field, coords, order=1, mode="grid-wrap")
