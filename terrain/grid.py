"""Grid helpers. The world wraps in x (cylinder) and clamps in y (poles)."""
import threading
from contextlib import contextmanager

import numpy as np
from scipy import fft as _fft
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as ndi_label

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIST8 = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SHIFT_IDX = {}


def _shift_index(shape, dy, dx):
    key = (shape, dy, dx)
    idx = _SHIFT_IDX.get(key)
    if idx is None:
        h, w = shape
        idx = (np.clip(np.arange(h) + dy, 0, h - 1)[:, None],
               ((np.arange(w) + dx) % w)[None, :])
        _SHIFT_IDX[key] = idx
    return idx


class Bounds:
    """Ghost samples just outside a tile, used only inside `use_bounds`."""

    __slots__ = ("left", "right", "top", "bottom", "clamp_x", "clamp_y")

    def __init__(self):
        self.left = self.right = self.top = self.bottom = None
        self.clamp_x = False
        self.clamp_y = False

_bounds_local = threading.local()


def active_bounds():
    """The bounds context on this thread, or None when the world is a cylinder."""
    return getattr(_bounds_local, "bounds", None)


def _bounds_open(bounds):
    return bounds is not None and (
        bounds.clamp_x or bounds.left is not None or bounds.right is not None
        or bounds.top is not None or bounds.bottom is not None)


@contextmanager
def use_bounds(bounds):
    prev = getattr(_bounds_local, "bounds", None)
    _bounds_local.bounds = bounds
    try:
        yield bounds
    finally:
        _bounds_local.bounds = prev


def shift(a, dy, dx):
    """Sample a[y+dy, x+dx]: wrap in x, replicate the edge row in y."""
    bounds = active_bounds()
    if bounds is None or not bounds.clamp_x:
        return a[_shift_index(a.shape, dy, dx)]
    h, w = a.shape
    yy = np.clip(np.arange(h) + dy, 0, h - 1)
    xx = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[yy[:, None], xx[None, :]]


class Halo:
    """A one-cell padded copy of a grid, so each neighbour offset is a view."""

    def __init__(self, shape, dtype=float):
        h, w = shape
        self.shape = shape
        self.buf = np.empty((h + 2, w + 2), dtype=dtype)

    def load(self, a, outside=None):
        b = self.buf
        b[1:-1, 1:-1] = a
        if outside == "zero":
            b[0, :] = 0
            b[-1, :] = 0
            b[:, 0] = 0
            b[:, -1] = 0
            return self
        bounds = None if outside == "wrap" else active_bounds()
        if not _bounds_open(bounds):
            b[1:-1, 0] = a[:, -1]       # x wraps
            b[1:-1, -1] = a[:, 0]
            b[0, :] = b[1, :]
            b[-1, :] = b[-2, :]
            return self
        if bounds.left is not None:
            b[1:-1, 0] = bounds.left
        elif bounds.clamp_x:
            b[1:-1, 0] = a[:, 0]
        else:
            b[1:-1, 0] = a[:, -1]
        if bounds.right is not None:
            b[1:-1, -1] = bounds.right
        elif bounds.clamp_x:
            b[1:-1, -1] = a[:, -1]
        else:
            b[1:-1, -1] = a[:, 0]
        b[0, :] = b[1, :]
        b[-1, :] = b[-2, :]
        if bounds.top is not None:
            b[0, 1:-1] = bounds.top
        if bounds.bottom is not None:
            b[-1, 1:-1] = bounds.bottom
        return self

    def at(self, dy, dx):
        h, w = self.shape
        return self.buf[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

_NEIGH_IDX = {}


def neighbour_index(shape, dy, dx):
    """Flat index of the wrapped neighbour. Constant per shape and offset, so cached."""
    key = (shape, dy, dx)
    out = _NEIGH_IDX.get(key)
    if out is None:
        h, w = shape
        yy = np.clip(np.arange(h) + dy, 0, h - 1)[:, None]
        xx = ((np.arange(w) + dx) % w)[None, :]
        out = yy * w + xx
        _NEIGH_IDX[key] = out
    return out


def min3x3(a, scratch, out):
    """3x3 neighbourhood minimum (wrap in x, clamp in y), into caller buffers."""
    np.minimum(a[:, :-1], a[:, 1:], out=scratch[:, :-1])
    bounds = active_bounds()
    if bounds is not None and bounds.clamp_x:
        scratch[:, -1] = a[:, -1]
        np.minimum(scratch[:, 1:], a[:, :-1], out=scratch[:, 1:])
    else:
        np.minimum(a[:, -1], a[:, 0], out=scratch[:, -1])
        np.minimum(scratch[:, 1:], a[:, :-1], out=scratch[:, 1:])
        np.minimum(scratch[:, 0], a[:, -1], out=scratch[:, 0])
    np.minimum(scratch[:-1], scratch[1:], out=out[:-1])
    out[-1] = scratch[-1]
    np.minimum(out[1:], scratch[:-1], out=out[1:])
    return out


def latitude(shape):
    """Signed latitude per row, -1 at one pole and +1 at the other."""
    h = shape[0]
    return (np.arange(h) / max(1, h - 1) * 2.0 - 1.0)[:, None]


def gradient(a):
    """Central-difference (dy, dx) with the grid's wrap/clamp conventions."""
    dy = (shift(a, 1, 0) - shift(a, -1, 0)) * 0.5
    dx = (shift(a, 0, 1) - shift(a, 0, -1)) * 0.5
    return dy, dx


def _blur_fft(a, sigma):
    """The same gaussian, multiplied in the frequency domain instead."""
    pad = int(np.ceil(4 * sigma))
    b = np.concatenate([a[:1].repeat(pad, 0), a, a[-1:].repeat(pad, 0)])
    ky = _fft.fftfreq(b.shape[0])[:, None]
    kx = _fft.rfftfreq(b.shape[1])[None, :]
    kernel = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2))
    spec = _fft.rfft2(b, workers=-1)
    spec *= kernel
    return _fft.irfft2(spec, s=b.shape, workers=-1)[pad:pad + a.shape[0]]


def blur(a, sigma):
    if sigma <= 0:
        return a
    a = a.astype(np.float32)
    bounds = active_bounds()
    if bounds is not None and bounds.clamp_x:
        return gaussian_filter(a, sigma, mode="nearest").astype(np.float32, copy=False)
    if sigma >= 12.0:
        return _blur_fft(a, sigma).astype(np.float32, copy=False)
    return gaussian_filter(a, sigma, mode=["nearest", "wrap"])


def edt(mask, return_indices=False):
    """Distance to the nearest False cell of `mask`, wrapping in x."""
    if not np.any(~mask):
        dist = np.full(mask.shape, np.inf, dtype=np.float64)
        if return_indices:
            return dist, (np.full(mask.shape, -1, dtype=np.intp),
                          np.full(mask.shape, -1, dtype=np.intp))
        return dist
    w = mask.shape[1]
    p = (w + 1) // 2
    tiled = np.concatenate([mask[:, -p:], mask, mask[:, :p]], axis=1)
    if not return_indices:
        return distance_transform_edt(tiled)[:, p:p + w]
    dist, idx = distance_transform_edt(tiled, return_indices=True)
    iy = idx[0][:, p:p + w]
    ix = (idx[1][:, p:p + w] - p) % w
    return dist[:, p:p + w], (iy, ix)


def label_wrap_x(mask, connectivity=8):
    """Connected component labeling with x-periodic topology, y clamped."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = (np.ones((3, 3), dtype=np.uint8) if connectivity == 8
                 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    labels, count = ndi_label(np.asarray(mask, bool), structure=structure)
    if count == 0 or mask.shape[1] == 1:
        return labels, count
    parent = np.arange(count + 1)
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a, b):
        a, b = root(int(a)), root(int(b))
        if a != b:
            parent[max(a,b)] = min(a,b)
    h = labels.shape[0]
    shifts = (-1, 0, 1) if connectivity == 8 else (0,)
    for y in range(h):
        a = labels[y, 0]
        if not a:
            continue
        for dy in shifts:
            ny = y + dy
            if 0 <= ny < h:
                b = labels[ny, -1]
                if b:
                    union(a, b)
    roots = np.array([root(i) for i in range(count + 1)], dtype=np.int32)
    unique = np.unique(roots[1:])
    lut = np.zeros(count + 1, dtype=np.int32)
    lut[unique] = np.arange(1, len(unique) + 1)
    return lut[roots[labels]], int(len(unique))


def normalize(a, lo=0.0, hi=1.0):
    amin, amax = float(a.min()), float(a.max())
    if amax - amin < 1e-12:
        return np.full_like(a, lo, dtype=float)
    return lo + (hi - lo) * (a - amin) / (amax - amin)


def sample_raster(field, y, x, extent_y, extent_x, wrap_x=True):
    """Bilinear sample of a macro raster covering `[0, extent)` metres."""
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("sample_raster expects a 2D field")
    extent_y = float(extent_y)
    extent_x = float(extent_x)
    if extent_y <= 0 or extent_x <= 0:
        raise ValueError("extents must be positive")
    yy, xx = np.broadcast_arrays(np.asarray(y, np.float64), np.asarray(x, np.float64))
    height, width = field.shape
    fy = yy / extent_y * height - 0.5
    fx = xx / extent_x * width - 0.5
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    ty = (fy - y0).astype(np.float32)
    tx = (fx - x0).astype(np.float32)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)
    if wrap_x:
        x0c = np.mod(x0, width)
        x1c = np.mod(x0 + 1, width)
    else:
        x0c = np.clip(x0, 0, width - 1)
        x1c = np.clip(x0 + 1, 0, width - 1)
    values = field.astype(np.float32, copy=False)
    v00 = values[y0c, x0c]
    v10 = values[y0c, x1c]
    v01 = values[y1c, x0c]
    v11 = values[y1c, x1c]
    return (v00 + (v10 - v00) * tx + (v01 - v00) * ty
            + (v00 - v10 - v01 + v11) * tx * ty).astype(np.float32)


def vertex_axes(origin_y, origin_x, rows, cols, dy, dx):
    """World-metre coordinates of a vertex grid, broadcast-ready as (y, x)."""
    y = np.float64(origin_y) + np.arange(rows, dtype=np.float64) * np.float64(dy)
    x = np.float64(origin_x) + np.arange(cols, dtype=np.float64) * np.float64(dx)
    return y[:, None], x[None, :]


def stamp(arr, y, x, amp, sigma):
    """Add a gaussian bump centred at (y, x), wrapping in x."""
    h, w = arr.shape
    r = int(max(2, sigma * 3))
    ys = np.arange(int(round(y)) - r, int(round(y)) + r + 1)
    xs = np.arange(int(round(x)) - r, int(round(x)) + r + 1)
    keep = (ys >= 0) & (ys < h)
    ys = ys[keep]
    if ys.size == 0:
        return
    dy = (ys - y)[:, None]
    dx = (xs - x)[None, :]
    bump = amp * np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    np.add.at(arr, (ys[:, None], xs[None, :] % w), bump)
