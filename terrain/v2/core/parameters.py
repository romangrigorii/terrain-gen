"""Spatial parameters. Solvers sample these instead of reading one global number."""
import numpy as np

from .grid import Grid


def _fade(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _hash2(iy, ix, seed):
    n = (np.asarray(iy, np.int64) * np.int64(374761393)
         + np.asarray(ix, np.int64) * np.int64(668265263)
         + np.int64(int(seed)) * np.int64(1442695041))
    n = n & np.int64(0x7FFFFFFF)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177) & np.int64(0x7FFFFFFF)
    return (n & np.int64(0xFFFFFF)) / float(0xFFFFFF)


class ConstantField:
    def __init__(self, value):
        self.value = float(value)

    def sample(self, x_m, y_m, time=0.0):
        return self.value

    def sample_grid(self, grid, time=0.0):
        return np.full(grid.shape, self.value, np.float32)


class RasterField:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32, copy=True)

    def sample_grid(self, grid, time=0.0):
        if self.data.shape != grid.shape:
            raise ValueError(f"raster is {self.data.shape}, grid is {grid.shape}")
        return self.data.copy()


class NoiseField:
    def __init__(self, frequency, amplitude, seed=0):
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.seed = int(seed)

    def _value(self, y_m, x_m):
        scale = self.frequency
        yf = np.asarray(y_m, np.float64) * scale
        xf = np.asarray(x_m, np.float64) * scale
        y0 = np.floor(yf).astype(np.int64)
        x0 = np.floor(xf).astype(np.int64)
        sy = _fade(yf - y0)
        sx = _fade(xf - x0)
        v00 = _hash2(y0, x0, self.seed)
        v10 = _hash2(y0, x0 + 1, self.seed)
        v01 = _hash2(y0 + 1, x0, self.seed)
        v11 = _hash2(y0 + 1, x0 + 1, self.seed)
        along = v00 * (1.0 - sx) + v10 * sx
        below = v01 * (1.0 - sx) + v11 * sx
        return self.amplitude * (along * (1.0 - sy) + below * sy)

    def sample(self, x_m, y_m, time=0.0):
        return float(self._value(y_m, x_m))

    def sample_grid(self, grid, time=0.0):
        yy, xx = grid.coordinates()
        return self._value(yy, xx).astype(np.float32)


class RadialField:
    def __init__(self, center, radius, amplitude, falloff):
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)
        self.amplitude = float(amplitude)
        self.falloff = float(falloff)

    def _weights(self, y_m, x_m):
        cy, cx = self.center
        distance = np.hypot(np.asarray(y_m) - cy, np.asarray(x_m) - cx)
        if self.falloff <= 0:
            return self.amplitude * (distance <= self.radius)
        edge = _fade((distance - self.radius) / self.falloff)
        return self.amplitude * (1.0 - edge)

    def sample(self, x_m, y_m, time=0.0):
        return float(self._weights(y_m, x_m))

    def sample_grid(self, grid, time=0.0):
        yy, xx = grid.coordinates()
        return self._weights(yy, xx).astype(np.float32)


class GradientField:
    """Varies from `lo` to `hi` along one axis. Axis 0 is north to south."""

    def __init__(self, lo, hi, axis=0):
        self.lo = float(lo)
        self.hi = float(hi)
        self.axis = int(axis)

    def sample_grid(self, grid, time=0.0):
        span = grid.rows if self.axis == 0 else grid.cols
        ramp = np.linspace(self.lo, self.hi, span, dtype=np.float64)
        if self.axis == 0:
            data = np.repeat(ramp[:, None], grid.cols, axis=1)
        else:
            data = np.repeat(ramp[None, :], grid.rows, axis=0)
        return data.astype(np.float32)


class BlendField:
    def __init__(self, fields, weights):
        if len(fields) != len(weights) or not fields:
            raise ValueError("blend needs one weight for each field")
        self.fields = list(fields)
        self.weights = list(weights)

    def sample_grid(self, grid, time=0.0):
        from .masks import normalize_weights
        values = [np.asarray(field.sample_grid(grid, time), np.float64)
                  for field in self.fields]
        weights = []
        for weight in self.weights:
            if hasattr(weight, "sample_grid"):
                weights.append(np.asarray(weight.sample_grid(grid, time), np.float64))
            else:
                weights.append(np.broadcast_to(np.asarray(weight, np.float64), grid.shape))
        mixed = np.zeros(grid.shape, np.float64)
        for value, weight in zip(values, normalize_weights(weights)):
            mixed += value * weight
        return mixed.astype(np.float32)


def as_grid(shape, cell_m):
    return Grid(tuple(shape), float(cell_m))
