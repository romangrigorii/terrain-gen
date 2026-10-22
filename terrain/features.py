"""Global landforms. A tile evaluates them. It does not invent its own.

Ridges, valleys, and rivers are polylines on the planet. The same feature id,
elevation profile, and falloff apply in every tile, in any order. Distance is
the great-circle distance, so the curve does not stop on a tile edge.
"""
from dataclasses import dataclass

import numpy as np

from .export import METRES_PER_UNIT


@dataclass
class TerrainFeature:
    feature_id: int
    kind: str
    points: np.ndarray
    elevation_m: np.ndarray
    width_m: float
    influence_m: float
    slope_constraints: np.ndarray = None


class FeatureSet:
    """One planet's ridges, valleys, and rivers."""

    def __init__(self, features, cells_y, cells_x, world_width_m):
        self.features = tuple(features)
        self.cells_y = int(cells_y)
        self.cells_x = int(cells_x)
        self.world_width_m = float(world_width_m)
        if self.cells_y < 2 or self.cells_x < 2:
            raise ValueError("feature grid must be at least 2 cells")
        if self.world_width_m <= 0:
            raise ValueError("world_width_m must be positive")
        self._xyz = []
        for feature in self.features:
            pts = np.asarray(feature.points, np.float64)
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
                raise ValueError(f"feature {feature.feature_id} needs a polyline")
            elev = np.asarray(feature.elevation_m, np.float64)
            if elev.shape != (pts.shape[0],):
                raise ValueError(f"feature {feature.feature_id} elevation does not match its points")
            self._xyz.append(_xyz(pts[:, 0], pts[:, 1], self.cells_y, self.cells_x))

    def evaluate(self, gy, gx):
        """Height contribution, in the world's height units, at vertex indices."""
        gy, gx = np.broadcast_arrays(np.asarray(gy, np.float64), np.asarray(gx, np.float64))
        point = _xyz(gy, gx, self.cells_y, self.cells_x)
        total = np.zeros(gy.shape, np.float64)
        for feature, xyz in zip(self.features, self._xyz):
            total += _feature_height(point, xyz, feature, self.world_width_m)
        return total / METRES_PER_UNIT

    def drainage(self):
        """Shared river outlets. A tile reads these. It does not choose a new one."""
        rows = []
        for feature in self.features:
            if feature.kind != "river":
                continue
            outlet = np.asarray(feature.points[-1], np.float64)
            rows.append({
                "feature_id": int(feature.feature_id),
                "outlet_y": float(outlet[0]),
                "outlet_x": float(outlet[1]),
                "basin": int(feature.feature_id),
            })
        return tuple(rows)

    def crossing(self, y0, y1, x0, x1):
        """Features whose polyline meets the cell window, wrapping in longitude."""
        found = []
        for feature in self.features:
            pts = np.asarray(feature.points, np.float64)
            y = pts[:, 0]
            x = np.mod(pts[:, 1], self.cells_x)
            hit = (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)
            if np.any(hit):
                found.append(feature)
        return tuple(found)


def build_features(seed, cells_y, cells_x, world_width_m, n_ridges=3, n_rivers=2):
    """Deterministic landforms for one seed. The same seed rebuilds the same ids."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    features = []
    next_id = 1
    span = max(4, int(n_ridges) + int(n_rivers))
    for index in range(max(0, int(n_ridges))):
        y = float(rng.uniform(0.2, 0.8) * cells_y)
        x = float(rng.uniform(0.0, cells_x))
        step = cells_x / span
        pts = [(y + rng.normal(0, cells_y * 0.02), x + k * step) for k in range(span)]
        elev = rng.uniform(1800.0, 4200.0, size=len(pts))
        features.append(_make(next_id, "ridge", pts, elev, 0.045 * world_width_m, 0.09 * world_width_m))
        next_id += 1
    for index in range(max(0, int(n_rivers))):
        y = float(rng.uniform(0.25, 0.75) * cells_y)
        x = float(rng.uniform(0.0, cells_x * 0.25))
        step = cells_x / (span + 1)
        pts = [(y + 0.15 * k, x + k * step) for k in range(span)]
        elev = np.linspace(900.0, 40.0, len(pts))
        features.append(_make(next_id, "river", pts, elev, 0.012 * world_width_m, 0.03 * world_width_m))
        next_id += 1
    return FeatureSet(features, cells_y, cells_x, world_width_m)


def _make(feature_id, kind, points, elevation_m, width_m, influence_m):
    pts = np.asarray(points, np.float64)
    elev = np.asarray(elevation_m, np.float64)
    slope = np.zeros(elev.shape, np.float64)
    if elev.size > 1:
        slope[:-1] = np.diff(elev)
        slope[-1] = slope[-2]
    return TerrainFeature(int(feature_id), kind, pts, elev, float(width_m), float(influence_m), slope)


def _xyz(gy, gx, cells_y, cells_x):
    phi = np.asarray(gy, np.float64) / float(cells_y) * np.pi - np.pi / 2.0
    lam = np.asarray(gx, np.float64) / float(cells_x) * (2.0 * np.pi)
    cos_phi = np.cos(phi)
    return np.stack((cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)), axis=-1)


def _feature_height(point, xyz, feature, world_width_m):
    elev = np.asarray(feature.elevation_m, np.float64)
    sigma = max(float(feature.width_m) / world_width_m * np.pi, 1e-4)
    reach = max(float(feature.influence_m) / world_width_m * np.pi, sigma)
    best = np.full(point.shape[:-1], np.inf, np.float64)
    amp = np.zeros(point.shape[:-1], np.float64)
    for i in range(xyz.shape[0] - 1):
        dist, weight = _segment(point, xyz[i], xyz[i + 1])
        nearer = dist < best
        best = np.where(nearer, dist, best)
        mixed = (1.0 - weight) * elev[i] + weight * elev[i + 1]
        amp = np.where(nearer, mixed, amp)
    field = np.where(best <= reach, amp * np.exp(-0.5 * (best / sigma) ** 2), 0.0)
    if feature.kind in ("valley", "river"):
        field = -np.abs(field)
    return field


def _segment(point, a, b):
    """Angular distance from each point to the great-circle arc, and the arc fraction."""
    n = np.cross(a, b)
    nlen = float(np.linalg.norm(n))
    pa = np.clip(np.sum(point * a, axis=-1), -1.0, 1.0)
    pb = np.clip(np.sum(point * b, axis=-1), -1.0, 1.0)
    da = np.arccos(pa)
    db = np.arccos(pb)
    ends = np.minimum(da, db)
    if nlen < 1e-8:
        pick = (da <= db).astype(np.float64)
        return ends, pick
    n = n / nlen
    proj = point - n * np.sum(point * n, axis=-1)[..., None]
    proj = proj / np.maximum(np.linalg.norm(proj, axis=-1, keepdims=True), 1e-12)
    ab = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    at = np.arccos(np.clip(np.sum(proj * a, axis=-1), -1.0, 1.0))
    side = np.sum(np.cross(a, proj) * n, axis=-1)
    on = (side >= -1e-7) & (at <= ab + 1e-7) & (ab > 1e-8)
    gap = np.arccos(np.clip(np.sum(point * proj, axis=-1), -1.0, 1.0))
    dist = np.where(on, gap, ends)
    frac = np.where(on, np.clip(at / max(ab, 1e-8), 0.0, 1.0), (da > db).astype(np.float64))
    return dist, frac
