"""Stage 1: plates, plate motion, and the strain fields along their boundaries."""
from dataclasses import dataclass

import numpy as np
from . import grid, noise


@dataclass
class Tectonics:
    plate: np.ndarray        # int plate id per cell
    seeds: np.ndarray        # (n, 2) plate centres, (y, x)
    vel: np.ndarray          # (n, 2) plate velocity, (vy, vx)
    plate_cont: np.ndarray   # (n,) bool, continental plate
    buoyancy: np.ndarray     # (n,) float, higher plate overrides at subduction
    crust: np.ndarray        # continentality per cell, 0 oceanic .. 1 cratonic
    is_cont: np.ndarray      # bool map, crust > 0.5
    vy: np.ndarray
    vx: np.ndarray
    div: np.ndarray          # smoothed divergence map
    curl: np.ndarray         # smoothed curl map
    boundary: np.ndarray     # bool map, cells adjacent to another plate
    dist: np.ndarray         # distance to nearest boundary cell
    conv: np.ndarray         # convergence rate of the nearest boundary  [0, 1]
    rift: np.ndarray         # extension rate of the nearest boundary    [0, 1]
    shear: np.ndarray        # shear rate of the nearest boundary        [0, 1]
    cont_self: np.ndarray    # nearest boundary: this side continental?
    cont_other: np.ndarray   # nearest boundary: far side continental?
    overriding: np.ndarray   # nearest boundary: is this side the overrider?


def _seed_points(n, h, w, rng):
    """Blue-noise-ish plate centres: rejection sampling, relaxing on failure."""
    rmin = 0.75 * np.sqrt(h * w / n)
    pts = []
    for _ in range(n * 200):
        if len(pts) >= n:
            break
        p = np.array([rng.uniform(0, h), rng.uniform(0, w)])
        if pts:
            q = np.array(pts)
            dy = np.abs(q[:, 0] - p[0])
            dx = np.abs(q[:, 1] - p[1])
            dx = np.minimum(dx, w - dx)  # x wraps
            if np.min(np.hypot(dy, dx)) < rmin:
                rmin *= 0.998
                continue
        pts.append(p)
    while len(pts) < n:
        pts.append(np.array([rng.uniform(0, h), rng.uniform(0, w)]))
    return np.array(pts)


def warp_field(h, w, rng, periods, amp, octaves=4):
    """A (dy, dx) displacement pair of fBm, as a 2-vector stack."""
    return np.stack([noise.fbm(h, w, rng, periods=periods, octaves=octaves) * amp,
                     noise.fbm(h, w, rng, periods=periods, octaves=octaves) * amp])


def _plate_map(seeds, reach, h, w, rng, cfg):
    """Additively weighted Voronoi, with the lookup coordinates noise-warped."""
    wy, wx = (warp_field(h, w, rng, cfg.plate_warp_periods, cfg.plate_warp) +
              warp_field(h, w, rng, cfg.plate_warp_periods * 4, cfg.plate_warp_fine))
    yy, xx = np.mgrid[0:h, 0:w]
    qy = np.clip(yy + wy, 0, h - 1)
    qx = (xx + wx) % w

    plate = np.zeros((h, w), dtype=int)
    best = np.full((h, w), np.inf)
    for i, (sy, sx) in enumerate(seeds):
        dx = np.abs(qx - sx)
        d = np.hypot(qy - sy, np.minimum(dx, w - dx)) - reach[i]
        take = d < best
        best[take] = d[take]
        plate[take] = i
    return plate


def _gate(x, lo, hi):
    """Suppress weak strain entirely, pass strong strain through unchanged."""
    t = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    return x * t * t * (3 - 2 * t)


def _crust(plate, plate_cont, cfg, rng):
    """Continuous continentality."""
    h, w = plate.shape
    n1 = noise.fbm(h, w, rng, periods=cfg.crust_periods, octaves=6)
    n2 = noise.fbm(h, w, rng, periods=cfg.crust_periods * 3, octaves=5)
    raw = plate_cont[plate] * cfg.crust_plate_weight + 0.75 * n1 + 0.35 * n2
    raw = noise.warp(raw, *warp_field(h, w, rng, cfg.crust_warp_periods, cfg.crust_warp))
    raw = grid.blur(raw, cfg.crust_blur)

    lat = np.abs(np.linspace(-1.0, 1.0, h))[:, None]
    lat = lat + noise.fbm(h, w, rng, periods=cfg.polar_wobble_periods,
                          octaves=4) * cfg.polar_wobble
    t = np.clip((lat - cfg.polar_start) / max(1e-6, 1.0 - cfg.polar_start), 0, 1)
    raw = raw - cfg.polar_ocean * raw.std() * (t * t * (3 - 2 * t)) ** 1.5

    thr = float(np.quantile(raw, 1.0 - cfg.cont_fraction))
    return np.clip(0.5 + 0.5 * np.tanh(cfg.crust_sharpness *
                                       (raw - thr) / (raw.std() + 1e-9)), 0, 1)


def _neighbour_plate(plate):
    """For every cell, a neighbouring cell's plate id if it differs, else -1."""
    other = np.full(plate.shape, -1, dtype=int)
    for dy, dx in grid.NEIGH4:
        nb = grid.shift(plate, dy, dx)
        take = (nb != plate) & (other < 0)
        other[take] = nb[take]
    return other


def build(cfg, rng):
    h, w = cfg.height, cfg.width
    n = cfg.n_plates

    seeds = _seed_points(n, h, w, rng)
    spacing = np.sqrt(h * w / n)
    reach = rng.uniform(-1.0, 1.0, n) * cfg.plate_size_var * spacing
    plate = _plate_map(seeds, reach, h, w, rng, cfg)

    plate_cont = rng.random(n) < cfg.cont_plate_fraction
    buoyancy = np.where(plate_cont, 1.0, 0.0) + rng.random(n) * 0.3

    ang = rng.uniform(0, 2 * np.pi, n)
    speed = rng.uniform(0.4, 1.0, n)
    vel = np.column_stack([np.sin(ang), np.cos(ang)]) * speed[:, None]

    crust = _crust(plate, plate_cont, cfg, rng)
    is_cont = crust > 0.5
    vy, vx = vel[plate, 0], vel[plate, 1]

    dvy_dy, dvy_dx = grid.gradient(vy)
    dvx_dy, dvx_dx = grid.gradient(vx)
    div = grid.blur(dvx_dx + dvy_dy, cfg.stress_blur)
    curl = grid.blur(dvy_dx - dvx_dy, cfg.stress_blur)

    other = _neighbour_plate(plate)
    boundary = other >= 0
    if not boundary.any():  # degenerate single-plate world
        boundary[0, 0] = True
        other[0, 0] = plate[0, 0]

    scale = 1.0 / max(1e-6, float(np.abs(div).max()))
    conv_b = _gate(np.clip(-div, 0, None) * scale, *cfg.conv_gate)
    rift_b = _gate(np.clip(div, 0, None) * scale, *cfg.rift_gate)
    shear_b = _gate(np.abs(curl) / max(1e-6, float(np.abs(curl).max())), *cfg.shear_gate)

    dist, (iy, ix) = grid.edt(~boundary, return_indices=True)

    def pull(f, spread=True):
        """Carry a boundary value inland, then soften."""
        out = f[iy, ix].astype(float)
        if not spread:
            return out
        sm = grid.blur(out, cfg.stress_spread)
        peak, smpeak = float(out.max()), float(sm.max())
        return sm * (peak / smpeak) if smpeak > 1e-9 else sm  # blur eats the peak

    op = np.where(other >= 0, other, plate)
    fields = Tectonics(
        plate=plate, seeds=seeds, vel=vel, plate_cont=plate_cont,
        buoyancy=buoyancy, crust=crust, is_cont=is_cont,
        vy=vy, vx=vx, div=div, curl=curl,
        boundary=boundary, dist=dist,
        conv=pull(conv_b * boundary),
        rift=pull(rift_b * boundary),
        shear=pull(shear_b * boundary),
        cont_self=pull(plate_cont[plate].astype(float)),
        cont_other=pull(plate_cont[op].astype(float)),
        overriding=pull((buoyancy[plate] >= buoyancy[op]).astype(float)),
    )
    return fields


def boundary_class(t, conv_thresh=0.25):
    """0 none, 1 convergent, 2 divergent, 3 transform - for the plate overlay."""
    cls = np.zeros(t.plate.shape, dtype=int)
    on = t.boundary
    conv = t.conv * on
    rift = t.rift * on
    shear = t.shear * on
    cls[on & (shear > np.maximum(conv, rift) * 1.2)] = 3
    cls[on & (conv > conv_thresh) & (conv >= rift)] = 1
    cls[on & (rift > conv_thresh) & (rift > conv)] = 2
    return cls
