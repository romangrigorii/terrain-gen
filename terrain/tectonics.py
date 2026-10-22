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
    """Plate centres packed on the map. Longitude wraps. The cold edge is a row, not a point."""
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
            dx = np.minimum(dx, w - dx)
            if np.min(np.hypot(dy, dx)) < rmin:
                rmin *= 0.998
                continue
        pts.append(p)
    while len(pts) < n:
        pts.append(np.array([rng.uniform(0, h), rng.uniform(0, w)]))
    return np.asarray(pts, np.float64)


def warp_field(h, w, rng, periods, amp, octaves=4):
    """A (dy, dx) displacement pair of fBm, as a 2-vector stack."""
    return np.stack([noise.fbm(h, w, rng, periods=periods, octaves=octaves) * amp,
                     noise.fbm(h, w, rng, periods=periods, octaves=octaves) * amp])


def _plate_map(seeds, reach, h, w, rng, cfg):
    """Weighted Voronoi on the map. A step east is the same length as a step north."""
    # The short warp is what ripples the border. Keep it only while the
    # border still steps forward; the broad bend of the plate is left whole.
    broad = warp_field(h, w, rng, cfg.plate_warp_periods, cfg.plate_warp)
    fine = warp_field(h, w, rng, cfg.plate_warp_periods * 4, cfg.plate_warp_fine)
    wy, wx = noise.stack_displacement((broad, fine))
    yy, xx = np.mgrid[0:h, 0:w]
    qy = np.clip(yy + wy, 0, h - 1)
    qx = (xx + wx) % w

    plate = np.zeros((h, w), dtype=int)
    best = np.full((h, w), np.inf)
    for i, (sy, sx) in enumerate(seeds):
        dx = np.abs(qx - sx)
        dist = np.hypot(qy - sy, np.minimum(dx, w - dx)) - reach[i]
        take = dist < best
        best[take] = dist[take]
        plate[take] = i
    return plate


def _gate(x, lo, hi):
    """Suppress weak strain entirely, pass strong strain through unchanged."""
    t = np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    return x * t * t * (3 - 2 * t)


def _accrete(raw, continental, div, plate):
    """Collisions thicken crust. Rifts thin it. Latitude is not a term."""
    here = np.asarray(continental, dtype=bool)
    there = np.zeros(plate.shape, dtype=bool)
    on = np.zeros(plate.shape, dtype=bool)
    for dy, dx in grid.NEIGH4:
        neighbour = grid.shift(plate, dy, dx)
        flag = grid.shift(here.astype(np.uint8), dy, dx).astype(bool)
        take = (neighbour != plate) & ~on
        there[take] = flag[take]
        on[take] = True
    conv = np.clip(-np.asarray(div, np.float64), 0.0, None)
    rift = np.clip(np.asarray(div, np.float64), 0.0, None)
    conv = conv / max(float(conv.max()), 1e-6)
    rift = rift / max(float(rift.max()), 1e-6)
    both = on & here & there
    margin = on & here & ~there
    opening = on & here
    built = raw + 0.65 * conv * both + 0.35 * conv * margin - 0.40 * rift * opening
    spread = max(1.25, 0.02 * min(raw.shape))
    return grid.blur(built.astype(np.float32), spread)


def _crust(plate, plate_cont, cfg, rng, div=None, other=None):
    """Continentality from fractal noise, then collisions and rifts.

    A plate can hold both kinds of crust, and a continent can cross a boundary.
    Latitude does not lower the crust. A polar world only lifts the cold edge
    where the noise is already high, so that edge is not one solid bar.
    `other` is accepted so older callers keep working; the neighbour crust is
    read from the cells themselves.
    """
    del other
    h, w = plate.shape
    broad = noise.fbm(h, w, rng, periods=cfg.crust_periods, octaves=6)
    detail = noise.fbm(h, w, rng, periods=cfg.crust_periods * 3, octaves=4)
    lean = plate_cont[plate].astype(np.float64) - 0.5
    raw = 0.85 * broad + 0.35 * detail + lean * cfg.crust_plate_weight * 0.55
    if cfg.geography == "poles":
        lat = np.abs(grid.latitude((h, w)))
        lobes = noise.fbm(h, w, rng, periods=3, octaves=4)
        raw = raw + lat * (0.9 + 1.2 * np.clip(lobes, 0.0, None))
    elif cfg.geography == "dispersed":
        raw = raw + 0.45 * noise.fbm(h, w, rng, periods=cfg.crust_periods * 2, octaves=4)
    raw = noise.warp(raw, *warp_field(h, w, rng, cfg.crust_warp_periods, cfg.crust_warp))
    raw = grid.blur(raw, cfg.crust_blur)
    if div is not None:
        preliminary = raw >= grid.area_quantile(raw, 1.0 - cfg.cont_fraction)
        raw = _accrete(raw, preliminary, div, plate)

    thr = grid.area_quantile(raw, 1.0 - cfg.cont_fraction)
    return np.clip(0.5 + 0.5 * np.tanh(cfg.crust_sharpness *
                                       (raw - thr) / (raw.std() + 1e-9)), 0, 1)


def _sides(plate, crust, steps):
    """Crust just inside each plate, far enough to clear the boundary blur."""
    h, w = plate.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    own = crust.copy()
    far = crust.copy()
    found = np.zeros(plate.shape, dtype=bool)
    for dy, dx in grid.NEIGH4:
        neighbour = grid.shift(plate, dy, dx)
        take = (neighbour != plate) & ~found
        if not np.any(take):
            continue
        near_y, near_x = rows, cols
        far_y, far_x = rows, cols
        for _ in range(max(1, int(steps))):
            near_y = np.clip(near_y - dy, 0, h - 1)
            near_x = (near_x - dx) % w
            far_y = np.clip(far_y + dy, 0, h - 1)
            far_x = (far_x + dx) % w
        own = np.where(take, crust[near_y, near_x], own)
        far = np.where(take, crust[far_y, far_x], far)
        found |= take
    return own, far


def _neighbour_plate(plate):
    """For every cell, a neighbouring cell's plate id if it differs, else -1."""
    other = np.full(plate.shape, -1, dtype=int)
    for dy, dx in grid.NEIGH4:
        nb = grid.shift(plate, dy, dx)
        take = (nb != plate) & (other < 0)
        other[take] = nb[take]
    return other


def _shift3(field, dy, dx):
    return np.stack([grid.shift(field[..., i], dy, dx) for i in range(field.shape[-1])], axis=-1)


def _boundary_motion(plate, omega, xyz):
    """Opening (positive is a rift) and shear where two plates meet.

    Each plate moves as v = ω × p. At a shared boundary the two angular
    velocities are compared at the same point, so a rigid plate, poles
    included, has no internal strain.
    """
    position = np.stack(xyz, axis=-1)
    opening = np.zeros(plate.shape, np.float64)
    shear = np.zeros(plate.shape, np.float64)
    hits = np.zeros(plate.shape, np.float64)

    def add(other_plate, other_position):
        differ = other_plate != plate
        if not np.any(differ):
            return
        delta = other_position - position
        delta = delta - np.sum(delta * position, axis=-1, keepdims=True) * position
        length = np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-8)
        normal = delta / length
        relative = np.cross(omega[other_plate] - omega[plate], position)
        closing = -np.sum(relative * normal, axis=-1)
        along = np.sum(relative * normal, axis=-1, keepdims=True)
        side = np.linalg.norm(relative - along * normal, axis=-1)
        opening[differ] += (-closing)[differ]
        shear[differ] += side[differ]
        hits[differ] += 1.0

    for dy, dx in grid.NEIGH8:
        add(grid.shift(plate, dy, dx), _shift3(position, dy, dx))
    rolled = np.roll(position, position.shape[1] // 2, axis=1)
    rolled_id = np.roll(plate, plate.shape[1] // 2, axis=1)
    for row in (0, -1):
        other_plate = plate.copy()
        other_position = position.copy()
        other_plate[row] = rolled_id[row]
        other_position[row] = rolled[row]
        add(other_plate, other_position)
    weight = np.maximum(hits, 1.0)
    return opening / weight, shear / weight


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
    vy, vx = vel[plate, 0], vel[plate, 1]
    dvy_dy, dvy_dx = grid.gradient(vy)
    dvx_dy, dvx_dx = grid.gradient(vx)
    div = grid.blur((dvx_dx + dvy_dy).astype(np.float32), cfg.stress_blur)
    curl = grid.blur((dvy_dx - dvx_dy).astype(np.float32), cfg.stress_blur)
    other = _neighbour_plate(plate)

    crust = _crust(plate, plate_cont, cfg, rng, div=div, other=other)
    is_cont = crust > 0.5
    boundary = other >= 0
    if not boundary.any():  # degenerate single-plate world
        boundary[0, 0] = True
        other[0, 0] = plate[0, 0]

    scale = 1.0 / max(1e-6, float(np.abs(div).max()))
    conv_b = _gate(np.clip(-div, 0, None) * scale, *cfg.conv_gate)
    rift_b = _gate(np.clip(div, 0, None) * scale, *cfg.rift_gate)
    shear_b = _gate(np.abs(curl) / max(1e-6, float(np.abs(curl).max())), *cfg.shear_gate)

    dist, (iy, ix) = grid.edt(~boundary, return_indices=True)
    # The pixel edge is a staircase. A ridge drawn on that staircase ripples
    # along the boundary; a short blur takes the stair out and leaves the line.
    dist = grid.blur(dist.astype(np.float32), 1.25)

    def pull(f, spread=True):
        """Carry a boundary value inland, then soften."""
        out = f[iy, ix].astype(float)
        if not spread:
            return out
        sm = grid.blur(out, cfg.stress_spread)
        peak, smpeak = float(out.max()), float(sm.max())
        return sm * (peak / smpeak) if smpeak > 1e-9 else sm  # blur eats the peak

    own, far = _sides(plate, is_cont.astype(np.float32),
                      max(3, int(round(float(cfg.crust_blur)))))
    fields = Tectonics(
        plate=plate, seeds=seeds, vel=vel, plate_cont=plate_cont,
        buoyancy=buoyancy, crust=crust, is_cont=is_cont,
        vy=vy, vx=vx, div=div, curl=curl,
        boundary=boundary, dist=dist,
        conv=pull(conv_b * boundary),
        rift=pull(rift_b * boundary),
        shear=pull(shear_b * boundary),
        cont_self=pull(own),
        cont_other=pull(far),
        overriding=pull((own >= far).astype(float)),
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
