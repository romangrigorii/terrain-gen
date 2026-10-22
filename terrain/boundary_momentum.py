"""Continue a tile by the way its terrain was already moving.

Distance into the new tile is `d` and position along the shared edge is `s`.
The continuation follows

    dH/dd + v(s) dH/ds = m(s)

`v` is how far a ridge, coast, or valley steps sideways per cell inward.
`m` is the elevation change that remains after that step. Both are read from
the incoming strip, whose last column is the shared edge and whose earlier
columns run back into the tile that already exists.

This is a geometric initial condition. It is not a tectonic simulation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import grid


# The integrated slope must not become a mountain. 0.02 elevation units is
# a few hundred metres, not a wall built out of a one-pixel estimate.
_TREND_CAP = 0.02


def continue_boundary(target, strip, side, reach=96, detail_reach=48,
                      tangent_sigma=4.0, max_drift=0.7, slope_decay=None,
                      velocity=None, residual=None):
    """Carry `strip` in from `side` and ease it into `target` by `reach`.

    `strip` has shape (length along the edge, depth), ordered from the old
    interior toward the shared edge. The shared edge of the result is that
    last column. Past `reach`, the result is `target` unchanged.
    """
    if side not in ("left", "right", "top", "bottom"):
        raise ValueError("side must be left, right, top, or bottom")
    out = np.array(target, np.float64, copy=True)
    rows, cols = out.shape
    span = rows if side in ("left", "right") else cols
    strip = np.asarray(strip, np.float64)
    if strip.ndim != 2 or strip.shape[0] != span or strip.shape[1] < 2:
        raise ValueError("strip must be (edge length, depth) and end on the boundary")
    depth = cols if side in ("left", "right") else rows
    reach = float(reach)
    if reach <= 0 or depth < 1:
        return out
    detail_reach = float(min(max(detail_reach, 0.0), reach))
    if slope_decay is None:
        # A long reach must not turn the decay length into a huge integral.
        slope_decay = min(8.0, max(reach / 3.0, 1.0))
    else:
        slope_decay = max(float(slope_decay), 1.0)
    if velocity is None:
        velocity, residual = momentum(strip, tangent_sigma, max_drift)
    else:
        velocity = np.asarray(velocity, np.float64)
        if velocity.shape != (span,):
            raise ValueError("velocity must be one value per edge cell")
        if residual is None:
            residual = np.zeros(span, np.float64)
        else:
            residual = np.asarray(residual, np.float64)
    boundary = strip[:, -1]
    radius = max(float(tangent_sigma), 2.0)
    low_b, high_b = _split(boundary, radius)
    index = _side_index(side)
    limit = min(depth, int(np.ceil(reach - 1e-9)))
    for distance in range(limit):
        if distance >= reach:
            break
        line = out[index(distance)]
        low_t, high_t = _split(line, radius)
        origin = _backtrack(velocity, distance)
        coarse = _window(distance, reach)
        fine = _window(distance, detail_reach)
        growth = slope_decay * (1.0 - np.exp(-distance / slope_decay))
        trend = np.clip(_at(residual, origin) * growth, -_TREND_CAP, _TREND_CAP)
        blended = ((1.0 - coarse) * low_t + coarse * _at(low_b, origin)
                   + (1.0 - fine) * high_t + fine * _at(high_b, origin)
                   + coarse * trend)
        out[index(distance)] = blended
    out[index(0)] = boundary
    return out


def momentum(strip, tangent_sigma=4.0, max_drift=0.7):
    """Sidestep per cell, and the elevation change left after that sidestep.

    Flat water matches every shift, so it stays at zero until a real feature
    is closer. Between two tracked edges the sidestep is filled in, so a
    landmass keeps the motion of its own coasts instead of one median for the
    whole tile.
    """
    field = np.asarray(strip, np.float64)
    span, depth = field.shape
    velocity = np.zeros(span, np.float64)
    residual = np.zeros(span, np.float64)
    if depth < 2 or span < 2:
        return velocity, residual
    gap = min(4, depth - 1)
    newer = field[:, -1]
    older = field[:, -1 - gap]
    radius = max(2, int(round(float(tangent_sigma))))
    search = max(6, int(np.ceil(float(max_drift) * gap)) + 2)
    shifts = np.arange(-search, search + 1)
    errors = np.empty((shifts.size, span), np.float64)
    for i, shift in enumerate(shifts):
        errors[i] = _local_sse(newer, _shift(older, int(shift)), radius)
    center = search
    best = np.argmin(errors, axis=0)
    columns = np.arange(span)
    err_best = errors[best, columns]
    err_zero = errors[center]
    scale = float(np.mean(newer * newer)) + 1.0
    confident = err_best < err_zero - 1e-6 * scale
    best = np.where(confident, best, center)
    err_best = errors[best, columns]
    lower = np.clip(best - 1, 0, shifts.size - 1)
    upper = np.clip(best + 1, 0, shifts.size - 1)
    err_lo = errors[lower, columns]
    err_hi = errors[upper, columns]
    denom = err_lo - 2.0 * err_best + err_hi
    delta = np.zeros(span, np.float64)
    curved = confident & (np.abs(denom) > 1e-8)
    delta[curved] = 0.5 * (err_lo[curved] - err_hi[curved]) / denom[curved]
    delta = np.clip(delta, -0.75, 0.75)
    velocity = (shifts[best].astype(np.float64) + delta) / float(gap)
    if np.any(confident):
        known = np.flatnonzero(confident)
        velocity = np.interp(columns.astype(np.float64), known.astype(np.float64), velocity[known])
    else:
        velocity = np.zeros(span, np.float64)
    limit = float(max_drift)
    velocity = np.clip(velocity, -limit, limit)
    if tangent_sigma > 0:
        velocity = gaussian_filter1d(velocity, float(tangent_sigma), mode="nearest")
        velocity = np.clip(velocity, -limit, limit)
    aligned = _shift(older, velocity * gap)
    residual = (newer - aligned) / float(gap)
    if tangent_sigma > 0:
        residual = gaussian_filter1d(residual, float(tangent_sigma), mode="nearest")
    return velocity, residual


def coast_distance(height, sea=0.0):
    """Positive on land, negative on water, measured to the shoreline."""
    land = np.asarray(height) > sea
    if not np.any(land):
        return np.full(land.shape, -1e3, np.float64)
    if not np.any(~land):
        return np.full(land.shape, 1e3, np.float64)
    return grid.edt(land) - grid.edt(~land)


def continue_shore(target, strip, side, sea=0.0, **kwargs):
    """Carry the shoreline with the sidestep measured on the height strip.

    The distance field is smooth, so its own match is ambiguous. The height
    edge is the thing that actually moved.
    """
    tangent = kwargs.get("tangent_sigma", 4.0)
    drift = kwargs.get("max_drift", 0.7)
    velocity, _residual = momentum(strip, tangent, drift)
    return continue_boundary(
        coast_distance(target, sea), coast_distance(strip, sea), side,
        velocity=velocity, residual=np.zeros(velocity.shape[0], np.float64),
        **kwargs)


def apply_boundary_slope(h, strip, side):
    """Start the inward cell on the slope that arrived with the strip.

    This is an initial condition. The erosion loop should not keep writing
    this second cell, or it pins sediment that ought to move.
    """
    if side not in ("left", "right", "top", "bottom"):
        raise ValueError("side must be left, right, top, or bottom")
    strip = np.asarray(strip, np.float64)
    if strip.ndim != 2 or strip.shape[1] < 2:
        raise ValueError("strip must carry at least the edge and the cell before it")
    edge = strip[:, -1]
    raw = edge - strip[:, -2]
    broad = gaussian_filter1d(raw, 2.0, mode="nearest")
    # A real cliff keeps its step. Small grain does not, or the next cell
    # becomes a ridge running the length of the new tile.
    incoming = np.where(np.abs(raw) > 0.2, raw, broad)
    if side == "left":
        h[:, 0] = edge
        h[:, 1] = edge + incoming
    elif side == "right":
        h[:, -1] = edge
        h[:, -2] = edge + incoming
    elif side == "top":
        h[0, :] = edge
        h[1, :] = edge + incoming
    else:
        h[-1, :] = edge
        h[-2, :] = edge + incoming
    return h


def diagnose_vertical_seam(left, right):
    """Height gap and slope gap where `left` meets `right`."""
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    height_error = a[:, -1] - b[:, 0]
    incoming = a[:, -1] - a[:, -2]
    outgoing = b[:, 1] - b[:, 0]
    slope_error = incoming - outgoing
    return {
        "height_max": float(np.max(np.abs(height_error))),
        "slope_max": float(np.max(np.abs(slope_error))),
        "slope_mean": float(np.mean(np.abs(slope_error))),
    }


def diagnose_shore_seam(left, right, sea=0.0):
    """The same seam, measured on signed distance to the coast."""
    return diagnose_vertical_seam(coast_distance(left, sea), coast_distance(right, sea))


def conform_shore(height, distance, sea=0.0, band=8.0):
    """Where the shore distance and the height disagree, follow the shore.

    Only cells near sea level move. A high plateau and deep water keep the
    elevation the momentum field already carried in.
    """
    h = np.array(height, np.float64, copy=True)
    distance = np.asarray(distance, np.float64)
    near_level = np.abs(h - sea) <= 0.15
    close = np.abs(distance) <= float(band)
    wrong = near_level & close & np.isfinite(distance) & ((h > sea) != (distance > 0.0))
    if np.any(wrong):
        magnitude = np.maximum(np.abs(h - sea), 0.02)
        h[wrong] = sea + np.sign(distance[wrong]) * magnitude[wrong]
    return h


def _window(distance, reach):
    """One at the edge, zero at `reach`, with flat slope at both ends."""
    if reach <= 0 or distance >= reach:
        return 0.0
    if distance <= 0:
        return 1.0
    t = float(distance) / float(reach)
    fade = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return 1.0 - fade


def _split(profile, radius):
    profile = np.asarray(profile, np.float64)
    low = gaussian_filter1d(profile, radius, mode="nearest")
    return low, profile - low


def _shift(values, dy):
    base = np.arange(values.shape[0], dtype=np.float64)
    return np.interp(base - np.asarray(dy, np.float64), base, np.asarray(values, np.float64))


def _at(values, position):
    base = np.arange(values.shape[0], dtype=np.float64)
    return np.interp(position, base, np.asarray(values, np.float64))


def _backtrack(velocity, distance, steps=4):
    """Where along the edge the sample now at `s` came from."""
    position = np.arange(velocity.shape[0], dtype=np.float64)
    if distance <= 0:
        return position
    step = float(distance) / float(steps)
    base = position.copy()
    for _ in range(steps):
        position = position - np.interp(position, base, velocity) * step
    return position


def _local_sse(left, right, radius):
    diff = (np.asarray(left, np.float64) - np.asarray(right, np.float64)) ** 2
    radius = int(max(0, radius))
    if radius == 0:
        return diff
    padded = np.pad(diff, radius, mode="edge")
    kernel = np.ones(radius * 2 + 1, np.float64) / (radius * 2 + 1)
    return np.convolve(padded, kernel, mode="valid")


def _side_index(side):
    if side == "left":
        return lambda distance: (slice(None), distance)
    if side == "right":
        return lambda distance: (slice(None), -1 - distance)
    if side == "top":
        return lambda distance: (distance, slice(None))
    return lambda distance: (-1 - distance, slice(None))


def side_weight(shape, side, reach):
    """One on the shared edge, zero at `reach`, the same window on every side."""
    rows, cols = shape
    if side == "left":
        distance = np.arange(cols, dtype=np.float64)[None, :]
    elif side == "right":
        distance = np.arange(cols, dtype=np.float64)[None, ::-1]
    elif side == "top":
        distance = np.arange(rows, dtype=np.float64)[:, None]
    else:
        distance = np.arange(rows, dtype=np.float64)[::-1, None]
    t = np.clip(distance / float(reach), 0.0, 1.0)
    fade = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    weight = np.broadcast_to(1.0 - fade, shape).copy()
    return weight
