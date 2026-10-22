"""Stage 2: turn plate strain into an elevation field."""
import math

import numpy as np

from . import grid, noise, tectonics


def _ramp(d, width):
    """1 at the boundary, decaying inland."""
    return np.exp(-d / max(1e-6, width))


def _band(d, offset, width):
    """A ridge sitting `offset` cells away from the boundary."""
    return np.exp(-((d - offset) ** 2) / (2 * max(1e-6, width) ** 2))


def tectonic_relief(t, cfg, rng):
    """Elevation from crust thickness plus boundary tectonics, before texture."""
    d = t.dist
    conv, rift, shear = t.conv, t.rift, t.shear
    crust = t.crust                    # local crust thickness, gates everything
    ocean = 1.0 - crust
    both_cont = t.cont_self * t.cont_other
    subduction = np.clip(np.abs(t.cont_self - t.cont_other) +
                         (1 - t.cont_self) * (1 - t.cont_other), 0, 1)
    over = t.overriding

    h = cfg.ocean_base + (cfg.cont_base - cfg.ocean_base) * crust

    hh, w = h.shape
    h += cfg.upland_h * noise.fbm(hh, w, rng, periods=cfg.upland_periods, octaves=5) * crust

    h += cfg.collision_h * conv * both_cont * crust * _ramp(d, cfg.collision_w)

    down = (1.0 - over) * subduction * conv
    up = over * subduction * conv
    h -= cfg.trench_h * down * _ramp(d, cfg.trench_w)
    segs = np.clip(noise.fbm(hh, w, rng, periods=cfg.arc_break_periods, octaves=4)
                   * 2.4 + cfg.arc_break_bias, 0, 1)
    h += cfg.arc_h * up * segs * _band(d, cfg.arc_offset, cfg.arc_w)
    h += cfg.cordillera_h * up * crust * _ramp(d, cfg.cordillera_w)
    h += 0.15 * cfg.arc_h * up * crust * _band(d, cfg.arc_offset * 2.2, cfg.arc_w * 2)

    cont_rift = rift * crust
    h -= cfg.rift_depth * cont_rift * _ramp(d, cfg.rift_w)
    h += cfg.rift_shoulder * cont_rift * _band(d, cfg.rift_w * 1.6, cfg.rift_w)
    h += cfg.ridge_h * rift * ocean * _ramp(d, cfg.ridge_w)

    h -= cfg.transform_h * shear * _ramp(d, cfg.transform_w)
    h += 0.4 * cfg.transform_h * shear * _band(d, cfg.transform_w * 3, cfg.transform_w * 2)

    ridge_src = (t.rift > cfg.ridge_age_thresh) & t.boundary & ~t.is_cont
    if ridge_src.any():
        d_ridge = grid.edt(~ridge_src)
        d_ridge = noise.warp(d_ridge, *tectonics.warp_field(
            hh, w, rng, cfg.age_warp_periods, cfg.age_warp))
        h -= cfg.age_depth * (1 - np.exp(-d_ridge / cfg.age_scale)) * ocean

    return h


def hotspot_chains(h, t, cfg, rng):
    """Intraplate volcanism smeared into a chain by the plate's own motion."""
    hh, w = h.shape
    for _ in range(cfg.n_hotspots):
        for _try in range(40):
            y = rng.uniform(hh * 0.1, hh * 0.9)
            x = rng.uniform(0, w)
            if t.crust[int(y), int(x)] < cfg.hotspot_max_crust:
                break
        else:
            continue
        p = t.plate[int(y), int(x)]
        vy, vx = t.vel[p]
        step = cfg.hotspot_spacing
        n_isl = rng.integers(cfg.hotspot_chain[0], cfg.hotspot_chain[1] + 1)
        amp = cfg.hotspot_h * rng.uniform(0.7, 1.3)
        for i in range(n_isl):
            cy = y - vy * step * i + rng.normal(0, step * 0.25)
            cx = x - vx * step * i + rng.normal(0, step * 0.25)
            if not (0 <= cy < hh):
                break
            decay = amp * (0.55 ** (i * 0.8)) * rng.uniform(0.7, 1.15)
            for _ in range(3):
                s = cfg.hotspot_sigma * rng.uniform(0.55, 1.15)
                grid.stamp(h, cy + rng.normal(0, s * 0.6), cx + rng.normal(0, s * 0.6),
                           decay * rng.uniform(0.35, 0.6), s)
    return h


def texture(h, t, cfg, rng):
    """Fractal detail, weighted so young mountains are rough and abyss is smooth."""
    hh, w = h.shape
    relief = grid.normalize(grid.blur(np.abs(h - grid.blur(h, 12)), 6))
    tectonic_act = grid.normalize(grid.blur(t.conv + 0.6 * t.rift + 0.4 * t.shear, 3))
    rough = np.clip(0.35 + 1.3 * np.maximum(relief, tectonic_act * _ramp(t.dist, 40)), 0, 1.6)

    ridged = noise.fbm(hh, w, rng, periods=cfg.texture_periods, octaves=7, ridged=True)
    fine = noise.fbm(hh, w, rng, periods=cfg.texture_periods * 2, octaves=5)
    wy = noise.fbm(hh, w, rng, periods=3, octaves=3) * cfg.texture_warp
    wx = noise.fbm(hh, w, rng, periods=3, octaves=3) * cfg.texture_warp
    ridged = noise.warp(ridged, wy, wx)

    land_ish = np.clip((h - cfg.ocean_base * 0.5) * 2, 0, 1)
    h = h + cfg.texture_h * rough * ridged * land_ish
    h = h + cfg.texture_h * 0.45 * fine
    h = h + cfg.abyss_h * fine * (1 - land_ish)
    return h


def coastal_plain(h, cfg, rng):
    """Bound how fast the land may climb as it leaves the shore."""
    land = h > 0.0
    if not land.any():
        return h
    inland = grid.edt(land)
    n = noise.fbm(*h.shape, rng, periods=cfg.coast_slope_periods, octaves=4)
    allowed = cfg.coast_slope * (0.5 + 1.5 * np.clip(0.5 + 0.5 * n, 0, 1) ** 1.5)
    zone = np.exp(-(inland / cfg.coast_plain_zone) ** 2)
    return np.where(land, h + (np.minimum(h, allowed * inland) - h) * zone, h)


def erode_margins(h, t, cfg, rng):
    """Fray the coast where a continental plate meets an oceanic one."""
    hh, w = h.shape
    land = h > 0.0
    if not land.any():
        return h

    sd = grid.edt(land) - grid.edt(~land)
    near = (np.abs(t.cont_self - t.cont_other)       # continent meets ocean
            * _ramp(t.dist, cfg.margin_zone))        # near that boundary

    slope = np.hypot(*grid.gradient(grid.blur(h, cfg.margin_slope_blur)))
    ref = float(np.median(slope[np.abs(sd) < 2.0]))
    gain = np.clip(slope / max(ref, 1e-9), 1.0, cfg.margin_gain_max)

    def unit(periods, octaves=4, ridged=False, cap=None):
        f = noise.fbm(hh, w, rng, periods=periods, octaves=octaves, ridged=ridged)
        u = (f - f.mean()) / max(float(f.std()), 1e-9)
        return u if cap is None else np.clip(u, -cap, cap)

    h = h + (near * np.exp(-(sd / cfg.margin_reach) ** 2) * gain
             * (cfg.margin_h * unit(cfg.margin_periods, cap=cfg.margin_cap)
                + cfg.margin_h_fine * unit(cfg.margin_periods * 3,
                                           cap=cfg.margin_cap)))

    cut = np.clip(unit(cfg.margin_cut_periods, octaves=5, ridged=True)
                  - cfg.margin_cut_thresh, 0.0, cfg.margin_cut_cap)
    return h - cfg.margin_cut_h * near * _band(sd, cfg.margin_cut_offset,
                                               cfg.margin_cut_w) * cut


def fray_coast(h, cfg, rng):
    """Break up the shoreline where it would otherwise be a smooth contour."""
    hh, w = h.shape
    big = noise.fbm(hh, w, rng, periods=cfg.coast_periods * 0.4, octaves=4)
    fine = noise.fbm(hh, w, rng, periods=cfg.coast_periods * 2, octaves=4)
    band = np.exp(-(h / cfg.coast_band) ** 2)
    return h + band * (cfg.coast_h * big + cfg.coast_h_fine * fine)


def _stroke_ok(stroke):
    if not isinstance(stroke, dict):
        return False
    kind = stroke.get("kind")
    if kind not in ("mountain", "valley"):
        return False
    try:
        metres = float(stroke.get("metres", 0.0))
        radius = float(stroke.get("radius", 0.0))
    except (TypeError, ValueError):
        return False
    points = stroke.get("points") or ()
    return (math.isfinite(metres) and metres > 0 and math.isfinite(radius)
            and radius > 0 and len(points) > 0)


def _stroke_mask(shape, points, radius):
    """1 at the crest of the stroke, falling to 0 at `radius`."""
    from scipy.ndimage import gaussian_filter

    rows, cols = shape
    coords = []
    for point in points:
        if len(point) < 2:
            continue
        try:
            y = float(point[0])
            x = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(y) and math.isfinite(x)):
            continue
        coords.append((y * (rows - 1), x * (cols - 1)))
    if not coords:
        return np.zeros((rows, cols), np.float32)
    sigma = max(0.8, float(radius) * cols / 2.0)
    pad = int(math.ceil(sigma * 3.0)) + 2
    ys = [p[0] for p in coords]
    xs = [p[1] for p in coords]
    y0 = int(math.floor(min(0.0, min(ys)))) - pad
    x0 = int(math.floor(min(0.0, min(xs)))) - pad
    y1 = int(math.ceil(max(float(rows - 1), max(ys)))) + pad
    x1 = int(math.ceil(max(float(cols - 1), max(xs)))) + pad
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.float32)
    samples = [coords[0]]
    for (py0, px0), (py1, px1) in zip(coords, coords[1:]):
        steps = int(max(abs(py1 - py0), abs(px1 - px0)))
        for i in range(1, steps + 1):
            t = i / steps
            samples.append((py0 + (py1 - py0) * t, px0 + (px1 - px0) * t))
    iy = np.rint([p[0] for p in samples])
    ix = np.rint([p[1] for p in samples])
    iy = np.clip(iy - y0, 0, canvas.shape[0] - 1).astype(int)
    ix = np.clip(ix - x0, 0, canvas.shape[1] - 1).astype(int)
    canvas[iy, ix] = 1.0
    soft = gaussian_filter(canvas, sigma, mode="constant", cval=0.0)
    peak = float(soft.max())
    if peak <= 0:
        return np.zeros((rows, cols), np.float32)
    tile = soft[-y0:-y0 + rows, -x0:-x0 + cols]
    return (tile / np.float32(peak)).astype(np.float32, copy=False)


def paint_relief(height, strokes, metres_per_unit=None):
    """Add painted mountains and cut painted valleys."""
    usable = [stroke for stroke in (strokes or ()) if _stroke_ok(stroke)]
    if not usable:
        return height
    if metres_per_unit is None:
        from .export import METRES_PER_UNIT
        metres_per_unit = METRES_PER_UNIT
    scale = float(metres_per_unit)
    out = np.array(height, np.float32, copy=True)
    for stroke in usable:
        mask = _stroke_mask(out.shape, stroke["points"], stroke["radius"])
        amp = float(stroke["metres"]) / scale
        if stroke["kind"] == "valley":
            out -= np.float32(amp) * mask
            continue
        before = out
        out = before + np.float32(amp) * mask
        water = np.clip(-before, 0, None)
        out += water * np.clip(mask * np.float32(3.0), 0, 1)
    return np.asarray(out, np.float32)


def _stroke_cover(shape, strokes):
    """The strongest brush at each cell, or a zero field when nothing paints."""
    cover = None
    for stroke in strokes or ():
        if not _stroke_ok(stroke):
            continue
        mask = _stroke_mask(shape, stroke["points"], stroke["radius"])
        cover = mask if cover is None else np.maximum(cover, mask)
    if cover is None:
        return np.zeros(shape, np.float32)
    return cover


def sculpt_grain(shape, strokes, cfg, origin=(0, 0)):
    """Ridged grain for a painted mountain or valley."""
    cover = _stroke_cover(shape, strokes)
    if float(cover.max()) <= 0.0 or float(getattr(cfg, "texture_h", 0.0)) == 0.0:
        return np.zeros(shape, np.float32)
    rows, cols = shape
    tx, ty = (int(origin[0]), int(origin[1])) if origin else (0, 0)
    rng = noise.stage_rng(int(getattr(cfg, "seed", 0)), "sculpt-texture", tx, ty)
    periods = max(1.0, float(cfg.texture_periods))
    ridged = noise.fbm(rows, cols, rng, periods=periods, octaves=7, ridged=True)
    fine = noise.fbm(rows, cols, rng, periods=periods * 2, octaves=5)
    warp = float(cfg.texture_warp)
    wy = noise.fbm(rows, cols, rng, periods=3, octaves=3) * warp
    wx = noise.fbm(rows, cols, rng, periods=3, octaves=3) * warp
    ridged = noise.warp(ridged, wy, wx)
    ridged = ridged - np.float32(ridged.mean())
    fine = fine - np.float32(fine.mean())
    detail = np.float32(cfg.texture_h) * ridged + np.float32(cfg.texture_h) * np.float32(0.45) * fine
    return (detail * cover).astype(np.float32)
