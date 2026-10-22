"""High-resolution terrain export."""
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from terrain.export import METRES_PER_UNIT, _write_png
from terrain.tiles import RELIEF_UNITS, hierarchical_window

PREVIEW_WIDTH = 640
DIRECT_MAX = 8192
TILE = 4096
GAME_MAX = 2048
EXPORT_WIDTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
EXPORT_LABELS = ("1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K")

DETAIL_FRACTION = 0.02
ATLAS_DETAIL_REFERENCE_SPAN = 2.0
DETAIL_PERIODS = (64, 32, 16, 8)


def map_size(width):
    """4:3, even height, so block reductions in the simulator stay aligned."""
    return int(width), int(width * 3 / 4) // 2 * 2


def describe(width):
    w, h = map_size(width)
    if width > DIRECT_MAX:
        tiles = math.ceil(w / TILE) * math.ceil(h / TILE)
        text = f"{w}×{h}, {tiles} tiles upscaled from the {PREVIEW_WIDTH}-wide preview"
        raw_gib = (w * h * 9) / (1024 ** 3)  # 16-bit height + RGBA + RGB
        if width >= 32768:
            text += f", ~{raw_gib:.0f} GiB raw before PNG compression"
        return text
    return f"{w}×{h}, upscaled from the {PREVIEW_WIDTH}-wide preview"


def stitch_fields(height, biome, cover, edges, band=None):
    """Build the tile from its borders, then let the new surface fill the inside."""
    height = from_borders(height, edges, "height", "outer", band)
    cover = np.clip(from_borders(cover, edges, "cover", "cover_outer", band), 0, 1)
    biome = np.array(biome, copy=True)
    paint_lip_biome(biome, edges)
    return height, biome, cover

TEXTURE_HOLD = 10
TEXTURE_FADE = 8


def from_borders(target, edges, key="height", outer_key="outer", band=None):
    """Border keeps the neighbor's texture. The new map takes over further in."""
    target = np.array(target, np.float32, copy=True)
    if not edges:
        return target
    cont, dist, band = _border_plane(target.shape, edges, key, outer_key, band)
    filled = target.copy()
    inside = dist <= band
    filled[inside] = cont[inside]
    sigma = max(16.0, band / 5.0)
    low_c = gaussian_filter(filled, sigma=sigma, mode="nearest")
    low_t = gaussian_filter(target, sigma=sigma, mode="nearest")
    fade_low = _quintic_arr(np.clip(dist / max(band, 1), 0, 1))
    fade_tex = np.ones(target.shape, np.float32)
    fade_tex[inside] = _quintic_arr(np.clip((dist[inside] - TEXTURE_HOLD) / TEXTURE_FADE, 0, 1))
    out = (filled * (1.0 - fade_tex) + target * fade_tex
           + (low_t - low_c) * (fade_low - fade_tex))
    out[dist > band] = target[dist > band]
    return out


def restore_lip(field, edges, key="height", outer_key="outer"):
    """Put the neighbor's next samples back after a later stage rewrote them."""
    field = np.array(field, np.float32, copy=True)
    if not edges:
        return field
    plane, dist, _ = _border_plane(field.shape, edges, key, outer_key)
    field[dist < TEXTURE_HOLD] = plane[dist < TEXTURE_HOLD]
    return field


def paint_lip_biome(biome, edges):
    reach = TEXTURE_HOLD + TEXTURE_FADE // 2
    for edge in edges:
        if "biome" not in edge:
            continue
        for i in range(reach):
            _edge_line(biome, edge["side"], i)[:] = edge["biome"]


def _quintic_arr(s):
    """Zero value and zero slope at 0, one at 1."""
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def _band_for(shape, band):
    h, w = shape
    band = min(band or max(36, w // 4), h // 2 - 1, w // 2 - 1)
    return max(8, int(band))


def _smooth_1d(delta, sigma=None):
    """Broad part of an edge profile. A one-pixel jag stays on the boundary."""
    delta = np.asarray(delta, np.float32)
    if sigma is None:
        sigma = max(2.0, delta.shape[0] / 40.0)
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(delta, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _edge_line(field, side, i):
    if side == "left":
        return field[:, i]
    if side == "right":
        return field[:, -1 - i]
    if side == "top":
        return field[i, :]
    return field[-1 - i, :]


def _border_plane(shape, edges, key, outer_key, band=None):
    """Neighbor texture carried across the border, grain included."""
    band = _band_for(shape, band)
    dist = np.full(shape, np.float32(band + 1))
    acc = np.zeros(shape, np.float32)
    weight = np.zeros(shape, np.float32)
    strip_key = "strip" if key == "height" else "cover_strip"
    for edge in edges:
        side = edge["side"]
        vertical = side in ("left", "right")
        span = shape[0] if vertical else shape[1]
        border, step, flow = _edge_motion(edge, key, outer_key, strip_key, span)
        if border is None:
            continue
        tone = max(10.0, span / 18.0)
        broad = _smooth_1d(step, tone)
        edge_s = _smooth_1d(border, tone)
        grain = border - edge_s
        reach = TEXTURE_HOLD + TEXTURE_FADE
        rows = np.arange(span)
        for i in range(band):
            if i == 0:
                values = border + step
            elif i < reach:
                src = np.clip(np.rint(rows - flow * i).astype(np.int32), 0, span - 1)
                values = edge_s + broad * np.float32(i + 1) + grain[src]
            else:
                values = edge_s + broad * np.float32(i + 1)
            _edge_line(acc, side, i)[:] += values
            _edge_line(weight, side, i)[:] += 1.0
            line = _edge_line(dist, side, i)
            np.minimum(line, np.float32(i), out=line)
    known = weight > 0
    acc[known] /= weight[known]
    return acc, dist, band


def _edge_motion(edge, key, outer_key, strip_key, span):
    """Border sample, per-pixel step, and how fast features slide along the edge."""
    strip = edge.get(strip_key)
    if strip is not None:
        strip = np.asarray(strip, np.float32)
        if strip.ndim == 2 and strip.shape[0] == span and strip.shape[1] >= 2:
            border = strip[:, -1]
            prev = strip[:, -2]
            flow = np.clip(_smooth_1d(_flow(strip)), -3.0, 3.0)
            return border, border - prev, flow
    border = np.asarray(edge[key], np.float32)
    if border.shape[0] != span:
        return None, None, None
    prev = np.asarray(edge.get(outer_key, border), np.float32)
    if prev.shape[0] != span:
        prev = border
    return border, border - prev, np.zeros(span, np.float32)


def _flow(strip):
    """Pixels a feature shifts along the edge for each step toward the new tile."""
    shifts = []
    pairs = min(3, strip.shape[1] - 1)
    for k in range(pairs):
        newer = strip[:, -1 - k]
        older = strip[:, -2 - k]
        shift, keep = _shift(newer, older)
        shifts.append(np.where(keep, shift, 0.0))
    if not shifts:
        return np.zeros(strip.shape[0], np.float32)
    return np.mean(shifts, axis=0).astype(np.float32)


def _shift(newer, older):
    """dy where newer[y] lines up with older[y - dy]. Reject a weak match."""
    err0 = _window((newer - older) ** 2)
    best = np.zeros(newer.shape[0], np.float32)
    err = err0.copy()
    for dy in range(-8, 9):
        if dy == 0:
            continue
        pred = _slide(older, dy)
        e = _window((newer - pred) ** 2)
        take = e < err
        best[take] = dy
        err[take] = e[take]
    return best, err < err0 * 0.8


def _slide(values, dy):
    """values[y - dy], clamped at the ends."""
    out = np.empty_like(values)
    if dy > 0:
        out[:dy] = values[0]
        out[dy:] = values[:-dy]
    elif dy < 0:
        out[dy:] = values[-1]
        out[:dy] = values[-dy:]
    else:
        out[:] = values
    return out


def _window(err):
    return np.convolve(np.pad(err, 2, mode="edge"), np.ones(5, np.float32) / 5.0, mode="valid")


def relight_edges(rgb, bed, edges, skip=None):
    """Fix the one-pixel lighting line where a tile was shaded against its own far side."""
    rgb = np.array(rgb, copy=True)
    if not edges:
        return rgb
    bed = np.asarray(bed, np.float32)
    wrong = _hillshade(bed)
    fixed = _hillshade(bed, edges)
    land = bed > 0
    lit0 = np.where(land, 0.55 + 0.75 * wrong, 0.82 + 0.28 * wrong)
    lit1 = np.where(land, 0.55 + 0.75 * fixed, 0.82 + 0.28 * fixed)
    factor = lit1 / np.maximum(lit0, 1e-3)
    mask = np.zeros(bed.shape, np.bool_)
    h, w = bed.shape
    for edge in edges:
        side = edge["side"]
        if side == "left":
            mask[:, 0] = True
        elif side == "right":
            mask[:, -1] = True
        elif side == "top":
            mask[0, :] = True
        else:
            mask[-1, :] = True
    if skip is not None:
        mask &= ~np.asarray(skip, np.bool_)
    lit = np.clip(np.rint(rgb[mask].astype(np.float32) * factor[mask, None]), 0, 255).astype(np.uint8)
    rgb[mask] = lit
    return rgb


def _hillshade(h, edges=None):
    """Same lighting as terrain.render.hillshade. Edge samples use the neighbor."""
    z = np.float32(28.0)
    hz = np.asarray(h, np.float32) * z
    height, width = hz.shape
    left = np.roll(hz, 1, axis=1)
    right = np.roll(hz, -1, axis=1)
    up = np.empty_like(hz)
    down = np.empty_like(hz)
    up[0] = hz[0]
    up[1:] = hz[:-1]
    down[-1] = hz[-1]
    down[:-1] = hz[1:]
    if edges:
        for edge in edges:
            outer = edge.get("shade", edge.get("outer"))
            if outer is None:
                continue
            outer = np.asarray(outer, np.float32) * z
            side = edge["side"]
            if side == "left" and outer.shape[0] == height:
                left[:, 0] = outer
            elif side == "right" and outer.shape[0] == height:
                right[:, -1] = outer
            elif side == "top" and outer.shape[0] == width:
                up[0, :] = outer
            elif side == "bottom" and outer.shape[0] == width:
                down[-1, :] = outer
    dy = (down - up) * np.float32(0.5)
    dx = (right - left) * np.float32(0.5)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(360.0 - 315.0 + 90.0)
    altitude = np.deg2rad(45.0)
    shade = (np.sin(altitude) * np.cos(slope) +
             np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    return np.clip(shade, 0, 1)


def downsample(image, max_edge):
    """Stride down to max_edge. Nearest sample, so it stays cheap on big maps."""
    image = np.asarray(image)
    h, w = image.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_edge)))
    if step == 1:
        return image
    return np.ascontiguousarray(image[::step, ::step])


def _hash(y, x, seed):
    y = y.astype(np.int64)
    x = x.astype(np.int64)
    n = (y * np.int64(374761393) + x * np.int64(668265263) + np.int64(seed)) & np.int64(0xFFFFFFFF)
    n ^= n >> np.int64(13)
    n = (n * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    return (n & np.int64(0xFFFFFF)).astype(np.float32) * np.float32(1.0 / 0xFFFFFF)


def _blend(grid, fy, fx):
    """Bilinear sample. `fy` is per row, `fx` per column, both in grid index space."""
    iy = np.floor(fy).astype(np.int32)
    ix = np.floor(fx).astype(np.int32)
    ty = (fy - iy).astype(np.float32)
    tx = (fx - ix).astype(np.float32)
    iy = np.clip(iy, 0, grid.shape[0] - 1)
    ix = np.clip(ix, 0, grid.shape[1] - 1)
    iy1 = np.clip(iy + 1, 0, grid.shape[0] - 1)
    ix1 = np.clip(ix + 1, 0, grid.shape[1] - 1)
    wide = grid[:, ix]
    wide = wide + (grid[:, ix1] - wide) * tx
    top = wide[iy]
    return top + (wide[iy1] - top) * ty[:, None]


def _value_noise(yy, xx, period, full_w, seed, wrap_x=True):
    """Smooth value noise on global pixel coordinates. x wraps with the world."""
    cells = max(1, int(full_w) // int(period))
    y_start = math.floor(float(yy[0, 0]) / period)
    y_end = math.floor(float(yy[-1, 0]) / period) + 1
    x_start = math.floor(float(xx[0, 0]) / period)
    x_end = math.floor(float(xx[0, -1]) / period) + 1
    ys = np.arange(y_start, y_end + 1, dtype=np.int64)
    xs = np.arange(x_start, x_end + 1, dtype=np.int64)
    lattice_x = xs % cells if wrap_x else xs
    grid = _hash(ys[:, None], lattice_x[None, :], seed)
    fy = (yy[:, 0] / period - y_start).astype(np.float32)
    fx = (xx[0, :] / period - x_start).astype(np.float32)
    ty = fy - np.floor(fy)
    tx = fx - np.floor(fx)
    fy = np.floor(fy) + ty * ty * (3.0 - 2.0 * ty)
    fx = np.floor(fx) + tx * tx * (3.0 - 2.0 * tx)
    return _blend(grid, fy, fx)


def detail(y0, x0, tile_h, tile_w, full_w, seed,
           wrap_x=True, origin=(0, 0)):
    """0..1 height grain for one tile. Neighbouring tiles meet without a seam."""
    yy = (origin[0] + y0 + np.arange(tile_h, dtype=np.float32))[:, None]
    xx = (origin[1] + x0 + np.arange(tile_w, dtype=np.float32))[None, :]
    acc = np.zeros((tile_h, tile_w), np.float32)
    amp, norm = 1.0, 0.0
    for i, period in enumerate(DETAIL_PERIODS):
        acc += np.float32(amp) * _value_noise(
            yy, xx, period, full_w, seed + i * 101, wrap_x=wrap_x)
        norm += amp
        amp *= 0.5
    acc /= np.float32(norm)
    return acc


def _sample_plane(field, y0, x0, tile_h, tile_w, full_h, full_w,
                  wrap_x=True):
    """Bilinear sample of one plane. x wraps, y clamps. Neighbouring tiles meet."""
    ch, cw = field.shape
    ys = (y0 + np.arange(tile_h, dtype=np.float32) + 0.5) * (ch / full_h) - 0.5
    xs = (x0 + np.arange(tile_w, dtype=np.float32) + 0.5) * (cw / full_w) - 0.5
    iy = np.floor(ys).astype(np.int32)
    ix = np.floor(xs).astype(np.int32)
    ty = (ys - iy).astype(np.float32)
    tx = (xs - ix).astype(np.float32)
    if wrap_x:
        x0i, x1i = ix % cw, (ix + 1) % cw
    else:
        x0i, x1i = np.clip(ix, 0, cw - 1), np.clip(ix + 1, 0, cw - 1)
    wide = field[:, x0i]
    wide = wide + (field[:, x1i] - wide) * tx
    y0i = np.clip(iy, 0, ch - 1)
    y1i = np.clip(iy + 1, 0, ch - 1)
    top = wide[y0i]
    return (top + (wide[y1i] - top) * ty[:, None]).astype(np.float32)


def _sample_tile(field, y0, x0, tile_h, tile_w, full_h, full_w,
                 wrap_x=True):
    if field.ndim == 2:
        return _sample_plane(field, y0, x0, tile_h, tile_w, full_h, full_w,
                             wrap_x=wrap_x)
    planes = [_sample_plane(field[..., c], y0, x0, tile_h, tile_w, full_h, full_w,
                            wrap_x=wrap_x)
              for c in range(field.shape[-1])]
    return np.stack(planes, axis=-1)


def _limits(height, detail_fraction=DETAIL_FRACTION, detail_reference_span=None):
    if not np.isfinite(detail_fraction) or not 0 <= detail_fraction <= 0.1:
        raise ValueError("detail_fraction must be finite and between 0 and 0.1")
    if not np.isfinite(height).all():
        raise ValueError("height contains NaN or infinite values")
    hmin = float(height.min())
    hmax = float(height.max())
    span = max(hmax - hmin, 1e-9)
    if detail_reference_span is not None:
        if not np.isfinite(detail_reference_span) or detail_reference_span <= 0:
            raise ValueError("detail_reference_span must be finite and positive")
    detail_amp = detail_fraction * (
        span if detail_reference_span is None else detail_reference_span)
    lo, hi = hmin - detail_amp, hmax + detail_amp
    return lo, hi, detail_amp, max(hi - lo, 1e-9)


def _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled,
              detail_fraction=DETAIL_FRACTION, generator_info=None,
              wrap_x=True, tile_origin=(0, 0), detail_reference_span=None,
              detail_mode="noise", world_width_m=200_000.0):
    meta = {
        "width": int(full_w),
        "height": int(full_h),
        "seed": int(seed),
        "simulated": [int(height.shape[1]), int(height.shape[0])],
        "wrap": "x" if wrap_x else "none (atlas tile)",
        "detail_coordinate_origin_yx": list(map(int, tile_origin)),
        "height_bits": 16,
        "lo": lo,
        "hi": hi,
        "units_per_code": scale / 65535.0,
        "metres_per_unit": METRES_PER_UNIT,
        "world_width_m": float(world_width_m),
        "sea_code": (0.0 - lo) / scale * 65535.0,
        "detail": (f"seeded value noise, +/- {detail_fraction:.1%} of "
                   + ("the simulated height range" if detail_reference_span is None
                      else f"a fixed {detail_reference_span:g}-unit atlas reference span")),
        "detail_fraction": float(detail_fraction),
        "detail_reference_span": detail_reference_span,
        "detail_mode": detail_mode,
        "simulation": (
            "hierarchical window regeneration: regional ridges, gullies and sediment "
            "are solved from world metres, then cropped"
            if detail_mode == "hierarchical"
            else "single preview-resolution terrain solved and upscaled; new noise is not eroded"
        ),
        "material_mask": "legacy game RGBA: R rock, G ground, B cover, A accent",
        "flow_mask": "legacy game RGB: R wetness, G/B zero",
        "advanced_bundle_materials": "separate format: RGBA sand / soil-grass / rock / snow",
    }
    if generator_info is not None:
        meta["generator"] = dict(generator_info)
    if tiled:
        meta.update({
            "tile": TILE,
            "tiles_x": math.ceil(full_w / TILE),
            "tiles_y": math.ceil(full_h / TILE),
            "files": {"height": "height/y######_x######.png",
                      "material_mask": "mask/y######_x######.png",
                      "flow": "flow/y######_x######.png"},
        })
    return meta


def _as_float(image):
    return np.ascontiguousarray(image, dtype=np.float32)


def _write_color(image, path, mode):
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode).save(
        path, compress_level=1)


def _apply_detail(sampled, y0, x0, full_w, seed, detail_amp,
                  wrap_x=True, tile_origin=(0, 0)):
    if detail_amp == 0:
        return sampled
    grain = detail(y0, x0, sampled.shape[0], sampled.shape[1], full_w, seed,
                   wrap_x=wrap_x, origin=tile_origin)
    return sampled + (grain - np.float32(0.5)) * np.float32(2.0 * detail_amp)


def _export_limits(height, detail_mode, detail_fraction, detail_reference_span):
    if detail_mode == "noise":
        return _limits(height, detail_fraction, detail_reference_span)
    if detail_mode != "hierarchical":
        raise ValueError("detail_mode must be noise or hierarchical")
    if not np.isfinite(height).all():
        raise ValueError("height contains NaN or infinite values")
    hmin = float(np.min(height))
    hmax = float(np.max(height))
    lo, hi = hmin - RELIEF_UNITS, hmax + RELIEF_UNITS
    return lo, hi, 0.0, max(hi - lo, 1e-9)


def _refine(macro, y0, x0, tile_h, tile_w, full_h, full_w, seed, detail_amp,
            detail_mode, wrap_x, tile_origin, world_width_m):
    """Upscale-and-grain, or regenerate this window from world metres."""
    if detail_mode == "noise":
        sampled = _sample_tile(macro, y0, x0, tile_h, tile_w, full_h, full_w,
                               wrap_x=wrap_x)
        return _apply_detail(sampled, y0, x0, full_w, seed, detail_amp,
                             wrap_x=wrap_x, tile_origin=tile_origin)
    world_height_m = float(world_width_m) * float(full_h) / float(full_w)
    return hierarchical_window(
        macro, y0, x0, tile_h, tile_w, full_h, full_w, seed,
        float(world_width_m), world_height_m, wrap_x=wrap_x)


def write_direct(height, mask, flow, full_w, full_h, seed, dest,
                 detail_fraction=DETAIL_FRACTION, generator_info=None,
                 wrap_x=True, tile_origin=(0, 0), detail_seed=None,
                 detail_reference_span=None, detail_mode="noise",
                 world_width_m=200_000.0):
    """One image per map. Used up to 8K, where the upscale still fits in memory."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    height_f, mask_f, flow_f = _as_float(height), _as_float(mask), _as_float(flow)
    lo, hi, detail_amp, scale = _export_limits(
        height_f, detail_mode, detail_fraction, detail_reference_span)
    sampled = _refine(
        height_f, 0, 0, full_h, full_w, full_h, full_w,
        seed if detail_seed is None else detail_seed, detail_amp, detail_mode,
        wrap_x, tile_origin, world_width_m)
    coded = np.rint(np.clip((sampled - lo) / scale, 0, 1) * 65535).astype(np.uint16)
    _write_png(dest / "height16.png", coded, 16, level=1)
    _write_color(_sample_tile(mask_f, 0, 0, full_h, full_w, full_h, full_w,
                              wrap_x=wrap_x), dest / "material_mask.png", "RGBA")
    _write_color(_sample_tile(flow_f, 0, 0, full_h, full_w, full_h, full_w,
                              wrap_x=wrap_x), dest / "flow_mask.png", "RGB")
    (dest / "manifest.json").write_text(json.dumps(
        _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled=False,
                  detail_fraction=detail_fraction, generator_info=generator_info,
                  wrap_x=wrap_x, tile_origin=tile_origin,
                  detail_reference_span=detail_reference_span,
                  detail_mode=detail_mode, world_width_m=world_width_m), indent=1))
    return dest


def write_tiles(height, mask, flow, full_w, full_h, seed, dest,
                on_status=None, on_tile=None, detail_fraction=DETAIL_FRACTION,
                generator_info=None, wrap_x=True, tile_origin=(0, 0),
                detail_seed=None, detail_reference_span=None, max_workers=2,
                detail_mode="noise", world_width_m=200_000.0):
    """Write a map wider than memory as 4096 tiles under `dest`."""
    dest = Path(dest)
    height_dir, mask_dir, flow_dir = dest / "height", dest / "mask", dest / "flow"
    for folder in (height_dir, mask_dir, flow_dir):
        folder.mkdir(parents=True, exist_ok=True)

    height_f, mask_f, flow_f = _as_float(height), _as_float(mask), _as_float(flow)
    lo, hi, detail_amp, scale = _export_limits(
        height_f, detail_mode, detail_fraction, detail_reference_span)
    tiles_y = math.ceil(full_h / TILE)
    tiles_x = math.ceil(full_w / TILE)
    total = tiles_y * tiles_x
    state = {"done": 0}
    lock = threading.Lock()

    def one(ty, tx):
        y0, x0 = ty * TILE, tx * TILE
        th, tw = min(TILE, full_h - y0), min(TILE, full_w - x0)
        name = f"y{y0:06d}_x{x0:06d}.png"
        sampled = _refine(
            height_f, y0, x0, th, tw, full_h, full_w,
            seed if detail_seed is None else detail_seed, detail_amp,
            detail_mode, wrap_x, tile_origin, world_width_m)
        coded = np.rint(np.clip((sampled - lo) / scale, 0, 1) * 65535).astype(np.uint16)
        _write_png(height_dir / name, coded, 16, level=1)
        _write_color(_sample_tile(mask_f, y0, x0, th, tw, full_h, full_w,
                                  wrap_x=wrap_x), mask_dir / name, "RGBA")
        _write_color(_sample_tile(flow_f, y0, x0, th, tw, full_h, full_w,
                                  wrap_x=wrap_x), flow_dir / name, "RGB")
        with lock:
            state["done"] += 1
            done = state["done"]
        if on_tile:
            on_tile(done, total)
        if on_status and (done == total or done % 8 == 0):
            on_status(f"writing tiles {done}/{total}")

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    workers = min(int(max_workers), os.cpu_count() or 2, total)
    coords = [(ty, tx) for ty in range(tiles_y) for tx in range(tiles_x)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: one(*pair), coords))

    (dest / "manifest.json").write_text(json.dumps(
        _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled=True,
                  detail_fraction=detail_fraction, generator_info=generator_info,
                  wrap_x=wrap_x, tile_origin=tile_origin,
                  detail_reference_span=detail_reference_span,
                  detail_mode=detail_mode, world_width_m=world_width_m), indent=1))
    return dest
