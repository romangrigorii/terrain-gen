"""Layer rendering: numpy height/plate fields -> uint8 RGB images."""
import numpy as np

from . import climate, grid

SEA_RAMP = [
    (-1.00, (11, 30, 64)), (-0.70, (13, 36, 75)), (-0.45, (16, 44, 88)),
    (-0.25, (20, 54, 100)), (-0.12, (25, 64, 112)), (-0.05, (30, 74, 123)),
    (0.0, (37, 86, 134)),
]
LAND_RAMP = [
    (0.00, (68, 108, 72)), (0.14, (88, 128, 76)), (0.34, (112, 142, 82)),
    (0.52, (142, 148, 94)), (0.68, (152, 132, 100)), (0.80, (136, 118, 106)),
    (0.90, (176, 174, 174)), (1.00, (252, 252, 255)),
]
PLATE_CLASS = {1: (220, 60, 50), 2: (60, 140, 235), 3: (250, 200, 60)}


def _ramp(vals, ramp):
    stops = np.array([s for s, _ in ramp])
    cols = np.array([c for _, c in ramp], dtype=float)
    out = np.empty(vals.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(vals, stops, cols[:, c])
    return out


def _rank(vals, pool, n=20000):
    """Where each value sits in `pool`'s distribution, 0..1."""
    if pool.size == 0:
        return np.zeros_like(vals)
    s = np.sort(pool if pool.size <= n else np.random.default_rng(0).choice(pool, n))
    return np.searchsorted(s, vals, side="right") / len(s)


def _cell_noise(shape, seed, origin=(0, 0)):
    """Stable value in -1..1 for each cell. `origin` is the global cell of local (0, 0)."""
    y, x = np.indices(shape)
    y = y.astype(np.int64) + np.int64(origin[0])
    x = x.astype(np.int64) + np.int64(origin[1])
    n = (y * np.int64(374761393)
         + x * np.int64(668265263)
         + np.int64(seed)) & np.int64(0xFFFFFFFF)
    n ^= n >> np.int64(13)
    n = (n * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    return n.astype(np.float32) * np.float32(2.0 / 4294967295.0) - np.float32(1.0)


def _smoother(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _sway(shape, seed, smooth=1.8, origin=(0, 0)):
    """Smooth noise brought back to about -1..1. Blur alone would flatten it."""
    field = grid.blur(_cell_noise(shape, int(seed), origin), float(smooth))
    scale = float(np.percentile(np.abs(field), 90))
    if scale < 1e-6:
        return field
    return np.clip(field / np.float32(scale), -1.0, 1.0).astype(np.float32)


def _shore_mix(h, seed=11, reach=3.4, origin=(0, 0)):
    """1 inland, 0 at sea. The join wanders, so the shore is not a cell staircase.

    The shared rim keeps the hard cut. A neighbor tile paints that same pixel.
    """
    h = np.asarray(h, np.float64)
    gy, gx = grid.gradient(h)
    slope = np.maximum(np.hypot(gy, gx), 1e-6)
    shifted = h + _sway(h.shape, seed, origin=origin) * slope * float(reach)
    mix = _smoother(shifted / (slope * 1.35) + 0.5).astype(np.float32)
    hard = (h > 0.0).astype(np.float32)
    mix[0, :] = hard[0, :]
    mix[-1, :] = hard[-1, :]
    mix[:, 0] = hard[:, 0]
    mix[:, -1] = hard[:, -1]
    return mix


def _fray_labels(ids, seed, reach=4.5, origin=(0, 0)):
    """Move a colour boundary off the grid. Cells far from a border stay put."""
    ids = np.asarray(ids)
    edge = np.zeros(ids.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        edge |= grid.shift(ids, dy, dx) != ids
    if not edge.any():
        return ids
    dist = grid.edt(~edge)
    fade = np.clip(1.0 - dist / (float(reach) + 1.0), 0.0, 1.0).astype(np.float32)
    near = fade > 0.0
    rows, cols = ids.shape
    yy = np.arange(rows, dtype=np.float32)[:, None]
    xx = np.arange(cols, dtype=np.float32)[None, :]
    sway_y = _sway(ids.shape, seed, smooth=2.4, origin=origin)
    sway_x = _sway(ids.shape, seed + 29, smooth=2.4, origin=origin)
    sy = np.clip(np.rint(yy + sway_y * float(reach) * fade), 0, rows - 1).astype(np.int32)
    sx = np.mod(np.rint(xx + sway_x * float(reach) * fade).astype(np.int32), cols)
    picked = ids[sy, sx]
    same = ((picked == 0) == (ids == 0)) & near
    out = np.array(ids, copy=True)
    out[same] = picked[same]
    out[0, :] = ids[0, :]
    out[-1, :] = ids[-1, :]
    out[:, 0] = ids[:, 0]
    out[:, -1] = ids[:, -1]
    return out


def _ragged_cover(mask, seed, reach=3.2, sigma=0.85, origin=(0, 0)):
    """Warp a binary mask so its outline is a curve, then soften that curve."""
    mask = np.asarray(mask, np.float32)
    rows, cols = mask.shape
    yy = np.arange(rows, dtype=np.float32)[:, None]
    xx = np.arange(cols, dtype=np.float32)[None, :]
    sy = np.clip(np.rint(yy + _sway(mask.shape, seed, origin=origin) * float(reach)), 0, rows - 1).astype(np.int32)
    sx = np.mod(np.rint(xx + _sway(mask.shape, int(seed) + 17, origin=origin) * float(reach)).astype(np.int32), cols)
    warped = mask[sy, sx]
    if sigma <= 0:
        return warped
    return grid.blur(warped, float(sigma))


def color_stats(h):
    """Land and sea scales for one height field. Reuse them for every tile of that field."""
    h = np.asarray(h, np.float64)
    land_v, sea_v = h[h > 0], h[h < 0]
    sea_scale = max(1e-6, -float(np.percentile(sea_v, 2)) if sea_v.size else 1.0)
    land_scale = max(1e-6, float(np.percentile(land_v, 98)) if land_v.size else 1.0)
    return {"land": land_v, "sea": sea_v, "land_scale": land_scale, "sea_scale": sea_scale}


def _view_origin(world):
    origin = getattr(world, "tile_origin", None)
    if not origin:
        return (0, 0)
    return int(origin[0]), int(origin[1])


def hypsometric(h, land_mix=0.35, sea_mix=0.22, stats=None, origin=(0, 0)):
    """Blue below sea level, green-to-white above, each side scaled separately."""
    if stats is None:
        stats = color_stats(h)
    land_v, sea_v = stats["land"], stats["sea"]
    sea_scale = float(stats["sea_scale"])
    land_scale = float(stats["land_scale"])
    sea = _ramp((1 - sea_mix) * np.clip(h / sea_scale, -1, 0) -
                sea_mix * (1 - _rank(h, sea_v)), SEA_RAMP)
    land = _ramp((1 - land_mix) * np.clip(h / land_scale, 0, 1) +
                 land_mix * _rank(h, land_v), LAND_RAMP)
    mix = _shore_mix(h, origin=origin)[..., None]
    return sea * (1.0 - mix) + land * mix


def hillshade(h, azimuth=315.0, altitude=45.0, z=28.0):
    dy, dx = grid.gradient(h * z)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    shade = (np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def relief(h, shade=True, surface=None, tint=None, tint_mix=0.0, stats=None, origin=(0, 0)):
    """Hypsometric tint of the bed, hillshaded off `surface` if water sits on it."""
    rgb = hypsometric(h, stats=stats, origin=origin)
    shore = _shore_mix(h, origin=origin)[..., None]
    if tint is not None and tint_mix > 0:
        gain = float(tint_mix) * shore
        rgb = rgb * (1.0 - gain) + tint * gain
    if shade:
        s = hillshade(h if surface is None else surface)[..., None]
        lit = (0.82 + 0.28 * s) * (1.0 - shore) + (0.55 + 0.75 * s) * shore
        rgb = np.clip(rgb * lit, 0, 255)
    return rgb

LAKE_RAMP = [(0.0, (104, 158, 188)), (0.35, (84, 138, 178)), (1.0, (60, 108, 158))]
LAKE_DEPTH_BLUR = 2.0   # softens the drowned channel out of the depth shading
RIVER_RAMP = [(0.0, (110, 165, 195)), (0.4, (78, 132, 176)), (1.0, (48, 96, 150))]


def overpaint_water(rgb, w, origin=(0, 0)):
    """Lay lakes and rivers over whatever the land was painted with."""
    wat = w.water
    if wat.lake_mask.any():
        soft = grid.blur(wat.lake_depth, LAKE_DEPTH_BLUR)
        d = np.clip(soft / max(1e-6, float(np.percentile(
            soft[wat.lake_mask], 92))), 0, 1)
        cover = _ragged_cover(wat.lake_mask, seed=int(w.cfg.seed) + 23, origin=origin)[..., None]
        rgb = rgb * (1.0 - cover) + _ramp(d, LAKE_RAMP) * cover
    if wat.river_mask.any():
        v = np.clip(wat.width / max(1e-6, w.cfg.river_width_max), 0, 1)
        cover = _ragged_cover(wat.river_mask, seed=int(w.cfg.seed) + 47,
                              reach=1.4, sigma=0.65, origin=origin)[..., None]
        rgb = rgb * (1.0 - cover) + _ramp(v, RIVER_RAMP) * cover
    return rgb


def tint_placed(rgb, world):
    """Pale ice, tan dunes, and a darker cave mouth on the finished map."""
    ice = getattr(world, "placed_ice", None)
    sand = getattr(world, "placed_sand", None)
    void = getattr(world, "placed_void", None)
    if ice is None and sand is None and void is None:
        return rgb
    out = np.asarray(rgb, np.float64)
    if ice is not None and float(np.max(ice)) > 0.0:
        gain = np.clip(np.asarray(ice, np.float64) / 80.0, 0.0, 1.0)[..., None]
        out = out * (1.0 - 0.55 * gain) + np.array([210.0, 225.0, 235.0]) * (0.55 * gain)
    if sand is not None and float(np.max(sand)) > 0.0:
        gain = np.clip(np.asarray(sand, np.float64) / 40.0, 0.0, 1.0)[..., None]
        out = out * (1.0 - 0.4 * gain) + np.array([214.0, 184.0, 120.0]) * (0.4 * gain)
    if void is not None and float(np.max(void)) > 0.0:
        gain = np.clip(np.asarray(void, np.float64) / 30.0, 0.0, 1.0)[..., None]
        out = out * (1.0 - 0.4 * gain)
    return out


def water_map(w, shade=True, tint_mix=None):
    """The main view: relief with lakes at their own flat level and rivers by width."""
    origin = _view_origin(w)
    stats = getattr(w, "color_stats", None)
    rgb = relief(w.height, shade, surface=w.surface, tint=biome_rgb(w),
                 tint_mix=w.cfg.biome_tint if tint_mix is None else tint_mix,
                 stats=stats, origin=origin)
    rgb = paint_trees(rgb, w.trees, w.height.shape, w.cfg.tree_relief)
    rgb = tint_placed(overpaint_water(rgb, w, origin), w)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def plate_colors(n, rng=None):
    rng = rng or np.random.default_rng(7)
    hs = rng.permutation(n) / max(1, n)
    i = (hs * 6).astype(int) % 6
    f = hs * 6 - np.floor(hs * 6)
    v, s = 0.88, 0.55
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255).astype(float)


def plates(t, cls=None, shade_height=None):
    cols = plate_colors(len(t.seeds))
    rgb = cols[t.plate]
    rgb = np.where(t.is_cont[..., None], rgb, rgb * 0.55 + 30)
    if shade_height is not None:
        rgb = np.clip(rgb * (0.65 + 0.5 * hillshade(shade_height)[..., None]), 0, 255)
    if cls is not None:
        for k, c in PLATE_CLASS.items():
            rgb = np.where((cls == k)[..., None], np.array(c, dtype=float), rgb)
    return rgb.astype(np.uint8)


def scalar(a, lo=None, hi=None, ramp=None):
    """Generic diverging/sequential view of any field."""
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    v = np.clip((a - lo) / max(1e-9, hi - lo), 0, 1)
    ramp = ramp or [(0.0, (12, 12, 30)), (0.35, (40, 70, 130)),
                    (0.6, (200, 120, 60)), (0.85, (240, 200, 90)), (1.0, (255, 255, 240))]
    return _ramp(v, ramp).astype(np.uint8)


def stress(t):
    """Red convergence, blue extension, green shear, all on one image."""
    rgb = np.zeros(t.plate.shape + (3,))
    rgb[..., 0] = t.conv
    rgb[..., 2] = t.rift
    rgb[..., 1] = t.shear * 0.8
    fade = np.exp(-t.dist / 25.0)[..., None]
    return np.clip(rgb * fade * 320 + 12, 0, 255).astype(np.uint8)


def flow(acc, h, threshold):
    """Log drainage area over a dim relief, so the whole network is visible."""
    base = relief(h, shade=True) * 0.55
    la = np.log10(np.maximum(acc, 1.0))
    lo = np.log10(max(2.0, threshold * 0.04))     # faintest tributary shown
    hi = np.log10(max(10.0, float(acc.max())))
    v = np.clip((la - lo) / (hi - lo), 0, 1)
    water = np.stack([60 + 60 * v, 130 + 90 * v, 190 + 60 * v], axis=-1)
    strength = np.where(h > 0, v ** 0.7, 0)[..., None]
    return np.clip(base * (1 - strength) + water * strength, 0, 255).astype(np.uint8)

RAIN_RAMP = [(0.0, (196, 172, 120)), (0.35, (176, 186, 120)),
             (0.7, (96, 158, 108)), (1.0, (36, 96, 132))]


def rain_map(w):
    """Runoff per cell over a dim relief: tan is desert, blue-green is soaked."""
    base = relief(w.height, shade=True) * 0.5
    v = _rank(w.runoff, w.runoff[w.land])
    rain = _ramp(v, RAIN_RAMP)
    return np.clip(np.where(w.land[..., None], rain * 0.72 + base * 0.5, base),
                   0, 255).astype(np.uint8)

BIOME_COLORS = np.array([
    (37, 86, 134),      # ocean, only ever a placeholder
    (240, 246, 252),    # ice
    (154, 158, 146),    # tundra
    (58, 94, 78),       # taiga
    (224, 190, 128),    # desert               one class, picked by rain alone
    (176, 182, 108),    # steppe
    (92, 134, 74),      # temperate forest
    (48, 102, 70),      # temperate rainforest
    (150, 138, 86),     # shrubland            olive-brown, against steppe's green
    (202, 176, 92),     # savanna
    (122, 152, 64),     # tropical seasonal forest
    (34, 96, 48),       # tropical rainforest
], dtype=float)


def biome_rgb(w, soften=None):
    """Biome colour. Borders wander off the cell grid; sea stays for the ramp."""
    origin = _view_origin(w)
    land_ids = np.where(w.land, w.biome, 0)
    rgb = BIOME_COLORS[_fray_labels(land_ids, seed=int(w.cfg.seed) + 41, origin=origin)]
    s = w.cfg.biome_soften if soften is None else soften
    if s <= 0:
        return rgb
    m = _shore_mix(w.height, seed=int(w.cfg.seed) + 7, origin=origin)
    num = np.stack([grid.blur(rgb[..., c] * m, s) for c in range(3)], axis=-1)
    return num / np.maximum(grid.blur(m, s), 1e-6)[..., None]


def biome_map(w):
    """Flat biome colour, hillshaded, with the water drawn back over it."""
    origin = _view_origin(w)
    stats = getattr(w, "color_stats", None)
    rgb = biome_rgb(w)
    shore = _shore_mix(w.height, seed=int(w.cfg.seed) + 7, origin=origin)[..., None]
    lit = rgb * (0.78 + 0.38 * hillshade(w.height)[..., None])
    rgb = hypsometric(w.height, stats=stats, origin=origin) * (1.0 - shore) + lit * shore
    return np.clip(overpaint_water(rgb, w, origin), 0, 255).astype(np.uint8)

LEGEND_COLORS = np.array([
    (37, 86, 134),      # ocean, only ever a placeholder
    (238, 244, 250),    # ice
    (176, 166, 150),    # tundra                warm grey
    (46, 92, 64),       # taiga                 dark green
    (234, 176, 104),    # desert                orange-tan
    (198, 206, 116),    # steppe                chartreuse, off the desert tan
    (92, 158, 72),      # temperate forest      mid green
    (22, 126, 124),     # temperate rainforest  teal, well off taiga
    (166, 140, 80),     # shrubland             olive-brown
    (214, 158, 60),     # savanna               gold, darker than steppe
    (140, 182, 62),     # tropical seasonal     lime
    (10, 74, 34),       # tropical rainforest   darkest green, clear of taiga
], dtype=float)

LEGEND_FONT = "georgia,timesnewroman,serif"
LEGEND_PAPER = np.array([250.0, 248.0, 243.0])
LEGEND_INK = np.array([28.0, 26.0, 24.0])
LEGEND_PAD = 8


def _legend_font(size, _cache={}):
    if size not in _cache:
        import pygame
        pygame.font.init()
        _cache[size] = pygame.font.SysFont(LEGEND_FONT, size)
    return _cache[size]


def _glyph(ch, size, _cache={}):
    """One rasterised letter as an alpha array, and how far to step after it."""
    import pygame
    key = (ch, size)
    if key not in _cache:
        img = _legend_font(size).render(ch, True, (255, 255, 255))
        _cache[key] = (pygame.surfarray.array_alpha(img).T.astype(float) / 255.0,
                       img.get_width())
    return _cache[key]


def _draw_text(rgb, text, size, y, x, ink):
    """Stamp `text` into `rgb` with its top-left at (y, x). Clipped, not wrapped: the panel is drawn at a size that fits, and a map too small for it would"""
    h, w = rgb.shape[:2]
    for ch in text:
        a, adv = _glyph(ch, size)
        th, tw = a.shape
        ys, xs = np.arange(y, y + th), np.arange(x, x + tw)
        ky, kx = (ys >= 0) & (ys < h), (xs >= 0) & (xs < w)
        if ky.any() and kx.any():
            al = a[ky][:, kx][..., None]
            box = rgb[ys[ky][:, None], xs[kx][None, :]]
            rgb[ys[ky][:, None], xs[kx][None, :]] = box * (1 - al) + ink * al
        x += adv
    return rgb


def _fill(rgb, y0, y1, x0, x1, colour):
    """Paint a rectangle, clipped to the image rather than wrapped."""
    h, w = rgb.shape[:2]
    y0, y1, x0, x1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if y1 > y0 and x1 > x0:
        rgb[y0:y1, x0:x1] = colour
    return rgb


def present_biomes(w):
    """The biomes this world actually has on land, in id order."""
    return [b for b in range(1, len(LEGEND_COLORS)) if (w.biome[w.land] == b).any()]


def _corner(shape, land, box_h, box_w):
    """Top-left of the emptiest corner: a key belongs over water, not a coast."""
    h, w = shape
    corners = [(8, 8), (8, w - box_w - 8), (h - box_h - 8, 8),
               (h - box_h - 8, w - box_w - 8)]
    return min(corners, key=lambda c: land[max(0, c[0]):c[0] + box_h,
                                           max(0, c[1]):c[1] + box_w].mean())


def draw_legend(rgb, ids, land, size=None):
    """A key in the emptiest corner: a swatch and a name per biome on this map."""
    h, w = rgb.shape[:2]
    size = size or int(np.clip(round(w / 54), 9, 16))
    f = _legend_font(size)
    row = int(f.get_height() * 1.3)
    sw = int(f.get_height() * 0.9)
    names = [climate.BIOME_NAMES[b] for b in ids]
    box_w = LEGEND_PAD * 2 + sw + 8 + max(
        sum(_glyph(c, size)[1] for c in n) for n in names)
    box_h = LEGEND_PAD * 2 + row * len(ids)
    y0, x0 = _corner((h, w), land, box_h, box_w)
    _fill(rgb, y0, y0 + box_h, x0, x0 + box_w, LEGEND_PAPER)
    for i, (b, name) in enumerate(zip(ids, names)):
        top, left = y0 + LEGEND_PAD + i * row, x0 + LEGEND_PAD
        _fill(rgb, top, top + sw, left, left + sw, LEGEND_INK)
        _fill(rgb, top + 1, top + sw - 1, left + 1, left + sw - 1, LEGEND_COLORS[b])
        _draw_text(rgb, name, size, top + (sw - f.get_height()) // 2,
                   left + sw + 8, LEGEND_INK)
    return rgb

BORDER = np.array([38.0, 36.0, 32.0])
BORDER_A = 0.42     # a line to follow, not a wall


def biome_edges(w):
    """Cells with a different land biome next door: the internal borders."""
    edge = np.zeros(w.biome.shape, bool)
    for dy, dx in grid.NEIGH4:
        edge |= (grid.shift(w.biome, dy, dx) != w.biome) & grid.shift(w.land, dy, dx)
    return edge & w.land & ~(w.rivers | w.lakes)


def legend_biome_map(w):
    """One colour per biome, with a ragged join, and a key to read them by."""
    origin = _view_origin(w)
    stats = getattr(w, "color_stats", None)
    land_ids = np.where(w.land, w.biome, 0)
    colours = LEGEND_COLORS[_fray_labels(land_ids, seed=int(w.cfg.seed) + 41, origin=origin)]
    lit = colours * (0.86 + 0.24 * hillshade(w.height)[..., None])
    shore = _shore_mix(w.height, seed=int(w.cfg.seed) + 7, origin=origin)[..., None]
    rgb = hypsometric(w.height, stats=stats, origin=origin) * (1.0 - shore) + lit * shore
    rgb = overpaint_water(rgb, w, origin)
    return draw_legend(np.clip(rgb, 0, 255), present_biomes(w), w.land).astype(np.uint8)

TEMP_RAMP = [(-30.0, (78, 96, 168)), (-10.0, (120, 168, 208)),
             (0.0, (226, 234, 240)), (12.0, (232, 206, 132)),
             (24.0, (216, 130, 66)), (34.0, (166, 48, 44))]


def temp_map(w):
    """Mean annual temperature, absolute, over a dim relief."""
    base = relief(w.height, shade=True) * 0.45
    return np.clip(_ramp(w.temp, TEMP_RAMP) * 0.75 + base * 0.5,
                   0, 255).astype(np.uint8)

TREE_STAMPS = [
    [(-2, 0, .40), (-1, 0, .85), (0, 0, 1.0), (0, -1, .45), (0, 1, .45), (1, 0, .65)],
    [(-1, 0, .75), (-1, -1, .35), (-1, 1, .35), (0, 0, 1.0), (0, -1, .85),
     (0, 1, .85), (1, 0, .70)],
    [(-1, 0, .45), (0, 0, .90), (0, -1, .80), (0, 1, .80), (0, -2, .50), (0, 2, .50)],
    [(0, 0, .75), (0, 1, .35)],
]
TREE_COLORS = np.array([
    (26, 58, 44),       # conifer     - blue-green, and the darkest of the four
    (52, 96, 50),       # broadleaf
    (96, 130, 58),      # palm        - yellower, which is what dry country does
    (114, 124, 72),     # scrub
], dtype=float)
TREE_SHADOW = np.array([16, 24, 20.0])
TREE_SHADOW_A = 0.38    # alpha of the offset shadow pass


def _splat(shape, ys, xs, offsets, weights):
    """Accumulate one stamp over every tree at once: one pass per offset, not per tree. Wraps in x and clamps in y, like every other neighbour read."""
    h, w = shape
    acc = np.zeros(shape)
    for dy, dx, a in offsets:
        np.add.at(acc, (np.clip(ys + dy, 0, h - 1), (xs + dx) % w), a * weights)
    return acc


def paint_trees(rgb, t, shape, strength=1.0):
    """Stamp individual trees over whatever the ground was painted with."""
    if len(t) == 0 or strength <= 0:
        return rgb
    ys = np.clip(t.pos[:, 0].astype(int), 0, shape[0] - 1)
    xs = t.pos[:, 1].astype(int) % shape[1]
    shade = np.clip(_splat(shape, ys + 1, xs + 1, [(0, 0, 1.0)], t.size), 0, 1)
    sa = (TREE_SHADOW_A * strength * shade)[..., None]
    rgb = rgb * (1 - sa) + TREE_SHADOW * sa
    for k, offsets in enumerate(TREE_STAMPS):
        m = t.kind == k
        if not m.any():
            continue
        a = strength * np.clip(
            _splat(shape, ys[m], xs[m], offsets, t.size[m]), 0, 1)[..., None]
        rgb = rgb * (1 - a) + TREE_COLORS[k] * a
    return rgb

GROUND_DESAT = 0.62     # how far the base is pulled towards its own luminance
GROUND_WARM = np.array([1.06, 1.00, 0.88])   # and then back towards earth


def tree_map(w):
    """The trees, on ground the colour of ground."""
    base = relief(w.height, surface=w.surface, tint=biome_rgb(w), tint_mix=0.5)
    lum = base @ np.array([0.30, 0.59, 0.11])
    ground = (base * (1 - GROUND_DESAT) + lum[..., None] * GROUND_DESAT) * GROUND_WARM
    base = np.where(w.land[..., None], ground * 0.94, base)
    return np.clip(overpaint_water(paint_trees(base, w.trees, w.height.shape), w),
                   0, 255).astype(np.uint8)


def canopy_map(w):
    """The density field the trees were scattered from, on its own."""
    base = relief(w.height, shade=True) * 0.45
    cov = _ramp(w.trees.cover, [(0.0, (198, 180, 144)), (0.35, (150, 164, 96)),
                                (0.7, (74, 122, 62)), (1.0, (26, 66, 42))])
    return np.clip(overpaint_water(
        np.where(w.land[..., None], cov * 0.78 + base * 0.5, base), w),
        0, 255).astype(np.uint8)


def land_mask(h, lakes=None):
    rgb = np.where((h > 0)[..., None], np.array([232, 226, 208.0]),
                   np.array([28, 52, 84.0]))
    coast = (h > 0) != (grid.blur((h > 0).astype(float), 1.2) > 0.5)
    rgb = np.where(coast[..., None], np.array([20, 20, 20.0]), rgb)
    if lakes is not None:
        rgb = np.where(lakes[..., None], np.array([70, 120, 170.0]), rgb)
    return rgb.astype(np.uint8)


def slope_map(h):
    dy, dx = grid.gradient(h)
    return scalar(np.hypot(dy, dx), 0, float(np.percentile(np.hypot(dy, dx), 99.5)))


def erosion_diff(pre, post):
    d = post - pre
    lim = max(1e-6, float(np.percentile(np.abs(d), 99.5)))
    v = np.clip(d / lim, -1, 1)
    return _ramp(v, [(-1.0, (200, 60, 40)), (0.0, (25, 25, 28)),
                     (1.0, (60, 150, 220))]).astype(np.uint8)

LAYERS = [
    ("main view", water_map),
    ("elevation (no water)", lambda w: relief(w.height).astype(np.uint8)),
    ("tectonic relief", lambda w: relief(w.height_raw).astype(np.uint8)),
    ("pre-erosion", lambda w: relief(w.height_pre).astype(np.uint8)),
    ("plates", lambda w: plates(w.tect, shade_height=w.height)),
    ("boundaries", lambda w: plates(w.tect, cls=_cls(w), shade_height=w.height)),
    ("stress", lambda w: stress(w.tect)),
    ("rainfall", rain_map),
    ("temperature", temp_map),
    ("biomes", biome_map),
    ("trees", tree_map),
    ("biome legend", legend_biome_map),
    ("main view (biomes)", lambda w: water_map(w, tint_mix=1.0)),
    ("drainage", lambda w: flow(w.flow, w.height, w.cfg.river_threshold * w.height.size)),
    ("erosion delta", lambda w: erosion_diff(w.height_pre, w.height_eroded)),
    ("slope", lambda w: slope_map(w.height)),
    ("land / coast", lambda w: land_mask(w.height, w.lakes)),
    ("river width", lambda w: river_width_map(w)),
    ("canopy", canopy_map),
]


def river_width_map(w):
    """Channel width and lake extent on a plain background, no shading."""
    rgb = np.where((w.height > 0)[..., None], np.array([238, 232, 216.0]),
                   np.array([32, 56, 88.0]))
    rgb = np.where(w.lakes[..., None], np.array([120, 160, 190.0]), rgb)
    v = np.clip(w.water.width / max(1e-6, w.cfg.river_width_max), 0, 1)
    return np.where(w.rivers[..., None],
                    _ramp(v, [(0.0, (150, 190, 214)), (0.5, (60, 118, 168)),
                              (1.0, (18, 48, 96))]), rgb).astype(np.uint8)


def _cls(w):
    from . import tectonics
    return tectonics.boundary_class(w.tect)


def layer(world, i):
    name, fn = LAYERS[i % len(LAYERS)]
    return name, fn(world)
