"""Stage 4: lakes and rivers as real features rather than a flow threshold."""
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import grid, hydrology


@dataclass
class Water:
    filled: np.ndarray        # depression-filled surface
    flow: np.ndarray          # drainage area, in cells
    receivers: np.ndarray
    lake_id: np.ndarray       # 0 = not a lake
    lake_level: np.ndarray    # flat water surface elevation, 0 outside lakes
    lake_depth: np.ndarray    # level - bed
    width: np.ndarray         # river width in cells, 0 where there is no river
    polylines: list = field(default_factory=list)   # (n, 2) arrays of (y, x)
    widths: list = field(default_factory=list)      # channel width per path node

    @property
    def lake_mask(self):
        return self.lake_id > 0

    @property
    def river_mask(self):
        return self.width > 0

    def surface(self, h):
        """Elevation of whatever is on top: lake surfaces are flat, land is bed."""
        return np.where(self.lake_mask, self.lake_level, h)

# --------------------------------------------------------------------------


def _label_lakes(h, filled, cfg, sea_level=0.0):
    """Closed basins, dropped if they are shallower or smaller than the cutoffs."""
    raw = (filled - h > cfg.lake_min_depth) & (h > sea_level)
    lbl, n = grid.label_wrap_x(raw, connectivity=8)
    if n == 0:
        return np.zeros_like(lbl), 0
    keep = np.zeros(n + 1, bool)
    keep[1:] = np.bincount(lbl.ravel(), minlength=n + 1)[1:] >= cfg.lake_min_area
    remap = np.cumsum(keep) * keep
    return remap[lbl], int(remap.max())

_Routed = namedtuple("_Routed", "filled rec plain flow lake_id n level outflow")


def _lake_balance(h, filled, rec, lbl, n, cfg, evap=None):
    """Give each lake the water surface its catchment can hold, not its rim."""
    idx = np.arange(1, n + 1)
    cells = np.flatnonzero(lbl.ravel() > 0)
    ids = lbl.ravel()[cells]
    order = np.lexsort((h.ravel()[cells], ids))
    cells, ids = cells[order], ids[order]
    beds = h.ravel()[cells]
    bounds = np.searchsorted(ids, np.arange(1, n + 2))    # lake i is [i-1:i]
    spill = np.concatenate([[0.0], np.atleast_1d(
        ndimage.minimum(filled, lbl, index=idx))])

    pos = np.atleast_2d(ndimage.minimum_position(filled, lbl, index=idx))
    exits = np.zeros(n + 1, int)
    exits[1:] = pos[:, 0] * h.shape[1] + pos[:, 1]

    rec = rec.copy()
    rf = rec.ravel()
    tgt = exits[ids]
    move = cells != tgt
    rf[cells[move]] = tgt[move]

    level, outflow = np.zeros(n + 1), np.zeros(n + 1)
    per = np.zeros(n + 1)
    per[1:] = (float(cfg.lake_evap) if evap is None
               else np.atleast_1d(ndimage.mean(evap, lbl, index=idx)))

    def gate(cell, inflow):
        lake = lbl.ravel()[cell]
        beg, size = bounds[lake - 1], bounds[lake] - bounds[lake - 1]
        ev = per[lake]
        wet = np.where(ev <= 0, size,
                       np.minimum((inflow / np.maximum(ev, 1e-9)).astype(int), size))
        full = wet >= size
        level[lake] = np.where(full, spill[lake],
                               beds[beg + np.minimum(wet, size - 1)])
        outflow[lake] = np.where(full, np.maximum(inflow - ev * size, 0.0), 0.0)
        return outflow[lake]

    mask = np.zeros(h.size, bool)
    mask[exits[1:]] = True
    return rec, (mask, gate), level, outflow


def _shift_local(a, dy, dx):
    """`grid.shift` without the wrap, for working inside one basin's bounding box."""
    out = np.zeros_like(a)
    hh, ww = a.shape
    out[max(0, -dy):hh - max(0, dy), max(0, -dx):ww - max(0, dx)] = \
        a[max(0, dy):hh - max(0, -dy), max(0, dx):ww - max(0, -dx)]
    return out


def _shift_periodic_x(a, dy, dx):
    """Shift inside a local y window that covers every column of the cylinder."""
    out = np.roll(a, shift=(-dy, -dx), axis=(0, 1))
    if dy > 0:
        out[-dy:, :] = 0
    elif dy < 0:
        out[:-dy, :] = 0
    return out


def _draw_receivers(rec, basin, lake, lbl):
    """Receivers for *drawing*: inside a basin, step towards the water."""
    if not basin.any() or not lake.any():
        return rec
    rec = rec.copy()
    W = rec.shape[1]
    for sl in ndimage.find_objects(lbl):
        if sl is None:
            continue
        ys = slice(max(0, sl[0].start - 1), min(rec.shape[0], sl[0].stop + 1))
        xs = slice(max(0, sl[1].start - 1), min(W, sl[1].stop + 1))
        todo, frontier = basin[ys, xs].copy(), lake[ys, xs]
        if not (todo.any() and frontier.any()):
            continue
        gidx = (np.arange(ys.start, ys.stop)[:, None] * W +
                np.arange(xs.start, xs.stop)[None, :])
        sub = rec[ys, xs]
        sample = _shift_periodic_x if (xs.start == 0 and xs.stop == W) else _shift_local
        while frontier.any() and todo.any():
            found = np.zeros_like(todo)
            for dy, dx in grid.NEIGH8:
                near = sample(frontier, dy, dx) & todo & ~found
                if not near.any():
                    continue
                sub[near] = sample(gidx, dy, dx)[near]
                found |= near
            todo &= ~found
            frontier = found
        rec[ys, xs] = sub
    return rec


def _route_water(h, cfg, sea_level=0.0, filled=None, rec=None,
                 runoff=None, evap=None):
    """Fill, route, label lakes, and accumulate under each lake's water balance."""
    if filled is None:
        filled = hydrology.fill_depressions(
            h, sea_level, method=getattr(cfg, "depression_method", "iterative"))
    if rec is None:
        rec, _, _ = hydrology.flow_routing(filled)
    lbl, n = _label_lakes(h, filled, cfg, sea_level)
    if n == 0:
        flow = hydrology.accumulate(filled, rec, weights=runoff)
        return _Routed(filled, rec, rec, flow, lbl, 0, np.zeros(1), np.zeros(1))
    plain = rec
    rec, gate, level, outflow = _lake_balance(h, filled, rec, lbl, n, cfg, evap)
    flow = hydrology.accumulate(filled, rec, weights=runoff, gate=gate)
    return _Routed(filled, rec, plain, flow, lbl, n, level, outflow)


def carve_outlets(h, cfg, sea_level=0.0, routed=None, runoff=None, evap=None):
    """Notch each lake's pour point and cut a channel downstream from it."""
    thresh = cfg.river_threshold * h.size
    for _ in range(cfg.outlet_carve_passes):
        known = routed[:2] if routed is not None else (None, None)
        routed = None
        r = _route_water(h, cfg, sea_level, *known, runoff=runoff, evap=evap)
        lbl, n = r.lake_id, r.n
        if n == 0:
            break

        rim = np.zeros_like(lbl)
        for dy, dx in grid.NEIGH8:
            nb = grid.shift(lbl, dy, dx)
            take = (lbl == 0) & (nb > 0) & (rim == 0)
            rim[take] = nb[take]
        if not rim.any():
            break
        idx = np.arange(1, n + 1)
        pours = ndimage.minimum_position(h, rim, index=idx)

        flat, recf = h.ravel(), r.rec.ravel()
        for lake, pos in enumerate(np.atleast_2d(pours), start=1):
            c = int(pos[0]) * h.shape[1] + int(pos[1])
            if r.outflow[lake] < thresh:   # too little water to cut, or to draw
                continue
            cur = flat[c] - cfg.outlet_carve_depth
            for _ in range(cfg.outlet_carve_len):
                flat[c] = min(flat[c], cur)
                nxt = recf[c]
                if nxt == c or flat[nxt] <= sea_level:
                    break
                cur -= cfg.outlet_carve_slope
                if flat[nxt] < cur:      # a natural channel takes over from here
                    break
                c = nxt
        h = flat.reshape(h.shape)
    return h

# --------------------------------------------------------------------------


def polylines(flow, rec, channel, cross=None):
    """Trace the channel network into paths, each running head -> mouth/junction."""
    hh, w = flow.shape
    up = np.bincount(rec.ravel()[channel.ravel()], minlength=flow.size)
    heads = np.flatnonzero(channel.ravel() & (up == 0))
    heads = heads[np.argsort(-flow.ravel()[heads])]  # longest rivers claim first
    recf, chan = rec.ravel(), channel.ravel()
    crossf = np.zeros(flow.size, bool) if cross is None else cross.ravel()
    visited = np.zeros(flow.size, bool)
    out = []
    for start in heads:
        c, path = int(start), []
        while True:
            path.append(c)
            visited[c] = True
            nxt = int(recf[c])
            if nxt == c:
                break
            path_end = visited[nxt] or not (chan[nxt] or crossf[nxt])
            if path_end:
                path.append(nxt)     # touch the trunk (or the water) and stop
                break
            c = nxt
        if len(path) >= 3:
            out.append(np.array([(p // w, p % w) for p in path], dtype=float))
    return out


def _unwrap_x(pts, w):
    """Undo the x seam inside one path so smoothing does not average across it."""
    dx = np.diff(pts[:, 1])
    pts[1:, 1] += np.cumsum(-w * (dx > w / 2) + w * (dx < -w / 2))
    return pts


def _smooth(pts, passes=2):
    """Moving average over the interior; D8 only knows eight directions."""
    for _ in range(passes):
        if len(pts) < 5:
            break
        mid = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4.0
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def _fbm1(n, rng, period, octaves=4):
    """1-D fractal noise along a path's arc length."""
    t = np.arange(n, dtype=float)
    out, amp, norm, p = np.zeros(n), 1.0, 0.0, float(period)
    for _ in range(octaves):
        k = int(np.ceil(n / max(2.0, p))) + 2
        ctrl = rng.normal(0, 1, k + 1)
        x = t / max(2.0, p)
        i = np.minimum(x.astype(int), k - 1)
        f = x - i
        f = f * f * (3 - 2 * f)
        out += amp * (ctrl[i] * (1 - f) + ctrl[i + 1] * f)
        norm += amp
        amp *= 0.5
        p *= 0.5
    return out / norm


def meander(pts, width, cfg, rng):
    """Push the path sideways by noise, hardest in the middle of its run."""
    n = len(pts)
    if n < 6:
        return pts
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
    taper = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / cfg.meander_taper, 0, 1)
    amp = cfg.meander_amp * np.sqrt(np.maximum(width, 0.4)) * taper
    return pts + normal * (amp * _fbm1(n, rng, cfg.meander_period))[:, None]

# --------------------------------------------------------------------------


def _disc(width_map, y, x, chan_w):
    """Paint a channel of width `chan_w` cells centred on (y, x), keeping the max."""
    hh, w = width_map.shape
    r = max(chan_w * 0.5, 0.5)
    ri = int(np.ceil(r))
    ys = np.arange(int(round(y)) - ri, int(round(y)) + ri + 1)
    xs = np.arange(int(round(x)) - ri, int(round(x)) + ri + 1)
    ys = ys[(ys >= 0) & (ys < hh)]
    if ys.size == 0:
        return
    sel = np.hypot((ys - y)[:, None], (xs - x)[None, :]) <= r
    if not sel.any():
        sel[np.argmin(np.abs(ys - y)), np.argmin(np.abs(xs - x))] = True
    yy, xx = np.nonzero(sel)
    np.maximum.at(width_map, (ys[yy], xs[xx] % w), chan_w)


def _connect_strays(width, lake_mask, sea, draw, cfg, budget=400):
    """Water a reader can follow: every channel must end at a lake or the sea."""
    river = width > 0
    if not river.any():
        return width
    w = width.shape[1]
    lab, n = ndimage.label(np.concatenate([river | lake_mask | sea] * 3, axis=1),
                           structure=np.ones((3, 3)))
    if n == 0:
        return width
    mid = lab[:, w:2 * w]
    reaches = np.zeros(n + 1, bool)
    reaches[np.unique(lab[np.concatenate([lake_mask | sea] * 3, axis=1)])] = True
    stray = river & ~reaches[mid]
    if not stray.any():
        return width
    width = width.copy()
    wf, drawf = width.ravel(), draw.ravel()
    water = ((lake_mask | sea) | (river & reaches[mid])).ravel()
    for i in np.unique(mid[stray]):
        cells = np.flatnonzero((mid == i).ravel() & (wf > 0))
        c = int(cells[np.argmax(wf[cells])])     # widest cell: the downstream end
        chan_w = float(wf[c])
        trail = []
        for _ in range(budget):
            nxt = int(drawf[c])
            if nxt == c:
                break
            c = nxt
            if water[c]:
                for t in trail:
                    _disc(width, t // w, t % w, chan_w)
                trail = None
                break
            trail.append(c)
        if trail is not None:                     # never found water: drop it
            wf[cells] = 0.0
    return width


def rasterize(paths, widths, shape):
    """Stamp every path into a width field, sampling densely enough to stay joined."""
    width_map = np.zeros(shape)
    for pts, wv in zip(paths, widths):
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            steps = max(1, int(np.ceil(np.hypot(*(b - a)) / 0.4)))
            for s in range(steps + 1):
                t = s / steps
                y, x = a + (b - a) * t
                _disc(width_map, y, x, wv[i] + (wv[i + 1] - wv[i]) * t)
    return width_map

# --------------------------------------------------------------------------


def carve_channel_profile(h, width, hardness, sea_level, lake_id, cfg):
    """Cut a thalweg and a separate, softer bank."""
    h = np.asarray(h, np.float32)
    width = np.asarray(width, np.float32)
    channel = width > 0.05
    if hardness is None:
        softness = np.ones(h.shape, np.float32)
    else:
        softness = np.clip(1.0 - np.asarray(hardness, np.float32), 0.15, 1.0)
    span = max(1e-6, float(cfg.river_width_max))
    bed = float(cfg.river_incision) * np.clip(width / span, 0.0, 1.0) ** 0.5
    reach = (1.2 + 5.0 * softness).astype(np.float32)
    dist = ndimage.distance_transform_edt(~channel).astype(np.float32)
    bank = (float(cfg.river_incision) * 0.45
            * np.clip(1.0 - dist / np.maximum(reach, 1e-3), 0.0, 1.0)
            * softness)
    cut = np.maximum(bed, np.where(channel, 0.0, bank)).astype(np.float32)
    cut = grid.blur(cut, 0.55)
    dry = (h > sea_level) & (np.asarray(lake_id) == 0)
    return (h - cut * dry).astype(np.float32)


def build(h, cfg, rng, sea_level=0.0, routed=None, runoff=None, evap=None,
          bed_hardness=None):
    """Run the whole stage. Returns (incised height, Water)."""
    h = carve_outlets(h, cfg, sea_level, routed, runoff, evap)

    r = _route_water(h, cfg, sea_level, runoff=runoff, evap=evap)
    filled, rec, flow = r.filled, r.rec, r.flow
    level = r.level[r.lake_id]
    depth = np.where(r.lake_id > 0, np.maximum(level - h, 0.0), 0.0)
    lake_id = r.lake_id * (depth > 0)
    if lake_id.any():
        piece, npieces = grid.label_wrap_x(lake_id > 0, connectivity=8)
        if npieces > 1:
            keep_piece = np.zeros(npieces + 1, bool)
            deep = ndimage.maximum_position(depth, r.lake_id,
                                            index=np.arange(1, r.n + 1))
            for pos in np.atleast_2d(deep):
                keep_piece[piece[int(pos[0]), int(pos[1])]] = True
            keep_piece[0] = False
            lake_id = lake_id * keep_piece[piece]
    depth = np.where(lake_id > 0, depth, 0.0)
    keep = np.zeros(r.n + 1, bool)
    keep[np.unique(lake_id)] = True
    keep[0] = False
    lake_id = (np.cumsum(keep) * keep)[lake_id]
    level = np.where(lake_id > 0, level, 0.0)

    thresh = cfg.river_threshold * h.size
    lake_mask = lake_id > 0
    sea = h <= sea_level
    hollow = (filled - h > 1e-9) & ~lake_mask & ~sea
    draw = _draw_receivers(r.plain, (r.lake_id > 0) & ~lake_mask, lake_mask, r.lake_id)
    channel = (flow >= thresh) & ~sea & ~lake_mask
    channel[:2] = channel[-2:] = False

    paths, widths = [], []
    for pts in polylines(flow, draw, channel, cross=hollow):
        q = flow[pts[:, 0].astype(int), pts[:, 1].astype(int)]
        q = np.maximum.accumulate(q)
        wv = np.clip(cfg.river_width *
                     (q / (cfg.river_width_ref * h.size)) ** cfg.river_width_exp,
                     0.7, cfg.river_width_max)
        pts = _unwrap_x(pts, h.shape[1])
        paths.append(meander(_smooth(pts), wv, cfg, rng))
        widths.append(wv)

    width = rasterize(paths, widths, h.shape)

    if getattr(cfg, "terrain_model", "classic") in ("layered", "full"):
        h = carve_channel_profile(h, width, bed_hardness, sea_level, lake_id, cfg)
    else:
        cut = cfg.river_incision * np.clip(width / max(1e-6, cfg.river_width_max), 0, 1) ** 0.5
        cut = grid.blur(cut, 0.7) * (h > sea_level) * (lake_id == 0)
        h = h - cut

    width = width * ((h > sea_level) & (lake_id == 0))
    width = _connect_strays(width, lake_mask, h <= sea_level, draw, cfg)

    return h, Water(filled=filled, flow=flow, receivers=rec, lake_id=lake_id,
                    lake_level=level, lake_depth=depth, width=width, polylines=paths,
                    widths=widths)
