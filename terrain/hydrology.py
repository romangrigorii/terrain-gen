"""Stage 3: erosion and drainage."""
import numpy as np
from collections import deque
from dataclasses import dataclass
from heapq import heappush, heappop

from . import grid


def thermal(h, iters=30, talus=0.035, rate=0.35, mask=None, pin=None):
    """Creep: any slope steeper than `talus` sheds material to its downhill neighbours."""
    h = h.copy()
    factor = rate / 8.0
    if mask is not None:
        factor = mask.astype(h.dtype) * factor
    delta = np.empty_like(h)
    around = grid.Halo(h.shape, h.dtype)
    sent = grid.Halo(h.shape, h.dtype)
    give = np.empty_like(h)
    open_edge = grid.active_bounds() is not None
    for _ in range(iters):
        delta[:] = 0.0
        around.load(h)
        for (dy, dx), dist in zip(grid.NEIGH8, grid.step_lengths(h.shape)):
            np.subtract(h, around.at(dy, dx), out=give)
            np.subtract(give, talus * dist, out=give)
            np.clip(give, 0, None, out=give)
            np.multiply(give, factor, out=give)
            np.subtract(delta, give, out=delta)
            if open_edge:
                sent.load(give, outside="zero")
            else:
                sent.load(give)
            np.add(delta, sent.at(-dy, -dx), out=delta)
        if not delta.any():     # nothing anywhere exceeds the angle of repose
            break
        h += delta
        if pin is not None:
            pin(h)
    return h


def _pool(a, op):
    """2x2 block reduce, edge-padded to even dimensions."""
    h, w = a.shape
    if h % 2:                       # duplicate, never drop: dropping a row
        a = np.concatenate([a, a[-1:]], axis=0)      # would lose a peak and
    if w % 2:                       # break the upper-bound guarantee
        a = np.concatenate([a, a[:, -1:]], axis=1)
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
    return op(op(b, axis=3), axis=1)


def _warm_start(h, outlet, min_size=64):
    """An upper bound on the filled surface, solved cheaply on a coarse grid."""
    if min(h.shape) <= min_size:
        return None
    hc, oc = _pool(h, np.max), _pool(outlet, np.all)
    if not oc.any():
        return None
    coarse = _solve(hc, oc, 0.0, _warm_start(hc, oc, min_size))
    return coarse.repeat(2, 0).repeat(2, 1)[:h.shape[0], :h.shape[1]]


def _finish_sparse(h, filled, eps, iters_left, refresh=32):
    """Run the same iteration, but only over the cells still above the terrain."""
    rows, cols = h.shape
    hf, ff = h.ravel(), filled.ravel()
    while iters_left > 0:
        live = np.flatnonzero(ff > hf)
        if live.size == 0:
            break
        ly, lx = np.divmod(live, cols)
        bounds = grid.active_bounds()
        if bounds is not None and bounds.clamp_x:
            nb = np.stack([np.clip(ly + dy, 0, rows - 1) * cols
                           + np.clip(lx + dx, 0, cols - 1)
                           for dy, dx in grid.NEIGH8])
        else:
            nb = np.stack([np.clip(ly + dy, 0, rows - 1) * cols + (lx + dx) % cols
                           for dy, dx in grid.NEIGH8])
        floor = hf[live]
        for _ in range(min(refresh, iters_left)):
            iters_left -= 1
            cur = ff[live]
            best = np.maximum(ff[nb].min(axis=0) + eps, floor)
            np.minimum(best, cur, out=best)
            if not (best < cur).any():
                return filled
            ff[live] = best
    return filled


def _solve(h, outlet, eps, init, max_iters=4000, dense_iters=16):
    """Iterate w = max(h, min(neighbours) + eps) until nothing moves."""
    filled = np.where(outlet, h, h.max() + 1.0 if init is None else init)
    work = np.empty_like(filled)
    best = np.empty_like(filled)
    lower = np.empty(filled.shape, dtype=bool)
    for i in range(max_iters):
        grid.min3x3(filled, work, best)
        np.add(best, eps, out=best)
        np.minimum(best, filled, out=best)
        np.maximum(best, h, out=best)
        np.copyto(best, filled, where=outlet)
        np.less(best, filled, out=lower)
        done = not lower.any()
        filled, best = best, filled
        if done:
            return filled
        if i + 1 >= dense_iters:
            return _finish_sparse(h, filled, eps, max_iters - i - 1)
    return filled


def priority_flood(h, sea_level=0.0, eps=1e-5):
    """Exact 8-connected drainage conditioning with x-periodic boundaries."""
    h = np.asarray(h, dtype=np.float32)
    if h.ndim != 2 or not np.isfinite(h).all():
        raise ValueError("DEM must be a finite 2D array")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    H, W = h.shape
    out = h.copy()
    original = h.ravel()
    flat = out.ravel()
    visited = np.zeros(h.size, bool)
    seeds = (h <= sea_level).copy()
    seeds[0] = seeds[-1] = True
    bounds = grid.active_bounds()
    wrap_x = not (bounds is not None and bounds.clamp_x)
    if bounds is not None:
        if bounds.left is not None:
            seeds[:, 0] |= np.asarray(bounds.left) <= h[:, 0]
        if bounds.right is not None:
            seeds[:, -1] |= np.asarray(bounds.right) <= h[:, -1]
    heap = []
    for idx in np.flatnonzero(seeds.ravel()):
        idx = int(idx)
        visited[idx] = True
        heappush(heap, (float(flat[idx]), idx))
    pits = deque()
    while heap or pits:
        if pits:
            level, cell = pits.popleft()
        else:
            level, cell = heappop(heap)
        y, x = divmod(cell, W)
        for dy, dx in grid.NEIGH8:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H:
                continue
            if wrap_x:
                nx %= W
            elif nx < 0 or nx >= W:
                continue
            k = ny * W + nx
            if visited[k]:
                continue
            visited[k] = True
            original_elevation = float(original[k])
            if original_elevation <= level:
                raised = max(original_elevation, level + eps)
                flat[k] = raised
                pits.append((float(flat[k]), k))
            else:
                heappush(heap, (original_elevation, k))
    return out


def fill_depressions(h, sea_level=0.0, eps=1e-5, max_iters=400, method="iterative"):
    """Planchon-Darboux filling. Ocean cells and the map's y edges are outlets."""
    if method == "priority_flood":
        return priority_flood(h, sea_level, eps)
    if method != "iterative":
        raise ValueError(f"unknown depression solver: {method}")
    outlet = h <= sea_level
    outlet[0, :] = True
    outlet[-1, :] = True
    bounds = grid.active_bounds()
    if bounds is not None:
        if bounds.left is not None:
            outlet[:, 0] |= np.asarray(bounds.left) <= h[:, 0]
        if bounds.right is not None:
            outlet[:, -1] |= np.asarray(bounds.right) <= h[:, -1]
    h = h.astype(np.float32, copy=False)
    init = _warm_start(h, outlet)
    if init is not None:
        init = np.maximum(h, init + eps * sum(h.shape))
    return _solve(h, outlet, eps, init, max_iters)


def flow_routing(filled):
    """D8 steepest descent. Returns (receiver index, distance to receiver)."""
    h, w = filled.shape
    rec = np.arange(h * w).reshape(h, w).copy()
    rec_d = np.ones_like(filled)
    best = np.zeros_like(filled)
    around = grid.Halo(filled.shape, filled.dtype).load(filled)
    slope = np.empty_like(filled)
    bounds = grid.active_bounds()
    here = np.arange(h * w).reshape(h, w) if bounds is not None and bounds.clamp_x else None
    lengths = grid.step_lengths(filled.shape)
    for (dy, dx), dist in zip(grid.NEIGH8, lengths):
        np.subtract(filled, around.at(dy, dx), out=slope)
        np.divide(slope, dist, out=slope)
        take = slope > best
        best[take] = slope[take]
        rec[take] = grid.neighbour_index(filled.shape, dy, dx)[take]
        rec_d[take] = np.broadcast_to(dist, filled.shape)[take]
        if here is not None and dx != 0:
            hit = take[:, :1] if dx < 0 else take[:, -1:]
            if dx < 0:
                rec[:, 0] = np.where(hit[:, 0], here[:, 0], rec[:, 0])
            else:
                rec[:, -1] = np.where(hit[:, 0], here[:, -1], rec[:, -1])
    return rec, rec_d, best


def accumulate(filled, rec, weights=None, gate=None):
    """Drainage area: every cell's water, plus everything draining into it."""
    n = filled.size
    acc = np.ones(n) if weights is None else weights.ravel().astype(float).copy()
    r = rec.ravel()
    flows = r != np.arange(n)              # false at sinks: sea, edges, pits
    indeg = np.bincount(r[flows], minlength=n)
    front = np.flatnonzero(indeg == 0)     # ridge cells, nothing upstream
    gate_mask, gate_fn = gate if gate else (None, None)
    stamp = np.empty(n, np.int64)
    while front.size:
        if gate_mask is not None:
            g = front[gate_mask[front]]
            if g.size:
                acc[g] = gate_fn(g, acc[g])
        front = front[flows[front]]
        if front.size == 0:
            break
        tgt = r[front]
        np.add.at(acc, tgt, acc[front])
        np.subtract.at(indeg, tgt, 1)
        cand = tgt[indeg[tgt] == 0]
        pos = np.arange(cand.size)
        stamp[cand] = pos
        front = cand[stamp[cand] == pos]
    return acc.reshape(filled.shape)


def stream_power(h, sea_level=0.0, passes=4, k=0.06, m=0.5, n=1.0,
                 thermal_iters=12, talus=0.045, weights=None, pin=None,
                 depression_method="iterative", *, layered=False):
    """Alternate hillslope creep and fluvial incision. Returns (h, filled, acc)."""
    if layered:
        return _stream_power_layered(
            h, sea_level, passes, k, m, n, weights, pin, depression_method)
    h = h.astype(np.float32, copy=False)
    if pin is not None:
        pin(h)
    for p in range(passes):
        land = h > sea_level - 0.02
        h = thermal(h, iters=thermal_iters, talus=talus, mask=land, pin=pin)
        filled = fill_depressions(h, sea_level, method=depression_method)
        rec, rec_d, _ = flow_routing(filled)
        acc = accumulate(filled, rec, weights)

        flat = h.ravel()
        slope = np.clip((flat - flat[rec.ravel()]) / rec_d.ravel(), 0, None)
        incision = k * (acc.ravel() ** m) * (slope ** n)
        drop = np.minimum(incision, (flat - flat[rec.ravel()]) * 0.5)
        drop = np.where(land.ravel(), np.clip(drop, 0, None), 0.0)
        h = np.subtract(flat, drop, dtype=np.float32).reshape(h.shape)
        h = np.where(land, grid.blur(h, 0.4), h)
        if pin is not None:
            pin(h)
    filled = fill_depressions(h, sea_level, method=depression_method)
    rec, rec_d, _ = flow_routing(filled)
    acc = accumulate(filled, rec, weights)
    return h, filled, acc, rec


def sediment_transport(bedrock, sediment, hardness, discharge=None, iters=8,
                       k=0.04, m=0.5, n=1.0, bedrock_rate=0.35, wrap_x=True):
    """Move loose sediment downhill. Deposit where the stream cannot carry it."""
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    bedrock = np.array(bedrock, np.float32, copy=True)
    sediment = np.array(sediment, np.float32, copy=True)
    hardness = np.clip(np.asarray(hardness, np.float32), 0.0, 1.0)
    if bedrock.shape != sediment.shape or bedrock.shape != hardness.shape:
        raise ValueError("bedrock, sediment and hardness must share a shape")
    if discharge is None:
        discharge = np.ones(bedrock.shape, np.float32)
    else:
        discharge = np.maximum(np.asarray(discharge, np.float32), 0.0)
        if discharge.shape != bedrock.shape:
            raise ValueError("discharge must match the height field")
    flow = np.log1p(discharge)
    flow = flow / (1.0 + flow)

    def run():
        nonlocal bedrock, sediment
        height, width = bedrock.shape
        index = np.arange(height * width, dtype=np.int32).reshape(height, width)
        for _ in range(iters):
            surface = bedrock + sediment
            best = np.zeros((height, width), np.float32)
            receiver = index.copy()
            for (dy, dx), dist in zip(grid.NEIGH8, grid.step_lengths(bedrock.shape)):
                neighbour = grid.shift(surface, dy, dx)
                neighbour_i = grid.shift(index, dy, dx)
                drop = (surface - neighbour) / np.float32(dist)
                better = drop > best
                best = np.where(better, drop, best)
                receiver = np.where(better, neighbour_i, receiver)
            slope = np.clip(best, 0.0, None)
            moving_room = np.float32(k) * (flow ** np.float32(m)) * (slope ** np.float32(n))
            take = np.minimum(sediment, moving_room)
            leftover = moving_room - take
            cut = np.minimum(bedrock, leftover * np.float32(bedrock_rate) * (1.0 - hardness))
            flowing = (slope > 0.0).astype(np.float32)
            take *= flowing
            cut *= flowing
            sediment -= take
            bedrock -= cut
            arrived = np.zeros(height * width, np.float32)
            np.add.at(arrived, receiver.ravel(), (take + cut).ravel())
            sediment += arrived.reshape(height, width)
        return bedrock, sediment, bedrock + sediment

    if wrap_x:
        return run()
    bounds = grid.Bounds()
    bounds.clamp_x = True
    bounds.clamp_y = True
    with grid.use_bounds(bounds):
        return run()


@dataclass
class ErosionState:
    """Bedrock, deposited sediment, and the load still in the water."""
    bedrock: np.ndarray
    sediment: np.ndarray
    suspended: np.ndarray
    erodability: np.ndarray
    flow: np.ndarray
    flowdir: np.ndarray

    @property
    def height(self):
        return self.bedrock + self.sediment


@dataclass(frozen=True)
class ErosionPass:
    """One physical scale of a multiscale stack."""
    feature_size_m: float
    iterations: int
    hydraulic_strength: float
    thermal_strength: float
    deposition_rate: float

EROSION_PRESET = (
    ErosionPass(96.0, 12, 0.60, 0.25, 0.45),
    ErosionPass(24.0, 20, 0.85, 0.15, 0.30),
    ErosionPass(8.0, 16, 0.35, 0.40, 0.20),
)


def _drainage_fronts(rec):
    """Upstream-to-downstream levels. Cells left in a cycle are omitted."""
    n = rec.size
    receiver = np.asarray(rec, np.int64).ravel()
    index = np.arange(n)
    flows = receiver != index
    indeg = np.bincount(receiver[flows], minlength=n).astype(np.int32)
    front = np.flatnonzero(indeg == 0)
    stamp = np.empty(n, np.int64)
    fronts = []
    seen = np.zeros(n, dtype=bool)
    while front.size:
        fronts.append(front)
        seen[front] = True
        movers = front[flows[front]]
        if movers.size == 0:
            break
        target = receiver[movers]
        np.subtract.at(indeg, target, 1)
        cand = target[indeg[target] == 0]
        pos = np.arange(cand.size)
        stamp[cand] = pos
        front = cand[stamp[cand] == pos]
    return fronts, seen


def _exit_mask(shape, wrap_x):
    """Cells with no interior outlet. Load that reaches them leaves the map."""
    height, width = shape
    exit_ = np.zeros((height, width), dtype=bool)
    exit_[0, :] = True
    exit_[-1, :] = True
    if not wrap_x:
        exit_[:, 0] = True
        exit_[:, -1] = True
    return exit_.ravel()


def _flow_capacity(discharge, shape):
    """Pointwise 0..1 discharge. A missing field carries at full capacity."""
    if discharge is None:
        return np.ones(shape, np.float32)
    flow = np.maximum(np.asarray(discharge, np.float32), 0.0)
    if flow.shape != shape:
        raise ValueError("discharge must match the height field")
    flow = np.log1p(flow)
    return (flow / (1.0 + flow)).astype(np.float32)


def hydraulic_erode(bedrock, sediment, erodability, discharge=None, iters=4,
                    kc=0.05, ke=0.8, kd=0.65, m=0.5, n=1.0, wrap_x=True):
    """Capacity-limited erosion with a suspended load."""
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    bedrock = np.array(bedrock, np.float32, copy=True)
    sediment = np.array(sediment, np.float32, copy=True)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")
    erodability = np.clip(np.asarray(erodability, np.float32), 0.0, 1.0)
    if erodability.shape != bedrock.shape:
        raise ValueError("erodability must match the height field")
    flow = _flow_capacity(discharge, bedrock.shape)
    exported = 0.0
    receivers = np.arange(bedrock.size, dtype=np.int32).reshape(bedrock.shape)

    def sweep():
        nonlocal bedrock, sediment, exported, receivers
        bed = bedrock.ravel()
        loose = sediment.ravel()
        soft = erodability.ravel()
        water = flow.ravel()
        leaving = _exit_mask(bedrock.shape, wrap_x)
        for _ in range(iters):
            surface = (bed + loose).reshape(bedrock.shape)
            receivers, rec_d, _ = flow_routing(surface)
            receiver = np.asarray(receivers, np.int64).ravel()
            dist = np.maximum(np.asarray(rec_d, np.float32).ravel(), np.float32(1e-6))
            fronts, seen = _drainage_fronts(receiver)
            incoming = np.zeros(bed.size, np.float32)
            index = np.arange(bed.size)
            for front in fronts:
                carried = incoming[front]
                here = bed[front] + loose[front]
                down = bed[receiver[front]] + loose[receiver[front]]
                sink = receiver[front] == index[front]
                slope = np.clip((here - down) / dist[front], 0.0, None)
                slope = np.where(sink, np.float32(0.0), slope)
                capacity = (np.float32(kc)
                            * np.power(np.maximum(water[front], 0.0), np.float32(m))
                            * np.power(slope, np.float32(n)))
                room = np.maximum(here - down, 0.0) * np.float32(0.5)
                room = np.where(sink, np.float32(0.0), room)
                spare = capacity - carried
                budget = np.minimum(np.float32(ke) * np.maximum(spare, 0.0), room)
                from_loose = np.minimum(loose[front], budget)
                loose[front] = loose[front] - from_loose
                rock_budget = (budget - from_loose) * soft[front]
                from_rock = np.minimum(np.maximum(room - from_loose, 0.0), rock_budget)
                bed[front] = bed[front] - from_rock
                carried = carried + from_loose + from_rock
                excess = np.maximum(carried - capacity, 0.0)
                settled = np.minimum(carried, np.float32(kd) * excess)
                exit_sink = sink & leaving[front]
                settled = np.where(sink, np.where(exit_sink, np.float32(0.0), carried), settled)
                loose[front] = loose[front] + settled
                departed = np.where(exit_sink, carried, np.float32(0.0))
                exported += float(departed.sum(dtype=np.float64))
                moving = carried - settled - departed
                np.add.at(incoming, receiver[front], moving)
            missed = np.flatnonzero(~seen)
            if missed.size:
                loose[missed] = loose[missed] + incoming[missed]
        bedrock = bed.reshape(bedrock.shape)
        sediment = loose.reshape(sediment.shape)

    if wrap_x and grid.active_bounds() is None:
        sweep()
    else:
        bounds = grid.Bounds()
        bounds.clamp_x = True
        bounds.clamp_y = True
        with grid.use_bounds(bounds):
            sweep()
    state = ErosionState(
        bedrock=bedrock,
        sediment=sediment,
        suspended=np.zeros_like(sediment),
        erodability=erodability,
        flow=flow,
        flowdir=np.asarray(receivers, np.int32))
    return state, float(exported)


def weather_bedrock(bedrock, sediment, erodability, rate=0.002):
    """Turn a thin skin of exposed bedrock into loose debris."""
    bedrock = np.array(bedrock, np.float32, copy=True)
    sediment = np.array(sediment, np.float32, copy=True)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")
    soft = np.clip(np.asarray(erodability, np.float32), 0.0, 1.0)
    if soft.shape != bedrock.shape:
        raise ValueError("erodability must match the height field")
    surface = bedrock + sediment
    dy, dx = grid.gradient(surface)
    dx = dx / grid.metric_cos(surface.shape)
    exposure = np.clip(np.hypot(dy, dx) / np.float32(0.25), 0.0, 1.0)
    produced = (np.float32(rate) * soft
                * (np.float32(0.2) + np.float32(0.8) * exposure))
    return bedrock - produced, sediment + produced


def slump_debris(bedrock, sediment, repose=0.70, rate=0.35, iters=4, wrap_x=True):
    """Slide loose debris until the surface is at the angle of repose."""
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    bedrock = np.asarray(bedrock, np.float32)
    sediment = np.array(sediment, np.float32, copy=True)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")

    def run():
        nonlocal sediment
        factor = np.float32(rate / 8.0)
        limit = np.float32(repose)
        delta = np.empty_like(sediment)
        surface = np.empty_like(sediment)
        give = np.empty_like(sediment)
        around = grid.Halo(sediment.shape, np.float32)
        sent = grid.Halo(sediment.shape, np.float32)
        open_edge = grid.active_bounds() is not None
        for _ in range(iters):
            np.add(bedrock, sediment, out=surface)
            delta[:] = 0.0
            around.load(surface)
            cap = sediment * np.float32(0.125)
            moved = False
            for (dy, dx), dist in zip(grid.NEIGH8, grid.step_lengths(sediment.shape)):
                np.subtract(surface, around.at(dy, dx), out=give)
                np.divide(give, np.float32(dist), out=give)
                np.subtract(give, limit, out=give)
                np.clip(give, 0.0, None, out=give)
                if not give.any():
                    continue
                moved = True
                np.multiply(give, factor, out=give)
                np.minimum(give, cap, out=give)
                np.subtract(delta, give, out=delta)
                if open_edge:
                    sent.load(give, outside="zero")
                else:
                    sent.load(give)
                np.add(delta, sent.at(-dy, -dx), out=delta)
            if not moved:
                break
            sediment += delta
        return sediment

    if wrap_x and grid.active_bounds() is None:
        return run()
    bounds = grid.Bounds()
    bounds.clamp_x = True
    bounds.clamp_y = True
    with grid.use_bounds(bounds):
        return run()


def _resample(field, shape):
    """Bilinear sample of `field` onto `shape`. Edges repeat the boundary."""
    from scipy.ndimage import map_coordinates

    field = np.asarray(field, np.float64)
    if field.shape == tuple(shape):
        return field.astype(np.float32, copy=True)
    rows = np.linspace(0.0, field.shape[0] - 1.0, int(shape[0]))
    cols = np.linspace(0.0, field.shape[1] - 1.0, int(shape[1]))
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    out = map_coordinates(field, [yy.ravel(), xx.ravel()], order=1, mode="nearest")
    return out.reshape(int(shape[0]), int(shape[1])).astype(np.float32)


def _box_mean(field, radius):
    """Mean of a square neighbourhood. `radius` 0 returns the field."""
    if radius <= 0:
        return np.asarray(field, np.float32)
    from scipy.ndimage import uniform_filter

    size = int(radius) * 2 + 1
    out = uniform_filter(np.asarray(field, np.float32), size=size, mode="nearest")
    return out.astype(np.float32, copy=False)


def _erode_local(bedrock, sediment, erodability, discharge, cell_m, max_radius,
                 wrap_x):
    """Hop-limited cascade. The whole dependency fits inside `max_radius`."""
    budget = int(max_radius)
    if budget < 0:
        raise ValueError("max_radius must be nonnegative")
    if budget == 0:
        return bedrock, sediment, bedrock + sediment
    bedrock, sediment = weather_bedrock(
        bedrock, sediment, erodability, rate=0.0003)
    slump_n = min(2, budget)
    remain = budget - slump_n
    collect = 0
    if cell_m > 0.0 and remain > 0:
        collect = int(round(24.0 / float(cell_m)))
        collect = max(0, min(collect, remain // 2))
    hops = remain - collect
    flow = discharge
    if collect:
        flow = _box_mean(discharge if discharge is not None
                         else np.ones_like(sediment), collect)
    hardness = np.clip(1.0 - np.asarray(erodability, np.float32), 0.0, 1.0)
    if hops:
        bedrock, sediment, surface = sediment_transport(
            bedrock, sediment, hardness, flow, iters=hops, k=0.05,
            wrap_x=wrap_x)
    else:
        surface = bedrock + sediment
    if slump_n:
        sediment = slump_debris(
            bedrock, sediment, repose=0.70, rate=0.4, iters=slump_n,
            wrap_x=wrap_x)
        surface = bedrock + sediment
    return bedrock, sediment, surface


def _erode_pyramid(bedrock, sediment, erodability, discharge, cell_m, passes,
                   wrap_x):
    """Solve each feature size on a grid of that size, then bring the change back."""
    for step in passes:
        if step.iterations < 0 or step.feature_size_m <= 0.0:
            raise ValueError("an erosion pass needs a positive feature size")
        factor = max(1, int(round(float(step.feature_size_m) / float(cell_m))))
        full = bedrock.shape
        coarse = full
        if factor > 1:
            coarse = (max(4, int(round(full[0] / factor))),
                      max(4, int(round(full[1] / factor))))
        if coarse == full:
            state, _exported = hydraulic_erode(
                bedrock, sediment, erodability, discharge,
                iters=int(step.iterations),
                kc=0.05 * float(step.hydraulic_strength),
                kd=float(step.deposition_rate), wrap_x=wrap_x)
            bedrock = state.bedrock
            sediment = state.sediment
        else:
            area = np.float32((full[0] / coarse[0]) * (full[1] / coarse[1]))
            bed_c = _resample(bedrock, coarse)
            sed_c = _resample(sediment, coarse)
            soft_c = _resample(erodability, coarse)
            if discharge is None:
                flow_c = None
            else:
                flow_c = _resample(discharge, coarse) * area
            state, _exported = hydraulic_erode(
                bed_c, sed_c, soft_c, flow_c, iters=int(step.iterations),
                kc=0.05 * float(step.hydraulic_strength),
                kd=float(step.deposition_rate), wrap_x=wrap_x)
            bedrock = bedrock + _resample(state.bedrock - bed_c, full)
            sediment = sediment + _resample(state.sediment - sed_c, full)
            sediment = np.maximum(sediment, 0.0)
        if step.thermal_strength > 0.0:
            sediment = slump_debris(
                bedrock, sediment, repose=0.70,
                rate=0.45 * float(step.thermal_strength),
                iters=max(1, int(step.iterations) // 4), wrap_x=wrap_x)
    return bedrock, sediment, bedrock + sediment


def erode_multiscale(bedrock, sediment, erodability, discharge=None, cell_m=1.0,
                     passes=None, max_radius=None, wrap_x=True):
    """Erode the same relief at more than one physical scale."""
    if cell_m <= 0.0:
        raise ValueError("cell_m must be positive")
    if passes is None:
        passes = EROSION_PRESET
    if max_radius is None:
        return _erode_pyramid(
            bedrock, sediment, erodability, discharge, cell_m, passes, wrap_x)
    return _erode_local(
        bedrock, sediment, erodability, discharge, cell_m, max_radius, wrap_x)


def _stream_power_layered(h, sea_level, passes, k, m, n, weights, pin,
                          depression_method):
    """Incision replaced by transport. Same four return values as `stream_power`."""
    h = np.array(h, np.float32, copy=True)
    if pin is not None:
        pin(h)
    erodability = np.where(h > sea_level, np.float32(0.7), np.float32(0.0))
    erodability = np.asarray(erodability, np.float32)
    mantle = np.full(h.shape, np.float32(0.008), dtype=np.float32)
    bedrock, sediment = weather_bedrock(h - mantle, mantle, erodability, rate=0.001)
    wrap = grid.active_bounds() is None
    state, _exported = hydraulic_erode(
        bedrock, sediment, erodability, weights, iters=max(int(passes), 1),
        kc=float(k), ke=0.8, kd=0.65, m=m, n=n, wrap_x=wrap)
    sediment = slump_debris(
        state.bedrock, state.sediment, repose=0.75, rate=0.3, iters=2,
        wrap_x=wrap)
    h = state.bedrock + sediment
    if pin is not None:
        pin(h)
    filled = fill_depressions(h, sea_level, method=depression_method)
    rec, _, _ = flow_routing(filled)
    acc = accumulate(filled, rec, weights)
    return h, filled, acc, rec


def erosion_mask(height, rainfall, hardness, cell_m, metres_per_unit=12000.0):
    """Where water is allowed to work. Steep, soft, wet ground scores higher."""
    cell = float(cell_m)
    if not np.isfinite(cell) or cell <= 0.0:
        raise ValueError("cell_m must be finite and positive")
    height = np.asarray(height, np.float64)
    dy, dx = np.gradient(height, cell)
    slope = np.hypot(dx, dy) * float(metres_per_unit)
    steepness = np.clip(slope / 0.7, 0.0, 1.0)
    if hardness is None:
        softness = np.ones(height.shape, np.float64)
    else:
        softness = np.clip(1.0 - np.asarray(hardness, np.float64), 0.0, 1.0)
    rain = np.clip(np.asarray(rainfall, np.float64), 0.0, 1.0)
    mask = ((0.15 + 0.85 * rain)
            * (0.3 + 0.7 * steepness)
            * (0.25 + 0.75 * softness))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def accumulate_mfd(height, weights=None, p=1.5):
    """Spread each cell's water across every lower neighbour."""
    height = np.asarray(height, np.float32)
    rows, cols = height.shape
    count = rows * cols
    if weights is None:
        acc = np.ones(count, np.float64)
    else:
        acc = np.asarray(weights, np.float64).ravel().copy()
        if acc.size != count:
            raise ValueError("weights must match the height field")
    around = grid.Halo(height.shape, np.float32).load(height)
    slopes = []
    dests = []
    for (dy, dx), dist in zip(grid.NEIGH8, grid.step_lengths(height.shape)):
        drop = (height - around.at(dy, dx)) / np.float32(dist)
        slopes.append(np.power(np.clip(drop, 0.0, None), np.float32(p)))
        dests.append(grid.neighbour_index(height.shape, dy, dx))
    stack = np.stack(slopes, axis=0)
    denom = stack.sum(axis=0)
    live = denom > 0
    indeg = np.zeros(count, np.int32)
    fractions = []
    for i in range(8):
        share = np.zeros_like(height)
        share[live] = stack[i][live] / denom[live]
        sending = share > 1e-8
        np.add.at(indeg, dests[i][sending], 1)
        fractions.append(share.ravel())
    front = np.flatnonzero(indeg == 0)
    seen = np.zeros(count, dtype=bool)
    stamp = np.empty(count, np.int64)
    steps = 0
    while front.size and steps <= count:
        steps += 1
        seen[front] = True
        ready = []
        for share, dest in zip(fractions, dests):
            moving = front[share[front] > 0.0]
            if moving.size == 0:
                continue
            target = dest.ravel()[moving]
            np.add.at(acc, target, acc[moving] * share[moving])
            np.subtract.at(indeg, target, 1)
            hit = target[indeg[target] == 0]
            if hit.size:
                ready.append(hit)
        if not ready:
            break
        cand = np.concatenate(ready)
        pos = np.arange(cand.size)
        stamp[cand] = pos
        front = cand[stamp[cand] == pos]
        front = front[~seen[front]]
    return acc.reshape(height.shape).astype(np.float32)


def flow_guided_warp(bedrock, sediment, receivers, intensity, displacement_cells=0.4,
                     substeps=4, wrap_x=True):
    """Nudge the ground a short distance along the drainage direction."""
    from scipy.ndimage import map_coordinates

    bedrock = np.asarray(bedrock, np.float32)
    sediment = np.asarray(sediment, np.float32)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")
    rows, cols = bedrock.shape
    yy, xx = np.indices((rows, cols))
    rec = np.asarray(receivers, np.int64)
    ry, rx = np.divmod(rec, cols)
    dx = (rx - xx + cols // 2) % cols - cols // 2
    dy = ry - yy
    norm = np.maximum(np.hypot(dy, dx), 1e-6)
    vy = dy / norm
    vx = dx / norm
    mag = np.log1p(np.maximum(np.asarray(intensity, np.float64), 0.0))
    mag = mag / (1.0 + mag)
    step = float(displacement_cells) / float(max(int(substeps), 1))
    sy = yy.astype(np.float64)
    sx = xx.astype(np.float64)
    for _ in range(max(int(substeps), 1)):
        sy = sy - vy * mag * step
        sx = sx - vx * mag * step
    if wrap_x:
        sx = np.mod(sx, cols)
    else:
        sx = np.clip(sx, 0.0, cols - 1.0)
    sy = np.clip(sy, 0.0, rows - 1.0)

    def sample(field):
        out = map_coordinates(np.asarray(field, np.float64),
                              [sy.ravel(), sx.ravel()], order=1, mode="nearest")
        return out.reshape(rows, cols).astype(np.float32)

    return sample(bedrock), sample(sediment)


def deposit_sediment(height, discharge, seed, iters=8, wrap_x=True,
                     rainfall=None, cell_m=None):
    """Split a solved surface into bedrock and a soft mantle, then transport it."""
    from . import geology, noise

    height = np.asarray(height, np.float32)
    rng = noise.stage_rng(seed, "strata")
    rough = noise.fbm(height.shape[0], height.shape[1], rng, periods=3, octaves=4)
    noise_hard = np.clip(0.25 + 0.55 * (rough * 0.5 + 0.5), 0.05, 0.95)
    rows = np.linspace(0.0, 0.02, height.shape[0], dtype=np.float64)[:, None]
    cols = np.linspace(0.0, 0.01, height.shape[1], dtype=np.float64)[None, :]
    reference = np.max(height) + rows + cols
    strata = geology.strata_erodability(height, reference)
    erodability = np.clip(strata * (1.15 - noise_hard), 0.05, 1.0).astype(np.float32)
    if rainfall is not None and cell_m is not None:
        preliminary = (1.0 - erodability).astype(np.float32)
        erodability = np.clip(
            erodability * (0.35 + 0.65 * erosion_mask(
                height, rainfall, preliminary, cell_m)),
            0.05, 1.0).astype(np.float32)
    resistance = (1.0 - erodability).astype(np.float32)
    dy, dx = grid.gradient(height)
    dx = dx / grid.metric_cos(height.shape)
    grade = np.hypot(dy, dx)
    mantle = (0.002 + 0.02 * np.exp(-grade * 40.0)).astype(np.float32)
    bedrock, sediment = weather_bedrock(
        height - mantle, mantle, erodability, rate=0.0008)
    state, _exported = hydraulic_erode(
        bedrock, sediment, erodability, discharge, iters=iters,
        kc=0.05, ke=0.8, kd=0.65, wrap_x=wrap_x)
    sediment = slump_debris(
        state.bedrock, state.sediment, repose=0.75, rate=0.3,
        iters=min(3, max(int(iters), 0)), wrap_x=wrap_x)
    surface = state.bedrock + sediment
    return surface, state.bedrock, sediment, resistance
