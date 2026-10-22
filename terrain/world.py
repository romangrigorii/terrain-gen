"""Stage order and the finished world. Settings live in `config`."""
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np
from . import climate, elevation, grid, hydrology, rivers, tectonics, trees, noise
from .config import Config, _scale_to_size


@dataclass
class World:
    cfg: Config
    tect: tectonics.Tectonics
    height_raw: np.ndarray      # tectonics only, before texture and erosion
    height_pre: np.ndarray      # after texture, before erosion
    height_eroded: np.ndarray   # after erosion, before rivers were cut in
    height: np.ndarray          # final
    water: rivers.Water
    runoff: np.ndarray          # water a cell contributes, 1 on average on land
    temp: np.ndarray            # mean annual temperature, C
    swing: np.ndarray           # seasonal half-range, C; warmest month is temp+swing
    biome: np.ndarray           # climate.BIOME_NAMES index per cell
    trees: trees.Trees          # canopy cover, and the individual trees on it
    timings: dict = field(default_factory=dict)
    bedrock: np.ndarray = None  # solid rock, present when sediment transport ran
    sediment: np.ndarray = None
    hardness: np.ndarray = None
    placed_ice: np.ndarray = None    # metres of ice stamped by a brush
    placed_sand: np.ndarray = None   # metres of dune sand
    placed_void: np.ndarray = None   # metres of cave void under the roof

    @property
    def land(self):
        return self.height > 0.0

    @property
    def flow(self):
        return self.water.flow

    @property
    def rivers(self):
        return self.water.river_mask

    @property
    def lakes(self):
        return self.water.lake_mask

    @property
    def surface(self):
        """Height of the top of the water where there is water, of the bed elsewhere."""
        return self.water.surface(self.height)

    def info_at(self, y, x):
        p = int(self.tect.plate[y, x])
        w = self.water
        out = {
            "plate": p,
            "crust": "continental" if self.tect.plate_cont[p] else "oceanic",
            "elev": float(self.height[y, x]),
            "flow": float(w.flow[y, x]),
            "rain": float(self.runoff[y, x]),
            "temp": float(self.temp[y, x]),
            "biome": climate.BIOME_NAMES[int(self.biome[y, x])],
            "canopy": float(self.trees.cover[y, x]) if self.trees else 0.0,
            "dist_to_boundary": float(self.tect.dist[y, x]),
            "river_w": float(w.width[y, x]),
            "lake_depth": float(w.lake_depth[y, x]),
        }
        if self.sediment is not None:
            out["sediment"] = float(self.sediment[y, x])
            out["bedrock"] = float(self.bedrock[y, x])
            out["hardness"] = float(self.hardness[y, x])
        return out

    def tile(self, spec):
        """One hierarchical tile of this world. See terrain.tiles.generate_tile."""
        from .tiles import generate_tile
        return generate_tile(self, spec)


def apply_border(h, border, inplace=False):
    """Copy each neighbor's rim onto that side of `h`."""
    if border is None:
        return h
    if not inplace:
        h = np.array(h, copy=True)
    rows, cols = h.shape
    for side, spec in border.items():
        edge = np.asarray(spec["edge"], np.float32)
        if side == "left" and edge.shape == (rows,):
            h[:, 0] = edge
        elif side == "right" and edge.shape == (rows,):
            h[:, -1] = edge
        elif side == "top" and edge.shape == (cols,):
            h[0, :] = edge
        elif side == "bottom" and edge.shape == (cols,):
            h[-1, :] = edge
    return h


def border_bounds(border, shape):
    """Ghost samples for `grid.use_bounds`: the pixel just past the shared rim."""
    rows, cols = shape
    bounds = grid.Bounds()
    if not border:
        return bounds

    def fit(side, n):
        if side not in border:
            return None
        spec = border[side]
        ghost = spec.get("outer", spec.get("edge"))
        ghost = np.asarray(ghost, np.float32)
        if ghost.shape != (n,):
            ghost = np.asarray(spec["edge"], np.float32)
        return ghost if ghost.shape == (n,) else None

    bounds.left = fit("left", rows)
    bounds.right = fit("right", rows)
    bounds.top = fit("top", cols)
    bounds.bottom = fit("bottom", cols)
    bounds.clamp_x = bounds.left is not None or bounds.right is not None
    bounds.clamp_y = bounds.top is not None or bounds.bottom is not None
    return bounds


def _hermite_join(src, border, reach):
    """Fade each edge onto the neighbor's rim."""
    src = np.asarray(src, np.float64)
    rows, cols = src.shape
    span = int(max(2, min(12, int(reach), rows // 2, cols // 2)))
    t = np.arange(span, dtype=np.float64) / float(span)
    fade = 1.0 - t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    acc = np.zeros(src.shape, np.float64)
    weight = np.zeros(src.shape, np.float64)

    def add(side, edge):
        if side == "left":
            rim = src[:, 0]
            band = (slice(None), slice(0, span))
            flip = transpose = False
        elif side == "right":
            rim = src[:, -1]
            band = (slice(None), slice(-span, None))
            flip, transpose = True, False
        elif side == "top":
            rim = src[0, :]
            band = (slice(0, span), slice(None))
            flip, transpose = False, True
        else:
            rim = src[-1, :]
            band = (slice(-span, None), slice(None))
            flip, transpose = True, True
        p0 = np.asarray(edge, np.float64) - rim
        ramp = fade * p0[:, None]
        w = np.broadcast_to(fade, ramp.shape).copy()
        if flip:
            ramp = ramp[:, ::-1]
            w = w[:, ::-1]
        if transpose:
            ramp = ramp.T
            w = w.T
        acc[band] += ramp * w
        weight[band] += w

    for side, spec in border.items():
        if side not in ("left", "right", "top", "bottom"):
            continue
        edge = np.asarray(spec["edge"], np.float64)
        expect = rows if side in ("left", "right") else cols
        if edge.shape != (expect,):
            continue
        add(side, edge)
    out = src.copy()
    known = weight > 1e-6
    out[known] += acc[known] / weight[known]
    return out


def meet_border(height, border, reach=None):
    """Rebuild `height` so each neighbor owns the rim."""
    if not border:
        return height
    src = np.asarray(height, np.float64)
    rows, cols = src.shape
    if rows < 4 or cols < 4:
        return np.asarray(apply_border(src, border), np.float32)
    if reach is None:
        reach = max(32.0, min(rows, cols) / 5.0)
    out = _hermite_join(src, border, reach)
    desired = apply_border(out, border)
    sides = {
        "left": (slice(None), 0),
        "right": (slice(None), -1),
        "top": (0, slice(None)),
        "bottom": (-1, slice(None)),
    }
    owned = {}
    for side, spec in border.items():
        edge = np.asarray(spec["edge"], np.float64)
        slot = sides.get(side)
        if slot is None:
            continue
        expect = rows if side in ("left", "right") else cols
        if edge.shape != (expect,):
            continue
        owned[side] = edge
        out[slot] = desired[slot]
    for horizontal, vertical, y, x in (
        ("left", "top", 0, 0),
        ("right", "top", 0, -1),
        ("left", "bottom", -1, 0),
        ("right", "bottom", -1, -1),
    ):
        if horizontal not in owned or vertical not in owned:
            continue
        out[y, x] = 0.5 * (float(owned[horizontal][y]) + float(owned[vertical][x]))
    return out.astype(np.float32, copy=False)


def _smootherstep(t):
    """Perlin's quintic. Zero slope at both ends, so a blend does not crease."""
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _shift_line(values, dy):
    base = np.arange(values.shape[0], dtype=np.float64)
    index = base - np.asarray(dy, np.float64)
    return np.interp(index, base, np.asarray(values, np.float64))


def _box_mean(values, radius):
    radius = int(max(0, radius))
    if radius == 0:
        return np.asarray(values, np.float64)
    padded = np.pad(np.asarray(values, np.float64), radius, mode="edge")
    kernel = np.ones(radius * 2 + 1, np.float64) / (radius * 2 + 1)
    return np.convolve(padded, kernel, mode="valid")


def _edge_drift(strip):
    """How far a feature slides along the edge for each step toward this tile."""
    span, depth = strip.shape
    if depth < 4:
        return np.zeros(span, np.float64)
    newer = strip[:, -1]
    older = strip[:, -3]
    # A flat coast matches every small slide. Start at zero and move only
    # when another slide is actually closer, so a straight shore stays straight.
    best = np.zeros(span, np.float64)
    err = _box_mean((newer - older) ** 2, 4)
    for dy in range(-6, 7):
        if dy == 0:
            continue
        mismatch = _box_mean((newer - _shift_line(older, dy)) ** 2, 4)
        take = mismatch < err
        best[take] = dy
        err[take] = mismatch[take]
    # The pair is two pixels apart, so one step inward is half that slide.
    return _box_mean(best / 2.0, 5)


def _side_line(field, side, distance):
    if side == "left":
        return field[:, distance]
    if side == "right":
        return field[:, -1 - distance]
    if side == "top":
        return field[distance, :]
    return field[-1 - distance, :]


def _land_runs(mask, shortest=4):
    runs = []
    start = None
    for index, on in enumerate(mask):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= shortest:
                runs.append((start, index))
            start = None
    if start is not None and len(mask) - start >= shortest:
        runs.append((start, len(mask)))
    return runs


def _body_weight(origin, start, stop, feather):
    outside = np.maximum(start - origin, origin - (stop - 1))
    outside = np.maximum(outside, 0.0)
    return 1.0 - _smootherstep(outside / feather)


def _carry_run(field, strip, side, start, stop, drift, sea):
    """Carry one landmass in by its drift, without a shelf along the seam.

    Burt and Adelson blend a coarse shape over a wide zone and fine detail over
    a short one. The wide zone uses Perlin's quintic, so the slope is flat
    where the mass meets the rim and where it rejoins this tile.
    """
    depth = strip.shape[1]
    thickness = 0
    for column in range(depth - 1, -1, -1):
        if float((strip[start:stop, column] > sea).mean()) < 0.45:
            break
        thickness += 1
    thickness = max(thickness, 3)
    flow = float(np.clip(np.median(drift[start:stop]), -1.5, 1.5))
    span = field.shape[0] if side in ("left", "right") else field.shape[1]
    limit = (field.shape[1] if side in ("left", "right") else field.shape[0]) // 2 - 1
    reach = int(np.clip(max(thickness * 2, 24), 16, max(16, limit)))
    detail = max(4, reach // 5)
    profile = strip[:, -1].astype(np.float64)
    radius = max(3, (stop - start) // 6)
    low = _box_mean(profile, radius)
    high = profile - low
    positions = np.arange(span, dtype=np.float64)
    feather = 8.0
    for distance in range(reach):
        if distance >= (field.shape[1] if side in ("left", "right") else field.shape[0]):
            break
        slide = flow * distance
        origin = positions - slide
        mask = _body_weight(origin, start, stop, feather)
        coarse = (1.0 - _smootherstep(distance / max(reach - 1, 1))) * mask
        fine = (1.0 - _smootherstep(min(distance / detail, 1.0))) * mask
        line = _side_line(field, side, distance)
        low_here = _box_mean(line, radius)
        high_here = np.asarray(line, np.float64) - low_here
        blended = (_shift_line(low, slide) * coarse + low_here * (1.0 - coarse)
                   + _shift_line(high, slide) * fine + high_here * (1.0 - fine))
        line[:] = blended.astype(np.float32)


def carry_mass(height, border, sea=0.0):
    """Continue a landmass that meets the edge, along the way it was already heading.

    The shared rim stays the neighbor's edge. The body's broad shape eases into
    this tile over a wide zone, and its small ridges only near the cut. Open
    water on the edge is left to the short rim blend.
    """
    if not border:
        return height
    field = np.array(height, np.float32, copy=True)
    for side, spec in border.items():
        if side not in ("left", "right", "top", "bottom"):
            continue
        strip = spec.get("strip")
        if strip is None:
            continue
        strip = np.asarray(strip, np.float32)
        span = field.shape[0] if side in ("left", "right") else field.shape[1]
        if strip.ndim != 2 or strip.shape[0] != span or strip.shape[1] < 4:
            continue
        drift = _edge_drift(strip)
        for start, stop in _land_runs(strip[:, -1] > sea):
            _carry_run(field, strip, side, start, stop, drift, sea)
    return field


def _pin_biome(biome, border):
    if not border:
        return
    rows, cols = biome.shape
    for side, spec in border.items():
        col = spec.get("biome")
        if col is None:
            continue
        col = np.asarray(col)
        if side == "left" and col.shape == (rows,):
            biome[:, 0] = col
        elif side == "right" and col.shape == (rows,):
            biome[:, -1] = col
        elif side == "top" and col.shape == (cols,):
            biome[0, :] = col
        elif side == "bottom" and col.shape == (cols,):
            biome[-1, :] = col


def _validate_config(cfg):
    """Reject invalid domain values before allocating world-sized arrays."""
    import math
    if not isinstance(cfg.width, int) or not isinstance(cfg.height, int) or cfg.width < 8 or cfg.height < 8:
        raise ValueError("width and height must be integers >= 8")
    if not isinstance(cfg.seed, (int, np.integer)) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not (0.0 < cfg.land_fraction < 1.0) or not (0.0 < cfg.cont_fraction < 1.0):
        raise ValueError("land_fraction and cont_fraction must be between 0 and 1")
    if cfg.n_plates < 2 or cfg.erosion_passes < 0 or cfg.thermal_iters < 0:
        raise ValueError("n_plates >= 2, erosion_passes >= 0, thermal_iters >= 0 required")
    if cfg.depression_method not in ("iterative", "priority_flood"):
        raise ValueError("depression_method must be iterative or priority_flood")
    if cfg.erosion_model not in ("stream_power", "thermal", "none"):
        raise ValueError("erosion_model must be stream_power, thermal or none")
    if cfg.wind_mode not in ("belts", "easterly", "westerly"):
        raise ValueError("wind_mode must be belts, easterly or westerly")
    if cfg.noise_backend not in ("numpy", "numba"):
        raise ValueError("noise_backend must be numpy or numba")
    nonnegative = ("river_threshold", "river_width", "lake_evap", "rain_base",
                   "rain_orog", "rain_ocean_gain", "rain_recycle", "rain_blur", "erosion_k")
    if any(not math.isfinite(getattr(cfg, k)) or getattr(cfg, k) < 0 for k in nonnegative):
        raise ValueError("rain, river and erosion parameters must be finite and nonnegative")
    if cfg.river_width_ref <= 0 or cfg.rain_evap_cap <= 0 or cfg.tree_spacing <= 0:
        raise ValueError("river_width_ref, rain_evap_cap and tree_spacing must be positive")
    if cfg.rain_ocean_gain > 1 or cfg.rain_recycle > 1:
        raise ValueError("rain_ocean_gain and rain_recycle must not exceed 1")
    if not math.isfinite(cfg.world_width_m) or cfg.world_width_m <= 0:
        raise ValueError("world_width_m must be finite and positive")
    if cfg.sediment_iters < 0:
        raise ValueError("sediment_iters must be nonnegative")
    if cfg.terrain_model not in ("classic", "layered", "full"):
        raise ValueError("terrain_model must be classic, layered or full")
    if cfg.geography not in ("free", "poles", "dispersed"):
        raise ValueError("geography must be free, poles or dispersed")
    if cfg.crust_features < 1:
        raise ValueError("crust_features must be positive")


def generate(cfg=None, verbose=True, on_stage=None, on_preview=None, border=None,
             sculpt=None, sculpt_at=(0, 0), **overrides):
    """`on_stage(name)` is called before each stage; a big map takes a minute and a caller with a window needs something to put on it."""
    cfg = replace(cfg) if cfg is not None else Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown terrain setting: {k}")
        setattr(cfg, k, v)
    _validate_config(cfg)
    cfg = _scale_to_size(cfg)
    rng = np.random.default_rng(cfg.seed)
    timings = {}

    def stage(name, fn):
        if on_stage:
            on_stage(name)
        t0 = time.perf_counter()
        with noise.use_backend(cfg.noise_backend):
            out = fn()
        timings[name] = time.perf_counter() - t0
        if verbose:
            print(f"  {name:<12} {timings[name]:6.2f}s")
        if on_preview:
            on_preview(name, out)
        return out

    if verbose:
        print(f"generating {cfg.width}x{cfg.height} seed={cfg.seed}")
    tect = stage("plates", lambda: tectonics.build(cfg, rng))
    raw = stage("relief", lambda: elevation.tectonic_relief(tect, cfg, rng))
    raw = stage("hotspots", lambda: elevation.hotspot_chains(raw, tect, cfg, rng))
    pre = stage("texture", lambda: elevation.texture(raw.copy(), tect, cfg, rng))

    # Plates and crust are finished. Sea level is a share of the sphere.
    # Rain is computed next so erosion has a climate, then temperature and
    # biomes are computed again on the eroded surface. Latitude drives that
    # climate. It does not decide where the crust is.
    level = grid.area_quantile(pre, 1.0 - cfg.land_fraction)
    raw -= level
    pre -= level
    def coastline():
        h = elevation.coastal_plain(pre, cfg, rng)
        h = elevation.erode_margins(h, tect, cfg, rng)
        h = elevation.fray_coast(h, cfg, rng)
        if sculpt:
            h = elevation.paint_relief(h, sculpt)
            h = h + elevation.sculpt_grain(h.shape, sculpt, cfg, sculpt_at)
        if not border:
            return h
        # The rim blend keeps the seam. A landmass on that edge then keeps
        # travelling into this tile instead of ending on the cut.
        return carry_mass(meet_border(h, border), border)

    pre = stage("coastline", coastline)

    def erosion_rain():
        t = climate.temperature(pre, cfg, np.random.default_rng([cfg.seed, 1]))[0]
        return climate.runoff(pre, cfg, np.random.default_rng([cfg.seed, 2]),
                              temp=t)[0]

    wet = stage("pre-rain", erosion_rain)
    strata = {}

    def _settle(carved, filled, flow, rec):
        layered = cfg.sediment or cfg.terrain_model in ("layered", "full")
        if not layered:
            return carved, filled, flow, rec
        from . import aeolian, erosion_particles

        wrap = grid.active_bounds() is None
        cell_m = float(cfg.world_width_m) / float(max(cfg.width, 1))
        discharge = flow
        rainfall = wet if cfg.terrain_model in ("layered", "full") else None
        if cfg.terrain_model == "full":
            spread = hydrology.accumulate_mfd(carved, wet)
            discharge = (0.65 * np.asarray(flow, np.float32) + 0.35 * spread).astype(np.float32)
        iters = int(cfg.sediment_iters)
        if cfg.terrain_model == "full":
            iters = min(iters, 4)
        surface, bedrock, sediment, hardness = hydrology.deposit_sediment(
            carved, discharge, cfg.seed, iters=iters, wrap_x=wrap,
            rainfall=rainfall, cell_m=cell_m)
        if cfg.terrain_model == "full":
            soft = (1.0 - hardness).astype(np.float32)
            passes = (
                hydrology.ErosionPass(max(cell_m * 8.0, 64.0), 2, 0.45, 0.20, 0.40),
                hydrology.ErosionPass(max(cell_m * 3.0, 24.0), 2, 0.35, 0.25, 0.30),
            )
            bedrock, sediment, surface = hydrology.erode_multiscale(
                bedrock, sediment, soft, discharge, cell_m, passes=passes,
                wrap_x=wrap)
            bedrock, sediment = erosion_particles.rain_splash(
                bedrock, sediment, soft, seed=cfg.seed, wrap_x=wrap)
            bedrock, sediment = hydrology.flow_guided_warp(
                bedrock, sediment, rec, flow, displacement_cells=0.35,
                substeps=3, wrap_x=wrap)
            dryness = np.clip(1.0 - np.clip(np.asarray(wet, np.float32), 0.0, 1.0), 0.0, 1.0)
            bedrock, sediment = aeolian.transport(
                bedrock, sediment, dryness, cfg.wind_mode, seed=cfg.seed, wrap_x=wrap)
            surface = bedrock + sediment
        filled = hydrology.fill_depressions(
            surface, 0.0, method=cfg.depression_method)
        rec, _, _ = hydrology.flow_routing(filled)
        flow = hydrology.accumulate(filled, rec, wet)
        strata["bedrock"] = bedrock
        strata["sediment"] = sediment
        strata["hardness"] = hardness
        return surface, filled, flow, rec

    def erode():
        passes = 0 if cfg.erosion_model == "none" else (
            1 if cfg.erosion_model == "thermal" else cfg.erosion_passes)
        k = 0.0 if cfg.erosion_model != "stream_power" else cfg.erosion_k
        iters = 0 if cfg.erosion_model == "none" else cfg.thermal_iters
        if border is None:
            carved, filled, flow, rec = hydrology.stream_power(
                pre.copy(), 0.0, passes, k, cfg.erosion_m,
                cfg.erosion_n, iters, cfg.talus, weights=wet,
                depression_method=cfg.depression_method)
            return _settle(carved, filled, flow, rec)
        with grid.use_bounds(border_bounds(border, pre.shape)):
            carved, filled, flow, rec = hydrology.stream_power(
                pre.copy(), 0.0, passes, k, cfg.erosion_m,
                cfg.erosion_n, iters, cfg.talus, weights=wet,
                depression_method=cfg.depression_method)
            carved, filled, flow, rec = _settle(carved, filled, flow, rec)
        return meet_border(carved, border), filled, flow, rec

    eroded, er_filled, er_flow, er_rec = stage("erosion", erode)
    sst = stage("temperature", lambda: climate.temperature(
        eroded, cfg, np.random.default_rng([cfg.seed, 1]))[0])
    runoff, evap = stage("climate", lambda: climate.runoff(
        eroded, cfg, np.random.default_rng([cfg.seed, 2]), temp=sst))
    def run_rivers():
        if border is None:
            return rivers.build(
                eroded.copy(), cfg, rng, routed=(er_filled, er_rec, er_flow),
                runoff=runoff, evap=evap, bed_hardness=strata.get("hardness"))
        bed = meet_border(eroded, border)
        with grid.use_bounds(border_bounds(border, bed.shape)):
            height, water = rivers.build(
                bed, cfg, rng, runoff=runoff, evap=evap,
                bed_hardness=strata.get("hardness"))
        return meet_border(height, border), water

    h, water = stage("rivers", run_rivers)
    placed_ice = placed_sand = placed_void = None
    stamps = None
    if sculpt:
        from .v2.generation.place import place_height, placed_strokes
        stamps = placed_strokes(sculpt)
    if stamps:
        cell_m = float(cfg.world_width_m) / float(max(int(cfg.width), 1))

        def stamp():
            return place_height(h, stamps, cell_m, temp=sst, rain=runoff)

        h, placed_ice, placed_sand, placed_void = stage("placed", stamp)

    def biomes():
        t, sw = climate.temperature(h, cfg, np.random.default_rng([cfg.seed, 1]),
                                    humid=runoff)
        biome = climate.biomes(h, t, sw, runoff, cfg)
        _pin_biome(biome, border)
        return t, sw, biome

    temp, swing, biome = stage("biomes", biomes)

    world = World(cfg=cfg, tect=tect, height_raw=raw, height_pre=pre,
                  height_eroded=eroded, height=h, water=water, runoff=runoff,
                  temp=temp, swing=swing, biome=biome, trees=None, timings=timings,
                  bedrock=strata.get("bedrock"), sediment=strata.get("sediment"),
                  hardness=strata.get("hardness"),
                  placed_ice=placed_ice, placed_sand=placed_sand, placed_void=placed_void)
    world.trees = stage("trees", lambda: trees.build(
        world, np.random.default_rng([cfg.seed, 3])))
    if verbose:
        print(f"  {'total':<12} {sum(timings.values()):6.2f}s  "
              f"land={world.land.mean():.0%}  max={h.max():.2f}  min={h.min():.2f}  "
              f"lakes={water.lake_id.max()}  rivers={len(water.polylines)}  "
              f"trees={len(world.trees)}")
    return world
