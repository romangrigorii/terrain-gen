"""Stage order and the finished world. Settings live in `config`."""
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np
from . import climate, elevation, grid, hydrology, rivers, tectonics, trees, noise
from .boundary_momentum import (
    apply_boundary_slope, coast_distance, conform_shore, continue_boundary,
    continue_shore, momentum, side_weight,
)
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
    sea_level: float = None          # subtracted from this tile; shared across a planet

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


def _border_sides(border):
    """Side records only. `sea_level` lives on the same dict and is a number."""
    if not border:
        return
    for side, spec in border.items():
        if side in ("left", "right", "top", "bottom") and isinstance(spec, dict):
            yield side, spec


def apply_border(h, border, inplace=False):
    """Copy each neighbor's rim onto that side of `h`."""
    if border is None:
        return h
    if not inplace:
        h = np.array(h, copy=True)
    rows, cols = h.shape
    for side, spec in _border_sides(border):
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
        strip = spec.get("strip")
        if strip is not None:
            strip = np.asarray(strip, np.float32)
            if strip.ndim == 2 and strip.shape[0] == n and strip.shape[1] >= 2:
                return np.asarray(strip[:, -2], np.float32)
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

    for side, spec in _border_sides(border):
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
    for side, spec in _border_sides(border):
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


def refresh_flow(height, weights, method="iterative"):
    """Route `height` as it stands. Call this after a border edit, not before it."""
    filled = hydrology.fill_depressions(np.asarray(height, np.float32), 0.0, method=method)
    rec, _, _ = hydrology.flow_routing(filled)
    flow = hydrology.accumulate(filled, rec, weights)
    return filled, flow, rec


def resolve_sea_level(field, land_fraction, border=None):
    """One sea level for a planet. A bordered tile does not choose another."""
    if border:
        shared = border.get("sea_level")
        if shared is not None:
            return float(shared)
        matched = _level_from_edges(field, border)
        if matched is not None:
            return matched
    return float(grid.area_quantile(field, 1.0 - land_fraction))


def _level_from_edges(field, border):
    """Shift that puts this tile's rim in the neighbor's sea frame."""
    field = np.asarray(field, np.float64)
    rows, cols = field.shape
    deltas = []
    for side, spec in border.items():
        if not isinstance(spec, dict) or "edge" not in spec:
            continue
        edge = np.asarray(spec["edge"], np.float64)
        if side == "left" and edge.shape == (rows,):
            deltas.append(field[:, 0] - edge)
        elif side == "right" and edge.shape == (rows,):
            deltas.append(field[:, -1] - edge)
        elif side == "top" and edge.shape == (cols,):
            deltas.append(field[0, :] - edge)
        elif side == "bottom" and edge.shape == (cols,):
            deltas.append(field[-1, :] - edge)
    if not deltas:
        return None
    return float(np.median(np.concatenate(deltas)))


def carry_mass(height, border, sea=0.0):
    """Continue every incoming edge by its own sideways motion.

    Broad shape and fine ridges both run in from the strip. The shoreline
    distance runs with them, so a diagonal coast is not only a height match.
    """
    if not border:
        return height
    source = np.asarray(height, np.float64)
    rows, cols = source.shape
    acc = np.zeros_like(source)
    weight = np.zeros_like(source)
    shore_acc = np.zeros_like(source)
    shore_w = np.zeros_like(source)
    pinned = []
    source_shore = None
    for side, spec in border.items():
        if side not in ("left", "right", "top", "bottom"):
            continue
        strip = spec.get("strip")
        if strip is None:
            continue
        strip = np.asarray(strip, np.float64)
        span = rows if side in ("left", "right") else cols
        if strip.ndim != 2 or strip.shape[0] != span or strip.shape[1] < 2:
            continue
        depth = cols if side in ("left", "right") else rows
        # A long copy of the edge profile is a wall perpendicular to the seam.
        # The coast may step sideways a little. The new tile keeps its own grain.
        reach = float(min(32, max(1, depth - 1)))
        detail = 0.0
        sigma = 4.0
        velocity, residual = momentum(strip, tangent_sigma=sigma)
        velocity = np.clip(velocity * 0.25, -0.7, 0.7)
        continued = continue_boundary(
            source, strip, side, reach=reach, detail_reach=detail,
            tangent_sigma=sigma, velocity=velocity, residual=residual)
        w = side_weight(source.shape, side, reach)
        acc += w * continued
        weight += w
        if source_shore is None:
            source_shore = coast_distance(source, sea)
        shore = continue_shore(
            source, strip, side, sea=sea, reach=reach, detail_reach=detail,
            tangent_sigma=sigma)
        shore_acc += w * shore
        shore_w += w
        pinned.append((side, strip))
    if not pinned:
        return np.asarray(height, np.float32)
    out = source.copy()
    known = weight > 1e-8
    out[known] = acc[known] / weight[known]
    if source_shore is not None:
        shore = source_shore.copy()
        known_shore = shore_w > 1e-8
        shore[known_shore] = shore_acc[known_shore] / shore_w[known_shore]
        out = conform_shore(out, shore, sea)
    _pin_edges(out, [(side, strip[:, -1]) for side, strip in pinned])
    return out.astype(np.float32)


def _seed_boundary_slope(height, border):
    """Give the solver the incoming slope once, before it moves sediment."""
    if not border:
        return height
    rows, cols = height.shape
    for side, spec in border.items():
        if side not in ("left", "right", "top", "bottom") or not isinstance(spec, dict):
            continue
        strip = spec.get("strip")
        if strip is None:
            continue
        strip = np.asarray(strip, np.float64)
        span = rows if side in ("left", "right") else cols
        if strip.ndim != 2 or strip.shape[0] != span or strip.shape[1] < 2:
            continue
        if (side in ("left", "right") and cols < 2) or (side in ("top", "bottom") and rows < 2):
            continue
        apply_boundary_slope(height, strip, side)
    return height


def check_corner(a, b, tolerance=1e-5):
    """A shared corner is one value. Two neighbors may not vote a third."""
    if abs(float(a) - float(b)) > tolerance:
        raise ValueError("Neighboring tiles disagree at their shared corner")
    return float(a)


def _pin_edges(field, pinned):
    """The shared pixel is the neighbor's. A corner must already be one value."""
    slots = {
        "left": (slice(None), 0),
        "right": (slice(None), -1),
        "top": (0, slice(None)),
        "bottom": (-1, slice(None)),
    }
    owned = {}
    for side, edge in pinned:
        field[slots[side]] = edge
        owned[side] = np.asarray(edge, np.float64)
    for horizontal, vertical, y, x in (
        ("left", "top", 0, 0),
        ("right", "top", 0, -1),
        ("left", "bottom", -1, 0),
        ("right", "bottom", -1, -1),
    ):
        if horizontal in owned and vertical in owned:
            field[y, x] = check_corner(owned[horizontal][y], owned[vertical][x])


def _pin_biome(biome, border):
    if not border:
        return
    rows, cols = biome.shape
    for side, spec in _border_sides(border):
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
    level = resolve_sea_level(pre, cfg.land_fraction, border)
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
        # The incoming strip sets the rim and carries its direction inward.
        # Later stages pin that rim. They do not fade the continuation away.
        return carry_mass(h, border)

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
        def pin(field):
            apply_border(field, border, inplace=True)

        started = np.array(pre, np.float32, copy=True)
        _seed_boundary_slope(started, border)
        with grid.use_bounds(border_bounds(border, pre.shape)):
            carved, filled, flow, rec = hydrology.stream_power(
                started, 0.0, passes, k, cfg.erosion_m,
                cfg.erosion_n, iters, cfg.talus, weights=wet,
                depression_method=cfg.depression_method, pin=pin)
            carved, filled, flow, rec = _settle(carved, filled, flow, rec)
            apply_border(carved, border, inplace=True)
            filled, flow, rec = refresh_flow(carved, wet, cfg.depression_method)
        return carved, filled, flow, rec

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
        bed = apply_border(np.array(eroded, np.float32, copy=True), border, inplace=True)
        with grid.use_bounds(border_bounds(border, bed.shape)):
            height, water = rivers.build(
                bed, cfg, rng, runoff=runoff, evap=evap,
                bed_hardness=strata.get("hardness"))
        apply_border(height, border, inplace=True)
        return height, water

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

    world = World(cfg=cfg, tect=tect, height_raw=raw, height_pre=pre, sea_level=float(level),
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
