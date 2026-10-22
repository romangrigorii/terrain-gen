"""Stage 5: where the rain falls, and so how much water each cell contributes."""
import numpy as np

from . import grid, noise


def _belts(shape, cfg, rng):
    """Zonal rain factor: three wet bands and three dry ones, wandering."""
    h, w = shape
    lat = grid.latitude(shape) + np.zeros((1, w))
    lat = lat + cfg.rain_wobble * noise.fbm(h, w, rng, cfg.rain_wobble_periods, 3)
    band = np.cos(3.0 * np.pi * np.clip(lat, -1, 1))
    return 1.0 - cfg.rain_belts * (1.0 - band) * 0.5


def _march(h, land, belts, cap, cfg, inbound=None, upwind=None, trace=False):
    """Blow moisture across longitude. `inbound` is the moisture arriving from upwind.

    A full cylinder omits `inbound` and wraps from its own last column. A tile of
    a larger field passes the moisture and the height the global march already
    carried to this tile's upwind edge.
    """
    hh, w = h.shape
    if inbound is None:
        m = cap[:, -1].copy()
    else:
        m = np.asarray(inbound, np.float64).copy()
        if m.shape != (hh,):
            raise ValueError("inbound moisture must be one value per row")
    rain = np.zeros((hh, w))
    entering = np.zeros((hh, w)) if trace else None
    if upwind is not None:
        upwind = np.asarray(upwind, np.float64)
        if upwind.shape != (hh,):
            raise ValueError("upwind height must be one value per row")
    passes = 1 if inbound is not None else 2
    for lap in range(passes):
        prev = h[:, -1] if upwind is None else upwind.copy()
        writing = lap == passes - 1
        for x in range(w):
            hx = h[:, x]
            climb = np.maximum(hx - prev, 0.0)   # upwind gradient, this cell
            prev = hx
            wet = land[:, x]
            r = np.minimum(m * belts[:, x] * (cfg.rain_base +
                                              cfg.rain_orog * climb), m)
            r = np.where(wet, r, 0.0)
            if writing and entering is not None:
                entering[:, x] = m
            m = m - r + np.where(wet, r * cfg.rain_recycle,
                                 (cap[:, x] - m) * cfg.rain_ocean_gain)
            if writing:
                rain[:, x] = r
    if trace:
        return rain, entering
    return rain


def capacity(temp, cfg, sea_level=0.0, h=None):
    """How much moisture the air can carry, from the temperature under it."""
    cap = np.exp(cfg.rain_capacity * (temp - float(temp.mean())))
    if h is not None:
        sea = h <= sea_level
        if sea.any():
            cap = cap / max(1e-6, float(cap[sea].mean()))
    return cap


def _shore_weight(land, reach, east):
    """How near the shore is, looking in one direction only, decaying with distance."""
    decay = np.exp(-1.0 / max(1e-6, reach))
    cols = range(land.shape[1] - 1, -1, -1) if east else range(land.shape[1])
    w = np.zeros(land.shape[0])
    out = np.zeros(land.shape)
    for lap in range(2):
        for x in cols:
            w = np.maximum(land[:, x], w * decay)
            if lap:
                out[:, x] = w
    return out


def currents(h, cfg, sea_level=0.0):
    """Sea surface temperature anomaly from the subtropical gyres, in degrees C."""
    land = h > sea_level
    sea = ~land
    if not sea.any():
        return np.zeros_like(h)
    lat = np.abs(grid.latitude(h.shape)) + np.zeros_like(h)
    cold = np.exp(-((lat - cfg.current_lat) / 0.18) ** 2)
    a = (cfg.current_warm * _shore_weight(land, cfg.current_reach, east=False) * lat
         - cfg.current_cold * _shore_weight(land, cfg.current_reach, east=True) * cold)
    m = sea.astype(float)
    a = grid.blur(a * m, cfg.current_blur) / np.maximum(grid.blur(m, cfg.current_blur), 1e-6)
    return np.where(sea, a - float(a[sea].mean()), 0.0)


def rainfall(h, cfg, rng, sea_level=0.0, temp=None):
    """Rain per cell. Winds are zonal: easterly in the tropics and at the poles, westerly in between, which is why the marching is done in two groups."""
    land = h > sea_level
    belts = _belts(h.shape, cfg, rng)
    cap = (np.ones_like(h) if temp is None
           else capacity(temp, cfg, sea_level, h))
    lat = np.abs(grid.latitude(h.shape))[:, 0]
    mode = getattr(cfg, "wind_mode", "belts")
    if mode == "easterly":
        westerly = np.zeros(lat.shape, dtype=bool)
    elif mode == "westerly":
        westerly = np.ones(lat.shape, dtype=bool)
    else:
        westerly = (lat > 1 / 3) & (lat < 2 / 3)

    rain = np.zeros_like(h)
    for sel, flip in ((westerly, False), (~westerly, True)):
        if not sel.any():
            continue
        sl = slice(None, None, -1) if flip else slice(None)
        out = _march(h[sel][:, sl], land[sel][:, sl], belts[sel][:, sl],
                     cap[sel][:, sl], cfg)
        rain[sel] = out[:, sl]
    return grid.blur(rain, cfg.rain_blur)


def runoff(h, cfg, rng, sea_level=0.0, temp=None):
    """Rainfall as accumulation weights: mean one over land, zero at sea."""
    land = h > sea_level
    rain = rainfall(h, cfg, rng, sea_level, temp)
    if not land.any():
        return np.zeros_like(h), np.full_like(h, cfg.lake_evap)
    mean = max(1e-9, float(rain[land].mean()))
    out = np.where(land, rain / mean, 0.0)
    evap = cfg.lake_evap / np.clip(np.where(land, out, 1.0), cfg.rain_evap_cap, None)
    return out, evap


def _lapse(humid, land, cfg):
    """C lost per unit of height, steeper where the air is dry."""
    if humid is None or not land.any():
        return cfg.temp_lapse
    vals = np.sort(humid[land])
    rank = 0.5 * (np.searchsorted(vals, humid, side="left") +
                  np.searchsorted(vals, humid, side="right")) / max(1, len(vals))
    return cfg.temp_lapse * (1.0 + cfg.temp_lapse_moist * (1.0 - 2.0 * rank))


def temperature(h, cfg, rng, sea_level=0.0, humid=None):
    """Mean annual temperature and the seasonal half-range, both in degrees C."""
    lat = grid.latitude(h.shape)
    lat = np.clip(lat + cfg.temp_wobble * noise.fbm(
        h.shape[0], h.shape[1], rng, cfg.temp_wobble_periods, 3), -1.0, 1.0)
    base = cfg.temp_equator + (cfg.temp_pole - cfg.temp_equator) * lat ** 2
    weight = (1.0 - cfg.temp_polar_amp) + 3.0 * cfg.temp_polar_amp * lat ** 2
    land = h > sea_level
    if land.any():
        weight = weight / max(1e-6, float(weight[land].mean()))
    temp = (base + cfg.temp_offset * weight
            - _lapse(humid, land, cfg) * np.maximum(h - sea_level, 0.0)
            + currents(h, cfg, sea_level))
    swing = _swing(lat, land, cfg)
    frozen = ~land & (temp + swing < ICE_C)
    if frozen.any() and not (land | frozen).all():
        swing = _swing(lat, land | frozen, cfg)
    return temp, swing


def _swing(lat, closed, cfg):
    """Seasonal half-range: latitude, damped by how near open water is."""
    cont = 1.0 - np.exp(-grid.edt(closed) / max(1e-6, cfg.temp_cont_reach))
    return (cfg.temp_swing * np.abs(lat)
            * (cfg.temp_maritime + (1.0 - cfg.temp_maritime) * cont))

(OCEAN, ICE, TUNDRA, TAIGA, DESERT, STEPPE, TEMPERATE_FOREST,
 TEMPERATE_RAINFOREST, SHRUBLAND, SAVANNA, TROPICAL_SEASONAL,
 TROPICAL_RAINFOREST) = range(12)

BIOME_NAMES = [
    "ocean", "ice", "tundra", "taiga", "desert", "steppe",
    "temperate forest", "temperate rainforest", "shrubland",
    "savanna", "tropical seasonal forest", "tropical rainforest",
]

MI_BANDS = (0.4, 0.7, 1.1, 1.8)
T_BANDS = (0.0, 6.0, 12.0, 17.0, 22.0)
ICE_C = 0.0     # warmest month below this and nothing ever thaws
TREE_C = 6.0    # warmest month below this and nothing grows tall

_DE, _TU, _TA, _ST, _SH, _SV = (DESERT, TUNDRA, TAIGA, STEPPE, SHRUBLAND,
                                SAVANNA)
_TF, _TR, _PS, _PR = (TEMPERATE_FOREST, TEMPERATE_RAINFOREST,
                      TROPICAL_SEASONAL, TROPICAL_RAINFOREST)
BIOME_MATRIX = np.array([
    [_DE, _DE, _DE, _DE, _DE, _DE],     # arid       mi < 0.25
    [_TU, _TU, _ST, _ST, _SH, _SV],     # semiarid   mi < 0.5
    [_TU, _TA, _TA, _TF, _PS, _PS],     # subhumid   mi < 1.1
    [_TU, _TA, _TF, _TF, _TF, _PR],     # humid      mi < 1.8
    [_TU, _TA, _TR, _TR, _PR, _PR],     # perhumid   mi >= 1.6
], dtype=np.int8)


def precip_mm(cfg):
    """What a runoff of 1.0 means in mm/yr, at this world's temperature."""
    return max(1.0, cfg.precip_mean_mm * (1.0 + cfg.rain_per_degree * cfg.temp_offset))


def moisture_index(temp, swing, runoff, cfg):
    """Rain relative to what the local heat can evaporate. The moisture axis."""
    biotemp = np.zeros_like(temp)
    for phase in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        biotemp += np.clip(temp + swing * np.cos(phase), 0.0, 30.0)
    biotemp /= 12.0
    pet = np.maximum(58.93 * biotemp, 58.93 * 1.5)      # mm/yr
    return precip_mm(cfg) * runoff / pet


def biomes(h, temp, swing, runoff, cfg, sea_level=0.0):
    """What grows where: temperature against rain *for that temperature*."""
    mi = moisture_index(temp, swing, runoff, cfg)
    out = BIOME_MATRIX[np.searchsorted(MI_BANDS, grid.blur(mi, cfg.biome_blur)),
                       np.searchsorted(T_BANDS, grid.blur(temp, cfg.biome_blur))]
    warmest = temp + swing
    out = np.where(warmest < TREE_C, TUNDRA, out)
    out = np.where(warmest < ICE_C, ICE, out)
    return np.where(h > sea_level, out, OCEAN).astype(np.int8)
