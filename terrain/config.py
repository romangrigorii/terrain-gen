"""Generation settings, and how pixel-sized knobs scale with the map."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    # --- grid ---
    width: int = 512
    height: int = 384
    seed: int = 0
    depression_method: str = "iterative"  # iterative | priority_flood
    erosion_model: str = "stream_power"  # stream_power | thermal | none
    wind_mode: str = "belts"  # belts | easterly | westerly
    noise_backend: str = "numpy"  # numpy | numba (optional JIT)
    ref_width: int = 512    # size these knobs are tuned at; see `_scale_to_size`

    # --- plates ---
    n_plates: int = 21
    plate_warp: float = 34.0        # px of noise displacement on plate borders
    plate_warp_fine: float = 15.0   # second, shorter-scale displacement
    plate_warp_periods: int = 3
    plate_size_var: float = 0.42    # 0 = all plates the same size
    cont_plate_fraction: float = 0.42
    geography: str = "free"         # free | poles | dispersed; never a polar sink
    crust_features: int = 24        # spherical blobs in the continental potential
    stress_blur: float = 2.0        # smoothing of the raw strain derivatives
    stress_spread: float = 5.0      # softens medial-axis rays after propagation
    conv_gate: tuple = (0.18, 0.48)
    rift_gate: tuple = (0.38, 0.78)
    shear_gate: tuple = (0.55, 0.92)

    # --- crust ---
    cont_base: float = 0.58
    ocean_base: float = -0.72
    cont_fraction: float = 0.38     # area of continental crust (shelves included)
    crust_periods: float = 2.0      # lower -> fewer, larger continents
    crust_plate_weight: float = 1.15  # how strongly plates dictate the outline
    crust_blur: float = 9.0
    crust_warp: float = 18.0        # detaches coastlines from the plate outline
    crust_warp_periods: float = 5.0
    crust_sharpness: float = 0.85    # higher -> narrower shelves, steeper margins
    upland_h: float = 0.30
    upland_periods: float = 3.0
    polar_start: float = 0.56       # kept so older configs load; geology ignores it
    polar_ocean: float = 0.0        # kept so older configs load; geology ignores it
    polar_wobble: float = 0.13      # kept so older configs load; geology ignores it
    polar_wobble_periods: float = 2.5
    land_fraction: float = 0.29

    # --- boundary relief (amplitude / falloff width in px) ---
    collision_h: float = 1.30
    collision_w: float = 26.0
    trench_h: float = 0.70
    trench_w: float = 5.0
    arc_h: float = 0.85
    arc_offset: float = 16.0
    arc_w: float = 7.0
    arc_break_periods: float = 16.0   # along-strike spacing of arc volcanoes
    arc_break_bias: float = 0.2       # lower -> more gaps in the chain
    cordillera_h: float = 0.55
    cordillera_w: float = 30.0
    rift_depth: float = 0.45
    rift_w: float = 9.0
    rift_shoulder: float = 0.40
    ridge_h: float = 0.28
    ridge_w: float = 18.0
    transform_h: float = 0.13
    transform_w: float = 4.0
    ridge_age_thresh: float = 0.15
    age_depth: float = 0.35
    age_scale: float = 90.0
    age_warp: float = 26.0          # breaks up the seafloor-age contours
    age_warp_periods: float = 4.0

    # --- hotspots ---
    n_hotspots: int = 5
    hotspot_max_crust: float = 0.35   # only seed hotspots in oceanic crust
    hotspot_h: float = 1.15
    hotspot_sigma: float = 3.2
    hotspot_spacing: float = 11.0
    hotspot_chain: tuple = (3, 9)

    # --- texture / coast ---
    texture_h: float = 0.36
    texture_periods: int = 6
    texture_warp: float = 12.0
    abyss_h: float = 0.10
    coast_h: float = 0.15           # bays and headlands
    coast_h_fine: float = 0.055     # fretted edge
    coast_band: float = 0.13        # elevation range the fraying reaches
    coast_periods: int = 10
    coast_slope: float = 0.024      # max rise per cell as land leaves the shore
    coast_slope_periods: float = 5.0  # how that cap varies along the coast
    coast_plain_zone: float = 30.0    # how far inland the cap reaches, in px
    margin_h: float = 0.16          # extra fray on a continent-ocean margin
    margin_h_fine: float = 0.07
    margin_periods: float = 8.0     # coarse: finer noise breaks the coast up
    margin_zone: float = 45.0       # falloff from the plate boundary, in px
    margin_reach: float = 22.0      # width of the coastal corridor, in px
    margin_gain_max: float = 3.0    # cap on the slope compensation
    margin_cap: float = 1.0         # clip on the fray noise, in std devs
    margin_slope_blur: float = 2.0
    margin_cut_h: float = 0.30      # drowned inlets cut into a margin coast
    margin_cut_thresh: float = 1.0  # in std devs; higher -> fewer inlets
    margin_cut_cap: float = 1.5
    margin_cut_periods: float = 10.0
    margin_cut_offset: float = 1.0  # centre of the cut window, px inland
    margin_cut_w: float = 7.0

    # --- erosion ---
    erosion_passes: int = 4
    erosion_k: float = 0.009
    erosion_m: float = 0.5
    erosion_n: float = 1.0
    thermal_iters: int = 12
    talus: float = 0.045

    # --- climate ---
    rain_base: float = 0.006        # moisture a land cell takes out of the air
    rain_orog: float = 1.0          # extra rain per unit of upwind climb
    rain_ocean_gain: float = 0.08   # moisture an ocean cell puts back
    rain_belts: float = 0.9
    rain_wobble: float = 0.10       # how far the bands wander, in latitude
    rain_wobble_periods: float = 2.0
    rain_recycle: float = 0.55      # share of rain a land cell puts back up
    rain_blur: float = 2.5          # px; weather is not one cell wide
    rain_evap_cap: float = 0.2      # driest runoff `lake_evap` is divided by
    rain_capacity: float = 0.07
    temp_equator: float = 27.0      # mean annual C at the equator, at sea level
    temp_pole: float = -25.0        # ditto at the poles
    temp_offset: float = 0.0
    temp_polar_amp: float = 0.5
    temp_lapse: float = 38.0
    temp_lapse_moist: float = 0.25
    # --- ocean currents ---
    current_cold: float = 6.0
    current_warm: float = 4.0
    current_reach: float = 40.0     # px offshore the anomaly reaches
    current_lat: float = 0.35       # |latitude| the upwelling band sits on
    current_blur: float = 8.0       # px; a gyre is not a coastline
    temp_wobble: float = 0.06       # how far the isotherms wander, in latitude
    temp_wobble_periods: float = 2.0
    temp_swing: float = 22.0        # seasonal half-range at the pole, deep inland
    temp_maritime: float = 0.35     # share of that swing a coast still gets
    temp_cont_reach: float = 15.0
    precip_mean_mm: float = 750.0   # what a runoff of 1.0 means, in mm/yr
    rain_per_degree: float = 0.05
    biome_blur: float = 2.0         # px
    biome_soften: float = 2.5       # px
    biome_tint: float = 0.35        # biome colour mixed into the relief layer

    # --- lakes and rivers ---
    lake_min_depth: float = 4e-3    # shallower closed basins are just wet ground
    lake_min_area: int = 30         # cells; below this a basin is wet ground
    lake_evap: float = 3.0
    outlet_carve_passes: int = 2
    outlet_carve_depth: float = 0.012   # notch cut into a lake's pour point
    outlet_carve_slope: float = 6e-4    # gradient of the carved outflow channel
    outlet_carve_len: int = 220         # max cells carved downstream of a lake
    river_threshold: float = 0.0006     # drainage area needed to become a river
    river_width: float = 0.85           # channel width in cells at that threshold
    river_width_max: float = 4.0
    river_width_cap: float = 6.0        # widest channel in cells, at any map size
    river_width_exp: float = 0.45       # width goes as discharge to this power
    river_width_ref: float = 0.0006
    river_incision: float = 0.02        # how deep the channel sits in the bed
    meander_amp: float = 1.5            # lateral swing in cells, times sqrt(width)
    meander_period: float = 30.0        # along-path wavelength, in cells
    meander_taper: float = 8.0          # vertices over which the swing fades out

    # --- trees ---
    tree_mi_open: float = 0.45
    tree_mi_closed: float = 1.5
    tree_line_fade: float = 5.0     # C of warmest month over which forest thins out
    tree_gallery: float = 0.75      # canopy a bank gets that its own rain would not
    tree_gallery_reach: float = 3.0  # px from water that reaches
    tree_grove: float = 0.55        # clumping, in std devs of the noise on `mi`
    tree_grove_periods: float = 22.0
    tree_spacing: float = 2.0       # px between trees under a closed canopy
    tree_size_var: float = 0.35     # lognormal sigma on trunk size
    tree_mix: float = 6.0
    tree_relief: float = 0.55

    world_width_m: float = 200_000.0
    sediment: bool = False
    sediment_iters: int = 8
    terrain_model: str = "classic"

_PX_FIELDS = (
    "plate_warp", "plate_warp_fine", "stress_blur", "stress_spread",
    "crust_blur", "crust_warp", "collision_w", "trench_w", "arc_offset",
    "arc_w", "cordillera_w", "rift_w", "ridge_w", "transform_w", "age_scale",
    "age_warp", "hotspot_sigma", "hotspot_spacing", "texture_warp",
    "coast_plain_zone", "rain_blur", "temp_cont_reach", "biome_blur",
    "biome_soften", "current_reach", "current_blur",
    "margin_zone", "margin_reach", "margin_cut_offset",
    "margin_cut_w", "margin_slope_blur", "meander_period", "meander_taper",
    "meander_amp", "river_width", "river_width_max",
    "tree_gallery_reach", "tree_mix",
)
_PER_CELL_FIELDS = ("coast_slope", "talus", "outlet_carve_slope",
                    "rain_base", "rain_ocean_gain")


def _scale_to_size(cfg):
    """Put every pixel-denominated knob back on the map's own scale."""
    s = cfg.width / max(1, cfg.ref_width)
    if s == 1.0:
        return cfg
    cfg = replace(cfg)
    for k in _PX_FIELDS:
        setattr(cfg, k, getattr(cfg, k) * s)
    for k in _PER_CELL_FIELDS:
        setattr(cfg, k, getattr(cfg, k) / s)
    cfg.outlet_carve_len = max(1, round(cfg.outlet_carve_len * s))
    cfg.lake_min_area = max(1, round(cfg.lake_min_area * s * s))   # an area
    if cfg.river_width_max > cfg.river_width_cap:
        cfg.river_width *= cfg.river_width_cap / cfg.river_width_max
        cfg.river_width_max = cfg.river_width_cap
    return cfg
