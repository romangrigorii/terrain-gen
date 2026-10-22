"""Optional world-generation profiles. All parameters remain overridable."""
from dataclasses import replace

from .world import Config

PRESETS = {
    "earthlike": {
        "description": "Balanced oceans, continents, erosion, biomes and forests.",
        "values": {},
    },
    "archipelago": {
        "description": "More coastlines, island chains and marine terrain.",
        "values": {"land_fraction": 0.18, "cont_fraction": 0.24,
                   "cont_plate_fraction": 0.25, "n_hotspots": 16,
                   "hotspot_h": 1.22, "coast_h": 0.20},
    },
    "supercontinent": {
        "description": "A few large continents, extensive continental interiors.",
        "values": {"land_fraction": 0.44, "cont_fraction": 0.54,
                   "crust_periods": 0.8, "crust_plate_weight": 0.45,
                   "cont_plate_fraction": 0.65, "n_hotspots": 3},
    },
    "alpine": {
        "description": "Stronger mountain belts and highly dissected valleys.",
        "values": {"land_fraction": 0.36, "collision_h": 1.9,
                   "collision_w": 20.0, "cordillera_h": 0.85,
                   "erosion_passes": 6, "erosion_k": 0.012,
                   "upland_h": 0.36},
    },
    "arid": {
        "description": "Drier regions, stronger rain shadows and sparse forests.",
        "values": {"temp_offset": 5.0, "rain_recycle": 0.24,
                   "rain_belts": 0.95, "tree_mi_open": 0.70,
                   "tree_mi_closed": 1.8, "river_threshold": 0.001},
    },
    "volcanic": {
        "description": "More intraplate volcanoes, ridges and island chains.",
        "values": {"n_hotspots": 25, "hotspot_h": 1.5,
                   "hotspot_sigma": 2.6, "arc_h": 1.15,
                   "ridge_h": 0.38, "land_fraction": 0.26},
    },
}


def make_config(name="earthlike", cfg=None, **overrides):
    """Build an independent Config with preset values and explicit overrides."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
    result = replace(cfg) if cfg is not None else Config()
    for key, value in {**PRESETS[name]["values"], **overrides}.items():
        if not hasattr(result, key):
            raise TypeError(f"unknown terrain setting: {key}")
        setattr(result, key, value)
    return result
