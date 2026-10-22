"""Environment settings. Extra processes turn on per scenario; the solvers stay shared."""
from ..core.parameters import ConstantField, GradientField, RadialField


def _processes(**flags):
    processes = {"glacial": False, "aeolian": False, "coastal": False, "karst": False}
    processes.update(flags)
    return processes


SCENARIOS = {
    "alpine": {
        "temperature_c": -2.0,
        "rainfall_mm_day": 3.5,
        "uplift_m_year": 0.0012,
        "hardness": 0.82,
        "preferred_c": 4.0,
        "erodibility": 6.0e-5,
        "porosity": 0.12,
        "wind_deg": 270.0,
        "processes": _processes(glacial=True),
        "summary": "Cold, hard rock, steady uplift.",
    },
    "arid": {
        "temperature_c": 26.0,
        "rainfall_mm_day": 0.35,
        "uplift_m_year": 0.00015,
        "hardness": 0.6,
        "preferred_c": 24.0,
        "erodibility": 4.0e-5,
        "porosity": 0.2,
        "wind_deg": 90.0,
        "processes": _processes(aeolian=True),
        "summary": "Hot and dry, so little water and little cover.",
    },
    "tropical": {
        "temperature_c": 25.0,
        "rainfall_mm_day": 11.0,
        "uplift_m_year": 0.0003,
        "hardness": 0.3,
        "preferred_c": 26.0,
        "erodibility": 8.0e-5,
        "porosity": 0.45,
        "wind_deg": 90.0,
        "processes": _processes(karst=True),
        "summary": "Warm, wet, and easy to incise.",
    },
    "volcanic": {
        "temperature_c": 18.0,
        "rainfall_mm_day": 4.0,
        "uplift_m_year": 0.004,
        "hardness": 0.75,
        "preferred_c": 16.0,
        "erodibility": 5.0e-5,
        "porosity": 0.15,
        "wind_deg": 90.0,
        "processes": _processes(),
        "summary": "Uplift concentrated around a vent.",
    },
    "temperate": {
        "temperature_c": 11.0,
        "rainfall_mm_day": 3.0,
        "uplift_m_year": 0.00025,
        "hardness": 0.5,
        "preferred_c": 12.0,
        "erodibility": 5.0e-5,
        "porosity": 0.25,
        "wind_deg": 270.0,
        "processes": _processes(),
        "summary": "Moderate rain, heat, and rock strength.",
    },
    "arctic": {
        "temperature_c": -18.0,
        "rainfall_mm_day": 0.8,
        "uplift_m_year": 0.0002,
        "hardness": 0.7,
        "preferred_c": -8.0,
        "erodibility": 3.0e-5,
        "porosity": 0.1,
        "wind_deg": 90.0,
        "processes": _processes(glacial=True),
        "summary": "Below freezing, with thin cover.",
    },
    "coastal": {
        "temperature_c": 14.0,
        "rainfall_mm_day": 7.0,
        "uplift_m_year": 0.0002,
        "hardness": 0.4,
        "preferred_c": 14.0,
        "erodibility": 7.0e-5,
        "porosity": 0.3,
        "wind_deg": 270.0,
        "processes": _processes(coastal=True),
        "summary": "Wetter toward the southern shore.",
    },
    "artificial": {
        "temperature_c": 15.0,
        "rainfall_mm_day": 1.5,
        "uplift_m_year": 0.0,
        "hardness": 0.95,
        "preferred_c": 15.0,
        "erodibility": 1.0e-5,
        "base_cohesion": 0.7,
        "porosity": 0.05,
        "wind_deg": 90.0,
        "processes": _processes(),
        "summary": "Stiff ground and weak incision, a stand-in for armoured land.",
    },
}


def scenario_names():
    return tuple(SCENARIOS)


def parameters_for(name, shape, cell_m, seed=0, overrides=None):
    if name not in SCENARIOS:
        known = ", ".join(SCENARIOS)
        raise ValueError(f"unknown scenario {name!r}; known: {known}")
    spec = dict(SCENARIOS[name])
    if overrides:
        spec.update(overrides)
    rows, cols = shape
    height_m = rows * float(cell_m)
    width_m = cols * float(cell_m)
    uplift_rate = float(spec["uplift_m_year"])
    rain_rate = float(spec["rainfall_mm_day"])
    uplift = ConstantField(uplift_rate)
    rain = ConstantField(rain_rate)
    if name == "volcanic":
        uplift = RadialField(
            center=(height_m * 0.5, width_m * 0.5),
            radius=0.0,
            amplitude=uplift_rate,
            falloff=0.35 * min(height_m, width_m),
        )
    if name == "coastal":
        rain = GradientField(rain_rate * 0.35, rain_rate, axis=0)
    return {
        "temperature": ConstantField(float(spec["temperature_c"])),
        "rainfall": rain,
        "uplift": uplift,
        "hardness": ConstantField(float(spec["hardness"])),
        "porosity": ConstantField(float(spec.get("porosity", 0.2))),
        "preferred_c": spec["preferred_c"],
        "erodibility": spec["erodibility"],
        "base_cohesion": spec.get("base_cohesion", 0.05),
        "wind_deg": float(spec.get("wind_deg", 90.0)),
        "processes": dict(spec["processes"]),
        "seed": int(seed),
        "summary": spec["summary"],
    }
