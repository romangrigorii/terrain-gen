"""Edit a scenario's numbers and which processes run, then build a landscape from that."""
import numpy as np

from .scenarios import SCENARIOS, parameters_for, scenario_names

EDITABLE = ("temperature_c", "rainfall_mm_day", "uplift_m_year", "hardness")
PROCESSES = ("glacial", "aeolian", "coastal", "karst")
LIMITS = {
    "temperature_c": (-40.0, 45.0),
    "rainfall_mm_day": (0.0, 30.0),
    "uplift_m_year": (0.0, 0.01),
    "hardness": (0.0, 1.0),
}


class ScenarioEditor:
    def __init__(self, name="temperate"):
        self.name = "temperate"
        self.values = {}
        self.processes = {}
        self.load(name)

    def load(self, name):
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}")
        spec = SCENARIOS[name]
        self.name = name
        self.values = {key: float(spec[key]) for key in EDITABLE}
        self.processes = dict(spec["processes"])

    def cycle(self, step=1):
        names = scenario_names()
        self.load(names[(names.index(self.name) + int(step)) % len(names)])

    def toggle(self, process):
        if process not in PROCESSES:
            raise ValueError(f"unknown process {process!r}")
        self.processes[process] = not self.processes[process]

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        if key not in LIMITS:
            raise ValueError(f"unknown setting {key!r}")
        lo, hi = LIMITS[key]
        self.values[key] = float(np.clip(value, lo, hi))

    def summary(self):
        return SCENARIOS[self.name]["summary"]

    def overrides(self):
        return {**self.values, "processes": dict(self.processes)}

    def parameters(self, shape, cell_m, seed=0):
        return parameters_for(self.name, shape, cell_m, seed, self.overrides())

    def run_preview(self, seed=0, shape=(48, 64), cell_m=90.0, passes=2, dt_years=80.0):
        from .world import run
        return run(self.name, shape=shape, cell_m=cell_m, seed=seed, passes=passes,
                   dt_years=dt_years, overrides=self.overrides())
