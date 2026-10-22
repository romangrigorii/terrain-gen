"""A stage reads named fields and writes them back. The pipeline runs a list of stages."""
from dataclasses import dataclass, field
import time


@dataclass
class Context:
    seed: int
    dt_years: float
    time_years: float = 0.0
    parameters: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class Stage:
    name = "stage"
    requires = ()
    writes = ()

    def run(self, world, ctx):
        raise NotImplementedError(self.name)


class Pipeline:
    def __init__(self, stages):
        self.stages = list(stages)

    def run(self, world, ctx):
        for stage in self.stages:
            self._require(stage, world)
            started = time.perf_counter()
            stage.run(world, ctx)
            ctx.timings[stage.name] = ctx.timings.get(stage.name, 0.0) + (
                time.perf_counter() - started)
            missing = [key for key in stage.writes
                       if key not in world.fields and key not in world.features]
            if missing:
                raise KeyError(f"{stage.name} did not write {missing}")

    @staticmethod
    def _require(stage, world):
        missing = [key for key in stage.requires
                   if key not in world.fields and key not in world.features]
        if missing:
            raise KeyError(f"{stage.name} missing {missing}")
