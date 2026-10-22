"""Geology is built once. Climate, water, erosion, and plants then share one state."""
from .pipeline import Pipeline


class Scheduler:
    def __init__(self, init, loop, passes=1):
        self.init = Pipeline(init)
        self.loop = Pipeline(loop)
        self.passes = int(passes)
        if self.passes < 0:
            raise ValueError("passes must be nonnegative")

    def run(self, world, ctx):
        self.init.run(world, ctx)
        for _ in range(self.passes):
            ctx.time_years += float(ctx.dt_years)
            self.loop.run(world, ctx)
        return world
