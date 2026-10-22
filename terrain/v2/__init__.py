"""Shared landscape state and interchangeable process stages.

Elevation is derived from material layers. The existing generator still
builds a heightmap; `import_world` lays that result onto the shared state.
"""
from .core.pipeline import Context, Pipeline, Stage
from .core.state import TerrainState
from .generation.legacy import import_world
from .generation.world import run

__all__ = [
    "Context",
    "Pipeline",
    "Stage",
    "TerrainState",
    "import_world",
    "run",
]
