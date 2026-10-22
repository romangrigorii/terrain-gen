"""Import path for the boundary-momentum continuation."""
from terrain.boundary_momentum import (
    coast_distance, conform_shore, continue_boundary, continue_shore, momentum,
)

__all__ = [
    "continue_boundary", "continue_shore", "momentum", "coast_distance", "conform_shore",
]
