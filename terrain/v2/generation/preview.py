"""A small picture of a layered landscape for the studio."""
import numpy as np

from terrain import render


def shade_state(state):
    """Hillshade the roof, then tint ice, sand, and open cavities."""
    elev = state.elevation
    span = max(float(elev.max() - elev.min()), 1e-6)
    norm = ((elev - float(elev.min())) / span).astype(np.float32)
    rgb = render.relief(norm).astype(np.float64)
    ice = np.clip(state.get("ice_m").astype(np.float64) / 8.0, 0.0, 1.0)[..., None]
    sand = np.clip(state.get("sand_m").astype(np.float64) / 6.0, 0.0, 1.0)[..., None]
    void = np.clip(state.get("void_m").astype(np.float64) / 4.0, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - 0.45 * ice) + np.array([210.0, 225.0, 235.0]) * (0.45 * ice)
    rgb = rgb * (1.0 - 0.35 * sand) + np.array([214.0, 184.0, 120.0]) * (0.35 * sand)
    rgb = rgb * (1.0 - 0.35 * void)
    return np.clip(rgb, 0, 255).astype(np.uint8)
