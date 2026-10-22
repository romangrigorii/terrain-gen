"""Checks that do not depend on a picture of the terrain looking right."""
import numpy as np

from .state import THICKNESS


def check_state(state):
    """Raise if a thickness, water depth, or cover fraction went negative or invalid."""
    problems = []
    for name in THICKNESS:
        array = np.asarray(state.get(name))
        if not np.isfinite(array).all():
            problems.append(f"{name} has invalid values")
        elif np.any(array < -1e-5):
            problems.append(f"{name} went negative")
    elevation = state.elevation
    if not np.isfinite(elevation).all():
        problems.append("elevation has invalid values")
    if problems:
        raise ValueError("; ".join(problems))


def thickness_budget(before_sum, after_sum, uplift_sum, exported_sum, atol=1e-2):
    """Uplift adds rock. Material that leaves the map is counted in `exported_sum`."""
    expected = before_sum + uplift_sum - exported_sum
    if abs(after_sum - expected) > atol:
        raise AssertionError(
            f"thickness budget {after_sum} != {expected} "
            f"(uplift {uplift_sum}, exported {exported_sum})")


def water_budget(before_sum, after_sum, rain_sum, atol=1e-3):
    expected = before_sum + rain_sum
    if abs(after_sum - expected) > atol:
        raise AssertionError(f"water budget {after_sum} != {expected}")


def drainage_problems(graph):
    problems = []
    if not graph.complete:
        problems.append("drainage graph has a cycle or a disconnected reach")
    for node in graph.nodes:
        if node.downstream_id is None:
            continue
        downstream = graph.nodes[node.downstream_id]
        if node.elevation_m + 1e-4 < downstream.elevation_m:
            problems.append(f"river node {node.id} flows uphill")
            break
    return problems
