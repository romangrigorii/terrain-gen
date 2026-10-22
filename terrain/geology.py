"""Rock layers sampled at the elevation that is exposed right now."""
import numpy as np


def strata_erodability(bedrock_height, reference_height, layer_thickness_m=40.0,
                       metres_per_unit=12000.0):
    """Erodability of the bed currently at the surface."""
    thickness = float(layer_thickness_m)
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("layer_thickness_m must be finite and positive")
    depth_m = (np.asarray(reference_height, np.float64)
               - np.asarray(bedrock_height, np.float64)) * float(metres_per_unit)
    phase = np.remainder(depth_m, thickness) / thickness
    weak = phase < 0.55
    return np.where(weak, np.float32(0.85), np.float32(0.15)).astype(np.float32)
