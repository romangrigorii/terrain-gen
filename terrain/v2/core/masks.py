"""Region weights. They are normalized so every cell's weights sum to one."""
import numpy as np


def normalize_weights(weights):
    stacked = np.stack([np.maximum(np.asarray(w, np.float64), 0.0) for w in weights], axis=0)
    total = stacked.sum(axis=0, keepdims=True)
    even = np.full_like(stacked, 1.0 / stacked.shape[0])
    mixed = np.where(total > 1e-12, stacked / np.maximum(total, 1e-12), even)
    return [part.astype(np.float32) for part in mixed]
