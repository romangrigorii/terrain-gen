"""Raindrop erosion. A drop picks up loose grains and carries them downhill."""
import numpy as np


def rain_splash(bedrock, sediment, erodability, seed=0, steps=10, wrap_x=True):
    """Move loose sediment a few cells downhill. Bedrock stays put."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    bedrock = np.asarray(bedrock, np.float32)
    sediment = np.array(sediment, np.float32, copy=True)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")
    soft = np.clip(np.asarray(erodability, np.float32), 0.0, 1.0)
    if soft.shape != sediment.shape:
        raise ValueError("erodability must match the height field")
    rows, cols = sediment.shape
    count = int(min(2500, max(48, (rows * cols) // 48)))
    count = min(count, rows * cols)
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 17])
    chosen = rng.choice(rows * cols, size=count, replace=False)
    y, x = np.divmod(chosen, cols)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    load = np.zeros(count, np.float32)
    flat = sediment.ravel()
    for _ in range(int(steps)):
        surface = bedrock + sediment
        here = surface[y, x]
        best = np.zeros(count, np.float32)
        ny = y.copy()
        nx = x.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = np.clip(y + dy, 0, rows - 1)
            if wrap_x:
                xx = (x + dx) % cols
            else:
                xx = np.clip(x + dx, 0, cols - 1)
            drop = here - surface[yy, xx]
            take_step = drop > best
            best = np.where(take_step, drop, best)
            ny = np.where(take_step, yy, ny)
            nx = np.where(take_step, xx, nx)
        moving = best > 0.0
        room = np.float32(0.02) * best * soft[y, x]
        order = y * cols + x
        held = flat[order]
        take = np.minimum(held, room)
        take = np.where(moving, take, np.float32(0.0)).astype(np.float32)
        requested = np.zeros(rows * cols, np.float32)
        np.add.at(requested, order, take)
        over = requested[order] > held + np.float32(1e-6)
        take = np.where(
            over, take * held / np.maximum(requested[order], np.float32(1e-6)),
            take).astype(np.float32)
        np.add.at(flat, order, -take)
        load = load + take
        settle = np.where(moving, np.minimum(load, load * np.float32(0.2)), load)
        np.add.at(flat, y * cols + x, settle)
        load = load - settle
        y = np.where(moving, ny, y).astype(np.int32)
        x = np.where(moving, nx, x).astype(np.int32)
        sediment = flat.reshape(rows, cols)
        flat = sediment.ravel()
    np.add.at(flat, y * cols + x, load)
    return bedrock, flat.reshape(rows, cols)
