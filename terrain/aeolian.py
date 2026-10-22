"""Wind transport for dry ground."""
import numpy as np

from . import grid, hydrology


def wind_components(shape, mode):
    """`(vy, vx)` in cell steps. Belts flip along latitude. Y is not the wind."""
    rows, cols = shape
    if mode == "easterly":
        vx, vy = -1.0, 0.0
    elif mode == "westerly":
        vx, vy = 1.0, 0.0
    elif mode == "belts":
        latitude = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        vx = np.sign(np.sin(latitude * np.pi * 3.0)).astype(np.float32)
        vx = np.broadcast_to(vx, shape).copy()
        vy = np.zeros(shape, np.float32)
        return vy, vx
    else:
        raise ValueError("wind mode must be belts, easterly or westerly")
    return (np.full(shape, np.float32(vy)),
            np.full(shape, np.float32(vx)))


def transport(bedrock, sediment, dryness, wind_mode="belts", seed=0, iters=3,
              wrap_x=True):
    """Blow loose grains downwind where `dryness` is high."""
    del seed  # the pass is deterministic without a draw
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    bedrock = np.asarray(bedrock, np.float32)
    sediment = np.array(sediment, np.float32, copy=True)
    if bedrock.shape != sediment.shape:
        raise ValueError("bedrock and sediment must share a shape")
    dryness = np.clip(np.asarray(dryness, np.float32), 0.0, 1.0)
    if dryness.shape != sediment.shape:
        raise ValueError("dryness must match the height field")
    vy, vx = wind_components(sediment.shape, wind_mode)
    use_x = np.abs(vx) >= np.abs(vy)
    step_x = np.where(use_x, np.sign(vx), 0.0).astype(np.int32)
    step_y = np.where(~use_x, np.sign(vy), 0.0).astype(np.int32)

    def run():
        nonlocal sediment
        for _ in range(int(iters)):
            flux = np.minimum(sediment, sediment * dryness * np.float32(0.22))
            arrived = np.zeros_like(flux)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                send = (step_y == dy) & (step_x == dx)
                if not send.any():
                    continue
                moved = np.where(send, flux, np.float32(0.0))
                arrived += grid.shift(moved, -dy, -dx)
            sediment = sediment - flux + arrived
        mobile = sediment * dryness
        held = sediment - mobile
        slumped = hydrology.slump_debris(
            bedrock, mobile, repose=0.55, rate=0.45, iters=2, wrap_x=wrap_x)
        return held + slumped

    if wrap_x and grid.active_bounds() is None:
        return bedrock, run()
    bounds = grid.Bounds()
    bounds.clamp_x = True
    bounds.clamp_y = True
    with grid.use_bounds(bounds):
        return bedrock, run()
