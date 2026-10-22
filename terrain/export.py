"""Handing a world to something else: greyscale PNGs plus what they mean."""
import hashlib
import itertools
import json
import struct
import warnings
import zlib
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import scipy

from . import climate
from .world import Config

MAX = 65535

METRES_PER_UNIT = 12000.0


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _write_png(path, a, depth, level=6):
    """Greyscale, 8- or 16-bit, filter type 0 on every row - all the format needs."""
    h, w = a.shape
    body = a.astype(">u2" if depth == 16 else np.uint8).view(np.uint8).reshape(h, -1)
    raw = np.empty((h, body.shape[1] + 1), np.uint8)
    raw[:, 0] = 0                                   # filter: none
    raw[:, 1:] = body
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0)))
        f.write(_chunk(b"IDAT", zlib.compress(raw.tobytes(), level)))
        f.write(_chunk(b"IEND", b""))


def _write_rgb_png(path, a, level=6):
    """Write RGB/RGBA 8-bit PNG without a runtime image dependency."""
    a = np.asarray(a)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("expected uint8 RGB or RGBA data")
    h, w, channels = a.shape
    raw = np.empty((h, 1 + w * channels), np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a.reshape(h, -1)
    color_type = 2 if channels == 3 else 6
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(_chunk(b"IDAT", zlib.compress(raw.tobytes(), int(level))))
        f.write(_chunk(b"IEND", b""))


# Feather is in preview pixels. One preview pixel becomes about a hundred
# pixels at 64K, so a hard biome edge stays a staircase unless it is rounded
# here before the upscale.
MASK_FEATHER = 1.5
FLOW_FEATHER = 1.0


def feather_color(image, wrap_x=True, sigma=MASK_FEATHER):
    """Round stepped color edges. A flat field stays flat."""
    image = np.asarray(image, np.float32)
    sigma = float(sigma)
    if sigma <= 0 or image.ndim < 2 or min(image.shape[:2]) < 2:
        return image

    def plane(field):
        if wrap_x and field.shape[1] > 1:
            pad = max(1, int(np.ceil(sigma * 3.0)))
            wide = np.concatenate((field[:, -pad:], field, field[:, :pad]), axis=1)
            wide = scipy.ndimage.gaussian_filter(wide, sigma=(0.0, sigma), mode="nearest")
            field = wide[:, pad:-pad]
        else:
            field = scipy.ndimage.gaussian_filter(field, sigma=(0.0, sigma), mode="nearest")
        field = scipy.ndimage.gaussian_filter(field, sigma=(sigma, 0.0), mode="nearest")
        return np.asarray(field, np.float32)

    if image.ndim == 2:
        return plane(image)
    return np.stack([plane(image[..., c]) for c in range(image.shape[-1])], axis=-1)


def advanced_bundle(world, folder, include_fields=True):
    """Export normals, four material weights, wetness and vector flow."""
    from . import derived
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    n = derived.normals(world.height)
    _write_rgb_png(d / "normals.png", np.rint(np.clip(n * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
    mat = feather_color(derived.material_weights(world), wrap_x=True, sigma=MASK_FEATHER)
    _write_rgb_png(d / "materials.png", np.rint(np.clip(mat, 0, 1) * 255).astype(np.uint8))
    wet = feather_color(derived.wetness(world), wrap_x=True, sigma=FLOW_FEATHER)
    field8(np.clip(wet, 0, 1), d / "wetness.png", units="relative surface wetness")
    (d / "materials.json").write_text(json.dumps({
        "channels": ["sand", "soil_grass", "rock", "snow"],
        "encoding": "unorm8_rgba", "land_sum": 1.0,
        "sea_or_lake": [0, 0, 0, 0],
        "normal_encoding": "rgb_unorm8 = (xyz + 1) / 2; y points down image",
        "notes": "Material and wetness layers are heuristic shader inputs, not physical predictions",
    }, indent=2))
    vectors = derived.flow_directions(world)
    flow_rgb = np.zeros(vectors.shape[:2] + (3,), np.uint8)
    flow_rgb[..., 0] = np.rint(np.clip(vectors[..., 1] * 0.5 + 0.5, 0, 1) * 255)
    flow_rgb[..., 1] = np.rint(np.clip(vectors[..., 0] * 0.5 + 0.5, 0, 1) * 255)
    flow_rgb[..., 2] = np.uint8(128)
    _write_rgb_png(d / "flowdir.png", flow_rgb)
    np.savez_compressed(d / "flow_vectors.npz", vectors=vectors,
                        discharge=world.water.flow.astype(np.float32),
                        river_width=world.water.width.astype(np.float32))
    extra = {}
    if world.sediment is not None:
        sediment = np.asarray(world.sediment, np.float32)
        ceiling = float(max(sediment.max(), 1e-6))
        field8(sediment / ceiling, d / "sediment.png",
               units="sediment thickness, normalised to this map")
        heightmap(np.asarray(world.bedrock, np.float32), d / "bedrock16.png")
        hardness = np.clip(np.asarray(world.hardness, np.float32), 0.0, 1.0)
        field8(1.0 - hardness, d / "erodability.png", units="erodability")
        extra = dict(sediment=sediment, bedrock=np.asarray(world.bedrock, np.float32),
                     erodability=(1.0 - hardness).astype(np.float32))
    if include_fields:
        np.savez_compressed(d / "terrain_fields.npz",
            elevation=world.height.astype(np.float32),
            lake_depth=world.water.lake_depth.astype(np.float32),
            lake_surface=world.water.lake_level.astype(np.float32),
            runoff=world.runoff.astype(np.float32),
            temperature_c=world.temp.astype(np.float32),
            seasonal_swing_c=world.swing.astype(np.float32),
            biome=world.biome.astype(np.uint8),
            canopy=world.trees.cover.astype(np.float32),
            water_flow=world.water.flow.astype(np.float32),
            **extra)
    return d


def _sidecar(path, meta):
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def elevation_bounds(*fields):
    """One physical range for every tile of a height field."""
    if not fields:
        raise ValueError("elevation_bounds needs at least one height field")
    lo = min(float(np.min(field)) for field in fields)
    hi = max(float(np.max(field)) for field in fields)
    return lo, hi


def heightmap(h, path, lo=None, hi=None):
    """Write `h` as a 16-bit PNG, and a JSON sidecar saying what the codes mean.

    Pass `lo` and `hi` from `elevation_bounds` when `h` is one tile of a larger
    field. Per-tile extrema make the same height encode as a different code.
    """
    lo = float(np.min(h)) if lo is None else float(lo)
    hi = float(np.max(h)) if hi is None else float(hi)
    span = max(1e-9, hi - lo)
    _write_png(path, np.rint(np.clip((h - lo) / span, 0, 1) * MAX).astype(np.uint16), 16)
    return _sidecar(path, {
        "width": int(h.shape[1]), "height": int(h.shape[0]), "bits": 16,
        "lo": lo, "hi": hi, "units_per_code": span / MAX,
        "metres_per_unit": METRES_PER_UNIT,
        "sea_code": (0.0 - lo) / span * MAX,
        "wrap": "x",    # the map is a cylinder; y clamps
    })


def indexmap(a, path, names):
    """Class indices as an 8-bit PNG, with the class names beside it."""
    a = np.asarray(a)
    if a.min() < 0 or a.max() > 255:
        raise ValueError(f"indices must fit in a byte, got {a.min()}..{a.max()}")
    _write_png(path, a.astype(np.uint8), 8)
    return _sidecar(path, {
        "width": int(a.shape[1]), "height": int(a.shape[0]), "bits": 8,
        "classes": list(names), "wrap": "x",
    })


def field8(a, path, lo=0.0, hi=1.0, units="fraction"):
    """A bounded continuous field as an 8-bit PNG, with its scale beside it."""
    span = max(1e-9, hi - lo)
    _write_png(path, np.rint(np.clip((np.asarray(a) - lo) / span, 0, 1) * 255
                             ).astype(np.uint8), 8)
    return _sidecar(path, {
        "width": int(a.shape[1]), "height": int(a.shape[0]), "bits": 8,
        "lo": lo, "hi": hi, "units_per_code": span / 255, "units": units,
        "wrap": "x",
    })


def river_paths(water, path):
    """Channel centrelines as vectors, which is what they were before rasterising."""
    paths = [{"points": [[round(float(x), 3), round(float(y), 3)] for y, x in pts],
              "width": [round(float(v), 3) for v in wv]}
             for pts, wv in zip(water.polylines, water.widths)]
    Path(path).write_text(json.dumps(
        {"coords": "xy, in cells", "wrap": "x", "rivers": paths}))
    return paths


def _stamp():
    """What produced a world, beyond the knobs: the code, and what it ran on."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return f"src:{h.hexdigest()[:12]} numpy:{np.__version__} scipy:{scipy.__version__}"


def config(cfg, path):
    """The seed and every knob, written so that loading it rebuilds this world."""
    meta = dict(asdict(cfg), ref_width=cfg.width, generator=_stamp())
    Path(path).write_text(json.dumps(meta, indent=1))
    return meta


def load_config(path):
    """Read one back. JSON has no tuples, and a few knobs are pairs."""
    d = json.loads(Path(path).read_text())
    stamp, here = d.pop("generator", None), _stamp()
    if stamp is not None and stamp != here:
        warnings.warn(f"config was written by [{stamp}], this is [{here}]; "
                      "the rebuilt world may not match")
    known = {f.name for f in fields(Config)}
    if d.keys() - known:
        warnings.warn(f"ignoring unknown config keys: {sorted(d.keys() - known)}")
    return Config(**{k: tuple(v) if isinstance(v, list) else v
                     for k, v in d.items() if k in known})


def new_dir(parent, name):
    """`name`, or `name-2`, `name-3`... - the first one that does not exist yet."""
    parent = Path(parent)
    for n in itertools.count(1):
        d = parent / (name if n == 1 else f"{name}-{n}")
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            continue


def bundle(world, parent="out/export", d=None, advanced=False):
    """Everything an engine needs, in a folder of its own. Returns that folder."""
    d = Path(d) if d is not None else new_dir(parent, f"seed{world.cfg.seed}")
    heightmap(world.height, d / "height16.png")
    heightmap(world.water.lake_depth, d / "lakes16.png")
    indexmap(world.biome, d / "biome.png", climate.BIOME_NAMES)
    field8(world.trees.cover, d / "canopy.png", units="canopy fraction")
    river_paths(world.water, d / "rivers.json")
    config(world.cfg, d / "config.json")
    if advanced:
        advanced_bundle(world, d)
    return d
