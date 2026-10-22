"""Sculpt, paint, and decorate an exported map without regenerating it.

The asset studio writes a height image, ground weights, and optional tiles.
This module opens that folder, applies brushes to the rasters already there,
and saves them back. The first save copies the untouched files to
`authored/original`.

Paint keeps one weight per surface. Weights in a cell sum to 1, and a stroke
blends toward the chosen texture. Water and ice are surfaces in that stack.
A pond dug with the water tool stays a separate mask on top.
"""
import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .climate import BIOME_NAMES
from .export import METRES_PER_UNIT, _write_png, heightmap
from .render import LEGEND_COLORS

# name, label, group, color, grain.
# Grain is stipple, speck, crack, strata, pebble, ripple, crystal, flow.
# The first four names stay sand, soil, rock, snow so older four-channel files still line up.
_SURFACES = (
    ("sand", "sand", "dry", (214, 186, 122), 1),
    ("soil", "grass", "soil", (78, 124, 62), 0),
    ("rock", "rock", "rock", (132, 126, 116), 3),
    ("snow", "snow", "cold", (236, 240, 244), 6),
    ("meadow", "meadow", "soil", (168, 184, 78), 0),
    ("forest", "forest", "soil", (42, 86, 46), 0),
    ("dirt", "dirt", "soil", (126, 88, 54), 1),
    ("clay", "clay", "soil", (158, 102, 74), 2),
    ("peat", "peat", "soil", (64, 48, 34), 1),
    ("moss", "moss", "soil", (74, 148, 72), 0),
    ("dune", "dune", "dry", (228, 198, 138), 1),
    ("gravel", "gravel", "dry", (148, 142, 132), 4),
    ("salt", "salt", "dry", (232, 228, 216), 1),
    ("hardpan", "hardpan", "dry", (178, 142, 96), 2),
    ("cliff", "cliff", "rock", (98, 96, 94), 3),
    ("cobble", "cobble", "rock", (118, 114, 108), 4),
    ("scree", "scree", "rock", (164, 160, 152), 4),
    ("lichen", "lichen", "rock", (126, 138, 100), 0),
    ("firn", "firn", "cold", (210, 222, 230), 6),
    ("glacier", "glacier", "cold", (146, 188, 214), 6),
    ("seaice", "sea ice", "cold", (188, 208, 216), 6),
    ("frost", "frost", "cold", (214, 226, 232), 6),
    ("mud", "mud", "wet", (98, 74, 50), 7),
    ("marsh", "marsh", "wet", (88, 108, 66), 0),
    ("riverbed", "riverbed", "wet", (96, 102, 94), 4),
    ("wetsand", "wet sand", "wet", (176, 156, 108), 5),
    ("shallows", "shallows", "water", (72, 176, 178), 5),
    ("deep", "deep", "water", (30, 68, 118), 5),
    ("river", "river", "water", (52, 118, 162), 7),
    ("swamp", "swamp", "water", (64, 88, 66), 5),
    ("foam", "foam", "water", (226, 232, 234), 5),
    ("ash", "ash", "fire", (102, 98, 94), 1),
    ("lava", "lava", "fire", (186, 72, 28), 2),
    ("basalt", "basalt", "fire", (44, 46, 50), 3),
)
MATERIALS = tuple(item[0] for item in _SURFACES)
MATERIAL_LABELS = {item[0]: item[1] for item in _SURFACES}
MATERIAL_GROUP = {item[0]: item[2] for item in _SURFACES}
MATERIAL_RGB = np.array([item[3] for item in _SURFACES], np.float32)
MATERIAL_PATTERN = tuple(item[4] for item in _SURFACES)
_MATERIAL_INDEX = {name: index for index, name in enumerate(MATERIALS)}
GROUPS = (
    ("soil", "Soil"),
    ("dry", "Dry"),
    ("rock", "Rock"),
    ("cold", "Cold"),
    ("wet", "Wet"),
    ("water", "Water"),
    ("fire", "Volc"),
)
_PATTERN_MEMBERS = tuple(
    tuple(index for index, pattern in enumerate(MATERIAL_PATTERN) if pattern == pat)
    for pat in range(8)
)


def _ids(*names):
    return tuple(_MATERIAL_INDEX[name] for name in names)


_SANDS = _ids("sand", "dune", "gravel", "salt", "hardpan", "wetsand")
_SOILS = _ids("soil", "meadow", "forest", "dirt", "clay", "peat", "moss", "mud", "marsh", "lichen")
_ROCKS = _ids("rock", "cliff", "cobble", "scree", "riverbed", "basalt", "ash", "lava")
_SNOWS = _ids("snow", "firn", "glacier", "seaice", "frost")
_WATERS = _ids("shallows", "deep", "river", "swamp", "foam")


def _mix(**weights):
    vec = np.zeros(len(MATERIALS), np.float32)
    for name, value in weights.items():
        vec[_MATERIAL_INDEX[name]] = value
    return vec


def _biome_table():
    """Twelve biome rows. The first four columns match the original sand, grass, rock, snow mix."""
    base = np.array([
        (0, 0, 0, 0),
        (0.02, 0.05, 0.13, 0.80),
        (0.08, 0.42, 0.22, 0.28),
        (0.04, 0.74, 0.18, 0.04),
        (0.82, 0.06, 0.12, 0),
        (0.22, 0.62, 0.16, 0),
        (0.06, 0.78, 0.16, 0),
        (0.04, 0.80, 0.16, 0),
        (0.28, 0.48, 0.24, 0),
        (0.34, 0.52, 0.14, 0),
        (0.08, 0.76, 0.16, 0),
        (0.04, 0.82, 0.14, 0),
    ], np.float32)
    full = np.zeros((base.shape[0], len(MATERIALS)), np.float32)
    full[:, :4] = base
    full[1, _MATERIAL_INDEX["glacier"]] = 0.28
    full[1, _MATERIAL_INDEX["firn"]] = 0.16
    full[1, _MATERIAL_INDEX["seaice"]] = 0.10
    full[2, _MATERIAL_INDEX["frost"]] = 0.16
    full[2, _MATERIAL_INDEX["moss"]] = 0.10
    full[3, _MATERIAL_INDEX["moss"]] = 0.12
    full[3, _MATERIAL_INDEX["forest"]] = 0.16
    return full


PROPS = ("tree", "pine", "rock", "shrub", "ruin", "hut")
PROP_COLOR = {
    "tree": (46, 110, 62),
    "pine": (28, 86, 72),
    "rock": (148, 142, 132),
    "shrub": (110, 140, 64),
    "ruin": (158, 148, 132),
    "hut": (132, 96, 64),
}
TOOLS = ("raise", "lower", "smooth", "flatten", "grain", "path", "water",
         "dry", "paint", "place", "erase")
PATH_MIX = _mix(sand=0.72, soil=0.06, rock=0.22)
SHORE_MIX = _mix(sand=0.80, soil=0.05, rock=0.15)
_BIOME_WEIGHTS = _biome_table()
_TILE_NAME = re.compile(r"y(\d+)_x(\d+)\.png$", re.IGNORECASE)
_MAX_CELLS = 8192 * 8192
_UNDO_LIMIT = 40
_PACK_CELLS = 1_500_000
_WORK_FILES = ("height.npz", "water.png", "props.json", "preview.png", "edit.json", "surfaces.json")
_LIBRARY_FILES = (
    "zemia_height16.png", "zemia_height16.json", "zemia_height.png",
    "material_mask.png", "preview.png",
)


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    path: Path

    @property
    def label(self):
        return f"y {self.y0}    x {self.x0}"


@dataclass(frozen=True)
class ExportEntry:
    folder: Path
    kind: str
    label: str
    mtime: float
    tiles: tuple


class TileRequired(Exception):
    """The export is stored as tiles. The caller picks one before editing."""

    def __init__(self, entry):
        super().__init__(f"{entry.folder.name} is split into {len(entry.tiles)} tiles")
        self.entry = entry


class _Stroke:
    def __init__(self, tool, radius, amount, material, prop, anchor, salt, erase=False):
        self.tool = tool
        self.radius = radius
        self.amount = amount
        self.material = material
        self.prop = prop
        self.anchor = anchor
        self.salt = salt
        self.erase = bool(erase)


class _Capture:
    def __init__(self, props):
        self.before_props = props
        self.rect = None
        self.before_h = None
        self.before_m = None
        self.before_w = None

    def include(self, doc, y0, y1, x0, x1):
        """Remember the original cells before this dab changes them."""
        if y1 <= y0 or x1 <= x0:
            return
        if self.rect is None:
            self.rect = (y0, y1, x0, x1)
            self.before_h = doc.height[y0:y1, x0:x1].copy()
            self.before_m = doc.materials[y0:y1, x0:x1].copy()
            self.before_w = doc.water[y0:y1, x0:x1].copy()
            return
        oy0, oy1, ox0, ox1 = self.rect
        ny0, ny1 = min(oy0, y0), max(oy1, y1)
        nx0, nx1 = min(ox0, x0), max(ox1, x1)
        if (ny0, ny1, nx0, nx1) == self.rect:
            return
        nh = doc.height[ny0:ny1, nx0:nx1].copy()
        nm = doc.materials[ny0:ny1, nx0:nx1].copy()
        nw = doc.water[ny0:ny1, nx0:nx1].copy()
        sy, sx = oy0 - ny0, ox0 - nx0
        hh, hw = self.before_h.shape
        nh[sy:sy + hh, sx:sx + hw] = self.before_h
        nm[sy:sy + hh, sx:sx + hw] = self.before_m
        nw[sy:sy + hh, sx:sx + hw] = self.before_w
        self.before_h, self.before_m, self.before_w = nh, nm, nw
        self.rect = (ny0, ny1, nx0, nx1)


class _Frame:
    def __init__(self, rect, before_h, before_m, before_w, before_props,
                 after_h, after_m, after_w, after_props, prev, serial):
        self.rect = rect
        self.before_h = before_h
        self.before_m = before_m
        self.before_w = before_w
        self.before_props = before_props
        self.after_h = after_h
        self.after_m = after_m
        self.after_w = after_w
        self.after_props = after_props
        self.prev = prev
        self.serial = serial


def assets_root():
    """The studio's `out` folder, or `TERRAIN_ASSETS_ROOT` when that is set."""
    env = os.environ.get("TERRAIN_ASSETS_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "out"


def classify(folder):
    """How an export folder stores its height, or None when it is not an export."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    manifest = folder / "manifest.json"
    tiles = folder / "height"
    if tiles.is_dir() and manifest.is_file():
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        if "tile" in data or any(tiles.glob("y*_x*.png")):
            return "tiled"
    if (folder / "height16.png").is_file() and manifest.is_file():
        return "hires"
    if (folder / "height16.png").is_file():
        return "bundle"
    if (folder / "zemia_height16.png").is_file():
        return "library"
    return None


def list_tiles(folder):
    tiles = []
    for path in sorted((Path(folder) / "height").glob("y*_x*.png")):
        match = _TILE_NAME.match(path.name)
        if match:
            tiles.append(Tile(int(match.group(1)), int(match.group(2)), path))
    return tiles


def _label(folder, kind, root):
    text = folder.name
    if root is not None:
        try:
            text = folder.resolve().relative_to(Path(root).resolve()).as_posix()
        except (OSError, ValueError):
            text = folder.name
    return f"{text}  ({kind})"


def _mtime(folder, kind):
    if kind == "tiled":
        path = folder / "manifest.json"
    elif kind == "library":
        path = folder / "zemia_height16.png"
    else:
        path = folder / "height16.png"
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def inspect(folder, root=None):
    folder = Path(folder)
    kind = classify(folder)
    if kind is None:
        raise ValueError(f"{folder} is not an exported terrain folder.")
    tiles = tuple(list_tiles(folder)) if kind == "tiled" else ()
    return ExportEntry(folder, kind, _label(folder, kind, root), _mtime(folder, kind), tiles)


def discover(root):
    """Export folders under the studio output tree, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    folders = []
    if classify(root):
        folders.append(root)
    for pattern in ("export/*", "library/terrain", "library/terrain/hires/*"):
        for path in root.glob(pattern):
            if path.is_dir() and classify(path):
                folders.append(path)
    seen = set()
    entries = []
    for folder in folders:
        try:
            key = str(folder.resolve())
        except OSError:
            key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        entries.append(inspect(folder, root))
    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    return entries


def read_u16(path):
    """16-bit greyscale PNG as numeric samples, whatever endian the file uses."""
    image = Image.open(path)
    if image.mode == "I;16B":
        raw = np.frombuffer(image.tobytes(), dtype=">u2")
        return raw.reshape(image.size[1], image.size[0]).astype(np.uint16)
    if image.mode == "I;16L":
        raw = np.frombuffer(image.tobytes(), dtype="<u2")
        return np.array(raw.reshape(image.size[1], image.size[0]))
    arr = np.array(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{Path(path).name} is not a greyscale height image")
    return arr.astype(np.uint16, copy=False)


def read_rgba(path):
    image = Image.open(path)
    if image.mode == "RGB":
        arr = np.array(image)
        zeros = np.zeros(arr.shape[:2] + (1,), np.uint8)
        return np.concatenate((arr, zeros), axis=-1)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return np.array(image)[..., :4]


def _decode_height(path, meta):
    codes = read_u16(path)
    lo = float(meta["lo"])
    hi = float(meta["hi"])
    if "units_per_code" in meta:
        step = float(meta["units_per_code"])
    else:
        step = max(hi - lo, 1e-9) / 65535.0
    height = lo + codes.astype(np.float32) * np.float32(step)
    return height, lo, hi


def _normalize(raw):
    raw = np.asarray(raw, np.float32)
    total = raw.sum(axis=-1, keepdims=True)
    out = np.zeros_like(raw)
    np.divide(raw, np.maximum(total, 1e-8), out=out, where=total > 1e-4)
    return out


def expand_channels(raw):
    """Place a four-ground image into the full surface stack. A full stack passes through."""
    raw = np.asarray(raw, np.float32)
    count = raw.shape[-1]
    if count == len(MATERIALS):
        return raw
    if count != 4:
        raise ValueError(f"ground image has {count} channels")
    out = np.zeros(raw.shape[:-1] + (len(MATERIALS),), np.float32)
    out[..., :4] = raw
    return out


def materials_from_rgba(image):
    raw = np.asarray(image, np.float32) / np.float32(255)
    if raw.shape[-1] not in (4, len(MATERIALS)):
        raw = raw[..., :4]
    return expand_channels(_normalize(raw))


def materials_from_legacy(mask):
    """Game RGBA is rock, ground, cover, accent. Fold that into sand, grass, rock, and snow."""
    mask = np.asarray(mask, np.float32) / np.float32(255)
    raw = np.stack((
        mask[..., 3],
        np.clip(mask[..., 1] * 0.85 + mask[..., 2] * 0.15, 0, 1),
        mask[..., 0],
        np.zeros(mask.shape[:2], np.float32),
    ), axis=-1)
    return expand_channels(_normalize(raw))


def materials_from_named(weights, names):
    """Scatter a saved stack onto the current palette by surface name."""
    weights = np.asarray(weights, np.float32)
    out = np.zeros(weights.shape[:2] + (len(MATERIALS),), np.float32)
    for index, name in enumerate(names):
        slot = _MATERIAL_INDEX.get(str(name))
        if slot is not None and index < weights.shape[-1]:
            out[..., slot] += weights[..., index]
    return _normalize(out)


def materials_from_biome(biome):
    idx = np.clip(np.asarray(biome), 0, len(_BIOME_WEIGHTS) - 1)
    return _normalize(_BIOME_WEIGHTS[idx])


def materials_from_height(height):
    raw = np.zeros(height.shape + (len(MATERIALS),), np.float32)
    raw[..., 1] = np.float32(height >= 0)
    return raw


def _gather(materials, indices):
    if not indices:
        return np.zeros(materials.shape[:2], np.float32)
    return np.asarray(materials, np.float32)[..., list(indices)].sum(axis=-1)


def project_four(materials):
    """Fold every surface into the sand, grass, rock, and snow image the bundle already uses."""
    materials = expand_channels(materials)
    raw = np.stack((
        _gather(materials, _SANDS),
        _gather(materials, _SOILS),
        _gather(materials, _ROCKS),
        _gather(materials, _SNOWS),
    ), axis=-1)
    return _normalize(raw)


def legacy_mask(materials):
    """Project the surface stack back to the game's rock, ground, cover, accent."""
    materials = expand_channels(materials)
    sand = _gather(materials, _SANDS)
    soil = _gather(materials, _SOILS)
    rock = _gather(materials, _ROCKS)
    snow = _gather(materials, _SNOWS)
    rgba = np.stack((
        rock,
        np.clip(soil + 0.35 * sand, 0, 1),
        np.clip(soil * 0.65 + snow, 0, 1),
        sand,
    ), axis=-1)
    return np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _pack_materials(materials, cells):
    """Wide maps keep the stack in float16. A 4K map of every surface would not fit in float32."""
    dtype = np.float16 if cells >= _PACK_CELLS else np.float32
    return np.ascontiguousarray(materials, dtype)


def _hash2(y, x, salt):
    n = (y.astype(np.uint32) * np.uint32(374761393)
         + x.astype(np.uint32) * np.uint32(668265263)
         + np.uint32(salt))
    n ^= n >> 13
    n *= np.uint32(1274126177)
    return (n & np.uint32(0xFFFFFF)).astype(np.float32) / np.float32(16777215.0)


def _patterns(y0, y1, x0, x1):
    """Cheap grain so a blend of two surfaces does not read as one flat color."""
    yy = np.arange(y0, y1, dtype=np.int32)[:, None]
    xx = np.arange(x0, x1, dtype=np.int32)[None, :]
    fine = _hash2(yy, xx, 1)
    coarse = _hash2(yy // 3, xx // 3, 2)
    diag = _hash2((yy + xx) // 2, np.abs(yy // 2 - xx // 4), 3)
    bands = ((yy % 5) == 0).astype(np.float32)
    phase = xx.astype(np.float32) + yy.astype(np.float32) * np.float32(0.35)
    rip = 0.5 + 0.5 * np.sin(phase * np.float32(0.9))
    flow = 0.5 + 0.5 * np.sin(xx.astype(np.float32) * np.float32(0.55))
    crystal = ((yy // 2 + xx // 3) & 1).astype(np.float32)
    out = np.empty((y1 - y0, x1 - x0, 8), np.float32)
    out[..., 0] = 0.90 + 0.16 * fine
    out[..., 1] = 0.86 + 0.22 * (coarse > 0.72).astype(np.float32)
    out[..., 2] = 0.82 + 0.28 * (diag > 0.84).astype(np.float32)
    out[..., 3] = 0.88 + 0.16 * bands
    pebbles = ((coarse > 0.55) & (fine > 0.35)).astype(np.float32)
    out[..., 4] = 0.84 + 0.22 * pebbles
    out[..., 5] = 0.90 + 0.12 * rip
    out[..., 6] = 0.88 + 0.16 * crystal
    out[..., 7] = 0.88 + 0.14 * flow
    return out


def _accumulate_color(color, mats, y0, x0):
    height, width = mats.shape[:2]
    patterns = _patterns(y0, y0 + height, x0, x0 + width)
    for pat, members in enumerate(_PATTERN_MEMBERS):
        if not members:
            continue
        idx = np.asarray(members, np.intp)
        if float(mats[..., idx].max()) < 0.01:
            continue
        mixed = mats[..., idx] @ MATERIAL_RGB[idx]
        color += mixed * patterns[..., pat:pat + 1]


def _ground_color(materials, y0, x0):
    mats = np.asarray(materials, np.float32)
    height, width = mats.shape[:2]
    color = np.zeros((height, width, 3), np.float32)
    if height == 0 or width == 0:
        return color
    rows = max(1, min(height, 400_000 // max(width, 1)))
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        _accumulate_color(color[start:stop], mats[start:stop], y0 + start, x0)
    return color


def _authored_materials(data, shape):
    if "materials" not in data.files:
        return None
    mats = data["materials"]
    if getattr(mats, "ndim", 0) != 3 or mats.shape[:2] != shape:
        return None
    channels = int(mats.shape[-1])
    if "channels" in data.files:
        names = [str(item) for item in np.array(data["channels"]).reshape(-1).tolist()]
        if len(names) == channels:
            return materials_from_named(mats, names)
    if channels in (4, len(MATERIALS)):
        return expand_channels(np.array(mats, np.float32))
    return None


def _bounds(shape, cy, cx, radius):
    height, width = shape
    reach = int(math.ceil(radius))
    y0 = max(0, int(math.floor(cy)) - reach)
    y1 = min(height, int(math.ceil(cy)) + reach + 1)
    x0 = max(0, int(math.floor(cx)) - reach)
    x1 = min(width, int(math.ceil(cx)) + reach + 1)
    return y0, y1, x0, x1


def _falloff(y0, y1, x0, x1, cy, cx, radius):
    yy = np.arange(y0, y1, dtype=np.float32)[:, None]
    xx = np.arange(x0, x1, dtype=np.float32)[None, :]
    dist = np.hypot(yy - np.float32(cy), xx - np.float32(cx))
    t = np.clip(1.0 - dist / np.float32(radius), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _box(values):
    if min(values.shape) < 1:
        return values
    padded = np.pad(values, 1, mode="edge")
    return (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / np.float32(9)


def _grain(shape, cy, cx, salt):
    seed = (int(round(cy * 8)) * 73856093
            ^ int(round(cx * 8)) * 19349663
            ^ int(salt) * 83492791) & 0xFFFFFFFF
    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return _box(noise)


def _stride(image, limit):
    image = np.asarray(image)
    step = max(1, int(math.ceil(max(image.shape[:2]) / max(1, limit))))
    return np.ascontiguousarray(image[::step, ::step]), step


def _stamp_props(image, props, step):
    out = np.array(image, copy=True)
    height, width = out.shape[:2]
    for prop in props:
        color = PROP_COLOR.get(prop["kind"], (240, 240, 240))
        y = int(round(float(prop["y"]) / step))
        x = int(round(float(prop["x"]) / step))
        rad = max(1, int(round(float(prop["scale"]))))
        y0, y1 = max(0, y - rad), min(height, y + rad + 1)
        x0, x1 = max(0, x - rad), min(width, x + rad + 1)
        out[y0:y1, x0:x1] = color
    return out


def _find_library(folder):
    lineage = [folder, *list(folder.parents)[:6]]
    for parent in lineage:
        if parent.name == "terrain" and parent.parent.name == "library":
            return parent
        candidate = parent / "library" / "terrain"
        if candidate.is_dir():
            return candidate
    return None


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _prop_record(prop):
    return {
        "id": int(prop["id"]),
        "kind": prop["kind"],
        "x": round(float(prop["x"]), 3),
        "y": round(float(prop["y"]), 3),
        "scale": round(float(prop["scale"]), 3),
        "rotation": round(float(prop["rotation"]), 4),
    }


class Document:
    """One editable export: height, four ground weights, pond mask, and props."""

    def __init__(self):
        self.folder = None
        self.kind = ""
        self.tile = None
        self.tile_path = None
        self.height_path = None
        self.height = None
        self.materials = None
        self.water = None
        self.biome = None
        self.props = []
        self.next_id = 1
        self.lo = 0.0
        self.hi = 1.0
        self.metres_per_unit = METRES_PER_UNIT
        self.cell_m = 1.0
        self.cell_exact = False
        self.full_shape = (0, 0)
        self.library_dir = None
        self.notes = []
        self.rng = np.random.default_rng(0)
        self.view_lo = 0.0
        self.view_hi = 1.0
        self.rgb = None
        self.rgb_mode = None
        self.view_token = 0
        self.serial = 0
        self.saved_serial = 0
        self._clock = 0
        self._salt = 0
        self._stroke = None
        self._capture = None
        self._undo = []
        self._redo = []
        self.clipped_on_save = False
        self.library_synced = False

    @property
    def stroking(self):
        return self._stroke is not None

    @property
    def dirty(self):
        if self.serial != self.saved_serial:
            return True
        return self._capture is not None and self._capture.rect is not None

    @property
    def can_undo(self):
        return bool(self._undo) and self._stroke is None

    @property
    def can_redo(self):
        return bool(self._redo) and self._stroke is None

    @property
    def has_original(self):
        original = self.folder / "authored" / "original"
        return original.is_dir() and any(original.rglob("*"))

    @property
    def summary(self):
        height, width = self.height.shape
        text = f"{self.kind}  {width}×{height}  {self.cell_m:.0f} m/cell"
        if not self.cell_exact:
            text += ", estimated"
        if self.tile is not None:
            text += f", tile y {self.tile[0]} x {self.tile[1]}"
        if self.notes:
            text += f". {self.notes[0]}"
        return text

    def _sample(self, y, x):
        height, width = self.height.shape
        iy = int(np.clip(round(float(y)), 0, height - 1))
        ix = int(np.clip(round(float(x)), 0, width - 1))
        return float(self.height[iy, ix])

    def describe_cell(self, y, x):
        height, width = self.height.shape
        if not (math.isfinite(y) and math.isfinite(x) and 0 <= y < height and 0 <= x < width):
            return "outside the map"
        iy, ix = int(y), int(x)
        metres = float(self.height[iy, ix]) * self.metres_per_unit
        if metres > 0.5:
            text = f"+{metres:,.0f} m"
        elif metres < -0.5:
            text = f"{metres:,.0f} m"
        else:
            text = "sea level"
        weights = np.asarray(self.materials[iy, ix], np.float32)
        if float(weights.sum()) >= 0.2:
            name = MATERIALS[int(np.argmax(weights))]
            if self.water[iy, ix] > 0.35 and MATERIAL_GROUP[name] != "water":
                ground = "water"
            else:
                ground = MATERIAL_LABELS[name]
        elif self.water[iy, ix] > 0.35 or self.height[iy, ix] < 0:
            ground = "water"
        else:
            ground = "open"
        return f"{text}  {ground}"

    def _measure(self):
        lo = float(self.height.min())
        hi = float(self.height.max())
        self.view_lo = min(0.0, lo)
        self.view_hi = max(lo + 1e-4, hi, 0.05)

    def begin_stroke(self, tool, y, x, radius, amount, material="sand", prop="tree", erase=False):
        if self._stroke is not None:
            self.end_stroke()
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool}")
        radius = float(radius)
        amount = float(amount)
        if not math.isfinite(radius) or radius < 0.5:
            raise ValueError("brush radius must be at least half a cell")
        if not math.isfinite(amount) or not math.isfinite(y) or not math.isfinite(x):
            raise ValueError("brush values must be finite")
        if tool == "paint" and material not in MATERIALS:
            raise ValueError(f"unknown ground {material}")
        if tool == "place" and prop not in PROPS:
            raise ValueError(f"object must be one of {', '.join(PROPS)}")
        self._salt += 1
        self._stroke = _Stroke(tool, radius, amount, material, prop,
                               self._sample(y, x), self._salt, erase)
        self._capture = _Capture(copy.deepcopy(self.props))

    def dab(self, y, x):
        if self._stroke is None:
            raise RuntimeError("begin a stroke before using the brush")
        y, x = float(y), float(x)
        if not math.isfinite(y) or not math.isfinite(x):
            return None
        y0, y1, x0, x1 = _bounds(self.height.shape, y, x, self._stroke.radius)
        if y1 <= y0 or x1 <= x0:
            return None
        self._capture.include(self, y0, y1, x0, x1)
        self._apply(y0, y1, x0, x1, y, x)
        window = self.height[y0:y1, x0:x1]
        self.view_lo = min(self.view_lo, float(window.min()))
        self.view_hi = max(self.view_hi, float(window.max()))
        self.render(self.rgb_mode or "shaded", (y0, y1, x0, x1))
        return y0, y1, x0, x1

    def end_stroke(self):
        stroke, capture = self._stroke, self._capture
        self._stroke = None
        self._capture = None
        if stroke is None or capture is None or capture.rect is None:
            return False
        y0, y1, x0, x1 = capture.rect
        after_h = self.height[y0:y1, x0:x1].copy()
        after_m = self.materials[y0:y1, x0:x1].copy()
        after_w = self.water[y0:y1, x0:x1].copy()
        after_props = copy.deepcopy(self.props)
        same = (np.array_equal(capture.before_h, after_h)
                and np.allclose(capture.before_m, after_m, atol=1e-6)
                and np.allclose(capture.before_w, after_w, atol=1e-6)
                and capture.before_props == after_props)
        self._measure()
        if self.rgb_mode == "height":
            self.render("height")
        if same:
            return False
        self._redo.clear()
        prev = self.serial
        self._clock += 1
        self.serial = self._clock
        self._undo.append(_Frame(
            capture.rect, capture.before_h, capture.before_m, capture.before_w,
            capture.before_props, after_h, after_m, after_w, after_props,
            prev, self.serial))
        if len(self._undo) > _UNDO_LIMIT:
            self._undo.pop(0)
        return True

    def undo(self):
        if not self.can_undo:
            return False
        frame = self._undo.pop()
        self._blit(frame, before=True)
        self._redo.append(frame)
        self.serial = frame.prev
        return True

    def redo(self):
        if not self.can_redo:
            return False
        frame = self._redo.pop()
        self._blit(frame, before=False)
        self._undo.append(frame)
        self.serial = frame.serial
        return True

    def _blit(self, frame, before):
        y0, y1, x0, x1 = frame.rect
        self.height[y0:y1, x0:x1] = frame.before_h if before else frame.after_h
        self.materials[y0:y1, x0:x1] = frame.before_m if before else frame.after_m
        self.water[y0:y1, x0:x1] = frame.before_w if before else frame.after_w
        self.props = copy.deepcopy(frame.before_props if before else frame.after_props)
        self.next_id = 1 + max((prop["id"] for prop in self.props), default=0)
        self._measure()
        if self.rgb_mode == "height":
            self.render("height")
        else:
            self.render(self.rgb_mode or "shaded", frame.rect)

    def _write_height(self, y0, y1, x0, x1, values):
        if not np.isfinite(values).all():
            raise RuntimeError("brush produced a non-finite height")
        np.clip(values, -2.0, 6.0, out=values)
        self.height[y0:y1, x0:x1] = values

    def _fade_water(self, y0, y1, x0, x1, fall, factor):
        self.water[y0:y1, x0:x1] = np.clip(
            self.water[y0:y1, x0:x1] * (1.0 - fall * np.float32(factor)), 0, 1)

    def _blend(self, y0, y1, x0, x1, weight, target):
        weight = np.clip(weight, 0, 1)[..., None].astype(np.float32)
        mixed = self.materials[y0:y1, x0:x1] * (1.0 - weight) + target * weight
        total = mixed.sum(axis=-1, keepdims=True)
        mixed = np.where(total > 1e-4, mixed / np.maximum(total, 1e-8), 0)
        self.materials[y0:y1, x0:x1] = mixed.astype(np.float32)

    def _apply(self, y0, y1, x0, x1, cy, cx):
        stroke = self._stroke
        fall = _falloff(y0, y1, x0, x1, cy, cx, stroke.radius)
        patch = self.height[y0:y1, x0:x1]
        tool = stroke.tool
        if tool in ("raise", "lower", "grain"):
            metres = float(np.clip(stroke.amount, 0, 5000)) / self.metres_per_unit
            if tool == "lower":
                delta = -metres * fall
            elif tool == "grain":
                delta = _grain(fall.shape, cy, cx, stroke.salt) * fall * np.float32(0.5 * metres)
            else:
                delta = np.float32(metres) * fall
            self._write_height(y0, y1, x0, x1, patch + delta)
            self._fade_water(y0, y1, x0, x1, fall, 0.9)
            return
        if tool in ("smooth", "flatten", "path"):
            strength = float(np.clip(stroke.amount, 0, 1))
            if tool == "smooth":
                target = _box(_box(patch))
            elif tool == "path":
                iy = int(np.clip(round(cy) - y0, 0, patch.shape[0] - 1))
                ix = int(np.clip(round(cx) - x0, 0, patch.shape[1] - 1))
                stroke.anchor = 0.82 * stroke.anchor + 0.18 * float(patch[iy, ix])
                target = np.float32(stroke.anchor)
            else:
                target = np.float32(stroke.anchor)
            self._write_height(y0, y1, x0, x1, patch + (target - patch) * fall * np.float32(strength))
            self._fade_water(y0, y1, x0, x1, fall, 0.9 if tool != "path" else 1.0)
            if tool == "path":
                self._blend(y0, y1, x0, x1, fall * np.float32(strength), PATH_MIX)
            return
        if tool == "water":
            depth = float(np.clip(stroke.amount, 0, 500)) / self.metres_per_unit
            bed = np.float32(stroke.anchor - depth)
            dug = np.minimum(patch, bed)
            self._write_height(y0, y1, x0, x1, patch * (1.0 - fall) + dug * fall)
            core = np.clip((fall - np.float32(0.35)) / np.float32(0.65), 0, 1) * fall
            self.water[y0:y1, x0:x1] = np.maximum(self.water[y0:y1, x0:x1], core)
            ring = np.clip(1.0 - np.abs(fall - np.float32(0.28)) / np.float32(0.22), 0, 1)
            self._blend(y0, y1, x0, x1, ring * fall * np.float32(0.65), SHORE_MIX)
            return
        if tool == "dry":
            self.water[y0:y1, x0:x1] = np.clip(self.water[y0:y1, x0:x1] * (1.0 - fall), 0, 1)
            return
        if tool == "paint":
            strength = float(np.clip(stroke.amount, 0, 1))
            fall_w = fall * np.float32(strength)
            slot = _MATERIAL_INDEX[stroke.material]
            if stroke.erase:
                mats = np.array(self.materials[y0:y1, x0:x1], np.float32, copy=True)
                mats[..., slot] *= (1.0 - fall_w)
                total = mats.sum(axis=-1, keepdims=True)
                mats = np.where(total > 1e-4, mats / np.maximum(total, 1e-8), 0)
                self.materials[y0:y1, x0:x1] = mats
                if MATERIAL_GROUP[stroke.material] == "water":
                    self._fade_water(y0, y1, x0, x1, fall, strength)
            else:
                target = np.zeros(self.materials.shape[-1], np.float32)
                target[slot] = 1
                self._blend(y0, y1, x0, x1, fall_w, target)
                if MATERIAL_GROUP[stroke.material] == "water":
                    self.water[y0:y1, x0:x1] = np.maximum(self.water[y0:y1, x0:x1], fall_w)
                else:
                    self._fade_water(y0, y1, x0, x1, fall, 0.65)
            return
        if tool == "place":
            scale = float(np.clip(stroke.amount, 0.2, 4))
            reach = max(2.0, stroke.radius * 0.55)
            for prop in self.props:
                if (prop["x"] - cx) ** 2 + (prop["y"] - cy) ** 2 < reach ** 2:
                    return
            self.props.append({
                "id": self.next_id,
                "kind": stroke.prop,
                "x": float(cx),
                "y": float(cy),
                "scale": scale,
                "rotation": float(self.rng.random() * math.tau),
            })
            self.next_id += 1
            return
        if tool == "erase":
            reach = stroke.radius ** 2
            self.props = [prop for prop in self.props
                          if (prop["x"] - cx) ** 2 + (prop["y"] - cy) ** 2 > reach]

    def render(self, mode="shaded", rect=None):
        height, width = self.height.shape
        full = (self.rgb is None or self.rgb_mode != mode or self.rgb.shape[:2] != (height, width))
        if full:
            rect = None
        if rect is None:
            y0, y1, x0, x1 = 0, height, 0, width
        else:
            y0, y1, x0, x1 = rect
            y0 = max(0, y0 - 2)
            y1 = min(height, y1 + 2)
            x0 = max(0, x0 - 2)
            x1 = min(width, x1 + 2)
        painted = self._color_rect(y0, y1, x0, x1, mode)
        if full:
            self.rgb = painted
        else:
            self.rgb[y0:y1, x0:x1] = painted
        self.rgb_mode = mode
        self.view_token += 1
        return self.rgb

    def _color_rect(self, y0, y1, x0, x1, mode):
        heights = self.height[y0:y1, x0:x1]
        materials = self.materials[y0:y1, x0:x1]
        if mode == "height":
            span = max(self.view_hi - self.view_lo, 1e-9)
            grey = 40.0 + 200.0 * np.clip((heights - self.view_lo) / span, 0, 1)
            rgb = np.repeat(grey[..., None], 3, axis=-1)
            sea = heights < 0
            rgb = np.where(sea[..., None], rgb * 0.35 + np.array((30.0, 70.0, 110.0)), rgb)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        if self.biome is not None:
            ids = np.clip(self.biome[y0:y1, x0:x1], 0, len(LEGEND_COLORS) - 1)
            base = LEGEND_COLORS[ids]
        else:
            base = self._hypso(heights)
        cover = np.clip(materials.sum(axis=-1, keepdims=True), 0, 1)
        tint = _ground_color(materials, y0, x0)
        ground = base * (1.0 - 0.82 * cover) + tint * (0.82 * cover)
        shade = _shade(self.height, y0, y1, x0, x1)
        if mode == "materials":
            lit = ground * (0.78 + 0.28 * shade[..., None])
        else:
            lit = ground * (0.46 + 0.72 * shade[..., None])
        sea_mix = np.clip(0.78 - 0.45 * cover[..., 0], 0.25, 0.85)
        sea = heights < 0
        lit = np.where(sea[..., None],
                       lit * (1.0 - sea_mix[..., None]) + np.array((42.0, 92.0, 138.0)) * sea_mix[..., None],
                       lit)
        painted_water = _gather(materials, _WATERS)
        extra = np.clip(self.water[y0:y1, x0:x1] - painted_water, 0, 1) * np.float32(0.85)
        lit = lit * (1.0 - extra[..., None]) + np.array((58.0, 122.0, 164.0)) * extra[..., None]
        return np.clip(lit, 0, 255).astype(np.uint8)

    def _hypso(self, heights):
        span = max(self.view_hi, 0.05)
        t = np.clip(heights / span, 0, 1)[..., None]
        low = np.array((168.0, 154.0, 96.0))
        mid = np.array((86.0, 124.0, 68.0))
        high = np.array((214.0, 210.0, 198.0))
        color = low * (1 - np.clip(t / 0.45, 0, 1)) + mid * np.clip(t / 0.45, 0, 1)
        color = color * (1 - np.clip((t - 0.45) / 0.55, 0, 1)) + high * np.clip((t - 0.45) / 0.55, 0, 1)
        return color

    def save(self):
        """Write the working map back over the export. The first save keeps originals."""
        self.end_stroke()
        self.clipped_on_save = False
        self.library_synced = False
        self._backup_sources()
        self._write_height_file()
        self._write_material_files()
        self._write_authored()
        self._sync_game_copy()
        self.saved_serial = self.serial
        note = f"Saved {self.folder.name}. The first export remains in authored/original."
        if self.props:
            note += f" {len(self.props)} objects."
        if self.clipped_on_save:
            note += " Heights past this tile's shared scale were clipped in the image."
        if self.library_synced:
            note += " Game copy updated."
        return note

    def restore(self):
        """Put the first saved rasters back and return the reopened map."""
        original = self.folder / "authored" / "original"
        files = [path for path in original.rglob("*") if path.is_file()]
        if not files:
            raise ValueError("Save once before restoring. Nothing has been written over this export yet.")
        backed = set()
        for src in files:
            key = src.relative_to(original).as_posix()
            backed.add(key)
            dest = self._path_for_key(key)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        if "materials.png" not in backed:
            created = self.folder / "materials.png"
            if created.is_file():
                created.unlink()
        if self.library_dir is not None:
            for name in _LIBRARY_FILES:
                if f"library/{name}" in backed:
                    continue
                path = self.library_dir / name
                if path.is_file() and self.library_dir.resolve() != self.folder.resolve():
                    path.unlink()
        for name in _WORK_FILES:
            path = self.folder / "authored" / name
            if path.is_file():
                path.unlink()
        return open_export(self.folder, tile=self.tile)

    def _should_backup(self):
        return not (self.folder / "authored" / "edit.json").is_file()

    def _backup_file(self, src, key):
        if not self._should_backup() or not Path(src).is_file():
            return
        dest = self.folder / "authored" / "original" / Path(key)
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def _backup_sources(self):
        if self.kind == "tiled":
            mask_path = self.folder / "mask" / self.tile_path.name
            pairs = [(self.tile_path, self.tile_path.relative_to(self.folder).as_posix()),
                     (mask_path, mask_path.relative_to(self.folder).as_posix()),
                     (self.folder / "manifest.json", "manifest.json")]
        elif self.kind == "library":
            pairs = [(self.folder / name, name) for name in _LIBRARY_FILES]
        else:
            pairs = [(self.height_path, self.height_path.name),
                     (self.height_path.with_suffix(".json"), self.height_path.with_suffix(".json").name),
                     (self.folder / "manifest.json", "manifest.json"),
                     (self.folder / "materials.png", "materials.png"),
                     (self.folder / "material_mask.png", "material_mask.png")]
        if (self.library_dir is not None and self.kind in ("bundle", "hires")
                and self.library_dir.resolve() != self.folder.resolve()):
            pairs.extend((self.library_dir / name, f"library/{name}") for name in _LIBRARY_FILES)
        for src, key in pairs:
            self._backup_file(src, key)

    def _path_for_key(self, key):
        rel = Path(key)
        if rel.parts and rel.parts[0] == "library":
            if self.library_dir is None:
                return None
            return self.library_dir.joinpath(*rel.parts[1:])
        return self.folder.joinpath(rel)

    def _write_height_file(self):
        if self.kind == "tiled":
            lo, hi = self.lo, self.hi
            if float(self.height.min()) < lo - 1e-8 or float(self.height.max()) > hi + 1e-8:
                self.clipped_on_save = True
            span = max(hi - lo, 1e-9)
            codes = np.rint(np.clip((np.clip(self.height, lo, hi) - lo) / span, 0, 1) * 65535)
            _write_png(self.tile_path, codes.astype(np.uint16), 16)
            return
        lo = min(self.lo, float(self.height.min()))
        hi = max(self.hi, float(self.height.max()))
        if hi <= lo:
            hi = lo + 1e-4
        self.lo, self.hi = lo, hi
        heightmap(self.height, self.height_path, lo=lo, hi=hi)
        path = self.folder / "manifest.json"
        if path.is_file():
            data = _load_json(path)
            span = max(hi - lo, 1e-9)
            data["lo"] = lo
            data["hi"] = hi
            data["units_per_code"] = span / 65535.0
            data["sea_code"] = (0.0 - lo) / span * 65535.0
            path.write_text(json.dumps(data, indent=1))

    def _write_material_files(self):
        folded = np.rint(np.clip(project_four(self.materials), 0, 1) * 255).astype(np.uint8)
        if self.kind == "tiled":
            mask_path = self.folder / "mask" / self.tile_path.name
            if mask_path.parent.is_dir():
                Image.fromarray(legacy_mask(self.materials), "RGBA").save(mask_path)
            return
        if self.kind == "library" or self.kind == "hires" or (self.folder / "material_mask.png").is_file():
            Image.fromarray(legacy_mask(self.materials), "RGBA").save(self.folder / "material_mask.png")
        if self.kind != "library":
            Image.fromarray(folded, "RGBA").save(self.folder / "materials.png")

    def _write_authored(self):
        authored = self.folder / "authored"
        authored.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(authored / "height.npz",
                            height=self.height.astype(np.float32),
                            materials=np.asarray(self.materials, np.float32),
                            water=self.water.astype(np.float32),
                            channels=np.array(MATERIALS))
        water = np.rint(np.clip(self.water, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(water, "L").save(authored / "water.png")
        (authored / "props.json").write_text(json.dumps({
            "coords": "xy cells, origin top-left, y down",
            "props": [_prop_record(prop) for prop in self.props],
        }, indent=1))
        preview, step = _stride(self._color_rect(0, self.height.shape[0], 0, self.height.shape[1], "shaded"), 1280)
        Image.fromarray(_stamp_props(preview, self.props, step), "RGB").save(authored / "preview.png")
        (authored / "surfaces.json").write_text(json.dumps({
            "blend": "Weights in a cell sum to 1. A paint stroke blends toward the chosen texture.",
            "channels": [
                {"name": name, "label": MATERIAL_LABELS[name], "group": MATERIAL_GROUP[name]}
                for name in MATERIALS
            ],
        }, indent=1))
        (authored / "edit.json").write_text(json.dumps({
            "tool": "embellish",
            "saved": datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": self.kind,
            "shape": [int(self.height.shape[0]), int(self.height.shape[1])],
            "tile": list(self.tile) if self.tile is not None else None,
            "metres_per_unit": self.metres_per_unit,
            "cell_m": self.cell_m,
            "props": len(self.props),
            "notes": [
                "Height, ground weights, ponds, and props are authored on this export.",
                "Ground is a stack of blendable surfaces. materials.png and the game mask fold that stack into sand, grass, rock, and snow.",
                "The first saved rasters are in authored/original.",
                "Rivers, flow, and canopy stay as they were exported.",
                "Props are markers in props.json and on the preview, not a mesh.",
            ],
        }, indent=1))

    def _sync_game_copy(self):
        library = self.library_dir
        if library is None or self.kind not in ("bundle", "hires"):
            return
        if library.resolve() == self.folder.resolve():
            return
        library.mkdir(parents=True, exist_ok=True)
        small, _step = _stride(self.height, 2048)
        heightmap(small, library / "zemia_height16.png", lo=self.lo, hi=self.hi)
        span = max(float(small.max() - small.min()), 1e-9)
        grey = np.rint(np.clip((small - float(small.min())) / span, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(np.repeat(grey[:, :, None], 3, axis=2), "RGB").save(library / "zemia_height.png")
        mask, _step = _stride(legacy_mask(self.materials), 2048)
        Image.fromarray(np.ascontiguousarray(mask), "RGBA").save(library / "material_mask.png")
        preview, step = _stride(self._color_rect(0, self.height.shape[0], 0, self.height.shape[1], "shaded"), 640)
        Image.fromarray(_stamp_props(preview, self.props, step), "RGB").save(library / "preview.png")
        self.library_synced = True


def _shade(height, y0, y1, x0, x1, z=28.0):
    """Hillshade for one window, using the neighbor just outside it."""
    py0, py1 = max(0, y0 - 1), min(height.shape[0], y1 + 1)
    px0, px1 = max(0, x0 - 1), min(height.shape[1], x1 + 1)
    patch = height[py0:py1, px0:px1]
    if min(patch.shape) < 2:
        return np.full((y1 - y0, x1 - x0), 0.65, np.float32)
    gy = np.zeros(patch.shape, np.float32)
    gx = np.zeros(patch.shape, np.float32)
    gy[1:-1, :] = (patch[2:, :] - patch[:-2, :]) * np.float32(0.5 * z)
    gy[0, :] = (patch[min(1, patch.shape[0] - 1), :] - patch[0, :]) * np.float32(z)
    gy[-1, :] = (patch[-1, :] - patch[max(0, patch.shape[0] - 2), :]) * np.float32(z)
    gx[:, 1:-1] = (patch[:, 2:] - patch[:, :-2]) * np.float32(0.5 * z)
    gx[:, 0] = (patch[:, min(1, patch.shape[1] - 1)] - patch[:, 0]) * np.float32(z)
    gx[:, -1] = (patch[:, -1] - patch[:, max(0, patch.shape[1] - 2)]) * np.float32(z)
    sy, sx = y0 - py0, x0 - px0
    gy = gy[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    gx = gx[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = float(np.deg2rad(135.0))
    altitude = float(np.deg2rad(45.0))
    shade = (math.sin(altitude) * np.cos(slope)
             + math.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    return np.clip(shade, 0, 1).astype(np.float32)


def _load_props(path):
    data = _load_json(path)
    props = []
    notes = []
    next_id = 1
    for raw in data.get("props", ()):
        try:
            kind = raw["kind"]
            if kind not in PROPS:
                raise ValueError(kind)
            prop = {
                "id": int(raw.get("id", next_id)),
                "kind": kind,
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "scale": float(raw.get("scale", 1)),
                "rotation": float(raw.get("rotation", 0)),
            }
        except (KeyError, TypeError, ValueError):
            notes.append("One placed object could not be read and was skipped.")
            continue
        if not all(math.isfinite(prop[key]) for key in ("x", "y", "scale", "rotation")):
            continue
        props.append(prop)
        next_id = max(next_id, prop["id"] + 1)
    return props, next_id, notes


def _load_biome(folder, shape):
    path = folder / "biome.png"
    if not path.is_file():
        return None
    image = Image.open(path)
    if image.mode != "L":
        image = image.convert("L")
    biome = np.array(image)
    if biome.shape != shape:
        return None
    side = _load_json(folder / "biome.json")
    classes = side.get("classes")
    if classes and list(classes) != list(BIOME_NAMES):
        table = {name: index for index, name in enumerate(BIOME_NAMES)}
        mapped = np.zeros_like(biome)
        for old, name in enumerate(classes):
            mapped[biome == old] = table.get(name, 0)
        biome = mapped
    return biome.astype(np.uint8)


def _world_width(folder, meta):
    for name in ("config.json", "manifest.json"):
        data = _load_json(folder / name)
        if "world_width_m" in data:
            return float(data["world_width_m"]), True
    if "world_width_m" in meta:
        return float(meta["world_width_m"]), True
    return 200_000.0, False


def _seed_of(folder, meta):
    for source in (_load_json(folder / "config.json"), meta):
        if "seed" in source:
            try:
                return int(source["seed"])
            except (TypeError, ValueError):
                return 0
    return 0


def open_export(path, tile=None, prefer_authored=True):
    """Open a studio export, one tile of a tiled export, or a height PNG."""
    path = Path(path)
    if path.is_file():
        match = _TILE_NAME.match(path.name)
        if match and path.parent.name == "height":
            tile = (int(match.group(1)), int(match.group(2)))
            path = path.parent.parent
        else:
            path = path.parent
    if not path.is_dir():
        raise ValueError(f"{path} is not an export folder.")
    kind = classify(path)
    if kind is None:
        children = [child for child in sorted(path.iterdir()) if classify(child)] if path.is_dir() else []
        if len(children) == 1:
            path = children[0]
            kind = classify(path)
        elif len(children) > 1:
            raise ValueError("This folder holds several exports. Open one of them from the list.")
        else:
            raise ValueError("No exported height map in this folder.")
    if kind == "tiled":
        tiles = list_tiles(path)
        if tile is None:
            raise TileRequired(inspect(path))
        y0, x0 = int(tile[0]), int(tile[1])
        chosen = next((item for item in tiles if item.y0 == y0 and item.x0 == x0), None)
        if chosen is None:
            raise ValueError(f"No tile at y {y0}, x {x0}.")
        tile = (y0, x0)
        height_path = chosen.path
        meta = _load_json(path / "manifest.json")
    elif kind == "library":
        height_path = path / "zemia_height16.png"
        meta = _load_json(height_path.with_suffix(".json"))
        tile = None
    elif kind == "hires":
        height_path = path / "height16.png"
        meta = _load_json(path / "manifest.json")
        tile = None
    else:
        height_path = path / "height16.png"
        meta = _load_json(height_path.with_suffix(".json"))
        tile = None
    if "lo" not in meta or "hi" not in meta:
        raise ValueError(f"{height_path.name} has no height scale beside it.")
    height, lo, hi = _decode_height(height_path, meta)
    if height.size > _MAX_CELLS:
        raise ValueError("This map is too large to embellish as one image. Open one terrain tile.")
    notes = []
    materials = None
    water = np.zeros(height.shape, np.float32)
    npz_path = path / "authored" / "height.npz"
    if prefer_authored and npz_path.is_file():
        try:
            with np.load(npz_path) as data:
                loaded = _authored_materials(data, height.shape)
                same = (data["height"].shape == height.shape
                        and data["water"].shape == height.shape
                        and loaded is not None)
                if same:
                    height = np.array(data["height"], np.float32)
                    materials = loaded
                    water = np.array(data["water"], np.float32)
                else:
                    notes.append("Authored height did not match this map, so the export image was loaded.")
        except (OSError, ValueError, KeyError):
            notes.append("Authored height could not be read, so the export image was loaded.")
    if materials is None:
        rgba_path = path / "materials.png"
        legacy_path = path / "material_mask.png"
        if rgba_path.is_file():
            rgba = read_rgba(rgba_path)
            if rgba.shape[:2] == height.shape:
                materials = materials_from_rgba(rgba)
        if materials is None and legacy_path.is_file():
            legacy = read_rgba(legacy_path)
            if legacy.shape[:2] == height.shape:
                materials = materials_from_legacy(legacy)
                notes.append("Ground was read from the game mask.")
        if materials is None:
            biome_guess = _load_biome(path, height.shape)
            if biome_guess is not None:
                materials = materials_from_biome(biome_guess)
                notes.append("Ground was inferred from biomes.")
            else:
                materials = materials_from_height(height)
                notes.append("Ground was inferred from sea level.")
    if prefer_authored and not npz_path.is_file():
        water_path = path / "authored" / "water.png"
        if water_path.is_file():
            image = np.array(Image.open(water_path).convert("L"))
            if image.shape == height.shape:
                water = image.astype(np.float32) / np.float32(255)
    props, next_id, prop_notes = ([], 1, [])
    props_path = path / "authored" / "props.json"
    if props_path.is_file():
        props, next_id, prop_notes = _load_props(props_path)
    notes.extend(prop_notes)
    if (path / "rivers.json").is_file():
        notes.append("Rivers stay as exported.")
    if kind == "tiled":
        notes.append("This tile shares a height scale with its neighbors, so extremes are clipped on save.")
    world_width, exact = _world_width(path, meta)
    full_w = int(meta.get("width", height.shape[1]))
    doc = Document()
    doc.folder = path
    doc.kind = kind
    doc.tile = tile
    doc.tile_path = height_path if kind == "tiled" else None
    doc.height_path = height_path
    doc.height = np.ascontiguousarray(height, np.float32)
    doc.materials = _pack_materials(materials, height.size)
    doc.water = np.ascontiguousarray(water, np.float32)
    doc.biome = _load_biome(path, doc.height.shape)
    doc.props = props
    doc.next_id = next_id
    doc.lo = lo
    doc.hi = hi
    doc.metres_per_unit = float(meta.get("metres_per_unit", METRES_PER_UNIT))
    doc.cell_m = float(world_width) / max(full_w, 1)
    doc.cell_exact = exact
    doc.full_shape = (int(meta.get("height", height.shape[0])), full_w)
    doc.library_dir = _find_library(path)
    doc.notes = notes
    doc.rng = np.random.default_rng(_seed_of(path, meta) + 17)
    doc._measure()
    if not doc.cell_exact:
        doc.notes.append("Cell size assumes a 200 km world.")
    return doc
