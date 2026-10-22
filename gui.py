"""Asset studio for the terrain generator."""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

import hires
from terrain import Config, climate, export, generate, grid, render
from terrain.elevation import paint_relief, sculpt_grain
from terrain.presets import PRESETS, make_config
from terrain.view3d import load_surface, mark_strokes, pan_delta, stitch_tiles
from terrain.world import apply_border

ASSETS = Path(os.environ.get("TERRAIN_ASSETS_ROOT", Path(__file__).resolve().parent / "out"))
LIBRARY_TERRAIN = ASSETS / "library" / "terrain"
BUNDLE_PARENT = ASSETS / "export"

OPTIONS = {
    "preset": tuple(PRESETS),
    "erosion_model": ("stream_power", "thermal", "none"),
    "terrain_model": ("classic", "layered", "full"),
    "wind_mode": ("belts", "easterly", "westerly"),
    "depression_method": ("iterative", "priority_flood"),
    "noise_backend": ("numpy", "numba"),
    "detail_level": ("off", "low", "medium", "high"),
    "detail_mode": ("noise", "hierarchical"),
}
OPTION_CAPTIONS = {
    "preset": "World",
    "erosion_model": "Erosion",
    "terrain_model": "Terrain",
    "wind_mode": "Wind",
    "depression_method": "Drainage",
    "noise_backend": "Noise",
    "detail_level": "Export detail",
    "detail_mode": "Hi-res",
}
DETAIL_AMPLITUDES = {"off": 0.0, "low": 0.01, "medium": 0.02, "high": 0.04}
PAINT_TOOLS = ("mountain", "valley", "glacial", "dune", "coast", "cave")
ASSET_RANGE = {
    "glacial": ("ice", 40, 1600),
    "dune": ("sand", 20, 600),
    "coast": ("cut", 40, 1600),
    "cave": ("void", 10, 500),
}
STROKE_COLOR = {
    "mountain": ((196, 112, 64, 84), (255, 220, 180)),
    "valley": ((64, 146, 206, 84), (190, 230, 255)),
    "glacial": ((186, 214, 230, 90), (230, 242, 248)),
    "dune": ((196, 164, 96, 90), (236, 214, 160)),
    "coast": ((64, 150, 140, 90), (180, 230, 220)),
    "cave": ((110, 86, 140, 100), (200, 180, 220)),
}
CURSOR_COLOR = {
    "mountain": (230, 150, 96),
    "valley": (120, 186, 230),
    "glacial": (210, 228, 238),
    "dune": (220, 186, 120),
    "coast": (120, 196, 184),
    "cave": (170, 140, 200),
}
KIND_NAME = {
    "mountain": "mountain",
    "valley": "valley",
    "glacial": "ice",
    "dune": "dunes",
    "coast": "coast",
    "cave": "caves",
}


def latest_pointer():
    """The studio's note of the newest export, kept beside the export folder."""
    return BUNDLE_PARENT.parent / "latest.json"


def read_latest(path):
    """Bundle and hi-res folders recorded by the last successful export."""
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    bundle, hires = data.get("bundle"), data.get("hires")
    return (str(bundle) if bundle else None, str(hires) if hires else None)


def embellish_command(bundle=None, hires=None):
    """Launch the embellish editor, preferring the hi-res export when it exists."""
    script = Path(__file__).resolve().parent / "edit_gui.py"
    command = [sys.executable, str(script)]
    target = hires or bundle
    if target:
        command.append(str(target))
    return command


def config_from_settings(settings, width, height):
    """Apply preset first, then GUI overrides; exclude export-only settings."""
    overrides = {key: value for key, value in settings.items()
                 if key not in ("preset", "detail_level", "detail_mode", "advanced_export")}
    return make_config(settings.get("preset", "earthlike"),
                       width=width, height=height, **overrides)

HEIGHT_STAGES = {"relief", "hotspots", "texture", "coastline", "erosion", "rivers", "placed"}

COVER = {
    climate.ICE, climate.TUNDRA, climate.TAIGA, climate.TEMPERATE_FOREST,
    climate.TEMPERATE_RAINFOREST, climate.TROPICAL_SEASONAL,
    climate.TROPICAL_RAINFOREST,
}
SPARSE = {climate.SHRUBLAND, climate.SAVANNA, climate.STEPPE}

BG = (18, 20, 24)
PANEL = (28, 31, 38)
LINE = (68, 74, 86)
TEXT = (230, 232, 236)
MUTED = (150, 156, 168)
ACCENT = (120, 176, 214)
WARN = (230, 186, 110)
BAR_H = 36

STAGE_WEIGHTS = (
    ("plates", 2),
    ("relief", 6),
    ("hotspots", 2),
    ("texture", 6),
    ("coastline", 5),
    ("pre-rain", 4),
    ("erosion", 28),
    ("temperature", 3),
    ("climate", 5),
    ("rivers", 12),
    ("biomes", 4),
    ("trees", 2),
)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image.astype(np.uint8))


def _surface(image):
    """pygame wants width, height, 3."""
    ordered = np.transpose(_rgb(image), (1, 0, 2))
    return pygame.surfarray.make_surface(np.ascontiguousarray(ordered))


def _blit_scaled(screen, sheet, dest, clip):
    """Draw `sheet` into `dest`. Only the visible part is scaled.

    Scaling the whole atlas made a surface wider than the display can hold,
    and that error closed the window.
    """
    visible = dest.clip(clip)
    if visible.width < 1 or visible.height < 1:
        return
    sw, sh = sheet.get_size()
    if sw < 1 or sh < 1 or dest.width < 1 or dest.height < 1:
        return
    if visible.size == dest.size and dest.width <= 4096 and dest.height <= 4096:
        screen.blit(pygame.transform.smoothscale(sheet, dest.size), dest.topleft)
        return
    x0 = (visible.x - dest.x) * sw / dest.width
    y0 = (visible.y - dest.y) * sh / dest.height
    x1 = (visible.right - dest.x) * sw / dest.width
    y1 = (visible.bottom - dest.y) * sh / dest.height
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(sw, max(left + 1, int(math.ceil(x1))))
    bottom = min(sh, max(top + 1, int(math.ceil(y1))))
    part = sheet.subsurface((left, top, right - left, bottom - top))
    screen.blit(pygame.transform.smoothscale(part, (visible.width, visible.height)), visible.topleft)


def _preview_height(stage, value):
    if stage not in HEIGHT_STAGES:
        return None
    height = value[0] if isinstance(value, tuple) else value
    if not isinstance(height, np.ndarray) or height.ndim != 2:
        return None
    height = hires.downsample(height, hires.PREVIEW_WIDTH)
    return render.relief(height).astype(np.uint8)


def material_mask(world):
    """R rock, G ground, B cover, A accent. Matches library/terrain/layout.json."""
    height = world.height
    land = height > 0
    slope = np.hypot(*grid.gradient(height))
    steep = np.percentile(slope[land], 88) if land.any() else 1.0
    rock = np.where(land, np.clip(slope / max(steep, 1e-6), 0, 1), 0).astype(np.float32)

    cover_amt = np.zeros(height.shape, np.float32)
    biome = world.biome
    cover_amt[np.isin(biome, [climate.ICE, climate.TUNDRA])] = 0.92
    cover_amt[np.isin(biome, list(COVER - {climate.ICE, climate.TUNDRA}))] = 0.74
    cover_amt[np.isin(biome, list(SPARSE))] = 0.30
    cover_amt[biome == climate.DESERT] = 0.06
    cover_amt *= land

    canopy = world.trees.cover if world.trees is not None else 0.0
    accent = np.clip(canopy, 0, 1).astype(np.float32) * land
    accent = np.maximum(accent, ((biome == climate.ICE) & land).astype(np.float32))
    accent *= 1.0 - rock

    cover = (1.0 - rock) * cover_amt
    ground = land.astype(np.float32) * (1.0 - rock) * (1.0 - cover_amt)
    total = np.maximum(rock + ground + cover + accent, 1e-6)
    stacked = np.stack((rock, ground, cover, accent), axis=-1) / total[..., None]
    return np.rint(np.clip(stacked, 0, 1) * 255).astype(np.uint8)


def flow_mask(world):
    """Red channel is wetness: rivers, lakes, and heavy runoff."""
    height = world.height
    land = height > 0
    wet = np.zeros(height.shape, np.float32)
    water = world.water
    if water is not None:
        wet = np.clip(water.width / max(world.cfg.river_width_max, 1e-6), 0, 1)
        wet = np.maximum(wet, water.lake_mask.astype(np.float32) * 0.55)
    if world.runoff is not None and land.any():
        scale = max(float(np.percentile(world.runoff[land], 95)), 1e-6)
        wet = np.maximum(wet, np.clip(world.runoff / scale, 0, 1) * 0.4 * land)
    red = np.rint(np.clip(wet, 0, 1) * 255).astype(np.uint8)
    rgb = np.zeros(height.shape + (3,), np.uint8)
    rgb[..., 0] = red
    return rgb


def height_rgb(world):
    span = max(float(world.height.max() - world.height.min()), 1e-9)
    grey = np.rint(np.clip((world.height - world.height.min()) / span, 0, 1) * 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def stroke_reaches(path, tx, ty, radius, y_scale=0.75):
    """True when the brush around an atlas polyline touches the tile."""
    tx, ty = int(tx), int(ty)
    limit = float(radius) * 1.6
    if limit <= 0:
        return False

    def hit(ax, ay):
        dx = 0.0 if tx <= ax <= tx + 1 else (tx - ax if ax < tx else ax - (tx + 1))
        dy = 0.0 if ty <= ay <= ty + 1 else (ty - ay if ay < ty else ay - (ty + 1))
        return math.hypot(dx, dy * float(y_scale)) <= limit

    pts = [(float(x), float(y)) for x, y in path]
    if any(hit(ax, ay) for ax, ay in pts):
        return True
    for (ax0, ay0), (ax1, ay1) in zip(pts, pts[1:]):
        length = math.hypot(ax1 - ax0, (ay1 - ay0) * float(y_scale))
        steps = max(1, int(math.ceil(length / 0.05)))
        for i in range(1, steps):
            t = i / steps
            if hit(ax0 + (ax1 - ax0) * t, ay0 + (ay1 - ay0) * t):
                return True
    return False


def _thaw_strokes(frozen):
    strokes = []
    for kind, metres, radius, points in frozen or ():
        strokes.append({
            "kind": kind,
            "metres": float(metres),
            "radius": float(radius),
            "points": [(float(y), float(x)) for y, x in points],
        })
    return strokes


def _without_paint(height, strokes, cfg=None, origin=(0, 0)):
    """Height with a previously applied brush taken back off."""
    if not strokes:
        return height
    bed = np.asarray(height, np.float32)
    painted = paint_relief(bed, strokes)
    if cfg is not None:
        painted = painted + sculpt_grain(painted.shape, strokes, cfg, origin)
    return bed - (painted - bed)


def _tiles_at(ax, ay, eps=1e-3):
    """Tile ids that own an atlas point. A rim point belongs to both neighbors."""
    txs = {int(math.floor(ax))}
    tys = {int(math.floor(ay))}
    if abs(ax - round(ax)) <= eps:
        edge = int(round(ax))
        txs.update((edge - 1, edge))
    if abs(ay - round(ay)) <= eps:
        edge = int(round(ay))
        tys.update((edge - 1, edge))
    for tx in sorted(txs):
        for ty in sorted(tys):
            raw_x = ax - tx
            raw_y = ay - ty
            if -eps <= raw_x <= 1.0 + eps and -eps <= raw_y <= 1.0 + eps:
                yield (tx, ty,
                       float(min(1.0, max(0.0, raw_y))),
                       float(min(1.0, max(0.0, raw_x))))


def stroke_crossings(start, end, present, step=0.015):
    """Sample a segment across tile boundaries."""
    ax = float(start[0]) + float(start[3])
    ay = float(start[1]) + float(start[2])
    bx = float(end[0]) + float(end[3])
    by = float(end[1]) + float(end[2])
    dist = math.hypot(bx - ax, by - ay)
    if dist <= 1e-8:
        return
    present = {(int(tx), int(ty)) for tx, ty in present}
    steps = max(1, int(math.ceil(dist / float(step))))
    seen = None
    for i in range(1, steps + 1):
        t = i / steps
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t
        for hit in _tiles_at(x, y):
            if (hit[0], hit[1]) not in present:
                continue
            key = (hit[0], hit[1], round(hit[2], 4), round(hit[3], 4))
            if key == seen:
                continue
            seen = key
            yield hit


def _fit_plane(image, shape):
    image = np.asarray(image)
    if image.shape[:2] == shape:
        return image
    ys = np.linspace(0, image.shape[0] - 1, shape[0]).astype(np.int32)
    xs = np.linspace(0, image.shape[1] - 1, shape[1]).astype(np.int32)
    return np.ascontiguousarray(image[ys][:, xs])


def stitch_atlas(tiles):
    """Join `(tx, ty, height, mask, flow, color)` into one sheet."""
    usable = []
    for tx, ty, height, mask, flow, color in tiles:
        height = np.asarray(height, np.float32)
        if height.ndim != 2 or height.shape[0] < 2 or height.shape[1] < 2:
            continue
        usable.append((int(tx), int(ty), height, mask, flow, color))
    if not usable:
        return None
    rows, cols = usable[0][2].shape
    fitted = []
    for tx, ty, height, mask, flow, color in usable:
        if height.shape != (rows, cols):
            height = _fit_plane(height, (rows, cols)).astype(np.float32)
        fitted.append((tx, ty, height,
                       _fit_plane(mask, (rows, cols)),
                       _fit_plane(flow, (rows, cols)),
                       _fit_plane(color, (rows, cols))))
    txs = [tx for tx, *_rest in fitted]
    tys = [ty for _tx, ty, *_rest in fitted]
    tx0, ty0 = min(txs), min(tys)
    nx = max(txs) - tx0 + 1
    ny = max(tys) - ty0 + 1
    if nx == 1 and ny == 1 and len(fitted) == 1:
        _tx, _ty, height, mask, flow, color = fitted[0]
        return height, mask, flow, color, 1, 1
    out_h = ny * (rows - 1) + 1
    out_w = nx * (cols - 1) + 1
    height = np.zeros((out_h, out_w), np.float32)
    mask = np.zeros((out_h, out_w) + fitted[0][3].shape[2:], fitted[0][3].dtype)
    flow = np.zeros((out_h, out_w) + fitted[0][4].shape[2:], fitted[0][4].dtype)
    color = np.zeros((out_h, out_w) + fitted[0][5].shape[2:], fitted[0][5].dtype)
    for tx, ty, bed, tile_mask, tile_flow, tile_color in fitted:
        y0 = (ty - ty0) * (rows - 1)
        x0 = (tx - tx0) * (cols - 1)
        height[y0:y0 + rows, x0:x0 + cols] = bed
        mask[y0:y0 + rows, x0:x0 + cols] = tile_mask
        flow[y0:y0 + rows, x0:x0 + cols] = tile_flow
        color[y0:y0 + rows, x0:x0 + cols] = tile_color
    return height, mask, flow, color, nx, ny


def stitch_images(tiles):
    """Place `(tx, ty, image)` on one sheet. Neighbouring tiles share their rim."""
    usable = []
    for tx, ty, image in tiles:
        image = np.asarray(image)
        if image.ndim < 2 or image.shape[0] < 2 or image.shape[1] < 2:
            continue
        usable.append((int(tx), int(ty), image))
    if not usable:
        return None
    rows, cols = usable[0][2].shape[:2]
    fitted = []
    for tx, ty, image in usable:
        if image.shape[:2] != (rows, cols):
            image = _fit_plane(image, (rows, cols))
        fitted.append((tx, ty, image))
    txs = [tx for tx, _ty, _image in fitted]
    tys = [ty for _tx, ty, _image in fitted]
    tx0, ty0 = min(txs), min(tys)
    nx = max(txs) - tx0 + 1
    ny = max(tys) - ty0 + 1
    sample = fitted[0][2]
    out = np.zeros((ny * (rows - 1) + 1, nx * (cols - 1) + 1) + sample.shape[2:], sample.dtype)
    for tx, ty, image in fitted:
        y0 = (ty - ty0) * (rows - 1)
        x0 = (tx - tx0) * (cols - 1)
        out[y0:y0 + rows, x0:x0 + cols] = image
    return out


def _edge_strip(field, side, n=24):
    """Columns up to the shared edge, last column on the border."""
    if side == "left":
        n = min(n, field.shape[1])
        return np.array(field[:, -n:], np.float32, copy=True)
    if side == "right":
        n = min(n, field.shape[1])
        return np.array(field[:, n - 1::-1], np.float32, copy=True)
    if side == "top":
        n = min(n, field.shape[0])
        return np.array(field[-n:, :].T, np.float32, copy=True)
    n = min(n, field.shape[0])
    return np.array(field[n - 1::-1, :].T, np.float32, copy=True)


def _border_from(edges):
    """Neighbor samples the generator pins, keyed by side."""
    if not edges:
        return None
    border = {
        edge["side"]: {
            "edge": edge["height"],
            "outer": edge["outer"],
            "strip": edge["strip"],
            "biome": edge["biome"],
        }
        for edge in edges
    }
    levels = [edge.get("sea_level") for edge in edges if edge.get("sea_level") is not None]
    if levels:
        border["sea_level"] = float(levels[0])
    return border


def _shown_color(image, sigma):
    """Soft edges for the on-screen mask. The stored mask stays the hard field."""
    soft = export.feather_color(image, wrap_x=True, sigma=sigma)
    return np.clip(np.rint(soft), 0, 255).astype(np.uint8)


def assets_of(world):
    biome = render.layer(world, next(i for i, (name, _) in enumerate(render.LAYERS) if name == "biomes"))[1]
    canopy = render.layer(world, next(i for i, (name, _) in enumerate(render.LAYERS) if name == "canopy"))[1]
    mask = material_mask(world)
    flow = flow_mask(world)
    return {
        "world": render.water_map(world).astype(np.uint8),
        "height": height_rgb(world),
        "biomes": biome,
        "material mask": _shown_color(mask, export.MASK_FEATHER)[:, :, :3],
        "flow": _shown_color(flow, export.FLOW_FEATHER),
        "canopy": canopy,
        "_mask": mask,
    }


def _save_rgb(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb(image), "RGB").save(path)


def write_game_copy(made, library_dir, world=None):
    """Maintain old RGB maps while providing the exact 16-bit terrain bed."""
    library_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(hires.downsample(made["height"], hires.GAME_MAX), library_dir / "zemia_height.png")
    if world is not None:
        export.heightmap(world.height, library_dir / "zemia_height16.png")
    mask = export.feather_color(hires.downsample(made["_mask"], hires.GAME_MAX),
                                wrap_x=True, sigma=export.MASK_FEATHER)
    Image.fromarray(np.clip(np.rint(mask), 0, 255).astype(np.uint8), "RGBA").save(
        library_dir / "material_mask.png")
    flow = export.feather_color(hires.downsample(made["flow"], hires.GAME_MAX),
                                wrap_x=True, sigma=export.FLOW_FEATHER)
    _save_rgb(np.clip(np.rint(flow), 0, 255).astype(np.uint8), library_dir / "flow_mask.png")
    _save_rgb(hires.downsample(made["world"], hires.PREVIEW_WIDTH), library_dir / "preview.png")
    return library_dir


def _same_settings(world, settings):
    """Simulation options, unlike export detail, require regeneration."""
    cfg = world.cfg
    expected = config_from_settings(settings, cfg.width, cfg.height)
    return all(getattr(cfg, key) == getattr(expected, key) for key in
               ("seed", "n_plates", "land_fraction", "temp_offset", "wind_mode",
                "erosion_model", "terrain_model", "depression_method", "noise_backend"))


def preview_thumbs(made):
    names = ("world", "height", "biomes", "material mask", "flow", "canopy")
    return {name: hires.downsample(made[name], hires.PREVIEW_WIDTH) for name in names}


class Slider:
    def __init__(self, label, lo, hi, value, fmt=str):
        self.label, self.lo, self.hi, self.fmt = label, lo, hi, fmt
        self.value = value
        self.rect = pygame.Rect(0, 0, 180, 14)
        self.grabbed = False

    def set_from(self, x):
        t = np.clip((x - self.rect.x) / max(1, self.rect.w), 0, 1)
        self.value = int(round(self.lo + (self.hi - self.lo) * t))

    def draw(self, screen, font):
        t = (self.value - self.lo) / max(1, self.hi - self.lo)
        y = self.rect.centery
        knob = int(self.rect.x + t * self.rect.w)
        pygame.draw.line(screen, (60, 64, 74), self.rect.midleft, self.rect.midright, 3)
        pygame.draw.line(screen, ACCENT, self.rect.midleft, (knob, y), 3)
        pygame.draw.circle(screen, (240, 244, 248), (knob, y), 6)
        screen.blit(font.render(f"{self.label}  {self.fmt(self.value)}", True, TEXT),
                    (self.rect.x, self.rect.y - 18))


class PreviewCancelled(Exception):
    """A newer preview replaced the one this worker is still building."""


class Button:
    def __init__(self, label, width=120):
        self.label = label
        self.rect = pygame.Rect(0, 0, width, 28)
        self.enabled = True

    def draw(self, screen, font, hot=False, fill=None):
        if not self.enabled:
            color = (40, 42, 48)
        elif fill is not None:
            color = fill
        elif hot:
            color = (64, 104, 140)
        else:
            color = (46, 78, 108)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, LINE, self.rect, 1, border_radius=4)
        img = font.render(self.label, True, TEXT if self.enabled else MUTED)
        screen.blit(img, img.get_rect(center=self.rect.center))


class Progress:
    """Weighted steps for a job, plus a time estimate from the steps already finished."""

    def __init__(self):
        self.active = False
        self.title = ""
        self.steps = []
        self.index = -1
        self.done_weight = 0.0
        self.total_weight = 1.0
        self.started = 0.0
        self.step_started = 0.0
        self.finished_at = None
        self.label = ""
        self.tile_frac = None

    def begin(self, title, steps):
        self.active = True
        self.title = title
        self.steps = [(name, float(weight)) for name, weight in steps]
        self.index = -1
        self.done_weight = 0.0
        self.total_weight = max(1e-6, sum(weight for _, weight in self.steps))
        now = time.perf_counter()
        self.started = now
        self.step_started = now
        self.finished_at = None
        self.label = "starting"
        self.tile_frac = None

    def advance(self, name):
        """`name` is starting, so the step before it has finished."""
        now = time.perf_counter()
        if self.index >= 0:
            self.done_weight += self.steps[self.index][1]
        found = next((i for i, (step, _) in enumerate(self.steps)
                      if step == name and i > self.index), None)
        if found is None:
            self.steps.append((name, 3.0))
            self.total_weight += 3.0
            found = len(self.steps) - 1
        while self.index + 1 < found:
            self.index += 1
            self.done_weight += self.steps[self.index][1]
        self.index = found
        self.step_started = now
        self.label = name
        self.tile_frac = None

    def tiles(self, done, total):
        if self.index < 0 or self.steps[self.index][0] != "tiles":
            self.advance("tiles")
        total = max(1, total)
        self.tile_frac = min(1.0, done / total)
        self.label = f"tiles {done}/{total}"
        if done >= total:
            self.done_weight += self.steps[self.index][1]
            self.steps[self.index] = ("tiles", 0.0)
            self.tile_frac = 1.0

    def complete(self):
        if self.index >= 0 and self.index < len(self.steps):
            self.done_weight += self.steps[self.index][1]
            self.steps[self.index] = (self.steps[self.index][0], 0.0)
        self.done_weight = self.total_weight
        self.active = False
        self.finished_at = time.perf_counter()
        self.label = "done"
        self.tile_frac = 1.0

    def stop(self):
        self.active = False
        self.finished_at = None
        self.label = ""

    def snapshot(self):
        """Fill fraction, elapsed seconds, seconds remaining, and the step name."""
        now = time.perf_counter()
        if not self.started:
            return 0.0, 0.0, None, ""
        elapsed = (self.finished_at or now) - self.started
        if not self.active:
            if self.finished_at:
                return 1.0, elapsed, 0.0, "done"
            return 0.0, 0.0, None, ""
        current = self.steps[self.index][1] if 0 <= self.index < len(self.steps) else 0.0
        if self.tile_frac is not None:
            creep = self.tile_frac
        else:
            time_in = now - self.step_started
            if self.done_weight > 0:
                expected = current * elapsed / self.done_weight
                creep = min(0.92, time_in / max(expected, 1e-3))
            else:
                creep = min(0.35, time_in / 8.0)
        credited = self.done_weight + current * creep
        fraction = min(0.995, credited / self.total_weight)
        eta = None
        if self.done_weight > 0 and elapsed > 0.3:
            eta = elapsed * max(0.0, self.total_weight - credited) / credited
        return fraction, elapsed, eta, self.label


class Studio:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Terrain asset studio")
        self.screen = pygame.display.set_mode((1360, 860), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("segoeui,calibri,consolas", 16)
        self.small = pygame.font.SysFont("segoeui,calibri,consolas", 13)
        self.big = pygame.font.SysFont("segoeui,calibri,consolas", 20, bold=True)
        self.seed = 7
        self.seed_text = "7"
        self.res_i = 2
        self.job = None
        self.plates = Slider("plates", 4, 48, 18)
        self.land = Slider("land", 15, 60, 29, fmt=lambda v: f"{v}%")
        self.temp = Slider("temperature", -20, 20, 0, fmt=lambda v: f"{v:+d} C")
        self.preset = "earthlike"
        self.erosion_model = "stream_power"
        self.terrain_model = "classic"
        self.wind_mode = "belts"
        self.depression_method = "iterative"
        self.noise_backend = "numpy"
        self.detail_level = "medium"
        self.detail_mode = "noise"
        self.advanced_export = True
        self.export_dir = None
        self.panel_scroll = 0
        self.panel_max_scroll = 0
        self.option_buttons = {name: Button("", 252) for name in OPTIONS}
        self.advanced_btn = Button("", 252)
        self.generate_btn = Button("Generate preview", 252)
        self.export_btn = Button("Export maps", 252)
        self.export_btn.enabled = False
        self.embellish_btn = Button("Embellish maps", 252)
        self.last_bundle, self.last_hires = read_latest(latest_pointer())
        self.size_rects = []
        self.thumb_rects = []
        self.view = pygame.Rect(300, 16, 1040, 680)
        self.zoom = 1.0
        self.cam = [0.0, 0.0]
        self.drag = None
        self.image = None
        self.image_name = ""
        self.thumbs = {}
        self.world = None
        self.made = None
        self.atlas = {}
        self.tx = 0
        self.ty = 0
        self.live_cell = None
        self.live_surface = None
        self.preset_dirty = True
        self._seed_from = (0, 0)
        self._fit_when_done = True
        self.busy = False
        self.status = ("Generate a preview. M mountain, V valley, I ice, G dunes, "
                       "O coast, C caves. Generate again to build the strokes in.")
        self.seed_box = pygame.Rect(16, 72, 164, 28)
        self.random_btn = Button("Random", 80)
        self.random_btn.rect.topleft = (188, 72)
        self.events = queue.Queue()
        self.clock = pygame.time.Clock()
        self.progress = Progress()
        self.tool = "view"
        self.mountain_m = 4000
        self.valley_m = 1500
        self.relief_m = Slider("height", 200, 8000, self.mountain_m,
                               fmt=lambda v: f"{v / 1000:.2f} km")
        self.brush = Slider("brush", 8, 160, 40, fmt=lambda v: f"{v} px")
        self.tool_buttons = {
            "view": Button("Pan", 64),
            "mountain": Button("Mountain", 96),
            "valley": Button("Valley", 80),
            "glacial": Button("Ice", 52),
            "dune": Button("Dunes", 68),
            "coast": Button("Coast", 64),
            "cave": Button("Cave", 60),
        }
        self.asset_m = {"glacial": 400, "dune": 160, "coast": 400, "cave": 80}
        self.undo_paint_btn = Button("Undo", 64)
        self.clear_paint_btn = Button("Clear", 64)
        self.view3d_btn = Button("3D", 52)
        self.view3d = False
        self.orbit = [0.55, 0.78, 1.2]
        self.look = [0.5, 0.5]
        self.pan_drag = None
        self.orbit_drag = None
        self._view3d_cache = None
        self._sheet_layout = None
        self._stroke_cell = None
        self.strokes = {}
        self.baked_strokes = {}
        self._undo_stack = []
        self._generate_rest = []
        self._inflight_sculpt = None
        self._preview_token = 0
        self._preview_cancel = threading.Event()
        self.painting = None
        self.pointer = None
        self.sculpt_bar = pygame.Rect(0, 0, 0, 0)

    def settings(self):
        return dict(
            seed=self.seed,
            n_plates=self.plates.value,
            land_fraction=self.land.value / 100,
            temp_offset=float(self.temp.value),
            preset=self.preset,
            erosion_model=self.erosion_model,
            terrain_model=self.terrain_model,
            wind_mode=self.wind_mode,
            depression_method=self.depression_method,
            noise_backend=self.noise_backend,
            detail_level=self.detail_level,
            detail_mode=self.detail_mode,
            advanced_export=self.advanced_export,
        )

    def config(self):
        width, height = hires.map_size(hires.PREVIEW_WIDTH)
        return config_from_settings(self.settings(), width, height)

    def cycle_option(self, option):
        values = OPTIONS[option]
        current = getattr(self, option)
        setattr(self, option, values[(values.index(current) + 1) % len(values)])
        if option == "preset":
            base = make_config(self.preset)
            self.plates.value = base.n_plates
            self.land.value = round(base.land_fraction * 100)
            self.temp.value = round(base.temp_offset)
        if option in ("detail_level", "detail_mode"):
            if option == "detail_mode" and self.detail_mode == "hierarchical":
                self.status = ("Hi-res export reruns erosion at the export cell size. "
                               "It does not change the preview.")
            else:
                self.status = "Microdetail changes the next export, not the simulated preview."
        else:
            self.preset_dirty = True
            notes = {
                "classic": "Classic incision. Existing seeds stay as they were.",
                "layered": "Layered terrain adds sediment, rock layers, talus and river banks.",
                "full": "Full terrain adds multiscale erosion, raindrops, flow warp and wind.",
            }
            if option == "terrain_model":
                self.status = notes[self.terrain_model] + " Generate preview to apply it."
            else:
                self.status = "Settings changed. Click Generate preview to apply them."

    def _freeze_strokes(self, strokes):
        frozen = []
        for stroke in strokes or ():
            frozen.append((
                stroke["kind"],
                float(stroke["metres"]),
                float(stroke["radius"]),
                tuple((float(y), float(x)) for y, x in stroke["points"]),
            ))
        return tuple(frozen)

    def _sculpt_payload(self, cell):
        payload = []
        for stroke in self.strokes.get(cell, ()):
            payload.append({
                "kind": stroke["kind"],
                "metres": float(stroke["metres"]),
                "radius": float(stroke["radius"]),
                "points": [(float(y), float(x)) for y, x in stroke["points"]],
            })
        return payload

    def _paint_pending(self):
        return any(not self._strokes_baked(cell) for cell in self.atlas)

    def _strokes_baked(self, cell):
        current = self._freeze_strokes(self.strokes.get(cell))
        return current == self.baked_strokes.get(cell, ())

    def _sync_relief_slider(self):
        if self.tool in ASSET_RANGE:
            label, lo, hi = ASSET_RANGE[self.tool]
            self.relief_m.label = label
            self.relief_m.lo = lo
            self.relief_m.hi = hi
            self.relief_m.value = self.asset_m[self.tool]
            self.relief_m.fmt = lambda v: f"{v} m"
            return
        self.relief_m.lo = 200
        self.relief_m.hi = 8000
        self.relief_m.fmt = lambda v: f"{v / 1000:.2f} km"
        if self.tool == "valley":
            self.relief_m.label = "depth"
            self.relief_m.value = self.valley_m
        else:
            self.relief_m.label = "height"
            self.relief_m.value = self.mountain_m

    def _store_relief(self):
        if self.tool in ASSET_RANGE:
            self.asset_m[self.tool] = int(self.relief_m.value)
        elif self.tool == "valley":
            self.valley_m = int(self.relief_m.value)
        else:
            self.mountain_m = int(self.relief_m.value)

    def _stroke_metres(self):
        if self.tool in ASSET_RANGE:
            return float(self.asset_m[self.tool])
        if self.tool == "valley":
            return float(self.valley_m)
        return float(self.mountain_m)

    def _set_tool(self, name):
        if self.painting is not None:
            self._commit_stroke()
        self.tool = name
        self._sync_relief_slider()
        notes = {
            "view": "Pan. Drag to move. M mountain, V valley, I ice, G dunes, O coast, C caves.",
            "mountain": ("Drag a mountain across the generated tiles. "
                         "Generate preview builds it into each tile it crosses."),
            "valley": ("Drag a valley across the generated tiles. "
                       "Generate preview builds it into each tile it crosses."),
            "glacial": ("Drag a glacier. Generate preview puts ice there and cuts the valley under it."),
            "dune": ("Drag a dune field. Generate preview piles sand there."),
            "coast": ("Drag along a shoreline. Generate preview cuts that shore and leaves a beach."),
            "cave": ("Drag a cave. Generate preview opens a hollow and a void under it."),
        }
        self.status = notes[name]

    def _toggle_view3d(self):
        if (self.tx, self.ty) not in self.atlas:
            self.status = "Generate a preview, then load the 3D surface."
            return
        self.view3d = not self.view3d
        self._view3d_cache = None
        self.pan_drag = None
        self.orbit_drag = None
        if self.view3d:
            self.image_name = "world"
            self.status = ("3D view of every generated tile. "
                           "Left-drag pans, right-drag rotates, scroll zooms.")
        else:
            self.status = "Map view. D loads the rendered surface in 3D."

    def start(self):
        if self.busy:
            return
        if self.painting is not None:
            self._commit_stroke()
        try:
            self.seed = int(self.seed_text)
        except ValueError:
            self.status = "Seed needs to be a whole number."
            return
        cell = (self.tx, self.ty)
        self._generate_rest = [other for other in sorted(self.atlas)
                               if other != cell and not self._strokes_baked(other)]
        self._launch(cell)

    def cancel_preview(self):
        """Stop the preview in progress. Tiles already finished stay on the map."""
        if not self.busy or self.job != "preview":
            return
        self._preview_cancel.set()
        self._preview_token += 1
        self._generate_rest = []
        self._inflight_sculpt = None
        self.live_surface = None
        self.live_cell = None
        self.progress.stop()
        self.busy = False
        self.job = None
        self.export_btn.enabled = self.world is not None
        self.status = "Generation stopped. Pan to a tile, then click Generate preview."
        self._release_map()

    def _launch(self, cell, keep=False):
        self.busy = True
        self.job = "preview"
        self._preview_token += 1
        self._preview_cancel = threading.Event()
        token = self._preview_token
        cancel = self._preview_cancel
        self.export_btn.enabled = False
        self.image_name = "world"
        self.tx, self.ty = cell
        self.live_cell = cell
        self.live_surface = None
        edges = self._edges_for(*cell)
        if keep and cell in self.atlas:
            settings = dict(self.atlas[cell]["settings"])
            self.seed = int(settings["seed"])
            self.seed_text = str(self.seed)
        else:
            settings = self._preset_for(*cell)
        title = "Extending" if edges else "Generating preview"
        left = len(self._generate_rest)
        extra = f"  {left + 1} tiles" if left else ""
        self.status = f"{title.lower()} tile {cell[0]},{cell[1]}{extra}"
        self.progress.begin(title, STAGE_WEIGHTS + (("maps", 4),))
        pw, ph = hires.map_size(hires.PREVIEW_WIDTH)
        cfg = config_from_settings(settings, pw, ph)
        sculpt = self._sculpt_payload(cell)
        self._inflight_sculpt = (cell, self._freeze_strokes(sculpt))
        threading.Thread(
            target=self._worker,
            args=(cfg, edges, settings, cell, sculpt, token, cancel),
            daemon=True).start()

    def _random_seed(self):
        return int(np.random.default_rng().integers(0, 0x80000000))

    def randomize_seed(self):
        """Put a new seed in the box. The next preview builds that world."""
        self.seed = self._random_seed()
        self.seed_text = str(self.seed)
        self.preset_dirty = True

    def _preset_for(self, tx, ty):
        """Sliders are the preset. A tile with no world yet takes a new seed unless one was just chosen."""
        settings = self.settings()
        if ((tx, ty) not in self.atlas and not self.preset_dirty
                and self._seed_from in self.atlas):
            settings["seed"] = self._random_seed()
        self.seed = int(settings["seed"])
        self.seed_text = str(self.seed)
        self.preset_dirty = False
        return settings

    def _edges_for(self, tx, ty):
        found = []
        for (dx, dy), side, take, outside in (
            ((-1, 0), "left", lambda a: a[:, -1], lambda a: a[:, -2]),
            ((1, 0), "right", lambda a: a[:, 0], lambda a: a[:, 1]),
            ((0, -1), "top", lambda a: a[-1, :], lambda a: a[-2, :]),
            ((0, 1), "bottom", lambda a: a[0, :], lambda a: a[1, :]),
        ):
            other = self.atlas.get((tx + dx, ty + dy))
            if other is None or other["world"].trees is None:
                continue
            world = other["world"]
            relief = world.height_pre if world.height_pre is not None else world.height
            relief = np.asarray(relief, np.float32)
            neighbor = (tx + dx, ty + dy)
            solve = relief
            current = self.strokes.get(neighbor) or []
            if current and not self._strokes_baked(neighbor):
                base = relief
                baked = self.baked_strokes.get(neighbor)
                if baked:
                    base = _without_paint(relief, _thaw_strokes(baked), world.cfg, neighbor)
                solve = paint_relief(base, current)
                solve = solve + sculpt_grain(solve.shape, current, world.cfg, neighbor)
            shade = np.asarray(world.surface, np.float32)
            canopy = np.asarray(world.trees.cover, np.float32)
            found.append({
                "side": side,
                "height": np.array(take(solve), dtype=np.float32, copy=True),
                "outer": np.array(outside(solve), dtype=np.float32, copy=True),
                "strip": _edge_strip(solve, side),
                "sea_level": getattr(world, "sea_level", None),
                "shade": np.array(outside(shade), dtype=np.float32, copy=True),
                "biome": np.array(take(world.biome), copy=True),
                "cover": np.array(take(canopy), dtype=np.float32, copy=True),
                "cover_outer": np.array(outside(canopy), dtype=np.float32, copy=True),
                "cover_strip": _edge_strip(canopy, side),
            })
        return found

    def _refresh_seam(self, cell):
        """Relight the shared line of this cell and the cells it touches."""
        tx, ty = cell
        keys = [(tx, ty)]
        keys.extend((tx + dx, ty + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        for key in keys:
            tile = self.atlas.get(key)
            if tile is None or "_plain" not in tile["made"]:
                continue
            edges = self._edges_for(*key)
            plain = tile["made"]["_plain"]
            if edges:
                water = tile["world"].water
                skip = None
                if water is not None:
                    skip = np.asarray(water.lake_mask) | np.asarray(water.river_mask)
                rgb = hires.relight_edges(
                    plain, np.asarray(tile["world"].surface, np.float32), edges, skip)
            else:
                rgb = plain
            tile["made"]["world"] = rgb
            tile["surface"] = _surface(rgb)

    def _worker(self, cfg, edges, settings, cell, sculpt=None, token=None, cancel=None):
        border = _border_from(edges)

        def abandon():
            return cancel is not None and cancel.is_set()

        def on_stage(name):
            if abandon():
                raise PreviewCancelled()
            self.events.put(("stage", name, token))

        def on_preview(name, value):
            if abandon():
                raise PreviewCancelled()
            if border and name not in ("coastline", "erosion", "rivers", "placed"):
                return
            if border:
                height = value[0] if isinstance(value, tuple) else value
                if isinstance(height, np.ndarray) and height.ndim == 2:
                    shown = hires.downsample(height, hires.PREVIEW_WIDTH)
                    self.events.put(("cell_image", cell,
                                     render.relief(shown).astype(np.uint8), token))
                    return
            image = _preview_height(name, value)
            if image is not None:
                self.events.put(("cell_image", cell, image, token))

        try:
            if abandon():
                return
            if border:
                level = float(np.mean([np.mean(edge["height"]) for edge in edges]))
                canvas = np.full((cfg.height, cfg.width), level, np.float32)
                self.events.put(("cell_image", cell,
                                 render.relief(apply_border(canvas, border)).astype(np.uint8),
                                 token))
            world = generate(cfg, verbose=False, on_stage=on_stage,
                             on_preview=on_preview, border=border, sculpt=sculpt or None,
                             sculpt_at=cell)
            rows, cols = world.height.shape
            world.tile_origin = (cell[1] * max(rows - 1, 1), cell[0] * max(cols - 1, 1))
            if abandon():
                return
            self.events.put(("stage", "maps", token))
            made = assets_of(world)
            made["_plain"] = np.array(made["world"], copy=True)
        except PreviewCancelled:
            return
        except Exception as err:
            if abandon():
                return
            self.events.put(("error", f"generation failed: {err}", token))
            return
        if abandon():
            return
        self.events.put(("done", world, made, settings, cell, token))

    def _preview_current(self, token):
        return token is None or token == self._preview_token

    def pump(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.status = event[1]
                elif kind == "stage":
                    if len(event) > 2 and not self._preview_current(event[2]):
                        continue
                    self.progress.advance(event[1])
                    self.status = event[1]
                elif kind == "tiles":
                    done, total = event[1], event[2]
                    self.progress.tiles(done, total)
                    if done == total or done % 8 == 0:
                        self.status = f"writing tiles {done}/{total}"
                elif kind == "cell_image":
                    if len(event) > 3 and not self._preview_current(event[3]):
                        continue
                    cell, image = event[1], event[2]
                    if self.job == "preview" and cell == (self.tx, self.ty):
                        self.live_cell = cell
                        self.live_surface = _surface(image)
                        if not self.atlas:
                            self.fit()
                elif kind == "image":
                    self._set_image(event[1], event[2], fit=self.image is None)
                elif kind == "preview_ready":
                    self.world, made = event[1], event[2]
                    self.made = made
                    self.thumbs = {name: made[name] for name in
                                   ("world", "height", "biomes", "material mask", "flow", "canopy")}
                    self._set_image("world", made["world"], fit=True)
                elif kind == "done":
                    if len(event) > 5 and not self._preview_current(event[5]):
                        continue
                    world, made, settings, cell = event[1], event[2], event[3], event[4]
                    self.live_surface = None
                    self.live_cell = None
                    self.atlas[cell] = {
                        "world": world,
                        "made": made,
                        "settings": settings,
                        "surface": _surface(made["world"]),
                    }
                    self._refresh_seam(cell)
                    if self._inflight_sculpt and self._inflight_sculpt[0] == cell:
                        self.baked_strokes[cell] = self._inflight_sculpt[1]
                    self._inflight_sculpt = None
                    if self._generate_rest:
                        nxt = self._generate_rest.pop(0)
                        self._show_cell(*cell, fit=False)
                        self.status = (f"tile {cell[0]},{cell[1]} is in the image. "
                                       f"Building {nxt[0]},{nxt[1]}.")
                        self._launch(nxt, keep=True)
                    else:
                        self.progress.complete()
                        self.busy = False
                        self.job = None
                        self._show_cell(*cell, fit=self._fit_when_done)
                        h, w = world.height.shape
                        baked = self.baked_strokes.get(cell) or ()
                        paint = f"  {len(baked)} painted" if baked else ""
                        if self._paint_pending():
                            paint = "  paint changed since this build"
                        self.status = (f"tile {cell[0]},{cell[1]}  {w}×{h}  "
                                       f"land {world.land.mean():.0%}  seed {settings['seed']}{paint}")
                elif kind == "exported":
                    self.progress.complete()
                    self.busy = False
                    self.job = None
                    self.export_btn.enabled = self.world is not None
                    self.status = event[1]
                    self.thumbs = event[2]
                    if len(event) > 4:
                        self.last_bundle = event[3]
                        self.last_hires = event[4]
                    self._set_image("world", self.thumbs["world"], fit=True)
                elif kind == "error":
                    if len(event) > 2 and not self._preview_current(event[2]):
                        continue
                    self.progress.stop()
                    self._inflight_sculpt = None
                    self._generate_rest = []
                    self.live_surface = None
                    self.live_cell = None
                    self.busy = False
                    self.job = None
                    self.export_btn.enabled = self.world is not None
                    self.status = event[1]
        except queue.Empty:
            pass

    def _set_image(self, name, image, fit=False):
        self.image_name = name
        self.image = _surface(hires.downsample(image, hires.PREVIEW_WIDTH))
        if fit:
            self.fit()

    def fit(self):
        layer = self.image_name not in ("", "world")
        if layer and self.image is not None:
            mw, mh = self.image.get_width(), self.image.get_height()
            self.zoom = min(self.view.w / mw, self.view.h / mh)
            self.cam = [(mw - self.view.w / self.zoom) / 2, (mh - self.view.h / self.zoom) / 2]
            return
        if not self.atlas and self.live_surface is None:
            return
        mw, mh, sx, sy = self._tile_stride()
        self.zoom = min(self.view.w / mw, self.view.h / mh)
        self.cam = [self.tx * sx + (mw - self.view.w / self.zoom) / 2,
                    self.ty * sy + (mh - self.view.h / self.zoom) / 2]

    def _tile_size(self):
        if self.atlas:
            surf = next(iter(self.atlas.values()))["surface"]
            return surf.get_width(), surf.get_height()
        if self.live_surface is not None:
            return self.live_surface.get_width(), self.live_surface.get_height()
        return hires.map_size(hires.PREVIEW_WIDTH)

    def _show_cell(self, tx, ty, fit=True):
        tile = self.atlas[(tx, ty)]
        self.tx, self.ty = tx, ty
        self.world = tile["world"]
        self.made = tile["made"]
        self.thumbs = {name: tile["made"][name] for name in
                       ("world", "height", "biomes", "material mask", "flow", "canopy")}
        self.image_name = "world"
        self.image = tile["surface"]
        settings = tile["settings"]
        self.seed = int(settings["seed"])
        self.seed_text = str(self.seed)
        self.plates.value = int(settings["n_plates"])
        self.land.value = int(round(settings["land_fraction"] * 100))
        self.temp.value = int(round(settings["temp_offset"]))
        for option in OPTIONS:
            setattr(self, option, settings.get(option, getattr(self, option)))
        self.advanced_export = settings.get("advanced_export", True)
        self.preset_dirty = False
        self.export_btn.enabled = True
        if fit:
            self.fit()

    def _step(self, dx, dy):
        if self.busy or not self.atlas:
            return
        self._seed_from = (self.tx, self.ty)
        self.tx += dx
        self.ty += dy
        self._fit_when_done = True
        if (self.tx, self.ty) in self.atlas:
            self._show_cell(self.tx, self.ty, fit=True)
            return
        self.randomize_seed()
        self.fit()
        self.status = (f"Tile {self.tx},{self.ty} is empty. "
                       "Click Generate preview to build it.")

    def _cell_under_center(self):
        if not self.atlas:
            return None
        mw, mh, sx, sy = self._tile_stride()
        cx = self.cam[0] + self.view.w / (2 * max(self.zoom, 1e-6))
        cy = self.cam[1] + self.view.h / (2 * max(self.zoom, 1e-6))
        return int(math.floor(cx / sx)), int(math.floor(cy / sy))

    def _touches_atlas(self, tx, ty):
        return any((tx + dx, ty + dy) in self.atlas
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def _release_map(self):
        if self.busy or not self.atlas or self.image_name not in ("", "world"):
            return
        cell = self._cell_under_center()
        if cell is None:
            return
        if cell in self.atlas:
            if cell != (self.tx, self.ty):
                self._show_cell(*cell, fit=False)
            return
        if self._touches_atlas(*cell):
            if cell == (self.tx, self.ty):
                return
            self._seed_from = (self.tx, self.ty)
            self.tx, self.ty = cell
            self._fit_when_done = False
            self.randomize_seed()
            self.status = (f"Tile {cell[0]},{cell[1]} is empty. "
                           "Click Generate preview to build it.")

    def open_embellish(self):
        command = embellish_command(self.last_bundle, self.last_hires)
        subprocess.Popen(command, cwd=str(Path(command[1]).parent))
        if len(command) > 2:
            self.status = f"Embellish editor is opening {Path(command[2]).name}."
        else:
            self.status = "Embellish editor is open. Choose an export."

    def _ask_export_folder(self):
        """Folder the user picked, None to use the studio default, False if they cancelled."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            self.status = "The folder dialog is unavailable, so maps go to the usual export folder."
            return None
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        start = self.export_dir or ASSETS
        if not Path(start).is_dir():
            start = Path.home()
        root.update()
        chosen = filedialog.askdirectory(
            title="Choose a folder for the exported maps",
            initialdir=str(start),
            mustexist=True,
        )
        root.destroy()
        if not chosen:
            return False
        self.export_dir = Path(chosen)
        return self.export_dir

    def export(self):
        if self.world is None or self.busy:
            return
        if self._paint_pending():
            self.status = ("Generate preview to build the painted mountains and valleys, "
                           "then export.")
            return
        destination = self._ask_export_folder()
        if destination is False:
            self.status = "Export cancelled."
            return
        self.busy = True
        self.job = "export"
        self.export_btn.enabled = False
        width = hires.EXPORT_WIDTHS[self.res_i]
        settings = dict(self.atlas[(self.tx, self.ty)]["settings"])
        settings["detail_level"] = self.detail_level
        settings["detail_mode"] = self.detail_mode
        settings["advanced_export"] = self.advanced_export
        resim = False
        w, h, _span = self._export_span(width)
        steps = list(STAGE_WEIGHTS) + [("maps", 4)] if resim else []
        if max(w, h) > hires.DIRECT_MAX:
            ntiles = math.ceil(w / hires.TILE) * math.ceil(h / hires.TILE)
            steps.append(("tiles", ntiles))
        else:
            steps.append(("writing", 8))
        where = destination if destination is not None else LIBRARY_TERRAIN
        self.progress.begin(f"Exporting {hires.EXPORT_LABELS[self.res_i]}", steps)
        self.status = f"Exporting to {where}."
        threading.Thread(
            target=self._export_worker,
            args=(width, settings, resim, destination), daemon=True).start()

    def _export_span(self, width):
        """Pixel size of the export. Every generated tile is part of that image."""
        tile_w, tile_h = hires.map_size(width)
        atlas = getattr(self, "atlas", None) or {}
        if len(atlas) <= 1:
            return tile_w, tile_h, None
        rows, cols = next(iter(atlas.values()))["world"].height.shape
        txs = [key[0] for key in atlas]
        tys = [key[1] for key in atlas]
        nx = max(txs) - min(txs) + 1
        ny = max(tys) - min(tys) + 1
        src_w = nx * (cols - 1) + 1
        src_h = ny * (rows - 1) + 1
        full_w = int(round((src_w - 1) * (tile_w - 1) / max(cols - 1, 1))) + 1
        full_h = int(round((src_h - 1) * (tile_h - 1) / max(rows - 1, 1))) + 1
        return full_w, full_h, (nx, ny)

    def _simulate(self, width, height, settings):
        cfg = config_from_settings(settings, width, height)

        def on_stage(name):
            self.events.put(("stage", f"{name}"))

        def on_preview(name, value):
            image = _preview_height(name, value)
            if image is not None:
                self.events.put(("image", name, image))

        return generate(cfg, verbose=False, on_stage=on_stage, on_preview=on_preview,
                        sculpt=self._sculpt_payload((self.tx, self.ty)) or None,
                        sculpt_at=(self.tx, self.ty))

    def _export_worker(self, width, settings, resim, destination=None):
        w, h, span = self._export_span(width)
        try:
            if resim:
                pw, ph = hires.map_size(hires.PREVIEW_WIDTH)
                self.events.put(("status", f"rebuilding preview at {pw}×{ph}"))
                world = self._simulate(pw, ph, settings)
                self.events.put(("stage", "maps"))
                made = assets_of(world)
                self.events.put(("preview_ready", world, made))
            else:
                world, made = self.world, self.made
            height = np.asarray(world.height, np.float32)
            mask = made["_mask"]
            flow = made["flow"]
            color = made["world"]
            game_made = made
            game_world = world
            origin = (self.ty * h, self.tx * w)
            world_width_m = float(world.cfg.world_width_m)
            seed = int(world.cfg.seed)
            keys = sorted(getattr(self, "atlas", None) or {})
            whole = len(keys) > 1
            if whole:
                tx0 = min(tx for tx, _ty in keys)
                tx1 = max(tx for tx, _ty in keys)
                ty0 = min(ty for _tx, ty in keys)
                ty1 = max(ty for _tx, ty in keys)
                corner = (tx0, ty0) if (tx0, ty0) in self.atlas else keys[0]
                seed = int(self.atlas[corner]["settings"]["seed"])
            if span is not None:
                tiles = []
                for (tx, ty), tile in self.atlas.items():
                    tile_made = tile["made"]
                    tiles.append((tx, ty, np.asarray(tile["world"].height, np.float32),
                                  tile_made["_mask"], tile_made["flow"], tile_made["world"]))
                stitched = stitch_atlas(tiles)
                if stitched is not None:
                    height, mask, flow, color, nx, ny = stitched
                    bed = type("Bed", (), {})()
                    bed.height = height
                    biomes = stitch_images(
                        (tx, ty, tile["made"]["biomes"])
                        for (tx, ty), tile in self.atlas.items()
                        if "biomes" in tile["made"])
                    canopy = stitch_images(
                        (tx, ty, tile["made"]["canopy"])
                        for (tx, ty), tile in self.atlas.items()
                        if "canopy" in tile["made"])
                    game_made = {
                        "height": height_rgb(bed),
                        "world": color,
                        "_mask": mask,
                        "flow": flow,
                        "material mask": np.asarray(mask)[:, :, :3],
                        "biomes": color if biomes is None else biomes,
                        "canopy": color if canopy is None else canopy,
                    }
                    game_world = bed
                    origin = (0, 0)
                    tile_w, _tile_h = hires.map_size(width)
                    world_width_m = float(world.cfg.world_width_m) * (w / max(tile_w, 1))
                    span = (nx, ny)
            preset = settings.get("preset", "earthlike")
            if whole:
                stem = (f"{preset}_x{tx0}_to_{tx1}_y{ty0}_to_{ty1}_{w}x{h}")
            else:
                stem = (f"{preset}_seed{seed}_{w}x{h}_x{self.tx}_y{self.ty}")
            if destination:
                hires_parent = Path(destination) / "hires"
                game_dir = Path(destination)
                bundle_parent = Path(destination)
            else:
                hires_parent = LIBRARY_TERRAIN / "hires"
                game_dir = LIBRARY_TERRAIN
                bundle_parent = BUNDLE_PARENT
            folder = hires_parent / stem
            index = 2
            while folder.exists():
                folder = hires_parent / f"{stem}-{index}"
                index += 1
            if whole:
                game_dir = folder
            where = (f"the world x{tx0} to {tx1}, y{ty0} to {ty1}"
                     if whole else f"{w}×{h}")
            self.events.put(("status", f"upscaling {where} to {w}×{h}"))
            metadata = {
                "preset": preset,
                "erosion_model": world.cfg.erosion_model,
                "terrain_model": world.cfg.terrain_model,
                "detail_mode": settings.get("detail_mode", "noise"),
                "wind_mode": world.cfg.wind_mode,
                "depression_method": world.cfg.depression_method,
                "noise_backend": world.cfg.noise_backend,
                "simulation_seed": seed,
                "tile": [tx0, ty0] if whole else [self.tx, self.ty],
            }
            if whole:
                metadata["span"] = [tx1 - tx0 + 1, ty1 - ty0 + 1]
                metadata["tiles"] = [
                    {"x": tx, "y": ty, "seed": int(self.atlas[(tx, ty)]["settings"]["seed"])}
                    for tx, ty in keys
                ]
            detail_fraction = DETAIL_AMPLITUDES[settings["detail_level"]]
            detail_mode = settings.get("detail_mode", "noise")
            atlas_sampling = {"wrap_x": False,
                              "tile_origin": origin,
                              "detail_seed": 0,
                              "detail_reference_span": hires.ATLAS_DETAIL_REFERENCE_SPAN,
                              "detail_mode": detail_mode,
                              "world_width_m": world_width_m}
            if max(w, h) > hires.DIRECT_MAX:
                hires.write_tiles(
                    height, mask, flow, w, h, seed, folder,
                    on_tile=lambda done, total: self.events.put(("tiles", done, total)),
                    detail_fraction=detail_fraction, generator_info=metadata,
                    **atlas_sampling,
                )
            else:
                self.events.put(("stage", "writing"))
                hires.write_direct(
                    height, mask, flow, w, h, seed, folder,
                    detail_fraction=detail_fraction, generator_info=metadata,
                    **atlas_sampling)
            game = write_game_copy(game_made, game_dir, world=game_world)
            if whole:
                # The seed bundle is one simulated tile. A world of several
                # tiles is the stitched map, so that bundle would be only the
                # tile that happened to be open.
                bundle = folder
                note = (f"exported the world x{tx0} to {tx1}, y{ty0} to {ty1} "
                        f"({len(keys)} tiles) at {w}×{h} to {folder.as_posix()}")
            else:
                bundle = export.bundle(world, parent=bundle_parent,
                                       advanced=settings["advanced_export"])
                Image.fromarray(_rgb(hires.downsample(color, hires.PREVIEW_WIDTH)), "RGB").save(
                    bundle / "main_view.png")
                note = (f"exported {w}×{h} to {folder.as_posix()}. "
                        f"Game copy is in {game.as_posix()}. Bundle is {bundle.as_posix()}")
            latest_pointer().write_text(json.dumps({
                "bundle": str(bundle),
                "hires": str(folder),
                "game": str(game_dir),
            }, indent=1))
            shown = game_made if whole else made
            self.events.put(("exported", note, preview_thumbs(shown), str(bundle), str(folder)))
        except Exception as err:
            self.events.put(("error", f"export failed: {err}"))

    def _tile_stride(self):
        """Pixel size of one tile, and the step to the next shared rim."""
        mw, mh = self._tile_size()
        return mw, mh, max(mw - 1, 1), max(mh - 1, 1)

    def _tile_rect(self, tx, ty, mw, mh):
        z = self.zoom
        sx, sy = max(mw - 1, 1), max(mh - 1, 1)
        x0 = int(round(self.view.x + (tx * sx - self.cam[0]) * z))
        y0 = int(round(self.view.y + (ty * sy - self.cam[1]) * z))
        x1 = int(round(self.view.x + (tx * sx + mw - self.cam[0]) * z))
        y1 = int(round(self.view.y + (ty * sy + mh - self.cam[1]) * z))
        return pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _map_sheet(self):
        """One picture of every generated tile, sharing the rim, scaled as a whole."""
        if not self.atlas:
            return None
        mw, mh, sx, sy = self._tile_stride()
        tiles = []
        for (tx, ty), tile in self.atlas.items():
            surf = tile["surface"]
            if self.live_surface is not None and (tx, ty) == self.live_cell:
                surf = self.live_surface
            if surf is None:
                continue
            tiles.append((tx, ty, surf))
        if not tiles:
            return None
        key = tuple(sorted((tx, ty, id(surf)) for tx, ty, surf in tiles))
        cached = getattr(self, "_map_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        tx0 = min(tx for tx, _, _ in tiles)
        ty0 = min(ty for _, ty, _ in tiles)
        tx1 = max(tx for tx, _, _ in tiles)
        ty1 = max(ty for _, ty, _ in tiles)
        sheet = pygame.Surface(((tx1 - tx0) * sx + mw, (ty1 - ty0) * sy + mh))
        sheet.fill((10, 12, 16))
        for tx, ty, surf in sorted(tiles, key=lambda item: (item[1], item[0])):
            if surf.get_size() != (mw, mh):
                surf = pygame.transform.smoothscale(surf, (mw, mh))
            sheet.blit(surf, ((tx - tx0) * sx, (ty - ty0) * sy))
        origin = (tx0 * sx, ty0 * sy)
        self._map_cache = (key, sheet, origin)
        return sheet, origin

    def _draw_atlas(self):
        mw, mh, sx, sy = self._tile_stride()
        empty = set()
        for tx, ty in self.atlas:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (tx + dx, ty + dy)
                if n not in self.atlas:
                    empty.add(n)
        if self.live_cell is not None and self.live_cell not in self.atlas:
            empty.add(self.live_cell)
        for tx, ty in empty:
            dest = self._tile_rect(tx, ty, mw, mh)
            if not dest.colliderect(self.view):
                continue
            pygame.draw.rect(self.screen, (16, 18, 22), dest)
            pygame.draw.rect(self.screen, LINE, dest, 1)
            if self.live_surface is not None and (tx, ty) == self.live_cell:
                continue
            word = "generating" if self.busy and (tx, ty) == (self.tx, self.ty) else "click Generate"
            img = self.small.render(word, True, MUTED)
            self.screen.blit(img, img.get_rect(center=dest.center))
        made = self._map_sheet()
        if made is not None:
            sheet, origin = made
            z = self.zoom
            dest = pygame.Rect(
                int(round(self.view.x + (origin[0] - self.cam[0]) * z)),
                int(round(self.view.y + (origin[1] - self.cam[1]) * z)),
                max(1, int(round(sheet.get_width() * z))),
                max(1, int(round(sheet.get_height() * z))),
            )
            _blit_scaled(self.screen, sheet, dest, self.view)
        if (self.live_surface is not None and self.live_cell is not None
                and self.live_cell not in self.atlas):
            dest = self._tile_rect(*self.live_cell, mw, mh)
            _blit_scaled(self.screen, self.live_surface, dest, self.view)
        if self.atlas or self.live_cell is not None:
            dest = self._tile_rect(self.tx, self.ty, mw, mh)
            pygame.draw.rect(self.screen, ACCENT, dest, 1)

    def _map_span(self):
        """Pixel size of the picture strokes are drawn on."""
        if self.image_name in ("", "world") and (self.atlas or self.live_surface is not None):
            return self._tile_size()
        if self.image is not None:
            return self.image.get_width(), self.image.get_height()
        return hires.map_size(hires.PREVIEW_WIDTH)

    def _norm_to_screen(self, tx, ty, y, x):
        z = self.zoom
        mw, mh = self._map_span()
        atlas = self.image_name in ("", "world") and (self.atlas or self.live_surface is not None)
        if atlas:
            _mw, _mh, step_x, step_y = self._tile_stride()
            px = self.view.x + ((tx + x) * step_x - self.cam[0]) * z
            py = self.view.y + ((ty + y) * step_y - self.cam[1]) * z
        else:
            px = self.view.x + (x * mw - self.cam[0]) * z
            py = self.view.y + (y * mh - self.cam[1]) * z
        return px, py

    def _brush_px(self, radius_frac):
        mw = self._map_span()[0]
        return max(2, int(round(float(radius_frac) * mw * self.zoom)))

    def _pointer_on_atlas(self, pos):
        """`(tx, ty, y, x)` on a generated tile, or None."""
        if self.view3d and self.image_name in ("", "world"):
            return self._sheet_pointer(pos)
        if pos is None or not self.view.collidepoint(pos):
            return None
        if self.sculpt_bar.collidepoint(pos):
            return None
        z = max(self.zoom, 1e-6)
        wx = self.cam[0] + (pos[0] - self.view.x) / z
        wy = self.cam[1] + (pos[1] - self.view.y) / z
        atlas = self.image_name in ("", "world") and (self.atlas or self.live_surface is not None)
        if not atlas:
            if self.image is None or (self.tx, self.ty) not in self.atlas and not self.atlas:
                return None
            mw, mh = self.image.get_width(), self.image.get_height()
            if not (0 <= wx < mw and 0 <= wy < mh):
                return None
            return (self.tx, self.ty, wy / mh, wx / mw)
        mw, mh, sx, sy = self._tile_stride()
        tx = int(math.floor(wx / sx))
        ty = int(math.floor(wy / sy))

        def owns(cx, cy):
            if (cx, cy) not in self.atlas:
                return False
            return cx * sx <= wx < cx * sx + mw and cy * sy <= wy < cy * sy + mh

        if not owns(tx, ty):
            if owns(tx - 1, ty):
                tx -= 1
            elif owns(tx, ty - 1):
                ty -= 1
            elif owns(tx - 1, ty - 1):
                tx -= 1
                ty -= 1
        if (tx, ty) not in self.atlas:
            return None
        y = (wy - ty * sy) / sy
        x = (wx - tx * sx) / sx
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return (tx, ty, float(y), float(x))

    def _pointer_on_selected(self, pos):
        """Normalized (y, x) on the selected tile, or None."""
        hit = self._pointer_on_atlas(pos)
        if hit is None or (hit[0], hit[1]) != (self.tx, self.ty):
            return None
        return (hit[2], hit[3])

    def _view3d_dest(self):
        """The 3D picture fills the same area as the topography map."""
        return pygame.Rect(self.view)

    def _sheet_key(self):
        parts = []
        for (tx, ty), tile in self.atlas.items():
            world = tile.get("world")
            made = tile.get("made") or {}
            if world is None or "world" not in made:
                continue
            parts.append((tx, ty, id(world.height), id(made["world"])))
        return tuple(sorted(parts))

    def _terrain_sheet(self):
        """Height, color, and mask for every generated tile, sharing rims."""
        tiles = []
        for (tx, ty), tile in self.atlas.items():
            world = tile.get("world")
            made = tile.get("made") or {}
            if world is None or "world" not in made:
                continue
            tiles.append((tx, ty, np.asarray(world.surface, np.float32), made["world"]))
        return stitch_tiles(tiles)

    def _view3d_frame(self):
        """Cached perspective of every generated tile on one surface."""
        tiles = self._sheet_key()
        if not tiles:
            self._sheet_layout = None
            return None
        dest = self._view3d_dest()
        render_w, render_h = dest.w, dest.h
        txs = [tx for tx, _ty, _h, _c in tiles]
        tys = [ty for _tx, ty, _h, _c in tiles]
        span = max(max(txs) - min(txs) + 1, max(tys) - min(tys) + 1)
        steps = max(28, math.ceil(min(220, 140 * span) / 5))
        key = (tiles, render_w, render_h, steps,
               round(self.orbit[0], 3), round(self.orbit[1], 3), round(self.orbit[2], 3),
               round(self.look[0], 4), round(self.look[1], 4))
        cached = self._view3d_cache
        if cached is None or cached[0] != key:
            sheet = self._terrain_sheet()
            if sheet is None:
                self._sheet_layout = None
                return None
            height, color, mask, layout = sheet
            rgb, pick_y, pick_x = load_surface(
                height, color,
                self.orbit[0], self.orbit[1], self.orbit[2],
                render_w, render_h, steps=steps, origin=self.look, mask=mask)
            cached = (key, rgb, pick_y, pick_x, layout)
            self._view3d_cache = cached
        self._sheet_layout = cached[4]
        return dest, cached[1], cached[2], cached[3]

    def _sheet_to_tile(self, y, x):
        """Sheet pick in 0..1 to `(tx, ty, local_y, local_x)` on a real tile."""
        layout = self._sheet_layout
        if layout is None:
            return (self.tx, self.ty, float(y), float(x))
        tx0, ty0, nx, ny = layout
        fx = min(max(float(x), 0.0), 0.999999) * nx
        fy = min(max(float(y), 0.0), 0.999999) * ny
        ix = min(nx - 1, int(math.floor(fx)))
        iy = min(ny - 1, int(math.floor(fy)))
        tx = tx0 + ix
        ty = ty0 + iy
        if (tx, ty) not in self.atlas:
            return None
        return (tx, ty, fy - iy, fx - ix)

    def _picks_on_tile(self, pick_y, pick_x, tx, ty):
        """Pick maps in one tile's 0..1 space. Other tiles and sky stay -1."""
        layout = self._sheet_layout
        if layout is None:
            return pick_y, pick_x
        tx0, ty0, nx, ny = layout
        local_y = pick_y * ny - (ty - ty0)
        local_x = pick_x * nx - (tx - tx0)
        inside = ((pick_y >= 0) & (local_y >= -1e-3) & (local_y <= 1.001)
                  & (local_x >= -1e-3) & (local_x <= 1.001))
        return (np.where(inside, local_y, np.float32(-1)).astype(np.float32),
                np.where(inside, local_x, np.float32(-1)).astype(np.float32))

    def _sheet_pointer(self, pos):
        if pos is None or not self.view.collidepoint(pos) or self.sculpt_bar.collidepoint(pos):
            return None
        frame = self._view3d_frame()
        if frame is None:
            return None
        dest, _rgb, pick_y, pick_x = frame
        ix = int((pos[0] - dest.x) * pick_x.shape[1] / max(dest.w, 1))
        iy = int((pos[1] - dest.y) * pick_y.shape[0] / max(dest.h, 1))
        if not (0 <= iy < pick_y.shape[0] and 0 <= ix < pick_x.shape[1]):
            return None
        y = float(pick_y[iy, ix])
        x = float(pick_x[iy, ix])
        if y < 0.0 or x < 0.0:
            return None
        return self._sheet_to_tile(y, x)

    def _draw_view3d(self):
        frame = self._view3d_frame()
        if frame is None:
            return
        dest, rgb, pick_y, pick_x = frame
        groups = {}
        for cell, items in self.strokes.items():
            if cell not in self.atlas or self._strokes_baked(cell):
                continue
            groups.setdefault(cell, []).extend(items)
        if self.painting is not None:
            for cell, points in self.painting["runs"].items():
                groups.setdefault(cell, []).append({
                    "kind": self.painting["kind"],
                    "radius": self.painting["radius"],
                    "points": points,
                })
        cursor_hit = None
        if self.tool in PAINT_TOOLS and self.pointer is not None:
            cursor_hit = self._sheet_pointer(self.pointer)
            if cursor_hit is not None:
                groups.setdefault((cursor_hit[0], cursor_hit[1]), [])
        shown = rgb
        radius = self.brush.value / float(hires.PREVIEW_WIDTH)
        for (tx, ty), strokes in groups.items():
            tile = self.atlas.get((tx, ty))
            if tile is None:
                continue
            local_y, local_x = self._picks_on_tile(pick_y, pick_x, tx, ty)
            cursor = None
            if cursor_hit is not None and (cursor_hit[0], cursor_hit[1]) == (tx, ty):
                cursor = (cursor_hit[2], cursor_hit[3], radius, self.tool)
            shape = tile["world"].height.shape
            shown = mark_strokes(shown, local_y, local_x, strokes, cursor,
                                 y_aspect=shape[0] / max(1, shape[1]))
        surf = _surface(shown)
        if surf.get_size() != dest.size:
            surf = pygame.transform.smoothscale(surf, dest.size)
        self.screen.blit(surf, dest.topleft)

    def _remember_point(self, hit):
        runs = self.painting["runs"]
        run = runs.setdefault((hit[0], hit[1]), [])
        point = (float(hit[2]), float(hit[3]))
        if not run or run[-1] != point:
            run.append(point)
        self.painting["last"] = hit

    def _begin_stroke(self, pos):
        if not self.atlas:
            self.status = "Generate a preview, then draw on it."
            return True
        hit = self._pointer_on_atlas(pos)
        if hit is None:
            return False
        self._store_relief()
        self.painting = {
            "kind": self.tool,
            "metres": self._stroke_metres(),
            "radius": self.brush.value / float(hires.PREVIEW_WIDTH),
            "runs": {},
            "path": [(float(hit[0]) + float(hit[3]), float(hit[1]) + float(hit[2]))],
            "last": hit,
        }
        self._remember_point(hit)
        self._stroke_cell = (hit[0], hit[1])
        return True

    def _extend_stroke(self, pos):
        if self.painting is None:
            return
        hit = self._pointer_on_atlas(pos)
        if hit is None:
            return
        last = self.painting["last"]
        dy = (hit[1] + hit[2]) - (last[1] + last[2])
        dx = (hit[0] + hit[3]) - (last[0] + last[3])
        gap = max(0.003, 0.12 * float(self.painting["radius"]))
        if dx * dx + dy * dy < gap * gap:
            return
        for point in stroke_crossings(last, hit, self.atlas):
            self._remember_point(point)
        self.painting["path"].append(
            (float(hit[0]) + float(hit[3]), float(hit[1]) + float(hit[2])))
        self.painting["last"] = hit

    def _commit_stroke(self):
        stroke = self.painting
        self.painting = None
        path = stroke.get("path") if stroke else None
        if not path:
            return
        radius = float(stroke["radius"])
        cells = []
        for tx, ty in sorted(self.atlas):
            if not stroke_reaches(path, tx, ty, radius):
                continue
            local = [(ay - ty, ax - tx) for ax, ay in path]
            self.strokes.setdefault((tx, ty), []).append({
                "kind": stroke["kind"],
                "metres": float(stroke["metres"]),
                "radius": radius,
                "points": local,
            })
            cells.append((tx, ty))
        if not cells:
            return
        self._undo_stack.append(cells)
        self._stroke_cell = cells[-1]
        kind = KIND_NAME.get(stroke["kind"], stroke["kind"])
        if len(cells) == 1:
            where = f"tile {cells[0][0]},{cells[0][1]}"
        else:
            where = f"{len(cells)} tiles"
        if stroke["kind"] in ASSET_RANGE:
            amount = f"{stroke['metres']:.0f} m"
        else:
            amount = f"{stroke['metres'] / 1000:.2f} km"
        self.status = (f"Drew {kind}, {amount}, across {where}. "
                       "Generate preview to build it in.")

    def _undo_stroke(self):
        if self.painting is not None:
            self.painting = None
            self.status = "Cancelled the stroke being drawn."
            return
        cells = self._undo_stack.pop() if self._undo_stack else [(self.tx, self.ty)]
        removed = False
        for cell in cells:
            strokes = self.strokes.get(cell)
            if not strokes:
                continue
            strokes.pop()
            removed = True
            if not strokes:
                self.strokes.pop(cell, None)
        if removed:
            self.status = "Removed the last stroke. Generate preview to rebuild the tiles it crossed."

    def _clear_strokes(self):
        self.painting = None
        cell = (self.tx, self.ty)
        if cell not in self.strokes:
            return
        self.strokes.pop(cell, None)
        self.status = "Cleared strokes on this tile. Generate preview to rebuild it."

    def _sculpt_click(self, pos):
        if not self.sculpt_bar.collidepoint(pos):
            return False
        for name, button in self.tool_buttons.items():
            if button.rect.collidepoint(pos):
                self._set_tool(name)
                return True
        if self.undo_paint_btn.enabled and self.undo_paint_btn.rect.collidepoint(pos):
            self._undo_stroke()
            return True
        if self.view3d_btn.rect.collidepoint(pos):
            self._toggle_view3d()
            return True
        if self.clear_paint_btn.enabled and self.clear_paint_btn.rect.collidepoint(pos):
            self._clear_strokes()
            return True
        for slider in (self.relief_m, self.brush):
            if slider.rect.inflate(0, 18).collidepoint(pos):
                slider.grabbed = True
                slider.set_from(pos[0])
                self._store_relief()
                return True
        return True

    def _draw_paint(self):
        atlas = self.image_name in ("", "world") and (self.atlas or self.live_surface is not None)
        listed = []
        if atlas:
            cells = list(self.strokes.items())
        else:
            cells = [((self.tx, self.ty), self.strokes.get((self.tx, self.ty), ()))]
        for cell, items in cells:
            for stroke in items:
                listed.append((cell, stroke, stroke is not self.painting and not self._strokes_baked(cell)))
        if self.painting is not None:
            for cell, points in self.painting["runs"].items():
                listed.append((cell, {
                    "kind": self.painting["kind"],
                    "radius": self.painting["radius"],
                    "points": points,
                }, True))
        show_cursor = (self.tool in PAINT_TOOLS and not self.busy
                       and self.pointer is not None
                       and self._pointer_on_atlas(self.pointer) is not None)
        if not listed and not show_cursor:
            return
        self.screen.set_clip(self.view)
        if listed:
            overlay = pygame.Surface(self.view.size, pygame.SRCALPHA)
            for (tx, ty), stroke, pending in listed:
                fill, spine_rgb = STROKE_COLOR.get(stroke["kind"], STROKE_COLOR["valley"])
                spine = spine_rgb + ((230 if pending else 160),)
                radius = self._brush_px(stroke["radius"])
                pts = []
                for y, x in stroke["points"]:
                    if not (-0.02 <= float(y) <= 1.02 and -0.02 <= float(x) <= 1.02):
                        continue
                    sx, sy = self._norm_to_screen(tx, ty, y, x)
                    pts.append((int(sx - self.view.x), int(sy - self.view.y)))
                if not pts:
                    continue
                if pending:
                    for point in pts:
                        pygame.draw.circle(overlay, fill, point, radius)
                if len(pts) >= 2:
                    width = max(2, radius // 5) if pending else 2
                    pygame.draw.lines(overlay, spine, False, pts, width)
                elif pts:
                    pygame.draw.circle(overlay, spine, pts[0], max(3, radius // 6) if pending else 4)
            self.screen.blit(overlay, self.view.topleft)
        if show_cursor:
            tx, ty, y, x = self._pointer_on_atlas(self.pointer)
            sx, sy = self._norm_to_screen(tx, ty, y, x)
            radius = self._brush_px(self.brush.value / float(hires.PREVIEW_WIDTH))
            color = CURSOR_COLOR.get(self.tool, CURSOR_COLOR["valley"])
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius, 2)
        self.screen.set_clip(None)

    def draw_sculpt_bar(self):
        self._sync_relief_slider()
        bar = pygame.Rect(self.view.x + 8, self.view.bottom - 112, max(240, self.view.w - 16), 104)
        self.sculpt_bar = bar
        pygame.draw.rect(self.screen, (24, 26, 32), bar, border_radius=6)
        pygame.draw.rect(self.screen, LINE, bar, 1, border_radius=6)
        x = bar.x + 8
        y = bar.y + 8
        fills = {
            "view": (64, 104, 140),
            "mountain": (128, 78, 48),
            "valley": (40, 92, 122),
            "glacial": (70, 100, 120),
            "dune": (128, 104, 58),
            "coast": (36, 96, 92),
            "cave": (78, 60, 100),
        }
        row = ("view", "mountain", "valley")
        for name in row:
            button = self.tool_buttons[name]
            button.rect.topleft = (x, y)
            button.enabled = True
            button.draw(self.screen, self.small,
                        fill=fills[name] if self.tool == name else None)
            x += button.rect.w + 6
        self.undo_paint_btn.rect.topleft = (x, y)
        self.undo_paint_btn.enabled = bool(
            self.painting or self._undo_stack or self.strokes.get((self.tx, self.ty)))
        self.undo_paint_btn.draw(self.screen, self.small)
        x += self.undo_paint_btn.rect.w + 6
        self.view3d_btn.rect.topleft = (x, y)
        self.view3d_btn.enabled = True
        self.view3d_btn.draw(self.screen, self.small,
                             fill=(58, 110, 86) if self.view3d else None)
        x += self.view3d_btn.rect.w + 6
        self.clear_paint_btn.rect.topleft = (x, y)
        self.clear_paint_btn.enabled = bool(self.strokes.get((self.tx, self.ty)))
        self.clear_paint_btn.draw(self.screen, self.small)
        x = bar.x + 8
        y = bar.y + 42
        for name in ("glacial", "dune", "coast", "cave"):
            button = self.tool_buttons[name]
            button.rect.topleft = (x, y)
            button.enabled = True
            button.draw(self.screen, self.small,
                        fill=fills[name] if self.tool == name else None)
            x += button.rect.w + 6
        x += 8
        remain = max(180, bar.right - 10 - x)
        slider_w = max(80, (remain - 16) // 2)
        self.relief_m.rect = pygame.Rect(x, bar.y + 68, slider_w, 14)
        self.brush.rect = pygame.Rect(x + slider_w + 16, bar.y + 68, slider_w, 14)
        self.relief_m.draw(self.screen, self.small)
        self.brush.draw(self.screen, self.small)

    def draw_map(self):
        pygame.draw.rect(self.screen, (10, 12, 16), self.view)
        self.screen.set_clip(self.view)
        showing_3d = (self.view3d and not self.busy and self.image_name in ("", "world")
                      and (self.tx, self.ty) in self.atlas)
        if showing_3d:
            self._draw_view3d()
        elif self.image_name in ("", "world") and (self.atlas or self.live_surface is not None):
            self._draw_atlas()
        elif self.image is not None:
            surf, z = self.image, self.zoom
            mw, mh = surf.get_width(), surf.get_height()
            dest = pygame.Rect(int(self.view.x - self.cam[0] * z), int(self.view.y - self.cam[1] * z),
                               max(1, int(mw * z)), max(1, int(mh * z)))
            _blit_scaled(self.screen, surf, dest, self.view)
        self.screen.set_clip(None)
        if not showing_3d:
            self._draw_paint()
        label = self.big.render("surface" if showing_3d else (self.image_name or "preview"),
                                True, (255, 226, 150))
        self.screen.blit(label, (self.view.x + 12, self.view.y + 10))
        if showing_3d:
            note = "All generated tiles. Left-drag pans, right-drag rotates, scroll zooms."
        elif self.tool == "mountain":
            note = "Drag a mountain across generated tiles. Generate preview builds it in."
        elif self.tool == "valley":
            note = "Drag a valley across generated tiles. Generate preview builds it in."
        elif self.tool == "glacial":
            note = "Drag a glacier. Generate preview puts ice there and cuts the valley."
        elif self.tool == "dune":
            note = "Drag a dune field. Generate preview piles sand there."
        elif self.tool == "coast":
            note = "Drag along a shoreline. Generate preview cuts it and leaves a beach."
        elif self.tool == "cave":
            note = "Drag a cave. Generate preview opens a hollow there."
        elif self._paint_pending():
            note = f"tile {self.tx},{self.ty}   paint is waiting — Generate preview rebuilds it"
        else:
            note = f"tile {self.tx},{self.ty}   M mountain   V valley   H pan   D 3D"
        self.screen.blit(self.small.render(note, True, (255, 226, 150)),
                         (self.view.x + 12, self.view.y + 34))

    def draw_panel(self):
        panel_bottom = self.screen.get_height() - BAR_H
        panel = pygame.Rect(0, 0, 284, panel_bottom)
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.line(self.screen, LINE, (284, 0), (284, panel_bottom))
        self.screen.set_clip(panel)
        y = 18 - self.panel_scroll
        self.screen.blit(self.big.render("Terrain assets", True, TEXT), (16, y))
        y += 36
        self.screen.blit(self.small.render("seed", True, MUTED), (16, y))
        box = pygame.Rect(16, y + 18, 164, 28)
        pygame.draw.rect(self.screen, (16, 18, 22), box, border_radius=4)
        pygame.draw.rect(self.screen, LINE, box, 1, border_radius=4)
        self.screen.blit(self.font.render(self.seed_text or " ", True, TEXT), (24, y + 22))
        self.seed_box = box
        self.random_btn.rect.topleft = (188, y + 18)
        self.random_btn.enabled = not self.busy
        self.random_btn.draw(self.screen, self.small)
        y += 62
        self.screen.blit(self.small.render("export resolution", True, MUTED), (16, y))
        y += 18
        self.size_rects = []
        bw, bh, gap, cols = 46, 24, 4, 5
        for i, label in enumerate(hires.EXPORT_LABELS):
            row, col = divmod(i, cols)
            rect = pygame.Rect(16 + col * (bw + gap), y + row * (bh + gap), bw, bh)
            on = i == self.res_i
            pygame.draw.rect(self.screen, (46, 78, 108) if on else (36, 40, 48), rect, border_radius=3)
            img = self.small.render(label, True, TEXT)
            self.screen.blit(img, img.get_rect(center=rect.center))
            self.size_rects.append(rect)
        y += 2 * (bh + gap) + 8
        chosen = hires.EXPORT_WIDTHS[self.res_i]
        y = self._wrapped(hires.describe(chosen), 16, y, 252, WARN if chosen >= 131072 else MUTED)
        if self.detail_mode == "hierarchical":
            y = self._wrapped("Hierarchical export reruns erosion per tile.", 16, y, 252, WARN)
        y += 10
        for slider in (self.plates, self.land, self.temp):
            slider.rect.topleft = (16, y + 18)
            slider.rect.w = 248
            slider.draw(self.screen, self.small)
            y += 48
        y += 8
        for option, button in self.option_buttons.items():
            button.rect.topleft = (16, y)
            value = getattr(self, option)
            button.label = f"{OPTION_CAPTIONS[option]}: {value}  >"
            button.enabled = not self.busy
            button.draw(self.screen, self.small)
            y += 32
        y += 4
        self.advanced_btn.rect.topleft = (16, y)
        self.advanced_btn.label = f"Advanced layers: {'on' if self.advanced_export else 'off'}  >"
        self.advanced_btn.enabled = not self.busy
        self.advanced_btn.draw(self.screen, self.small)
        y += 38
        self.generate_btn.rect.topleft = (16, y)
        self.generate_btn.enabled = not self.busy or self.job == "preview"
        if self.busy and self.job == "preview":
            self.generate_btn.label = "Cancel"
        elif self._paint_pending():
            self.generate_btn.label = "Rebuild with paint"
        else:
            self.generate_btn.label = "Generate preview"
        self.generate_btn.draw(self.screen, self.font)
        y += 38
        self.export_btn.rect.topleft = (16, y)
        self.export_btn.label = "Exporting..." if self.busy and self.job == "export" else "Export maps"
        self.export_btn.draw(self.screen, self.font)
        y += 36
        self.embellish_btn.rect.topleft = (16, y)
        self.embellish_btn.enabled = not self.busy
        self.embellish_btn.draw(self.screen, self.font)
        y += 40
        y = self._wrapped(self.status, 16, y, 252)
        if self.panel_max_scroll > 0:
            y = self._wrapped("Scroll the panel for the rest of the controls.", 16, y, 252)
        self.screen.set_clip(None)
        content_bottom = y + self.panel_scroll
        self.panel_max_scroll = max(0, content_bottom - panel_bottom + 8)
        self.panel_scroll = int(min(self.panel_scroll, self.panel_max_scroll))

    def _wrapped(self, text, x, y, width, color=MUTED):
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if self.small.size(trial)[0] > width and line:
                self.screen.blit(self.small.render(line, True, color), (x, y))
                y += 18
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.small.render(line, True, color), (x, y))
            y += 18
        return y

    def draw_thumbs(self):
        names = ["world", "height", "biomes", "material mask", "flow", "canopy"]
        self.thumb_rects = []
        y = self.screen.get_height() - 128 - BAR_H
        x = 300
        for name in names:
            rect = pygame.Rect(x, y, 150, 112)
            pygame.draw.rect(self.screen, (12, 14, 18), rect)
            pygame.draw.rect(self.screen, ACCENT if name == self.image_name else LINE, rect, 1)
            image = self.thumbs.get(name)
            if image is not None:
                thumb = pygame.transform.smoothscale(_surface(image), (148, 88))
                self.screen.blit(thumb, (rect.x + 1, rect.y + 1))
            caption = self.small.render(name, True, TEXT)
            self.screen.blit(caption, (rect.x + 6, rect.bottom - 20))
            self.thumb_rects.append((rect, name))
            x += 162

    def draw_bar(self):
        width, height = self.screen.get_size()
        rect = pygame.Rect(0, height - BAR_H, width, BAR_H)
        pygame.draw.rect(self.screen, (12, 14, 18), rect)
        pygame.draw.line(self.screen, LINE, (0, rect.y), (width, rect.y))
        fraction, elapsed, eta, label = self.progress.snapshot()
        recent = self.progress.finished_at and (time.perf_counter() - self.progress.finished_at < 10)
        track = pygame.Rect(12, rect.y + 16, width - 24, 12)
        pygame.draw.rect(self.screen, (36, 40, 48), track, border_radius=4)
        if self.progress.active or recent:
            fill = track.copy()
            fill.w = max(0, int(track.w * fraction))
            if fill.w:
                pygame.draw.rect(self.screen, ACCENT, fill, border_radius=4)
            if self.progress.active:
                title = f"{self.progress.title}   {label}   {fraction * 100:.0f}%"
                if eta is None:
                    right = f"{format_duration(elapsed)} elapsed   estimating"
                else:
                    right = f"{format_duration(elapsed)} elapsed   {format_duration(eta)} left"
            else:
                title = f"{self.progress.title}   finished in {format_duration(elapsed)}"
                right = ""
        else:
            title, right = "Ready", ""
        self.screen.blit(self.small.render(title, True, TEXT), (12, rect.y + 1))
        if right:
            img = self.small.render(right, True, MUTED)
            self.screen.blit(img, (width - 12 - img.get_width(), rect.y + 1))

    def draw(self):
        self.screen.fill(BG)
        self.view = pygame.Rect(300, 12, self.screen.get_width() - 316,
                                self.screen.get_height() - 156 - BAR_H)
        self.draw_map()
        self.draw_sculpt_bar()
        self.draw_thumbs()
        self.draw_panel()
        self.draw_bar()
        pygame.display.flip()

    def on_click(self, pos):
        in_panel = pos[0] < 284 and pos[1] < self.screen.get_height() - BAR_H
        if in_panel and self.random_btn.rect.collidepoint(pos):
            if self.random_btn.enabled:
                self.randomize_seed()
                self.status = f"Seed {self.seed}."
            return
        if in_panel and self.seed_box.collidepoint(pos):
            return
        if in_panel and any(slider.rect.inflate(0, 10).collidepoint(pos) for slider in
               (self.plates, self.land, self.temp)):
            slider = next(s for s in (self.plates, self.land, self.temp)
                          if s.rect.inflate(0, 10).collidepoint(pos))
            slider.grabbed = True
            slider.set_from(pos[0])
            return
        if in_panel:
            for i, rect in enumerate(self.size_rects):
                if rect.collidepoint(pos) and not self.busy:
                    self.res_i = i
                    return
        if in_panel:
            for option, button in self.option_buttons.items():
                if button.enabled and button.rect.collidepoint(pos):
                    self.cycle_option(option)
                    return
            if self.advanced_btn.enabled and self.advanced_btn.rect.collidepoint(pos):
                self.advanced_export = not self.advanced_export
                self.status = "Advanced layers are now " + ("on" if self.advanced_export else "off")
                return
            if self.generate_btn.rect.collidepoint(pos):
                if self.busy and self.job == "preview":
                    self.cancel_preview()
                    return
                self._seed_from = (self.tx, self.ty)
                self._fit_when_done = True
                self.preset_dirty = True
                self.start()
                return
            if self.export_btn.rect.collidepoint(pos) and self.export_btn.enabled:
                self.export()
                return
            if self.embellish_btn.enabled and self.embellish_btn.rect.collidepoint(pos):
                self.open_embellish()
                return
        for rect, name in self.thumb_rects:
            if rect.collidepoint(pos) and name in self.thumbs:
                self._set_image(name, self.thumbs[name], fit=True)
                return
        if self._sculpt_click(pos):
            return
        if self.view.collidepoint(pos):
            if self.view3d and not self.busy and self.image_name in ("", "world"):
                self.pan_drag = (pos, self.look[0], self.look[1])
                return
            if not self.busy and self.tool in PAINT_TOOLS:
                self._begin_stroke(pos)
                return
            self.drag = pos

    def on_key(self, event):
        arrows = {pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
                  pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1)}
        if event.key in arrows:
            self._step(*arrows[event.key])
            return
        if event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL):
            self._undo_stroke()
            return
        if event.key == pygame.K_m:
            self._set_tool("mountain")
            return
        if event.key == pygame.K_v:
            self._set_tool("valley")
            return
        if event.key == pygame.K_i:
            self._set_tool("glacial")
            return
        if event.key == pygame.K_g:
            self._set_tool("dune")
            return
        if event.key == pygame.K_o:
            self._set_tool("coast")
            return
        if event.key == pygame.K_c:
            self._set_tool("cave")
            return
        if event.key == pygame.K_h:
            self._set_tool("view")
            return
        if event.key == pygame.K_d:
            self._toggle_view3d()
            return
        if event.key == pygame.K_BACKSPACE:
            self.seed_text = self.seed_text[:-1]
            self.preset_dirty = True
        elif event.key == pygame.K_RETURN:
            if self.busy and self.job == "preview":
                self.cancel_preview()
                return
            self._seed_from = (self.tx, self.ty)
            self._fit_when_done = True
            self.preset_dirty = True
            self.start()
        elif event.unicode.isascii() and event.unicode.isdigit() and len(self.seed_text) < 10:
            self.seed_text += event.unicode
            self.preset_dirty = True
        elif event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def run(self):
        while True:
            self.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                if event.type == pygame.KEYDOWN:
                    self.on_key(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    if (self.view.collidepoint(event.pos)
                            and not self.sculpt_bar.collidepoint(event.pos)):
                        if self.view3d and not self.busy and self.image_name in ("", "world"):
                            self.orbit_drag = (event.pos, self.orbit[0], self.orbit[1])
                        else:
                            self.drag = event.pos
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    painted = self.painting is not None
                    if painted:
                        self._commit_stroke()
                    was_drag = self.drag is not None
                    was_pan = self.pan_drag is not None
                    self.drag = None
                    self.pan_drag = None
                    for slider in (self.plates, self.land, self.temp):
                        if slider.grabbed:
                            self.preset_dirty = True
                        slider.grabbed = False
                    for slider in (self.relief_m, self.brush):
                        if slider.grabbed:
                            self._store_relief()
                        slider.grabbed = False
                    if was_drag and not painted and not was_pan:
                        self._release_map()
                if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                    self.drag = None
                    self.orbit_drag = None
                if event.type == pygame.MOUSEMOTION:
                    self.pointer = event.pos
                    grabbed = next((s for s in (self.plates, self.land, self.temp,
                                                self.relief_m, self.brush) if s.grabbed), None)
                    if grabbed:
                        grabbed.set_from(event.pos[0])
                        if grabbed in (self.relief_m, self.brush):
                            self._store_relief()
                    elif self.painting is not None:
                        self._extend_stroke(event.pos)
                    elif self.orbit_drag is not None:
                        origin, yaw, pitch = self.orbit_drag
                        dx = event.pos[0] - origin[0]
                        dy = event.pos[1] - origin[1]
                        self.orbit[0] = yaw + dx * 0.004
                        self.orbit[1] = float(np.clip(pitch - dy * 0.0035, 0.15, 1.25))
                    elif self.pan_drag is not None:
                        origin, look_x, look_z = self.pan_drag
                        dest = self._view3d_dest()
                        shift_x, shift_z = pan_delta(
                            self.orbit[0], self.orbit[1], self.orbit[2],
                            event.pos[0] - origin[0], event.pos[1] - origin[1],
                            dest.w, dest.h)
                        self.look[0] = float(np.clip(look_x + shift_x, -0.35, 1.35))
                        self.look[1] = float(np.clip(look_z + shift_z, -0.35, 1.35))
                    elif self.drag and self.image is not None:
                        self.cam[0] -= (event.pos[0] - self.drag[0]) / self.zoom
                        self.cam[1] -= (event.pos[1] - self.drag[1]) / self.zoom
                        self.drag = event.pos
                if event.type == pygame.MOUSEWHEEL:
                    pointer = pygame.mouse.get_pos()
                    if pointer[0] < 284:
                        self.panel_scroll = int(np.clip(
                            self.panel_scroll - event.y * 36, 0, self.panel_max_scroll))
                    elif (self.view3d and self.image_name in ("", "world")
                          and self.view.collidepoint(pointer)):
                        self.orbit[2] = float(np.clip(self.orbit[2] / (1.12 ** event.y), 0.4, 6.0))
                    elif self.sculpt_bar.collidepoint(pointer):
                        pass
                    elif self.image is not None:
                        self.zoom = float(np.clip(self.zoom * (1.12 ** event.y), 0.15, 12))
            try:
                self.draw()
            except MemoryError:
                pygame.display.set_caption("Terrain asset studio — out of memory")
                self.status = "The view ran out of memory. Export keeps going if it already started."
            except Exception as err:
                pygame.display.set_caption("Terrain asset studio — view failed")
                self.status = f"The view failed: {err}"
            self.clock.tick(60)

if __name__ == "__main__":
    Studio().run()
