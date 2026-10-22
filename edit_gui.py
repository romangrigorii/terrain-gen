"""Embellish an exported terrain map: sculpt, paint ground, and place objects.

Opens the folders written by gui.py. A tiled export asks for one tile first,
the same way a world is edited one section at a time. The first save keeps
the untouched rasters in authored/original.

    python edit_gui.py
    python edit_gui.py out/library/terrain/hires/<export>
"""
import math
import sys
from pathlib import Path

import numpy as np
import pygame

from terrain.embellish import (
    GROUPS, MATERIALS, MATERIAL_GROUP, MATERIAL_LABELS, MATERIAL_RGB,
    PROP_COLOR, PROPS, TileRequired, assets_root, discover, inspect, open_export,
)

BG = (18, 20, 24)
PANEL = (28, 31, 38)
LINE = (68, 74, 86)
TEXT = (230, 232, 236)
MUTED = (150, 156, 168)
ACCENT = (120, 176, 214)
BAR_H = 36
PANEL_W = 300

TOOL_BUTTONS = (
    ("pan", "Pan"), ("raise", "Raise"), ("lower", "Lower"),
    ("smooth", "Smooth"), ("flatten", "Flat"), ("grain", "Grain"),
    ("path", "Path"), ("water", "Water"), ("paint", "Paint"),
    ("place", "Place"), ("erase", "Erase"),
)
TOOL_ORDER = tuple(name for name, _label in TOOL_BUTTONS)
# label, minimum, maximum, unit, default. Unit "%" is stored on the map as a fraction.
AMOUNT = {
    "raise": ("height", 1, 800, "m", 40),
    "lower": ("depth", 1, 800, "m", 40),
    "smooth": ("strength", 5, 100, "%", 55),
    "flatten": ("strength", 5, 100, "%", 70),
    "grain": ("grain", 1, 200, "m", 12),
    "path": ("strength", 5, 100, "%", 65),
    "water": ("depth", 1, 80, "m", 12),
    "paint": ("strength", 5, 100, "%", 70),
    "place": ("size", 40, 220, "%", 100),
}
HINTS = {
    "pan": "Drag to move the map. The wheel zooms. F fits it.",
    "raise": "Drag to lift the ground. [ and ] change the brush. Shift changes it faster.",
    "lower": "Drag to dig. The brush size is in cells; the bar shows that width on the ground.",
    "smooth": "Drag to soften a ridge or a ditch.",
    "flatten": "Drag toward the height where you pressed. The stroke holds that level.",
    "grain": "Drag to break up a surface that looks too smooth.",
    "path": "Drag a road. It follows the hills and cuts the bumps, then paints a track.",
    "water": "Drag a pond at the height you first click. Shift-drag lifts the water paint off.",
    "paint": "Drag to blend a texture into the ground. M cycles this set, G the next set. Shift-drag erases it. I samples the cell under the cursor.",
    "place": "Click or drag to scatter objects. N cycles them. The brush is their spacing.",
    "erase": "Drag to remove trees, rocks, and buildings.",
}
CURSOR = {
    "raise": (196, 164, 110),
    "lower": (120, 164, 196),
    "smooth": (186, 186, 176),
    "flatten": (206, 176, 120),
    "grain": (176, 154, 112),
    "path": (186, 150, 96),
    "water": (90, 156, 196),
    "place": (170, 196, 150),
    "erase": (206, 120, 110),
}
KEYS = {
    pygame.K_1: "pan", pygame.K_2: "raise", pygame.K_3: "lower",
    pygame.K_4: "smooth", pygame.K_5: "flatten", pygame.K_6: "grain",
    pygame.K_7: "path", pygame.K_8: "water", pygame.K_9: "paint",
    pygame.K_0: "place", pygame.K_e: "erase",
}
VIEWS = ("shaded", "height", "materials")
VIEW_LABELS = {"shaded": "Shaded", "height": "Height", "materials": "Ground"}


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
        knob = int(self.rect.x + t * self.rect.w)
        pygame.draw.line(screen, (60, 64, 74), self.rect.midleft, self.rect.midright, 3)
        pygame.draw.line(screen, ACCENT, self.rect.midleft, (knob, self.rect.centery), 3)
        pygame.draw.circle(screen, (240, 244, 248), (knob, self.rect.centery), 6)
        screen.blit(font.render(f"{self.label}  {self.fmt(self.value)}", True, TEXT),
                    (self.rect.x, self.rect.y - 18))


class Button:
    def __init__(self, label, width=120):
        self.label = label
        self.rect = pygame.Rect(0, 0, width, 28)
        self.enabled = True

    def draw(self, screen, font, hot=False, fill=None, ink=None):
        if not self.enabled:
            color = (40, 42, 48)
        elif fill is not None:
            color = fill
        elif hot:
            color = (64, 104, 140)
        else:
            color = (46, 78, 108)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (230, 236, 242) if hot else LINE, self.rect, 1, border_radius=4)
        img = font.render(self.label, True, ink or (TEXT if self.enabled else MUTED))
        screen.blit(img, img.get_rect(center=self.rect.center))


def _ink(rgb):
    lum = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    return (24, 26, 28) if lum > 150 else (244, 246, 248)


def _draw_prop(screen, kind, x, y, scale, zoom):
    size = int(np.clip(7 * scale * min(max(zoom, 0.35), 2.2), 4, 22))
    color = PROP_COLOR.get(kind, (220, 220, 220))
    if kind in ("tree", "pine", "shrub"):
        radius = max(2, size // 2)
        pygame.draw.circle(screen, color, (int(x), int(y - size // 5)), radius)
        if kind != "shrub":
            pygame.draw.line(screen, (92, 70, 48), (int(x), int(y - size // 5)),
                             (int(x), int(y + size // 2)), 2)
    elif kind == "rock":
        pygame.draw.polygon(screen, color, (
            (x, y - size / 2), (x + size / 2, y), (x + size / 6, y + size / 2), (x - size / 2, y),
        ))
    elif kind == "ruin":
        pygame.draw.rect(screen, color, pygame.Rect(x - size / 2, y - size / 2, size, size), 2)
    else:
        pygame.draw.polygon(screen, color, (
            (x, y - size / 2), (x + size / 2, y), (x - size / 2, y),
        ))
        pygame.draw.rect(screen, color, pygame.Rect(x - size / 3, y, size * 2 / 3, size / 2), 1)


class Editor:
    def __init__(self, path=None):
        pygame.init()
        pygame.display.set_caption("Embellish terrain")
        self.screen = pygame.display.set_mode((1360, 860), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("segoeui,calibri,consolas", 16)
        self.small = pygame.font.SysFont("segoeui,calibri,consolas", 13)
        self.big = pygame.font.SysFont("segoeui,calibri,consolas", 20, bold=True)
        self.clock = pygame.time.Clock()
        self.root = assets_root()
        self.entries = discover(self.root)
        self.doc = None
        self.mode = "library"
        self.tiles = ()
        self.tile_folder = None
        self.tool = "pan"
        self.material = "sand"
        self.group = MATERIAL_GROUP[self.material]
        self.group_choice = {}
        for name in MATERIALS:
            self.group_choice.setdefault(MATERIAL_GROUP[name], name)
        self.group_choice[self.group] = self.material
        self.prop = "tree"
        self.view_mode = "shaded"
        self.amounts = {name: spec[4] for name, spec in AMOUNT.items()}
        self.brush = Slider("brush", 2, 160, 18, fmt=lambda v: f"{v} px")
        self.amount_slider = Slider("height", 1, 800, 40, fmt=lambda v: f"{v} m")
        self.open_btn = Button("Open folder", 252)
        self.save_btn = Button("Save", 123)
        self.close_btn = Button("Close", 123)
        self.restore_btn = Button("Restore original", 252)
        self.undo_btn = Button("Undo", 123)
        self.redo_btn = Button("Redo", 123)
        self.refresh_btn = Button("Refresh", 88)
        self.tool_btns = {name: Button(label, 84) for name, label in TOOL_BUTTONS}
        self.group_btns = {name: Button(label, 64) for name, label in GROUPS}
        self.material_btns = {name: Button(MATERIAL_LABELS[name], 84) for name in MATERIALS}
        self.prop_btns = {name: Button(name.title(), 84) for name in PROPS}
        self.view_btns = {name: Button(VIEW_LABELS[name], 84) for name in VIEWS}
        self.view = pygame.Rect(312, 12, 1036, 800)
        self.zoom = 1.0
        self.cam = [0.0, 0.0]
        self.list_scroll = 0
        self.tile_scroll = 0
        self.list_hits = []
        self.tile_hits = []
        self._pan_from = None
        self._last = None
        self._map_token = None
        self._map_surf = None
        self._map_origin = (0, 0)
        self._arm_close = False
        self._needs_fit = False
        self.status = ("Open an export to sculpt height, paint ground, cut paths, "
                       "and place trees, rocks, and buildings.")
        if not self.entries and path is None:
            self.status = f"No exports in {self.root}. Export maps from the asset studio, then open that folder."
        if path:
            self._open(Path(path))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._up(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._motion(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._wheel(event)
            self._draw()
            self.clock.tick(60)

    def _set_tool(self, name):
        if name not in TOOL_ORDER:
            return
        if self.doc is not None and self.doc.stroking:
            self.doc.end_stroke()
            self._last = None
        if self.tool in AMOUNT:
            self.amounts[self.tool] = int(self.amount_slider.value)
        self.tool = name
        self.brush.label = "spacing" if name == "place" else "brush"
        self._load_amount()
        self._arm_close = False

    def _load_amount(self):
        spec = AMOUNT.get(self.tool)
        if spec is None:
            return
        label, lo, hi, unit, default = spec
        self.amount_slider.label = label
        self.amount_slider.lo = lo
        self.amount_slider.hi = hi
        self.amount_slider.value = int(np.clip(self.amounts.get(self.tool, default), lo, hi))
        self.amount_slider.fmt = (lambda v: f"{v} m") if unit == "m" else (lambda v: f"{v}%")

    def _amount(self):
        unit = AMOUNT[self.tool][3]
        if unit == "%":
            return self.amount_slider.value / 100.0
        return float(self.amount_slider.value)

    def _open(self, path, tile=None):
        if self.doc is not None and self.doc.dirty:
            self.status = "Save or close this map before opening another."
            return
        try:
            doc = open_export(path, tile=tile)
        except TileRequired as err:
            self.doc = None
            self.mode = "tiles"
            self.tile_folder = err.entry.folder
            self.tiles = err.entry.tiles
            self.tile_scroll = 0
            self._map_token = None
            self.status = (f"{err.entry.folder.name} has {len(self.tiles)} tiles. "
                           "Choose one section to embellish.")
            return
        except (OSError, ValueError) as err:
            self.status = str(err)
            return
        self.doc = doc
        self.mode = "edit"
        self._map_token = None
        self._needs_fit = True
        self._arm_close = False
        self.status = doc.summary
        try:
            folder = doc.folder.resolve()
            known = {entry.folder.resolve() for entry in self.entries}
        except OSError:
            folder, known = doc.folder, set()
        if folder not in known:
            try:
                self.entries.insert(0, inspect(doc.folder, self.root))
            except ValueError:
                pass

    def _browse(self):
        if self.doc is not None and self.doc.dirty:
            self.status = "Save or close this map before opening another."
            return
        chosen = self._ask_folder()
        if chosen:
            self._open(Path(chosen))

    def _ask_folder(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            self.status = "The folder dialog is unavailable. Start with python edit_gui.py <export folder>."
            return None
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        chosen = filedialog.askdirectory(title="Open an exported terrain folder")
        root.destroy()
        return chosen or None

    def _save(self):
        if self.doc is None:
            return
        try:
            self.status = self.doc.save()
        except (OSError, ValueError, RuntimeError) as err:
            self.status = f"Save failed: {err}"

    def _restore(self):
        if self.doc is None:
            return
        try:
            self.doc = self.doc.restore()
        except (OSError, ValueError) as err:
            self.status = str(err)
            return
        self._map_token = None
        self.doc.render(self.view_mode)
        self.status = "Restored the maps from the first save."

    def _close(self):
        if self.mode == "tiles":
            self.mode = "library"
            self.tiles = ()
            self.status = "Choose an export."
            return
        if self.doc is None:
            return
        if self.doc.dirty and not self._arm_close:
            self._arm_close = True
            self.status = "Unsaved edits. Click Close again to drop them."
            return
        self.doc = None
        self.mode = "library"
        self._arm_close = False
        self._map_token = None
        self.status = "Choose an export."

    def _rescan(self):
        self.entries = discover(self.root)
        self.list_scroll = 0
        self.status = f"{len(self.entries)} exports in {self.root}."

    def _undo(self):
        if self.doc is not None and self.doc.undo():
            self._map_token = None
            self.status = "Undid the last stroke."

    def _redo(self):
        if self.doc is not None and self.doc.redo():
            self._map_token = None
            self.status = "Redid the stroke."

    def _fit(self):
        if self.doc is None:
            return
        height, width = self.doc.height.shape
        self.zoom = min(self.view.w / max(width, 1), self.view.h / max(height, 1))
        self.zoom = float(max(self.zoom, 0.05))
        self.cam[0] = -(self.view.w / self.zoom - width) / 2
        self.cam[1] = -(self.view.h / self.zoom - height) / 2
        self._map_token = None

    def _cell(self, pos):
        x = (pos[0] - self.view.x) / self.zoom + self.cam[0]
        y = (pos[1] - self.view.y) / self.zoom + self.cam[1]
        return y, x

    def _inside(self, y, x):
        if self.doc is None:
            return False
        height, width = self.doc.height.shape
        return math.isfinite(y) and math.isfinite(x) and 0 <= y < height and 0 <= x < width

    def _names_in(self, group):
        return [name for name in MATERIALS if MATERIAL_GROUP[name] == group]

    def _remember_material(self):
        self.group = MATERIAL_GROUP[self.material]
        self.group_choice[self.group] = self.material

    def _set_group(self, group):
        if MATERIAL_GROUP.get(self.material) == self.group:
            self.group_choice[self.group] = self.material
        self.group = group
        names = self._names_in(group)
        chosen = self.group_choice.get(group)
        self.material = chosen if chosen in names else names[0]

    def _cycle_material(self, step=1):
        names = self._names_in(self.group)
        if not names:
            return
        if self.material not in names:
            self.material = names[0]
        else:
            self.material = names[(names.index(self.material) + step) % len(names)]
        self.group_choice[self.group] = self.material

    def _cycle_group(self, step=1):
        ids = [name for name, _label in GROUPS]
        idx = ids.index(self.group) if self.group in ids else -1
        self._set_group(ids[(idx + step) % len(ids)])

    def _sample_ground(self, pos):
        if self.doc is None:
            return
        y, x = self._cell(pos)
        if not self._inside(y, x):
            return
        weights = np.asarray(self.doc.materials[int(y), int(x)], np.float32)
        if float(weights.sum()) < 0.2:
            self.status = "open ground"
            return
        self.material = MATERIALS[int(np.argmax(weights))]
        self._remember_material()
        self.status = MATERIAL_LABELS[self.material]

    def _draw_textures(self, y):
        for index, (name, _label) in enumerate(GROUPS):
            row, col = divmod(index, 4)
            button = self.group_btns[name]
            button.rect.topleft = (16 + col * 68, y + row * 32)
            button.draw(self.screen, self.small, hot=self.group == name)
        y += 32 * math.ceil(len(GROUPS) / 4) + 4
        shown = self._names_in(self.group)
        shown_set = set(shown)
        for name, button in self.material_btns.items():
            if name not in shown_set:
                button.rect = pygame.Rect(0, -40, 0, 0)
        for index, name in enumerate(shown):
            row, col = divmod(index, 3)
            button = self.material_btns[name]
            button.rect.topleft = (16 + col * 90, y + row * 32)
            swatch = tuple(int(channel) for channel in MATERIAL_RGB[MATERIALS.index(name)])
            button.draw(self.screen, self.small, hot=self.material == name,
                        fill=swatch, ink=_ink(swatch))
        y += 32 * math.ceil(len(shown) / 3) + 6
        return y

    def _stroke_down(self, pos):
        y, x = self._cell(pos)
        if not self._inside(y, x) or self.tool == "pan":
            self._pan_from = pos
            return
        tool = self.tool
        amount = self._amount()
        erase = False
        if tool == "water" and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            tool, amount = "dry", 1.0
        elif tool == "paint" and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            erase = True
        try:
            self.doc.begin_stroke(tool, y, x, float(self.brush.value), amount,
                                  self.material, self.prop, erase=erase)
        except ValueError as err:
            self.status = str(err)
            return
        self._last = (y, x)
        self.doc.dab(y, x)
        self._map_token = None

    def _stroke_to(self, pos):
        y, x = self._cell(pos)
        if not self._inside(y, x) or self._last is None:
            return
        ly, lx = self._last
        dist = math.hypot(y - ly, x - lx)
        step = max(0.75, self.brush.value * 0.28)
        count = int(dist / step)
        if count < 1:
            return
        for i in range(1, count + 1):
            t = i / count
            self.doc.dab(ly + (y - ly) * t, lx + (x - lx) * t)
        self._last = (y, x)
        self._map_token = None

    def _key(self, event):
        ctrl = event.mod & pygame.KMOD_CTRL
        if event.key == pygame.K_ESCAPE:
            if self.doc is not None and self.doc.stroking:
                self.doc.end_stroke()
                self._last = None
            elif self.mode == "tiles":
                self._close()
            else:
                self._set_tool("pan")
            return
        if ctrl and event.key == pygame.K_s:
            self._save()
            return
        if ctrl and event.key == pygame.K_z:
            if event.mod & pygame.KMOD_SHIFT:
                self._redo()
            else:
                self._undo()
            return
        if ctrl and event.key == pygame.K_y:
            self._redo()
            return
        if event.key == pygame.K_f:
            self._fit()
            return
        if event.key == pygame.K_v and self.doc is not None:
            self.view_mode = VIEWS[(VIEWS.index(self.view_mode) + 1) % len(VIEWS)]
            self.doc.render(self.view_mode)
            self._map_token = None
            return
        if event.key == pygame.K_m and self.tool == "paint":
            step = -1 if event.mod & pygame.KMOD_SHIFT else 1
            self._cycle_material(step)
            return
        if not ctrl and event.key == pygame.K_g and self.tool == "paint":
            self._cycle_group(-1 if event.mod & pygame.KMOD_SHIFT else 1)
            return
        if not ctrl and event.key == pygame.K_i and self.tool == "paint":
            self._sample_ground(pygame.mouse.get_pos())
            return
        if event.key == pygame.K_n and self.tool == "place":
            self.prop = PROPS[(PROPS.index(self.prop) + 1) % len(PROPS)]
            return
        if event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            step = 8 if event.mod & pygame.KMOD_SHIFT else 1
            if event.key == pygame.K_LEFTBRACKET:
                step = -step
            self.brush.value = int(np.clip(self.brush.value + step, self.brush.lo, self.brush.hi))
            return
        if event.key in KEYS and not ctrl:
            self._set_tool(KEYS[event.key])

    def _down(self, event):
        pos = event.pos
        if event.button == 1:
            if pos[0] < PANEL_W:
                self._panel(pos)
                return
            if self.mode == "tiles" and self._pick_tile(pos):
                return
            if self.mode == "edit" and self.doc is not None and self.view.collidepoint(pos):
                self._stroke_down(pos)
            return
        if event.button in (2, 3) and self.view.collidepoint(pos):
            if self.doc is not None and self.doc.stroking:
                self.doc.end_stroke()
                self._last = None
            self._pan_from = pos

    def _up(self, event):
        if event.button == 1:
            self.brush.grabbed = False
            self.amount_slider.grabbed = False
            self._pan_from = None
            if self.doc is not None and self.doc.stroking:
                self.doc.end_stroke()
                self._last = None
                self._map_token = None
        elif event.button in (2, 3):
            self._pan_from = None

    def _motion(self, event):
        if self.brush.grabbed:
            self.brush.set_from(event.pos[0])
            return
        if self.amount_slider.grabbed:
            self.amount_slider.set_from(event.pos[0])
            if self.tool in self.amounts:
                self.amounts[self.tool] = int(self.amount_slider.value)
            return
        if self._pan_from is not None:
            self.cam[0] -= (event.pos[0] - self._pan_from[0]) / max(self.zoom, 1e-6)
            self.cam[1] -= (event.pos[1] - self._pan_from[1]) / max(self.zoom, 1e-6)
            self._pan_from = event.pos
            return
        if self.doc is not None and self.doc.stroking:
            self._stroke_to(event.pos)

    def _wheel(self, event):
        pos = pygame.mouse.get_pos()
        if pos[0] < PANEL_W:
            self.list_scroll = max(0, self.list_scroll - event.y * 28)
            return
        if self.mode == "tiles":
            self.tile_scroll = max(0, self.tile_scroll - event.y * 48)
            return
        if self.doc is None or not self.view.collidepoint(pos):
            return
        y, x = self._cell(pos)
        self.zoom = float(np.clip(self.zoom * (1.12 ** event.y), 0.05, 32))
        self.cam[0] = x - (pos[0] - self.view.x) / self.zoom
        self.cam[1] = y - (pos[1] - self.view.y) / self.zoom
        self._map_token = None

    def _panel(self, pos):
        hit_close = self.close_btn.enabled and self.close_btn.rect.collidepoint(pos)
        if not hit_close:
            self._arm_close = False
        if self.open_btn.rect.collidepoint(pos):
            self._browse()
            return
        if hit_close:
            self._close()
            return
        if self.save_btn.enabled and self.save_btn.rect.collidepoint(pos):
            self._save()
            return
        if self.restore_btn.enabled and self.restore_btn.rect.collidepoint(pos):
            self._restore()
            return
        if self.undo_btn.enabled and self.undo_btn.rect.collidepoint(pos):
            self._undo()
            return
        if self.redo_btn.enabled and self.redo_btn.rect.collidepoint(pos):
            self._redo()
            return
        if self.refresh_btn.enabled and self.refresh_btn.rect.collidepoint(pos):
            self._rescan()
            return
        if self.brush.rect.inflate(0, 10).collidepoint(pos):
            self.brush.grabbed = True
            self.brush.set_from(pos[0])
            return
        if self.tool in AMOUNT and self.amount_slider.rect.inflate(0, 10).collidepoint(pos):
            self.amount_slider.grabbed = True
            self.amount_slider.set_from(pos[0])
            self.amounts[self.tool] = int(self.amount_slider.value)
            return
        for name, button in self.tool_btns.items():
            if button.rect.collidepoint(pos):
                self._set_tool(name)
                return
        if self.tool == "paint":
            for name, button in self.group_btns.items():
                if button.rect.collidepoint(pos):
                    self._set_group(name)
                    return
            for name, button in self.material_btns.items():
                if button.rect.collidepoint(pos):
                    self.material = name
                    self._remember_material()
                    return
        if self.tool == "place":
            for name, button in self.prop_btns.items():
                if button.rect.collidepoint(pos):
                    self.prop = name
                    return
        for name, button in self.view_btns.items():
            if button.rect.collidepoint(pos) and self.doc is not None:
                self.view_mode = name
                self.doc.render(name)
                self._map_token = None
                return
        for rect, entry in self.list_hits:
            if rect.collidepoint(pos):
                self._open(entry.folder)
                return

    def _pick_tile(self, pos):
        for rect, tile in self.tile_hits:
            if rect.collidepoint(pos):
                self._open(self.tile_folder, tile=(tile.y0, tile.x0))
                return True
        return False

    def _draw(self):
        width, height = self.screen.get_size()
        self.view = pygame.Rect(PANEL_W + 12, 12, max(10, width - PANEL_W - 24),
                                max(10, height - BAR_H - 24))
        if self._needs_fit and self.doc is not None:
            self._fit()
            self._needs_fit = False
        self.screen.fill(BG)
        self._draw_view()
        self._draw_panel()
        self.screen.set_clip(None)
        self._draw_bar()
        star = " *" if self.doc is not None and self.doc.dirty else ""
        pygame.display.set_caption("Embellish terrain" + star)
        pygame.display.flip()

    def _draw_view(self):
        pygame.draw.rect(self.screen, (14, 16, 18), self.view)
        if self.mode == "tiles":
            self._draw_tiles()
            return
        if self.doc is None or self.doc.height is None:
            self._draw_empty()
            return
        self._draw_map()

    def _draw_empty(self):
        lines = (
            "Export a map from the asset studio.",
            "Open it here to sculpt the height, paint the ground,",
            "cut paths and ponds, and place trees, rocks, and buildings.",
        )
        y = self.view.centery - 36
        for line in lines:
            img = self.font.render(line, True, MUTED)
            self.screen.blit(img, img.get_rect(centerx=self.view.centerx, y=y))
            y += 26

    def _draw_tiles(self):
        self.tile_hits = []
        self.screen.set_clip(self.view)
        self.screen.blit(self.font.render("Choose a terrain tile", True, TEXT),
                         (self.view.x + 16, self.view.y + 14))
        cols = max(1, (self.view.w - 32) // 180)
        row_h, col_w = 48, 180
        rows = math.ceil(len(self.tiles) / cols) if self.tiles else 0
        self.tile_scroll = int(min(self.tile_scroll, max(0, rows * row_h + 48 - self.view.h)))
        for index, tile in enumerate(self.tiles):
            row, col = divmod(index, cols)
            rect = pygame.Rect(self.view.x + 16 + col * col_w,
                               self.view.y + 46 + row * row_h - self.tile_scroll,
                               col_w - 12, row_h - 8)
            if rect.bottom < self.view.top or rect.top > self.view.bottom:
                continue
            pygame.draw.rect(self.screen, (46, 78, 108), rect, border_radius=4)
            img = self.small.render(tile.label, True, TEXT)
            self.screen.blit(img, img.get_rect(center=rect.center))
            self.tile_hits.append((rect, tile))
        self.screen.set_clip(None)

    def _draw_map(self):
        if self.doc.rgb is None or self.doc.rgb_mode != self.view_mode:
            self.doc.render(self.view_mode)
        rgb = self.doc.rgb
        height, width = rgb.shape[:2]
        x0 = max(0, int(math.floor(self.cam[0])))
        y0 = max(0, int(math.floor(self.cam[1])))
        x1 = min(width, int(math.ceil(self.cam[0] + self.view.w / self.zoom)) + 1)
        y1 = min(height, int(math.ceil(self.cam[1] + self.view.h / self.zoom)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        token = (self.doc.view_token, self.view_mode, y0, y1, x0, x1, round(self.zoom, 3))
        if token != self._map_token:
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            surface = pygame.surfarray.make_surface(np.transpose(crop, (1, 0, 2)))
            tw = max(1, int(round(crop.shape[1] * self.zoom)))
            th = max(1, int(round(crop.shape[0] * self.zoom)))
            if (tw, th) != surface.get_size():
                scaler = pygame.transform.smoothscale if self.zoom < 1.25 else pygame.transform.scale
                surface = scaler(surface, (tw, th))
            self._map_surf = surface
            self._map_origin = (x0, y0)
            self._map_token = token
        ox, oy = self._map_origin
        dest = (self.view.x + (ox - self.cam[0]) * self.zoom,
                self.view.y + (oy - self.cam[1]) * self.zoom)
        self.screen.set_clip(self.view)
        self.screen.blit(self._map_surf, dest)
        for prop in self.doc.props:
            sx = self.view.x + (prop["x"] - self.cam[0]) * self.zoom
            sy = self.view.y + (prop["y"] - self.cam[1]) * self.zoom
            if not self.view.collidepoint((sx, sy)):
                continue
            _draw_prop(self.screen, prop["kind"], sx, sy, prop["scale"], self.zoom)
        pointer = pygame.mouse.get_pos()
        if self.tool != "pan" and self.view.collidepoint(pointer):
            color = CURSOR.get(self.tool, (220, 220, 220))
            if self.tool == "paint":
                color = tuple(int(channel) for channel in MATERIAL_RGB[MATERIALS.index(self.material)])
            radius = max(2, int(round(self.brush.value * self.zoom)))
            pygame.draw.circle(self.screen, color, pointer, radius, 1)
        self.screen.set_clip(None)

    def _draw_panel(self):
        panel_bottom = self.screen.get_height() - BAR_H
        panel = pygame.Rect(0, 0, PANEL_W, panel_bottom)
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.line(self.screen, LINE, (PANEL_W, 0), (PANEL_W, panel_bottom))
        self.screen.set_clip(panel)
        y = 14
        self.screen.blit(self.big.render("Embellish", True, TEXT), (16, y))
        y += 28
        editing = self.doc is not None
        self.open_btn.rect.topleft = (16, y)
        self.open_btn.draw(self.screen, self.font)
        y += 34
        self.save_btn.rect.topleft = (16, y)
        self.save_btn.enabled = editing
        self.save_btn.draw(self.screen, self.small)
        self.close_btn.rect.topleft = (145, y)
        self.close_btn.enabled = editing or self.mode == "tiles"
        self.close_btn.label = "Back" if self.mode == "tiles" else "Close"
        self.close_btn.draw(self.screen, self.small)
        y += 32
        self.restore_btn.rect.topleft = (16, y)
        self.restore_btn.enabled = editing and self.doc.has_original
        self.restore_btn.draw(self.screen, self.small)
        y += 32
        self.undo_btn.rect.topleft = (16, y)
        self.undo_btn.enabled = editing and self.doc.can_undo
        self.undo_btn.draw(self.screen, self.small)
        self.redo_btn.rect.topleft = (145, y)
        self.redo_btn.enabled = editing and self.doc.can_redo
        self.redo_btn.draw(self.screen, self.small)
        y += 38
        for index, name in enumerate(TOOL_ORDER):
            row, col = divmod(index, 3)
            button = self.tool_btns[name]
            button.rect.topleft = (16 + col * 90, y + row * 32)
            button.draw(self.screen, self.small, hot=self.tool == name)
        y += 32 * math.ceil(len(TOOL_ORDER) / 3) + 8
        if self.tool != "paint":
            for button in self.material_btns.values():
                button.rect = pygame.Rect(0, -40, 0, 0)
            for button in self.group_btns.values():
                button.rect = pygame.Rect(0, -40, 0, 0)
        if self.tool != "place":
            for button in self.prop_btns.values():
                button.rect = pygame.Rect(0, -40, 0, 0)
        if self.tool == "paint":
            y = self._draw_textures(y)
        elif self.tool == "place":
            for index, name in enumerate(PROPS):
                row, col = divmod(index, 3)
                button = self.prop_btns[name]
                button.rect.topleft = (16 + col * 90, y + row * 32)
                button.draw(self.screen, self.small, hot=self.prop == name)
            y += 68
        self.brush.rect.topleft = (16, y + 18)
        self.brush.rect.w = 252
        self.brush.draw(self.screen, self.small)
        y += 46
        if self.tool in AMOUNT:
            self.amount_slider.rect.topleft = (16, y + 18)
            self.amount_slider.rect.w = 252
            self.amount_slider.draw(self.screen, self.small)
            y += 46
        else:
            self.amount_slider.rect = pygame.Rect(0, -40, 0, 0)
        for index, name in enumerate(VIEWS):
            button = self.view_btns[name]
            button.rect.topleft = (16 + index * 90, y)
            button.enabled = editing
            button.draw(self.screen, self.small, hot=self.view_mode == name)
        y += 36
        y = self._wrap(HINTS[self.tool], 16, y, 268)
        y = self._wrap(self.status, 16, y + 4, 268, TEXT)
        y += 8
        self.screen.blit(self.small.render("Exports", True, MUTED), (16, y + 4))
        self.refresh_btn.rect.topleft = (196, y)
        self.refresh_btn.enabled = True
        self.refresh_btn.draw(self.screen, self.small)
        y += 32
        self._draw_list(y, panel_bottom)
        self.screen.set_clip(None)

    def _draw_list(self, top, bottom):
        self.list_hits = []
        row_h = 24
        available = bottom - top
        if available < row_h or not self.entries:
            if not self.entries:
                self.screen.blit(self.small.render("No exports yet.", True, MUTED), (16, top))
            return
        max_scroll = max(0, len(self.entries) * row_h - available)
        self.list_scroll = int(min(max(0, self.list_scroll), max_scroll))
        clip = pygame.Rect(8, top, PANEL_W - 16, available)
        self.screen.set_clip(clip)
        for index, entry in enumerate(self.entries):
            item_top = top + index * row_h - self.list_scroll
            if item_top + row_h < top or item_top > bottom:
                continue
            rect = pygame.Rect(16, item_top, 268, row_h - 3)
            selected = self.doc is not None and entry.folder == self.doc.folder
            pygame.draw.rect(self.screen, (46, 78, 108) if selected else (36, 40, 48),
                             rect, border_radius=3)
            img = self.small.render(entry.label, True, TEXT)
            self.screen.blit(img, (rect.x + 8, rect.y + 3))
            self.list_hits.append((rect, entry))

    def _draw_bar(self):
        height = self.screen.get_height()
        width = self.screen.get_width()
        rect = pygame.Rect(0, height - BAR_H, width, BAR_H)
        pygame.draw.rect(self.screen, PANEL, rect)
        pygame.draw.line(self.screen, LINE, (0, rect.y), (width, rect.y))
        parts = []
        if self.doc is not None:
            parts.append(self.doc.folder.name)
            parts.append(f"{self.doc.height.shape[1]}×{self.doc.height.shape[0]}")
            reach = 2 * self.brush.value * self.doc.cell_m
            if reach >= 1000:
                parts.append(f"brush {reach / 1000:.1f} km")
            else:
                parts.append(f"brush {reach:.0f} m")
            pointer = pygame.mouse.get_pos()
            if self.view.collidepoint(pointer):
                parts.append(self.doc.describe_cell(*self._cell(pointer)))
            parts.append("unsaved" if self.doc.dirty else "saved")
        else:
            parts.append("Choose an export")
        self.screen.blit(self.small.render("   ".join(parts), True, TEXT), (12, rect.y + 8))

    def _wrap(self, text, x, y, width, color=MUTED):
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if self.small.size(trial)[0] > width and line:
                self.screen.blit(self.small.render(line, True, color), (x, y))
                y += 16
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.small.render(line, True, color), (x, y))
            y += 16
        return y


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    Editor(args[0] if args else None).run()


if __name__ == "__main__":
    main()
