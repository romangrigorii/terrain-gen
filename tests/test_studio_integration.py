"""Compatibility checks for the original GUI/highres workflow with terrain v2."""
import importlib
import json
import queue
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hires
from terrain.presets import make_config
from terrain.world import generate


def _mock_gui(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygame", types.ModuleType("pygame"))
    sys.modules.pop("gui", None)
    return importlib.import_module("gui")


def _fields():
    y, x = np.mgrid[:24, :32]
    height = (0.3 * np.cos(x / 7) + 0.02 * y).astype("float32")
    mask = np.tile(np.array([50, 100, 150, 200], np.uint8), (24, 32, 1))
    flow = np.tile(np.array([180, 0, 0], np.uint8), (24, 32, 1))
    return height, mask, flow


def _read_height(path, manifest):
    codes = np.array(Image.open(path)).astype(np.float32)
    return manifest["lo"] + codes * manifest["units_per_code"]


def test_preset_options_build_config(monkeypatch):
    gui = _mock_gui(monkeypatch)
    settings = dict(preset="alpine", seed=17, n_plates=19,
                    land_fraction=0.4, temp_offset=4.0,
                    erosion_model="thermal", wind_mode="westerly",
                    depression_method="iterative", noise_backend="numpy",
                    detail_level="off", advanced_export=True)
    cfg = gui.config_from_settings(settings, 128, 96)
    assert cfg.collision_h == make_config("alpine").collision_h
    assert cfg.land_fraction == 0.4
    assert cfg.erosion_model == "thermal"
    assert cfg.wind_mode == "westerly"
    assert cfg.terrain_model == "classic"
    layered = dict(settings, terrain_model="full", detail_mode="hierarchical")
    cfg = gui.config_from_settings(layered, 128, 96)
    assert cfg.terrain_model == "full"
    assert not hasattr(cfg, "detail_mode")
    assert (cfg.width, cfg.height) == (128, 96)


def test_preset_button_resets_main_sliders(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.preset = "earthlike"
    studio.plates = types.SimpleNamespace(value=18)
    studio.land = types.SimpleNamespace(value=29)
    studio.temp = types.SimpleNamespace(value=0)
    studio.preset_dirty = False
    studio.cycle_option("preset")
    assert studio.preset == "archipelago"
    assert studio.land.value == 15
    assert studio.preset_dirty


def test_worker_preview_and_legacy_assets(monkeypatch, tmp_path):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    settings = dict(preset="archipelago", seed=8, n_plates=10,
                    land_fraction=0.18, temp_offset=0.,
                    erosion_model="thermal", wind_mode="belts",
                    depression_method="iterative", noise_backend="numpy",
                    detail_level="off", advanced_export=True)
    cfg = gui.config_from_settings(settings, 128, 96)
    studio._worker(cfg, None, settings, (0, 0))
    events = []
    while not studio.events.empty():
        events.append(studio.events.get_nowait())
    errors = [ev for ev in events if ev[0] == "error"]
    assert not errors, errors
    done = next(ev for ev in events if ev[0] == "done")
    world, made = done[1:3]
    assert world.height.shape == (96, 128)
    assert made["_mask"].shape == (96, 128, 4)
    assert set(("flow", "world", "canopy", "biomes")) <= set(made)
    gui.write_game_copy(made, tmp_path, world)
    assert (tmp_path / "zemia_height16.png").exists()
    assert json.loads((tmp_path / "zemia_height16.json").read_text())["bits"] == 16
    assert Image.open(tmp_path / "material_mask.png").mode == "RGBA"


def test_hires_direct_and_tiles_identical(tmp_path, monkeypatch):
    height, mask, flow = _fields()
    monkeypatch.setattr(hires, "TILE", 40)
    kwargs = dict(wrap_x=False, tile_origin=(144, -96), detail_seed=0,
                  detail_fraction=0.03,
                  generator_info={"preset": "alpine", "wind_mode": "westerly"})
    direct = tmp_path / "direct"
    tiles = tmp_path / "tiles"
    hires.write_direct(height, mask, flow, 96, 72, 5, direct, **kwargs)
    hires.write_tiles(height, mask, flow, 96, 72, 5, tiles, **kwargs)
    a = np.array(Image.open(direct / "height16.png"))
    b = np.empty_like(a)
    for f in (tiles / "height").glob("*.png"):
        y = int(f.stem.split("_")[0][1:]); x = int(f.stem.split("_")[1][1:])
        part = np.array(Image.open(f))
        b[y:y+part.shape[0], x:x+part.shape[1]] = part
    np.testing.assert_array_equal(a, b)
    meta = json.loads((direct / "manifest.json").read_text())
    assert meta["generator"]["preset"] == "alpine"
    assert meta["wrap"] == "none (atlas tile)"
    assert meta["detail_fraction"] == 0.03
    assert Image.open(direct / "material_mask.png").mode == "RGBA"


def test_upscaled_mask_softens_a_stepped_edge(tmp_path):
    mask = np.zeros((24, 32, 4), np.float32)
    y, x = np.mgrid[:24, :32]
    mask[(y * 2 + x) > 28, 0] = 255
    flow = np.zeros((24, 32, 3), np.float32)
    height = np.zeros((24, 32), np.float32)
    dest = tmp_path / "soft"
    hires.write_direct(height, mask, flow, 96, 128, 1, dest,
                       detail_fraction=0, wrap_x=False)
    band = np.array(Image.open(dest / "material_mask.png"))[:, :, 0]
    soft = np.logical_and(band > 8, band < 247).sum()
    assert soft > 30


def test_microdetail_shared_atlas_coordinates():
    b = hires.detail(0, 64, 16, 1, 64, 11, wrap_x=False, origin=(0, 0))
    a = hires.detail(0, 0, 16, 1, 64, 11, wrap_x=False, origin=(0, 64))
    np.testing.assert_array_equal(a, b)


def test_hires_atlas_samples_own_edges_not_opposite_edges():
    a = np.tile(np.arange(8, dtype=np.float32), (4, 1))
    clamped = hires._sample_tile(a, 0, 0, 8, 16, 8, 16, wrap_x=False)
    wrapped = hires._sample_tile(a, 0, 0, 8, 16, 8, 16, wrap_x=True)
    assert clamped[0, 0] == pytest.approx(0)
    assert wrapped[0, 0] > 0
    assert clamped[0, -1] == pytest.approx(7)


def test_advanced_bundle_from_gui_world(monkeypatch, tmp_path):
    gui = _mock_gui(monkeypatch)
    cfg = make_config("arid", width=96, height=72, seed=3,
                      erosion_model="none", wind_mode="easterly")
    world = generate(cfg, verbose=False)
    from terrain.export import bundle
    bundle(world, parent=tmp_path, advanced=True)
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1
    folder = dirs[0]
    for name in ["height16.png", "normals.png", "materials.png",
                 "wetness.png", "flow_vectors.npz", "terrain_fields.npz"]:
        assert (folder / name).exists()


def test_border_erosion_mode_observed(monkeypatch):
    from terrain import hydrology
    calls = []
    original = hydrology.stream_power

    def capture(h, sea_level, passes, k, *args, **kwargs):
        calls.append((passes, k))
        return original(h, sea_level, passes, k, *args, **kwargs)

    monkeypatch.setattr(hydrology, "stream_power", capture)
    cfg = make_config("earthlike", width=96, height=72, seed=7,
                      erosion_passes=3, erosion_model="none")
    border = {"left": {"edge": np.zeros(72, np.float32),
                       "outer": np.zeros(72, np.float32),
                       "biome": np.zeros(72, np.int8)}}
    generate(cfg, verbose=False, border=border)
    assert calls and calls[0] == (0, 0.0)


def test_invalid_detail(tmp_path):
    h, mask, flow = _fields()
    with pytest.raises(ValueError, match="detail_fraction"):
        hires.write_direct(h, mask, flow, 64, 48, 1, tmp_path, detail_fraction=-1)


def test_gui_export_worker_writes_advanced_outputs(monkeypatch, tmp_path):
    gui = _mock_gui(monkeypatch)
    monkeypatch.setattr(gui, "LIBRARY_TERRAIN", tmp_path / "library" / "terrain")
    monkeypatch.setattr(gui, "BUNDLE_PARENT", tmp_path / "export")
    settings = dict(preset="earthlike", seed=18, n_plates=8,
                    land_fraction=0.29, temp_offset=0.,
                    erosion_model="none", wind_mode="belts",
                    depression_method="iterative", noise_backend="numpy",
                    detail_level="low", advanced_export=True)
    world = generate(gui.config_from_settings(settings, 96, 72), verbose=False)
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    studio.world = world
    studio.made = gui.assets_of(world)
    studio.tx = 0
    studio.ty = 0
    studio._export_worker(256, settings, False)
    events = []
    while not studio.events.empty():
        events.append(studio.events.get_nowait())
    errors = [ev[1] for ev in events if ev[0] == "error"]
    assert not errors, errors
    assert any(ev[0] == "exported" for ev in events)
    hires_dirs = list((tmp_path / "library" / "terrain" / "hires").iterdir())
    assert len(hires_dirs) == 1
    manifest = json.loads((hires_dirs[0] / "manifest.json").read_text())
    assert manifest["generator"]["preset"] == "earthlike"
    assert manifest["detail_fraction"] == 0.01
    assert manifest["wrap"] == "none (atlas tile)"
    assert (tmp_path / "library" / "terrain" / "zemia_height16.png").exists()
    advanced = next((tmp_path / "export").iterdir())
    assert (advanced / "flow_vectors.npz").exists()


def test_export_writes_into_the_folder_the_user_picked(monkeypatch, tmp_path):
    gui = _mock_gui(monkeypatch)
    monkeypatch.setattr(gui, "LIBRARY_TERRAIN", tmp_path / "library" / "terrain")
    monkeypatch.setattr(gui, "BUNDLE_PARENT", tmp_path / "export")
    chosen = tmp_path / "picked"
    chosen.mkdir()
    seen = {}

    def write_direct(height, mask, flow, w, h, seed, folder, **kwargs):
        seen["hires"] = Path(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    def write_game(made, library_dir, world=None):
        seen["game"] = Path(library_dir)
        Path(library_dir).mkdir(parents=True, exist_ok=True)
        return Path(library_dir)

    def bundle(world, parent="out/export", d=None, advanced=False):
        seen["bundle"] = Path(parent)
        out = Path(parent) / "seed3"
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(gui.hires, "write_direct", write_direct)
    monkeypatch.setattr(gui, "write_game_copy", write_game)
    monkeypatch.setattr(gui.export, "bundle", bundle)
    blank = np.zeros((8, 8, 3), np.uint8)
    world = types.SimpleNamespace(
        height=np.zeros((8, 8), np.float32),
        cfg=types.SimpleNamespace(
            erosion_model="none", terrain_model="classic", wind_mode="belts",
            depression_method="iterative", noise_backend="numpy",
            world_width_m=1000.0, seed=3,
        ),
    )
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    studio.world = world
    studio.made = {
        "world": blank, "height": blank, "biomes": blank,
        "material mask": blank, "flow": blank, "canopy": blank,
        "_mask": np.zeros((8, 8, 4), np.uint8),
    }
    studio.tx = 0
    studio.ty = 0
    studio.atlas = {}
    settings = dict(preset="earthlike", seed=3, detail_level="off",
                    detail_mode="noise", advanced_export=False)
    studio._export_worker(64, settings, False, chosen)
    events = []
    while not studio.events.empty():
        events.append(studio.events.get_nowait())
    errors = [ev[1] for ev in events if ev[0] == "error"]
    assert not errors, errors
    assert seen["hires"].is_relative_to(chosen / "hires")
    assert seen["game"] == chosen
    assert seen["bundle"] == chosen
    assert not (tmp_path / "library").exists()


def _atlas_tile(seed, height_value):
    height = np.full((8, 12), height_value, np.float32)
    mask = np.zeros((8, 12, 4), np.uint8)
    flow = np.zeros((8, 12, 3), np.uint8)
    color = np.zeros((8, 12, 3), np.uint8)
    cfg = types.SimpleNamespace(
        erosion_model="none", terrain_model="classic", wind_mode="belts",
        depression_method="iterative", noise_backend="numpy",
        world_width_m=1000.0, seed=seed,
    )
    made = {
        "world": color, "height": color, "biomes": color, "canopy": color,
        "material mask": mask[:, :, :3], "flow": flow, "_mask": mask,
    }
    return {
        "world": types.SimpleNamespace(height=height, cfg=cfg),
        "made": made,
        "settings": {"seed": seed, "preset": "earthlike"},
    }


def test_export_is_the_whole_world_not_the_open_tile(monkeypatch, tmp_path):
    gui = _mock_gui(monkeypatch)
    monkeypatch.setattr(gui, "LIBRARY_TERRAIN", tmp_path / "library" / "terrain")
    monkeypatch.setattr(gui, "BUNDLE_PARENT", tmp_path / "export")
    called = {"bundle": False}

    def bundle(*args, **kwargs):
        called["bundle"] = True
        raise AssertionError("a multi-tile world must not bundle only the open tile")

    monkeypatch.setattr(gui.export, "bundle", bundle)
    left = _atlas_tile(7, 0.0)
    right = _atlas_tile(99, 3.0)
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    studio.atlas = {(0, 0): left, (1, 0): right}
    studio.tx, studio.ty = 1, 0
    studio.world = right["world"]
    studio.made = right["made"]
    settings = dict(preset="earthlike", seed=99, detail_level="off",
                    detail_mode="noise", advanced_export=True)
    studio._export_worker(64, settings, False, tmp_path / "picked")
    events = []
    while not studio.events.empty():
        events.append(studio.events.get_nowait())
    errors = [ev[1] for ev in events if ev[0] == "error"]
    assert not errors, errors
    assert called["bundle"] is False
    folders = list((tmp_path / "picked" / "hires").iterdir())
    assert len(folders) == 1
    folder = folders[0]
    assert "x0_to_1" in folder.name and "y0_to_0" in folder.name
    assert "seed99" not in folder.name
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["generator"]["tile"] == [0, 0]
    assert {item["seed"] for item in manifest["generator"]["tiles"]} == {7, 99}
    height = np.array(Image.open(folder / "height16.png"))
    assert height.shape[1] > 64
    assert height[:, : height.shape[1] // 2].mean() < height[:, height.shape[1] // 2 :].mean()


def test_neighbor_atlas_exports_have_continuous_height(tmp_path):
    left = np.broadcast_to(np.linspace(0, 1, 32, dtype=np.float32), (24, 32)).copy()
    right = np.broadcast_to(np.linspace(1, 2, 32, dtype=np.float32), (24, 32)).copy()
    mask = np.zeros((24, 32, 4), np.uint8)
    flow = np.zeros((24, 32, 3), np.uint8)
    for name, height, x in (("left", left, 0), ("right", right, 256)):
        hires.write_direct(height, mask, flow, 256, 192, 12, tmp_path / name,
                           wrap_x=False, tile_origin=(0, x), detail_seed=0,
                           detail_fraction=.03, detail_reference_span=2.0)
    left_meta = json.loads((tmp_path / "left" / "manifest.json").read_text())
    right_meta = json.loads((tmp_path / "right" / "manifest.json").read_text())
    a = _read_height(tmp_path / "left" / "height16.png", left_meta)
    b = _read_height(tmp_path / "right" / "height16.png", right_meta)
    assert np.max(np.abs(a[:, -1] - b[:, 0])) < .01
    assert left_meta["detail_reference_span"] == right_meta["detail_reference_span"] == 2.0


def test_stroke_crosses_a_tile_boundary(monkeypatch):
    gui = _mock_gui(monkeypatch)
    points = list(gui.stroke_crossings(
        (0, 0, 0.5, 0.85), (1, 0, 0.5, 0.2), {(0, 0), (1, 0)}))
    assert {point[0] for point in points} == {0, 1}
    assert max(point[3] for point in points if point[0] == 0) > 0.98
    assert min(point[3] for point in points if point[0] == 1) < 0.02
    stayed = list(gui.stroke_crossings(
        (0, 0, 0.5, 0.2), (1, 0, 0.5, 0.2), {(0, 0)}))
    assert stayed and all(point[0] == 0 for point in stayed)


def test_brush_reaches_the_tiles_it_touches(monkeypatch):
    gui = _mock_gui(monkeypatch)
    path = [(0.2, 0.5), (0.8, 0.5)]
    assert gui.stroke_reaches(path, 0, 0, 0.15)
    assert gui.stroke_reaches(path, 1, 0, 0.3)
    assert not gui.stroke_reaches(path, 1, 0, 0.05)
    corner = [(1.0, 1.0)]
    assert gui.stroke_reaches(corner, 0, 0, 0.1)
    assert gui.stroke_reaches(corner, 1, 1, 0.1)
    assert not gui.stroke_reaches(corner, 3, 3, 0.2)


def test_stitch_atlas_joins_generated_tiles(monkeypatch):
    gui = _mock_gui(monkeypatch)
    left = np.full((4, 5), 1, np.float32)
    right = np.full((4, 5), 4, np.float32)
    mask = np.zeros((4, 5, 4), np.uint8)
    mask[..., 3] = 255
    flow = np.zeros((4, 5, 3), np.uint8)
    color = np.zeros((4, 5, 3), np.uint8)
    color[..., 1] = 30
    height, out_mask, _flow, out_color, nx, ny = gui.stitch_atlas([
        (0, 0, left, mask, flow, color),
        (1, 0, right, mask, flow, color),
    ])
    assert (nx, ny) == (2, 1)
    assert height.shape == (4, 9)
    assert height[1, 0] == 1 and height[1, -1] == 4
    assert out_mask[1, 0, 3] == 255 and out_color[1, -1, 1] == 30


def _preview_studio(gui):
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = True
    studio.job = "preview"
    studio.painting = None
    studio.seed_text = "3"
    studio.atlas = {}
    studio.tx = 1
    studio.ty = 2
    studio._preview_cancel = threading.Event()
    return studio


def test_panning_onto_an_empty_tile_does_not_generate(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}}
    studio.image_name = ""
    studio.tx, studio.ty = 0, 0
    studio.seed = 7
    studio.seed_text = "7"
    studio.preset_dirty = False
    studio._cell_under_center = lambda: (1, 0)
    studio._random_seed = lambda: 333
    started = []
    studio.start = lambda: started.append(True)
    studio._release_map()
    assert started == []
    assert (studio.tx, studio.ty) == (1, 0)
    assert studio._seed_from == (0, 0)
    assert studio.seed == 333
    assert studio.seed_text == "333"
    assert "Generate preview" in studio.status


def test_panning_onto_a_built_tile_shows_it(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}, (1, 0): {}}
    studio.image_name = "world"
    studio.tx, studio.ty = 0, 0
    studio._cell_under_center = lambda: (1, 0)
    shown = []
    started = []
    studio._show_cell = lambda tx, ty, fit=False: shown.append((tx, ty, fit))
    studio.start = lambda: started.append(True)
    studio._release_map()
    assert shown == [(1, 0, False)]
    assert started == []


def test_arrow_to_an_empty_tile_does_not_generate(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}}
    studio.tx, studio.ty = 0, 0
    studio.seed = 7
    studio.seed_text = "7"
    studio.preset_dirty = False
    studio.fit = lambda: None
    studio._random_seed = lambda: 111
    started = []
    studio.start = lambda: started.append(True)
    studio._step(1, 0)
    assert started == []
    assert (studio.tx, studio.ty) == (1, 0)
    assert studio.seed == 111
    assert studio.seed_text == "111"
    assert studio.preset_dirty
    assert "Generate preview" in studio.status


def test_each_empty_tile_gets_its_own_seed(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}}
    studio.tx, studio.ty = 0, 0
    studio.seed = 7
    studio.seed_text = "7"
    studio.preset_dirty = True
    studio.fit = lambda: None
    rolls = iter([111, 222])
    studio._random_seed = lambda: next(rolls)
    studio._step(1, 0)
    assert studio.seed == 111 and studio.seed_text == "111"
    studio._step(1, 0)
    assert (studio.tx, studio.ty) == (2, 0)
    assert studio.seed == 222 and studio.seed_text == "222"


def test_a_seed_chosen_for_this_tile_is_kept(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.atlas = {(0, 0): {"settings": {"seed": 7}}}
    studio._seed_from = (0, 0)
    studio.preset_dirty = True
    studio.seed = 42
    studio.seed_text = "42"
    studio.settings = lambda: {"seed": studio.seed}
    studio._random_seed = lambda: 999
    assert studio._preset_for(1, 0)["seed"] == 42


def test_arrow_to_a_built_tile_keeps_its_seed(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}, (1, 0): {}}
    studio.tx, studio.ty = 0, 0
    studio.seed = 7
    studio.seed_text = "7"
    studio.fit = lambda: None
    shown = []
    studio._show_cell = lambda tx, ty, fit=True: shown.append((tx, ty, fit))
    studio._random_seed = lambda: 999
    studio._step(1, 0)
    assert shown == [(1, 0, True)]
    assert studio.seed == 7


def test_panning_inside_an_empty_tile_keeps_the_seed(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.busy = False
    studio.atlas = {(0, 0): {}}
    studio.image_name = ""
    studio.tx, studio.ty = 1, 0
    studio.seed = 111
    studio.seed_text = "111"
    studio.preset_dirty = True
    studio._cell_under_center = lambda: (1, 0)
    studio._random_seed = lambda: 222
    studio._release_map()
    assert studio.seed == 111
    assert (studio.tx, studio.ty) == (1, 0)


def test_randomize_seed_fills_the_box(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.seed = 7
    studio.seed_text = "7"
    studio.preset_dirty = False
    studio._random_seed = lambda: 4242
    studio.randomize_seed()
    assert studio.seed == 4242
    assert studio.seed_text == "4242"
    assert studio.preset_dirty


def _running_preview(gui):
    studio = _preview_studio(gui)
    studio._preview_token = 4
    studio._generate_rest = [(2, 2)]
    studio._inflight_sculpt = ((1, 2), ())
    studio.live_surface = "partial"
    studio.live_cell = (1, 2)
    studio.progress = gui.Progress()
    studio.progress.begin("Generating preview", (("plates", 1),))
    studio.export_btn = types.SimpleNamespace(enabled=False)
    studio.world = None
    studio.image_name = "world"
    studio._cell_under_center = lambda: None
    kept = {"world": object()}
    studio.atlas = {(0, 0): kept}
    return studio, kept


def test_cancel_stops_a_preview_and_keeps_finished_tiles(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio, kept = _running_preview(gui)
    launched = []
    studio._launch = lambda cell, keep=False: launched.append(cell)
    studio.cancel_preview()
    assert studio._preview_cancel.is_set()
    assert studio._preview_token == 5
    assert not studio.busy
    assert studio.job is None
    assert launched == []
    assert studio.atlas == {(0, 0): kept}
    assert studio._generate_rest == []
    assert studio.live_surface is None
    assert "Pan to a tile" in studio.status


def test_after_cancel_a_new_tile_can_be_generated(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio, kept = _running_preview(gui)
    studio.cancel_preview()
    studio._cell_under_center = lambda: (1, 0)
    studio._release_map()
    assert (studio.tx, studio.ty) == (1, 0)
    launched = []
    studio._launch = lambda cell, keep=False: launched.append(cell)
    studio.seed_text = "3"
    studio.painting = None
    studio._strokes_baked = lambda cell: True
    studio.start()
    assert launched == [(1, 0)]
    assert studio.atlas[(0, 0)] is kept


def test_start_does_not_restart_a_running_preview(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = _preview_studio(gui)
    studio.seed_text = "nope"
    launched = []
    studio._launch = lambda cell, keep=False: launched.append(cell)
    studio.start()
    assert not studio._preview_cancel.is_set()
    assert launched == []
    assert studio.busy


def test_cancel_does_not_stop_an_export(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = _preview_studio(gui)
    studio.job = "export"
    studio.cancel_preview()
    assert not studio._preview_cancel.is_set()
    assert studio.busy
    assert studio.job == "export"


def test_cancelled_preview_posts_no_result(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    settings = dict(preset="archipelago", seed=8, n_plates=6,
                    land_fraction=0.18, temp_offset=0.,
                    erosion_model="none", wind_mode="belts",
                    depression_method="iterative", noise_backend="numpy",
                    detail_level="off", advanced_export=False)
    cfg = gui.config_from_settings(settings, 32, 24)
    cancel = threading.Event()
    cancel.set()
    studio._worker(cfg, None, settings, (0, 0), None, 1, cancel)
    assert studio.events.empty()


def test_a_replaced_preview_cannot_finish_over_the_new_one(monkeypatch):
    gui = _mock_gui(monkeypatch)
    studio = gui.Studio.__new__(gui.Studio)
    studio.events = queue.Queue()
    studio.job = "preview"
    studio.busy = True
    studio.tx, studio.ty = 0, 0
    studio.atlas = {}
    studio._preview_token = 2
    studio.progress = gui.Progress()
    studio.progress.begin("Generating preview", (("plates", 1),))
    studio.status = "generating"
    studio.live_surface = "current"
    studio.events.put(("stage", "plates", 1))
    studio.events.put(("cell_image", (0, 0), "stale", 1))
    studio.events.put(("error", "generation failed: boom", 1))
    studio.events.put(("done", None, None, None, (0, 0), 1))
    studio.events.put(("stage", "relief", 2))
    studio.pump()
    assert studio.busy
    assert studio.job == "preview"
    assert studio.atlas == {}
    assert studio.live_surface == "current"
    assert studio.status == "relief"
