"""Compatibility checks for the original GUI/highres workflow with terrain v2."""
import importlib
import json
import queue
import sys
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
    assert studio.land.value == 18
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
