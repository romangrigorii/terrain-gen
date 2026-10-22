"""Brushes and round-trips for the export embellisher."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hires
from terrain import export
from terrain.embellish import (
    MATERIALS, MATERIAL_GROUP, TOOLS, TileRequired, discover, legacy_mask, open_export,
)


def _ramp():
    return np.linspace(-0.05, 0.3, 24 * 32, dtype=np.float32).reshape(24, 32)


def _bundle(root, height=None, materials=None, biome=None, mask=None, world=None):
    height = _ramp() if height is None else np.asarray(height, np.float32)
    folder = Path(root) / "export" / "seed1"
    folder.mkdir(parents=True, exist_ok=True)
    export.heightmap(height, folder / "height16.png")
    payload = {"seed": 3}
    if world is not False:
        payload["world_width_m"] = float(height.shape[1] * 100 if world is None else world)
    (folder / "config.json").write_text(json.dumps(payload))
    if materials is not None:
        Image.fromarray(np.asarray(materials, np.uint8), "RGBA").save(folder / "materials.png")
    if biome is not None:
        Image.fromarray(np.asarray(biome, np.uint8), "L").save(folder / "biome.png")
    if mask is not None:
        Image.fromarray(np.asarray(mask, np.uint8), "RGBA").save(folder / "material_mask.png")
    return folder


def _zeros(root):
    return _bundle(root, np.zeros((24, 32), np.float32))


def test_open_roundtrip_and_cell_size(tmp_path):
    height = _ramp()
    folder = _bundle(tmp_path, height)
    doc = open_export(folder)
    assert doc.kind == "bundle"
    assert doc.cell_exact
    assert doc.cell_m == pytest.approx(100)
    assert np.allclose(doc.height, height, atol=1e-3)
    assert doc.render("shaded").shape == height.shape + (3,)
    assert doc.render("height").dtype == np.uint8
    assert doc.render("materials").shape[2] == 3


def test_missing_folder_explains_itself(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="height"):
        open_export(empty)


def test_estimated_cell_size_without_world_width(tmp_path):
    folder = _bundle(tmp_path, world=False)
    doc = open_export(folder)
    assert not doc.cell_exact
    assert doc.cell_m == pytest.approx(200_000 / doc.height.shape[1])


def test_raise_is_local_and_one_undo_restores_the_stroke(tmp_path):
    folder = _zeros(tmp_path)
    doc = open_export(folder)
    before = doc.height.copy()
    doc.begin_stroke("raise", 5, 5, 2, 120)
    doc.dab(5, 5)
    doc.dab(20, 20)
    doc.end_stroke()
    assert doc.height[5, 5] == pytest.approx(120 / 12000, abs=1e-5)
    assert doc.height[20, 20] == pytest.approx(120 / 12000, abs=1e-5)
    assert doc.height[12, 12] == 0
    assert doc.dirty
    assert doc.undo()
    assert np.array_equal(doc.height, before)
    assert not doc.dirty
    assert doc.redo()
    assert doc.height[5, 5] > 0
    doc.save()
    assert not doc.dirty
    doc.begin_stroke("raise", 8, 8, 2, 30)
    doc.dab(8, 8)
    doc.end_stroke()
    assert doc.dirty
    doc.undo()
    assert not doc.dirty


def test_lower_smooth_flatten_and_grain(tmp_path):
    height = np.zeros((24, 32), np.float32)
    height[16, 16] = 0.2
    doc = open_export(_bundle(tmp_path, height))
    doc.begin_stroke("lower", 16, 16, 3, 240)
    doc.dab(16, 16)
    doc.end_stroke()
    assert doc.height[16, 16] == pytest.approx(0.2 - 240 / 12000, abs=1e-5)

    height = np.zeros((24, 32), np.float32)
    height[12, 12] = 1
    doc = open_export(_bundle(tmp_path / "smooth", height))
    doc.begin_stroke("smooth", 12, 12, 4, 1)
    doc.dab(12, 12)
    doc.end_stroke()
    assert doc.height[12, 12] < 0.5
    assert doc.height[0, 0] == 0

    height = np.zeros((24, 32), np.float32)
    height[16, 16] = 1
    doc = open_export(_bundle(tmp_path / "flat", height))
    doc.begin_stroke("flatten", 8, 8, 14, 1)
    doc.dab(16, 16)
    doc.end_stroke()
    assert doc.height[16, 16] < 0.2
    assert doc.height[0, 0] == 0

    doc = open_export(_zeros(tmp_path / "grain"))
    doc.begin_stroke("grain", 15, 15, 4, 50)
    doc.dab(15, 15)
    doc.end_stroke()
    first = doc.height.copy()
    assert np.any(first[12:19, 12:19] != 0)
    assert np.all(first[:5, :5] == 0)
    again = open_export(_zeros(tmp_path / "grain2"))
    again.begin_stroke("grain", 15, 15, 4, 50)
    again.dab(15, 15)
    again.end_stroke()
    assert np.array_equal(first, again.height)


def test_path_cuts_bumps_and_paints_a_track(tmp_path):
    height = np.full((30, 40), 0.02, np.float32)
    height[15, 5::6] = 0.2
    doc = open_export(_bundle(tmp_path, height))
    before = doc.height.copy()
    doc.begin_stroke("path", 15, 2, 3, 1)
    for x in range(2, 38):
        doc.dab(15, x)
    doc.end_stroke()
    assert doc.height[15, 8:32].std() < before[15, 8:32].std() * 0.6
    assert doc.materials[15, 20, 0] > 0.5
    assert doc.height[0, 0] == pytest.approx(before[0, 0])


def test_water_dry_and_raise(tmp_path):
    height = np.full((24, 32), 0.1, np.float32)
    doc = open_export(_bundle(tmp_path, height))
    doc.begin_stroke("water", 10, 10, 5, 20)
    doc.dab(10, 10)
    doc.end_stroke()
    assert doc.water[10, 10] > 0.9
    assert doc.height[10, 10] == pytest.approx(0.1 - 20 / 12000, abs=1e-6)
    assert doc.height[0, 0] == pytest.approx(0.1)
    held = float(doc.height[10, 10])
    doc.begin_stroke("dry", 10, 10, 5, 1)
    doc.dab(10, 10)
    doc.end_stroke()
    assert doc.water[10, 10] < 0.2
    assert doc.height[10, 10] == pytest.approx(held)
    doc.begin_stroke("water", 10, 10, 5, 20)
    doc.dab(10, 10)
    doc.end_stroke()
    doc.begin_stroke("raise", 10, 10, 4, 30)
    doc.dab(10, 10)
    doc.end_stroke()
    assert doc.water[10, 10] < 0.2


def test_paint_place_and_erase(tmp_path):
    materials = np.zeros((24, 32, 4), np.uint8)
    materials[..., 1] = 255
    doc = open_export(_bundle(tmp_path, materials=materials))
    doc.begin_stroke("paint", 8, 8, 3, 1, material="sand")
    doc.dab(8, 8)
    doc.end_stroke()
    assert doc.materials[8, 8, 0] > 0.9
    assert doc.materials[8, 8].sum() == pytest.approx(1)
    assert doc.materials[0, 0, 1] == pytest.approx(1)

    doc.begin_stroke("place", 5, 5, 4, 1.2, prop="tree")
    doc.dab(5, 5)
    doc.dab(5.2, 5.2)
    doc.dab(14, 14)
    doc.end_stroke()
    assert [prop["kind"] for prop in doc.props] == ["tree", "tree"]
    assert doc.props[0]["scale"] == pytest.approx(1.2)
    doc.undo()
    assert doc.props == []
    doc.redo()
    assert len(doc.props) == 2
    doc.begin_stroke("erase", 5, 5, 3, 1)
    doc.dab(5, 5)
    doc.end_stroke()
    assert len(doc.props) == 1
    assert doc.props[0]["y"] == pytest.approx(14)


def test_surface_palette_blends_water_ice_and_roundtrips(tmp_path):
    assert MATERIALS[:4] == ("sand", "soil", "rock", "snow")
    assert len(MATERIALS) >= 24
    assert MATERIAL_GROUP["glacier"] == "cold"
    assert MATERIAL_GROUP["shallows"] == "water"
    assert MATERIAL_GROUP["lava"] == "fire"
    materials = np.zeros((24, 32, 4), np.uint8)
    materials[..., 1] = 255
    folder = _bundle(tmp_path, materials=materials)
    doc = open_export(folder)
    assert doc.materials.shape[-1] == len(MATERIALS)
    ice = MATERIALS.index("glacier")
    moss = MATERIALS.index("moss")
    doc.begin_stroke("paint", 8, 8, 3, 1, material="glacier")
    doc.dab(8, 8)
    doc.end_stroke()
    assert doc.materials[8, 8, ice] > 0.9
    assert doc.materials[8, 8].sum() == pytest.approx(1)
    assert doc.materials[0, 0, 1] == pytest.approx(1)
    doc.begin_stroke("paint", 8, 8, 3, 0.45, material="moss")
    doc.dab(8, 8)
    doc.end_stroke()
    weights = np.asarray(doc.materials[8, 8], np.float32)
    assert weights[ice] > 0.4
    assert weights[moss] > 0.3
    assert weights.sum() == pytest.approx(1, abs=1e-3)
    doc.begin_stroke("paint", 12, 12, 3, 1, material="shallows")
    doc.dab(12, 12)
    doc.end_stroke()
    shallows = MATERIALS.index("shallows")
    assert doc.materials[12, 12, shallows] > 0.9
    assert doc.water[12, 12] > 0.8
    doc.begin_stroke("paint", 12, 12, 3, 1, material="shallows", erase=True)
    doc.dab(12, 12)
    doc.end_stroke()
    assert doc.materials[12, 12, shallows] < 0.15
    assert doc.water[12, 12] < 0.25
    image = doc.render("materials")
    assert image.dtype == np.uint8 and image.shape == doc.height.shape + (3,)
    doc.save()
    assert (folder / "authored" / "surfaces.json").is_file()
    again = open_export(folder)
    assert again.materials.shape[-1] == len(MATERIALS)
    assert float(again.materials[8, 8, moss]) == pytest.approx(float(doc.materials[8, 8, moss]), abs=1e-4)

    glacier = np.zeros((2, 2, len(MATERIALS)), np.float32)
    glacier[..., ice] = 1
    mask = legacy_mask(glacier)
    assert mask[0, 0, 2] == 255
    assert int(mask[0, 0, 0]) == 0
    water = np.zeros((2, 2, len(MATERIALS)), np.float32)
    water[..., shallows] = 1
    assert int(legacy_mask(water).sum()) == 0

    height = again.height
    four = np.zeros(height.shape + (4,), np.float32)
    four[..., 0] = 1
    np.savez_compressed(folder / "authored" / "height.npz",
                        height=height, materials=four, water=np.zeros(height.shape, np.float32))
    reopened = open_export(folder)
    assert reopened.materials.shape[-1] == len(MATERIALS)
    assert reopened.materials[2, 2, 0] == pytest.approx(1)
    assert reopened.materials[2, 2, 1] == pytest.approx(0)


def test_biome_and_legacy_mask_become_ground(tmp_path):
    biome = np.full((24, 32), 6, np.uint8)
    biome[4, 4] = 4
    biome[0, 0] = 0
    doc = open_export(_bundle(tmp_path / "biome", biome=biome))
    assert doc.materials[4, 4, 0] > 0.7
    assert doc.materials[0, 0].sum() == 0

    mask = np.zeros((24, 32, 4), np.uint8)
    mask[..., 0] = 255
    doc = open_export(_bundle(tmp_path / "mask", mask=mask))
    assert doc.materials[3, 3, 2] > 0.9
    assert "game mask" in doc.summary


def test_save_keeps_original_and_sub_code_edits(tmp_path):
    folder = _bundle(tmp_path)
    doc = open_export(folder)
    before = float(doc.height[3, 3])
    original = (folder / "height16.png").read_bytes()
    doc.height[3, 3] = before + 0.25
    doc.save()
    backup = folder / "authored" / "original" / "height16.png"
    assert backup.read_bytes() == original
    doc.height[3, 3] = before + 0.5
    doc.save()
    assert backup.read_bytes() == original
    target = float(doc.height[4, 4]) + 1e-8
    doc.height[4, 4] = target
    doc.save()
    exact = open_export(folder)
    assert exact.height[4, 4] == pytest.approx(target, abs=1e-12)
    quantized = open_export(folder, prefer_authored=False)
    assert abs(float(quantized.height[4, 4]) - target) > 1e-9
    restored = exact.restore()
    assert restored.height[3, 3] == pytest.approx(before, abs=1e-5)
    assert not (folder / "authored" / "height.npz").exists()


def test_save_updates_and_restore_removes_a_new_game_copy(tmp_path):
    root = tmp_path / "out"
    folder = _bundle(root)
    library = root / "library" / "terrain"
    library.mkdir(parents=True)
    doc = open_export(folder)
    assert doc.library_dir == library
    doc.begin_stroke("raise", 6, 6, 2, 80)
    doc.dab(6, 6)
    doc.end_stroke()
    note = doc.save()
    assert "Game copy" in note
    assert (library / "zemia_height16.png").is_file()
    assert (library / "preview.png").is_file()
    restored = doc.restore()
    assert restored.height[6, 6] < doc.height[6, 6]
    assert not (library / "zemia_height16.png").exists()


def test_hires_manifest_supplies_cell_size(tmp_path):
    height = np.zeros((6, 8), np.float32)
    height[1, 1] = 0.2
    mask = np.zeros((6, 8, 4), np.uint8)
    mask[..., 1] = 200
    flow = np.zeros((6, 8, 3), np.uint8)
    hires.write_direct(height, mask, flow, 8, 6, 3, tmp_path,
                       detail_fraction=0, world_width_m=8000, wrap_x=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["world_width_m"] == 8000
    doc = open_export(tmp_path)
    assert doc.kind == "hires"
    assert doc.height.shape == (6, 8)
    assert doc.cell_exact
    assert doc.cell_m == pytest.approx(1000)
    assert np.isfinite(doc.height).all()


def test_tiled_save_clips_to_the_shared_scale(tmp_path):
    folder = tmp_path / "continent"
    (folder / "height").mkdir(parents=True)
    (folder / "mask").mkdir()
    export._write_png(folder / "height" / "y000000_x000000.png",
                      np.full((4, 4), 1000, np.uint16), 16)
    Image.fromarray(np.zeros((4, 4, 4), np.uint8), "RGBA").save(
        folder / "mask" / "y000000_x000000.png")
    (folder / "manifest.json").write_text(json.dumps({
        "lo": 0.0, "hi": 1.0, "units_per_code": 1 / 65535,
        "width": 8, "height": 4, "metres_per_unit": 12000,
        "world_width_m": 8000, "tile": 4, "seed": 1,
    }))
    with pytest.raises(TileRequired) as raised:
        open_export(folder)
    assert len(raised.value.entry.tiles) == 1
    doc = open_export(folder, tile=(0, 0))
    assert doc.kind == "tiled"
    assert doc.cell_m == pytest.approx(1000)
    doc.height[:] = 5
    note = doc.save()
    assert doc.clipped_on_save
    assert "clipped" in note
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["hi"] == 1.0
    assert open_export(folder, tile=(0, 0)).height[0, 0] == pytest.approx(5)
    image = open_export(folder, tile=(0, 0), prefer_authored=False)
    assert image.height.max() == pytest.approx(1.0, abs=1e-4)
    assert (folder / "mask" / "y000000_x000000.png").is_file()


def test_discover_lists_bundle_and_hires(tmp_path):
    root = tmp_path / "out"
    _bundle(root)
    ridge = root / "library" / "terrain" / "hires" / "ridge"
    height = np.zeros((4, 4), np.float32)
    mask = np.zeros((4, 4, 4), np.uint8)
    flow = np.zeros((4, 4, 3), np.uint8)
    hires.write_direct(height, mask, flow, 4, 4, 1, ridge,
                       detail_fraction=0, world_width_m=4000, wrap_x=False)
    labels = [entry.label for entry in discover(root)]
    assert any("seed1" in label and "bundle" in label for label in labels)
    assert any("ridge" in label and "hires" in label for label in labels)


def test_editor_tools_and_studio_launch(tmp_path):
    import edit_gui
    import gui

    assert set(edit_gui.TOOL_ORDER) == set(edit_gui.HINTS)
    assert set(edit_gui.AMOUNT) == set(edit_gui.TOOL_ORDER) - {"pan", "erase"}
    assert set(edit_gui.TOOL_ORDER) - {"pan"} <= set(TOOLS)
    command = gui.embellish_command(r"C:\bundle", r"C:\hires")
    assert command[-1] == r"C:\hires"
    assert command[1].endswith("edit_gui.py")
    command = gui.embellish_command(r"C:\bundle", None)
    assert command[-1] == r"C:\bundle"
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"bundle": "b", "hires": "h"}))
    assert gui.read_latest(pointer) == ("b", "h")
    assert gui.read_latest(tmp_path / "missing.json") == (None, None)
