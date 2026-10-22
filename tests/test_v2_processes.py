"""Glaciers, dunes, coasts, caves, tile flux, and scenario editing."""
import numpy as np
import pytest

from terrain.v2 import TerrainState, run
from terrain.v2.core.validation import check_state, thickness_budget
from terrain.v2.erosion.aeolian import blow
from terrain.v2.erosion.coastal import retreat
from terrain.v2.erosion.glacial import glaciate
from terrain.v2.erosion.hydraulic import step_water, tiled_step
from terrain.v2.erosion.karst import dissolve
from terrain.v2.generation.editor import ScenarioEditor
from terrain.v2.generation.preview import shade_state


def test_ice_deepens_the_valley_floor_more_than_the_ridge():
    rows, cols = 14, 21
    center = cols // 2
    profile = 50.0 * np.abs(np.arange(cols) - center) / center
    height = np.tile(profile, (rows, 1)) + np.linspace(20.0, 0.0, rows)[:, None]
    state = TerrainState((rows, cols), 25.0)
    state.set("bedrock_m", height)
    state.set("temperature_c", np.full(height.shape, -6.0))
    state.set("rainfall_mm_day", np.full(height.shape, 4.0))
    ice = np.zeros(height.shape, np.float32)
    ice[:, center - 1:center + 2] = 25.0
    state.set("ice_m", ice)
    before = state.elevation.copy()
    for _ in range(5):
        glaciate(state, k=0.6, snow_m=1.0)
    check_state(state)
    floor = float((before[:, center] - state.elevation[:, center]).mean())
    ridge = float((before[:, 0] - state.elevation[:, 0]).mean())
    shoulder = float((before[:, center - 2] - state.elevation[:, center - 2]).mean())
    assert floor > ridge + 0.4
    assert shoulder > ridge
    assert state.get("ice_m").min() >= 0


def test_warm_ice_melts_into_till():
    state = TerrainState((6, 6), 20.0)
    state.set("temperature_c", np.full((6, 6), 8.0))
    state.set("ice_m", np.full((6, 6), 10.0))
    glaciate(state, k=0.0, snow_m=0.0)
    assert state.get("ice_m").mean() == pytest.approx(2.0, abs=0.05)
    assert state.get("regolith_m").mean() == pytest.approx(1.2, abs=0.05)


def test_sand_blows_downwind_and_stops_at_plants():
    state = TerrainState((10, 16), 20.0)
    sand = np.zeros((10, 16), np.float32)
    sand[:, 2] = 8.0
    state.set("sand_m", sand)
    before = float(state.get("sand_m").sum())
    for _ in range(5):
        blow(state, wind_deg=90.0, rate=0.5)
    assert float(state.get("sand_m").sum()) == pytest.approx(before, abs=1e-2)
    cols = np.arange(16)
    centroid = float((state.get("sand_m").sum(axis=0) * cols).sum() / before)
    assert centroid > 3.0

    blocked = TerrainState((8, 14), 20.0)
    patch = np.zeros((8, 14), np.float32)
    patch[:, 2] = 6.0
    blocked.set("sand_m", patch)
    veg = np.zeros((8, 14), np.float32)
    veg[:, 8] = 1.0
    blocked.set("vegetation", veg)
    for _ in range(12):
        blow(blocked, wind_deg=90.0, rate=0.6)
    assert blocked.get("sand_m")[:, 9:].sum() == pytest.approx(0.0, abs=1e-3)
    assert blocked.get("sand_m")[:, 7].sum() > blocked.get("sand_m")[:, 2].sum()


def test_waves_cut_the_shore_and_leave_a_beach():
    state = TerrainState((12, 14), 30.0)
    height = np.full((12, 14), -8.0)
    height[:, 5:] = 24.0
    state.set("bedrock_m", height)
    state.set("hardness", np.zeros((12, 14)))
    before = state.elevation.copy()
    retreat(state, k=3.0)
    check_state(state)
    assert state.elevation[6, 5] < before[6, 5] - 1.0
    assert state.elevation[6, 8] == pytest.approx(before[6, 8], abs=1e-3)
    assert state.get("sand_m").sum() > 0
    hard = TerrainState((12, 14), 30.0)
    hard.set("bedrock_m", height)
    hard.set("hardness", np.ones((12, 14)))
    retreat(hard, k=3.0)
    assert hard.elevation[6, 5] > state.elevation[6, 5]


def test_caves_keep_the_roof_until_it_is_too_thin():
    state = TerrainState((8, 8), 20.0)
    state.set("bedrock_m", np.full((8, 8), 40.0))
    state.set("porosity", np.full((8, 8), 0.8))
    state.set("hardness", np.zeros((8, 8)))
    state.set("rainfall_mm_day", np.full((8, 8), 10.0))
    roof = state.elevation.copy()
    dissolve(state, k=0.01, roof_m=8.0, dt_years=20.0)
    assert state.elevation == pytest.approx(roof)
    assert state.get("void_m").mean() > 0.5
    assert state.features["dissolved_thickness_sum"] == pytest.approx(
        float(state.get("void_m").sum()), rel=1e-5, abs=1e-2)
    state.set("void_m", np.full((8, 8), 38.0))
    before = float(state.elevation.sum())
    dissolve(state, k=0.0, roof_m=8.0, dt_years=1.0)
    assert state.elevation.mean() < 40.0
    assert state.get("void_m").max() <= 32.0 + 1e-3
    assert before > float(state.elevation.sum())
    check_state(state)


def test_tiled_water_matches_one_unsplit_step():
    rng = np.random.default_rng(3)
    elevation = rng.normal(size=(17, 19)).cumsum(axis=0)
    water = rng.random((17, 19)) * 2.0
    rain = np.full((17, 19), 0.1)
    whole = step_water(elevation, water, rain)
    tiled = tiled_step(elevation, water, rain, tile_h=8, tile_w=6)
    assert tiled == pytest.approx(whole, abs=1e-8)
    assert float(tiled.sum()) == pytest.approx(float(water.sum() + rain.sum()), abs=1e-6)


def test_scenario_editor_changes_the_run_without_a_new_solver():
    editor = ScenarioEditor("arid")
    assert editor.processes["aeolian"]
    editor.toggle("aeolian")
    assert not editor.processes["aeolian"]
    editor.set("rainfall_mm_day", 80)
    assert editor.get("rainfall_mm_day") == 30.0
    editor.cycle()
    assert editor.name != "arid"
    assert "processes" in editor.overrides()

    shape = (16, 20)
    dry = run("temperate", shape=shape, cell_m=80, seed=1, passes=1, dt_years=100,
              overrides={"rainfall_mm_day": 0.3, "processes": {
                  "glacial": False, "aeolian": False, "coastal": False, "karst": False}})
    wet = run("temperate", shape=shape, cell_m=80, seed=1, passes=1, dt_years=100,
              overrides={"rainfall_mm_day": 12.0, "processes": {
                  "glacial": False, "aeolian": False, "coastal": False, "karst": False}})
    assert wet.get("bedrock_m").sum() < dry.get("bedrock_m").sum()

    alpine = run("alpine", shape=shape, cell_m=80, seed=1, passes=1, dt_years=40)
    tropical = run("tropical", shape=shape, cell_m=80, seed=1, passes=1, dt_years=40)
    assert alpine.get("ice_m").mean() > 0
    assert tropical.get("void_m").mean() > 0
    picture = shade_state(alpine)
    assert picture.shape == alpine.elevation.shape + (3,)
    assert picture.dtype == np.uint8


def test_coastal_cut_stays_in_the_thickness_budget():
    state = TerrainState((10, 12), 25.0)
    height = np.full((10, 12), -4.0)
    height[:, 4:] = 18.0
    state.set("bedrock_m", height)
    before = float(state.elevation.sum())
    retreat(state, k=2.5)
    thickness_budget(before, float(state.elevation.sum()), 0.0,
                     float(state.features["offshore_thickness_sum"]), atol=0.05)
