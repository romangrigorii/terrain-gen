"""Shared state, physical stages, and scenario fields."""
import numpy as np
import pytest

from terrain.v2 import TerrainState, import_world, run
from terrain.v2.core.masks import normalize_weights
from terrain.v2.core.parameters import BlendField, ConstantField, NoiseField, RadialField
from terrain.v2.core.pipeline import Context, Pipeline, Stage
from terrain.v2.core.registry import create, known
from terrain.v2.core.scheduler import Scheduler
from terrain.v2.core.validation import (
    check_state, drainage_problems, thickness_budget, water_budget,
)
from terrain.v2.environment.vegetation import viability
from terrain.v2.erosion.hydraulic import step_water
from terrain.v2.erosion.stream_power import incise
from terrain.v2.erosion.thermal import relax
from terrain.v2.generation.scenarios import parameters_for, scenario_names
from terrain.v2.generation.world import PaintClimate, SeedRelief
from terrain.v2.hydrology.river_graph import build_graph, enforce_descending
from terrain.v2.hydrology.routing import accumulate, route


def test_elevation_is_the_sum_of_the_solid_layers():
    state = TerrainState((4, 5), 25.0)
    state.set("bedrock_m", np.full((4, 5), 10.0))
    state.set("regolith_m", np.full((4, 5), 1.0))
    state.set("sand_m", np.full((4, 5), 2.0))
    state.set("soil_m", np.full((4, 5), 0.5))
    assert state.elevation == pytest.approx(13.5)


def test_peel_takes_cover_before_bedrock_and_refuses_bad_values():
    state = TerrainState((3, 3), 10.0)
    soil = np.zeros((3, 3), np.float32)
    sand = np.zeros((3, 3), np.float32)
    soil[1, 1] = 2.0
    sand[1, 1] = 3.0
    state.set("soil_m", soil)
    state.set("sand_m", sand)
    state.set("bedrock_m", np.full((3, 3), 20.0))
    state.peel(np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]], np.float64))
    assert state.get("soil_m")[1, 1] == pytest.approx(0.0, abs=1e-5)
    assert state.get("sand_m")[1, 1] == pytest.approx(1.0, abs=1e-5)
    assert state.get("bedrock_m")[1, 1] == pytest.approx(20.0, abs=1e-4)
    state.peel(np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], np.float64))
    assert state.get("sand_m")[1, 1] == pytest.approx(0.0, abs=1e-5)
    assert state.get("bedrock_m")[1, 1] == pytest.approx(16.0, abs=1e-3)
    assert state.get("regolith_m").min() >= 0
    with pytest.raises(ValueError):
        state.set("water_m", np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        state.set("soil_m", np.zeros((2, 2)))


def test_pipeline_rejects_a_stage_whose_inputs_are_absent():
    class NeedsIce(Stage):
        name = "ice"
        requires = ("permafrost_m",)

        def run(self, world, ctx):
            pass

    state = TerrainState((4, 4), 10.0)
    with pytest.raises(KeyError, match="permafrost_m"):
        Pipeline([NeedsIce()]).run(state, Context(seed=1, dt_years=1.0))


def test_region_weights_sum_to_one_and_blend_the_parameters():
    weights = normalize_weights([
        np.array([[1.0, 0.0], [0.2, 3.0]]),
        np.array([[1.0, 0.0], [0.2, 1.0]]),
    ])
    assert weights[0] + weights[1] == pytest.approx(1.0)
    state = TerrainState((4, 6), 50.0)
    mixed = BlendField(
        [ConstantField(2.0), ConstantField(8.0)],
        [ConstantField(1.0), ConstantField(3.0)],
    ).sample_grid(state.grid)
    assert mixed == pytest.approx(6.5)
    first = NoiseField(0.01, 3.0, seed=4).sample_grid(state.grid)
    second = NoiseField(0.01, 3.0, seed=4).sample_grid(state.grid)
    other = NoiseField(0.01, 3.0, seed=9).sample_grid(state.grid)
    assert first == pytest.approx(second)
    assert not np.allclose(first, other)


def test_water_step_conserves_volume_and_stays_nonnegative():
    bed = np.zeros((8, 8), np.float64)
    water = np.zeros((8, 8), np.float64)
    water[3, 3] = 4.0
    moved = step_water(bed, water, np.zeros_like(bed))
    water_budget(4.0, float(moved.sum()), 0.0)
    assert moved.min() >= -1e-8
    rain = np.full((8, 8), 0.25)
    wet = step_water(bed, np.zeros_like(bed), rain)
    water_budget(0.0, float(wet.sum()), float(rain.sum()))
    assert wet.min() >= -1e-8


def test_thermal_relaxation_conserves_cover_and_roots_hold_the_slope():
    def pile(repose_deg):
        state = TerrainState((5, 5), 10.0)
        sand = np.zeros((5, 5), np.float32)
        sand[2, 2] = 8.0
        state.set("sand_m", sand)
        before = float(state.elevation.sum())
        relax(state, np.deg2rad(repose_deg), rate=0.5)
        return state, before

    bare, before = pile(30.0)
    held, _ = pile(44.0)
    assert bare.elevation.sum() == pytest.approx(before, abs=1e-2)
    assert held.get("sand_m")[2, 2] == pytest.approx(8.0, abs=1e-3)
    assert bare.get("sand_m")[2, 2] < 7.0
    assert bare.get("sand_m").min() >= -1e-5


def test_stream_power_exports_a_budget_and_follows_rain_and_hardness():
    def ramp(rain, hardness):
        state = TerrainState((16, 10), 40.0)
        height = np.linspace(100.0, 0.0, 16)[:, None] * np.ones((1, 10))
        state.set("bedrock_m", height)
        state.set("rainfall_mm_day", np.full(state.grid.shape, rain))
        state.set("hardness", np.full(state.grid.shape, hardness))
        state.set("uplift_m_year", np.zeros(state.grid.shape))
        before = float(state.elevation.sum())
        incise(state, k=2.0e-4, m=0.45, n=1.0, dt_years=80.0,
               rain=state.get("rainfall_mm_day"))
        exported = float(state.features["exported_thickness_sum"])
        thickness_budget(before, float(state.elevation.sum()), 0.0, exported, atol=0.05)
        check_state(state)
        return state

    wet = ramp(8.0, 0.0)
    dry = ramp(0.2, 0.0)
    hard = ramp(8.0, 0.95)
    # The ridge loses rock. Downstream cells receive that rock as sand, so a
    # mid-slope average barely moves. The outlet sand pile is the carried load.
    assert wet.elevation[0].mean() < dry.elevation[0].mean() - 0.3
    assert hard.elevation[0].mean() > wet.elevation[0].mean() + 0.15
    assert wet.get("sand_m")[-1].mean() > dry.get("sand_m")[-1].mean() + 0.5


def test_confluence_is_a_higher_strahler_order_and_uphill_reaches_are_lifted():
    elevation = np.array([
        [12.0, 9.0, 12.0],
        [9.0, 8.0, 9.0],
        [6.0, 5.0, 6.0],
        [4.0, 2.0, 4.0],
    ], np.float64)
    receivers = -np.ones(elevation.shape, np.int32)
    receivers[0, 0] = 1 * 3 + 0
    receivers[1, 0] = 2 * 3 + 0
    receivers[2, 0] = 2 * 3 + 1
    receivers[0, 2] = 1 * 3 + 2
    receivers[1, 2] = 2 * 3 + 1
    receivers[2, 1] = 3 * 3 + 1
    accumulation = np.zeros_like(elevation)
    accumulation[0, 0] = accumulation[1, 0] = accumulation[2, 0] = 2
    accumulation[0, 2] = accumulation[1, 2] = 2
    accumulation[2, 1] = 5
    accumulation[3, 1] = 8
    graph = build_graph(elevation, receivers, accumulation, threshold=2)
    assert graph.complete
    assert drainage_problems(graph) == []
    junction = graph.node_at(2, 1)
    outlet = graph.node_at(3, 1)
    assert junction.strahler == 2
    assert outlet.strahler == 2
    assert graph.node_at(0, 0).strahler == 1
    junction.elevation_m = 0.5
    enforce_descending(graph, min_drop=0.2)
    assert junction.elevation_m >= outlet.elevation_m + 0.2
    assert drainage_problems(graph) == []


def test_a_cycle_is_reported():
    elevation = np.array([[3.0, 1.0], [1.0, 0.0]])
    receivers = np.array([[1, 0], [-1, -1]], np.int32)
    accumulation = np.full((2, 2), 4.0)
    graph = build_graph(elevation, receivers, accumulation, threshold=1.0)
    assert drainage_problems(graph)


def test_downhill_routing_reaches_the_low_edge():
    elevation = np.linspace(40.0, 0.0, 12)[:, None] * np.ones((1, 8))
    receivers, slope = route(elevation, cell_m=20.0)
    area = accumulate(receivers, np.ones_like(elevation))
    assert area[-1].sum() > area[0].sum()
    assert np.all(receivers[-1] == -1)
    assert slope[0].mean() > 0


def test_viability_is_limited_by_the_scarcest_factor():
    cold = viability(-20.0, 1.0, 1.0, preferred_c=10.0)
    dry = viability(10.0, 0.05, 1.0, preferred_c=10.0)
    fit = viability(10.0, 0.9, 1.0, preferred_c=10.0)
    assert cold < 0.2
    assert dry == pytest.approx(0.05)
    assert fit > 0.8


def test_scenarios_share_stages_and_change_the_landscape():
    assert set(scenario_names()) >= {
        "alpine", "arid", "tropical", "volcanic", "temperate",
        "arctic", "coastal", "artificial",
    }
    shape = (20, 24)
    cell = 80.0
    arid = run("arid", shape=shape, cell_m=cell, seed=2, passes=1, dt_years=120.0)
    tropical = run("tropical", shape=shape, cell_m=cell, seed=2, passes=1, dt_years=120.0)
    assert tropical.get("vegetation").mean() > arid.get("vegetation").mean() + 0.4
    # Rain raises the incision. The cut rock is not put back into the bedrock.
    assert tropical.get("bedrock_m").sum() < arid.get("bedrock_m").sum()
    assert arid.features["time_years"] == pytest.approx(120.0)
    again = run("tropical", shape=shape, cell_m=cell, seed=2, passes=1, dt_years=120.0)
    assert again.elevation == pytest.approx(tropical.elevation)

    grid = TerrainState(shape, cell).grid
    volcanic = parameters_for("volcanic", shape, cell, seed=1)
    uplift = volcanic["uplift"].sample_grid(grid)
    assert uplift[shape[0] // 2, shape[1] // 2] > uplift[0, 0] * 3
    coastal = parameters_for("coastal", shape, cell, seed=1)["rainfall"].sample_grid(grid)
    assert coastal[-1].mean() > coastal[0].mean()

    state = TerrainState((6, 6), 30.0)
    ctx = Context(seed=0, dt_years=10.0, parameters=parameters_for("temperate", (6, 6), 30.0))
    Scheduler([SeedRelief(), PaintClimate()], [], passes=0).run(state, ctx)
    assert "stream_power" in known()
    create("stream_power")
    assert RadialField((0, 0), 10.0, 1.0, 20.0).sample(0.0, 0.0) == pytest.approx(1.0, abs=0.05)


def test_legacy_world_keeps_its_height_on_the_layered_state():
    from terrain import generate
    from terrain.export import METRES_PER_UNIT

    world = generate(
        verbose=False, width=32, height=24, seed=5, n_plates=4,
        erosion_model="none", erosion_passes=0, thermal_iters=0, n_hotspots=1,
    )
    state = import_world(world)
    expected = np.asarray(world.height, np.float64) * METRES_PER_UNIT
    assert state.elevation == pytest.approx(expected, rel=1e-5, abs=1e-2)
    assert state.get("water_m").min() >= 0
    assert state.get("vegetation").min() >= 0
    check_state(state)
    pipe = TerrainState(world.height.shape, state.grid.cell_m)
    Pipeline([create("legacy_import")]).run(
        pipe, Context(seed=5, dt_years=1.0, parameters={"legacy_world": world}))
    assert pipe.elevation == pytest.approx(state.elevation)
