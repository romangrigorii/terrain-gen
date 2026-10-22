"""Overlapping patches stay continuous across a tile edge, and the pole keeps its slope."""
import numpy as np
import pytest

from terrain.planet_tiles import TileConfig, TiledPlanet, build_planet, sample_coarse_height
from terrain.trend_patches import (
    PatchBlender, continue_terrain_trend, continue_terrain_trend_quintic,
    example_patch_sampler, quintic_window,
)


def _window_derivatives(t):
    step = 1e-5
    here = quintic_window(t)
    ahead = quintic_window(t + step)
    behind = quintic_window(t - step)
    first = (ahead - behind) / (2 * step)
    second = (ahead - 2 * here + behind) / step ** 2
    return here, first, second


def test_quintic_window_is_flat_at_both_ends():
    for end in (0.0, 1.0):
        value, first, second = _window_derivatives(end)
        assert value == 0.0
        assert abs(first) < 1e-6
        assert abs(second) < 1e-2
    assert quintic_window(0.5) == 1.0
    _, first, second = _window_derivatives(0.5)
    assert abs(first) < 1e-6
    assert abs(second) < 1e-2


def test_patch_weights_cover_the_planet_once():
    blender = _sine_blender()
    y = np.linspace(0.0, blender.world_height, 9)
    x = np.linspace(0.0, blender.world_width - 1.0, 16)
    total = blender.weight_sum(y[:, None], x[None, :])
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_a_global_wave_has_no_kink_on_a_tile_boundary():
    blender = _sine_blender()
    stride = blender.stride
    x = np.arange(blender.world_width, dtype=np.float64)
    y = np.full(x.shape, float(stride))
    got = blender.correction(y, x)
    want = np.sin(2.0 * np.pi * x / blender.world_width)
    np.testing.assert_allclose(got, want, atol=1e-5)
    curve = np.diff(got, n=2)
    boundary = []
    for edge in range(stride, blender.world_width - 1, stride):
        boundary.append(curve[edge - 1])
    assert max(abs(value) for value in boundary) <= np.max(np.abs(curve)) + 1e-5


def test_a_canyon_is_still_deep_on_the_tile_edge():
    stride, support = 16, 32
    sampler = example_patch_sampler(stride, 64, seed=3, amplitude=1.0)
    blender = PatchBlender(64, 32, stride, sampler, support=support)
    center = sampler(0, 1)[support // 2, support // 2]
    edge = blender.correction(stride, 2 * stride)
    assert center < -0.6
    np.testing.assert_allclose(edge, center, atol=1e-5)
    left = set(blender.patch_ids(1, 0, tile=stride))
    right = set(blender.patch_ids(2, 0, tile=stride))
    assert left & right


def test_tiles_match_without_their_neighbor_and_across_the_date_line():
    planet, other, blender = _planets()
    alone = planet.tile(1, 0, halo=4, smooth_passes=1)
    other.tile(2, 0, halo=4, smooth_passes=1)
    other.tile(0, 0, halo=4, smooth_passes=1)
    later = other.tile(1, 0, halo=4, smooth_passes=1)
    np.testing.assert_array_equal(alone.height, later.height)
    np.testing.assert_array_equal(alone.normal, later.normal)
    right = planet.tile(2, 0, halo=4, smooth_passes=1)
    np.testing.assert_array_equal(alone.height[:, -1], right.height[:, 0])
    np.testing.assert_array_equal(alone.normal[:, -1], right.normal[:, 0])
    west = planet.tile(0, 0, halo=4, smooth_passes=1)
    east = planet.tile(planet.state.grid.tiles_x - 1, 0, halo=4, smooth_passes=1)
    np.testing.assert_array_equal(west.height[:, 0], east.height[:, -1])
    shared = blender.correction(np.arange(17), np.full(17, 32.0))
    assert float(np.min(shared)) < -0.4


def test_a_cached_patch_reloads_the_same_correction(tmp_path):
    sampler = example_patch_sampler(16, 64, seed=5)
    first = PatchBlender(64, 32, 16, sampler, cache_dir=tmp_path)
    created = first.correction(8, 8)
    second = PatchBlender(64, 32, 16, lambda py, px: np.zeros((33, 33)), cache_dir=tmp_path)
    np.testing.assert_allclose(second.correction(8, 8), created, atol=1e-6)


def test_hermite_carries_height_and_slope_into_the_next_region():
    width = 10.0
    start, end = 2.0, 5.0
    slope_start, slope_end = 0.4, -0.25
    assert continue_terrain_trend(0.0, width, start, slope_start, end, slope_end) == start
    assert continue_terrain_trend(width, width, start, slope_start, end, slope_end) == end
    eps = 1e-4
    rising = continue_terrain_trend(
        np.array([0.0, eps]), width, start, slope_start, end, slope_end)
    arriving = continue_terrain_trend(
        np.array([width - eps, width]), width, start, slope_start, end, slope_end)
    assert (rising[1] - rising[0]) / eps == pytest.approx(slope_start, abs=1e-4)
    assert (arriving[1] - arriving[0]) / eps == pytest.approx(slope_end, abs=1e-4)
    line = continue_terrain_trend(np.linspace(0, width, 5), width, 0.0, 1.0, width, 1.0)
    np.testing.assert_allclose(line, np.linspace(0, width, 5), atol=1e-6)


def test_quintic_trend_also_carries_curvature():
    width = 8.0
    curve = -0.2
    eps = 1e-3
    samples = continue_terrain_trend_quintic(
        np.array([0.0, eps, 2 * eps]), width, 1.0, 0.3, curve, 4.0, -0.1, 0.05)
    second = (samples[2] - 2 * samples[1] + samples[0]) / eps ** 2
    assert second == pytest.approx(curve, abs=2e-3)


def test_the_pole_keeps_a_slope_instead_of_a_flat_ring():
    cells_y, cells_x = 40, 80
    coarse = np.zeros((cells_y, cells_x), np.float32)
    longitude = (np.arange(cells_x) + 0.5) / cells_x * (2.0 * np.pi)
    for row in range(10):
        coarse[row] = np.cos(longitude) * (10 - row)
        coarse[-1 - row] = np.cos(longitude) * (10 - row)
    state = build_planet(
        1, cells_x=cells_x, cells_y=cells_y, tile=10, halo=4,
        coarse=coarse, features=1, polar_transition=8)
    pole = np.array([
        sample_coarse_height(0, x, state) for x in (0, 20, 40, 60)
    ])
    np.testing.assert_allclose(pole, pole[0], atol=1e-5)
    assert abs(float(pole[0])) < 1.0
    gx = cells_x // 2
    column = np.array([sample_coarse_height(y, gx, state) for y in range(5)])
    step = np.diff(column)
    assert step[0] < -0.5
    assert np.all(step < 0.0)
    assert abs(step[0] - step[1]) < abs(step[0])


def _sine_blender():
    stride, support = 16, 32
    world_width, world_height = 64, 32

    def sampler(py, px):
        xs = px * stride + np.arange(support + 1, dtype=np.float64)
        row = np.sin(2.0 * np.pi * xs / world_width)
        return np.repeat(row[None, :], support + 1, axis=0)

    return PatchBlender(world_width, world_height, stride, sampler, support=support)


def _planets():
    coarse = np.zeros((32, 64), np.float32)
    cfg = TileConfig(world_width=64, world_height=32, tile_cells=16, seed=4, halo=4)
    sampler = example_patch_sampler(16, 64, seed=4)
    blender = PatchBlender(
        64, 32, 16, sampler, polar_transition_cells=6, support=32)

    def make():
        planet = TiledPlanet(coarse, cfg)
        planet.set_patch_blender(blender)
        return planet

    first = make()
    second = TiledPlanet(coarse.copy(), cfg)
    second.set_patch_blender(PatchBlender(
        64, 32, 16, example_patch_sampler(16, 64, seed=4),
        polar_transition_cells=6, support=32))
    return first, second, blender
