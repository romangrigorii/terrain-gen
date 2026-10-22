"""A tile matches its neighbors without those neighbors having been built."""
import numpy as np

from terrain.planet_tiles import (
    GLOBAL_OPERATIONS, LOCAL_OPERATIONS, build_planet, canonical_vertices,
    generate_planet_tile, operation_scope, sample_global_height,
)


def _planet():
    return build_planet(7, cells_x=64, cells_y=32, tile=16, halo=4)


def test_adjacent_tiles_share_the_vertex_and_the_normal():
    state = _planet()
    left = generate_planet_tile(state, 1, 1)
    right = generate_planet_tile(state, 2, 1)
    np.testing.assert_array_equal(left.height[:, -1], right.height[:, 0])
    np.testing.assert_array_equal(left.normal[:, -1], right.normal[:, 0])
    np.testing.assert_array_equal(left.bedrock[:, -1], right.bedrock[:, 0])
    assert left.height.shape == (17, 17)
    assert left.normal.shape == (17, 17, 3)
    length = np.linalg.norm(left.normal, axis=-1)
    np.testing.assert_allclose(length, 1.0, atol=1e-5)


def test_the_date_line_is_the_same_vertex():
    state = _planet()
    west = generate_planet_tile(state, 0, 1)
    east = generate_planet_tile(state, state.grid.tiles_x - 1, 1)
    np.testing.assert_array_equal(west.height[:, 0], east.height[:, -1])
    np.testing.assert_array_equal(west.normal[:, 0], east.normal[:, -1])


def test_both_poles_match_across_a_shared_edge():
    state = _planet()
    for ty in (0, state.grid.tiles_y - 1):
        left = generate_planet_tile(state, 0, ty)
        right = generate_planet_tile(state, 1, ty)
        row = 0 if ty == 0 else -1
        np.testing.assert_array_equal(left.height[:, -1], right.height[:, 0])
        np.testing.assert_array_equal(left.normal[:, -1], right.normal[:, 0])
        assert np.allclose(left.bedrock[row], left.bedrock[row, 0])
        assert np.isfinite(left.normal[row]).all()


def test_stepping_past_a_pole_lands_on_the_opposite_meridian():
    state = _planet()
    grid = state.grid
    half = grid.cells_x // 2
    direct = sample_global_height(np.array([1]), np.array([half + 3]), state)
    wrapped = sample_global_height(np.array([-1]), np.array([3]), state)
    np.testing.assert_array_equal(direct, wrapped)
    north = sample_global_height(np.array([grid.cells_y - 1]), np.array([half + 5]), state)
    beyond = sample_global_height(np.array([grid.cells_y + 1]), np.array([5]), state)
    np.testing.assert_array_equal(north, beyond)
    gy, gx = canonical_vertices(np.array([-1, grid.cells_y + 1]), np.array([3, 5]),
                                grid.cells_y, grid.cells_x)
    np.testing.assert_array_equal(gy, np.array([1, grid.cells_y - 1]))
    np.testing.assert_array_equal(gx, np.array([half + 3, half + 5]))


def test_generation_order_does_not_change_a_tile():
    first = _planet()
    second = _planet()
    alone = generate_planet_tile(first, 4, 1)
    generate_planet_tile(second, 3, 0)
    generate_planet_tile(second, 0, 1)
    later = generate_planet_tile(second, 4, 1)
    np.testing.assert_array_equal(alone.height, later.height)
    np.testing.assert_array_equal(alone.normal, later.normal)
    np.testing.assert_array_equal(alone.bedrock, later.bedrock)


def test_a_short_halo_cannot_pretend_to_smooth():
    state = build_planet(1, cells_x=32, cells_y=16, tile=16, halo=1)
    try:
        generate_planet_tile(state, 0, 0, smooth_passes=1)
    except ValueError as exc:
        assert "halo" in str(exc)
    else:
        raise AssertionError("a one-cell halo was accepted for a smooth pass")


def test_local_and_global_operations_stay_apart():
    for name in LOCAL_OPERATIONS:
        assert operation_scope(name) == "local"
    for name in GLOBAL_OPERATIONS:
        assert operation_scope(name) == "global"


def test_a_coarse_world_is_sampled_the_same_way_from_every_tile():
    coarse = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    state = build_planet(3, cells_x=64, cells_y=32, tile=16, halo=4, coarse=coarse)
    left = generate_planet_tile(state, -1, 0)
    right = generate_planet_tile(state, 0, 0)
    np.testing.assert_array_equal(left.height[:, -1], right.height[:, 0])
    np.testing.assert_array_equal(left.bedrock[:, -1], right.bedrock[:, 0])
