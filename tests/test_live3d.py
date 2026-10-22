"""The live view builds a mesh without OpenGL, and draws when a context exists."""
import numpy as np
import pytest

from terrain.export import METRES_PER_UNIT
from terrain.live3d import _mvp, build_geometry, frame_distance, vertical_scale


def _corners_inside(yaw, pitch, distance, plane, view_w, view_h):
    mvp, _eye = _mvp(yaw, pitch, distance, (0.5, 0.5),
                     np.zeros((2, 2), np.float32), view_w, view_h, plane=plane)
    for z in (0.0, 1.0):
        for x in (0.0, 1.0):
            clip = mvp.astype(np.float64) @ np.array(
                [x * plane[0], 0.0, z * plane[1], 1.0], np.float64)
            ndc = clip[:2] / clip[3]
            if float(np.max(np.abs(ndc))) > 1.0:
                return False
    return True


def test_the_camera_fits_every_tile_of_a_tall_sheet():
    _y, z_span = vertical_scale(1438, 640, (0, 0, 1, 3), 200_000)
    plane = (1.0, z_span)
    distance = frame_distance(0.55, 0.78, plane, 1040, 680)
    assert 1.0 < distance < 6.0
    assert _corners_inside(0.55, 0.78, distance, plane, 1040, 680)
    wide_y, wide_z = vertical_scale(480, 1279, (0, 0, 2, 1), 200_000)
    wide = (1.0, wide_z)
    wide_distance = frame_distance(0.9, 0.4, wide, 1040, 680)
    assert wide_distance < 8.0
    assert _corners_inside(0.9, 0.4, wide_distance, wide, 1040, 680)
    assert wide_y > 0


def test_height_uses_the_same_metres_as_the_ground():
    y_scale, z_span = vertical_scale(480, 640, (0, 0, 1, 1), 200_000)
    assert y_scale == pytest.approx(METRES_PER_UNIT / 200_000)
    assert z_span == pytest.approx(479 / 639)
    wide, z_wide = vertical_scale(480, 1279, (0, 0, 2, 1), 200_000)
    assert wide == pytest.approx(METRES_PER_UNIT / 400_000)
    assert z_wide == pytest.approx(479 / 1278)
    # A 12 km peak is 12 km across the ground, not a spike several times that.
    assert 1.0 * y_scale * 200_000 == pytest.approx(METRES_PER_UNIT)


def test_the_mesh_keeps_a_preview_sized_grid():
    from terrain.live3d import _resolution, MAX_EDGE
    assert _resolution(480, 640, MAX_EDGE) == (480, 640)


def test_build_geometry_skips_gaps():
    height = np.full((8, 10), -0.2, np.float32)
    height[2:5, 3:6] = 0.4
    color = np.zeros((8, 10, 3), np.uint8)
    color[..., 1] = 180
    mask = np.ones((8, 10), np.float32)
    mask[:, 7:] = 0
    verts, indices, albedo, bed = build_geometry(height, color, mask, max_edge=32)
    assert verts.shape == (80, 8)
    assert indices.size == 7 * 6 * 6
    assert albedo.shape == (8, 10, 3)
    assert albedo[4, 4, 1] == 180
    assert bed[0, 0] == pytest.approx(-0.2)
    assert bed[3, 4] == pytest.approx(0.4)


def test_live_view_draws_land_sea_and_a_moving_surface():
    pytest.importorskip("moderngl")
    from terrain.live3d import LiveTerrain

    try:
        view = LiveTerrain()
    except Exception as err:
        pytest.skip(f"no OpenGL context: {err}")
    height = np.full((48, 64), -0.35, np.float32)
    height[4:16, 6:18] = 0.7
    color = np.zeros(height.shape + (3,), np.uint8)
    color[..., 1] = 150
    color[:8, :8] = (40, 90, 40)
    try:
        view.set_sheet(height, color, None, (0, 0, 1, 1), ("sheet",))
        first, pick_y, pick_x = view.draw(0.4, 0.85, 1.15, 160, 110, (0.5, 0.5), 0.0)
        second, _, _ = view.draw(0.4, 0.85, 1.15, 160, 110, (0.5, 0.5), 2.4)
    finally:
        view.close()
    assert first.shape == (110, 160, 3)
    sky = np.array([166, 196, 230], np.int16)
    assert int(np.abs(first.astype(np.int16) - sky).sum()) > 1000
    assert int(np.abs(first.astype(np.int16) - second.astype(np.int16)).sum()) > 0
    hit = (pick_y >= 0) & (pick_x >= 0)
    assert hit.any()
    mid = hit[50:60, 75:85]
    assert mid.any()
    assert pick_y[50:60, 75:85][mid].mean() == pytest.approx(0.5, abs=0.2)
    assert pick_x[50:60, 75:85][mid].mean() == pytest.approx(0.5, abs=0.2)


def test_a_framed_view_includes_both_ends_of_the_sheet():
    pytest.importorskip("moderngl")
    from terrain.live3d import LiveTerrain, frame_distance

    try:
        view = LiveTerrain()
    except Exception as err:
        pytest.skip(f"no OpenGL context: {err}")
    height = np.full((36, 96), 0.25, np.float32)
    color = np.zeros(height.shape + (3,), np.uint8)
    color[:, :48, 0] = 220
    color[:, 48:, 2] = 220
    try:
        view.set_sheet(height, color, None, (0, 0, 2, 1), ("wide",), 200_000.0)
        distance = frame_distance(0.55, 0.78, view.plane, 320, 200)
        _rgb, pick_y, pick_x = view.draw(0.55, 0.78, distance, 320, 200, (0.5, 0.5), 0.0)
    finally:
        view.close()
    hit = (pick_y >= 0) & (pick_x >= 0)
    assert (hit & (pick_x < 0.35)).any()
    assert (hit & (pick_x > 0.65)).any()
