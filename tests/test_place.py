"""Brushes that stamp ice, dunes, shores, and caves onto a generated heightmap."""
import numpy as np
import pytest

from terrain import generate
from terrain.v2.generation.place import place_height


def _dab(kind, metres, radius=0.16, at=(0.5, 0.5)):
    return {"kind": kind, "metres": metres, "radius": radius, "points": [at]}


def test_mountains_are_not_stamped_here():
    height = np.full((12, 16), np.float32(0.04))
    same, ice, sand, void = place_height(height, [_dab("mountain", 2000)], 40.0)
    assert same is height
    assert ice is None and sand is None and void is None


def test_ice_cuts_the_painted_ground_and_leaves_the_rim():
    height = np.full((18, 24), np.float32(0.05))
    out, ice, sand, void = place_height(height, [_dab("glacial", 400)], 40.0)
    assert np.array_equal(out[0], height[0])
    assert np.array_equal(out[:, 0], height[:, 0])
    assert out[9, 12] < height[9, 12] - 20.0 / 12000.0
    assert out[2, 2] == pytest.approx(height[2, 2])
    assert ice[9, 12] > 50.0
    assert ice[0, 0] == 0.0


def test_dunes_pile_sand_inside_the_brush():
    height = np.full((16, 20), np.float32(0.04))
    out, ice, sand, void = place_height(height, [_dab("dune", 160, radius=0.2)], 30.0)
    assert sand.sum() > 200.0
    assert sand[0].sum() == 0.0
    assert out.max() > height.max()
    assert out[2, 2] == pytest.approx(float(height[2, 2]))
    assert np.array_equal(out[0], height[0])


def test_coast_brush_cuts_only_the_painted_shore():
    height = np.full((16, 22), np.float32(400.0 / 12000.0))
    height[:, :8] = np.float32(-40.0 / 12000.0)
    stroke = {
        "kind": "coast",
        "metres": 200.0,
        "radius": 0.1,
        "points": [(y, 8 / 21) for y in (0.35, 0.5, 0.65)],
    }
    out, ice, sand, void = place_height(height, [stroke], 40.0)
    assert out[8, 8] < height[8, 8] - 40.0 / 12000.0
    assert out[8, 16] == pytest.approx(float(height[8, 16]))
    assert np.array_equal(out[:, -1], height[:, -1])
    assert sand[8, 8] > 0.0


def test_cave_opens_a_hollow_and_keeps_a_void():
    height = np.full((16, 20), np.float32(800.0 / 12000.0))
    out, ice, sand, void = place_height(height, [_dab("cave", 80, radius=0.18)], 40.0)
    assert out[8, 10] < height[8, 10] - 10.0 / 12000.0
    assert void[8, 10] > 20.0
    assert out[2, 2] == pytest.approx(float(height[2, 2]))
    assert void[2, 2] == pytest.approx(0.0)
    assert np.array_equal(out[-1], height[-1])


def test_generate_bakes_a_dune_into_the_tile():
    common = dict(verbose=False, width=32, height=24, seed=5, n_plates=4,
                  erosion_model="none", erosion_passes=0, thermal_iters=0, n_hotspots=1)
    plain = generate(**common)
    stamped = generate(**common, sculpt=[_dab("dune", 200, radius=0.22)])
    assert plain.placed_sand is None
    assert stamped.placed_sand.max() > 0.0
    assert np.array_equal(stamped.height[0], plain.height[0])
    assert not np.array_equal(stamped.height, plain.height)
