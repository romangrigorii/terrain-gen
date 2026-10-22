"""Shared landforms, colors, and moisture agree across a tile cut."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from terrain.climate import _march
from terrain.export import elevation_bounds, heightmap
from terrain.features import FeatureSet, TerrainFeature, build_features
from terrain.poisson_reconcile import reconcile
from terrain.render import _cell_noise, color_stats, hypsometric
from terrain.trend_patches import blend_frequency_bands
from terrain.world import refresh_flow


def _ridge():
    points = np.stack((np.full(9, 16.0), np.linspace(4.0, 60.0, 9)), axis=1)
    feature = TerrainFeature(
        7, "ridge", points, np.full(9, 3600.0), width_m=5000.0, influence_m=12000.0)
    return FeatureSet((feature,), 32, 64, 64000.0)


def test_one_ridge_is_the_same_field_in_every_tile():
    features = _ridge()
    gy, gx = np.mgrid[0:33, 0:65]
    whole = features.evaluate(gy, gx)
    tile = 16
    order = [(1, 0), (0, 1), (1, 2), (0, 3), (0, 0), (1, 1), (0, 2), (1, 3)]
    for ty, tx in order:
        ys = ty * tile + np.arange(tile + 1)
        xs = tx * tile + np.arange(tile + 1)
        part = features.evaluate(ys[:, None], xs[None, :])
        np.testing.assert_allclose(part, whole[np.ix_(ys, xs)], atol=1e-6)
    along = [float(features.evaluate(16, x)) for x in (8, 24, 40, 56)]
    assert min(along) > 0.2
    assert max(along) - min(along) < 0.05


def test_a_river_keeps_one_outlet_across_three_tiles():
    points = np.stack((np.full(8, 12.0), np.linspace(2.0, 58.0, 8)), axis=1)
    river = TerrainFeature(
        4, "river", points, np.linspace(800.0, 20.0, 8), width_m=2000.0, influence_m=6000.0)
    features = FeatureSet((river,), 32, 64, 64000.0)
    first = features.drainage()
    again = build_features(4, 32, 64, 64000.0).drainage()
    assert first[0]["feature_id"] == 4
    assert first == features.drainage()
    windows = [features.crossing(0, 32, x, x + 16) for x in (0, 16, 32)]
    assert all(hit and hit[0].feature_id == 4 for hit in windows)
    assert again


def test_global_cell_noise_does_not_restart_in_a_tile():
    full = _cell_noise((12, 20), 3)
    tile = _cell_noise((4, 5), 3, origin=(2, 6))
    np.testing.assert_array_equal(tile, full[2:6, 6:11])


def test_the_same_elevation_keeps_its_color_in_every_tile():
    field = np.linspace(-1.0, 3.0, 200, dtype=np.float64).reshape(10, 20)
    stats = color_stats(field)
    low = np.full((2, 2), 0.4)
    high = np.array([[0.4, 2.8], [2.2, 1.5]])
    shared_low = hypsometric(low, stats=stats)
    shared_high = hypsometric(high, stats=stats)
    np.testing.assert_allclose(shared_low[0, 0], shared_high[0, 0], atol=1e-5)
    assert not np.allclose(hypsometric(low)[0, 0], shared_low[0, 0])


def test_shared_height_bounds_encode_one_physical_height(tmp_path):
    left = np.array([[0.0, 1.0]], np.float32)
    right = np.array([[1.0, 5.0]], np.float32)
    lo, hi = elevation_bounds(left, right)
    heightmap(left, tmp_path / "left.png", lo=lo, hi=hi)
    heightmap(right, tmp_path / "right.png", lo=lo, hi=hi)
    heightmap(left, tmp_path / "own.png")
    shared_left = np.array(Image.open(tmp_path / "left.png"))
    shared_right = np.array(Image.open(tmp_path / "right.png"))
    own = np.array(Image.open(tmp_path / "own.png"))
    assert shared_left[0, 1] == shared_right[0, 0]
    assert own[0, 1] != shared_left[0, 1]


def test_moisture_arriving_at_a_tile_matches_the_global_march():
    cfg = SimpleNamespace(rain_base=0.25, rain_orog=1.0, rain_recycle=0.3, rain_ocean_gain=0.4)
    height = np.tile(np.linspace(0.2, 1.4, 8), (3, 1))
    land = np.ones(height.shape, dtype=bool)
    belts = np.ones(height.shape)
    cap = np.full(height.shape, 1.2)
    rain, entering = _march(height, land, belts, cap, cfg, trace=True)
    mid = 4
    carried = _march(
        height[:, mid:], land[:, mid:], belts[:, mid:], cap[:, mid:], cfg,
        inbound=entering[:, mid], upwind=height[:, mid - 1])
    np.testing.assert_allclose(carried, rain[:, mid:], atol=1e-8)


def test_poisson_join_keeps_the_interior_when_the_border_already_matches():
    source = np.random.default_rng(0).normal(size=(10, 12))
    free = np.zeros(source.shape, dtype=bool)
    free[1:-1, 1:-1] = True
    out = reconcile(source, source, free)
    np.testing.assert_allclose(out, source, atol=1e-5)
    np.testing.assert_allclose(out[~free], source[~free], atol=1e-6)


def test_coarse_bands_cross_a_seam_and_repeat():
    left = np.ones((16, 16))
    right = np.full((16, 16), 3.0)
    ramp = np.linspace(1.0, 0.0, 16)
    weight_left = np.broadcast_to(ramp, (16, 16)).copy()
    weight_right = 1.0 - weight_left
    blended = blend_frequency_bands([left, right], [weight_left, weight_right], levels=2)
    again = blend_frequency_bands([left, right], [weight_left, weight_right], levels=2)
    np.testing.assert_array_equal(blended, again)
    assert 1.4 < float(blended[8, 8]) < 2.6


def test_flow_follows_the_height_it_was_given():
    height = np.repeat(np.linspace(1.0, 0.0, 8), 6).reshape(8, 6).astype(np.float32)
    weights = np.ones_like(height)
    _filled, flow, _rec = refresh_flow(height, weights)
    assert flow.shape == height.shape
    assert float(flow[-1].mean()) > float(flow[0].mean())
