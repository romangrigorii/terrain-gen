"""Fast regression tests. Run with PYTHONPATH=. pytest -q."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from terrain import elevation, grid, hydrology, derived, export
from terrain.world import Config, generate
from terrain.presets import PRESETS, make_config


def make_world(**kw):
    return generate(Config(width=96, height=72, seed=9, **kw), verbose=False)


def test_presets_do_not_mutate_input():
    cfg = Config(width=96, height=72, seed=1)
    original = asdict(cfg)
    opts = make_config('arid', cfg, seed=44)
    assert opts.seed == 44 and opts.wind_mode == 'belts'
    assert asdict(cfg) == original
    w = generate(cfg, verbose=False, seed=3, wind_mode='westerly')
    assert w.cfg.seed == 3
    assert cfg.seed == 1 and cfg.wind_mode == 'belts'
    assert asdict(cfg) == original


def test_bad_options_fail_early():
    for args, error in [({'widht': 100}, TypeError),
                        ({'land_fraction': 1.2}, ValueError),
                        ({'width': 2}, ValueError),
                        ({'erosion_model': 'broken'}, ValueError),
                        ({'wind_mode': 'broken'}, ValueError),
                        ({'terrain_model': 'broken'}, ValueError)]:
        with pytest.raises(error):
            generate(verbose=False, **args)


def test_distance_to_missing_source_is_infinite():
    mask = np.ones((10, 15), bool)
    assert np.isinf(grid.edt(mask)).all()
    assert np.all(grid.edt(~mask) == 0)


def test_edt_wrap_seam():
    m = np.ones((11, 19), bool)
    m[5, 0] = False
    assert grid.edt(m)[5, -1] == 1.0
    assert grid.edt(m)[5, 1] == 1.0


def test_priority_flood_drains_interior_basin():
    h = np.ones((11, 15), dtype=np.float32)
    h[2:-2, :] = 4
    h[4:7, 5:10] = 2
    filled = hydrology.priority_flood(h, eps=1e-5)
    assert np.isfinite(filled).all()
    assert (filled >= h).all()
    rec, _, _ = hydrology.flow_routing(filled)
    for y in range(1, 10):
        for x in range(15):
            k = y * 15 + x
            for _ in range(11*15):
                nxt = int(rec.ravel()[k])
                if nxt == k:
                    break
                k = nxt
            else:
                pytest.fail('cycle in priority flooded DEM')
            assert k // 15 in (0, 10), (y, x, k)


@pytest.mark.parametrize('mode', ['iterative', 'priority_flood'])
@pytest.mark.parametrize('erosion', ['stream_power', 'thermal', 'none'])
def test_world_modes(mode, erosion):
    w = make_world(depression_method=mode, erosion_model=erosion)
    assert w.height.shape == (72, 96)
    for a in (w.height, w.runoff, w.temp, w.swing, w.water.flow, w.water.lake_depth):
        assert np.isfinite(a).all()
    assert w.water.lake_depth.min() >= 0
    assert set(np.unique(w.biome)).issubset(set(range(12)))
    assert (w.trees.cover >= 0).all() and (w.trees.cover <= 1).all()
    if erosion == 'none':
        np.testing.assert_allclose(w.height_eroded, w.height_pre)


@pytest.mark.parametrize('preset', list(PRESETS))
def test_presets(preset):
    w = generate(make_config(preset, width=96, height=72, seed=2), verbose=False)
    assert np.isfinite(w.height).all()
    assert 0 < w.land.mean() < 1


def test_deterministic_default():
    w1, w2 = make_world(), make_world()
    np.testing.assert_array_equal(w1.height, w2.height)
    np.testing.assert_array_equal(w1.biome, w2.biome)
    np.testing.assert_array_equal(w1.trees.pos, w2.trees.pos)


def test_advanced_layers(tmp_path):
    PIL = pytest.importorskip('PIL.Image')
    w = make_world()
    n = derived.normals(w.height)
    np.testing.assert_allclose(np.linalg.norm(n, axis=2), 1., atol=1e-6)
    m = derived.material_weights(w)
    np.testing.assert_allclose(m[w.land & ~w.lakes & ~w.rivers].sum(axis=1), 1., atol=1e-6)
    assert not m[(~w.land) | w.lakes | w.rivers].any()
    assert derived.flow_directions(w).shape == (*w.height.shape, 2)
    d = export.bundle(w, parent=tmp_path, advanced=True)
    for name in ('height16.png','materials.png', 'normals.png', 'wetness.png'):
        im = PIL.open(d / name)
        assert im.size == (96, 72), name
        im.load()
    with np.load(d / 'terrain_fields.npz') as arr:
        np.testing.assert_array_equal(arr['elevation'], w.height)
    with open(d / 'config.json') as f:
        cfg = json.load(f)
    assert cfg['seed'] == 9


def test_wind_modes_distinct():
    a = make_world(wind_mode='easterly')
    b = make_world(wind_mode='westerly')
    assert np.mean(abs(a.runoff-b.runoff)) > 0.01


def test_lake_label_at_periodic_seam():
    m = np.zeros((6, 8), dtype=bool)
    m[2, 0] = m[3, 7] = True
    m[5, 4] = True
    labels, n = grid.label_wrap_x(m)
    assert n == 2
    assert labels[2, 0] == labels[3, 7] != 0
    assert labels[5, 4] != labels[2, 0]
    assert not labels[0].any()


def _half_corr_lag(field, axis, limit):
    field = np.asarray(field, np.float64)
    field = field - field.mean()
    var = float(np.mean(field * field)) + 1e-12
    for lag in range(1, limit):
        if axis == 0:
            corr = float(np.mean(field[:-lag] * field[lag:])) / var
        else:
            corr = float(np.mean(field[:, :-lag] * field[:, lag:])) / var
        if corr < 0.5:
            return lag
    return limit


def test_noise_cells_are_square_on_a_wide_map():
    from terrain import noise
    field = noise.fbm(90, 120, np.random.default_rng(0), periods=2, octaves=1)
    y_lag = _half_corr_lag(field, 0, 40)
    x_lag = _half_corr_lag(field, 1, 40)
    assert abs(y_lag - x_lag) <= max(2, 0.2 * max(y_lag, x_lag))


def test_a_strong_warp_does_not_fold():
    from terrain import noise
    rng = np.random.default_rng(1)
    dy = noise.fbm(48, 64, rng, periods=2, octaves=4) * 80
    dx = noise.fbm(48, 64, rng, periods=2, octaves=4) * 80
    dy, dx = noise.limit_displacement(dy, dx)
    assert noise._peak_gradient(dy, dx) <= 0.85 + 1e-6


def _boundary_world(kind):
    """A straight boundary down the columns, so a profile is just the rows."""
    from terrain.config import Config
    from terrain.tectonics import Tectonics
    h, w = 96, 8
    dist = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)).copy()
    zeros = np.zeros((h, w), np.float32)
    ones = np.ones((h, w), np.float32)
    conv = rift = shear = zeros
    cont_self = cont_other = overriding = ones
    crust = ones.copy()
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0] = True
    if kind == "subduction":
        conv = ones.copy()
        cont_other = zeros.copy()
    elif kind == "rift":
        rift = ones.copy()
    elif kind == "transform":
        shear = ones.copy()
    else:
        raise ValueError(kind)
    return Tectonics(
        plate=np.zeros((h, w), np.int32), seeds=np.zeros((1, 2)),
        vel=np.zeros((1, 2)), plate_cont=np.zeros(1, dtype=bool),
        buoyancy=np.zeros(1), crust=crust, is_cont=crust > 0.5,
        vy=zeros, vx=zeros, div=zeros, curl=zeros, boundary=boundary,
        dist=dist, conv=conv, rift=rift, shear=shear,
        cont_self=cont_self, cont_other=cont_other, overriding=overriding,
    ), Config(width=w, height=h, upland_h=0.0, age_depth=0.0, arc_h=0.0)


def _interior_peaks(profile):
    return np.flatnonzero(
        (profile[1:-1] > profile[:-2] + 1e-4) & (profile[1:-1] > profile[2:] + 1e-4))


def test_a_plate_boundary_is_one_ridge():
    from terrain.elevation import tectonic_relief
    tect, cfg = _boundary_world("subduction")
    relief = tectonic_relief(tect, cfg, np.random.default_rng(0))
    for column in relief.T:
        assert len(_interior_peaks(column)) == 1
    tect, cfg = _boundary_world("rift")
    relief = tectonic_relief(tect, cfg, np.random.default_rng(0))
    for column in relief.T:
        assert len(_interior_peaks(column)) == 0
        assert np.all(np.diff(column) >= -1e-5)
    tect, cfg = _boundary_world("transform")
    relief = tectonic_relief(tect, cfg, np.random.default_rng(0))
    for column in relief.T:
        assert len(_interior_peaks(column)) == 0


def test_a_fine_warp_does_not_ripple_the_broad_one():
    from terrain import noise
    rng = np.random.default_rng(2)
    coarse = noise.fbm(48, 64, rng, periods=2, octaves=2) * 6
    coarse_x = noise.fbm(48, 64, rng, periods=2, octaves=2) * 6
    fine = noise.fbm(48, 64, rng, periods=10, octaves=4) * 40
    fine_x = noise.fbm(48, 64, rng, periods=10, octaves=4) * 40
    dy, dx = noise.stack_displacement(((coarse, coarse_x), (fine, fine_x)))
    assert noise._peak_gradient(dy, dx) <= 0.85 + 1e-6
    # The broad bend is the plate. The fine layer must not replace it.
    keep = float(np.corrcoef(dy.ravel(), np.asarray(coarse, np.float64).ravel())[0, 1])
    assert keep > 0.9


def test_numba_noise_matches_numpy():
    pytest.importorskip('numba')
    from terrain import noise
    with noise.use_backend('numpy'):
        a = noise.perlin(64, 96, 4, 8, np.random.default_rng(29))
    with noise.use_backend('numba'):
        b = noise.perlin(64, 96, 4, 8, np.random.default_rng(29))
    np.testing.assert_allclose(a, b, atol=5e-7, rtol=0)


def test_sediment_stage_matches_the_eroded_surface():
    world = generate(Config(width=48, height=36, seed=4, erosion_model="none",
                            sediment=True, sediment_iters=3), verbose=False)
    assert world.bedrock is not None and world.sediment is not None
    np.testing.assert_allclose(world.bedrock + world.sediment, world.height_eroded, atol=1e-4)
    assert np.isfinite(world.height).all()
    plain = generate(Config(width=48, height=36, seed=4, erosion_model="none"), verbose=False)
    assert plain.sediment is None


def test_sediment_moves_downhill_and_conserves_mass():
    h, w = 12, 20
    column = np.linspace(1.0, 0.0, w, dtype=np.float32)
    bedrock = np.broadcast_to(column, (h, w)).copy()
    sediment = np.full((h, w), 0.15, np.float32)
    hardness = np.full((h, w), 0.2, np.float32)
    before = float(bedrock.sum() + sediment.sum())
    bedrock2, sediment2, surface = hydrology.sediment_transport(
        bedrock, sediment, hardness, iters=6, k=0.2, bedrock_rate=0.5, wrap_x=False)
    after = float(bedrock2.sum() + sediment2.sum())
    assert abs(after - before) < 1e-3
    assert sediment2[:, -1].mean() > sediment[:, -1].mean()
    assert sediment2[:, 0].mean() < sediment[:, 0].mean()
    np.testing.assert_allclose(surface, bedrock2 + sediment2, atol=1e-5)


def test_adjacent_tiles_share_the_edge_vertex():
    from types import SimpleNamespace
    from terrain.tiles import TileSpec, generate_tile

    height_n, width_n = 24, 32
    yy, xx = np.mgrid[0:height_n, 0:width_n]
    height = np.sin(xx / 4.0).astype(np.float32) + (yy / height_n).astype(np.float32)
    world = SimpleNamespace(
        cfg=Config(width=width_n, height=height_n, seed=3, world_width_m=8000.0),
        height=height,
        runoff=np.full_like(height, 0.6),
        temp=np.full_like(height, 8.0),
        water=SimpleNamespace(flow=np.full_like(height, 2.0)),
    )
    spec = dict(cells=16, meters_per_cell=10.0, halo=8)
    left = generate_tile(world, TileSpec(0, 0, **spec))
    right = generate_tile(world, TileSpec(1, 0, **spec))
    again = generate_tile(world, TileSpec(0, 0, **spec))
    np.testing.assert_array_equal(left.height, again.height)
    np.testing.assert_allclose(left.height[:, -1], right.height[:, 0], atol=1e-5)
    np.testing.assert_allclose(left.sediment[:, -1], right.sediment[:, 0], atol=1e-5)
    assert left.height.shape == (17, 17)
    assert np.isfinite(left.materials).all()
    np.testing.assert_allclose(left.materials.sum(axis=-1), 1.0, atol=1e-5)


def test_hierarchical_windows_meet():
    from terrain.tiles import hierarchical_window

    macro = np.linspace(0.2, 1.1, 18, dtype=np.float32).reshape(3, 6)
    macro = np.repeat(np.repeat(macro, 2, axis=0), 2, axis=1)
    kwargs = dict(full_h=40, full_w=48, seed=9, world_width_m=4800.0,
                  world_height_m=4000.0, halo=8, iters=4, wrap_x=True)
    whole = hierarchical_window(macro, 4, 8, 12, 20, **kwargs)
    left = hierarchical_window(macro, 4, 8, 12, 10, **kwargs)
    right = hierarchical_window(macro, 4, 18, 12, 10, **kwargs)
    np.testing.assert_allclose(whole, np.hstack([left, right]), atol=1e-5)


def test_hydraulic_conserves_mass_and_deposits_at_a_slope_break():
    width = 24
    column = np.arange(width, dtype=np.float32)
    steep = 1.0 - column * 0.08
    gentle = 0.2 - (column - 10.0) * 0.004
    profile = np.where(column < 10, steep, gentle).astype(np.float32)
    bedrock = np.broadcast_to(profile, (6, width)).copy()
    sediment = np.full_like(bedrock, 0.08)
    soft = np.ones_like(bedrock)
    before = float(bedrock.sum() + sediment.sum())
    state, exported = hydrology.hydraulic_erode(
        bedrock, sediment, soft, iters=3, kc=0.25, ke=0.9, kd=0.7, wrap_x=False)
    after = float(state.bedrock.sum() + state.sediment.sum() + state.suspended.sum())
    assert abs(after + exported - before) < 1e-2
    assert state.sediment[:, 12:18].mean() > state.sediment[:, 1:8].mean()
    assert state.sediment[:, 0].mean() < sediment[:, 0].mean()
    np.testing.assert_allclose(state.height, state.bedrock + state.sediment, atol=1e-6)


def test_hard_strata_erode_slower_than_soft():
    column = np.linspace(1.0, 0.2, 18, dtype=np.float32)
    bedrock = np.broadcast_to(column, (5, 18)).copy()
    sediment = np.zeros_like(bedrock)
    soft = np.ones_like(bedrock)
    hard = np.full_like(bedrock, 0.05)
    soft_state, _ = hydrology.hydraulic_erode(
        bedrock, sediment, soft, iters=3, kc=0.3, wrap_x=False)
    hard_state, _ = hydrology.hydraulic_erode(
        bedrock, sediment, hard, iters=3, kc=0.3, wrap_x=False)
    soft_cut = float((bedrock - soft_state.bedrock).sum())
    hard_cut = float((bedrock - hard_state.bedrock).sum())
    assert soft_cut > hard_cut + 1e-3


def test_strata_follow_the_exposed_bed():
    from terrain.geology import strata_erodability

    bedrock = np.array([100.0, 80.0, 70.0], dtype=np.float64)
    erodability = strata_erodability(bedrock, 100.0, layer_thickness_m=40.0,
                                     metres_per_unit=1.0)
    assert erodability[0] == np.float32(0.85)
    assert erodability[1] == np.float32(0.85)
    assert erodability[2] == np.float32(0.15)
    opened = strata_erodability(np.array([70.0]), 100.0, layer_thickness_m=40.0,
                                metres_per_unit=1.0)
    assert opened[0] == np.float32(0.15)


def test_slump_moves_debris_off_the_pile_and_keeps_it():
    bedrock = np.zeros((16, 16), np.float32)
    sediment = np.zeros((16, 16), np.float32)
    sediment[8, 8] = 1.5
    before = float(sediment.sum())
    out = hydrology.slump_debris(
        bedrock, sediment, repose=0.6, rate=0.8, iters=8, wrap_x=False)
    assert abs(float(out.sum()) - before) < 1e-3
    assert out[8, 8] < sediment[8, 8]
    assert out[8, 7] > 0.0 and out[7, 8] > 0.0
    np.testing.assert_array_equal(bedrock, 0.0)


def test_layered_stream_power_is_opt_in_and_finite():
    column = np.linspace(1.0, 0.0, 16, dtype=np.float32)
    height = np.broadcast_to(column, (8, 16)).copy()
    first = hydrology.stream_power(height, passes=1, thermal_iters=1, k=0.05)
    second = hydrology.stream_power(height, passes=1, thermal_iters=1, k=0.05)
    np.testing.assert_array_equal(first[0], second[0])
    layered = hydrology.stream_power(
        height.copy(), passes=2, thermal_iters=0, k=0.08, layered=True)
    assert layered[0].shape == height.shape
    assert np.isfinite(layered[0]).all()


def test_multiscale_keeps_the_peak_and_adds_a_finer_pass():
    yy, xx = np.mgrid[-16:16, -16:16]
    hill = np.exp(-(yy ** 2 + xx ** 2) / 30.0).astype(np.float32)
    bedrock = hill * np.float32(0.4)
    sediment = np.full_like(hill, np.float32(0.02))
    soft = np.full_like(hill, np.float32(0.8))
    coarse_pass = hydrology.ErosionPass(32.0, 2, 0.6, 0.2, 0.5)
    fine_pass = hydrology.ErosionPass(8.0, 2, 0.5, 0.3, 0.4)
    coarse_b, coarse_s, coarse_h = hydrology.erode_multiscale(
        bedrock, sediment, soft, cell_m=4.0, passes=(coarse_pass,))
    _b, _s, both = hydrology.erode_multiscale(
        bedrock, sediment, soft, cell_m=4.0, passes=(coarse_pass, fine_pass))
    assert both.shape == hill.shape and np.isfinite(both).all()
    peak = np.unravel_index(int(hill.argmax()), hill.shape)
    moved = np.unravel_index(int(both.argmax()), both.shape)
    assert abs(peak[0] - moved[0]) <= 3 and abs(peak[1] - moved[1]) <= 3
    assert not np.allclose(coarse_h, both, atol=1e-5)
    assert coarse_b.shape == hill.shape and coarse_s.shape == hill.shape


def test_lake_seam_in_actual_river_labelling():
    from terrain import rivers
    h = np.full((7, 13), 3.0, dtype=np.float32)
    filled = h.copy()
    filled[3, 0] = filled[4, -1] = 5
    cfg = Config(lake_min_area=1, lake_min_depth=0.1)
    lbl, n = rivers._label_lakes(h, filled, cfg)
    assert n == 1
    assert lbl[3, 0] == lbl[4, -1] == 1


def test_mfd_sinks_hold_the_water_that_entered():
    column = np.linspace(1.0, 0.0, 12, dtype=np.float32)
    height = np.broadcast_to(column, (6, 12)).copy()
    weights = np.full(height.shape, 2.0, np.float32)
    flow = hydrology.accumulate_mfd(height, weights, p=1.5)
    sinks = flow[:, -1].sum()
    assert abs(float(sinks) - float(weights.sum())) < 1e-2


def test_erosion_mask_stays_in_range():
    height = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    height = np.broadcast_to(height, (8, 16)).copy()
    rain = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    rain = np.broadcast_to(rain, height.shape).copy()
    hard = np.full(height.shape, 0.4, np.float32)
    mask = hydrology.erosion_mask(height, rain, hard, cell_m=20.0)
    assert mask.shape == height.shape
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_soft_banks_open_wider_than_hard_banks():
    from terrain import rivers
    height = np.full((14, 18), 1.0, np.float32)
    width = np.zeros_like(height)
    width[:, 9] = 3.0
    cfg = Config(river_incision=0.08, river_width_max=4.0)
    lakes = np.zeros(height.shape, np.int32)
    hard = rivers.carve_channel_profile(height, width, np.ones_like(height), 0.0, lakes, cfg)
    soft = rivers.carve_channel_profile(height, width, np.zeros_like(height), 0.0, lakes, cfg)
    assert int((soft < height - 1e-4).sum()) > int((hard < height - 1e-4).sum())


def test_raindrops_conserve_sediment():
    from terrain.erosion_particles import rain_splash
    column = np.linspace(1.0, 0.0, 16, dtype=np.float32)
    bedrock = np.broadcast_to(column, (8, 16)).copy()
    sediment = np.full_like(bedrock, 0.2)
    before = float(sediment.sum())
    _rock, out = rain_splash(bedrock, sediment, np.ones_like(bedrock), seed=3, steps=6, wrap_x=False)
    assert abs(float(out.sum()) - before) < 1e-2
    assert not np.allclose(out, sediment)


def test_wind_moves_dry_sediment_and_leaves_wet_ground():
    from terrain.aeolian import transport
    bedrock = np.zeros((10, 16), np.float32)
    sediment = np.zeros_like(bedrock)
    sediment[:, :6] = 0.3
    dryness = np.zeros_like(bedrock)
    dryness[:, :8] = 1.0
    before = float(sediment.sum())
    _rock, out = transport(bedrock, sediment, dryness, "westerly", iters=4, wrap_x=True)
    assert abs(float(out.sum()) - before) < 1e-2
    assert out[:, 6:].sum() > sediment[:, 6:].sum()
    wet = np.zeros_like(bedrock)
    _rock, still = transport(bedrock, sediment, wet, "westerly", iters=4, wrap_x=True)
    np.testing.assert_allclose(still, sediment, atol=1e-6)


def test_flow_warp_keeps_the_grid_shape():
    column = np.linspace(1.0, 0.0, 12, dtype=np.float32)
    bedrock = np.broadcast_to(column, (6, 12)).copy()
    sediment = np.full_like(bedrock, 0.05)
    rec = np.arange(bedrock.size, dtype=np.int32).reshape(bedrock.shape)
    rec[:, :-1] = rec[:, 1:]
    rock, loose = hydrology.flow_guided_warp(
        bedrock, sediment, rec, np.ones_like(bedrock), displacement_cells=0.4, substeps=2)
    assert rock.shape == bedrock.shape and loose.shape == sediment.shape
    assert np.isfinite(rock).all() and np.isfinite(loose).all()


def test_layered_and_full_models_settle_sediment():
    layered = generate(Config(width=40, height=32, seed=4, erosion_model="none",
                              terrain_model="layered", sediment_iters=2), verbose=False)
    assert layered.sediment is not None
    np.testing.assert_allclose(layered.bedrock + layered.sediment, layered.height_eroded, atol=1e-3)
    assert np.isfinite(layered.height).all()
    full = generate(Config(width=40, height=32, seed=4, erosion_model="none",
                           terrain_model="full", sediment_iters=2), verbose=False)
    assert full.sediment is not None and np.isfinite(full.height).all()
    assert not np.allclose(full.height_eroded, layered.height_eroded, atol=1e-4)


def test_load_surface_picks_the_rendered_ground():
    from terrain.view3d import load_surface, mark_strokes

    height = np.zeros((48, 64), np.float32)
    color = np.zeros(height.shape + (3,), np.uint8)
    color[..., 1] = 140
    rgb, pick_y, pick_x = load_surface(
        height, color[::2, ::2], yaw=0.4, pitch=0.85, distance=1.15,
        view_w=160, view_h=110, steps=36)
    assert rgb.shape == (110, 160, 3)
    assert pick_y[55, 80] == pytest.approx(0.5, abs=0.15)
    assert pick_x[55, 80] == pytest.approx(0.5, abs=0.15)
    assert rgb[55, 80, 1] > 30
    height[10:16, 12:18] = 1.1
    rgb, pick_y, pick_x = load_surface(
        height, color, yaw=0.2, pitch=0.7, distance=1.3,
        view_w=160, view_h=110, steps=40)
    spike = ((pick_y >= 0)
             & (np.abs(pick_y - 12.5 / 47) < 0.1)
             & (np.abs(pick_x - 14.5 / 63) < 0.1))
    assert spike.any()
    y = float(pick_y[spike][0])
    x = float(pick_x[spike][0])
    marked = mark_strokes(rgb, pick_y, pick_x, [{
        "kind": "mountain", "radius": 0.12, "points": [(y, x)],
    }], y_aspect=48 / 64)
    assert marked.shape == rgb.shape
    assert int(np.abs(marked.astype(np.int16) - rgb.astype(np.int16)).sum()) > 0


def test_stitch_tiles_shares_the_rim_and_leaves_gaps():
    from terrain.view3d import stitch_tiles

    left = np.full((5, 7), 1, np.float32)
    right = np.full((5, 7), 3, np.float32)
    black = np.zeros((5, 7, 3), np.uint8)
    red = np.zeros((5, 7, 3), np.uint8)
    red[..., 0] = 200
    sheet, color, mask, layout = stitch_tiles([
        (0, 0, left, black),
        (1, 0, right, red),
    ])
    assert layout == (0, 0, 2, 1)
    assert sheet.shape == (5, 13)
    assert mask.all()
    assert sheet[2, 0] == 1
    assert sheet[2, 6] == 3  # the shared rim is written by the later tile
    assert sheet[2, -1] == 3
    gapped, _color, gap_mask, gap_layout = stitch_tiles([
        (0, 0, left, black),
        (2, 0, right, red),
    ])
    assert gap_layout == (0, 0, 3, 1)
    assert gapped.shape == (5, 19)
    assert gap_mask[:, :7].all() and gap_mask[:, 12:].all()
    assert not gap_mask[:, 8:12].any()


def test_load_surface_draws_every_tile_and_skips_gaps():
    from terrain.view3d import load_surface, stitch_tiles

    rows, cols = 12, 16
    left_h = np.full((rows, cols), 0.2, np.float32)
    right_h = np.full((rows, cols), 0.2, np.float32)
    green = np.zeros((rows, cols, 3), np.uint8)
    green[..., 1] = 180
    red = np.zeros((rows, cols, 3), np.uint8)
    red[..., 0] = 180
    height, color, mask, _layout = stitch_tiles([
        (0, 0, left_h, green),
        (2, 0, right_h, red),
    ])
    rgb, pick_y, pick_x = load_surface(
        height, color, yaw=0.15, pitch=1.05, distance=1.6,
        view_w=180, view_h=120, steps=48, mask=mask)
    valid = pick_y >= 0
    left = valid & (pick_x < 0.4)
    right = valid & (pick_x > 0.6)
    assert left.any() and right.any()
    assert not (valid & (pick_x > 0.42) & (pick_x < 0.58)).any()
    assert rgb[left, 1].mean() > rgb[left, 0].mean() + 20
    assert rgb[right, 0].mean() > rgb[right, 1].mean() + 20


def _dab(kind, metres, y=0.5, x=0.5, radius=0.18):
    return {"kind": kind, "metres": metres, "radius": radius, "points": [(y, x)]}


def test_paint_relief_adds_height_and_cuts_depth():
    land = np.full((32, 48), 0.25, np.float32)
    mountain = elevation.paint_relief(land, [_dab("mountain", 2400)])
    valley = elevation.paint_relief(land, [_dab("valley", 1200)])
    amp_m = 2400 / export.METRES_PER_UNIT
    amp_v = 1200 / export.METRES_PER_UNIT
    assert mountain[16, 24] == pytest.approx(0.25 + amp_m, abs=1e-3)
    assert valley[16, 24] == pytest.approx(0.25 - amp_v, abs=1e-3)
    both = elevation.paint_relief(land, [_dab("mountain", 2400), _dab("valley", 1200, radius=0.08)])
    assert both[16, 24] == pytest.approx(0.25 + amp_m - amp_v, abs=1e-3)
    assert land[16, 24] == pytest.approx(0.25)
    assert elevation.paint_relief(land, []) is land


def test_paint_relief_lifts_an_island_to_the_requested_crest():
    sea = np.full((32, 48), -0.8, np.float32)
    out = elevation.paint_relief(sea, [_dab("mountain", 3600, radius=0.2)])
    assert out[16, 24] == pytest.approx(3600 / export.METRES_PER_UNIT, abs=1e-3)
    assert out[0, 0] == pytest.approx(-0.8, abs=1e-3)


def test_painted_mountain_survives_into_the_world():
    cfg = Config(width=48, height=36, seed=4, erosion_model="none")
    base = generate(cfg, verbose=False)
    strokes = [_dab("mountain", 4800, y=0.5, x=0.55, radius=0.2)]
    built = generate(cfg, verbose=False, sculpt=strokes)
    y = int(round(0.5 * (cfg.height - 1)))
    x = int(round(0.55 * (cfg.width - 1)))
    amp = 4800 / export.METRES_PER_UNIT
    core = built.height_pre[y - 2:y + 3, x - 2:x + 3]
    base_core = base.height_pre[y - 2:y + 3, x - 2:x + 3]
    assert core.mean() >= base_core.mean() + amp * 0.55
    patch = built.height[y - 2:y + 3, x - 2:x + 3]
    base_patch = base.height[y - 2:y + 3, x - 2:x + 3]
    assert patch.max() > base_patch.max() + amp * 0.4


def test_painted_mountain_wears_the_preset_texture():
    """A stroke picks up the preset's ridge grain and leaves the far land alone."""
    cfg = Config(width=48, height=36, seed=4, erosion_model="none")
    strokes = [_dab("mountain", 3600, y=0.5, x=0.5, radius=0.16)]
    shape = (cfg.height, cfg.width)
    grain = elevation.sculpt_grain(shape, strokes, cfg, origin=(2, -1))
    assert grain[0, 0] == pytest.approx(0.0, abs=1e-6)
    y = int(round(0.5 * (cfg.height - 1)))
    x = int(round(0.5 * (cfg.width - 1)))
    assert float(np.std(grain[y - 3:y + 4, x - 3:x + 4])) > 0.02
    assert abs(float(grain[y - 4:y + 5, x - 4:x + 5].mean())) < 0.08
    silent = elevation.sculpt_grain(shape, strokes, replace(cfg, texture_h=0.0))
    assert float(np.max(np.abs(silent))) == 0.0
    other = elevation.sculpt_grain(shape, strokes, replace(cfg, texture_periods=2), origin=(2, -1))
    assert not np.allclose(grain, other)
    again = elevation.sculpt_grain(shape, strokes, cfg, origin=(2, -1))
    assert grain == pytest.approx(again)
    shifted = elevation.sculpt_grain(shape, strokes, cfg, origin=(3, -1))
    assert not np.allclose(grain, shifted)
    built = generate(cfg, verbose=False, sculpt=strokes, sculpt_at=(1, 0))
    base = generate(cfg, verbose=False)
    assert built.height_pre[0, 0] == pytest.approx(float(base.height_pre[0, 0]), abs=0.02)


def test_one_stroke_is_one_mountain_across_four_tiles():
    """A crest on the shared corner is the same hill on every tile."""
    land = np.full((36, 48), 0.2, np.float32)
    metres = 4800
    amp = metres / export.METRES_PER_UNIT
    radius = 0.28

    def stroke(y, x):
        return {"kind": "mountain", "metres": metres, "radius": radius, "points": [(y, x)]}

    painted = {
        (0, 0): elevation.paint_relief(land, [stroke(1.0, 1.0)]),
        (1, 0): elevation.paint_relief(land, [stroke(1.0, 0.0)]),
        (0, 1): elevation.paint_relief(land, [stroke(0.0, 1.0)]),
        (1, 1): elevation.paint_relief(land, [stroke(0.0, 0.0)]),
    }
    crest = painted[(0, 0)][-1, -1]
    assert crest == pytest.approx(0.2 + amp, abs=1e-3)
    assert painted[(1, 0)][-1, 0] == pytest.approx(crest, abs=1e-3)
    assert painted[(0, 1)][0, -1] == pytest.approx(crest, abs=1e-3)
    assert painted[(1, 1)][0, 0] == pytest.approx(crest, abs=1e-3)
    assert painted[(0, 0)][-1, -5] == pytest.approx(painted[(1, 0)][-1, 4], abs=0.02)
    assert painted[(0, 0)][-5, -1] == pytest.approx(painted[(0, 1)][4, -1], abs=0.02)
    assert painted[(0, 0)][3, 3] < 0.2 + amp * 0.15


def test_brush_halo_is_not_its_own_mountain():
    land = np.full((36, 48), 0.2, np.float32)
    metres = 4800
    amp = metres / export.METRES_PER_UNIT
    radius = 0.34
    left = elevation.paint_relief(land, [{
        "kind": "mountain", "metres": metres, "radius": radius,
        "points": [(0.5, 0.45), (0.5, 0.7)],
    }])
    right = elevation.paint_relief(land, [{
        "kind": "mountain", "metres": metres, "radius": radius,
        "points": [(0.5, -0.55), (0.5, -0.3)],
    }])
    assert left.max() == pytest.approx(0.2 + amp, abs=1e-2)
    assert left[:, -1] == pytest.approx(right[:, 0], abs=0.03)
    assert float(right.max()) < 0.2 + amp * 0.65


def test_shared_crest_survives_the_border_solve():
    from terrain.world import meet_border

    land = np.full((36, 48), 0.25, np.float32)
    metres = 3600
    amp = metres / export.METRES_PER_UNIT
    radius = 0.3
    left_stroke = {"kind": "mountain", "metres": metres, "radius": radius,
                   "points": [(0.5, 1.0)]}
    right_stroke = {"kind": "mountain", "metres": metres, "radius": radius,
                    "points": [(0.5, 0.0)]}
    left = elevation.paint_relief(land, [left_stroke])
    target = elevation.paint_relief(land, [right_stroke])
    border = {
        "right": {
            "edge": target[:, 0].copy(),
            "outer": target[:, 1].copy(),
        }
    }
    solved = meet_border(left, border)
    crest = 0.25 + amp
    assert left[:, -1].max() == pytest.approx(crest, abs=1e-2)
    assert solved[:, -1].max() == pytest.approx(crest, rel=0.08, abs=0.04)
    assert solved[18, -6] > 0.25 + amp * 0.45


def test_shared_rim_is_the_neighbors_pixel():
    """The stitched seam is one sample, so the slope is not applied twice."""
    from terrain.view3d import stitch_tiles
    from terrain.world import meet_border

    rows, cols = 24, 32
    left = np.tile(np.linspace(0.0, 1.0, cols, dtype=np.float32), (rows, 1))
    right = np.full((rows, cols), 4.0, np.float32)
    solved = meet_border(right, {
        "left": {"edge": left[:, -1].copy(), "outer": left[:, -2].copy()},
    })
    assert solved[:, 0] == pytest.approx(left[:, -1], abs=1e-5)
    step = float(left[0, -1] - left[0, -2])
    blank = np.zeros((rows, cols, 3), np.uint8)
    sheet, _color, _mask, _layout = stitch_tiles([
        (0, 0, left, blank),
        (1, 0, solved, blank),
    ])
    seam = cols - 1
    assert sheet[:, seam] == pytest.approx(left[:, -1], abs=1e-5)
    assert sheet[0, seam] - sheet[0, seam - 1] == pytest.approx(step, abs=1e-4)


def test_seam_fade_does_not_raise_a_ridge():
    """A slope across the rim must not lift a stripe beside the edge."""
    from terrain.world import meet_border

    rows, cols = 48, 64
    right = np.zeros((rows, cols), np.float32)
    edge = np.zeros(rows, np.float32)
    outer = np.full(rows, -0.08, np.float32)
    solved = meet_border(right, {"left": {"edge": edge, "outer": outer}})
    assert solved[:, 0] == pytest.approx(0.0, abs=1e-5)
    assert float(np.max(np.abs(solved[:, 1:]))) < 0.02
    assert solved[:, -1] == pytest.approx(0.0, abs=1e-5)


def test_corner_of_two_rims_is_their_mean():
    from terrain.world import meet_border

    height = np.zeros((18, 22), np.float32)
    solved = meet_border(height, {
        "left": {"edge": np.full(18, 2.0), "outer": np.full(18, 2.0)},
        "top": {"edge": np.full(22, 4.0), "outer": np.full(22, 4.0)},
    })
    assert solved[0, 0] == pytest.approx(3.0, abs=1e-4)
    assert solved[0, 1] == pytest.approx(4.0, abs=1e-4)
    assert solved[1, 0] == pytest.approx(2.0, abs=1e-4)
    assert abs(float(solved[1, 1]) - 3.0) < 1.0


def test_landmass_on_the_edge_continues_into_the_tile():
    """A body of land on the shared edge keeps going; open water does not."""
    from terrain.world import carry_mass

    rows, cols = 48, 64
    tile = np.full((rows, cols), -0.2, np.float32)
    depth = 16
    strip = np.full((rows, depth), -0.2, np.float32)
    strip[16:32, :] = 0.35
    solved = carry_mass(tile, {"left": {
        "edge": strip[:, -1].copy(),
        "outer": strip[:, -2].copy(),
        "strip": strip,
    }})
    assert solved[:, 0] == pytest.approx(strip[:, -1], abs=1e-4)
    assert solved[24, 6] > 0.25
    assert solved[4, 6] < 0.0
    assert solved[24, -1] == pytest.approx(-0.2, abs=1e-4)


def test_landmass_keeps_sliding_along_the_edge():
    """A coast that was drifting along the edge keeps that drift inward."""
    from terrain.world import carry_mass

    rows, cols = 72, 64
    depth = 12
    strip = np.full((rows, depth), -0.3, np.float32)
    for column in range(depth):
        top = 8 + column
        strip[top:top + 14, column] = 0.45
    tile = np.full((rows, cols), -0.3, np.float32)
    solved = carry_mass(tile, {"left": {
        "edge": strip[:, -1].copy(),
        "outer": strip[:, -2].copy(),
        "strip": strip,
    }})
    # The rim land starts on row 19 and was drifting one pixel per step.
    # The new tile keeps a fraction of that drift, then returns to itself.
    assert solved[19, 0] > 0.3
    assert solved[28, 8] > 0.2
    assert solved[10, 8] < 0.0
    assert solved[22, -1] == pytest.approx(-0.3, abs=1e-4)


def test_continued_land_eases_in_without_a_shelf():
    """A wide quintic join has no flat hold and no slope break to hillshade."""
    from terrain.world import carry_mass

    rows, cols = 48, 64
    tile = np.full((rows, cols), -0.2, np.float32)
    depth = 16
    strip = np.full((rows, depth), -0.2, np.float32)
    strip[16:32, :] = 0.35
    solved = carry_mass(tile, {"left": {
        "edge": strip[:, -1].copy(),
        "outer": strip[:, -2].copy(),
        "strip": strip,
    }})
    profile = solved[24].astype(np.float64)
    # The window is long and quintic: no step at the rim, no crease, and the
    # far side of the tile is still the candidate.
    assert profile.max() <= profile[0] + 0.02
    assert profile[2] > profile[0] - 0.03
    assert profile[31] < profile[0] - 0.1
    assert profile[31] > -0.2
    assert abs(profile[-1] - (-0.2)) < 1e-4
    assert abs(float(profile[-1] - profile[-2])) < 0.04
    assert float(np.max(np.abs(np.diff(profile, n=2)))) < 0.04


def test_a_small_edge_slope_does_not_build_a_wall():
    """A one-cell slope on the seam must not be integrated into a ridge."""
    from terrain.world import carry_mass

    rows, cols, depth = 80, 96, 16
    strip = np.zeros((rows, depth), np.float32)
    for column in range(depth):
        strip[:, column] = -0.2 + 0.01 * column
    strip[30:50, :] += 0.35
    tile = np.full((rows, cols), -0.2, np.float32)
    solved = carry_mass(tile, {"left": {
        "edge": strip[:, -1].copy(),
        "outer": strip[:, -2].copy(),
        "strip": strip,
    }})
    assert solved[:, 0] == pytest.approx(strip[:, -1], abs=1e-4)
    assert float(np.max(solved)) < float(np.max(strip[:, -1])) + 0.02
    assert abs(float(solved[40, -1]) - (-0.2)) < 1e-4


def test_continued_land_keeps_the_tile_texture():
    """The new tile keeps its own grain. Only the shared edge is replaced."""
    from terrain.world import carry_mass

    rows, cols = 48, 160
    tile = np.full((rows, cols), -0.2, np.float32)
    tile[24, 4] += 0.18
    tile[24, -1] += 0.18
    depth = 16
    strip = np.full((rows, depth), -0.2, np.float32)
    strip[16:32, :] = 0.35
    solved = carry_mass(tile, {"left": {
        "edge": strip[:, -1].copy(),
        "outer": strip[:, -2].copy(),
        "strip": strip,
    }})
    near = 0.5 * (float(solved[24, 3]) + float(solved[24, 5]))
    assert float(solved[24, 4]) > near + 0.1
    assert float(solved[24, -1]) > float(solved[24, -2]) + 0.1
    assert solved[:, 0] == pytest.approx(strip[:, -1], abs=1e-4)
