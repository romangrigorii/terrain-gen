"""Continents come from the sphere. Latitude changes climate and the metric."""
import numpy as np

from terrain import generate, grid, tectonics
from terrain.config import Config
from terrain.presets import make_config


def _band_land(land, rows):
    weights = np.broadcast_to(grid.cell_area(land.shape), land.shape)
    return float((weights[rows] * land[rows]).sum() / weights[rows].sum())


def test_an_east_step_matches_a_north_step():
    lengths = grid.step_lengths((21, 12))
    east = lengths[grid.NEIGH8.index((0, 1))]
    north = lengths[grid.NEIGH8.index((1, 0))]
    assert abs(float(east[0, 0]) - 1.0) < 1e-6
    assert abs(float(east[10, 0]) - 1.0) < 1e-6
    assert abs(float(north[5, 0]) - 1.0) < 1e-6


def test_distance_across_the_seam_is_one_cell():
    mask = np.ones((21, 16), dtype=bool)
    mask[0, 0] = False
    dist = grid.edt(mask)
    assert abs(float(dist[0, 1]) - 1.0) < 1e-5
    assert abs(float(dist[0, -1]) - 1.0) < 1e-5


def test_sea_level_counts_every_pixel_the_same():
    equator = np.zeros((40, 10))
    equator[18:23] = 1.0
    pole = np.zeros((40, 10))
    pole[:5] = 1.0
    assert grid.area_quantile(equator, 0.9) == np.quantile(equator, 0.9)
    assert grid.area_quantile(pole, 0.9) == np.quantile(pole, 0.9)


def test_crust_is_not_stripped_at_the_poles():
    cfg = Config(width=64, height=48, seed=2)
    rng = np.random.default_rng(2)
    plate = np.zeros((48, 64), dtype=int)
    plate_cont = np.array([True])
    crust = tectonics._crust(plate, plate_cont, cfg, rng)
    lat = np.abs(grid.latitude(crust.shape))[:, 0]
    polar = float(crust[lat > 0.75].mean())
    tropic = float(crust[lat < 0.25].mean())
    assert polar > 0.15
    assert polar > tropic - 0.35


def test_a_collision_thickens_crust_and_a_rift_thins_it():
    raw = np.zeros((12, 16), np.float64)
    plate = np.zeros((12, 16), dtype=int)
    plate[:, 8:] = 1
    continental = np.ones(plate.shape, dtype=bool)
    div = np.zeros((12, 16))
    div[:, 7:9] = -1.0
    built = tectonics._accrete(raw, continental, div, plate)
    assert built[:, 8].mean() > built[:, 2].mean() + 0.05
    div[:, 7:9] = 1.0
    opened = tectonics._accrete(raw, continental, div, plate)
    assert opened[:, 8].mean() < built[:, 8].mean()


def test_a_rigid_plate_has_no_internal_strain():
    plate = np.zeros((24, 32), dtype=int)
    plate[:, 16:] = 1
    same = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    opening, shear = tectonics._boundary_motion(plate, same, grid.sphere_xyz(plate.shape))
    assert np.abs(opening).max() < 1e-6
    assert shear.max() < 1e-6
    split = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    opening, _shear = tectonics._boundary_motion(plate, split, grid.sphere_xyz(plate.shape))
    assert np.abs(opening[1:-1, 4:12]).max() < 1e-6
    assert np.abs(opening[:, 15:17]).max() > 1e-3
    # Plate 1 spinning so its equator moves east, away from a still plate 0: a rift.
    rift = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    opening, _shear = tectonics._boundary_motion(plate, rift, grid.sphere_xyz(plate.shape))
    assert opening[12, 15] > 0.0


def test_an_unwarped_seed_owns_its_cell():
    cfg = Config(width=64, height=48, seed=5, n_plates=10,
                 plate_warp=0.0, plate_warp_fine=0.0)
    tect = tectonics.build(cfg, np.random.default_rng(5))
    for index, (sy, sx) in enumerate(tect.seeds):
        y = int(np.clip(round(float(sy)), 0, tect.plate.shape[0] - 1))
        x = int(np.clip(round(float(sx)), 0, tect.plate.shape[1] - 1))
        assert tect.plate[y, x] == index


def test_a_plate_can_hold_both_kinds_of_crust():
    cfg = Config(width=80, height=48, seed=4, n_plates=8)
    tect = tectonics.build(cfg, np.random.default_rng(4))
    mixed = 0
    for index in range(cfg.n_plates):
        cells = tect.plate == index
        if int(cells.sum()) < 30:
            continue
        share = float(tect.is_cont[cells].mean())
        if 0.1 < share < 0.9:
            mixed += 1
    assert mixed >= 1


def test_polar_placement_is_a_choice_not_a_penalty():
    cfg = make_config("polar", width=64, height=48, seed=3)
    tect = tectonics.build(cfg, np.random.default_rng(3))
    lat = np.abs(grid.latitude(tect.crust.shape))[:, 0]
    assert float(tect.crust[lat > 0.65].mean()) > float(tect.crust[lat < 0.25].mean())


def test_land_is_not_smeared_into_a_bar():
    """A cold-edge continent is a blob. It is not one pole copied across the map."""
    for seed in (1, 3, 7, 11, 21):
        world = generate(Config(
            width=96, height=72, seed=seed, erosion_model="none",
            thermal_iters=0, n_hotspots=0, n_plates=8), verbose=False)
        row = world.land.mean(axis=1)
        assert float(row.max()) < 0.98, (seed, float(row.max()))


def test_seeds_do_not_share_an_equatorial_belt():
    belts = 0
    for seed in (1, 2, 3, 4):
        world = generate(Config(
            width=72, height=48, seed=seed, erosion_model="none",
            thermal_iters=0, n_hotspots=0, n_plates=8), verbose=False)
        lat = np.abs(grid.latitude(world.land.shape))[:, 0]
        polar = _band_land(world.land, lat > 0.7)
        tropic = _band_land(world.land, lat < 0.25)
        if tropic > 0.55 and polar < 0.03:
            belts += 1
        edge = float(np.abs(world.height_raw[0] - world.height_raw[1]).mean())
        mid = float(np.abs(np.diff(world.height_raw[16:32], axis=0)).mean())
        assert edge < mid * 5.0 + 0.08
    assert belts == 0
