"""Colour joins wander off the cell grid instead of stepping in squares."""
import numpy as np

from terrain import render


def test_a_diagonal_shore_does_not_step_in_lockstep():
    rows, cols = 48, 64
    y = np.arange(rows)[:, None]
    x = np.arange(cols)[None, :]
    height = ((x - y) * 0.02).astype(np.float32)
    mix = render._shore_mix(height)
    cross = np.argmax(mix > 0.5, axis=1)
    deviation = cross[2:-2] - np.arange(2, rows - 2)
    assert deviation.max() - deviation.min() >= 3
    assert mix[2, -2] > 0.9
    assert mix[-2, 2] < 0.1


def test_a_straight_biome_wall_breaks_into_a_wander():
    ids = np.full((40, 56), 2, np.int8)
    ids[:, 24:] = 6
    out = render._fray_labels(ids, seed=4, reach=4.5)
    cross = np.argmax(out == 6, axis=1)
    assert cross[2:-2].max() - cross[2:-2].min() >= 3
    assert np.all(out[:, 10:16] == 2)
    assert np.all(out[:, 34:40] == 6)
    assert np.array_equal(out[:, 0], ids[:, 0])
    assert np.array_equal(out[:, -1], ids[:, -1])


def test_a_square_lake_outline_is_not_one_column():
    mask = np.zeros((40, 50), dtype=bool)
    mask[8:32, 10:40] = True
    cover = render._ragged_cover(mask, seed=3)
    assert cover[20, 25] > 0.95
    assert cover[1, 1] < 0.05
    band = cover[12:28, :]
    cross = np.argmax(band > 0.5, axis=1)
    assert cross.max() - cross.min() >= 2
