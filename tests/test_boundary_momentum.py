"""A valley that meets an edge keeps its direction in the next tile."""
import unittest

import numpy as np

from terrain.boundary_momentum import (
    apply_boundary_slope, diagnose_vertical_seam, continue_boundary,
    continue_shore, momentum,
)
from terrain.world import carry_mass, check_corner, resolve_sea_level


class BoundaryMomentumTests(unittest.TestCase):
    def setUp(self):
        self.s = np.arange(129)
        # Moving valley crossing the old tile's boundary.
        self.strip = np.stack([
            -np.exp(-0.5 * ((self.s - (50 + 0.55 * j)) / 11.0) ** 2)
            for j in range(16)
        ], axis=1)

    def test_exact_boundary_four_sides(self):
        for side in ("left", "right", "top", "bottom"):
            target = np.zeros((129, 129), np.float32)
            result = continue_boundary(target, self.strip, side, reach=75)
            expected = self.strip[:, -1]
            actual = {
                "left": result[:, 0],
                "right": result[:, -1],
                "top": result[0, :],
                "bottom": result[-1, :],
            }[side]
            np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_ends_as_target(self):
        out = continue_boundary(np.zeros((129, 129)), self.strip, "left", reach=75)
        np.testing.assert_allclose(out[:, 75], 0.0, atol=1e-7)
        np.testing.assert_allclose(out[:, 76:], 0.0, atol=1e-7)

    def test_directional_continuation(self):
        out = continue_boundary(np.zeros((129, 129)), self.strip, "left", reach=75)
        start = int(np.argmin(out[:, 0]))
        next_center = int(np.argmin(out[:, 9]))
        self.assertGreater(next_center, start)


class CoastAndMotionTests(unittest.TestCase):
    def test_a_diagonal_shore_keeps_heading(self):
        span = 96
        depth = 16
        strip = np.full((span, depth), -0.3)
        for column in range(depth):
            shore = int(30 + 0.8 * column)
            strip[shore:, column] = 0.4
        target = np.full((span, span), -0.3)
        out = continue_boundary(target, strip, "left", reach=60, max_drift=1.0)
        start = int(np.argmax(out[:, 0] > 0.0))
        later = int(np.argmax(out[:, 8] > 0.0))
        self.assertGreater(later, start + 2)

    def test_shore_distance_moves_with_the_coast(self):
        span = 96
        depth = 16
        strip = np.full((span, depth), -0.3)
        for column in range(depth):
            shore = int(30 + 0.8 * column)
            strip[shore:, column] = 0.4
        target = np.full((span, span), -0.3)
        out = continue_shore(target, strip, "left", reach=60, max_drift=1.0, detail_reach=60)
        start = int(np.argmax(out[:, 0] > 0.0))
        later = int(np.argmax(out[:, 8] > 0.0))
        self.assertGreater(later, start + 2)

    def test_one_landmass_does_not_share_a_single_median_with_another(self):
        span = 80
        depth = 12
        strip = np.full((span, depth), -0.2)
        for column in range(depth):
            strip[10 + column:22 + column, column] = 0.5
            strip[50 - column:62 - column, column] = 0.5
        velocity, _residual = momentum(strip, tangent_sigma=2.0, max_drift=1.2)
        self.assertGreater(float(velocity[16]), 0.4)
        self.assertLess(float(velocity[56]), -0.4)


class SeamContinuityTests(unittest.TestCase):
    def test_slope_is_seeded_from_the_strip_and_not_left_as_a_step(self):
        span = 24
        strip = np.tile(np.linspace(0.2, 1.1, 6), (span, 1))
        right = np.zeros((span, 8))
        apply_boundary_slope(right, strip, "left")
        left = np.zeros((span, 6))
        left[:, :] = strip
        report = diagnose_vertical_seam(left, right)
        self.assertLess(report["height_max"], 1e-6)
        self.assertLess(report["slope_max"], 1e-6)

    def test_a_ridge_is_still_there_well_inside_the_detail_window(self):
        rows, cols = 48, 80
        along = np.arange(rows)
        strip = np.zeros((rows, 12))
        for column in range(12):
            center = 16 + 0.4 * column
            strip[:, column] = -np.exp(-0.5 * ((along - center) / 3.5) ** 2)
        solved = carry_mass(np.zeros((rows, cols)), {"left": {
            "edge": strip[:, -1],
            "outer": strip[:, -2],
            "strip": strip,
        }})
        # The valley steps in, then the new tile takes over. It must not
        # be dug into a wall deeper than the edge, and it must be gone inland.
        edge = float(np.min(solved[:, 0]))
        self.assertLess(float(np.min(solved[:, 6])), edge * 0.4)
        self.assertGreater(float(np.min(solved)), edge - 0.03)
        self.assertGreater(float(np.min(solved[:, 48])), -0.08)

    def test_shore_distance_agrees_across_a_continued_coast(self):
        span, depth = 48, 10
        strip = np.full((span, depth), -0.4)
        for column in range(depth):
            strip[int(12 + column):, column] = 0.5
        tile = np.full((span, 40), -0.4)
        solved = carry_mass(tile, {"left": {
            "edge": strip[:, -1],
            "outer": strip[:, -2],
            "strip": strip,
        }})
        report = diagnose_vertical_seam(strip[:, -2:], solved[:, :2])
        self.assertLess(report["height_max"], 1e-4)
        # The cliff is eased in, not copied onto the next pixel as a wall.
        outgoing = np.abs(solved[:, 1] - solved[:, 0])
        incoming = np.abs(strip[:, -1] - strip[:, -2])
        self.assertLess(float(np.max(outgoing)), float(np.max(incoming)))
        from terrain.boundary_momentum import coast_distance
        self.assertTrue(np.array_equal(
            coast_distance(strip)[:, -1] > 0, coast_distance(solved)[:, 0] > 0))
        start = int(np.argmax(solved[:, 0] > 0))
        later = int(np.argmax(solved[:, 6] > 0))
        self.assertGreater(later, start)

    def test_a_shared_corner_must_already_agree(self):
        self.assertEqual(check_corner(1.25, 1.25), 1.25)
        with self.assertRaises(ValueError):
            check_corner(1.0, 1.4)

    def test_sea_level_on_the_border_is_not_treated_as_an_edge(self):
        from terrain.world import apply_border
        height = np.zeros((8, 8), np.float32)
        edge = np.arange(8, dtype=np.float32)
        height = apply_border(height, {"sea_level": 1.5, "left": {"edge": edge}})
        np.testing.assert_array_equal(height[:, 0], edge)

    def test_a_bordered_tile_keeps_the_planet_sea_level(self):
        field = np.linspace(-2.0, 4.0, 64).reshape(8, 8)
        border = {"sea_level": 1.75, "left": {"edge": np.zeros(8), "strip": np.zeros((8, 4))}}
        self.assertEqual(resolve_sea_level(field, 0.29, border), 1.75)
        local = resolve_sea_level(field, 0.29, None)
        self.assertNotEqual(local, 1.75)


if __name__ == "__main__":
    unittest.main()
