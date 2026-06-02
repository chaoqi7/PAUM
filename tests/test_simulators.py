from __future__ import annotations

import unittest

import numpy as np

from pointcloud_noise.params import LiDARPhysicsParams
from pointcloud_noise.scanner import PointCloudScanner
from pointcloud_noise.simulators import GaussianSimulator, PhysicsAwareSimulator


class SimulatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            dtype=float,
        )
        self.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(self.points), 1))
        self.sensor = np.array([0.0, 0.0, 2.5])

    def test_level_presets_are_available(self) -> None:
        for level in range(5):
            params = LiDARPhysicsParams.from_level(level)
            self.assertEqual(params.level, level)

    def test_gaussian_level_zero_keeps_points(self) -> None:
        simulator = GaussianSimulator(level=0, seed=123)
        result = simulator.add_noise(self.points)
        np.testing.assert_allclose(result.points, self.points)
        np.testing.assert_allclose(result.reliability, np.ones(len(self.points)))

    def test_physics_result_shapes_and_reliability_range(self) -> None:
        simulator = PhysicsAwareSimulator(level=2, seed=123)
        result = simulator.add_noise(self.points, self.normals, self.sensor)
        self.assertEqual(result.points.shape, self.points.shape)
        self.assertEqual(result.reliability.shape, (len(self.points),))
        self.assertTrue(np.all(result.reliability >= 0.0))
        self.assertTrue(np.all(result.reliability <= 1.0))

    def test_gaussian_scanner_all_view_does_not_require_open3d(self) -> None:
        data = np.column_stack([self.points, np.arange(len(self.points))])
        scanner = PointCloudScanner(mode="gaussian", level=0, seed=123)
        result = scanner.scan_array(data, view_mode="all")
        np.testing.assert_allclose(result.data, data)


if __name__ == "__main__":
    unittest.main()
