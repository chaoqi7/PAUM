"""Point-cloud scanner wrapper around the core simulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulators import GaussianSimulator, NoiseResult, PhysicsAwareSimulator


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for normal estimation and visibility scanning. "
            "Install it with `pip install open3d` or `pip install -r requirements.txt`."
        ) from exc
    return o3d


@dataclass(frozen=True)
class ScanOutput:
    """Noisy array and per-point uncertainty metadata."""

    data: np.ndarray
    reliability: np.ndarray
    sigma: np.ndarray
    bias_magnitude: np.ndarray


class PointCloudScanner:
    """Apply point-cloud noise to arrays or text-file data."""

    def __init__(
        self,
        *,
        mode: str = "physics",
        level: int = 2,
        gaussian_std: float | None = None,
        seed: int | None = None,
        sensor_radius: float = 2.5,
        normal_radius: float = 0.1,
        normal_max_nn: int = 30,
    ) -> None:
        self.mode = mode
        self.level = level
        self.sensor_radius = sensor_radius
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.rng = np.random.default_rng(seed)

        if mode == "physics":
            self.simulator = PhysicsAwareSimulator(level=level, seed=self.rng)
        elif mode == "gaussian":
            self.simulator = GaussianSimulator(level=level, std=gaussian_std, seed=self.rng)
        else:
            raise ValueError("mode must be either 'physics' or 'gaussian'")

    def scan_array(
        self,
        data: np.ndarray,
        *,
        view_mode: str = "surround",
        append_reliability: bool = False,
        preserve_attributes: bool = True,
        normal_columns: tuple[int, int, int] | None = None,
    ) -> ScanOutput:
        """Add noise to an ``N x K`` array.

        The first three columns are treated as XYZ. Remaining columns are copied
        to the output as generic attributes when ``preserve_attributes`` is true.
        """

        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("input data must be a 2D array with at least three XYZ columns")

        points = data[:, :3]
        attributes = data[:, 3:] if preserve_attributes and data.shape[1] > 3 else np.empty((len(data), 0))
        supplied_normals = data[:, list(normal_columns)] if normal_columns is not None else None
        if supplied_normals is not None and supplied_normals.shape[1] != 3:
            raise ValueError("normal_columns must select exactly three columns")
        sensor_positions = self._make_sensor_positions(view_mode)

        if view_mode == "all":
            if supplied_normals is None and self.mode == "gaussian":
                result = self.simulator.add_noise(points)
            else:
                normals = supplied_normals if supplied_normals is not None else self._estimate_normals(points)[1]
                result = self.simulator.add_noise(points, normals, sensor_positions[0])

            output = self._merge_columns(result.points, attributes, result.reliability, append_reliability)
            return ScanOutput(output, result.reliability, result.sigma, result.bias_magnitude)

        pcd = self._make_point_cloud(points)
        normals = supplied_normals
        if normals is None:
            pcd, normals = self._estimate_normals(points, pcd=pcd)

        chunks: list[np.ndarray] = []
        attribute_chunks: list[np.ndarray] = []
        reliability_chunks: list[np.ndarray] = []
        sigma_chunks: list[np.ndarray] = []
        bias_chunks: list[np.ndarray] = []
        all_points = np.asarray(pcd.points)

        for sensor_position in sensor_positions:
            visible_indices = self._visible_indices(pcd, sensor_position)
            if len(visible_indices) < 3:
                continue

            visible_points = all_points[visible_indices]
            visible_normals = normals[visible_indices]
            visible_attrs = attributes[visible_indices]
            result = self.simulator.add_noise(visible_points, visible_normals, sensor_position)

            chunks.append(result.points)
            attribute_chunks.append(visible_attrs)
            reliability_chunks.append(result.reliability)
            sigma_chunks.append(result.sigma)
            bias_chunks.append(result.bias_magnitude)

        if not chunks:
            reliability = np.ones(len(points), dtype=float)
            sigma = np.zeros(len(points), dtype=float)
            bias = np.zeros(len(points), dtype=float)
            output = self._merge_columns(points, attributes, reliability, append_reliability)
            return ScanOutput(output, reliability, sigma, bias)

        noisy_points = np.vstack(chunks)
        noisy_attrs = np.vstack(attribute_chunks) if attributes.shape[1] else np.empty((len(noisy_points), 0))
        reliability = np.concatenate(reliability_chunks)
        sigma = np.concatenate(sigma_chunks)
        bias = np.concatenate(bias_chunks)

        sample_idx = self._resample_indices(len(noisy_points), len(points))
        resampled_points = noisy_points[sample_idx]
        resampled_attrs = noisy_attrs[sample_idx] if noisy_attrs.shape[1] else noisy_attrs
        resampled_reliability = reliability[sample_idx]
        resampled_sigma = sigma[sample_idx]
        resampled_bias = bias[sample_idx]

        output = self._merge_columns(
            resampled_points,
            resampled_attrs,
            resampled_reliability,
            append_reliability,
        )
        return ScanOutput(output, resampled_reliability, resampled_sigma, resampled_bias)

    def _make_point_cloud(self, points: np.ndarray):
        o3d = _require_open3d()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _estimate_normals(self, points: np.ndarray, *, pcd=None):
        o3d = _require_open3d()
        if pcd is None:
            pcd = self._make_point_cloud(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
        return pcd, np.asarray(pcd.normals)

    def _make_sensor_positions(self, view_mode: str) -> list[np.ndarray]:
        radius = self.sensor_radius
        if view_mode == "surround":
            return [
                np.array([radius, 0.0, 0.0]),
                np.array([-radius, 0.0, 0.0]),
                np.array([0.0, radius, 0.0]),
                np.array([0.0, -radius, 0.0]),
                np.array([0.0, 0.0, radius]),
            ]
        if view_mode == "front" or view_mode == "all":
            return [np.array([radius, 0.0, 0.0])]
        if view_mode == "single":
            phi = self.rng.uniform(0.0, 2.0 * np.pi)
            theta = self.rng.uniform(0.0, np.pi)
            return [
                np.array(
                    [
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.cos(theta),
                    ]
                )
            ]
        raise ValueError("view_mode must be one of 'surround', 'single', 'front', or 'all'")

    def _visible_indices(self, pcd, sensor_position: np.ndarray) -> np.ndarray:
        _, indices = pcd.hidden_point_removal(sensor_position, radius=1000.0)
        return np.asarray(indices, dtype=int)

    def _resample_indices(self, source_count: int, target_count: int) -> np.ndarray:
        replace = source_count < target_count
        return self.rng.choice(source_count, size=target_count, replace=replace)

    @staticmethod
    def _merge_columns(
        points: np.ndarray,
        attributes: np.ndarray,
        reliability: np.ndarray,
        append_reliability: bool,
    ) -> np.ndarray:
        columns = [points]
        if attributes.size:
            columns.append(attributes)
        if append_reliability:
            columns.append(reliability.reshape(-1, 1))
        return np.hstack(columns)
