"""Core point-level noise simulators."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .params import LiDARPhysicsParams


ArrayLike = np.ndarray


@dataclass(frozen=True)
class NoiseResult:
    """Simulation output for one point cloud scan."""

    points: np.ndarray
    reliability: np.ndarray
    sigma: np.ndarray
    bias_magnitude: np.ndarray


def _make_rng(seed_or_rng: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def _normalize_vectors(vectors: ArrayLike) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class PhysicsAwareSimulator:
    """Physics-aware Gaussian random field noise model."""

    def __init__(
        self,
        params: LiDARPhysicsParams | None = None,
        *,
        level: int = 2,
        seed: int | np.random.Generator | None = None,
        jitter: float = 1e-10,
    ) -> None:
        self.params = params if params is not None else LiDARPhysicsParams.from_level(level)
        self.rng = _make_rng(seed)
        self.jitter = jitter

    def add_noise(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        sensor_position: ArrayLike,
    ) -> NoiseResult:
        """Add physics-aware noise to visible points."""

        points = np.asarray(points, dtype=float)
        normals = _normalize_vectors(normals)
        sensor_position = np.asarray(sensor_position, dtype=float)

        if points.size == 0:
            empty = np.zeros((0,), dtype=float)
            return NoiseResult(points=np.zeros((0, 3)), reliability=empty, sigma=empty, bias_magnitude=empty)

        ranges, thetas = self._compute_geometric_factors(points, normals, sensor_position)
        reflectivity = np.abs(np.cos(thetas)) * 0.9 + 0.1
        scalar_bias, vector_bias = self._compute_systematic_bias(ranges, thetas, reflectivity)
        sigmas = self._compute_aleatoric_sigma(ranges, thetas, reflectivity)
        reliability = self._compute_reliability(sigmas)
        grf_noise = self._sample_grf(points, sigmas, ranges)

        bias_displacement = normals * scalar_bias[:, None] + np.asarray(vector_bias, dtype=float)
        noisy_points = points + bias_displacement + grf_noise

        return NoiseResult(
            points=noisy_points,
            reliability=reliability,
            sigma=sigmas,
            bias_magnitude=scalar_bias,
        )

    def simulate_scan(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        sensor_position: ArrayLike,
    ) -> np.ndarray:
        """Compatibility wrapper returning only noisy coordinates."""

        return self.add_noise(points, normals, sensor_position).points

    def _compute_geometric_factors(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        sensor_position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        view_vectors = points - sensor_position
        ranges = np.linalg.norm(view_vectors, axis=1)
        view_dirs = view_vectors / np.maximum(ranges[:, None], 1e-12)
        dots = np.sum(view_dirs * normals, axis=1)
        thetas = np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
        return ranges, thetas

    def _compute_systematic_bias(
        self,
        ranges: np.ndarray,
        thetas: np.ndarray,
        reflectivity: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        clipped_theta = np.clip(thetas, 0.0, self.params.grazing_threshold)
        geometric_bias = self.params.kappa * ranges * np.tan(clipped_theta)

        radiometric_bias = np.zeros_like(ranges)
        saturated = reflectivity > self.params.tau_reflectivity
        radiometric_bias[saturated] = self.params.saturation_offset

        return geometric_bias + radiometric_bias, self.params.kinematic_drift

    def _compute_aleatoric_sigma(
        self,
        ranges: np.ndarray,
        thetas: np.ndarray,
        reflectivity: np.ndarray,
    ) -> np.ndarray:
        clipped_theta = np.clip(thetas, 0.0, self.params.grazing_threshold)
        sigma_geo_sq = (0.001 * ranges * np.tan(clipped_theta)) ** 2
        sigma_rad_sq = (1e-5 * (ranges**2)) / (reflectivity + 1e-4)
        sigma_env_sq = self.params.env_noise_base**2
        return np.sqrt(sigma_geo_sq + sigma_rad_sq + sigma_env_sq)

    def _compute_reliability(self, sigmas: np.ndarray) -> np.ndarray:
        reliability = np.exp(-self.params.lambda_sensitivity * (sigmas**2))
        return np.clip(reliability, 0.0, 1.0)

    def _sample_grf(self, points: np.ndarray, sigmas: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        point_count = len(points)
        if point_count == 0:
            return np.zeros((0, 3), dtype=float)

        diffs = points[:, None, :] - points[None, :, :]
        dists_sq = np.sum(diffs * diffs, axis=2)
        avg_ranges = (ranges[:, None] + ranges[None, :]) / 2.0
        length_scales = np.maximum(
            self.params.beam_divergence * avg_ranges * self.params.correlation_scale,
            1e-12,
        )
        correlation = np.exp(-dists_sq / (2.0 * length_scales**2 + 1e-12))
        covariance = np.outer(sigmas, sigmas) * correlation
        covariance += np.eye(point_count) * self.jitter

        try:
            factor = np.linalg.cholesky(covariance)
            white_noise = self.rng.standard_normal((point_count, 3))
            return factor @ white_noise
        except np.linalg.LinAlgError:
            warnings.warn(
                "GRF covariance was not positive definite; falling back to independent noise.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.rng.normal(0.0, sigmas[:, None], size=(point_count, 3))


class GaussianSimulator:
    """Simple independent Gaussian jitter baseline."""

    STD_BY_LEVEL = {
        0: 0.0,
        1: 0.002,
        2: 0.005,
        3: 0.010,
        4: 0.020,
    }

    def __init__(
        self,
        *,
        std: float | None = None,
        level: int = 2,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if std is None:
            if level not in self.STD_BY_LEVEL:
                raise ValueError(f"level must be one of 0, 1, 2, 3, 4; got {level!r}")
            std = self.STD_BY_LEVEL[level]
        self.std = float(std)
        self.rng = _make_rng(seed)

    def add_noise(
        self,
        points: ArrayLike,
        normals: ArrayLike | None = None,
        sensor_position: ArrayLike | None = None,
    ) -> NoiseResult:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            empty = np.zeros((0,), dtype=float)
            return NoiseResult(points=np.zeros((0, 3)), reliability=empty, sigma=empty, bias_magnitude=empty)

        noise = self.rng.normal(0.0, self.std, size=points.shape)
        point_count = len(points)
        return NoiseResult(
            points=points + noise,
            reliability=np.ones(point_count, dtype=float),
            sigma=np.full(point_count, self.std, dtype=float),
            bias_magnitude=np.zeros(point_count, dtype=float),
        )

    def simulate_scan(
        self,
        points: ArrayLike,
        normals: ArrayLike | None = None,
        sensor_position: ArrayLike | None = None,
    ) -> np.ndarray:
        return self.add_noise(points, normals, sensor_position).points

