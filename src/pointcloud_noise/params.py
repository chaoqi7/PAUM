"""Noise-level presets for the LiDAR physics simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class LiDARPhysicsParams:
    """Physical parameters used by the noise simulator.

    Levels 0-4 roughly map from clean data to severe acquisition noise.
    Units are meters unless stated otherwise.
    """

    level: int
    beam_divergence: float
    kappa: float
    saturation_offset: float
    kinematic_drift: tuple[float, float, float]
    env_noise_base: float
    correlation_scale: float
    lambda_sensitivity: float = 500.0
    grazing_threshold: float = radians(75.0)
    tau_reflectivity: float = 0.8

    @classmethod
    def from_level(cls, level: int) -> "LiDARPhysicsParams":
        """Create a preset by level."""

        presets = {
            0: {
                "beam_divergence": 1e-6,
                "kappa": 0.0,
                "saturation_offset": 0.0,
                "kinematic_drift": (0.0, 0.0, 0.0),
                "env_noise_base": 0.0001,
                "correlation_scale": 1.0,
            },
            1: {
                "beam_divergence": 0.0002,
                "kappa": 0.001,
                "saturation_offset": 0.002,
                "kinematic_drift": (0.001, 0.001, 0.0005),
                "env_noise_base": 0.001,
                "correlation_scale": 10.0,
            },
            2: {
                "beam_divergence": 0.0015,
                "kappa": 0.003,
                "saturation_offset": 0.005,
                "kinematic_drift": (0.003, 0.003, 0.001),
                "env_noise_base": 0.003,
                "correlation_scale": 20.0,
            },
            3: {
                "beam_divergence": 0.0035,
                "kappa": 0.008,
                "saturation_offset": 0.015,
                "kinematic_drift": (0.008, 0.008, 0.003),
                "env_noise_base": 0.008,
                "correlation_scale": 40.0,
            },
            4: {
                "beam_divergence": 0.006,
                "kappa": 0.015,
                "saturation_offset": 0.03,
                "kinematic_drift": (0.015, 0.015, 0.005),
                "env_noise_base": 0.015,
                "correlation_scale": 60.0,
            },
        }
        if level not in presets:
            raise ValueError(f"level must be one of 0, 1, 2, 3, 4; got {level!r}")
        return cls(level=level, **presets[level])


NOISE_LEVEL_TABLE = [
    {
        "level": 0,
        "name": "Ideal",
        "description": "Nearly clean synthetic data.",
    },
    {
        "level": 1,
        "name": "High Precision",
        "description": "Industrial-grade scanning with millimeter-level noise.",
    },
    {
        "level": 2,
        "name": "Standard",
        "description": "Balanced default for common LiDAR-like degradation.",
    },
    {
        "level": 3,
        "name": "Noisy",
        "description": "Handheld or low-cost sensor noise.",
    },
    {
        "level": 4,
        "name": "Severe",
        "description": "Strong drift, larger beam divergence, and adverse conditions.",
    },
]

