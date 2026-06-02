"""Generate a tiny demo point cloud for local smoke tests."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def make_sphere(samples: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, samples)
    costheta = rng.uniform(-1.0, 1.0, samples)
    theta = np.arccos(costheta)
    radius = 0.5 + 0.02 * rng.standard_normal(samples)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    intensity = np.clip(0.6 + 0.4 * costheta, 0.0, 1.0)
    return np.column_stack([x, y, z, intensity])


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/demo_input.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, make_sphere(), fmt="%.6f")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

