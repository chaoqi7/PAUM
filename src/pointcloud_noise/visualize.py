"""Optional visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_noise_comparison(
    original_points: np.ndarray,
    noisy_points: np.ndarray,
    *,
    reliability: np.ndarray | None = None,
    output: str | Path = "noise_comparison.png",
) -> Path:
    """Save a side-by-side point-cloud visualization."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to use visualization: `pip install matplotlib`.") from exc

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_count = 3 if reliability is not None else 2
    fig = plt.figure(figsize=(6 * column_count, 6))

    ax1 = fig.add_subplot(1, column_count, 1, projection="3d")
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c="#6b7280", s=2, alpha=0.6)
    ax1.set_title("Original")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, column_count, 2, projection="3d")
    ax2.scatter(noisy_points[:, 0], noisy_points[:, 1], noisy_points[:, 2], c="#dc2626", s=2, alpha=0.65)
    ax2.set_title("Noisy")
    ax2.set_axis_off()

    if reliability is not None:
        ax3 = fig.add_subplot(1, column_count, 3, projection="3d")
        points = ax3.scatter(
            noisy_points[:, 0],
            noisy_points[:, 1],
            noisy_points[:, 2],
            c=reliability,
            cmap="viridis",
            s=2,
            alpha=0.85,
        )
        ax3.set_title("Reliability")
        ax3.set_axis_off()
        fig.colorbar(points, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

