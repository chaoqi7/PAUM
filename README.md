# PointCloud Noise Simulator

Physics-aware and Gaussian noise generation for point-cloud datasets.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-required-0F766E?style=flat-square)
![Package](https://img.shields.io/badge/package-src_layout-2563EB?style=flat-square)

This repository provides a compact data-preparation toolkit for corrupting clean point clouds with controllable measurement noise. It is designed for dataset augmentation, robustness evaluation, and simulation-to-real transfer studies where sensor uncertainty should be represented before downstream model training.

![Measurement uncertainty pipeline](assets/figures/measurement_uncertainty_pipeline.png)

## Features

- Physics-aware point-cloud corruption with range, incidence-angle, reflectivity, motion-drift, and environment-noise terms.
- Gaussian random field sampling for spatially correlated point perturbations.
- Independent Gaussian jitter baseline for simple control experiments.
- Five predefined noise levels from clean data to severe acquisition degradation.
- Single-file and recursive folder processing.
- Optional reliability score output for each noisy point.
- Attribute-preserving text-file I/O: only `XYZ` is modified by default.
- Python API and command-line interface.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Core dependencies:

| Package | Version | Purpose |
|---|---:|---|
| Python | `>=3.10` | Runtime |
| NumPy | `>=1.24` | Numerical computation and GRF sampling |
| Open3D | `>=0.18` | Normal estimation and hidden-point removal |
| Matplotlib | `>=3.7` | Optional visualization utilities |

## Quick Start

Create a small synthetic demo cloud:

```bash
python examples/generate_synthetic_cloud.py examples/demo_input.txt
```

Add level-3 physics-aware noise and append reliability scores:

```bash
pointcloud-noise ^
  --input examples/demo_input.txt ^
  --output outputs/demo_physics_l3.txt ^
  --mode physics ^
  --level 3 ^
  --view surround ^
  --append-reliability ^
  --seed 42
```

Process a full dataset folder:

```bash
pointcloud-noise ^
  --input data/clean ^
  --output data/noisy_physics_l2 ^
  --mode physics ^
  --level 2 ^
  --view surround
```

Generate a Gaussian baseline:

```bash
pointcloud-noise ^
  --input data/clean ^
  --output data/noisy_gaussian_l2 ^
  --mode gaussian ^
  --level 2
```

Use existing normal columns when the input format is `x y z nx ny nz ...`:

```bash
pointcloud-noise ^
  --input data/clean ^
  --output data/noisy_physics_l3 ^
  --mode physics ^
  --level 3 ^
  --view all ^
  --normal-columns 3,4,5
```

## Input and Output

Input files are plain text point clouds with at least three columns.

| Columns | Meaning | Default behavior |
|---|---|---|
| `0:3` | `x, y, z` coordinates | Corrupted by the selected noise model |
| `3:` | Intensity, RGB, normals, labels, or other attributes | Copied unchanged |
| Last column | Reliability score | Added only with `--append-reliability` |

Supported extensions are `.txt`, `.xyz`, `.pts`, and `.csv`. CSV files are read with comma delimiters; other files default to whitespace delimiters.

## Noise Levels

| Level | Name | Description | Typical use |
|---:|---|---|---|
| 0 | Ideal | Nearly clean data with minimal background noise | Sanity checks |
| 1 | High Precision | Millimeter-level acquisition noise | Industrial metrology-style data |
| 2 | Standard | Moderate LiDAR-like degradation | Default augmentation |
| 3 | Noisy | Visible drift and stronger environment noise | Low-cost or handheld sensors |
| 4 | Severe | Large beam divergence, drift, and adverse conditions | Robustness stress tests |

## Model Components

| Component | Implementation | Effect |
|---|---|---|
| Geometric bias | `kappa * range * tan(theta)` | Larger drift for distant or grazing-angle points |
| Radiometric bias | Saturation offset above reflectivity threshold | Bias from highly reflective surfaces |
| Kinematic drift | Fixed 3D drift vector | Motion-induced systematic offset |
| Aleatoric sigma | Geometry, radiometry, and environment terms | Point-wise heteroscedastic noise scale |
| GRF covariance | Range-adaptive spatial kernel | Correlated perturbations for neighboring points |
| Reliability | `exp(-lambda * sigma^2)` | Lower score for more uncertain points |

## Command-Line Options

| Option | Default | Description |
|---|---:|---|
| `--mode` | `physics` | `physics` or `gaussian` |
| `--level` | `2` | Noise intensity from `0` to `4` |
| `--view` | `surround` | `surround`, `single`, `front`, or `all` |
| `--append-reliability` | off | Append one reliability column |
| `--drop-extra` | off | Output noisy `XYZ` only |
| `--normal-columns` | none | Existing normal columns, for example `3,4,5` |
| `--sample-step` | `1` | Process every Nth file in folder mode |
| `--max-files` | none | Limit the number of processed files |
| `--seed` | none | Random seed for reproducibility |

## Python API

```python
import numpy as np
from pointcloud_noise import PointCloudScanner

data = np.loadtxt("examples/demo_input.txt")

scanner = PointCloudScanner(mode="physics", level=3, seed=42)
result = scanner.scan_array(
    data,
    view_mode="surround",
    append_reliability=True,
)

np.savetxt("outputs/demo_physics_l3.txt", result.data, fmt="%.6f")
```

## Repository Layout

| Path | Description |
|---|---|
| `src/pointcloud_noise/params.py` | LiDAR-style noise-level presets |
| `src/pointcloud_noise/simulators.py` | Physics-aware GRF simulator and Gaussian baseline |
| `src/pointcloud_noise/scanner.py` | Normal estimation, visibility scanning, resampling, and column merging |
| `src/pointcloud_noise/io.py` | Text point-cloud loading, saving, and folder traversal |
| `src/pointcloud_noise/cli.py` | Command-line interface |
| `src/pointcloud_noise/visualize.py` | Optional visualization helper |
| `examples/` | Demo point-cloud generation script |
| `tests/` | Lightweight simulator tests |
| `assets/figures/` | Manuscript figure used by the README |

## Tests

```bash
python -m unittest discover -s tests
```

The simulator tests do not require Open3D. Open3D is required when running scan modes that estimate normals or perform hidden-point removal.

