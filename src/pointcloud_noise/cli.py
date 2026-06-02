"""Command-line interface for point-cloud noise generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from .io import collect_point_files, output_path_for, parse_extensions, read_point_file, write_point_file
from .scanner import PointCloudScanner


@dataclass
class RunStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0


def parse_normal_columns(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    try:
        columns = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("normal columns must look like 3,4,5") from exc
    if len(columns) != 3:
        raise argparse.ArgumentTypeError("normal columns must contain exactly three column indexes")
    if any(column < 0 for column in columns):
        raise argparse.ArgumentTypeError("normal column indexes must be zero-based and non-negative")
    return columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointcloud-noise",
        description="Add physics-aware or Gaussian noise to point-cloud text files.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input point file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory.")
    parser.add_argument("--mode", choices=("physics", "gaussian"), default="physics", help="Noise model.")
    parser.add_argument("--level", type=int, choices=range(0, 5), default=2, help="Noise level from 0 to 4.")
    parser.add_argument("--gaussian-std", type=float, default=None, help="Override Gaussian std in meters.")
    parser.add_argument(
        "--view",
        choices=("surround", "single", "front", "all"),
        default="surround",
        help="Visibility strategy before adding noise.",
    )
    parser.add_argument("--append-reliability", action="store_true", help="Append one reliability column.")
    parser.add_argument("--drop-extra", action="store_true", help="Drop columns after XYZ instead of copying them.")
    parser.add_argument(
        "--normal-columns",
        type=parse_normal_columns,
        default=None,
        help="Optional zero-based normal columns, for example 3,4,5.",
    )
    parser.add_argument(
        "--delimiter",
        choices=("auto", "space", "comma", "tab"),
        default="auto",
        help="Input and output delimiter.",
    )
    parser.add_argument("--fmt", default="%.6f", help="np.savetxt format string.")
    parser.add_argument("--extensions", default=".txt,.xyz,.pts,.csv", help="Comma-separated extensions for folders.")
    parser.add_argument("--sample-step", type=int, default=1, help="Process every Nth file in a folder.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of processed files.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--sensor-radius", type=float, default=2.5, help="Virtual sensor radius.")
    parser.add_argument("--normal-radius", type=float, default=0.1, help="Open3D normal-estimation radius.")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Open3D normal-estimation max neighbors.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    files = collect_point_files(
        input_path,
        extensions=parse_extensions(args.extensions),
        sample_step=args.sample_step,
        max_files=args.max_files,
    )

    if not files:
        print(f"No point-cloud files found under {input_path}", file=sys.stderr)
        return 2

    scanner = PointCloudScanner(
        mode=args.mode,
        level=args.level,
        gaussian_std=args.gaussian_std,
        seed=args.seed,
        sensor_radius=args.sensor_radius,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
    )

    stats = RunStats(total=len(files))
    if not args.quiet:
        print(f"Mode={args.mode} | Level={args.level} | View={args.view} | Files={len(files)}")

    for file_path in files:
        target_path = output_path_for(file_path, input_path, output_path)
        try:
            data = read_point_file(file_path, delimiter=args.delimiter)
            output = scanner.scan_array(
                data,
                view_mode=args.view,
                append_reliability=args.append_reliability,
                preserve_attributes=not args.drop_extra,
                normal_columns=args.normal_columns,
            )
            write_point_file(target_path, output.data, delimiter=args.delimiter, fmt=args.fmt)
            stats.succeeded += 1
            if not args.quiet:
                print(f"[ok] {file_path} -> {target_path}")
        except Exception as exc:  # noqa: BLE001 - CLI should report and continue.
            stats.failed += 1
            print(f"[failed] {file_path}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(f"Done. succeeded={stats.succeeded}, failed={stats.failed}, total={stats.total}")

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
