"""Text-file input/output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_EXTENSIONS = (".txt", ".xyz", ".pts", ".csv")


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        extensions.append(item if item.startswith(".") else f".{item}")
    return tuple(extensions) if extensions else DEFAULT_EXTENSIONS


def delimiter_value(delimiter: str, path: str | Path | None = None) -> str | None:
    if delimiter == "space":
        return None
    if delimiter == "comma":
        return ","
    if delimiter == "tab":
        return "\t"
    if delimiter != "auto":
        raise ValueError("delimiter must be one of auto, space, comma, or tab")
    if path is not None and Path(path).suffix.lower() == ".csv":
        return ","
    return None


def read_point_file(path: str | Path, *, delimiter: str = "auto") -> np.ndarray:
    sep = delimiter_value(delimiter, path)
    data = np.loadtxt(path, delimiter=sep)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def write_point_file(
    path: str | Path,
    data: np.ndarray,
    *,
    delimiter: str = "auto",
    fmt: str = "%.6f",
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = delimiter_value(delimiter, output_path)
    np.savetxt(output_path, data, delimiter=" " if sep is None else sep, fmt=fmt)


def collect_point_files(
    input_path: str | Path,
    *,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    sample_step: int = 1,
    max_files: int | None = None,
) -> list[Path]:
    source = Path(input_path)
    normalized_ext = {ext.lower() for ext in extensions}
    if source.is_file():
        files = [source]
    else:
        files = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_ext
        )

    if sample_step < 1:
        raise ValueError("sample_step must be >= 1")
    files = files[::sample_step]
    if max_files is not None:
        files = files[:max_files]
    return files


def output_path_for(input_file: Path, input_root: str | Path, output_root: str | Path) -> Path:
    source = Path(input_root)
    output = Path(output_root)
    if source.is_file():
        if output.suffix:
            return output
        return output / source.name
    return output / input_file.relative_to(source)

