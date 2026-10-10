"""CSV epoch log, JSON dumps that survive numpy types, and the git hash of the working tree."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path


def git_hash(root: str | os.PathLike | None = None) -> str | None:
    """Short hash of HEAD, with '-dirty' when the tree has changes; None outside a repository."""
    try:
        cwd = str(root) if root else None
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(obj):
    """Recursively convert numpy scalars and arrays, Paths and non-finite floats (to None) for json.dump."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def dump_json(path: str | os.PathLike, obj) -> Path:
    """Write JSON atomically (temp file then rename) so a killed job never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


class CsvLogger:
    """Append one row per epoch to a CSV with a fixed column order; the header is written once."""

    def __init__(self, path: str | os.PathLike, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.columns)

    def log(self, **row) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"unknown log columns {sorted(unknown)}")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([_fmt(row.get(c)) for c in self.columns])


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return v
