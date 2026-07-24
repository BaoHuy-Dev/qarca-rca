#!/usr/bin/env python3
"""Write a machine-readable manifest for sources, data, and final results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "fse-baro",
    "matplotlib",
    "numpy",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "seaborn",
)
ARCHIVES = ("RE1-OB.zip", "RE1-SS.zip", "RE1-TT.zip")


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def relative_record(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "summary" / "run_manifest.json",
    )
    parser.add_argument("--result", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = sorted(
        {
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("scripts/*.py"),
            *ROOT.glob("tests/*.py"),
            *ROOT.glob("generated/*.tex"),
            *ROOT.glob("figures/*"),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "PROTOCOL.md",
            ROOT / "PROTOCOL_CHANGELOG.md",
            ROOT / "README.md",
            ROOT / "data" / "README.md",
            ROOT / "main.tex",
            ROOT / "references.bib",
            ROOT / "llncs.cls",
            ROOT / "splncs04.bst",
        },
        key=lambda path: path.as_posix(),
    )
    archives = []
    for name in ARCHIVES:
        path = ROOT / "data" / "downloads" / name
        if path.is_file():
            record = relative_record(path)
            record["md5"] = digest(path, "md5")
            archives.append(record)
    results = [relative_record(path) for path in args.result if path.is_file()]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_relative": Path(sys.executable).resolve().relative_to(ROOT).as_posix(),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "packages": {
            package: importlib.metadata.version(package) for package in PACKAGES
        },
        "source_files": [relative_record(path) for path in source_paths if path.is_file()],
        "data_archives": archives,
        "result_files": results,
        "notes": [
            "Raw extracted telemetry is not redistributed.",
            "Git was unavailable in the execution environment; source SHA-256 records replace a commit identifier for this run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
