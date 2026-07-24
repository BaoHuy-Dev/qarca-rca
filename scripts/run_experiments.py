#!/usr/bin/env python3
"""Run the preregistered RCAEval RE1 robustness matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from trailrca.data import discover_cases
from trailrca.evaluation import evaluate_case
from trailrca.methods import METHODS


RUNNER_VERSION = "baseline-v2-20260723"
CONFIRMATORY_METHODS = ("nsigma", "baro", "median_shift")
CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "trailrca" / "candidates.py",
    ROOT / "src" / "trailrca" / "data.py",
    ROOT / "src" / "trailrca" / "evaluation.py",
    ROOT / "src" / "trailrca" / "methods.py",
    ROOT / "src" / "trailrca" / "missingness.py",
)


def fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


CODE_HASH = fingerprint(
    [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in CODE_PATHS
    ]
)


def source_system(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("RE1-"):
            return parent.name.removeprefix("RE1-")
    raise ValueError(f"cannot infer system from {path}")


def source_descriptor(path: Path) -> dict[str, object]:
    source = path.resolve()
    inject = source.with_name("inject_time.txt")
    source_stat, inject_stat = source.stat(), inject.stat()
    return {
        "path": source.as_posix(),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "inject_time": int(inject.read_text(encoding="utf-8").strip()),
        "inject_size": inject_stat.st_size,
        "inject_mtime_ns": inject_stat.st_mtime_ns,
    }


def source_fingerprint(path: Path) -> str:
    return fingerprint(source_descriptor(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "raw" / "predictions.csv"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "results" / "cache" / "baselines"
    )
    parser.add_argument("--systems", nargs="+", default=["OB", "SS", "TT"])
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(CONFIRMATORY_METHODS)
    )
    parser.add_argument(
        "--mechanisms", nargs="+", default=["point", "burst", "stream", "incident"]
    )
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def config_fingerprint(args: argparse.Namespace) -> str:
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "fse-baro")
    }
    return fingerprint(
        {
            "version": RUNNER_VERSION,
            "code_hash": CODE_HASH,
            "python": platform.python_version(),
            "dependencies": dependencies,
            "methods": tuple(args.methods),
            "mechanisms": tuple(args.mechanisms),
            "rates": tuple(args.rates),
            "replicates": args.replicates,
            "window": args.window,
            "candidate_tie_rule": "worst_exact_tie",
        }
    )


def case_cache_path(path: Path, cache_dir: Path, config: str) -> Path:
    key = hashlib.blake2b(path.as_posix().encode(), digest_size=12).hexdigest()
    return cache_dir / config / source_system(path) / f"{key}.csv"


def expected_scenarios(
    mechanisms: tuple[str, ...], rates: tuple[float, ...], replicates: int
) -> set[tuple[str, float, int]]:
    return {("none", 0.0, 0)} | {
        (mechanism, round(float(rate), 12), replicate)
        for mechanism in mechanisms
        if mechanism != "none"
        for rate in rates
        for replicate in range(replicates)
    }


def cache_valid(
    cache: Path,
    source: Path,
    config: str,
    methods: tuple[str, ...],
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
) -> bool:
    if not cache.is_file():
        return False
    try:
        data = pd.read_csv(cache)
        scenarios = expected_scenarios(mechanisms, rates, replicates)
        exact = all(
            {
                (str(row.mechanism), round(float(row.rate), 12), int(row.replicate))
                for row in data.loc[data["method"] == method].itertuples()
            }
            == scenarios
            for method in methods
        )
        return bool(
            len(data) == len(methods) * len(scenarios)
            and not data.duplicated(
                ["case_id", "mechanism", "rate", "replicate", "method"]
            ).any()
            and set(data["method"]) == set(methods)
            and set(data["runner_version"]) == {RUNNER_VERSION}
            and set(data["code_hash"]) == {CODE_HASH}
            and set(data["config_fingerprint"]) == {config}
            and set(data["source_fingerprint"]) == {source_fingerprint(source)}
            and exact
        )
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        return False


def evaluate_and_checkpoint(
    csv_path: str,
    cache_path: str,
    config: str,
    methods: tuple[str, ...],
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
    window: int,
    force: bool,
) -> tuple[str, bool]:
    source, cache = Path(csv_path), Path(cache_path)
    if not force and cache_valid(
        cache, source, config, methods, mechanisms, rates, replicates
    ):
        return str(cache), False
    rows = evaluate_case(
        source,
        methods=methods,
        mechanisms=mechanisms,
        rates=rates,
        replicates=replicates,
        window=window,
    )
    result = pd.DataFrame(rows)
    descriptor = source_descriptor(source)
    result["runner_version"] = RUNNER_VERSION
    result["code_hash"] = CODE_HASH
    result["config_fingerprint"] = config
    result["source_fingerprint"] = source_fingerprint(source)
    result["source_size"] = descriptor["size"]
    result["source_mtime_ns"] = descriptor["mtime_ns"]
    result["window"] = window
    result = result.sort_values(
        ["mechanism", "rate", "replicate", "method"], kind="stable"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, cache)
    return str(cache), True


def main() -> int:
    args = parse_args()
    config = config_fingerprint(args)
    paths = [
        p
        for p in discover_cases(args.data_root)
        if any(part == f"RE1-{system}" for part in p.parts for system in args.systems)
    ]
    if args.limit is None:
        counts = {
            system: sum(source_system(path) == system for path in paths)
            for system in args.systems
        }
        wrong_counts = {system: count for system, count in counts.items() if count != 125}
        if wrong_counts:
            raise SystemExit(
                f"Expected exactly 125 RCAEval RE1 cases per system: {wrong_counts}"
            )
    if args.limit is not None:
        paths = paths[: args.limit]
    case_keys = [
        f"{source_system(path)}/{path.parent.parent.name}/{path.parent.name}"
        for path in paths
    ]
    if len(case_keys) != len(set(case_keys)):
        raise SystemExit("Duplicate RCAEval case identifiers discovered")
    if not paths:
        raise SystemExit("No cases matched --systems")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths: list[Path] = []
    completed = 0
    kwargs = {
        "methods": tuple(args.methods),
        "mechanisms": tuple(args.mechanisms),
        "rates": tuple(args.rates),
        "replicates": args.replicates,
        "window": args.window,
    }

    if args.workers == 1:
        for path in paths:
            prediction_path, _ = evaluate_and_checkpoint(
                str(path),
                str(case_cache_path(path, args.cache_dir, config)),
                config,
                force=args.force_cache,
                **kwargs,
            )
            prediction_paths.append(Path(prediction_path))
            completed += 1
            print(f"[{completed}/{len(paths)}] {path.parent.parent.name}/{path.parent.name}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    evaluate_and_checkpoint,
                    str(path),
                    str(case_cache_path(path, args.cache_dir, config)),
                    config,
                    force=args.force_cache,
                    **kwargs,
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                path = futures[future]
                prediction_path, _ = future.result()
                prediction_paths.append(Path(prediction_path))
                completed += 1
                print(
                    f"[{completed}/{len(paths)}] {path.parent.parent.name}/{path.parent.name}",
                    flush=True,
                )

    result = pd.concat(
        (pd.read_csv(path) for path in sorted(prediction_paths)), ignore_index=True
    ).sort_values(
        ["case_id", "mechanism", "rate", "replicate", "method"], kind="stable"
    )
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
