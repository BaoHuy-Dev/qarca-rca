#!/usr/bin/env python3
"""Run leakage-safe leave-one-system-out experiments for QARCA.

Raw windows are loaded only inside workers.  Label-free service features are
cached per incident for the frozen training augmentations, then each held-out
system is scored by models trained exclusively on the other systems.  Test
predictions are checkpointed per incident, so an interrupted full run resumes
without changing masks or refitting on test outcomes.
"""

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
from time import perf_counter
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trailrca.data import discover_cases, load_case
from trailrca.evaluation import (
    Prediction,
    evaluate_service_scores,
    observability_summary,
)
from trailrca.missingness import case_seed, inject_missingness
from trailrca.quality_ranker import (
    FEATURE_COLUMNS,
    NO_QUALITY_FEATURE_COLUMNS,
    QUALITY_ONLY_FEATURE_COLUMNS,
    STRUCTURAL_FEATURE_COLUMNS,
    QualityAwareServiceRanker,
    extract_service_features,
)


RUNNER_VERSION = "qarca-loso-v2-20260722"
MODEL_NAMES = (
    "qarca",
    "qarca_original_train",
    "qarca_no_quality",
    "qarca_no_quality_original",
    "qarca_structural_only",
    "qarca_quality_only",
    "qarca_no_incident_train",
)
DEFAULT_MECHANISMS = ("point", "burst", "stream", "incident")
DEFAULT_RATES = (0.1, 0.3, 0.5)
SCHEMA_HASH = hashlib.sha256("\x1f".join(FEATURE_COLUMNS).encode()).hexdigest()
CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "trailrca" / "candidates.py",
    ROOT / "src" / "trailrca" / "data.py",
    ROOT / "src" / "trailrca" / "evaluation.py",
    ROOT / "src" / "trailrca" / "missingness.py",
    ROOT / "src" / "trailrca" / "quality_ranker.py",
)


def _fingerprint(payload: object) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


CODE_HASH = _fingerprint(
    [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in CODE_PATHS
    ]
)
DEPENDENCY_VERSIONS = {
    distribution: importlib.metadata.version(distribution)
    for distribution in ("numpy", "pandas", "scikit-learn", "scipy")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "raw" / "qarca.csv"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "results" / "cache" / "qarca"
    )
    parser.add_argument("--systems", nargs="+", default=["OB", "SS", "TT"])
    parser.add_argument("--mechanisms", nargs="+", default=list(DEFAULT_MECHANISMS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--limit-per-system",
        type=int,
        default=None,
        help="Deterministic smoke-test limit applied separately to every system.",
    )
    parser.add_argument("--force-training-cache", action="store_true")
    parser.add_argument("--force-test-cache", action="store_true")
    return parser.parse_args()


def _source_system(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("RE1-"):
            return parent.name.removeprefix("RE1-")
    raise ValueError(f"cannot infer system from {path}")


def _key(path: Path) -> str:
    return hashlib.blake2b(path.as_posix().encode(), digest_size=12).hexdigest()


def _source_descriptor(path: Path) -> dict[str, object]:
    source = path.resolve()
    inject_path = source.with_name("inject_time.txt")
    source_stat = source.stat()
    inject_stat = inject_path.stat()
    return {
        "path": source.as_posix(),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "inject_time": int(inject_path.read_text(encoding="utf-8").strip()),
        "inject_size": inject_stat.st_size,
        "inject_mtime_ns": inject_stat.st_mtime_ns,
    }


def _source_fingerprint(path: Path) -> str:
    return _fingerprint(_source_descriptor(path))


def _training_config_fingerprint(
    mechanisms: tuple[str, ...], rates: tuple[float, ...], window: int
) -> str:
    return _fingerprint(
        {
            "version": RUNNER_VERSION,
            "code_hash": CODE_HASH,
            "schema_hash": SCHEMA_HASH,
            "python": platform.python_version(),
            "dependencies": DEPENDENCY_VERSIONS,
            "window": window,
            "mechanisms": mechanisms,
            "rates": rates,
            "training_replicates": 1,
            "training_namespace": "train:",
            "candidate_tie_rule": "worst_exact_tie",
        }
    )


def _fold_fingerprint(
    training_config: str,
    held_out: str,
    training_paths: list[Path],
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
) -> str:
    return _fingerprint(
        {
            "training_config": training_config,
            "held_out": held_out,
            "training_sources": sorted(
                _source_fingerprint(path) for path in training_paths
            ),
            "test_mechanisms": mechanisms,
            "test_rates": rates,
            "test_replicates": replicates,
            "models": MODEL_NAMES,
        }
    )


def _training_cache_path(
    path: Path, cache_dir: Path, config_fingerprint: str
) -> Path:
    return (
        cache_dir
        / "training"
        / config_fingerprint
        / _source_system(path)
        / f"{_key(path)}.npz"
    )


def _test_cache_path(path: Path, cache_dir: Path, fold_fingerprint: str) -> Path:
    return (
        cache_dir
        / "test"
        / fold_fingerprint
        / _source_system(path)
        / f"{_key(path)}.csv"
    )


def _training_cache_valid(
    cache_path: Path,
    source: Path,
    window: int,
    config_fingerprint: str,
    expected_variants: int,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            return bool(
                str(cached["version"]) == RUNNER_VERSION
                and str(cached["config_fingerprint"]) == config_fingerprint
                and str(cached["schema_hash"]) == SCHEMA_HASH
                and str(cached["code_hash"]) == CODE_HASH
                and int(cached["window"]) == window
                and str(cached["source_fingerprint"])
                == _source_fingerprint(source)
                and cached["X_augmented"].shape[0] == expected_variants
                and cached["X_augmented"].shape[2] == len(FEATURE_COLUMNS)
            )
    except (OSError, KeyError, ValueError, IndexError):
        return False


def _training_scenarios(
    mechanisms: tuple[str, ...], rates: tuple[float, ...]
) -> list[tuple[str, float]]:
    return [("none", 0.0)] + [
        (mechanism, rate) for mechanism in mechanisms for rate in rates
    ]


def build_training_cache(
    csv_path: str,
    cache_path: str,
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    window: int,
    config_fingerprint: str,
    force: bool,
) -> tuple[str, bool]:
    """Worker: cache clean and frozen augmented features for one incident."""

    source = Path(csv_path)
    destination = Path(cache_path)
    expected_variants = len(_training_scenarios(mechanisms, rates))
    if not force and _training_cache_valid(
        destination,
        source,
        window,
        config_fingerprint,
        expected_variants,
    ):
        return str(destination), False

    case = load_case(source, window=window)
    feature_blocks: list[np.ndarray] = []
    services: tuple[str, ...] | None = None
    for mechanism, rate in _training_scenarios(mechanisms, rates):
        if mechanism == "none":
            observed = case.frame
        else:
            seed = case_seed(f"train:{case.case_id}", mechanism, rate, 0)
            observed, _ = inject_missingness(
                case.frame, mechanism, rate, seed, n_pre=case.n_pre
            )
        features = extract_service_features(observed, case.n_pre)
        current_services = tuple(str(value) for value in features.index)
        if services is None:
            services = current_services
        elif services != current_services:
            raise RuntimeError("candidate services changed across training masks")
        feature_blocks.append(
            features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        )

    assert services is not None
    labels = np.asarray([service == case.service for service in services], dtype=np.int8)
    if int(labels.sum()) != 1:
        raise ValueError(f"root {case.service!r} is not a unique candidate in {case.case_id}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    stat = source.stat()
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(RUNNER_VERSION),
            config_fingerprint=np.asarray(config_fingerprint),
            schema_hash=np.asarray(SCHEMA_HASH),
            code_hash=np.asarray(CODE_HASH),
            window=np.asarray(window, dtype=np.int32),
            source_fingerprint=np.asarray(_source_fingerprint(source)),
            source_size=np.asarray(stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
            case_id=np.asarray(case.case_id),
            system=np.asarray(case.system),
            X_original=feature_blocks[0],
            X_augmented=np.stack(feature_blocks),
            y=labels,
        )
    os.replace(temporary, destination)
    return str(destination), True


def _incident_weights(labels: np.ndarray) -> np.ndarray:
    candidate_count = labels.size
    if candidate_count < 2 or int(labels.sum()) != 1:
        raise ValueError("each incident needs one root and at least one non-root")
    weights = np.full(candidate_count, 0.5 / (candidate_count - 1), dtype=np.float64)
    weights[labels == 1] = 0.5
    return weights


def load_training_matrix(
    cache_paths: list[Path],
    training_systems: set[str],
    *,
    augmented: bool,
    variant_indices: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """Load only feature arrays belonging to the current training systems."""

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    incidents = 0
    for cache_path in sorted(cache_paths):
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["system"]) not in training_systems:
                continue
            labels = cached["y"].astype(np.int8, copy=False)
            base_weights = _incident_weights(labels)
            if augmented:
                blocks = cached["X_augmented"].astype(np.float32, copy=False)
                if variant_indices is not None:
                    blocks = blocks[list(variant_indices)]
                variants = blocks.shape[0]
                feature_parts.append(blocks.reshape(-1, blocks.shape[-1]))
                label_parts.append(np.tile(labels, variants))
                # Every original incident has unit total weight irrespective of
                # candidate count and number of augmented variants.
                weight_parts.append(np.tile(base_weights / variants, variants))
            else:
                feature_parts.append(cached["X_original"].astype(np.float32, copy=False))
                label_parts.append(labels)
                weight_parts.append(base_weights)
            incidents += 1
    if not feature_parts:
        raise ValueError("no cached incidents matched the training systems")
    matrix = pd.DataFrame(
        np.concatenate(feature_parts, axis=0), columns=list(FEATURE_COLUMNS)
    )
    return (
        matrix,
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
        incidents,
    )


def fit_fold_models(
    cache_paths: list[Path],
    training_systems: set[str],
    training_scenarios: list[tuple[str, float]],
) -> dict[str, QualityAwareServiceRanker]:
    def fitted(
        feature_columns: tuple[str, ...],
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        incidents: int,
        variants: int,
    ) -> QualityAwareServiceRanker:
        started = perf_counter()
        model = QualityAwareServiceRanker(feature_columns=feature_columns).fit_precomputed(
            X, y, weights, n_training_incidents=incidents
        )
        model.training_elapsed_ms_ = (perf_counter() - started) * 1000.0
        model.training_variants_per_incident_ = variants
        return model

    augmented_X, augmented_y, augmented_w, augmented_n = load_training_matrix(
        cache_paths, training_systems, augmented=True
    )
    augmented_variants = len(training_scenarios)
    full = fitted(
        FEATURE_COLUMNS, augmented_X, augmented_y, augmented_w,
        augmented_n, augmented_variants,
    )
    no_quality = fitted(
        NO_QUALITY_FEATURE_COLUMNS, augmented_X, augmented_y, augmented_w,
        augmented_n, augmented_variants,
    )
    structural_only = fitted(
        STRUCTURAL_FEATURE_COLUMNS, augmented_X, augmented_y, augmented_w,
        augmented_n, augmented_variants,
    )
    quality_only = fitted(
        QUALITY_ONLY_FEATURE_COLUMNS, augmented_X, augmented_y, augmented_w,
        augmented_n, augmented_variants,
    )
    del augmented_X, augmented_y, augmented_w

    original_X, original_y, original_w, original_n = load_training_matrix(
        cache_paths, training_systems, augmented=False
    )
    original = fitted(
        FEATURE_COLUMNS, original_X, original_y, original_w, original_n, 1,
    )
    no_quality_original = fitted(
        NO_QUALITY_FEATURE_COLUMNS,
        original_X,
        original_y,
        original_w,
        original_n,
        1,
    )
    del original_X, original_y, original_w

    non_incident_indices = tuple(
        index
        for index, (mechanism, _) in enumerate(training_scenarios)
        if mechanism != "incident"
    )
    no_incident_X, no_incident_y, no_incident_w, no_incident_n = load_training_matrix(
        cache_paths,
        training_systems,
        augmented=True,
        variant_indices=non_incident_indices,
    )
    no_incident = fitted(
        FEATURE_COLUMNS,
        no_incident_X,
        no_incident_y,
        no_incident_w,
        no_incident_n,
        len(non_incident_indices),
    )
    return {
        "qarca": full,
        "qarca_original_train": original,
        "qarca_no_quality": no_quality,
        "qarca_no_quality_original": no_quality_original,
        "qarca_structural_only": structural_only,
        "qarca_quality_only": quality_only,
        "qarca_no_incident_train": no_incident,
    }


def _test_scenarios(
    mechanisms: tuple[str, ...], rates: tuple[float, ...], replicates: int
) -> list[tuple[str, float, int]]:
    return [("none", 0.0, 0)] + [
        (mechanism, rate, replicate)
        for mechanism in mechanisms
        for rate in rates
        for replicate in range(replicates)
    ]


def _test_cache_valid(
    path: Path,
    expected_rows: int,
    source: Path,
    window: int,
    fold_fingerprint: str,
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        cached = pd.read_csv(path)
        key_columns = ["mechanism", "rate", "replicate", "method"]
        expected_scenarios = {
            (mechanism, round(float(rate), 12), int(replicate))
            for mechanism, rate, replicate in _test_scenarios(
                mechanisms, rates, replicates
            )
        }
        exact_scenarios = all(
            {
                (str(row.mechanism), round(float(row.rate), 12), int(row.replicate))
                for row in cached.loc[cached["method"] == method].itertuples()
            }
            == expected_scenarios
            for method in MODEL_NAMES
        )
        return bool(
            len(cached) == expected_rows
            and not cached.duplicated(key_columns).any()
            and set(cached["method"]) == set(MODEL_NAMES)
            and set(cached["runner_version"]) == {RUNNER_VERSION}
            and set(cached["code_hash"]) == {CODE_HASH}
            and set(cached["fold_fingerprint"]) == {fold_fingerprint}
            and set(cached["source_fingerprint"])
            == {_source_fingerprint(source)}
            and set(cached["source_size"]) == {source.stat().st_size}
            and set(cached["source_mtime_ns"]) == {source.stat().st_mtime_ns}
            and set(cached["window"]) == {window}
            and exact_scenarios
        )
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        return False


def evaluate_test_case(
    csv_path: str,
    output_path: str,
    models: Mapping[str, QualityAwareServiceRanker],
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
    window: int,
    fold_fingerprint: str,
    force: bool,
) -> tuple[str, bool]:
    """Worker: score one held-out incident and atomically checkpoint rows."""

    source = Path(csv_path)
    destination = Path(output_path)
    scenarios = _test_scenarios(mechanisms, rates, replicates)
    expected_rows = len(scenarios) * len(models)
    if not force and _test_cache_valid(
        destination,
        expected_rows,
        source,
        window,
        fold_fingerprint,
        mechanisms,
        rates,
        replicates,
    ):
        return str(destination), False

    case = load_case(source, window=window)
    rows: list[dict[str, object]] = []
    source_stat = source.stat()
    source_fingerprint = _source_fingerprint(source)
    for mechanism, rate, replicate in scenarios:
        if mechanism == "none":
            observed, realized = case.frame, 0.0
        else:
            seed = case_seed(case.case_id, mechanism, rate, replicate)
            observed, realized = inject_missingness(
                case.frame, mechanism, rate, seed, n_pre=case.n_pre
            )

        feature_started = perf_counter()
        features = extract_service_features(observed, case.n_pre)
        feature_ms = (perf_counter() - feature_started) * 1000.0
        observability = observability_summary(
            observed, case.n_pre, case.service
        )

        for method, model in models.items():
            prediction_started = perf_counter()
            scores = model.predict_feature_scores(features)
            service_ranks = sorted(
                scores.index, key=lambda service: (-scores[service], service)
            )
            prediction_ms = (perf_counter() - prediction_started) * 1000.0
            rank, rr, hit1, hit3, hit5 = evaluate_service_scores(
                case.service, scores.to_dict()
            )
            avg5 = (6 - rank) / 5.0 if rr > 0.0 and rank <= 5 else 0.0
            normalized_rank = (
                (rank - 1) / (len(service_ranks) - 1)
                if rr > 0.0 and len(service_ranks) > 1
                else (0.0 if rr > 0.0 else 1.0)
            )
            ordered_scores = scores.loc[service_ranks].to_numpy(dtype=float)
            margin = (
                float(ordered_scores[0] - ordered_scores[1])
                if ordered_scores.size > 1
                else float(ordered_scores[0])
            )
            base = Prediction(
                case_id=case.case_id,
                system=case.system,
                root_service=case.service,
                fault=case.fault,
                repetition=case.repetition,
                n_pre=case.n_pre,
                n_post=len(case.frame) - case.n_pre,
                mechanism=mechanism,
                rate=rate,
                replicate=replicate,
                realized_rate=realized,
                method=method,
                rank=rank,
                reciprocal_rank=rr,
                avg_at_5=avg5,
                normalized_rank=normalized_rank,
                hit_at_1=hit1,
                hit_at_3=hit3,
                hit_at_5=hit5,
                candidate_count=len(service_ranks),
                elapsed_ms=feature_ms + prediction_ms,
                top_5="|".join(service_ranks[:5]),
                top_evidence_metric="",
                **observability,
            ).row()
            base.update(
                {
                    "top_score": float(ordered_scores[0]),
                    "score_margin": margin,
                    "root_score": float(scores.get(case.service, np.nan)),
                    "runner_version": RUNNER_VERSION,
                    "code_hash": CODE_HASH,
                    "fold_fingerprint": fold_fingerprint,
                    "source_fingerprint": source_fingerprint,
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "window": window,
                    "training_incidents": model.n_training_incidents_,
                    "training_variants_per_incident": (
                        model.training_variants_per_incident_
                    ),
                    "training_elapsed_ms": model.training_elapsed_ms_,
                }
            )
            rows.append(base)

    result = pd.DataFrame(rows).sort_values(
        ["mechanism", "rate", "replicate", "method"], kind="stable"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return str(destination), True


def _select_paths(args: argparse.Namespace) -> list[Path]:
    by_system: dict[str, list[Path]] = {system: [] for system in args.systems}
    for path in discover_cases(args.data_root):
        system = _source_system(path)
        if system in by_system:
            by_system[system].append(path)
    missing = [system for system, paths in by_system.items() if not paths]
    if missing:
        raise SystemExit(f"No cases found for systems: {missing}")
    if args.limit_per_system is None:
        wrong_counts = {
            system: len(system_paths)
            for system, system_paths in by_system.items()
            if len(system_paths) != 125
        }
        if wrong_counts:
            raise SystemExit(
                f"Expected exactly 125 RCAEval RE1 cases per system: {wrong_counts}"
            )
    selected: list[Path] = []
    for system in args.systems:
        paths = by_system[system]
        if args.limit_per_system is not None:
            paths = paths[: args.limit_per_system]
        selected.extend(paths)
    case_keys = [
        f"{_source_system(path)}/{path.parent.parent.name}/{path.parent.name}"
        for path in selected
    ]
    if len(case_keys) != len(set(case_keys)):
        raise SystemExit("Duplicate RCAEval case identifiers discovered")
    return selected


def main() -> int:
    args = parse_args()
    if len(set(args.systems)) < 2:
        raise SystemExit("LOSO evaluation requires at least two systems")
    if args.replicates < 1 or args.workers < 1:
        raise SystemExit("--replicates and --workers must be positive")
    paths = _select_paths(args)
    mechanisms = tuple(args.mechanisms)
    rates = tuple(args.rates)
    training_config = _training_config_fingerprint(
        mechanisms, rates, args.window
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing frozen training features for {len(paths)} incidents", flush=True)
    cache_paths: list[Path] = []
    rebuilt = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_training_cache,
                str(path),
                str(
                    _training_cache_path(
                        path, args.cache_dir, training_config
                    )
                ),
                mechanisms,
                rates,
                args.window,
                training_config,
                args.force_training_cache,
            ): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            cache_path, was_rebuilt = future.result()
            cache_paths.append(Path(cache_path))
            rebuilt += int(was_rebuilt)
            if completed % 10 == 0 or completed == len(futures):
                print(f"  training cache {completed}/{len(futures)}", flush=True)
    print(f"Training cache ready ({rebuilt} rebuilt)", flush=True)
    cache_paths.sort()

    prediction_paths: list[Path] = []
    for held_out in args.systems:
        training_systems = set(args.systems) - {held_out}
        training_paths = [
            path for path in paths if _source_system(path) in training_systems
        ]
        fold_fingerprint = _fold_fingerprint(
            training_config,
            held_out,
            training_paths,
            mechanisms,
            rates,
            args.replicates,
        )
        print(
            f"Fitting fold held_out={held_out} from {sorted(training_systems)}",
            flush=True,
        )
        models = fit_fold_models(
            cache_paths,
            training_systems,
            _training_scenarios(mechanisms, rates),
        )
        test_paths = [path for path in paths if _source_system(path) == held_out]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    evaluate_test_case,
                    str(path),
                    str(
                        _test_cache_path(
                            path, args.cache_dir, fold_fingerprint
                        )
                    ),
                    models,
                    mechanisms,
                    rates,
                    args.replicates,
                    args.window,
                    fold_fingerprint,
                    args.force_test_cache,
                ): path
                for path in test_paths
            }
            rebuilt_test = 0
            for completed, future in enumerate(as_completed(futures), start=1):
                prediction_path, was_rebuilt = future.result()
                prediction_paths.append(Path(prediction_path))
                rebuilt_test += int(was_rebuilt)
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"  test {held_out} {completed}/{len(futures)}",
                        flush=True,
                    )
        print(f"Fold {held_out} complete ({rebuilt_test} cases rebuilt)", flush=True)

    combined = pd.concat(
        (pd.read_csv(path) for path in sorted(prediction_paths)), ignore_index=True
    ).sort_values(
        ["case_id", "mechanism", "rate", "replicate", "method"], kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} predictions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
