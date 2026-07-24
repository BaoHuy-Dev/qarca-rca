#!/usr/bin/env python3
"""Validate the confirmatory matrix and create data-derived paper assets.

The default path is deliberately fail closed: no output is created until the
entire pre-specified matrix, its provenance, and cross-method mask invariants
have been validated.  ``--allow-incomplete`` is only for diagnostics and is
confined to directories whose path contains an ``exploratory`` component.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASELINES = ("nsigma", "baro", "median_shift")
PROPOSED = "qarca"
QARCA_METHODS = (
    "qarca",
    "qarca_original_train",
    "qarca_no_quality",
    "qarca_no_quality_original",
    "qarca_structural_only",
    "qarca_quality_only",
    "qarca_no_incident_train",
)
ALL_METHODS = BASELINES + QARCA_METHODS
MECHANISMS = ("point", "burst", "stream", "incident")
RATES = (0.1, 0.3, 0.5)
SYSTEMS = ("OB", "SS", "TT")
METRICS = (
    "reciprocal_rank",
    "avg_at_5",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
)
OBSERVABILITY_COLUMNS = (
    "root_observed_fraction",
    "root_streams_available",
    "root_streams_total",
    "root_pre_observed_fraction",
    "root_post_observed_fraction",
    "root_streams_pre_available",
    "root_streams_post_available",
    "root_streams_scorable",
    "observed_fraction",
)
FRACTION_COLUMNS = (
    "root_observed_fraction",
    "root_pre_observed_fraction",
    "root_post_observed_fraction",
    "observed_fraction",
)
MASK_INVARIANT_COLUMNS = ("candidate_count", "realized_rate") + OBSERVABILITY_COLUMNS
INCIDENT_METADATA_COLUMNS = (
    "root_service",
    "fault",
    "repetition",
    "n_pre",
    "n_post",
    "source_fingerprint",
    "source_size",
    "source_mtime_ns",
    "window",
)
REQUIRED_COLUMNS = {
    "case_id",
    "system",
    "root_service",
    "fault",
    "repetition",
    "n_pre",
    "n_post",
    "mechanism",
    "rate",
    "replicate",
    "realized_rate",
    "method",
    "rank",
    "reciprocal_rank",
    "avg_at_5",
    "normalized_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "candidate_count",
    "elapsed_ms",
    "top_5",
    "top_evidence_metric",
    *OBSERVABILITY_COLUMNS,
    "runner_version",
    "code_hash",
    "config_fingerprint",
    "fold_fingerprint",
    "source_fingerprint",
    "source_size",
    "source_mtime_ns",
    "window",
    "top_score",
    "score_margin",
    "root_score",
    "training_incidents",
    "training_variants_per_incident",
    "training_elapsed_ms",
}
NUMERIC_COLUMNS = {
    "repetition",
    "n_pre",
    "n_post",
    "rate",
    "replicate",
    "realized_rate",
    "rank",
    "reciprocal_rank",
    "avg_at_5",
    "normalized_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "candidate_count",
    "elapsed_ms",
    *OBSERVABILITY_COLUMNS,
    "source_size",
    "source_mtime_ns",
    "window",
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
METHOD_LABELS = {
    "nsigma": r"$n$-sigma",
    "baro": "BARO",
    "median_shift": "Median shift",
    "qarca": "QARCA",
    "qarca_original_train": "QARCA (original-only train)",
    "qarca_no_quality": "QARCA (no quality)",
    "qarca_no_quality_original": "No quality, original-only train",
    "qarca_structural_only": "Structural only",
    "qarca_quality_only": "Quality only",
    "qarca_no_incident_train": "No incident-loss train",
}


class ValidationError(ValueError):
    """Raised before any confirmatory output is written."""


@dataclass(frozen=True)
class StudyDesign:
    systems: tuple[str, ...] = SYSTEMS
    cases_per_system: int = 125
    mechanisms: tuple[str, ...] = MECHANISMS
    rates: tuple[float, ...] = RATES
    replicates: int = 10
    baselines: tuple[str, ...] = BASELINES
    qarca_methods: tuple[str, ...] = QARCA_METHODS

    @property
    def methods(self) -> tuple[str, ...]:
        return self.baselines + self.qarca_methods

    @property
    def scenarios(self) -> tuple[tuple[str, float, int], ...]:
        return (("none", 0.0, 0),) + tuple(
            (mechanism, rate, replicate)
            for mechanism in self.mechanisms
            for rate in self.rates
            for replicate in range(self.replicates)
        )

    @property
    def incidents(self) -> int:
        return len(self.systems) * self.cases_per_system

    @property
    def rows_per_method(self) -> int:
        return self.incidents * len(self.scenarios)


@dataclass(frozen=True)
class Destinations:
    summary: Path
    figures: Path
    generated: Path


@dataclass
class AnalysisArtifacts:
    frames: dict[str, pd.DataFrame]
    generated_text: dict[str, str]
    alt_text: str
    figure_data: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze a partial run, but only under separate exploratory paths.",
    )
    return parser.parse_args()


def resolve_destinations(args: argparse.Namespace) -> Destinations:
    if args.allow_incomplete:
        destinations = Destinations(
            args.output or ROOT / "results" / "exploratory" / "summary",
            args.figures or ROOT / "figures" / "exploratory",
            args.generated or ROOT / "generated" / "exploratory",
        )
        for path in destinations.__dict__.values():
            if "exploratory" not in {part.lower() for part in path.resolve().parts}:
                raise ValidationError(
                    "--allow-incomplete outputs must have an 'exploratory' path component"
                )
        return destinations
    return Destinations(
        args.output or ROOT / "results" / "summary",
        args.figures or ROOT / "figures",
        args.generated or ROOT / "generated",
    )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load current runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_runner_provenance() -> dict[str, tuple[str, str]]:
    """Read the version and code hash computed by the current runner sources."""

    baseline = _load_module(ROOT / "scripts" / "run_experiments.py", "_current_baseline")
    qarca = _load_module(
        ROOT / "scripts" / "run_qarca_experiments.py", "_current_qarca"
    )
    return {
        "baseline": (str(baseline.RUNNER_VERSION), str(baseline.CODE_HASH)),
        "qarca": (str(qarca.RUNNER_VERSION), str(qarca.CODE_HASH)),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _nonempty_strings(series: pd.Series, name: str) -> None:
    values = series.astype("string")
    _require(not values.isna().any(), f"{name} contains missing values")
    _require(values.str.strip().ne("").all(), f"{name} contains empty values")


def _validate_hashes(series: pd.Series, name: str) -> None:
    _nonempty_strings(series, name)
    invalid = ~series.astype(str).str.match(HASH_PATTERN)
    _require(not invalid.any(), f"{name} must contain lowercase SHA-256 values")


def _one_value(frame: pd.DataFrame, column: str, label: str) -> str:
    values = frame[column].dropna().astype(str).unique()
    _require(len(values) == 1, f"{label} must have exactly one {column}; found {len(values)}")
    return str(values[0])


def _integer_like(series: pd.Series, name: str) -> None:
    values = series.to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"{name} contains non-finite values")
    _require(np.allclose(values, np.rint(values), atol=1e-10), f"{name} must be integral")


def _validate_provenance(
    data: pd.DataFrame,
    design: StudyDesign,
    expected: Mapping[str, tuple[str, str]] | None,
    complete: bool,
) -> None:
    baseline = data.loc[data["method"].isin(design.baselines)]
    qarca = data.loc[data["method"].isin(design.qarca_methods)]
    _require(not baseline.empty or not complete, "confirmatory baseline rows are absent")
    _require(not qarca.empty or not complete, "confirmatory QARCA rows are absent")

    if not baseline.empty:
        version = _one_value(baseline, "runner_version", "baseline rows")
        code_hash = _one_value(baseline, "code_hash", "baseline rows")
        _validate_hashes(baseline["code_hash"], "baseline code_hash")
        _one_value(baseline, "config_fingerprint", "baseline rows")
        _validate_hashes(baseline["config_fingerprint"], "baseline config_fingerprint")
        _require(
            baseline["fold_fingerprint"].isna().all(),
            "baseline rows must not carry QARCA fold_fingerprint values",
        )
        if expected is not None:
            _require(
                (version, code_hash) == expected["baseline"],
                "baseline output does not match the current runner version/code hash",
            )

    if not qarca.empty:
        version = _one_value(qarca, "runner_version", "QARCA rows")
        code_hash = _one_value(qarca, "code_hash", "QARCA rows")
        _validate_hashes(qarca["code_hash"], "QARCA code_hash")
        _require(
            qarca["config_fingerprint"].isna().all(),
            "QARCA rows must use fold_fingerprint, not baseline config_fingerprint",
        )
        _validate_hashes(qarca["fold_fingerprint"], "QARCA fold_fingerprint")
        fold_counts = qarca.groupby("system", observed=True)["fold_fingerprint"].nunique()
        _require((fold_counts == 1).all(), "each held-out system must have one QARCA fold fingerprint")
        fold_map = qarca.groupby("system", observed=True)["fold_fingerprint"].first()
        _require(
            fold_map.nunique() == len(fold_map),
            "QARCA fold fingerprints must differ across held-out systems",
        )
        if expected is not None:
            _require(
                (version, code_hash) == expected["qarca"],
                "QARCA output does not match the current runner version/code hash",
            )


def _validate_numeric_semantics(data: pd.DataFrame, design: StudyDesign) -> None:
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        _require(not converted.isna().any(), f"{column} contains missing or non-numeric values")
        data[column] = converted
    for column in (
        "repetition",
        "n_pre",
        "n_post",
        "replicate",
        "rank",
        "candidate_count",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "root_streams_available",
        "root_streams_total",
        "root_streams_pre_available",
        "root_streams_post_available",
        "root_streams_scorable",
        "source_size",
        "source_mtime_ns",
        "window",
    ):
        _integer_like(data[column], column)

    for column in (
        "reciprocal_rank",
        "avg_at_5",
        "normalized_rank",
        "realized_rate",
        *FRACTION_COLUMNS,
    ):
        values = data[column].to_numpy(dtype=float)
        _require(np.isfinite(values).all(), f"{column} contains non-finite values")
        _require(((values >= -1e-12) & (values <= 1.0 + 1e-12)).all(), f"{column} is outside [0,1]")
    _require((data["candidate_count"] >= 2).all(), "candidate_count must be at least two")
    _require(
        ((data["rank"] >= 1) & (data["rank"] <= data["candidate_count"])).all(),
        "rank is outside the candidate set",
    )
    _require((data[["n_pre", "n_post", "source_size", "window"]] > 0).all().all(), "window/source sizes must be positive")
    _require(np.isfinite(data["elapsed_ms"]).all() and (data["elapsed_ms"] >= 0).all(), "elapsed_ms must be finite and non-negative")

    rank = data["rank"].to_numpy(dtype=float)
    candidates = data["candidate_count"].to_numpy(dtype=float)
    expected_rr = 1.0 / rank
    expected_avg5 = np.where(rank <= 5, (6.0 - rank) / 5.0, 0.0)
    expected_normalized = np.where(candidates > 1, (rank - 1.0) / (candidates - 1.0), 0.0)
    _require(np.allclose(data["reciprocal_rank"], expected_rr, atol=1e-10), "reciprocal_rank is inconsistent with rank")
    _require(np.allclose(data["avg_at_5"], expected_avg5, atol=1e-10), "avg_at_5 is inconsistent with rank")
    _require(np.allclose(data["normalized_rank"], expected_normalized, atol=1e-10), "normalized_rank is inconsistent with rank")
    for cutoff, column in ((1, "hit_at_1"), (3, "hit_at_3"), (5, "hit_at_5")):
        _require(
            np.array_equal(data[column].to_numpy(dtype=int), (rank <= cutoff).astype(int)),
            f"{column} is inconsistent with rank",
        )

    totals = data["root_streams_total"]
    for column in (
        "root_streams_available",
        "root_streams_pre_available",
        "root_streams_post_available",
        "root_streams_scorable",
    ):
        _require(((data[column] >= 0) & (data[column] <= totals)).all(), f"{column} is inconsistent with root_streams_total")
    _require(
        (data["root_streams_scorable"] <= data["root_streams_pre_available"]).all()
        and (data["root_streams_scorable"] <= data["root_streams_post_available"]).all(),
        "root_streams_scorable exceeds pre/post available streams",
    )

    unperturbed = data["mechanism"].eq("none")
    _require(np.allclose(data.loc[unperturbed, "rate"], 0.0), "unperturbed rows must have rate zero")
    _require((data.loc[unperturbed, "replicate"] == 0).all(), "unperturbed rows must use replicate zero")
    _require(np.allclose(data.loc[unperturbed, "realized_rate"], 0.0), "unperturbed rows must have realized_rate zero")

    qmask = data["method"].isin(design.qarca_methods)
    for column in ("top_score", "score_margin", "root_score", "training_incidents", "training_variants_per_incident", "training_elapsed_ms"):
        values = pd.to_numeric(data.loc[qmask, column], errors="coerce")
        _require(not values.isna().any(), f"QARCA {column} contains missing/non-numeric values")
        _require(np.isfinite(values).all(), f"QARCA {column} contains non-finite values")
        data.loc[qmask, column] = values
    if qmask.any():
        _require((data.loc[qmask, "score_margin"] >= -1e-12).all(), "QARCA score_margin must be non-negative")
        _require((data.loc[qmask, "training_elapsed_ms"] >= 0).all(), "QARCA training_elapsed_ms must be non-negative")
        _require((data.loc[qmask, "training_incidents"] > 0).all(), "QARCA training_incidents must be positive")
        _require((data.loc[qmask, "training_variants_per_incident"] > 0).all(), "training variants must be positive")
        _integer_like(data.loc[qmask, "training_incidents"], "training_incidents")
        _integer_like(data.loc[qmask, "training_variants_per_incident"], "training_variants_per_incident")
    baseline = ~qmask
    for column in ("top_score", "score_margin", "root_score", "training_incidents", "training_variants_per_incident", "training_elapsed_ms"):
        _require(data.loc[baseline, column].isna().all(), f"baseline {column} must be empty")


def _validate_matrix(data: pd.DataFrame, design: StudyDesign, complete: bool) -> None:
    allowed_methods = set(design.methods)
    actual_methods = set(data["method"].astype(str))
    _require(actual_methods <= allowed_methods, f"unexpected methods: {sorted(actual_methods - allowed_methods)}")
    if complete:
        _require(actual_methods == allowed_methods, f"method set is incomplete: {sorted(allowed_methods - actual_methods)}")

    allowed_systems = set(design.systems)
    actual_systems = set(data["system"].astype(str))
    _require(actual_systems <= allowed_systems, f"unexpected systems: {sorted(actual_systems - allowed_systems)}")
    if complete:
        _require(actual_systems == allowed_systems, f"system set is incomplete: {sorted(allowed_systems - actual_systems)}")

    data["rate"] = data["rate"].astype(float).round(12)
    data["replicate"] = data["replicate"].astype(int)
    key = ["system", "case_id", "mechanism", "rate", "replicate", "method"]
    duplicates = data.duplicated(key, keep=False)
    _require(not duplicates.any(), f"duplicate prediction keys: {int(duplicates.sum())} rows")

    expected_scenarios = {(m, round(float(r), 12), int(k)) for m, r, k in design.scenarios}
    actual_scenarios = set(
        data[["mechanism", "rate", "replicate"]].itertuples(index=False, name=None)
    )
    _require(actual_scenarios <= expected_scenarios, f"unexpected scenarios: {sorted(actual_scenarios - expected_scenarios)[:5]}")
    if complete:
        _require(actual_scenarios == expected_scenarios, "scenario set is incomplete")

    incidents = data[["system", "case_id"]].drop_duplicates()
    counts = incidents.groupby("system", observed=True)["case_id"].size()
    if complete:
        _require((counts.reindex(design.systems, fill_value=0) == design.cases_per_system).all(), f"expected {design.cases_per_system} incidents per system; found {counts.to_dict()}")
        method_counts = data.groupby("method", observed=True).size()
        _require((method_counts == design.rows_per_method).all(), f"each method must have {design.rows_per_method:,} rows; found {method_counts.to_dict()}")
        scenario_counts = data.groupby(["system", "case_id", "method"], observed=True).size()
        _require((scenario_counts == len(design.scenarios)).all(), "at least one incident/method lacks the exact scenario matrix")
        method_incidents = data.groupby("method", observed=True)[["system", "case_id"]].apply(lambda x: len(x.drop_duplicates()))
        _require((method_incidents == design.incidents).all(), "incident sets differ across methods")

    mask_key = ["system", "case_id", "mechanism", "rate", "replicate"]
    grouped = data.groupby(mask_key, observed=True, sort=False)
    for column in MASK_INVARIANT_COLUMNS:
        numeric = grouped[column].agg(["min", "max"])
        _require(
            np.allclose(numeric["min"], numeric["max"], rtol=1e-9, atol=1e-11),
            f"{column} differs across methods for the same telemetry mask",
        )
    incident_groups = data.groupby(["system", "case_id"], observed=True, sort=False)
    for column in INCIDENT_METADATA_COLUMNS:
        _require((incident_groups[column].nunique(dropna=False) == 1).all(), f"{column} differs within an incident")

    qmask = data["method"].isin(design.qarca_methods)
    if qmask.any():
        qdata = data.loc[qmask]
        for column in ("training_elapsed_ms", "training_incidents", "training_variants_per_incident"):
            stable = qdata.groupby(["system", "method"], observed=True)[column].nunique(dropna=False)
            _require((stable == 1).all(), f"{column} is not constant within a QARCA fold/model")
        if complete:
            expected_training = design.incidents - design.cases_per_system
            _require((qdata["training_incidents"].astype(int) == expected_training).all(), f"QARCA folds must train on {expected_training} incidents")


def validate_results(
    frame: pd.DataFrame,
    design: StudyDesign = StudyDesign(),
    *,
    complete: bool = True,
    expected_provenance: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return a normalized copy only after all requested invariants hold."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    _require(not missing, f"missing required columns: {sorted(missing)}")
    _require(not frame.empty, "result matrix is empty")
    data = frame.copy()
    for column in ("case_id", "system", "root_service", "fault", "mechanism", "method", "runner_version", "code_hash", "source_fingerprint"):
        _nonempty_strings(data[column], column)
    _validate_hashes(data["source_fingerprint"], "source_fingerprint")
    _validate_numeric_semantics(data, design)
    _validate_matrix(data, design, complete)
    _validate_provenance(data, design, expected_provenance, complete)
    return data.sort_values(
        ["system", "case_id", "mechanism", "rate", "replicate", "method"],
        kind="stable",
    ).reset_index(drop=True)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (total - position) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def sign_flip_pvalue(differences: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    observed = abs(float(differences.mean()))
    exceed = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(np.mean(signs * differences))) >= observed - 1e-15
    return (exceed + 1.0) / (draws + 1.0)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    estimates = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        estimates[start : start + count] = rng.choice(
            values, size=(count, len(values)), replace=True
        ).mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def incident_condition_means(data: pd.DataFrame) -> pd.DataFrame:
    """Average replicate masks within incidents before any comparison."""

    keys = ["system", "case_id", "root_service", "fault", "method", "mechanism", "rate"]
    aggregations: dict[str, tuple[str, str]] = {
        metric: (metric, "mean") for metric in (*METRICS, "normalized_rank")
    }
    aggregations.update(
        {
            "rank": ("rank", "mean"),
            "candidate_count": ("candidate_count", "first"),
            "elapsed_ms": ("elapsed_ms", "mean"),
            "realized_rate": ("realized_rate", "mean"),
            "root_streams_scorable": ("root_streams_scorable", "mean"),
            "observed_fraction": ("observed_fraction", "mean"),
            "score_margin": ("score_margin", "mean"),
            "replicate_masks": ("replicate", "nunique"),
        }
    )
    return data.groupby(keys, observed=True, as_index=False).agg(**aggregations)


def per_case_auc(means: pd.DataFrame, metric: str = "reciprocal_rank") -> pd.DataFrame:
    original = means.loc[means["mechanism"].eq("none"), ["system", "case_id", "method", metric]].rename(columns={metric: "original"})
    rows: list[dict[str, object]] = []
    for mechanism in MECHANISMS:
        subset = means.loc[means["mechanism"].eq(mechanism)]
        for (system, case_id, method), group in subset.groupby(["system", "case_id", "method"], observed=True):
            base = original.loc[(original["system"] == system) & (original["case_id"] == case_id) & (original["method"] == method), "original"]
            if base.empty:
                continue
            mapping = dict(zip(group["rate"], group[metric]))
            rates = sorted(mapping)
            x = np.asarray([0.0, *rates], dtype=float)
            y = np.asarray([float(base.iloc[0]), *(float(mapping[r]) for r in rates)])
            rows.append(
                {
                    "system": system,
                    "case_id": case_id,
                    "method": method,
                    "mechanism": mechanism,
                    "auc": float(np.trapezoid(y, x) / x[-1]),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(
    auc: pd.DataFrame,
    design: StudyDesign,
    resamples: int,
    *,
    complete: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260723)
    rows: list[dict[str, object]] = []
    available = set(auc["method"])
    if PROPOSED not in available:
        return pd.DataFrame(columns=["mechanism", "baseline", "n_incidents", "mean_difference", "ci_low", "ci_high", "p_raw", "p_holm"])
    for mechanism in design.mechanisms:
        pivot = auc.loc[auc["mechanism"].eq(mechanism)].pivot(index=["system", "case_id"], columns="method", values="auc")
        for baseline in design.baselines:
            if baseline not in pivot or PROPOSED not in pivot:
                if complete:
                    raise ValidationError(f"missing confirmatory comparison: {PROPOSED} vs {baseline} under {mechanism}")
                continue
            paired = pivot[[PROPOSED, baseline]].dropna()
            differences = (paired[PROPOSED] - paired[baseline]).to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(differences, rng, resamples)
            rows.append(
                {
                    "mechanism": mechanism,
                    "baseline": baseline,
                    "n_incidents": len(differences),
                    "mean_difference": float(differences.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw": sign_flip_pvalue(differences, rng, resamples),
                }
            )
    if complete:
        _require(len(rows) == 12, "the confirmatory Holm family must contain exactly 12 comparisons")
        expected_pairs = {(m, b) for m in design.mechanisms for b in design.baselines}
        _require({(str(r["mechanism"]), str(r["baseline"])) for r in rows} == expected_pairs, "the Holm family differs from QARCA vs three baselines by four mechanisms")
    adjusted = holm_adjust(float(row["p_raw"]) for row in rows)
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    return pd.DataFrame(rows)


def _summary_table(means: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = means.groupby(keys, observed=True)
    output = grouped.agg(
        incidents=("case_id", "size"),
        reciprocal_rank=("reciprocal_rank", "mean"),
        reciprocal_rank_sd=("reciprocal_rank", "std"),
        avg_at_5=("avg_at_5", "mean"),
        hit_at_1=("hit_at_1", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        hit_at_5=("hit_at_5", "mean"),
        normalized_rank=("normalized_rank", "mean"),
        elapsed_ms=("elapsed_ms", "mean"),
        realized_rate=("realized_rate", "mean"),
    ).reset_index()
    output["reciprocal_rank_se"] = output["reciprocal_rank_sd"] / np.sqrt(output["incidents"])
    return output


def root_evidence_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["system", "case_id", "method", "mechanism", "rate"]
    work = data.copy()
    work["has_root_evidence"] = work["root_streams_scorable"].gt(0)
    work["worst_rank"] = work["rank"].eq(work["candidate_count"]).astype(float)
    unconditional = work.groupby(keys, observed=True, as_index=False).agg(
        evidence_mask_fraction=("has_root_evidence", "mean"),
        mrr_unconditional=("reciprocal_rank", "mean"),
        hit1_unconditional=("hit_at_1", "mean"),
        normalized_rank_unconditional=("normalized_rank", "mean"),
        worst_rank_fraction=("worst_rank", "mean"),
    )
    conditional = work.loc[work["has_root_evidence"]].groupby(keys, observed=True, as_index=False).agg(
        mrr_conditional=("reciprocal_rank", "mean"),
        hit1_conditional=("hit_at_1", "mean"),
        normalized_rank_conditional=("normalized_rank", "mean"),
    )
    incident = unconditional.merge(conditional, on=keys, how="left")
    grouped = incident.groupby(["method", "mechanism", "rate"], observed=True)
    return grouped.agg(
        incidents=("case_id", "size"),
        incidents_with_evidence=("mrr_conditional", "count"),
        evidence_mask_fraction=("evidence_mask_fraction", "mean"),
        mrr_unconditional=("mrr_unconditional", "mean"),
        mrr_conditional=("mrr_conditional", "mean"),
        hit1_unconditional=("hit1_unconditional", "mean"),
        hit1_conditional=("hit1_conditional", "mean"),
        normalized_rank_unconditional=("normalized_rank_unconditional", "mean"),
        normalized_rank_conditional=("normalized_rank_conditional", "mean"),
        worst_rank_fraction=("worst_rank_fraction", "mean"),
    ).reset_index()


def failure_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["system", "case_id", "method", "mechanism", "rate"]
    work = data.copy()
    work["miss_at_1"] = 1.0 - work["hit_at_1"]
    work["miss_at_3"] = 1.0 - work["hit_at_3"]
    work["miss_at_5"] = 1.0 - work["hit_at_5"]
    work["worst_rank"] = work["rank"].eq(work["candidate_count"]).astype(float)
    work["root_unscorable"] = work["root_streams_scorable"].eq(0).astype(float)
    incident = work.groupby(keys, observed=True, as_index=False).agg(
        normalized_rank=("normalized_rank", "mean"),
        miss_at_1=("miss_at_1", "mean"),
        miss_at_3=("miss_at_3", "mean"),
        miss_at_5=("miss_at_5", "mean"),
        worst_rank=("worst_rank", "mean"),
        root_unscorable=("root_unscorable", "mean"),
    )
    return incident.groupby(["method", "mechanism", "rate"], observed=True).agg(
        incidents=("case_id", "size"),
        normalized_rank_mean=("normalized_rank", "mean"),
        normalized_rank_p90=("normalized_rank", lambda x: x.quantile(0.90)),
        miss_at_1=("miss_at_1", "mean"),
        miss_at_3=("miss_at_3", "mean"),
        miss_at_5=("miss_at_5", "mean"),
        worst_rank_fraction=("worst_rank", "mean"),
        root_unscorable_fraction=("root_unscorable", "mean"),
    ).reset_index()


def risk_coverage(means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposed = means.loc[means["method"].eq(PROPOSED)].copy()
    points: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for (mechanism, rate), group in proposed.groupby(["mechanism", "rate"], observed=True, sort=False):
        ranked = group.sort_values("score_margin", ascending=False, kind="stable")
        total = len(ranked)
        cumulative_n = 0
        cumulative_errors = 0.0
        condition_points: list[dict[str, object]] = []
        for margin, tied in ranked.groupby("score_margin", sort=False, observed=True):
            cumulative_n += len(tied)
            cumulative_errors += float((1.0 - tied["hit_at_1"]).sum())
            condition_points.append(
                {
                    "mechanism": mechanism,
                    "rate": rate,
                    "threshold": float(margin),
                    "accepted_incidents": cumulative_n,
                    "coverage": cumulative_n / total,
                    "selective_risk": cumulative_errors / cumulative_n,
                }
            )
        points.extend(condition_points)
        increments = np.diff(
            np.asarray([0.0, *(p["coverage"] for p in condition_points)], dtype=float)
        )
        risk = np.asarray([p["selective_risk"] for p in condition_points], dtype=float)
        summary: dict[str, object] = {
            "mechanism": mechanism,
            "rate": rate,
            "incidents": total,
            # Empirical selective-risk integral.  Tied margins enter together,
            # so neither service/case names nor an arbitrary tie order affect it.
            "aurc": float(np.sum(increments * risk)),
            "full_coverage_risk": float(condition_points[-1]["selective_risk"]),
        }
        for target in (0.5, 0.8):
            chosen = next(point for point in condition_points if float(point["coverage"]) >= target)
            summary[f"risk_at_{int(target * 100)}pct_coverage"] = float(chosen["selective_risk"])
            summary[f"actual_coverage_at_{int(target * 100)}pct"] = float(chosen["coverage"])
        summaries.append(summary)
    return pd.DataFrame(points), pd.DataFrame(summaries)


def ablation_tables(auc: pd.DataFrame, means: pd.DataFrame, design: StudyDesign) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = [method for method in design.qarca_methods if method in set(auc["method"])]
    detailed_rows: list[dict[str, object]] = []
    for mechanism in design.mechanisms:
        pivot = auc.loc[auc["mechanism"].eq(mechanism) & auc["method"].isin(available)].pivot(index=["system", "case_id"], columns="method", values="auc")
        for method in available:
            values = pivot[method].dropna()
            paired = (
                pivot[[PROPOSED, method]].dropna()
                if PROPOSED in pivot and method != PROPOSED
                else pd.DataFrame()
            )
            detailed_rows.append(
                {
                    "mechanism": mechanism,
                    "method": method,
                    "incidents": len(values),
                    "mean_auc": float(values.mean()),
                    "delta_vs_qarca": (
                        0.0
                        if method == PROPOSED
                        else float((paired[method] - paired[PROPOSED]).mean())
                        if not paired.empty
                        else np.nan
                    ),
                }
            )
    detailed = pd.DataFrame(detailed_rows)
    original = means.loc[means["mechanism"].eq("none")].groupby("method", observed=True)["reciprocal_rank"].mean()
    high_loss = means.loc[means["mechanism"].ne("none") & np.isclose(means["rate"], max(design.rates))].groupby("method", observed=True)["reciprocal_rank"].mean()
    overall = auc.groupby("method", observed=True)["auc"].mean()
    rows = []
    for method in available:
        rows.append(
            {
                "method": method,
                "original_mrr": float(original.get(method, np.nan)),
                "robustness_auc": float(overall.get(method, np.nan)),
                "delta_auc_vs_qarca": float(overall.get(method, np.nan) - overall.get(PROPOSED, np.nan)),
                "mrr_at_max_loss": float(high_loss.get(method, np.nan)),
            }
        )
    return pd.DataFrame(rows), detailed


def runtime_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    incident = data.groupby(["system", "case_id", "method", "mechanism", "rate"], observed=True, as_index=False)["elapsed_ms"].mean()
    inference = incident.groupby(["method", "mechanism", "rate"], observed=True)["elapsed_ms"].agg(
        incidents="size", mean_ms="mean", median_ms="median", p95_ms=lambda x: x.quantile(0.95)
    ).reset_index()
    qdata = data.loc[data["method"].isin(QARCA_METHODS)]
    if qdata.empty:
        training = pd.DataFrame(columns=["system", "method", "training_elapsed_ms", "training_incidents", "training_variants_per_incident"])
    else:
        training = qdata.groupby(["system", "method"], observed=True, as_index=False).agg(
            training_elapsed_ms=("training_elapsed_ms", "first"),
            training_incidents=("training_incidents", "first"),
            training_variants_per_incident=("training_variants_per_incident", "first"),
            fold_fingerprint=("fold_fingerprint", "first"),
        )
    return inference, training


def native_missingness(data: pd.DataFrame) -> pd.DataFrame:
    original = data.loc[data["mechanism"].eq("none")].drop_duplicates(["system", "case_id"])
    original = original.assign(native_missing_fraction=1.0 - original["observed_fraction"])
    by_system = original.groupby("system", observed=True).agg(
        incidents=("case_id", "size"),
        observed_fraction_mean=("observed_fraction", "mean"),
        observed_fraction_min=("observed_fraction", "min"),
        native_missing_fraction_mean=("native_missing_fraction", "mean"),
    ).reset_index()
    overall = pd.DataFrame(
        [
            {
                "system": "ALL",
                "incidents": len(original),
                "observed_fraction_mean": original["observed_fraction"].mean(),
                "observed_fraction_min": original["observed_fraction"].min(),
                "native_missing_fraction_mean": original["native_missing_fraction"].mean(),
            }
        ]
    )
    return pd.concat([by_system, overall], ignore_index=True)


def _tex_escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, _tex_escape(method))


def _main_results_tex(original: pd.DataFrame, auc_summary: pd.DataFrame, design: StudyDesign) -> str:
    main = [method for method in (PROPOSED, *design.baselines) if method in set(original["method"])]
    auc_overall = auc_summary.groupby("method", observed=True).apply(
        lambda group: np.average(group["mean_auc"], weights=group["incidents"]),
        include_groups=False,
    )
    lines = [
        "% Generated by scripts/analyze_results.py; do not edit.",
        r"\begin{table}[t]",
        r"\caption{Service-level localization on original telemetry and robustness under synthetic telemetry loss. AUC is the per-incident MRR area from 0\% to 50\% loss, normalized by the rate interval and then averaged. Higher is better.}",
        r"\label{tab:main-results}",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        "Method & Hit@1 & Hit@3 & MRR & MRR AUC \\\\",
        r"\hline",
    ]
    indexed = original.set_index("method")
    for method in main:
        row = indexed.loc[method]
        lines.append(f"{_method_label(method)} & {row.hit_at_1:.3f} & {row.hit_at_3:.3f} & {row.reciprocal_rank:.3f} & {float(auc_overall.get(method, np.nan)):.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _ablation_tex(
    ablation: pd.DataFrame,
    root_evidence: pd.DataFrame,
    risk_summary: pd.DataFrame,
    inference: pd.DataFrame,
    training: pd.DataFrame,
    native: pd.DataFrame,
    design: StudyDesign,
) -> str:
    lines = [
        "% Generated by scripts/analyze_results.py; do not edit.",
        r"\begin{table}[t]",
        r"\caption{Pre-specified QARCA ablations and diagnostics. Delta is relative to QARCA; higher MRR and AUC are better.}",
        r"\label{tab:ablation}",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        "Variant & Original MRR & MRR AUC & $\\Delta$ AUC \\\\",
        r"\hline",
    ]
    for row in ablation.itertuples(index=False):
        lines.append(f"{_method_label(str(row.method))} & {row.original_mrr:.3f} & {row.robustness_auc:.3f} & {row.delta_auc_vs_qarca:+.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])

    maximum = max(design.rates)
    evidence = root_evidence.loc[
        root_evidence["method"].eq(PROPOSED)
        & root_evidence["mechanism"].isin(design.mechanisms)
        & np.isclose(root_evidence["rate"], maximum)
    ]
    risk = risk_summary.loc[
        risk_summary["mechanism"].isin(design.mechanisms)
        & np.isclose(risk_summary["rate"], maximum)
    ]
    q_runtime = inference.loc[inference["method"].eq(PROPOSED)]
    q_training = training.loc[training["method"].eq(PROPOSED)]
    native_all = native.loc[native["system"].eq("ALL")]

    if not native_all.empty:
        native_fraction = float(native_all.iloc[0]["native_missing_fraction_mean"])
        lines.append(
            f"Before imposed loss, the mean native missing-value fraction was {native_fraction:.1%}; "
            "all synthetic masks were applied only to originally finite values."
        )
    if not evidence.empty:
        incidents = evidence["incidents"].to_numpy(dtype=float)
        conditional_weights = evidence["incidents_with_evidence"].to_numpy(dtype=float)
        evidence_fraction = float(np.average(evidence["evidence_mask_fraction"], weights=incidents))
        unconditional = float(np.average(evidence["mrr_unconditional"], weights=incidents))
        valid = evidence["mrr_conditional"].notna() & conditional_weights.astype(bool)
        conditional = (
            float(np.average(evidence.loc[valid, "mrr_conditional"], weights=conditional_weights[valid]))
            if valid.any()
            else np.nan
        )
        conditional_text = f"{conditional:.3f}" if np.isfinite(conditional) else "not estimable"
        lines.append(
            f"At {maximum:.0%} imposed loss, averaged across mechanisms, {evidence_fraction:.1%} of QARCA masks retained at least one diagnostically scorable root stream. "
            f"Its unconditional MRR was {unconditional:.3f}, versus {conditional_text} when summarized only over incidents with retained root evidence; "
            "the unconditional value remains the primary result."
        )
    if not risk.empty:
        risk50 = float(risk["risk_at_50pct_coverage"].mean())
        actual50 = float(risk["actual_coverage_at_50pct"].mean())
        full = float(risk["full_coverage_risk"].mean())
        lines.append(
            f"At the same loss endpoint, the label-free top-two margin yielded mean Hit@1 error {risk50:.3f} at {actual50:.1%} empirical coverage, compared with {full:.3f} at full coverage. "
            "This post-hoc risk--coverage description does not validate a deployment threshold."
        )
    if not q_runtime.empty:
        inference_ms = float(
            np.average(q_runtime["mean_ms"], weights=q_runtime["incidents"])
        )
        runtime_text = (
            f"Recorded QARCA inference averaged {inference_ms:.1f} ms per incident-condition"
        )
        if not q_training.empty:
            training_ms = float(q_training["training_elapsed_ms"].mean())
            runtime_text += (
                f", while fitting a held-out-system fold averaged {training_ms / 1000.0:.1f} s"
            )
        lines.append(
            runtime_text
            + "; these implementation- and hardware-specific timings are descriptive."
        )
    lines.append("")
    return "\n".join(lines)


def _data_derived_text(
    original: pd.DataFrame,
    auc: pd.DataFrame,
    tests: pd.DataFrame,
    design: StudyDesign,
) -> dict[str, str]:
    overall = auc.groupby("method", observed=True)["auc"].mean()
    cases = auc[["system", "case_id"]].drop_duplicates().shape[0]
    if PROPOSED not in overall or not any(b in overall for b in design.baselines):
        sentence = "The exploratory input did not contain all confirmatory methods, so no confirmatory comparison is reported."
        return {name: sentence + "\n" for name in ("results_summary.tex", "abstract_result.tex", "conclusion_result.tex")}
    available_baselines = [b for b in design.baselines if b in overall]
    best = max(available_baselines, key=lambda method: float(overall[method]))
    proposed_auc = float(overall[PROPOSED])
    baseline_auc = float(overall[best])
    delta = proposed_auc - baseline_auc
    wins = int(((tests["p_holm"] < 0.05) & (tests["mean_difference"] > 0)).sum()) if not tests.empty else 0
    losses = int(((tests["p_holm"] < 0.05) & (tests["mean_difference"] < 0)).sum()) if not tests.empty else 0
    original_index = original.set_index("method")
    original_mrr = float(original_index.loc[PROPOSED, "reciprocal_rank"])
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
    results = (
        f"After averaging masks within each incident, QARCA obtained an original-telemetry MRR of {original_mrr:.3f} and a four-mechanism mean MRR AUC of {proposed_auc:.3f}. "
        f"The strongest baseline by the same AUC was {_method_label(best)} at {baseline_auc:.3f}; QARCA was {abs(delta):.3f} {direction}. "
        f"Within the pre-specified family of {len(tests)} paired sign-flip comparisons, {wins} favored QARCA and {losses} favored a baseline at Holm-adjusted $p<0.05$."
    )
    abstract = (
        f"Across {cases} RCAEval incidents and four synthetic metric-loss mechanisms up to 50\\%, QARCA achieved mean per-incident MRR AUC {proposed_auc:.3f}, compared with {baseline_auc:.3f} for the strongest evaluated baseline."
    )
    conclusion = (
        f"Under the evaluated synthetic metric-loss masks, QARCA's mean MRR AUC was {proposed_auc:.3f} versus {baseline_auc:.3f} for the strongest baseline ({delta:+.3f}); {wins} of {len(tests)} Holm-adjusted comparisons favored QARCA. "
        "These results do not establish performance under naturally occurring telemetry outages or telemetry modalities absent from RCAEval RE1."
    )
    return {
        "results_summary.tex": "% Generated; do not edit.\n" + results + "\n",
        "abstract_result.tex": "% Generated; do not edit.\n" + abstract + "\n",
        "conclusion_result.tex": "% Generated; do not edit.\n" + conclusion + "\n",
    }


def exact_alt_text(aggregate: pd.DataFrame, design: StudyDesign) -> str:
    main = [PROPOSED, *design.baselines]
    original = aggregate.loc[aggregate["mechanism"].eq("none")].set_index("method")
    parts = [
        "Four line-chart panels report incident-weighted mean reciprocal rank versus the imposed missing-telemetry rate for point, channel-local burst, whole-stream, and incident-correlated loss."
    ]
    values = [f"{METHOD_LABELS.get(method, method).replace('$', '')} {float(original.loc[method, 'reciprocal_rank']):.3f}" for method in main if method in original.index]
    parts.append("Original telemetry at zero imposed loss has mean reciprocal rank " + ", ".join(values) + ".")
    maximum = max(design.rates)
    for mechanism in design.mechanisms:
        subset = aggregate.loc[aggregate["mechanism"].eq(mechanism) & np.isclose(aggregate["rate"], maximum)].set_index("method")
        values = [f"{METHOD_LABELS.get(method, method).replace('$', '')} {float(subset.loc[method, 'reciprocal_rank']):.3f}" for method in main if method in subset.index]
        parts.append(f"At {maximum:.0%} {mechanism} loss, mean reciprocal rank is " + ", ".join(values) + ".")
    parts.append("Higher values are better; lines connect the evaluated rates and do not imply measurements between them.")
    return " ".join(parts) + "\n"


def build_artifacts(
    data: pd.DataFrame,
    design: StudyDesign = StudyDesign(),
    *,
    resamples: int = 10_000,
    complete: bool = True,
) -> AnalysisArtifacts:
    means = incident_condition_means(data)
    aggregate = _summary_table(means, ["mechanism", "rate", "method"])
    by_system = _summary_table(means, ["system", "mechanism", "rate", "method"])
    by_fault = _summary_table(means, ["system", "fault", "mechanism", "rate", "method"])
    original = aggregate.loc[aggregate["mechanism"].eq("none")].drop(columns=["mechanism", "rate"]).reset_index(drop=True)
    auc = per_case_auc(means)
    auc_summary = auc.groupby(["mechanism", "method"], observed=True).agg(
        incidents=("case_id", "size"), mean_auc=("auc", "mean"), sd_auc=("auc", "std")
    ).reset_index()
    tests = paired_tests(auc, design, resamples, complete=complete)
    root_evidence = root_evidence_summary(data)
    failures = failure_summary(data)
    risk_points, risk_summary = risk_coverage(means)
    ablation, ablation_detail = ablation_tables(auc, means, design)
    inference, training = runtime_tables(data)
    native = native_missingness(data)

    generated = {
        "results.tex": _main_results_tex(original, auc_summary, design),
        "ablation_summary.tex": _ablation_tex(
            ablation,
            root_evidence,
            risk_summary,
            inference,
            training,
            native,
            design,
        ),
    }
    generated.update(_data_derived_text(original, auc, tests, design))
    frames = {
        "incident_condition_means.csv": means,
        "aggregate.csv": aggregate,
        "by_system.csv": by_system,
        "by_system_fault.csv": by_fault,
        "original_results.csv": original,
        "per_case_robustness_auc.csv": auc,
        "robustness_auc.csv": auc_summary,
        "paired_tests.csv": tests,
        "native_missingness.csv": native,
        "root_evidence.csv": root_evidence,
        "normalized_rank_failures.csv": failures,
        "risk_coverage.csv": risk_points,
        "risk_coverage_summary.csv": risk_summary,
        "ablation_diagnostics.csv": ablation,
        "ablation_by_mechanism.csv": ablation_detail,
        "inference_runtime.csv": inference,
        "training_runtime.csv": training,
    }
    figure_data = aggregate.loc[aggregate["method"].isin((PROPOSED, *design.baselines))].copy()
    return AnalysisArtifacts(frames, generated, exact_alt_text(aggregate, design), figure_data)


def plot_robustness(data: pd.DataFrame, output_base: Path, design: StudyDesign) -> None:
    methods = [PROPOSED, *design.baselines]
    styles = {
        "qarca": ("black", "o", "-"),
        "nsigma": ("0.30", "s", "--"),
        "baro": ("0.52", "^", "-."),
        "median_shift": ("0.72", "D", ":"),
    }
    original = data.loc[data["mechanism"].eq("none")].set_index("method")
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.0), sharey=True)
    for axis, mechanism in zip(axes, design.mechanisms):
        subset = data.loc[data["mechanism"].eq(mechanism)]
        for method in methods:
            if method not in set(subset["method"]) or method not in original.index:
                continue
            grouped = subset.loc[subset["method"].eq(method)].sort_values("rate")
            x = np.asarray([0.0, *grouped["rate"].to_list()])
            y = np.asarray([float(original.loc[method, "reciprocal_rank"]), *grouped["reciprocal_rank"].to_list()])
            color, marker, linestyle = styles[method]
            axis.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=1.4, markersize=4, label=METHOD_LABELS[method].replace("$", ""))
        axis.set_title(mechanism.replace("_", " ").title())
        axis.set_xlabel("Missing-telemetry rate")
        axis.set_xlim(0.0, max(design.rates) + 0.01)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Mean reciprocal rank")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_artifacts(
    artifacts: AnalysisArtifacts,
    destinations: Destinations,
    design: StudyDesign,
) -> None:
    """Stage every file first, then atomically replace final files."""

    with tempfile.TemporaryDirectory(prefix="analysis-stage-", dir=ROOT) as temporary:
        stage = Path(temporary)
        stage_summary = stage / "summary"
        stage_figures = stage / "figures"
        stage_generated = stage / "generated"
        stage_summary.mkdir()
        stage_figures.mkdir()
        stage_generated.mkdir()
        for name, frame in artifacts.frames.items():
            frame.to_csv(stage_summary / name, index=False)
        for name, content in artifacts.generated_text.items():
            (stage_generated / name).write_text(content, encoding="utf-8")
        (stage_figures / "robustness-alt.txt").write_text(artifacts.alt_text, encoding="utf-8")
        (stage_figures / "alt-text.txt").write_text(artifacts.alt_text, encoding="utf-8")
        plot_robustness(artifacts.figure_data, stage_figures / "robustness", design)

        for final_directory in (destinations.summary, destinations.figures, destinations.generated):
            final_directory.mkdir(parents=True, exist_ok=True)
        for source in stage_summary.iterdir():
            os.replace(source, destinations.summary / source.name)
        for source in stage_figures.iterdir():
            os.replace(source, destinations.figures / source.name)
        for source in stage_generated.iterdir():
            os.replace(source, destinations.generated / source.name)


def run_pipeline(
    input_path: Path,
    destinations: Destinations,
    *,
    design: StudyDesign = StudyDesign(),
    resamples: int = 10_000,
    complete: bool = True,
    expected_provenance: Mapping[str, tuple[str, str]] | None = None,
) -> AnalysisArtifacts:
    raw = pd.read_csv(input_path)
    validated = validate_results(
        raw,
        design,
        complete=complete,
        expected_provenance=expected_provenance,
    )
    artifacts = build_artifacts(validated, design, resamples=resamples, complete=complete)
    write_artifacts(artifacts, destinations, design)
    return artifacts


def main() -> int:
    args = parse_args()
    if args.resamples < 100:
        raise SystemExit("--resamples must be at least 100")
    try:
        destinations = resolve_destinations(args)
        run_pipeline(
            args.input,
            destinations,
            resamples=args.resamples,
            complete=not args.allow_incomplete,
            expected_provenance=current_runner_provenance(),
        )
    except ValidationError as exc:
        raise SystemExit(f"Validation failed before output generation: {exc}") from exc
    print(f"Validated input and wrote summaries to {destinations.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
