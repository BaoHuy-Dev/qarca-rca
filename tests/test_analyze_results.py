from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_results_under_test", ROOT / "scripts" / "analyze_results.py"
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope="module")
def tiny_design():
    return analysis.StudyDesign(
        systems=("OB", "SS", "TT"),
        cases_per_system=1,
        mechanisms=analysis.MECHANISMS,
        rates=(0.1, 0.5),
        replicates=2,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture()
def complete_matrix(tiny_design):
    fold_hash = {system: _sha(f"fold-{system}") for system in tiny_design.systems}
    variants = {
        "qarca_original_train": 1,
        "qarca_no_quality_original": 1,
        "qarca_no_incident_train": 10,
    }
    rows: list[dict[str, object]] = []
    for system_index, system in enumerate(tiny_design.systems):
        case_id = f"{system}-case"
        source_hash = _sha(f"source-{case_id}")
        for mechanism, rate, replicate in tiny_design.scenarios:
            realized = 0.0 if mechanism == "none" else rate * 0.8
            scorable = 2 if mechanism == "stream" and rate == 0.5 else 4
            root_fraction = 0.92 - realized * 0.40
            pre_fraction = 0.93 - realized * 0.25
            post_fraction = 0.91 - realized * 0.55
            observed_fraction = 0.95 - realized * 0.35
            for method_index, method in enumerate(tiny_design.methods):
                rank = 1 + (
                    method_index
                    + tiny_design.mechanisms.index(mechanism)
                    if mechanism != "none"
                    else method_index
                ) % 5
                rank = 1 + (rank - 1 + replicate) % 5
                qmethod = method in tiny_design.qarca_methods
                row = {
                    "case_id": case_id,
                    "system": system,
                    "root_service": f"root-{system}",
                    "fault": "cpu",
                    "repetition": 0,
                    "n_pre": 60,
                    "n_post": 60,
                    "mechanism": mechanism,
                    "rate": rate,
                    "replicate": replicate,
                    "realized_rate": realized,
                    "method": method,
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "avg_at_5": (6 - rank) / 5.0,
                    "normalized_rank": (rank - 1) / 4.0,
                    "hit_at_1": int(rank <= 1),
                    "hit_at_3": int(rank <= 3),
                    "hit_at_5": int(rank <= 5),
                    "candidate_count": 5,
                    "elapsed_ms": 1.0 + method_index / 10,
                    "top_5": "a|b|c|d|e",
                    "top_evidence_metric": "",
                    "root_observed_fraction": root_fraction,
                    "root_streams_available": 4,
                    "root_streams_total": 4,
                    "root_pre_observed_fraction": pre_fraction,
                    "root_post_observed_fraction": post_fraction,
                    "root_streams_pre_available": 4,
                    "root_streams_post_available": 4,
                    "root_streams_scorable": scorable,
                    "observed_fraction": observed_fraction,
                    "runner_version": "qarca-test" if qmethod else "baseline-test",
                    "code_hash": _sha("qarca-code") if qmethod else _sha("baseline-code"),
                    "config_fingerprint": np.nan if qmethod else _sha("baseline-config"),
                    "fold_fingerprint": fold_hash[system] if qmethod else np.nan,
                    "source_fingerprint": source_hash,
                    "source_size": 1000 + system_index,
                    "source_mtime_ns": 123456 + system_index,
                    "window": 300,
                    "top_score": 0.8 + method_index / 100 if qmethod else np.nan,
                    "score_margin": 0.2 + method_index / 100 if qmethod else np.nan,
                    "root_score": 0.6 + method_index / 100 if qmethod else np.nan,
                    "training_incidents": 2 if qmethod else np.nan,
                    "training_variants_per_incident": variants.get(method, 13) if qmethod else np.nan,
                    "training_elapsed_ms": 10.0 + method_index + system_index if qmethod else np.nan,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_complete_matrix_validates_and_builds_exact_confirmatory_family(
    complete_matrix, tiny_design
):
    validated = analysis.validate_results(
        complete_matrix,
        tiny_design,
        complete=True,
        expected_provenance=None,
    )
    assert len(validated) == len(tiny_design.methods) * tiny_design.rows_per_method

    artifacts = analysis.build_artifacts(
        validated, tiny_design, resamples=100, complete=True
    )
    tests = artifacts.frames["paired_tests.csv"]
    assert len(tests) == 12
    assert set(zip(tests.mechanism, tests.baseline)) == {
        (mechanism, baseline)
        for mechanism in tiny_design.mechanisms
        for baseline in tiny_design.baselines
    }
    assert set(artifacts.generated_text) == {
        "results.tex",
        "results_summary.tex",
        "ablation_summary.tex",
        "abstract_result.tex",
        "conclusion_result.tex",
    }
    incident_means = artifacts.frames["incident_condition_means.csv"]
    perturbed = incident_means.loc[incident_means.mechanism.ne("none")]
    assert (perturbed.replicate_masks == tiny_design.replicates).all()
    assert "Original telemetry" in artifacts.alt_text


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate prediction"),
        (lambda frame: frame.iloc[1:].copy(), "rows"),
        (
            lambda frame: frame.assign(
                method=np.where(frame.index == 0, "smoke_method", frame.method)
            ),
            "unexpected methods",
        ),
    ],
)
def test_invalid_matrix_shape_or_method_is_rejected(
    complete_matrix, tiny_design, mutation, match
):
    with pytest.raises(analysis.ValidationError, match=match):
        analysis.validate_results(
            mutation(complete_matrix),
            tiny_design,
            complete=True,
            expected_provenance=None,
        )


def test_cross_method_mask_mismatch_is_rejected(complete_matrix, tiny_design):
    invalid = complete_matrix.copy()
    target = (
        invalid.method.eq("baro")
        & invalid.system.eq("OB")
        & invalid.mechanism.eq("point")
        & np.isclose(invalid.rate, 0.1)
        & invalid.replicate.eq(0)
    )
    invalid.loc[target, "root_post_observed_fraction"] += 0.01
    with pytest.raises(analysis.ValidationError, match="differs across methods"):
        analysis.validate_results(
            invalid, tiny_design, complete=True, expected_provenance=None
        )


def test_mixed_fold_fingerprint_is_rejected(complete_matrix, tiny_design):
    invalid = complete_matrix.copy()
    target_index = invalid.loc[
        invalid.method.eq("qarca") & invalid.system.eq("OB")
    ].index[0]
    invalid.loc[target_index, "fold_fingerprint"] = _sha("stale-fold")
    with pytest.raises(analysis.ValidationError, match="one QARCA fold fingerprint"):
        analysis.validate_results(
            invalid, tiny_design, complete=True, expected_provenance=None
        )


def test_missing_column_fails_before_any_output(
    complete_matrix, tiny_design, tmp_path
):
    invalid_path = tmp_path / "invalid.csv"
    complete_matrix.drop(columns="root_streams_scorable").to_csv(
        invalid_path, index=False
    )
    destinations = analysis.Destinations(
        tmp_path / "summary", tmp_path / "figures", tmp_path / "generated"
    )
    with pytest.raises(analysis.ValidationError, match="missing required columns"):
        analysis.run_pipeline(
            invalid_path,
            destinations,
            design=tiny_design,
            resamples=100,
            complete=True,
            expected_provenance=None,
        )
    assert not destinations.summary.exists()
    assert not destinations.figures.exists()
    assert not destinations.generated.exists()


def test_incomplete_escape_hatch_requires_exploratory_paths(tmp_path):
    args = SimpleNamespace(
        allow_incomplete=True,
        output=tmp_path / "summary",
        figures=tmp_path / "figures",
        generated=tmp_path / "generated",
    )
    with pytest.raises(analysis.ValidationError, match="exploratory"):
        analysis.resolve_destinations(args)

    args.output = tmp_path / "exploratory" / "summary"
    args.figures = tmp_path / "exploratory" / "figures"
    args.generated = tmp_path / "exploratory" / "generated"
    resolved = analysis.resolve_destinations(args)
    assert "exploratory" in resolved.generated.parts
