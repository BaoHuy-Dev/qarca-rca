from dataclasses import replace

import numpy as np
import pandas as pd

from trailrca.data import FailureCase
from trailrca.quality_ranker import (
    FEATURE_COLUMNS,
    NO_QUALITY_FEATURE_COLUMNS,
    QualityAwareServiceRanker,
    build_training_matrix,
    extract_case_features,
    extract_service_features,
    leave_one_system_out_indices,
)


def incident_frame(
    services: tuple[str, ...],
    shifted_service: str,
    *,
    seed: int,
    n_pre: int = 30,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = 2 * n_pre
    data: dict[str, np.ndarray] = {"time": np.arange(length, dtype=float)}
    for service in services:
        for metric_type in ("cpu", "mem", "latency"):
            values = rng.normal(0.0, 0.15, size=length)
            if service == shifted_service:
                values[n_pre:] += {"cpu": 4.0, "mem": 2.5, "latency": 3.0}[
                    metric_type
                ]
            data[f"{service}_{metric_type}"] = values
    return pd.DataFrame(data)


def failure_case(
    case_number: int,
    system: str,
    services: tuple[str, ...],
    root: str,
) -> FailureCase:
    frame = incident_frame(services, root, seed=case_number)
    return FailureCase(
        case_id=f"opaque-{case_number}",
        system=system,
        service=root,
        fault="irrelevant-label",
        repetition=case_number,
        inject_time=30,
        frame=frame,
        n_pre=30,
    )


def test_feature_extraction_is_label_and_system_invariant() -> None:
    case = failure_case(1, "train-system", ("root1", "peer1"), "root1")
    relabelled = replace(
        case,
        case_id="contains/a/different/label",
        system="unseen-system",
        service="peer1",
        fault="different-fault",
        repetition=999,
        inject_time=-10,
    )

    first = extract_case_features(case)
    second = extract_case_features(relabelled)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == FEATURE_COLUMNS
    assert all(np.issubdtype(dtype, np.number) for dtype in first.dtypes)
    assert not any("identity" in column or "system" in column for column in first)


def test_anomaly_quality_gap_and_relative_features() -> None:
    frame = incident_frame(("cause", "peer"), "cause", seed=3)
    # One contiguous outage is visible in every peer stream after the boundary.
    peer_columns = [column for column in frame if column.startswith("peer_")]
    frame.loc[35:49, peer_columns] = np.nan

    features = extract_service_features(frame, n_pre=30, blocks=10)

    assert features.loc["cause", "robust_median_shift_max"] > features.loc[
        "peer", "robust_median_shift_max"
    ]
    assert features.loc["peer", "post_availability_mean"] == 0.5
    assert features.loc["peer", "post_longest_gap_mean"] == 0.5
    assert features.loc["peer", "block_longest_valid_run_mean"] < features.loc[
        "cause", "block_longest_valid_run_mean"
    ]
    assert features.loc[
        "cause", "relative_robust_median_shift_max_percentile"
    ] == 1.0


def test_missing_pre_reference_is_not_filled_from_post() -> None:
    frame = incident_frame(("cold", "peer"), "peer", seed=5)
    frame.loc[:29, "cold_cpu"] = np.nan
    frame.loc[30:, "cold_cpu"] = 1_000_000.0

    features = extract_service_features(frame, n_pre=30)

    # The unavailable reference produces no anomaly evidence; the availability
    # and validity features preserve why, rather than borrowing post values.
    assert features.loc["cold", "kind_cpu_standard_max_max"] == 0.0
    assert features.loc["cold", "kind_cpu_robust_max_max"] == 0.0
    assert features.loc["cold", "pre_availability_min"] < 1.0
    assert features.loc["cold", "reference_valid_min"] == 0.0


def test_per_incident_weights_and_deterministic_unseen_service_ranking() -> None:
    cases: list[FailureCase] = []
    for index in range(6):
        services = (f"cause{index}", f"peer{index}a", f"peer{index}b")
        cases.append(failure_case(10 + index, "training", services, services[0]))

    matrix = build_training_matrix(cases)
    assert tuple(matrix.X.columns) == FEATURE_COLUMNS
    for incident_index in range(len(cases)):
        selected = matrix.incident_index == incident_index
        assert np.isclose(matrix.sample_weight[selected].sum(), 1.0)
        assert matrix.y[selected].sum() == 1

    held_out = failure_case(
        100,
        "held-out",
        ("novelcause", "novelpeer1", "novelpeer2"),
        "novelcause",
    )
    first = QualityAwareServiceRanker().fit(cases)
    second = QualityAwareServiceRanker().fit(cases)

    first_scores = first.predict_case_scores(held_out)
    second_scores = second.predict_case_scores(held_out)
    np.testing.assert_allclose(first_scores, second_scores, rtol=0.0, atol=0.0)
    assert first.rank_case(held_out)[0] == "novelcause"

    # Changing the prediction-time root/fault labels cannot alter any score.
    wrong_labels = replace(held_out, service="novelpeer1", fault="another-fault")
    pd.testing.assert_series_equal(
        first.predict_case_scores(held_out), first.predict_case_scores(wrong_labels)
    )


def test_leave_one_system_out_indices_are_disjoint_and_complete() -> None:
    cases = [
        failure_case(201, "A", ("a1", "a2"), "a1"),
        failure_case(202, "B", ("b1", "b2"), "b1"),
        failure_case(203, "A", ("a3", "a4"), "a3"),
    ]
    train, test = leave_one_system_out_indices(cases, "B")

    assert train.tolist() == [0, 2]
    assert test.tolist() == [1]
    assert set(train).isdisjoint(test)
    assert sorted(np.concatenate((train, test)).tolist()) == [0, 1, 2]


def test_precomputed_fit_and_quality_ablation_schema() -> None:
    cases = [
        failure_case(301, "A", ("roota", "peera"), "roota"),
        failure_case(302, "A", ("rootb", "peerb"), "rootb"),
    ]
    matrix = build_training_matrix(cases)
    model = QualityAwareServiceRanker(
        feature_columns=NO_QUALITY_FEATURE_COLUMNS
    ).fit_precomputed(
        matrix.X,
        matrix.y,
        matrix.sample_weight,
        n_training_incidents=len(cases),
    )
    features = extract_case_features(cases[0])
    scores = model.predict_feature_scores(features)

    assert set(scores.index) == set(features.index)
    assert np.isfinite(scores).all()
    assert 0 < len(NO_QUALITY_FEATURE_COLUMNS) < len(FEATURE_COLUMNS)
    quality_markers = (
        "availability",
        "longest_gap",
        "block_coverage",
        "block_valid",
        "block_longest",
        "reference_valid",
        "post_valid",
        "evidence_quality",
    )
    assert not any(
        marker in column
        for column in NO_QUALITY_FEATURE_COLUMNS
        for marker in quality_markers
    )
