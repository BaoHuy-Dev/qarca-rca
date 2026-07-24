"""Experiment execution and ranking metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .candidates import candidate_metric_columns, service_of
from .data import load_case
from .methods import localize
from .missingness import case_seed, inject_missingness


@dataclass(frozen=True)
class Prediction:
    case_id: str
    system: str
    root_service: str
    fault: str
    repetition: int
    n_pre: int
    n_post: int
    mechanism: str
    rate: float
    replicate: int
    realized_rate: float
    method: str
    rank: int
    reciprocal_rank: float
    avg_at_5: float
    normalized_rank: float
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    candidate_count: int
    elapsed_ms: float
    top_5: str
    top_evidence_metric: str
    root_observed_fraction: float
    root_streams_available: int
    root_streams_total: int
    root_pre_observed_fraction: float
    root_post_observed_fraction: float
    root_streams_pre_available: int
    root_streams_post_available: int
    root_streams_scorable: int
    observed_fraction: float

    def row(self) -> dict[str, object]:
        return asdict(self)


def evaluate_ranking(root: str, ranks: list[str]) -> tuple[int, float, int, int, int]:
    try:
        rank = ranks.index(root) + 1
        rr = 1.0 / rank
        present = True
    except ValueError:
        rank = len(ranks) + 1
        rr = 0.0
        present = False
    return (
        rank,
        rr,
        int(present and rank <= 1),
        int(present and rank <= 3),
        int(present and rank <= 5),
    )


def evaluate_service_scores(
    root: str, scores: dict[str, float]
) -> tuple[int, float, int, int, int]:
    """Evaluate with the worst rank in an exact tie, independent of names."""

    if root not in scores:
        rank = len(scores) + 1
        return rank, 0.0, 0, 0, 0
    root_score = float(scores[root])
    rank = sum(float(score) >= root_score for score in scores.values())
    reciprocal_rank = 1.0 / rank
    return (
        rank,
        reciprocal_rank,
        int(rank <= 1),
        int(rank <= 3),
        int(rank <= 5),
    )


def observability_summary(
    frame, n_pre: int, root_service: str, minimum_samples: int = 3
) -> dict[str, float | int]:
    """Summarize root evidence separately before and after the boundary."""

    metric_columns = candidate_metric_columns(frame)
    root_columns = [
        column for column in metric_columns if service_of(column) == root_service
    ]
    root_values = frame[root_columns]
    pre = root_values.iloc[:n_pre]
    post = root_values.iloc[n_pre:]
    pre_counts = pre.notna().sum(axis=0)
    post_counts = post.notna().sum(axis=0)
    all_values = frame[metric_columns]
    return {
        "root_observed_fraction": (
            float(root_values.notna().to_numpy().mean()) if root_columns else 0.0
        ),
        "root_streams_available": int(
            sum(root_values[column].notna().any() for column in root_columns)
        ),
        "root_streams_total": len(root_columns),
        "root_pre_observed_fraction": (
            float(pre.notna().to_numpy().mean()) if root_columns else 0.0
        ),
        "root_post_observed_fraction": (
            float(post.notna().to_numpy().mean()) if root_columns else 0.0
        ),
        "root_streams_pre_available": int((pre_counts > 0).sum()),
        "root_streams_post_available": int((post_counts > 0).sum()),
        "root_streams_scorable": int(
            ((pre_counts >= minimum_samples) & (post_counts >= minimum_samples)).sum()
        ),
        "observed_fraction": (
            float(all_values.notna().to_numpy().mean()) if metric_columns else 0.0
        ),
    }


def evaluate_case(
    csv_path: str | Path,
    methods: tuple[str, ...],
    mechanisms: tuple[str, ...],
    rates: tuple[float, ...],
    replicates: int,
    window: int = 300,
) -> list[dict[str, object]]:
    """Evaluate every requested scenario for one original incident."""

    case = load_case(csv_path, window=window)
    scenarios: list[tuple[str, float, int]] = [("none", 0.0, 0)]
    for mechanism in mechanisms:
        if mechanism == "none":
            continue
        for rate in rates:
            for replicate in range(replicates):
                scenarios.append((mechanism, rate, replicate))

    predictions: list[dict[str, object]] = []
    for mechanism, rate, replicate in scenarios:
        seed = case_seed(case.case_id, mechanism, rate, replicate)
        corrupted, realized = inject_missingness(
            case.frame,
            mechanism=mechanism,
            rate=rate,
            seed=seed,
            n_pre=case.n_pre,
        )
        for method in methods:
            result = localize(method, corrupted, case.n_pre)
            service_ranks = list(result["service_ranks"])
            metric_ranks = list(result["metric_ranks"])
            service_scores = dict(result["service_scores"])
            rank, rr, hit1, hit3, hit5 = evaluate_service_scores(
                case.service, service_scores
            )
            avg5 = (6 - rank) / 5.0 if rr > 0.0 and rank <= 5 else 0.0
            normalized_rank = (
                (rank - 1) / (len(service_ranks) - 1)
                if rr > 0.0 and len(service_ranks) > 1
                else (0.0 if rr > 0.0 else 1.0)
            )
            observability = observability_summary(
                corrupted, case.n_pre, case.service
            )
            prediction = Prediction(
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
                elapsed_ms=float(result["elapsed_ms"]),
                top_5="|".join(service_ranks[:5]),
                top_evidence_metric=next(
                    (
                        metric
                        for metric in metric_ranks
                        if service_of(metric)
                        in set(result["service_ranks"])
                    ),
                    "",
                ),
                **observability,
            )
            predictions.append(prediction.row())
    return predictions
