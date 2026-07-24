"""A fixed service-candidate universe shared by every localizer."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


# RCAEval's Online Boutique export contains an aggregate passthrough cluster
# that is not an application service and is excluded by the benchmark's BARO
# preprocessing.  All other named services/infrastructure components remain
# candidates for every method, whether or not any values survive a mask.
EXCLUDED_SERVICES = frozenset({"PassthroughCluster"})


def service_of(metric: str) -> str:
    """Map RCAEval's ``service_metric`` convention to a service identifier."""

    return str(metric).split("_", 1)[0]


def candidate_metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return candidate metric names based on schema, never observed values."""

    return [
        str(column)
        for column in frame.columns
        if column != "time" and service_of(str(column)) not in EXCLUDED_SERVICES
    ]


def candidate_services(frame: pd.DataFrame) -> list[str]:
    """Return the same lexical service universe under every telemetry mask."""

    return sorted({service_of(column) for column in candidate_metric_columns(frame)})


def complete_service_ranking(
    ranked_services: Iterable[str], frame: pd.DataFrame
) -> list[str]:
    """Filter to the fixed universe and append no-evidence services.

    A baseline may drop a constant or wholly unavailable stream internally.
    Such services remain valid candidates and are placed after all services
    with method-specific evidence.  Lexical order is used only to serialize a
    readable list; scientific metrics use conservative tie-aware ranks from
    :func:`ranked_service_scores` rather than this display order.
    """

    universe = candidate_services(frame)
    allowed = set(universe)
    seen: set[str] = set()
    completed: list[str] = []
    for raw_service in ranked_services:
        service = str(raw_service)
        if service in allowed and service not in seen:
            completed.append(service)
            seen.add(service)
    completed.extend(service for service in universe if service not in seen)
    return completed


def ranked_service_scores(
    ranked_services: Iterable[str], frame: pd.DataFrame
) -> dict[str, float]:
    """Turn an evidence order into scores over the complete candidate set.

    Services retained by a method get distinct positive ordinal scores.
    Services omitted because every usable stream disappeared share score zero;
    their root rank is therefore the worst position in that tie, independent of
    service spelling.
    """

    completed = complete_service_ranking(ranked_services, frame)
    evidence: list[str] = []
    supplied = set(str(service) for service in ranked_services)
    for service in completed:
        if service in supplied:
            evidence.append(service)
    count = len(evidence)
    scores = {
        service: float(count - position)
        for position, service in enumerate(evidence)
    }
    scores.update(
        {service: 0.0 for service in completed if service not in scores}
    )
    return scores


__all__ = [
    "EXCLUDED_SERVICES",
    "candidate_metric_columns",
    "candidate_services",
    "complete_service_ranking",
    "ranked_service_scores",
    "service_of",
]
