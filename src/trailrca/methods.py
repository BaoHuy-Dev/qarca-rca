"""Root-cause localization methods and the proposed TRAIL ranker."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import pandas as pd
from baro.root_cause_analysis import nsigma, robust_scorer

from .candidates import (
    candidate_metric_columns,
    complete_service_ranking,
    ranked_service_scores,
    service_of,
)
from .missingness import causal_fill


METHODS = ("nsigma", "baro", "median_shift", "trail_no_reliability", "trail")


def unique_services(metric_ranks: list[str]) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for metric in metric_ranks:
        service = service_of(metric)
        if service not in seen:
            seen.add(service)
            ranked.append(service)
    return ranked


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in candidate_metric_columns(frame)
        if not c.endswith("_error")
    ]


def _robust_scale(reference: np.ndarray) -> tuple[float, float]:
    reference = reference[np.isfinite(reference)]
    if reference.size < 3:
        return np.nan, np.nan
    location = float(np.median(reference))
    mad = float(np.median(np.abs(reference - location))) * 1.4826
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    floor = 1e-9 * max(1.0, abs(location))
    return location, max(mad, iqr_scale, floor)


def median_shift(frame: pd.DataFrame, n_pre: int) -> tuple[list[str], dict[str, float]]:
    """Complete-case robust location-shift baseline."""

    scores: dict[str, float] = {}
    for col in _candidate_columns(frame):
        before = frame[col].iloc[:n_pre].to_numpy(dtype=float)
        after = frame[col].iloc[n_pre:].to_numpy(dtype=float)
        location, scale = _robust_scale(before)
        observed_after = after[np.isfinite(after)]
        if not np.isfinite(scale) or observed_after.size < 3:
            continue
        effect = abs(float(np.median(observed_after)) - location) / scale
        scores[col] = float(np.log1p(min(effect, 1e6)))
    ranks = sorted(scores, key=lambda c: (-scores[c], c))
    return ranks, scores


def trail(
    frame: pd.DataFrame,
    n_pre: int,
    reliability: bool = True,
    blocks: int = 10,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> tuple[list[str], dict[str, float]]:
    """Telemetry-Reliability-Aware Incident Localization.

    TRAIL compares robust distribution summaries in contiguous post-incident
    blocks. It retains the observation mask, discounts fragmented evidence, and
    pools only the two strongest indicators per service. No label, fault type,
    imputation, topology, or training data is used.
    """

    columns = _candidate_columns(frame)
    post_length = len(frame) - n_pre
    block_edges = np.linspace(0, post_length, blocks + 1, dtype=int)
    minimum_pre = max(10, int(0.10 * n_pre))
    metric_score: dict[str, float] = {}
    metric_reliability: dict[str, float] = {}

    def longest_gap(observed: np.ndarray) -> int:
        longest = current = 0
        for present in observed:
            if present:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    for col in columns:
        before = frame[col].iloc[:n_pre].to_numpy(dtype=float)
        after = frame[col].iloc[n_pre:].to_numpy(dtype=float)
        before_mask = np.isfinite(before)
        after_mask = np.isfinite(after)
        observed_before = before[before_mask]
        if observed_before.size < minimum_pre:
            continue
        _, scale = _robust_scale(observed_before)
        if not np.isfinite(scale):
            continue
        reference_quantiles = np.quantile(observed_before, quantiles)

        block_effects: list[float] = []
        for start, stop in zip(block_edges[:-1], block_edges[1:]):
            block = after[start:stop]
            observed_block = block[np.isfinite(block)]
            minimum_block = max(3, int(np.ceil(0.20 * len(block))))
            if observed_block.size < minimum_block:
                continue
            shifted_quantiles = np.quantile(observed_block, quantiles)
            distance = float(np.mean(np.abs(shifted_quantiles - reference_quantiles)))
            block_effects.append(min(distance / scale, 1e6))

        if not block_effects:
            continue
        effects = np.asarray(block_effects, dtype=float)
        q25, q75 = np.quantile(effects, [0.25, 0.75])
        # A lower-tail summary favors sustained changes over isolated spikes.
        anomaly = max(0.0, float(np.median(effects) - 0.25 * (q75 - q25)))
        pre_coverage = float(before_mask.mean())
        post_coverage = float(after_mask.mean())
        valid_fraction = len(block_effects) / blocks
        continuity = np.exp(-longest_gap(after_mask) / max(1, post_length))
        evidence_reliability = (
            np.sqrt(pre_coverage * post_coverage) * valid_fraction * continuity
        )
        metric_reliability[col] = float(evidence_reliability)
        weight = evidence_reliability if reliability else 1.0
        metric_score[col] = float(np.log1p(anomaly) * weight)

    service_metrics: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for metric, score in metric_score.items():
        service_metrics[service_of(metric)].append((metric, score))

    service_score: dict[str, float] = {}
    service_evidence: dict[str, str] = {}
    for service, entries in service_metrics.items():
        entries.sort(key=lambda item: (-item[1], item[0]))
        selected = entries[:2]
        if reliability:
            weights = np.asarray(
                [max(metric_reliability.get(metric, 0.0), 1e-12) for metric, _ in selected]
            )
            values = np.asarray([score for _, score in selected])
            pooled = float(np.average(values, weights=weights))
            availability = 1.0 - np.exp(-float(weights.sum()))
            service_score[service] = pooled * availability
        else:
            top = selected[0][1]
            second = selected[1][1] if len(selected) > 1 else 0.0
            service_score[service] = top + 0.25 * second
        service_evidence[service] = entries[0][0]

    service_ranks = sorted(service_score, key=lambda s: (-service_score[s], s))
    # Return metric-like evidence ranks too, retaining service ordering.
    evidence_ranks = [service_evidence[s] for s in service_ranks]
    return evidence_ranks, dict(metric_score)


def localize(method: str, frame: pd.DataFrame, n_pre: int) -> dict[str, object]:
    """Run one method and return service/metric rankings plus elapsed time."""

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    started = perf_counter()

    if method in {"nsigma", "baro"}:
        columns = ["time", *candidate_metric_columns(frame)]
        prepared = causal_fill(frame.loc[:, columns], n_pre)
        # BARO accepts an anomaly index; this gives every method the same known
        # incident boundary and isolates localization from detection errors.
        fn = nsigma if method == "nsigma" else robust_scorer
        output = fn(prepared, anomalies=[n_pre])
        metric_ranks = list(output["ranks"])
        evidence_ranks = unique_services(metric_ranks)
        service_scores = ranked_service_scores(evidence_ranks, frame)
        service_ranks = complete_service_ranking(evidence_ranks, frame)
        scores: dict[str, float] = {}
    elif method == "median_shift":
        metric_ranks, scores = median_shift(frame, n_pre)
        evidence_ranks = unique_services(metric_ranks)
        service_scores = ranked_service_scores(evidence_ranks, frame)
        service_ranks = complete_service_ranking(evidence_ranks, frame)
    else:
        metric_ranks, scores = trail(
            frame, n_pre, reliability=(method == "trail")
        )
        evidence_ranks = unique_services(metric_ranks)
        service_scores = ranked_service_scores(evidence_ranks, frame)
        service_ranks = complete_service_ranking(evidence_ranks, frame)

    elapsed_ms = (perf_counter() - started) * 1000.0
    return {
        "metric_ranks": metric_ranks,
        "service_ranks": service_ranks,
        "service_scores": service_scores,
        "scores": scores,
        "elapsed_ms": elapsed_ms,
    }
