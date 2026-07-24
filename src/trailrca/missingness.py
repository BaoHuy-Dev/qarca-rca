"""Deterministic telemetry-loss mechanisms used by the experiment."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


MECHANISMS = ("none", "point", "burst", "stream", "incident")


def case_seed(case_id: str, mechanism: str, rate: float, replicate: int) -> int:
    """Create an order-independent 64-bit seed for a case/scenario."""

    payload = f"{case_id}|{mechanism}|{rate:.6f}|{replicate}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def inject_missingness(
    frame: pd.DataFrame,
    mechanism: str,
    rate: float,
    seed: int,
    n_pre: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Mask telemetry values and return the realized loss over observed cells.

    ``point`` masks a fixed cell budget, ``burst`` masks one contiguous interval
    per stream, ``stream`` removes whole metric streams, and ``incident`` masks
    post-incident cells with probability proportional to their hidden robust
    deviation. The corruption generator may see hidden values; RCA methods never
    receive them. Time is never masked.
    """

    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")

    out = frame.copy(deep=True)
    cols = [c for c in out.columns if c != "time"]
    values = out[cols].to_numpy(dtype=float, copy=True)
    originally_observed = np.isfinite(values)
    rng = np.random.default_rng(seed)

    if mechanism == "none" or rate == 0.0:
        return out, 0.0

    n_rows, n_cols = values.shape
    mask = np.zeros_like(values, dtype=bool)
    def exact_uniform(target: np.ndarray, eligible: np.ndarray, fraction: float) -> None:
        indexes = np.flatnonzero(eligible)
        count = int(round(fraction * len(indexes)))
        if count:
            target.flat[rng.choice(indexes, size=count, replace=False)] = True

    if mechanism == "point":
        if n_pre is None:
            exact_uniform(mask, originally_observed, rate)
        else:
            exact_uniform(mask[:n_pre], originally_observed[:n_pre], rate)
            exact_uniform(mask[n_pre:], originally_observed[n_pre:], rate)
    elif mechanism == "burst":
        block = max(1, min(n_rows, int(round(rate * n_rows))))
        # A synthetic outage is applied only to channels with at least one
        # originally observed cell in the fixed window. Native RE1 gaps remain
        # untouched and are accounted for separately.
        eligible_columns = np.flatnonzero(originally_observed.any(axis=0))
        for j in eligible_columns:
            start = int(rng.integers(0, n_rows - block + 1))
            mask[start : start + block, j] = True
    elif mechanism == "stream":
        eligible_columns = np.flatnonzero(originally_observed.any(axis=0))
        # The nominal stream-loss rate is a fraction of channels that carried
        # evidence in the original window. At least one such channel is retained
        # whenever possible; for a single-channel frame the added loss is zero.
        if eligible_columns.size > 1:
            count = max(1, int(round(rate * eligible_columns.size)))
            count = min(eligible_columns.size - 1, count)
            selected = rng.choice(eligible_columns, size=count, replace=False)
            mask[:, selected] = True
    elif mechanism == "incident":
        if n_pre is None:
            raise ValueError("incident missingness requires n_pre")
        # Keep the nominal budget balanced across the two periods. Pre-incident
        # loss is uniform; post-incident loss preferentially hides large shifts.
        exact_uniform(mask[:n_pre], originally_observed[:n_pre], rate)
        post = values[n_pre:]
        post_observed = originally_observed[n_pre:]
        weights = np.zeros_like(post, dtype=float)
        for j in range(n_cols):
            reference = values[:n_pre, j]
            reference = reference[np.isfinite(reference)]
            if reference.size < 3:
                weights[:, j] = 1.0
                continue
            location = float(np.median(reference))
            mad = float(np.median(np.abs(reference - location))) * 1.4826
            q25, q75 = np.quantile(reference, [0.25, 0.75])
            scale = max(mad, float((q75 - q25) / 1.349), 1e-9 * max(1.0, abs(location)))
            z = np.abs(post[:, j] - location) / scale
            weights[:, j] = np.exp(2.0 * np.clip(z, 0.0, 5.0))
        eligible_indexes = np.flatnonzero(post_observed)
        count = int(round(rate * len(eligible_indexes)))
        if count:
            probabilities = weights.flat[eligible_indexes]
            probabilities = probabilities / probabilities.sum()
            chosen = rng.choice(
                eligible_indexes, size=count, replace=False, p=probabilities
            )
            mask[n_pre:].flat[chosen] = True

    effective_mask = mask & originally_observed
    values[effective_mask] = np.nan
    # Some RCAEval streams (notably Train Ticket counters) are parsed as
    # integer columns.  ``.loc`` assignment attempts an in-place dtype-preserving
    # write and therefore rejects the NaNs introduced above.  Reconstruct the
    # metric block from the floating-point array so missing values are represented
    # faithfully, then restore the original column order.
    metric_frame = pd.DataFrame(values, index=out.index, columns=cols)
    out = pd.concat((out.drop(columns=cols), metric_frame), axis=1)
    out = out.loc[:, frame.columns]
    denominator = int(originally_observed.sum())
    realized = float(effective_mask.sum() / denominator) if denominator else 0.0
    return out, realized


def causal_fill(frame: pd.DataFrame, n_pre: int) -> pd.DataFrame:
    """Prepare incomplete data for baselines without post-to-pre leakage.

    Forward filling respects time. Values still missing at the beginning are
    filled with a median estimated only from the pre-incident window. A fully
    absent stream becomes constant zero and is subsequently ignored.
    """

    out = frame.copy(deep=True)
    cols = [c for c in out.columns if c != "time"]
    numeric = out[cols].ffill()
    pre_medians = numeric.iloc[:n_pre].median(axis=0, skipna=True).fillna(0.0)
    numeric = numeric.fillna(pre_medians).fillna(0.0)
    out.loc[:, cols] = numeric
    return out
