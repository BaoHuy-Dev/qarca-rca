"""Leakage-safe supervised ranking from anomaly and telemetry-quality evidence.

The feature extractor deliberately has no label, fault, case, or system argument.
Service names are retained only as row identifiers so predictions can be mapped
back to candidates; they are never columns presented to the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .candidates import candidate_metric_columns, service_of
from .data import FailureCase


RANDOM_STATE = 1729
METRIC_KINDS = ("cpu", "memory", "load", "latency", "error", "other")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

ANOMALY_FEATURES = (
    "standard_max",
    "robust_max",
    "standard_median_shift",
    "robust_median_shift",
    "robust_quantile_shift_mean",
    "robust_quantile_shift_max",
)
QUALITY_FEATURES = (
    "pre_availability",
    "post_availability",
    "joint_availability",
    "pre_longest_gap",
    "post_longest_gap",
    "block_coverage_mean",
    "block_coverage_min",
    "block_valid_fraction",
    "block_longest_valid_run",
    "reference_valid",
    "post_valid",
    "evidence_quality",
)


def _feature_columns() -> tuple[str, ...]:
    columns = [
        "metric_count_log1p",
        "metric_share",
        "metric_kind_fraction",
    ]
    for feature in ANOMALY_FEATURES:
        columns.extend(
            (f"{feature}_max", f"{feature}_mean", f"{feature}_median")
        )
    for feature in QUALITY_FEATURES:
        columns.extend((f"{feature}_mean", f"{feature}_min"))
    for kind in METRIC_KINDS:
        columns.extend((f"kind_{kind}_present", f"kind_{kind}_metric_fraction"))
        columns.extend(f"kind_{kind}_{feature}_max" for feature in ANOMALY_FEATURES)
        columns.extend(
            (
                f"kind_{kind}_post_availability_mean",
                f"kind_{kind}_evidence_quality_mean",
            )
        )
    relative_sources = (
        "standard_max_max",
        "robust_max_max",
        "standard_median_shift_max",
        "robust_median_shift_max",
        "robust_quantile_shift_mean_max",
        "robust_quantile_shift_max_max",
        "post_availability_mean",
        "block_longest_valid_run_mean",
        "evidence_quality_mean",
    )
    columns.extend(f"relative_{source}_percentile" for source in relative_sources)
    return tuple(columns)


FEATURE_COLUMNS = _feature_columns()

# Fixed ablation schema.  Every feature that directly measures missingness,
# coverage, continuity, or evidence validity is excluded; anomaly summaries and
# system-independent structural summaries remain.  Keeping this list public
# makes the manuscript's quality-feature ablation mechanically auditable.
QUALITY_COLUMN_MARKERS = (
    "availability",
    "longest_gap",
    "block_coverage",
    "block_valid",
    "block_longest",
    "reference_valid",
    "post_valid",
    "evidence_quality",
)
NO_QUALITY_FEATURE_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if not any(marker in column for marker in QUALITY_COLUMN_MARKERS)
)
STRUCTURAL_FEATURE_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if column in {"metric_count_log1p", "metric_share", "metric_kind_fraction"}
    or column.endswith("_present")
    or column.endswith("_metric_fraction")
)
QUALITY_ONLY_FEATURE_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if any(marker in column for marker in QUALITY_COLUMN_MARKERS)
)


def metric_service_and_kind(metric: str) -> tuple[str, str]:
    """Parse a metric name without encoding the service identity as a feature."""

    if "_" in metric:
        service, metric_type = service_of(metric), metric.split("_", 1)[1]
    else:
        service, metric_type = metric, "other"

    tokens = set(filter(None, re.split(r"[^a-z0-9]+", metric_type.lower())))
    if tokens & {"cpu", "processor"}:
        kind = "cpu"
    elif tokens & {"mem", "memory", "rss", "heap"}:
        kind = "memory"
    elif tokens & {"load", "throughput", "qps", "rate"}:
        kind = "load"
    elif tokens & {"latency", "delay", "duration", "response"}:
        kind = "latency"
    elif tokens & {"error", "errors", "failure", "failures", "loss"}:
        kind = "error"
    else:
        kind = "other"
    return service, kind


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _longest_missing_fraction(observed: np.ndarray) -> float:
    if observed.size == 0:
        return 1.0
    return _longest_true_run(~observed) / observed.size


def _scale(values: np.ndarray, robust: bool) -> tuple[float, float] | None:
    minimum = 3 if robust else 2
    if values.size < minimum:
        return None
    location = float(np.median(values) if robust else np.mean(values))
    if robust:
        mad = 1.4826 * float(np.median(np.abs(values - location)))
        q25, q75 = np.quantile(values, (0.25, 0.75))
        dispersion = max(mad, float((q75 - q25) / 1.349))
    else:
        dispersion = float(np.std(values, ddof=1))
    floor = 1e-9 * max(1.0, abs(location))
    return location, max(dispersion, floor)


def _bounded_log_effect(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.log1p(np.clip(abs(value), 0.0, 1e6)))


def _metric_features(
    values: np.ndarray,
    n_pre: int,
    blocks: int,
) -> dict[str, float]:
    before = values[:n_pre]
    after = values[n_pre:]
    before_mask = np.isfinite(before)
    after_mask = np.isfinite(after)
    observed_before = before[before_mask]
    observed_after = after[after_mask]

    pre_availability = float(before_mask.mean())
    post_availability = float(after_mask.mean())
    if pre_availability + post_availability:
        joint_availability = (
            2.0
            * pre_availability
            * post_availability
            / (pre_availability + post_availability)
        )
    else:
        joint_availability = 0.0

    block_masks = np.array_split(after_mask, min(blocks, after_mask.size))
    block_coverages = np.asarray(
        [float(mask.mean()) for mask in block_masks], dtype=float
    )
    valid_blocks = block_coverages >= 0.5
    block_valid_fraction = float(valid_blocks.mean())
    block_longest_valid_run = (
        _longest_true_run(valid_blocks) / valid_blocks.size
        if valid_blocks.size
        else 0.0
    )

    standard = _scale(observed_before, robust=False)
    robust = _scale(observed_before, robust=True)
    reference_valid = float(standard is not None and robust is not None)
    post_valid = float(observed_after.size > 0)

    effects = {feature: 0.0 for feature in ANOMALY_FEATURES}
    if observed_after.size and standard is not None:
        location, scale = standard
        effects["standard_max"] = _bounded_log_effect(
            np.max(np.abs(observed_after - location)) / scale
        )
        effects["standard_median_shift"] = _bounded_log_effect(
            (float(np.median(observed_after)) - location) / scale
        )
    if observed_after.size and robust is not None:
        location, scale = robust
        effects["robust_max"] = _bounded_log_effect(
            np.max(np.abs(observed_after - location)) / scale
        )
        effects["robust_median_shift"] = _bounded_log_effect(
            (float(np.median(observed_after)) - location) / scale
        )
        before_quantiles = np.quantile(observed_before, QUANTILES)
        after_quantiles = np.quantile(observed_after, QUANTILES)
        quantile_effects = np.abs(after_quantiles - before_quantiles) / scale
        effects["robust_quantile_shift_mean"] = _bounded_log_effect(
            float(np.mean(quantile_effects))
        )
        effects["robust_quantile_shift_max"] = _bounded_log_effect(
            float(np.max(quantile_effects))
        )

    evidence_quality = (
        joint_availability * block_valid_fraction * block_longest_valid_run
    )
    effects.update(
        {
            "pre_availability": pre_availability,
            "post_availability": post_availability,
            "joint_availability": joint_availability,
            "pre_longest_gap": _longest_missing_fraction(before_mask),
            "post_longest_gap": _longest_missing_fraction(after_mask),
            "block_coverage_mean": float(block_coverages.mean()),
            "block_coverage_min": float(block_coverages.min()),
            "block_valid_fraction": block_valid_fraction,
            "block_longest_valid_run": block_longest_valid_run,
            "reference_valid": reference_valid,
            "post_valid": post_valid,
            "evidence_quality": evidence_quality,
        }
    )
    return effects


def _relative_percentile(values: pd.Series) -> pd.Series:
    """Return an empirical percentile with neutral 0.5 for a singleton."""

    if len(values) == 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(values) - 1.0)


def extract_service_features(
    frame: pd.DataFrame,
    n_pre: int,
    *,
    blocks: int = 10,
) -> pd.DataFrame:
    """Build fixed-schema, label-free feature rows for all candidate services.

    Missing cells remain missing while each metric is summarized; post-incident
    values are never used to estimate a pre-incident location or scale. Service
    strings are used only to group metric rows and form the returned index.
    """

    if not isinstance(n_pre, (int, np.integer)) or not 0 < n_pre < len(frame):
        raise ValueError("n_pre must be an integer strictly inside the frame")
    if not isinstance(blocks, int) or blocks < 1:
        raise ValueError("blocks must be a positive integer")
    if "time" not in frame.columns:
        raise ValueError("frame must contain a time column")
    if not frame.columns.is_unique:
        raise ValueError("frame columns must be unique")

    metric_columns = candidate_metric_columns(frame)
    if not metric_columns:
        raise ValueError("frame must contain at least one metric column")

    by_service: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for metric in metric_columns:
        service, kind = metric_service_and_kind(str(metric))
        try:
            values = frame[metric].to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric column {metric!r} must be numeric") from exc
        # Infinite telemetry is unavailable evidence, just like the data loader.
        values = np.where(np.isfinite(values), values, np.nan)
        by_service.setdefault(service, []).append(
            (kind, _metric_features(values, int(n_pre), blocks))
        )

    total_metrics = len(metric_columns)
    rows: dict[str, dict[str, float]] = {}
    for service, metric_rows in by_service.items():
        row: dict[str, float] = {
            "metric_count_log1p": float(np.log1p(len(metric_rows))),
            "metric_share": len(metric_rows) / total_metrics,
            "metric_kind_fraction": len({kind for kind, _ in metric_rows})
            / len(METRIC_KINDS),
        }
        for feature in ANOMALY_FEATURES:
            values = np.asarray([item[feature] for _, item in metric_rows])
            row[f"{feature}_max"] = float(values.max())
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_median"] = float(np.median(values))
        for feature in QUALITY_FEATURES:
            values = np.asarray([item[feature] for _, item in metric_rows])
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_min"] = float(values.min())

        for kind in METRIC_KINDS:
            kind_rows = [item for item_kind, item in metric_rows if item_kind == kind]
            row[f"kind_{kind}_present"] = float(bool(kind_rows))
            row[f"kind_{kind}_metric_fraction"] = len(kind_rows) / len(metric_rows)
            for feature in ANOMALY_FEATURES:
                row[f"kind_{kind}_{feature}_max"] = (
                    float(max(item[feature] for item in kind_rows))
                    if kind_rows
                    else 0.0
                )
            for feature in ("post_availability", "evidence_quality"):
                row[f"kind_{kind}_{feature}_mean"] = (
                    float(np.mean([item[feature] for item in kind_rows]))
                    if kind_rows
                    else 0.0
                )
        rows[service] = row

    features = pd.DataFrame.from_dict(rows, orient="index")
    features.index.name = "service"
    relative_sources = (
        "standard_max_max",
        "robust_max_max",
        "standard_median_shift_max",
        "robust_median_shift_max",
        "robust_quantile_shift_mean_max",
        "robust_quantile_shift_max_max",
        "post_availability_mean",
        "block_longest_valid_run_mean",
        "evidence_quality_mean",
    )
    for source in relative_sources:
        features[f"relative_{source}_percentile"] = _relative_percentile(
            features[source]
        )

    # Selecting a public constant both fixes training/prediction order and makes
    # accidental metadata columns impossible to pass to sklearn.
    features = features.loc[sorted(features.index), list(FEATURE_COLUMNS)]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise RuntimeError("feature extraction produced a non-finite value")
    return features


def extract_case_features(
    case: FailureCase,
    frame: pd.DataFrame | None = None,
    *,
    blocks: int = 10,
) -> pd.DataFrame:
    """Convenience wrapper that reads only ``frame`` and ``n_pre`` from a case."""

    observed_frame = case.frame if frame is None else frame
    return extract_service_features(observed_frame, case.n_pre, blocks=blocks)


@dataclass(frozen=True)
class TrainingMatrix:
    """Numeric model inputs plus non-feature bookkeeping for audit/tests."""

    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray
    incident_index: np.ndarray
    candidate_services: tuple[str, ...]


def _aligned_frames(
    cases: Sequence[FailureCase],
    frames: Sequence[pd.DataFrame] | None,
) -> list[pd.DataFrame | None]:
    if frames is None:
        return [None] * len(cases)
    if len(frames) != len(cases):
        raise ValueError("frames must have the same length as cases")
    return list(frames)


def build_training_matrix(
    cases: Sequence[FailureCase],
    frames: Sequence[pd.DataFrame] | None = None,
    *,
    blocks: int = 10,
) -> TrainingMatrix:
    """Create binary root/non-root rows with equal total weight per incident."""

    if not cases:
        raise ValueError("at least one training case is required")
    aligned_frames = _aligned_frames(cases, frames)
    feature_parts: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    incident_indices: list[np.ndarray] = []
    candidates: list[str] = []

    for incident_index, (case, observed_frame) in enumerate(
        zip(cases, aligned_frames, strict=True)
    ):
        features = extract_case_features(case, observed_frame, blocks=blocks)
        # The root label is consulted only here, never during feature extraction.
        y = (features.index.to_numpy(dtype=object) == case.service).astype(np.int8)
        if int(y.sum()) != 1:
            raise ValueError(
                "each incident root service must occur exactly once among candidates"
            )
        candidate_count = len(features)
        if candidate_count < 2:
            raise ValueError("each training incident needs at least two candidates")
        feature_parts.append(features.reset_index(drop=True))
        labels.append(y)
        # Give every incident total mass one and balance its single positive
        # against all of its negatives without a system-size-dependent class
        # weight inside the estimator.
        incident_weight = np.full(candidate_count, 0.5 / (candidate_count - 1))
        incident_weight[y == 1] = 0.5
        weights.append(incident_weight)
        incident_indices.append(np.full(candidate_count, incident_index, dtype=int))
        candidates.extend(str(service) for service in features.index)

    X = pd.concat(feature_parts, ignore_index=True).loc[:, list(FEATURE_COLUMNS)]
    return TrainingMatrix(
        X=X,
        y=np.concatenate(labels),
        sample_weight=np.concatenate(weights),
        incident_index=np.concatenate(incident_indices),
        candidate_services=tuple(candidates),
    )


def leave_one_system_out_indices(
    cases: Sequence[FailureCase], held_out_system: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable train/test positions for a leave-one-system-out fold."""

    train = np.asarray(
        [i for i, case in enumerate(cases) if case.system != held_out_system],
        dtype=int,
    )
    test = np.asarray(
        [i for i, case in enumerate(cases) if case.system == held_out_system],
        dtype=int,
    )
    if train.size == 0 or test.size == 0:
        raise ValueError("a LOSO fold requires non-empty training and test partitions")
    return train, test


class QualityAwareServiceRanker:
    """Deterministic logistic service ranker with a fixed feature schema."""

    def __init__(
        self,
        *,
        blocks: int = 10,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
    ) -> None:
        if not isinstance(blocks, int) or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        selected = tuple(feature_columns)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("feature_columns must be non-empty and unique")
        unknown = set(selected) - set(FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"unknown feature columns: {sorted(unknown)}")
        self.blocks = blocks
        self.feature_columns = selected
        self.estimator = Pipeline(
            steps=(
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.0,
                        solver="liblinear",
                        class_weight=None,
                        fit_intercept=True,
                        max_iter=2000,
                        tol=1e-7,
                        random_state=RANDOM_STATE,
                    ),
                ),
            )
        )

    def fit(
        self,
        cases: Sequence[FailureCase],
        frames: Sequence[pd.DataFrame] | None = None,
    ) -> QualityAwareServiceRanker:
        """Fit using root labels only for ``y`` and equal incident weights."""

        training = build_training_matrix(cases, frames, blocks=self.blocks)
        if np.unique(training.y).size != 2:
            raise ValueError("training data must contain root and non-root candidates")
        return self.fit_precomputed(
            training.X,
            training.y,
            training.sample_weight,
            n_training_incidents=len(cases),
        )

    def fit_precomputed(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray,
        *,
        n_training_incidents: int,
    ) -> QualityAwareServiceRanker:
        """Fit from label-free extracted features and separately built labels.

        This entry point lets the full LOSO runner cache expensive feature
        extraction without caching raw telemetry or passing metadata to sklearn.
        """

        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"precomputed matrix misses columns: {sorted(missing)}")
        labels = np.asarray(y, dtype=np.int8)
        weights = np.asarray(sample_weight, dtype=float)
        if len(X) != labels.size or labels.size != weights.size:
            raise ValueError("X, y, and sample_weight must have equal lengths")
        if np.unique(labels).size != 2:
            raise ValueError("training data must contain root and non-root candidates")
        selected = X.loc[:, list(self.feature_columns)]
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError("precomputed features must be finite")
        self.estimator.fit(
            selected,
            labels,
            scale__sample_weight=weights,
            classifier__sample_weight=weights,
        )
        self.n_training_incidents_ = int(n_training_incidents)
        self.feature_names_in_ = np.asarray(self.feature_columns, dtype=object)
        return self

    def predict_feature_scores(self, features: pd.DataFrame) -> pd.Series:
        """Score already extracted rows while preserving service identifiers."""

        check_is_fitted(self.estimator)
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"feature frame misses columns: {sorted(missing)}")
        selected = features.loc[:, list(self.feature_columns)]
        probabilities = self.estimator.predict_proba(selected)
        classes = self.estimator.named_steps["classifier"].classes_
        positive_column = int(np.flatnonzero(classes == 1)[0])
        return pd.Series(
            probabilities[:, positive_column],
            index=features.index,
            name="root_probability",
            dtype=float,
        )

    def predict_scores(self, frame: pd.DataFrame, n_pre: int) -> pd.Series:
        """Return root probabilities without accepting any ground-truth label."""

        check_is_fitted(self.estimator)
        features = extract_service_features(frame, n_pre, blocks=self.blocks)
        return self.predict_feature_scores(features)

    def rank(self, frame: pd.DataFrame, n_pre: int) -> list[str]:
        """Rank services by probability, resolving exact ties by name."""

        scores = self.predict_scores(frame, n_pre)
        return sorted(scores.index, key=lambda service: (-scores[service], service))

    def predict_case_scores(
        self, case: FailureCase, frame: pd.DataFrame | None = None
    ) -> pd.Series:
        """Case wrapper that intentionally ignores all label metadata."""

        observed_frame = case.frame if frame is None else frame
        return self.predict_scores(observed_frame, case.n_pre)

    def rank_case(
        self, case: FailureCase, frame: pd.DataFrame | None = None
    ) -> list[str]:
        """Case wrapper that intentionally ignores all label metadata."""

        observed_frame = case.frame if frame is None else frame
        return self.rank(observed_frame, case.n_pre)


def fit_quality_ranker(
    cases: Sequence[FailureCase],
    frames: Sequence[pd.DataFrame] | None = None,
    *,
    blocks: int = 10,
) -> QualityAwareServiceRanker:
    """Fit and return the fixed-hyperparameter ranker."""

    return QualityAwareServiceRanker(blocks=blocks).fit(cases, frames)


__all__ = [
    "FEATURE_COLUMNS",
    "METRIC_KINDS",
    "NO_QUALITY_FEATURE_COLUMNS",
    "QUALITY_ONLY_FEATURE_COLUMNS",
    "QUALITY_COLUMN_MARKERS",
    "STRUCTURAL_FEATURE_COLUMNS",
    "QualityAwareServiceRanker",
    "TrainingMatrix",
    "build_training_matrix",
    "extract_case_features",
    "extract_service_features",
    "fit_quality_ranker",
    "leave_one_system_out_indices",
    "metric_service_and_kind",
]
