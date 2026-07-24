"""Loading and validating RCAEval RE1 failure cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureCase:
    """One labelled failure window from RCAEval RE1."""

    case_id: str
    system: str
    service: str
    fault: str
    repetition: int
    inject_time: int
    frame: pd.DataFrame
    n_pre: int

    @property
    def metric_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != "time"]


def _dataset_root(csv_path: Path) -> Path:
    for parent in csv_path.parents:
        if parent.name.startswith("RE1-"):
            return parent
    raise ValueError(f"Cannot infer RE1 system from {csv_path}")


def discover_cases(data_root: str | Path) -> list[Path]:
    """Return data files in a deterministic case order."""

    root = Path(data_root)
    paths = sorted(root.rglob("data.csv"), key=lambda p: p.as_posix())
    if not paths:
        raise FileNotFoundError(f"No RCAEval data.csv files under {root}")
    return paths


def load_case(csv_path: str | Path, window: int = 300) -> FailureCase:
    """Load equal pre/post incident windows without imputing missing values."""

    path = Path(csv_path)
    label = path.parent.parent.name
    try:
        service, fault = label.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Unexpected RCAEval case label: {label}") from exc

    repetition = int(path.parent.name)
    inject_path = path.with_name("inject_time.txt")
    inject_time = int(inject_path.read_text(encoding="utf-8").strip())
    dataset_root = _dataset_root(path)
    system = dataset_root.name.removeprefix("RE1-")

    frame = pd.read_csv(path)
    # Pandas renames the duplicated time header in RCAEval to time.1.
    frame = frame.drop(columns=["time.1"], errors="ignore")
    if "time" not in frame:
        raise ValueError(f"Missing time column in {path}")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)

    pre = frame.loc[frame["time"] < inject_time].tail(window)
    post = frame.loc[frame["time"] >= inject_time].head(window)
    minimum = min(60, window)
    if len(pre) < minimum or len(post) < minimum:
        raise ValueError(
            f"{path} has too little telemetry: pre={len(pre)}, post={len(post)}"
        )
    sliced = pd.concat([pre, post], ignore_index=True)
    case_id = f"{system}/{label}/{repetition}"
    return FailureCase(
        case_id=case_id,
        system=system,
        service=service,
        fault=fault,
        repetition=repetition,
        inject_time=inject_time,
        frame=sliced,
        n_pre=len(pre),
    )
