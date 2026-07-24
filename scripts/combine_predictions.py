#!/usr/bin/env python3
"""Combine independently checkpointed baseline and QARCA prediction tables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


KEY = ["system", "case_id", "mechanism", "rate", "replicate", "method"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("qarca", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    qarca = pd.read_csv(args.qarca)
    overlap = set(baseline["method"]) & set(qarca["method"])
    if overlap:
        raise SystemExit(f"Method sets overlap: {sorted(overlap)}")
    combined = pd.concat((baseline, qarca), ignore_index=True, sort=False)
    duplicates = combined.duplicated(KEY, keep=False)
    if duplicates.any():
        raise SystemExit(f"Duplicate prediction keys: {int(duplicates.sum())} rows")
    combined = combined.sort_values(KEY, kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, args.output)
    print(
        f"Wrote {len(combined):,} rows from {combined['method'].nunique()} methods "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
