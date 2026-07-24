import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from trailrca.evaluation import observability_summary
from trailrca.methods import localize
from trailrca.quality_ranker import extract_service_features

ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(80, 4))
    values[40:, 0] += 5.0
    return pd.DataFrame(
        {
            "time": np.arange(80),
            "root_cpu": values[:, 0],
            "root_mem": values[:, 1],
            "peer_cpu": values[:, 2],
            "peer_mem": values[:, 3],
        }
    )


def load_runner():
    path = ROOT / "scripts" / "run_qarca_experiments.py"
    spec = importlib.util.spec_from_file_location("qarca_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_source(tmp_path: Path, system: str, name: str) -> Path:
    directory = tmp_path / f"RE1-{system}" / name / "1"
    directory.mkdir(parents=True)
    source = directory / "data.csv"
    source.write_text("time,service_cpu\n0,1\n", encoding="utf-8")
    (directory / "inject_time.txt").write_text("1\n", encoding="utf-8")
    return source


def test_config_and_fold_fingerprints_cover_scenarios_and_training_set(tmp_path) -> None:
    runner = load_runner()
    first = fake_source(tmp_path, "A", "root_cpu")
    second = fake_source(tmp_path, "B", "root_cpu")
    mechanisms = ("point", "burst", "stream", "incident")

    base = runner._training_config_fingerprint(mechanisms, (0.1, 0.3, 0.5), 300)
    changed_rate = runner._training_config_fingerprint(mechanisms, (0.1, 0.2, 0.5), 300)
    assert base != changed_rate

    fold = runner._fold_fingerprint(base, "C", [first], mechanisms, (0.1,), 10)
    changed_cases = runner._fold_fingerprint(
        base, "C", [first, second], mechanisms, (0.1,), 10
    )
    changed_replicates = runner._fold_fingerprint(
        base, "C", [first], mechanisms, (0.1,), 9
    )
    assert len({fold, changed_cases, changed_replicates}) == 3


def test_source_fingerprint_covers_injection_metadata(tmp_path) -> None:
    runner = load_runner()
    source = fake_source(tmp_path, "A", "root_cpu")
    before = runner._source_fingerprint(source)
    source.with_name("inject_time.txt").write_text("2\n", encoding="utf-8")
    after = runner._source_fingerprint(source)
    assert before != after


def test_all_localizers_retain_same_mask_invariant_candidate_universe() -> None:
    frame = synthetic_frame()
    frame["PassthroughCluster_load"] = np.arange(len(frame), dtype=float)
    frame.loc[:, ["root_cpu", "root_mem"]] = np.nan
    expected = set(extract_service_features(frame, n_pre=40).index)

    assert expected == {"peer", "root"}
    for method in ("nsigma", "baro", "median_shift"):
        result = localize(method, frame, n_pre=40)
        assert set(result["service_ranks"]) == expected
        assert set(result["service_scores"]) == expected


def test_observability_requires_pre_and_post_evidence() -> None:
    frame = synthetic_frame()
    frame.loc[:39, "root_cpu"] = np.nan
    frame.loc[40:, "root_mem"] = np.nan
    summary = observability_summary(frame, 40, "root", minimum_samples=3)

    assert summary["root_streams_total"] == 2
    assert summary["root_streams_available"] == 2
    assert summary["root_streams_scorable"] == 0
    assert summary["root_streams_pre_available"] == 1
    assert summary["root_streams_post_available"] == 1
