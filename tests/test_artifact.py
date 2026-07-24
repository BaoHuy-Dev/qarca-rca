from pathlib import Path

import numpy as np
import pandas as pd

from trailrca.evaluation import evaluate_ranking, evaluate_service_scores
from trailrca.candidates import candidate_services, complete_service_ranking
from trailrca.methods import localize
from trailrca.missingness import case_seed, inject_missingness


def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pre = rng.normal(0.0, 0.1, size=(40, 4))
    post = rng.normal(0.0, 0.1, size=(40, 4))
    post[:, 0] += 5.0
    values = np.vstack([pre, post])
    return pd.DataFrame(
        {
            "time": np.arange(80),
            "root_cpu": values[:, 0],
            "root_mem": values[:, 1],
            "peer_cpu": values[:, 2],
            "peer_mem": values[:, 3],
        }
    )


def test_trail_localizes_persistent_shift() -> None:
    result = localize("trail", synthetic_frame(), n_pre=40)
    assert result["service_ranks"][0] == "root"


def test_missingness_is_deterministic_and_preserves_time() -> None:
    frame = synthetic_frame()
    seed = case_seed("OB/root_cpu/1", "point", 0.3, 2)
    first, first_rate = inject_missingness(frame, "point", 0.3, seed)
    second, second_rate = inject_missingness(frame, "point", 0.3, seed)
    pd.testing.assert_frame_equal(first, second)
    assert first["time"].notna().all()
    assert first_rate == second_rate
    assert 0.2 < first_rate < 0.4


def test_structured_loss_budget() -> None:
    frame = synthetic_frame()
    for mechanism in ("burst", "stream", "incident"):
        masked, realized = inject_missingness(
            frame, mechanism, 0.25, seed=11, n_pre=40
        )
        assert masked.drop(columns="time").isna().any().any()
        assert abs(realized - 0.25) < 1e-12


def test_stream_loss_ignores_natively_empty_channel() -> None:
    frame = synthetic_frame()
    frame["empty_cpu"] = np.nan
    masked, _ = inject_missingness(frame, "stream", 0.25, seed=19, n_pre=40)

    originally_observed = ["root_cpu", "root_mem", "peer_cpu", "peer_mem"]
    removed = [column for column in originally_observed if masked[column].isna().all()]
    assert len(removed) == 1
    assert masked["empty_cpu"].isna().all()


def test_ranking_metrics_penalize_absent_root() -> None:
    rank, rr, hit1, hit3, hit5 = evaluate_ranking("missing", ["a", "b"])
    assert (rank, rr, hit1, hit3, hit5) == (3, 0.0, 0, 0, 0)


def test_candidate_universe_is_mask_invariant_and_completed() -> None:
    frame = synthetic_frame()
    frame["PassthroughCluster_load"] = 1.0
    masked = frame.copy()
    masked.loc[:, ["root_cpu", "root_mem"]] = np.nan

    assert candidate_services(frame) == ["peer", "root"]
    assert candidate_services(masked) == candidate_services(frame)
    assert complete_service_ranking(["peer"], masked) == ["peer", "root"]


def test_score_ties_use_worst_rank_not_service_name() -> None:
    first = evaluate_service_scores("alpha", {"alpha": 0.0, "zulu": 0.0})
    second = evaluate_service_scores("zulu", {"alpha": 0.0, "zulu": 0.0})
    assert first == second == (2, 0.5, 0, 1, 1)
