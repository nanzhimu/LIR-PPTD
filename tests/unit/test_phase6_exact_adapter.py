from __future__ import annotations

from pathlib import Path

from lir_pptd.experiments.phase6_exact_adapter import build_exact_task_input
from lir_pptd.experiments.phase6_generator import generate_stream

ROOT = Path(__file__).parents[2]


def test_exact_adapter_builds_frozen_input() -> None:
    stream = generate_stream(ROOT, "validation", 1001, "synthetic_num_small")
    task = stream["tasks"][0]
    candidate = {
        "K": 10,
        "lambda_tau": "1/10",
        "epsilon_c": "1/1024",
        "kappa": "2",
        "eta": "1/10",
        "c0": "1/2",
    }
    exact = build_exact_task_input(task, candidate)
    assert exact.task_id == task["task_id"]
    assert exact.task_kind == task["modality"]
    assert exact.participant_ids == tuple(task["participant_ids"])
    assert exact.K == 10
    assert exact.tau == "1/2"
    assert exact.report_lower == "0"
    assert exact.report_upper == "1"
    assert exact.reputations[exact.participant_ids[0]] == "1/2"
    assert exact.epochs[exact.participant_ids[0]] == 0
