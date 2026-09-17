from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .calibration_anchored import CalibrationSchedule, run_calibration_task
from .phase6_exact_adapter import LongitudinalState
from .phase_r1.methods import StatefulPrediction, run_lir_task
from .phase_r1.real_data_loader import RealTask


ALGORITHM_ID = "calibration_anchored_lir_pptd_v2_h1"


@dataclass(frozen=True)
class CalibrationAnchoredH1SequenceResult:
    """Sequence result for the recovered calibration-only persistence semantics.

    Ordinary tasks run the unchanged LIR-PPTD within-task refinement and may
    compute a counterfactual ``next_state`` as part of ``StatefulPrediction``.
    That transition is deliberately *not* committed. Persistent reputation is
    advanced only by trusted calibration rounds.
    """

    evaluated_predictions: tuple[StatefulPrediction, ...]
    calibration_results: tuple[Any, ...]
    evaluation_task_ids: tuple[str, ...]
    calibration_task_ids: tuple[str, ...]
    final_state: LongitudinalState
    ordinary_persistent_commit_count: int
    calibration_persistent_commit_count: int


def run_calibration_anchored_h1_sequence(
    tasks: Sequence[RealTask],
    candidate_config: Mapping[str, Any],
    initial_state: LongitudinalState,
    *,
    schedule: CalibrationSchedule = CalibrationSchedule(period=20),
    dps: int = 80,
) -> CalibrationAnchoredH1SequenceResult:
    """Run frozen H1 semantics: only calibration rounds persist reputation.

    The function intentionally has no runtime ``ordinary_state_commit`` switch.
    H1 is a fixed candidate, not a tunable semantic axis.
    """

    state = initial_state
    predictions: list[StatefulPrediction] = []
    calibration_results: list[Any] = []
    eval_ids: list[str] = []
    cal_ids: list[str] = []
    calibration_commits = 0

    for one_based_round, task in enumerate(tasks, start=1):
        if schedule.is_calibration_round(one_based_round):
            result, state = run_calibration_task(task, candidate_config, state, dps=dps)
            calibration_results.append(result)
            cal_ids.append(task.task_id)
            calibration_commits += 1
            continue

        pred = run_lir_task(task, candidate_config, state, ablation="full", dps=dps)
        predictions.append(pred)
        eval_ids.append(task.task_id)
        # H1 invariant: do NOT assign ``state = pred.next_state`` here.

    return CalibrationAnchoredH1SequenceResult(
        evaluated_predictions=tuple(predictions),
        calibration_results=tuple(calibration_results),
        evaluation_task_ids=tuple(eval_ids),
        calibration_task_ids=tuple(cal_ids),
        final_state=state,
        ordinary_persistent_commit_count=0,
        calibration_persistent_commit_count=calibration_commits,
    )
