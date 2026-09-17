from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from ..core.calibration import ExactCalibrationTaskInput, ExactCalibrationTaskResult, run_exact_calibration_update
from ..core.consistency_scale import DEFAULT_CONSISTENCY_SCALE_MODE, resolve_from_modality
from .phase6_exact_adapter import LongitudinalState
from .phase_r1.longitudinal_state import merge_participant_result
from .phase_r1.methods import StatefulPrediction, run_lir_task
from .phase_r1.real_data_loader import RealTask


@dataclass(frozen=True)
class CalibrationSchedule:
    period: int = 20
    one_based_offset: int = 0

    def is_calibration_round(self, one_based_round: int) -> bool:
        if self.period <= 0:
            raise ValueError("calibration period must be positive")
        if one_based_round <= 0:
            raise ValueError("round index must be one-based and positive")
        return (one_based_round - self.one_based_offset) % self.period == 0


def _fraction_text(value: Any) -> str:
    fraction = value if isinstance(value, Fraction) else Fraction(str(value))
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def run_calibration_task(
    task: RealTask,
    candidate_config: Mapping[str, Any],
    global_state: LongitudinalState,
    *,
    dps: int = 80,
) -> tuple[ExactCalibrationTaskResult, LongitudinalState]:
    reps = dict(global_state.reputations or {})
    epochs = dict(global_state.epochs or {})
    reports = {worker: tuple(task.reports[worker]) for worker in task.participant_ids}
    dimension = len(next(iter(reports.values())))
    scale = resolve_from_modality(
        task.modality,
        dimension,
        candidate_config["lambda_tau"],
        str(candidate_config.get("consistency_scale_mode", DEFAULT_CONSISTENCY_SCALE_MODE)),
    )
    tau = _fraction_text(scale.resolved_tau)

    exact = ExactCalibrationTaskInput(
        task_id=task.task_id,
        task_kind=task.modality,
        tau=tau,
        epsilon_c=candidate_config["epsilon_c"],
        kappa=candidate_config["kappa"],
        eta=candidate_config["eta"],
        reports=reports,
        reputations={w: reps.get(w, candidate_config["c0"]) for w in reps},
        epochs={w: int(epochs.get(w, 0)) for w in reps},
        participant_ids=task.participant_ids,
        trusted_truth=task.truth_vector,
    )
    result = run_exact_calibration_update(exact, dps=dps)
    state = LongitudinalState(
        reputations=dict(result.next_reputation),
        epochs=dict(result.next_epochs),
    )
    return result, state


@dataclass(frozen=True)
class CalibrationAnchoredSequenceResult:
    evaluated_predictions: tuple[StatefulPrediction, ...]
    calibration_results: tuple[ExactCalibrationTaskResult, ...]
    evaluation_task_ids: tuple[str, ...]
    calibration_task_ids: tuple[str, ...]
    final_state: LongitudinalState


def run_calibration_anchored_sequence(
    tasks: Sequence[RealTask],
    candidate_config: Mapping[str, Any],
    initial_state: LongitudinalState,
    *,
    schedule: CalibrationSchedule = CalibrationSchedule(),
    dps: int = 80,
) -> CalibrationAnchoredSequenceResult:
    state = initial_state
    predictions: list[StatefulPrediction] = []
    calibration_results: list[ExactCalibrationTaskResult] = []
    eval_ids: list[str] = []
    cal_ids: list[str] = []

    for one_based_round, task in enumerate(tasks, start=1):
        if schedule.is_calibration_round(one_based_round):
            result, state = run_calibration_task(task, candidate_config, state, dps=dps)
            calibration_results.append(result)
            cal_ids.append(task.task_id)
            continue

        pred = run_lir_task(task, candidate_config, state, ablation="full", dps=dps)
        predictions.append(pred)
        eval_ids.append(task.task_id)
        state = pred.next_state

    return CalibrationAnchoredSequenceResult(
        evaluated_predictions=tuple(predictions),
        calibration_results=tuple(calibration_results),
        evaluation_task_ids=tuple(eval_ids),
        calibration_task_ids=tuple(cal_ids),
        final_state=state,
    )
