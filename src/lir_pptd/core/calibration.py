from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..identifiers import WorkerID
from .exact import ExactTaskInput, _mp, _prepare_exact_task, decimal_string
from .types import ExactReputationTransition


@dataclass(frozen=True)
class ExactCalibrationTaskInput:
    """Trusted-calibration task for calibration-anchored LIR-PPTD.

    Reports and reputation state use the same exact-arithmetic conventions as
    ``ExactTaskInput``. ``trusted_truth`` is an authenticated known truth and is
    never inferred from worker reports.
    """

    task_id: str
    task_kind: Literal["numerical", "categorical"] | str
    tau: object
    epsilon_c: object
    kappa: object
    eta: object
    reports: dict[str, tuple[object, ...]]
    reputations: dict[str, object]
    epochs: dict[str, int]
    participant_ids: tuple[str, ...]
    trusted_truth: tuple[object, ...]
    report_lower: object = "0"
    report_upper: object = "1"


@dataclass(frozen=True)
class ExactCalibrationEvidence:
    worker_id: WorkerID
    distance_to_trusted_truth: str
    evidence: str


@dataclass(frozen=True)
class ExactCalibrationTaskResult:
    success: bool
    task_id: str
    task_kind: str
    trusted_truth: tuple[str, ...]
    participant_ids: tuple[WorkerID, ...]
    evidence: tuple[ExactCalibrationEvidence, ...]
    reputation_transitions: tuple[ExactReputationTransition, ...]
    next_reputation: dict[WorkerID, str]
    next_epochs: dict[WorkerID, int]
    released_truth_estimate: None = None
    evaluation_eligible: bool = False


def _validate_trusted_truth(prepared, task: ExactCalibrationTaskInput):
    truth = tuple(_mp(prepared.ctx, x) for x in task.trusted_truth)
    if len(truth) != prepared.dimension:
        raise ValueError("trusted truth dimension mismatch")
    if task.task_kind == "numerical":
        if any(x < prepared.lower or x > prepared.upper for x in truth):
            raise ValueError("trusted numerical truth outside configured domain")
    elif task.task_kind == "categorical":
        if any(x not in (0, 1) for x in truth) or sum(truth) != 1:
            raise ValueError("trusted categorical truth must be one-hot")
    else:
        raise ValueError("unknown task kind")
    return truth


def run_exact_calibration_update(
    task: ExactCalibrationTaskInput,
    *,
    dps: int = 80,
) -> ExactCalibrationTaskResult:
    """Update reputation once from an authenticated trusted truth.

    Calibration tasks do not run the within-task truth iteration and do not
    release an inferred truth. They only compute report-to-trusted-truth
    evidence and apply the same bounded reputation transition as ordinary
    LIR-PPTD tasks.
    """

    proxy = ExactTaskInput(
        task_id=task.task_id,
        task_kind=task.task_kind,
        K=1,  # validation-only; no truth iteration is executed below
        tau=task.tau,
        epsilon_c=task.epsilon_c,
        kappa=task.kappa,
        eta=task.eta,
        reports=task.reports,
        reputations=task.reputations,
        epochs=task.epochs,
        participant_ids=task.participant_ids,
        report_lower=task.report_lower,
        report_upper=task.report_upper,
    )
    prepared = _prepare_exact_task(proxy, dps=dps)
    trusted_truth = _validate_trusted_truth(prepared, task)

    distances = {
        worker: sum(
            (prepared.reports[worker][h] - trusted_truth[h]) ** 2
            for h in range(prepared.dimension)
        )
        for worker in prepared.ids
    }
    q_cal = {
        worker: prepared.tau / (prepared.tau + distances[worker])
        for worker in prepared.ids
    }

    transitions: list[ExactReputationTransition] = []
    next_reputation: dict[WorkerID, str] = {
        WorkerID(worker): str(value)
        for worker, value in task.reputations.items()
    }
    next_epochs: dict[WorkerID, int] = {
        WorkerID(worker): int(task.epochs.get(worker, 0))
        for worker in task.reputations
    }

    for worker in prepared.ids:
        c = prepared.reputations[worker]
        q = q_cal[worker]
        direct = c + prepared.eta * ((1 - c) * q - prepared.kappa * c * (1 - q))
        positive = (
            (1 - prepared.eta * prepared.kappa) * c
            + prepared.eta * q
            + prepared.eta * (prepared.kappa - 1) * c * q
        )
        prev_epoch = prepared.epochs.get(worker, 0)
        next_value = decimal_string(prepared.ctx, positive, prepared.ctx.dps)
        transitions.append(
            ExactReputationTransition(
                worker_id=worker,
                previous=decimal_string(prepared.ctx, c, prepared.ctx.dps),
                direct=decimal_string(prepared.ctx, direct, prepared.ctx.dps),
                nonnegative=next_value,
                next=next_value,
                update_count=1,
                previous_epoch=prev_epoch,
                next_epoch=prev_epoch + 1,
            )
        )
        next_reputation[worker] = next_value
        next_epochs[worker] = prev_epoch + 1

    return ExactCalibrationTaskResult(
        success=True,
        task_id=task.task_id,
        task_kind=task.task_kind,
        trusted_truth=tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in trusted_truth),
        participant_ids=prepared.ids,
        evidence=tuple(
            ExactCalibrationEvidence(
                worker_id=worker,
                distance_to_trusted_truth=decimal_string(
                    prepared.ctx, distances[worker], prepared.ctx.dps
                ),
                evidence=decimal_string(prepared.ctx, q_cal[worker], prepared.ctx.dps),
            )
            for worker in prepared.ids
        ),
        reputation_transitions=tuple(transitions),
        next_reputation=next_reputation,
        next_epochs=next_epochs,
    )
