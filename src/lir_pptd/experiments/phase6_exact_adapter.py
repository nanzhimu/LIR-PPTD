from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from ..core.exact import ExactTaskInput
from ..core.consistency_scale import DEFAULT_CONSISTENCY_SCALE_MODE, resolve_from_modality


@dataclass(frozen=True)
class LongitudinalState:
    reputations: Mapping[str, Any] | None = None
    epochs: Mapping[str, int] | None = None


def _fraction_text(value: Any) -> str:
    fraction = Fraction(str(value)) if not isinstance(value, Fraction) else value
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _parse_ratio(value: Any) -> Fraction:
    return Fraction(str(value)) if not isinstance(value, Fraction) else value


def _task_dimension(task_artifact: Mapping[str, Any]) -> int:
    reports = task_artifact.get("reports", {})
    first_report = next(iter(reports.values()))
    return len(first_report)


def build_exact_task_input(
    task_artifact: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    longitudinal_state: Mapping[str, Any] | LongitudinalState | None = None,
) -> ExactTaskInput:
    if "task_id" not in task_artifact or "modality" not in task_artifact:
        raise ValueError("task artifact missing required fields")
    if "reports" not in task_artifact or "participant_ids" not in task_artifact:
        raise ValueError("task artifact missing report or participant fields")

    if isinstance(longitudinal_state, LongitudinalState):
        state_reputations = dict(longitudinal_state.reputations or {})
        state_epochs = dict(longitudinal_state.epochs or {})
    else:
        longitudinal_state = longitudinal_state or {}
        state_reputations = dict(longitudinal_state.get("reputations", {}))
        state_epochs = dict(longitudinal_state.get("epochs", {}))

    participant_ids = tuple(task_artifact["participant_ids"])
    reports = {
        worker: tuple(task_artifact["reports"][worker])
        for worker in participant_ids
    }
    dimension = _task_dimension(task_artifact)
    scale = resolve_from_modality(
        str(task_artifact["modality"]),
        dimension,
        candidate_config["lambda_tau"],
        str(candidate_config.get("consistency_scale_mode", DEFAULT_CONSISTENCY_SCALE_MODE)),
    )
    tau = _fraction_text(scale.resolved_tau)
    c0 = candidate_config["c0"]

    reputations = {
        worker: state_reputations.get(worker, c0)
        for worker in participant_ids
    }
    epochs = {worker: int(state_epochs.get(worker, 0)) for worker in participant_ids}

    return ExactTaskInput(
        task_id=str(task_artifact["task_id"]),
        task_kind=str(task_artifact["modality"]),
        K=int(candidate_config["K"]),
        tau=tau,
        epsilon_c=candidate_config["epsilon_c"],
        kappa=candidate_config["kappa"],
        eta=candidate_config["eta"],
        reports=reports,
        reputations=reputations,
        epochs=epochs,
        participant_ids=participant_ids,
        report_lower="0",
        report_upper="1",
    )
