from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...baselines.crh import predict as crh_predict
from ...baselines.mean_vote import predict as mean_vote_predict
from ...core.types import ExactTaskResult
from ..phase6_ablation_runner import run_ablation_once
from ..phase6_exact_adapter import LongitudinalState
from .longitudinal_state import merge_participant_result
from .real_data_loader import RealTask, task_artifact


class MethodExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StatefulPrediction:
    prediction_vector: tuple[float, ...]
    prediction_class: int | None
    next_state: LongitudinalState
    result: ExactTaskResult


def _class_from_vector(values: tuple[float, ...]) -> int:
    if not values:
        raise MethodExecutionError("empty categorical prediction")
    maximum = max(values)
    return min(i for i, value in enumerate(values) if value == maximum)


def run_stateless_baseline(task: RealTask, method: str) -> tuple[tuple[float, ...], int | None]:
    if method == "mean_vote":
        raw = mean_vote_predict(task.reports, task.modality)
    elif method == "crh":
        raw = crh_predict(task.reports, task.modality)
    else:
        raise MethodExecutionError(f"unsupported stateless baseline: {method}")
    vector = tuple(float(x) for x in raw)
    prediction_class = _class_from_vector(vector) if task.modality == "categorical" else None
    return vector, prediction_class


def run_lir_task(
    task: RealTask,
    candidate_config: Mapping[str, Any],
    global_state: LongitudinalState,
    *,
    ablation: str = "full",
    dps: int = 80,
) -> StatefulPrediction:
    result = run_ablation_once(
        task_artifact(task),
        candidate_config,
        global_state,
        ablation=ablation,
        dps=dps,
    )
    vector = tuple(float(x) for x in result.final_output)
    prediction_class = _class_from_vector(vector) if task.modality == "categorical" else None
    if ablation == "nr":
        # NR deliberately has no persistent historical reputation. Preserve the
        # global state object unchanged so later tasks cannot accidentally use
        # the just-computed NR transition.
        next_state = global_state
    else:
        next_state = merge_participant_result(global_state, result)
    return StatefulPrediction(
        prediction_vector=vector,
        prediction_class=prediction_class,
        next_state=next_state,
        result=result,
    )
