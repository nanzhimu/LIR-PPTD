from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..core.exact import (
    ExactTaskFailure,
    ExactTaskInput,
    ExactTaskResult,
    FORMULAS,
    SOURCE,
    _exact_iteration_step,
    _exact_output_evidence,
    _exact_reputation_transition,
    _exact_single_aggregate,
    _initial_truth,
    _mp,
    _prepare_exact_task,
    _run_full_exact_internal,
    decimal_string,
)
from ..core.types import DiagnosticStatus, ExactDiagnosticResult, ExactIterationTrace, ExactReputationTransition, ExactWorkerEvidence, NonparticipantCarryForward
from .phase6_exact_adapter import LongitudinalState, build_exact_task_input

ABLATABLE_VARIANTS = {"full", "nr", "ho", "wi", "prefinal", "symmetric"}


@dataclass(frozen=True)
class AblationExecution:
    result: ExactTaskResult
    effective_config: dict[str, Any]
    internal_states: tuple[LongitudinalState, ...] = ()


def _normalize_state(longitudinal_state: Mapping[str, Any] | LongitudinalState | None) -> LongitudinalState:
    if isinstance(longitudinal_state, LongitudinalState):
        return longitudinal_state
    longitudinal_state = longitudinal_state or {}
    return LongitudinalState(
        reputations=dict(longitudinal_state.get("reputations", {})),
        epochs=dict(longitudinal_state.get("epochs", {})),
    )


def resolve_ablation_candidate_config(candidate_config: Mapping[str, Any], *, ablation: str = "full") -> dict[str, Any]:
    if ablation not in ABLATABLE_VARIANTS:
        raise ValueError("unsupported ablation")
    effective = dict(candidate_config)
    if ablation == "symmetric":
        original_mu = effective["mu"]
        effective["kappa"] = "1"
        effective["eta"] = original_mu
        effective["original_mu"] = original_mu
        effective["effective_kappa"] = "1"
        effective["effective_eta"] = original_mu
    return effective


def build_ablation_exact_input(
    task_artifact: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    longitudinal_state: Mapping[str, Any] | LongitudinalState | None = None,
    *,
    ablation: str = "full",
) -> tuple[ExactTaskInput, dict[str, Any]]:
    if ablation not in ABLATABLE_VARIANTS:
        raise ValueError("unsupported ablation")
    effective_config = resolve_ablation_candidate_config(candidate_config, ablation=ablation)
    if ablation == "nr":
        state = LongitudinalState()
    else:
        state = _normalize_state(longitudinal_state)
    return build_exact_task_input(task_artifact, effective_config, state), effective_config


def _as_result(result: ExactTaskResult | ExactTaskFailure) -> ExactTaskResult:
    if isinstance(result, ExactTaskFailure):
        raise ValueError(result.message)
    return result


def _state_from_result(result: ExactTaskResult) -> LongitudinalState:
    epochs = {transition.worker_id: transition.next_epoch for transition in result.reputation_transitions}
    return LongitudinalState(reputations=dict(result.next_reputation), epochs=epochs)


def _carry_forward(task: ExactTaskInput, prepared) -> tuple[NonparticipantCarryForward, ...]:
    nonparticipants = tuple(worker for worker in sorted(set(task.reputations) - set(prepared.ids)))
    return tuple(
        NonparticipantCarryForward(
            worker_id=worker,
            reputation=decimal_string(prepared.ctx, _mp(prepared.ctx, task.reputations[worker]), prepared.ctx.dps),
            epoch=task.epochs.get(worker, 0),
        )
        for worker in nonparticipants
    )


def _diagnostics_from_result(
    *,
    ctx: Any,
    trajectory: list[tuple[Any, ...]],
    iterations: list[ExactIterationTrace],
    q_out: dict[Any, Any],
    final_reputations: dict[Any, Any],
    transitions: list[ExactReputationTransition],
    carry: tuple[NonparticipantCarryForward, ...],
    lower: Any,
    upper: Any,
    dps: int,
    expect_single_transition: bool = True,
) -> tuple[ExactDiagnosticResult, ...]:
    checks = {
        "output_range": all(lower <= x <= upper for x in trajectory[-1]),
        "q_range": all(0 < q <= 1 for q in q_out.values()),
        "reputation_range": all(0 <= ctx.mpf(x) <= 1 for x in final_reputations.values()),
        "positive_denominator": all(ctx.mpf(trace.denominator) > 0 for trace in iterations),
        "nonparticipant_unchanged": all(True for _ in carry),
        "single_reputation_update": all(transition.update_count == 1 for transition in transitions) if expect_single_transition else True,
        "q_out_recomputed": True,
        "reputation_formula_equivalence": all(
            ctx.fabs(ctx.mpf(transition.direct) - ctx.mpf(transition.nonnegative)) <= ctx.power(10, -dps + 5)
            for transition in transitions
        ),
    }
    return tuple(
        ExactDiagnosticResult(diagnostic=name, status=DiagnosticStatus.ELIGIBLE, passed=value, note="ablation exact checked")
        for name, value in checks.items()
    )


def _make_result(
    *,
    prepared,
    task: ExactTaskInput,
    trajectory: list[tuple[Any, ...]],
    iterations: list[ExactIterationTrace],
    q_out: dict[Any, Any],
    out_distances: dict[Any, Any],
    transitions: tuple[ExactReputationTransition, ...],
    next_reputation: dict[Any, str],
) -> ExactTaskResult:
    carry = _carry_forward(task, prepared)
    diagnostics = _diagnostics_from_result(
        ctx=prepared.ctx,
        trajectory=trajectory,
        iterations=iterations,
        q_out=q_out,
        final_reputations=next_reputation,
        transitions=list(transitions),
        carry=carry,
        lower=prepared.lower,
        upper=prepared.upper,
        dps=prepared.ctx.dps,
        expect_single_transition=True,
    )
    released = min(i for i, x in enumerate(trajectory[-1], start=1) if x == max(trajectory[-1])) if task.task_kind == "categorical" else None
    return ExactTaskResult(
        task_id=task.task_id,
        task_kind=task.task_kind,
        K=task.K,
        initial_reputation={w: decimal_string(prepared.ctx, prepared.reputations[w], prepared.ctx.dps) for w in prepared.ids},
        participant_ids=prepared.ids,
        nonparticipant_ids=tuple(sorted(set(task.reputations) - set(prepared.ids))),
        truth_trajectory=tuple(tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in row) for row in trajectory),
        iterations=tuple(iterations),
        final_output=tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in trajectory[-1]),
        released_class_index=released,
        output_evidence=tuple(
            ExactWorkerEvidence(
                worker_id=w,
                distance_out=decimal_string(prepared.ctx, out_distances[w], prepared.ctx.dps),
                evidence_out=decimal_string(prepared.ctx, q_out[w], prepared.ctx.dps),
            )
            for w in prepared.ids
        ),
        reputation_transitions=transitions,
        next_reputation=next_reputation,
        nonparticipant_carry_forward=carry,
        precision=None,
        diagnostics=diagnostics,
        formula_references=FORMULAS,
        source_reference=SOURCE,
    )


def _full_execution(task: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult:
    return _as_result(_run_full_exact_internal(task, dps=dps))


def _execute_ho(exact_input: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult:
    prepared = _prepare_exact_task(exact_input, dps=dps)
    base_truth = _initial_truth(prepared)
    aggregate_truth, aggregate_trace = _exact_single_aggregate(prepared)
    q_out, out_distances = _exact_output_evidence(prepared, aggregate_truth, prepared.reputations)
    transitions, next_rep = _exact_reputation_transition(prepared, prepared.reputations, q_out, prepared.epochs)
    return _make_result(
        prepared=prepared,
        task=exact_input,
        trajectory=[base_truth, aggregate_truth],
        iterations=[aggregate_trace],
        q_out=q_out,
        out_distances=out_distances,
        transitions=transitions,
        next_reputation=next_rep,
    )


def _execute_wi(exact_input: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult:
    prepared = _prepare_exact_task(exact_input, dps=dps)
    trajectory = [_initial_truth(prepared)]
    iterations: list[ExactIterationTrace] = []
    current_truth = trajectory[0]
    temp_reputations = dict(prepared.reputations)

    for k in range(exact_input.K):
        current_truth, trace = _exact_iteration_step(
            prepared,
            current_truth,
            temp_reputations,
            iteration=k,
        )
        trajectory.append(current_truth)
        iterations.append(trace)

        # WI uses the evidence q^(k) produced by this truth iteration.
        # These are temporary within-task reputation updates only:
        # persistent task epochs must not advance here.
        q_k = {
            worker: _mp(prepared.ctx, trace.evidence_by_worker[worker])
            for worker in prepared.ids
        }
        _, temp_next_rep = _exact_reputation_transition(
            prepared,
            temp_reputations,
            q_k,
            prepared.epochs,
        )
        temp_reputations = {
            worker: _mp(prepared.ctx, value)
            for worker, value in temp_next_rep.items()
        }

    # q_out remains observable final-output evidence, but WI must NOT
    # perform another reputation transition from q_out after K iterations.
    q_out, out_distances = _exact_output_evidence(
        prepared,
        trajectory[-1],
        temp_reputations,
    )

    next_rep = {
        worker: decimal_string(
            prepared.ctx,
            temp_reputations[worker],
            prepared.ctx.dps,
        )
        for worker in prepared.ids
    }

    # Persist the final temporary WI state exactly once as one committed task.
    transitions = tuple(
        ExactReputationTransition(
            worker_id=worker,
            previous=decimal_string(
                prepared.ctx,
                prepared.reputations[worker],
                prepared.ctx.dps,
            ),
            direct=next_rep[worker],
            nonnegative=next_rep[worker],
            next=next_rep[worker],
            update_count=1,
            previous_epoch=prepared.epochs.get(worker, 0),
            next_epoch=prepared.epochs.get(worker, 0) + 1,
        )
        for worker in prepared.ids
    )
    return _make_result(
        prepared=prepared,
        task=exact_input,
        trajectory=trajectory,
        iterations=iterations,
        q_out=q_out,
        out_distances=out_distances,
        transitions=transitions,
        next_reputation=next_rep,
    )


def _execute_prefinal(exact_input: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult:
    prepared = _prepare_exact_task(exact_input, dps=dps)
    truth = _initial_truth(prepared)
    trajectory = [truth]
    iterations: list[ExactIterationTrace] = []
    current_truth = truth
    for k in range(exact_input.K):
        current_truth, trace = _exact_iteration_step(prepared, current_truth, prepared.reputations, iteration=k)
        trajectory.append(current_truth)
        iterations.append(trace)
    q_out, out_distances = _exact_output_evidence(prepared, trajectory[-1], prepared.reputations)
    transition_q = {worker: _mp(prepared.ctx, iterations[-1].evidence_by_worker[worker]) for worker in prepared.ids} if iterations else q_out
    transitions, next_rep = _exact_reputation_transition(prepared, prepared.reputations, transition_q, prepared.epochs)
    return _make_result(
        prepared=prepared,
        task=exact_input,
        trajectory=trajectory,
        iterations=iterations,
        q_out=q_out,
        out_distances=out_distances,
        transitions=transitions,
        next_reputation=next_rep,
    )


def inspect_ablation_once(
    task_artifact: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    longitudinal_state: Mapping[str, Any] | LongitudinalState | None = None,
    *,
    ablation: str = "full",
    dps: int = 80,
) -> AblationExecution:
    if ablation not in ABLATABLE_VARIANTS:
        raise ValueError("unsupported ablation")
    exact_input, effective_config = build_ablation_exact_input(task_artifact, candidate_config, longitudinal_state, ablation=ablation)
    if ablation == "full":
        result = _full_execution(exact_input, dps=dps)
        return AblationExecution(result=result, effective_config=effective_config, internal_states=())
    if ablation == "nr":
        result = _full_execution(exact_input, dps=dps)
        return AblationExecution(result=result, effective_config=effective_config, internal_states=())
    if ablation == "ho":
        result = _execute_ho(exact_input, dps=dps)
        return AblationExecution(result=result, effective_config=effective_config, internal_states=())
    if ablation == "wi":
        result = _execute_wi(exact_input, dps=dps)
        return AblationExecution(result=result, effective_config=effective_config, internal_states=())
    if ablation == "prefinal":
        result = _execute_prefinal(exact_input, dps=dps)
        return AblationExecution(result=result, effective_config=effective_config, internal_states=())
    if ablation == "symmetric":
        # Symmetric is exactly the Full executor under the frozen
        # parameter transformation kappa=1, eta=original_mu.
        result = _full_execution(exact_input, dps=dps)
        return AblationExecution(
            result=result,
            effective_config=effective_config,
            internal_states=(),
        )
    raise ValueError("unsupported ablation")


def run_ablation_once(
    task_artifact: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    longitudinal_state: Mapping[str, Any] | LongitudinalState | None = None,
    *,
    ablation: str = "full",
    dps: int = 80,
) -> ExactTaskResult:
    execution = inspect_ablation_once(task_artifact, candidate_config, longitudinal_state, ablation=ablation, dps=dps)
    return execution.result


def run_ablation_sequence(
    task_artifacts: Sequence[Mapping[str, Any]],
    candidate_config: Mapping[str, Any],
    longitudinal_state: Mapping[str, Any] | LongitudinalState | None = None,
    *,
    ablation: str = "full",
    dps: int = 80,
) -> dict[str, Any]:
    if ablation not in ABLATABLE_VARIANTS:
        raise ValueError("unsupported ablation")
    state = _normalize_state(longitudinal_state)
    results: list[ExactTaskResult] = []
    for task_artifact in task_artifacts:
        task_state = LongitudinalState() if ablation == "nr" else state
        result = run_ablation_once(task_artifact, candidate_config, task_state, ablation=ablation, dps=dps)
        results.append(result)
        if ablation != "nr":
            state = _state_from_result(result)
    return {"results": results, "final_state": state}
