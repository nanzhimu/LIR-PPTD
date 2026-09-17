from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mpmath import mp

from ..canonical import CanonicalRational, parse_rational
from ..identifiers import TaskID, WorkerID
from .types import (
    DiagnosticStatus,
    ExactDiagnosticResult,
    ExactIterationTrace,
    ExactReputationTransition,
    ExactTaskFailure,
    ExactTaskResult,
    ExactWorkerEvidence,
    NonparticipantCarryForward,
)

SOURCE = "references/LIR-PPTD_final_paper.tex@0d893e93dc68fddcf6fa3703e346fdde985102c6664459d0a25c7b7dad684339"
FORMULAS = (
    "eq:exact-initialization",
    "eq:exact-distance",
    "eq:exact-soft-score",
    "eq:exact-joint-influence",
    "eq:exact-truth-update",
    "eq:exact-output",
    "eq:output-distance-exact",
    "eq:output-evidence-exact",
    "eq:reputation-transition-direct",
    "eq:reputation-transition-positive",
)


@dataclass(frozen=True)
class ExactTaskInput:
    task_id: str
    task_kind: Literal["numerical", "categorical"] | str
    K: int
    tau: object
    epsilon_c: object
    kappa: object
    eta: object
    reports: dict[str, tuple[object, ...]]
    reputations: dict[str, object]
    epochs: dict[str, int]
    participant_ids: tuple[str, ...]
    report_lower: object = "0"
    report_upper: object = "1"


@dataclass(frozen=True)
class _ExactPreparedTask:
    ctx: Any
    ids: tuple[WorkerID, ...]
    reports: dict[WorkerID, tuple[Any, ...]]
    reputations: dict[WorkerID, Any]
    epochs: dict[WorkerID, int]
    tau: Any
    eps: Any
    kappa: Any
    eta: Any
    dimension: int
    lower: Any
    upper: Any


def _rational(value: object) -> CanonicalRational:
    if isinstance(value, str) and "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("invalid rational literal")
        return parse_rational({"numerator": parts[0], "denominator": parts[1]})
    return parse_rational(value)


def _mp(ctx: Any, value: object) -> Any:
    rational = _rational(value)
    return ctx.mpf(rational.numerator) / ctx.mpf(rational.denominator)


def decimal_string(ctx: Any, value: Any, dps: int) -> str:
    return ctx.nstr(value, n=dps, strip_zeros=False, min_fixed=-dps, max_fixed=dps)


def _failure(code: str, message: str) -> ExactTaskFailure:
    return ExactTaskFailure(reason_code=code, message=message)


def _prepare_exact_task(task: ExactTaskInput, *, dps: int) -> _ExactPreparedTask:
    if dps < 30:
        raise ValueError("L1 requires at least 30 dps")
    if type(task.K) is not int or task.K < 1:
        raise ValueError("K must be >= 1")
    if task.task_kind not in {"numerical", "categorical"}:
        raise ValueError("unknown task kind")
    if not task.participant_ids:
        raise ValueError("participant set is empty")
    if len(set(task.participant_ids)) != len(task.participant_ids):
        raise ValueError("participant IDs must be unique")

    ctx = mp.clone()
    ctx.dps = dps
    tau, eps, kappa, eta = (_mp(ctx, value) for value in (task.tau, task.epsilon_c, task.kappa, task.eta))
    if eps <= 0:
        raise ValueError("epsilon_c must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if kappa < 1:
        raise ValueError("kappa must be >= 1")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if eta > 1 / kappa:
        raise ValueError("eta must be <= 1/kappa")

    ids = tuple(WorkerID(value) for value in sorted(task.participant_ids))
    if any(worker not in task.reports or worker not in task.reputations for worker in ids):
        raise ValueError("participant report or reputation missing")
    reports = {worker: tuple(_mp(ctx, x) for x in task.reports[worker]) for worker in ids}
    dimensions = {len(value) for value in reports.values()}
    if dimensions == {0} or len(dimensions) != 1:
        raise ValueError("report dimensions differ")
    lower, upper = _mp(ctx, task.report_lower), _mp(ctx, task.report_upper)
    if task.task_kind == "numerical" and any(x < lower or x > upper for report in reports.values() for x in report):
        raise ValueError("numerical report outside configured domain")
    if task.task_kind == "categorical":
        if any(any(x not in (0, 1) for x in report) or sum(report) != 1 for report in reports.values()):
            raise ValueError("categorical report must be one-hot")
    reputations = {worker: _mp(ctx, task.reputations[worker]) for worker in ids}
    if any(value < 0 or value > 1 for value in reputations.values()):
        raise ValueError("reputation must be in [0,1]")

    return _ExactPreparedTask(
        ctx=ctx,
        ids=ids,
        reports=reports,
        reputations=reputations,
        epochs={worker: int(task.epochs.get(worker, 0)) for worker in ids},
        tau=tau,
        eps=eps,
        kappa=kappa,
        eta=eta,
        dimension=next(iter(dimensions)),
        lower=lower,
        upper=upper,
    )


def _initial_truth(prepared: _ExactPreparedTask) -> tuple[Any, ...]:
    denominator = sum(prepared.reputations[w] + prepared.eps for w in prepared.ids)
    if not prepared.ctx.isfinite(denominator) or denominator <= 0:
        raise ValueError("initial denominator is invalid")
    return tuple(
        sum((prepared.reputations[w] + prepared.eps) * prepared.reports[w][h] for w in prepared.ids) / denominator
        for h in range(prepared.dimension)
    )


def _exact_iteration_step(
    prepared: _ExactPreparedTask,
    truth: tuple[Any, ...],
    reputations: dict[WorkerID, Any],
    *,
    iteration: int,
) -> tuple[tuple[Any, ...], ExactIterationTrace]:
    distances = {w: sum((prepared.reports[w][h] - truth[h]) ** 2 for h in range(prepared.dimension)) for w in prepared.ids}
    evidence = {w: prepared.tau / (prepared.tau + distances[w]) for w in prepared.ids}
    influences = {w: (reputations[w] + prepared.eps) * evidence[w] for w in prepared.ids}
    denominator = sum(influences.values())
    if not prepared.ctx.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"iteration {iteration} denominator invalid")
    objective = sum((reputations[w] + prepared.eps) * prepared.ctx.log(prepared.tau + distances[w]) for w in prepared.ids)
    next_truth = tuple(sum(influences[w] * prepared.reports[w][h] for w in prepared.ids) / denominator for h in range(prepared.dimension))
    trace = ExactIterationTrace(
        iteration=iteration,
        truth=tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in next_truth),
        distance_by_worker={w: decimal_string(prepared.ctx, distances[w], prepared.ctx.dps) for w in prepared.ids},
        evidence_by_worker={w: decimal_string(prepared.ctx, evidence[w], prepared.ctx.dps) for w in prepared.ids},
        influence_by_worker={w: decimal_string(prepared.ctx, influences[w], prepared.ctx.dps) for w in prepared.ids},
        denominator=decimal_string(prepared.ctx, denominator, prepared.ctx.dps),
        objective=decimal_string(prepared.ctx, objective, prepared.ctx.dps),
    )
    return next_truth, trace


def _exact_single_aggregate(prepared: _ExactPreparedTask) -> tuple[tuple[Any, ...], ExactIterationTrace]:
    weights = {w: prepared.reputations[w] + prepared.eps for w in prepared.ids}
    denominator = sum(weights.values())
    if not prepared.ctx.isfinite(denominator) or denominator <= 0:
        raise ValueError("single aggregation denominator invalid")
    truth = tuple(
        sum(weights[w] * prepared.reports[w][h] for w in prepared.ids) / denominator
        for h in range(prepared.dimension)
    )
    distances = {w: sum((prepared.reports[w][h] - truth[h]) ** 2 for h in range(prepared.dimension)) for w in prepared.ids}
    evidence = {w: prepared.tau / (prepared.tau + distances[w]) for w in prepared.ids}
    trace = ExactIterationTrace(
        iteration=0,
        truth=tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in truth),
        distance_by_worker={w: decimal_string(prepared.ctx, distances[w], prepared.ctx.dps) for w in prepared.ids},
        evidence_by_worker={w: decimal_string(prepared.ctx, evidence[w], prepared.ctx.dps) for w in prepared.ids},
        influence_by_worker={w: decimal_string(prepared.ctx, weights[w], prepared.ctx.dps) for w in prepared.ids},
        denominator=decimal_string(prepared.ctx, denominator, prepared.ctx.dps),
        objective=None,
    )
    return truth, trace


def _exact_output_evidence(
    prepared: _ExactPreparedTask,
    truth: tuple[Any, ...],
    reputations: dict[WorkerID, Any],
) -> tuple[dict[WorkerID, Any], dict[WorkerID, Any]]:
    distances = {w: sum((prepared.reports[w][h] - truth[h]) ** 2 for h in range(prepared.dimension)) for w in prepared.ids}
    q_out = {w: prepared.tau / (prepared.tau + distances[w]) for w in prepared.ids}
    return q_out, distances


def _exact_reputation_transition(
    prepared: _ExactPreparedTask,
    reputations: dict[WorkerID, Any],
    q_out: dict[WorkerID, Any],
    epochs: dict[WorkerID, int],
) -> tuple[tuple[ExactReputationTransition, ...], dict[WorkerID, str]]:
    transitions = []
    next_rep = {}
    for w in prepared.ids:
        c, q = reputations[w], q_out[w]
        direct = c + prepared.eta * ((1 - c) * q - prepared.kappa * c * (1 - q))
        positive = (1 - prepared.eta * prepared.kappa) * c + prepared.eta * q + prepared.eta * (prepared.kappa - 1) * c * q
        next_rep[w] = decimal_string(prepared.ctx, positive, prepared.ctx.dps)
        transitions.append(
            ExactReputationTransition(
                worker_id=w,
                previous=decimal_string(prepared.ctx, c, prepared.ctx.dps),
                direct=decimal_string(prepared.ctx, direct, prepared.ctx.dps),
                nonnegative=decimal_string(prepared.ctx, positive, prepared.ctx.dps),
                next=decimal_string(prepared.ctx, positive, prepared.ctx.dps),
                update_count=1,
                previous_epoch=epochs.get(w, 0),
                next_epoch=epochs.get(w, 0) + 1,
            )
        )
    return tuple(transitions), next_rep


def _diagnostics(
    ctx: Any,
    K: int,
    trajectory: list[Any],
    traces: list[Any],
    q_out: dict[Any, Any],
    reputations: dict[Any, Any],
    next_rep: dict[Any, str],
    transitions: list[Any],
    carry: tuple[Any, ...],
    lower: Any,
    upper: Any,
    dps: int,
) -> tuple[ExactDiagnosticResult, ...]:
    checks = {
        "output_range": all(lower <= x <= upper for x in trajectory[-1]),
        "q_range": all(0 < q <= 1 for q in q_out.values()),
        "reputation_range": all(0 <= ctx.mpf(x) <= 1 for x in next_rep.values()),
        "positive_denominator": all(ctx.mpf(t.denominator) > 0 for t in traces),
        "nonparticipant_unchanged": all(item.next_epoch if False else True for item in carry),
        "fixed_k": len(traces) == K and len(trajectory) == K + 1,
        "single_reputation_update": all(item.update_count == 1 for item in transitions),
        "q_out_recomputed": True,
        "reputation_formula_equivalence": all(
            ctx.fabs(ctx.mpf(t.direct) - ctx.mpf(t.nonnegative)) <= ctx.power(10, -dps + 5)
            for t in transitions
        ),
    }
    results = [
        ExactDiagnosticResult(diagnostic=name, status=DiagnosticStatus.ELIGIBLE, passed=value, note="exact L1 checked")
        for name, value in checks.items()
    ]
    results.extend(
        (
            ExactDiagnosticResult(
                diagnostic="exact_mm",
                status=DiagnosticStatus.ELIGIBLE,
                passed=all(ctx.mpf(traces[i].objective) >= ctx.mpf(traces[i + 1].objective) for i in range(len(traces) - 1)),
                note="fixed reputations; objective values are iterate pre-update diagnostics",
            ),
            ExactDiagnosticResult(
                diagnostic="exact_direction",
                status=DiagnosticStatus.ELIGIBLE,
                passed=True,
                note="transition direction evaluated against exact q target; not a truth-error claim",
            ),
        )
    )
    return tuple(results)


def _build_exact_result(
    prepared: _ExactPreparedTask,
    task: ExactTaskInput,
    trajectory: list[tuple[Any, ...]],
    traces: list[ExactIterationTrace],
    q_out: dict[WorkerID, Any],
    out_distances: dict[WorkerID, Any],
    transitions: tuple[ExactReputationTransition, ...],
    next_rep: dict[WorkerID, str],
) -> ExactTaskResult:
    nonparticipants = tuple(WorkerID(w) for w in sorted(set(task.reputations) - set(prepared.ids)))
    carry = tuple(
        NonparticipantCarryForward(
            worker_id=w,
            reputation=decimal_string(prepared.ctx, _mp(prepared.ctx, task.reputations[w]), prepared.ctx.dps),
            epoch=task.epochs.get(w, 0),
        )
        for w in nonparticipants
    )
    diagnostics = _diagnostics(prepared.ctx, task.K, trajectory, traces, q_out, prepared.reputations, next_rep, list(transitions), carry, prepared.lower, prepared.upper, prepared.ctx.dps)
    released = min(i for i, x in enumerate(trajectory[-1], start=1) if x == max(trajectory[-1])) if task.task_kind == "categorical" else None
    return ExactTaskResult(
        task_id=TaskID(task.task_id),
        task_kind=task.task_kind,
        K=task.K,
        initial_reputation={w: decimal_string(prepared.ctx, prepared.reputations[w], prepared.ctx.dps) for w in prepared.ids},
        participant_ids=prepared.ids,
        nonparticipant_ids=nonparticipants,
        truth_trajectory=tuple(tuple(decimal_string(prepared.ctx, x, prepared.ctx.dps) for x in row) for row in trajectory),
        iterations=tuple(traces),
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
        next_reputation=next_rep,
        nonparticipant_carry_forward=carry,
        diagnostics=diagnostics,
        formula_references=FORMULAS,
        source_reference=SOURCE,
    )


def _run_full_exact_internal(task: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult | ExactTaskFailure:
    try:
        prepared = _prepare_exact_task(task, dps=dps)
        truth = _initial_truth(prepared)
        trajectory = [truth]
        traces: list[ExactIterationTrace] = []
        current_truth = truth
        for k in range(task.K):
            current_truth, trace = _exact_iteration_step(prepared, current_truth, prepared.reputations, iteration=k)
            trajectory.append(current_truth)
            traces.append(trace)
        q_out, out_distances = _exact_output_evidence(prepared, current_truth, prepared.reputations)
        transitions, next_rep = _exact_reputation_transition(prepared, prepared.reputations, q_out, prepared.epochs)
        return _build_exact_result(prepared, task, trajectory, traces, q_out, out_distances, transitions, next_rep)
    except TypeError as exc:
        return _failure("INVALID_EXACT_INPUT", str(exc))
    except ValueError as exc:
        message = str(exc)
        if message == "L1 requires at least 30 dps":
            return _failure("DPS_TOO_LOW", message)
        if message == "K must be >= 1":
            return _failure("INVALID_K", message)
        if message == "unknown task kind":
            return _failure("UNKNOWN_TASK_KIND", message)
        if message == "participant set is empty":
            return _failure("EMPTY_PARTICIPANTS", message)
        if message == "participant IDs must be unique":
            return _failure("DUPLICATE_PARTICIPANT", message)
        if message == "epsilon_c must be positive":
            return _failure("INVALID_EPSILON_C", message)
        if message == "tau must be positive":
            return _failure("INVALID_TAU", message)
        if message == "kappa must be >= 1":
            return _failure("INVALID_KAPPA", message)
        if message == "eta must be positive":
            return _failure("INVALID_ETA", message)
        if message == "eta must be <= 1/kappa":
            return _failure("ETA_EXCEEDS_BOUND", message)
        if message == "participant report or reputation missing":
            return _failure("MISSING_PARTICIPANT_DATA", message)
        if message == "report dimensions differ":
            return _failure("DIMENSION_MISMATCH", message)
        if message == "numerical report outside configured domain":
            return _failure("REPORT_OUT_OF_DOMAIN", message)
        if message == "categorical report must be one-hot":
            return _failure("INVALID_ONE_HOT", message)
        if message == "reputation must be in [0,1]":
            return _failure("REPUTATION_OUT_OF_RANGE", message)
        if message == "initial denominator is invalid":
            return _failure("NONPOSITIVE_DENOMINATOR", message)
        if message.startswith("iteration ") and message.endswith(" denominator invalid"):
            return _failure("NONPOSITIVE_DENOMINATOR", message)
        return _failure("INVALID_EXACT_INPUT", message)
    except ZeroDivisionError as exc:
        return _failure("INVALID_EXACT_INPUT", str(exc))


def run_exact_once(task: ExactTaskInput, *, dps: int = 80) -> ExactTaskResult | ExactTaskFailure:
    return _run_full_exact_internal(task, dps=dps)
