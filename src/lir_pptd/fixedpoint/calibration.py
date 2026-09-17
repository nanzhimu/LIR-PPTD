from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..core.calibration import ExactCalibrationTaskInput
from .arithmetic import div_positive, local_add, local_sub, square
from .encoding import FixedPointError, decimal, trunc_scaled
from .profile import FixedBackendProfile, default_profile
from ..core.consistency_scale import squared_distance_bound_from_modality
from .range_analysis import PreflightStatus, preflight
from .reputation import update_reputation
from .types import FixedNonparticipantCarryForward, FixedReputationTransition


@dataclass(frozen=True)
class FixedCalibrationEvidence:
    worker_id: str
    distance_integer: int
    evidence_integer: int
    distance_decoded: str
    evidence_decoded: str


@dataclass(frozen=True)
class FixedCalibrationTaskFailure:
    success: bool
    reason_code: str
    message: str


@dataclass(frozen=True)
class FixedCalibrationTaskResult:
    success: bool
    task_id: str
    task_kind: str
    profile_id: str
    backend_profile_hash: str
    participant_ids: tuple[str, ...]
    nonparticipant_ids: tuple[str, ...]
    trusted_truth_integer: tuple[int, ...]
    trusted_truth_decoded: tuple[str, ...]
    input_report_integers: Mapping[str, tuple[int, ...]]
    initial_reputation_integers: Mapping[str, int]
    evidence: tuple[FixedCalibrationEvidence, ...]
    reputation_transitions: tuple[FixedReputationTransition, ...]
    nonparticipant_carry_forward: tuple[FixedNonparticipantCarryForward, ...]
    released_truth_estimate: None = None
    evaluation_eligible: bool = False
    iterations_executed: int = 0
    reputation_updates_per_participant: int = 1


def _validate_fixed_trusted_truth(task: ExactCalibrationTaskInput, truth: tuple[int, ...], delta: int) -> None:
    if not truth:
        raise ValueError("trusted truth dimension mismatch")
    if task.task_kind == "categorical":
        if any(x not in (0, delta) for x in truth) or sum(truth) != delta:
            raise ValueError("trusted categorical truth must be one-hot")
    elif task.task_kind == "numerical":
        if any(x < 0 or x > delta for x in truth):
            raise ValueError("trusted numerical truth outside configured domain")
    else:
        raise ValueError("unknown task kind")


def run_fixed_calibration_update(
    task: ExactCalibrationTaskInput,
    p: FixedBackendProfile | None = None,
) -> FixedCalibrationTaskResult | FixedCalibrationTaskFailure:
    """Deterministic fixed-point implementation of trusted calibration.

    No truth discovery loop is executed.  The authenticated truth is encoded on
    the same f-bit grid used by the ordinary LIR-PPTD fixed-point path; each
    participant receives exactly one bounded reputation transition.
    """
    p = p or default_profile()
    try:
        ids = tuple(sorted(task.participant_ids))
        if not ids:
            raise ValueError("empty participants")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate participant")
        if task.task_kind not in {"categorical", "numerical"}:
            raise ValueError("unknown task kind")
        dims = {len(task.reports[w]) for w in ids}
        if len(dims) != 1 or not dims or next(iter(dims)) < 1:
            raise ValueError("dimension mismatch")
        dimension = next(iter(dims))
        distance_bound = int(squared_distance_bound_from_modality(task.task_kind, dimension))
        if len(task.trusted_truth) != dimension:
            raise ValueError("trusted truth dimension mismatch")

        cert = preflight(
            p,
            participants=len(ids),
            dimension=dimension,
            K=1,
            epsilon_c=task.epsilon_c,
            tau=task.tau,
            eta=task.eta,
            kappa=task.kappa,
            task_kind=task.task_kind,
        )
        if cert.preflight_status != PreflightStatus.PASSED:
            return FixedCalibrationTaskFailure(False, cert.reason_code or "PREFLIGHT_REJECTED", "static preflight did not pass")

        delta = p.scale_delta
        reports = {w: tuple(trunc_scaled(x, delta) for x in task.reports[w]) for w in ids}
        if any(x < 0 or x > delta for row in reports.values() for x in row):
            raise ValueError("report outside [0,1]")
        if task.task_kind == "categorical" and any(
            any(x not in (0, delta) for x in row) or sum(row) != delta for row in reports.values()
        ):
            raise ValueError("one-hot required")

        trusted = tuple(trunc_scaled(x, delta) for x in task.trusted_truth)
        _validate_fixed_trusted_truth(task, trusted, delta)

        reps = {w: trunc_scaled(task.reputations[w], delta) for w in ids}
        if any(x < 0 or x > delta for x in reps.values()):
            raise ValueError("reputation outside [0,1]")
        tau = trunc_scaled(task.tau, delta)
        if tau <= 0:
            raise ValueError("tau must be positive")

        evidence = []
        transitions = []
        for w in ids:
            d = 0
            for h in range(dimension):
                diff = local_sub(reports[w][h], trusted[h], p).output
                d = local_add(d, square(diff, p).output, p).output
            q = div_positive(tau, tau + d, p, tau, tau + distance_bound * delta).output
            evidence.append(
                FixedCalibrationEvidence(
                    worker_id=w,
                    distance_integer=d,
                    evidence_integer=q,
                    distance_decoded=decimal(d, delta),
                    evidence_decoded=decimal(q, delta),
                )
            )
            transitions.append(
                update_reputation(w, reps[w], q, task.eta, task.kappa, task.epochs.get(w, 0), p)
            )

        non = tuple(sorted(set(task.reputations) - set(ids)))
        carry = tuple(
            FixedNonparticipantCarryForward(
                worker_id=w,
                reputation_integer=trunc_scaled(task.reputations[w], delta),
                epoch=task.epochs.get(w, 0),
            )
            for w in non
        )
        return FixedCalibrationTaskResult(
            success=True,
            task_id=task.task_id,
            task_kind=task.task_kind,
            profile_id=p.profile_id,
            backend_profile_hash=str(p.digest()),
            participant_ids=ids,
            nonparticipant_ids=non,
            trusted_truth_integer=trusted,
            trusted_truth_decoded=tuple(decimal(x, delta) for x in trusted),
            input_report_integers=reports,
            initial_reputation_integers=reps,
            evidence=tuple(evidence),
            reputation_transitions=tuple(transitions),
            nonparticipant_carry_forward=carry,
        )
    except (FixedPointError, ValueError, KeyError, TypeError) as exc:
        return FixedCalibrationTaskFailure(False, getattr(exc, "reason_code", "INVALID_FIXED_CALIBRATION_INPUT"), str(exc))
