from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..core.calibration import ExactCalibrationTaskInput
from ..fixedpoint import default_profile
from ..fixedpoint.encoding import trunc_scaled
from ..core.consistency_scale import squared_distance_bound_from_modality
from ..fixedpoint.range_analysis import PreflightStatus, preflight
from .simulated_shamir import SimulatedBackendError, SimulatedShamirBackend
from .types import TranscriptEntry


@dataclass(frozen=True)
class SecureCalibrationEvidence:
    worker_id: str
    distance_integer: int
    evidence_integer: int


@dataclass(frozen=True)
class SecureCalibrationTransition:
    worker_id: str
    previous_integer: int
    candidate_integer: int
    update_count: int
    previous_epoch: int
    next_epoch: int


@dataclass(frozen=True)
class SecureCalibrationTaskResult:
    status: str
    task_id: str
    task_kind: str
    participant_ids: tuple[str, ...]
    trusted_truth_integer: tuple[int, ...]
    evidence: tuple[SecureCalibrationEvidence, ...]
    reputation_transitions: tuple[SecureCalibrationTransition, ...]
    nonparticipant_carry_forward: tuple[dict[str, int | str], ...]
    iterations_executed: int
    reputation_updates_per_participant: int
    released_truth_estimate: None
    evaluation_eligible: bool
    operation_trace: tuple[TranscriptEntry, ...]
    backend_kind: str = "simulated_shamir"
    production_mpc_used: bool = False
    cryptographic_security_claim: bool = False


def _ok(result):
    if result.status != "completed" or result.output is None:
        raise SimulatedBackendError(result.reason_code or "OPERATION_REJECTED", "backend operation rejected")
    return result.output


def _value(backend: SimulatedShamirBackend, bundle):
    result = backend.reconstruct(bundle)
    if result.status != "reconstructed" or result.value is None:
        raise SimulatedBackendError(result.reason_code or "RECONSTRUCTION_FAILED", "reconstruction failed")
    return result.value if result.value <= backend.profile.field_prime_p // 2 else result.value - backend.profile.field_prime_p


def _validate_trusted(task: ExactCalibrationTaskInput, trusted: tuple[int, ...], delta: int) -> None:
    if task.task_kind == "categorical":
        if any(x not in (0, delta) for x in trusted) or sum(trusted) != delta:
            raise ValueError("trusted categorical truth must be one-hot")
    elif task.task_kind == "numerical":
        if any(x < 0 or x > delta for x in trusted):
            raise ValueError("trusted numerical truth outside configured domain")
    else:
        raise ValueError("unknown task kind")


def run_secure_calibration_update(
    task: ExactCalibrationTaskInput,
    backend: SimulatedShamirBackend | None = None,
) -> SecureCalibrationTaskResult:
    """Simulated-Shamir execution of a trusted calibration reputation update.

    The trusted truth is treated as authenticated input to the secure arithmetic
    path. No ordinary truth-estimation loop runs and no inferred task output is
    prepared/released.
    """
    backend = backend or SimulatedShamirBackend()
    p = default_profile()
    ids = tuple(sorted(task.participant_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("invalid participant set")
    dims = {len(task.reports[w]) for w in ids}
    if len(dims) != 1 or next(iter(dims)) < 1:
        raise ValueError("dimension mismatch")
    D = next(iter(dims))
    distance_bound = int(squared_distance_bound_from_modality(task.task_kind, D))
    if len(task.trusted_truth) != D:
        raise ValueError("trusted truth dimension mismatch")

    cert = preflight(p, participants=len(ids), dimension=D, K=1, epsilon_c=task.epsilon_c,
                     tau=task.tau, eta=task.eta, kappa=task.kappa, task_kind=task.task_kind)
    if cert.preflight_status != PreflightStatus.PASSED:
        raise ValueError(cert.reason_code or "PREFLIGHT_REJECTED")

    delta = p.scale_delta
    reports = {w: tuple(trunc_scaled(x, delta) for x in task.reports[w]) for w in ids}
    reps = {w: trunc_scaled(task.reputations[w], delta) for w in ids}
    trusted = tuple(trunc_scaled(x, delta) for x in task.trusted_truth)
    _validate_trusted(task, trusted, delta)
    if task.task_kind == "categorical" and any(
        any(x not in (0, delta) for x in row) or sum(row) != delta for row in reports.values()
    ):
        raise ValueError("INVALID_ONE_HOT")

    tau = trunc_scaled(task.tau, delta)
    et = Fraction(str(task.eta))
    kappa = Fraction(str(task.kappa))
    beta = tuple(trunc_scaled(x, delta) for x in (1 - et * kappa, et, et * (kappa - 1)))

    def share(secret, value, namespace="algorithm", worker=None, coordinate=None):
        backend.set_context(task_id=task.task_id, iteration=None, worker_id=worker, coordinate=coordinate)
        return backend.share_secret(secret, value, namespace=namespace)

    report_shares = {
        w: tuple(share(f"cal-report:{w}:{h}", x, "calibration-input", worker=w, coordinate=h) for h, x in enumerate(row))
        for w, row in reports.items()
    }
    rep_shares = {w: share(f"cal-reputation:{w}", x, "calibration-input", worker=w) for w, x in reps.items()}
    truth_shares = tuple(share(f"cal-trusted-truth:{h}", x, "calibration-trusted", coordinate=h) for h, x in enumerate(trusted))
    tau_s = share("cal-tau", tau, "public")

    evidence = []
    transitions = []
    for w in ids:
        squares = []
        for h in range(D):
            backend.set_context(task_id=task.task_id, iteration=None, worker_id=w, coordinate=h)
            diff = _ok(backend.local_sub(report_shares[w][h], truth_shares[h]))
            squares.append(_ok(backend.SecSqr(diff)))
        distance = squares[0]
        for sq in squares[1:]:
            distance = _ok(backend.local_add(distance, sq))
        q = _ok(backend.SecDivPositive(tau_s, _ok(backend.local_add(tau_s, distance)), tau, tau + distance_bound * delta))
        cq = _ok(backend.SecMulPositive(rep_shares[w], q))
        terms = [
            _ok(backend.PubMulPositive(rep_shares[w], beta[0])),
            _ok(backend.PubMulPositive(q, beta[1])),
            _ok(backend.PubMulPositive(cq, beta[2])),
        ]
        candidate = _ok(backend.local_add(_ok(backend.local_add(terms[0], terms[1])), terms[2]))
        prepared = backend.prepare_reputation(w, task.epochs.get(w, 0), task.epochs.get(w, 0) + 1)
        evidence.append(SecureCalibrationEvidence(w, _value(backend, distance), _value(backend, q)))
        transitions.append(SecureCalibrationTransition(
            worker_id=w,
            previous_integer=reps[w],
            candidate_integer=_value(backend, candidate),
            update_count=prepared.update_count,
            previous_epoch=prepared.previous_epoch,
            next_epoch=prepared.next_epoch,
        ))

    non = tuple(
        {
            "worker_id": w,
            "reputation_integer": trunc_scaled(task.reputations[w], delta),
            "epoch": task.epochs.get(w, 0),
        }
        for w in sorted(set(task.reputations) - set(ids))
    )
    return SecureCalibrationTaskResult(
        status="completed",
        task_id=task.task_id,
        task_kind=task.task_kind,
        participant_ids=ids,
        trusted_truth_integer=trusted,
        evidence=tuple(evidence),
        reputation_transitions=tuple(transitions),
        nonparticipant_carry_forward=non,
        iterations_executed=0,
        reputation_updates_per_participant=1,
        released_truth_estimate=None,
        evaluation_eligible=False,
        operation_trace=backend.export_transcript(public=False).entries,
    )
