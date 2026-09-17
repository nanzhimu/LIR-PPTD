from __future__ import annotations

from fractions import Fraction

from lir_pptd.core.calibration import ExactCalibrationTaskInput, run_exact_calibration_update
from lir_pptd.fixedpoint import default_profile, run_fixed_calibration_update
from lir_pptd.mpc import SimulatedShamirBackend, run_secure_calibration_update


def _cat_task(**changes):
    base = dict(
        task_id="cal-cat-20",
        task_kind="categorical",
        tau="2/5",
        epsilon_c="1/1024",
        kappa="2",
        eta="1/10",
        reports={
            "h1": ("1", "0"),
            "h2": ("1", "0"),
            "m1": ("0", "1"),
        },
        reputations={"h1": "1/2", "h2": "3/5", "m1": "2/5", "absent": "7/10"},
        epochs={"h1": 1, "h2": 2, "m1": 3, "absent": 9},
        participant_ids=("h1", "h2", "m1"),
        trusted_truth=("1", "0"),
    )
    base.update(changes)
    return ExactCalibrationTaskInput(**base)


def _num_task(**changes):
    base = dict(
        task_id="cal-num-20",
        task_kind="numerical",
        tau="1/5",
        epsilon_c="1/1024",
        kappa="2",
        eta="1/10",
        reports={"h": ("1/4",), "m": ("3/4",)},
        reputations={"h": "1/2", "m": "1/2", "absent": "1/3"},
        epochs={"h": 0, "m": 4, "absent": 8},
        participant_ids=("h", "m"),
        trusted_truth=("1/4",),
    )
    base.update(changes)
    return ExactCalibrationTaskInput(**base)


def _as_float(text: str) -> float:
    return float(Fraction(text)) if "/" in text else float(text)


def _assert_exact_fixed_close(task):
    exact = run_exact_calibration_update(task)
    fixed = run_fixed_calibration_update(task)
    assert fixed.success
    delta = default_profile().scale_delta
    fx_q = {e.worker_id: e.evidence_integer / delta for e in fixed.evidence}
    ex_q = {e.worker_id: float(e.evidence) for e in exact.evidence}
    for w in ex_q:
        assert abs(fx_q[w] - ex_q[w]) <= 4 / delta
    fx_next = {t.worker_id: t.candidate_integer / delta for t in fixed.reputation_transitions}
    ex_next = {t.worker_id: float(t.next) for t in exact.reputation_transitions}
    for w in ex_next:
        assert abs(fx_next[w] - ex_next[w]) <= 12 / delta
    assert fixed.iterations_executed == 0
    assert fixed.released_truth_estimate is None
    assert fixed.evaluation_eligible is False
    assert fixed.reputation_updates_per_participant == 1
    carry = {x.worker_id: (x.reputation_integer, x.epoch) for x in fixed.nonparticipant_carry_forward}
    assert carry["absent"][1] in {8, 9}


def _assert_fixed_secure_equal(task):
    fixed = run_fixed_calibration_update(task)
    assert fixed.success
    secure = run_secure_calibration_update(task, SimulatedShamirBackend(master_seed=20260816))
    assert secure.status == "completed"
    assert secure.iterations_executed == 0
    assert secure.released_truth_estimate is None
    assert secure.reputation_updates_per_participant == 1
    fq = {e.worker_id: (e.distance_integer, e.evidence_integer) for e in fixed.evidence}
    sq = {e.worker_id: (e.distance_integer, e.evidence_integer) for e in secure.evidence}
    assert sq == fq
    ft = {t.worker_id: (t.previous_integer, t.candidate_integer, t.previous_epoch, t.next_epoch) for t in fixed.reputation_transitions}
    st = {t.worker_id: (t.previous_integer, t.candidate_integer, t.previous_epoch, t.next_epoch) for t in secure.reputation_transitions}
    assert st == ft
    assert len(secure.operation_trace) > 0


def test_categorical_exact_fixed_and_secure_conformance():
    task = _cat_task()
    _assert_exact_fixed_close(task)
    _assert_fixed_secure_equal(task)


def test_numerical_exact_fixed_and_secure_conformance():
    task = _num_task()
    _assert_exact_fixed_close(task)
    _assert_fixed_secure_equal(task)


def test_invalid_trusted_truth_rejected_by_fixed_and_secure():
    task = _cat_task(trusted_truth=("1/2", "1/2"))
    fixed = run_fixed_calibration_update(task)
    assert not fixed.success
    try:
        run_secure_calibration_update(task)
    except ValueError as exc:
        assert "one-hot" in str(exc)
    else:
        raise AssertionError("secure calibration must reject invalid trusted truth")
