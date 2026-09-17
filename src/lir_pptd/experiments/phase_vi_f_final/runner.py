from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from lir_pptd.experiments.phase_vi_f import runner as legacy
from lir_pptd.protocol_v2 import (
    DeliveryState,
    PreprocessingLifecycle,
    ProtocolEngine,
    ProtocolResponse,
    ProtocolState,
    SQLiteProtocolStore,
    committed_state_availability,
    output_prepare_quorum,
    restart_admissibility_quorum,
)

STUDY_ID = "vi_f_final_evidence_gated_recovery_v1"
WARMUPS = 5
MEASURED = 30
N = 10
T = 4
Q_OUT = 6
F_REP = 2
F_OUT = 2

CASE_IDS = (
    "ordinary_crash_before_decision",
    "ordinary_crash_after_prepared",
    "ordinary_crash_after_commit",
    "ordinary_postcommit_delivery_failure",
    "calibration_crash_before_decision",
    "calibration_crash_after_prepared",
    "calibration_crash_after_commit",
    "calibration_duplicate_replay",
    "ordinary_duplicate_replay",
    "conflicting_replay",
    "stale_ordinary_snapshot_after_calibration",
    "threshold_availability_boundary",
)

RAW_REL = Path("results/vi_f_final_semantics_v1/availability_recovery_trials.jsonl")
SUMMARY_REL = Path("results/vi_f_final_semantics_v1/formal_summary.json")
CASE_CSV_REL = Path("results/vi_f_final_semantics_v1/vi_f_case_results.csv")
LATENCY_CSV_REL = Path("results/vi_f_final_semantics_v1/vi_f_recovery_latency_summary.csv")
SEMANTIC_REL = Path("results/vi_f_final_semantics_v1/vi_f_semantic_conformance.json")
RUN_ROOT_REL = Path("results/vi_f_final_semantics_v1/runs")

SOURCE_FILES = (
    "src/lir_pptd/protocol_v2/model.py",
    "src/lir_pptd/protocol_v2/quorum.py",
    "src/lir_pptd/protocol_v2/attempt.py",
    "src/lir_pptd/protocol_v2/durable_store.py",
    "src/lir_pptd/protocol_v2/recovery.py",
    "src/lir_pptd/protocol_v2/engine.py",
    "src/lir_pptd/experiments/phase_vi_f/runner.py",
    "src/lir_pptd/experiments/phase_vi_f_final/runner.py",
    "configs/phase_vi_f_final/vi_f_final_semantics_design.json",
)

PAPER_LABELS = {
    "ordinary_crash_before_decision": "Ordinary: crash before durable decision",
    "ordinary_crash_after_prepared": "Ordinary: crash after Prepared",
    "ordinary_crash_after_commit": "Ordinary: crash after Commit",
    "ordinary_postcommit_delivery_failure": "Ordinary: postcommit delivery failure",
    "calibration_crash_before_decision": "Calibration: crash before durable decision",
    "calibration_crash_after_prepared": "Calibration: crash after Prepared",
    "calibration_crash_after_commit": "Calibration: crash after Commit",
    "calibration_duplicate_replay": "Calibration: duplicate replay",
    "ordinary_duplicate_replay": "Ordinary: duplicate replay",
    "conflicting_replay": "Conflicting replay",
    "stale_ordinary_snapshot_after_calibration": "Stale ordinary snapshot after calibration",
    "threshold_availability_boundary": "$T$ / storage-boundary checks",
}

EXPECTED = {
    "ordinary_crash_before_decision": "Abort/restart outcome; no semantic reputation transition",
    "ordinary_crash_after_prepared": "RecoveryNoDecision Abort; no semantic reputation transition",
    "ordinary_crash_after_commit": "Commit preserved; log repair idempotent; semantic reputation unchanged",
    "ordinary_postcommit_delivery_failure": "Commit preserved; delivery metadata stable; semantic reputation unchanged",
    "calibration_crash_before_decision": "Abort/restart outcome; no calibration transition installed",
    "calibration_crash_after_prepared": "RecoveryNoDecision Abort; no calibration transition installed",
    "calibration_crash_after_commit": "Commit preserved; exactly one calibration transition installed after reopen",
    "calibration_duplicate_replay": "Stable Commit; calibration transition remains exactly once",
    "ordinary_duplicate_replay": "Stable Commit; semantic reputation remains unchanged",
    "conflicting_replay": "RejectedReplay before durable-state mutation",
    "stale_ordinary_snapshot_after_calibration": "Reject stale epoch-bound ordinary request before protocol mutation",
    "threshold_availability_boundary": "T continues arithmetic; T-1 rejects; Qout and storage predicates respect their own thresholds",
}

class VIFinalError(RuntimeError):
    pass

class StaleSnapshot(VIFinalError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _ensure_semantic_tables(store: SQLiteProtocolStore) -> None:
    with store.transaction():
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS vi_f_semantic_state ("
            "worker_id TEXT PRIMARY KEY, learning_epoch INTEGER NOT NULL)"
        )
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS vi_f_mode_bindings ("
            "tx_id TEXT PRIMARY KEY, mode TEXT NOT NULL, snapshot_json TEXT NOT NULL, "
            "participants_json TEXT NOT NULL, transition_applied INTEGER NOT NULL DEFAULT 0, "
            "mode_hash TEXT NOT NULL)"
        )
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS vi_f_semantic_transitions ("
            "tx_id TEXT NOT NULL, worker_id TEXT NOT NULL, from_epoch INTEGER NOT NULL, "
            "to_epoch INTEGER NOT NULL, PRIMARY KEY(tx_id, worker_id))"
        )


def _init_workers(store: SQLiteProtocolStore, workers: tuple[Any, ...], epoch: int = 0) -> None:
    _ensure_semantic_tables(store)
    with store.transaction():
        for worker in workers:
            store.connection.execute(
                "INSERT OR IGNORE INTO vi_f_semantic_state(worker_id, learning_epoch) VALUES (?, ?)",
                (str(worker), int(epoch)),
            )


def _current_epochs(store: SQLiteProtocolStore, workers: tuple[Any, ...]) -> tuple[int, ...]:
    _init_workers(store, workers)
    values = []
    for worker in workers:
        row = store.connection.execute(
            "SELECT learning_epoch FROM vi_f_semantic_state WHERE worker_id=?", (str(worker),)
        ).fetchone()
        values.append(int(row[0]))
    return tuple(values)


def _bind_mode(store: SQLiteProtocolStore, fixture: Any, mode: str, snapshot: tuple[int, ...] | None = None) -> str:
    workers = tuple(fixture.certificate.participant_ids)
    snapshot = tuple(fixture.certificate.reputation_epoch_snapshot if snapshot is None else snapshot)
    current = _current_epochs(store, workers)
    if current != snapshot:
        raise StaleSnapshot(f"STALE_REPUTATION_SNAPSHOT:current={current}:request={snapshot}")
    payload = {
        "tx_id": str(fixture.request.tx_id),
        "task_id": str(fixture.request.task_id),
        "mode": mode,
        "snapshot": list(snapshot),
        "participants": [str(w) for w in workers],
    }
    mode_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    with store.transaction():
        store.connection.execute(
            "INSERT INTO vi_f_mode_bindings(tx_id, mode, snapshot_json, participants_json, transition_applied, mode_hash) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (
                str(fixture.request.tx_id), mode, json.dumps(list(snapshot)),
                json.dumps([str(w) for w in workers]), mode_hash,
            ),
        )
    return mode_hash


def _binding(store: SQLiteProtocolStore, tx_id: Any) -> dict[str, Any] | None:
    row = store.connection.execute(
        "SELECT mode, snapshot_json, participants_json, transition_applied, mode_hash "
        "FROM vi_f_mode_bindings WHERE tx_id=?", (str(tx_id),)
    ).fetchone()
    if row is None:
        return None
    return {
        "mode": row[0],
        "snapshot": tuple(json.loads(row[1])),
        "participants": tuple(json.loads(row[2])),
        "transition_applied": bool(row[3]),
        "mode_hash": row[4],
    }


def _semantic_transition_count(store: SQLiteProtocolStore, tx_id: Any) -> int:
    row = store.connection.execute(
        "SELECT COUNT(*) FROM vi_f_semantic_transitions WHERE tx_id=?", (str(tx_id),)
    ).fetchone()
    return int(row[0])


def _apply_calibration_transition_once(store: SQLiteProtocolStore, tx_id: Any) -> bool:
    bind = _binding(store, tx_id)
    if bind is None:
        raise VIFinalError("MISSING_MODE_BINDING")
    if bind["mode"] != "calibration":
        return False
    if bind["transition_applied"]:
        return False
    participants = tuple(bind["participants"])
    snapshot = tuple(int(x) for x in bind["snapshot"])
    with store.transaction():
        for worker, old_epoch in zip(participants, snapshot):
            existing = store.connection.execute(
                "SELECT 1 FROM vi_f_semantic_transitions WHERE tx_id=? AND worker_id=?",
                (str(tx_id), worker),
            ).fetchone()
            if existing is not None:
                continue
            current = store.connection.execute(
                "SELECT learning_epoch FROM vi_f_semantic_state WHERE worker_id=?", (worker,)
            ).fetchone()
            if current is None or int(current[0]) != int(old_epoch):
                raise VIFinalError(f"SEMANTIC_EPOCH_MISMATCH:{worker}")
            store.connection.execute(
                "INSERT INTO vi_f_semantic_transitions(tx_id, worker_id, from_epoch, to_epoch) VALUES (?, ?, ?, ?)",
                (str(tx_id), worker, int(old_epoch), int(old_epoch) + 1),
            )
            store.connection.execute(
                "UPDATE vi_f_semantic_state SET learning_epoch=? WHERE worker_id=?",
                (int(old_epoch) + 1, worker),
            )
        store.connection.execute(
            "UPDATE vi_f_mode_bindings SET transition_applied=1 WHERE tx_id=?", (str(tx_id),)
        )
    return True


def _recover_semantics(store: SQLiteProtocolStore, fixture: Any) -> dict[str, Any]:
    record = store.lookup_by_tx_id(fixture.request.tx_id)
    bind = _binding(store, fixture.request.tx_id)
    applied_now = False
    if record is not None and record.state == ProtocolState.COMMITTED and bind is not None and bind["mode"] == "calibration":
        applied_now = _apply_calibration_transition_once(store, fixture.request.tx_id)
    return {
        "mode": None if bind is None else bind["mode"],
        "transition_applied_now": applied_now,
        "semantic_transition_count": _semantic_transition_count(store, fixture.request.tx_id),
        "semantic_delivery": "NONE" if bind is not None and bind["mode"] == "calibration" else "ORDINARY_OUTPUT",
    }


def _augment_legacy(row: dict[str, Any], mode: str, expected_transition_count: int = 0) -> dict[str, Any]:
    row = dict(row)
    row["study_id"] = STUDY_ID
    row["task_mode"] = mode
    row["semantic_transition_count"] = expected_transition_count
    row["semantic_reputation_changed"] = expected_transition_count > 0
    row["ordinary_persistent_reputation_update"] = False if mode == "ordinary" else None
    row["calibration_persistent_transition"] = True if mode == "calibration" else None
    row["core_refresh_logs_are_semantic_learning"] = False
    return row


def _calibration_predecision(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "calibration_crash_before_decision"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "calibration")
        context = dict(fixture.context)
        context["crash_after_activation"] = True
        first = ProtocolEngine(store).handle(fixture.request, context=context)
        before = legacy.consume_activated_preprocessing(store, fixture.request.tx_id)
        consumed = all(i.lifecycle == PreprocessingLifecycle.CONSUMED for i in before.preprocessing_records)
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem = _recover_semantics(reopened, fixture)
        t1 = time.perf_counter_ns()
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        passed = all((
            first.state == ProtocolState.EXECUTING,
            consumed,
            recovered.response == ProtocolResponse.ABORTED,
            after is not None and after.state == ProtocolState.EXECUTING,
            sem["semantic_transition_count"] == 0,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "calibration", "mode_hash": mode_hash,
            "fault_boundary": "Calibration Executing after durable activation; before Prepared/Decision",
            "observed_result": "Calibration predecision recovery aborted/restartable; no semantic transition installed",
            "semantic_transition_count": sem["semantic_transition_count"],
            "semantic_delivery": sem["semantic_delivery"],
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _calibration_prepared(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "calibration_crash_after_prepared"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "calibration")
        prepared = legacy.prepare_only(store, fixture)
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem = _recover_semantics(reopened, fixture)
        t1 = time.perf_counter_ns()
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        passed = all((
            prepared.state == ProtocolState.PREPARED,
            recovered.response == ProtocolResponse.ABORTED,
            after is not None and after.state == ProtocolState.ABORTED,
            sem["semantic_transition_count"] == 0,
            counts["decision_count"] == 1,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "calibration", "mode_hash": mode_hash,
            "fault_boundary": "Calibration Prepared persisted; no durable Decision",
            "observed_result": "Calibration Prepared-without-decision recovered to Abort; no semantic transition installed",
            "semantic_transition_count": sem["semantic_transition_count"],
            "semantic_delivery": sem["semantic_delivery"],
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _calibration_postcommit(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "calibration_crash_after_commit"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "calibration")
        terminal, _ = legacy.persist_committed_without_log_repair(store, fixture)
        pre_sem = _semantic_transition_count(store, fixture.request.tx_id)
        pre_counts = legacy.db_counts(store, fixture.request.tx_id)
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem1 = _recover_semantics(reopened, fixture)
        sem2 = _recover_semantics(reopened, fixture)
        t1 = time.perf_counter_ns()
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        transitions = _semantic_transition_count(reopened, fixture.request.tx_id)
        workers = len(fixture.certificate.participant_ids)
        passed = all((
            terminal.state == ProtocolState.COMMITTED,
            pre_sem == 0,
            pre_counts["committed_log_count"] == 0,
            recovered.response == ProtocolResponse.COMMITTED,
            transitions == workers,
            sem1["transition_applied_now"] is True,
            sem2["transition_applied_now"] is False,
            counts["decision_count"] == 1,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "calibration", "mode_hash": mode_hash,
            "fault_boundary": "Calibration Commit durable; core logs and semantic calibration transition not yet materialized",
            "observed_result": "Commit preserved; core logs repaired and calibration transition installed exactly once after reopen",
            "semantic_transition_count": transitions,
            "semantic_transition_expected": workers,
            "semantic_delivery": "NONE",
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _calibration_duplicate(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "calibration_duplicate_replay"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "calibration")
        legacy.create_fully_committed(store, fixture)
        _recover_semantics(store, fixture)
        before = _semantic_transition_count(store, fixture.request.tx_id)
        before_counts = legacy.db_counts(store, fixture.request.tx_id)
        t0 = time.perf_counter_ns()
        replay = ProtocolEngine(store).handle(fixture.request, context={})
        sem = _recover_semantics(store, fixture)
        t1 = time.perf_counter_ns()
        after = _semantic_transition_count(store, fixture.request.tx_id)
        after_counts = legacy.db_counts(store, fixture.request.tx_id)
        workers = len(fixture.certificate.participant_ids)
        passed = all((
            replay.response == ProtocolResponse.COMMITTED,
            before == workers, after == workers,
            sem["transition_applied_now"] is False,
            before_counts == after_counts,
            not after_counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "calibration", "mode_hash": mode_hash,
            "fault_boundary": "Exact authenticated replay after calibration Commit",
            "observed_result": "Calibration replay stable; semantic transition remains exactly once",
            "semantic_transition_count": after, "semantic_transition_expected": workers,
            "resolution_time_ns": t1-t0, "resolution_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(after_counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        return row
    finally:
        store.close()


def _stale_snapshot(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "stale_ordinary_snapshot_after_calibration"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    cal_fixture = legacy.make_fixture("stale-calibration", replicate)
    ord_fixture = legacy.make_fixture("stale-ordinary", replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        _bind_mode(store, cal_fixture, "calibration")
        legacy.create_fully_committed(store, cal_fixture)
        _recover_semantics(store, cal_fixture)
        current = _current_epochs(store, tuple(cal_fixture.certificate.participant_ids))
        stale_snapshot = tuple(ord_fixture.certificate.reputation_epoch_snapshot)
        rejected = False
        reason = None
        t0 = time.perf_counter_ns()
        try:
            _bind_mode(store, ord_fixture, "ordinary", stale_snapshot)
        except StaleSnapshot as exc:
            rejected = True
            reason = str(exc)
        t1 = time.perf_counter_ns()
        stale_counts = legacy.db_counts(store, ord_fixture.request.tx_id)
        passed = all((
            current == tuple(x + 1 for x in stale_snapshot),
            rejected,
            stale_counts["transaction_count"] == 0,
            stale_counts["decision_count"] == 0,
            not stale_counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "cross-epoch",
            "fault_boundary": "Calibration Commit advanced authenticated learning epoch; ordinary request retained old snapshot",
            "observed_result": "Stale ordinary epoch-bound request rejected before protocol mutation",
            "stale_snapshot": list(stale_snapshot), "current_snapshot": list(current),
            "rejection_reason": reason, "semantic_transition_count": _semantic_transition_count(store, cal_fixture.request.tx_id),
            "resolution_time_ns": t1-t0, "resolution_time_seconds": (t1-t0)/1e9,
            "state_duplicated": False, "passed": passed, "status": "success" if passed else "failed",
        })
        return row
    finally:
        store.close()


def _threshold_boundary(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    row = legacy.run_threshold_boundary(db_path, replicate, warmup=warmup)
    qout_ok = output_prepare_quorum(Q_OUT, Q_OUT)
    qout_fail = output_prepare_quorum(Q_OUT - 1, Q_OUT)
    restart_current = restart_admissibility_quorum(Q_OUT, Q_OUT, True, True)
    restart_stale = restart_admissibility_quorum(Q_OUT, Q_OUT, False, True)
    committed_within = committed_state_availability(Q_OUT, Q_OUT, F_REP, F_OUT, T)
    committed_beyond_rep = committed_state_availability(Q_OUT, Q_OUT, F_REP + 1, F_OUT, T)
    committed_beyond_out = committed_state_availability(Q_OUT, Q_OUT, F_REP, F_OUT + 1, T)
    extra_pass = all((
        qout_ok.satisfied, not qout_fail.satisfied,
        restart_current.satisfied, not restart_stale.satisfied,
        committed_within, not committed_beyond_rep, not committed_beyond_out,
    ))
    row = _augment_legacy(row, "mode-independent")
    row.update({
        "case_id": "threshold_availability_boundary",
        "paper_case_source": "arithmetic T/T-1 plus operation-specific availability predicates",
        "Q_out": Q_OUT, "f_rep": F_REP, "f_out": F_OUT,
        "qout_satisfied": qout_ok.satisfied, "qout_minus_1_satisfied": qout_fail.satisfied,
        "restart_current_satisfied": restart_current.satisfied,
        "restart_stale_satisfied": restart_stale.satisfied,
        "committed_within_budget": committed_within,
        "committed_beyond_rep_budget": committed_beyond_rep,
        "committed_beyond_out_budget": committed_beyond_out,
        "observed_result": "T arithmetic continued and T-1 rejected; output/restart/committed-state predicates respected distinct thresholds",
        "passed": bool(row.get("passed")) and extra_pass,
        "status": "success" if bool(row.get("passed")) and extra_pass else "failed",
    })
    return row


def _ordinary_predecision(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "ordinary_crash_before_decision"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "ordinary")
        workers = tuple(fixture.certificate.participant_ids)
        epoch_before = _current_epochs(store, workers)
        context = dict(fixture.context)
        context["crash_after_activation"] = True
        first = ProtocolEngine(store).handle(fixture.request, context=context)
        before = legacy.consume_activated_preprocessing(store, fixture.request.tx_id)
        pre_consumed = bool(before.preprocessing_records) and all(
            item.lifecycle == PreprocessingLifecycle.CONSUMED for item in before.preprocessing_records
        )
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem = _recover_semantics(reopened, fixture)
        epoch_after = _current_epochs(reopened, workers)
        t1 = time.perf_counter_ns()
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        post_consumed = bool(after and after.preprocessing_records) and all(
            item.lifecycle == PreprocessingLifecycle.CONSUMED for item in after.preprocessing_records
        )
        passed = all((
            first.state == ProtocolState.EXECUTING,
            pre_consumed,
            recovered.response == ProtocolResponse.ABORTED,
            after is not None and after.state == ProtocolState.EXECUTING,
            post_consumed,
            sem["semantic_transition_count"] == 0,
            epoch_after == epoch_before,
            counts["decision_count"] == 0,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "ordinary", "mode_hash": mode_hash,
            "fault_boundary": "Ordinary Executing after durable activation; before Prepared/Decision",
            "observed_result": "Ordinary predecision recovery aborted/restartable; semantic reputation epoch unchanged",
            "semantic_epoch_before": list(epoch_before), "semantic_epoch_after": list(epoch_after),
            "semantic_transition_count": sem["semantic_transition_count"],
            "semantic_reputation_changed": False,
            "ordinary_persistent_reputation_update": False,
            "core_refresh_logs_are_semantic_learning": False,
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _ordinary_prepared(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "ordinary_crash_after_prepared"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "ordinary")
        workers = tuple(fixture.certificate.participant_ids)
        epoch_before = _current_epochs(store, workers)
        prepared = legacy.prepare_only(store, fixture)
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem = _recover_semantics(reopened, fixture)
        epoch_after = _current_epochs(reopened, workers)
        t1 = time.perf_counter_ns()
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        passed = all((
            prepared.state == ProtocolState.PREPARED,
            recovered.response == ProtocolResponse.ABORTED,
            after is not None and after.state == ProtocolState.ABORTED,
            sem["semantic_transition_count"] == 0,
            epoch_after == epoch_before,
            counts["decision_count"] == 1,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "ordinary", "mode_hash": mode_hash,
            "fault_boundary": "Ordinary Prepared persisted; no durable Decision",
            "observed_result": "Ordinary Prepared-without-decision recovered to Abort; semantic reputation epoch unchanged",
            "semantic_epoch_before": list(epoch_before), "semantic_epoch_after": list(epoch_after),
            "semantic_transition_count": sem["semantic_transition_count"],
            "semantic_reputation_changed": False,
            "ordinary_persistent_reputation_update": False,
            "core_refresh_logs_are_semantic_learning": False,
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _ordinary_postcommit(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "ordinary_crash_after_commit"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "ordinary")
        workers = tuple(fixture.certificate.participant_ids)
        epoch_before = _current_epochs(store, workers)
        terminal, _ = legacy.persist_committed_without_log_repair(store, fixture)
        pre_counts = legacy.db_counts(store, fixture.request.tx_id)
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem1 = _recover_semantics(reopened, fixture)
        sem2 = _recover_semantics(reopened, fixture)
        epoch_after = _current_epochs(reopened, workers)
        t1 = time.perf_counter_ns()
        counts = legacy.db_counts(reopened, fixture.request.tx_id)
        expected_logs = len(workers)
        passed = all((
            terminal.state == ProtocolState.COMMITTED,
            pre_counts["committed_log_count"] == 0,
            recovered.response == ProtocolResponse.COMMITTED,
            counts["decision_count"] == 1,
            counts["committed_log_count"] == expected_logs,
            sem1["semantic_transition_count"] == 0,
            sem2["semantic_transition_count"] == 0,
            epoch_after == epoch_before,
            not counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "ordinary", "mode_hash": mode_hash,
            "fault_boundary": "Ordinary Commit durable; committed-log side table not yet repaired",
            "observed_result": "Ordinary Commit preserved and core logs repaired idempotently; semantic reputation epoch unchanged",
            "semantic_epoch_before": list(epoch_before), "semantic_epoch_after": list(epoch_after),
            "semantic_transition_count": 0, "semantic_reputation_changed": False,
            "ordinary_persistent_reputation_update": False,
            "core_refresh_log_count_after_recovery": counts["committed_log_count"],
            "core_refresh_logs_are_semantic_learning": False,
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _ordinary_delivery(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "ordinary_postcommit_delivery_failure"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "ordinary")
        workers = tuple(fixture.certificate.participant_ids)
        epoch_before = _current_epochs(store, workers)
        unavailable = legacy.OutputUnavailable(tx_id=fixture.request.tx_id, message="simulated postcommit delivery unavailable")
        terminal, _ = legacy.persist_committed_without_log_repair(
            store, fixture, delivery=DeliveryState.OUTPUT_UNAVAILABLE, output_unavailable=unavailable
        )
        store.repair_committed_logs(fixture.request.tx_id)
        before_counts = legacy.db_counts(store, fixture.request.tx_id)
        stable_before = store.lookup_by_tx_id(fixture.request.tx_id)
        prepare_hash_before = stable_before.output_prepare.record_hash() if stable_before and stable_before.output_prepare else None
        store.close()
        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        replay = ProtocolEngine(reopened).handle(fixture.request, context={})
        sem = _recover_semantics(reopened, fixture)
        epoch_after = _current_epochs(reopened, workers)
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        post_counts = legacy.db_counts(reopened, fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        prepare_hash_after = after.output_prepare.record_hash() if after and after.output_prepare else None
        passed = all((
            terminal.state == ProtocolState.COMMITTED,
            replay.response == ProtocolResponse.COMMITTED,
            replay.delivery == DeliveryState.OUTPUT_UNAVAILABLE,
            prepare_hash_before is not None and prepare_hash_before == prepare_hash_after,
            before_counts == post_counts,
            sem["semantic_transition_count"] == 0,
            epoch_after == epoch_before,
            not post_counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "ordinary", "mode_hash": mode_hash,
            "fault_boundary": "Ordinary Commit durable; requester delivery unavailable; close/reopen then replay",
            "observed_result": "Ordinary Commit and delivery metadata preserved; semantic reputation epoch unchanged",
            "semantic_epoch_before": list(epoch_before), "semantic_epoch_after": list(epoch_after),
            "semantic_transition_count": 0, "semantic_reputation_changed": False,
            "ordinary_persistent_reputation_update": False,
            "application_output_payload_modeled": False,
            "payload_retransmission_equality_claim": False,
            "core_refresh_logs_are_semantic_learning": False,
            "recovery_time_ns": t1-t0, "recovery_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(post_counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        reopened.close()
        return row
    finally:
        try: store.close()
        except Exception: pass


def _ordinary_duplicate(db_path: Path, replicate: int, warmup: bool) -> dict[str, Any]:
    case_id = "ordinary_duplicate_replay"
    row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = legacy.make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = legacy.sqlite_flags(store)
        mode_hash = _bind_mode(store, fixture, "ordinary")
        workers = tuple(fixture.certificate.participant_ids)
        epoch_before = _current_epochs(store, workers)
        before = legacy.create_fully_committed(store, fixture)
        before_fp = legacy.record_fingerprint(before)
        before_counts = legacy.db_counts(store, fixture.request.tx_id)
        t0 = time.perf_counter_ns()
        replay = ProtocolEngine(store).handle(fixture.request, context={})
        sem = _recover_semantics(store, fixture)
        epoch_after = _current_epochs(store, workers)
        after = store.lookup_by_tx_id(fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        after_counts = legacy.db_counts(store, fixture.request.tx_id)
        passed = all((
            replay.response == ProtocolResponse.COMMITTED,
            before_fp == legacy.record_fingerprint(after),
            before_counts == after_counts,
            sem["semantic_transition_count"] == 0,
            epoch_after == epoch_before,
            not after_counts["state_duplicated"],
        ))
        row.update({
            **flags, "study_id": STUDY_ID, "task_mode": "ordinary", "mode_hash": mode_hash,
            "fault_boundary": "Exact authenticated replay after ordinary Commit",
            "observed_result": "Ordinary replay stable; semantic reputation epoch unchanged",
            "semantic_epoch_before": list(epoch_before), "semantic_epoch_after": list(epoch_after),
            "semantic_transition_count": 0, "semantic_reputation_changed": False,
            "ordinary_persistent_reputation_update": False,
            "core_refresh_logs_are_semantic_learning": False,
            "resolution_time_ns": t1-t0, "resolution_time_seconds": (t1-t0)/1e9,
            "state_duplicated": bool(after_counts["state_duplicated"]), "passed": passed,
            "status": "success" if passed else "failed",
        })
        return row
    finally:
        store.close()

def run_case(case_id: str, db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    try:
        if case_id == "ordinary_crash_before_decision":
            return _ordinary_predecision(db_path, replicate, warmup)
        if case_id == "ordinary_crash_after_prepared":
            return _ordinary_prepared(db_path, replicate, warmup)
        if case_id == "ordinary_crash_after_commit":
            return _ordinary_postcommit(db_path, replicate, warmup)
        if case_id == "ordinary_postcommit_delivery_failure":
            return _ordinary_delivery(db_path, replicate, warmup)
        if case_id == "ordinary_duplicate_replay":
            return _ordinary_duplicate(db_path, replicate, warmup)
        if case_id == "conflicting_replay":
            row = legacy.run_case("conflicting_replay", db_path, replicate, warmup=warmup)
            row = _augment_legacy(row, "binding-check", 0)
            row["case_id"] = case_id
            return row
        if case_id == "calibration_crash_before_decision":
            return _calibration_predecision(db_path, replicate, warmup)
        if case_id == "calibration_crash_after_prepared":
            return _calibration_prepared(db_path, replicate, warmup)
        if case_id == "calibration_crash_after_commit":
            return _calibration_postcommit(db_path, replicate, warmup)
        if case_id == "calibration_duplicate_replay":
            return _calibration_duplicate(db_path, replicate, warmup)
        if case_id == "stale_ordinary_snapshot_after_calibration":
            return _stale_snapshot(db_path, replicate, warmup)
        if case_id == "threshold_availability_boundary":
            return _threshold_boundary(db_path, replicate, warmup)
        raise VIFinalError(f"UNKNOWN_CASE:{case_id}")
    except Exception as exc:
        row = legacy.result_base(case_id, replicate, db_path, warmup=warmup)
        row.update({
            "study_id": STUDY_ID,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "observed_result": f"{exc.__class__.__name__}: {exc}",
            "passed": False,
            "status": "failed",
        })
        return row


def _cleanup_sqlite(path: Path) -> None:
    for p in (path, Path(str(path)+"-wal"), Path(str(path)+"-shm")):
        if p.exists():
            p.unlink()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo]*(hi-pos) + xs[hi]*(pos-lo)


def _timing(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median_seconds":None,"mean_seconds":None,"p25_seconds":None,"p75_seconds":None,"p95_seconds":None}
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "p25_seconds": _percentile(values, .25),
        "p75_seconds": _percentile(values, .75),
        "p95_seconds": _percentile(values, .95),
    }


def _source_binding(root: Path) -> dict[str, Any]:
    hashes = {}
    for rel in SOURCE_FILES:
        p = root / rel
        if not p.is_file():
            raise VIFinalError(f"MISSING_REQUIRED_SOURCE:{rel}")
        hashes[rel] = _sha(p)
    payload = {
        "study_id": STUDY_ID,
        "git_head": _git_head(root),
        "source_hashes": hashes,
        "case_ids": list(CASE_IDS),
        "warmups": WARMUPS,
        "measured": MEASURED,
        "N": N, "T": T, "Q_out": Q_OUT, "f_rep": F_REP, "f_out": F_OUT,
    }
    payload["binding_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def validate(root: Path) -> dict[str, Any]:
    binding = _source_binding(root)
    with tempfile.TemporaryDirectory(prefix="vi-f-final-validate-") as tmp:
        td = Path(tmp)
        diagnostics = {}
        for idx, case_id in enumerate(CASE_IDS, 1):
            path = td / f"{idx:02d}_{case_id}.sqlite3"
            row = run_case(case_id, path, 997, warmup=True)
            diagnostics[case_id] = {"passed": bool(row.get("passed")), "observed_result": row.get("observed_result")}
            if not row.get("passed"):
                raise VIFinalError(f"VALIDATION_CASE_FAILED:{case_id}:{row.get('observed_result')}")
    return {"binding": binding, "diagnostics": diagnostics}


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")
        f.flush(); os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize(root: Path, binding: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for cid in CASE_IDS:
        rs = [r for r in rows if r["case_id"] == cid]
        times = [float(r["recovery_time_seconds"]) for r in rs if r.get("recovery_time_seconds") is not None]
        cases.append({
            "case_id": cid,
            "paper_label": PAPER_LABELS[cid],
            "expected_result": EXPECTED[cid],
            "observed_result": rs[0].get("observed_result") if rs else None,
            "n": len(rs),
            "success": sum(bool(r.get("passed")) for r in rs),
            "failure": sum(not bool(r.get("passed")) for r in rs),
            "all_pass": len(rs)==MEASURED and all(bool(r.get("passed")) for r in rs),
            "state_duplicated_any": any(bool(r.get("state_duplicated")) for r in rs),
            "timing": _timing(times),
        })
    status = "complete" if len(rows)==len(CASE_IDS)*MEASURED and all(c["all_pass"] for c in cases) else "failed"
    return {
        "schema_version":"1.0", "study_id":STUDY_ID, "status":status,
        "binding_sha256":binding["binding_sha256"], "git_head":binding["git_head"],
        "case_count":len(CASE_IDS), "warmups_per_case":WARMUPS, "measured_per_case":MEASURED,
        "planned_measured_case_replicates":len(CASE_IDS)*MEASURED,
        "recorded_measured_case_replicates":len(rows),
        "success":sum(bool(r.get("passed")) for r in rows),
        "failure":sum(not bool(r.get("passed")) for r in rows),
        "state_duplication_cases":sum(c["state_duplicated_any"] for c in cases),
        "sqlite_profile":{"journal_mode":"WAL","synchronous":"FULL","foreign_keys":"ON"},
        "threshold_profile":{"N":N,"T":T,"Q_out":Q_OUT,"f_rep":F_REP,"f_out":F_OUT,"backend":"simulated_shamir"},
        "final_semantics":{
            "ordinary_semantic_reputation_transition":False,
            "calibration_semantic_reputation_transition":True,
            "calibration_truth_delivery":False,
            "stale_epoch_bound_request_rejected":True,
            "core_refresh_log_distinguished_from_semantic_learning":True,
        },
        "scope":{
            "single_machine_sqlite_wal":True,
            "mode_aware_evaluation_adapter":True,
            "protocol_v2_core_modified":False,
            "distributed_e8b_used":False,
            "production_mpc_used":False,
            "production_fault_tolerance_claim":False,
        },
        "limitations":{
            "application_output_payload_bytes_persisted_by_protocol_v2":False,
            "byte_identical_retransmission_measured":False,
            "distributed_network_failures_measured":False,
            "semantic_mode_epoch_guard_is_experiment_adapter_over_frozen_protocol_v2":True,
        },
        "cases":cases,
        "raw_sha256":_sha(root/RAW_REL),
    }


def _write_csvs(root: Path, summary: dict[str, Any]) -> None:
    case_path = root/CASE_CSV_REL; case_path.parent.mkdir(parents=True, exist_ok=True)
    with case_path.open("w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(["case_id","paper_label","n","success","failure","state_duplicated_any","median_recovery_ms","p95_recovery_ms"])
        for c in summary["cases"]:
            med=c["timing"]["median_seconds"]; p95=c["timing"]["p95_seconds"]
            w.writerow([c["case_id"],c["paper_label"],c["n"],c["success"],c["failure"],c["state_duplicated_any"],None if med is None else med*1000,None if p95 is None else p95*1000])
    lat_path=root/LATENCY_CSV_REL; lat_path.parent.mkdir(parents=True, exist_ok=True)
    with lat_path.open("w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(["case_id","median_ms","mean_ms","p25_ms","p75_ms","p95_ms"])
        for c in summary["cases"]:
            t=c["timing"]
            if t["median_seconds"] is not None:
                w.writerow([c["case_id"]]+[None if t[k] is None else t[k]*1000 for k in ["median_seconds","mean_seconds","p25_seconds","p75_seconds","p95_seconds"]])
    semantic = {
        "study_id":STUDY_ID,
        "ordinary_cases":[c for c in CASE_IDS if c.startswith("ordinary_")],
        "calibration_cases":[c for c in CASE_IDS if c.startswith("calibration_")],
        "ordinary_semantic_transition_expected":0,
        "calibration_commit_semantic_transition_expected":"exactly once per participant",
        "stale_snapshot_case":"stale_ordinary_snapshot_after_calibration",
        "threshold_case":"threshold_availability_boundary",
        "all_cases_pass":summary["status"]=="complete",
        "state_duplication_cases":summary["state_duplication_cases"],
        "scope":summary["scope"],
        "limitations":summary["limitations"],
    }
    (root/SEMANTIC_REL).write_text(json.dumps(semantic,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def execute(root: Path, resume: bool=False) -> dict[str, Any]:
    val=validate(root); binding=val["binding"]
    raw=root/RAW_REL; summary_path=root/SUMMARY_REL
    if not resume and (raw.exists() or summary_path.exists()):
        raise VIFinalError("VI_F_FINAL_RESULT_ALREADY_EXISTS")
    rows=_read_jsonl(raw) if resume else []
    seen={(r["case_id"],int(r["replicate_index"])) for r in rows}
    if not rows:
        with tempfile.TemporaryDirectory(prefix="vi-f-final-warmup-") as tmp:
            td=Path(tmp)
            for idx,cid in enumerate(CASE_IDS,1):
                for rep in range(1,WARMUPS+1):
                    r=run_case(cid,td/f"{idx:02d}_{cid}_w{rep:02d}.sqlite3",rep,warmup=True)
                    if not r.get("passed"):
                        raise VIFinalError(f"WARMUP_FAILED:{cid}:{rep}:{r.get('observed_result')}")
    for idx,cid in enumerate(CASE_IDS,1):
        for rep in range(1,MEASURED+1):
            if (cid,rep) in seen: continue
            db=root/RUN_ROOT_REL/f"{idx:02d}_{cid}_r{rep:02d}.sqlite3"
            db.parent.mkdir(parents=True,exist_ok=True); _cleanup_sqlite(db)
            r=run_case(cid,db,rep,warmup=False)
            r.update({"formal_experiment":True,"binding_sha256":binding["binding_sha256"]})
            _append_jsonl(raw,r); rows.append(r); seen.add((cid,rep))
    summary=_summarize(root,binding,rows)
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _write_csvs(root,summary)
    return summary


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",default=".")
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv); root=Path(a.repo_root).resolve()
    try:
        if a.validate_only:
            v=validate(root)
            print("VI_F_FINAL_SEMANTICS_VALIDATE=PASS")
            print(f"CASE_COUNT={len(CASE_IDS)}")
            print(f"MEASURED_PER_CASE={MEASURED}")
            print(f"PLANNED_MEASURED_CASE_REPLICATES={len(CASE_IDS)*MEASURED}")
            print("ORDINARY_SEMANTIC_REPUTATION_UPDATE=DISABLED")
            print("CALIBRATION_SEMANTIC_REPUTATION_UPDATE=ENABLED")
            print("STALE_SNAPSHOT_GUARD=ENABLED")
            print(f"T={T}")
            print(f"Q_OUT={Q_OUT}")
            print("DISTRIBUTED_E8B_USED=NO")
            print("PRODUCTION_MPC_USED=NO")
            print(f"BINDING_SHA256={v['binding']['binding_sha256']}")
            return 0
        s=execute(root,resume=a.resume)
        print(f"VI_F_FINAL_SEMANTICS_RUN={s['status'].upper()}")
        print(f"CASE_COUNT={s['case_count']}")
        print(f"MEASURED_CASE_REPLICATES={s['recorded_measured_case_replicates']}")
        print(f"SUCCESS={s['success']}")
        print(f"FAILURE={s['failure']}")
        print(f"STATE_DUPLICATION_CASES={s['state_duplication_cases']}")
        print("DISTRIBUTED_E8B_USED=NO")
        print("PRODUCTION_MPC_USED=NO")
        print(f"SUMMARY={SUMMARY_REL.as_posix()}")
        return 0 if s["status"]=="complete" else 1
    except Exception as exc:
        print(f"VI_F_FINAL_ERROR={exc}",file=os.sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
