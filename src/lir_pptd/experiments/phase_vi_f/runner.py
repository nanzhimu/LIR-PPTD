from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from lir_pptd.fixedpoint import default_profile
from lir_pptd.identifiers import AttemptID, TaskID, TransactionID, WorkerID
from lir_pptd.mpc import SimulatedShamirBackend
from lir_pptd.mpc.simulated_shamir import SimulatedBackendProfile
from lir_pptd.mpc.types import ShareBundle
from lir_pptd.protocol_v2 import (
    CommittedLogEntry,
    DeliveryState,
    OperationKind,
    OutputUnavailable,
    PreprocessingLifecycle,
    ProtocolEngine,
    ProtocolRecord,
    ProtocolResponse,
    ProtocolState,
    SQLiteProtocolStore,
    build_abort_auth,
    build_commit_certificate,
    build_participant_certificate,
    build_transaction_request,
    canonical_part_hash,
    fixed_attempt_continuation_quorum,
)

DESIGN_REL = Path("configs/phase_vi_f/vi_f_formal_design.json")
BINDING_REL = Path("results/summary/vi_f_start_binding.json")
APPROVAL_REL = Path("approvals/phase_vi_f/VI_F_START_APPROVAL.json")
RAW_REL = Path("results/raw/vi_f/availability_recovery_trials.jsonl")
CHECKPOINT_REL = Path("results/summary/vi_f_checkpoint.json")
SUMMARY_REL = Path("results/summary/vi_f_formal_summary.json")
TABLE_REL = Path("results/summary/vi_f_paper_table.tex")

CASE_IDS = (
    "crash_before_durable_decision",
    "crash_after_prepared",
    "crash_after_commit",
    "duplicate_replay",
    "conflicting_replay",
    "postcommit_delivery_failure",
    "arithmetic_boundary_t_vs_tminus1",
)
WARMUPS = 5
MEASURED = 30
N = 10
T = 4

EXPECTED_HASHES = {
    "src/lir_pptd/protocol_v2/model.py": "b3078280f902e400a7bb923f08cef3c545f82db1b9647bc5784f1315d17488f7",
    "src/lir_pptd/protocol_v2/binding.py": "11e330aee0ff08a259557c1fb773080b0e699ca28e10c380b17f990594f828bf",
    "src/lir_pptd/protocol_v2/durable_store.py": "71dd05eacb06c6db699d06952e990dc0465cb1b76ef0c32594942322dff77ebe",
    "src/lir_pptd/protocol_v2/engine.py": "522a345e8112b1bff94bb0b8a9b6487a6ed381f6eb155e2e0e2b3cd96bfbfd5b",
    "src/lir_pptd/protocol_v2/recovery.py": "d22555a378c67c8ab490c458417efe020157c90023e08efae5c2cf2f4113d1a6",
    "src/lir_pptd/protocol_v2/locking.py": "bf7741c2b7ac5e861ac96381eda08bb4ff2ddf838ad86c3b6b3e4931a52ad3f1",
    "src/lir_pptd/protocol_v2/quorum.py": "8cc1dce428842d5f2302f9ecc96b0618d76d0c529fb76bd3061fc9af3b26b381",
    "src/lir_pptd/protocol_v2/decision.py": "7be7c1e5bbf4170cba4ab1335ddb97039146b68951fd2f4d7799937bdabf016b",
    "src/lir_pptd/protocol_v2/preprocessing.py": "6abf2dfecde1954aca14d4e71af149cdcac04fc5597309b86594575f4135100d",
    "src/lir_pptd/protocol_v2/attempt.py": "13f94cd173d2f03be1cd9728398d34648f97db76e4740e0424b4ec5f851f889c",
    "src/lir_pptd/mpc/simulated_shamir.py": "d0d2b123a872aa2f88aeaec1270ea28dce9ff16adf529ebb151418995967e940",
    "src/lir_pptd/mpc/types.py": "7fd70decd0aea8d4971c76d726abb6824564cb058378ca4fa97e5bfa6ed2bd30",
    "src/lir_pptd/fixedpoint/profile.py": "b9b504c5221d5ef5c67311076db3102cf9cec60fa3a56743fb93cb1e952dbedd",
    "configs/backend_profiles/simulated_shamir.yaml": "8953d5a6c4587f90c23a6b3a31dc3bece7863eb31037769a071587f12d817ab9",
    "docs/PROTOCOL_SPEC.md": "e47428f3b7c8e7f69e555c7ea71fc5bf3d164dce5333d62f42040da3ea286d89",
    "docs/EXPERIMENT_SPEC.md": "fc83881328cdda0de78990afd4b9736bf913d41781293f9744cb5b9317cc56af",
    "docs/PHASE5_V2_FINAL_REPORT.md": "66d670653de7954d227597cb2ad6e14c27ec42f060448dc026a07386d5ab5020",
}


class VIFError(RuntimeError):
    pass


def sha(path: Path) -> str:
    if not path.is_file():
        raise VIFError(f"MISSING_FILE:{path.as_posix()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise VIFError(f"INVALID_JSONL:{path}:{index}") from exc
    return rows


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        raise VIFError(f"GIT_HEAD_FAILED:{exc}") from exc


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def timing_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median_seconds": None,
            "mean_seconds": None,
            "p25_seconds": None,
            "p75_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "p25_seconds": percentile(values, 0.25),
        "p75_seconds": percentile(values, 0.75),
        "p95_seconds": percentile(values, 0.95),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def benchmark_profile() -> SimulatedBackendProfile:
    fixed = default_profile()
    return SimulatedBackendProfile(
        profile_id="vi-f-simulated-shamir-n10-t4-v1",
        field_prime_p=fixed.prime_p,
        threshold_t=T,
        committee_size_n=N,
        server_ids=tuple(f"s{i}" for i in range(1, N + 1)),
        x_coordinates=tuple(range(1, N + 1)),
        fixed_backend_profile_id=fixed.profile_id,
        fixed_backend_profile_hash=str(fixed.digest()),
    )


@dataclass(frozen=True)
class CaseFixture:
    request: Any
    certificate: Any
    context: dict[str, Any]


def make_fixture(case_id: str, replicate: int) -> CaseFixture:
    task_id = TaskID(f"vi-f-{case_id[:16]}-{replicate:02d}")
    participants = (WorkerID("worker-a"), WorkerID("worker-b"))
    storage = (WorkerID("worker-s1"), WorkerID("worker-s2"), WorkerID("worker-s3"))
    certificate = build_participant_certificate(
        task_id=task_id,
        cfg_hash="a" * 64,
        participant_ids=participants,
        participant_count=2,
        dimension=2,
        common_report_storage_set=storage,
        selected_upload_ids=("upload-a", "upload-b"),
        reputation_epoch_snapshot=(0, 0),
        schema_domain_version="phase5-v2",
        requester_attestation="signed",
    )
    part_hash = canonical_part_hash(certificate)
    request = build_transaction_request(
        task_id=task_id,
        cfg_hash="a" * 64,
        part_hash=part_hash,
        attempt_id=AttemptID("attempt-0001"),
        tx_id=TransactionID(f"tx-vif-{case_id[:12]}-{replicate:02d}"),
        requester_id=WorkerID("worker-dr"),
    )
    context = {
        "complete_candidates": 2,
        "minimum_participants": 2,
        "maximum_participants": 4,
        "epoch_snapshot": (0, 0),
        "participant_certificate": certificate,
        "requester_id": request.requester_id,
        "committee_hash": "committee-hash",
        "record_specs": (
            ("record-1", "op-1", OperationKind.OUTPUT_PREPARE),
            ("record-2", "op-2", OperationKind.REPUTATION_PREPARE),
        ),
        "output_storage_set": storage,
        "reputation_storage_set": storage,
        "output_record_id": "output-record",
        "reputation_record_id": "reputation-record",
    }
    return CaseFixture(request=request, certificate=certificate, context=context)


def build_valid_commit_certificate(fixture: CaseFixture, record: ProtocolRecord):
    if record.prepared is None or record.output_prepare is None or record.reputation_prepare is None:
        raise VIFError("COMMIT_CERTIFICATE_REQUIRES_PREPARED_RECORD")
    return build_commit_certificate(
        task_id=fixture.request.task_id,
        tx_id=fixture.request.tx_id,
        attempt_id=fixture.request.attempt_id,
        cfg_hash=fixture.request.cfg_hash,
        part_hash=fixture.request.part_hash,
        epoch_snapshot=(0, 0),
        output_committee=fixture.certificate.common_report_storage_set,
        reputation_committee=fixture.certificate.common_report_storage_set,
        output_prepare_hash=record.output_prepare.record_hash(),
        reputation_prepare_hash=record.reputation_prepare.record_hash(),
        output_manifest_hash=record.prepared.output_manifest_hash,
        reputation_manifest_hash=record.prepared.reputation_manifest_hash,
        requester_id=fixture.request.requester_id,
    )


def committed_logs_for(fixture: CaseFixture, commit_hash: str) -> tuple[CommittedLogEntry, ...]:
    return tuple(
        CommittedLogEntry(
            worker_id=worker_id,
            epoch=1,
            refresh_id=f"refresh-{worker_id}",
            tx_id=fixture.request.tx_id,
            commit_certificate_hash=commit_hash,
        )
        for worker_id in fixture.certificate.participant_ids
    )


def materialized_committed_record(
    fixture: CaseFixture,
    prepared_record: ProtocolRecord,
    commit_hash: str,
    *,
    delivery: DeliveryState = DeliveryState.SECURE_OUTPUT,
    output_unavailable: OutputUnavailable | None = None,
) -> ProtocolRecord:
    return prepared_record.model_copy(
        update={
            "state": ProtocolState.COMMITTED,
            "response": ProtocolResponse.COMMITTED,
            "delivery": delivery,
            "decision_winner": ProtocolResponse.COMMITTED,
            "commit_certificate_hash": commit_hash,
            "committed_epoch_vector": (1, 1),
            "committed_output_share_count": len(fixture.certificate.common_report_storage_set),
            "committed_reputation_share_count": len(fixture.certificate.participant_ids),
            "committed_logs": committed_logs_for(fixture, commit_hash),
            "output_unavailable": output_unavailable,
        }
    )


def sqlite_flags(store: SQLiteProtocolStore) -> dict[str, bool | str | int]:
    journal = str(store.connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    sync = int(store.connection.execute("PRAGMA synchronous").fetchone()[0])
    fk = int(store.connection.execute("PRAGMA foreign_keys").fetchone()[0])
    return {
        "journal_mode": journal,
        "wal_enabled": journal == "wal",
        "synchronous_value": sync,
        "synchronous_full": sync == 2,
        "foreign_keys_value": fk,
        "foreign_keys_on": fk == 1,
    }


def db_counts(store: SQLiteProtocolStore, tx_id: Any) -> dict[str, Any]:
    tx = str(tx_id)
    transaction_count = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM protocol_transactions WHERE tx_id=?", (tx,)
        ).fetchone()[0]
    )
    decision_count = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM protocol_transactions WHERE tx_id=? AND decision_winner IS NOT NULL",
            (tx,),
        ).fetchone()[0]
    )
    terminal_record_count = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM protocol_transactions WHERE tx_id=? AND protocol_state IN ('Aborted','Committed')",
            (tx,),
        ).fetchone()[0]
    )
    committed_log_count = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM protocol_committed_logs WHERE tx_id=?", (tx,)
        ).fetchone()[0]
    )
    duplicate_refresh_rows = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM (SELECT refresh_id, COUNT(*) c FROM protocol_committed_logs WHERE tx_id=? GROUP BY refresh_id HAVING c>1)",
            (tx,),
        ).fetchone()[0]
    )
    duplicate_worker_epoch_rows = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM (SELECT worker_id, epoch, COUNT(*) c FROM protocol_committed_logs WHERE tx_id=? GROUP BY worker_id, epoch HAVING c>1)",
            (tx,),
        ).fetchone()[0]
    )
    return {
        "transaction_count": transaction_count,
        "decision_count": decision_count,
        "terminal_record_count": terminal_record_count,
        "committed_log_count": committed_log_count,
        "duplicate_refresh_rows": duplicate_refresh_rows,
        "duplicate_worker_epoch_rows": duplicate_worker_epoch_rows,
        "state_duplicated": any(
            (
                transaction_count > 1,
                decision_count > 1,
                terminal_record_count > 1,
                duplicate_refresh_rows > 0,
                duplicate_worker_epoch_rows > 0,
            )
        ),
    }


def record_fingerprint(record: ProtocolRecord | None) -> str | None:
    if record is None:
        return None
    return hashlib.sha256(canonical_bytes(record.model_dump(mode="json"))).hexdigest()


def consume_activated_preprocessing(store: SQLiteProtocolStore, tx_id: Any) -> ProtocolRecord:
    """Mirror the frozen Phase-5 E8-A crash harness's durable consume step.

    The current public ``consume_preprocessing_record`` helper updates an obsolete
    ``consumed`` key instead of the model's lifecycle field, so VI-F deliberately
    follows the already-approved E8-A persistence path rather than changing
    ``protocol_v2`` semantics.
    """
    record = store.lookup_by_tx_id(tx_id)
    if record is None:
        raise VIFError("MISSING_EXECUTING_RECORD")
    consumed = tuple(
        item.model_copy(update={"lifecycle": PreprocessingLifecycle.CONSUMED})
        for item in record.preprocessing_records
    )
    with store.transaction():
        store.connection.execute(
            "UPDATE protocol_transactions SET preprocessing_records_json=? WHERE tx_id=?",
            (json.dumps([item.model_dump(mode="json") for item in consumed]), str(tx_id)),
        )
        for item in consumed:
            store.connection.execute(
                "INSERT OR REPLACE INTO protocol_preprocessing_records"
                "(record_id, tx_id, op_id, operation_kind, record_json, consumed) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (
                    item.record_id,
                    str(item.tx_id),
                    item.op_id,
                    item.operation_kind.value,
                    item.model_dump_json(),
                ),
            )
    updated = store.lookup_by_tx_id(tx_id)
    if updated is None:
        raise VIFError("MISSING_EXECUTING_RECORD_AFTER_CONSUME")
    return updated


def result_base(
    case_id: str,
    replicate: int,
    db_path: Path | None,
    *,
    warmup: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "VI-F-RAW-v1",
        "case_id": case_id,
        "replicate_index": replicate,
        "warmup": warmup,
        "sqlite_path": None if db_path is None else str(db_path),
        "status": "failed",
        "passed": False,
        "state_duplicated": False,
        "recovery_time_ns": None,
        "recovery_time_seconds": None,
        "resolution_time_ns": None,
        "resolution_time_seconds": None,
        "production_mpc_used": False,
        "distributed_e8b_used": False,
        "distributed_availability_claim": False,
        "production_fault_tolerance_claim": False,
    }


def run_crash_before_durable_decision(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "crash_before_durable_decision"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        context = dict(fixture.context)
        context["crash_after_activation"] = True
        first = ProtocolEngine(store).handle(fixture.request, context=context)
        before = consume_activated_preprocessing(store, fixture.request.tx_id)
        pre_counts = db_counts(store, fixture.request.tx_id)
        pre_state = before.state.value
        pre_consumed = bool(before.preprocessing_records) and all(
            item.lifecycle == PreprocessingLifecycle.CONSUMED for item in before.preprocessing_records
        )
        store.close()

        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        post_counts = db_counts(reopened, fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        post_consumed = bool(after and after.preprocessing_records) and all(
            item.lifecycle == PreprocessingLifecycle.CONSUMED for item in after.preprocessing_records
        )
        passed = all(
            (
                first.state == ProtocolState.EXECUTING,
                pre_state == ProtocolState.EXECUTING.value,
                pre_consumed,
                recovered.response == ProtocolResponse.ABORTED,
                recovered.state == ProtocolState.ABORTED,
                after is not None and after.state == ProtocolState.EXECUTING,
                post_consumed,
                post_counts["decision_count"] == 0,
                not post_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "Executing after durable activation; before Prepared/Decision",
                "pre_state": pre_state,
                "recovery_response": recovered.response.value,
                "recovery_result_state": recovered.state.value if recovered.state else None,
                "durable_state_after_recovery": after.state.value if after else None,
                "durable_decision_before_crash": None,
                "preprocessing_consumed_before_close": pre_consumed,
                "preprocessing_consumed_after_reopen": post_consumed,
                "decision_count_before_crash": pre_counts["decision_count"],
                "decision_count_after_recovery": post_counts["decision_count"],
                "pre_counts": pre_counts,
                "post_counts": post_counts,
                "state_duplicated": bool(post_counts["state_duplicated"]),
                "recovery_time_ns": t1 - t0,
                "recovery_time_seconds": (t1 - t0) / 1e9,
                "observed_result": (
                    "Recovery returned Aborted; durable record remained Executing for explicit restart; "
                    "activated preprocessing remained consumed"
                ),
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        reopened.close()
        return row
    finally:
        try:
            store.close()
        except Exception:
            pass


def prepare_only(store: SQLiteProtocolStore, fixture: CaseFixture) -> ProtocolRecord:
    context = dict(fixture.context)
    context["prepare_only"] = True
    context["crash_before_decision"] = True
    result = ProtocolEngine(store).handle(fixture.request, context=context)
    record = store.lookup_by_tx_id(fixture.request.tx_id)
    if result.state != ProtocolState.PREPARED or record is None or record.state != ProtocolState.PREPARED:
        raise VIFError("PREPARE_ONLY_DID_NOT_REACH_PREPARED")
    if record.prepared is None or record.output_prepare is None or record.reputation_prepare is None:
        raise VIFError("PREPARED_RECORD_INCOMPLETE")
    return record


def run_crash_after_prepared(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "crash_after_prepared"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        prepared = prepare_only(store, fixture)
        pre_counts = db_counts(store, fixture.request.tx_id)
        store.close()

        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        post_counts = db_counts(reopened, fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        expected_abort_auth = build_abort_auth(
            task_id=fixture.request.task_id,
            tx_id=fixture.request.tx_id,
            attempt_id=fixture.request.attempt_id,
            requester_id=fixture.request.requester_id,
            cfg_hash=fixture.request.cfg_hash,
            part_hash=fixture.request.part_hash,
            prepared_hash=prepared.prepared.prepared_hash(),
            reason_code="RecoveryNoDecision",
        )
        expected_abort_hash = expected_abort_auth.auth_hash()
        passed = all(
            (
                prepared.state == ProtocolState.PREPARED,
                pre_counts["decision_count"] == 0,
                recovered.response == ProtocolResponse.ABORTED,
                recovered.state == ProtocolState.ABORTED,
                after is not None and after.state == ProtocolState.ABORTED,
                after.decision_winner == ProtocolResponse.ABORTED,
                after.abort_auth_hash == expected_abort_hash,
                post_counts["decision_count"] == 1,
                post_counts["terminal_record_count"] == 1,
                not post_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "Prepared persisted; no Decision",
                "pre_state": prepared.state.value,
                "durable_decision_before_crash": None,
                "decision_count_before_crash": pre_counts["decision_count"],
                "pre_counts": pre_counts,
                "recovery_response": recovered.response.value,
                "recovery_result_state": recovered.state.value if recovered.state else None,
                "durable_state_after_recovery": after.state.value if after else None,
                "decision_winner_after_recovery": after.decision_winner.value if after and after.decision_winner else None,
                "abort_auth_present": bool(after and after.abort_auth_hash),
                "recovery_reason": "RecoveryNoDecision",
                "abort_auth_matches_recovery_no_decision": bool(after and after.abort_auth_hash == expected_abort_hash),
                "decision_count_after_recovery": post_counts["decision_count"],
                "post_counts": post_counts,
                "state_duplicated": bool(post_counts["state_duplicated"]),
                "recovery_time_ns": t1 - t0,
                "recovery_time_seconds": (t1 - t0) / 1e9,
                "observed_result": "Prepared-without-decision recovered to one authenticated Abort (RecoveryNoDecision)",
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        reopened.close()
        return row
    finally:
        try:
            store.close()
        except Exception:
            pass


def persist_committed_without_log_repair(
    store: SQLiteProtocolStore,
    fixture: CaseFixture,
    *,
    delivery: DeliveryState = DeliveryState.SECURE_OUTPUT,
    output_unavailable: OutputUnavailable | None = None,
) -> tuple[ProtocolRecord, str]:
    prepared = prepare_only(store, fixture)
    certificate = build_valid_commit_certificate(fixture, prepared)
    commit_hash = certificate.certificate_hash()
    winner = store.record_decision(fixture.request.tx_id, commit_hash, ProtocolResponse.COMMITTED)
    if winner != ProtocolResponse.COMMITTED:
        raise VIFError("COMMIT_DECISION_NOT_WINNER")
    terminal = materialized_committed_record(
        fixture,
        prepared,
        commit_hash,
        delivery=delivery,
        output_unavailable=output_unavailable,
    )
    store.commit_terminal_state(terminal)
    return terminal, commit_hash


def run_crash_after_commit(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "crash_after_commit"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        terminal, _ = persist_committed_without_log_repair(store, fixture)
        pre_counts = db_counts(store, fixture.request.tx_id)
        if pre_counts["committed_log_count"] != 0:
            raise VIFError("LOGS_ALREADY_REPAIRED_BEFORE_CRASH")
        store.close()

        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        recovered = ProtocolEngine(reopened).handle(fixture.request, context={})
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        post_counts = db_counts(reopened, fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        expected_logs = len(fixture.certificate.participant_ids)
        # A second replay must be idempotent and must not add rows.
        ProtocolEngine(reopened).handle(fixture.request, context={})
        after_second_counts = db_counts(reopened, fixture.request.tx_id)
        passed = all(
            (
                terminal.state == ProtocolState.COMMITTED,
                pre_counts["decision_count"] == 1,
                recovered.response == ProtocolResponse.COMMITTED,
                recovered.state == ProtocolState.COMMITTED,
                after is not None and after.state == ProtocolState.COMMITTED,
                after.decision_winner == ProtocolResponse.COMMITTED,
                post_counts["decision_count"] == 1,
                post_counts["committed_log_count"] == expected_logs,
                after_second_counts["committed_log_count"] == expected_logs,
                not post_counts["state_duplicated"],
                not after_second_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "Committed terminal state durable; committed-log side table not yet repaired",
                "pre_state": terminal.state.value,
                "durable_decision_before_crash": ProtocolResponse.COMMITTED.value,
                "decision_count_before_crash": pre_counts["decision_count"],
                "committed_log_count_before_crash": pre_counts["committed_log_count"],
                "pre_counts": pre_counts,
                "recovery_response": recovered.response.value,
                "recovery_result_state": recovered.state.value if recovered.state else None,
                "durable_state_after_recovery": after.state.value if after else None,
                "decision_winner_after_recovery": after.decision_winner.value if after and after.decision_winner else None,
                "decision_count_after_recovery": post_counts["decision_count"],
                "committed_log_count_after_recovery": post_counts["committed_log_count"],
                "post_counts": post_counts,
                "post_second_replay_counts": after_second_counts,
                "expected_committed_log_count": expected_logs,
                "state_duplicated": bool(post_counts["state_duplicated"] or after_second_counts["state_duplicated"]),
                "recovery_time_ns": t1 - t0,
                "recovery_time_seconds": (t1 - t0) / 1e9,
                "observed_result": "Unique Commit preserved; missing committed logs repaired idempotently after reopen",
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        reopened.close()
        return row
    finally:
        try:
            store.close()
        except Exception:
            pass


def create_fully_committed(store: SQLiteProtocolStore, fixture: CaseFixture) -> ProtocolRecord:
    terminal, _ = persist_committed_without_log_repair(store, fixture)
    store.repair_committed_logs(fixture.request.tx_id)
    record = store.lookup_by_tx_id(fixture.request.tx_id)
    if record is None:
        raise VIFError("MISSING_COMMITTED_RECORD")
    return record


def run_duplicate_replay(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "duplicate_replay"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        before = create_fully_committed(store, fixture)
        before_fp = record_fingerprint(before)
        before_counts = db_counts(store, fixture.request.tx_id)
        t0 = time.perf_counter_ns()
        replay = ProtocolEngine(store).handle(fixture.request, context={})
        after = store.lookup_by_tx_id(fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        after_fp = record_fingerprint(after)
        after_counts = db_counts(store, fixture.request.tx_id)
        passed = all(
            (
                replay.response == ProtocolResponse.COMMITTED,
                replay.state == ProtocolState.COMMITTED,
                before_fp == after_fp,
                before_counts == after_counts,
                after_counts["decision_count"] == 1,
                not after_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "exact replay of committed request",
                "pre_state": before.state.value,
                "replay_response": replay.response.value,
                "replay_state": replay.state.value if replay.state else None,
                "record_fingerprint_before": before_fp,
                "record_fingerprint_after": after_fp,
                "record_fingerprint_unchanged": before_fp == after_fp,
                "db_counts_unchanged": before_counts == after_counts,
                "pre_counts": before_counts,
                "post_counts": after_counts,
                "state_duplicated": bool(after_counts["state_duplicated"]),
                "resolution_time_ns": t1 - t0,
                "resolution_time_seconds": (t1 - t0) / 1e9,
                "observed_result": "Identical replay returned stable Committed state with no duplicate durable/log state",
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        return row
    finally:
        store.close()


def run_conflicting_replay(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "conflicting_replay"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        before = create_fully_committed(store, fixture)
        before_fp = record_fingerprint(before)
        before_counts = db_counts(store, fixture.request.tx_id)
        conflicting = fixture.request.model_copy(update={"attempt_id": AttemptID("attempt-conflicting")})
        t0 = time.perf_counter_ns()
        replay = ProtocolEngine(store).handle(conflicting, context={})
        t1 = time.perf_counter_ns()
        after = store.lookup_by_tx_id(fixture.request.tx_id)
        after_fp = record_fingerprint(after)
        after_counts = db_counts(store, fixture.request.tx_id)
        passed = all(
            (
                replay.response == ProtocolResponse.REJECTED_REPLAY,
                before_fp == after_fp,
                before_counts == after_counts,
                not after_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "same txID; changed attemptID",
                "mutation_field": "attempt_id",
                "replay_response": replay.response.value,
                "record_fingerprint_before": before_fp,
                "record_fingerprint_after": after_fp,
                "record_fingerprint_unchanged": before_fp == after_fp,
                "db_counts_unchanged": before_counts == after_counts,
                "pre_counts": before_counts,
                "post_counts": after_counts,
                "state_duplicated": bool(after_counts["state_duplicated"]),
                "resolution_time_ns": t1 - t0,
                "resolution_time_seconds": (t1 - t0) / 1e9,
                "observed_result": "Conflicting tx binding rejected as RejectedReplay; committed durable state unchanged",
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        return row
    finally:
        store.close()


def run_postcommit_delivery_failure(db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "postcommit_delivery_failure"
    row = result_base(case_id, replicate, db_path, warmup=warmup)
    fixture = make_fixture(case_id, replicate)
    store = SQLiteProtocolStore(db_path)
    try:
        flags = sqlite_flags(store)
        output_unavailable = OutputUnavailable(
            tx_id=fixture.request.tx_id,
            message="simulated postcommit delivery unavailable",
        )
        terminal, _ = persist_committed_without_log_repair(
            store,
            fixture,
            delivery=DeliveryState.OUTPUT_UNAVAILABLE,
            output_unavailable=output_unavailable,
        )
        store.repair_committed_logs(fixture.request.tx_id)
        stable_before = store.lookup_by_tx_id(fixture.request.tx_id)
        before_counts = db_counts(store, fixture.request.tx_id)
        prepare_hash_before = stable_before.output_prepare.record_hash() if stable_before and stable_before.output_prepare else None
        store.close()

        t0 = time.perf_counter_ns()
        reopened = SQLiteProtocolStore(db_path)
        replay = ProtocolEngine(reopened).handle(fixture.request, context={})
        after = reopened.lookup_by_tx_id(fixture.request.tx_id)
        post_counts = db_counts(reopened, fixture.request.tx_id)
        t1 = time.perf_counter_ns()
        prepare_hash_after = after.output_prepare.record_hash() if after and after.output_prepare else None
        passed = all(
            (
                terminal.state == ProtocolState.COMMITTED,
                replay.response == ProtocolResponse.COMMITTED,
                replay.state == ProtocolState.COMMITTED,
                replay.delivery == DeliveryState.OUTPUT_UNAVAILABLE,
                after is not None and after.state == ProtocolState.COMMITTED,
                after.delivery == DeliveryState.OUTPUT_UNAVAILABLE,
                after.output_unavailable is not None,
                prepare_hash_before is not None and prepare_hash_before == prepare_hash_after,
                before_counts == post_counts,
                post_counts["decision_count"] == 1,
                not post_counts["state_duplicated"],
            )
        )
        row.update(
            {
                **flags,
                "fault_boundary": "Commit durable; requester delivery unavailable; close/reopen then replay",
                "pre_state": terminal.state.value,
                "pre_delivery": terminal.delivery.value if terminal.delivery else None,
                "recovery_response": replay.response.value,
                "recovery_result_state": replay.state.value if replay.state else None,
                "recovery_delivery": replay.delivery.value if replay.delivery else None,
                "durable_state_after_recovery": after.state.value if after else None,
                "output_prepare_hash_before": prepare_hash_before,
                "output_prepare_hash_after": prepare_hash_after,
                "output_prepare_hash_unchanged": prepare_hash_before is not None and prepare_hash_before == prepare_hash_after,
                "application_output_payload_modeled": False,
                "payload_retransmission_equality_claim": False,
                "pre_counts": before_counts,
                "post_counts": post_counts,
                "state_duplicated": bool(post_counts["state_duplicated"]),
                "recovery_time_ns": t1 - t0,
                "recovery_time_seconds": (t1 - t0) / 1e9,
                "observed_result": (
                    "Commit persisted and replay returned Committed + OutputUnavailable; prepared-output metadata/logs stable; "
                    "application output payload bytes are not modeled"
                ),
                "passed": passed,
                "status": "success" if passed else "failed",
            }
        )
        reopened.close()
        return row
    finally:
        try:
            store.close()
        except Exception:
            pass


def run_threshold_boundary(_db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    case_id = "arithmetic_boundary_t_vs_tminus1"
    row = result_base(case_id, replicate, None, warmup=warmup)
    profile = benchmark_profile()
    backend = SimulatedShamirBackend(profile, master_seed=910_000 + replicate)
    value = 123456 + replicate
    bundle = backend.share_secret(f"vi-f-threshold-{replicate}", value, namespace="vi-f")
    t_bundle = ShareBundle(
        shares=tuple(bundle.shares[:T]),
        secret_id=bundle.secret_id,
        field_prime=bundle.field_prime,
        threshold=bundle.threshold,
        committee_size=bundle.committee_size,
        backend_profile_hash=bundle.backend_profile_hash,
        share_epoch=bundle.share_epoch,
    )
    tm1_bundle = ShareBundle(
        shares=tuple(bundle.shares[: T - 1]),
        secret_id=bundle.secret_id,
        field_prime=bundle.field_prime,
        threshold=bundle.threshold,
        committee_size=bundle.committee_size,
        backend_profile_hash=bundle.backend_profile_hash,
        share_epoch=bundle.share_epoch,
    )
    order = ("T", "T-1") if replicate % 2 else ("T-1", "T")
    results: dict[str, Any] = {}
    timings: dict[str, int] = {}
    for label in order:
        selected = t_bundle if label == "T" else tm1_bundle
        t0 = time.perf_counter_ns()
        result = backend.reconstruct(selected)
        t1 = time.perf_counter_ns()
        results[label] = result
        timings[label] = t1 - t0
    q_t = fixed_attempt_continuation_quorum(T, T, True, True, True)
    q_tm1 = fixed_attempt_continuation_quorum(T - 1, T, True, True, True)
    t_result = results["T"]
    tm1_result = results["T-1"]
    passed = all(
        (
            t_result.status == "reconstructed",
            t_result.value == value,
            tm1_result.status == "rejected",
            tm1_result.reason_code == "INSUFFICIENT_SHARES",
            q_t.satisfied is True,
            q_tm1.satisfied is False,
        )
    )
    row.update(
        {
            "fault_boundary": "simulated threshold reconstruction and fixed-attempt continuation predicate",
            "simulated_profile_N": N,
            "simulated_profile_T": T,
            "test_order": list(order),
            "t_share_count": T,
            "t_minus_1_share_count": T - 1,
            "t_reconstruction_status": t_result.status,
            "t_reconstruction_value_matches": t_result.value == value,
            "t_minus_1_reconstruction_status": tm1_result.status,
            "t_minus_1_reason_code": tm1_result.reason_code,
            "t_continuation_quorum_satisfied": q_t.satisfied,
            "t_minus_1_continuation_quorum_satisfied": q_tm1.satisfied,
            "t_reconstruction_time_ns": timings["T"],
            "t_minus_1_reconstruction_time_ns": timings["T-1"],
            "whole_protocol_completion_inferred": False,
            "output_preparation_inferred": False,
            "joint_commit_inferred": False,
            "state_duplicated": False,
            "observed_result": "T reconstructed and satisfied arithmetic-continuation predicate; T-1 rejected; no full-protocol completion inferred",
            "passed": passed,
            "status": "success" if passed else "failed",
        }
    )
    return row


CASE_RUNNERS: dict[str, Callable[[Path, int], dict[str, Any]]] = {
    "crash_before_durable_decision": run_crash_before_durable_decision,
    "crash_after_prepared": run_crash_after_prepared,
    "crash_after_commit": run_crash_after_commit,
    "duplicate_replay": run_duplicate_replay,
    "conflicting_replay": run_conflicting_replay,
    "postcommit_delivery_failure": run_postcommit_delivery_failure,
    "arithmetic_boundary_t_vs_tminus1": run_threshold_boundary,
}


def run_case(case_id: str, db_path: Path, replicate: int, *, warmup: bool) -> dict[str, Any]:
    try:
        return CASE_RUNNERS[case_id](db_path, replicate, warmup=warmup)
    except Exception as exc:
        row = result_base(case_id, replicate, db_path if case_id != "arithmetic_boundary_t_vs_tminus1" else None, warmup=warmup)
        row.update(
            {
                "status": "failed",
                "passed": False,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "observed_result": f"{exc.__class__.__name__}: {exc}",
            }
        )
        return row


def static_preflight(root: Path) -> dict[str, Any]:
    actual: dict[str, str] = {}
    for rel, expected in EXPECTED_HASHES.items():
        got = sha(root / rel)
        actual[rel] = got
        if got != expected:
            raise VIFError(f"BOUND_SOURCE_SHA_MISMATCH:{rel}:{got}:{expected}")

    design = read_json(root / DESIGN_REL)
    if [item["case_id"] for item in design["cases"]] != list(CASE_IDS):
        raise VIFError("DESIGN_CASE_SET_OR_ORDER_MISMATCH")
    if design["repetitions"]["warmup_replicates_per_case"] != WARMUPS:
        raise VIFError("DESIGN_WARMUP_COUNT_MISMATCH")
    if design["repetitions"]["measured_replicates_per_case"] != MEASURED:
        raise VIFError("DESIGN_MEASURED_COUNT_MISMATCH")
    if design["repetitions"]["measured_case_replicates_total"] != len(CASE_IDS) * MEASURED:
        raise VIFError("DESIGN_TOTAL_COUNT_MISMATCH")

    profile = benchmark_profile()
    if profile.committee_size_n != N or profile.threshold_t != T:
        raise VIFError("THRESHOLD_PROFILE_MISMATCH")
    if profile.production_ready or profile.cryptographic_security_claim:
        raise VIFError("SIMULATED_PROFILE_SCOPE_DRIFT")

    diagnostics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vi-f-preflight-") as tmp:
        tmp_root = Path(tmp)
        for index, case_id in enumerate(CASE_IDS, 1):
            db_path = tmp_root / f"{index:02d}_{case_id}.sqlite3"
            row = run_case(case_id, db_path, 999, warmup=True)
            diagnostics[case_id] = {
                "passed": bool(row.get("passed")),
                "observed_result": row.get("observed_result"),
            }
            if not row.get("passed"):
                raise VIFError(
                    f"CASE_PREFLIGHT_FAILED:{case_id}:{row.get('error_type')}:{row.get('error_message')}:{row.get('observed_result')}"
                )

    phase5_summary_path = root / "results/summary/phase5_v2_e8a.json"
    if not phase5_summary_path.is_file():
        fallback = root / "summary/phase5_v2_e8a.json"
        phase5_summary_path = fallback if fallback.is_file() else phase5_summary_path
    phase5_summary = None
    phase5_summary_hash = None
    if phase5_summary_path.is_file():
        phase5_summary = read_json(phase5_summary_path)
        phase5_summary_hash = sha(phase5_summary_path)
        if phase5_summary.get("status") != "passed" or not phase5_summary.get("all_passed"):
            raise VIFError("PHASE5_E8A_NOT_PASSED")

    experiment_runner_rel = Path("src/lir_pptd/experiments/phase_vi_f/runner.py")
    experiment_runner_sha256 = sha(root / experiment_runner_rel)

    return {
        "bound_source_hashes": actual,
        "experiment_runner_path": experiment_runner_rel.as_posix(),
        "experiment_runner_sha256": experiment_runner_sha256,
        "phase5_e8a_summary_path": str(phase5_summary_path.relative_to(root)) if phase5_summary_path.is_file() else None,
        "phase5_e8a_summary_sha256": phase5_summary_hash,
        "phase5_e8a_status": phase5_summary.get("status") if phase5_summary else None,
        "diagnostic_cases": diagnostics,
        "simulated_threshold_profile": {
            "N": N,
            "T": T,
            "backend": "simulated_shamir",
            "production_ready": profile.production_ready,
            "cryptographic_security_claim": profile.cryptographic_security_claim,
        },
        "scope": {
            "production_mpc_used": False,
            "distributed_e8b_used": False,
            "distributed_availability_claim": False,
            "production_fault_tolerance_claim": False,
        },
    }


def environment_binding() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def make_start_binding(root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    design_hash = sha(root / DESIGN_REL)
    payload = {
        "schema_version": "1.0",
        "experiment_namespace": "VI-F",
        "git_head": git_head(root),
        "design_path": DESIGN_REL.as_posix(),
        "design_sha256": design_hash,
        "source_hashes": preflight["bound_source_hashes"],
        "experiment_runner_path": preflight["experiment_runner_path"],
        "experiment_runner_sha256": preflight["experiment_runner_sha256"],
        "phase5_e8a_summary_path": preflight["phase5_e8a_summary_path"],
        "phase5_e8a_summary_sha256": preflight["phase5_e8a_summary_sha256"],
        "case_ids": list(CASE_IDS),
        "warmup_replicates_per_case": WARMUPS,
        "measured_replicates_per_case": MEASURED,
        "planned_measured_case_replicates": len(CASE_IDS) * MEASURED,
        "threshold_profile": preflight["simulated_threshold_profile"],
        "scope": preflight["scope"],
        "environment": environment_binding(),
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return {**payload, "start_binding_sha256": digest}


def preflight_and_bind(root: Path) -> int:
    preflight = static_preflight(root)
    binding_path = root / BINDING_REL
    raw_path = root / RAW_REL
    approval_path = root / APPROVAL_REL
    if raw_path.is_file() and not binding_path.is_file():
        raise VIFError("RAW_EXISTS_WITHOUT_BINDING")
    binding = make_start_binding(root, preflight)
    if binding_path.is_file():
        existing = read_json(binding_path)
        if existing.get("start_binding_sha256") != binding["start_binding_sha256"]:
            raise VIFError("EXISTING_BINDING_MISMATCH")
    else:
        write_json(binding_path, binding)
    if approval_path.is_file():
        approval = read_json(approval_path)
        if approval.get("start_binding_sha256") != binding["start_binding_sha256"]:
            raise VIFError("EXISTING_APPROVAL_BINDING_MISMATCH")
    print("VI_F_PREFLIGHT=PASS")
    print("FORMAL_EXPERIMENTS_RUN=0")
    print("CASE_CONFORMANCE_PREFLIGHT=PASS")
    print(f"CASE_COUNT={len(CASE_IDS)}")
    print(f"WARMUP_REPLICATES_PER_CASE={WARMUPS}")
    print(f"PLANNED_MEASURED_CASE_REPLICATES={len(CASE_IDS) * MEASURED}")
    print(f"SIMULATED_THRESHOLD_N={N}")
    print(f"SIMULATED_THRESHOLD_T={T}")
    print("PRODUCTION_MPC_USED=NO")
    print("DISTRIBUTED_E8B_USED=NO")
    print("DISTRIBUTED_AVAILABILITY_CLAIM=NO")
    print("RUNTIME_GATE_ENABLED=NO")
    print(f"VI_F_START_BINDING_SHA256={binding['start_binding_sha256']}")
    return 0


def record_approval(root: Path, binding: str, approval_text: str) -> int:
    binding_obj = read_json(root / BINDING_REL)
    expected_binding = binding_obj["start_binding_sha256"]
    if binding != expected_binding:
        raise VIFError("APPROVAL_BINDING_ARGUMENT_MISMATCH")
    expected_text = f"APPROVE VI-F {binding}"
    if approval_text != expected_text:
        raise VIFError("APPROVAL_TEXT_MISMATCH")
    approval_path = root / APPROVAL_REL
    if approval_path.is_file():
        existing = read_json(approval_path)
        if existing.get("start_binding_sha256") != binding or existing.get("approval_text") != expected_text:
            raise VIFError("EXISTING_APPROVAL_CONFLICT")
        print("VI_F_START_APPROVAL=EXISTS_VALID")
        return 0
    write_json(
        approval_path,
        {
            "schema_version": "1.0",
            "experiment_namespace": "VI-F",
            "start_binding_sha256": binding,
            "approval_text": expected_text,
            "recorded_unix": time.time(),
        },
    )
    print("VI_F_START_APPROVAL=RECORDED")
    return 0


def ensure_binding_and_approval(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = root / BINDING_REL
    approval_path = root / APPROVAL_REL
    if not binding_path.is_file():
        raise VIFError("MISSING_START_BINDING")
    if not approval_path.is_file():
        raise VIFError("MISSING_START_APPROVAL")
    binding = read_json(binding_path)
    approval = read_json(approval_path)
    if approval.get("start_binding_sha256") != binding.get("start_binding_sha256"):
        raise VIFError("APPROVAL_BINDING_MISMATCH")
    expected_text = f"APPROVE VI-F {binding['start_binding_sha256']}"
    if approval.get("approval_text") != expected_text:
        raise VIFError("APPROVAL_TEXT_INVALID")
    current = make_start_binding(root, static_preflight(root))
    if current["start_binding_sha256"] != binding["start_binding_sha256"]:
        raise VIFError("CURRENT_REPO_OR_DESIGN_NO_LONGER_MATCHES_START_BINDING")
    return binding, approval


def formal_db_path(root: Path, binding: str, case_index: int, case_id: str, replicate: int) -> Path:
    run_root = root / "results" / "vi_f_runs" / binding[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root / f"{case_index:02d}_{case_id}_r{replicate:02d}.sqlite3"


def cleanup_sqlite(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def run_warmups(root: Path, binding: str) -> None:
    with tempfile.TemporaryDirectory(prefix="vi-f-warmups-", dir=str(root / "results")) as tmp:
        tmp_root = Path(tmp)
        for case_index, case_id in enumerate(CASE_IDS, 1):
            for replicate in range(1, WARMUPS + 1):
                db_path = tmp_root / f"{case_index:02d}_{case_id}_w{replicate:02d}.sqlite3"
                row = run_case(case_id, db_path, replicate, warmup=True)
                if not row.get("passed"):
                    raise VIFError(f"WARMUP_FAILED:{case_id}:{replicate}:{row.get('observed_result')}")


def measured_key(case_id: str, replicate: int) -> str:
    return f"{case_id}:{replicate:02d}"


def generate_summary(root: Path, binding: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    design = read_json(root / DESIGN_REL)
    design_by_case = {item["case_id"]: item for item in design["cases"]}
    grouped: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in CASE_IDS}
    for row in rows:
        grouped[row["case_id"]].append(row)

    case_summaries: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        items = sorted(grouped[case_id], key=lambda item: int(item["replicate_index"]))
        times = [float(item["recovery_time_seconds"]) for item in items if item.get("recovery_time_seconds") is not None]
        timing = timing_summary(times)
        all_pass = len(items) == MEASURED and all(bool(item.get("passed")) for item in items)
        duplicated_any = any(bool(item.get("state_duplicated")) for item in items)
        observed = items[0].get("observed_result") if items else "No measured records"
        if items and len({item.get("observed_result") for item in items}) > 1:
            observed = f"Mixed observed results across {len(items)} replicates"
        case_summaries.append(
            {
                "case_id": case_id,
                "paper_label": design_by_case[case_id]["paper_label"],
                "expected_result": design_by_case[case_id]["expected_result"],
                "observed_result": observed,
                "n_measured": len(items),
                "success_replicates": sum(1 for item in items if item.get("passed")),
                "failure_replicates": sum(1 for item in items if not item.get("passed")),
                "all_replicates_pass": all_pass,
                "state_duplicated_any": duplicated_any,
                "recovery_time_applicable": bool(design_by_case[case_id]["recovery_time_applicable"]),
                "recovery_time": timing,
            }
        )

    raw_path = root / RAW_REL
    successes = sum(1 for row in rows if row.get("passed"))
    failures = len(rows) - successes
    status = "complete" if len(rows) == len(CASE_IDS) * MEASURED and failures == 0 else "failed"
    return {
        "schema_version": "1.0",
        "experiment_namespace": "VI-F",
        "formal_experiment": True,
        "status": status,
        "start_binding_sha256": binding["start_binding_sha256"],
        "case_count": len(CASE_IDS),
        "warmup_replicates_per_case": WARMUPS,
        "measured_replicates_per_case": MEASURED,
        "planned_measured_case_replicates": len(CASE_IDS) * MEASURED,
        "committed_measured_case_replicates": len(rows),
        "success_replicates": successes,
        "failure_replicates": failures,
        "state_duplication_cases": sum(1 for item in case_summaries if item["state_duplicated_any"]),
        "raw_path": RAW_REL.as_posix(),
        "raw_sha256": sha(raw_path),
        "sqlite_profile": {
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "foreign_keys": "ON",
        },
        "threshold_boundary": {
            "backend": "simulated_shamir",
            "N": N,
            "T": T,
            "production_mpc_used": False,
            "cryptographic_security_claim": False,
        },
        "scope": {
            "single_machine_sqlite_wal": True,
            "protocol_state_machine_conformance": True,
            "distributed_e8b_used": False,
            "distributed_availability_claim": False,
            "production_mpc_used": False,
            "production_fault_tolerance_claim": False,
        },
        "measurement_limitations": {
            "postcommit_application_output_payload_bytes_persisted": False,
            "postcommit_payload_retransmission_equality_claim": False,
            "distributed_network_faults_measured": False,
            "production_mpc_recovery_measured": False,
        },
        "cases": case_summaries,
    }


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def paper_table(summary: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{VI-F single-machine SQLite-WAL durable recovery and protocol-conformance results. The threshold row is a simulated arithmetic-boundary check; no distributed E8-B or production fault-tolerance claim is made.}",
        r"\label{tab:vi-f-availability-recovery}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}p{0.16\textwidth}p{0.25\textwidth}Xp{0.09\textwidth}p{0.11\textwidth}@{}}",
        r"\toprule",
        r"Failure / recovery case & Expected result & Observed result & State duplicated? & Recovery time \\",
        r"\midrule",
    ]
    for case in summary["cases"]:
        timing = case["recovery_time"]
        if case["recovery_time_applicable"]:
            median = timing["median_seconds"]
            recovery = "N/A" if median is None else f"{1000.0 * float(median):.3f} ms"
        elif case["case_id"] in {"duplicate_replay", "conflicting_replay"}:
            recovery = "N/A (replay)"
        else:
            recovery = "N/A (boundary)"
        duplicated = "Yes" if case["state_duplicated_any"] else "No"
        lines.append(
            f"{latex_escape(case['paper_label'])} & {latex_escape(case['expected_result'])} & "
            f"{latex_escape(case['observed_result'])} & {duplicated} & {latex_escape(recovery)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def formal_run(root: Path) -> int:
    binding, _approval = ensure_binding_and_approval(root)
    raw_path = root / RAW_REL
    checkpoint_path = root / CHECKPOINT_REL
    summary_path = root / SUMMARY_REL
    table_path = root / TABLE_REL

    existing_rows = read_jsonl(raw_path)
    for row in existing_rows:
        if row.get("start_binding_sha256") != binding["start_binding_sha256"]:
            raise VIFError("RAW_BINDING_MISMATCH")
    seen_keys = {measured_key(row["case_id"], int(row["replicate_index"])) for row in existing_rows}
    if len(seen_keys) != len(existing_rows):
        raise VIFError("DUPLICATE_RAW_MEASURED_KEY")

    if len(existing_rows) == 0:
        run_warmups(root, binding["start_binding_sha256"])

    for case_index, case_id in enumerate(CASE_IDS, 1):
        for replicate in range(1, MEASURED + 1):
            key = measured_key(case_id, replicate)
            if key in seen_keys:
                continue
            db_path = formal_db_path(root, binding["start_binding_sha256"], case_index, case_id, replicate)
            cleanup_sqlite(db_path)
            row = run_case(case_id, db_path, replicate, warmup=False)
            row.update(
                {
                    "run_id": f"{binding['start_binding_sha256'][:12]}:{case_index:02d}:{replicate:02d}",
                    "start_binding_sha256": binding["start_binding_sha256"],
                    "formal_experiment": True,
                }
            )
            append_jsonl(raw_path, row)
            existing_rows.append(row)
            seen_keys.add(key)
            write_json(
                checkpoint_path,
                {
                    "schema_version": "1.0",
                    "experiment_namespace": "VI-F",
                    "start_binding_sha256": binding["start_binding_sha256"],
                    "committed_measured_case_replicates": len(existing_rows),
                    "last_committed_key": key,
                    "planned_measured_case_replicates": len(CASE_IDS) * MEASURED,
                },
            )

    rows = read_jsonl(raw_path)
    summary = generate_summary(root, binding, rows)
    write_json(summary_path, summary)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(paper_table(summary), encoding="utf-8")

    print(f"VI_F_FORMAL_STATUS={summary['status']}")
    print(f"CASE_COUNT={summary['case_count']}")
    print(f"COMMITTED_MEASURED_CASE_REPLICATES={summary['committed_measured_case_replicates']}")
    print(f"SUCCESS_REPLICATES={summary['success_replicates']}")
    print(f"FAILURE_REPLICATES={summary['failure_replicates']}")
    print(f"STATE_DUPLICATION_CASES={summary['state_duplication_cases']}")
    print("DISTRIBUTED_E8B_USED=NO")
    print("DISTRIBUTED_AVAILABILITY_CLAIM=NO")
    print("PRODUCTION_MPC_USED=NO")
    print(f"RAW_SHA256={summary['raw_sha256']}")
    print(f"SUMMARY={SUMMARY_REL.as_posix()}")
    print(f"TABLE={TABLE_REL.as_posix()}")
    if summary["status"] == "complete":
        print("VI_F_FORMAL_INVOCATION=PASS")
        return 0
    print("VI_F_FORMAL_INVOCATION=FAIL")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lir_pptd.experiments.phase_vi_f.runner")
    parser.add_argument("--repo-root", default=".")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-and-bind", action="store_true")
    group.add_argument("--record-approval", action="store_true")
    group.add_argument("--formal-run", action="store_true")
    parser.add_argument("--binding")
    parser.add_argument("--approval-text")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    try:
        if args.preflight_and_bind:
            return preflight_and_bind(root)
        if args.record_approval:
            if not args.binding or args.approval_text is None:
                raise VIFError("RECORD_APPROVAL_REQUIRES_BINDING_AND_TEXT")
            return record_approval(root, args.binding, args.approval_text)
        if args.formal_run:
            return formal_run(root)
        raise VIFError("NO_ACTION")
    except VIFError as exc:
        print(f"VI_F_ERROR={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
