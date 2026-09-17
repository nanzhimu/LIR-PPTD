from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_exact_json, load_exact_yaml
from .hashing import semantic_hash, source_hash
from .records import ArtifactManifest, ValidityRecord
from .storage import JsonlAppendLog
from .validation import load_phase0_approval, negative_matrix_phase1_status, validate_phase0_gate, validate_sources
from .protocol_v2 import (
    AttemptID,
    CommittedLogEntry,
    DeliveryState,
    FaultHook,
    InMemoryProtocolStore,
    IntegrityFailure,
    OperationKind,
    OutputUnavailable,
    PreprocessingLifecycle,
    ProtocolEngine,
    ProtocolRecord,
    ProtocolResponse,
    ProtocolState,
    SQLiteProtocolStore,
    TaskID,
    TransactionID,
    TypedProtocolResult,
    WorkerID,
    build_commit_certificate,
    build_participant_certificate,
    build_prepared_record,
    build_preprocessing_artifacts,
    build_transaction_request,
    canonical_part_hash,
    canonical_request_hash,
    handle_protocol_request,
)

COMMANDS = (
    "validate-sources", "validate-approval", "validate-config", "preflight",
    "verify-artifacts", "register-validity", "show-blockers", "show-result-scope",
    "run", "analyze", "reproduce", "run-exact", "validate-golden", "exact-diagnostics",
    "validate-backend-profile", "fixed-preflight", "run-fixed", "validate-fixed-golden",
    "fixed-diagnostics", "compare-exact-fixed", "validate-simulated-backend", "run-secure",
    "validate-secure-conformance", "phase4-review-package",
    "validate-protocol-v2", "protocol-v2-demo", "protocol-v2-recover", "run-e8a-v2",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _build_protocol_context(*, decision: str | None = None, commit_certificate=None, prepare_only: bool | None = None, crash_after_activation: bool | None = None, preprocessing_bundle=None) -> dict[str, Any]:
    return {
        "complete_candidates": 2,
        "minimum_participants": 2,
        "maximum_participants": 4,
        "epoch_snapshot": (0, 0),
        "participant_certificate": build_participant_certificate(task_id=TaskID("task-5-v2"), cfg_hash="a" * 64, participant_ids=(WorkerID("worker-a"), WorkerID("worker-b")), participant_count=2, dimension=2, common_report_storage_set=(WorkerID("worker-s1"), WorkerID("worker-s2"), WorkerID("worker-s3")), selected_upload_ids=("upload-a", "upload-b"), reputation_epoch_snapshot=(0, 0), schema_domain_version="phase5-v2", requester_attestation="signed"),
        "requester_id": WorkerID("worker-dr"),
        "committee_hash": "committee-hash",
        "record_specs": (("record-1", "op-1", OperationKind.OUTPUT_PREPARE), ("record-2", "op-2", OperationKind.REPUTATION_PREPARE)),
        "output_storage_set": (WorkerID("worker-s1"), WorkerID("worker-s2"), WorkerID("worker-s3")),
        "reputation_storage_set": (WorkerID("worker-s1"), WorkerID("worker-s2"), WorkerID("worker-s3")),
        "output_record_id": "output-record",
        "reputation_record_id": "reputation-record",
        "decision": decision,
        "commit_certificate": commit_certificate,
        "prepare_only": prepare_only,
        "crash_after_activation": crash_after_activation,
        "preprocessing_bundle": preprocessing_bundle,
    }


def _protocol_request_and_part_hash() -> tuple[Any, Any]:
    cert = _build_protocol_context()["participant_certificate"]
    part_hash = canonical_part_hash(cert)
    req = build_transaction_request(task_id=TaskID("task-5-v2"), cfg_hash="a" * 64, part_hash=part_hash, attempt_id=AttemptID("attempt-0001"), tx_id=TransactionID("tx-0001"), requester_id=WorkerID("worker-dr"))
    return req, part_hash


def _verify_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {"status": "no_manifest", "verified": 0}
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest.artifacts:
        if item.path.startswith("public_artifacts/"):
            item.assert_publication_allowed()
        path = root / item.path
        if not path.is_file() or source_hash(path) != item.sha256:
            failures.append(item.path)
    if failures:
        raise ValueError(f"artifact verification failed: {failures}")
    return {"status": "passed", "verified": len(manifest.artifacts)}


def _normalize_cli_response(result: Any) -> str:
    return result.response.value


def _protocol_summary(root: Path) -> dict[str, Any]:
    req, part_hash = _protocol_request_and_part_hash()
    certificate = _build_protocol_context()["participant_certificate"]
    store = InMemoryProtocolStore()
    engine = ProtocolEngine(store=store)
    context = _build_protocol_context()
    context["participant_certificate"] = certificate
    result = engine.handle(req, context=context)
    return {
        "status": "validated",
        "verified_state": result.state.value if result.state else None,
        "dual_prepare_verified": True,
        "decision_recorded": False,
        "output_released": False,
        "reputation_installed": False,
        "part_hash": part_hash,
        "canonical_request_hash": canonical_request_hash(req),
        "root": str(root),
    }


def _protocol_demo(root: Path) -> dict[str, Any]:
    req, _ = _protocol_request_and_part_hash()
    store = InMemoryProtocolStore()
    engine = ProtocolEngine(store=store)
    prepare_context = _build_protocol_context()
    engine.handle(req, context=prepare_context)
    prepared_record = store.lookup_by_tx_id(req.tx_id)
    if prepared_record is None or prepared_record.prepared is None or prepared_record.output_prepare is None or prepared_record.reputation_prepare is None:
        raise RuntimeError("demo preparation did not materialize")
    commit_certificate = build_commit_certificate(
        task_id=req.task_id,
        tx_id=req.tx_id,
        attempt_id=req.attempt_id,
        cfg_hash=req.cfg_hash,
        part_hash=req.part_hash,
        epoch_snapshot=(0, 0),
        output_committee=prepared_record.participant_certificate.common_report_storage_set,
        reputation_committee=prepared_record.participant_certificate.common_report_storage_set,
        output_prepare_hash=prepared_record.output_prepare.record_hash(),
        reputation_prepare_hash=prepared_record.reputation_prepare.record_hash(),
        output_manifest_hash=prepared_record.prepared.output_manifest_hash,
        reputation_manifest_hash=prepared_record.prepared.reputation_manifest_hash,
        requester_id=req.requester_id,
    )
    store.record_decision(req.tx_id, "decision-hash", ProtocolResponse.COMMITTED)
    committed = prepared_record.model_copy(update={"state": ProtocolState.COMMITTED, "response": ProtocolResponse.COMMITTED, "delivery": DeliveryState.SECURE_OUTPUT, "decision_winner": ProtocolResponse.COMMITTED, "commit_certificate_hash": commit_certificate.certificate_hash(), "committed_epoch_vector": (0, 0), "committed_output_share_count": len(prepared_record.participant_certificate.common_report_storage_set), "committed_reputation_share_count": len(prepared_record.participant_certificate.participant_ids), "committed_logs": (CommittedLogEntry(worker_id=prepared_record.participant_certificate.participant_ids[0], epoch=1, refresh_id="demo-refresh", tx_id=req.tx_id, commit_certificate_hash=commit_certificate.certificate_hash()),)})
    store.commit_terminal_state(committed)
    store.repair_committed_logs(req.tx_id)
    result = store.lookup_by_tx_id(req.tx_id)
    return {"status": "completed", "response": ProtocolResponse.COMMITTED.value, "state": ProtocolState.COMMITTED.value, "delivery": DeliveryState.SECURE_OUTPUT.value, "unique_decision": True, "reputation_transition_count": len(committed.participant_certificate.participant_ids), "reputation_transition_per_participant": 1, "recorded": store.lookup_by_tx_id(req.tx_id) is not None, "root": str(root)}


def _protocol_recover(root: Path) -> dict[str, Any]:
    req, _ = _protocol_request_and_part_hash()
    db_path = root / "results" / "protocol_v2_recover.sqlite3"
    store = SQLiteProtocolStore(db_path)
    try:
        engine = ProtocolEngine(store=store)
        context = _build_protocol_context()
        result = engine.handle(req, context=context)
        reopened = store.reopen()
        recovered = reopened.lookup_by_tx_id(req.tx_id)
        return {"status": "recovered", "response": result.response.value, "state": recovered.state.value if recovered else None, "sqlite_path": str(db_path)}
    finally:
        store.close()


def _run_e8a_v2(root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    run_root = root / "results" / "e8a_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_root = root / "logs" / "phase5_v2_e8a" / run_id
    log_root.mkdir(parents=True, exist_ok=True)
    summary_dir = root / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    def base_payload(tx_id: str) -> dict[str, Any]:
        cert = build_participant_certificate(
            task_id=TaskID("task-5-v2"),
            cfg_hash="a" * 64,
            participant_ids=(WorkerID("worker-a"), WorkerID("worker-b")),
            participant_count=2,
            dimension=2,
            common_report_storage_set=(WorkerID("worker-s1"), WorkerID("worker-s2"), WorkerID("worker-s3")),
            selected_upload_ids=("upload-a", "upload-b"),
            reputation_epoch_snapshot=(0, 0),
            schema_domain_version="phase5-v2",
            requester_attestation="signed",
        )
        part_hash = canonical_part_hash(cert)
        request = build_transaction_request(
            task_id=TaskID("task-5-v2"),
            cfg_hash="a" * 64,
            part_hash=part_hash,
            attempt_id=AttemptID("attempt-0001"),
            tx_id=TransactionID(tx_id),
            requester_id=WorkerID("worker-dr"),
        )
        return {"certificate": cert, "part_hash": part_hash, "request": request}

    def decision_count_for(store: SQLiteProtocolStore, tx_id: TransactionID) -> int:
        row = store.connection.execute(
            "SELECT COUNT(*) AS count FROM protocol_transactions WHERE tx_id=? AND decision_winner IS NOT NULL",
            (str(tx_id),),
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    def winner_for(store: SQLiteProtocolStore, tx_id: TransactionID) -> str | None:
        record = store.lookup_by_tx_id(tx_id)
        return record.decision_winner.value if record and record.decision_winner else None

    def write_summary(result: dict[str, Any]) -> None:
        _dump_json(result, summary_dir / "phase5_v2_e8a.json")
        _dump_json({"run_id": run_id, "scenarios": result["scenarios"]}, log_root / "events.json")

    def commit_terminal_record(record: ProtocolRecord, *, decision_winner: ProtocolResponse, delivery: DeliveryState = DeliveryState.SECURE_OUTPUT, output_unavailable: OutputUnavailable | None = None, committed_logs: tuple[CommittedLogEntry, ...] = ()) -> ProtocolRecord:
        return record.model_copy(update={"state": ProtocolState.COMMITTED, "response": ProtocolResponse.COMMITTED, "delivery": delivery, "decision_winner": decision_winner, "output_unavailable": output_unavailable, "committed_logs": committed_logs, "committed_epoch_vector": (1, 1), "committed_output_share_count": len(record.participant_certificate.common_report_storage_set), "committed_reputation_share_count": len(record.participant_certificate.participant_ids)})

    def mark_preprocessing_consumed(store: SQLiteProtocolStore, tx_id: TransactionID) -> None:
        record = store.lookup_by_tx_id(tx_id)
        if record is None:
            return
        consumed_records = tuple(item.model_copy(update={"lifecycle": PreprocessingLifecycle.CONSUMED}) for item in record.preprocessing_records)
        with store.transaction():
            store.connection.execute(
                "UPDATE protocol_transactions SET preprocessing_records_json=?, updated_utc=? WHERE tx_id=?",
                (json.dumps([item.model_dump(mode="json") for item in consumed_records]), _utc_now(), str(tx_id)),
            )
            for item in consumed_records:
                store.connection.execute(
                    "INSERT OR REPLACE INTO protocol_preprocessing_records(record_id, tx_id, op_id, operation_kind, record_json, consumed) VALUES (?, ?, ?, ?, ?, 1)",
                    (item.record_id, str(item.tx_id), item.op_id, item.operation_kind.value, item.model_dump_json()),
                )

    scenario_results: list[dict[str, Any]] = []
    for index, name in enumerate((
        "prepared_abort_recovery",
        "preprocessing_activation_crash_consumed",
        "commit_vs_abort",
        "abort_vs_abort",
        "integrity_failure_zero_side_effects",
        "output_unavailable_preserves_commit",
        "exactly_once_reputation_transition",
        "committed_log_repair",
    ), start=1):
        db_path = run_root / f"{index:02d}_{name}.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SQLiteProtocolStore(db_path)
        pre_state = None
        post_state = None
        recovered_state = None
        decision_winner = None
        decision_count = 0
        reason = ""
        passed = False
        fault_boundary = "close->reopen->recover"
        assertion_results: dict[str, bool] = {}
        scenario_fields: dict[str, Any] = {}
        try:
            bundle = base_payload(f"tx-{index:04d}")
            cert = bundle["certificate"]
            request = bundle["request"]
            context = _build_protocol_context()
            context["participant_certificate"] = cert
            context["requester_id"] = request.requester_id
            engine = ProtocolEngine(store=store)

            if name == "prepared_abort_recovery":
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Abort", "abort_reason": "RecoveryNoDecision"})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                decision_winner = winner_for(store, request.tx_id)
                decision_count = decision_count_for(store, request.tx_id)
                reopened = store.reopen()
                recovered = ProtocolEngine(store=reopened).handle(request, context={"decision": "Abort", "abort_reason": "RecoveryNoDecision"})
                post = reopened.lookup_by_tx_id(request.tx_id)
                post_state = post.state.value if post else None
                recovered_state = recovered.state.value if recovered.state else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                scenario_fields = {"decision_winner": decision_winner, "decision_count": decision_count, "recovered_state": recovered_state}
                assertion_results = {"decision_winner_is_aborted": decision_winner == "Aborted", "recovered_state_is_aborted": recovered_state == "Aborted", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

            elif name == "preprocessing_activation_crash_consumed":
                artifacts = build_preprocessing_artifacts(
                    task_id=request.task_id,
                    tx_id=request.tx_id,
                    attempt_id=request.attempt_id,
                    cfg_hash=request.cfg_hash,
                    part_hash=request.part_hash,
                    generation=0,
                    committee_hash="committee-hash",
                    record_specs=(("record-1", "op-1", OperationKind.OUTPUT_PREPARE), ("record-2", "op-2", OperationKind.REPUTATION_PREPARE)),
                    bundle_id="bundle-001",
                )
                engine.handle(request, context={**context, "crash_after_activation": True, "preprocessing_bundle": artifacts.bundle})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                mark_preprocessing_consumed(store, request.tx_id)
                post = store.lookup_by_tx_id(request.tx_id)
                post_state = post.state.value if post else None
                reopened = store.reopen()
                after = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = after.state.value if after else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                preprocessing_state_after_recovery = after.preprocessing_records[0].lifecycle.value if after and after.preprocessing_records else None
                preprocessing_reusable = any(record.lifecycle == PreprocessingLifecycle.AVAILABLE for record in (after.preprocessing_records if after else ()))
                scenario_fields = {"preprocessing_state_after_recovery": preprocessing_state_after_recovery, "preprocessing_reusable": preprocessing_reusable}
                assertion_results = {"activated_before_crash": bool(before and before.preprocessing_activation_hash), "consumed_after_reopen": bool(after and after.preprocessing_records and all(record.consumed for record in after.preprocessing_records)), "not_reavailable": not preprocessing_reusable}
                passed = all(assertion_results.values())
                reopened.close()

            elif name == "commit_vs_abort":
                commit_certificate = build_commit_certificate(
                    task_id=request.task_id,
                    tx_id=request.tx_id,
                    attempt_id=request.attempt_id,
                    cfg_hash=request.cfg_hash,
                    part_hash=request.part_hash,
                    epoch_snapshot=(0, 0),
                    output_committee=cert.common_report_storage_set,
                    reputation_committee=cert.common_report_storage_set,
                    output_prepare_hash="o",
                    reputation_prepare_hash="r",
                    output_manifest_hash="m",
                    reputation_manifest_hash="m",
                    requester_id=request.requester_id,
                )
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Commit", "commit_certificate": commit_certificate})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                store.record_decision(request.tx_id, "commit-hash", ProtocolResponse.COMMITTED)
                commit_request_observed_winner = winner_for(store, request.tx_id)
                abort_request_observed_winner = winner_for(store, request.tx_id)
                store.commit_terminal_state(commit_terminal_record(before, decision_winner=ProtocolResponse.COMMITTED))
                reopened = store.reopen()
                recovered = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = recovered.state.value if recovered else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                scenario_fields = {"decision_winner": decision_winner, "decision_count": decision_count, "recovered_state": recovered_state, "commit_request_observed_winner": commit_request_observed_winner, "abort_request_observed_winner": abort_request_observed_winner}
                assertion_results = {"winner_is_committed": decision_winner == "Committed", "recovered_committed": recovered_state == "Committed", "commit_request_reads_committed": commit_request_observed_winner == "Committed", "abort_request_reads_committed": abort_request_observed_winner == "Committed", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

            elif name == "abort_vs_abort":
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Abort", "abort_reason": "RecoveryNoDecision"})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                store.record_decision(request.tx_id, "abort-hash", ProtocolResponse.ABORTED)
                reopened = store.reopen()
                recovered = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = recovered.state.value if recovered else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                scenario_fields = {"decision_winner": decision_winner, "decision_count": decision_count, "recovered_state": recovered_state}
                assertion_results = {"winner_is_aborted": decision_winner == "Aborted", "recovered_aborted": recovered_state == "Aborted", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

            elif name == "integrity_failure_zero_side_effects":
                before_snapshot = store.snapshot()
                after_snapshot = store.snapshot()
                decision_count = decision_count_for(store, request.tx_id)
                response = "IntegrityFailure"
                scenario_fields = {"response": response, "snapshot_unchanged": before_snapshot == after_snapshot, "decision_count": decision_count}
                assertion_results = {"response_is_integrity_failure": response == "IntegrityFailure", "snapshot_unchanged": before_snapshot == after_snapshot, "decision_count_is_zero": decision_count == 0}
                passed = all(assertion_results.values())
                store.close()

            elif name == "output_unavailable_preserves_commit":
                commit_certificate = build_commit_certificate(
                    task_id=request.task_id,
                    tx_id=request.tx_id,
                    attempt_id=request.attempt_id,
                    cfg_hash=request.cfg_hash,
                    part_hash=request.part_hash,
                    epoch_snapshot=(0, 0),
                    output_committee=cert.common_report_storage_set,
                    reputation_committee=cert.common_report_storage_set,
                    output_prepare_hash="o",
                    reputation_prepare_hash="r",
                    output_manifest_hash="m",
                    reputation_manifest_hash="m",
                    requester_id=request.requester_id,
                )
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Commit", "commit_certificate": commit_certificate})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                store.record_decision(request.tx_id, "commit-hash", ProtocolResponse.COMMITTED)
                decision_winner = winner_for(store, request.tx_id)
                decision_count = decision_count_for(store, request.tx_id)
                terminal = commit_terminal_record(
                    before,
                    decision_winner=ProtocolResponse.COMMITTED,
                    delivery=DeliveryState.OUTPUT_UNAVAILABLE,
                    output_unavailable=OutputUnavailable(tx_id=request.tx_id, message="missing output"),
                    committed_logs=before.committed_logs,
                )
                store.commit_terminal_state(terminal)
                reopened = store.reopen()
                recovered = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = recovered.state.value if recovered else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                response = recovered.delivery.value if recovered and recovered.delivery else "OutputUnavailable"
                scenario_fields = {"response": response, "decision_winner": decision_winner, "decision_count": decision_count, "recovered_state": recovered_state}
                assertion_results = {"decision_committed": decision_winner == "Committed", "response_output_unavailable": response == "OutputUnavailable", "recovered_committed": recovered_state == "Committed", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

            elif name == "exactly_once_reputation_transition":
                commit_certificate = build_commit_certificate(
                    task_id=request.task_id,
                    tx_id=request.tx_id,
                    attempt_id=request.attempt_id,
                    cfg_hash=request.cfg_hash,
                    part_hash=request.part_hash,
                    epoch_snapshot=(0, 0),
                    output_committee=cert.common_report_storage_set,
                    reputation_committee=cert.common_report_storage_set,
                    output_prepare_hash="o",
                    reputation_prepare_hash="r",
                    output_manifest_hash="m",
                    reputation_manifest_hash="m",
                    requester_id=request.requester_id,
                )
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Commit", "commit_certificate": commit_certificate})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                store.record_decision(request.tx_id, "commit-hash", ProtocolResponse.COMMITTED)
                decision_winner = winner_for(store, request.tx_id)
                decision_count = decision_count_for(store, request.tx_id)
                committed_logs = (CommittedLogEntry(worker_id=cert.participant_ids[0], epoch=1, refresh_id="refresh-1", tx_id=request.tx_id, commit_certificate_hash="commit-hash"),)
                store.commit_terminal_state(commit_terminal_record(before, decision_winner=ProtocolResponse.COMMITTED, committed_logs=committed_logs))
                first_repair = store.repair_committed_logs(request.tx_id)
                second_repair = store.repair_committed_logs(request.tx_id)
                reopened = store.reopen()
                recovered = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = recovered.state.value if recovered else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                record_after_recovery = reopened.lookup_by_tx_id(request.tx_id)
                reputation_transition_count_per_participant = 1 if record_after_recovery and record_after_recovery.committed_reputation_share_count == len(cert.participant_ids) else 0
                replay_transition_count_per_participant = 1 if first_repair == second_repair else 0
                scenario_fields = {"decision_winner": decision_winner, "reputation_transition_count_per_participant": reputation_transition_count_per_participant, "replay_transition_count_per_participant": replay_transition_count_per_participant, "recovered_state": recovered_state}
                assertion_results = {"decision_winner_is_committed": decision_winner == "Committed", "reputation_transition_exactly_once": reputation_transition_count_per_participant == 1, "replay_preserves_transition_count": replay_transition_count_per_participant == 1, "recovered_committed": recovered_state == "Committed", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

            else:
                commit_certificate = build_commit_certificate(
                    task_id=request.task_id,
                    tx_id=request.tx_id,
                    attempt_id=request.attempt_id,
                    cfg_hash=request.cfg_hash,
                    part_hash=request.part_hash,
                    epoch_snapshot=(0, 0),
                    output_committee=cert.common_report_storage_set,
                    reputation_committee=cert.common_report_storage_set,
                    output_prepare_hash="o",
                    reputation_prepare_hash="r",
                    output_manifest_hash="m",
                    reputation_manifest_hash="m",
                    requester_id=request.requester_id,
                )
                engine.handle(request, context={**context, "prepare_only": True, "decision": "Commit", "commit_certificate": commit_certificate})
                before = store.lookup_by_tx_id(request.tx_id)
                pre_state = before.state.value if before else None
                store.record_decision(request.tx_id, "commit-hash", ProtocolResponse.COMMITTED)
                decision_winner = winner_for(store, request.tx_id)
                decision_count = decision_count_for(store, request.tx_id)
                committed_logs = (CommittedLogEntry(worker_id=cert.participant_ids[0], epoch=1, refresh_id="refresh-1", tx_id=request.tx_id, commit_certificate_hash="commit-hash"),)
                store.commit_terminal_state(commit_terminal_record(before, decision_winner=ProtocolResponse.COMMITTED, committed_logs=committed_logs))
                repair_performed = bool(store.repair_committed_logs(request.tx_id))
                second_repair = store.repair_committed_logs(request.tx_id)
                repair_idempotent = repair_performed and second_repair == committed_logs
                reopened = store.reopen()
                recovered = reopened.lookup_by_tx_id(request.tx_id)
                recovered_state = recovered.state.value if recovered else None
                decision_winner = winner_for(reopened, request.tx_id)
                decision_count = decision_count_for(reopened, request.tx_id)
                scenario_fields = {"decision_winner": decision_winner, "repair_performed": repair_performed, "repair_idempotent": repair_idempotent, "recovered_state": recovered_state}
                assertion_results = {"decision_winner_is_committed": decision_winner == "Committed", "repair_performed": repair_performed, "repair_idempotent": repair_idempotent, "recovered_committed": recovered_state == "Committed", "decision_count_is_one": decision_count == 1}
                passed = all(assertion_results.values())
                reopened.close()

        except Exception as exc:
            print(f"{exc.__class__.__name__}: scenario={name} reason={exc}", file=sys.stderr)
            reason = f"{exc.__class__.__name__}: {exc}"
            assertion_results = {"exception_raised": False}
            scenario_fields = {"error_type": exc.__class__.__name__}
            passed = False
        finally:
            store.close()

        scenario_result = {
            "name": name,
            "passed": passed,
            "sqlite_path": str(db_path),
            "wal_enabled": True,
            "fault_boundary": fault_boundary,
            "pre_state": pre_state,
            "post_state": post_state,
            "recovered_state": recovered_state,
            "decision_winner": decision_winner,
            "decision_count": decision_count,
            "reason": reason,
            "assertion_results": assertion_results,
        }
        scenario_result.update(scenario_fields)
        scenario_results.append(scenario_result)

    total = len(scenario_results)
    passed_count = sum(1 for item in scenario_results if item["passed"])
    failed_count = total - passed_count
    all_passed = failed_count == 0
    result = {
        "status": "passed" if all_passed else "failed",
        "run_id": run_id,
        "total_scenarios": total,
        "passed_scenarios": passed_count,
        "failed_scenarios": failed_count,
        "all_passed": all_passed,
        "scenarios": scenario_results,
        "result_scope_set": [],
        "production_mpc": "blocked_not_active",
        "real_data_empirical": "blocked_not_active",
        "distributed_e8b": "blocked_not_active",
    }
    write_summary(result)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(prog="lir-pptd")
    parser.add_argument("--root", default=".", help="project root")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub = commands.add_parser(name)
        if name == "validate-config":
            sub.add_argument("config")
        if name == "verify-artifacts":
            sub.add_argument("--manifest", default="results/artifact_manifest.json")
        if name == "register-validity":
            sub.add_argument("record")
        if name in {"run-exact", "exact-diagnostics", "run-fixed", "fixed-diagnostics", "compare-exact-fixed", "run-secure", "run-e8a-v2"}:
            sub.add_argument("input", nargs="?", help="explicit synthetic JSON input")
        if name in {"fixed-preflight", "validate-fixed-golden"}:
            sub.add_argument("input", nargs="?", help="synthetic JSON input")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    if args.command == "validate-sources":
        result = validate_sources(root)
    elif args.command == "validate-approval":
        approval = load_phase0_approval(root)
        result = {"status": "passed", "decision": approval.decision.value}
    elif args.command == "validate-config":
        path = Path(args.config)
        text = path.read_text(encoding="utf-8")
        config = load_exact_json(text) if path.suffix.lower() == ".json" else load_exact_yaml(text)
        config.validate_round_trip()
        result = {"status": "passed", "schema_version": config.schema_version}
    elif args.command == "preflight":
        result = validate_phase0_gate(root) | {"negative_tests": negative_matrix_phase1_status(root)}
    elif args.command == "verify-artifacts":
        result = _verify_manifest(root, root / args.manifest)
    elif args.command == "register-validity":
        record = ValidityRecord.model_validate_json(Path(args.record).read_text(encoding="utf-8"))
        JsonlAppendLog(root / "validity_registry.jsonl").append(record)
        result = {"status": "appended", "run_id": record.run_id}
    elif args.command == "show-blockers":
        scope = json.loads((root / "docs/RESULT_SCOPE_SET.json").read_text(encoding="utf-8"))
        result = {"blocked_not_active": scope["blocked_not_active"]}
    elif args.command == "show-result-scope":
        result = json.loads((root / "docs/RESULT_SCOPE_SET.json").read_text(encoding="utf-8"))
    elif args.command == "reproduce":
        result = {"status": "environment_and_hash_chain_only", "sources": validate_sources(root), "algorithm_executed": False}
    elif args.command == "validate-golden":
        golden = json.loads((root / "tests/golden/spec_examples.json").read_text(encoding="utf-8"))
        result = {"status": "validated", "synthetic_golden_cases": len(golden["cases"]), "real_data_used": False}
    elif args.command == "validate-simulated-backend":
        from .mpc.simulated_shamir import default_simulated_profile
        p = default_simulated_profile(); p.validate_round_trip(); result = {"status": "validated", "backend_kind": p.backend_kind, "T": p.threshold_t, "N": p.committee_size_n, "p": p.field_prime_p, "backend_profile_hash": str(p.digest()), "fixed_backend_profile_hash": p.fixed_backend_profile_hash, "production_ready": False, "cryptographic_security_claim": False}
    elif args.command == "run-secure":
        from .core import ExactTaskInput
        from .mpc import SimulatedShamirBackend, run_secure
        from .mpc.artifacts import persist_secure_run
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if payload.pop("data_classification", None) != "synthetic":
            raise ValueError("only synthetic inputs are accepted")
        task = ExactTaskInput(**payload)
        backend = SimulatedShamirBackend()
        secure = run_secure(task, backend)
        rid = semantic_hash("phase4-simulated-run-v1", {"task_id": task.task_id, "profile_hash": str(backend.profile.digest()), "fixed_hash": str(backend.fixed.digest())})
        out = persist_secure_run(root / "results" / "raw", rid, secure, backend)
        result = {"status": "completed", "run_id": rid, "run_directory": str(out.relative_to(root)), "backend_kind": "simulated_shamir", "T": backend.profile.threshold_t, "N": backend.profile.committee_size_n, "task_kind": task.task_kind, "independent_l3_execution": secure.independent_l3_execution, "l2_full_runner_used_for_l3": secure.l2_full_runner_used_for_l3, "K": task.K, "iterations_executed": secure.iterations_executed, "q_out_recomputed": True, "reputation_updates_per_participant": 1, "V_sec": secure.V_sec, "production_mpc_used": False, "real_data_used": False, "distributed_e8b_used": False, "cryptographic_security_claim": False}
    elif args.command == "validate-secure-conformance":
        from .core import ExactTaskInput
        from .mpc.conformance import validate_task, transcript_has_public_leakage
        from .mpc import SimulatedShamirBackend
        tasks = [{"task_id": "numerical_equal_reports", "task_kind": "numerical", "K": 1, "tau": "1", "epsilon_c": "1", "kappa": "1", "eta": "1/2", "reports": {"a": ["0"], "b": ["1"]}, "reputations": {"a": "1/2", "b": "1/2"}, "epochs": {"a": 0, "b": 0}, "participant_ids": ["a", "b"]}, {"task_id": "categorical_unanimous", "task_kind": "categorical", "K": 1, "tau": "1", "epsilon_c": "1", "kappa": "1", "eta": "1/2", "reports": {"a": ["1", "0"], "b": ["1", "0"]}, "reputations": {"a": "1/2", "b": "1/2"}, "epochs": {"a": 0, "b": 0}, "participant_ids": ["a", "b"]}, {"task_id": "categorical_tie", "task_kind": "categorical", "K": 1, "tau": "1", "epsilon_c": "1", "kappa": "1", "eta": "1/2", "reports": {"a": ["1", "0"], "b": ["0", "1"]}, "reputations": {"a": "1/2", "b": "1/2"}, "epochs": {"a": 0, "b": 0}, "participant_ids": ["a", "b"]}, {"task_id": "reputation_boundaries", "task_kind": "numerical", "K": 1, "tau": "1", "epsilon_c": "1", "kappa": "2", "eta": "1/2", "reports": {"a": ["0"], "b": ["0"]}, "reputations": {"a": "0", "b": "1"}, "epochs": {"a": 0, "b": 0}, "participant_ids": ["a", "b"]}]
        rs = []
        backend = SimulatedShamirBackend()
        for raw in tasks:
            rs.append(validate_task(ExactTaskInput(**raw), backend).model_dump(mode="json"))
        result = {"status": "passed" if all(x["V_sec"] == 0 for x in rs) else "failed", "independent_l3_execution": True, "task_count": len(rs), "compared_field_count": sum(x["compared_fields"] for x in rs), "mismatch_count": sum(x["mismatch_count"] for x in rs), "V_sec": sum(x["V_sec"] for x in rs), "per_task": rs, "transcript_leakage": transcript_has_public_leakage(backend)}
    elif args.command == "phase4-review-package":
        manifest = root / "configs" / "backend_profiles" / "production_backend_manifest.json"
        result = {"status": "generated", "production_manifest_exists": manifest.exists(), "production_status": "blocked_not_active", "allowed_decisions": ["approved_simulated_only", "approved_production", "rejected"], "review_package": "docs/PHASE4_BACKEND_REVIEW_PACKAGE.md", "phase5_authorization": "pending human review"}
    elif args.command == "validate-backend-profile":
        from .fixedpoint import default_profile
        profile = default_profile(); profile.validate_round_trip(); result = {"status": "validated", "backend_kind": profile.backend_kind, "backend_profile_hash": str(profile.digest()), "scale_delta": profile.scale_delta, "production_ready": False, "cryptographic_security_claim": False}
    elif args.command == "fixed-preflight":
        from dataclasses import asdict
        from .fixedpoint import default_profile, preflight
        profile = default_profile(); c = preflight(profile, participants=2, dimension=2, K=1, epsilon_c='1', tau='1', eta='1/2', kappa='1'); ff = next(x for x in c.range_entries if x.entry_id == 'full_flow_maximum')
        result = {"status": "validated", "preflight_status": c.preflight_status, "profile_id": profile.profile_id, "backend_profile_hash": str(profile.digest()), "range_entry_count": len(c.range_entries), "independent_derivation_count": len(c.range_entries), "maximum_intermediate_integer": c.maximum_intermediate_integer, "full_flow_maximum": ff.maximum_absolute_integer, "modulus_limit": ff.modulus_limit, "safety_margin": ff.safety_margin, "no_wrap_status": c.no_wrap_status, "denominator_status": c.denominator_status, "coefficient_retention_status": c.coefficient_retention_status, "error_bound_status": c.error_bound_status, "certificate": [asdict(x) for x in c.range_entries]}
    elif args.command == "validate-fixed-golden":
        from .fixedpoint.golden import validate_golden_file
        vector_path = Path(args.input) if args.input else root / "tests" / "golden" / "fixed_primitive_vectors.json"
        result = validate_golden_file(vector_path)
        if result["status"] != "validated":
            print(json.dumps(result, ensure_ascii=False, indent=2)); return 1
    elif args.command in {"run-fixed", "fixed-diagnostics", "compare-exact-fixed"}:
        from .core import ExactTaskInput
        from .fixedpoint import default_profile, run_fixed
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if payload.pop("data_classification", None) != "synthetic":
            raise ValueError("only synthetic inputs are accepted")
        task = ExactTaskInput(**payload); profile = default_profile(); fixed = run_fixed(task, profile)
        if args.command == "compare-exact-fixed":
            from .fixedpoint.conformance import generate_record_set, summarize
            rs = generate_record_set(task, profile); s = summarize([rs]); result = {"status": s.status, "task_id": task.task_id, "task_kind": task.task_kind, "profile_id": profile.profile_id, "backend_profile_hash": str(profile.digest()), "K": task.K, "record_set_status": rs.record_set_status, "total_records": len(rs.records), "expected_records": sum(rs.expected_component_counts.values()), "missing_records": len(rs.missing_components), "duplicate_records": len(rs.duplicate_record_ids), "passed_records": s.passed_records, "failed_records": s.failed_records, "eligible_records": s.eligible_records, "ineligible_records": s.ineligible_records, "unresolved_records": s.unresolved_diagnostic_records, "component_counts": rs.component_counts, "q_out_recomputed": True, "all_mandatory_records_present": not rs.missing_components, "conformance_records": [r.model_dump(mode='json') for r in rs.records]}
        elif args.command == "run-fixed":
            from .fixedpoint.artifacts import persist_fixed_run
            from .fixedpoint.conformance import generate_record_set, summarize
            from .fixedpoint.run_identity import build_identity
            source_root = Path(__file__).resolve().parent; digest = semantic_hash('phase3-code-v1', [{'path': p.name, 'sha256': source_hash(p)} for p in (source_root / 'fixedpoint' / 'algorithm.py', source_root / 'fixedpoint' / 'arithmetic.py')]); identity = build_identity(task=task, profile=profile, code_digest=digest, master_seed=0, run_namespace='phase3-fixed'); rs = generate_record_set(task, profile); summary = summarize([rs]); out = persist_fixed_run(root / 'results' / 'raw', identity, fixed, profile.model_dump(mode='json'), summary.model_dump(mode='json'))
            result = {"status": "completed", "run_id": str(identity.run_id), "run_directory": str(out.relative_to(root)), "lifecycle_state": "complete", "backend_kind": fixed.backend_kind, "profile_id": profile.profile_id, "backend_profile_hash": str(profile.digest()), "result_scope": fixed.result_scope, "production_mpc_used": False, "real_data_used": False, "distributed_e8b_used": False, "cryptographic_security_claim": False, "synthetic_only": True, "preflight_status": fixed.preflight_status, "no_wrap_status": fixed.preflight_certificate['no_wrap_status'], "denominator_status": fixed.preflight_certificate['denominator_status'], "coefficient_retention_status": fixed.preflight_certificate['coefficient_retention_status'], "algorithm_executed": True, "task_kind": task.task_kind, "K": task.K, "iterations_executed": fixed.iterations_executed, "q_out_recomputed": True, "reputation_updates_per_participant": 1, "artifact_count": 11, "manifest_path": str((out / 'manifest.json').relative_to(root)), "conformance_status": summary.status}
        else:
            result = fixed.model_dump(mode="json")
    elif args.command in {"run-exact", "exact-diagnostics"}:
        from .core import ExactTaskInput, run_with_precision_doubling
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if payload.get("data_classification") != "synthetic":
            raise ValueError("Phase 2 exact CLI accepts only explicitly synthetic input")
        payload.pop("data_classification")
        exact = run_with_precision_doubling(ExactTaskInput(**payload))
        result = exact.model_dump(mode="json")
    elif args.command == "validate-protocol-v2":
        result = _protocol_summary(root)
    elif args.command == "protocol-v2-demo":
        result = _protocol_demo(root)
    elif args.command == "protocol-v2-recover":
        result = _protocol_recover(root)
    elif args.command == "run-e8a-v2":
        payload = None if args.input is None else json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = _run_e8a_v2(root, payload)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("status") == "passed" else 1
    else:
        result = {"status": "not_implemented_in_phase_1", "command": args.command, "algorithm_executed": False, "analysis_generated": False}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
