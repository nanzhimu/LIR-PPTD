from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..hashing import semantic_hash
from .binding import canonical_request_hash, validate_participant_certificate, validate_prepared_record
from .model import (
    CommittedLogEntry,
    DeliveryState,
    DecisionRecord,
    IntegrityFailure,
    LookupResult,
    OutputPrepareRecord,
    OutputUnavailable,
    ParticipantCertificate,
    PreparedRecord,
    PreprocessingRecord,
    ProtocolRecord,
    ProtocolResponse,
    ProtocolState,
    ReputationPrepareRecord,
    TransactionRequest,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class StoreOutcome:
    inserted: bool
    record: ProtocolRecord


class ProtocolStoreError(RuntimeError):
    pass


class InMemoryProtocolStore:
    def __init__(self) -> None:
        self._records: dict[str, ProtocolRecord] = {}
        self._request_index: dict[str, str] = {}
        self._decision_index: dict[str, str] = {}
        self._committed_logs: dict[str, CommittedLogEntry] = {}

    def close(self) -> None:
        return None

    def _store(self, record: ProtocolRecord) -> ProtocolRecord:
        if record.request.task_id != record.task_id or record.request.tx_id != record.tx_id:
            raise ProtocolStoreError("task/tx mismatch")
        if not validate_participant_certificate(record.participant_certificate, record.request.part_hash):
            raise ProtocolStoreError("participant certificate mismatch")
        self._records[str(record.tx_id)] = record
        self._request_index[canonical_request_hash(record.request)] = str(record.tx_id)
        return record

    def lookup_by_request(self, request: TransactionRequest) -> LookupResult:
        tx_id = self._request_index.get(canonical_request_hash(request))
        if tx_id is None:
            return LookupResult(status="absent")
        record = self._records[tx_id]
        if record.request != request:
            return LookupResult(status="conflict", conflicting_record=record, reason="request mismatch")
        return LookupResult(status="exact", record=record)

    def lookup_by_tx_id(self, tx_id) -> ProtocolRecord | None:
        return self._records.get(str(tx_id))

    def _existing(self, record: ProtocolRecord) -> ProtocolRecord | None:
        existing = self._records.get(str(record.tx_id))
        if existing is not None and existing.request != record.request:
            raise ProtocolStoreError("conflicting request identity")
        return existing

    def create_locked_if_absent(self, record: ProtocolRecord) -> StoreOutcome:
        existing = self._existing(record)
        if existing is not None:
            return StoreOutcome(inserted=False, record=existing)
        if record.state != ProtocolState.LOCKED:
            raise ProtocolStoreError("expected locked state")
        self._store(record)
        return StoreOutcome(inserted=True, record=record)

    def insert_locked(self, record: ProtocolRecord) -> StoreOutcome:
        return self.create_locked_if_absent(record)

    def activate_and_transition_to_executing(self, record: ProtocolRecord) -> StoreOutcome:
        existing = self._existing(record)
        if existing is None:
            if record.state != ProtocolState.EXECUTING:
                raise ProtocolStoreError("expected executing state")
            self._store(record)
            return StoreOutcome(inserted=True, record=record)
        if existing.state != ProtocolState.LOCKED:
            raise ProtocolStoreError("activation requires locked state")
        if record.state != ProtocolState.EXECUTING:
            raise ProtocolStoreError("expected executing state")
        self._store(record)
        return StoreOutcome(inserted=False, record=record)

    def insert_executing(self, record: ProtocolRecord) -> StoreOutcome:
        return self.activate_and_transition_to_executing(record)

    def save_execution_checkpoint(self, record: ProtocolRecord) -> StoreOutcome:
        existing = self._existing(record)
        if existing is None:
            if record.state not in {ProtocolState.EXECUTING, ProtocolState.PREPARED}:
                raise ProtocolStoreError("expected executing or prepared state")
            self._store(record)
            return StoreOutcome(inserted=True, record=record)
        if existing.state == ProtocolState.LOCKED and record.state in {ProtocolState.EXECUTING, ProtocolState.PREPARED}:
            self._store(record)
            return StoreOutcome(inserted=False, record=record)
        if existing.state not in {ProtocolState.EXECUTING, ProtocolState.PREPARED}:
            raise ProtocolStoreError("checkpoint requires executing or prepared state")
        self._store(record)
        return StoreOutcome(inserted=False, record=record)

    def save_output_prepare(self, record: ProtocolRecord) -> StoreOutcome:
        return self.save_execution_checkpoint(record)

    def save_reputation_prepare(self, record: ProtocolRecord) -> StoreOutcome:
        return self.save_execution_checkpoint(record)

    def promote_to_prepared(self, record: ProtocolRecord) -> StoreOutcome:
        existing = self._existing(record)
        if existing is None:
            if record.state != ProtocolState.PREPARED or record.prepared is None:
                raise ProtocolStoreError("expected prepared record")
            self._store(record)
            return StoreOutcome(inserted=True, record=record)
        if existing.state == ProtocolState.LOCKED:
            if record.state != ProtocolState.PREPARED or record.prepared is None:
                raise ProtocolStoreError("expected prepared record")
            self._store(record)
            return StoreOutcome(inserted=False, record=record)
        if existing.state != ProtocolState.EXECUTING:
            raise ProtocolStoreError("prepared promotion requires lock or executing state")
        if record.state != ProtocolState.PREPARED or record.prepared is None or record.output_prepare is None or record.reputation_prepare is None:
            raise ProtocolStoreError("both prepare records required")
        if not validate_prepared_record(record.prepared):
            raise ProtocolStoreError("invalid prepared record")
        self._store(record)
        return StoreOutcome(inserted=False, record=record)

    def insert_prepared(self, record: ProtocolRecord) -> StoreOutcome:
        return self.promote_to_prepared(record)

    def create_restart(self, record: ProtocolRecord) -> StoreOutcome:
        return self.create_locked_if_absent(record)

    def record_decision(self, tx_id, decision_hash: str, decision: ProtocolResponse) -> ProtocolResponse:
        existing = self._decision_index.get(str(tx_id))
        if existing is not None:
            return ProtocolResponse(existing)
        self._decision_index[str(tx_id)] = decision.value
        record = self._records.get(str(tx_id))
        if record is not None:
            self._store(record.model_copy(update={"decision_winner": decision}))
        return decision

    def commit_terminal_state(self, record: ProtocolRecord) -> None:
        if record.state not in {ProtocolState.ABORTED, ProtocolState.COMMITTED}:
            raise ProtocolStoreError("terminal state required")
        if record.decision_winner is None:
            raise ProtocolStoreError("authorized decision required")
        self._store(record)

    def materialize_committed_logs(self, tx_id) -> tuple[CommittedLogEntry, ...]:
        record = self._records.get(str(tx_id))
        return () if record is None else record.committed_logs

    def repair_committed_logs(self, tx_id) -> tuple[CommittedLogEntry, ...]:
        record = self._records.get(str(tx_id))
        if record is None:
            return ()
        for entry in record.committed_logs:
            self._committed_logs[entry.refresh_id] = entry
        return record.committed_logs

    def inspect_tx(self, tx_id) -> ProtocolRecord | None:
        return self._records.get(str(tx_id))

    def reopen(self) -> "InMemoryProtocolStore":
        return self

    def snapshot(self) -> dict[str, Any]:
        return {"records": {tx_id: record.model_dump(mode="json") for tx_id, record in self._records.items()}, "committed_logs": {rid: entry.model_dump(mode="json") for rid, entry in self._committed_logs.items()}}


class SQLiteProtocolStore:
    schema_version = 3

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._decision_index: dict[str, str] = {}
        self._load_decision_index()
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def _load_decision_index(self) -> None:
        try:
            rows = self.connection.execute("SELECT tx_id, decision_winner FROM protocol_transactions WHERE decision_winner IS NOT NULL").fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            self._decision_index[str(row["tx_id"])] = row["decision_winner"]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _ensure_schema(self) -> None:
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS protocol_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("INSERT OR IGNORE INTO protocol_meta(key, value) VALUES ('schema_version', ?)", (str(self.schema_version),))
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS protocol_transactions (
                    tx_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL UNIQUE,
                    request_json TEXT NOT NULL,
                    participant_json TEXT NOT NULL,
                    protocol_state TEXT NOT NULL,
                    response TEXT,
                    delivery TEXT,
                    lock_certificate_hash TEXT,
                    preprocessing_manifest_hash TEXT,
                    preprocessing_bundle_hash TEXT,
                    preprocessing_activation_hash TEXT,
                    preprocessing_records_json TEXT NOT NULL DEFAULT '[]',
                    output_prepare_json TEXT,
                    reputation_prepare_json TEXT,
                    prepared_json TEXT,
                    decision_json TEXT,
                    decision_hash TEXT,
                    commit_certificate_hash TEXT,
                    abort_auth_hash TEXT,
                    decision_winner TEXT,
                    committed_epoch_vector TEXT,
                    committed_output_share_count INTEGER NOT NULL DEFAULT 0,
                    committed_reputation_share_count INTEGER NOT NULL DEFAULT 0,
                    committed_logs_json TEXT NOT NULL DEFAULT '[]',
                    output_unavailable_json TEXT,
                    integrity_failure_json TEXT,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    cfg_hash TEXT NOT NULL,
                    part_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    requester_id TEXT NOT NULL
                )
                """
            )
            self.connection.execute("CREATE TABLE IF NOT EXISTS protocol_preprocessing_records (record_id TEXT PRIMARY KEY, tx_id TEXT NOT NULL, op_id TEXT NOT NULL, operation_kind TEXT NOT NULL, record_json TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0, UNIQUE(tx_id, op_id, operation_kind))")
            self.connection.execute("CREATE TABLE IF NOT EXISTS protocol_prepared_outputs (record_id TEXT PRIMARY KEY, tx_id TEXT NOT NULL, record_kind TEXT NOT NULL, worker_id TEXT NOT NULL, coordinate_index INTEGER NOT NULL, record_json TEXT NOT NULL, UNIQUE(tx_id, record_kind, worker_id, coordinate_index))")
            self.connection.execute("CREATE TABLE IF NOT EXISTS protocol_committed_logs (refresh_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, epoch INTEGER NOT NULL, tx_id TEXT NOT NULL, commit_certificate_hash TEXT NOT NULL, log_json TEXT NOT NULL, UNIQUE(tx_id, worker_id, epoch))")
            self.connection.execute("CREATE TABLE IF NOT EXISTS protocol_fault_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, tx_id TEXT, hook TEXT NOT NULL, payload_json TEXT NOT NULL)")

    def _row_to_record(self, row: sqlite3.Row) -> ProtocolRecord:
        request = TransactionRequest.model_validate_json(row["request_json"])
        participant = ParticipantCertificate.model_validate_json(row["participant_json"])
        output_prepare = OutputPrepareRecord.model_validate_json(row["output_prepare_json"]) if row["output_prepare_json"] else None
        reputation_prepare = ReputationPrepareRecord.model_validate_json(row["reputation_prepare_json"]) if row["reputation_prepare_json"] else None
        prepared = PreparedRecord.model_validate_json(row["prepared_json"]) if row["prepared_json"] else None
        decision = DecisionRecord.model_validate_json(row["decision_json"]) if row["decision_json"] else None
        return ProtocolRecord(
            task_id=request.task_id,
            tx_id=request.tx_id,
            request=request,
            participant_certificate=participant,
            state=ProtocolState(row["protocol_state"]),
            response=ProtocolResponse(row["response"]) if row["response"] else None,
            delivery=DeliveryState(row["delivery"]) if row["delivery"] else None,
            lock_certificate_hash=row["lock_certificate_hash"],
            preprocessing_manifest_hash=row["preprocessing_manifest_hash"],
            preprocessing_bundle_hash=row["preprocessing_bundle_hash"],
            preprocessing_activation_hash=row["preprocessing_activation_hash"],
            preprocessing_records=tuple(PreprocessingRecord.model_validate(item) for item in json.loads(row["preprocessing_records_json"])),
            output_prepare=output_prepare,
            reputation_prepare=reputation_prepare,
            prepared=prepared,
            decision=decision,
            commit_certificate_hash=row["commit_certificate_hash"],
            abort_auth_hash=row["abort_auth_hash"],
            decision_winner=ProtocolResponse(row["decision_winner"]) if row["decision_winner"] else None,
            committed_epoch_vector=tuple(json.loads(row["committed_epoch_vector"])) if row["committed_epoch_vector"] else None,
            committed_output_share_count=row["committed_output_share_count"],
            committed_reputation_share_count=row["committed_reputation_share_count"],
            committed_logs=tuple(CommittedLogEntry.model_validate(item) for item in json.loads(row["committed_logs_json"])),
            output_unavailable=OutputUnavailable.model_validate_json(row["output_unavailable_json"]) if row["output_unavailable_json"] else None,
            integrity_failure=IntegrityFailure.model_validate_json(row["integrity_failure_json"]) if row["integrity_failure_json"] else None,
            created_utc=row["created_utc"],
            updated_utc=row["updated_utc"],
            metadata={},
        )

    def lookup_by_request(self, request: TransactionRequest) -> LookupResult:
        row = self.connection.execute("SELECT * FROM protocol_transactions WHERE request_hash=?", (canonical_request_hash(request),)).fetchone()
        if row is None:
            return LookupResult(status="absent")
        record = self._row_to_record(row)
        if record.request != request:
            return LookupResult(status="conflict", conflicting_record=record, reason="request mismatch")
        return LookupResult(status="exact", record=record)

    def lookup_by_tx_id(self, tx_id) -> ProtocolRecord | None:
        row = self.connection.execute("SELECT * FROM protocol_transactions WHERE tx_id=?", (str(tx_id),)).fetchone()
        return None if row is None else self._row_to_record(row)

    def _store_record(self, record: ProtocolRecord) -> None:
        if record.request.task_id != record.task_id or record.request.tx_id != record.tx_id:
            raise ProtocolStoreError("task/tx mismatch")
        if not validate_participant_certificate(record.participant_certificate, record.request.part_hash):
            raise ProtocolStoreError("participant certificate mismatch")
        existing = self.lookup_by_tx_id(record.tx_id)
        if existing is None:
            self._insert_new(record)
            return
        if existing.request != record.request:
            raise ProtocolStoreError("conflicting request identity")
        self._write_update(record)

    def _upsert(self, record: ProtocolRecord) -> StoreOutcome:
        existing = self.lookup_by_tx_id(record.tx_id)
        if existing is None:
            return StoreOutcome(inserted=True, record=self._insert_new(record))
        if existing.request != record.request:
            raise ProtocolStoreError("conflicting request identity")
        self._write_update(record)
        return StoreOutcome(inserted=False, record=record)

    def _insert_new(self, record: ProtocolRecord) -> ProtocolRecord:
        with self.transaction():
            self.connection.execute(
                """
                INSERT INTO protocol_transactions(
                    tx_id, request_hash, request_json, participant_json, protocol_state,
                    response, delivery, lock_certificate_hash, preprocessing_manifest_hash,
                    preprocessing_bundle_hash, preprocessing_activation_hash,
                    preprocessing_records_json, output_prepare_json, reputation_prepare_json,
                    prepared_json, decision_json, decision_hash, commit_certificate_hash,
                    abort_auth_hash, decision_winner, committed_epoch_vector,
                    committed_output_share_count, committed_reputation_share_count,
                    committed_logs_json, output_unavailable_json, integrity_failure_json,
                    created_utc, updated_utc, cfg_hash, part_hash, task_id, attempt_id, requester_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.tx_id),
                    canonical_request_hash(record.request),
                    record.request.model_dump_json(),
                    record.participant_certificate.model_dump_json(),
                    record.state.value,
                    record.response.value if record.response else None,
                    record.delivery.value if record.delivery else None,
                    record.lock_certificate_hash,
                    record.preprocessing_manifest_hash,
                    record.preprocessing_bundle_hash,
                    record.preprocessing_activation_hash,
                    _json([item.model_dump(mode="json") for item in record.preprocessing_records]),
                    record.output_prepare.model_dump_json() if record.output_prepare else None,
                    record.reputation_prepare.model_dump_json() if record.reputation_prepare else None,
                    record.prepared.model_dump_json() if record.prepared else None,
                    record.decision.model_dump_json() if record.decision else None,
                    record.decision.decision_hash() if record.decision else None,
                    record.commit_certificate_hash,
                    record.abort_auth_hash,
                    record.decision_winner.value if record.decision_winner else None,
                    _json(list(record.committed_epoch_vector)) if record.committed_epoch_vector is not None else None,
                    record.committed_output_share_count,
                    record.committed_reputation_share_count,
                    _json([item.model_dump(mode="json") for item in record.committed_logs]),
                    record.output_unavailable.model_dump_json() if record.output_unavailable else None,
                    record.integrity_failure.model_dump_json() if record.integrity_failure else None,
                    record.created_utc or _utc_now(),
                    record.updated_utc or _utc_now(),
                    record.request.cfg_hash,
                    record.request.part_hash,
                    record.request.task_id,
                    record.request.attempt_id,
                    record.request.requester_id,
                ),
            )
        return record

    def _write_update(self, record: ProtocolRecord) -> None:
        with self.transaction():
            self.connection.execute(
                """
                UPDATE protocol_transactions SET
                    request_hash=?, request_json=?, participant_json=?, protocol_state=?, response=?, delivery=?,
                    lock_certificate_hash=?, preprocessing_manifest_hash=?, preprocessing_bundle_hash=?, preprocessing_activation_hash=?,
                    preprocessing_records_json=?, output_prepare_json=?, reputation_prepare_json=?, prepared_json=?,
                    decision_json=?, decision_hash=?, commit_certificate_hash=?, abort_auth_hash=?, decision_winner=?,
                    committed_epoch_vector=?, committed_output_share_count=?, committed_reputation_share_count=?, committed_logs_json=?,
                    output_unavailable_json=?, integrity_failure_json=?, updated_utc=?, cfg_hash=?, part_hash=?, task_id=?, attempt_id=?, requester_id=?
                WHERE tx_id=?
                """,
                (
                    canonical_request_hash(record.request),
                    record.request.model_dump_json(),
                    record.participant_certificate.model_dump_json(),
                    record.state.value,
                    record.response.value if record.response else None,
                    record.delivery.value if record.delivery else None,
                    record.lock_certificate_hash,
                    record.preprocessing_manifest_hash,
                    record.preprocessing_bundle_hash,
                    record.preprocessing_activation_hash,
                    _json([item.model_dump(mode="json") for item in record.preprocessing_records]),
                    record.output_prepare.model_dump_json() if record.output_prepare else None,
                    record.reputation_prepare.model_dump_json() if record.reputation_prepare else None,
                    record.prepared.model_dump_json() if record.prepared else None,
                    record.decision.model_dump_json() if record.decision else None,
                    record.decision.decision_hash() if record.decision else None,
                    record.commit_certificate_hash,
                    record.abort_auth_hash,
                    record.decision_winner.value if record.decision_winner else None,
                    _json(list(record.committed_epoch_vector)) if record.committed_epoch_vector is not None else None,
                    record.committed_output_share_count,
                    record.committed_reputation_share_count,
                    _json([item.model_dump(mode="json") for item in record.committed_logs]),
                    record.output_unavailable.model_dump_json() if record.output_unavailable else None,
                    record.integrity_failure.model_dump_json() if record.integrity_failure else None,
                    record.updated_utc or _utc_now(),
                    record.request.cfg_hash,
                    record.request.part_hash,
                    record.request.task_id,
                    record.request.attempt_id,
                    record.request.requester_id,
                    str(record.tx_id),
                ),
            )

    def create_locked_if_absent(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record)

    def insert_locked(self, record: ProtocolRecord) -> StoreOutcome:
        return self.create_locked_if_absent(record)

    def activate_and_transition_to_executing(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record.model_copy(update={"state": ProtocolState.EXECUTING}))

    def insert_executing(self, record: ProtocolRecord) -> StoreOutcome:
        return self.activate_and_transition_to_executing(record)

    def save_execution_checkpoint(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record)

    def save_output_prepare(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record)

    def save_reputation_prepare(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record)

    def promote_to_prepared(self, record: ProtocolRecord) -> StoreOutcome:
        return self._upsert(record.model_copy(update={"state": ProtocolState.PREPARED}))

    def insert_prepared(self, record: ProtocolRecord) -> StoreOutcome:
        return self.promote_to_prepared(record)

    def create_restart(self, record: ProtocolRecord) -> StoreOutcome:
        return self.create_locked_if_absent(record)

    def record_decision(self, tx_id, decision_hash: str, decision: ProtocolResponse) -> ProtocolResponse:
        existing = self._decision_index.get(str(tx_id))
        if existing is not None:
            return ProtocolResponse(existing)
        with self.transaction():
            row = self.connection.execute("SELECT decision_winner FROM protocol_transactions WHERE tx_id=?", (str(tx_id),)).fetchone()
            if row is None:
                raise ProtocolStoreError("unknown transaction")
            if row[0]:
                self._decision_index[str(tx_id)] = row[0]
                return ProtocolResponse(row[0])
            self.connection.execute(
                "UPDATE protocol_transactions SET decision_hash=?, decision_winner=?, updated_utc=? WHERE tx_id=?",
                (decision_hash, decision.value, _utc_now(), str(tx_id)),
            )
        self._decision_index[str(tx_id)] = decision.value
        return decision

    def commit_terminal_state(self, record: ProtocolRecord) -> None:
        if record.state not in {ProtocolState.ABORTED, ProtocolState.COMMITTED}:
            raise ProtocolStoreError("terminal state required")
        if record.decision_winner is None:
            raise ProtocolStoreError("authorized decision required")
        self._store_record(record)

    def materialize_committed_logs(self, tx_id) -> tuple[CommittedLogEntry, ...]:
        record = self.lookup_by_tx_id(tx_id)
        return () if record is None else record.committed_logs

    def repair_committed_logs(self, tx_id) -> tuple[CommittedLogEntry, ...]:
        record = self.lookup_by_tx_id(tx_id)
        if record is None:
            return ()
        with self.transaction():
            for entry in record.committed_logs:
                self.connection.execute(
                    "INSERT OR IGNORE INTO protocol_committed_logs(refresh_id, worker_id, epoch, tx_id, commit_certificate_hash, log_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (entry.refresh_id, str(entry.worker_id), entry.epoch, str(entry.tx_id), entry.commit_certificate_hash, entry.model_dump_json()),
                )
        return record.committed_logs

    def inspect_tx(self, tx_id) -> ProtocolRecord | None:
        return self.lookup_by_tx_id(tx_id)

    def reopen(self) -> "SQLiteProtocolStore":
        self.connection.close()
        return SQLiteProtocolStore(self.path)

    def snapshot(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT * FROM protocol_transactions").fetchall()
        return {"records": [dict(row) for row in rows]}
