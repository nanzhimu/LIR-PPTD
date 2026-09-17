from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .binding import (
    build_commit_certificate,
    canonical_part_hash,
    canonical_request_hash,
    failure_result,
    validate_commit_certificate,
    validate_participant_certificate,
    validate_prepared_record,
    validate_preprocessing_bundle,
    validate_preprocessing_manifest,
)
from .decision import build_abort_decision, build_commit_decision, build_output_prepare, build_prepared_record, build_reputation_prepare, record_decision as build_decision_record, validate_prepared_bundle
from .locking import acquire_lock, authenticate_request_before_lookup
from .model import (
    DeliveryState,
    FaultHook,
    ExecutionCheckpoint,
    IntegrityFailure,
    OperationKind,
    ParticipantCertificate,
    PreprocessingRecord,
    ProtocolRecord,
    ProtocolResponse,
    ProtocolState,
    TypedProtocolResult,
)
from .preprocessing import activate_preprocessing, consume_preprocessing_record, generate_preprocessing_artifacts
from .quorum import (
    committed_state_availability,
    fixed_attempt_continuation_quorum,
    input_readiness_quorum,
    joint_commit_authorization_quorum,
    output_prepare_quorum,
    output_reconstruction_quorum,
    output_storage_quorum,
    preprocessing_readiness_quorum,
    reputation_lock_quorum,
    reputation_prepare_quorum,
    reputation_recovery_quorum,
    reputation_storage_quorum,
    restart_admissibility_quorum,
    whole_task_restart_quorum,
    upload_storage_quorum,
)
from .recovery import recover_record
from .store import InMemoryProtocolStore, SQLiteProtocolStore


@dataclass
class ProtocolEngine:
    store: InMemoryProtocolStore | SQLiteProtocolStore
    fault_hooks: set[FaultHook] = field(default_factory=set)

    def handle(self, request, *, context: dict[str, Any] | None = None) -> TypedProtocolResult:
        context = {} if context is None else dict(context)
        try:
            existing = authenticate_request_before_lookup(request, self.store)
        except ValueError:
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REPLAY, message="conflicting replay")

        if existing is not None:
            if existing.request != request:
                return TypedProtocolResult(response=ProtocolResponse.REJECTED_REPLAY, state=existing.state, message="conflicting request identity")
            return self._recover_existing(existing, request, context=context)

        return self._run_new(request, context=context)

    def _recover_existing(self, record: ProtocolRecord, request, *, context: dict[str, Any]) -> TypedProtocolResult:
        if record.request != request:
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REPLAY, state=record.state, message="conflicting request identity")
        if record.integrity_failure is not None:
            return TypedProtocolResult(response=ProtocolResponse.INTEGRITY_FAILURE, integrity_failure=record.integrity_failure)
        return recover_record(self.store, request, context=context)

    def _run_new(self, request, *, context: dict[str, Any]) -> TypedProtocolResult:
        participant_certificate: ParticipantCertificate | None = context.get("participant_certificate")
        if participant_certificate is None:
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, message="missing participant certificate")
        if request.requester_id != context.get("requester_id", request.requester_id):
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, message="requester attestation mismatch")
        if not validate_participant_certificate(participant_certificate, request.part_hash):
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, message="participant certificate mismatch")

        complete_candidates = context.get("complete_candidates", 0)
        minimum_participants = context.get("minimum_participants", 0)
        maximum_participants = context.get("maximum_participants", 0)
        if not (minimum_participants <= complete_candidates <= maximum_participants):
            return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=DeliveryState.NONE, message="InsufficientParticipants")

        if not context.get("request_authenticated", True):
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, message="request authentication failed")

        epoch_snapshot = tuple(context.get("epoch_snapshot", participant_certificate.reputation_epoch_snapshot))
        if not epoch_snapshot:
            epoch_snapshot = participant_certificate.reputation_epoch_snapshot
        lock_certificate = acquire_lock(store=self.store, request=request, epoch_snapshot=epoch_snapshot, locked_committee=tuple(context.get("locked_committee", ())), reason_code=context.get("lock_reason"))
        locked = ProtocolRecord(
            task_id=request.task_id,
            tx_id=request.tx_id,
            request=request,
            participant_certificate=participant_certificate,
            state=ProtocolState.LOCKED,
            response=ProtocolResponse.ABORTED,
            delivery=DeliveryState.NONE,
            lock_certificate_hash=lock_certificate.lock_hash(),
        )
        self.store.create_locked_if_absent(locked)

        if context.get("activate_after_lock", True) is False:
            return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.LOCKED, delivery=DeliveryState.NONE, message="locked only")

        artifacts = generate_preprocessing_artifacts(
            task_id=request.task_id,
            tx_id=request.tx_id,
            attempt_id=request.attempt_id,
            cfg_hash=request.cfg_hash,
            part_hash=request.part_hash,
            generation=context.get("generation", 0),
            committee_hash=context.get("committee_hash", "committee-hash"),
            record_specs=context.get(
                "record_specs",
                (("output-record", "op-output", OperationKind.OUTPUT_PREPARE), ("reputation-record", "op-reputation", OperationKind.REPUTATION_PREPARE)),
            ),
            bundle_id=context.get("bundle_id", "bundle-001"),
        )
        if not validate_preprocessing_manifest(artifacts.manifest):
            return TypedProtocolResult(response=ProtocolResponse.INTEGRITY_FAILURE, integrity_failure=IntegrityFailure(reason_code="INVALID_MANIFEST", message="invalid preprocessing manifest", tx_id=request.tx_id))
        if not validate_preprocessing_bundle(artifacts.bundle, artifacts.manifest):
            return TypedProtocolResult(response=ProtocolResponse.INTEGRITY_FAILURE, integrity_failure=IntegrityFailure(reason_code="INVALID_BUNDLE", message="invalid preprocessing bundle", tx_id=request.tx_id))

        checkpoint = ExecutionCheckpoint(
            task_id=request.task_id,
            tx_id=request.tx_id,
            attempt_id=request.attempt_id,
            cfg_hash=request.cfg_hash,
            part_hash=request.part_hash,
            generation=context.get("generation", 0),
            lock_certificate_hash=lock_certificate.lock_hash(),
            preprocessing_manifest_hash=artifacts.manifest.manifest_hash(),
            preprocessing_bundle_hash=artifacts.bundle.bundle_hash(),
        )
        self.store.save_execution_checkpoint(locked.model_copy(update={
            "state": ProtocolState.EXECUTING,
            "preprocessing_manifest_hash": artifacts.manifest.manifest_hash(),
            "preprocessing_bundle_hash": artifacts.bundle.bundle_hash(),
            "preprocessing_records": artifacts.records,
            "preprocessing_activation_hash": artifacts.bundle.bundle_hash(),
        }))

        if context.get("crash_after_activation"):
            return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.EXECUTING, delivery=DeliveryState.NONE, message="activation boundary")

        if context.get("prepare_only", False):
            output_prepare = build_output_prepare(
                task_id=request.task_id,
                tx_id=request.tx_id,
                part_hash=request.part_hash,
                attempt_id=request.attempt_id,
                record_id=context.get("output_record_id", "output-record"),
                manifest_hash=artifacts.manifest.manifest_hash(),
                committee_hash=context.get("committee_hash", "committee-hash"),
                storage_set=tuple(context.get("output_storage_set", participant_certificate.common_report_storage_set)),
            )
            reputation_prepare = build_reputation_prepare(
                task_id=request.task_id,
                tx_id=request.tx_id,
                part_hash=request.part_hash,
                attempt_id=request.attempt_id,
                record_id=context.get("reputation_record_id", "reputation-record"),
                manifest_hash=artifacts.manifest.manifest_hash(),
                committee_hash=context.get("committee_hash", "committee-hash"),
                storage_set=tuple(context.get("reputation_storage_set", participant_certificate.common_report_storage_set)),
            )
            prepared = build_prepared_record(
                task_id=request.task_id,
                tx_id=request.tx_id,
                attempt_id=request.attempt_id,
                cfg_hash=request.cfg_hash,
                part_hash=request.part_hash,
                lock_certificate_hash=lock_certificate.lock_hash(),
                epoch_snapshot=epoch_snapshot,
                output_prepare_hash=output_prepare.record_hash(),
                reputation_prepare_hash=reputation_prepare.record_hash(),
                output_manifest_hash=artifacts.manifest.manifest_hash(),
                reputation_manifest_hash=artifacts.manifest.manifest_hash(),
            )
            if not validate_prepared_record(prepared):
                return TypedProtocolResult(response=ProtocolResponse.INTEGRITY_FAILURE, integrity_failure=IntegrityFailure(reason_code="INVALID_PREPARED", message="invalid prepared record", tx_id=request.tx_id))
            self.store.promote_to_prepared(locked.model_copy(update={"state": ProtocolState.PREPARED, "output_prepare": output_prepare, "reputation_prepare": reputation_prepare, "prepared": prepared}))
            if context.get("crash_before_decision"):
                return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.PREPARED, delivery=DeliveryState.NONE, message="prepared boundary")
            if context.get("decision") == "Commit" and context.get("commit_certificate") is not None:
                certificate = context["commit_certificate"]
                if not validate_commit_certificate(certificate, prepared):
                    return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, state=ProtocolState.PREPARED, message="invalid commit certificate")
                self.store.record_decision(request.tx_id, certificate.certificate_hash(), ProtocolResponse.COMMITTED)
                committed = locked.model_copy(update={
                    "state": ProtocolState.COMMITTED,
                    "response": ProtocolResponse.COMMITTED,
                    "delivery": DeliveryState.SECURE_OUTPUT,
                    "output_prepare": output_prepare,
                    "reputation_prepare": reputation_prepare,
                    "prepared": prepared,
                    "decision_winner": ProtocolResponse.COMMITTED,
                    "commit_certificate_hash": certificate.certificate_hash(),
                    "committed_epoch_vector": certificate.epoch_snapshot,
                    "committed_output_share_count": len(output_prepare.storage_set),
                    "committed_reputation_share_count": len(reputation_prepare.storage_set),
                })
                self.store.commit_terminal_state(committed)
                self.store.repair_committed_logs(request.tx_id)
                return TypedProtocolResult(response=ProtocolResponse.COMMITTED, state=ProtocolState.COMMITTED, delivery=DeliveryState.SECURE_OUTPUT, committed_epoch_vector=certificate.epoch_snapshot)
            if context.get("decision") == "Abort":
                reason = context.get("abort_reason", "RecoveryNoDecision")
                auth, _decision = build_abort_decision(request=request, prepared=prepared, reason_code=reason, requester_id=request.requester_id)
                self.store.record_decision(request.tx_id, auth.auth_hash(), ProtocolResponse.ABORTED)
                aborted = locked.model_copy(update={
                    "state": ProtocolState.ABORTED,
                    "response": ProtocolResponse.ABORTED,
                    "delivery": DeliveryState.NONE,
                    "output_prepare": output_prepare,
                    "reputation_prepare": reputation_prepare,
                    "prepared": prepared,
                    "decision_winner": ProtocolResponse.ABORTED,
                    "abort_auth_hash": auth.auth_hash(),
                })
                self.store.commit_terminal_state(aborted)
                return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=DeliveryState.NONE)
            return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.PREPARED, delivery=DeliveryState.NONE)

        return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.EXECUTING, delivery=DeliveryState.NONE)


def handle_protocol_request(store, request, *, context: dict[str, Any] | None = None) -> TypedProtocolResult:
    return ProtocolEngine(store=store).handle(request, context=context)
