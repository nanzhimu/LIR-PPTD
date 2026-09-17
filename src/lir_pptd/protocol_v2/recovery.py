from __future__ import annotations

from .binding import (
    build_abort_auth,
    validate_abort_auth,
    validate_commit_certificate,
    validate_prepared_record,
)
from .model import (
    DeliveryState,
    IntegrityFailure,
    PreparedRecord,
    ProtocolRecord,
    ProtocolResponse,
    ProtocolState,
    RecoveryOutcome,
    TypedProtocolResult,
)


def _failure(reason_code: str, message: str, tx_id=None) -> TypedProtocolResult:
    return TypedProtocolResult(
        response=ProtocolResponse.INTEGRITY_FAILURE,
        integrity_failure=IntegrityFailure(reason_code=reason_code, message=message, tx_id=tx_id),
    )


def recover_record(store, request, *, context: dict | None = None) -> TypedProtocolResult:
    existing = store.lookup_by_tx_id(request.tx_id)
    if existing is None:
        return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, message="unknown transaction")
    if existing.request != request:
        return TypedProtocolResult(response=ProtocolResponse.REJECTED_REPLAY, state=existing.state, message="conflicting request identity")
    if existing.integrity_failure is not None:
        return TypedProtocolResult(response=ProtocolResponse.INTEGRITY_FAILURE, integrity_failure=existing.integrity_failure)

    if existing.state == ProtocolState.COMMITTED:
        store.repair_committed_logs(request.tx_id)
        delivery = existing.delivery or DeliveryState.SECURE_OUTPUT
        return TypedProtocolResult(
            response=ProtocolResponse.COMMITTED,
            state=ProtocolState.COMMITTED,
            delivery=delivery,
            committed_epoch_vector=existing.committed_epoch_vector,
        )
    if existing.state == ProtocolState.ABORTED:
        return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=existing.delivery or DeliveryState.NONE)
    if existing.state == ProtocolState.PREPARED:
        prepared = existing.prepared
        if prepared is None or not validate_prepared_record(prepared):
            return _failure("INVALID_PREPARED", "durable prepared record could not be verified", request.tx_id)
        if context and context.get("decision") == "Commit":
            commit_certificate = context.get("commit_certificate")
            if commit_certificate is None or not validate_commit_certificate(commit_certificate, prepared):
                return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, state=ProtocolState.PREPARED, message="invalid commit certificate")
            store.record_decision(request.tx_id, commit_certificate.certificate_hash(), ProtocolResponse.COMMITTED)
            committed = existing.model_copy(update={"state": ProtocolState.COMMITTED, "response": ProtocolResponse.COMMITTED, "delivery": DeliveryState.SECURE_OUTPUT, "decision_winner": ProtocolResponse.COMMITTED, "commit_certificate_hash": commit_certificate.certificate_hash(), "committed_epoch_vector": commit_certificate.epoch_snapshot})
            store.commit_terminal_state(committed)
            store.repair_committed_logs(request.tx_id)
            return TypedProtocolResult(response=ProtocolResponse.COMMITTED, state=ProtocolState.COMMITTED, delivery=DeliveryState.SECURE_OUTPUT, committed_epoch_vector=commit_certificate.epoch_snapshot)
        abort_reason = "RecoveryNoDecision" if context is None else context.get("abort_reason", "RecoveryNoDecision")
        auth = build_abort_auth(
            task_id=request.task_id,
            tx_id=request.tx_id,
            attempt_id=request.attempt_id,
            requester_id=request.requester_id,
            cfg_hash=request.cfg_hash,
            part_hash=request.part_hash,
            prepared_hash=prepared.prepared_hash(),
            reason_code=abort_reason,
        )
        if not validate_abort_auth(auth, prepared, abort_reason):
            return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, state=ProtocolState.PREPARED, message="invalid abort authorization")
        store.record_decision(request.tx_id, auth.auth_hash(), ProtocolResponse.ABORTED)
        aborted = existing.model_copy(update={"state": ProtocolState.ABORTED, "response": ProtocolResponse.ABORTED, "delivery": DeliveryState.NONE, "decision_winner": ProtocolResponse.ABORTED, "abort_auth_hash": auth.auth_hash()})
        store.commit_terminal_state(aborted)
        return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=DeliveryState.NONE)
    if existing.state == ProtocolState.EXECUTING:
        return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=DeliveryState.NONE, message="executing abort")
    if existing.state == ProtocolState.LOCKED:
        return TypedProtocolResult(response=ProtocolResponse.ABORTED, state=ProtocolState.ABORTED, delivery=DeliveryState.NONE, message="locked abort")
    return TypedProtocolResult(response=ProtocolResponse.REJECTED_REQUEST, state=existing.state, message="unsupported state")
