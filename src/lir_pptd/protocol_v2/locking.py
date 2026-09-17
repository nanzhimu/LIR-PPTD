from __future__ import annotations

from .binding import build_lock_certificate, validate_lock_certificate
from .model import ProtocolRecord, ProtocolResponse, TransactionRequest


def authenticate_request_before_lookup(request: TransactionRequest, store) -> ProtocolRecord | None:
    lookup = store.lookup_by_request(request)
    if lookup.status == "conflict":
        raise ValueError("RejectedReplay")
    if lookup.status == "exact":
        return lookup.record
    existing_by_tx = store.lookup_by_tx_id(request.tx_id)
    if existing_by_tx is not None and existing_by_tx.request != request:
        raise ValueError("RejectedReplay")
    return existing_by_tx


def acquire_lock(
    *,
    store,
    request: TransactionRequest,
    epoch_snapshot: tuple[int, ...],
    locked_committee: tuple = (),
    reason_code: str | None = None,
) -> ProtocolRecord:
    certificate = build_lock_certificate(
        task_id=request.task_id,
        cfg_hash=request.cfg_hash,
        part_hash=request.part_hash,
        attempt_id=request.attempt_id,
        tx_id=request.tx_id,
        requester_id=request.requester_id,
        epoch_snapshot=epoch_snapshot,
        locked_committee=locked_committee,
        reason_code=reason_code,
    )
    if not validate_lock_certificate(certificate, request, epoch_snapshot):
        raise ValueError("invalid lock certificate")
    return certificate
