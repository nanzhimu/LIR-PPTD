from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .binding import (
    build_abort_auth,
    build_commit_certificate,
    validate_abort_auth,
    validate_commit_certificate,
)
from .model import (
    AbortAuth,
    CommitCertificate,
    DecisionRecord,
    PreparedRecord,
    ProtocolResponse,
    ProtocolState,
    TransactionRequest,
)


@dataclass(frozen=True)
class DecisionOutcome:
    decision: ProtocolResponse
    record: DecisionRecord


def build_output_prepare(
    *,
    task_id,
    tx_id,
    part_hash: str,
    attempt_id,
    record_id: str,
    manifest_hash: str,
    committee_hash: str,
    storage_set: Sequence,
    q_out_recomputed: bool = True,
) -> object:
    from .model import OutputPrepareRecord

    return OutputPrepareRecord(
        tx_id=tx_id,
        task_id=task_id,
        part_hash=part_hash,
        attempt_id=attempt_id,
        record_id=record_id,
        manifest_hash=manifest_hash,
        committee_hash=committee_hash,
        storage_set=tuple(storage_set),
        q_out_recomputed=q_out_recomputed,
    )


def build_reputation_prepare(
    *,
    task_id,
    tx_id,
    part_hash: str,
    attempt_id,
    record_id: str,
    manifest_hash: str,
    committee_hash: str,
    storage_set: Sequence,
    update_count: int = 1,
) -> object:
    from .model import ReputationPrepareRecord

    return ReputationPrepareRecord(
        tx_id=tx_id,
        task_id=task_id,
        part_hash=part_hash,
        attempt_id=attempt_id,
        record_id=record_id,
        manifest_hash=manifest_hash,
        committee_hash=committee_hash,
        storage_set=tuple(storage_set),
        update_count=update_count,
    )


def build_prepared_record(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    lock_certificate_hash: str,
    epoch_snapshot: Sequence[int],
    output_prepare_hash: str,
    reputation_prepare_hash: str,
    output_manifest_hash: str,
    reputation_manifest_hash: str,
) -> PreparedRecord:
    return PreparedRecord(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        lock_certificate_hash=lock_certificate_hash,
        epoch_snapshot=tuple(epoch_snapshot),
        output_prepare_hash=output_prepare_hash,
        reputation_prepare_hash=reputation_prepare_hash,
        output_manifest_hash=output_manifest_hash,
        reputation_manifest_hash=reputation_manifest_hash,
    )


def build_abort_decision(
    *,
    request: TransactionRequest,
    prepared: PreparedRecord,
    reason_code: str,
    requester_id,
) -> tuple[AbortAuth, DecisionRecord]:
    auth = build_abort_auth(
        task_id=request.task_id,
        tx_id=request.tx_id,
        attempt_id=request.attempt_id,
        requester_id=requester_id,
        cfg_hash=request.cfg_hash,
        part_hash=request.part_hash,
        prepared_hash=prepared.prepared_hash(),
        reason_code=reason_code,
    )
    decision = DecisionRecord(
        task_id=request.task_id,
        tx_id=request.tx_id,
        attempt_id=request.attempt_id,
        cfg_hash=request.cfg_hash,
        part_hash=request.part_hash,
        prepared_hash=prepared.prepared_hash(),
        decision="Abort",
        authorization_hash=auth.auth_hash(),
    )
    return auth, decision


def build_commit_decision(
    *,
    request: TransactionRequest,
    prepared: PreparedRecord,
    certificate: CommitCertificate,
) -> DecisionRecord:
    return DecisionRecord(
        task_id=request.task_id,
        tx_id=request.tx_id,
        attempt_id=request.attempt_id,
        cfg_hash=request.cfg_hash,
        part_hash=request.part_hash,
        prepared_hash=prepared.prepared_hash(),
        decision="Commit",
        authorization_hash=certificate.certificate_hash(),
    )


def validate_prepared_bundle(output_prepare, reputation_prepare, prepared: PreparedRecord) -> bool:
    return (
        output_prepare.tx_id == reputation_prepare.tx_id == prepared.tx_id
        and output_prepare.task_id == reputation_prepare.task_id == prepared.task_id
        and output_prepare.part_hash == reputation_prepare.part_hash == prepared.part_hash
        and output_prepare.attempt_id == reputation_prepare.attempt_id == prepared.attempt_id
        and output_prepare.manifest_hash == prepared.output_manifest_hash
        and reputation_prepare.manifest_hash == prepared.reputation_manifest_hash
    )


def record_decision(
    *,
    request: TransactionRequest,
    prepared: PreparedRecord,
    decision: ProtocolResponse,
    authorization: AbortAuth | CommitCertificate,
) -> DecisionRecord:
    if decision == ProtocolResponse.COMMITTED:
        if not isinstance(authorization, CommitCertificate) or not validate_commit_certificate(authorization, prepared):
            raise ValueError("invalid commit authorization")
        return build_commit_decision(request=request, prepared=prepared, certificate=authorization)
    if decision == ProtocolResponse.ABORTED:
        if not isinstance(authorization, AbortAuth) or not validate_abort_auth(authorization, prepared, authorization.reason_code):
            raise ValueError("invalid abort authorization")
        return DecisionRecord(
            task_id=request.task_id,
            tx_id=request.tx_id,
            attempt_id=request.attempt_id,
            cfg_hash=request.cfg_hash,
            part_hash=request.part_hash,
            prepared_hash=prepared.prepared_hash(),
            decision="Abort",
            authorization_hash=authorization.auth_hash(),
        )
    raise ValueError("unsupported decision")


class RecordDecision:
    def __init__(self, store) -> None:
        self.store = store

    def execute(
        self,
        *,
        request: TransactionRequest,
        prepared: PreparedRecord,
        decision: ProtocolResponse,
        authorization: AbortAuth | CommitCertificate,
    ) -> ProtocolResponse:
        record = record_decision(request=request, prepared=prepared, decision=decision, authorization=authorization)
        return self.store.record_decision(request.tx_id, record.decision_hash(), ProtocolResponse.COMMITTED if decision == ProtocolResponse.COMMITTED else ProtocolResponse.ABORTED)
