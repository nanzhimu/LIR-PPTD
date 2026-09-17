from __future__ import annotations

from typing import Sequence

from ..hashing import semantic_hash
from .model import (
    AbortAuth,
    CommitCertificate,
    DecisionRecord,
    ExecutionCheckpoint,
    IntegrityFailure,
    LockCertificate,
    ParticipantCertificate,
    PreparedRecord,
    PreprocessingBundle,
    PreprocessingManifest,
    PreprocessingRecord,
    ProtocolResponse,
    ProtocolState,
    TransactionRequest,
    TypedProtocolResult,
)


def canonical_part_hash(certificate: ParticipantCertificate) -> str:
    return certificate.canonical_part_hash()


def canonical_request_hash(request: TransactionRequest) -> str:
    return request.request_hash()


def build_participant_certificate(
    *,
    task_id,
    cfg_hash: str,
    participant_ids: Sequence,
    participant_count: int,
    dimension: int,
    common_report_storage_set: Sequence,
    selected_upload_ids: Sequence,
    reputation_epoch_snapshot: Sequence[int],
    schema_domain_version: str,
    requester_attestation: str,
) -> ParticipantCertificate:
    return ParticipantCertificate(
        task_id=task_id,
        cfg_hash=cfg_hash,
        participant_ids=tuple(participant_ids),
        participant_count=participant_count,
        dimension=dimension,
        common_report_storage_set=tuple(common_report_storage_set),
        selected_upload_ids=tuple(selected_upload_ids),
        reputation_epoch_snapshot=tuple(reputation_epoch_snapshot),
        schema_domain_version=schema_domain_version,
        requester_attestation=requester_attestation,
    )


def build_transaction_request(
    *,
    task_id,
    cfg_hash: str,
    part_hash: str,
    attempt_id,
    tx_id,
    requester_id,
) -> TransactionRequest:
    return TransactionRequest(
        task_id=task_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        attempt_id=attempt_id,
        tx_id=tx_id,
        requester_id=requester_id,
    )


def build_lock_certificate(
    *,
    task_id,
    cfg_hash: str,
    part_hash: str,
    attempt_id,
    tx_id,
    requester_id,
    epoch_snapshot: Sequence[int],
    locked_committee: Sequence = (),
    reason_code: str | None = None,
) -> LockCertificate:
    return LockCertificate(
        task_id=task_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        attempt_id=attempt_id,
        tx_id=tx_id,
        requester_id=requester_id,
        epoch_snapshot=tuple(epoch_snapshot),
        locked_committee=tuple(locked_committee),
        reason_code=reason_code,
    )


def build_preprocessing_record(
    *,
    record_id: str,
    op_id: str,
    operation_kind,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    generation: int,
    committee_hash: str,
    manifest_hash: str,
    bundle_hash: str,
) -> PreprocessingRecord:
    return PreprocessingRecord(
        record_id=record_id,
        op_id=op_id,
        operation_kind=operation_kind,
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        generation=generation,
        committee_hash=committee_hash,
        manifest_hash=manifest_hash,
        bundle_hash=bundle_hash,
    )


def build_preprocessing_manifest(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    generation: int,
    committee_hash: str,
    records: Sequence[PreprocessingRecord],
) -> PreprocessingManifest:
    return PreprocessingManifest(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        generation=generation,
        committee_hash=committee_hash,
        records=tuple(records),
    )


def build_preprocessing_bundle(*, bundle_id: str, manifest: PreprocessingManifest, lifecycle=None) -> PreprocessingBundle:
    return PreprocessingBundle(
        bundle_id=bundle_id,
        manifest_hash=manifest.manifest_hash(),
        task_id=manifest.task_id,
        tx_id=manifest.tx_id,
        attempt_id=manifest.attempt_id,
        cfg_hash=manifest.cfg_hash,
        part_hash=manifest.part_hash,
        generation=manifest.generation,
        committee_hash=manifest.committee_hash,
        lifecycle=lifecycle if lifecycle is not None else (manifest.records[0].lifecycle if manifest.records else None),
    )


def build_preprocessing_artifacts(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    generation: int,
    committee_hash: str,
    record_specs: Sequence[tuple[str, str, object]],
    bundle_id: str,
):
    records = tuple(
        build_preprocessing_record(
            record_id=record_id,
            op_id=op_id,
            operation_kind=operation_kind,
            task_id=task_id,
            tx_id=tx_id,
            attempt_id=attempt_id,
            cfg_hash=cfg_hash,
            part_hash=part_hash,
            generation=generation,
            committee_hash=committee_hash,
            manifest_hash="pending",
            bundle_hash="pending",
        )
        for record_id, op_id, operation_kind in record_specs
    )
    manifest = build_preprocessing_manifest(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        generation=generation,
        committee_hash=committee_hash,
        records=records,
    )
    bundle = build_preprocessing_bundle(bundle_id=bundle_id, manifest=manifest)
    manifest_hash = manifest.manifest_hash()
    bundle_hash = bundle.bundle_hash()
    records = tuple(record.model_copy(update={"manifest_hash": manifest_hash, "bundle_hash": bundle_hash}) for record in records)
    manifest = manifest.model_copy(update={"records": records})
    bundle = build_preprocessing_bundle(bundle_id=bundle_id, manifest=manifest)
    bundle_hash = bundle.bundle_hash()
    records = tuple(record.model_copy(update={"bundle_hash": bundle_hash}) for record in records)
    manifest = manifest.model_copy(update={"records": records})
    bundle = build_preprocessing_bundle(bundle_id=bundle_id, manifest=manifest)
    return type("PreprocessingArtifacts", (), {"manifest": manifest, "bundle": bundle, "records": records})()


def build_commit_certificate(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    epoch_snapshot: Sequence[int],
    output_committee: Sequence,
    reputation_committee: Sequence,
    output_prepare_hash: str,
    reputation_prepare_hash: str,
    output_manifest_hash: str,
    reputation_manifest_hash: str,
    requester_id,
) -> CommitCertificate:
    return CommitCertificate(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        epoch_snapshot=tuple(epoch_snapshot),
        output_committee=tuple(output_committee),
        reputation_committee=tuple(reputation_committee),
        output_prepare_hash=output_prepare_hash,
        reputation_prepare_hash=reputation_prepare_hash,
        output_manifest_hash=output_manifest_hash,
        reputation_manifest_hash=reputation_manifest_hash,
        requester_id=requester_id,
    )


def build_abort_auth(
    *,
    task_id,
    tx_id,
    attempt_id,
    requester_id,
    cfg_hash: str,
    part_hash: str,
    prepared_hash: str,
    reason_code: str,
) -> AbortAuth:
    return AbortAuth(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        requester_id=requester_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        prepared_hash=prepared_hash,
        reason_code=reason_code,
    )


def build_execution_checkpoint(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    generation: int,
    lock_certificate_hash: str,
    preprocessing_manifest_hash: str,
    preprocessing_bundle_hash: str,
) -> ExecutionCheckpoint:
    return ExecutionCheckpoint(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        generation=generation,
        lock_certificate_hash=lock_certificate_hash,
        preprocessing_manifest_hash=preprocessing_manifest_hash,
        preprocessing_bundle_hash=preprocessing_bundle_hash,
    )


def validate_participant_certificate(certificate: ParticipantCertificate, expected_part_hash: str) -> bool:
    return canonical_part_hash(certificate) == expected_part_hash


def validate_lock_certificate(certificate: LockCertificate, request: TransactionRequest, expected_epoch_snapshot: Sequence[int]) -> bool:
    return (
        certificate.task_id == request.task_id
        and certificate.cfg_hash == request.cfg_hash
        and certificate.part_hash == request.part_hash
        and certificate.attempt_id == request.attempt_id
        and certificate.tx_id == request.tx_id
        and certificate.requester_id == request.requester_id
        and tuple(certificate.epoch_snapshot) == tuple(expected_epoch_snapshot)
        and certificate.lock_hash() == semantic_hash("protocol-v2-lock-certificate", certificate.model_dump(mode="python"))
    )


def validate_preprocessing_manifest(manifest: PreprocessingManifest) -> bool:
    if len({record.record_id for record in manifest.records}) != len(manifest.records):
        return False
    if len({record.op_id for record in manifest.records}) != len(manifest.records):
        return False
    return all(
        record.task_id == manifest.task_id
        and record.tx_id == manifest.tx_id
        and record.attempt_id == manifest.attempt_id
        and record.cfg_hash == manifest.cfg_hash
        and record.part_hash == manifest.part_hash
        and record.generation == manifest.generation
        and record.committee_hash == manifest.committee_hash
        for record in manifest.records
    )


def validate_preprocessing_bundle(bundle: PreprocessingBundle, manifest: PreprocessingManifest) -> bool:
    return (
        bundle.manifest_hash == manifest.manifest_hash()
        and bundle.task_id == manifest.task_id
        and bundle.tx_id == manifest.tx_id
        and bundle.attempt_id == manifest.attempt_id
        and bundle.cfg_hash == manifest.cfg_hash
        and bundle.part_hash == manifest.part_hash
        and bundle.generation == manifest.generation
        and bundle.committee_hash == manifest.committee_hash
    )


def validate_commit_certificate(certificate: CommitCertificate, prepared: PreparedRecord) -> bool:
    return (
        certificate.task_id == prepared.task_id
        and certificate.tx_id == prepared.tx_id
        and certificate.attempt_id == prepared.attempt_id
        and certificate.cfg_hash == prepared.cfg_hash
        and certificate.part_hash == prepared.part_hash
        and certificate.epoch_snapshot == prepared.epoch_snapshot
        and certificate.output_prepare_hash == prepared.output_prepare_hash
        and certificate.reputation_prepare_hash == prepared.reputation_prepare_hash
        and certificate.output_manifest_hash == prepared.output_manifest_hash
        and certificate.reputation_manifest_hash == prepared.reputation_manifest_hash
    )


def validate_abort_auth(auth: AbortAuth, prepared: PreparedRecord, reason_code: str) -> bool:
    return (
        auth.task_id == prepared.task_id
        and auth.tx_id == prepared.tx_id
        and auth.attempt_id == prepared.attempt_id
        and auth.cfg_hash == prepared.cfg_hash
        and auth.part_hash == prepared.part_hash
        and auth.prepared_hash == prepared.prepared_hash()
        and auth.reason_code == reason_code
    )


def validate_prepared_record(record: PreparedRecord) -> bool:
    return record.state == ProtocolState.PREPARED


def validate_execution_checkpoint(record: ExecutionCheckpoint) -> bool:
    return record.checkpoint_stage == "ExecutionCheckpoint"


def validate_prepared_bundle(output_prepare, reputation_prepare, prepared: PreparedRecord) -> bool:
    return (
        output_prepare.tx_id == reputation_prepare.tx_id == prepared.tx_id
        and output_prepare.task_id == reputation_prepare.task_id == prepared.task_id
        and output_prepare.part_hash == reputation_prepare.part_hash == prepared.part_hash
        and output_prepare.attempt_id == reputation_prepare.attempt_id == prepared.attempt_id
        and output_prepare.manifest_hash == prepared.output_manifest_hash
        and reputation_prepare.manifest_hash == prepared.reputation_manifest_hash
    )


def ensure_protocol_response(value: str | ProtocolResponse) -> ProtocolResponse:
    return value if isinstance(value, ProtocolResponse) else ProtocolResponse(value)


def failure_result(reason_code: str, message: str, tx_id=None) -> TypedProtocolResult:
    return TypedProtocolResult(
        response=ProtocolResponse.INTEGRITY_FAILURE,
        integrity_failure=IntegrityFailure(reason_code=reason_code, message=message, tx_id=tx_id),
    )
