from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..hashing import semantic_hash
from ..identifiers import AttemptID, TaskID, TransactionID, UploadID, WorkerID


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProtocolState(StrEnum):
    LOCKED = "Locked"
    EXECUTING = "Executing"
    PREPARED = "Prepared"
    ABORTED = "Aborted"
    COMMITTED = "Committed"


class ProtocolResponse(StrEnum):
    ABORTED = "Aborted"
    COMMITTED = "Committed"
    REJECTED_REQUEST = "RejectedRequest"
    REJECTED_REPLAY = "RejectedReplay"
    INTEGRITY_FAILURE = "IntegrityFailure"


class DeliveryState(StrEnum):
    SECURE_OUTPUT = "secure_output"
    OUTPUT_UNAVAILABLE = "OutputUnavailable"
    NONE = "None"


class OperationKind(StrEnum):
    UPLOAD_COMPLETENESS = "UploadCompleteness"
    INPUT_READINESS = "InputReadiness"
    REPUTATION_LOCK = "ReputationLock"
    PREPROCESSING_READINESS = "PreprocessingReadiness"
    FIXED_ATTEMPT_CONTINUATION = "FixedAttemptContinuation"
    OUTPUT_PREPARE = "OutputPrepare"
    REPUTATION_PREPARE = "ReputationPrepare"
    JOINT_COMMIT_AUTHORIZATION = "JointCommitAuthorization"
    OUTPUT_RECONSTRUCTION = "OutputReconstruction"
    REPUTATION_RECOVERY = "ReputationRecovery"
    WHOLE_TASK_RESTART = "WholeTaskRestart"


class PreprocessingLifecycle(StrEnum):
    AVAILABLE = "AVAILABLE"
    ACTIVATED = "ACTIVATED"
    CONSUMED = "CONSUMED"


class FaultHook(StrEnum):
    BEFORE_LOCK_PERSISTENCE = "before_lock_persistence"
    AFTER_LOCK_BEFORE_ACTIVATION = "after_lock_before_activation"
    AFTER_ACTIVATION_BEFORE_ARITHMETIC = "after_activation_before_arithmetic"
    BETWEEN_OUTPUT_AND_REPUTATION_PREPARE = "between_output_and_reputation_prepare"
    AFTER_PREPARED_BEFORE_DECISION = "after_prepared_before_decision"
    AFTER_DECISION_BEFORE_MATERIALIZATION = "after_decision_before_materialization"
    AT_COMMIT_LINEARIZATION = "at_commit_linearization"
    AFTER_COMMIT_BEFORE_DELIVERY = "after_commit_before_delivery"
    PREPARED_RECORD_CORRUPTION = "prepared_record_corruption"
    INVALID_COMMIT_CERTIFICATE = "invalid_commit_certificate"
    BARE_ABORT_BYPASS = "bare_abort_bypass"
    MISMATCHED_ABORT_AUTH = "mismatched_abort_auth"
    CONFLICTING_REPLAY = "conflicting_replay"
    INVALID_REQUEST_PARSING = "invalid_request_parsing"


class IntegrityFailure(FrozenModel):
    reason_code: str
    message: str
    tx_id: TransactionID | None = None


class OutputUnavailable(FrozenModel):
    tx_id: TransactionID
    message: str


class ExactlyOnceTransition(FrozenModel):
    worker_id: WorkerID
    previous_epoch: int
    next_epoch: int
    update_count: int = Field(default=1, ge=1)
    applied: Literal[True] = True


class CommittedLogEntry(FrozenModel):
    worker_id: WorkerID
    epoch: int
    refresh_id: str
    tx_id: TransactionID
    commit_certificate_hash: str
    status: ProtocolState = ProtocolState.COMMITTED


class FaultInjectionPlan(FrozenModel):
    hooks: tuple[FaultHook, ...] = ()
    notes: str | None = None


class WorkerBinding(FrozenModel):
    worker_id: WorkerID
    epoch: int = Field(ge=0)
    share_count: int = Field(ge=0)


class TaskBinding(FrozenModel):
    task_id: TaskID
    cfg_hash: str
    part_hash: str
    attempt_id: AttemptID
    tx_id: TransactionID


class RequestAuthenticationEvidence(FrozenModel):
    request_hash: str
    canonical_parse_hash: str
    attestation_hash: str
    verifier_name: str
    verified: bool = True


class CertificateAttestation(FrozenModel):
    issuer: str
    subject: str
    signature_hash: str
    verified: bool = True


class EvidencePredicate(FrozenModel):
    operation: OperationKind
    member_ids: tuple[WorkerID, ...]
    threshold: int = Field(ge=0)
    task_id: TaskID
    attempt_id: AttemptID
    tx_id: TransactionID
    cfg_hash: str
    part_hash: str
    epoch: int | None = None
    generation: int | None = None
    operation_id: str | None = None
    evidence_hash: str

    @field_validator("member_ids")
    @classmethod
    def unique_members(cls, value: tuple[WorkerID, ...]) -> tuple[WorkerID, ...]:
        if len(set(value)) != len(value):
            raise ValueError("member_ids must be unique")
        return value

    @field_validator("threshold")
    @classmethod
    def threshold_not_negative(cls, value: int, info: Any) -> int:
        member_ids = info.data.get("member_ids")
        if member_ids is not None and value > len(member_ids):
            raise ValueError("threshold cannot exceed member count")
        return value

    def satisfied(self) -> bool:
        return len(self.member_ids) >= self.threshold

    def predicate_hash(self) -> str:
        return semantic_hash("protocol-v2-evidence-predicate", self.model_dump(mode="python"))


class OperationQuorum(FrozenModel):
    operation: OperationKind
    required: int
    available: int
    satisfied: bool
    predicate: EvidencePredicate | None = None


class ParticipantCertificate(FrozenModel):
    task_id: TaskID
    cfg_hash: str
    participant_ids: tuple[WorkerID, ...]
    participant_count: int = Field(ge=1)
    dimension: int = Field(ge=1)
    common_report_storage_set: tuple[WorkerID, ...]
    selected_upload_ids: tuple[UploadID, ...]
    reputation_epoch_snapshot: tuple[int, ...]
    schema_domain_version: str
    requester_attestation: str

    @field_validator("participant_ids")
    @classmethod
    def unique_participants(cls, value: tuple[WorkerID, ...]) -> tuple[WorkerID, ...]:
        if len(set(value)) != len(value):
            raise ValueError("participant ids must be unique")
        return value

    @field_validator("common_report_storage_set")
    @classmethod
    def unique_storage(cls, value: tuple[WorkerID, ...]) -> tuple[WorkerID, ...]:
        if len(set(value)) != len(value):
            raise ValueError("common report storage set must be unique")
        return value

    @field_validator("selected_upload_ids")
    @classmethod
    def unique_uploads(cls, value: tuple[UploadID, ...]) -> tuple[UploadID, ...]:
        if len(set(value)) != len(value):
            raise ValueError("selected upload ids must be unique")
        return value

    @field_validator("participant_count")
    @classmethod
    def count_matches(cls, value: int, info: Any) -> int:
        participant_ids = info.data.get("participant_ids")
        if participant_ids is not None and value != len(participant_ids):
            raise ValueError("participant_count must match participant_ids length")
        return value

    @field_validator("reputation_epoch_snapshot")
    @classmethod
    def epoch_snapshot_matches(cls, value: tuple[int, ...], info: Any) -> tuple[int, ...]:
        participant_ids = info.data.get("participant_ids")
        if participant_ids is not None and len(value) != len(participant_ids):
            raise ValueError("reputation_epoch_snapshot must match participant_ids length")
        return value

    def canonical_part_hash(self) -> str:
        return semantic_hash("protocol-v2-participant-certificate", self.model_dump(mode="python"))


class TransactionRequest(FrozenModel):
    task_id: TaskID
    cfg_hash: str
    part_hash: str
    attempt_id: AttemptID
    tx_id: TransactionID
    requester_id: WorkerID

    def request_hash(self) -> str:
        return semantic_hash("protocol-v2-transaction-request", self.model_dump(mode="python"))


class LockCertificate(FrozenModel):
    task_id: TaskID
    cfg_hash: str
    part_hash: str
    attempt_id: AttemptID
    tx_id: TransactionID
    requester_id: WorkerID
    epoch_snapshot: tuple[int, ...]
    locked_committee: tuple[WorkerID, ...] = ()
    reason_code: str | None = None

    def lock_hash(self) -> str:
        return semantic_hash("protocol-v2-lock-certificate", self.model_dump(mode="python"))


class PreprocessingRecord(FrozenModel):
    record_id: str
    op_id: str
    operation_kind: OperationKind
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    generation: int = Field(ge=0)
    committee_hash: str
    manifest_hash: str
    bundle_hash: str
    lifecycle: PreprocessingLifecycle = PreprocessingLifecycle.AVAILABLE

    @field_validator("record_id", "op_id")
    @classmethod
    def nonempty_identifiers(cls, value: str) -> str:
        if not value:
            raise ValueError("identifier cannot be empty")
        return value

    def canonical_record_hash(self) -> str:
        return semantic_hash("protocol-v2-preprocessing-record", self.model_dump(mode="python"))

    @property
    def consumed(self) -> bool:
        return self.lifecycle == PreprocessingLifecycle.CONSUMED


class PreprocessingManifest(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    generation: int = Field(ge=0)
    committee_hash: str
    records: tuple[PreprocessingRecord, ...]

    @field_validator("records")
    @classmethod
    def unique_records(cls, value: tuple[PreprocessingRecord, ...]) -> tuple[PreprocessingRecord, ...]:
        record_ids = [record.record_id for record in value]
        op_ids = [record.op_id for record in value]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("preprocessing record ids must be unique")
        if len(set(op_ids)) != len(op_ids):
            raise ValueError("preprocessing operation ids must be unique")
        return value

    def manifest_hash(self) -> str:
        payload = self.model_dump(mode="python")
        payload["records"] = [
            {k: v for k, v in record.items() if k not in {"manifest_hash", "bundle_hash"}}
            for record in payload["records"]
        ]
        return semantic_hash("protocol-v2-preprocessing-manifest", payload)


class PreprocessingBundle(FrozenModel):
    bundle_id: str
    manifest_hash: str
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    generation: int = Field(ge=0)
    committee_hash: str
    lifecycle: PreprocessingLifecycle = PreprocessingLifecycle.AVAILABLE

    def bundle_hash(self) -> str:
        payload = self.model_dump(mode="python")
        return semantic_hash("protocol-v2-preprocessing-bundle", {k: v for k, v in payload.items() if k != "lifecycle"})


class OutputPrepareRecord(FrozenModel):
    tx_id: TransactionID
    task_id: TaskID
    part_hash: str
    attempt_id: AttemptID
    record_id: str
    manifest_hash: str
    committee_hash: str
    storage_set: tuple[WorkerID, ...]
    q_out_recomputed: bool = True
    activated: bool = False

    def record_hash(self) -> str:
        return semantic_hash("protocol-v2-output-prepare", self.model_dump(mode="python"))


class ReputationPrepareRecord(FrozenModel):
    tx_id: TransactionID
    task_id: TaskID
    part_hash: str
    attempt_id: AttemptID
    record_id: str
    manifest_hash: str
    committee_hash: str
    storage_set: tuple[WorkerID, ...]
    update_count: int = Field(default=1, ge=1)
    activated: bool = False

    def record_hash(self) -> str:
        return semantic_hash("protocol-v2-reputation-prepare", self.model_dump(mode="python"))


class PreparedRecord(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    lock_certificate_hash: str
    epoch_snapshot: tuple[int, ...]
    output_prepare_hash: str
    reputation_prepare_hash: str
    output_manifest_hash: str
    reputation_manifest_hash: str
    state: ProtocolState = ProtocolState.PREPARED

    def prepared_hash(self) -> str:
        return semantic_hash("protocol-v2-prepared-record", self.model_dump(mode="python"))


class ExecutionCheckpoint(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    generation: int = Field(ge=0)
    lock_certificate_hash: str
    preprocessing_manifest_hash: str
    preprocessing_bundle_hash: str
    checkpoint_stage: Literal["ExecutionCheckpoint"] = "ExecutionCheckpoint"

    def checkpoint_hash(self) -> str:
        return semantic_hash("protocol-v2-execution-checkpoint", self.model_dump(mode="python"))


class DecisionRecord(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    prepared_hash: str
    decision: Literal["Commit", "Abort"]
    authorization_hash: str

    def decision_hash(self) -> str:
        return semantic_hash("protocol-v2-decision-record", self.model_dump(mode="python"))


class CommitCertificate(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    cfg_hash: str
    part_hash: str
    epoch_snapshot: tuple[int, ...]
    output_committee: tuple[WorkerID, ...]
    reputation_committee: tuple[WorkerID, ...]
    output_prepare_hash: str
    reputation_prepare_hash: str
    output_manifest_hash: str
    reputation_manifest_hash: str
    requester_id: WorkerID

    def certificate_hash(self) -> str:
        return semantic_hash("protocol-v2-commit-certificate", self.model_dump(mode="python"))


class AbortAuth(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    attempt_id: AttemptID
    requester_id: WorkerID
    cfg_hash: str
    part_hash: str
    prepared_hash: str
    reason_code: str

    def auth_hash(self) -> str:
        return semantic_hash("protocol-v2-abort-auth", self.model_dump(mode="python"))


class RecoveryOutcome(FrozenModel):
    response: ProtocolResponse
    state: ProtocolState | None = None
    delivery: DeliveryState | None = None
    message: str | None = None
    committed_epoch_vector: tuple[int, ...] | None = None
    integrity_failure: IntegrityFailure | None = None


class TypedProtocolResult(FrozenModel):
    response: ProtocolResponse
    state: ProtocolState | None = None
    delivery: DeliveryState | None = None
    committed_epoch_vector: tuple[int, ...] | None = None
    message: str | None = None
    integrity_failure: IntegrityFailure | None = None


class ProtocolRecord(FrozenModel):
    task_id: TaskID
    tx_id: TransactionID
    request: TransactionRequest
    participant_certificate: ParticipantCertificate
    state: ProtocolState
    response: ProtocolResponse | None = None
    delivery: DeliveryState | None = None
    lock_certificate_hash: str | None = None
    preprocessing_manifest_hash: str | None = None
    preprocessing_bundle_hash: str | None = None
    preprocessing_activation_hash: str | None = None
    preprocessing_records: tuple[PreprocessingRecord, ...] = ()
    output_prepare: OutputPrepareRecord | None = None
    reputation_prepare: ReputationPrepareRecord | None = None
    prepared: PreparedRecord | None = None
    decision: DecisionRecord | None = None
    commit_certificate_hash: str | None = None
    abort_auth_hash: str | None = None
    decision_winner: ProtocolResponse | None = None
    committed_epoch_vector: tuple[int, ...] | None = None
    committed_output_share_count: int = 0
    committed_reputation_share_count: int = 0
    committed_logs: tuple[CommittedLogEntry, ...] = ()
    output_unavailable: OutputUnavailable | None = None
    integrity_failure: IntegrityFailure | None = None
    created_utc: str | None = None
    updated_utc: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def record_hash(self) -> str:
        return semantic_hash("protocol-v2-protocol-record", self.model_dump(mode="python"))


class LookupResult(FrozenModel):
    status: Literal["absent", "exact", "conflict"]
    record: ProtocolRecord | None = None
    conflicting_record: ProtocolRecord | None = None
    reason: str | None = None


class RestartRequest(FrozenModel):
    previous_tx_id: TransactionID
    task_id: TaskID
    new_attempt_id: AttemptID
    new_tx_id: TransactionID
    generation: int = Field(ge=0)
    fresh_manifest_id: str
    fresh_bundle_id: str
    fresh_operation_ids: tuple[str, ...]
    fresh_record_ids: tuple[str, ...]
    snapshot_current: bool = True
    responsive_control_plane: bool = True
