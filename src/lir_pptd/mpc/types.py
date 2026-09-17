from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class SimulatedBackendError(ValueError):
    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(message)

class SharePoint(M):
    server_id: str
    x_coordinate: int
    y_value: int
    field_prime: int
    threshold: int
    committee_size: int
    secret_id: str
    backend_profile_hash: str
    share_epoch: int = 0

class ShareBundle(M):
    shares: tuple[SharePoint, ...]
    secret_id: str
    field_prime: int
    threshold: int
    committee_size: int
    backend_profile_hash: str
    share_epoch: int = 0

class ReconstructionResult(M):
    status: Literal["reconstructed", "rejected"]
    value: int | None = None
    reason_code: str | None = None

class BackendOperationResult(M):
    status: Literal["completed", "rejected"]
    output: ShareBundle | None = None
    reason_code: str | None = None

class PreparedOutput(M):
    task_id: str
    output_secret_id: str
    q_out_recomputed: bool
    privacy_class: Literal["private"] = "private"

class PreparedReputation(M):
    worker_id: str
    previous_epoch: int
    next_epoch: int
    update_count: int
    privacy_class: Literal["private"] = "private"

class TranscriptEntry(M):
    operation_id: str
    operation_name: str
    backend_kind: Literal["simulated_shamir"]
    task_id: str = "phase4"
    iteration: int | None = None
    worker_id: str | None = None
    coordinate: int | None = None
    input_secret_ids: tuple[str, ...] = ()
    output_secret_ids: tuple[str, ...] = ()
    threshold: int
    committee_size: int
    share_epoch: int
    backend_profile_hash: str
    fixed_profile_hash: str
    simulated_internal_opening: bool = False
    status: Literal["completed", "rejected"]
    reason_code: str | None = None
    privacy_class: Literal["public", "private"]

class TranscriptBundle(M):
    entries: tuple[TranscriptEntry, ...]

class SecureTaskResult(M):
    status: Literal["completed"] = "completed"
    task_id: str
    task_kind: str
    K: int
    iterations_executed: int
    q_out_recomputed: bool
    reputation_updates_per_participant: int
    nonparticipant_carry_forward: bool
    V_sec: int = 0
    l2_result: dict[str, Any] = Field(default_factory=dict)
    l3_result: dict[str, Any]
    conformance_fields: int = 0
    operation_trace: tuple[TranscriptEntry, ...]
    independent_l3_execution: Literal[True] = True
    l2_full_runner_used_for_l3: Literal[False] = False
    backend_kind: Literal["simulated_shamir"] = "simulated_shamir"
    production_mpc_used: Literal[False] = False
    real_data_used: Literal[False] = False
    distributed_e8b_used: Literal[False] = False
    cryptographic_security_claim: Literal[False] = False
    synthetic_only: Literal[True] = True

class SecureConformanceResult(M):
    task_id: str
    status: Literal["passed", "failed"]
    compared_fields: int
    mismatch_count: int
    V_sec: int
