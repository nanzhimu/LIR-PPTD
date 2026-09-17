from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..canonical import CanonicalRational
from ..identifiers import TaskID, WorkerID


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnosticStatus(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNRESOLVED = "unresolved"


class StabilityStatus(StrEnum):
    STABLE = "stable"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class ExactTaskFailure(FrozenModel):
    success: Literal[False] = False
    reason_code: str
    message: str
    reputation_updated: Literal[False] = False
    raw_success_written: Literal[False] = False


class ExactDiagnosticResult(FrozenModel):
    diagnostic: str
    status: DiagnosticStatus
    passed: bool
    note: str


class ExactIterationTrace(FrozenModel):
    iteration: int
    truth: tuple[str, ...]
    distance_by_worker: dict[WorkerID, str]
    evidence_by_worker: dict[WorkerID, str]
    influence_by_worker: dict[WorkerID, str]
    denominator: str
    objective: str | None = None


class ExactWorkerEvidence(FrozenModel):
    worker_id: WorkerID
    distance_out: str
    evidence_out: str


class ExactReputationTransition(FrozenModel):
    worker_id: WorkerID
    previous: str
    direct: str
    nonnegative: str
    next: str
    update_count: int
    previous_epoch: int
    next_epoch: int


class NonparticipantCarryForward(FrozenModel):
    worker_id: WorkerID
    reputation: str
    epoch: int


class PrecisionRunSummary(FrozenModel):
    dps: int
    final_output: tuple[str, ...]
    q_out: dict[WorkerID, str]
    next_reputation: dict[WorkerID, str]


class PrecisionDoublingResult(FrozenModel):
    initial_dps: int
    final_dps: int
    precision_doubling_steps: int
    epsilon_hp: CanonicalRational
    stability_status: StabilityStatus
    runs: tuple[PrecisionRunSummary, ...]


class ExactTaskResult(FrozenModel):
    success: Literal[True] = True
    schema_version: str = "1.0"
    backend_kind: Literal["exact_mpmath"] = "exact_mpmath"
    result_scope: Literal["algorithmic_conformance"] = "algorithmic_conformance"
    task_id: TaskID
    task_kind: Literal["numerical", "categorical"]
    K: int
    initial_reputation: dict[WorkerID, str]
    participant_ids: tuple[WorkerID, ...]
    nonparticipant_ids: tuple[WorkerID, ...]
    truth_trajectory: tuple[tuple[str, ...], ...]
    iterations: tuple[ExactIterationTrace, ...]
    final_output: tuple[str, ...]
    released_class_index: int | None = None
    output_evidence: tuple[ExactWorkerEvidence, ...]
    reputation_transitions: tuple[ExactReputationTransition, ...]
    next_reputation: dict[WorkerID, str]
    nonparticipant_carry_forward: tuple[NonparticipantCarryForward, ...]
    precision: PrecisionDoublingResult | None = None
    diagnostics: tuple[ExactDiagnosticResult, ...]
    formula_references: tuple[str, ...]
    source_reference: str
    production_mpc_used: Literal[False] = False
    real_data_used: Literal[False] = False
    distributed_e8b_used: Literal[False] = False
