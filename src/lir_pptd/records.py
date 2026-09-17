from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .identifiers import ArtifactHash, BackendProfileHash, CodeHash, ConfigHash, DataHash, RunID


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactClassification(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class ScopeStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED_NOT_ACTIVE = "blocked_not_active"


class Phase0Decision(StrEnum):
    APPROVED_WITH_DOWNSTREAM_BLOCKERS = "approved_with_downstream_blockers"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class FailureRecord(FrozenRecord):
    reason_code: str
    message: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RunRecord(FrozenRecord):
    run_id: RunID
    created_at_utc: datetime = Field(default_factory=utc_now)
    config_hash: ConfigHash
    code_hash: CodeHash
    data_hash: DataHash
    backend_profile_hash: BackendProfileHash
    coordinator_duration_seconds: float | None = Field(default=None, ge=0)
    status: Literal["validated", "failed", "not_run"]
    failure: FailureRecord | None = None
    original_literals: dict[str, str] = Field(default_factory=dict)

    @field_validator("created_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("timestamp must be timezone-aware UTC")
        return value


class ArtifactEntry(FrozenRecord):
    path: str
    sha256: ArtifactHash
    classification: ArtifactClassification

    def assert_publication_allowed(self, destination: str = "public_artifacts") -> None:
        if destination == "public_artifacts" and self.classification is not ArtifactClassification.PUBLIC:
            raise ValueError(f"{self.classification.value} artifact cannot enter public_artifacts")


class ArtifactManifest(FrozenRecord):
    schema_version: str = "1.0"
    run_id: str
    artifacts: list[ArtifactEntry]


class ValidityRecord(FrozenRecord):
    run_id: str
    valid: bool
    reason: str
    superseding_commit: str | None = None
    approver: str
    timestamp_utc: datetime = Field(default_factory=utc_now)


class BlockerRecord(FrozenRecord):
    blocker_id: str
    scope: str
    status: ScopeStatus
    reason: str
    activation_condition: str


class ApprovalRecord(FrozenRecord):
    phase: int
    author_name: str
    decision: Phase0Decision
    note: str
    timestamp: datetime
    allowed_decisions: list[Phase0Decision] = Field(default_factory=list)
    template_only: bool = False
    cursor_must_not_approve: bool = True
