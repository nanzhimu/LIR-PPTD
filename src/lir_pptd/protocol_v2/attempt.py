from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .model import AttemptID, ProtocolRecord, TaskID, TransactionID


@dataclass(frozen=True)
class RestartPlan:
    task_id: TaskID
    attempt_id: AttemptID
    tx_id: TransactionID
    generation: int
    fresh_manifest_id: str
    fresh_bundle_id: str
    fresh_operation_ids: tuple[str, ...]
    fresh_record_ids: tuple[str, ...]


def build_fresh_restart(
    *,
    task_id: TaskID,
    attempt_id: AttemptID,
    tx_id: TransactionID,
    generation: int,
    manifest_id: str,
    bundle_id: str,
    operation_ids: Sequence[str],
    record_ids: Sequence[str],
) -> RestartPlan:
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    return RestartPlan(
        task_id=task_id,
        attempt_id=attempt_id,
        tx_id=tx_id,
        generation=generation,
        fresh_manifest_id=manifest_id,
        fresh_bundle_id=bundle_id,
        fresh_operation_ids=tuple(operation_ids),
        fresh_record_ids=tuple(record_ids),
    )


def restart_from_terminal(
    previous: ProtocolRecord,
    *,
    attempt_id: AttemptID,
    tx_id: TransactionID,
    generation: int,
    manifest_id: str,
    bundle_id: str,
    operation_ids: Sequence[str],
    record_ids: Sequence[str],
) -> RestartPlan:
    if previous.state not in {previous.state.ABORTED, previous.state.COMMITTED}:
        raise ValueError("restart requires a terminal record")
    return build_fresh_restart(
        task_id=previous.task_id,
        attempt_id=attempt_id,
        tx_id=tx_id,
        generation=generation,
        manifest_id=manifest_id,
        bundle_id=bundle_id,
        operation_ids=operation_ids,
        record_ids=record_ids,
    )
