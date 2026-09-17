from __future__ import annotations

from .model import EvidencePredicate, OperationKind, OperationQuorum


def _predicate(
    operation: OperationKind,
    *,
    member_ids,
    threshold: int,
    task_id,
    attempt_id,
    tx_id,
    cfg_hash: str,
    part_hash: str,
    epoch: int | None = None,
    generation: int | None = None,
    operation_id: str | None = None,
    evidence_hash: str,
) -> EvidencePredicate:
    return EvidencePredicate(
        operation=operation,
        member_ids=tuple(member_ids),
        threshold=threshold,
        task_id=task_id,
        attempt_id=attempt_id,
        tx_id=tx_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        epoch=epoch,
        generation=generation,
        operation_id=operation_id,
        evidence_hash=evidence_hash,
    )


def operation_quorum(operation: OperationKind, available: int, required: int, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return OperationQuorum(
        operation=operation,
        required=required,
        available=available,
        satisfied=available >= required,
        predicate=predicate,
    )


def upload_completeness_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.UPLOAD_COMPLETENESS, available, required, predicate)


def input_readiness_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.INPUT_READINESS, available, required, predicate)


def reputation_lock_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.REPUTATION_LOCK, available, required, predicate)


def preprocessing_readiness_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.PREPROCESSING_READINESS, available, required, predicate)


def fixed_attempt_continuation_quorum(
    surviving_original_active_members: int,
    required_threshold: int,
    survivor_subset_backend: bool,
    compatible_state: bool = True,
    compatible_preprocessing: bool = True,
    *,
    predicate: EvidencePredicate | None = None,
) -> OperationQuorum:
    available = surviving_original_active_members if survivor_subset_backend and compatible_state and compatible_preprocessing else 0
    return operation_quorum(OperationKind.FIXED_ATTEMPT_CONTINUATION, available, required_threshold, predicate)


def output_prepare_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.OUTPUT_PREPARE, available, required, predicate)


def reputation_prepare_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.REPUTATION_PREPARE, available, required, predicate)


def joint_commit_authorization_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.JOINT_COMMIT_AUTHORIZATION, available, required, predicate)


def output_reconstruction_quorum(available: int, threshold: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.OUTPUT_RECONSTRUCTION, available, threshold, predicate)


def reputation_recovery_quorum(available: int, threshold: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.REPUTATION_RECOVERY, available, threshold, predicate)


def whole_task_restart_quorum(available: int, required: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.WHOLE_TASK_RESTART, available, required, predicate)


def upload_storage_quorum(report_storage_size: int, failure_budget: int, output_storage_size: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.UPLOAD_COMPLETENESS, report_storage_size, output_storage_size + failure_budget, predicate)


def output_storage_quorum(output_storage_size: int, threshold: int, failure_budget: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.OUTPUT_PREPARE, output_storage_size, threshold + failure_budget, predicate)


def reputation_prepare_quorum(available: int, required: int, failure_budget: int = 0, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.REPUTATION_PREPARE, available, required + failure_budget, predicate)


def reputation_storage_quorum(reputation_storage_size: int, threshold: int, failure_budget: int, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.REPUTATION_RECOVERY, reputation_storage_size, threshold + failure_budget, predicate)


def restart_admissibility_quorum(available: int, required: int, snapshot_current: bool, responsive_control_plane: bool, *, predicate: EvidencePredicate | None = None) -> OperationQuorum:
    return operation_quorum(OperationKind.WHOLE_TASK_RESTART, available if snapshot_current and responsive_control_plane else 0, required, predicate)


def committed_state_availability(reputation_storage_size: int, output_storage_size: int, reputation_losses: int, output_losses: int, threshold: int) -> bool:
    return reputation_storage_size - reputation_losses >= threshold and output_storage_size - output_losses >= threshold
