from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..phase6_exact_adapter import LongitudinalState
from ...core.types import ExactTaskResult


class LongitudinalStateError(ValueError):
    pass


def initial_global_state(worker_ids: tuple[str, ...], c0: object) -> LongitudinalState:
    if not worker_ids:
        raise LongitudinalStateError("empty worker universe")
    return LongitudinalState(
        reputations={worker: c0 for worker in worker_ids},
        epochs={worker: 0 for worker in worker_ids},
    )


def merge_participant_result(
    previous: LongitudinalState,
    result: ExactTaskResult,
) -> LongitudinalState:
    prev_rep = dict(previous.reputations or {})
    prev_epochs = {k: int(v) for k, v in dict(previous.epochs or {}).items()}
    if not prev_rep:
        raise LongitudinalStateError("previous global reputation state is empty")
    if set(prev_rep) != set(prev_epochs):
        raise LongitudinalStateError("reputation and epoch worker universes differ")

    participants = set(str(w) for w in result.participant_ids)
    unknown = participants - set(prev_rep)
    if unknown:
        raise LongitudinalStateError(f"result contains worker outside global universe: {sorted(unknown)[0]}")

    next_rep = dict(prev_rep)
    next_epochs = dict(prev_epochs)
    transition_workers: set[str] = set()
    for transition in result.reputation_transitions:
        worker = str(transition.worker_id)
        if worker in transition_workers:
            raise LongitudinalStateError(f"duplicate transition for {worker}")
        transition_workers.add(worker)
        if worker not in participants:
            raise LongitudinalStateError(f"transition for nonparticipant {worker}")
        if int(transition.previous_epoch) != prev_epochs[worker]:
            raise LongitudinalStateError(f"previous epoch mismatch for {worker}")
        if int(transition.next_epoch) != prev_epochs[worker] + 1:
            raise LongitudinalStateError(f"next epoch mismatch for {worker}")
        next_rep[worker] = transition.next
        next_epochs[worker] = int(transition.next_epoch)

    if transition_workers != participants:
        missing = sorted(participants - transition_workers)
        raise LongitudinalStateError(f"missing participant transition: {missing[0] if missing else 'unknown'}")

    # Nonparticipants are copied from previous maps unchanged by construction.
    return LongitudinalState(reputations=next_rep, epochs=next_epochs)
