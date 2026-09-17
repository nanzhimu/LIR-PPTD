from __future__ import annotations

import math

from lir_pptd.experiments.phase_r1.attacks import (
    malicious_count_half_up,
    select_malicious_workers,
)

DEFAULT_NAMESPACE = "phase-r1-malicious-subset-v1"


class FreshSubsetError(RuntimeError):
    """Raised when a fully unseen malicious-worker subset cannot be frozen."""


def fresh_subsets(
    worker_ids,
    *,
    dataset_id: str,
    rho: str,
    requested: int,
    excluded: set[tuple[str, ...]],
    seed_start: int,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Deterministically select unseen malicious-worker subsets.

    The returned subsets are disjoint from ``excluded`` within one dataset/rho
    cell.  If fewer than ``requested`` unseen unique subsets exist, all
    remaining unseen subsets are selected.  This is used by the fresh Target-B
    confirmation freeze and is intentionally result-blind.
    """

    workers = tuple(sorted(set(map(str, worker_ids))))
    k = malicious_count_half_up(len(workers), rho)
    possible = math.comb(len(workers), k)
    remaining = possible - len(excluded)
    target = min(requested, remaining)
    if target <= 0:
        raise FreshSubsetError(f"NO_UNSEEN_SUBSET:{dataset_id}:{rho}")

    chosen = []
    seen: set[tuple[str, ...]] = set()
    seed = seed_start
    while len(chosen) < target and seed < seed_start + 1_000_000:
        subset = select_malicious_workers(
            workers,
            dataset_id=dataset_id,
            rho=rho,
            seed=seed,
            namespace=namespace,
        )
        if subset not in excluded and subset not in seen:
            seen.add(subset)
            chosen.append({"seed": seed, "workers": list(subset)})
        seed += 1

    if len(chosen) != target:
        raise FreshSubsetError(
            f"UNABLE_TO_FILL:{dataset_id}:{rho}:{len(chosen)}/{target}"
        )
    return target, possible, remaining, chosen


# Internal compatibility name used by the freeze script/tests.
_fresh_subsets = fresh_subsets
