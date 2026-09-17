from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Iterable, Mapping


class AttackPlanError(ValueError):
    pass


def _ratio(rho: str | Fraction) -> Fraction:
    value = rho if isinstance(rho, Fraction) else Fraction(str(rho))
    if value < 0 or value > 1:
        raise AttackPlanError("rho must be in [0,1]")
    return value


def malicious_count_half_up(worker_count: int, rho: str | Fraction) -> int:
    if worker_count <= 0:
        raise AttackPlanError("worker_count must be positive")
    r = _ratio(rho)
    numerator = worker_count * r.numerator
    denominator = r.denominator
    # exact half-up rounding for nonnegative rational values
    return (2 * numerator + denominator) // (2 * denominator)


def select_malicious_workers(
    worker_ids: Iterable[str],
    *,
    dataset_id: str,
    rho: str | Fraction,
    seed: int,
    namespace: str = "phase-r1-malicious-subset-v1",
) -> tuple[str, ...]:
    workers = tuple(sorted(set(worker_ids)))
    if not workers:
        raise AttackPlanError("worker universe is empty")
    k = malicious_count_half_up(len(workers), rho)
    ranked = sorted(
        workers,
        key=lambda worker: hashlib.sha256(
            f"{namespace}|{dataset_id}|{rho}|{seed}|{worker}".encode("utf-8")
        ).digest(),
    )
    return tuple(sorted(ranked[:k]))


def build_unique_malicious_subsets(
    worker_ids: Iterable[str],
    *,
    dataset_id: str,
    rho: str | Fraction,
    requested: int,
    seed_start: int = 5001,
    max_seed_scan: int = 100000,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    workers = tuple(sorted(set(worker_ids)))
    if requested <= 0:
        raise AttackPlanError("requested must be positive")
    k = malicious_count_half_up(len(workers), rho)
    possible = math.comb(len(workers), k)
    target = min(requested, possible)
    seen: set[tuple[str, ...]] = set()
    chosen: list[tuple[int, tuple[str, ...]]] = []
    seed = seed_start
    while len(chosen) < target and seed < seed_start + max_seed_scan:
        subset = select_malicious_workers(workers, dataset_id=dataset_id, rho=rho, seed=seed)
        if subset not in seen:
            seen.add(subset)
            chosen.append((seed, subset))
        seed += 1
    if len(chosen) != target:
        raise AttackPlanError("unable to generate required unique malicious subsets")
    return tuple(chosen)


def apply_fixed_target(
    reports: Mapping[str, tuple[str | float | int, ...]],
    malicious_workers: Iterable[str],
    *,
    modality: str,
) -> dict[str, tuple[str | float | int, ...]]:
    malicious = set(malicious_workers)
    if not reports:
        raise AttackPlanError("empty reports")
    width = len(next(iter(reports.values())))
    if width <= 0:
        raise AttackPlanError("empty report vector")
    out: dict[str, tuple[str | float | int, ...]] = {}
    for worker, report in reports.items():
        if worker not in malicious:
            out[worker] = tuple(report)
            continue
        if modality == "numerical":
            out[worker] = tuple(1 for _ in range(width))
        elif modality == "categorical":
            out[worker] = tuple(1 if i == 0 else 0 for i in range(width))
        else:
            raise AttackPlanError(f"unsupported modality: {modality}")
    return out


def realized_malicious_report_fraction(
    tasks: Iterable[Mapping[str, object]],
    malicious_workers: Iterable[str],
) -> float:
    malicious = set(malicious_workers)
    total = 0
    attacked = 0
    for task in tasks:
        participants = tuple(str(x) for x in task["participant_ids"])
        total += len(participants)
        attacked += sum(worker in malicious for worker in participants)
    if total == 0:
        raise AttackPlanError("no task reports")
    return attacked / total


def _fraction_text(value):
    from fractions import Fraction
    f = value if isinstance(value, Fraction) else Fraction(str(value))
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def corrupt_report_truth_independent(
    report: tuple[str | float | int, ...],
    *,
    modality: str,
) -> tuple[str | float | int, ...]:
    """Type-correct response corruption that never inspects task truth.

    Binary categorical: flip class 0<->1.
    Multiclass categorical: cyclic shift class (y+1) mod K.
    Numerical scalar/vector in [0,1]: mirror x -> 1-x exactly.
    """
    if not report:
        raise AttackPlanError("empty report vector")
    if modality == "categorical":
        values = tuple(int(x) for x in report)
        if any(x not in (0, 1) for x in values) or sum(values) != 1:
            raise AttackPlanError("categorical report must be one-hot")
        width = len(values)
        if width < 2:
            raise AttackPlanError("categorical report width must be >=2")
        current = values.index(1)
        target = 1 - current if width == 2 else (current + 1) % width
        return tuple(1 if i == target else 0 for i in range(width))
    if modality == "numerical":
        from decimal import Decimal
        out = []
        for raw in report:
            value = Decimal(str(raw))
            if value < 0 or value > 1:
                raise AttackPlanError("numerical report outside [0,1]")
            mirrored = Decimal(1) - value
            # Decimal text remains accepted by both the float baselines and the
            # exact LIR-PPTD adapter; avoid rational strings such as 1/3.
            out.append(format(mirrored, "f"))
        return tuple(out)
    raise AttackPlanError(f"unsupported modality: {modality}")


def schedule_is_on(task_index: int, on_count: int, off_count: int) -> bool:
    if task_index < 0:
        raise AttackPlanError("task_index must be nonnegative")
    if on_count <= 0 or off_count <= 0:
        raise AttackPlanError("on/off counts must be positive")
    period = on_count + off_count
    return (task_index % period) < on_count


def apply_response_corruption(
    reports: Mapping[str, tuple[str | float | int, ...]],
    malicious_workers: Iterable[str],
    *,
    modality: str,
    active: bool = True,
) -> dict[str, tuple[str | float | int, ...]]:
    malicious = set(malicious_workers)
    out: dict[str, tuple[str | float | int, ...]] = {}
    for worker, report in reports.items():
        if active and worker in malicious:
            out[worker] = corrupt_report_truth_independent(tuple(report), modality=modality)
        else:
            out[worker] = tuple(report)
    return out
