from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ..canonical import canonical_json_bytes

WORKER_IDS = tuple(f"w{i:03d}" for i in range(20))
SCORING_TASK_COUNT = 100
GENERATOR_VERSION = "lir-pptd-phase6-r2-v1"
RHO_GRID = ("0", "1/10", "3/10", "1/2", "7/10", "9/10")
MALICIOUS_COUNTS = {"0": 0, "1/10": 2, "3/10": 6, "1/2": 10, "7/10": 14, "9/10": 18}
WARM_PREFIX_CANDIDATES = (10, 20, 30)
ON_OFF_SCHEDULES = ((5, 1), (5, 5), (1, 5))
FIXED_TARGET_NUMERICAL_CANDIDATES = ("all_zero_vector", "all_one_vector")
FIXED_TARGET_CATEGORICAL_CANDIDATES = ("class_0", "class_1")
ORACLE_DELTA_SCALE_CANDIDATES = ("0.1", "0.25", "0.5", "1.0")
ORACLE_CATEGORICAL_VARIANTS = ("uniform_over_incorrect_labels", "deterministic_minimum_index_incorrect_label")


def _decision(root: Path) -> dict[str, Any]:
    return json.loads((root / "docs/PHASE6_R2_GENERATOR_DECISIONS.json").read_text(encoding="utf-8-sig"))


def _digest(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _u01(seed: int, namespace: str) -> float:
    return int.from_bytes(_digest(GENERATOR_VERSION, seed, namespace), "big") / 2**256


def _normal(seed: int, namespace: str) -> float:
    u1 = max(_u01(seed, namespace + "/u1"), 2**-256)
    u2 = _u01(seed, namespace + "/u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _uniform(seed: int, namespace: str, low: float, high: float) -> float:
    return low + (high - low) * _u01(seed, namespace)


def _number(value: float) -> str:
    return format(value, ".17g")


def _rank_workers(seed: int, namespace: str, workers: Iterable[str] = WORKER_IDS) -> list[str]:
    return sorted(workers, key=lambda worker: _digest(GENERATOR_VERSION, seed, namespace, worker))


def malicious_ids(seed: int, rho: str) -> tuple[str, ...]:
    return tuple(_rank_workers(seed, f"malicious_identity/{rho}")[:MALICIOUS_COUNTS[rho]])


def _one_hot(label: int) -> list[int]:
    return [1, 0] if label == 0 else [0, 1]


def _truth_numerical(seed: int, family: str, task_index: int, namespace_prefix: str) -> list[str]:
    if family == "synthetic_num_small":
        return [_number(_uniform(seed, f"{namespace_prefix}/{family}/{task_index}/{h}", 0.25, 0.75)) for h in range(5)]
    if family == "synthetic_num_shifted":
        low, high = ((0.20, 0.50) if task_index <= 50 else (0.50, 0.80))
        return [_number(_uniform(seed, f"{namespace_prefix}/{family}/{task_index}/{h}", low, high)) for h in range(5)]
    if family == "synthetic_num_longitudinal":
        if task_index == 1:
            return [_number(_uniform(seed, f"{namespace_prefix}/{family}/{task_index}/{h}", 0.25, 0.75)) for h in range(5)]
        previous = _truth_numerical(seed, family, task_index - 1, namespace_prefix)
        return [
            _number(min(0.90, max(0.10, float(previous[h]) + 0.03 * _normal(seed, f"{namespace_prefix}/{family}/{task_index}/{h}"))))
            for h in range(5)
        ]
    raise ValueError(f"not numerical: {family}")


def _truth_categorical(seed: int, family: str, task_index: int, namespace_prefix: str) -> list[int]:
    if family == "synthetic_cat_binary":
        label = 0 if _u01(seed, f"{namespace_prefix}/{family}/{task_index}") < 0.5 else 1
        return _one_hot(label)
    if family == "synthetic_cat_tie":
        return _one_hot(0 if task_index % 2 == 1 else 1)
    raise ValueError(f"not categorical: {family}")


def _honest_numerical_report(seed: int, family: str, task_index: int, worker: str, truth: list[str], namespace_prefix: str) -> list[str]:
    return [
        _number(min(1.0, max(0.0, float(truth[h]) + 0.05 * _normal(seed, f"{namespace_prefix}/{family}/{task_index}/{worker}/{h}"))))
        for h in range(5)
    ]


def _honest_categorical_report(seed: int, family: str, task_index: int, worker: str, truth: list[int], namespace_prefix: str) -> list[int]:
    label = 0 if truth[0] == 1 else 1
    if _u01(seed, f"{namespace_prefix}/{family}/{task_index}/{worker}") < 0.95:
        return _one_hot(label)
    return _one_hot(1 - label)


def _random_numerical_attack(seed: int, task_index: int, worker: str) -> list[str]:
    return [_number(_u01(seed, f"attack/random_numerical/{task_index}/{worker}/{h}")) for h in range(5)]


def _random_categorical_attack(seed: int, task_index: int, worker: str) -> list[int]:
    label = 0 if _u01(seed, f"attack/random_categorical/{task_index}/{worker}") < 0.5 else 1
    return _one_hot(label)


def _fixed_target_numerical_candidate(candidate: str) -> list[str]:
    if candidate == "all_zero_vector":
        return ["0", "0", "0", "0", "0"]
    if candidate == "all_one_vector":
        return ["1", "1", "1", "1", "1"]
    raise ValueError(f"unsupported candidate: {candidate}")


def _fixed_target_categorical_candidate(candidate: str) -> list[int]:
    if candidate == "class_0":
        return [1, 0]
    if candidate == "class_1":
        return [0, 1]
    raise ValueError(f"unsupported candidate: {candidate}")


def score_fixed_target_candidates(
    validation_truths_by_family: dict[str, list[list[str]] | list[list[int]]],
    family: str,
) -> dict[str, Any]:
    truths = validation_truths_by_family.get(family)
    if truths is None:
        raise ValueError(f"missing validation truths for {family}")
    if family.startswith("synthetic_num"):
        scores: dict[str, float] = {}
        for candidate in FIXED_TARGET_NUMERICAL_CANDIDATES:
            candidate_truth = _fixed_target_numerical_candidate(candidate)
            scores[candidate] = sum(
                math.sqrt(sum((float(truth[h]) - float(candidate_truth[h])) ** 2 for h in range(len(candidate_truth))))
                for truth in truths  # type: ignore[arg-type]
            ) / len(truths)
        return {"candidates": list(FIXED_TARGET_NUMERICAL_CANDIDATES), "scores": {key: _number(value) for key, value in scores.items()}}
    if family.startswith("synthetic_cat"):
        scores = {}
        for candidate in FIXED_TARGET_CATEGORICAL_CANDIDATES:
            candidate_truth = _fixed_target_categorical_candidate(candidate)
            scores[candidate] = sum(1 for truth in truths if truth != candidate_truth) / len(truths)  # type: ignore[arg-type]
        return {"candidates": list(FIXED_TARGET_CATEGORICAL_CANDIDATES), "scores": {key: _number(value) for key, value in scores.items()}}
    raise ValueError(f"unsupported family: {family}")


def score_warm_prefix_candidates(validation_primary_losses_by_length: dict[int, float]) -> dict[str, Any]:
    if set(validation_primary_losses_by_length) != set(WARM_PREFIX_CANDIDATES):
        raise ValueError("warm-prefix scores must cover exactly 10, 20, and 30")
    return {
        "candidates": list(WARM_PREFIX_CANDIDATES),
        "scores": {str(length): _number(validation_primary_losses_by_length[length]) for length in WARM_PREFIX_CANDIDATES},
    }


def _oracle_direction(seed: int, family: str, task_index: int, worker: str) -> list[float]:
    raw = [_normal(seed, f"attack/oracle_numerical/{family}/{task_index}/{worker}/{h}") for h in range(5)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def _oracle_numerical_attack(seed: int, family: str, task_index: int, worker: str, truth: list[str], delta_scale: str = "0.1") -> list[str]:
    delta = float(delta_scale)
    direction = _oracle_direction(seed, family, task_index, worker)
    return [_number(min(1.0, max(0.0, float(truth[h]) + delta * direction[h]))) for h in range(5)]


def _oracle_categorical_attack(
    seed: int,
    family: str,
    task_index: int,
    worker: str,
    truth: list[int],
    variant: str = "deterministic_minimum_index_incorrect_label",
) -> list[int]:
    label = 0 if truth[0] == 1 else 1
    if variant == "uniform_over_incorrect_labels":
        return _one_hot(1 - label)
    if variant == "deterministic_minimum_index_incorrect_label":
        return _one_hot(1 - label)
    raise ValueError(f"unsupported oracle variant: {variant}")


def _fixed_target_attack(attack_id: str, explicit_target: str | None) -> list[str] | list[int]:
    if explicit_target is None:
        raise ValueError("fixed-target attack requires explicit frozen target selection")
    if attack_id == "fixed_target_numerical":
        return _fixed_target_numerical_candidate(explicit_target)
    if attack_id == "fixed_target_categorical":
        return _fixed_target_categorical_candidate(explicit_target)
    raise ValueError(f"unsupported fixed-target attack: {attack_id}")


def _validate_on_off_schedule(schedule: tuple[int, int] | None) -> tuple[int, int] | None:
    if schedule is None:
        return None
    if schedule not in ON_OFF_SCHEDULES:
        raise ValueError("invalid on_off schedule")
    return schedule


def _on_off_state(task_index: int, malicious: bool, schedule: tuple[int, int] | None = None) -> dict[str, Any]:
    if schedule is None:
        schedule = (5, 5)
    benign_length, attack_length = schedule
    if not malicious:
        return {
            "active": False,
            "phase": "benign",
            "actual_participation_count": task_index,
            "benign_length": benign_length,
            "attack_length": attack_length,
        }
    cycle = benign_length + attack_length
    index = (task_index - 1) % cycle
    active = index >= benign_length
    return {
        "active": active,
        "phase": "attack" if active else "benign",
        "actual_participation_count": task_index,
        "benign_length": benign_length,
        "attack_length": attack_length,
    }


def _build_task_payload(
    seed: int,
    family: str,
    task_index: int,
    rho: str,
    attack_id: str,
    *,
    truth_namespace_prefix: str,
    honest_noise_namespace_prefix: str,
    categorical_tie_namespace_prefix: str,
    fixed_target_selection: str | None,
    is_warm_prefix: bool,
    on_off_schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    malicious = set(malicious_ids(seed, rho)) if not is_warm_prefix else set()
    if family == "synthetic_cat_tie":
        truth = _truth_categorical(seed, family, task_index, f"{truth_namespace_prefix}/{family}/{task_index}")
        ranked = _rank_workers(seed, f"{categorical_tie_namespace_prefix}/{task_index}")
        labels = {worker: (0 if index < 10 else 1) for index, worker in enumerate(ranked)}
        reports = {worker: _one_hot(labels[worker]) for worker in WORKER_IDS}
        dimension_or_vocabulary: Any = ["class_0", "class_1"]
    elif family.startswith("synthetic_cat"):
        truth = _truth_categorical(seed, family, task_index, f"{truth_namespace_prefix}/{family}/{task_index}")
        reports = {}
        for worker in WORKER_IDS:
            if worker in malicious and attack_id != "no_attack":
                if attack_id == "fixed_target_categorical":
                    reports[worker] = _fixed_target_attack(attack_id, fixed_target_selection)
                elif attack_id == "random_categorical":
                    reports[worker] = _random_categorical_attack(seed, task_index, worker)
                elif attack_id == "oracle_categorical":
                    reports[worker] = _oracle_categorical_attack(seed, family, task_index, worker, truth)
                elif attack_id == "on_off":
                    if on_off_schedule is None:
                        raise ValueError("on_off schedule required")
                    benign_length, attack_length = on_off_schedule
                    cycle = benign_length + attack_length
                    phase_index = (task_index - 1) % cycle
                    if phase_index < benign_length:
                        reports[worker] = _honest_categorical_report(seed, family, task_index, worker, truth, f"{honest_noise_namespace_prefix}/{family}/{task_index}")
                    else:
                        reports[worker] = _fixed_target_categorical_candidate(fixed_target_selection) if fixed_target_selection is not None else _fixed_target_attack(attack_id, fixed_target_selection)
                else:
                    raise ValueError(f"unsupported attack_id: {attack_id}")
            else:
                reports[worker] = _honest_categorical_report(seed, family, task_index, worker, truth, f"{honest_noise_namespace_prefix}/{family}/{task_index}")
        dimension_or_vocabulary = ["class_0", "class_1"]
    else:
        truth = _truth_numerical(seed, family, task_index, f"{truth_namespace_prefix}/{family}/{task_index}")
        reports = {}
        for worker in WORKER_IDS:
            if worker in malicious and attack_id != "no_attack":
                if attack_id == "fixed_target_numerical":
                    reports[worker] = _fixed_target_attack(attack_id, fixed_target_selection)
                elif attack_id == "random_numerical":
                    reports[worker] = _random_numerical_attack(seed, task_index, worker)
                elif attack_id == "oracle_numerical":
                    reports[worker] = _oracle_numerical_attack(seed, family, task_index, worker, truth)
                elif attack_id == "on_off":
                    if on_off_schedule is None:
                        raise ValueError("on_off schedule required")
                    benign_length, attack_length = on_off_schedule
                    cycle = benign_length + attack_length
                    phase_index = (task_index - 1) % cycle
                    if phase_index < benign_length:
                        reports[worker] = _honest_numerical_report(seed, family, task_index, worker, truth, f"{honest_noise_namespace_prefix}/{family}/{task_index}/{worker}")
                    else:
                        reports[worker] = _fixed_target_numerical_candidate(fixed_target_selection) if fixed_target_selection is not None else _fixed_target_attack(attack_id, fixed_target_selection)
                else:
                    raise ValueError(f"unsupported attack_id: {attack_id}")
            else:
                reports[worker] = _honest_numerical_report(seed, family, task_index, worker, truth, f"{honest_noise_namespace_prefix}/{family}/{task_index}/{worker}")
        dimension_or_vocabulary = 5

    if attack_id == "on_off":
        if on_off_schedule is None:
            raise ValueError("on_off schedule required")
        attack_state = _on_off_state(task_index, True, on_off_schedule)
    else:
        attack_state = {
            "scored": not is_warm_prefix,
            "is_warm_prefix": is_warm_prefix,
            "attack_family": attack_id,
            "on_off_counter": 0 if is_warm_prefix else task_index if attack_id == "on_off" else 0,
        }

    return {
        "task_id": f"{family}__task-{task_index:03d}" if not is_warm_prefix else f"warm_prefix__{family}__task-{task_index:03d}",
        "task_index": task_index,
        "modality": "categorical" if family.startswith("synthetic_cat") else "numerical",
        "dimension_or_vocabulary": dimension_or_vocabulary,
        "ground_truth": truth,
        "participant_ids": list(WORKER_IDS),
        "reports": reports,
        "malicious_ids": sorted(malicious),
        "attack_id": attack_id,
        "is_warm_prefix": is_warm_prefix,
        "attack_state": attack_state,
        "recovery": {
            "fault_profile": "phase5_e8a_frozen_fault_profile",
            "backend": "sqlite_wal_single_process",
            "actual_participation_count": task_index,
            "recovery_latency_ms": 0,
        },
        "source_seed": seed,
        "generator_namespace": f"{truth_namespace_prefix}/{family}/{task_index}" if not is_warm_prefix else f"warm_prefix/{family}/{task_index}",
    }


def _build_scored_task(
    seed: int,
    family: str,
    task_index: int,
    rho: str,
    attack_id: str,
    *,
    fixed_target_selection: str | None,
    on_off_schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return _build_task_payload(
        seed,
        family,
        task_index,
        rho,
        attack_id,
        truth_namespace_prefix="truth",
        honest_noise_namespace_prefix="honest_noise",
        categorical_tie_namespace_prefix="categorical_tie",
        fixed_target_selection=fixed_target_selection,
        is_warm_prefix=False,
        on_off_schedule=_validate_on_off_schedule(on_off_schedule if attack_id == "on_off" else None),
    )


def _build_warm_prefix_task(seed: int, family: str, task_index: int) -> dict[str, Any]:
    return _build_task_payload(
        seed,
        family,
        task_index,
        "0",
        "no_attack",
        truth_namespace_prefix="warm_prefix/truth",
        honest_noise_namespace_prefix="warm_prefix/honest_noise",
        categorical_tie_namespace_prefix="warm_prefix/categorical_tie",
        fixed_target_selection=None,
        is_warm_prefix=True,
        on_off_schedule=None,
    )


def _build_warm_prefix(seed: int, family: str, warm_prefix_length: int | None) -> dict[str, Any] | None:
    if warm_prefix_length in (None, 0):
        return None
    if warm_prefix_length not in WARM_PREFIX_CANDIDATES:
        raise ValueError("warm_prefix_length must be one of 10, 20, or 30")
    tasks = [_build_warm_prefix_task(seed, family, index) for index in range(1, warm_prefix_length + 1)]
    return {
        "length": warm_prefix_length,
        "attack_free": True,
        "scored": False,
        "tasks": tasks,
        "task_artifact_hashes": [hashlib.sha256(canonical_json_bytes(task)).hexdigest() for task in tasks],
    }


def generate_stream(
    root: Path,
    split: str,
    seed: int,
    family: str,
    rho: str = "0",
    attack_id: str = "no_attack",
    *,
    fixed_target_selection: str | None = None,
    warm_prefix_length: int | None = None,
    on_off_schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    _ = _decision(root)
    allowed = {"validation": range(1001, 1031), "formal_candidate": range(2001, 2031)}
    if split == "final_test" or split not in allowed or seed not in allowed[split]:
        raise ValueError("only validation and formal_candidate seeds are permitted")
    if attack_id in {"fixed_target_numerical", "fixed_target_categorical"} and fixed_target_selection is None:
        raise ValueError("fixed-target attack requires explicit frozen target selection")
    if attack_id == "on_off" and fixed_target_selection is None:
        raise ValueError("on_off requires explicit frozen fixed target selection")
    if attack_id == "on_off" and on_off_schedule is None:
        raise ValueError("on_off schedule requires explicit schedule")
    if warm_prefix_length not in (None, 0, 10, 20, 30):
        raise ValueError("warm_prefix_length must be one of 10, 20, or 30")

    scored_tasks = [
        _build_scored_task(seed, family, index, rho, attack_id, fixed_target_selection=fixed_target_selection, on_off_schedule=on_off_schedule)
        for index in range(1, SCORING_TASK_COUNT + 1)
    ]
    warm_prefix = _build_warm_prefix(seed, family, warm_prefix_length)
    task_hashes = [hashlib.sha256(canonical_json_bytes(task)).hexdigest() for task in scored_tasks]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "paired_seed": seed,
        "task_family": family,
        "task_count": len(scored_tasks),
        "worker_ids": list(WORKER_IDS),
        "warm_prefix_length": warm_prefix_length,
        "warm_prefix_attack_free": True,
        "warm_prefix_primary_endpoint_scored": False,
        "validation_formal_disjoint": True,
        "fixed_target_candidates": {
            "numerical": list(FIXED_TARGET_NUMERICAL_CANDIDATES),
            "categorical": list(FIXED_TARGET_CATEGORICAL_CANDIDATES),
        },
        "fixed_target_selection": fixed_target_selection,
        "oracle_variants": {
            "numerical_delta_scales_times_sqrt_D": list(ORACLE_DELTA_SCALE_CANDIDATES),
            "categorical_variants": list(ORACLE_CATEGORICAL_VARIANTS),
        },
        "generator_config_hash": hashlib.sha256(canonical_json_bytes(_decision(root))).hexdigest(),
        "task_artifact_hashes": task_hashes,
        "scored_tasks": scored_tasks,
        "tasks": scored_tasks,
        "malicious_ids": list(malicious_ids(seed, rho)),
        "attack_id": attack_id,
    }
    if warm_prefix is not None:
        manifest["warm_prefix"] = warm_prefix
    manifest["stream_hash"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest
