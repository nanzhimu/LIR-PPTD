from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .experiments.phase6_ablation_runner import run_ablation_sequence
from .experiments.phase6_candidate_runner import run_candidate
from .experiments.phase6_generator import (
    FIXED_TARGET_CATEGORICAL_CANDIDATES,
    FIXED_TARGET_NUMERICAL_CANDIDATES,
    WARM_PREFIX_CANDIDATES,
    generate_stream,
)
from .experiments.phase6_exact_adapter import LongitudinalState

E4_FAMILIES = (
    "synthetic_num_small",
    "synthetic_num_shifted",
    "synthetic_num_longitudinal",
    "synthetic_cat_binary",
    "synthetic_cat_tie",
)
E4_ATTACK_CLASSES = ("fixed_target", "on_off")
E4_ON_OFF_SCHEDULES = ((5, 1), (5, 5), (1, 5))
E4_RHO_GRID = ("0", "1/10", "3/10", "1/2", "7/10", "9/10")
E4_START_CONDITIONS = ("cold", "warm")
E4_VARIANTS = ("full", "nr", "ho", "wi", "prefinal", "symmetric")
REFERENCE_LIR_CONFIG_ID = "lir_cfg_01_reference"

StreamLoader = Callable[..., dict[str, Any]]
CandidateRunner = Callable[..., dict[str, Any]]
AblationSequenceRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SelectionCell:
    seed: int
    family: str
    status: str
    denominator: int
    loss: float | None
    artifact_hash: str | None = None
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class SelectionResult:
    status: str
    winner: str | None
    candidates: tuple[Any, ...]
    aggregate_scores: dict[str, float | None]
    per_seed_scores: dict[int, dict[str, float | None]]
    per_seed_family_scores: dict[int, dict[str, dict[str, float | None]]]
    cells: tuple[SelectionCell, ...]
    denominator: int
    status_counts: dict[str, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class E4MatrixCell:
    seed: int
    family: str
    attack_class: str
    attack_id: str
    rho: str
    start: str
    schedule: tuple[int, int] | None
    variant: str
    status: str
    denominator: int
    primary_loss: float | None
    same_artifact_reused: bool
    cold_warm_hash_equal: bool
    scored_task_artifact_hashes: tuple[str, ...]
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class E4MatrixResult:
    status: str
    run_count: int
    denominator: int
    cells: tuple[E4MatrixCell, ...]
    condition_keys: tuple[str, ...]
    metadata: dict[str, Any]


def _fail_closed_for_final_test(split: str) -> None:
    if split == "final_test":
        raise ValueError("final_test access is blocked")


def _load_method_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json").read_text(encoding="utf-8-sig"))


def _candidate_configs(root: Path) -> list[dict[str, Any]]:
    return list(_load_method_config(root)["candidate_configs"])


def _config_map(root: Path) -> dict[str, dict[str, Any]]:
    return {config["config_id"]: config for config in _candidate_configs(root)}


def _reference_config(root: Path) -> dict[str, Any]:
    return _config_map(root)[REFERENCE_LIR_CONFIG_ID]


def _fixed_target_vector(candidate: str) -> list[str]:
    if candidate == "all_zero_vector":
        return ["0", "0", "0", "0", "0"]
    if candidate == "all_one_vector":
        return ["1", "1", "1", "1", "1"]
    raise ValueError(f"unsupported numerical candidate: {candidate}")


def _fixed_target_label(candidate: str) -> int:
    if candidate == "class_0":
        return 0
    if candidate == "class_1":
        return 1
    raise ValueError(f"unsupported categorical candidate: {candidate}")


def _ground_truth_class_index(ground_truth: Sequence[Any]) -> int:
    return 0 if float(ground_truth[0]) >= float(ground_truth[1]) else 1


def _numerical_l2(candidate: Sequence[Any], truth: Sequence[Any]) -> float:
    return math.sqrt(sum((float(c) - float(t)) ** 2 for c, t in zip(candidate, truth)))


def _categorical_disagreement(candidate_index: int | None, truth: Sequence[Any]) -> float:
    if candidate_index is None:
        return 1.0
    return 0.0 if candidate_index == _ground_truth_class_index(truth) else 1.0


def _stream_loader_default(
    root: Path,
    split: str,
    seed: int,
    family: str,
    *,
    rho: str = "0",
    attack_id: str = "no_attack",
    fixed_target_selection: str | None = None,
    warm_prefix_length: int | None = None,
    on_off_schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return generate_stream(
        root,
        split,
        seed,
        family,
        rho=rho,
        attack_id=attack_id,
        fixed_target_selection=fixed_target_selection,
        warm_prefix_length=warm_prefix_length,
        on_off_schedule=on_off_schedule,
    )


def _candidate_runner_default(**kwargs: Any) -> dict[str, Any]:
    return run_candidate(**kwargs)


def _ablation_sequence_runner_default(**kwargs: Any) -> dict[str, Any]:
    return run_ablation_sequence(**kwargs)


def _cell_status_counts(cells: Iterable[SelectionCell | E4MatrixCell]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.status] = counts.get(cell.status, 0) + 1
    return counts


def _fixed_target_scores_for_stream(stream: Mapping[str, Any]) -> dict[str, float]:
    truths = [task["ground_truth"] for task in stream["scored_tasks"]]
    if stream["scored_tasks"][0]["modality"] == "numerical":
        return {
            candidate: sum(_numerical_l2(_fixed_target_vector(candidate), truth) for truth in truths) / len(truths)
            for candidate in FIXED_TARGET_NUMERICAL_CANDIDATES
        }
    return {
        candidate: sum(1 for truth in truths if _fixed_target_label(candidate) != _ground_truth_class_index(truth)) / len(truths)
        for candidate in FIXED_TARGET_CATEGORICAL_CANDIDATES
    }


def select_global_fixed_targets(
    root: Path,
    seeds: Sequence[int],
    *,
    split: str = "validation",
    stream_loader: StreamLoader = _stream_loader_default,
    families: Sequence[str] = E4_FAMILIES,
) -> SelectionResult:
    _fail_closed_for_final_test(split)
    if not seeds:
        raise ValueError("seed list must be supplied explicitly")

    per_seed_scores: dict[int, dict[str, float | None]] = {seed: {} for seed in seeds}
    per_seed_family_scores: dict[int, dict[str, dict[str, float | None]]] = {seed: {} for seed in seeds}
    cells: list[SelectionCell] = []

    numerical_agg: dict[str, list[float]] = {candidate: [] for candidate in FIXED_TARGET_NUMERICAL_CANDIDATES}
    categorical_agg: dict[str, list[float]] = {candidate: [] for candidate in FIXED_TARGET_CATEGORICAL_CANDIDATES}

    for seed in seeds:
        per_seed_family_scores[seed] = {}
        for family in families:
            stream = stream_loader(root, split, seed, family)
            family_scores = _fixed_target_scores_for_stream(stream)
            per_seed_family_scores[seed][family] = dict(family_scores)
            cells.extend(
                SelectionCell(seed=seed, family=family, status="success", denominator=1, loss=score, detail={"candidate": candidate})
                for candidate, score in family_scores.items()
            )
            if family.startswith("synthetic_num"):
                for candidate, score in family_scores.items():
                    numerical_agg[candidate].append(score)
            else:
                for candidate, score in family_scores.items():
                    categorical_agg[candidate].append(score)

    numerical_scores = {candidate: (sum(values) / len(values) if values else None) for candidate, values in numerical_agg.items()}
    categorical_scores = {candidate: (sum(values) / len(values) if values else None) for candidate, values in categorical_agg.items()}

    for seed in seeds:
        numerical_family_scores = [per_seed_family_scores[seed][family] for family in families if family.startswith("synthetic_num")]
        categorical_family_scores = [per_seed_family_scores[seed][family] for family in families if family.startswith("synthetic_cat")]
        if numerical_family_scores:
            per_seed_scores[seed]["all_zero_vector"] = sum(score["all_zero_vector"] for score in numerical_family_scores) / len(numerical_family_scores)
            per_seed_scores[seed]["all_one_vector"] = sum(score["all_one_vector"] for score in numerical_family_scores) / len(numerical_family_scores)
        if categorical_family_scores:
            per_seed_scores[seed]["class_0"] = sum(score["class_0"] for score in categorical_family_scores) / len(categorical_family_scores)
            per_seed_scores[seed]["class_1"] = sum(score["class_1"] for score in categorical_family_scores) / len(categorical_family_scores)

    numerical_winner = max(FIXED_TARGET_NUMERICAL_CANDIDATES, key=lambda candidate: (numerical_scores[candidate], 1 if candidate == "all_zero_vector" else 0))
    categorical_winner = max(FIXED_TARGET_CATEGORICAL_CANDIDATES, key=lambda candidate: (categorical_scores[candidate], 1 if candidate == "class_0" else 0))
    aggregate_scores: dict[str, float | None] = {**numerical_scores, **categorical_scores}
    return SelectionResult(
        status="selected",
        winner=None,
        candidates=("numerical", "categorical"),
        aggregate_scores=aggregate_scores,
        per_seed_scores=per_seed_scores,
        per_seed_family_scores=per_seed_family_scores,
        cells=tuple(cells),
        denominator=len(seeds),
        status_counts=_cell_status_counts(cells),
        metadata={"numerical_winner": numerical_winner, "categorical_winner": categorical_winner},
    )


def _task_loss(task_artifact: Mapping[str, Any], task_record: Mapping[str, Any]) -> tuple[float | None, str]:
    exact_record = next((record for record in task_record.get("method_records", []) if record.get("method") == "lir_pptd_full"), None)
    if exact_record is None:
        return None, "incomplete"
    if exact_record.get("status") != "success":
        return None, str(exact_record.get("status", "failure"))
    if task_artifact["modality"] == "numerical":
        return _numerical_l2(exact_record["output"], task_artifact["ground_truth"]), "success"
    return _categorical_disagreement(exact_record.get("released_class_index"), task_artifact["ground_truth"]), "success"


def _candidate_losses_for_condition(
    *,
    root: Path,
    split: str,
    seed: int,
    family: str,
    rho: str,
    attack_id: str,
    fixed_target_selection: str,
    config_id: str,
    warm_prefix_length: int | None,
    on_off_schedule: tuple[int, int] | None,
    candidate_runner: CandidateRunner,
) -> tuple[float | None, str, int, dict[str, Any]]:
    candidate_output = candidate_runner(
        root=root,
        split=split,
        seed=seed,
        family=family,
        rho=rho,
        attack_id=attack_id,
        config_id=config_id,
        fixed_target_selection=fixed_target_selection,
        warm_prefix_length=warm_prefix_length,
        on_off_schedule=on_off_schedule,
    )
    stream = candidate_output["stream"]
    task_records = list(candidate_output.get("task_records", []))
    task_artifacts = list(stream.get("scored_tasks", []))
    if len(task_records) != len(task_artifacts):
        return None, "incomplete", len(task_artifacts), {"reason": "task count mismatch", "task_count": len(task_artifacts)}
    task_losses: list[float] = []
    statuses: list[str] = []
    for task_artifact, task_record in zip(task_artifacts, task_records):
        loss, status = _task_loss(task_artifact, task_record)
        statuses.append(status)
        if loss is not None:
            task_losses.append(loss)
    if any(status != "success" for status in statuses):
        return None, "failure", len(task_artifacts), {"statuses": statuses}
    return (sum(task_losses) / len(task_losses)) if task_losses else None, "success", len(task_artifacts), {"statuses": statuses}


def _select_from_candidates(
    *,
    candidates: Sequence[Any],
    per_seed_family_scores: dict[int, dict[str, dict[str, float | None]]],
    per_seed_scores: dict[int, dict[str, float | None]],
    seed_family_conditions: dict[int, dict[str, dict[str, float | None]]],
    seed_order: Sequence[int],
    winner_key_fn: Callable[[Any, dict[str, float | None]], tuple[Any, ...]],
    blocked: bool,
    blocked_denominator: int,
    tie_break: Sequence[Any],
) -> tuple[str | None, dict[str, float | None], dict[str, int], int]:
    status_counts = {"success": 0, "failure": 0, "abort": 0, "timeout": 0, "incomplete": 0}
    aggregate_scores: dict[str, float | None] = {str(candidate): None for candidate in candidates}
    if blocked:
        return None, aggregate_scores, status_counts, blocked_denominator
    for candidate in candidates:
        candidate_key = str(candidate)
        seed_means: list[float] = []
        for seed in seed_order:
            family_means = [seed_family_conditions[seed][family][candidate_key] for family in seed_family_conditions[seed] if seed_family_conditions[seed][family][candidate_key] is not None]
            if len(family_means) != len(seed_family_conditions[seed]):
                return None, aggregate_scores, status_counts, blocked_denominator
            seed_means.append(sum(family_means) / len(family_means))
        aggregate_scores[candidate_key] = sum(seed_means) / len(seed_means) if seed_means else None
    winner = min(candidates, key=lambda candidate: winner_key_fn(candidate, aggregate_scores))
    return str(winner), aggregate_scores, status_counts, blocked_denominator


def _selection_from_condition_results(
    *,
    candidates: Sequence[Any],
    results_by_seed_family: dict[int, dict[str, dict[str, float | None]]],
    seed_order: Sequence[int],
    tie_break: Callable[[str], Any],
    blocked: bool,
) -> tuple[str | None, dict[str, float | None], int]:
    aggregate_scores: dict[str, float | None] = {str(candidate): None for candidate in candidates}
    if blocked:
        return None, aggregate_scores, len(seed_order)
    for candidate in candidates:
        candidate_key = str(candidate)
        seed_means: list[float] = []
        for seed in seed_order:
            family_means = [score[candidate_key] for score in results_by_seed_family[seed].values() if score[candidate_key] is not None]
            if len(family_means) != len(results_by_seed_family[seed]):
                return None, aggregate_scores, len(seed_order)
            seed_means.append(sum(family_means) / len(family_means))
        aggregate_scores[candidate_key] = sum(seed_means) / len(seed_means) if seed_means else None
    winner = min(candidates, key=lambda candidate: (aggregate_scores[str(candidate)], tie_break(str(candidate))))
    return str(winner), aggregate_scores, len(seed_order)


def _warm_prefix_state(
    prefix_tasks: Sequence[Mapping[str, Any]],
    candidate_config: Mapping[str, Any],
    *,
    ablation_sequence_runner: AblationSequenceRunner,
) -> LongitudinalState:
    if not prefix_tasks:
        return LongitudinalState()
    prefix_result = ablation_sequence_runner(task_artifacts=tuple(prefix_tasks), candidate_config=candidate_config, longitudinal_state=LongitudinalState(), ablation="full")
    final_state = prefix_result.get("final_state")
    if isinstance(final_state, LongitudinalState):
        return final_state
    return LongitudinalState(reputations=dict(final_state.get("reputations", {})), epochs=dict(final_state.get("epochs", {})))


def select_global_warm_prefix(
    root: Path,
    seeds: Sequence[int],
    *,
    fixed_targets: Mapping[str, str],
    split: str = "validation",
    config_id: str = REFERENCE_LIR_CONFIG_ID,
    candidate_runner: CandidateRunner = _candidate_runner_default,
    families: Sequence[str] = E4_FAMILIES,
    rhos: Sequence[str] = E4_RHO_GRID,
) -> SelectionResult:
    _fail_closed_for_final_test(split)
    if set(fixed_targets) != {"numerical", "categorical"}:
        raise ValueError("global fixed targets must be supplied explicitly")
    if config_id != REFERENCE_LIR_CONFIG_ID:
        raise ValueError("warm-prefix selection must use lir_cfg_01_reference")
    if not seeds:
        raise ValueError("seed list must be supplied explicitly")

    configs = _config_map(root)
    if config_id not in configs:
        raise ValueError("unknown frozen config id")

    per_seed_scores: dict[int, dict[str, float | None]] = {seed: {} for seed in seeds}
    per_seed_family_scores: dict[int, dict[str, dict[str, float | None]]] = {seed: {} for seed in seeds}
    cells: list[SelectionCell] = []
    candidate_condition_scores: dict[int, dict[int, dict[str, float | None]]] = {seed: {} for seed in seeds}
    blocked = False
    blocked_denominator = 0
    aggregate_scores: dict[str, float | None] = {str(length): None for length in WARM_PREFIX_CANDIDATES}

    for seed in seeds:
        for family in families:
            per_seed_family_scores[seed][family] = {str(length): None for length in WARM_PREFIX_CANDIDATES}
        for length in WARM_PREFIX_CANDIDATES:
            condition_losses: list[float] = []
            condition_statuses: list[str] = []
            for family in families:
                for rho in rhos:
                    attack_id = "fixed_target_numerical" if family.startswith("synthetic_num") else "fixed_target_categorical"
                    fixed_target_selection = fixed_targets["numerical"] if family.startswith("synthetic_num") else fixed_targets["categorical"]
                    loss, status, denominator, detail = _candidate_losses_for_condition(
                        root=root,
                        split=split,
                        seed=seed,
                        family=family,
                        rho=rho,
                        attack_id=attack_id,
                        fixed_target_selection=fixed_target_selection,
                        config_id=config_id,
                        warm_prefix_length=length,
                        on_off_schedule=None,
                        candidate_runner=candidate_runner,
                    )
                    blocked_denominator += denominator
                    condition_statuses.append(status)
                    if loss is None:
                        blocked = True
                    else:
                        condition_losses.append(loss)
                    cells.append(
                        SelectionCell(
                            seed=seed,
                            family=family,
                            status=status,
                            denominator=denominator,
                            loss=loss,
                            detail={"warm_prefix_length": length, **detail},
                        )
                    )
            per_seed_family_scores[seed]["warm_prefix"] = {str(length): (sum(condition_losses) / len(condition_losses) if condition_losses else None)}
            per_seed_scores[seed][str(length)] = per_seed_family_scores[seed]["warm_prefix"][str(length)]
            candidate_condition_scores[seed][length] = {str(length): per_seed_scores[seed][str(length)]}
            if any(status != "success" for status in condition_statuses):
                blocked = True

    if blocked:
        return SelectionResult(
            status="blocked",
            winner=None,
            candidates=WARM_PREFIX_CANDIDATES,
            aggregate_scores=aggregate_scores,
            per_seed_scores=per_seed_scores,
            per_seed_family_scores=per_seed_family_scores,
            cells=tuple(cells),
            denominator=blocked_denominator,
            status_counts=_cell_status_counts(cells),
            metadata={"config_id": config_id},
        )

    for length in WARM_PREFIX_CANDIDATES:
        seed_means = [per_seed_scores[seed][str(length)] for seed in seeds]
        aggregate_scores[str(length)] = sum(seed_means) / len(seed_means) if seed_means else None
    winner = min(WARM_PREFIX_CANDIDATES, key=lambda length: (aggregate_scores[str(length)], length))
    return SelectionResult(
        status="selected",
        winner=str(winner),
        candidates=WARM_PREFIX_CANDIDATES,
        aggregate_scores=aggregate_scores,
        per_seed_scores=per_seed_scores,
        per_seed_family_scores=per_seed_family_scores,
        cells=tuple(cells),
        denominator=blocked_denominator,
        status_counts=_cell_status_counts(cells),
        metadata={"config_id": config_id},
    )


def select_lir_config(
    root: Path,
    seeds: Sequence[int],
    *,
    fixed_targets: Mapping[str, str],
    warm_prefix_length: int,
    split: str = "validation",
    candidate_runner: CandidateRunner = _candidate_runner_default,
    families: Sequence[str] = E4_FAMILIES,
    rhos: Sequence[str] = E4_RHO_GRID,
) -> SelectionResult:
    _fail_closed_for_final_test(split)
    if set(fixed_targets) != {"numerical", "categorical"}:
        raise ValueError("global fixed targets must be supplied explicitly")
    if warm_prefix_length not in WARM_PREFIX_CANDIDATES:
        raise ValueError("warm-prefix length must be one of 10, 20, or 30")
    if not seeds:
        raise ValueError("seed list must be supplied explicitly")

    configs = _candidate_configs(root)
    if len(configs) != 12:
        raise ValueError("expected exactly 12 frozen configs")

    per_seed_scores: dict[int, dict[str, float | None]] = {seed: {} for seed in seeds}
    per_seed_family_scores: dict[int, dict[str, dict[str, float | None]]] = {seed: {} for seed in seeds}
    cells: list[SelectionCell] = []
    blocked = False
    blocked_denominator = 0
    aggregate_scores: dict[str, float | None] = {config["config_id"]: None for config in configs}

    for seed in seeds:
        for family in families:
            per_seed_family_scores[seed][family] = {config["config_id"]: None for config in configs}
        for config in configs:
            config_id = config["config_id"]
            condition_losses: list[float] = []
            condition_statuses: list[str] = []
            for family in families:
                for rho in rhos:
                    attack_id = "fixed_target_numerical" if family.startswith("synthetic_num") else "fixed_target_categorical"
                    fixed_target_selection = fixed_targets["numerical"] if family.startswith("synthetic_num") else fixed_targets["categorical"]
                    loss, status, denominator, detail = _candidate_losses_for_condition(
                        root=root,
                        split=split,
                        seed=seed,
                        family=family,
                        rho=rho,
                        attack_id=attack_id,
                        fixed_target_selection=fixed_target_selection,
                        config_id=config_id,
                        warm_prefix_length=warm_prefix_length,
                        on_off_schedule=None,
                        candidate_runner=candidate_runner,
                    )
                    blocked_denominator += denominator
                    condition_statuses.append(status)
                    if loss is None:
                        blocked = True
                    else:
                        condition_losses.append(loss)
                    cells.append(
                        SelectionCell(
                            seed=seed,
                            family=family,
                            status=status,
                            denominator=denominator,
                            loss=loss,
                            detail={"config_id": config_id, **detail},
                        )
                    )
            per_seed_scores[seed][config_id] = sum(condition_losses) / len(condition_losses) if condition_losses else None
            for family in families:
                per_seed_family_scores[seed][family][config_id] = per_seed_scores[seed][config_id]
            if any(status != "success" for status in condition_statuses):
                blocked = True

    if blocked:
        return SelectionResult(
            status="blocked",
            winner=None,
            candidates=tuple(config["config_id"] for config in configs),
            aggregate_scores=aggregate_scores,
            per_seed_scores=per_seed_scores,
            per_seed_family_scores=per_seed_family_scores,
            cells=tuple(cells),
            denominator=blocked_denominator,
            status_counts=_cell_status_counts(cells),
            metadata={"warm_prefix_length": warm_prefix_length},
        )

    for config in configs:
        config_id = config["config_id"]
        seed_means = [per_seed_scores[seed][config_id] for seed in seeds]
        aggregate_scores[config_id] = sum(seed_means) / len(seed_means) if seed_means else None
    winner = min((config["config_id"] for config in configs), key=lambda config_id: (aggregate_scores[config_id], config_id))
    return SelectionResult(
        status="selected",
        winner=winner,
        candidates=tuple(config["config_id"] for config in configs),
        aggregate_scores=aggregate_scores,
        per_seed_scores=per_seed_scores,
        per_seed_family_scores=per_seed_family_scores,
        cells=tuple(cells),
        denominator=blocked_denominator,
        status_counts=_cell_status_counts(cells),
        metadata={"warm_prefix_length": warm_prefix_length},
    )


def build_e4_condition_matrix(
    *,
    fixed_targets: Mapping[str, str],
    warm_prefix_length: int,
    lir_config_id: str,
    seeds: Sequence[int],
    rhos: Sequence[str] = E4_RHO_GRID,
    schedules: Sequence[tuple[int, int]] = E4_ON_OFF_SCHEDULES,
    families: Sequence[str] = E4_FAMILIES,
    starts: Sequence[str] = E4_START_CONDITIONS,
    variants: Sequence[str] = E4_VARIANTS,
) -> list[dict[str, Any]]:
    if set(fixed_targets) != {"numerical", "categorical"}:
        raise ValueError("global fixed targets must be supplied explicitly")
    if warm_prefix_length not in WARM_PREFIX_CANDIDATES:
        raise ValueError("warm-prefix length must be one of 10, 20, or 30")
    if not lir_config_id:
        raise ValueError("lir config must be supplied explicitly")
    if not seeds:
        raise ValueError("seed list must be supplied explicitly")

    rows: list[dict[str, Any]] = []
    for family in families:
        for attack_class in E4_ATTACK_CLASSES:
            if attack_class == "fixed_target":
                attack_ids = ("fixed_target_numerical",) if family.startswith("synthetic_num") else ("fixed_target_categorical",)
                schedules_to_use = (None,)
            else:
                attack_ids = ("on_off",)
                schedules_to_use = schedules
            for attack_id in attack_ids:
                for rho in rhos:
                    for start in starts:
                        for variant in variants:
                            for schedule in schedules_to_use:
                                rows.append(
                                    {
                                        "family": family,
                                        "attack_class": attack_class,
                                        "attack_id": attack_id,
                                        "rho": rho,
                                        "start": start,
                                        "schedule": schedule,
                                        "variant": variant,
                                        "seed_count": len(seeds),
                                    }
                                )
    return rows


def _base_condition_key(
    *,
    seed: int,
    family: str,
    attack_class: str,
    attack_id: str,
    rho: str,
    start: str,
    schedule: tuple[int, int] | None,
) -> str:
    return f"seed={seed}|family={family}|attack_class={attack_class}|attack_id={attack_id}|rho={rho}|start={start}|schedule={schedule}"


def _load_base_condition(
    *,
    root: Path,
    split: str,
    seed: int,
    family: str,
    attack_id: str,
    rho: str,
    start: str,
    schedule: tuple[int, int] | None,
    fixed_targets: Mapping[str, str],
    warm_prefix_length: int,
    stream_loader: StreamLoader,
    ablation_sequence_runner: AblationSequenceRunner,
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_target_selection = fixed_targets["numerical"] if family.startswith("synthetic_num") else fixed_targets["categorical"]
    cold_stream = None
    warm_stream = None
    same_hash = False
    cold_stream = stream_loader(
        root,
        split,
        seed,
        family,
        rho=rho,
        attack_id=attack_id,
        fixed_target_selection=fixed_target_selection,
        warm_prefix_length=None,
        on_off_schedule=schedule,
    )
    warm_stream = stream_loader(
        root,
        split,
        seed,
        family,
        rho=rho,
        attack_id=attack_id,
        fixed_target_selection=fixed_target_selection,
        warm_prefix_length=warm_prefix_length,
        on_off_schedule=schedule,
    )
    same_hash = cold_stream["task_artifact_hashes"] == warm_stream["task_artifact_hashes"]
    canonical_scored_tasks = tuple(cold_stream["scored_tasks"])
    prefix_tasks = tuple(warm_stream.get("warm_prefix", {}).get("tasks", []))
    prefix_state = _warm_prefix_state(prefix_tasks, candidate_config, ablation_sequence_runner=ablation_sequence_runner)
    start_state = LongitudinalState() if start == "cold" else prefix_state
    return {
        "same_hash": same_hash,
        "canonical_scored_tasks": canonical_scored_tasks,
        "cold_stream": cold_stream,
        "warm_stream": warm_stream,
        "start_state": start_state,
        "prefix_state": prefix_state,
    }


def run_e4_matrix(
    root: Path,
    seeds: Sequence[int],
    *,
    fixed_targets: Mapping[str, str] | None = None,
    warm_prefix_length: int | None = None,
    lir_config_id: str | None = None,
    split: str = "validation",
    stream_loader: StreamLoader = _stream_loader_default,
    ablation_sequence_runner: AblationSequenceRunner = _ablation_sequence_runner_default,
    families: Sequence[str] = E4_FAMILIES,
    rhos: Sequence[str] = E4_RHO_GRID,
    schedules: Sequence[tuple[int, int]] = E4_ON_OFF_SCHEDULES,
    starts: Sequence[str] = E4_START_CONDITIONS,
    variants: Sequence[str] = E4_VARIANTS,
) -> E4MatrixResult:
    _fail_closed_for_final_test(split)
    if fixed_targets is None or warm_prefix_length is None or lir_config_id is None:
        raise ValueError("Stage A/B/C frozen selections are required")
    if set(fixed_targets) != {"numerical", "categorical"}:
        raise ValueError("global fixed targets must be supplied explicitly")
    if warm_prefix_length not in WARM_PREFIX_CANDIDATES:
        raise ValueError("warm-prefix length must be one of 10, 20, or 30")
    if lir_config_id not in _config_map(root):
        raise ValueError("unknown frozen config id")
    if not seeds:
        raise ValueError("seed list must be supplied explicitly")

    candidate_config = _config_map(root)[lir_config_id]
    condition_keys: list[str] = []
    cells: list[E4MatrixCell] = []
    run_count = 0
    base_denominator = 0
    cache: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        for family in families:
            for attack_class in E4_ATTACK_CLASSES:
                if attack_class == "fixed_target":
                    attack_ids = ("fixed_target_numerical",) if family.startswith("synthetic_num") else ("fixed_target_categorical",)
                    schedule_grid = (None,)
                else:
                    attack_ids = ("on_off",)
                    schedule_grid = schedules
                for attack_id in attack_ids:
                    for rho in rhos:
                        for start in starts:
                            for schedule in schedule_grid:
                                base_key = _base_condition_key(
                                    seed=seed,
                                    family=family,
                                    attack_class=attack_class,
                                    attack_id=attack_id,
                                    rho=rho,
                                    start=start,
                                    schedule=schedule,
                                )
                                if base_key not in cache:
                                    cache[base_key] = _load_base_condition(
                                        root=root,
                                        split=split,
                                        seed=seed,
                                        family=family,
                                        attack_id=attack_id,
                                        rho=rho,
                                        start=start,
                                        schedule=schedule,
                                        fixed_targets=fixed_targets,
                                        warm_prefix_length=warm_prefix_length,
                                        stream_loader=stream_loader,
                                        ablation_sequence_runner=ablation_sequence_runner,
                                        candidate_config=candidate_config,
                                    )
                                base = cache[base_key]
                                base_denominator += 1
                                condition_keys.append(base_key)
                                for variant in variants:
                                    same_artifact_reused = bool(base["same_hash"])
                                    canonical_scored_tasks = base["canonical_scored_tasks"]
                                    if not same_artifact_reused:
                                        primary_loss = None
                                        status = "failure"
                                        result_detail = {
                                            "error": "cold/warm scored suffix hash mismatch"
                                        }
                                    else:
                                        try:
                                            runner_result = ablation_sequence_runner(
                                                task_artifacts=canonical_scored_tasks,
                                                candidate_config=candidate_config,
                                                longitudinal_state=base["start_state"],
                                                ablation=variant,
                                            )
                                            results = list(runner_result.get("results", []))
                                            if len(results) != len(canonical_scored_tasks):
                                                raise ValueError("incomplete ablation execution")
                                            losses: list[float] = []
                                            for result, task_artifact in zip(results, canonical_scored_tasks):
                                                if not getattr(result, "success", True):
                                                    raise ValueError("ablation failure")
                                                if task_artifact["modality"] == "numerical":
                                                    losses.append(_numerical_l2(result.final_output, task_artifact["ground_truth"]))
                                                else:
                                                    losses.append(_categorical_disagreement(result.released_class_index, task_artifact["ground_truth"]))
                                            primary_loss = sum(losses) / len(losses) if losses else None
                                            status = "success"
                                            run_count += 1
                                        except Exception as exc:
                                            primary_loss = None
                                            status = "failure"
                                            result_detail = {"error": str(exc)}
                                        else:
                                            result_detail = {
                                                "variant": variant,
                                                "artifact_id": id(canonical_scored_tasks),
                                                "prefix_state": base["prefix_state"],
                                            }
                                    cells.append(
                                        E4MatrixCell(
                                            seed=seed,
                                            family=family,
                                            attack_class=attack_class,
                                            attack_id=attack_id,
                                            rho=rho,
                                            start=start,
                                            schedule=schedule,
                                            variant=variant,
                                            status=status,
                                            denominator=1,
                                            primary_loss=primary_loss,
                                            same_artifact_reused=same_artifact_reused,
                                            cold_warm_hash_equal=same_artifact_reused,
                                            scored_task_artifact_hashes=tuple(base["cold_stream"]["task_artifact_hashes"]),
                                            detail=result_detail,
                                        )
                                    )

    return E4MatrixResult(
        status="completed" if all(cell.status == "success" for cell in cells) else "failed",
        run_count=run_count,
        denominator=base_denominator,
        cells=tuple(cells),
        condition_keys=tuple(condition_keys),
        metadata={"fixed_targets": dict(fixed_targets), "warm_prefix_length": warm_prefix_length, "lir_config_id": lir_config_id},
    )
