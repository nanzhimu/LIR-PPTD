from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..baselines.mean_vote import predict as mean_vote_predict
from ..canonical import canonical_json_bytes
from .phase10_r22_metrics import (
    argmax_zero_based,
    categorical_primary_loss,
    numerical_abs_error_sum,
    numerical_mae_from_abs_error_sums,
)
from .phase6_ablation_runner import run_ablation_sequence
from .phase6_exact_adapter import LongitudinalState
from .phase6_generator import _build_scored_task, _build_warm_prefix

REVISION = "r2.2"
PRIMARY_COMPARATOR = "mean_vote"
FINAL_SEEDS = tuple(range(3001, 3031))
BENCHMARK_SEEDS = frozenset(range(1001, 1031)) | frozenset(range(2001, 2031))
FAMILIES = (
    "synthetic_num_small",
    "synthetic_num_shifted",
    "synthetic_num_longitudinal",
    "synthetic_cat_binary",
    "synthetic_cat_tie",
)
RHO_GRID = ("0", "1/10", "3/10", "1/2", "7/10", "9/10")
ON_OFF_SCHEDULES = ((5, 1), (5, 5), (1, 5))
STARTS = ("cold", "warm")
WARM_PREFIX_LENGTH = 10
SCORED_TASK_COUNT = 100
CONFIG_ID = "lir_cfg_05_lambda020"
FIXED_TARGETS = {"numerical": "all_one_vector", "categorical": "class_0"}
ATTACK_CONDITION_COUNT = 4
BASE_CONDITIONS_PER_SEED = len(FAMILIES) * ATTACK_CONDITION_COUNT * len(RHO_GRID) * len(STARTS)
FINAL_BASE_CONDITION_COUNT = len(FINAL_SEEDS) * BASE_CONDITIONS_PER_SEED
EXPENSIVE_SCORED_FULL_TASKS = FINAL_BASE_CONDITION_COUNT * SCORED_TASK_COUNT

R21_RESULT_RELATIVE_PATH = Path("results/summary/phase10_r21_final_test.json")
R21_RESULT_SHA256 = "114cef6e667f3e73a600fdfc78862ab107d1dd22cf2b527fd19a9b5aecc25c5c"
RAW_OUTPUT_RELATIVE_PATH = Path("results/raw/phase10_r22_corrective")
SUMMARY_OUTPUT_RELATIVE_PATH = Path("results/summary/phase10_r22_corrective.json")
GATE_RELATIVE_PATH = Path("docs/PHASE10_R22_CORRECTIVE_EXECUTION_GATE.json")
CONFIG_ARTIFACT_RELATIVE_PATH = Path("docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json")
RUNNER_RELATIVE_PATH = Path("src/lir_pptd/experiments/phase10_r22_corrective_runner.py")
METRICS_RELATIVE_PATH = Path("src/lir_pptd/experiments/phase10_r22_metrics.py")

# These bindings are already frozen and known. The corrective gate additionally
# binds the r2.2 runner, metrics module, and method-config artifact by SHA256.
IMPLEMENTATION_HASHES = {
    "src/lir_pptd/experiments/phase6_ablation_runner.py": "40561ac46a4d8d42171caa13aa2dff035b1df2a0aaa2e2305c8a51f4ed2dc87c",
    "src/lir_pptd/experiments/phase6_generator.py": "1950c3dee2d2a322c69bf0345cd336c3b2cde3d0f09e1ce88963b2773d0962d3",
    "src/lir_pptd/baselines/mean_vote.py": "76d8921882fbf7ae838a37ec6d3a0d9f59c9df823358153c3b77e171e396c87e",
}

REQUIRED_GATE_BOUND_FILES = (
    str(RUNNER_RELATIVE_PATH).replace("\\", "/"),
    str(METRICS_RELATIVE_PATH).replace("\\", "/"),
    "src/lir_pptd/experiments/phase6_ablation_runner.py",
    "src/lir_pptd/experiments/phase6_generator.py",
    "src/lir_pptd/baselines/mean_vote.py",
    str(CONFIG_ARTIFACT_RELATIVE_PATH).replace("\\", "/"),
)


class CorrectiveRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorrectivePlan:
    revision: str = REVISION
    primary_comparator: str = PRIMARY_COMPARATOR
    final_seed_count: int = len(FINAL_SEEDS)
    family_count: int = len(FAMILIES)
    attack_condition_count: int = ATTACK_CONDITION_COUNT
    rho_count: int = len(RHO_GRID)
    start_count: int = len(STARTS)
    base_condition_count: int = FINAL_BASE_CONDITION_COUNT
    full_sequence_count: int = FINAL_BASE_CONDITION_COUNT
    scored_tasks_per_sequence: int = SCORED_TASK_COUNT
    expensive_scored_full_tasks: int = EXPENSIVE_SCORED_FULL_TASKS
    final_arithmetic_executed: bool = False
    result_written: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_seed: int
    base_condition_count: int
    full_sequence_count: int
    scored_full_task_count: int
    elapsed_seconds: float
    seconds_per_full_sequence: float
    projected_30_seed_hours: float
    config_artifact_sha256: str
    runner_sha256: str
    metrics_sha256: str


@dataclass(frozen=True)
class CorrectiveCondition:
    seed: int
    family: str
    attack_class: str
    attack_id: str
    schedule: tuple[int, int] | None
    rho: str
    start: str
    status: str
    error: str | None
    scored_task_count: int
    task_hashes: tuple[str, ...]
    cold_warm_hash_equal: bool
    full_primary_loss: float | None
    mean_vote_primary_loss: float | None
    delta: float | None
    task_evidence: tuple[dict[str, Any], ...]


StreamLoader = Callable[..., dict[str, Any]]
WarmPrefixLoader = Callable[..., dict[str, Any] | None]
AblationRunner = Callable[..., dict[str, Any]]
CandidateConfigLoader = Callable[[Path, str], dict[str, Any]]


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise CorrectiveRunnerError(f"BOUND_FILE_MISSING:{path.as_posix()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_implementation_bindings(root: Path) -> None:
    for relative_path, expected in IMPLEMENTATION_HASHES.items():
        actual = _sha256(root / relative_path)
        if actual != expected:
            label = "MEAN_VOTE_SHA_MISMATCH" if relative_path.endswith("mean_vote.py") else "IMPLEMENTATION_BINDING_MISMATCH"
            raise CorrectiveRunnerError(f"{label}:{relative_path}")


def _load_candidate_config(root: Path, config_id: str) -> dict[str, Any]:
    """Load the selected config directly from the frozen decision artifact.

    This intentionally avoids importing the drifted Phase6 candidate-runner
    module. Formal execution
    binds this artifact itself in the r2.2 execution gate.
    """
    path = root / CONFIG_ARTIFACT_RELATIVE_PATH
    if not path.is_file():
        raise CorrectiveRunnerError("METHOD_CONFIG_ARTIFACT_MISSING")
    decision = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = [
        candidate
        for candidate in decision.get("candidate_configs", [])
        if candidate.get("config_id") == config_id
    ]
    if len(candidates) != 1:
        raise CorrectiveRunnerError("FROZEN_METHOD_CONFIG_NOT_UNIQUE")
    return dict(candidates[0])


def verify_task_hash_binding(expected: Sequence[str], actual: Sequence[str]) -> None:
    if tuple(expected) != tuple(actual):
        raise CorrectiveRunnerError("FINAL_TASK_HASH_BINDING_MISMATCH")


def _attack_specs(family: str) -> tuple[tuple[str, str, tuple[int, int] | None], ...]:
    fixed_attack = "fixed_target_numerical" if family.startswith("synthetic_num") else "fixed_target_categorical"
    return (("fixed_target", fixed_attack, None),) + tuple(
        ("on_off", "on_off", schedule) for schedule in ON_OFF_SCHEDULES
    )


def _condition_key(
    seed: int,
    family: str,
    attack_class: str,
    schedule: tuple[int, int] | None,
    rho: str,
    start: str,
) -> tuple[Any, ...]:
    return seed, family, attack_class, schedule, rho, start


def _plan_lines(plan: CorrectivePlan) -> str:
    return "\n".join(
        (
            f"REVISION={plan.revision}",
            f"PRIMARY_COMPARATOR={plan.primary_comparator}",
            f"FINAL_SEED_COUNT={plan.final_seed_count}",
            f"FAMILY_COUNT={plan.family_count}",
            f"ATTACK_CONDITION_COUNT={plan.attack_condition_count}",
            f"RHO_COUNT={plan.rho_count}",
            f"START_COUNT={plan.start_count}",
            f"BASE_CONDITION_COUNT={plan.base_condition_count}",
            f"FULL_SEQUENCE_COUNT={plan.full_sequence_count}",
            f"SCORED_TASKS_PER_SEQUENCE={plan.scored_tasks_per_sequence}",
            f"EXPENSIVE_SCORED_FULL_TASKS={plan.expensive_scored_full_tasks}",
            "FINAL_ARITHMETIC_EXECUTED=false",
            "RESULT_WRITTEN=false",
        )
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _technical_freeze_commit(gate: Mapping[str, Any]) -> str:
    direct = gate.get("technical_freeze_commit")
    if isinstance(direct, str):
        value = direct
    else:
        value = gate.get("bound_technical_freeze", {}).get("commit", "")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise CorrectiveRunnerError("CORRECTIVE_EXECUTION_NOT_AUTHORIZED")
    return value.lower()


class CorrectiveRunner:
    def __init__(
        self,
        root: Path,
        *,
        scored_task_loader: StreamLoader = _build_scored_task,
        warm_prefix_loader: WarmPrefixLoader = _build_warm_prefix,
        ablation_runner: AblationRunner = run_ablation_sequence,
        mean_vote_runner: Callable[[dict[str, Sequence[float | int]], str], list[float | int]] = mean_vote_predict,
        candidate_config_loader: CandidateConfigLoader = _load_candidate_config,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.root = root
        self.scored_task_loader = scored_task_loader
        self.warm_prefix_loader = warm_prefix_loader
        self.ablation_runner = ablation_runner
        self.mean_vote_runner = mean_vote_runner
        self.candidate_config_loader = candidate_config_loader
        self.clock = clock
        self.materialized_task_count = 0

    def plan(self) -> CorrectivePlan:
        return CorrectivePlan()

    def _current_gate_bindings(self) -> dict[str, str]:
        return {path: _sha256(self.root / path) for path in REQUIRED_GATE_BOUND_FILES}

    def _load_and_verify_formal_gate(self) -> dict[str, Any]:
        gate_path = self.root / GATE_RELATIVE_PATH
        if not gate_path.is_file():
            raise CorrectiveRunnerError("CORRECTIVE_EXECUTION_NOT_AUTHORIZED")
        gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        if gate.get("authorized") is not True:
            raise CorrectiveRunnerError("CORRECTIVE_EXECUTION_NOT_AUTHORIZED")
        if gate.get("bound_r21_result", {}).get("sha256") != R21_RESULT_SHA256:
            raise CorrectiveRunnerError("CORRECTIVE_EXECUTION_NOT_AUTHORIZED")
        _technical_freeze_commit(gate)
        bound_files = gate.get("bound_files")
        if not isinstance(bound_files, Mapping):
            raise CorrectiveRunnerError("CORRECTIVE_EXECUTION_NOT_AUTHORIZED")
        current = self._current_gate_bindings()
        for relative_path, actual_sha in current.items():
            if bound_files.get(relative_path) != actual_sha:
                raise CorrectiveRunnerError(f"CORRECTIVE_GATE_BINDING_MISMATCH:{relative_path}")
        # Also retain the independently frozen known hashes.
        verify_implementation_bindings(self.root)
        return dict(gate)

    def _assert_new_execution_paths(self) -> None:
        summary_path = self.root / SUMMARY_OUTPUT_RELATIVE_PATH
        raw_dir = self.root / RAW_OUTPUT_RELATIVE_PATH
        if summary_path.exists():
            raise CorrectiveRunnerError("CORRECTIVE_RESULT_ALREADY_EXISTS")
        # Even an empty raw directory is rejected for a fresh run so a run can
        # never finish hours later and fail only at mkdir(exist_ok=False).
        if raw_dir.exists():
            raise CorrectiveRunnerError("CORRECTIVE_RAW_ALREADY_EXISTS")

    def _assert_resume_paths(self) -> None:
        summary_path = self.root / SUMMARY_OUTPUT_RELATIVE_PATH
        raw_dir = self.root / RAW_OUTPUT_RELATIVE_PATH
        if summary_path.exists():
            raise CorrectiveRunnerError("CORRECTIVE_RESULT_ALREADY_EXISTS")
        if not raw_dir.is_dir():
            raise CorrectiveRunnerError("CORRECTIVE_RESUME_STATE_MISSING")

    def _build_scored_tasks(
        self,
        seed: int,
        family: str,
        rho: str,
        attack_id: str,
        schedule: tuple[int, int] | None,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        target = FIXED_TARGETS["numerical"] if family.startswith("synthetic_num") else FIXED_TARGETS["categorical"]
        tasks = tuple(
            self.scored_task_loader(
                seed,
                family,
                task_index,
                rho,
                attack_id,
                fixed_target_selection=target,
                on_off_schedule=schedule,
            )
            for task_index in range(1, SCORED_TASK_COUNT + 1)
        )
        self.materialized_task_count += len(tasks)
        hashes = tuple(hashlib.sha256(canonical_json_bytes(task)).hexdigest() for task in tasks)
        return tasks, hashes

    def _warm_state(self, seed: int, family: str, config: Mapping[str, Any]) -> LongitudinalState:
        prefix = self.warm_prefix_loader(seed, family, WARM_PREFIX_LENGTH)
        prefix_tasks = tuple(prefix["tasks"]) if prefix else tuple()
        if not prefix_tasks:
            return LongitudinalState()
        result = self.ablation_runner(
            task_artifacts=prefix_tasks,
            candidate_config=config,
            longitudinal_state=LongitudinalState(),
            ablation="full",
        )
        prefix_results = tuple(result.get("results", ()))
        if len(prefix_results) != len(prefix_tasks):
            raise CorrectiveRunnerError("INCOMPLETE_WARM_PREFIX")
        if any(not getattr(item, "success", True) for item in prefix_results):
            raise CorrectiveRunnerError("WARM_PREFIX_TASK_FAILURE")
        return result.get("final_state", LongitudinalState())

    def _condition_metrics(
        self,
        family: str,
        tasks: Sequence[Mapping[str, Any]],
        results: Sequence[Any],
        task_hashes: Sequence[str],
    ) -> tuple[float, float, tuple[dict[str, Any], ...]]:
        if len(results) != len(tasks):
            raise CorrectiveRunnerError("INCOMPLETE_FULL_SEQUENCE")
        full_errors: list[Any] = []
        mean_errors: list[Any] = []
        dimensions: list[int] = []
        truths: list[int] = []
        full_predictions: list[int] = []
        mean_predictions: list[int] = []
        evidence: list[dict[str, Any]] = []
        for task_index, (task, result, task_hash) in enumerate(zip(tasks, results, task_hashes), start=1):
            if not getattr(result, "success", True):
                raise CorrectiveRunnerError("FULL_TASK_FAILURE")
            full_output = tuple(result.final_output)
            mean_output = tuple(self.mean_vote_runner(task["reports"], task["modality"]))
            if task["modality"] == "numerical":
                full_error = numerical_abs_error_sum(full_output, task["ground_truth"])
                mean_error = numerical_abs_error_sum(mean_output, task["ground_truth"])
                dimension = len(task["ground_truth"])
                full_errors.append(full_error)
                mean_errors.append(mean_error)
                dimensions.append(dimension)
                evidence.append(
                    {
                        "task_index": task_index,
                        "task_hash": task_hash,
                        "dimension": dimension,
                        "full_abs_error_sum": str(full_error),
                        "mean_vote_abs_error_sum": str(mean_error),
                    }
                )
            else:
                truth_class = argmax_zero_based(task["ground_truth"])
                full_prediction = argmax_zero_based(full_output)
                mean_prediction = argmax_zero_based(mean_output)
                truths.append(truth_class)
                full_predictions.append(full_prediction)
                mean_predictions.append(mean_prediction)
                evidence.append(
                    {
                        "task_index": task_index,
                        "task_hash": task_hash,
                        "truth_class": truth_class,
                        "full_prediction": full_prediction,
                        "mean_vote_prediction": mean_prediction,
                    }
                )
        if family.startswith("synthetic_num"):
            full_mae = numerical_mae_from_abs_error_sums(full_errors, dimensions)
            mean_mae = numerical_mae_from_abs_error_sums(mean_errors, dimensions)
            return float(full_mae), float(mean_mae), tuple(evidence)
        return (
            categorical_primary_loss(family, full_predictions, truths),
            categorical_primary_loss(family, mean_predictions, truths),
            tuple(evidence),
        )

    def _execute_start(
        self,
        *,
        seed: int,
        family: str,
        attack_class: str,
        attack_id: str,
        schedule: tuple[int, int] | None,
        rho: str,
        start: str,
        tasks: Sequence[Mapping[str, Any]],
        task_hashes: tuple[str, ...],
        config: Mapping[str, Any],
    ) -> CorrectiveCondition:
        status = "success"
        error = None
        full_loss = None
        mean_loss = None
        delta = None
        evidence: tuple[dict[str, Any], ...] = tuple()
        try:
            state = LongitudinalState() if start == "cold" else self._warm_state(seed, family, config)
            sequence = self.ablation_runner(
                task_artifacts=tasks,
                candidate_config=config,
                longitudinal_state=state,
                ablation="full",
            )
            results = tuple(sequence.get("results", ()))
            full_loss, mean_loss, evidence = self._condition_metrics(family, tasks, results, task_hashes)
            delta = full_loss - mean_loss
        except TimeoutError as exc:
            status = "timeout"
            error = str(exc)
        except Exception as exc:
            status = "failure"
            error = str(exc)
        return CorrectiveCondition(
            seed=seed,
            family=family,
            attack_class=attack_class,
            attack_id=attack_id,
            schedule=schedule,
            rho=rho,
            start=start,
            status=status,
            error=error,
            scored_task_count=len(tasks),
            task_hashes=task_hashes,
            cold_warm_hash_equal=True,
            full_primary_loss=full_loss,
            mean_vote_primary_loss=mean_loss,
            delta=delta,
            task_evidence=evidence,
        )

    def _execute_seed(
        self,
        seed: int,
        *,
        expected_hashes: Mapping[tuple[Any, ...], tuple[str, ...]] | None,
    ) -> tuple[CorrectiveCondition, ...]:
        config = self.candidate_config_loader(self.root, CONFIG_ID)
        conditions: list[CorrectiveCondition] = []
        for family in FAMILIES:
            for attack_class, attack_id, schedule in _attack_specs(family):
                for rho in RHO_GRID:
                    tasks, task_hashes = self._build_scored_tasks(seed, family, rho, attack_id, schedule)

                    # Formal task identity is checked before *any* warm-prefix or
                    # scored Full arithmetic for this base condition.
                    if expected_hashes is not None:
                        for start in STARTS:
                            key = _condition_key(seed, family, attack_class, schedule, rho, start)
                            if key not in expected_hashes:
                                raise CorrectiveRunnerError("FINAL_TASK_HASH_BINDING_MISMATCH")
                            verify_task_hash_binding(expected_hashes[key], task_hashes)
                    verify_task_hash_binding(task_hashes, task_hashes)

                    # Cold and warm are retained independently. A warm-prefix
                    # failure cannot erase the cold condition from the denominator.
                    for start in STARTS:
                        conditions.append(
                            self._execute_start(
                                seed=seed,
                                family=family,
                                attack_class=attack_class,
                                attack_id=attack_id,
                                schedule=schedule,
                                rho=rho,
                                start=start,
                                tasks=tasks,
                                task_hashes=task_hashes,
                                config=config,
                            )
                        )
        return tuple(conditions)

    def benchmark(self, seed: int) -> BenchmarkResult:
        if seed not in BENCHMARK_SEEDS or seed in FINAL_SEEDS:
            raise CorrectiveRunnerError("BENCHMARK_SEED_NOT_ALLOWED")
        verify_implementation_bindings(self.root)
        # Fail early if the direct frozen config loader cannot resolve the selected config.
        self.candidate_config_loader(self.root, CONFIG_ID)
        started = self.clock()
        conditions = self._execute_seed(seed, expected_hashes=None)
        elapsed = self.clock() - started
        count = len(conditions)
        if count != BASE_CONDITIONS_PER_SEED:
            raise CorrectiveRunnerError("BENCHMARK_DENOMINATOR_MISMATCH")
        return BenchmarkResult(
            benchmark_seed=seed,
            base_condition_count=count,
            full_sequence_count=count,
            scored_full_task_count=count * SCORED_TASK_COUNT,
            elapsed_seconds=elapsed,
            seconds_per_full_sequence=elapsed / count,
            projected_30_seed_hours=elapsed * len(FINAL_SEEDS) / 3600.0,
            config_artifact_sha256=_sha256(self.root / CONFIG_ARTIFACT_RELATIVE_PATH),
            runner_sha256=_sha256(self.root / RUNNER_RELATIVE_PATH),
            metrics_sha256=_sha256(self.root / METRICS_RELATIVE_PATH),
        )

    def _load_r21_hashes(self) -> dict[tuple[Any, ...], tuple[str, ...]]:
        path = self.root / R21_RESULT_RELATIVE_PATH
        if _sha256(path) != R21_RESULT_SHA256:
            raise CorrectiveRunnerError("R21_RESULT_SHA_MISMATCH")
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        hashes: dict[tuple[Any, ...], tuple[str, ...]] = {}
        for cell in result["cells"]:
            if cell["variant"] != "full":
                continue
            schedule = tuple(cell["schedule"]) if cell["schedule"] is not None else None
            key = _condition_key(
                cell["seed"],
                cell["family"],
                cell["attack_class"],
                schedule,
                cell["rho"],
                cell["start"],
            )
            if key in hashes:
                raise CorrectiveRunnerError("R21_FULL_CELL_BINDING_DUPLICATE")
            hashes[key] = tuple(cell["scored_task_hashes"])
        if len(hashes) != FINAL_BASE_CONDITION_COUNT:
            raise CorrectiveRunnerError("R21_FULL_CELL_BINDING_INCOMPLETE")
        return hashes

    def _checkpoint_path(self, seed: int) -> Path:
        return self.root / RAW_OUTPUT_RELATIVE_PATH / f"seed-{seed}.json"

    def _checkpoint_tmp_path(self, seed: int) -> Path:
        return self._checkpoint_path(seed).with_name(f"seed-{seed}.json.tmp")

    def _checkpoint_payload(
        self,
        seed: int,
        conditions: Sequence[CorrectiveCondition],
        gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": "phase10-r22-seed-checkpoint-v1",
            "revision": REVISION,
            "seed": seed,
            "r21_result_sha256": R21_RESULT_SHA256,
            "technical_freeze_commit": _technical_freeze_commit(gate),
            "bound_files": dict(gate["bound_files"]),
            "condition_count": len(conditions),
            "conditions": [asdict(condition) for condition in conditions],
        }

    def _validate_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        seed: int,
        expected_hashes: Mapping[tuple[Any, ...], tuple[str, ...]],
        gate: Mapping[str, Any],
    ) -> tuple[CorrectiveCondition, ...]:
        if payload.get("schema_version") != "phase10-r22-seed-checkpoint-v1":
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
        if payload.get("revision") != REVISION or payload.get("seed") != seed:
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
        if payload.get("r21_result_sha256") != R21_RESULT_SHA256:
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
        if payload.get("technical_freeze_commit") != _technical_freeze_commit(gate):
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_BINDING_MISMATCH")
        if payload.get("bound_files") != gate.get("bound_files"):
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_BINDING_MISMATCH")
        raw_conditions = payload.get("conditions")
        if not isinstance(raw_conditions, list) or len(raw_conditions) != BASE_CONDITIONS_PER_SEED:
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_DENOMINATOR_MISMATCH")
        conditions: list[CorrectiveCondition] = []
        seen: set[tuple[Any, ...]] = set()
        for raw in raw_conditions:
            if not isinstance(raw, Mapping):
                raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
            schedule_raw = raw.get("schedule")
            schedule = tuple(schedule_raw) if schedule_raw is not None else None
            task_hashes = tuple(raw.get("task_hashes", ()))
            evidence_raw = raw.get("task_evidence", ())
            evidence = tuple(evidence_raw) if isinstance(evidence_raw, list) else tuple(evidence_raw)
            condition = CorrectiveCondition(
                seed=int(raw["seed"]),
                family=str(raw["family"]),
                attack_class=str(raw["attack_class"]),
                attack_id=str(raw["attack_id"]),
                schedule=schedule,
                rho=str(raw["rho"]),
                start=str(raw["start"]),
                status=str(raw["status"]),
                error=raw.get("error"),
                scored_task_count=int(raw["scored_task_count"]),
                task_hashes=task_hashes,
                cold_warm_hash_equal=bool(raw["cold_warm_hash_equal"]),
                full_primary_loss=raw.get("full_primary_loss"),
                mean_vote_primary_loss=raw.get("mean_vote_primary_loss"),
                delta=raw.get("delta"),
                task_evidence=evidence,
            )
            if condition.seed != seed or condition.start not in STARTS:
                raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
            if condition.status not in {"success", "failure", "timeout"}:
                raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
            if condition.scored_task_count != SCORED_TASK_COUNT or not condition.cold_warm_hash_equal:
                raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
            if condition.status == "success":
                if len(condition.task_evidence) != SCORED_TASK_COUNT:
                    raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_EVIDENCE_INCOMPLETE")
                evidence_hashes = tuple(str(item.get("task_hash", "")) for item in condition.task_evidence)
                verify_task_hash_binding(condition.task_hashes, evidence_hashes)
            key = _condition_key(
                condition.seed,
                condition.family,
                condition.attack_class,
                condition.schedule,
                condition.rho,
                condition.start,
            )
            if key in seen or key not in expected_hashes:
                raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_INVALID")
            verify_task_hash_binding(expected_hashes[key], condition.task_hashes)
            seen.add(key)
            conditions.append(condition)
        expected_seed_keys = {key for key in expected_hashes if key[0] == seed}
        if seen != expected_seed_keys:
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_DENOMINATOR_MISMATCH")
        return tuple(conditions)

    def _load_checkpoint(
        self,
        seed: int,
        *,
        expected_hashes: Mapping[tuple[Any, ...], tuple[str, ...]],
        gate: Mapping[str, Any],
    ) -> tuple[CorrectiveCondition, ...]:
        path = self._checkpoint_path(seed)
        if not path.is_file():
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_MISSING")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._validate_checkpoint(payload, seed=seed, expected_hashes=expected_hashes, gate=gate)

    def _write_seed_checkpoint(
        self,
        seed: int,
        conditions: Sequence[CorrectiveCondition],
        gate: Mapping[str, Any],
    ) -> None:
        path = self._checkpoint_path(seed)
        if path.exists():
            raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_ALREADY_EXISTS")
        _atomic_write_json(path, self._checkpoint_payload(seed, conditions, gate))

    def _build_summary(self, conditions: Sequence[CorrectiveCondition], gate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "revision": REVISION,
            "primary_comparator": PRIMARY_COMPARATOR,
            "status": "completed" if all(condition.status == "success" for condition in conditions) else "completed_with_retained_failures",
            "denominator": len(conditions),
            "status_counts": {
                status: sum(condition.status == status for condition in conditions)
                for status in ("success", "failure", "timeout")
            },
            "r21_result_sha256": R21_RESULT_SHA256,
            "technical_freeze_commit": _technical_freeze_commit(gate),
            "bound_files": dict(gate["bound_files"]),
            "raw_directory": str(RAW_OUTPUT_RELATIVE_PATH).replace("\\", "/"),
            "checkpoint_count": len(FINAL_SEEDS),
            "conditions": [
                {key: value for key, value in asdict(condition).items() if key != "task_evidence"}
                for condition in conditions
            ],
        }

    def _run_formal(self, *, resume: bool) -> dict[str, Any]:
        gate = self._load_and_verify_formal_gate()
        if resume:
            self._assert_resume_paths()
        else:
            self._assert_new_execution_paths()
            (self.root / RAW_OUTPUT_RELATIVE_PATH).mkdir(parents=True, exist_ok=False)

        expected_hashes = self._load_r21_hashes()
        all_conditions: list[CorrectiveCondition] = []
        for seed in FINAL_SEEDS:
            checkpoint = self._checkpoint_path(seed)
            tmp_checkpoint = self._checkpoint_tmp_path(seed)
            if checkpoint.exists():
                if not resume:
                    raise CorrectiveRunnerError("CORRECTIVE_CHECKPOINT_ALREADY_EXISTS")
                conditions = self._load_checkpoint(seed, expected_hashes=expected_hashes, gate=gate)
            else:
                # A stale .tmp can only be an interrupted, never-accepted atomic write.
                # Removing it does not retry an accepted experimental condition.
                if tmp_checkpoint.exists():
                    tmp_checkpoint.unlink()
                conditions = self._execute_seed(seed, expected_hashes=expected_hashes)
                if len(conditions) != BASE_CONDITIONS_PER_SEED:
                    raise CorrectiveRunnerError("CORRECTIVE_SEED_DENOMINATOR_MISMATCH")
                self._write_seed_checkpoint(seed, conditions, gate)
                # Immediately read back and validate the durable checkpoint before continuing.
                conditions = self._load_checkpoint(seed, expected_hashes=expected_hashes, gate=gate)
            all_conditions.extend(conditions)

        if len(all_conditions) != FINAL_BASE_CONDITION_COUNT:
            raise CorrectiveRunnerError("CORRECTIVE_DENOMINATOR_MISMATCH")
        summary = self._build_summary(all_conditions, gate)
        _atomic_write_json(self.root / SUMMARY_OUTPUT_RELATIVE_PATH, summary)
        return summary

    def execute_formal(self) -> dict[str, Any]:
        return self._run_formal(resume=False)

    def resume_formal(self) -> dict[str, Any]:
        return self._run_formal(resume=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase10 r2.2 corrective recomputation runner")
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--benchmark-seed", type=int)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--resume", action="store_true")
    return parser


def _benchmark_lines(result: BenchmarkResult) -> str:
    return "\n".join(f"{key.upper()}={value}" for key, value in asdict(result).items())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = CorrectiveRunner(args.root)
    if args.plan_only:
        print(_plan_lines(runner.plan()))
        return 0
    if args.benchmark_seed is not None:
        print(_benchmark_lines(runner.benchmark(args.benchmark_seed)))
        return 0
    if args.resume:
        runner.resume_formal()
        return 0
    runner.execute_formal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
