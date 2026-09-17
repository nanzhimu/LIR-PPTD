from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phase6_ablation_runner import run_ablation_sequence
from .phase6_candidate_runner import _candidate_config
from .phase6_generator import _build_scored_task, _build_warm_prefix
from .phase6_exact_adapter import LongitudinalState

FINAL_TEST_RAW_DIR = Path("results/raw/phase10_r21_final_test")
FINAL_TEST_SUMMARY_PATH = Path("results/summary/phase10_r21_final_test.json")
FINAL_TEST_SPLIT = "final_test"
FINAL_TEST_CONFIG_ID = "lir_cfg_05_lambda020"


@dataclass(frozen=True)
class FinalTestPlan:
    manifest_sha256: str
    gate_status: str
    n_final: int
    base_condition_count: int
    expected_result_cell_count: int
    task_families: tuple[str, ...]
    attack_classes: tuple[str, ...]
    rhos: tuple[str, ...]
    schedules: tuple[tuple[int, int], ...]
    starts: tuple[str, ...]
    variants: tuple[str, ...]


@dataclass(frozen=True)
class FinalTestCell:
    seed: int
    family: str
    rho: str
    attack_class: str
    attack_id: str
    schedule: tuple[int, int] | None
    start: str
    variant: str
    status: str
    primary_loss: float | None
    output: tuple[str, ...] | None
    released_class_index: int | None
    scored_task_hashes: tuple[str, ...]
    cold_warm_hash_equal: bool
    error: str | None = None


@dataclass(frozen=True)
class FinalTestResult:
    plan: FinalTestPlan
    raw_directory: str
    summary_path: str
    cells: tuple[FinalTestCell, ...]
    run_count: int
    denominator: int
    manifest_sha256: str
    gate_sha256: str
    authorized: bool


class FinalTestAuthorizationError(RuntimeError):
    pass


class FinalTestMaterializer:
    def __init__(
        self,
        root: Path,
        *,
        manifest_path: Path | None = None,
        gate_path: Path | None = None,
        candidate_config_loader=_candidate_config,
        stream_loader=_build_scored_task,
        warm_prefix_loader=_build_warm_prefix,
        ablation_sequence_runner=run_ablation_sequence,
    ) -> None:
        self.root = root
        self.manifest_path = manifest_path or root / "configs/frozen/phase10_r21_final_test_manifest.json"
        self.gate_path = gate_path or root / "docs/PHASE10_R21_FINAL_EXECUTION_GATE.json"
        self.candidate_config_loader = candidate_config_loader
        self.stream_loader = stream_loader
        self.warm_prefix_loader = warm_prefix_loader
        self.ablation_sequence_runner = ablation_sequence_runner
        self._materialized = False

    def load_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))

    def load_gate(self) -> dict[str, Any]:
        return json.loads(self.gate_path.read_text(encoding="utf-8-sig"))

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def gate_sha256(self) -> str:
        return hashlib.sha256(self.gate_path.read_bytes()).hexdigest()

    def _manifest_frozen_selection(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        selection = manifest.get("frozen_selection")
        if not isinstance(selection, Mapping):
            raise FinalTestAuthorizationError("manifest frozen_selection missing")
        return dict(selection)

    def _bound_manifest_sha256(self, gate: Mapping[str, Any]) -> str:
        bound_manifest = gate.get("bound_manifest")
        if not isinstance(bound_manifest, Mapping) or "sha256" not in bound_manifest:
            raise FinalTestAuthorizationError("gate bound_manifest.sha256 missing")
        return str(bound_manifest["sha256"])

    def assert_authorized(self) -> None:
        gate = self.load_gate()
        manifest = self.load_manifest()
        if gate.get("status") != "PASS":
            raise FinalTestAuthorizationError("gate status is not PASS")
        if manifest.get("schema_version") != "1.0":
            raise FinalTestAuthorizationError("manifest schema invalid")
        if self._bound_manifest_sha256(gate) != self.manifest_sha256():
            raise FinalTestAuthorizationError("bound manifest sha256 mismatch")
        if self._manifest_frozen_selection(manifest) != dict(gate.get("frozen_selection", {})):
            raise FinalTestAuthorizationError("gate selection does not match manifest")

    def plan(self) -> FinalTestPlan:
        manifest = self.load_manifest()
        split_contract = manifest["split_contract"]
        execution_contract = manifest["execution_contract"]
        attack_contract = manifest["attack_contract"]
        task_contract = manifest["task_contract"]
        return FinalTestPlan(
            manifest_sha256=self.manifest_sha256(),
            gate_status="NOT_REQUIRED",
            n_final=split_contract["N_final"],
            base_condition_count=execution_contract["base_condition_count"],
            expected_result_cell_count=execution_contract["expected_result_cell_count"],
            task_families=tuple(task_contract["accuracy_families"]),
            attack_classes=tuple(attack_contract["classes"]),
            rhos=tuple(attack_contract["rho_grid"]),
            schedules=tuple(tuple(schedule) for schedule in attack_contract["on_off_schedules"]),
            starts=tuple(execution_contract["start_conditions"]),
            variants=tuple(execution_contract["variants"]),
        )

    def _resolve_output_path(self, value: Any) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _task_hash(self, task: Mapping[str, Any]) -> str:
        return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _build_scored_stream(
        self,
        *,
        seed: int,
        family: str,
        rho: str,
        attack_id: str,
        fixed_target_selection: str,
        schedule: tuple[int, int] | None,
        scored_task_count: int,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        scored_tasks = tuple(
            _build_scored_task(
                seed,
                family,
                task_index,
                rho,
                attack_id,
                fixed_target_selection=fixed_target_selection,
                on_off_schedule=schedule,
            )
            for task_index in range(1, scored_task_count + 1)
        )
        return scored_tasks, tuple(self._task_hash(task) for task in scored_tasks)

    def _build_prefix_stream(
        self,
        *,
        seed: int,
        family: str,
        warm_prefix_length: int,
    ) -> tuple[dict[str, Any], ...]:
        prefix = _build_warm_prefix(seed, family, warm_prefix_length)
        if prefix is None:
            return tuple()
        return tuple(prefix["tasks"])

    def _primary_loss_from_sequence(
        self,
        results: Sequence[Any],
        task_artifacts: Sequence[Mapping[str, Any]],
    ) -> float:
        if len(results) != len(task_artifacts):
            raise RuntimeError("incomplete ablation result set")
        losses: list[float] = []
        for result, task_artifact in zip(results, task_artifacts):
            if not getattr(result, "success", True):
                raise RuntimeError("ablation failure")
            if task_artifact["modality"] == "numerical":
                loss = sum((float(output) - float(truth)) ** 2 for output, truth in zip(result.final_output, task_artifact["ground_truth"])) ** 0.5
            else:
                truth_index = 0 if float(task_artifact["ground_truth"][0]) >= float(task_artifact["ground_truth"][1]) else 1
                predicted_index = getattr(result, "released_class_index", None)
                loss = 0.0 if predicted_index == truth_index else 1.0
            losses.append(loss)
        return sum(losses) / len(losses) if losses else 0.0

    def _output_paths(self, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
        formal_output = manifest.get("formal_output")
        if not isinstance(formal_output, Mapping):
            raise FinalTestAuthorizationError("manifest formal_output missing")
        raw_dir = self._resolve_output_path(formal_output["raw_directory"])
        summary_path = self._resolve_output_path(formal_output["summary"])
        return raw_dir, summary_path

    def materialize(self) -> FinalTestResult:
        self.assert_authorized()
        if self._materialized:
            raise FinalTestAuthorizationError("write-once final test materialization already completed")

        manifest = self.load_manifest()
        raw_dir, summary_path = self._output_paths(manifest)
        if raw_dir.exists() or summary_path.exists():
            raise FinalTestAuthorizationError("formal final-test outputs already exist")

        selection = self._manifest_frozen_selection(manifest)
        fixed_targets = selection["fixed_targets"]
        warm_prefix_length = selection["warm_prefix_length"]
        config_id = selection["lir_config_id"]
        candidate_config = self.candidate_config_loader(self.root, config_id)

        split_contract = manifest["split_contract"]
        task_contract = manifest["task_contract"]
        attack_contract = manifest["attack_contract"]
        execution_contract = manifest["execution_contract"]

        scored_task_count = int(task_contract["scored_task_count_per_stream"])
        cells: list[FinalTestCell] = []
        run_count = 0
        denominator = 0
        cache: dict[str, dict[str, Any]] = {}

        raw_dir.mkdir(parents=True, exist_ok=False)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        for seed in split_contract["seeds"]:
            for family in task_contract["accuracy_families"]:
                for attack_class in attack_contract["classes"]:
                    if attack_class == "fixed_target":
                        attack_ids = ("fixed_target_numerical",) if family.startswith("synthetic_num") else ("fixed_target_categorical",)
                        schedules = (None,)
                    else:
                        attack_ids = (attack_contract["on_off_attack_id"],)
                        schedules = tuple(tuple(schedule) for schedule in attack_contract["on_off_schedules"])
                    for attack_id in attack_ids:
                        for rho in attack_contract["rho_grid"]:
                            for schedule in schedules:
                                base_key = f"seed={seed}|family={family}|attack_class={attack_class}|attack_id={attack_id}|rho={rho}|schedule={schedule}"
                                if base_key not in cache:
                                    fixed_target_selection = (
                                        fixed_targets["numerical"] if family.startswith("synthetic_num") else fixed_targets["categorical"]
                                    )
                                    scored_tasks, scored_hashes = self._build_scored_stream(
                                        seed=seed,
                                        family=family,
                                        rho=rho,
                                        attack_id=attack_id,
                                        fixed_target_selection=fixed_target_selection,
                                        schedule=schedule,
                                        scored_task_count=scored_task_count,
                                    )
                                    prefix_tasks = self._build_prefix_stream(
                                        seed=seed,
                                        family=family,
                                        warm_prefix_length=warm_prefix_length,
                                    )
                                    prefix_state = LongitudinalState()
                                    if prefix_tasks:
                                        prefix_result = self.ablation_sequence_runner(
                                            task_artifacts=prefix_tasks,
                                            candidate_config=candidate_config,
                                            longitudinal_state=LongitudinalState(),
                                            ablation="full",
                                        )
                                        prefix_state = prefix_result.get("final_state", LongitudinalState())
                                    cache[base_key] = {
                                        "scored_tasks": scored_tasks,
                                        "scored_hashes": scored_hashes,
                                        "prefix_tasks": prefix_tasks,
                                        "prefix_state": prefix_state,
                                    }
                                base = cache[base_key]
                                for start in execution_contract["start_conditions"]:
                                    denominator += 1
                                    for variant in execution_contract["variants"]:
                                        state = LongitudinalState() if start == "cold" else base["prefix_state"]
                                        try:
                                            runner_result = self.ablation_sequence_runner(
                                                task_artifacts=base["scored_tasks"],
                                                candidate_config=candidate_config,
                                                longitudinal_state=state,
                                                ablation=variant,
                                            )
                                            results = list(runner_result.get("results", []))
                                            primary_loss = self._primary_loss_from_sequence(results, base["scored_tasks"])
                                            output = tuple(results[-1].final_output)
                                            released = results[-1].released_class_index
                                            status = "success"
                                            error = None
                                            run_count += 1
                                        except TimeoutError as exc:
                                            primary_loss = None
                                            output = None
                                            released = None
                                            status = "timeout"
                                            error = str(exc)
                                        except Exception as exc:
                                            primary_loss = None
                                            output = None
                                            released = None
                                            status = "failure"
                                            error = str(exc)
                                        cells.append(
                                            FinalTestCell(
                                                seed=seed,
                                                family=family,
                                                rho=rho,
                                                attack_class=attack_class,
                                                attack_id=attack_id,
                                                schedule=schedule,
                                                start=start,
                                                variant=variant,
                                                status=status,
                                                primary_loss=primary_loss,
                                                output=output,
                                                released_class_index=released,
                                                scored_task_hashes=base["scored_hashes"],
                                                cold_warm_hash_equal=True,
                                                error=error,
                                            )
                                        )

        result = FinalTestResult(
            plan=self.plan(),
            raw_directory=str(raw_dir),
            summary_path=str(summary_path),
            cells=tuple(cells),
            run_count=run_count,
            denominator=denominator,
            manifest_sha256=self.manifest_sha256(),
            gate_sha256=self.gate_sha256(),
            authorized=True,
        )
        payload = json.dumps(asdict(result), indent=2, sort_keys=True, default=str)
        (raw_dir / "result.json").write_text(payload, encoding="utf-8")
        summary_path.write_text(payload, encoding="utf-8")
        self._materialized = True
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase10 r2.1 final-test materializer")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    materializer = FinalTestMaterializer(args.root)
    if args.plan_only:
        plan = materializer.plan()
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return 0
    result = materializer.materialize()
    print(json.dumps(asdict(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
