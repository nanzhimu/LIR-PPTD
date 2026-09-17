from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from .attacks import (
    apply_fixed_target,
    apply_response_corruption,
    build_unique_malicious_subsets,
    realized_malicious_report_fraction,
    schedule_is_on,
)
from .contract import DATASET_CONTRACTS
from .longitudinal_state import initial_global_state
from .methods import run_lir_task, run_stateless_baseline
from .metrics import dataset_losses
from .real_data_loader import RealDataset, RealTask, load_real_dataset

DATASET_ORDER = ("product", "duck", "dog", "weather")
CORE_METHODS = ("mean_vote", "crh", "lir_pptd_full")
ABLATION_METHODS = ("lir_pptd_full", "lir_pptd_nr", "lir_pptd_ho")
CONFIG_ID = "lir_cfg_05_lambda020"
NONZERO_RHOS = ("1/10", "3/10", "1/2", "7/10", "9/10")
E2_RHOS = ("3/10", "1/2", "7/10")
E2_SCHEDULES = ((5, 1), (5, 5), (1, 5))
ORDER_SEEDS = (6101, 6102, 6103, 6104, 6105)


class FormalRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedStream:
    stage: str
    dataset_id: str
    attack_condition: str
    rho: str
    replicate_index: int
    subset_seed: int | None
    malicious_workers: tuple[str, ...]
    schedule: tuple[int, int] | None = None
    order_seed: int | None = None

    @property
    def pairing_id(self) -> str:
        subset_digest = hashlib.sha256("\n".join(self.malicious_workers).encode("utf-8")).hexdigest()
        schedule_text = "none" if self.schedule is None else f"{self.schedule[0]}_{self.schedule[1]}"
        order_text = "source" if self.order_seed is None else str(self.order_seed)
        return (
            f"stage={self.stage}|dataset={self.dataset_id}|attack={self.attack_condition}|rho={self.rho}"
            f"|rep={self.replicate_index}|subset={subset_digest}|schedule={schedule_text}|order={order_text}"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _load_config(path: Path) -> dict[str, Any]:
    obj = _load_json(path)
    for config in obj.get("candidate_configs", []):
        if config.get("config_id") == CONFIG_ID:
            return dict(config)
    raise FormalRunnerError(f"CONFIG_NOT_FOUND:{CONFIG_ID}")


def verify_bound_files(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = _load_json(binding_manifest_path)
    if binding.get("phase") != "Phase-R1":
        raise FormalRunnerError("BINDING_PHASE_MISMATCH")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise FormalRunnerError("BINDING_FILES_INVALID")
    for rel, expected in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            raise FormalRunnerError(f"BOUND_FILE_MISSING:{rel}")
        actual = _sha256(path)
        if actual != expected:
            raise FormalRunnerError(f"BOUND_FILE_HASH_MISMATCH:{rel}")
    return binding


def verify_start_approval(root: Path, binding: Mapping[str, Any], approval_path: Path) -> dict[str, Any]:
    approval = _load_json(approval_path)
    if approval.get("phase") != "Phase-R1" or approval.get("approval_type") != "formal_start_approval":
        raise FormalRunnerError("APPROVAL_TYPE_INVALID")
    if approval.get("status") != "PASS":
        raise FormalRunnerError("FORMAL_START_NOT_APPROVED")
    expected = str(binding.get("start_binding_sha256"))
    if not expected or approval.get("start_binding_sha256") != expected:
        raise FormalRunnerError("APPROVAL_BINDING_MISMATCH")
    return approval


def _dataset_manifest_map(dataset_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    ds = dataset_manifest.get("datasets")
    if not isinstance(ds, dict):
        raise FormalRunnerError("DATASET_MANIFEST_INVALID")
    return ds


def load_bound_datasets(root: Path, dataset_manifest: Mapping[str, Any]) -> dict[str, RealDataset]:
    out: dict[str, RealDataset] = {}
    entries = _dataset_manifest_map(dataset_manifest)
    for dataset_id in DATASET_ORDER:
        entry = entries[dataset_id]
        dataset = load_real_dataset(
            root / "data",
            dataset_id,
            expected_answer_sha256=str(entry["answer_sha256"]),
            expected_truth_sha256=str(entry["truth_sha256"]),
        )
        contract = DATASET_CONTRACTS[dataset_id]
        if dataset.modality != contract.modality or dataset.class_count != contract.class_count:
            raise FormalRunnerError(f"DATASET_CONTRACT_MISMATCH:{dataset_id}")
        out[dataset_id] = dataset
    return out


def _replicate_count(dataset_id: str, rho: str, attack_manifest: Mapping[str, Any]) -> int:
    raw = attack_manifest["E1_core_fixed_target"]["replicates"][dataset_id][rho]
    return int(raw)


def _subsets(dataset: RealDataset, rho: str, requested: int) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return build_unique_malicious_subsets(
        dataset.worker_ids,
        dataset_id=dataset.dataset_id,
        rho=rho,
        requested=requested,
        seed_start=5001,
    )


def plan_streams(
    datasets: Mapping[str, RealDataset],
    attack_manifest: Mapping[str, Any],
) -> dict[str, tuple[PlannedStream, ...]]:
    stages: dict[str, list[PlannedStream]] = {"E1": [], "E2": [], "E3": [], "E4": []}

    for dataset_id in DATASET_ORDER:
        dataset = datasets[dataset_id]
        stages["E1"].append(PlannedStream("E1", dataset_id, "fixed_target", "0", 0, None, ()))
        for rho in NONZERO_RHOS:
            requested = _replicate_count(dataset_id, rho, attack_manifest)
            for rep, (seed, workers) in enumerate(_subsets(dataset, rho, requested), start=1):
                stages["E1"].append(
                    PlannedStream("E1", dataset_id, "fixed_target", rho, rep, seed, workers)
                )

        for rho in E2_RHOS:
            subset_list = _subsets(dataset, rho, 10)
            for schedule in E2_SCHEDULES:
                attack_id = f"on_off_{schedule[0]}_{schedule[1]}"
                for rep, (seed, workers) in enumerate(subset_list, start=1):
                    stages["E2"].append(
                        PlannedStream("E2", dataset_id, attack_id, rho, rep, seed, workers, schedule=schedule)
                    )

        for rho in E2_RHOS:
            for rep, (seed, workers) in enumerate(_subsets(dataset, rho, 10), start=1):
                stages["E3"].append(
                    PlannedStream("E3", dataset_id, "response_corruption", rho, rep, seed, workers)
                )

        rho = "1/2"
        seed, workers = _subsets(dataset, rho, 1)[0]
        for order_seed in ORDER_SEEDS:
            stages["E4"].append(
                PlannedStream("E4", dataset_id, "response_corruption", rho, 1, seed, workers, order_seed=order_seed)
            )

    expected = {"E1": 562, "E2": 360, "E3": 120, "E4": 20}
    for stage, count in expected.items():
        if len(stages[stage]) != count:
            raise FormalRunnerError(f"PLAN_CARDINALITY_MISMATCH:{stage}:{len(stages[stage])}:{count}")
    return {k: tuple(v) for k, v in stages.items()}


def _permuted_tasks(tasks: tuple[RealTask, ...], order_seed: int | None) -> tuple[RealTask, ...]:
    if order_seed is None:
        return tasks
    return tuple(sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"phase-r1-order-v1|{order_seed}|{task.task_id}".encode("utf-8")).digest(),
    ))


def _attack_tasks(stream: PlannedStream, dataset: RealDataset) -> tuple[RealTask, ...]:
    ordered = _permuted_tasks(dataset.tasks, stream.order_seed)
    attacked: list[RealTask] = []
    for index, task in enumerate(ordered):
        if stream.rho == "0":
            reports = dict(task.reports)
        elif stream.attack_condition == "fixed_target":
            reports = apply_fixed_target(task.reports, stream.malicious_workers, modality=task.modality)
        elif stream.attack_condition == "response_corruption":
            reports = apply_response_corruption(
                task.reports, stream.malicious_workers, modality=task.modality, active=True
            )
        elif stream.attack_condition.startswith("on_off_"):
            if stream.schedule is None:
                raise FormalRunnerError("ON_OFF_SCHEDULE_MISSING")
            active = schedule_is_on(index, stream.schedule[0], stream.schedule[1])
            reports = apply_response_corruption(
                task.reports, stream.malicious_workers, modality=task.modality, active=active
            )
        else:
            raise FormalRunnerError(f"UNKNOWN_ATTACK:{stream.attack_condition}")
        attacked.append(replace(task, reports=reports))
    return tuple(attacked)


def _run_method(
    dataset: RealDataset,
    tasks: tuple[RealTask, ...],
    method: str,
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    predicted_classes: list[int] = []
    truth_classes: list[int] = []
    predicted_values: list[float] = []
    truth_values: list[float] = []

    if method in {"mean_vote", "crh"}:
        for task in tasks:
            vector, prediction_class = run_stateless_baseline(task, method)
            if dataset.modality == "categorical":
                if prediction_class is None or task.truth_class is None:
                    raise FormalRunnerError("CATEGORICAL_PREDICTION_MISSING")
                predicted_classes.append(prediction_class)
                truth_classes.append(task.truth_class)
            else:
                predicted_values.append(float(vector[0]))
                truth_values.append(float(task.truth_vector[0]))
    else:
        ablation_map = {
            "lir_pptd_full": "full",
            "lir_pptd_nr": "nr",
            "lir_pptd_ho": "ho",
        }
        if method not in ablation_map:
            raise FormalRunnerError(f"UNKNOWN_METHOD:{method}")
        state = initial_global_state(dataset.worker_ids, candidate_config["c0"])
        for task in tasks:
            result = run_lir_task(
                task,
                candidate_config,
                state,
                ablation=ablation_map[method],
                dps=80,
            )
            state = result.next_state
            if dataset.modality == "categorical":
                if result.prediction_class is None or task.truth_class is None:
                    raise FormalRunnerError("CATEGORICAL_PREDICTION_MISSING")
                predicted_classes.append(result.prediction_class)
                truth_classes.append(task.truth_class)
            else:
                predicted_values.append(float(result.prediction_vector[0]))
                truth_values.append(float(task.truth_vector[0]))

    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=predicted_classes if dataset.modality == "categorical" else None,
        truth_classes=truth_classes if dataset.modality == "categorical" else None,
        predicted_values=predicted_values if dataset.modality == "numerical" else None,
        truth_values=truth_values if dataset.modality == "numerical" else None,
    )
    elapsed = time.perf_counter() - started
    return {
        **losses,
        "task_count": len(tasks),
        "elapsed_seconds": elapsed,
    }


def _method_list(stage: str) -> tuple[str, ...]:
    if stage in {"E1", "E2"}:
        return CORE_METHODS
    if stage in {"E3", "E4"}:
        return ABLATION_METHODS
    raise FormalRunnerError(f"UNKNOWN_STAGE:{stage}")


def _run_id(stream: PlannedStream, method: str) -> str:
    return hashlib.sha256(f"{stream.pairing_id}|method={method}".encode("utf-8")).hexdigest()


def _load_existing_raw(path: Path, binding_sha: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            run_id = str(obj.get("run_id"))
            if not run_id or run_id in records:
                raise FormalRunnerError(f"RAW_DUPLICATE_OR_INVALID_RUN_ID:line={line_no}")
            if obj.get("start_binding_sha256") != binding_sha:
                raise FormalRunnerError(f"RAW_BINDING_MISMATCH:line={line_no}")
            records[run_id] = obj
    return records


def validate_plan(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = verify_bound_files(root, binding_manifest_path)
    dataset_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_dataset_manifest.json")
    attack_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_attack_manifest.json")
    datasets = load_bound_datasets(root, dataset_manifest)
    plan = plan_streams(datasets, attack_manifest)
    method_runs = sum(len(streams) * len(_method_list(stage)) for stage, streams in plan.items())
    if method_runs != 3186:
        raise FormalRunnerError(f"METHOD_RUN_CARDINALITY_MISMATCH:{method_runs}")
    return {
        "start_binding_sha256": binding["start_binding_sha256"],
        "stream_counts": {stage: len(streams) for stage, streams in plan.items()},
        "method_run_count": method_runs,
        "dataset_contracts": {
            dataset_id: {
                "modality": datasets[dataset_id].modality,
                "class_count": datasets[dataset_id].class_count,
                "tasks": len(datasets[dataset_id].tasks),
                "workers": len(datasets[dataset_id].worker_ids),
            }
            for dataset_id in DATASET_ORDER
        },
    }


def execute(root: Path, binding_manifest_path: Path, approval_path: Path) -> dict[str, Any]:
    binding = verify_bound_files(root, binding_manifest_path)
    approval = verify_start_approval(root, binding, approval_path)
    binding_sha = str(binding["start_binding_sha256"])

    dataset_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_dataset_manifest.json")
    attack_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_attack_manifest.json")
    runtime_budget = _load_json(root / "configs/phase_r1/frozen/phase_r1_runtime_budget.json")
    datasets = load_bound_datasets(root, dataset_manifest)
    plan = plan_streams(datasets, attack_manifest)
    candidate_config = _load_config(root / "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json")

    raw_path = root / "results/raw/phase_r1/formal_runs.jsonl"
    checkpoint_path = root / "results/summary/phase_r1_formal_checkpoint.json"
    final_path = root / "results/summary/phase_r1_formal_summary.json"
    if final_path.exists():
        raise FormalRunnerError("FORMAL_SUMMARY_ALREADY_EXISTS")
    existing = _load_existing_raw(raw_path, binding_sha)

    started_wall = time.time()
    perf_started = time.perf_counter()
    stage_elapsed: dict[str, float] = {}
    stage_status: dict[str, str] = {}

    for stage in ("E1", "E2", "E3", "E4"):
        # Runtime-only stage gate before secondary stages.
        if stage != "E1" and stage_status.get("E1") == "complete":
            predicted_total = float(runtime_budget["stage_projection_hours"]["total"])
            predicted_e1 = float(runtime_budget["stage_projection_hours"]["E1_core"])
            actual_e1_h = stage_elapsed["E1"] / 3600.0
            scale = actual_e1_h / predicted_e1 if predicted_e1 > 0 else 1.0
            runtime_projection = predicted_total * max(scale, 1.0)
            if runtime_projection > float(runtime_budget["hard_cap_hours"]):
                stage_status[stage] = "deferred_by_prefrozen_runtime_gate"
                for later in ("E2", "E3", "E4")[("E2", "E3", "E4").index(stage):]:
                    stage_status[later] = "deferred_by_prefrozen_runtime_gate"
                break

        stage_start = time.perf_counter()
        for stream in plan[stage]:
            dataset = datasets[stream.dataset_id]
            attacked_tasks = _attack_tasks(stream, dataset)
            realized = realized_malicious_report_fraction(
                ({"participant_ids": task.participant_ids} for task in attacked_tasks),
                stream.malicious_workers,
            ) if stream.malicious_workers else 0.0
            subset_digest = hashlib.sha256("\n".join(stream.malicious_workers).encode("utf-8")).hexdigest()

            for method in _method_list(stage):
                run_id = _run_id(stream, method)
                if run_id in existing:
                    continue
                base = {
                    "schema_version": "1.0",
                    "phase": "Phase-R1",
                    "start_binding_sha256": binding_sha,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "stage": stage,
                    "dataset_id": stream.dataset_id,
                    "modality": dataset.modality,
                    "class_count": dataset.class_count,
                    "attack_condition": stream.attack_condition,
                    "rho": stream.rho,
                    "replicate_index": stream.replicate_index,
                    "subset_seed": stream.subset_seed,
                    "malicious_worker_count": len(stream.malicious_workers),
                    "malicious_subset_sha256": subset_digest,
                    "realized_malicious_report_fraction": realized,
                    "schedule": list(stream.schedule) if stream.schedule is not None else None,
                    "task_order_seed": stream.order_seed,
                    "method": method,
                }
                try:
                    result = _run_method(dataset, attacked_tasks, method, candidate_config)
                    record = {**base, "status": "success", **result}
                except Exception as exc:  # retain formal failure; never silently retry committed failures
                    record = {
                        **base,
                        "status": "failure",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                _append_jsonl(raw_path, record)
                existing[run_id] = record
                _atomic_json(checkpoint_path, {
                    "schema_version": "1.0",
                    "phase": "Phase-R1",
                    "start_binding_sha256": binding_sha,
                    "completed_method_runs": len(existing),
                    "last_run_id": run_id,
                    "last_stage": stage,
                    "updated_unix": time.time(),
                })

        stage_elapsed[stage] = time.perf_counter() - stage_start
        stage_status[stage] = "complete"

    success = sum(r.get("status") == "success" for r in existing.values())
    failure = sum(r.get("status") == "failure" for r in existing.values())
    planned_total = 3186
    deferred = planned_total - len(existing)
    complete = len(existing) == planned_total

    summary = {
        "schema_version": "1.0",
        "phase": "Phase-R1",
        "status": "complete" if complete else "partial_runtime_deferred",
        "formal_experiment": True,
        "start_binding_sha256": binding_sha,
        "approval_sha256": _sha256(approval_path),
        "raw_path": str(raw_path.relative_to(root)).replace("\\", "/"),
        "raw_sha256": _sha256(raw_path),
        "planned_method_runs": planned_total,
        "committed_method_runs": len(existing),
        "success_method_runs": success,
        "failure_method_runs": failure,
        "deferred_method_runs": deferred,
        "stage_status": stage_status,
        "stage_elapsed_seconds": stage_elapsed,
        "wall_elapsed_seconds_this_invocation": time.perf_counter() - perf_started,
        "started_unix_this_invocation": started_wall,
        "prohibitions": [
            "do_not_delete_failures",
            "do_not_replace_unfavorable_results",
            "do_not_change frozen plan after start approval",
        ],
    }
    _atomic_json(final_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-R1 formal real-data runner")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--binding-manifest",
        default="results/summary/phase_r1_g7_start_binding_manifest.json",
    )
    parser.add_argument(
        "--approval",
        default="approvals/phase_r1/PHASE_R1_START_APPROVAL.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    binding = root / args.binding_manifest
    if args.validate_only:
        result = validate_plan(root, binding)
        print("R1_FORMAL_PLAN_VALIDATE=PASS")
        print(f"R1_START_BINDING_SHA256={result['start_binding_sha256']}")
        print(f"PLANNED_METHOD_RUNS={result['method_run_count']}")
        print("FORMAL_EXPERIMENTS_RUN=0")
        return 0
    summary = execute(root, binding, root / args.approval)
    print(f"R1_FORMAL_STATUS={summary['status']}")
    print(f"COMMITTED_METHOD_RUNS={summary['committed_method_runs']}")
    print(f"SUCCESS_METHOD_RUNS={summary['success_method_runs']}")
    print(f"FAILURE_METHOD_RUNS={summary['failure_method_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
