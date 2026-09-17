from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from lir_pptd.experiments.phase_r1.attacks import realized_malicious_report_fraction
from lir_pptd.experiments.phase_r1.formal_runner import (
    DATASET_ORDER,
    _append_jsonl,
    _atomic_json,
    _attack_tasks,
    _load_config,
    _load_json,
    _method_list,
    _run_id,
    _run_method,
    _sha256,
    load_bound_datasets,
    plan_streams,
    verify_bound_files as verify_phase_r1_bound_files,
)

UPSTREAM_PHASE_R1_START_BINDING_SHA256 = (
    "cdfc8abca3a954ce4f4f0f46cd72eaf424ed5b3affc1424187c2daf883bf536e"
)
UPSTREAM_E1_RAW_SHA256 = "09b7439c7f514cd6b547f249be1c3dea258489cc506ea1349f609460220714cb"
UPSTREAM_E1_SUMMARY_SHA256 = "57a3dc0ce6871cde7ae27e19f9145a68eaa299deacd3d567811510c7ccc10688"
UPSTREAM_E1_APPROVAL_SHA256 = "625d51c2c642ee6665b2ee5372007efb19878325de7d9f29169f029c3294f283"
R1B_STAGES = ("E2", "E3", "E4")
R1B_STREAM_COUNTS = {"E2": 360, "E3": 120, "E4": 20}
R1B_METHOD_RUN_COUNTS = {"E2": 1080, "E3": 360, "E4": 60}
R1B_PLANNED_METHOD_RUNS = 1500


class R1BRunnerError(RuntimeError):
    pass


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
                raise R1BRunnerError(f"R1B_RAW_DUPLICATE_OR_INVALID_RUN_ID:line={line_no}")
            if obj.get("r1b_start_binding_sha256") != binding_sha:
                raise R1BRunnerError(f"R1B_RAW_BINDING_MISMATCH:line={line_no}")
            if obj.get("phase") != "Phase-R1B":
                raise R1BRunnerError(f"R1B_RAW_PHASE_MISMATCH:line={line_no}")
            if obj.get("stage") not in R1B_STAGES:
                raise R1BRunnerError(f"R1B_RAW_STAGE_INVALID:line={line_no}")
            records[run_id] = obj
    return records


def verify_r1b_bound_files(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = _load_json(binding_manifest_path)
    if binding.get("phase") != "Phase-R1B":
        raise R1BRunnerError("R1B_BINDING_PHASE_MISMATCH")
    if binding.get("status") != "START_BINDING_READY_NOT_APPROVED":
        raise R1BRunnerError("R1B_BINDING_STATUS_INVALID")
    if binding.get("upstream_phase_r1_start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
        raise R1BRunnerError("R1B_UPSTREAM_BINDING_MISMATCH")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise R1BRunnerError("R1B_BINDING_FILES_INVALID")
    for rel, expected in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            raise R1BRunnerError(f"R1B_BOUND_FILE_MISSING:{rel}")
        actual = _sha256(path)
        if actual != expected:
            raise R1BRunnerError(f"R1B_BOUND_FILE_HASH_MISMATCH:{rel}")
    return binding


def verify_r1b_start_approval(root: Path, binding: Mapping[str, Any], approval_path: Path) -> dict[str, Any]:
    approval = _load_json(approval_path)
    if approval.get("phase") != "Phase-R1B" or approval.get("approval_type") != "formal_start_approval":
        raise R1BRunnerError("R1B_APPROVAL_TYPE_INVALID")
    if approval.get("status") != "PASS":
        raise R1BRunnerError("R1B_FORMAL_START_NOT_APPROVED")
    expected = str(binding.get("start_binding_sha256"))
    if not expected or approval.get("start_binding_sha256") != expected:
        raise R1BRunnerError("R1B_APPROVAL_BINDING_MISMATCH")
    return approval


def verify_upstream_e1(root: Path) -> dict[str, Any]:
    summary_path = root / "results/summary/phase_r1_formal_summary.json"
    raw_path = root / "results/raw/phase_r1/formal_runs.jsonl"
    approval_path = root / "approvals/phase_r1/PHASE_R1_START_APPROVAL.json"
    binding_path = root / "results/summary/phase_r1_g7_start_binding_manifest.json"

    for path in (summary_path, raw_path, approval_path, binding_path):
        if not path.is_file():
            raise R1BRunnerError(f"UPSTREAM_E1_REQUIRED_FILE_MISSING:{path.relative_to(root)}")

    if _sha256(summary_path) != UPSTREAM_E1_SUMMARY_SHA256:
        raise R1BRunnerError("UPSTREAM_E1_SUMMARY_HASH_MISMATCH")
    if _sha256(raw_path) != UPSTREAM_E1_RAW_SHA256:
        raise R1BRunnerError("UPSTREAM_E1_RAW_HASH_MISMATCH")
    if _sha256(approval_path) != UPSTREAM_E1_APPROVAL_SHA256:
        raise R1BRunnerError("UPSTREAM_E1_APPROVAL_HASH_MISMATCH")

    upstream_binding = verify_phase_r1_bound_files(root, binding_path)
    if upstream_binding.get("start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
        raise R1BRunnerError("UPSTREAM_G7_BINDING_MISMATCH")

    summary = _load_json(summary_path)
    if summary.get("start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
        raise R1BRunnerError("UPSTREAM_E1_START_BINDING_MISMATCH")
    if summary.get("status") != "partial_runtime_deferred":
        raise R1BRunnerError("UPSTREAM_E1_SUMMARY_STATUS_MISMATCH")
    expected_stage_status = {
        "E1": "complete",
        "E2": "deferred_by_prefrozen_runtime_gate",
        "E3": "deferred_by_prefrozen_runtime_gate",
        "E4": "deferred_by_prefrozen_runtime_gate",
    }
    if summary.get("stage_status") != expected_stage_status:
        raise R1BRunnerError("UPSTREAM_E1_STAGE_STATUS_MISMATCH")
    expected_counts = {
        "planned_method_runs": 3186,
        "committed_method_runs": 1686,
        "success_method_runs": 1686,
        "failure_method_runs": 0,
        "deferred_method_runs": 1500,
    }
    for key, expected in expected_counts.items():
        if int(summary.get(key, -1)) != expected:
            raise R1BRunnerError(f"UPSTREAM_E1_COUNT_MISMATCH:{key}")

    seen: set[str] = set()
    record_count = 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            record_count += 1
            run_id = str(obj.get("run_id"))
            if not run_id or run_id in seen:
                raise R1BRunnerError(f"UPSTREAM_E1_DUPLICATE_RUN_ID:line={line_no}")
            seen.add(run_id)
            if obj.get("stage") != "E1" or obj.get("status") != "success":
                raise R1BRunnerError(f"UPSTREAM_E1_RECORD_INVALID:line={line_no}")
            if obj.get("start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
                raise R1BRunnerError(f"UPSTREAM_E1_RECORD_BINDING_MISMATCH:line={line_no}")
    if record_count != 1686:
        raise R1BRunnerError(f"UPSTREAM_E1_RECORD_COUNT_MISMATCH:{record_count}")

    # R1B was deliberately planned before E1 inferential outcomes were inspected.
    if (root / "results/summary/phase_r1_statistical_analysis.json").exists():
        raise R1BRunnerError("UPSTREAM_E1_STATISTICAL_ANALYSIS_ALREADY_EXISTS")
    if (root / "results/summary/phase_r1_combined_statistical_analysis.json").exists():
        raise R1BRunnerError("UPSTREAM_COMBINED_STATISTICAL_ANALYSIS_ALREADY_EXISTS")

    return {
        "summary": summary,
        "run_ids": seen,
        "record_count": record_count,
        "raw_sha256": UPSTREAM_E1_RAW_SHA256,
        "summary_sha256": UPSTREAM_E1_SUMMARY_SHA256,
        "approval_sha256": UPSTREAM_E1_APPROVAL_SHA256,
    }


def _load_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, tuple[Any, ...]]]:
    dataset_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_dataset_manifest.json")
    attack_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_attack_manifest.json")
    datasets = load_bound_datasets(root, dataset_manifest)
    full_plan = plan_streams(datasets, attack_manifest)
    r1b_plan = {stage: full_plan[stage] for stage in R1B_STAGES}
    for stage, expected in R1B_STREAM_COUNTS.items():
        if len(r1b_plan[stage]) != expected:
            raise R1BRunnerError(f"R1B_PLAN_STREAM_CARDINALITY_MISMATCH:{stage}:{len(r1b_plan[stage])}:{expected}")
    method_runs = sum(len(r1b_plan[stage]) * len(_method_list(stage)) for stage in R1B_STAGES)
    if method_runs != R1B_PLANNED_METHOD_RUNS:
        raise R1BRunnerError(f"R1B_PLAN_METHOD_CARDINALITY_MISMATCH:{method_runs}")
    return dataset_manifest, attack_manifest, datasets, r1b_plan


def validate_plan(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = verify_r1b_bound_files(root, binding_manifest_path)
    upstream = verify_upstream_e1(root)
    _, _, datasets, r1b_plan = _load_plan(root)

    planned_run_ids = {
        _run_id(stream, method)
        for stage in R1B_STAGES
        for stream in r1b_plan[stage]
        for method in _method_list(stage)
    }
    if len(planned_run_ids) != R1B_PLANNED_METHOD_RUNS:
        raise R1BRunnerError("R1B_PLANNED_RUN_ID_UNIQUENESS_FAILURE")
    overlap = upstream["run_ids"] & planned_run_ids
    if overlap:
        raise R1BRunnerError(f"R1B_RUN_ID_OVERLAPS_E1:{len(overlap)}")

    return {
        "start_binding_sha256": binding["start_binding_sha256"],
        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
        "stream_counts": {stage: len(r1b_plan[stage]) for stage in R1B_STAGES},
        "method_run_count": R1B_PLANNED_METHOD_RUNS,
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


def _stage_projection_seconds(runtime_budget: Mapping[str, Any], stage: str) -> float:
    return float(runtime_budget["stage_projection_hours"][stage]) * 3600.0


def execute(root: Path, binding_manifest_path: Path, approval_path: Path) -> dict[str, Any]:
    binding = verify_r1b_bound_files(root, binding_manifest_path)
    approval = verify_r1b_start_approval(root, binding, approval_path)
    upstream = verify_upstream_e1(root)
    binding_sha = str(binding["start_binding_sha256"])

    _, _, datasets, plan = _load_plan(root)
    candidate_config = _load_config(root / "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json")
    runtime_budget = _load_json(root / "configs/phase_r1b/frozen/phase_r1b_runtime_budget.json")

    raw_path = root / "results/raw/phase_r1b/deferred_runs.jsonl"
    checkpoint_path = root / "results/summary/phase_r1b_formal_checkpoint.json"
    final_path = root / "results/summary/phase_r1b_formal_summary.json"
    if final_path.exists():
        raise R1BRunnerError("R1B_FORMAL_SUMMARY_ALREADY_EXISTS")
    existing = _load_existing_raw(raw_path, binding_sha)

    planned_run_ids = {
        _run_id(stream, method)
        for stage in R1B_STAGES
        for stream in plan[stage]
        for method in _method_list(stage)
    }
    if set(existing) - planned_run_ids:
        raise R1BRunnerError("R1B_RAW_CONTAINS_UNPLANNED_RUN_ID")
    if upstream["run_ids"] & planned_run_ids:
        raise R1BRunnerError("R1B_PLAN_OVERLAPS_UPSTREAM_E1")

    started_wall = time.time()
    perf_started = time.perf_counter()
    stage_elapsed: dict[str, float] = {}
    stage_status: dict[str, str] = {}
    hard_cap_seconds = float(runtime_budget["hard_cap_hours"]) * 3600.0

    for stage_index, stage in enumerate(R1B_STAGES):
        if stage_index > 0:
            completed_projection = sum(
                _stage_projection_seconds(runtime_budget, s)
                for s in R1B_STAGES[:stage_index]
                if stage_status.get(s) == "complete"
            )
            completed_actual = sum(stage_elapsed.get(s, 0.0) for s in R1B_STAGES[:stage_index])
            scale = completed_actual / completed_projection if completed_projection > 0 else 1.0
            scale = max(scale, 1.0)
            remaining_projection = sum(
                _stage_projection_seconds(runtime_budget, s)
                for s in R1B_STAGES[stage_index:]
            )
            projected_total = completed_actual + remaining_projection * scale
            if projected_total > hard_cap_seconds:
                for later in R1B_STAGES[stage_index:]:
                    stage_status[later] = "deferred_by_r1b_prefrozen_runtime_gate"
                break

        stage_start = time.perf_counter()
        for stream in plan[stage]:
            dataset = datasets[stream.dataset_id]
            attacked_tasks = _attack_tasks(stream, dataset)
            realized = (
                realized_malicious_report_fraction(
                    ({"participant_ids": task.participant_ids} for task in attacked_tasks),
                    stream.malicious_workers,
                )
                if stream.malicious_workers
                else 0.0
            )
            subset_digest = hashlib.sha256("\n".join(stream.malicious_workers).encode("utf-8")).hexdigest()

            for method in _method_list(stage):
                run_id = _run_id(stream, method)
                if run_id in existing:
                    continue
                base = {
                    "schema_version": "1.0",
                    "phase": "Phase-R1B",
                    "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
                    "upstream_e1_raw_sha256": UPSTREAM_E1_RAW_SHA256,
                    "r1b_start_binding_sha256": binding_sha,
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
                except Exception as exc:  # committed formal failures are retained; no silent retry
                    record = {
                        **base,
                        "status": "failure",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                _append_jsonl(raw_path, record)
                existing[run_id] = record
                _atomic_json(
                    checkpoint_path,
                    {
                        "schema_version": "1.0",
                        "phase": "Phase-R1B",
                        "r1b_start_binding_sha256": binding_sha,
                        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
                        "completed_method_runs": len(existing),
                        "last_run_id": run_id,
                        "last_stage": stage,
                        "updated_unix": time.time(),
                    },
                )

        stage_elapsed[stage] = time.perf_counter() - stage_start
        stage_status[stage] = "complete"

    success = sum(r.get("status") == "success" for r in existing.values())
    failure = sum(r.get("status") == "failure" for r in existing.values())
    deferred = R1B_PLANNED_METHOD_RUNS - len(existing)
    complete = len(existing) == R1B_PLANNED_METHOD_RUNS

    summary = {
        "schema_version": "1.0",
        "phase": "Phase-R1B",
        "status": "complete" if complete else "partial_runtime_deferred",
        "formal_experiment": True,
        "r1b_start_binding_sha256": binding_sha,
        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
        "upstream_e1_raw_sha256": UPSTREAM_E1_RAW_SHA256,
        "upstream_e1_summary_sha256": UPSTREAM_E1_SUMMARY_SHA256,
        "approval_sha256": _sha256(approval_path),
        "raw_path": str(raw_path.relative_to(root)).replace("\\", "/"),
        "raw_sha256": _sha256(raw_path),
        "planned_method_runs": R1B_PLANNED_METHOD_RUNS,
        "committed_method_runs": len(existing),
        "success_method_runs": success,
        "failure_method_runs": failure,
        "deferred_method_runs": deferred,
        "stage_status": stage_status,
        "stage_elapsed_seconds": stage_elapsed,
        "wall_elapsed_seconds_this_invocation": time.perf_counter() - perf_started,
        "started_unix_this_invocation": started_wall,
        "prohibitions": [
            "do_not_modify_or_rerun_E1",
            "do_not_delete_failures",
            "do_not_replace_unfavorable_results",
            "do_not_change_frozen_R1B_plan_after_start_approval",
            "do_not_inspect_effect_statistics_to_control_runtime_gate",
        ],
    }
    _atomic_json(final_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-R1B deferred secondary formal runner")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--binding-manifest",
        default="results/summary/phase_r1b_start_binding_manifest.json",
    )
    parser.add_argument(
        "--approval",
        default="approvals/phase_r1b/PHASE_R1B_START_APPROVAL.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    binding = root / args.binding_manifest
    if args.validate_only:
        result = validate_plan(root, binding)
        print("R1B_FORMAL_PLAN_VALIDATE=PASS")
        print(f"R1B_START_BINDING_SHA256={result['start_binding_sha256']}")
        print(f"R1B_PLANNED_METHOD_RUNS={result['method_run_count']}")
        print("E1_RERUN_METHOD_RUNS=0")
        print("FORMAL_EXPERIMENTS_RUN=0")
        return 0
    summary = execute(root, binding, root / args.approval)
    print(f"R1B_FORMAL_STATUS={summary['status']}")
    print(f"COMMITTED_METHOD_RUNS={summary['committed_method_runs']}")
    print(f"SUCCESS_METHOD_RUNS={summary['success_method_runs']}")
    print(f"FAILURE_METHOD_RUNS={summary['failure_method_runs']}")
    print(f"DEFERRED_METHOD_RUNS={summary['deferred_method_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
