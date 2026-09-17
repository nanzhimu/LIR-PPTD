from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from lir_pptd.experiments.phase_r1.attacks import realized_malicious_report_fraction
from lir_pptd.experiments.phase_r1.formal_runner import (
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
)
from lir_pptd.experiments.phase_r1b.runner import (
    verify_upstream_e1,
    verify_r1b_bound_files,
    verify_r1b_start_approval,
)

UPSTREAM_PHASE_R1_START_BINDING_SHA256 = (
    "cdfc8abca3a954ce4f4f0f46cd72eaf424ed5b3affc1424187c2daf883bf536e"
)
UPSTREAM_E1_RAW_SHA256 = "09b7439c7f514cd6b547f249be1c3dea258489cc506ea1349f609460220714cb"
UPSTREAM_R1B_START_BINDING_SHA256 = "09ff8192e3bdc664237589cc8925b1b2265bddd5a0c1e99f5c19e2523720f0e5"
UPSTREAM_E2_RAW_SHA256 = "3eebb2fe187908e62325c95fb8bd4cdb49eb338712a3744d79e344092889495d"
UPSTREAM_E2_SUMMARY_SHA256 = "7f67841f8789fb0fab34f3950728bba4ae1e8e46ec91ae6009248f1e362d3aaa"
UPSTREAM_R1B_APPROVAL_SHA256 = "3e11f241b6634443fce1f8217a0c6e382bc91bf368870a46d27b48d795c4c84c"

R1C_STAGES = ("E3", "E4")
R1C_STREAM_COUNTS = {"E3": 120, "E4": 20}
R1C_METHOD_RUN_COUNTS = {"E3": 360, "E4": 60}
R1C_PLANNED_METHOD_RUNS = 420


class R1CRunnerError(RuntimeError):
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
                raise R1CRunnerError(f"R1C_RAW_DUPLICATE_OR_INVALID_RUN_ID:line={line_no}")
            if obj.get("r1c_start_binding_sha256") != binding_sha:
                raise R1CRunnerError(f"R1C_RAW_BINDING_MISMATCH:line={line_no}")
            if obj.get("phase") != "Phase-R1C":
                raise R1CRunnerError(f"R1C_RAW_PHASE_MISMATCH:line={line_no}")
            if obj.get("stage") not in R1C_STAGES:
                raise R1CRunnerError(f"R1C_RAW_STAGE_INVALID:line={line_no}")
            records[run_id] = obj
    return records


def verify_r1c_bound_files(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = _load_json(binding_manifest_path)
    if binding.get("phase") != "Phase-R1C":
        raise R1CRunnerError("R1C_BINDING_PHASE_MISMATCH")
    if binding.get("status") != "START_BINDING_READY_NOT_APPROVED":
        raise R1CRunnerError("R1C_BINDING_STATUS_INVALID")
    if binding.get("upstream_phase_r1_start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
        raise R1CRunnerError("R1C_UPSTREAM_R1_BINDING_MISMATCH")
    if binding.get("upstream_r1b_start_binding_sha256") != UPSTREAM_R1B_START_BINDING_SHA256:
        raise R1CRunnerError("R1C_UPSTREAM_R1B_BINDING_MISMATCH")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise R1CRunnerError("R1C_BINDING_FILES_INVALID")
    for rel, expected in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            raise R1CRunnerError(f"R1C_BOUND_FILE_MISSING:{rel}")
        if _sha256(path) != expected:
            raise R1CRunnerError(f"R1C_BOUND_FILE_HASH_MISMATCH:{rel}")
    return binding


def verify_r1c_start_approval(root: Path, binding: Mapping[str, Any], approval_path: Path) -> dict[str, Any]:
    approval = _load_json(approval_path)
    if approval.get("phase") != "Phase-R1C" or approval.get("approval_type") != "formal_start_approval":
        raise R1CRunnerError("R1C_APPROVAL_TYPE_INVALID")
    if approval.get("status") != "PASS":
        raise R1CRunnerError("R1C_FORMAL_START_NOT_APPROVED")
    expected = str(binding.get("start_binding_sha256"))
    if not expected or approval.get("start_binding_sha256") != expected:
        raise R1CRunnerError("R1C_APPROVAL_BINDING_MISMATCH")
    return approval


def _load_full_plan(root: Path):
    dataset_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_dataset_manifest.json")
    attack_manifest = _load_json(root / "configs/phase_r1/frozen/phase_r1_attack_manifest.json")
    datasets = load_bound_datasets(root, dataset_manifest)
    plan = plan_streams(datasets, attack_manifest)
    return dataset_manifest, attack_manifest, datasets, plan


def verify_upstream_e2(root: Path) -> dict[str, Any]:
    # Reverify E1 through the already-frozen R1B verifier.
    e1 = verify_upstream_e1(root)

    r1b_binding_path = root / "results/summary/phase_r1b_start_binding_manifest.json"
    r1b_approval_path = root / "approvals/phase_r1b/PHASE_R1B_START_APPROVAL.json"
    e2_summary_path = root / "results/summary/phase_r1b_formal_summary.json"
    e2_raw_path = root / "results/raw/phase_r1b/deferred_runs.jsonl"

    for path in (r1b_binding_path, r1b_approval_path, e2_summary_path, e2_raw_path):
        if not path.is_file():
            raise R1CRunnerError(f"UPSTREAM_E2_REQUIRED_FILE_MISSING:{path.relative_to(root)}")

    r1b_binding = verify_r1b_bound_files(root, r1b_binding_path)
    if r1b_binding.get("start_binding_sha256") != UPSTREAM_R1B_START_BINDING_SHA256:
        raise R1CRunnerError("UPSTREAM_R1B_BINDING_MISMATCH")
    verify_r1b_start_approval(root, r1b_binding, r1b_approval_path)

    if _sha256(r1b_approval_path) != UPSTREAM_R1B_APPROVAL_SHA256:
        raise R1CRunnerError("UPSTREAM_R1B_APPROVAL_HASH_MISMATCH")
    if _sha256(e2_summary_path) != UPSTREAM_E2_SUMMARY_SHA256:
        raise R1CRunnerError("UPSTREAM_E2_SUMMARY_HASH_MISMATCH")
    if _sha256(e2_raw_path) != UPSTREAM_E2_RAW_SHA256:
        raise R1CRunnerError("UPSTREAM_E2_RAW_HASH_MISMATCH")

    summary = _load_json(e2_summary_path)
    if summary.get("r1b_start_binding_sha256") != UPSTREAM_R1B_START_BINDING_SHA256:
        raise R1CRunnerError("UPSTREAM_E2_START_BINDING_MISMATCH")
    if summary.get("status") != "partial_runtime_deferred":
        raise R1CRunnerError("UPSTREAM_E2_SUMMARY_STATUS_MISMATCH")
    expected_stage_status = {
        "E2": "complete",
        "E3": "deferred_by_r1b_prefrozen_runtime_gate",
        "E4": "deferred_by_r1b_prefrozen_runtime_gate",
    }
    if summary.get("stage_status") != expected_stage_status:
        raise R1CRunnerError("UPSTREAM_E2_STAGE_STATUS_MISMATCH")
    expected_counts = {
        "planned_method_runs": 1500,
        "committed_method_runs": 1080,
        "success_method_runs": 1080,
        "failure_method_runs": 0,
        "deferred_method_runs": 420,
    }
    for key, expected in expected_counts.items():
        if int(summary.get(key, -1)) != expected:
            raise R1CRunnerError(f"UPSTREAM_E2_COUNT_MISMATCH:{key}")

    _, _, _, full_plan = _load_full_plan(root)
    planned_e2_ids = {
        _run_id(stream, method)
        for stream in full_plan["E2"]
        for method in _method_list("E2")
    }
    if len(planned_e2_ids) != 1080:
        raise R1CRunnerError("UPSTREAM_E2_PLAN_CARDINALITY_MISMATCH")

    seen: set[str] = set()
    with e2_raw_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            run_id = str(obj.get("run_id"))
            if not run_id or run_id in seen:
                raise R1CRunnerError(f"UPSTREAM_E2_DUPLICATE_RUN_ID:line={line_no}")
            seen.add(run_id)
            if obj.get("phase") != "Phase-R1B" or obj.get("stage") != "E2":
                raise R1CRunnerError(f"UPSTREAM_E2_RECORD_STAGE_INVALID:line={line_no}")
            if obj.get("status") != "success":
                raise R1CRunnerError(f"UPSTREAM_E2_RECORD_NON_SUCCESS:line={line_no}")
            if obj.get("r1b_start_binding_sha256") != UPSTREAM_R1B_START_BINDING_SHA256:
                raise R1CRunnerError(f"UPSTREAM_E2_RECORD_BINDING_MISMATCH:line={line_no}")
    if seen != planned_e2_ids:
        raise R1CRunnerError(
            f"UPSTREAM_E2_RUN_ID_SET_MISMATCH:seen={len(seen)}:planned={len(planned_e2_ids)}"
        )

    # R1C is decided before inferential effect outcomes are inspected.
    forbidden_analysis_outputs = (
        "results/summary/phase_r1_statistical_analysis.json",
        "results/summary/phase_r1_combined_statistical_analysis.json",
        "results/summary/phase_r1_combined_statistical_analysis_v2.json",
    )
    for rel in forbidden_analysis_outputs:
        if (root / rel).exists():
            raise R1CRunnerError(f"UPSTREAM_EFFECT_STATISTICAL_ANALYSIS_ALREADY_EXISTS:{rel}")

    return {
        "e1": e1,
        "summary": summary,
        "run_ids": seen,
        "record_count": len(seen),
        "raw_sha256": UPSTREAM_E2_RAW_SHA256,
        "summary_sha256": UPSTREAM_E2_SUMMARY_SHA256,
        "approval_sha256": UPSTREAM_R1B_APPROVAL_SHA256,
    }


def _load_plan(root: Path):
    dataset_manifest, attack_manifest, datasets, full_plan = _load_full_plan(root)
    r1c_plan = {stage: full_plan[stage] for stage in R1C_STAGES}
    for stage, expected in R1C_STREAM_COUNTS.items():
        if len(r1c_plan[stage]) != expected:
            raise R1CRunnerError(
                f"R1C_PLAN_STREAM_CARDINALITY_MISMATCH:{stage}:{len(r1c_plan[stage])}:{expected}"
            )
    method_runs = sum(
        len(r1c_plan[stage]) * len(_method_list(stage))
        for stage in R1C_STAGES
    )
    if method_runs != R1C_PLANNED_METHOD_RUNS:
        raise R1CRunnerError(f"R1C_PLAN_METHOD_CARDINALITY_MISMATCH:{method_runs}")
    return dataset_manifest, attack_manifest, datasets, r1c_plan


def validate_plan(root: Path, binding_manifest_path: Path) -> dict[str, Any]:
    binding = verify_r1c_bound_files(root, binding_manifest_path)
    upstream = verify_upstream_e2(root)
    _, _, datasets, r1c_plan = _load_plan(root)

    planned_run_ids = {
        _run_id(stream, method)
        for stage in R1C_STAGES
        for stream in r1c_plan[stage]
        for method in _method_list(stage)
    }
    if len(planned_run_ids) != R1C_PLANNED_METHOD_RUNS:
        raise R1CRunnerError("R1C_PLANNED_RUN_ID_UNIQUENESS_FAILURE")
    if upstream["e1"]["run_ids"] & planned_run_ids:
        raise R1CRunnerError("R1C_RUN_ID_OVERLAPS_E1")
    if upstream["run_ids"] & planned_run_ids:
        raise R1CRunnerError("R1C_RUN_ID_OVERLAPS_E2")

    return {
        "start_binding_sha256": binding["start_binding_sha256"],
        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
        "upstream_r1b_start_binding_sha256": UPSTREAM_R1B_START_BINDING_SHA256,
        "stream_counts": {stage: len(r1c_plan[stage]) for stage in R1C_STAGES},
        "method_run_count": R1C_PLANNED_METHOD_RUNS,
        "dataset_contracts": {
            dataset_id: {
                "modality": datasets[dataset_id].modality,
                "class_count": datasets[dataset_id].class_count,
                "tasks": len(datasets[dataset_id].tasks),
                "workers": len(datasets[dataset_id].worker_ids),
            }
            for dataset_id in sorted(datasets)
        },
    }


def execute(root: Path, binding_manifest_path: Path, approval_path: Path) -> dict[str, Any]:
    binding = verify_r1c_bound_files(root, binding_manifest_path)
    approval = verify_r1c_start_approval(root, binding, approval_path)
    upstream = verify_upstream_e2(root)
    binding_sha = str(binding["start_binding_sha256"])

    _, _, datasets, plan = _load_plan(root)
    candidate_config = _load_config(root / "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json")
    runtime_policy = _load_json(root / "configs/phase_r1c/frozen/phase_r1c_runtime_budget.json")
    if runtime_policy.get("runtime_gate_enabled") is not False:
        raise R1CRunnerError("R1C_RUNTIME_DEFERRAL_POLICY_NOT_DISABLED")
    if runtime_policy.get("decision") != "RUN_ALL_E3_E4_TO_COMPLETION_NO_RUNTIME_DEFERRAL":
        raise R1CRunnerError("R1C_RUNTIME_COMPLETION_POLICY_MISMATCH")

    raw_path = root / "results/raw/phase_r1c/deferred_runs.jsonl"
    checkpoint_path = root / "results/summary/phase_r1c_formal_checkpoint.json"
    final_path = root / "results/summary/phase_r1c_formal_summary.json"
    if final_path.exists():
        raise R1CRunnerError("R1C_FORMAL_SUMMARY_ALREADY_EXISTS")
    existing = _load_existing_raw(raw_path, binding_sha)

    planned_run_ids = {
        _run_id(stream, method)
        for stage in R1C_STAGES
        for stream in plan[stage]
        for method in _method_list(stage)
    }
    if set(existing) - planned_run_ids:
        raise R1CRunnerError("R1C_RAW_CONTAINS_UNPLANNED_RUN_ID")
    if upstream["e1"]["run_ids"] & planned_run_ids:
        raise R1CRunnerError("R1C_PLAN_OVERLAPS_E1")
    if upstream["run_ids"] & planned_run_ids:
        raise R1CRunnerError("R1C_PLAN_OVERLAPS_E2")

    started_wall = time.time()
    perf_started = time.perf_counter()
    stage_elapsed: dict[str, float] = {}
    stage_status: dict[str, str] = {}

    # Pre-start policy: no runtime-based stage deferral. E3 and E4 are both executed.
    # External interruption is recoverable because each committed run is appended to raw JSONL
    # and identified by an immutable run_id; a restart skips existing committed run_ids.
    for stage in R1C_STAGES:
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
            subset_digest = hashlib.sha256(
                "\n".join(stream.malicious_workers).encode("utf-8")
            ).hexdigest()

            for method in _method_list(stage):
                run_id = _run_id(stream, method)
                if run_id in existing:
                    continue
                base = {
                    "schema_version": "1.0",
                    "phase": "Phase-R1C",
                    "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
                    "upstream_e1_raw_sha256": UPSTREAM_E1_RAW_SHA256,
                    "upstream_r1b_start_binding_sha256": UPSTREAM_R1B_START_BINDING_SHA256,
                    "upstream_e2_raw_sha256": UPSTREAM_E2_RAW_SHA256,
                    "r1c_start_binding_sha256": binding_sha,
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
                except Exception as exc:
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
                        "phase": "Phase-R1C",
                        "r1c_start_binding_sha256": binding_sha,
                        "upstream_r1b_start_binding_sha256": UPSTREAM_R1B_START_BINDING_SHA256,
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
    deferred = R1C_PLANNED_METHOD_RUNS - len(existing)
    complete = len(existing) == R1C_PLANNED_METHOD_RUNS

    summary = {
        "schema_version": "1.0",
        "phase": "Phase-R1C",
        "status": "complete" if complete else "incomplete_unexpected",
        "formal_experiment": True,
        "r1c_start_binding_sha256": binding_sha,
        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
        "upstream_e1_raw_sha256": UPSTREAM_E1_RAW_SHA256,
        "upstream_r1b_start_binding_sha256": UPSTREAM_R1B_START_BINDING_SHA256,
        "upstream_e2_raw_sha256": UPSTREAM_E2_RAW_SHA256,
        "upstream_e2_summary_sha256": UPSTREAM_E2_SUMMARY_SHA256,
        "approval_sha256": _sha256(approval_path),
        "raw_path": str(raw_path.relative_to(root)).replace("\\", "/"),
        "raw_sha256": _sha256(raw_path),
        "planned_method_runs": R1C_PLANNED_METHOD_RUNS,
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
            "do_not_modify_or_rerun_E2",
            "do_not_delete_failures",
            "do_not_replace_unfavorable_results",
            "do_not_change_frozen_R1C_plan_after_start_approval",
            "runtime_based_stage_deferral_disabled_before_start_approval",
            "resume_same_binding_after_external_interruption_without_deleting_raw_or_checkpoint",
        ],
    }
    _atomic_json(final_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-R1C deferred E3/E4 formal runner")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--binding-manifest",
        default="results/summary/phase_r1c_start_binding_manifest.json",
    )
    parser.add_argument(
        "--approval",
        default="approvals/phase_r1c/PHASE_R1C_START_APPROVAL.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-upstream-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    if args.verify_upstream_only:
        result = verify_upstream_e2(root)
        print("R1C_UPSTREAM_E1_E2=PASS")
        print(f"UPSTREAM_E1_RAW_SHA256={result['e1']['raw_sha256']}")
        print(f"UPSTREAM_E2_RAW_SHA256={result['raw_sha256']}")
        print(f"UPSTREAM_E2_SUMMARY_SHA256={result['summary_sha256']}")
        print("UPSTREAM_E1_METHOD_RUNS=1686")
        print("UPSTREAM_E2_METHOD_RUNS=1080")
        print("E1_E2_EFFECT_STATISTICS_USED_FOR_R1C_DECISION=NO")
        return 0
    binding = root / args.binding_manifest
    if args.validate_only:
        result = validate_plan(root, binding)
        print("R1C_FORMAL_PLAN_VALIDATE=PASS")
        print(f"R1C_START_BINDING_SHA256={result['start_binding_sha256']}")
        print(f"R1C_PLANNED_METHOD_RUNS={result['method_run_count']}")
        print("E1_RERUN_METHOD_RUNS=0")
        print("E2_RERUN_METHOD_RUNS=0")
        print("FORMAL_EXPERIMENTS_RUN=0")
        return 0
    summary = execute(root, binding, root / args.approval)
    print(f"R1C_FORMAL_STATUS={summary['status']}")
    print(f"COMMITTED_METHOD_RUNS={summary['committed_method_runs']}")
    print(f"SUCCESS_METHOD_RUNS={summary['success_method_runs']}")
    print(f"FAILURE_METHOD_RUNS={summary['failure_method_runs']}")
    print(f"DEFERRED_METHOD_RUNS={summary['deferred_method_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
