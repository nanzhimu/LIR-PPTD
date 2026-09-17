from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..baselines.crh import predict as crh_predict
from ..baselines.mean_vote import predict as mean_vote_predict
from ..core.exact import ExactTaskFailure, run_exact_once
from .phase6_exact_adapter import LongitudinalState, build_exact_task_input
from .phase6_generator import generate_stream

ALLOWED_SPLITS = {"validation", "formal_candidate"}
METHODS = ("lir_pptd_full", "mean_vote", "crh")
RECORD_STATUSES = {"success", "failure", "abort", "timeout"}


def validate_phase6_readiness(
    root: Path,
    active_fields: dict[str, Any],
    *,
    frozen_artifacts_refrozen: bool = False,
) -> dict[str, Any]:
    decision = json.loads((root / "docs/PHASE6_R2_GENERATOR_DECISIONS.json").read_text(encoding="utf-8-sig"))
    rejected_markers = set(decision["phase6_readiness_gate"]["reject_active_scope_markers"])
    allowed_blocked = set(decision["phase6_readiness_gate"]["allow_blocked_not_active_only_for"])
    failures: list[str] = []

    for field, value in active_fields.items():
        text = str(value)
        if field in allowed_blocked and text == "blocked_not_active":
            continue
        if any(marker in text for marker in rejected_markers):
            failures.append(f"{field}: unresolved active marker {text}")
        if value in (None, "", False) and field not in allowed_blocked:
            failures.append(f"{field}: missing active implementation")

    code_readiness = "passed" if not failures else "failed"
    frozen_artifact_readiness = "artifacts_refrozen" if frozen_artifacts_refrozen else "artifacts_not_yet_refrozen"
    overall = "passed" if code_readiness == "passed" and frozen_artifact_readiness == "artifacts_refrozen" else "not_ready"
    return {
        "status": code_readiness,
        "code_readiness": code_readiness,
        "frozen_artifact_readiness": frozen_artifact_readiness,
        "overall": overall,
        "failures": failures,
    }


def _candidate_config(root: Path, config_id: str) -> dict[str, Any]:
    decision = json.loads((root / "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json").read_text(encoding="utf-8-sig"))
    for candidate in decision["candidate_configs"]:
        if candidate["config_id"] == config_id:
            return candidate
    raise ValueError(f"unknown frozen candidate config: {config_id}")


def _retention_record(method: str, status: str, **fields: Any) -> dict[str, Any]:
    if status not in RECORD_STATUSES:
        raise ValueError(f"unsupported record status: {status}")
    record = {"method": method, "status": status, "retained": True}
    record.update(fields)
    return record


def _abort_record(method: str, *, reason: str, **fields: Any) -> dict[str, Any]:
    return _retention_record(method, "abort", reason=reason, **fields)


def _baseline_record(method: str, task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if method == "mean_vote":
            output = mean_vote_predict(task["reports"], task["modality"])
        elif method == "crh":
            output = crh_predict(task["reports"], task["modality"])
        else:
            raise ValueError(f"unsupported baseline: {method}")
        return _retention_record(method, "success", output=output)
    except TimeoutError as exc:
        return _retention_record(method, "timeout", error=str(exc))
    except Exception as exc:
        return _retention_record(method, "failure", error=str(exc))


def _next_state_from_result(result: Any) -> LongitudinalState:
    next_epochs = {transition.worker_id: transition.next_epoch for transition in result.reputation_transitions}
    return LongitudinalState(reputations=dict(result.next_reputation), epochs=next_epochs)


def _exact_record(
    task: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    state: LongitudinalState,
    *,
    phase: str,
) -> tuple[dict[str, Any], LongitudinalState]:
    try:
        exact_input = build_exact_task_input(task, candidate_config, state)
        result = run_exact_once(exact_input)
    except TimeoutError as exc:
        return _retention_record("lir_pptd_full", "timeout", phase=phase, error=str(exc)), state
    except Exception as exc:
        return _retention_record("lir_pptd_full", "failure", phase=phase, error=str(exc)), state
    if isinstance(result, ExactTaskFailure):
        return _retention_record(
            "lir_pptd_full",
            "failure",
            phase=phase,
            reason_code=result.reason_code,
            error=result.message,
        ), state
    return _retention_record(
        "lir_pptd_full",
        "success",
        phase=phase,
        output=list(result.final_output),
        released_class_index=result.released_class_index,
    ), _next_state_from_result(result)


def _run_exact_sequence(
    tasks: list[Mapping[str, Any]],
    candidate_config: Mapping[str, Any],
    state: LongitudinalState,
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], LongitudinalState, bool]:
    records: list[dict[str, Any]] = []
    complete = True
    for task in tasks:
        record, state = _exact_record(task, candidate_config, state, phase=phase)
        if record["status"] != "success":
            complete = False
        records.append(record)
    return records, state, complete


def run_candidate(
    root: Path,
    *,
    split: str,
    seed: int,
    family: str,
    rho: str = "0",
    attack_id: str = "no_attack",
    config_id: str = "lir_cfg_01_reference",
    fixed_target_selection: str | None = None,
    warm_prefix_length: int | None = None,
    on_off_schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("final_test and all other splits fail closed")

    candidate_config = _candidate_config(root, config_id)
    stream = generate_stream(
        root,
        split,
        seed,
        family,
        rho,
        attack_id,
        fixed_target_selection=fixed_target_selection,
        warm_prefix_length=warm_prefix_length,
        on_off_schedule=on_off_schedule,
    )

    state = LongitudinalState()
    prefix_records: list[dict[str, Any]] = []
    prefix_complete = True
    prefix_tasks = list(stream.get("warm_prefix", {}).get("tasks", []))
    if prefix_tasks:
        prefix_records, state, prefix_complete = _run_exact_sequence(prefix_tasks, candidate_config, state, phase="warm_prefix")

    scored_task_records: list[dict[str, Any]] = []
    scored_complete = True
    for index, task in enumerate(stream["scored_tasks"]):
        exact_record, state = _exact_record(task, candidate_config, state, phase="scored_suffix")
        method_records = [exact_record, _baseline_record("mean_vote", task), _baseline_record("crh", task)]
        if any(record["status"] != "success" for record in method_records):
            scored_complete = False
        scored_task_records.append(
            {
                "task_id": task["task_id"],
                "task_artifact_hash": stream["task_artifact_hashes"][index],
                "task_artifact_shared": True,
                "method_records": method_records,
            }
        )

    complete = prefix_complete and scored_complete
    return {
        "status": "completed" if complete else "failed",
        "split": split,
        "seed": seed,
        "family": family,
        "rho": rho,
        "attack_id": attack_id,
        "config_id": config_id,
        "fixed_target_selection": fixed_target_selection,
        "warm_prefix_length": warm_prefix_length,
        "prefix_task_count": len(prefix_records),
        "task_count": stream["task_count"],
        "accuracy_denominator": stream["task_count"],
        "stream_hash": stream["stream_hash"],
        "retained_failures": True,
        "prefix_records": prefix_records,
        "task_records": scored_task_records,
        "stream": stream,
        "abort_record_interface": _abort_record,
    }
