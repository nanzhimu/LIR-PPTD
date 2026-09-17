from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


class Phase6ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Phase6CandidatePackage:
    evaluation_task_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    attack_manifest: dict[str, Any]
    baseline_manifest: dict[str, Any]
    ablation_manifest: dict[str, Any]
    fairness_manifest: dict[str, Any]
    fault_profiles: dict[str, Any]
    seed_manifest: dict[str, Any]
    generation_config: dict[str, Any]
    negative_summary: dict[str, Any]


def _read_json(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise Phase6ManifestError(f"missing candidate artifact: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase6ManifestError(f"candidate artifact must be an object: {relative_path}")
    return value


def _sha256(root: Path, relative_path: str) -> str:
    return sha256((root / relative_path).read_bytes()).hexdigest()


def _require(value: Any, path: str) -> None:
    if value is None or value == "" or value == "placeholder_pending_hash":
        raise Phase6ManifestError(f"unresolved candidate field: {path}")


def load_phase6_candidate_package(root: Path) -> Phase6CandidatePackage:
    package = Phase6CandidatePackage(
        evaluation_task_manifest=_read_json(root, "configs/frozen/evaluation_task_manifest.json"),
        dataset_manifest=_read_json(root, "configs/frozen/dataset_manifest.phase6.synthetic.json"),
        attack_manifest=_read_json(root, "configs/frozen/attack_manifest.phase6.candidate.json"),
        baseline_manifest=_read_json(root, "configs/frozen/baseline_manifest.phase6.candidate.json"),
        ablation_manifest=_read_json(root, "configs/frozen/ablation_manifest.phase6.candidate.json"),
        fairness_manifest=_read_json(root, "configs/frozen/comparison_fairness_manifest.json"),
        fault_profiles=_read_json(root, "configs/frozen/fault_profiles.phase6.candidate.json"),
        seed_manifest=_read_json(root, "configs/frozen/synthetic_seed_manifest.phase6.candidate.json"),
        generation_config=_read_json(root, "configs/frozen/synthetic_generation_config.phase6.candidate.json"),
        negative_summary=_read_json(root, "configs/frozen/phase6_negative_test_summary.json"),
    )
    validate_phase6_candidate_package(root, package)
    return package


def validate_phase6_candidate_package(root: Path, package: Phase6CandidatePackage) -> None:
    tasks = package.evaluation_task_manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise Phase6ManifestError("evaluation task manifest has no tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise Phase6ManifestError("evaluation task IDs must be unique strings")
    recovery = next((task for task in tasks if task.get("task_id") == "synthetic_recovery"), None)
    if recovery is None or recovery.get("attack_family") != "not_applicable":
        raise Phase6ManifestError("recovery task must not use an attack family")
    if recovery.get("fault_profile_id") != "phase5_e8a_frozen_fault_profile":
        raise Phase6ManifestError("recovery task has an undefined fault profile")

    fault_profile = next((item for item in package.fault_profiles.get("fault_profiles", []) if item.get("fault_profile_id") == recovery["fault_profile_id"]), None)
    if fault_profile is None or fault_profile.get("backend") != "sqlite_wal_single_process":
        raise Phase6ManifestError("E8-A fault profile is missing or not SQLite WAL")
    if set(fault_profile.get("explicitly_not_extended_to", [])) != {"production_mpc", "real_data_empirical", "distributed_e8b"}:
        raise Phase6ManifestError("E8-A fault profile boundary is too broad")

    expected_task_hash = package.fairness_manifest.get("evaluation_task_manifest_hash")
    _require(expected_task_hash, "comparison_fairness_manifest.evaluation_task_manifest_hash")
    actual_task_hash = _sha256(root, "configs/frozen/evaluation_task_manifest.json")
    if expected_task_hash != actual_task_hash:
        raise Phase6ManifestError("evaluation task manifest hash mismatch")

    attack = package.attack_manifest
    for field in ("eligible_threshold",):
        _require(attack.get(field), f"attack_manifest.{field}")
    for name, entry in attack.get("fixed_targets", {}).items():
        if entry.get("status") != "blocked_requires_human_parameter_decision":
            _require(entry.get("values"), f"attack_manifest.fixed_targets.{name}")
    for name in ("duty_cycle", "recovery_neighborhood", "hold_length", "censoring_horizon"):
        entry = attack.get("on_off", {}).get(name)
        if not isinstance(entry, dict) or entry.get("value") is not None and entry.get("status") == "blocked_requires_human_parameter_decision":
            raise Phase6ManifestError(f"attack boundary is malformed: {name}")
        if entry.get("status") != "blocked_requires_human_parameter_decision":
            _require(entry.get("value"), f"attack_manifest.on_off.{name}")

    seeds = package.seed_manifest
    validation = set(seeds.get("seed_sets", {}).get("validation", []))
    formal = set(seeds.get("seed_sets", {}).get("formal_candidate", []))
    final_test = seeds.get("seed_sets", {}).get("final_test", [])
    if not validation or not formal or validation & formal or final_test:
        raise Phase6ManifestError("synthetic seed sets overlap or expose final-test seeds")
    config_hash = seeds.get("generator", {}).get("config_sha256")
    _require(config_hash, "synthetic_seed_manifest.generator.config_sha256")
    if config_hash != _sha256(root, "configs/frozen/synthetic_generation_config.phase6.candidate.json"):
        raise Phase6ManifestError("synthetic generation config hash mismatch")

    for method in package.fairness_manifest.get("methods", []):
        if method.get("comparison_status") == "eligible_primary":
            if method.get("evaluated_configs") != 1 and method.get("method_id") not in {"lir_pptd_full"}:
                raise Phase6ManifestError(f"eligible baseline must use one frozen config: {method.get('method_id')}")
            if method.get("compute_budget", {}).get("seed_repeats") != 30:
                raise Phase6ManifestError(f"eligible method seed count is not 30: {method.get('method_id')}")
    if package.negative_summary.get("active_mandatory_total") != package.negative_summary.get("active_mandatory_passed") or package.negative_summary.get("active_mandatory_failed") != 0:
        raise Phase6ManifestError("active mandatory negative-test coverage is not 100 percent")
    if (root / "approvals/PHASE_6_APPROVAL.json").exists():
        raise Phase6ManifestError("approved Phase 6 approval file must be created by a human")


def run_phase6_candidate_validation(root: Path) -> dict[str, Any]:
    package = load_phase6_candidate_package(root)
    return {
        "status": "validated_candidate_only",
        "phase7_authorized": False,
        "test_set_access": "prohibited",
        "final_selection": "prohibited",
        "blocked_not_active": ["production_mpc", "real_data_empirical", "distributed_e8b"],
        "active_mandatory_negative_test_coverage": "100%",
        "task_count": len(package.evaluation_task_manifest["tasks"]),
    }
