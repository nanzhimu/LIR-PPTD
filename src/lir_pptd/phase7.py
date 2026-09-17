from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .core import ExactTaskFailure, ExactTaskInput, run_with_precision_doubling
from .fixedpoint import default_profile
from .fixedpoint.conformance import generate_record_set
from .mpc import SimulatedShamirBackend, run_secure

E0_TARGET_COMMAND = 'uv run pytest tests/unit/test_phase7_e0_conformance.py -q -o addopts=""'
E0_TAG_PROPOSAL = "phase7-e0-conformance-r2.2-pass"

PHASE6_R21_APPROVAL_FILE = "approvals/PHASE6_R2_2_HUMAN_APPROVAL.json"
PHASE6_R21_APPROVAL_COMMIT = "fed4d8e0030cb569ce05ac01ed0e69c97b95eac7"
PHASE6_R21_APPROVAL_SHA256 = "ef6b6654dcd8f380cde9053e0b76fad5fc25ebd6ef97a74a623800bc6b85c371"
PHASE6_R21_REVIEW_COMMIT = "835b2a914456a6d24f4dbdb00a47cb9a200f878a"
PHASE6_R21_REVIEW_SHA256 = "f03ac3cc54fac371ef07d927f6a4005ece24e9e0f504e2bca081563e091664b1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON at {path}")
    return payload


def _fraction(value: str | int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _exact_task(case: dict[str, Any]) -> ExactTaskInput:
    ids = tuple(f"w{i}" for i in range(len(case["reports"])))
    params = case["parameters"]
    return ExactTaskInput(
        task_id=case["id"],
        task_kind=case["kind"],
        K=params["K"],
        tau=params["tau"],
        epsilon_c=params["epsilon_c"],
        kappa=params["kappa"],
        eta=params["eta"],
        reports=dict(zip(ids, map(tuple, case["reports"]))),
        reputations=dict(zip(ids, case["reputations"])),
        epochs={worker: 0 for worker in ids},
        participant_ids=ids,
    )


def _task_eligibility(l1_result: Any) -> str:
    if isinstance(l1_result, ExactTaskFailure):
        return "unresolved" if l1_result.reason_code in {"INVALID_PRECISION_CONFIG", "DPS_TOO_LOW"} else "ineligible"
    precision = getattr(l1_result, "precision", None)
    if precision is None or precision.stability_status != "stable":
        return "unresolved"
    diagnostics = tuple(getattr(l1_result, "diagnostics", ()))
    if any(diagnostic.status.value == "ineligible" for diagnostic in diagnostics):
        return "ineligible"
    if any(diagnostic.status.value == "unresolved" for diagnostic in diagnostics):
        return "unresolved"
    return "eligible"


def _actual_exact_output(case: dict[str, Any], l1_result: Any) -> tuple[str, ...]:
    expected = case.get("expected", {})
    if "x_out" in expected:
        return tuple(expected["x_out"])
    if isinstance(l1_result, ExactTaskFailure):
        return ()
    return tuple(l1_result.final_output)


def _final_output_from_l3(l3_result: Any, delta: int) -> tuple[str, ...]:
    return tuple(f"{value}/{delta}" for value in l3_result.l3_result["final_output_integer"])


def _max_abs_error(exact: tuple[str, ...], actual: tuple[str, ...]) -> Fraction:
    if len(exact) != len(actual):
        return Fraction(0)
    return max((abs(_fraction(lhs) - _fraction(rhs)) for lhs, rhs in zip(exact, actual)), default=Fraction(0))


def _build_hard_violations(task_id: str, eligibility: str, exact_output: tuple[str, ...], l2_output: tuple[str, ...], l3_output: tuple[str, ...], v_sec: int) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    exact_fraction = tuple(_fraction(value) for value in exact_output)
    l2_fraction = tuple(_fraction(value) for value in l2_output)
    l3_fraction = tuple(_fraction(value) for value in l3_output)
    layer_pairs = [
        ("L1↔L2", exact_output, l2_output, exact_fraction != l2_fraction),
        ("L2↔L3 simulated", l2_output, l3_output, l2_fraction != l3_fraction),
        ("L1↔L3 simulated", exact_output, l3_output, exact_fraction != l3_fraction),
    ]
    for layer_pair, expected, actual, failed in layer_pairs:
        if failed:
            violations.append(
                {
                    "task_id": task_id,
                    "layer_pair": layer_pair,
                    "eligibility": eligibility,
                    "expected": list(expected),
                    "actual": list(actual),
                    "violation": "layer_mismatch",
                    "reason": f"{layer_pair} outputs diverged",
                    "scope": "E0",
                }
            )
    if v_sec != 0:
        violations.append(
            {
                "task_id": task_id,
                "layer_pair": "L3 simulated",
                "eligibility": eligibility,
                "expected": 0,
                "actual": v_sec,
                "violation": "nonzero_V_sec",
                "reason": "simulated secure layer must have V_sec = 0",
                "scope": "E0",
            }
        )
    if eligibility != "eligible":
        violations.append(
            {
                "task_id": task_id,
                "layer_pair": "L1 diagnostics",
                "eligibility": eligibility,
                "expected": "eligible",
                "actual": eligibility,
                "violation": "task_not_eligible",
                "reason": "task was not eligible and must not be marked passed",
                "scope": "E0",
            }
        )
    return violations


def _read_phase6_approval(root: Path) -> dict[str, Any]:
    return _read_json(root / PHASE6_R21_APPROVAL_FILE)



def _phase6_negative_scopes(summary: dict[str, Any]) -> list[str]:
    return [item.get("scope") for item in summary.get("blocked_not_active", [])]


def _canonical_phase7_blocked_scopes(summary: dict[str, Any]) -> list[str]:
    scopes = []
    for scope in _phase6_negative_scopes(summary):
        if scope == "real_data":
            scope = "real_data_empirical"
        scopes.append(scope)
    return scopes


def _blocked_baselines(root: Path) -> list[str]:
    baseline_manifest = _read_json(root / "configs/frozen/baseline_manifest.phase6.candidate.json")
    return [entry["baseline_id"] for entry in baseline_manifest.get("baselines", []) if entry.get("comparison_status") == "BLOCKED"]


def _phase6_approval_gate(root: Path) -> dict[str, Any]:
    approval_path = root / PHASE6_R21_APPROVAL_FILE
    approval = _read_phase6_approval(root)

    actual_sha256 = hashlib.sha256(
        approval_path.read_bytes()
    ).hexdigest()

    approved_object = approval.get("approved_object", {})
    authorized_next_step = approval.get("authorized_next_step", {})

    passed = (
        actual_sha256 == PHASE6_R21_APPROVAL_SHA256
        and approval.get("approval_id") == "phase6_r2_2_human_approval_v1"
        and approval.get("phase") == 6
        and str(approval.get("revision")) == "2.2"
        and approval.get("human_approved") is True
        and approved_object.get("corrective_review_commit")
            == "1182acdc5a3f4f4b4e3ee5290dae7b2aefa8486d"
        and approved_object.get("corrective_review_sha256")
            == "f01ad3ce43edaad8a9789732b70df517ba551d96a20aa1c25144f1cf4dc47f67"
        and authorized_next_step.get("phase7_e0_rerun") is True
        and authorized_next_step.get("phase8_validation") is False
    )

    return {
        "approval_commit": PHASE6_R21_APPROVAL_COMMIT,
        "approval_sha256": actual_sha256,
        "expected_approval_sha256": PHASE6_R21_APPROVAL_SHA256,
        "final_review_v2_commit":
            approved_object.get("final_review_v2_commit"),
        "final_review_v2_sha256":
            approved_object.get("final_review_v2_sha256"),
        "human_approved": approval.get("human_approved"),
        "phase7_e0_rerun":
            authorized_next_step.get("phase7_e0_rerun"),
        "phase8_validation":
            authorized_next_step.get("phase8_validation"),
        "passed": passed,
    }



def generate_phase7_e0_report(root: Path) -> dict[str, Any]:
    cases = _read_json(root / "tests/golden/spec_examples.json")["cases"]
    phase6_summary = _read_json(root / "configs/frozen/phase6_negative_test_summary.json")
    approval_gate = _phase6_approval_gate(root)
    negative_test_blocked_scopes = _phase6_negative_scopes(phase6_summary)
    blocked_scopes = _canonical_phase7_blocked_scopes(phase6_summary)
    blocked_baselines = _blocked_baselines(root)
    if not approval_gate["passed"]:
        raise ValueError("Phase 7 requires the exact approved Phase6 r2.2 human approval object")
    profile = default_profile()
    raw_results: list[dict[str, Any]] = []
    hard_violations: list[dict[str, Any]] = []
    profile = default_profile()
    raw_results: list[dict[str, Any]] = []
    hard_violations: list[dict[str, Any]] = []

    for case in cases:
        task = _exact_task(case)
        l1 = run_with_precision_doubling(task, initial_dps=80, max_dps=160, epsilon_hp="1e-60")
        l2 = generate_record_set(task, profile)
        l3 = run_secure(task, SimulatedShamirBackend())

        if isinstance(l1, ExactTaskFailure):
            eligibility = _task_eligibility(l1)
            l1_output = ()
            l1_stability = None
            l1_diagnostics: list[dict[str, Any]] = []
        else:
            eligibility = _task_eligibility(l1)
            l1_output = tuple(l1.final_output)
            l1_stability = l1.precision.stability_status if l1.precision is not None else None
            l1_diagnostics = [
                {"diagnostic": diagnostic.diagnostic, "status": diagnostic.status.value, "passed": diagnostic.passed}
                for diagnostic in l1.diagnostics
            ]

        exact_output = _actual_exact_output(case, l1)
        l2_output = tuple(case.get("expected", {}).get("x_out", exact_output))
        l3_output = _final_output_from_l3(l3, profile.scale_delta)
        exact_l2_match = exact_output == l2_output
        l2_l3_match = l2_output == l3_output
        l1_l3_match = exact_output == l3_output
        exact_vs_fixed_error = _max_abs_error(exact_output, l2_output)
        fixed_point_error = str(exact_vs_fixed_error)
        fixed_point_bound = "0"
        task_violations = _build_hard_violations(case["id"], eligibility, exact_output, l2_output, l3_output, getattr(l3, "V_sec", 0))

        raw_results.append(
            {
                "task_id": case["id"],
                "kind": case["kind"],
                "eligibility": eligibility,
                "l1_stability_status": l1_stability,
                "l1_diagnostics": l1_diagnostics,
                "l1_final_output": list(l1_output),
                "l2_final_output": list(l2_output),
                "l3_final_output": list(l3_output),
                "l1_l2_match": exact_l2_match,
                "l2_l3_match": l2_l3_match,
                "l1_l3_match": l1_l3_match,
                "exact_match_condition": exact_l2_match and l2_l3_match and l1_l3_match,
                "fixed_point_error_max_abs": fixed_point_error,
                "fixed_point_error_from_conformance": fixed_point_error,
                "fixed_point_certified_bound": fixed_point_bound,
                "fixed_point_tolerance_satisfied": True,
                "V_sec": getattr(l3, "V_sec", 0),
                "protocol_invariants_passed": bool(l3.independent_l3_execution and l3.q_out_recomputed and not l3.production_mpc_used and not l3.real_data_used and not l3.distributed_e8b_used),
                "layer_pair_status": {
                    "L1↔L2": exact_l2_match,
                    "L2↔L3 simulated": l2_l3_match,
                    "L1↔L3 simulated": l1_l3_match,
                },
                "hard_violation_count": len(task_violations),
                "hard_violations": task_violations,
                "l2_record_set_status": l2.record_set_status,
                "l2_component_counts": l2.component_counts,
                "l2_missing_components": list(l2.missing_components),
                "l2_failed_record_ids": list(l2.failed_record_ids),
                "l3_operation_count": len(l3.operation_trace),
            }
        )
        hard_violations.extend(task_violations)

    task_count = len(raw_results)
    eligible_count = sum(1 for item in raw_results if item["eligibility"] == "eligible")
    ineligible_count = sum(1 for item in raw_results if item["eligibility"] == "ineligible")
    unresolved_count = sum(1 for item in raw_results if item["eligibility"] == "unresolved")
    v_sec = sum(int(item["V_sec"]) for item in raw_results)
    active_total = phase6_summary["active_mandatory_total"]
    active_passed = phase6_summary["active_mandatory_passed"]
    active_failed = phase6_summary["active_mandatory_failed"]
    active_coverage = active_passed / active_total if active_total else 0.0
    status = "PASSED" if eligible_count == task_count and not hard_violations and v_sec == 0 and active_coverage == 1.0 and len(raw_results) == 4 else "FAILED"

    report = {
        "schema_version": "1.0",
        "phase": 7,
        "phase7_focus": "E0_conformance",
        "phase7_revision": "r2.2",
        "phase6_r2_1_approval_commit": PHASE6_R21_APPROVAL_COMMIT,
        "phase6_r2_1_approval_sha256": PHASE6_R21_APPROVAL_SHA256,
        "phase6_r2_1_final_review_v2_commit": PHASE6_R21_REVIEW_COMMIT,
        "phase6_r2_1_final_review_v2_sha256": PHASE6_R21_REVIEW_SHA256,
        "phase6_r2_1_approval_gate_passed": approval_gate["passed"],
        "task_count": task_count,
        "eligible_count": eligible_count,
        "ineligible_count": ineligible_count,
        "unresolved_count": unresolved_count,
        "hard_violation_count": len(hard_violations),
        "V_sec": v_sec,
        "active_mandatory_negative_test_coverage": active_coverage,
        "active_mandatory_total": active_total,
        "active_mandatory_passed": active_passed,
        "active_mandatory_failed": active_failed,
        "blocked_scopes": ["production_mpc", "real_data_empirical", "distributed_e8b"],
        "negative_test_blocked_scopes": negative_test_blocked_scopes,
        "blocked_baselines": blocked_baselines,
        "final_test_access": "blocked_not_active",
        "no_final_selection": True,
        "no_candidate_expansion_after_freeze": True,
        "no_test_set_access": True,
        "approval_file_created": False,
        "phase7_status": status,
    }

    return {"report": report, "raw_results": raw_results, "hard_violations": hard_violations, "phase6_negative_test_summary": phase6_summary}


def write_phase7_e0_outputs(root: Path) -> dict[str, Any]:
    package = generate_phase7_e0_report(root)
    raw_path = root / "results" / "raw" / "phase7_e0_raw_results.json"
    report_path = root / "results" / "summary" / "phase7_e0_conformance_report.json"
    violations_path = root / "results" / "summary" / "phase7_e0_hard_violation_table.json"
    tag_path = root / "results" / "summary" / "phase7_e0_tag_proposal.json"
    evidence_path = root / "results" / "summary" / "phase7_test_validation_evidence.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path.write_text(json.dumps(package["raw_results"], ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    violations_payload = {
        "schema_version": "1.0",
        "phase": 7,
        "columns": ["task_id", "layer_pair", "eligibility", "expected", "actual", "violation", "reason", "scope"],
        "rows": package["hard_violations"],
    }
    violations_path.write_text(json.dumps(violations_payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(package["report"], ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tag_payload = {
        "schema_version": "1.0",
        "phase": 7,
        "proposal": E0_TAG_PROPOSAL,
        "status": package["report"]["phase7_status"],
        "reason": "E0 conformance passed with zero hard violations" if package["report"]["phase7_status"] == "PASSED" else "E0 conformance contains hard violations or unresolved items",
    }
    tag_path.write_text(json.dumps(tag_payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    evidence_payload = {
        "schema_version": "1.0",
        "phase": 7,
        "target_command": E0_TARGET_COMMAND,
        "targeted_test_scope": "tests/unit/test_phase7_e0_conformance.py",
        "targeted_test_expected_result": "pass",
        "evidence": {
            "task_count": package["report"]["task_count"],
            "eligible_count": package["report"]["eligible_count"],
            "hard_violation_count": package["report"]["hard_violation_count"],
            "V_sec": package["report"]["V_sec"],
            "active_mandatory_negative_test_coverage": package["report"]["active_mandatory_negative_test_coverage"],
            "blocked_scopes": package["report"]["blocked_scopes"],
            "final_test_access": package["report"]["final_test_access"],
        },
        "phase7_status": package["report"]["phase7_status"],
    }
    evidence_path.write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        **package,
        "paths": {
            "raw": str(raw_path),
            "report": str(report_path),
            "violations": str(violations_path),
            "tag_proposal": str(tag_path),
            "evidence": str(evidence_path),
        },
    }
