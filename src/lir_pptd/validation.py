from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .records import ApprovalRecord, Phase0Decision

REQUIRED_SPECS = (
    "docs/source_manifest.json",
    "docs/ALGORITHM_SPEC.md",
    "docs/PROTOCOL_SPEC.md",
    "docs/EXPERIMENT_SPEC.md",
    "docs/STATISTICS_SPEC.md",
    "docs/canonical_spec.json",
)
BLOCKED_SCOPES = {"production_mpc", "real_data_empirical", "distributed_e8b"}


def load_phase0_approval(root: Path) -> ApprovalRecord:
    approval = ApprovalRecord.model_validate_json(
        (root / "approvals/PHASE_0_AUTHOR_CHECKPOINT.json").read_text(encoding="utf-8")
    )
    if approval.decision is not Phase0Decision.APPROVED_WITH_DOWNSTREAM_BLOCKERS:
        raise ValueError("Phase 0 approval decision does not permit Phase 1")
    return approval


def validate_phase0_gate(root: Path) -> dict[str, Any]:
    approval = load_phase0_approval(root)
    missing = [path for path in REQUIRED_SPECS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 0 artifacts: {missing}")
    scope = json.loads((root / "docs/RESULT_SCOPE_SET.json").read_text(encoding="utf-8"))
    if scope["current_RESULT_SCOPE_SET"] != ["algorithmic_conformance"]:
        raise ValueError("Phase 1 requires algorithmic_conformance as the only current target")
    if set(scope["blocked_not_active"]) != BLOCKED_SCOPES:
        raise ValueError("required downstream scopes are not blocked_not_active")
    return {"decision": approval.decision.value, "scope": scope}


def validate_sources(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            failures.append(f"hash mismatch: {entry['path']}")
    if failures:
        raise ValueError("; ".join(failures))
    return {"validated_sources": len(manifest["files"]), "status": "passed"}


def negative_matrix_phase1_status(root: Path) -> dict[str, Any]:
    with (root / "docs/negative_test_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    blocked = [row for row in rows if row["status"] == "blocked_not_active"]
    if {row["scope"] for row in blocked} != {"production_mpc", "real_data", "distributed_e8b"}:
        raise ValueError("negative matrix downstream blocked scopes are inconsistent")
    active_phase1 = [
        row for row in rows
        if row["scope"] == "core_simulated"
        and row["rule_type"] in {"parameter_preflight"}
    ]
    return {
        "matrix_rows": len(rows),
        "active_phase1_rule_ids": [row["rule_id"] for row in active_phase1],
        "deferred_unimplemented": len(rows) - len(active_phase1) - len(blocked),
        "blocked_not_active": len(blocked),
    }
