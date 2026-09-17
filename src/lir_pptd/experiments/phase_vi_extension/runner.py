from __future__ import annotations

import argparse
import dataclasses
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from fractions import Fraction
from typing import Any, Mapping, Sequence

from lir_pptd.experiments.phase_r1.formal_runner import (
    PlannedStream,
    _attack_tasks,
    _load_config,
    _load_json,
    _run_method,
    _subsets,
    load_bound_datasets,
)
from lir_pptd.experiments.phase_r1.longitudinal_state import initial_global_state
from lir_pptd.experiments.phase_r1.methods import run_lir_task


class VIExtensionError(RuntimeError):
    pass


E1_RAW_SHA = "09b7439c7f514cd6b547f249be1c3dea258489cc506ea1349f609460220714cb"
E2_RAW_SHA = "3eebb2fe187908e62325c95fb8bd4cdb49eb338712a3744d79e344092889495d"
R1C_RAW_SHA = "0b246bfc427578b163f159193ba07e967b73c2ca80ad8b812c5dc317a2f2a632"

PLAN_REL = Path("configs/phase_vi_extension/vi_c3_d_plan.json")
BINDING_REL = Path("results/summary/vi_c3_d_start_binding.json")
APPROVAL_REL = Path("approvals/phase_vi_extension/VI_C3_D_START_APPROVAL.json")
C3_RAW_REL = Path("results/raw/vi_c3_d/trajectory_runs.jsonl")
D_RAW_REL = Path("results/raw/vi_c3_d/sensitivity_runs.jsonl")
CHECKPOINT_REL = Path("results/summary/vi_c3_d_checkpoint.json")
SUMMARY_REL = Path("results/summary/vi_c3_d_formal_summary.json")

E1_RAW_REL = Path("results/raw/phase_r1/formal_runs.jsonl")
E2_RAW_REL = Path("results/raw/phase_r1b/deferred_runs.jsonl")
R1C_RAW_REL = Path("results/raw/phase_r1c/deferred_runs.jsonl")

BASE_CONFIG_REL = Path("docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json")
DATASET_MANIFEST_REL = Path("configs/phase_r1/frozen/phase_r1_dataset_manifest.json")

BOUND_SOURCE_FILES = (
    "src/lir_pptd/experiments/phase_vi_extension/runner.py",
    "src/lir_pptd/experiments/phase_r1/formal_runner.py",
    "src/lir_pptd/experiments/phase_r1/attacks.py",
    "src/lir_pptd/experiments/phase_r1/methods.py",
    "src/lir_pptd/experiments/phase_r1/longitudinal_state.py",
    "src/lir_pptd/experiments/phase_r1/real_data_loader.py",
    "src/lir_pptd/experiments/phase_r1/metrics.py",
    BASE_CONFIG_REL.as_posix(),
    DATASET_MANIFEST_REL.as_posix(),
    PLAN_REL.as_posix(),
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise VIExtensionError(f"MISSING_FILE:{path.as_posix()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise VIExtensionError(f"INVALID_JSONL:{path}:{line_no}") from exc
    return rows


def rho_fraction(rho_text: str) -> Fraction:
    """Native _subsets accepts numeric rho; preserve exact rational arithmetic."""
    try:
        return Fraction(rho_text)
    except Exception as exc:
        raise VIExtensionError(f"INVALID_RHO_RATIONAL:{rho_text}") from exc


def subset_rows(dataset, rho_text: str, requested: int):
    """
    Call the native Phase-R1 subset generator with an exact Fraction.
    Stream metadata still uses the canonical textual rho (e.g. '1/2').
    """
    return _subsets(dataset, rho_fraction(rho_text), requested)


def exact_to_float(value: Any) -> float:
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, str):
        try:
            return float(Fraction(value))
        except Exception:
            return float(value)
    return float(value)


def next_state_from_prediction(pred: Any) -> Any:
    """Resolve StatefulPrediction without assuming a single field spelling."""
    for name in ("next_state", "global_state", "state"):
        if hasattr(pred, name):
            candidate = getattr(pred, name)
            if hasattr(candidate, "reputations") or (
                isinstance(candidate, Mapping) and "reputations" in candidate
            ):
                return candidate

    if dataclasses.is_dataclass(pred):
        for field in dataclasses.fields(pred):
            candidate = getattr(pred, field.name)
            if hasattr(candidate, "reputations") or (
                isinstance(candidate, Mapping) and "reputations" in candidate
            ):
                return candidate

    if isinstance(pred, tuple):
        for candidate in reversed(pred):
            if hasattr(candidate, "reputations") or (
                isinstance(candidate, Mapping) and "reputations" in candidate
            ):
                return candidate

    raise VIExtensionError(
        "STATEFUL_PREDICTION_NEXT_STATE_NOT_FOUND:"
        + type(pred).__name__
    )


def reputations_as_float(state: Any) -> dict[str, float]:
    if hasattr(state, "reputations"):
        src = getattr(state, "reputations")
    elif isinstance(state, Mapping) and "reputations" in state:
        src = state["reputations"]
    else:
        raise VIExtensionError("GLOBAL_STATE_REPUTATIONS_NOT_FOUND")
    return {str(k): exact_to_float(v) for k, v in src.items()}


def task_participant_report_map(task: Any) -> dict[str, Any]:
    reports = getattr(task, "reports", None)
    if isinstance(reports, Mapping):
        return {str(k): v for k, v in reports.items()}

    participants = getattr(task, "participant_ids", None)
    if participants is None or reports is None:
        raise VIExtensionError("TASK_PARTICIPANT_REPORT_FIELDS_NOT_FOUND")

    participants = list(participants)
    reports = list(reports)
    if len(participants) != len(reports):
        raise VIExtensionError("TASK_PARTICIPANT_REPORT_LENGTH_MISMATCH")
    return {str(w): x for w, x in zip(participants, reports)}


def attack_phase_from_native_output(original_task: Any, attacked_task: Any, malicious: set[str]) -> str:
    """
    Infer the displayed benign/attack phase from the actual native attacked
    reports rather than reimplementing the schedule clock.
    """
    before = task_participant_report_map(original_task)
    after = task_participant_report_map(attacked_task)
    participating_malicious = [w for w in after if w in malicious]
    if not participating_malicious:
        return "none"
    changed = any(before.get(w) != after.get(w) for w in participating_malicious)
    return "attack" if changed else "benign"


def load_datasets(root: Path):
    manifest = _load_json(root / DATASET_MANIFEST_REL)
    all_ds = load_bound_datasets(root, manifest)
    return {"dog": all_ds["dog"], "weather": all_ds["weather"]}


def load_base_config(root: Path) -> dict[str, Any]:
    cfg = _load_config(root / BASE_CONFIG_REL)
    expected = {
        "config_id": "lir_cfg_05_lambda020",
        "K": 10,
        "lambda_tau": "1/5",
        "epsilon_c": "1/1024",
        "kappa": "2",
        "mu": "1/5",
        "eta": "1/10",
        "c0": "1/2",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise VIExtensionError(f"BASE_CONFIG_MISMATCH:{key}:{cfg.get(key)!r}!={value!r}")
    return cfg


def config_with(
    base: Mapping[str, Any],
    *,
    K: int | None = None,
    lambda_tau: str | None = None,
    kappa: str | None = None,
    eta: str | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base))
    if K is not None:
        cfg["K"] = int(K)
    if lambda_tau is not None:
        if not isinstance(lambda_tau, str):
            raise VIExtensionError("LAMBDA_TAU_MUST_BE_EXACT_STRING")
        cfg["lambda_tau"] = lambda_tau
    if kappa is not None:
        if not isinstance(kappa, str) or not isinstance(eta, str):
            raise VIExtensionError("KAPPA_ETA_MUST_BE_EXACT_STRINGS")
        cfg["kappa"] = kappa
        cfg["mu"] = "1/5"
        cfg["eta"] = eta
    return cfg


def sensitivity_configs(base: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for K in (1, 3, 5, 20, 30):
        out.append(("K", str(K), config_with(base, K=K)))
    for label, value in (
        ("0.05", "1/20"),
        ("0.10", "1/10"),
        ("0.40", "2/5"),
        ("0.80", "4/5"),
    ):
        out.append(("lambda_tau", label, config_with(base, lambda_tau=value)))
    for label, kappa, eta in (
        ("1", "1", "1/5"),
        ("1.5", "3/2", "2/15"),
        ("3", "3", "1/15"),
        ("4", "4", "1/20"),
    ):
        out.append(("kappa", label, config_with(base, kappa=kappa, eta=eta)))
    if len(out) != 13:
        raise VIExtensionError("SENSITIVITY_CONFIG_COUNT_ERROR")
    return out


def load_one_frozen(
    path: Path,
    *,
    stage: str,
    dataset: str,
    rho: str,
    replicate: int,
    method: str,
    schedule: tuple[int, int] | None = None,
) -> dict[str, Any]:
    matches = []
    for r in load_jsonl(path):
        if r.get("stage") != stage:
            continue
        if r.get("dataset_id") != dataset:
            continue
        if r.get("rho") != rho:
            continue
        if int(r.get("replicate_index", -999)) != replicate:
            continue
        if r.get("method") != method:
            continue
        if schedule is not None and tuple(r.get("schedule") or ()) != schedule:
            continue
        matches.append(r)
    if len(matches) != 1:
        raise VIExtensionError(
            f"FROZEN_RECORD_NOT_UNIQUE:{stage}:{dataset}:{rho}:{replicate}:{method}:{schedule}:{len(matches)}"
        )
    return matches[0]


def almost_equal(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def subset_digest(workers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(w) for w in workers)).encode("utf-8")).hexdigest()


def validate_native_semantics(root: Path, datasets, base_cfg: Mapping[str, Any]) -> None:
    if sha256(root / E1_RAW_REL) != E1_RAW_SHA:
        raise VIExtensionError("E1_RAW_SHA_MISMATCH")
    if sha256(root / E2_RAW_REL) != E2_RAW_SHA:
        raise VIExtensionError("E2_RAW_SHA_MISMATCH")
    if sha256(root / R1C_RAW_REL) != R1C_RAW_SHA:
        raise VIExtensionError("R1C_RAW_SHA_MISMATCH")

    # VI-D uses fixed-target, so validate against frozen E1 (not E3).
    for dsid in ("dog", "weather"):
        ds = datasets[dsid]
        seed, workers = subset_rows(ds, "3/10", 10)[0]
        stream = PlannedStream(
            "E1", dsid, "fixed_target", "3/10", 1, seed, workers
        )
        rec = _run_method(ds, _attack_tasks(stream, ds), "lir_pptd_full", base_cfg)
        frozen = load_one_frozen(
            root / E1_RAW_REL,
            stage="E1", dataset=dsid, rho="3/10",
            replicate=1, method="lir_pptd_full",
        )
        if not almost_equal(rec["primary_loss"], frozen["primary_loss"]):
            raise VIExtensionError(
                f"E1_FIXED_TARGET_RECONSTRUCTION_MISMATCH:{dsid}:{rec['primary_loss']}:{frozen['primary_loss']}"
            )
        if int(seed) != int(frozen["subset_seed"]):
            raise VIExtensionError(f"E1_SUBSET_SEED_RECONSTRUCTION_MISMATCH:{dsid}")
        if len(workers) != int(frozen["malicious_worker_count"]):
            raise VIExtensionError(f"E1_SUBSET_COUNT_RECONSTRUCTION_MISMATCH:{dsid}")

    # VI-C3 uses the native on-off attack path. Validate it against frozen E2.
    for dsid in ("dog", "weather"):
        ds = datasets[dsid]
        seed, workers = subset_rows(ds, "1/2", 10)[0]
        stream = PlannedStream(
            "E2", dsid, "on_off_5_5", "1/2", 1, seed, workers, schedule=(5, 5)
        )
        rec = _run_method(ds, _attack_tasks(stream, ds), "lir_pptd_full", base_cfg)
        frozen = load_one_frozen(
            root / E2_RAW_REL,
            stage="E2", dataset=dsid, rho="1/2",
            replicate=1, method="lir_pptd_full", schedule=(5, 5),
        )
        if not almost_equal(rec["primary_loss"], frozen["primary_loss"]):
            raise VIExtensionError(
                f"E2_ONOFF_RECONSTRUCTION_MISMATCH:{dsid}:{rec['primary_loss']}:{frozen['primary_loss']}"
            )
        if int(seed) != int(frozen["subset_seed"]):
            raise VIExtensionError(f"E2_SUBSET_SEED_RECONSTRUCTION_MISMATCH:{dsid}")
        if len(workers) != int(frozen["malicious_worker_count"]):
            raise VIExtensionError(f"E2_SUBSET_COUNT_RECONSTRUCTION_MISMATCH:{dsid}")


def validate_subset_extension(datasets) -> None:
    for dsid, ds in datasets.items():
        first10 = subset_rows(ds, "1/2", 10)
        first20 = subset_rows(ds, "1/2", 20)
        if first20[:10] != first10:
            raise VIExtensionError(f"SUBSET_PREFIX_MISMATCH:{dsid}")
        digests = [subset_digest(workers) for _, workers in first20]
        if len(set(digests)) != 20:
            raise VIExtensionError(f"SUBSET_NOT_UNIQUE:{dsid}")


def validate_parameter_api(datasets, base_cfg: Mapping[str, Any]) -> None:
    ds = datasets["dog"]
    task = ds.tasks[0]
    for _, _, cfg in sensitivity_configs(base_cfg):
        state = initial_global_state(ds.worker_ids, cfg["c0"])
        pred = run_lir_task(task, cfg, state)
        state2 = next_state_from_prediction(pred)
        reps = reputations_as_float(state2)
        if not reps or not all(0.0 <= x <= 1.0 for x in reps.values()):
            raise VIExtensionError("PARAMETER_API_REPUTATION_RANGE_FAILURE")


def binding_payload(root: Path) -> dict[str, Any]:
    datasets = load_datasets(root)
    base_cfg = load_base_config(root)
    files = {rel: sha256(root / rel) for rel in BOUND_SOURCE_FILES}
    files[E1_RAW_REL.as_posix()] = sha256(root / E1_RAW_REL)
    files[E2_RAW_REL.as_posix()] = sha256(root / E2_RAW_REL)
    files[R1C_RAW_REL.as_posix()] = sha256(root / R1C_RAW_REL)
    return {
        "schema_version": "1.3",
        "experiment_namespace": "VI-C3+VI-D",
        "status": "START_BINDING_READY_NOT_APPROVED",
        "plan": read_json(root / PLAN_REL),
        "base_config": base_cfg,
        "bound_files": files,
        "native_attack_path": "phase_r1.formal_runner.PlannedStream + _attack_tasks",
        "replicate_index_contract": {
            "frozen_E1_E2_E3": "1-based for non-control subsets",
            "VI-C3": "11..20",
            "VI-D": "1..10"
        },
        "runtime_gate_enabled": False,
        "planned_new_units": 540,
        "formal_runs_before_approval": 0,
    }


def compute_binding_sha(payload: Mapping[str, Any]) -> str:
    p = dict(payload)
    p.pop("start_binding_sha256", None)
    return hashlib.sha256(canonical_bytes(p)).hexdigest()


def preflight_and_bind(root: Path) -> None:
    datasets = load_datasets(root)
    base_cfg = load_base_config(root)
    validate_subset_extension(datasets)
    validate_native_semantics(root, datasets, base_cfg)
    validate_parameter_api(datasets, base_cfg)

    payload = binding_payload(root)
    payload["start_binding_sha256"] = compute_binding_sha(payload)
    path = root / BINDING_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise VIExtensionError("EXISTING_BINDING_DIFFERS_REFUSING_OVERWRITE")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("VI_C3_D_PREFLIGHT=PASS")
    print("NATIVE_ATTACK_SEMANTICS=PASS")
    print("REPLICATE_INDEX_CONTRACT=PASS")
    print("VI_D_BASELINE_SOURCE=FROZEN_E1")
    print("RUNTIME_GATE_ENABLED=NO")
    print("PLANNED_NEW_UNITS=540")
    print("FORMAL_EXPERIMENTS_RUN=0")
    print("VI_C3_D_START_BINDING_SHA256=" + payload["start_binding_sha256"])
    print("BINDING_PATH=" + BINDING_REL.as_posix())


def verify_binding(root: Path) -> dict[str, Any]:
    path = root / BINDING_REL
    if not path.is_file():
        raise VIExtensionError("START_BINDING_MISSING")
    existing = read_json(path)
    current = binding_payload(root)
    current["start_binding_sha256"] = compute_binding_sha(current)
    if current != existing:
        raise VIExtensionError("START_BINDING_DRIFT")
    return existing


def record_approval(root: Path, binding: str, approval_text: str) -> None:
    b = verify_binding(root)
    if b["start_binding_sha256"] != binding:
        raise VIExtensionError("APPROVAL_BINDING_MISMATCH")
    expected = f"APPROVE VI-C3+VI-D {binding}"
    if approval_text.strip() != expected:
        raise VIExtensionError("HUMAN_APPROVAL_TEXT_MISMATCH")
    p = root / APPROVAL_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "experiment_namespace": "VI-C3+VI-D",
        "approval_type": "formal_start_approval",
        "status": "PASS",
        "start_binding_sha256": binding,
        "human_approval_source": "explicit_terminal_entry_by_user",
        "runtime_gate_enabled": False,
        "planned_new_units": 540,
    }
    if p.exists():
        if read_json(p) != payload:
            raise VIExtensionError("EXISTING_APPROVAL_DIFFERS")
    else:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("VI_C3_D_START_APPROVAL=PASS")
    print("APPROVAL_SHA256=" + sha256(p))


def verify_approval(root: Path, binding: str) -> None:
    p = root / APPROVAL_REL
    if not p.is_file():
        raise VIExtensionError("START_APPROVAL_MISSING")
    a = read_json(p)
    if a.get("status") != "PASS" or a.get("start_binding_sha256") != binding:
        raise VIExtensionError("START_APPROVAL_INVALID")


def checkpoint(root: Path, binding: str, c3_ids: set[str], d_ids: set[str], failures: int, stage: str) -> None:
    p = root / CHECKPOINT_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema_version": "1.3",
        "experiment_namespace": "VI-C3+VI-D",
        "start_binding_sha256": binding,
        "current_stage": stage,
        "committed_vi_c3": len(c3_ids),
        "committed_vi_d": len(d_ids),
        "committed_total": len(c3_ids) + len(d_ids),
        "retained_failure_count": failures,
        "runtime_gate_enabled": False,
        "updated_unix": time.time(),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def c3_record(
    dataset,
    dataset_id: str,
    rep: int,
    seed: int,
    workers: tuple[str, ...],
    base_cfg: Mapping[str, Any],
    binding: str,
) -> dict[str, Any]:
    stream = PlannedStream(
        "VI-C3", dataset_id, "on_off_5_5", "1/2", rep, seed, workers, schedule=(5, 5)
    )
    tasks = _attack_tasks(stream, dataset)
    malicious = set(workers)
    honest = set(dataset.worker_ids) - malicious
    state = initial_global_state(dataset.worker_ids, base_cfg["c0"])
    trajectory = []
    started = time.perf_counter()

    for zero_index, task in enumerate(tasks):
        pred = run_lir_task(task, base_cfg, state)
        state = next_state_from_prediction(pred)
        reps = reputations_as_float(state)
        h = [reps[w] for w in honest if w in reps]
        m = [reps[w] for w in malicious if w in reps]
        if not h or not m:
            raise VIExtensionError(f"C3_EMPTY_GROUP:{dataset_id}:{rep}:{zero_index}")
        original_task = dataset.tasks[zero_index]
        phase = attack_phase_from_native_output(original_task, task, malicious)
        participant_map = task_participant_report_map(task)
        task_malicious_participants = sum(1 for w in participant_map if w in malicious)
        trajectory.append({
            "task_index": zero_index + 1,
            "phase": phase,
            "honest_mean_reputation": sum(h) / len(h),
            "malicious_mean_reputation": sum(m) / len(m),
            "reputation_gap": (sum(h) / len(h)) - (sum(m) / len(m)),
            "honest_count": len(h),
            "malicious_count": len(m),
            "malicious_participants": task_malicious_participants,
        })

    return {
        "schema_version": "1.3",
        "run_id": f"VI-C3|{dataset_id}|rho=1/2|schedule=5_5|rep={rep}",
        "stage": "VI-C3",
        "status": "success",
        "start_binding_sha256": binding,
        "dataset_id": dataset_id,
        "modality": dataset.modality,
        "class_count": dataset.class_count,
        "rho": "1/2",
        "replicate_index": rep,
        "subset_seed": seed,
        "malicious_worker_count": len(workers),
        "vi_extension_subset_sha256": subset_digest(workers),
        "schedule": [5, 5],
        "task_count": len(tasks),
        "elapsed_seconds": time.perf_counter() - started,
        "trajectory": trajectory,
    }


def d_record(
    dataset,
    dataset_id: str,
    rho: str,
    rep: int,
    seed: int,
    workers: tuple[str, ...],
    sweep: str,
    value: str,
    cfg: Mapping[str, Any],
    binding: str,
) -> dict[str, Any]:
    stream = PlannedStream(
        "VI-D", dataset_id, "fixed_target", rho, rep, seed, workers
    )
    started = time.perf_counter()
    result = _run_method(dataset, _attack_tasks(stream, dataset), "lir_pptd_full", cfg)
    return {
        "schema_version": "1.3",
        "run_id": f"VI-D|{dataset_id}|rho={rho}|rep={rep}|{sweep}={value}",
        "stage": "VI-D",
        "status": "success",
        "start_binding_sha256": binding,
        "dataset_id": dataset_id,
        "modality": dataset.modality,
        "class_count": dataset.class_count,
        "rho": rho,
        "replicate_index": rep,
        "subset_seed": seed,
        "malicious_worker_count": len(workers),
        "vi_extension_subset_sha256": subset_digest(workers),
        "attack_condition": "fixed_target",
        "sweep": sweep,
        "sweep_value": value,
        "effective_config": dict(cfg),
        "primary_loss": result["primary_loss"],
        "secondary_loss": result["secondary_loss"],
        "task_count": result["task_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "outer_elapsed_seconds": time.perf_counter() - started,
    }


def formal_run(root: Path) -> None:
    b = verify_binding(root)
    binding = b["start_binding_sha256"]
    verify_approval(root, binding)

    final = root / SUMMARY_REL
    if final.exists():
        s = read_json(final)
        if s.get("status") == "complete":
            print("VI_C3_D_FORMAL_ALREADY_COMPLETE=YES")
            print("SUMMARY=" + SUMMARY_REL.as_posix())
            return
        raise VIExtensionError("INCOMPLETE_SUMMARY_ALREADY_EXISTS")

    datasets = load_datasets(root)
    base_cfg = load_base_config(root)

    c3_existing = load_jsonl(root / C3_RAW_REL)
    d_existing = load_jsonl(root / D_RAW_REL)
    c3_ids = {r["run_id"] for r in c3_existing}
    d_ids = {r["run_id"] for r in d_existing}
    if len(c3_ids) != len(c3_existing) or len(d_ids) != len(d_existing):
        raise VIExtensionError("RAW_DUPLICATE_RUN_ID")
    for r in c3_existing + d_existing:
        if r.get("start_binding_sha256") != binding:
            raise VIExtensionError("RAW_BINDING_MISMATCH")

    failures = sum(r.get("status") != "success" for r in c3_existing + d_existing)
    wall0 = time.perf_counter()
    started_unix = time.time()

    print("=== VI-C3 + VI-D FORMAL RUN TO COMPLETION ===")
    print("VI_C3_D_FORMAL_START_GATE=PASS")
    print("START_BINDING_SHA256=" + binding)
    print("RUNTIME_GATE_ENABLED=NO")
    print("PLANNED_NEW_UNITS=540")
    print("RESUME_EXISTING_UNITS=" + str(len(c3_ids) + len(d_ids)))

    # VI-C3: native subset sequence entries 11..20 (1-based).
    for dsid in ("dog", "weather"):
        ds = datasets[dsid]
        subsets20 = subset_rows(ds, "1/2", 20)
        for rep, (seed, workers) in enumerate(subsets20[10:20], start=11):
            run_id = f"VI-C3|{dsid}|rho=1/2|schedule=5_5|rep={rep}"
            if run_id in c3_ids:
                continue
            try:
                rec = c3_record(ds, dsid, rep, seed, workers, base_cfg, binding)
            except Exception as exc:
                rec = {
                    "schema_version": "1.3",
                    "run_id": run_id,
                    "stage": "VI-C3",
                    "status": "failure",
                    "start_binding_sha256": binding,
                    "dataset_id": dsid,
                    "rho": "1/2",
                    "replicate_index": rep,
                    "subset_seed": seed,
                    "vi_extension_subset_sha256": subset_digest(workers),
                    "schedule": [5, 5],
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
                failures += 1
            append_jsonl(root / C3_RAW_REL, rec)
            c3_ids.add(run_id)
            checkpoint(root, binding, c3_ids, d_ids, failures, "VI-C3")
            print(f"PROGRESS VI-C3 {len(c3_ids)}/20 total={len(c3_ids)+len(d_ids)}/540 failures={failures}")

    cfgs = sensitivity_configs(base_cfg)
    for dsid in ("dog", "weather"):
        ds = datasets[dsid]
        for rho in ("3/10", "7/10"):
            subsets = subset_rows(ds, rho, 10)
            for rep, (seed, workers) in enumerate(subsets, start=1):
                for sweep, value, cfg in cfgs:
                    run_id = f"VI-D|{dsid}|rho={rho}|rep={rep}|{sweep}={value}"
                    if run_id in d_ids:
                        continue
                    try:
                        rec = d_record(ds, dsid, rho, rep, seed, workers, sweep, value, cfg, binding)
                    except Exception as exc:
                        rec = {
                            "schema_version": "1.3",
                            "run_id": run_id,
                            "stage": "VI-D",
                            "status": "failure",
                            "start_binding_sha256": binding,
                            "dataset_id": dsid,
                            "rho": rho,
                            "replicate_index": rep,
                            "subset_seed": seed,
                            "vi_extension_subset_sha256": subset_digest(workers),
                            "attack_condition": "fixed_target",
                            "sweep": sweep,
                            "sweep_value": value,
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        }
                        failures += 1
                    append_jsonl(root / D_RAW_REL, rec)
                    d_ids.add(run_id)
                    checkpoint(root, binding, c3_ids, d_ids, failures, "VI-D")
                    if len(d_ids) % 10 == 0 or len(d_ids) == 520:
                        print(f"PROGRESS VI-D {len(d_ids)}/520 total={len(c3_ids)+len(d_ids)}/540 failures={failures}")

    if len(c3_ids) != 20 or len(d_ids) != 520:
        raise VIExtensionError(f"COMPLETENESS_FAILURE:C3={len(c3_ids)}:D={len(d_ids)}")

    summary = {
        "schema_version": "1.3",
        "experiment_namespace": "VI-C3+VI-D",
        "formal_experiment": True,
        "status": "complete",
        "start_binding_sha256": binding,
        "runtime_gate_enabled": False,
        "planned_new_units": 540,
        "committed_new_units": 540,
        "success_new_units": 540 - failures,
        "failure_new_units": failures,
        "deferred_new_units": 0,
        "vi_c3": {
            "planned": 20,
            "committed": 20,
            "raw_path": C3_RAW_REL.as_posix(),
            "raw_sha256": sha256(root / C3_RAW_REL),
            "subset_replicates": "11..20",
            "attack_semantics": "native Phase-R1 _attack_tasks",
        },
        "vi_d": {
            "planned_new_method_runs": 520,
            "committed_new_method_runs": 520,
            "raw_path": D_RAW_REL.as_posix(),
            "raw_sha256": sha256(root / D_RAW_REL),
            "baseline_reused_from_frozen_E1": True,
            "frozen_e1_raw_sha256": E1_RAW_SHA,
            "new_unique_configs_per_condition": 13,
            "replicate_indices": "1..10",
        },
        "wall_elapsed_seconds_this_invocation": time.perf_counter() - wall0,
        "started_unix_this_invocation": started_unix,
    }
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    checkpoint(root, binding, c3_ids, d_ids, failures, "COMPLETE")

    print("VI_C3_D_FORMAL_STATUS=complete")
    print("COMMITTED_NEW_UNITS=540")
    print("SUCCESS_NEW_UNITS=" + str(540 - failures))
    print("FAILURE_NEW_UNITS=" + str(failures))
    print("DEFERRED_NEW_UNITS=0")
    print("RUNTIME_GATE_ENABLED=NO")
    print("C3_RAW_SHA256=" + summary["vi_c3"]["raw_sha256"])
    print("D_RAW_SHA256=" + summary["vi_d"]["raw_sha256"])
    print("SUMMARY=" + SUMMARY_REL.as_posix())
    print("VI_C3_D_FORMAL_INVOCATION=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--preflight-and-bind", action="store_true")
    ap.add_argument("--record-approval", action="store_true")
    ap.add_argument("--binding")
    ap.add_argument("--approval-text")
    ap.add_argument("--formal-run", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    try:
        if args.preflight_and_bind:
            preflight_and_bind(root)
        elif args.record_approval:
            if not args.binding or args.approval_text is None:
                raise VIExtensionError("APPROVAL_ARGUMENTS_REQUIRED")
            record_approval(root, args.binding, args.approval_text)
        elif args.formal_run:
            formal_run(root)
        else:
            raise VIExtensionError("NO_ACTION")
    except Exception as exc:
        print(f"VI_C3_D_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
