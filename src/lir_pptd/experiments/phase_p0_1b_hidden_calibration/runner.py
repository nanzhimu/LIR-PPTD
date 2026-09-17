from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import statistics
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from lir_pptd.core.hidden_calibration import HiddenCalibrationSchedule
from lir_pptd.experiments.calibration_anchored import run_calibration_task
from lir_pptd.experiments.phase6_exact_adapter import LongitudinalState
from lir_pptd.experiments.phase_r1.attacks import (
    apply_response_corruption,
    build_unique_malicious_subsets,
)
from lir_pptd.experiments.phase_r1.longitudinal_state import initial_global_state
from lir_pptd.experiments.phase_r1.methods import StatefulPrediction, run_lir_task
from lir_pptd.experiments.phase_r1.metrics import dataset_losses
from lir_pptd.experiments.phase_r1.real_data_loader import RealDataset, RealTask, load_real_dataset

STUDY_ID = "p0_1b_hidden_calibration_evasion_probe_v1"
DATASETS = ("dog", "weather")
RHO = "7/10"
REPLICATES = 5
PERIOD = 20
SEED_START = 5001
CONDITIONS = (
    "hidden_protected",
    "hidden_mask_leaked_oracle",
    "legacy_periodic_public",
)
CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "eta": "1/10",
}
OUT_REL = Path("results/p0_1b_hidden_calibration_probe_v1")
RAW_NAME = "formal_runs.jsonl"
SUMMARY_NAME = "formal_summary.json"
PAIR_NAME = "paired_evasion_summary.csv"
AUDIT_NAME = "public_selector_audit.json"
PRIVATE_NAME = "private_selector_seeds.json"


class ProbeError(RuntimeError):
    pass


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _dataset_hashes(root: Path) -> dict[str, Any] | None:
    candidates = (
        root / "configs/calibration_final_validation/dataset_hashes.json",
        root / "configs/gold20_discrepancy/dataset_hashes.json",
    )
    for path in candidates:
        if path.is_file():
            obj = _load_json(path)
            return obj.get("datasets", obj)
    return None


def _load_datasets(root: Path) -> dict[str, RealDataset]:
    hashes = _dataset_hashes(root)
    out: dict[str, RealDataset] = {}
    for ds in DATASETS:
        if hashes is not None and ds in hashes:
            h = hashes[ds]
            out[ds] = load_real_dataset(
                root / "data",
                ds,
                expected_answer_sha256=h.get("answer_sha256"),
                expected_truth_sha256=h.get("truth_sha256"),
            )
        else:
            out[ds] = load_real_dataset(root / "data", ds)
    return out


def _participant_hash(task: RealTask) -> str:
    return _sha_obj({"task_id": task.task_id, "participants": list(task.participant_ids)})


def _periodic_ids(tasks: Sequence[RealTask]) -> tuple[str, ...]:
    return tuple(task.task_id for i, task in enumerate(tasks, start=1) if i % PERIOD == 0)


def _run_sequence(
    tasks: Sequence[RealTask],
    calibration_ids: frozenset[str],
    initial_state: LongitudinalState,
) -> tuple[tuple[StatefulPrediction, ...], tuple[str, ...], LongitudinalState, int]:
    state = initial_state
    predictions: list[StatefulPrediction] = []
    eval_ids: list[str] = []
    cal_count = 0
    for task in tasks:
        if task.task_id in calibration_ids:
            _, state = run_calibration_task(task, CONFIG, state, dps=80)
            cal_count += 1
            continue
        pred = run_lir_task(task, CONFIG, state, ablation="full", dps=80)
        predictions.append(pred)
        eval_ids.append(task.task_id)
        # Final H1 semantics: ordinary task-local counterfactual next_state is NOT persisted.
    return tuple(predictions), tuple(eval_ids), state, cal_count


def _metric(dataset: RealDataset, predictions: Sequence[StatefulPrediction], eval_ids: Sequence[str]) -> dict[str, Any]:
    by_id = {t.task_id: t for t in dataset.tasks}
    if dataset.modality == "categorical":
        pred = [int(p.prediction_class) for p in predictions]
        truth = [int(by_id[tid].truth_class) for tid in eval_ids]
        losses = dataset_losses(dataset.dataset_id, predicted_classes=pred, truth_classes=truth)
    else:
        pred = [float(p.prediction_vector[0]) for p in predictions]
        truth = [float(by_id[tid].truth_vector[0]) for tid in eval_ids]
        losses = dataset_losses(dataset.dataset_id, predicted_values=pred, truth_values=truth)
    if dataset.dataset_id == "weather":
        paper_name = "MAE"
        paper_value = float(losses["primary_loss"])
        higher_is_better = False
    else:
        paper_name = "MacroF1"
        paper_value = 1.0 - float(losses["primary_loss"])
        higher_is_better = True
    return {
        **losses,
        "paper_metric_name": paper_name,
        "paper_metric_value": paper_value,
        "higher_is_better": higher_is_better,
        "evaluation_task_count": len(eval_ids),
    }


def _numeric_reputation(value: Any) -> float:
    """Convert exact reputation values to float for reporting only.

    The longitudinal state may legitimately retain the textual prior "1/2"
    for workers that never participate in a calibration event.  Using
    float("1/2") raises ValueError, even though the exact state itself is
    valid.  Fraction handles both rational strings and ordinary decimals.
    """
    return float(Fraction(str(value)))


def _reputation_summary(dataset: RealDataset, state: LongitudinalState, malicious: frozenset[str]) -> dict[str, float]:
    reps = {str(k): _numeric_reputation(v) for k, v in dict(state.reputations or {}).items()}
    honest_vals = [reps[w] for w in dataset.worker_ids if w not in malicious]
    malicious_vals = [reps[w] for w in dataset.worker_ids if w in malicious]
    if not honest_vals or not malicious_vals:
        raise ProbeError("empty honest or malicious reputation group")
    h = statistics.fmean(honest_vals)
    m = statistics.fmean(malicious_vals)
    return {"honest_mean_reputation": h, "malicious_mean_reputation": m, "reputation_gap": h - m}


def _attack_tasks(
    dataset: RealDataset,
    malicious: frozenset[str],
    calibration_ids: frozenset[str],
    *,
    leak_designation: bool,
) -> tuple[tuple[RealTask, ...], int, int]:
    out: list[RealTask] = []
    attacked_mal_cal_reports = 0
    total_mal_cal_reports = 0
    for task in dataset.tasks:
        is_cal = task.task_id in calibration_ids
        mal_here = sum(w in malicious for w in task.participant_ids)
        if is_cal:
            total_mal_cal_reports += mal_here
        active = not (leak_designation and is_cal)
        if is_cal and active:
            attacked_mal_cal_reports += mal_here
        reports = apply_response_corruption(
            task.reports,
            malicious,
            modality=task.modality,
            active=active,
        )
        out.append(replace(task, reports=reports))
    return tuple(out), attacked_mal_cal_reports, total_mal_cal_reports


def _seed_key(dataset: str, replicate: int) -> str:
    return f"{dataset}|rep={replicate}"


def _load_or_create_private_seeds(path: Path, *, resume: bool) -> dict[str, str]:
    keys = [_seed_key(ds, rep) for ds in DATASETS for rep in range(1, REPLICATES + 1)]
    if path.is_file():
        if not resume:
            raise ProbeError(f"PRIVATE_SELECTOR_SEED_FILE_EXISTS_USE_RESUME:{path}")
        obj = _load_json(path)
        values = dict(obj.get("seeds", {}))
        missing = [k for k in keys if k not in values]
        if missing:
            raise ProbeError(f"PRIVATE_SELECTOR_SEED_MISSING:{missing[0]}")
        return values
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {k: secrets.token_hex(32) for k in keys}
    path.write_text(json.dumps({"schema_version": "1.0", "study_id": STUDY_ID, "seeds": values}, indent=2) + "\n", encoding="utf-8")
    return values


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["run_id"])] = row
    return out


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def _rewrite_canonical_runs(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write exactly one final record per run_id after resume/retry."""
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            DATASETS.index(str(r["dataset"])),
            int(r["replicate"]),
            CONDITIONS.index(str(r["condition"])),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _run_id(dataset: str, replicate: int, condition: str, mask_hash: str) -> str:
    return hashlib.sha256(f"{STUDY_ID}|{dataset}|{replicate}|{condition}|{mask_hash}".encode()).hexdigest()


def validate(root: Path) -> None:
    design = root / "configs/phase_p0_1b_hidden_calibration/design.json"
    required = (
        root / "src/lir_pptd/core/hidden_calibration.py",
        root / "src/lir_pptd/experiments/phase_p0_1b_hidden_calibration/runner.py",
        design,
    )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ProbeError("MISSING:" + ",".join(missing))
    obj = _load_json(design)
    if obj.get("study_id") != STUDY_ID:
        raise ProbeError("STUDY_ID_MISMATCH")
    print("P0_1B_HIDDEN_SELECTOR_VALIDATE=PASS")
    print("DATASETS=dog,weather")
    print(f"RHO={RHO}")
    print(f"REPLICATES={REPLICATES}")
    print(f"CONDITIONS={len(CONDITIONS)}")
    print(f"PLANNED_RUNS={len(DATASETS)*REPLICATES*len(CONDITIONS)}")
    print(f"NOMINAL_PERIOD={PERIOD}")
    print("HIDDEN_PAIR_SAME_MASK=YES")
    print("PRE_FREEZE_MODE_DISCLOSURE_PROTECTED=NO")
    print("LEAKED_ORACLE_NEGATIVE_CONTROL=YES")
    print("LEGACY_PERIODIC_DIAGNOSTIC=YES")
    print("FORMAL_B1_B2_MIGRATION=NO")


def _pair_analysis(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success = [r for r in rows if r.get("status") == "success"]
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in success:
        by_key.setdefault((r["dataset"], int(r["replicate"])), {})[r["condition"]] = r
    pairs: list[dict[str, Any]] = []
    for (ds, rep), g in sorted(by_key.items()):
        if "hidden_protected" not in g or "hidden_mask_leaked_oracle" not in g:
            continue
        a = g["hidden_protected"]
        b = g["hidden_mask_leaked_oracle"]
        if a["calibration_mask_sha256"] != b["calibration_mask_sha256"]:
            raise ProbeError(f"HIDDEN_PAIR_MASK_MISMATCH:{ds}:{rep}")
        higher = bool(a["higher_is_better"])
        metric_improvement = (float(a["paper_metric_value"]) - float(b["paper_metric_value"])) if higher else (float(b["paper_metric_value"]) - float(a["paper_metric_value"]))
        pairs.append({
            "dataset": ds,
            "replicate": rep,
            "calibration_task_count": int(a["calibration_task_count"]),
            "protected_attacked_malicious_calibration_reports": int(a["attacked_malicious_calibration_reports"]),
            "leaked_attacked_malicious_calibration_reports": int(b["attacked_malicious_calibration_reports"]),
            "protected_reputation_gap": float(a["reputation_gap"]),
            "leaked_reputation_gap": float(b["reputation_gap"]),
            "reputation_gap_gain": float(a["reputation_gap"]) - float(b["reputation_gap"]),
            "protected_metric": float(a["paper_metric_value"]),
            "leaked_metric": float(b["paper_metric_value"]),
            "metric_improvement_protected_vs_leaked": metric_improvement,
        })
    by_ds: dict[str, list[dict[str, Any]]] = {ds: [p for p in pairs if p["dataset"] == ds] for ds in DATASETS}
    ds_summary = {}
    for ds, vals in by_ds.items():
        if not vals:
            continue
        ds_summary[ds] = {
            "pairs": len(vals),
            "mean_reputation_gap_gain": statistics.fmean(v["reputation_gap_gain"] for v in vals),
            "rep_gap_gain_positive_pairs": sum(v["reputation_gap_gain"] > 0 for v in vals),
            "mean_metric_improvement": statistics.fmean(v["metric_improvement_protected_vs_leaked"] for v in vals),
            "metric_improved_pairs": sum(v["metric_improvement_protected_vs_leaked"] > 0 for v in vals),
        }
    concealment_integrity = (
        len(pairs) == len(DATASETS) * REPLICATES
        and all(p["protected_attacked_malicious_calibration_reports"] > 0 for p in pairs)
        and all(p["leaked_attacked_malicious_calibration_reports"] == 0 for p in pairs)
    )
    rep_effect = all(ds_summary.get(ds, {}).get("mean_reputation_gap_gain", 0.0) > 0 for ds in DATASETS)
    return pairs, {
        "hidden_pair_count": len(pairs),
        "concealment_integrity_pass": concealment_integrity,
        "reputation_evasion_blocking_pass": rep_effect,
        "formal_runner_migration_ready": bool(concealment_integrity and rep_effect),
        "dataset_summary": ds_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    validate(root)
    if args.validate_only:
        return 0

    outdir = root / OUT_REL
    raw_path = outdir / RAW_NAME
    private_path = outdir / PRIVATE_NAME
    if outdir.exists() and not args.resume and any((outdir / n).exists() for n in (RAW_NAME, SUMMARY_NAME, PAIR_NAME, AUDIT_NAME, PRIVATE_NAME)):
        raise ProbeError(f"FORMAL_OUTPUT_EXISTS_USE_RESUME:{outdir}")

    datasets = _load_datasets(root)
    private_seeds = _load_or_create_private_seeds(private_path, resume=args.resume)
    existing = _read_existing(raw_path)
    selector_audit: list[dict[str, Any]] = []

    for ds in DATASETS:
        dataset = datasets[ds]
        subsets = build_unique_malicious_subsets(
            dataset.worker_ids,
            dataset_id=ds,
            rho=RHO,
            requested=REPLICATES,
            seed_start=SEED_START,
        )
        for rep, (subset_seed, workers) in enumerate(subsets, start=1):
            malicious = frozenset(map(str, workers))
            seed_hex = private_seeds[_seed_key(ds, rep)]
            selector = HiddenCalibrationSchedule(
                seed=bytes.fromhex(seed_hex),
                session_id=f"{STUDY_ID}:{ds}:rep={rep}",
                selector_epoch=0,
                config_hash=_sha_obj({"study_id": STUDY_ID, "dataset": ds, "config": CONFIG, "rho": RHO}),
                period=PERIOD,
            )
            hidden_ids = frozenset(selector.selected_task_ids([t.task_id for t in dataset.tasks]))
            periodic_ids = frozenset(_periodic_ids(dataset.tasks))
            hidden_mask_hash = _sha_obj(sorted(hidden_ids))
            periodic_mask_hash = _sha_obj(sorted(periodic_ids))
            selector_audit.append({
                "dataset": ds,
                "replicate": rep,
                "seed_commitment_sha256": selector.seed_commitment_sha256,
                "retired_seed_hex": seed_hex,
                "hidden_calibration_task_count": len(hidden_ids),
                "hidden_calibration_rate": len(hidden_ids) / len(dataset.tasks),
                "hidden_calibration_mask_sha256": hidden_mask_hash,
                "legacy_periodic_task_count": len(periodic_ids),
                "legacy_periodic_mask_sha256": periodic_mask_hash,
            })

            for condition in CONDITIONS:
                if condition == "hidden_protected":
                    cal_ids = hidden_ids
                    leak = False
                    mask_hash = hidden_mask_hash
                elif condition == "hidden_mask_leaked_oracle":
                    cal_ids = hidden_ids
                    leak = True
                    mask_hash = hidden_mask_hash
                elif condition == "legacy_periodic_public":
                    cal_ids = periodic_ids
                    leak = True
                    mask_hash = periodic_mask_hash
                else:
                    raise ProbeError(condition)
                rid = _run_id(ds, rep, condition, mask_hash)
                if rid in existing and existing[rid].get("status") == "success":
                    continue
                t0 = time.perf_counter()
                try:
                    attacked, attacked_cal_reports, total_mal_cal_reports = _attack_tasks(
                        dataset, malicious, cal_ids, leak_designation=leak
                    )
                    state0 = initial_global_state(dataset.worker_ids, CONFIG["c0"])
                    preds, eval_ids, final_state, cal_count = _run_sequence(attacked, cal_ids, state0)
                    metric = _metric(dataset, preds, eval_ids)
                    rep_summary = _reputation_summary(dataset, final_state, malicious)
                    row = {
                        "schema_version": "1.0",
                        "study_id": STUDY_ID,
                        "run_id": rid,
                        "dataset": ds,
                        "rho": RHO,
                        "replicate": rep,
                        "subset_seed": subset_seed,
                        "malicious_workers": sorted(malicious),
                        "condition": condition,
                        "status": "success",
                        "nominal_period": PERIOD,
                        "calibration_task_count": cal_count,
                        "calibration_rate": cal_count / len(dataset.tasks),
                        "calibration_mask_sha256": mask_hash,
                        "seed_commitment_sha256": selector.seed_commitment_sha256 if condition.startswith("hidden") else None,
                        "pre_freeze_designation_visible_to_attacker": leak,
                        "attacked_malicious_calibration_reports": attacked_cal_reports,
                        "total_malicious_calibration_reports": total_mal_cal_reports,
                        **metric,
                        **rep_summary,
                        "elapsed_seconds": time.perf_counter() - t0,
                    }
                except Exception as exc:
                    row = {
                        "schema_version": "1.0",
                        "study_id": STUDY_ID,
                        "run_id": rid,
                        "dataset": ds,
                        "rho": RHO,
                        "replicate": rep,
                        "subset_seed": subset_seed,
                        "condition": condition,
                        "status": "failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - t0,
                    }
                _append(raw_path, row)
                existing[rid] = row
            print(f"PROGRESS={ds}:{rep}/{REPLICATES} RUNS={len(existing)}/{len(DATASETS)*REPLICATES*len(CONDITIONS)}")

    rows = list(existing.values())
    _rewrite_canonical_runs(raw_path, rows)
    pairs, decision = _pair_analysis(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / PAIR_NAME).open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(pairs[0].keys()) if pairs else ["dataset", "replicate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)
    (outdir / AUDIT_NAME).write_text(json.dumps({"schema_version": "1.0", "study_id": STUDY_ID, "selectors": selector_audit}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "planned_runs": len(DATASETS) * REPLICATES * len(CONDITIONS),
        "completed_runs": len(rows),
        "success": sum(r.get("status") == "success" for r in rows),
        "failure": sum(r.get("status") != "success" for r in rows),
        "datasets": list(DATASETS),
        "rho": RHO,
        "replicates": REPLICATES,
        "nominal_period": PERIOD,
        "decision": decision,
        "scope": {
            "probe_only": True,
            "formal_b1_b2_migration": False,
            "production_vrf": False,
            "production_mpc": False,
        },
    }
    (outdir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0_1B_HIDDEN_SELECTOR_PROBE=COMPLETE")
    print(f"PLANNED_RUNS={summary['planned_runs']}")
    print(f"SUCCESS={summary['success']}")
    print(f"FAILURE={summary['failure']}")
    print(f"HIDDEN_PAIR_COUNT={decision['hidden_pair_count']}")
    print(f"CONCEALMENT_INTEGRITY={'PASS' if decision['concealment_integrity_pass'] else 'FAIL'}")
    print(f"REPUTATION_EVASION_BLOCKING={'PASS' if decision['reputation_evasion_blocking_pass'] else 'FAIL'}")
    print(f"FORMAL_RUNNER_MIGRATION_READY={'YES' if decision['formal_runner_migration_ready'] else 'NO'}")
    print(f"SUMMARY={OUT_REL / SUMMARY_NAME}")
    return 0 if summary["failure"] == 0 and decision["formal_runner_migration_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
