from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lir_pptd.core.hidden_calibration import HiddenCalibrationSchedule
from lir_pptd.experiments.calibration_anchored import run_calibration_task
from lir_pptd.experiments.phase_r1.attacks import apply_response_corruption
from lir_pptd.experiments.phase_r1.longitudinal_state import initial_global_state
from lir_pptd.experiments.phase_r1.methods import run_lir_task
from lir_pptd.experiments.phase_r1.metrics import dataset_losses
from lir_pptd.experiments.phase_r1.real_data_loader import RealDataset, RealTask, load_real_dataset
from lir_pptd.experiments.phase_r1.statistics import (
    bootstrap_mean_ci,
    exact_two_sided_sign_flip,
    holm_adjust,
    paired_rank_biserial,
)

STUDY_ID = "vi_c_hidden_selector_migration_v1"
P0_1C_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

DATASETS = ("product", "duck", "dog", "weather")
RHOS = ("3/10", "1/2", "7/10")
REPLICATES = 20

FULL_METHOD = "lir_pptd"
ABLATIONS = ("no_hr", "no_cw", "no_eg")
METHODS = (FULL_METHOD,) + ABLATIONS

DISPLAY_NAMES = {
    "lir_pptd": "LIR-PPTD",
    "no_hr": "No-HR",
    "no_cw": "No-CW",
    "no_eg": "No-EG",
}

CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}

PERIOD = 20
TRAJECTORY_DATASETS = ("dog", "weather")
TRAJECTORY_RHO = "7/10"

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
P0_1C_RAW_REL = "results/p0_1c_hidden_b1_v1/formal_runs.jsonl"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"

OUT_REL = "results/vi_c_hidden_selector_migration_v1"

class VICHiddenError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stream:
    dataset_id: str
    rho: str
    replicate: int
    subset_seed: int
    malicious_workers: tuple[str, ...]

    @property
    def subset_digest(self) -> str:
        return hashlib.sha256("\n".join(self.malicious_workers).encode("utf-8")).hexdigest()

    @property
    def pairing_id(self) -> str:
        return (
            f"dataset={self.dataset_id}|rho={self.rho}|rep={self.replicate}"
            f"|subset={self.subset_digest}"
        )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    text = str(value).strip()
    if "/" in text:
        return Fraction(text)
    return Fraction(Decimal(text))


def _as_float(value: Any) -> float:
    return float(_fraction(value))


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VICHiddenError(f"MISSING_JSONL:{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_order = {d: i for i, d in enumerate(DATASETS)}
    rho_order = {r: i for i, r in enumerate(RHOS)}
    method_order = {m: i for i, m in enumerate(ABLATIONS)}
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_order[str(r["dataset"])],
            rho_order[str(r["rho"])],
            int(r["replicate"]),
            method_order[str(r["method"])],
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_datasets(root: Path) -> dict[str, RealDataset]:
    hashes = _load_json(root / DATASET_HASHES_REL)["datasets"]
    return {
        ds: load_real_dataset(
            root / "data",
            ds,
            expected_answer_sha256=hashes[ds]["answer_sha256"],
            expected_truth_sha256=hashes[ds]["truth_sha256"],
        )
        for ds in DATASETS
    }


def _plan_streams(manifest: Mapping[str, Any], datasets: Mapping[str, RealDataset]) -> list[Stream]:
    out: list[Stream] = []
    for ds in DATASETS:
        valid = set(map(str, datasets[ds].worker_ids))
        for rho in RHOS:
            rows = manifest["datasets"][ds][rho]["subsets"]
            if len(rows) != REPLICATES:
                raise VICHiddenError(f"CELL_N:{ds}:{rho}:{len(rows)}:{REPLICATES}")
            for rep, item in enumerate(rows, start=1):
                malicious = tuple(sorted(map(str, item["workers"])))
                if not set(malicious) <= valid:
                    raise VICHiddenError(f"UNKNOWN_WORKER:{ds}:{rho}:{rep}")
                out.append(
                    Stream(
                        dataset_id=ds,
                        rho=rho,
                        replicate=rep,
                        subset_seed=int(item["seed"]),
                        malicious_workers=malicious,
                    )
                )
    expected = len(DATASETS) * len(RHOS) * REPLICATES
    if len(out) != expected:
        raise VICHiddenError(f"STREAM_COUNT:{len(out)}:{expected}")
    return out


def _selector_config_hash(dataset: RealDataset) -> str:
    # Reproduce the exact selector namespace frozen by P0-1C.
    return _sha_obj({
        "study": P0_1C_STUDY_ID,
        "dataset": dataset.dataset_id,
        "answer_sha256": dataset.answer_sha256,
        "truth_sha256": dataset.truth_sha256,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "algorithm_config": CONFIG,
        "nominal_period": PERIOD,
        "selector_rule": "HMAC-SHA256 over immutable task_id; Bernoulli probability 1/P",
    })


def _load_hidden_masks(
    root: Path,
    datasets: Mapping[str, RealDataset],
) -> tuple[
    dict[str, HiddenCalibrationSchedule],
    dict[str, frozenset[str]],
    dict[str, str],
    dict[str, int],
]:
    seed_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seed_obj.get("seeds", {}).items()}
    if set(seeds) != set(DATASETS):
        raise VICHiddenError("SELECTOR_SEED_DATASET_SET_MISMATCH")

    audit_obj = _load_json(root / P0_1C_AUDIT_REL)
    expected = {str(x["dataset"]): dict(x) for x in audit_obj["selectors"]}

    selectors: dict[str, HiddenCalibrationSchedule] = {}
    masks: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}

    for ds in DATASETS:
        selector = HiddenCalibrationSchedule(
            seed=bytes.fromhex(seeds[ds]),
            session_id=f"LIR-PPTD/final-real-data/{ds}",
            selector_epoch=0,
            config_hash=_selector_config_hash(datasets[ds]),
            period=PERIOD,
        )
        task_ids = [task.task_id for task in datasets[ds].tasks]
        cal_ids = frozenset(selector.selected_task_ids(task_ids))
        mask_hash = _sha_obj(sorted(cal_ids))
        exp = expected[ds]

        if mask_hash != str(exp["calibration_mask_sha256"]):
            raise VICHiddenError(f"CALIBRATION_MASK_HASH_MISMATCH:{ds}")
        if len(cal_ids) != int(exp["calibration_task_count"]):
            raise VICHiddenError(f"CALIBRATION_COUNT_MISMATCH:{ds}")
        if selector.seed_commitment_sha256 != str(exp["seed_commitment_sha256"]):
            raise VICHiddenError(f"SEED_COMMITMENT_MISMATCH:{ds}")

        selectors[ds] = selector
        masks[ds] = cal_ids
        hashes[ds] = mask_hash
        counts[ds] = len(cal_ids)

    return selectors, masks, hashes, counts


def _load_full_reference(
    root: Path,
    streams: Sequence[Stream],
    mask_hashes: Mapping[str, str],
    calibration_counts: Mapping[str, int],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = _read_jsonl(root / P0_1C_RAW_REL)
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in rows:
        if row.get("study_id") != P0_1C_STUDY_ID:
            continue
        if row.get("method") != "lir_pptd":
            continue
        if row.get("rho") not in RHOS:
            continue
        key = (str(row["dataset"]), str(row["rho"]), int(row["replicate"]))
        if key in selected:
            raise VICHiddenError(f"DUPLICATE_FULL_REFERENCE:{key}")
        selected[key] = row

    expected = len(DATASETS) * len(RHOS) * REPLICATES
    if len(selected) != expected:
        raise VICHiddenError(f"FULL_REFERENCE_COUNT:{len(selected)}:{expected}")

    stream_by_key = {(s.dataset_id, s.rho, s.replicate): s for s in streams}
    for key, row in selected.items():
        stream = stream_by_key.get(key)
        if stream is None:
            raise VICHiddenError(f"FULL_REFERENCE_FOREIGN_STREAM:{key}")
        if row.get("status") != "success":
            raise VICHiddenError(f"FULL_REFERENCE_FAILURE:{key}")
        if str(row.get("pairing_id")) != stream.pairing_id:
            raise VICHiddenError(f"FULL_REFERENCE_PAIRING_MISMATCH:{key}")
        if tuple(sorted(map(str, row.get("malicious_workers", [])))) != stream.malicious_workers:
            raise VICHiddenError(f"FULL_REFERENCE_MALICIOUS_SET_MISMATCH:{key}")
        if str(row.get("p0_2_manifest_sha256")) != P0_2_MANIFEST_SHA256:
            raise VICHiddenError(f"FULL_REFERENCE_MANIFEST_MISMATCH:{key}")
        if str(row.get("calibration_mask_sha256")) != mask_hashes[stream.dataset_id]:
            raise VICHiddenError(f"FULL_REFERENCE_MASK_MISMATCH:{key}")
        if row.get("attack_generation_before_mode_designation") is not True:
            raise VICHiddenError(f"FULL_REFERENCE_ATTACK_ORDER_MISMATCH:{key}")
        if int(row.get("ordinary_persistent_commit_count", -1)) != 0:
            raise VICHiddenError(f"FULL_REFERENCE_ORDINARY_COMMIT:{key}")
        if int(row.get("calibration_persistent_commit_count", -1)) != calibration_counts[stream.dataset_id]:
            raise VICHiddenError(f"FULL_REFERENCE_CALIBRATION_COMMIT:{key}")

    return selected


def attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    # Critical ordering invariant: this function has no selector/mask input.
    return tuple(
        replace(
            task,
            reports=apply_response_corruption(
                task.reports,
                stream.malicious_workers,
                modality=task.modality,
                active=True,
            ),
        )
        for task in dataset.tasks
    )


def method_policy(method: str) -> dict[str, Any]:
    if method == "no_hr":
        return {
            "ordinary_ablation": "nr",
            "apply_calibration": False,
            "persist_ordinary": False,
        }
    if method == "no_cw":
        return {
            "ordinary_ablation": "ho",
            "apply_calibration": True,
            "persist_ordinary": False,
        }
    if method == "no_eg":
        return {
            "ordinary_ablation": "full",
            "apply_calibration": True,
            "persist_ordinary": True,
        }
    raise VICHiddenError(f"UNKNOWN_ABLATION:{method}")


def _group_reputation(
    state: Any,
    malicious_workers: Sequence[str],
    worker_ids: Sequence[str],
) -> tuple[float, float, float]:
    reps = dict(state.reputations or {})
    malicious = set(map(str, malicious_workers))
    honest = set(map(str, worker_ids)) - malicious
    honest_values = [_as_float(reps[w]) for w in sorted(honest)]
    malicious_values = [_as_float(reps[w]) for w in sorted(malicious)]
    if not honest_values or not malicious_values:
        raise VICHiddenError("EMPTY_REPUTATION_GROUP")
    mean_h = statistics.fmean(honest_values)
    mean_m = statistics.fmean(malicious_values)
    return mean_h, mean_m, mean_h - mean_m


def _record_prediction(
    dataset: RealDataset,
    task: RealTask,
    vector: Sequence[float],
    cls: int | None,
    pc: list[int],
    tc: list[int],
    pv: list[float],
    tv: list[float],
) -> None:
    if dataset.modality == "categorical":
        if cls is None or task.truth_class is None:
            raise VICHiddenError("CATEGORICAL_PREDICTION_MISSING")
        pc.append(int(cls))
        tc.append(int(task.truth_class))
    else:
        pv.append(float(vector[0]))
        tv.append(float(task.truth_vector[0]))


def _collect_metric(dataset: RealDataset, pc, tc, pv, tv) -> dict[str, Any]:
    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=pc if dataset.modality == "categorical" else None,
        truth_classes=tc if dataset.modality == "categorical" else None,
        predicted_values=pv if dataset.modality == "numerical" else None,
        truth_values=tv if dataset.modality == "numerical" else None,
    )
    primary_loss = float(losses["primary_loss"])
    if dataset.dataset_id == "weather":
        return {
            **losses,
            "paper_metric_name": "MAE",
            "paper_metric": primary_loss,
            "higher_is_better": False,
        }
    return {
        **losses,
        "paper_metric_name": "Accuracy" if dataset.dataset_id == "duck" else "MacroF1",
        "paper_metric": 1.0 - primary_loss,
        "higher_is_better": True,
    }


def run_ablation(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
    method: str,
) -> dict[str, Any]:
    policy = method_policy(method)
    state = initial_global_state(dataset.worker_ids, CONFIG["c0"])

    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []

    ordinary_task_count = 0
    ordinary_persistent_commits = 0
    calibration_task_count = 0
    calibration_persistent_commits = 0

    capture_trajectory = (
        dataset.dataset_id in TRAJECTORY_DATASETS
        and stream.rho == TRAJECTORY_RHO
        and method in {"no_cw", "no_eg"}
    )
    trajectory: list[dict[str, Any]] = []

    for task_index, task in enumerate(tasks, start=1):
        if task.task_id in cal_ids:
            calibration_task_count += 1

            if policy["apply_calibration"]:
                _, state = run_calibration_task(task, CONFIG, state, dps=80)
                calibration_persistent_commits += 1

            if capture_trajectory:
                mean_h, mean_m, gap = _group_reputation(
                    state,
                    stream.malicious_workers,
                    dataset.worker_ids,
                )
                trajectory.append({
                    "calibration_epoch": calibration_task_count,
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "mean_reputation_honest": mean_h,
                    "mean_reputation_malicious": mean_m,
                    "reputation_gap_honest_minus_malicious": gap,
                    "trajectory_role": (
                        "full_persistent_state_via_no_cw_equivalence"
                        if method == "no_cw"
                        else "no_evidence_gate_state"
                    ),
                })
            continue

        pred = run_lir_task(
            task,
            CONFIG,
            state,
            ablation=policy["ordinary_ablation"],
            dps=80,
        )
        ordinary_task_count += 1

        if policy["persist_ordinary"]:
            state = pred.next_state
            ordinary_persistent_commits += 1

        _record_prediction(
            dataset,
            task,
            pred.prediction_vector,
            pred.prediction_class,
            pc,
            tc,
            pv,
            tv,
        )

    if calibration_task_count != len(cal_ids):
        raise VICHiddenError(
            f"CALIBRATION_TASK_COUNT:{stream.dataset_id}:{stream.rho}:{method}:"
            f"{calibration_task_count}:{len(cal_ids)}"
        )

    if method == "no_hr":
        if calibration_persistent_commits != 0 or ordinary_persistent_commits != 0:
            raise VICHiddenError("NO_HR_PERSISTENCE_VIOLATION")
    elif method == "no_cw":
        if calibration_persistent_commits != len(cal_ids) or ordinary_persistent_commits != 0:
            raise VICHiddenError("NO_CW_PERSISTENCE_VIOLATION")
    elif method == "no_eg":
        if calibration_persistent_commits != len(cal_ids):
            raise VICHiddenError("NO_EG_CALIBRATION_PERSISTENCE_VIOLATION")
        if ordinary_persistent_commits != ordinary_task_count:
            raise VICHiddenError("NO_EG_ORDINARY_PERSISTENCE_VIOLATION")

    final_h, final_m, final_gap = _group_reputation(
        state,
        stream.malicious_workers,
        dataset.worker_ids,
    )

    return {
        **_collect_metric(dataset, pc, tc, pv, tv),
        "ordinary_task_count": ordinary_task_count,
        "calibration_task_count": calibration_task_count,
        "ordinary_persistent_commit_count": ordinary_persistent_commits,
        "calibration_persistent_commit_count": calibration_persistent_commits,
        "final_mean_reputation_honest": final_h,
        "final_mean_reputation_malicious": final_m,
        "final_reputation_gap_honest_minus_malicious": final_gap,
        "trajectory": trajectory,
    }


def _run_id(stream: Stream, method: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{STUDY_ID}|{P0_2_MANIFEST_SHA256}|{mask_hash}|{stream.pairing_id}|{method}".encode("utf-8")
    ).hexdigest()


def _mean_ci(values: Sequence[Fraction], seed_text: str) -> tuple[float, float, float]:
    mean = float(sum(values, Fraction(0, 1)) / len(values))
    low, high, _ = bootstrap_mean_ci(values, seed_text=seed_text)
    return mean, low, high


def _analyze(
    full_reference: Mapping[tuple[str, str, int], Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    success_rows = [dict(r) for r in new_rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["rho"]), int(r["replicate"]), str(r["method"])): r
        for r in success_rows
    }

    mean_rows: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    raw_p: dict[str, Fraction] = {}

    for ds in DATASETS:
        for rho in RHOS:
            reps = range(1, REPLICATES + 1)

            for method in METHODS:
                if method == FULL_METHOD:
                    rows = [full_reference[(ds, rho, rep)] for rep in reps]
                else:
                    rows = [by[(ds, rho, rep, method)] for rep in reps]
                vals = [_fraction(r["paper_metric"]) for r in rows]
                mean, low, high = _mean_ci(
                    vals,
                    seed_text=f"{STUDY_ID}|mean|{ds}|{rho}|{method}",
                )
                mean_rows.append({
                    "dataset": ds,
                    "rho": rho,
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "paper_metric": rows[0]["paper_metric_name"],
                    "n": REPLICATES,
                    "mean": mean,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                })

            for comp in ABLATIONS:
                deltas: list[Fraction] = []
                full_losses: list[float] = []
                comp_losses: list[float] = []
                for rep in reps:
                    full = full_reference[(ds, rho, rep)]
                    ablated = by[(ds, rho, rep, comp)]
                    f = _fraction(full["primary_loss"])
                    c = _fraction(ablated["primary_loss"])
                    deltas.append(f - c)
                    full_losses.append(float(f))
                    comp_losses.append(float(c))

                hid = f"{ds}|{rho}|full_vs_{comp}"
                p = exact_two_sided_sign_flip(deltas)
                raw_p[hid] = p
                low, high, seed = bootstrap_mean_ci(
                    deltas,
                    seed_text=f"{STUDY_ID}|pair|{hid}",
                )
                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                pairwise.append({
                    "hypothesis_id": hid,
                    "dataset": ds,
                    "rho": rho,
                    "comparator": comp,
                    "comparator_display_name": DISPLAY_NAMES[comp],
                    "paper_metric": full_reference[(ds, rho, 1)]["paper_metric_name"],
                    "n": REPLICATES,
                    "full_mean_loss": statistics.fmean(full_losses),
                    "comparator_mean_loss": statistics.fmean(comp_losses),
                    "mean_loss_delta_full_minus_comparator": float(mean_delta),
                    "wins": sum(d < 0 for d in deltas),
                    "ties": sum(d == 0 for d in deltas),
                    "losses": sum(d > 0 for d in deltas),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "bootstrap_seed": seed,
                    "exact_two_sided_sign_flip_p": float(p),
                    "paired_rank_biserial": float(paired_rank_biserial(deltas)),
                })

    # Preserve the final VI-C multiplicity discipline: one conservative family
    # over 4 datasets × 3 ratios × 3 ablations = 36 primary contrasts.
    adjusted = holm_adjust(raw_p)
    for row in pairwise:
        adj = adjusted[row["hypothesis_id"]]
        row["holm_family"] = "VI-C|primary|4-datasets|3-rhos|3-ablations|36"
        row["holm_family_size"] = len(raw_p)
        row["holm_adjusted_p"] = float(adj)
        delta = float(row["mean_loss_delta_full_minus_comparator"])
        row["direction"] = (
            "full_win" if delta < 0 else "full_loss" if delta > 0 else "tie"
        )
        row["holm_direction"] = (
            row["direction"] if adj <= Fraction(1, 20) and row["direction"] != "tie"
            else "not_significant"
        )

    comparison_summary: dict[str, Any] = {}
    for comp in ABLATIONS:
        rows = [r for r in pairwise if r["comparator"] == comp]
        comparison_summary[comp] = {
            "cells": len(rows),
            "mean_wins": sum(r["direction"] == "full_win" for r in rows),
            "mean_ties": sum(r["direction"] == "tie" for r in rows),
            "mean_losses": sum(r["direction"] == "full_loss" for r in rows),
            "holm_significant_wins": sum(r["holm_direction"] == "full_win" for r in rows),
            "holm_significant_losses": sum(r["holm_direction"] == "full_loss" for r in rows),
        }

    # Hidden-calibration reputation trajectories:
    # No-CW has exactly the same persistent state trajectory as Full because
    # both receive the same calibration updates and neither commits ordinary
    # candidate next_state. We therefore use No-CW state snapshots as the exact
    # Full persistent-state trajectory without rerunning Full.
    traj_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in success_rows:
        if row["dataset"] not in TRAJECTORY_DATASETS:
            continue
        if row["rho"] != TRAJECTORY_RHO:
            continue
        if row["method"] not in {"no_cw", "no_eg"}:
            continue
        role = "lir_pptd" if row["method"] == "no_cw" else "no_eg"
        for point in row.get("trajectory", []):
            traj_groups[
                (
                    str(row["dataset"]),
                    int(point["calibration_epoch"]),
                    str(point["task_id"]),
                    role,
                )
            ].append(point)

    trajectory_rows: list[dict[str, Any]] = []
    for (ds, epoch, task_id, role), points in sorted(traj_groups.items()):
        if len(points) != REPLICATES:
            raise VICHiddenError(
                f"TRAJECTORY_N:{ds}:{epoch}:{role}:{len(points)}:{REPLICATES}"
            )
        task_indices = {int(p["task_index"]) for p in points}
        if len(task_indices) != 1:
            raise VICHiddenError(f"TRAJECTORY_TASK_INDEX_MISMATCH:{ds}:{epoch}:{role}")
        task_index = next(iter(task_indices))

        for quantity in (
            "mean_reputation_honest",
            "mean_reputation_malicious",
            "reputation_gap_honest_minus_malicious",
        ):
            vals = [_fraction(p[quantity]) for p in points]
            mean, low, high = _mean_ci(
                vals,
                seed_text=f"{STUDY_ID}|traj|{ds}|{epoch}|{role}|{quantity}",
            )
            trajectory_rows.append({
                "dataset": ds,
                "rho": TRAJECTORY_RHO,
                "calibration_epoch": epoch,
                "task_index": task_index,
                "task_id": task_id,
                "method": role,
                "display_name": DISPLAY_NAMES[role],
                "quantity": quantity,
                "n": REPLICATES,
                "mean": mean,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
            })

    final_gaps: dict[str, Any] = {}
    for ds in TRAJECTORY_DATASETS:
        ds_gap_rows = [
            r
            for r in trajectory_rows
            if r["dataset"] == ds
            and r["quantity"] == "reputation_gap_honest_minus_malicious"
        ]
        if not ds_gap_rows:
            raise VICHiddenError(f"TRAJECTORY_MISSING:{ds}")
        max_epoch = max(int(r["calibration_epoch"]) for r in ds_gap_rows)
        final_gaps[ds] = {
            "calibration_epoch": max_epoch,
            "lir_pptd": next(
                float(r["mean"])
                for r in ds_gap_rows
                if int(r["calibration_epoch"]) == max_epoch and r["method"] == "lir_pptd"
            ),
            "no_eg": next(
                float(r["mean"])
                for r in ds_gap_rows
                if int(r["calibration_epoch"]) == max_epoch and r["method"] == "no_eg"
            ),
        }

    analysis = {
        "primary_contrast_count": len(pairwise),
        "holm_family_size": len(pairwise),
        "comparison_summary": comparison_summary,
        "trajectory_conditions": {
            "datasets": list(TRAJECTORY_DATASETS),
            "rho": TRAJECTORY_RHO,
        },
        "final_reputation_gaps": final_gaps,
        "scientific_decision": "REVIEW_AFTER_RESULT",
        "retuning_allowed": False,
    }
    return mean_rows, pairwise, trajectory_rows, analysis


def validate(root: Path):
    required = [
        root / P0_2_MANIFEST_REL,
        root / P0_1C_SEEDS_REL,
        root / P0_1C_AUDIT_REL,
        root / P0_1C_RAW_REL,
        root / DATASET_HASHES_REL,
        root / "src/lir_pptd/core/hidden_calibration.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise VICHiddenError("MISSING:" + ",".join(missing))

    manifest_sha = _sha256(root / P0_2_MANIFEST_REL)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise VICHiddenError(
            f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}"
        )

    manifest = _load_json(root / P0_2_MANIFEST_REL)
    datasets = _load_datasets(root)
    streams = _plan_streams(manifest, datasets)

    selectors, masks, mask_hashes, calibration_counts = _load_hidden_masks(
        root,
        datasets,
    )
    full_reference = _load_full_reference(
        root,
        streams,
        mask_hashes,
        calibration_counts,
    )

    print("VI_C_HIDDEN_SELECTOR_VALIDATE=PASS")
    print("DATASETS=product,duck,dog,weather")
    print("RHO_GRID=3/10,1/2,7/10")
    print("REPLICATES_PER_CELL=20")
    print(f"STREAMS={len(streams)}")
    print(f"REUSED_FULL_RUNS={len(full_reference)}")
    print("FULL_SOURCE=P0_1C_HIDDEN_B1_FORMAL_RESULTS")
    print("FULL_RERUN=NO")
    print(f"NEW_ABLATIONS={','.join(ABLATIONS)}")
    print(f"PLANNED_NEW_METHOD_RUNS={len(streams)*len(ABLATIONS)}")
    print("P0_1C_HIDDEN_SELECTOR_MASK_REUSE=EXACT")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("SAME_ATTACKED_STREAM_AS_FULL_REFERENCE=YES")
    print("SAME_SCORING_MASK_ALL_METHODS=YES")
    print("NO_HR_CALIBRATION_PERSISTENCE=DISABLED")
    print("NO_HR_ORDINARY_PERSISTENCE=DISABLED")
    print("NO_CW_CALIBRATION_PERSISTENCE=ENABLED")
    print("NO_CW_ORDINARY_PERSISTENCE=DISABLED")
    print("NO_EG_CALIBRATION_PERSISTENCE=ENABLED")
    print("NO_EG_ORDINARY_PERSISTENCE=ENABLED")
    print("FULL_TRAJECTORY_SOURCE=NO_CW_EXACT_PERSISTENT_STATE_EQUIVALENCE")
    print("TRAJECTORY_DATASETS=dog,weather")
    print("TRAJECTORY_RHO=7/10")
    print("HOLM_PRIMARY_FAMILY_SIZE=36")
    print("ALGORITHM_RETUNING=NO")
    print("SELECTOR_RESELECTION=NO")
    for ds in DATASETS:
        print(f"CALIBRATION_COUNT_{ds.upper()}={calibration_counts[ds]}")

    return (
        datasets,
        streams,
        selectors,
        masks,
        mask_hashes,
        calibration_counts,
        full_reference,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()

    (
        datasets,
        streams,
        selectors,
        masks,
        mask_hashes,
        calibration_counts,
        full_reference,
    ) = validate(root)

    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"

    if raw_path.exists() and not args.resume:
        raise VICHiddenError(f"OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)

    for index, stream in enumerate(streams, start=1):
        dataset = datasets[stream.dataset_id]

        # The complete attacked report sequence is created before the hidden
        # calibration mask is supplied to any ablation method.
        tasks = attacked_tasks(dataset, stream)
        cal_ids = masks[stream.dataset_id]
        mask_hash = mask_hashes[stream.dataset_id]

        for method in ABLATIONS:
            run_id = _run_id(stream, method, mask_hash)
            if run_id in existing and existing[run_id].get("status") == "success":
                continue

            started = time.perf_counter()
            try:
                result = run_ablation(
                    dataset,
                    tasks,
                    cal_ids,
                    stream,
                    method,
                )
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "subset_seed": stream.subset_seed,
                    "malicious_workers": list(stream.malicious_workers),
                    "method": method,
                    "status": "success",
                    "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
                    "calibration_mask_sha256": mask_hash,
                    "seed_commitment_sha256": selectors[
                        stream.dataset_id
                    ].seed_commitment_sha256,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
                    "same_attacked_stream_as_full_reference": True,
                    **result,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "subset_seed": stream.subset_seed,
                    "method": method,
                    "status": "failure",
                    "calibration_mask_sha256": mask_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }

            _append(raw_path, row)
            existing[run_id] = row

        if index % 10 == 0 or index == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"PROGRESS={index}/{len(streams)} "
                f"NEW_METHOD_RUNS={len(existing)}/{len(streams)*len(ABLATIONS)} "
                f"SUCCESS={success}",
                flush=True,
            )

    final_rows = list(existing.values())
    _rewrite_canonical(raw_path, final_rows)
    final_rows = list(_read_existing(raw_path).values())

    success = sum(r.get("status") == "success" for r in final_rows)
    failure = sum(r.get("status") != "success" for r in final_rows)
    planned_new = len(streams) * len(ABLATIONS)

    mean_rows: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}

    if success == planned_new and failure == 0:
        mean_rows, pairwise, trajectory_rows, analysis = _analyze(
            full_reference,
            final_rows,
        )
        _write_csv(
            outdir / "figure_vi_c_hidden_ablation_metric_means.csv",
            mean_rows,
        )
        _write_csv(
            outdir / "paired_comparisons.csv",
            pairwise,
        )
        _write_csv(
            outdir / "reputation_trajectory_means.csv",
            trajectory_rows,
        )

    audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "full_reference": {
            "source": P0_1C_RAW_REL,
            "study_id": P0_1C_STUDY_ID,
            "selected_full_rows": len(full_reference),
            "rerun": False,
        },
        "hidden_selector": {
            "same_p0_1c_masks": True,
            "nominal_rate": 1 / PERIOD,
            "same_mask_all_methods_rhos_replicates": True,
            "attack_generation_before_mode_designation": True,
            "selector_reselection": False,
        },
        "ablation_semantics": {
            "no_hr": "neutral c0 on ordinary tasks; calibration updates disabled; no persistent updates",
            "no_cw": "authenticated calibration reputation retained; ordinary q_out set to 1; ordinary candidate state not committed",
            "no_eg": "full ordinary consistency retained; ordinary candidate transitions are committed in addition to authenticated calibration transitions",
        },
        "trajectory": {
            "datasets": list(TRAJECTORY_DATASETS),
            "rho": TRAJECTORY_RHO,
            "full_state_source": "No-CW state snapshots",
            "full_no_cw_persistent_state_equivalence": True,
            "equivalence_reason": "Full and No-CW receive identical authenticated calibration transitions and neither persists ordinary candidate next_state; ordinary prediction differences therefore cannot alter persistent state.",
        },
        "multiplicity": {
            "primary_family": "all 36 Full-vs-ablation dataset×rho contrasts",
            "holm_family_size": 36,
        },
        "algorithm_retuning": False,
    }
    _atomic_json(outdir / "migration_audit.json", audit)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "datasets": list(DATASETS),
        "rho_grid": list(RHOS),
        "replicates_per_cell": REPLICATES,
        "streams": len(streams),
        "reused_full_runs": len(full_reference),
        "full_rerun": False,
        "new_methods": list(ABLATIONS),
        "planned_new_method_runs": planned_new,
        "completed_unique_new_method_runs": len(final_rows),
        "success": success,
        "failure": failure,
        "selector": {
            "reuse_p0_1c_masks": True,
            "nominal_period": PERIOD,
            "nominal_rate": 1 / PERIOD,
            "pre_freeze_mode_visible_to_attack": False,
            "seed_revealed": False,
        },
        "analysis": analysis,
        "scope": {
            "formal_vi_c_hidden_selector_migration": True,
            "algorithm_retuning": False,
            "selector_reselection": False,
            "full_model_rerun": False,
        },
    }
    _atomic_json(outdir / "formal_summary.json", summary)

    print("VI_C_HIDDEN_SELECTOR_MIGRATION=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"REUSED_FULL_RUNS={len(full_reference)}")
    print("FULL_RERUN=NO")
    print(f"PLANNED_NEW_METHOD_RUNS={planned_new}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")

    if analysis:
        for comp in ABLATIONS:
            s = analysis["comparison_summary"][comp]
            print(
                f"{comp.upper()}_FULL_MEAN_WINS_TIES_LOSSES="
                f"{s['mean_wins']}/{s['mean_ties']}/{s['mean_losses']}"
            )
            print(
                f"{comp.upper()}_FULL_HOLM_SIG_WINS_LOSSES="
                f"{s['holm_significant_wins']}/{s['holm_significant_losses']}"
            )
        for ds in TRAJECTORY_DATASETS:
            gaps = analysis["final_reputation_gaps"][ds]
            print(
                f"FINAL_REP_GAP_{ds.upper()}="
                f"FULL:{gaps['lir_pptd']},NO_EG:{gaps['no_eg']},"
                f"EPOCH:{gaps['calibration_epoch']}"
            )
        print("VI_C_SCIENTIFIC_DECISION=REVIEW_AFTER_RESULT")

    print(f"SUMMARY={outdir / 'formal_summary.json'}")
    return 0 if success == planned_new and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
