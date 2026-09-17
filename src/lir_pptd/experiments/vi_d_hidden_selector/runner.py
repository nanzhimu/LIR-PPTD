from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
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
from lir_pptd.experiments.phase_r1.statistics import bootstrap_mean_ci

STUDY_ID = "vi_d_hidden_selector_migration_v1"
P0_1C_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

DATASETS = ("dog", "weather")
RHOS = ("3/10", "7/10")
REPLICATES = 20

BASE_CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}

SWEEPS = {
    "K": ("1", "3", "5", "10", "20", "30"),
    "lambda_tau": ("1/20", "1/10", "1/5", "2/5", "4/5"),
    "kappa": ("1", "2", "4"),
    "calibration_period": ("10", "20", "40"),
}
DEFAULT_VALUES = {
    "K": "10",
    "lambda_tau": "1/5",
    "kappa": "2",
    "calibration_period": "20",
}
PERIODS = (10, 20, 40)
DEFAULT_PERIOD = 20
MU_FIXED = Fraction(1, 5)

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
P0_1C_RAW_REL = "results/p0_1c_hidden_b1_v1/formal_runs.jsonl"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"

OUT_REL = "results/vi_d_hidden_selector_migration_v1"


class VIDHiddenError(RuntimeError):
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


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


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
        raise VIDHiddenError(f"MISSING_JSONL:{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_order = {d: i for i, d in enumerate(DATASETS)}
    rho_order = {r: i for i, r in enumerate(RHOS)}
    sweep_order = {s: i for i, s in enumerate(SWEEPS)}
    value_order = {
        sweep: {v: i for i, v in enumerate(values)}
        for sweep, values in SWEEPS.items()
    }
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_order[str(r["dataset"])],
            rho_order[str(r["rho"])],
            int(r["replicate"]),
            sweep_order[str(r["sweep"])],
            value_order[str(r["sweep"])][str(r["value"])],
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


def _plan_streams(
    manifest: Mapping[str, Any],
    datasets: Mapping[str, RealDataset],
) -> list[Stream]:
    out: list[Stream] = []
    for ds in DATASETS:
        valid = set(map(str, datasets[ds].worker_ids))
        for rho in RHOS:
            rows = manifest["datasets"][ds][rho]["subsets"]
            if len(rows) != REPLICATES:
                raise VIDHiddenError(f"CELL_N:{ds}:{rho}:{len(rows)}:{REPLICATES}")
            for rep, item in enumerate(rows, start=1):
                malicious = tuple(sorted(map(str, item["workers"])))
                if not set(malicious) <= valid:
                    raise VIDHiddenError(f"UNKNOWN_WORKER:{ds}:{rho}:{rep}")
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
        raise VIDHiddenError(f"STREAM_COUNT:{len(out)}:{expected}")
    return out


def _selector_config_hash(dataset: RealDataset) -> str:
    # Exact P0-1C PRF-score namespace.  Sensitivity sweeps deliberately hold
    # the selector score fixed.  The calibration probability P is separately
    # bound by HiddenCalibrationSchedule's public commitment.
    return _sha_obj({
        "study": P0_1C_STUDY_ID,
        "dataset": dataset.dataset_id,
        "answer_sha256": dataset.answer_sha256,
        "truth_sha256": dataset.truth_sha256,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "algorithm_config": BASE_CONFIG,
        "nominal_period": DEFAULT_PERIOD,
        "selector_rule": "HMAC-SHA256 over immutable task_id; Bernoulli probability 1/P",
    })


def _load_selector_family(
    root: Path,
    datasets: Mapping[str, RealDataset],
) -> tuple[
    dict[str, dict[int, HiddenCalibrationSchedule]],
    dict[str, dict[int, frozenset[str]]],
    dict[str, dict[int, str]],
    dict[str, Any],
]:
    seed_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seed_obj.get("seeds", {}).items()}
    if not set(DATASETS) <= set(seeds):
        raise VIDHiddenError("SELECTOR_SEEDS_MISSING_REQUIRED_DATASET")

    audit_obj = _load_json(root / P0_1C_AUDIT_REL)
    expected_p20 = {str(x["dataset"]): dict(x) for x in audit_obj["selectors"]}

    selectors: dict[str, dict[int, HiddenCalibrationSchedule]] = {}
    masks: dict[str, dict[int, frozenset[str]]] = {}
    hashes: dict[str, dict[int, str]] = {}
    audit: dict[str, Any] = {}

    for ds in DATASETS:
        selectors[ds] = {}
        masks[ds] = {}
        hashes[ds] = {}
        task_ids = [task.task_id for task in datasets[ds].tasks]
        config_hash = _selector_config_hash(datasets[ds])

        for period in PERIODS:
            selector = HiddenCalibrationSchedule(
                seed=bytes.fromhex(seeds[ds]),
                session_id=f"LIR-PPTD/final-real-data/{ds}",
                selector_epoch=0,
                config_hash=config_hash,
                period=period,
            )
            cal_ids = frozenset(selector.selected_task_ids(task_ids))
            if not cal_ids:
                raise VIDHiddenError(f"ZERO_CALIBRATION_TASKS:{ds}:P{period}")
            selectors[ds][period] = selector
            masks[ds][period] = cal_ids
            hashes[ds][period] = _sha_obj(sorted(cal_ids))

        if not masks[ds][40] <= masks[ds][20] <= masks[ds][10]:
            raise VIDHiddenError(f"NESTING_VIOLATION:{ds}")

        exp = expected_p20[ds]
        if hashes[ds][20] != str(exp["calibration_mask_sha256"]):
            raise VIDHiddenError(f"P20_MASK_NOT_EXACT_P0_1C:{ds}")
        if selectors[ds][20].seed_commitment_sha256 != str(exp["seed_commitment_sha256"]):
            raise VIDHiddenError(f"P20_COMMITMENT_NOT_EXACT_P0_1C:{ds}")

        common_scored = len(set(task_ids) - masks[ds][10])
        audit[ds] = {
            "task_count": len(task_ids),
            "selector_score_namespace": "exact P0-1C PRF namespace",
            "nested": True,
            "common_scoring_mask": "complement of P10 calibration set",
            "common_scored_task_count": common_scored,
            "periods": {
                str(period): {
                    "nominal_probability": 1 / period,
                    "calibration_task_count": len(masks[ds][period]),
                    "realized_rate": len(masks[ds][period]) / len(task_ids),
                    "calibration_mask_sha256": hashes[ds][period],
                    "seed_commitment_sha256": selectors[ds][period].seed_commitment_sha256,
                }
                for period in PERIODS
            },
        }

    return selectors, masks, hashes, audit


def _load_full_reference(
    root: Path,
    streams: Sequence[Stream],
    mask_hashes: Mapping[str, Mapping[int, str]],
    calibration_masks: Mapping[str, Mapping[int, frozenset[str]]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = _read_jsonl(root / P0_1C_RAW_REL)
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in rows:
        if row.get("study_id") != P0_1C_STUDY_ID:
            continue
        if row.get("method") != "lir_pptd":
            continue
        if row.get("dataset") not in DATASETS or row.get("rho") not in RHOS:
            continue
        key = (str(row["dataset"]), str(row["rho"]), int(row["replicate"]))
        if key in selected:
            raise VIDHiddenError(f"DUPLICATE_FULL_REFERENCE:{key}")
        selected[key] = row

    expected = len(DATASETS) * len(RHOS) * REPLICATES
    if len(selected) != expected:
        raise VIDHiddenError(f"FULL_REFERENCE_COUNT:{len(selected)}:{expected}")

    planned = {(s.dataset_id, s.rho, s.replicate): s for s in streams}
    for key, row in selected.items():
        stream = planned.get(key)
        if stream is None:
            raise VIDHiddenError(f"FOREIGN_FULL_REFERENCE:{key}")
        if row.get("status") != "success":
            raise VIDHiddenError(f"FAILED_FULL_REFERENCE:{key}")
        if str(row.get("pairing_id")) != stream.pairing_id:
            raise VIDHiddenError(f"FULL_PAIRING_MISMATCH:{key}")
        if tuple(sorted(map(str, row.get("malicious_workers", [])))) != stream.malicious_workers:
            raise VIDHiddenError(f"FULL_MALICIOUS_SET_MISMATCH:{key}")
        if str(row.get("calibration_mask_sha256")) != mask_hashes[stream.dataset_id][20]:
            raise VIDHiddenError(f"FULL_P20_MASK_MISMATCH:{key}")
        if row.get("attack_generation_before_mode_designation") is not True:
            raise VIDHiddenError(f"FULL_ATTACK_ORDER_MISMATCH:{key}")
        if int(row.get("ordinary_persistent_commit_count", -1)) != 0:
            raise VIDHiddenError(f"FULL_ORDINARY_COMMIT_MISMATCH:{key}")
        if int(row.get("calibration_persistent_commit_count", -1)) != len(
            calibration_masks[stream.dataset_id][20]
        ):
            raise VIDHiddenError(f"FULL_CALIBRATION_COMMIT_MISMATCH:{key}")

    return selected


def attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    # No calibration mask, selector, period, or swept parameter is visible here.
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


def config_for_point(sweep: str, value: str) -> dict[str, Any]:
    cfg = dict(BASE_CONFIG)
    if sweep == "K":
        cfg["K"] = int(value)
    elif sweep == "lambda_tau":
        cfg["lambda_tau"] = value
    elif sweep == "kappa":
        kappa = Fraction(value)
        cfg["kappa"] = _fraction_text(kappa)
        cfg["eta"] = _fraction_text(MU_FIXED / kappa)
    elif sweep == "calibration_period":
        pass
    else:
        raise VIDHiddenError(f"UNKNOWN_SWEEP:{sweep}")
    return cfg


def calibration_period_for_point(sweep: str, value: str) -> int:
    return int(value) if sweep == "calibration_period" else DEFAULT_PERIOD


def planned_new_points() -> tuple[tuple[str, str], ...]:
    points: list[tuple[str, str]] = []
    for sweep, values in SWEEPS.items():
        for value in values:
            # K/lambda/kappa defaults are exact P0-1C Full references.
            # P=20 must be rerun because P-sensitivity uses the common P10
            # scoring mask, which differs from the normal P20 scoring set.
            if sweep != "calibration_period" and value == DEFAULT_VALUES[sweep]:
                continue
            points.append((sweep, value))
    return tuple(points)


def _metric(
    dataset: RealDataset,
    pc: list[int],
    tc: list[int],
    pv: list[float],
    tv: list[float],
) -> dict[str, Any]:
    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=pc if dataset.modality == "categorical" else None,
        truth_classes=tc if dataset.modality == "categorical" else None,
        predicted_values=pv if dataset.modality == "numerical" else None,
        truth_values=tv if dataset.modality == "numerical" else None,
    )
    primary_loss = float(losses["primary_loss"])
    if dataset.dataset_id == "dog":
        return {
            **losses,
            "paper_metric_name": "MacroF1",
            "paper_metric": 1.0 - primary_loss,
            "higher_is_better": True,
        }
    return {
        **losses,
        "paper_metric_name": "MAE",
        "paper_metric": primary_loss,
        "higher_is_better": False,
    }


def _group_reputation(
    state: Any,
    malicious_workers: Sequence[str],
    worker_ids: Sequence[str],
) -> tuple[float, float, float]:
    reps = dict(state.reputations or {})
    malicious = set(map(str, malicious_workers))
    honest = set(map(str, worker_ids)) - malicious
    h = [float(_fraction(reps[w])) for w in sorted(honest)]
    m = [float(_fraction(reps[w])) for w in sorted(malicious)]
    if not h or not m:
        raise VIDHiddenError("EMPTY_REPUTATION_GROUP")
    mean_h = statistics.fmean(h)
    mean_m = statistics.fmean(m)
    return mean_h, mean_m, mean_h - mean_m


def run_point(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    stream: Stream,
    sweep: str,
    value: str,
    masks: Mapping[int, frozenset[str]],
) -> dict[str, Any]:
    cfg = config_for_point(sweep, value)
    period = calibration_period_for_point(sweep, value)
    cal_ids = masks[period]

    # For P sensitivity, all P values must be evaluated on the same task set.
    # Nested masks make the largest calibration set P10 the union.
    common_score_ids = (
        frozenset(task.task_id for task in tasks if task.task_id not in masks[10])
        if sweep == "calibration_period"
        else frozenset(task.task_id for task in tasks if task.task_id not in masks[20])
    )

    state = initial_global_state(dataset.worker_ids, cfg["c0"])
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    ordinary_count = 0
    calibration_count = 0
    scored_count = 0

    for task in tasks:
        if task.task_id in cal_ids:
            calibration_count += 1
            _, state = run_calibration_task(task, cfg, state, dps=80)
            continue

        pred = run_lir_task(task, cfg, state, ablation="full", dps=80)
        ordinary_count += 1

        # Evidence Gate: ordinary candidate next_state is never committed.
        if task.task_id not in common_score_ids:
            continue

        scored_count += 1
        if dataset.modality == "categorical":
            if pred.prediction_class is None or task.truth_class is None:
                raise VIDHiddenError("CATEGORICAL_PREDICTION_MISSING")
            pc.append(int(pred.prediction_class))
            tc.append(int(task.truth_class))
        else:
            pv.append(float(pred.prediction_vector[0]))
            tv.append(float(task.truth_vector[0]))

    if calibration_count != len(cal_ids):
        raise VIDHiddenError(
            f"CALIBRATION_COUNT:{dataset.dataset_id}:{sweep}:{value}:"
            f"{calibration_count}:{len(cal_ids)}"
        )

    mean_h, mean_m, gap = _group_reputation(
        state,
        stream.malicious_workers,
        dataset.worker_ids,
    )
    return {
        **_metric(dataset, pc, tc, pv, tv),
        "sweep": sweep,
        "value": value,
        "config": cfg,
        "nominal_calibration_period": period,
        "nominal_calibration_probability": 1 / period,
        "calibration_task_count": calibration_count,
        "ordinary_task_count": ordinary_count,
        "scored_task_count": scored_count,
        "scoring_mask": (
            "common_complement_of_P10"
            if sweep == "calibration_period"
            else "ordinary_under_exact_P20_mask"
        ),
        "ordinary_persistent_commit_count": 0,
        "calibration_persistent_commit_count": calibration_count,
        "final_mean_reputation_honest": mean_h,
        "final_mean_reputation_malicious": mean_m,
        "final_reputation_gap_honest_minus_malicious": gap,
    }


def _run_id(
    stream: Stream,
    sweep: str,
    value: str,
    mask_hash: str,
) -> str:
    return hashlib.sha256(
        (
            f"{STUDY_ID}|{P0_2_MANIFEST_SHA256}|{mask_hash}|"
            f"{stream.pairing_id}|sweep={sweep}|value={value}"
        ).encode("utf-8")
    ).hexdigest()


def _loss_from_metric(dataset: str, metric: float) -> float:
    return 1.0 - metric if dataset == "dog" else metric


def _analyze(
    full_reference: Mapping[tuple[str, str, int], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    success_rows = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (
            str(r["dataset"]),
            str(r["rho"]),
            int(r["replicate"]),
            str(r["sweep"]),
            str(r["value"]),
        ): r
        for r in success_rows
    }

    mean_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for ds in DATASETS:
        for rho in RHOS:
            for sweep, values in SWEEPS.items():
                if sweep == "calibration_period":
                    default_records = [
                        by[(ds, rho, rep, sweep, DEFAULT_VALUES[sweep])]
                        for rep in range(1, REPLICATES + 1)
                    ]
                    default_metrics = [float(r["paper_metric"]) for r in default_records]
                else:
                    default_records = [
                        full_reference[(ds, rho, rep)]
                        for rep in range(1, REPLICATES + 1)
                    ]
                    default_metrics = [float(r["paper_metric"]) for r in default_records]

                default_losses = [_loss_from_metric(ds, x) for x in default_metrics]

                for value in values:
                    is_default = value == DEFAULT_VALUES[sweep]
                    if sweep != "calibration_period" and is_default:
                        records = [
                            full_reference[(ds, rho, rep)]
                            for rep in range(1, REPLICATES + 1)
                        ]
                        metrics = [float(r["paper_metric"]) for r in records]
                        source = "reused_p0_1c_full"
                        gaps: list[float] = []
                    else:
                        records = [
                            by[(ds, rho, rep, sweep, value)]
                            for rep in range(1, REPLICATES + 1)
                        ]
                        metrics = [float(r["paper_metric"]) for r in records]
                        source = "new_hidden_selector_run"
                        gaps = [
                            float(r["final_reputation_gap_honest_minus_malicious"])
                            for r in records
                        ]

                    metric_fracs = [_fraction(x) for x in metrics]
                    mean_metric = float(sum(metric_fracs, Fraction(0, 1)) / len(metric_fracs))
                    mlo, mhi, mseed = bootstrap_mean_ci(
                        metric_fracs,
                        seed_text=f"{STUDY_ID}|metric|{ds}|{rho}|{sweep}|{value}",
                    )
                    mean_rows.append({
                        "dataset": ds,
                        "rho": rho,
                        "sweep": sweep,
                        "value": value,
                        "is_default": is_default,
                        "paper_metric": "MacroF1" if ds == "dog" else "MAE",
                        "n": REPLICATES,
                        "mean": mean_metric,
                        "bootstrap_95_ci_low": mlo,
                        "bootstrap_95_ci_high": mhi,
                        "bootstrap_seed": mseed,
                        "source": source,
                    })

                    losses = [_loss_from_metric(ds, x) for x in metrics]
                    deltas = [
                        _fraction(a) - _fraction(b)
                        for a, b in zip(losses, default_losses)
                    ]
                    dmean = float(sum(deltas, Fraction(0, 1)) / len(deltas))
                    dlo, dhi, dseed = bootstrap_mean_ci(
                        deltas,
                        seed_text=f"{STUDY_ID}|delta|{ds}|{rho}|{sweep}|{value}",
                    )
                    delta_rows.append({
                        "dataset": ds,
                        "rho": rho,
                        "sweep": sweep,
                        "value": value,
                        "is_default": is_default,
                        "n": REPLICATES,
                        "mean_loss_delta_vs_default": dmean,
                        "bootstrap_95_ci_low": dlo,
                        "bootstrap_95_ci_high": dhi,
                        "bootstrap_seed": dseed,
                        "interpretation": "negative_is_better_than_default",
                        "inference_policy": "descriptive sensitivity; no post-hoc parameter selection",
                    })

                    if gaps:
                        gap_fracs = [_fraction(x) for x in gaps]
                        gmean = float(sum(gap_fracs, Fraction(0, 1)) / len(gap_fracs))
                        glo, ghi, gseed = bootstrap_mean_ci(
                            gap_fracs,
                            seed_text=f"{STUDY_ID}|gap|{ds}|{rho}|{sweep}|{value}",
                        )
                        gap_rows.append({
                            "dataset": ds,
                            "rho": rho,
                            "sweep": sweep,
                            "value": value,
                            "n": REPLICATES,
                            "mean_final_reputation_gap": gmean,
                            "bootstrap_95_ci_low": glo,
                            "bootstrap_95_ci_high": ghi,
                            "bootstrap_seed": gseed,
                        })

    sensitivity_summary: dict[str, Any] = {}
    for sweep in SWEEPS:
        sensitivity_summary[sweep] = {}
        for ds in DATASETS:
            sensitivity_summary[sweep][ds] = {}
            for rho in RHOS:
                cell = [
                    r for r in mean_rows
                    if r["sweep"] == sweep and r["dataset"] == ds and r["rho"] == rho
                ]
                best = min(
                    cell,
                    key=lambda r: _loss_from_metric(ds, float(r["mean"])),
                )
                worst = max(
                    cell,
                    key=lambda r: _loss_from_metric(ds, float(r["mean"])),
                )
                sensitivity_summary[sweep][ds][rho] = {
                    "descriptive_best_value": best["value"],
                    "descriptive_best_mean": best["mean"],
                    "descriptive_worst_value": worst["value"],
                    "descriptive_worst_mean": worst["mean"],
                    "default_value": DEFAULT_VALUES[sweep],
                    "note": "descriptive only; the preregistered default remains frozen",
                }

    # Compact robustness summaries around the frozen default.
    sweep_spreads: dict[str, Any] = {}
    for sweep in SWEEPS:
        relevant = [r for r in delta_rows if r["sweep"] == sweep and not r["is_default"]]
        sweep_spreads[sweep] = {
            "nondefault_cell_count": len(relevant),
            "max_absolute_mean_loss_delta_vs_default": max(
                abs(float(r["mean_loss_delta_vs_default"])) for r in relevant
            ) if relevant else 0.0,
            "cells_better_than_default_by_mean": sum(
                float(r["mean_loss_delta_vs_default"]) < 0 for r in relevant
            ),
            "cells_worse_than_default_by_mean": sum(
                float(r["mean_loss_delta_vs_default"]) > 0 for r in relevant
            ),
        }

    analysis = {
        "sensitivity_summary": sensitivity_summary,
        "sweep_spreads": sweep_spreads,
        "default_configuration_remains_frozen": True,
        "parameter_selection": "NONE",
        "inference_policy": (
            "Paired bootstrap intervals and loss deltas are descriptive. "
            "No multiplicity-based winner selection is used for parameter tuning."
        ),
        "scientific_decision": "REVIEW_AFTER_RESULT",
    }
    return mean_rows, delta_rows, gap_rows, analysis


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
        raise VIDHiddenError("MISSING:" + ",".join(missing))

    manifest_sha = _sha256(root / P0_2_MANIFEST_REL)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise VIDHiddenError(
            f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}"
        )

    manifest = _load_json(root / P0_2_MANIFEST_REL)
    datasets = _load_datasets(root)
    streams = _plan_streams(manifest, datasets)
    selectors, masks, mask_hashes, selector_audit = _load_selector_family(
        root,
        datasets,
    )
    full_reference = _load_full_reference(
        root,
        streams,
        mask_hashes,
        masks,
    )

    points = planned_new_points()
    planned_new = len(streams) * len(points)

    print("VI_D_HIDDEN_SELECTOR_VALIDATE=PASS")
    print("DATASETS=dog,weather")
    print("RHO_GRID=3/10,7/10")
    print(f"REPLICATES_PER_CELL={REPLICATES}")
    print(f"STREAMS={len(streams)}")
    print(f"REUSED_DEFAULT_FULL_RUNS={len(full_reference)}")
    print("DEFAULT_FULL_SOURCE=P0_1C_HIDDEN_B1_FORMAL_RESULTS")
    print("DEFAULT_FULL_RERUN=NO")
    print(f"NEW_POINTS_PER_STREAM={len(points)}")
    print(f"PLANNED_NEW_METHOD_RUNS={planned_new}")
    print(f"K_GRID={','.join(SWEEPS['K'])}")
    print(f"LAMBDA_TAU_GRID={','.join(SWEEPS['lambda_tau'])}")
    print(f"KAPPA_GRID={','.join(SWEEPS['kappa'])}")
    print(f"CALIBRATION_PERIOD_GRID={','.join(SWEEPS['calibration_period'])}")
    print("CALIBRATION_PERIOD_SEMANTICS=HIDDEN_BERNOULLI_PI_EQUALS_1_OVER_P")
    print("P10_P20_P40_USE_SAME_SECRET_SEED_AND_PRF_SCORES=YES")
    print("P40_SUBSET_P20_SUBSET_P10=YES")
    print("P20_MASK_REUSE_P0_1C=EXACT")
    print("P_SWEEP_COMMON_SCORING_MASK=COMPLEMENT_OF_P10_CALIBRATION_SET")
    print("NON_P_SWEEPS_USE_EXACT_P20_MASK=YES")
    print("SAME_ATTACKED_STREAM_ALL_POINTS=YES")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("ORDINARY_PERSISTENT_REPUTATION_UPDATE=DISABLED")
    print("CALIBRATION_PERSISTENT_REPUTATION_UPDATE=ENABLED")
    print("KAPPA_RULE=MU_EQUALS_ETA_TIMES_KAPPA_EQUALS_1_OVER_5_FIXED")
    print("PARAMETER_RETUNING_FOR_PAPER_SELECTION=PROHIBITED")
    print("SELECTOR_RESELECTION=NO")

    for ds in DATASETS:
        a = selector_audit[ds]
        print(
            f"CALIBRATION_COUNTS_{ds.upper()}="
            f"P10:{a['periods']['10']['calibration_task_count']},"
            f"P20:{a['periods']['20']['calibration_task_count']},"
            f"P40:{a['periods']['40']['calibration_task_count']}"
        )
        print(
            f"P_SWEEP_COMMON_SCORED_TASKS_{ds.upper()}="
            f"{a['common_scored_task_count']}"
        )

    return (
        datasets,
        streams,
        selectors,
        masks,
        mask_hashes,
        selector_audit,
        full_reference,
        points,
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
        selector_audit,
        full_reference,
        points,
    ) = validate(root)

    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"

    if raw_path.exists() and not args.resume:
        raise VIDHiddenError(f"OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)

    for stream_index, stream in enumerate(streams, start=1):
        dataset = datasets[stream.dataset_id]

        # Complete corruption stream first; no hidden calibration information
        # is passed into attack construction.
        tasks = attacked_tasks(dataset, stream)

        for sweep, value in points:
            period = calibration_period_for_point(sweep, value)
            mask_hash = mask_hashes[stream.dataset_id][period]
            run_id = _run_id(stream, sweep, value, mask_hash)

            if run_id in existing and existing[run_id].get("status") == "success":
                continue

            started = time.perf_counter()
            try:
                result = run_point(
                    dataset,
                    tasks,
                    stream,
                    sweep,
                    value,
                    masks[stream.dataset_id],
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
                    "status": "success",
                    "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
                    "sweep": sweep,
                    "value": value,
                    "calibration_mask_sha256": mask_hash,
                    "selector_seed_commitment_sha256": selectors[
                        stream.dataset_id
                    ][period].seed_commitment_sha256,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
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
                    "sweep": sweep,
                    "value": value,
                    "status": "failure",
                    "calibration_mask_sha256": mask_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }

            _append(raw_path, row)
            existing[run_id] = row

        if stream_index % 5 == 0 or stream_index == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"PROGRESS={stream_index}/{len(streams)} "
                f"NEW_METHOD_RUNS={len(existing)}/{len(streams)*len(points)} "
                f"SUCCESS={success}",
                flush=True,
            )

    final_rows = list(existing.values())
    _rewrite_canonical(raw_path, final_rows)
    final_rows = list(_read_existing(raw_path).values())

    planned_new = len(streams) * len(points)
    success = sum(r.get("status") == "success" for r in final_rows)
    failure = sum(r.get("status") != "success" for r in final_rows)

    mean_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}

    if success == planned_new and failure == 0:
        mean_rows, delta_rows, gap_rows, analysis = _analyze(
            full_reference,
            final_rows,
        )
        _write_csv(outdir / "vi_d_hidden_sensitivity_means.csv", mean_rows)
        _write_csv(outdir / "vi_d_hidden_delta_vs_default.csv", delta_rows)
        _write_csv(outdir / "vi_d_hidden_final_reputation_gap_means.csv", gap_rows)

    selector_audit_obj = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "same_private_seed_as_p0_1c": True,
        "same_prf_score_namespace_as_p0_1c": True,
        "p20_mask_exact_p0_1c": True,
        "nested_schedule_contract": "R_P40 subset R_P20 subset R_P10",
        "calibration_period_interpretation": "P is nominal hidden Bernoulli interval; pi=1/P",
        "datasets": selector_audit,
        "seed_revealed": False,
        "selector_reselection": False,
    }
    _atomic_json(outdir / "selector_rate_audit.json", selector_audit_obj)

    migration_audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "full_default_reference": {
            "source": P0_1C_RAW_REL,
            "selected_rows": len(full_reference),
            "rerun": False,
            "scope": "K/lambda_tau/kappa default only",
        },
        "hidden_selector": {
            "same_p0_1c_secret_seed": True,
            "p20_mask_exact_p0_1c": True,
            "p10_p20_p40_nested": True,
            "p_sweep_common_scoring_mask": "complement of P10 calibration set",
            "non_p_sweeps_fixed_mask": "exact P20 calibration set",
            "attack_generation_before_mode_designation": True,
        },
        "kappa_sweep": {
            "mu_equals_eta_times_kappa": "1/5 fixed",
            "eta_by_kappa": {
                "1": "1/5",
                "2": "1/10",
                "4": "1/20",
            },
        },
        "statistics": {
            "paired_bootstrap_resamples": 10000,
            "role": "descriptive sensitivity",
            "multiplicity_selection": "none",
            "post_hoc_parameter_selection": False,
        },
        "algorithm_retuning": False,
        "selector_reselection": False,
    }
    _atomic_json(outdir / "migration_audit.json", migration_audit)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "datasets": list(DATASETS),
        "rho_grid": list(RHOS),
        "replicates_per_cell": REPLICATES,
        "streams": len(streams),
        "reused_default_full_runs": len(full_reference),
        "default_full_rerun": False,
        "new_points_per_stream": len(points),
        "planned_new_method_runs": planned_new,
        "completed_unique_new_method_runs": len(final_rows),
        "success": success,
        "failure": failure,
        "base_config": BASE_CONFIG,
        "sweeps": {k: list(v) for k, v in SWEEPS.items()},
        "default_values": DEFAULT_VALUES,
        "selector": {
            "hidden": True,
            "p_semantics": "pi=1/P Bernoulli threshold over common secret PRF scores",
            "p20_exact_p0_1c": True,
            "nested_p_masks": True,
            "pre_freeze_mode_visible_to_attack": False,
        },
        "analysis": analysis,
        "scope": {
            "formal_vi_d_hidden_selector_migration": True,
            "one_factor_at_a_time": True,
            "algorithm_retuning": False,
            "selector_reselection": False,
            "parameter_selection": "NONE",
        },
    }
    _atomic_json(outdir / "formal_summary.json", summary)

    print("VI_D_HIDDEN_SELECTOR_MIGRATION=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"REUSED_DEFAULT_FULL_RUNS={len(full_reference)}")
    print("DEFAULT_FULL_RERUN=NO")
    print(f"NEW_POINTS_PER_STREAM={len(points)}")
    print(f"PLANNED_NEW_METHOD_RUNS={planned_new}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")

    if analysis:
        for sweep in SWEEPS:
            s = analysis["sweep_spreads"][sweep]
            print(
                f"SWEEP_{sweep.upper()}="
                f"NONDEFAULT_CELLS:{s['nondefault_cell_count']},"
                f"BETTER:{s['cells_better_than_default_by_mean']},"
                f"WORSE:{s['cells_worse_than_default_by_mean']},"
                f"MAX_ABS_LOSS_DELTA:{s['max_absolute_mean_loss_delta_vs_default']}"
            )
        print("PARAMETER_SELECTION=NONE")
        print("VI_D_SCIENTIFIC_DECISION=REVIEW_AFTER_RESULT")

    print(f"SUMMARY={outdir / 'formal_summary.json'}")
    return 0 if success == planned_new and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
