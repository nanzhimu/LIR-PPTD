from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from lir_pptd.experiments.phase_r1.statistics import (
    bootstrap_mean_ci,
    exact_two_sided_sign_flip,
    holm_adjust,
    paired_rank_biserial,
)

STUDY_ID = "vi_d2_hidden_selector_migration_v1"
CAL_QUALITY_STUDY_ID = "vi_d2_hidden_calibration_quality_v1"
ORDER_STUDY_ID = "vi_d2_hidden_task_order_v1"
P0_1C_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

ALL_DATASETS = ("product", "duck", "dog", "weather")
QUALITY_DATASETS = ("dog", "weather")
RHOS = ("3/10", "7/10")
REPLICATES = 20
ORDER_SEEDS = (7101, 7102, 7103, 7104, 7105)
PERIOD = 20

REFERENCE_NOISE_LEVELS = ("1/20", "1/10", "1/5")
MISSING_LEVELS = ("1/4", "1/2")
QUALITY_VARIANTS = tuple(
    [("reference_noise", x) for x in REFERENCE_NOISE_LEVELS]
    + [("missing_calibration", x) for x in MISSING_LEVELS]
)

CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
P0_1C_RAW_REL = "results/p0_1c_hidden_b1_v1/formal_runs.jsonl"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
OUT_REL = "results/vi_d2_hidden_selector_migration_v1"


class VID2HiddenError(RuntimeError):
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VID2HiddenError(f"MISSING_JSONL:{path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        out[str(row["run_id"])] = row
    return out


def _rewrite_canonical_quality(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_ord = {d: i for i, d in enumerate(QUALITY_DATASETS)}
    rho_ord = {r: i for i, r in enumerate(RHOS)}
    axis_ord = {"reference_noise": 0, "missing_calibration": 1}
    level_ord = {
        "reference_noise": {v: i for i, v in enumerate(REFERENCE_NOISE_LEVELS)},
        "missing_calibration": {v: i for i, v in enumerate(MISSING_LEVELS)},
    }
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_ord[str(r["dataset"])],
            rho_ord[str(r["rho"])],
            int(r["replicate"]),
            axis_ord[str(r["axis"])],
            level_ord[str(r["axis"])][str(r["level"])],
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rewrite_canonical_order(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_ord = {d: i for i, d in enumerate(ALL_DATASETS)}
    rho_ord = {r: i for i, r in enumerate(RHOS)}
    seed_ord = {s: i for i, s in enumerate(ORDER_SEEDS)}
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_ord[str(r["dataset"])],
            rho_ord[str(r["rho"])],
            int(r["replicate"]),
            seed_ord[int(r["order_seed"])],
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
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load_datasets(root: Path) -> dict[str, RealDataset]:
    hashes = _load_json(root / DATASET_HASHES_REL)["datasets"]
    return {
        ds: load_real_dataset(
            root / "data",
            ds,
            expected_answer_sha256=hashes[ds]["answer_sha256"],
            expected_truth_sha256=hashes[ds]["truth_sha256"],
        )
        for ds in ALL_DATASETS
    }


def _plan_streams(
    manifest: Mapping[str, Any],
    datasets: Mapping[str, RealDataset],
    dataset_ids: Sequence[str],
) -> list[Stream]:
    out: list[Stream] = []
    for ds in dataset_ids:
        valid = set(map(str, datasets[ds].worker_ids))
        for rho in RHOS:
            rows = manifest["datasets"][ds][rho]["subsets"]
            if len(rows) != REPLICATES:
                raise VID2HiddenError(f"CELL_N:{ds}:{rho}:{len(rows)}:{REPLICATES}")
            for rep, item in enumerate(rows, start=1):
                malicious = tuple(sorted(map(str, item["workers"])))
                if not set(malicious) <= valid:
                    raise VID2HiddenError(f"UNKNOWN_WORKER:{ds}:{rho}:{rep}")
                out.append(
                    Stream(
                        dataset_id=ds,
                        rho=rho,
                        replicate=rep,
                        subset_seed=int(item["seed"]),
                        malicious_workers=malicious,
                    )
                )
    return out


def _selector_config_hash(dataset: RealDataset) -> str:
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
    seeds_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seeds_obj.get("seeds", {}).items()}
    if set(seeds) != set(ALL_DATASETS):
        raise VID2HiddenError("P0_1C_SELECTOR_SEED_DATASET_SET_MISMATCH")

    audit_obj = _load_json(root / P0_1C_AUDIT_REL)
    expected = {str(x["dataset"]): dict(x) for x in audit_obj["selectors"]}

    selectors = {}
    masks = {}
    hashes = {}
    counts = {}

    for ds in ALL_DATASETS:
        selector = HiddenCalibrationSchedule(
            seed=bytes.fromhex(seeds[ds]),
            session_id=f"LIR-PPTD/final-real-data/{ds}",
            selector_epoch=0,
            config_hash=_selector_config_hash(datasets[ds]),
            period=PERIOD,
        )
        task_ids = [t.task_id for t in datasets[ds].tasks]
        mask = frozenset(selector.selected_task_ids(task_ids))
        mask_hash = _sha_obj(sorted(mask))
        exp = expected[ds]
        if mask_hash != str(exp["calibration_mask_sha256"]):
            raise VID2HiddenError(f"P0_1C_MASK_HASH_MISMATCH:{ds}")
        if len(mask) != int(exp["calibration_task_count"]):
            raise VID2HiddenError(f"P0_1C_MASK_COUNT_MISMATCH:{ds}")
        if selector.seed_commitment_sha256 != str(exp["seed_commitment_sha256"]):
            raise VID2HiddenError(f"P0_1C_COMMITMENT_MISMATCH:{ds}")
        selectors[ds] = selector
        masks[ds] = mask
        hashes[ds] = mask_hash
        counts[ds] = len(mask)

    return selectors, masks, hashes, counts


def _load_p0_references(
    root: Path,
    streams: Sequence[Stream],
    methods: Sequence[str],
    mask_hashes: Mapping[str, str],
    calibration_counts: Mapping[str, int],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    wanted_streams = {(s.dataset_id, s.rho, s.replicate): s for s in streams}
    rows = _read_jsonl(root / P0_1C_RAW_REL)
    selected: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    for row in rows:
        if row.get("study_id") != P0_1C_STUDY_ID:
            continue
        method = str(row.get("method"))
        if method not in methods:
            continue
        stream_key = (str(row["dataset"]), str(row["rho"]), int(row["replicate"]))
        stream = wanted_streams.get(stream_key)
        if stream is None:
            continue
        key = (*stream_key, method)
        if key in selected:
            raise VID2HiddenError(f"DUPLICATE_P0_REFERENCE:{key}")
        if row.get("status") != "success":
            raise VID2HiddenError(f"FAILED_P0_REFERENCE:{key}")
        if str(row.get("pairing_id")) != stream.pairing_id:
            raise VID2HiddenError(f"P0_REFERENCE_PAIRING_MISMATCH:{key}")
        if tuple(sorted(map(str, row.get("malicious_workers", [])))) != stream.malicious_workers:
            raise VID2HiddenError(f"P0_REFERENCE_SUBSET_MISMATCH:{key}")
        if str(row.get("calibration_mask_sha256")) != mask_hashes[stream.dataset_id]:
            raise VID2HiddenError(f"P0_REFERENCE_MASK_MISMATCH:{key}")
        if row.get("attack_generation_before_mode_designation") is not True:
            raise VID2HiddenError(f"P0_REFERENCE_ATTACK_ORDER_MISMATCH:{key}")
        if method == "lir_pptd":
            if int(row.get("ordinary_persistent_commit_count", -1)) != 0:
                raise VID2HiddenError(f"P0_REFERENCE_ORDINARY_COMMIT:{key}")
            if int(row.get("calibration_persistent_commit_count", -1)) != calibration_counts[stream.dataset_id]:
                raise VID2HiddenError(f"P0_REFERENCE_CAL_COMMIT:{key}")
        selected[key] = row

    expected = len(wanted_streams) * len(methods)
    if len(selected) != expected:
        raise VID2HiddenError(f"P0_REFERENCE_COUNT:{len(selected)}:{expected}")
    return selected


def attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    # Attack creation is completely independent of hidden calibration identity.
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


def perturbation_count(total: int, level: str) -> int:
    fraction = _fraction(level)
    # Nearest integer with exact half-up rounding.  Levels are nominal because
    # realized hidden-calibration counts need not be divisible by 20/10/5/etc.
    count = (2 * total * fraction.numerator + fraction.denominator) // (2 * fraction.denominator)
    return max(1, min(total, int(count)))


def _perturbation_rank(dataset_id: str, axis: str, task_id: str) -> bytes:
    return hashlib.sha256(
        f"{CAL_QUALITY_STUDY_ID}|dataset={dataset_id}|axis={axis}|task={task_id}".encode("utf-8")
    ).digest()


def selected_perturbation_ids(
    dataset_id: str,
    cal_ids: frozenset[str],
    axis: str,
    level: str,
) -> frozenset[str]:
    ranked = sorted(cal_ids, key=lambda tid: (_perturbation_rank(dataset_id, axis, tid), tid))
    return frozenset(ranked[:perturbation_count(len(ranked), level)])


def _corrupted_reference(task: RealTask) -> tuple[tuple[str | int, ...], int | None]:
    if task.modality == "categorical":
        width = len(task.truth_vector)
        if width < 2 or task.truth_class is None:
            raise VID2HiddenError("INVALID_CATEGORICAL_REFERENCE")
        wrong_class = (int(task.truth_class) + 1) % width
        vector = tuple(1 if i == wrong_class else 0 for i in range(width))
        return vector, wrong_class

    values: list[str] = []
    for raw in task.truth_vector:
        value = _fraction(raw)
        if value < 0 or value > 1:
            raise VID2HiddenError(f"NUMERICAL_REFERENCE_OUTSIDE_UNIT_INTERVAL:{task.task_id}")
        corrupted = Fraction(0, 1) if value == Fraction(1, 2) else Fraction(1, 1) - value
        values.append(_fraction_text(corrupted))
    return tuple(values), None


def corrupt_calibration_reference(task: RealTask) -> RealTask:
    vector, cls = _corrupted_reference(task)
    return replace(task, truth_vector=vector, truth_class=cls)


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
            raise VID2HiddenError("CATEGORICAL_PREDICTION_MISSING")
        pc.append(int(cls))
        tc.append(int(task.truth_class))
    else:
        pv.append(float(vector[0]))
        tv.append(float(task.truth_vector[0]))


def _metric(dataset: RealDataset, pc, tc, pv, tv) -> dict[str, Any]:
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
        raise VID2HiddenError("EMPTY_REPUTATION_GROUP")
    mh = statistics.fmean(h)
    mm = statistics.fmean(m)
    return mh, mm, mh - mm


def run_calibration_quality_variant(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
    axis: str,
    level: str,
) -> dict[str, Any]:
    selected = selected_perturbation_ids(dataset.dataset_id, cal_ids, axis, level)
    state = initial_global_state(dataset.worker_ids, CONFIG["c0"])

    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    installed = 0
    missing = 0
    corrupted = 0

    for task in tasks:
        if task.task_id in cal_ids:
            if axis == "missing_calibration" and task.task_id in selected:
                missing += 1
                # The task remains a hidden calibration task and remains
                # excluded from the ordinary scoring denominator; only the
                # authenticated update is unavailable.
                continue

            cal_task = task
            if axis == "reference_noise" and task.task_id in selected:
                cal_task = corrupt_calibration_reference(task)
                corrupted += 1

            _, state = run_calibration_task(cal_task, CONFIG, state, dps=80)
            installed += 1
            continue

        pred = run_lir_task(task, CONFIG, state, ablation="full", dps=80)
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
        # Evidence Gate: never commit pred.next_state on ordinary tasks.

    expected_installed = len(cal_ids) - (len(selected) if axis == "missing_calibration" else 0)
    if installed != expected_installed:
        raise VID2HiddenError(
            f"QUALITY_INSTALLED_COUNT:{dataset.dataset_id}:{axis}:{level}:{installed}:{expected_installed}"
        )
    if axis == "reference_noise" and corrupted != len(selected):
        raise VID2HiddenError("REFERENCE_CORRUPTION_COUNT_MISMATCH")
    if axis == "missing_calibration" and missing != len(selected):
        raise VID2HiddenError("MISSING_CALIBRATION_COUNT_MISMATCH")

    mh, mm, gap = _group_reputation(state, stream.malicious_workers, dataset.worker_ids)
    return {
        **_metric(dataset, pc, tc, pv, tv),
        "axis": axis,
        "level": level,
        "nominal_fraction": float(_fraction(level)),
        "hidden_calibration_event_count": len(cal_ids),
        "selected_calibration_event_count": len(selected),
        "realized_fraction_of_hidden_calibrations": len(selected) / len(cal_ids),
        "calibration_transitions_installed": installed,
        "realized_reference_corruptions": corrupted,
        "realized_missing_calibrations": missing,
        "ordinary_persistent_commit_count": 0,
        "final_mean_reputation_honest": mh,
        "final_mean_reputation_malicious": mm,
        "final_reputation_gap_honest_minus_malicious": gap,
    }


def stable_permutation(tasks: Sequence[RealTask], order_seed: int) -> tuple[RealTask, ...]:
    # SHA-based deterministic ranking avoids Python RNG-version dependence.
    return tuple(
        sorted(
            tasks,
            key=lambda task: (
                hashlib.sha256(
                    f"{ORDER_STUDY_ID}|seed={order_seed}|task={task.task_id}".encode("utf-8")
                ).digest(),
                task.task_id,
            ),
        )
    )


def task_order_sha256(tasks: Sequence[RealTask]) -> str:
    return hashlib.sha256("\n".join(t.task_id for t in tasks).encode("utf-8")).hexdigest()


def run_order_lir(
    dataset: RealDataset,
    permuted_tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
) -> dict[str, Any]:
    if frozenset(t.task_id for t in permuted_tasks if t.task_id in cal_ids) != cal_ids:
        raise VID2HiddenError("ORDER_CALIBRATION_SET_CHANGED")

    state = initial_global_state(dataset.worker_ids, CONFIG["c0"])
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    cal_count = 0

    for task in permuted_tasks:
        if task.task_id in cal_ids:
            _, state = run_calibration_task(task, CONFIG, state, dps=80)
            cal_count += 1
            continue

        pred = run_lir_task(task, CONFIG, state, ablation="full", dps=80)
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

    if cal_count != len(cal_ids):
        raise VID2HiddenError("ORDER_CALIBRATION_COUNT_MISMATCH")

    mh, mm, gap = _group_reputation(state, stream.malicious_workers, dataset.worker_ids)
    return {
        **_metric(dataset, pc, tc, pv, tv),
        "calibration_event_count": cal_count,
        "ordinary_persistent_commit_count": 0,
        "final_mean_reputation_honest": mh,
        "final_mean_reputation_malicious": mm,
        "final_reputation_gap_honest_minus_malicious": gap,
    }


def _quality_run_id(stream: Stream, axis: str, level: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{CAL_QUALITY_STUDY_ID}|{mask_hash}|{stream.pairing_id}|{axis}|{level}".encode("utf-8")
    ).hexdigest()


def _order_run_id(stream: Stream, order_seed: int, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{ORDER_STUDY_ID}|{mask_hash}|{stream.pairing_id}|order={order_seed}".encode("utf-8")
    ).hexdigest()


def _analyze_calibration_quality(
    default_lir: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    success = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["rho"]), int(r["replicate"]), str(r["axis"]), str(r["level"])): r
        for r in success
    }

    means: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []

    for ds in QUALITY_DATASETS:
        for rho in RHOS:
            defaults = [default_lir[(ds, rho, rep, "lir_pptd")] for rep in range(1, REPLICATES + 1)]
            dmetric = [_fraction(r["paper_metric"]) for r in defaults]
            dlo, dhi, dseed = bootstrap_mean_ci(
                dmetric,
                seed_text=f"{CAL_QUALITY_STUDY_ID}|mean|{ds}|{rho}|default",
            )
            means.append({
                "dataset": ds,
                "rho": rho,
                "axis": "default",
                "level": "0",
                "n": REPLICATES,
                "paper_metric": defaults[0]["paper_metric_name"],
                "mean": float(sum(dmetric, Fraction(0, 1)) / len(dmetric)),
                "bootstrap_95_ci_low": dlo,
                "bootstrap_95_ci_high": dhi,
                "source": "reused_p0_1c_lir",
                "bootstrap_seed": dseed,
            })

            for axis, levels in (
                ("reference_noise", REFERENCE_NOISE_LEVELS),
                ("missing_calibration", MISSING_LEVELS),
            ):
                for level in levels:
                    variants = [by[(ds, rho, rep, axis, level)] for rep in range(1, REPLICATES + 1)]
                    vals = [_fraction(r["paper_metric"]) for r in variants]
                    lo, hi, seed = bootstrap_mean_ci(
                        vals,
                        seed_text=f"{CAL_QUALITY_STUDY_ID}|mean|{ds}|{rho}|{axis}|{level}",
                    )
                    means.append({
                        "dataset": ds,
                        "rho": rho,
                        "axis": axis,
                        "level": level,
                        "n": REPLICATES,
                        "paper_metric": variants[0]["paper_metric_name"],
                        "mean": float(sum(vals, Fraction(0, 1)) / len(vals)),
                        "bootstrap_95_ci_low": lo,
                        "bootstrap_95_ci_high": hi,
                        "source": "new_hidden_selector_run",
                        "bootstrap_seed": seed,
                        "selected_calibration_event_count": variants[0]["selected_calibration_event_count"],
                        "realized_fraction_of_hidden_calibrations": variants[0]["realized_fraction_of_hidden_calibrations"],
                    })

                    deltas = [
                        _fraction(v["primary_loss"]) - _fraction(d["primary_loss"])
                        for v, d in zip(variants, defaults)
                    ]
                    delta_lo, delta_hi, delta_seed = bootstrap_mean_ci(
                        deltas,
                        seed_text=f"{CAL_QUALITY_STUDY_ID}|delta|{ds}|{rho}|{axis}|{level}",
                    )
                    p = exact_two_sided_sign_flip(deltas)
                    mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                    contrasts.append({
                        "hypothesis_id": f"{axis}|{ds}|{rho}|{level}",
                        "dataset": ds,
                        "rho": rho,
                        "axis": axis,
                        "level": level,
                        "n": REPLICATES,
                        "mean_loss_delta_variant_minus_default": float(mean_delta),
                        "variant_degraded_reps": sum(d > 0 for d in deltas),
                        "ties": sum(d == 0 for d in deltas),
                        "variant_improved_reps": sum(d < 0 for d in deltas),
                        "bootstrap_95_ci_low": delta_lo,
                        "bootstrap_95_ci_high": delta_hi,
                        "bootstrap_seed": delta_seed,
                        "exact_two_sided_sign_flip_p": float(p),
                        "paired_rank_biserial": float(paired_rank_biserial(deltas)),
                        "_p": p,
                    })

    # Keep the historical multiplicity logic: one family for the 12 reference
    # corruption contrasts and one family for the 8 missing-calibration contrasts.
    for axis in ("reference_noise", "missing_calibration"):
        family = [r for r in contrasts if r["axis"] == axis]
        adjusted = holm_adjust({r["hypothesis_id"]: r["_p"] for r in family})
        for row in family:
            adj = adjusted[row["hypothesis_id"]]
            row["holm_family"] = (
                "calibration_quality|reference_noise|12"
                if axis == "reference_noise"
                else "calibration_quality|missing_calibration|8"
            )
            row["holm_family_size"] = len(family)
            row["holm_adjusted_p"] = float(adj)
            delta = float(row["mean_loss_delta_variant_minus_default"])
            row["direction"] = "degraded" if delta > 0 else "improved" if delta < 0 else "tie"
            row["holm_direction"] = (
                row["direction"] if adj <= Fraction(1, 20) and row["direction"] != "tie"
                else "not_significant"
            )
            row.pop("_p", None)

    summary = {}
    for axis in ("reference_noise", "missing_calibration"):
        family = [r for r in contrasts if r["axis"] == axis]
        summary[axis] = {
            "cells": len(family),
            "mean_degradation_cells": sum(r["direction"] == "degraded" for r in family),
            "mean_improvement_cells": sum(r["direction"] == "improved" for r in family),
            "holm_significant_degradation_cells": sum(r["holm_direction"] == "degraded" for r in family),
            "holm_significant_improvement_cells": sum(r["holm_direction"] == "improved" for r in family),
        }
    return means, contrasts, summary


def _analyze_order(
    crh_reference: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    success = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["rho"]), int(r["replicate"]), int(r["order_seed"])): r
        for r in success
    }

    seed_rows: list[dict[str, Any]] = []
    raw_p: dict[str, Fraction] = {}

    for ds in ALL_DATASETS:
        for rho in RHOS:
            for order_seed in ORDER_SEEDS:
                deltas: list[Fraction] = []
                lir_losses: list[float] = []
                crh_losses: list[float] = []
                for rep in range(1, REPLICATES + 1):
                    lir = by[(ds, rho, rep, order_seed)]
                    crh = crh_reference[(ds, rho, rep, "crh")]
                    l = _fraction(lir["primary_loss"])
                    c = _fraction(crh["primary_loss"])
                    deltas.append(l - c)
                    lir_losses.append(float(l))
                    crh_losses.append(float(c))

                hid = f"{ds}|{rho}|order={order_seed}"
                p = exact_two_sided_sign_flip(deltas)
                raw_p[hid] = p
                lo, hi, bseed = bootstrap_mean_ci(
                    deltas,
                    seed_text=f"{ORDER_STUDY_ID}|delta|{hid}",
                )
                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                seed_rows.append({
                    "hypothesis_id": hid,
                    "dataset": ds,
                    "rho": rho,
                    "order_seed": order_seed,
                    "n": REPLICATES,
                    "lir_mean_loss": statistics.fmean(lir_losses),
                    "crh_mean_loss": statistics.fmean(crh_losses),
                    "mean_loss_delta_lir_minus_crh": float(mean_delta),
                    "lir_pair_wins": sum(d < 0 for d in deltas),
                    "ties": sum(d == 0 for d in deltas),
                    "crh_pair_wins": sum(d > 0 for d in deltas),
                    "bootstrap_95_ci_low": lo,
                    "bootstrap_95_ci_high": hi,
                    "bootstrap_seed": bseed,
                    "exact_two_sided_sign_flip_p": float(p),
                    "paired_rank_biserial": float(paired_rank_biserial(deltas)),
                })

    adjusted = holm_adjust(raw_p)
    for row in seed_rows:
        adj = adjusted[row["hypothesis_id"]]
        row["holm_family"] = "task_order|4-datasets|2-rhos|5-orders|40"
        row["holm_family_size"] = len(seed_rows)
        row["holm_adjusted_p"] = float(adj)
        delta = float(row["mean_loss_delta_lir_minus_crh"])
        row["direction"] = "lir_win" if delta < 0 else "crh_win" if delta > 0 else "tie"
        row["holm_direction"] = (
            row["direction"] if adj <= Fraction(1, 20) and row["direction"] != "tie"
            else "not_significant"
        )

    aggregate: list[dict[str, Any]] = []
    for ds in ALL_DATASETS:
        for rho in RHOS:
            cells = [r for r in seed_rows if r["dataset"] == ds and r["rho"] == rho]
            means = [float(r["mean_loss_delta_lir_minus_crh"]) for r in cells]
            aggregate.append({
                "dataset": ds,
                "rho": rho,
                "order_count": len(cells),
                "lir_favored_orders": sum(x < 0 for x in means),
                "tie_orders": sum(x == 0 for x in means),
                "crh_favored_orders": sum(x > 0 for x in means),
                "min_order_mean_delta": min(means),
                "max_order_mean_delta": max(means),
                "mean_of_order_mean_deltas": statistics.fmean(means),
                "holm_significant_lir_orders": sum(r["holm_direction"] == "lir_win" for r in cells),
                "holm_significant_crh_orders": sum(r["holm_direction"] == "crh_win" for r in cells),
            })

    summary = {
        "regime_count": len(aggregate),
        "regimes_lir_favored_all_5_orders": sum(r["lir_favored_orders"] == 5 for r in aggregate),
        "regimes_crh_favored_all_5_orders": sum(r["crh_favored_orders"] == 5 for r in aggregate),
        "regimes_mixed_or_near_parity": sum(
            r["lir_favored_orders"] != 5 and r["crh_favored_orders"] != 5 for r in aggregate
        ),
    }
    return seed_rows, aggregate, summary


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
        raise VID2HiddenError("MISSING:" + ",".join(missing))

    manifest_sha = _sha256(root / P0_2_MANIFEST_REL)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise VID2HiddenError(
            f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}"
        )

    manifest = _load_json(root / P0_2_MANIFEST_REL)
    datasets = _load_datasets(root)
    quality_streams = _plan_streams(manifest, datasets, QUALITY_DATASETS)
    order_streams = _plan_streams(manifest, datasets, ALL_DATASETS)
    selectors, masks, mask_hashes, calibration_counts = _load_hidden_masks(root, datasets)

    quality_defaults = _load_p0_references(
        root,
        quality_streams,
        ("lir_pptd",),
        mask_hashes,
        calibration_counts,
    )
    order_crh = _load_p0_references(
        root,
        order_streams,
        ("crh",),
        mask_hashes,
        calibration_counts,
    )

    # Validate nested perturbation-selection prefixes.
    perturbation_audit = {}
    for ds in QUALITY_DATASETS:
        noise_sets = [
            selected_perturbation_ids(ds, masks[ds], "reference_noise", level)
            for level in REFERENCE_NOISE_LEVELS
        ]
        missing_sets = [
            selected_perturbation_ids(ds, masks[ds], "missing_calibration", level)
            for level in MISSING_LEVELS
        ]
        if not noise_sets[0] <= noise_sets[1] <= noise_sets[2]:
            raise VID2HiddenError(f"REFERENCE_NOISE_NESTING:{ds}")
        if not missing_sets[0] <= missing_sets[1]:
            raise VID2HiddenError(f"MISSING_NESTING:{ds}")
        perturbation_audit[ds] = {
            "hidden_calibration_count": len(masks[ds]),
            "reference_noise": {
                level: {
                    "selected_count": len(selected_perturbation_ids(ds, masks[ds], "reference_noise", level)),
                    "realized_fraction": len(selected_perturbation_ids(ds, masks[ds], "reference_noise", level)) / len(masks[ds]),
                }
                for level in REFERENCE_NOISE_LEVELS
            },
            "missing_calibration": {
                level: {
                    "selected_count": len(selected_perturbation_ids(ds, masks[ds], "missing_calibration", level)),
                    "realized_fraction": len(selected_perturbation_ids(ds, masks[ds], "missing_calibration", level)) / len(masks[ds]),
                }
                for level in MISSING_LEVELS
            },
        }

    # Verify each order seed yields a true permutation while calibration identity
    # remains exactly the same set of immutable task IDs.
    order_audit = {}
    for ds in ALL_DATASETS:
        original_ids = {t.task_id for t in datasets[ds].tasks}
        per_seed = {}
        hashes_seen = set()
        for seed in ORDER_SEEDS:
            perm = stable_permutation(datasets[ds].tasks, seed)
            if {t.task_id for t in perm} != original_ids:
                raise VID2HiddenError(f"ORDER_NOT_PERMUTATION:{ds}:{seed}")
            cal_in_perm = frozenset(t.task_id for t in perm if t.task_id in masks[ds])
            if cal_in_perm != masks[ds]:
                raise VID2HiddenError(f"ORDER_CHANGED_CALIBRATION_SET:{ds}:{seed}")
            digest = task_order_sha256(perm)
            hashes_seen.add(digest)
            per_seed[str(seed)] = {
                "permuted_task_order_sha256": digest,
                "calibration_set_exact_p0_1c": True,
            }
        if len(hashes_seen) != len(ORDER_SEEDS):
            raise VID2HiddenError(f"ORDER_HASH_NOT_UNIQUE:{ds}")
        order_audit[ds] = per_seed

    quality_new_runs = len(quality_streams) * len(QUALITY_VARIANTS)
    order_new_runs = len(order_streams) * len(ORDER_SEEDS)

    print("VI_D2_HIDDEN_SELECTOR_VALIDATE=PASS")
    print("COMPONENTS=calibration_quality,task_order")
    print("P0_1C_HIDDEN_SELECTOR_MASK_REUSE=EXACT")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("ALGORITHM_RETUNING=NO")
    print("SELECTOR_RESELECTION=NO")

    print("CALIBRATION_QUALITY_DATASETS=dog,weather")
    print("CALIBRATION_QUALITY_RHO_GRID=3/10,7/10")
    print("CALIBRATION_QUALITY_REPLICATES_PER_CELL=20")
    print(f"CALIBRATION_QUALITY_STREAMS={len(quality_streams)}")
    print(f"CALIBRATION_QUALITY_REUSED_DEFAULT_RUNS={len(quality_defaults)}")
    print("CALIBRATION_QUALITY_DEFAULT_RERUN=NO")
    print("REFERENCE_NOISE_LEVELS=1/20,1/10,1/5")
    print("MISSING_CALIBRATION_LEVELS=1/4,1/2")
    print("QUALITY_PERTURBATION_SELECTION=TASK_ID_HASH_RANKED_NESTED_PREFIX")
    print("REFERENCE_CORRUPTION_CATEGORICAL=NEXT_WRONG_CLASS")
    print("REFERENCE_CORRUPTION_NUMERICAL=UNIT_INTERVAL_COMPLEMENT")
    print("MISSING_CALIBRATION_SCORING_MASK=UNCHANGED")
    print(f"CALIBRATION_QUALITY_PLANNED_NEW_RUNS={quality_new_runs}")
    print("CALIBRATION_QUALITY_HOLM_FAMILIES=REFERENCE_NOISE_12,MISSING_8")

    print("TASK_ORDER_DATASETS=product,duck,dog,weather")
    print("TASK_ORDER_RHO_GRID=3/10,7/10")
    print("TASK_ORDER_REPLICATES_PER_CELL=20")
    print("TASK_ORDER_SEEDS=7101,7102,7103,7104,7105")
    print(f"TASK_ORDER_BASE_STREAMS={len(order_streams)}")
    print(f"TASK_ORDER_REUSED_CRH_RUNS={len(order_crh)}")
    print("TASK_ORDER_CRH_RERUN=NO")
    print("TASK_ORDER_PERMUTATION=STABLE_SHA256_TASK_ID_RANK")
    print("TASK_ORDER_CALIBRATION_MEMBERSHIP=IMMUTABLE_TASK_ID_BASED")
    print("SAME_CALIBRATION_TASK_SET_ALL_5_ORDERS=YES")
    print(f"TASK_ORDER_PLANNED_NEW_LIR_RUNS={order_new_runs}")
    print("TASK_ORDER_HOLM_FAMILY_SIZE=40")

    print(f"TOTAL_PLANNED_NEW_RUNS={quality_new_runs + order_new_runs}")
    for ds in ALL_DATASETS:
        print(f"CALIBRATION_COUNT_{ds.upper()}={calibration_counts[ds]}")
    for ds in QUALITY_DATASETS:
        qa = perturbation_audit[ds]
        print(
            f"QUALITY_COUNTS_{ds.upper()}="
            f"NOISE5:{qa['reference_noise']['1/20']['selected_count']},"
            f"NOISE10:{qa['reference_noise']['1/10']['selected_count']},"
            f"NOISE20:{qa['reference_noise']['1/5']['selected_count']},"
            f"MISS25:{qa['missing_calibration']['1/4']['selected_count']},"
            f"MISS50:{qa['missing_calibration']['1/2']['selected_count']}"
        )

    return {
        "datasets": datasets,
        "quality_streams": quality_streams,
        "order_streams": order_streams,
        "selectors": selectors,
        "masks": masks,
        "mask_hashes": mask_hashes,
        "calibration_counts": calibration_counts,
        "quality_defaults": quality_defaults,
        "order_crh": order_crh,
        "perturbation_audit": perturbation_audit,
        "order_audit": order_audit,
    }


def _execute_quality(root: Path, ctx: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    outdir = root / OUT_REL / "calibration_quality"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"
    if raw_path.exists() and not resume:
        raise VID2HiddenError(f"QUALITY_OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)
    streams = ctx["quality_streams"]
    masks = ctx["masks"]
    hashes = ctx["mask_hashes"]
    selectors = ctx["selectors"]

    for idx, stream in enumerate(streams, start=1):
        dataset = ctx["datasets"][stream.dataset_id]
        tasks = attacked_tasks(dataset, stream)
        cal_ids = masks[stream.dataset_id]
        mask_hash = hashes[stream.dataset_id]

        for axis, level in QUALITY_VARIANTS:
            rid = _quality_run_id(stream, axis, level, mask_hash)
            if rid in existing and existing[rid].get("status") == "success":
                continue
            started = time.perf_counter()
            try:
                result = run_calibration_quality_variant(
                    dataset, tasks, cal_ids, stream, axis, level
                )
                row = {
                    "schema_version": "1.0",
                    "study_id": CAL_QUALITY_STUDY_ID,
                    "run_id": rid,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "subset_seed": stream.subset_seed,
                    "malicious_workers": list(stream.malicious_workers),
                    "status": "success",
                    "axis": axis,
                    "level": level,
                    "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
                    "calibration_mask_sha256": mask_hash,
                    "seed_commitment_sha256": selectors[stream.dataset_id].seed_commitment_sha256,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
                    **result,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "study_id": CAL_QUALITY_STUDY_ID,
                    "run_id": rid,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "axis": axis,
                    "level": level,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            _append(raw_path, row)
            existing[rid] = row

        if idx % 10 == 0 or idx == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"QUALITY_PROGRESS={idx}/{len(streams)} "
                f"NEW_RUNS={len(existing)}/{len(streams)*len(QUALITY_VARIANTS)} "
                f"SUCCESS={success}",
                flush=True,
            )

    _rewrite_canonical_quality(raw_path, existing.values())
    final = list(_read_existing(raw_path).values())
    planned = len(streams) * len(QUALITY_VARIANTS)
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)

    analysis = {}
    if success == planned and failure == 0:
        means, contrasts, analysis = _analyze_calibration_quality(
            ctx["quality_defaults"], final
        )
        _write_csv(outdir / "calibration_quality_means.csv", means)
        _write_csv(outdir / "calibration_quality_contrasts.csv", contrasts)

    summary = {
        "schema_version": "1.0",
        "study_id": CAL_QUALITY_STUDY_ID,
        "streams": len(streams),
        "reused_default_runs": len(ctx["quality_defaults"]),
        "default_rerun": False,
        "planned_new_runs": planned,
        "completed_unique_new_runs": len(final),
        "success": success,
        "failure": failure,
        "analysis": analysis,
    }
    _atomic_json(outdir / "formal_summary.json", summary)
    return summary


def _execute_order(root: Path, ctx: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    outdir = root / OUT_REL / "task_order"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"
    if raw_path.exists() and not resume:
        raise VID2HiddenError(f"ORDER_OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)
    streams = ctx["order_streams"]
    masks = ctx["masks"]
    hashes = ctx["mask_hashes"]
    selectors = ctx["selectors"]

    for idx, stream in enumerate(streams, start=1):
        dataset = ctx["datasets"][stream.dataset_id]
        attacked = attacked_tasks(dataset, stream)
        cal_ids = masks[stream.dataset_id]
        mask_hash = hashes[stream.dataset_id]

        for order_seed in ORDER_SEEDS:
            rid = _order_run_id(stream, order_seed, mask_hash)
            if rid in existing and existing[rid].get("status") == "success":
                continue
            started = time.perf_counter()
            try:
                permuted = stable_permutation(attacked, order_seed)
                result = run_order_lir(dataset, permuted, cal_ids, stream)
                crh = ctx["order_crh"][
                    (stream.dataset_id, stream.rho, stream.replicate, "crh")
                ]
                row = {
                    "schema_version": "1.0",
                    "study_id": ORDER_STUDY_ID,
                    "run_id": rid,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "subset_seed": stream.subset_seed,
                    "malicious_workers": list(stream.malicious_workers),
                    "order_seed": order_seed,
                    "status": "success",
                    "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
                    "calibration_mask_sha256": mask_hash,
                    "seed_commitment_sha256": selectors[stream.dataset_id].seed_commitment_sha256,
                    "permuted_task_order_sha256": task_order_sha256(permuted),
                    "same_calibration_set_as_p0_1c": True,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
                    "crh_source": "reused_p0_1c_stateless_reference",
                    "crh_primary_loss": float(crh["primary_loss"]),
                    "crh_paper_metric": float(crh["paper_metric"]),
                    **result,
                    "delta_lir_minus_crh": float(result["primary_loss"]) - float(crh["primary_loss"]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "study_id": ORDER_STUDY_ID,
                    "run_id": rid,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": stream.rho,
                    "replicate": stream.replicate,
                    "order_seed": order_seed,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            _append(raw_path, row)
            existing[rid] = row

        if idx % 10 == 0 or idx == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"ORDER_PROGRESS={idx}/{len(streams)} "
                f"NEW_LIR_RUNS={len(existing)}/{len(streams)*len(ORDER_SEEDS)} "
                f"SUCCESS={success}",
                flush=True,
            )

    _rewrite_canonical_order(raw_path, existing.values())
    final = list(_read_existing(raw_path).values())
    planned = len(streams) * len(ORDER_SEEDS)
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)

    analysis = {}
    if success == planned and failure == 0:
        seed_rows, aggregate, analysis = _analyze_order(ctx["order_crh"], final)
        _write_csv(outdir / "order_seed_contrasts.csv", seed_rows)
        _write_csv(outdir / "order_aggregate.csv", aggregate)

    summary = {
        "schema_version": "1.0",
        "study_id": ORDER_STUDY_ID,
        "base_streams": len(streams),
        "order_seeds": list(ORDER_SEEDS),
        "reused_crh_runs": len(ctx["order_crh"]),
        "crh_rerun": False,
        "planned_new_lir_runs": planned,
        "completed_unique_new_lir_runs": len(final),
        "success": success,
        "failure": failure,
        "analysis": analysis,
    }
    _atomic_json(outdir / "formal_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--component",
        choices=("all", "calibration_quality", "task_order"),
        default="all",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    ctx = validate(root)

    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    quality_summary = None
    order_summary = None

    if args.component in {"all", "calibration_quality"}:
        quality_summary = _execute_quality(root, ctx, args.resume)
        print("CALIBRATION_QUALITY_HIDDEN_MIGRATION=COMPLETE")
        print(f"QUALITY_SUCCESS={quality_summary['success']}")
        print(f"QUALITY_FAILURE={quality_summary['failure']}")
        if quality_summary["analysis"]:
            q = quality_summary["analysis"]
            print(
                "REFERENCE_NOISE_MEAN_DEGRADE_SIG="
                f"{q['reference_noise']['mean_degradation_cells']}/"
                f"{q['reference_noise']['holm_significant_degradation_cells']}"
            )
            print(
                "MISSING_CALIBRATION_MEAN_DEGRADE_SIG="
                f"{q['missing_calibration']['mean_degradation_cells']}/"
                f"{q['missing_calibration']['holm_significant_degradation_cells']}"
            )

    if args.component in {"all", "task_order"}:
        order_summary = _execute_order(root, ctx, args.resume)
        print("TASK_ORDER_HIDDEN_MIGRATION=COMPLETE")
        print(f"ORDER_SUCCESS={order_summary['success']}")
        print(f"ORDER_FAILURE={order_summary['failure']}")
        if order_summary["analysis"]:
            o = order_summary["analysis"]
            print(
                "ORDER_REGIMES_LIR_FAVORED_ALL_5="
                f"{o['regimes_lir_favored_all_5_orders']}/{o['regime_count']}"
            )

    outdir = root / OUT_REL
    audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "hidden_selector": {
            "same_p0_1c_private_seeds": True,
            "same_p0_1c_masks": True,
            "task_id_based_membership": True,
            "attack_generation_before_mode_designation": True,
            "selector_reselection": False,
        },
        "calibration_quality": {
            "datasets": list(QUALITY_DATASETS),
            "rho_grid": list(RHOS),
            "replicates": REPLICATES,
            "reference_noise_levels": list(REFERENCE_NOISE_LEVELS),
            "missing_levels": list(MISSING_LEVELS),
            "default_source": P0_1C_RAW_REL,
            "default_rerun": False,
            "perturbation_selection": "deterministic task-ID hash ranking; nested prefixes within the frozen hidden calibration set",
            "reference_corruption": {
                "categorical": "next incorrect one-hot class modulo class count",
                "numerical": "unit-interval complement 1-y; exact 0.5 maps to 0 to guarantee a changed reference",
            },
            "missing_update_scoring_mask": "unchanged P0-1C hidden calibration exclusion mask",
            "holm_families": {
                "reference_noise": 12,
                "missing_calibration": 8,
            },
            "perturbation_audit": ctx["perturbation_audit"],
        },
        "task_order": {
            "datasets": list(ALL_DATASETS),
            "rho_grid": list(RHOS),
            "replicates": REPLICATES,
            "order_seeds": list(ORDER_SEEDS),
            "permutation": "stable SHA-256 ranking of immutable task IDs",
            "same_calibration_task_ids_across_all_orders": True,
            "crh_reference_source": P0_1C_RAW_REL,
            "crh_rerun": False,
            "holm_family_size": 40,
            "order_audit": ctx["order_audit"],
        },
        "algorithm_retuning": False,
        "parameter_selection": "NONE",
    }
    _atomic_json(outdir / "migration_audit.json", audit)

    combined = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "component_requested": args.component,
        "calibration_quality": quality_summary,
        "task_order": order_summary,
        "scope": {
            "hidden_selector_migration": True,
            "algorithm_retuning": False,
            "selector_reselection": False,
        },
    }
    _atomic_json(outdir / "formal_summary.json", combined)

    total_success = sum(
        int(x["success"]) for x in (quality_summary, order_summary) if x is not None
    )
    total_failure = sum(
        int(x["failure"]) for x in (quality_summary, order_summary) if x is not None
    )
    print("VI_D2_HIDDEN_SELECTOR_MIGRATION=COMPLETE")
    print(f"TOTAL_NEW_SUCCESS={total_success}")
    print(f"TOTAL_NEW_FAILURE={total_failure}")
    print(f"SUMMARY={outdir / 'formal_summary.json'}")
    return 0 if total_failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
