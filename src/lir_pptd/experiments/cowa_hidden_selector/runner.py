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
from lir_pptd.experiments.phase_r1.metrics import dataset_losses
from lir_pptd.experiments.phase_r1.real_data_loader import RealDataset, RealTask, load_real_dataset
from lir_pptd.experiments.phase_r1.statistics import (
    bootstrap_mean_ci,
    exact_two_sided_sign_flip,
    holm_adjust,
    paired_rank_biserial,
)

STUDY_ID = "cowa_hidden_selector_migration_v1"
P0_1C_SELECTOR_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"
DATASETS = ("product", "duck", "dog", "weather")
RHOS = ("1/10", "3/10", "1/2", "7/10", "9/10")
PERIOD = 20
P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
P0_1C_SUMMARY_REL = "results/p0_1c_hidden_b1_v1/formal_summary.json"
P0_1C_RAW_REL = "results/p0_1c_hidden_b1_v1/formal_runs.jsonl"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
OUT_REL = "results/cowa_hidden_selector_migration_v1"
RAW_NAME = "formal_runs.jsonl"
SUMMARY_NAME = "formal_summary.json"
COMPARISONS_NAME = "paired_comparisons.csv"
MEANS_NAME = "figure_cowa_hidden_metric_means.csv"
AUDIT_NAME = "migration_audit.json"
CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}


class COWAHiddenMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stream:
    dataset_id: str
    rho: str
    replicate: int
    subset_seed: int
    malicious_workers: tuple[str, ...]

    @property
    def pairing_id(self) -> str:
        digest = hashlib.sha256("\n".join(self.malicious_workers).encode("utf-8")).hexdigest()
        return f"dataset={self.dataset_id}|rho={self.rho}|rep={self.replicate}|subset={digest}"


@dataclass
class COWAHistory:
    sums: dict[str, Fraction]
    counts: dict[str, int]
    c0: Fraction

    @classmethod
    def initialize(cls, worker_ids: Sequence[str], c0: Any = "1/2") -> "COWAHistory":
        return cls(
            sums={str(w): Fraction(0, 1) for w in worker_ids},
            counts={str(w): 0 for w in worker_ids},
            c0=_fraction(c0),
        )

    def weight(self, worker_id: str) -> Fraction:
        w = str(worker_id)
        count = self.counts.get(w, 0)
        if count <= 0:
            return self.c0
        return self.sums[w] / count

    def observe(self, worker_id: str, q_cal: Any) -> None:
        w = str(worker_id)
        if w not in self.sums:
            self.sums[w] = Fraction(0, 1)
            self.counts[w] = 0
        self.sums[w] += _fraction(q_cal)
        self.counts[w] += 1

    def workers_with_evidence(self) -> int:
        return sum(v > 0 for v in self.counts.values())


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    text = str(value)
    if "/" in text:
        return Fraction(text)
    return Fraction(Decimal(text))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def expected_n(dataset: str, rho: str) -> int:
    return 9 if dataset == "weather" and rho in {"1/10", "9/10"} else 20


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> int:
    total = 0
    for ds in DATASETS:
        for rho in RHOS:
            rows = manifest["datasets"][ds][rho]["subsets"]
            n = expected_n(ds, rho)
            if len(rows) != n:
                raise COWAHiddenMigrationError(f"CELL_N:{ds}:{rho}:{len(rows)}:{n}")
            subsets = [tuple(sorted(map(str, x["workers"]))) for x in rows]
            if len(subsets) != len(set(subsets)):
                raise COWAHiddenMigrationError(f"DUPLICATE_SUBSET:{ds}:{rho}")
            total += n
    if total != 378:
        raise COWAHiddenMigrationError(f"STREAM_COUNT:{total}:378")
    return total


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
        valid = set(datasets[ds].worker_ids)
        for rho in RHOS:
            for rep, row in enumerate(manifest["datasets"][ds][rho]["subsets"], start=1):
                workers = tuple(sorted(map(str, row["workers"])))
                if not set(workers) <= valid:
                    raise COWAHiddenMigrationError(f"UNKNOWN_WORKER:{ds}:{rho}:{rep}")
                out.append(Stream(ds, rho, rep, int(row["seed"]), workers))
    if len(out) != 378:
        raise COWAHiddenMigrationError(f"PLANNED_STREAMS:{len(out)}:378")
    return out


def _selector_config_hash(dataset: RealDataset, manifest_sha: str) -> str:
    # MUST exactly reproduce P0-1C.  We intentionally bind to the P0-1C study
    # identifier rather than this migration's identifier so the same private
    # seed reproduces the exact same hidden calibration task set.
    return _sha_obj({
        "study": P0_1C_SELECTOR_STUDY_ID,
        "dataset": dataset.dataset_id,
        "answer_sha256": dataset.answer_sha256,
        "truth_sha256": dataset.truth_sha256,
        "p0_2_manifest_sha256": manifest_sha,
        "algorithm_config": CONFIG,
        "nominal_period": PERIOD,
        "selector_rule": "HMAC-SHA256 over immutable task_id; Bernoulli probability 1/P",
    })


def _load_selectors(root: Path, datasets: Mapping[str, RealDataset], manifest_sha: str) -> dict[str, HiddenCalibrationSchedule]:
    seed_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seed_obj.get("seeds", {}).items()}
    if set(seeds) != set(DATASETS):
        raise COWAHiddenMigrationError("P0_1C_SELECTOR_SEED_DATASET_SET_MISMATCH")
    return {
        ds: HiddenCalibrationSchedule(
            seed=bytes.fromhex(seeds[ds]),
            session_id=f"LIR-PPTD/final-real-data/{ds}",
            selector_epoch=0,
            config_hash=_selector_config_hash(datasets[ds], manifest_sha),
            period=PERIOD,
        )
        for ds in DATASETS
    }


def _verify_masks(
    datasets: Mapping[str, RealDataset],
    selectors: Mapping[str, HiddenCalibrationSchedule],
    p0_1c_audit: Mapping[str, Any],
) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    expected = {str(x["dataset"]): dict(x) for x in p0_1c_audit["selectors"]}
    masks: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    for ds in DATASETS:
        task_ids = [t.task_id for t in datasets[ds].tasks]
        cal_ids = frozenset(selectors[ds].selected_task_ids(task_ids))
        mask_hash = _sha_obj(sorted(cal_ids))
        e = expected[ds]
        if mask_hash != str(e["calibration_mask_sha256"]):
            raise COWAHiddenMigrationError(f"P0_1C_MASK_HASH_MISMATCH:{ds}:{mask_hash}:{e['calibration_mask_sha256']}")
        if len(cal_ids) != int(e["calibration_task_count"]):
            raise COWAHiddenMigrationError(f"P0_1C_CAL_COUNT_MISMATCH:{ds}")
        if selectors[ds].seed_commitment_sha256 != str(e["seed_commitment_sha256"]):
            raise COWAHiddenMigrationError(f"P0_1C_SEED_COMMITMENT_MISMATCH:{ds}")
        masks[ds] = cal_ids
        hashes[ds] = mask_hash
    return masks, hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_lir_source(
    root: Path,
    streams: Sequence[Stream],
    mask_hashes: Mapping[str, str],
    manifest_sha: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    summary = _load_json(root / P0_1C_SUMMARY_REL)
    if int(summary.get("success", -1)) != 1512 or int(summary.get("failure", -1)) != 0:
        raise COWAHiddenMigrationError("P0_1C_FORMAL_SUMMARY_NOT_COMPLETE")
    if str(summary.get("p0_2_manifest_sha256")) != manifest_sha:
        raise COWAHiddenMigrationError("P0_1C_MANIFEST_BINDING_MISMATCH")
    raw = _read_jsonl(root / P0_1C_RAW_REL)
    lir = [r for r in raw if r.get("method") == "lir_pptd" and r.get("status") == "success"]
    if len(lir) != 378:
        raise COWAHiddenMigrationError(f"P0_1C_LIR_ROW_COUNT:{len(lir)}:378")
    by: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in lir:
        key = (str(row["dataset"]), str(row["rho"]), int(row["replicate"]))
        if key in by:
            raise COWAHiddenMigrationError(f"DUPLICATE_LIR_SOURCE:{key}")
        if str(row["calibration_mask_sha256"]) != mask_hashes[key[0]]:
            raise COWAHiddenMigrationError(f"LIR_MASK_BINDING_MISMATCH:{key}")
        by[key] = row
    for stream in streams:
        key = (stream.dataset_id, stream.rho, stream.replicate)
        if key not in by:
            raise COWAHiddenMigrationError(f"MISSING_LIR_SOURCE:{key}")
        if str(by[key]["pairing_id"]) != stream.pairing_id:
            raise COWAHiddenMigrationError(f"LIR_PAIRING_ID_MISMATCH:{key}")
    return by


def _attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    # Same P0-1C ordering: attack has no access to hidden mode designation.
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


def _class_from_vector(values: Sequence[float]) -> int:
    maximum = max(values)
    return min(i for i, value in enumerate(values) if value == maximum)


def cowa_predict(task: RealTask, history: COWAHistory) -> tuple[tuple[float, ...], int | None]:
    if not task.participant_ids:
        raise COWAHiddenMigrationError("EMPTY_TASK")
    dimension = len(next(iter(task.reports.values())))
    numerator = [Fraction(0, 1) for _ in range(dimension)]
    denominator = Fraction(0, 1)
    for worker in task.participant_ids:
        weight = history.weight(worker)
        if weight < 0:
            raise COWAHiddenMigrationError("NEGATIVE_COWA_WEIGHT")
        denominator += weight
        report = task.reports[worker]
        for h in range(dimension):
            numerator[h] += weight * _fraction(report[h])
    if denominator <= 0:
        raise COWAHiddenMigrationError("NONPOSITIVE_COWA_DENOMINATOR")
    vector = tuple(float(x / denominator) for x in numerator)
    cls = _class_from_vector(vector) if task.modality == "categorical" else None
    return vector, cls


def _record_prediction(
    dataset: RealDataset, task: RealTask, vector: Sequence[float], cls: int | None,
    pc: list[int], tc: list[int], pv: list[float], tv: list[float],
) -> None:
    if dataset.modality == "categorical":
        if cls is None or task.truth_class is None:
            raise COWAHiddenMigrationError("CATEGORICAL_PREDICTION_MISSING")
        pc.append(int(cls)); tc.append(int(task.truth_class))
    else:
        pv.append(float(vector[0])); tv.append(float(task.truth_vector[0]))


def _collect(dataset: RealDataset, pc, tc, pv, tv) -> dict[str, Any]:
    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=pc if dataset.modality == "categorical" else None,
        truth_classes=tc if dataset.modality == "categorical" else None,
        predicted_values=pv if dataset.modality == "numerical" else None,
        truth_values=tv if dataset.modality == "numerical" else None,
    )
    primary_loss = float(losses["primary_loss"])
    if dataset.dataset_id == "weather":
        metric_name = "MAE"; paper_metric = primary_loss; higher = False
    else:
        metric_name = "Accuracy" if dataset.dataset_id == "duck" else "MacroF1"
        paper_metric = 1.0 - primary_loss; higher = True
    return {
        **losses,
        "paper_metric_name": metric_name,
        "paper_metric": paper_metric,
        "higher_is_better": higher,
        "evaluation_task_count": len(pc) if dataset.modality == "categorical" else len(pv),
    }


def _run_cowa(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str]) -> dict[str, Any]:
    history = COWAHistory.initialize(dataset.worker_ids, CONFIG["c0"])
    # Calibration evidence is q_cal from exactly the same attacked calibration
    # report and authenticated reference used by LIR.  The bounded F(c,q)
    # dynamics are intentionally NOT used by COWA.
    evidence_state = initial_global_state(dataset.worker_ids, CONFIG["c0"])
    pc: list[int] = []; tc: list[int] = []; pv: list[float] = []; tv: list[float] = []
    cal_events = 0
    evidence_observations = 0
    for task in tasks:
        if task.task_id in cal_ids:
            result, _unused = run_calibration_task(task, CONFIG, evidence_state, dps=80)
            for item in result.evidence:
                history.observe(str(item.worker_id), item.evidence)
                evidence_observations += 1
            cal_events += 1
            continue
        vector, cls = cowa_predict(task, history)
        _record_prediction(dataset, task, vector, cls, pc, tc, pv, tv)
    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "calibration_events": cal_events,
        "cowa_evidence_observations": evidence_observations,
        "cowa_workers_with_calibration_evidence": history.workers_with_evidence(),
        "cowa_weight_rule": "mean_q_cal_so_far_else_c0",
        "cowa_ordinary_weight_update_count": 0,
    }


def _run_id(stream: Stream, mask_hash: str) -> str:
    return hashlib.sha256(f"{STUDY_ID}|{mask_hash}|{stream.pairing_id}|cowa".encode("utf-8")).hexdigest()


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        out[str(row["run_id"])] = row
    return out


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    order_ds = {d: i for i, d in enumerate(DATASETS)}
    order_rho = {r: i for i, r in enumerate(RHOS)}
    ordered = sorted((dict(r) for r in rows), key=lambda r: (order_ds[str(r["dataset"])], order_rho[str(r["rho"])], int(r["replicate"])))
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def _analyze(
    cowa_rows: Sequence[Mapping[str, Any]],
    lir_source: Mapping[tuple[str, str, int], Mapping[str, Any]],
    manifest_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cowa = {(str(r["dataset"]), str(r["rho"]), int(r["replicate"])): dict(r) for r in cowa_rows if r.get("status") == "success"}
    comparisons: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    for ds in DATASETS:
        for rho in RHOS:
            n = expected_n(ds, rho)
            reps = range(1, n + 1)
            lir_losses = [_fraction(lir_source[(ds, rho, rep)]["primary_loss"]) for rep in reps]
            cowa_losses = [_fraction(cowa[(ds, rho, rep)]["primary_loss"]) for rep in reps]
            deltas = [a - b for a, b in zip(lir_losses, cowa_losses)]
            mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
            ci_low, ci_high, boot_seed = bootstrap_mean_ci(
                deltas, seed_text=f"{STUDY_ID}|{manifest_sha}|{ds}|{rho}|LIR-vs-COWA-hidden"
            )
            p = exact_two_sided_sign_flip(deltas)
            rb = paired_rank_biserial(deltas)
            lir_metrics = [float(lir_source[(ds, rho, rep)]["paper_metric"]) for rep in reps]
            cowa_metrics = [float(cowa[(ds, rho, rep)]["paper_metric"]) for rep in reps]
            means.append({
                "dataset": ds, "rho": rho, "n": n,
                "lir_pptd": statistics.fmean(lir_metrics),
                "cowa": statistics.fmean(cowa_metrics),
                "paper_metric_name": str(cowa[(ds, rho, 1)]["paper_metric_name"]),
            })
            comparisons.append({
                "dataset": ds, "rho": rho, "n": n,
                "candidate": "lir_pptd", "comparator": "cowa",
                "lir_mean_loss": float(statistics.fmean(float(x) for x in lir_losses)),
                "cowa_mean_loss": float(statistics.fmean(float(x) for x in cowa_losses)),
                "mean_loss_delta_lir_minus_cowa": float(mean_delta),
                "lir_pair_wins": sum(x < 0 for x in deltas),
                "pair_ties": sum(x == 0 for x in deltas),
                "lir_pair_losses": sum(x > 0 for x in deltas),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "bootstrap_seed": boot_seed,
                "exact_two_sided_sign_flip_p": float(p),
                "paired_rank_biserial": float(rb),
                "_exact_p_fraction": p,
            })

    raw = {f"{r['dataset']}|{r['rho']}": r["_exact_p_fraction"] for r in comparisons}
    adjusted = holm_adjust(raw)
    for r in comparisons:
        key = f"{r['dataset']}|{r['rho']}"
        r["holm_family"] = "COWA-HIDDEN|20-cells"
        r["holm_adjusted_p"] = float(adjusted[key])
        r["holm_significant"] = adjusted[key] <= Fraction(1, 20)
        delta = float(r["mean_loss_delta_lir_minus_cowa"])
        r["direction"] = "lir_win" if delta < 0 else ("cowa_win" if delta > 0 else "tie")
        r["holm_direction"] = r["direction"] if r["holm_significant"] else "not_significant"
        r.pop("_exact_p_fraction", None)

    by_dataset: dict[str, Any] = {}
    for ds in DATASETS:
        fam = [r for r in comparisons if r["dataset"] == ds]
        by_dataset[ds] = {
            "mean_lir_wins": sum(r["direction"] == "lir_win" for r in fam),
            "mean_ties": sum(r["direction"] == "tie" for r in fam),
            "mean_cowa_wins": sum(r["direction"] == "cowa_win" for r in fam),
            "holm_lir_wins": sum(r["holm_direction"] == "lir_win" for r in fam),
            "holm_cowa_wins": sum(r["holm_direction"] == "cowa_win" for r in fam),
        }
    high = [r for r in comparisons if r["rho"] in {"1/2", "7/10", "9/10"}]
    summary = {
        "cell_count": 20,
        "mean_lir_wins": sum(r["direction"] == "lir_win" for r in comparisons),
        "mean_ties": sum(r["direction"] == "tie" for r in comparisons),
        "mean_cowa_wins": sum(r["direction"] == "cowa_win" for r in comparisons),
        "holm_significant_lir_wins": sum(r["holm_direction"] == "lir_win" for r in comparisons),
        "holm_significant_cowa_wins": sum(r["holm_direction"] == "cowa_win" for r in comparisons),
        "by_dataset": by_dataset,
        "rho_ge_0_5": {
            "cell_count": 12,
            "mean_lir_wins": sum(r["direction"] == "lir_win" for r in high),
            "mean_ties": sum(r["direction"] == "tie" for r in high),
            "mean_cowa_wins": sum(r["direction"] == "cowa_win" for r in high),
            "holm_lir_wins": sum(r["holm_direction"] == "lir_win" for r in high),
            "holm_cowa_wins": sum(r["holm_direction"] == "cowa_win" for r in high),
        },
        "interpretation_gate": "STATIC_EQUAL_INFORMATION_RESULT_ONLY",
        "retuning_allowed": False,
    }
    return comparisons, means, summary


def validate(root: Path):
    required = [
        root / P0_2_MANIFEST_REL,
        root / P0_1C_SEEDS_REL,
        root / P0_1C_AUDIT_REL,
        root / P0_1C_SUMMARY_REL,
        root / P0_1C_RAW_REL,
        root / DATASET_HASHES_REL,
        root / "src/lir_pptd/core/hidden_calibration.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise COWAHiddenMigrationError("MISSING:" + ",".join(missing))
    manifest_path = root / P0_2_MANIFEST_REL
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise COWAHiddenMigrationError(f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}")
    manifest = _load_json(manifest_path)
    stream_count = _validate_manifest_shape(manifest)
    datasets = _load_datasets(root)
    streams = _plan_streams(manifest, datasets)
    selectors = _load_selectors(root, datasets, manifest_sha)
    masks, mask_hashes = _verify_masks(datasets, selectors, _load_json(root / P0_1C_AUDIT_REL))
    lir_source = _load_lir_source(root, streams, mask_hashes, manifest_sha)
    print("COWA_HIDDEN_SELECTOR_VALIDATE=PASS")
    print(f"P0_2_MANIFEST_SHA256={manifest_sha}")
    print(f"STREAMS={stream_count}")
    print("STANDARD_CELLS=18x20")
    print("WEATHER_EXTREME_CELLS=2x9")
    print("LIR_SOURCE=P0_1C_FORMAL_RESULTS")
    print("LIR_RERUN=NO")
    print(f"PLANNED_COWA_RUNS={len(streams)}")
    print("P0_1C_PRIVATE_SELECTOR_SEEDS_REUSED=YES")
    print("P0_1C_CALIBRATION_MASK_REUSE=EXACT")
    print("SAME_ATTACKED_STREAM_AS_P0_1C=YES")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("COWA_RECEIVES_SAME_CALIBRATION_TASKS_AND_REFERENCES_AS_LIR=YES")
    print("COWA_WEIGHT_RULE=MEAN_Q_CAL_SO_FAR_ELSE_C0")
    print("COWA_CURRENT_TASK_CONSISTENCY=NO")
    print("COWA_BOUNDED_REPUTATION_DYNAMICS=NO")
    print("COWA_ORDINARY_WEIGHT_UPDATE=NO")
    print("FORMAL_B1_OR_LIR_RERUN=NO")
    return datasets, streams, manifest_sha, selectors, masks, mask_hashes, lir_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    datasets, streams, manifest_sha, selectors, masks, mask_hashes, lir_source = validate(root)
    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / RAW_NAME
    if raw_path.exists() and not args.resume:
        raise COWAHiddenMigrationError(f"FORMAL_OUTPUT_EXISTS_USE_RESUME:{raw_path}")
    existing = _read_existing(raw_path)
    for idx, stream in enumerate(streams, start=1):
        rid = _run_id(stream, mask_hashes[stream.dataset_id])
        if rid in existing and existing[rid].get("status") == "success":
            continue
        dataset = datasets[stream.dataset_id]
        attacked = _attacked_tasks(dataset, stream)
        t0 = time.perf_counter()
        try:
            result = _run_cowa(dataset, attacked, masks[stream.dataset_id])
            source = lir_source[(stream.dataset_id, stream.rho, stream.replicate)]
            if int(result["evaluation_task_count"]) != int(source["evaluation_task_count"]):
                raise COWAHiddenMigrationError("SCORING_MASK_COUNT_MISMATCH")
            row = {
                "schema_version": "1.0",
                "study_id": STUDY_ID,
                "run_id": rid,
                "pairing_id": stream.pairing_id,
                "dataset": stream.dataset_id,
                "rho": stream.rho,
                "replicate": stream.replicate,
                "subset_seed": stream.subset_seed,
                "malicious_workers": list(stream.malicious_workers),
                "status": "success",
                "method": "cowa",
                "p0_2_manifest_sha256": manifest_sha,
                "p0_1c_lir_run_id": str(source["run_id"]),
                "calibration_mask_sha256": mask_hashes[stream.dataset_id],
                "seed_commitment_sha256": selectors[stream.dataset_id].seed_commitment_sha256,
                "pre_freeze_mode_visible_to_attack": False,
                "same_calibration_reference_as_lir": True,
                "same_scoring_mask_as_lir": True,
                **result,
                "elapsed_seconds": time.perf_counter() - t0,
            }
        except Exception as exc:
            row = {
                "schema_version": "1.0", "study_id": STUDY_ID, "run_id": rid,
                "pairing_id": stream.pairing_id, "dataset": stream.dataset_id,
                "rho": stream.rho, "replicate": stream.replicate,
                "subset_seed": stream.subset_seed, "malicious_workers": list(stream.malicious_workers),
                "status": "failure", "method": "cowa",
                "calibration_mask_sha256": mask_hashes[stream.dataset_id],
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.perf_counter() - t0,
            }
        _append(raw_path, row); existing[rid] = row
        if idx % 10 == 0 or idx == len(streams):
            success_now = sum(r.get("status") == "success" for r in existing.values())
            print(f"PROGRESS={idx}/{len(streams)} SUCCESS={success_now}/{len(streams)}", flush=True)

    _rewrite_canonical(raw_path, existing.values())
    final = list(_read_existing(raw_path).values())
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)
    analysis = None
    if success == 378 and failure == 0:
        comparisons, means, analysis = _analyze(final, lir_source, manifest_sha)
        _write_csv(outdir / COMPARISONS_NAME, comparisons)
        _write_csv(outdir / MEANS_NAME, means)

    audit = {
        "schema_version": "1.0", "study_id": STUDY_ID,
        "p0_2_manifest_sha256": manifest_sha,
        "p0_1c_lir_reused": True,
        "p0_1c_lir_rerun": False,
        "p0_1c_private_selector_seeds_reused": True,
        "p0_1c_calibration_masks_reused_exactly": True,
        "same_attacked_streams": True,
        "same_scoring_masks": True,
        "cowa_semantics": {
            "same_authenticated_calibration_tasks_and_references_as_lir": True,
            "historical_weight": "mean authenticated q_cal observed so far, c0 before first evidence",
            "ordinary_categorical": "weighted vote",
            "ordinary_numerical": "weighted mean",
            "current_task_consistency": False,
            "bounded_reputation_dynamics_F": False,
            "ordinary_weight_update": False,
        },
        "selector_masks": [
            {"dataset": ds, "count": len(masks[ds]), "mask_sha256": mask_hashes[ds], "seed_commitment_sha256": selectors[ds].seed_commitment_sha256}
            for ds in DATASETS
        ],
    }
    _atomic_json(outdir / AUDIT_NAME, audit)
    summary = {
        "schema_version": "1.0", "study_id": STUDY_ID,
        "stream_count": 378, "planned_cowa_runs": 378,
        "completed_unique_cowa_runs": len(final), "success": success, "failure": failure,
        "lir_source": "results/p0_1c_hidden_b1_v1/formal_runs.jsonl",
        "lir_rerun": False,
        "analysis": analysis,
        "scope": {"static_equal_information_control": True, "dynamic_cowa_stress": False, "algorithm_retuning": False},
    }
    _atomic_json(outdir / SUMMARY_NAME, summary)
    print("COWA_HIDDEN_SELECTOR_MIGRATION=COMPLETE")
    print("STREAMS=378")
    print("LIR_RERUN=NO")
    print("PLANNED_COWA_RUNS=378")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")
    if analysis is not None:
        print(f"LIR_COWA_MEAN_WINS_TIES_LOSSES={analysis['mean_lir_wins']}/{analysis['mean_ties']}/{analysis['mean_cowa_wins']}")
        print(f"LIR_COWA_HOLM_SIG_WINS_LOSSES={analysis['holm_significant_lir_wins']}/{analysis['holm_significant_cowa_wins']}")
        print(f"RHO_GE_0_5_LIR_COWA_MEAN_WINS_TIES_LOSSES={analysis['rho_ge_0_5']['mean_lir_wins']}/{analysis['rho_ge_0_5']['mean_ties']}/{analysis['rho_ge_0_5']['mean_cowa_wins']}")
        for ds in DATASETS:
            x = analysis["by_dataset"][ds]
            print(f"{ds.upper()}_LIR_COWA_MEAN_WINS_TIES_LOSSES={x['mean_lir_wins']}/{x['mean_ties']}/{x['mean_cowa_wins']}")
        print("DYNAMIC_COWA_STRESS_DECISION=REVIEW_AFTER_STATIC_RESULT")
    print(f"SUMMARY={(outdir / SUMMARY_NAME).relative_to(root)}")
    return 0 if success == 378 and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
