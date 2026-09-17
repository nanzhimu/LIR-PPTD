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

from lir_pptd.baselines.pptd_source_semantic import predict as pptd_predict
from lir_pptd.baselines.tqpp_source_semantic import (
    TQPPAdaptationProfile,
    initial_state as tqpp_initial_state,
    run_task as tqpp_run_task,
)
from lir_pptd.core.hidden_calibration import HiddenCalibrationSchedule
from lir_pptd.experiments.calibration_anchored import run_calibration_task
from lir_pptd.experiments.phase_r1.formal_runner import _load_config
from lir_pptd.experiments.phase_r1.longitudinal_state import initial_global_state
from lir_pptd.experiments.phase_r1.methods import run_lir_task, run_stateless_baseline
from lir_pptd.experiments.phase_r1.metrics import dataset_losses
from lir_pptd.experiments.phase_r1.real_data_loader import RealDataset, RealTask, load_real_dataset
from lir_pptd.experiments.phase_r1.statistics import (
    bootstrap_mean_ci,
    exact_two_sided_sign_flip,
    holm_adjust,
    paired_rank_biserial,
)

STUDY_ID = "b2_hidden_selector_migration_v1"
P0_1C_SELECTOR_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

DATASETS = ("product", "duck", "dog", "weather")
ATTACKS = ("random", "collusive", "on_off")
METHODS = ("lir_pptd", "crh", "pptd_liang", "tqpp_liu")
BASELINES = ("crh", "pptd_liang", "tqpp_liu")

RHO = "7/10"
REPLICATES = 20
SEED_START = 83001
PERIOD = 20
TQPP_PROFILE = TQPPAdaptationProfile()

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
METHOD_CONFIG_REL = "docs/PHASE6_R2_METHOD_CONFIG_DECISIONS.json"

OUT_REL = "results/b2_hidden_selector_migration_v1"
RAW_NAME = "formal_runs.jsonl"
SUMMARY_NAME = "formal_summary.json"
COMPARISONS_NAME = "paired_comparisons.csv"
MEANS_NAME = "figure_b2_hidden_metric_means.csv"
SECONDARY_MEANS_NAME = "figure_b2_hidden_secondary_metric_means.csv"
AUDIT_NAME = "migration_audit.json"
SUBSET_MANIFEST_NAME = "b2_subset_manifest.json"


class B2HiddenError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaseSubset:
    dataset_id: str
    replicate: int
    subset_seed: int
    malicious_workers: tuple[str, ...]
    trusted_workers: tuple[str, ...]

    @property
    def subset_digest(self) -> str:
        return hashlib.sha256("\n".join(self.malicious_workers).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Stream:
    base: BaseSubset
    attack: str

    @property
    def pairing_id(self) -> str:
        phase = f"|phase={phase_offset(self.base.replicate)}" if self.attack == "on_off" else ""
        return (
            f"dataset={self.base.dataset_id}|rho={RHO}|rep={self.base.replicate}"
            f"|subset={self.base.subset_digest}|attack={self.attack}{phase}"
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
        if line.strip():
            row = json.loads(line)
            out[str(row["run_id"])] = row
    return out


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_order = {d: i for i, d in enumerate(DATASETS)}
    attack_order = {a: i for i, a in enumerate(ATTACKS)}
    method_order = {m: i for i, m in enumerate(METHODS)}
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_order[str(r["dataset"])],
            attack_order[str(r["attack"])],
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fraction(value: Any) -> Fraction:
    text = str(value)
    if "/" in text:
        return Fraction(text)
    return Fraction(Decimal(text))


def load_datasets(root: Path) -> dict[str, RealDataset]:
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


def _coverage(dataset: RealDataset) -> dict[str, int]:
    counts = {str(w): 0 for w in dataset.worker_ids}
    for task in dataset.tasks:
        for worker in task.participant_ids:
            counts[str(worker)] += 1
    return counts


def fixed_tqpp_trusted_workers(dataset: RealDataset) -> tuple[str, ...]:
    """Choose TQPP's trusted set once per dataset before attack construction."""
    coverage = _coverage(dataset)
    count = TQPP_PROFILE.trusted_count(len(dataset.worker_ids))
    ranked = sorted(
        map(str, dataset.worker_ids),
        key=lambda w: (-coverage.get(w, 0), w),
    )
    return tuple(ranked[:count])


def _malicious_count(total_workers: int) -> int:
    # Exact half-up rounding for 0.7*N.
    return (14 * total_workers + 10) // 20


def _rank_unknown(unknown: tuple[str, ...], dataset_id: str, seed: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            unknown,
            key=lambda w: hashlib.sha256(
                f"B2-v1|{dataset_id}|{RHO}|{seed}|{w}".encode("utf-8")
            ).digest(),
        )
    )


def plan_base_subsets(dataset: RealDataset) -> list[BaseSubset]:
    trusted = fixed_tqpp_trusted_workers(dataset)
    trusted_set = set(trusted)
    unknown = tuple(sorted(w for w in map(str, dataset.worker_ids) if w not in trusted_set))
    malicious_count = _malicious_count(len(dataset.worker_ids))
    if malicious_count > len(unknown):
        raise B2HiddenError(
            f"NOT_ENOUGH_NONTRUSTED_WORKERS:{dataset.dataset_id}:{malicious_count}>{len(unknown)}"
        )

    seen: set[tuple[str, ...]] = set()
    out: list[BaseSubset] = []
    seed = SEED_START
    while len(out) < REPLICATES and seed < SEED_START + 100000:
        ranked = _rank_unknown(unknown, dataset.dataset_id, seed)
        subset = tuple(sorted(ranked[:malicious_count]))
        if subset not in seen:
            if set(subset) & trusted_set:
                raise B2HiddenError(f"TRUSTED_MALICIOUS_OVERLAP:{dataset.dataset_id}")
            seen.add(subset)
            out.append(
                BaseSubset(
                    dataset_id=dataset.dataset_id,
                    replicate=len(out) + 1,
                    subset_seed=seed,
                    malicious_workers=subset,
                    trusted_workers=trusted,
                )
            )
        seed += 1

    if len(out) != REPLICATES:
        raise B2HiddenError(f"SUBSET_COUNT_MISMATCH:{dataset.dataset_id}:{len(out)}")
    return out


def plan_streams(datasets: Mapping[str, RealDataset]) -> tuple[list[Stream], dict[str, list[BaseSubset]]]:
    by_dataset: dict[str, list[BaseSubset]] = {}
    streams: list[Stream] = []
    for ds in DATASETS:
        bases = plan_base_subsets(datasets[ds])
        by_dataset[ds] = bases
        for base in bases:
            for attack in ATTACKS:
                streams.append(Stream(base=base, attack=attack))
    expected = len(DATASETS) * REPLICATES * len(ATTACKS)
    if len(streams) != expected:
        raise B2HiddenError(f"STREAM_COUNT_MISMATCH:{len(streams)}:{expected}")
    return streams, by_dataset


def _selector_config_hash(dataset: RealDataset) -> str:
    # Must exactly reproduce the selector namespace used by P0-1C.
    config = {
        "K": 10,
        "c0": "1/2",
        "epsilon_c": "1/1024",
        "lambda_tau": "1/5",
        "kappa": "2",
        "mu": "1/5",
        "eta": "1/10",
    }
    return _sha_obj(
        {
            "study": P0_1C_SELECTOR_STUDY_ID,
            "dataset": dataset.dataset_id,
            "answer_sha256": dataset.answer_sha256,
            "truth_sha256": dataset.truth_sha256,
            "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
            "algorithm_config": config,
            "nominal_period": PERIOD,
            "selector_rule": "HMAC-SHA256 over immutable task_id; Bernoulli probability 1/P",
        }
    )


def load_hidden_masks(
    root: Path,
    datasets: Mapping[str, RealDataset],
) -> tuple[
    dict[str, HiddenCalibrationSchedule],
    dict[str, frozenset[str]],
    dict[str, str],
    dict[str, int],
]:
    seed_obj = _load_json(root / P0_1C_SEEDS_REL)
    seed_map = {str(k): str(v) for k, v in seed_obj.get("seeds", {}).items()}
    if set(seed_map) != set(DATASETS):
        raise B2HiddenError("P0_1C_SELECTOR_SEED_DATASET_SET_MISMATCH")

    audit_obj = _load_json(root / P0_1C_AUDIT_REL)
    expected = {str(x["dataset"]): dict(x) for x in audit_obj["selectors"]}

    selectors: dict[str, HiddenCalibrationSchedule] = {}
    masks: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}

    for ds in DATASETS:
        selector = HiddenCalibrationSchedule(
            seed=bytes.fromhex(seed_map[ds]),
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
            raise B2HiddenError(f"P0_1C_MASK_HASH_MISMATCH:{ds}")
        if len(cal_ids) != int(exp["calibration_task_count"]):
            raise B2HiddenError(f"P0_1C_MASK_COUNT_MISMATCH:{ds}")
        if selector.seed_commitment_sha256 != str(exp["seed_commitment_sha256"]):
            raise B2HiddenError(f"P0_1C_COMMITMENT_MISMATCH:{ds}")

        selectors[ds] = selector
        masks[ds] = cal_ids
        hashes[ds] = mask_hash
        counts[ds] = len(cal_ids)

    return selectors, masks, hashes, counts


def _hash_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _random_report(
    report: tuple[str | float | int, ...],
    *,
    modality: str,
    namespace: str,
) -> tuple[str | float | int, ...]:
    width = len(report)
    if modality == "categorical":
        if width < 2:
            raise B2HiddenError("CATEGORICAL_WIDTH_LT_2")
        cls = _hash_u64(namespace + "|class") % width
        return tuple(1 if i == cls else 0 for i in range(width))
    if modality == "numerical":
        vals: list[str] = []
        for j in range(width):
            raw = _hash_u64(namespace + f"|coord={j}")
            vals.append(format(raw / 2**64, ".17g"))
        return tuple(vals)
    raise B2HiddenError(f"UNSUPPORTED_MODALITY:{modality}")


def _fixed_target_reports(
    reports: Mapping[str, Sequence[str | float | int]],
    malicious_workers: set[str],
    *,
    modality: str,
) -> dict[str, tuple[str | float | int, ...]]:
    out: dict[str, tuple[str | float | int, ...]] = {}
    for worker, report in reports.items():
        current = tuple(report)
        if str(worker) not in malicious_workers:
            out[str(worker)] = current
            continue
        if modality == "categorical":
            out[str(worker)] = tuple(1 if i == 0 else 0 for i in range(len(current)))
        elif modality == "numerical":
            out[str(worker)] = tuple("1" for _ in current)
        else:
            raise B2HiddenError(f"UNSUPPORTED_MODALITY:{modality}")
    return out


def phase_offset(replicate: int) -> int:
    # 20 replicates -> offsets 0..9 exactly twice.
    return (replicate - 1) % 10


def onoff_active(round_index_1based: int, offset: int) -> bool:
    pos = ((round_index_1based - 1) + offset) % 10
    return pos >= 5


def attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    """Generate all submitted reports without access to calibration designation."""
    malicious = set(stream.base.malicious_workers)
    offset = phase_offset(stream.base.replicate)
    out: list[RealTask] = []

    for round_index, task in enumerate(dataset.tasks, start=1):
        if stream.attack == "random":
            reports: dict[str, tuple[str | float | int, ...]] = {}
            for worker, report in task.reports.items():
                if str(worker) in malicious:
                    reports[str(worker)] = _random_report(
                        tuple(report),
                        modality=task.modality,
                        namespace=f"{stream.pairing_id}|round={round_index}|worker={worker}",
                    )
                else:
                    reports[str(worker)] = tuple(report)
        elif stream.attack == "collusive":
            reports = _fixed_target_reports(
                task.reports,
                malicious,
                modality=task.modality,
            )
        elif stream.attack == "on_off":
            if onoff_active(round_index, offset):
                reports = _fixed_target_reports(
                    task.reports,
                    malicious,
                    modality=task.modality,
                )
            else:
                reports = {str(w): tuple(v) for w, v in task.reports.items()}
        else:
            raise B2HiddenError(f"UNKNOWN_ATTACK:{stream.attack}")

        out.append(replace(task, reports=reports))
    return tuple(out)


def _record(
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
            raise B2HiddenError("CATEGORICAL_PREDICTION_MISSING")
        pc.append(int(cls))
        tc.append(int(task.truth_class))
    else:
        pv.append(float(vector[0]))
        tv.append(float(task.truth_vector[0]))


def _collect(dataset: RealDataset, pc, tc, pv, tv) -> dict[str, Any]:
    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=pc if dataset.modality == "categorical" else None,
        truth_classes=tc if dataset.modality == "categorical" else None,
        predicted_values=pv if dataset.modality == "numerical" else None,
        truth_values=tv if dataset.modality == "numerical" else None,
    )
    primary_loss = float(losses["primary_loss"])
    secondary_loss = float(losses["secondary_loss"])
    if dataset.dataset_id == "weather":
        return {
            **losses,
            "paper_metric_name": "MAE",
            "paper_metric": primary_loss,
            "secondary_metric_name": "RMSE",
            "secondary_metric": secondary_loss,
            "higher_is_better": False,
        }
    return {
        **losses,
        "paper_metric_name": "Accuracy" if dataset.dataset_id == "duck" else "MacroF1",
        "paper_metric": 1.0 - primary_loss,
        "secondary_metric_name": "MacroF1" if dataset.dataset_id == "duck" else "Accuracy",
        "secondary_metric": 1.0 - secondary_loss,
        "higher_is_better": True,
    }


def run_lir(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    state = initial_global_state(dataset.worker_ids, config["c0"])
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    calibration_commits = 0

    for task in tasks:
        if task.task_id in cal_ids:
            _, state = run_calibration_task(task, config, state, dps=80)
            calibration_commits += 1
            continue

        result = run_lir_task(task, config, state, ablation="full", dps=80)
        _record(
            dataset,
            task,
            result.prediction_vector,
            result.prediction_class,
            pc,
            tc,
            pv,
            tv,
        )
        # Evidence gate: ordinary result.next_state is intentionally not committed.

    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "calibration_persistent_commit_count": calibration_commits,
        "ordinary_persistent_commit_count": 0,
    }


def run_crh(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
) -> dict[str, Any]:
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    for task in tasks:
        if task.task_id in cal_ids:
            continue
        vector, cls = run_stateless_baseline(task, "crh")
        _record(dataset, task, vector, cls, pc, tc, pv, tv)
    return {**_collect(dataset, pc, tc, pv, tv), "calibration_task_count": len(cal_ids)}


def run_pptd(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
) -> dict[str, Any]:
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []
    iterations: list[int] = []
    for task in tasks:
        if task.task_id in cal_ids:
            continue
        result = pptd_predict(task.reports, task.modality, weight_mode="source_quadratic")
        iterations.append(int(result.iterations))
        _record(
            dataset,
            task,
            result.truth_vector,
            result.prediction_class,
            pc,
            tc,
            pv,
            tv,
        )
    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "mean_iterations": statistics.fmean(iterations) if iterations else 0.0,
    }


def run_tqpp(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
) -> dict[str, Any]:
    trusted = stream.base.trusted_workers
    malicious = set(stream.base.malicious_workers)
    if set(trusted) & malicious:
        raise B2HiddenError("TQPP_TRUSTED_SET_OVERLAPS_MALICIOUS")

    state = tqpp_initial_state(dataset.worker_ids, trusted)
    top_h_global = TQPP_PROFILE.top_h_count(len(dataset.worker_ids))
    pc: list[int] = []
    tc: list[int] = []
    pv: list[float] = []
    tv: list[float] = []

    for round_index, task in enumerate(tasks, start=1):
        result = tqpp_run_task(
            task.reports,
            task.modality,
            state,
            top_h_global=top_h_global,
            seed_namespace=f"{stream.pairing_id}|round={round_index}",
            profile=TQPP_PROFILE,
        )
        state = result.next_state
        if task.task_id in cal_ids:
            # TQPP receives no authenticated LIR calibration reference.
            # It processes the task under its native state evolution, but the
            # task is excluded from the shared scored set.
            continue
        _record(
            dataset,
            task,
            result.truth_vector,
            result.prediction_class,
            pc,
            tc,
            pv,
            tv,
        )

    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "trusted_worker_count": len(trusted),
        "trusted_workers": list(trusted),
        "trusted_set_fixed_before_attack_construction": True,
        "trusted_set_attack_independent": True,
        "top_h_global": top_h_global,
    }


def _run_method(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
    method: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if method == "lir_pptd":
        return run_lir(dataset, tasks, cal_ids, config)
    if method == "crh":
        return run_crh(dataset, tasks, cal_ids)
    if method == "pptd_liang":
        return run_pptd(dataset, tasks, cal_ids)
    if method == "tqpp_liu":
        return run_tqpp(dataset, tasks, cal_ids, stream)
    raise B2HiddenError(f"UNKNOWN_METHOD:{method}")


def _run_id(stream: Stream, method: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{STUDY_ID}|{mask_hash}|{stream.pairing_id}|{method}".encode("utf-8")
    ).hexdigest()


def _analyze(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    success_rows = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["attack"]), int(r["replicate"]), str(r["method"])): r
        for r in success_rows
    }

    means: list[dict[str, Any]] = []
    secondary_means: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for ds in DATASETS:
        for attack in ATTACKS:
            reps = list(range(1, REPLICATES + 1))

            for method in METHODS:
                rs = [by[(ds, attack, rep, method)] for rep in reps]
                vals = [_fraction(r["paper_metric"]) for r in rs]
                lo, hi, seed = bootstrap_mean_ci(
                    vals,
                    seed_text=f"{STUDY_ID}|metric|{ds}|{attack}|{method}",
                )
                means.append(
                    {
                        "dataset": ds,
                        "attack": attack,
                        "method": method,
                        "n": REPLICATES,
                        "metric": rs[0]["paper_metric_name"],
                        "mean": float(sum(vals, Fraction(0, 1)) / len(vals)),
                        "bootstrap_95_ci_low": lo,
                        "bootstrap_95_ci_high": hi,
                        "bootstrap_seed": seed,
                    }
                )

                svals = [_fraction(r["secondary_metric"]) for r in rs]
                slo, shi, sseed = bootstrap_mean_ci(
                    svals,
                    seed_text=f"{STUDY_ID}|secondary|{ds}|{attack}|{method}",
                )
                secondary_means.append(
                    {
                        "dataset": ds,
                        "attack": attack,
                        "method": method,
                        "n": REPLICATES,
                        "metric": rs[0]["secondary_metric_name"],
                        "mean": float(sum(svals, Fraction(0, 1)) / len(svals)),
                        "bootstrap_95_ci_low": slo,
                        "bootstrap_95_ci_high": shi,
                        "bootstrap_seed": sseed,
                    }
                )

            for baseline in BASELINES:
                deltas = [
                    _fraction(by[(ds, attack, rep, "lir_pptd")]["primary_loss"])
                    - _fraction(by[(ds, attack, rep, baseline)]["primary_loss"])
                    for rep in reps
                ]
                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                lo, hi, seed = bootstrap_mean_ci(
                    deltas,
                    seed_text=f"{STUDY_ID}|primary|{ds}|{attack}|{baseline}",
                )
                p = exact_two_sided_sign_flip(deltas)
                comparisons.append(
                    {
                        "dataset": ds,
                        "attack": attack,
                        "baseline": baseline,
                        "n": REPLICATES,
                        "candidate_mean_loss": statistics.fmean(
                            float(by[(ds, attack, rep, "lir_pptd")]["primary_loss"])
                            for rep in reps
                        ),
                        "baseline_mean_loss": statistics.fmean(
                            float(by[(ds, attack, rep, baseline)]["primary_loss"])
                            for rep in reps
                        ),
                        "mean_loss_delta_lir_minus_baseline": float(mean_delta),
                        "wins": sum(x < 0 for x in deltas),
                        "ties": sum(x == 0 for x in deltas),
                        "losses": sum(x > 0 for x in deltas),
                        "bootstrap_95_ci_low": lo,
                        "bootstrap_95_ci_high": hi,
                        "bootstrap_seed": seed,
                        "exact_two_sided_sign_flip_p": float(p),
                        "paired_rank_biserial": float(paired_rank_biserial(deltas)),
                        "_p_fraction": p,
                    }
                )

    # Preserve the original final B2 multiplicity discipline:
    # one global Holm family across 4 datasets × 3 attacks × 3 comparators = 36.
    raw_family = {
        f"{r['dataset']}|{r['attack']}|{r['baseline']}": r["_p_fraction"]
        for r in comparisons
    }
    adjusted = holm_adjust(raw_family)

    for row in comparisons:
        key = f"{row['dataset']}|{row['attack']}|{row['baseline']}"
        adj = adjusted[key]
        row["holm_family"] = "B2|primary|4-datasets|3-attacks|3-comparators|36"
        row["holm_family_size"] = len(raw_family)
        row["holm_adjusted_p"] = float(adj)
        row["holm_significant"] = adj <= Fraction(1, 20)
        delta = float(row["mean_loss_delta_lir_minus_baseline"])
        row["direction"] = "lir_win" if delta < 0 else ("lir_loss" if delta > 0 else "tie")
        row["holm_direction"] = row["direction"] if row["holm_significant"] else "not_significant"
        row.pop("_p_fraction", None)

    per_comparator: dict[str, Any] = {}
    for baseline in BASELINES:
        family = [r for r in comparisons if r["baseline"] == baseline]
        per_comparator[baseline] = {
            "cells": len(family),
            "mean_wins": sum(r["direction"] == "lir_win" for r in family),
            "mean_ties": sum(r["direction"] == "tie" for r in family),
            "mean_losses": sum(r["direction"] == "lir_loss" for r in family),
            "holm_significant_wins": sum(r["holm_direction"] == "lir_win" for r in family),
            "holm_significant_losses": sum(r["holm_direction"] == "lir_loss" for r in family),
        }

    per_attack: dict[str, Any] = {}
    for attack in ATTACKS:
        family = [r for r in comparisons if r["attack"] == attack]
        per_attack[attack] = {
            "pairwise_contrasts": len(family),
            "lir_mean_wins": sum(r["direction"] == "lir_win" for r in family),
            "lir_mean_ties": sum(r["direction"] == "tie" for r in family),
            "lir_mean_losses": sum(r["direction"] == "lir_loss" for r in family),
            "lir_holm_wins": sum(r["holm_direction"] == "lir_win" for r in family),
            "lir_holm_losses": sum(r["holm_direction"] == "lir_loss" for r in family),
        }

    crh_pptd_cells = 0
    for ds in DATASETS:
        for attack in ATTACKS:
            crh = next(
                r for r in comparisons
                if r["dataset"] == ds and r["attack"] == attack and r["baseline"] == "crh"
            )
            pptd = next(
                r for r in comparisons
                if r["dataset"] == ds and r["attack"] == attack and r["baseline"] == "pptd_liang"
            )
            if crh["direction"] in {"lir_win", "tie"} and pptd["direction"] in {"lir_win", "tie"}:
                crh_pptd_cells += 1

    analysis = {
        "cell_count": 12,
        "holm_family_size": 36,
        "per_comparator": per_comparator,
        "per_attack": per_attack,
        "cells_improve_or_tie_both_crh_and_pptd": crh_pptd_cells,
        "scientific_decision": "REVIEW_AFTER_RESULT",
        "retuning_allowed": False,
    }
    return comparisons, means, secondary_means, analysis


def _onoff_calibration_phase_audit(
    datasets: Mapping[str, RealDataset],
    masks: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for ds in DATASETS:
        task_positions = {
            task.task_id: index
            for index, task in enumerate(datasets[ds].tasks, start=1)
        }
        attack_exposures = 0
        benign_exposures = 0
        for rep in range(1, REPLICATES + 1):
            offset = phase_offset(rep)
            for task_id in masks[ds]:
                if onoff_active(task_positions[task_id], offset):
                    attack_exposures += 1
                else:
                    benign_exposures += 1
        expected_each = len(masks[ds]) * REPLICATES // 2
        if attack_exposures != expected_each or benign_exposures != expected_each:
            raise B2HiddenError(
                f"ONOFF_CALIBRATION_PHASE_UNBALANCED:{ds}:{attack_exposures}:{benign_exposures}:{expected_each}"
            )
        audit[ds] = {
            "calibration_task_count": len(masks[ds]),
            "replicates": REPLICATES,
            "attack_phase_calibration_exposures": attack_exposures,
            "benign_phase_calibration_exposures": benign_exposures,
            "balanced_exactly": True,
        }
    return audit


def validate(root: Path):
    required = [
        root / P0_2_MANIFEST_REL,
        root / P0_1C_SEEDS_REL,
        root / P0_1C_AUDIT_REL,
        root / DATASET_HASHES_REL,
        root / METHOD_CONFIG_REL,
        root / "src/lir_pptd/core/hidden_calibration.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise B2HiddenError("MISSING:" + ",".join(missing))

    p0_2_sha = _sha256(root / P0_2_MANIFEST_REL)
    if p0_2_sha != P0_2_MANIFEST_SHA256:
        raise B2HiddenError(f"P0_2_MANIFEST_SHA256:{p0_2_sha}:{P0_2_MANIFEST_SHA256}")

    datasets = load_datasets(root)
    streams, base_subsets = plan_streams(datasets)
    selectors, masks, mask_hashes, calibration_counts = load_hidden_masks(root, datasets)
    phase_audit = _onoff_calibration_phase_audit(datasets, masks)

    for ds in DATASETS:
        trusted_sets = {tuple(base.trusted_workers) for base in base_subsets[ds]}
        if len(trusted_sets) != 1:
            raise B2HiddenError(f"TQPP_TRUSTED_SET_NOT_FIXED:{ds}")
        trusted = next(iter(trusted_sets))
        if any(set(base.malicious_workers) & set(trusted) for base in base_subsets[ds]):
            raise B2HiddenError(f"TQPP_TRUSTED_SET_ATTACK_OVERLAP:{ds}")

    print("B2_HIDDEN_SELECTOR_VALIDATE=PASS")
    print("RHO=7/10")
    print("DATASETS=product,duck,dog,weather")
    print("ATTACKS=random,collusive,on_off")
    print("REPLICATES_PER_DATASET_ATTACK=20")
    print(f"STREAMS={len(streams)}")
    print(f"METHODS={len(METHODS)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print("P0_1C_HIDDEN_SELECTOR_MASK_REUSE=EXACT")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("SAME_CALIBRATION_MASK_ALL_METHODS_ATTACKS_REPLICATES=YES")
    print("RANDOM_ATTACK=TRUTH_INDEPENDENT_UNIFORM_DOMAIN")
    print("COLLUSIVE_ATTACK=FIXED_TARGET_CLASS0_OR_NUMERIC1")
    print("ON_OFF_SCHEDULE=5_BENIGN_5_COLLUSIVE")
    print("ON_OFF_PHASE_OFFSETS=0..9_EACH_TWICE_PER_DATASET")
    print("ON_OFF_HIDDEN_CALIBRATION_PHASE_BALANCE=EXACT_50_50")
    print("TQPP_TRUSTED_SET=DATASET_FIXED_PRE_ATTACK")
    print("TQPP_TRUSTED_SET_ATTACK_INDEPENDENT=YES")
    print("TQPP_TRUSTED_IDENTITIES_EXCLUDED_FROM_MALICIOUS_SUBSETS=YES")
    print("TQPP_RECEIVES_CALIBRATION_REFERENCE=NO")
    print("LIR_ORDINARY_PERSISTENT_REPUTATION_UPDATE=DISABLED")
    print("LIR_CALIBRATION_PERSISTENT_REPUTATION_UPDATE=ENABLED")
    print("WEATHER_PRIMARY=MAE")
    print("WEATHER_SECONDARY=RMSE")
    print("HOLM_PRIMARY_FAMILY_SIZE=36")
    print("ALGORITHM_RETUNING=NO")
    for ds in DATASETS:
        print(f"CALIBRATION_COUNT_{ds.upper()}={calibration_counts[ds]}")
        print(
            f"ONOFF_CAL_PHASE_{ds.upper()}="
            f"{phase_audit[ds]['attack_phase_calibration_exposures']}/"
            f"{phase_audit[ds]['benign_phase_calibration_exposures']}"
        )
    return datasets, streams, base_subsets, selectors, masks, mask_hashes, phase_audit


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
        base_subsets,
        selectors,
        masks,
        mask_hashes,
        phase_audit,
    ) = validate(root)

    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    config = _load_config(root / METHOD_CONFIG_REL)
    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / RAW_NAME
    if raw_path.exists() and not args.resume:
        raise B2HiddenError(f"OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    # Persist the deterministic B2 subset contract before execution.
    subset_manifest = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "rho": RHO,
        "seed_start": SEED_START,
        "tqpp_trusted_set_policy": "dataset-fixed before malicious-subset construction; trusted identities excluded from malicious subsets",
        "datasets": {
            ds: {
                "trusted_workers": list(base_subsets[ds][0].trusted_workers),
                "trusted_worker_count": len(base_subsets[ds][0].trusted_workers),
                "subsets": [
                    {
                        "replicate": base.replicate,
                        "seed": base.subset_seed,
                        "workers": list(base.malicious_workers),
                        "subset_sha256": base.subset_digest,
                    }
                    for base in base_subsets[ds]
                ],
            }
            for ds in DATASETS
        },
    }
    _atomic_json(outdir / SUBSET_MANIFEST_NAME, subset_manifest)

    existing = _read_existing(raw_path)

    for index, stream in enumerate(streams, start=1):
        dataset = datasets[stream.base.dataset_id]
        # Critical ordering: attack construction has no calibration input.
        tasks = attacked_tasks(dataset, stream)
        cal_ids = masks[stream.base.dataset_id]
        mask_hash = mask_hashes[stream.base.dataset_id]

        for method in METHODS:
            run_id = _run_id(stream, method, mask_hash)
            if run_id in existing and existing[run_id].get("status") == "success":
                continue

            started = time.perf_counter()
            try:
                result = _run_method(dataset, tasks, cal_ids, stream, method, config)
                if int(result["calibration_task_count"]) != len(cal_ids):
                    raise B2HiddenError(
                        f"CALIBRATION_COUNT_MISMATCH:{stream.base.dataset_id}:{stream.attack}:{method}"
                    )
                if method == "lir_pptd":
                    if int(result["ordinary_persistent_commit_count"]) != 0:
                        raise B2HiddenError("ORDINARY_PERSISTENCE_VIOLATION")
                    if int(result["calibration_persistent_commit_count"]) != len(cal_ids):
                        raise B2HiddenError("CALIBRATION_PERSISTENCE_COUNT_MISMATCH")

                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.base.dataset_id,
                    "rho": RHO,
                    "attack": stream.attack,
                    "phase_offset": phase_offset(stream.base.replicate)
                    if stream.attack == "on_off"
                    else None,
                    "replicate": stream.base.replicate,
                    "subset_seed": stream.base.subset_seed,
                    "malicious_workers": list(stream.base.malicious_workers),
                    "trusted_workers": list(stream.base.trusted_workers)
                    if method == "tqpp_liu"
                    else None,
                    "method": method,
                    "status": "success",
                    "calibration_mask_sha256": mask_hash,
                    "seed_commitment_sha256": selectors[
                        stream.base.dataset_id
                    ].seed_commitment_sha256,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
                    "same_mask_all_methods": True,
                    **result,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.base.dataset_id,
                    "rho": RHO,
                    "attack": stream.attack,
                    "replicate": stream.base.replicate,
                    "subset_seed": stream.base.subset_seed,
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
                f"METHOD_RUNS={len(existing)}/{len(streams)*len(METHODS)} "
                f"SUCCESS={success}",
                flush=True,
            )

    final = list(existing.values())
    _rewrite_canonical(raw_path, final)
    final = list(_read_existing(raw_path).values())
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)
    planned = len(streams) * len(METHODS)

    comparisons: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    secondary_means: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}

    if success == planned and failure == 0:
        comparisons, means, secondary_means, analysis = _analyze(final)
        _write_csv(outdir / COMPARISONS_NAME, comparisons)
        _write_csv(outdir / MEANS_NAME, means)
        _write_csv(outdir / SECONDARY_MEANS_NAME, secondary_means)

    audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256_for_selector_namespace": P0_2_MANIFEST_SHA256,
        "same_p0_1c_hidden_selector_masks": True,
        "attack_generation_before_mode_designation": True,
        "same_scoring_mask_all_methods": True,
        "attack_definitions": {
            "random": "truth-independent deterministic pseudo-random report over the admissible public domain",
            "collusive": "fixed target: class 0 for categorical tasks and 1.0 for normalized numerical tasks",
            "on_off": "5 benign tasks followed by 5 fixed-target collusive tasks, with replicate phase offsets 0..9 repeated exactly twice",
        },
        "on_off_hidden_calibration_phase_audit": phase_audit,
        "tqpp": {
            "trusted_set_policy": "fixed once per dataset before malicious-subset construction",
            "trusted_set_attack_independent": True,
            "trusted_identities_excluded_from_malicious_subsets": True,
            "receives_lir_calibration_reference": False,
            "trusted_fraction": TQPP_PROFILE.trusted_fraction,
            "top_h_fraction": TQPP_PROFILE.top_h_fraction,
            "reputation_mode": TQPP_PROFILE.reputation_mode,
        },
        "multiplicity": {
            "primary_family": "all 36 LIR-vs-baseline dataset×attack contrasts",
            "holm_family_size": 36,
        },
        "algorithm_retuning": False,
        "selector_reselection": False,
    }
    _atomic_json(outdir / AUDIT_NAME, audit)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "rho": RHO,
        "streams": len(streams),
        "planned_method_runs": planned,
        "completed_unique_method_runs": len(final),
        "success": success,
        "failure": failure,
        "datasets": list(DATASETS),
        "attacks": list(ATTACKS),
        "methods": list(METHODS),
        "replicates_per_dataset_attack": REPLICATES,
        "selector": {
            "reuse_p0_1c_masks": True,
            "nominal_period": PERIOD,
            "nominal_rate": 1 / PERIOD,
            "pre_freeze_mode_visible_to_attack": False,
            "seed_revealed": False,
        },
        "analysis": analysis,
        "scope": {
            "formal_b2_hidden_selector_migration": True,
            "algorithm_retuning": False,
            "selector_reselection": False,
            "production_vrf": False,
        },
    }
    _atomic_json(outdir / SUMMARY_NAME, summary)

    print("B2_HIDDEN_SELECTOR_MIGRATION=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"PLANNED_METHOD_RUNS={planned}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")
    if analysis:
        print(
            "B2_CRH_PPTD_IMPROVE_OR_TIE_CELLS="
            f"{analysis['cells_improve_or_tie_both_crh_and_pptd']}/12"
        )
        for baseline in BASELINES:
            x = analysis["per_comparator"][baseline]
            print(
                f"{baseline.upper()}_MEAN_WINS_TIES_LOSSES="
                f"{x['mean_wins']}/{x['mean_ties']}/{x['mean_losses']}"
            )
            print(
                f"{baseline.upper()}_HOLM_SIG_WINS_LOSSES="
                f"{x['holm_significant_wins']}/{x['holm_significant_losses']}"
            )
        for attack in ATTACKS:
            x = analysis["per_attack"][attack]
            print(
                f"ATTACK_{attack.upper()}_PAIRWISE_MEAN_WINS_TIES_LOSSES="
                f"{x['lir_mean_wins']}/{x['lir_mean_ties']}/{x['lir_mean_losses']}"
            )
        print("B2_SCIENTIFIC_DECISION=REVIEW_AFTER_RESULT")
    print(f"SUMMARY={outdir / SUMMARY_NAME}")
    return 0 if success == planned and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
