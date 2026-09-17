from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
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
    choose_initial_trusted_workers,
    initial_state as tqpp_initial_state,
    run_task as tqpp_run_task,
)
from lir_pptd.core.hidden_calibration import HiddenCalibrationSchedule
from lir_pptd.experiments.calibration_anchored import run_calibration_task
from lir_pptd.experiments.phase_r1.attacks import apply_response_corruption
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

STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"
DATASETS = ("product", "duck", "dog", "weather")
RHOS = ("1/10", "3/10", "1/2", "7/10", "9/10")
METHODS = ("lir_pptd", "crh", "pptd_liang", "tqpp_liu")
PERIOD = 20
P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
SELECTOR_DIR_REL = "configs/p0_1c_hidden_b1"
PRIVATE_SEEDS_NAME = "private_selector_seeds.json"
PUBLIC_COMMITMENTS_NAME = "selector_commitments.json"
OUT_REL = "results/p0_1c_hidden_b1_v1"
RAW_NAME = "formal_runs.jsonl"
SUMMARY_NAME = "formal_summary.json"
COMPARISONS_NAME = "paired_comparisons.csv"
MEANS_NAME = "figure_b1_hidden_metric_means.csv"
SELECTOR_AUDIT_NAME = "selector_audit.json"
TQPP_PROFILE = TQPPAdaptationProfile()
CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}


class P01CHiddenB1Error(RuntimeError):
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
                raise P01CHiddenB1Error(f"CELL_N:{ds}:{rho}:{len(rows)}:{n}")
            subsets = [tuple(sorted(map(str, x["workers"]))) for x in rows]
            if len(subsets) != len(set(subsets)):
                raise P01CHiddenB1Error(f"DUPLICATE_SUBSET:{ds}:{rho}")
            total += len(rows)
    if total != 378:
        raise P01CHiddenB1Error(f"STREAM_COUNT:{total}:378")
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
            rows = manifest["datasets"][ds][rho]["subsets"]
            for rep, row in enumerate(rows, start=1):
                workers = tuple(sorted(map(str, row["workers"])))
                if not set(workers) <= valid:
                    raise P01CHiddenB1Error(f"UNKNOWN_WORKER:{ds}:{rho}:{rep}")
                out.append(Stream(ds, rho, rep, int(row["seed"]), workers))
    if len(out) != 378:
        raise P01CHiddenB1Error(f"PLANNED_STREAMS:{len(out)}:378")
    return out


def _selector_config_hash(dataset: RealDataset, manifest_sha: str) -> str:
    return _sha_obj({
        "study": STUDY_ID,
        "dataset": dataset.dataset_id,
        "answer_sha256": dataset.answer_sha256,
        "truth_sha256": dataset.truth_sha256,
        "p0_2_manifest_sha256": manifest_sha,
        "algorithm_config": CONFIG,
        "nominal_period": PERIOD,
        "selector_rule": "HMAC-SHA256 over immutable task_id; Bernoulli probability 1/P",
    })


def _load_or_create_selector_seeds(path: Path, *, resume: bool) -> dict[str, str]:
    if path.is_file():
        obj = _load_json(path)
        seeds = {str(k): str(v) for k, v in obj.get("seeds", {}).items()}
        if set(seeds) != set(DATASETS):
            raise P01CHiddenB1Error("SELECTOR_SEED_DATASET_SET_MISMATCH")
        for ds, seed_hex in seeds.items():
            if len(bytes.fromhex(seed_hex)) < 32:
                raise P01CHiddenB1Error(f"SHORT_SELECTOR_SEED:{ds}")
        return seeds
    if resume:
        raise P01CHiddenB1Error(f"RESUME_REQUIRES_SELECTOR_SEEDS:{path}")
    seeds = {ds: secrets.token_bytes(32).hex() for ds in DATASETS}
    _atomic_json(path, {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "scope": "private until all hidden-calibration real-data experiments using this selector epoch are complete",
        "seeds": seeds,
    })
    return seeds


def _build_selectors(
    datasets: Mapping[str, RealDataset], seeds: Mapping[str, str], manifest_sha: str
) -> dict[str, HiddenCalibrationSchedule]:
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


def _selector_masks(
    datasets: Mapping[str, RealDataset], selectors: Mapping[str, HiddenCalibrationSchedule]
) -> tuple[dict[str, frozenset[str]], list[dict[str, Any]], dict[str, Any]]:
    masks: dict[str, frozenset[str]] = {}
    audit: list[dict[str, Any]] = []
    commitments: dict[str, Any] = {"schema_version": "1.0", "study_id": STUDY_ID, "datasets": {}}
    for ds in DATASETS:
        selector = selectors[ds]
        task_ids = [t.task_id for t in datasets[ds].tasks]
        cal_ids = frozenset(selector.selected_task_ids(task_ids))
        if not cal_ids:
            raise P01CHiddenB1Error(f"ZERO_CALIBRATION_TASKS:{ds}")
        masks[ds] = cal_ids
        mask_hash = _sha_obj(sorted(cal_ids))
        public = selector.public_commitment()
        commitments["datasets"][ds] = {
            "schema_version": public.schema_version,
            "session_id": public.session_id,
            "selector_epoch": public.selector_epoch,
            "config_hash": public.config_hash,
            "probability_numerator": public.probability_numerator,
            "probability_denominator": public.probability_denominator,
            "seed_commitment_sha256": public.seed_commitment_sha256,
            "calibration_mask_sha256": mask_hash,
        }
        audit.append({
            "dataset": ds,
            "task_count": len(task_ids),
            "calibration_task_count": len(cal_ids),
            "realized_calibration_rate": len(cal_ids) / len(task_ids),
            "nominal_calibration_rate": 1 / PERIOD,
            "calibration_mask_sha256": mask_hash,
            "seed_commitment_sha256": public.seed_commitment_sha256,
            "task_order_invariant": True,
            "attack_generation_before_mode_designation": True,
            "same_mask_all_methods_rhos_replicates": True,
            "seed_revealed": False,
        })
    return masks, audit, commitments


def _attacked_tasks(dataset: RealDataset, stream: Stream) -> tuple[RealTask, ...]:
    # Critical semantic ordering: adversarial reports are generated without any
    # calibration designation input.  The hidden selector mask is consumed only
    # after this attacked report sequence has been frozen in memory.
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


def _record_prediction(
    dataset: RealDataset, task: RealTask, vector: Sequence[float], cls: int | None,
    pc: list[int], tc: list[int], pv: list[float], tv: list[float]
) -> None:
    if dataset.modality == "categorical":
        if cls is None or task.truth_class is None:
            raise P01CHiddenB1Error("CATEGORICAL_PREDICTION_MISSING")
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
        metric_name = "MAE"
        paper_metric = primary_loss
        higher = False
    else:
        metric_name = "Accuracy" if dataset.dataset_id == "duck" else "MacroF1"
        paper_metric = 1.0 - primary_loss
        higher = True
    return {
        **losses,
        "paper_metric_name": metric_name,
        "paper_metric": paper_metric,
        "higher_is_better": higher,
        "evaluation_task_count": len(pc) if dataset.modality == "categorical" else len(pv),
    }


def _run_lir(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str]) -> dict[str, Any]:
    state = initial_global_state(dataset.worker_ids, CONFIG["c0"])
    pc: list[int] = []; tc: list[int] = []; pv: list[float] = []; tv: list[float] = []
    ordinary_commits = 0
    calibration_commits = 0
    for task in tasks:
        if task.task_id in cal_ids:
            _, state = run_calibration_task(task, CONFIG, state, dps=80)
            calibration_commits += 1
            continue
        pred = run_lir_task(task, CONFIG, state, ablation="full", dps=80)
        _record_prediction(dataset, task, pred.prediction_vector, pred.prediction_class, pc, tc, pv, tv)
        # Evidence-gated invariant: ordinary pred.next_state is never committed.
    return {
        **_collect(dataset, pc, tc, pv, tv),
        "ordinary_persistent_commit_count": ordinary_commits,
        "calibration_persistent_commit_count": calibration_commits,
        "calibration_task_count": len(cal_ids),
    }


def _run_crh(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str]) -> dict[str, Any]:
    pc: list[int] = []; tc: list[int] = []; pv: list[float] = []; tv: list[float] = []
    for task in tasks:
        if task.task_id in cal_ids:
            continue
        vector, cls = run_stateless_baseline(task, "crh")
        _record_prediction(dataset, task, vector, cls, pc, tc, pv, tv)
    return {**_collect(dataset, pc, tc, pv, tv), "calibration_task_count": len(cal_ids)}


def _run_pptd(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str]) -> dict[str, Any]:
    pc: list[int] = []; tc: list[int] = []; pv: list[float] = []; tv: list[float] = []
    iters: list[int] = []
    for task in tasks:
        if task.task_id in cal_ids:
            continue
        result = pptd_predict(task.reports, task.modality, weight_mode="source_quadratic")
        iters.append(int(result.iterations))
        _record_prediction(dataset, task, result.truth_vector, result.prediction_class, pc, tc, pv, tv)
    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "mean_iterations": statistics.fmean(iters) if iters else 0.0,
    }


def _coverage(dataset: RealDataset) -> dict[str, int]:
    out = {w: 0 for w in dataset.worker_ids}
    for task in dataset.tasks:
        for w in task.participant_ids:
            out[w] += 1
    return out


def _run_tqpp(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str], stream: Stream) -> dict[str, Any]:
    malicious = set(stream.malicious_workers)
    honest = [w for w in dataset.worker_ids if w not in malicious]
    trusted = choose_initial_trusted_workers(dataset.worker_ids, honest, _coverage(dataset), TQPP_PROFILE)
    state = tqpp_initial_state(dataset.worker_ids, trusted)
    top_h_global = TQPP_PROFILE.top_h_count(len(dataset.worker_ids))
    pc: list[int] = []; tc: list[int] = []; pv: list[float] = []; tv: list[float] = []
    for task_index, task in enumerate(tasks, start=1):
        result = tqpp_run_task(
            task.reports,
            task.modality,
            state,
            top_h_global=top_h_global,
            seed_namespace=f"{stream.pairing_id}|round={task_index}",
            profile=TQPP_PROFILE,
        )
        state = result.next_state
        if task.task_id in cal_ids:
            # TQPP does not receive LIR calibration truth. It processes the task
            # natively for its own longitudinal state, while the task is excluded
            # from the shared scored set for all methods.
            continue
        _record_prediction(dataset, task, result.truth_vector, result.prediction_class, pc, tc, pv, tv)
    return {
        **_collect(dataset, pc, tc, pv, tv),
        "calibration_task_count": len(cal_ids),
        "trusted_worker_count": len(trusted),
        "top_h_global": top_h_global,
        "trusted_workers": list(trusted),
        "trusted_pool_excludes_realized_malicious_workers": True,
    }


def _run_method(dataset: RealDataset, tasks: Sequence[RealTask], cal_ids: frozenset[str], stream: Stream, method: str) -> dict[str, Any]:
    if method == "lir_pptd":
        return _run_lir(dataset, tasks, cal_ids)
    if method == "crh":
        return _run_crh(dataset, tasks, cal_ids)
    if method == "pptd_liang":
        return _run_pptd(dataset, tasks, cal_ids)
    if method == "tqpp_liu":
        return _run_tqpp(dataset, tasks, cal_ids, stream)
    raise P01CHiddenB1Error(f"UNKNOWN_METHOD:{method}")


def _run_id(stream: Stream, method: str, manifest_sha: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{STUDY_ID}|{manifest_sha}|{mask_hash}|{stream.pairing_id}|{method}".encode("utf-8")
    ).hexdigest()


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row["run_id"])
        # Last attempt wins, which makes retry/resume failure-safe.
        out[rid] = row
    return out


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    order_method = {m: i for i, m in enumerate(METHODS)}
    order_ds = {d: i for i, d in enumerate(DATASETS)}
    order_rho = {r: i for i, r in enumerate(RHOS)}
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            order_ds[str(r["dataset"])],
            order_rho[str(r["rho"])],
            int(r["replicate"]),
            order_method[str(r["method"])],
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _frac(value: Any) -> Fraction:
    return Fraction(Decimal(str(value)))


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
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _analyze(rows: Sequence[Mapping[str, Any]], manifest_sha: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    success = [dict(r) for r in rows if r.get("status") == "success"]
    by = {(r["dataset"], r["rho"], int(r["replicate"]), r["method"]): r for r in success}
    comparisons: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    for ds in DATASETS:
        for rho in RHOS:
            n = expected_n(ds, rho)
            reps = list(range(1, n + 1))
            candidate = [by[(ds, rho, rep, "lir_pptd")] for rep in reps]
            mean_row: dict[str, Any] = {"dataset": ds, "rho": rho, "n": n}
            for method in METHODS:
                vals = [float(by[(ds, rho, rep, method)]["paper_metric"]) for rep in reps]
                mean_row[method] = statistics.fmean(vals)
            means.append(mean_row)
            for comparator in ("crh", "pptd_liang", "tqpp_liu"):
                deltas = [
                    _frac(by[(ds, rho, rep, "lir_pptd")]["primary_loss"])
                    - _frac(by[(ds, rho, rep, comparator)]["primary_loss"])
                    for rep in reps
                ]
                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                ci_low, ci_high, boot_seed = bootstrap_mean_ci(
                    deltas,
                    seed_text=f"{STUDY_ID}|{manifest_sha}|{ds}|{rho}|{comparator}",
                )
                p = exact_two_sided_sign_flip(deltas)
                rb = paired_rank_biserial(deltas)
                comparisons.append({
                    "dataset": ds,
                    "rho": rho,
                    "n": n,
                    "comparator": comparator,
                    "candidate_mean_loss": float(statistics.fmean(float(by[(ds, rho, rep, "lir_pptd")]["primary_loss"]) for rep in reps)),
                    "comparator_mean_loss": float(statistics.fmean(float(by[(ds, rho, rep, comparator)]["primary_loss"]) for rep in reps)),
                    "mean_loss_delta_lir_minus_comparator": float(mean_delta),
                    "wins": sum(x < 0 for x in deltas),
                    "ties": sum(x == 0 for x in deltas),
                    "losses": sum(x > 0 for x in deltas),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "bootstrap_seed": boot_seed,
                    "exact_two_sided_sign_flip_p": float(p),
                    "paired_rank_biserial": float(rb),
                    "_exact_p_fraction": p,
                })

    # One explicit Holm family per comparator over the 20 nonzero B1 cells.
    for comparator in ("crh", "pptd_liang", "tqpp_liu"):
        family = [r for r in comparisons if r["comparator"] == comparator]
        raw = {
            f"{r['dataset']}|{r['rho']}|{comparator}": r["_exact_p_fraction"]
            for r in family
        }
        adjusted = holm_adjust(raw)
        for r in family:
            key = f"{r['dataset']}|{r['rho']}|{comparator}"
            r["holm_family"] = f"B1|{comparator}|20-cells"
            r["holm_adjusted_p"] = float(adjusted[key])
            r["holm_significant"] = adjusted[key] <= Fraction(1, 20)
            delta = float(r["mean_loss_delta_lir_minus_comparator"])
            r["direction"] = "lir_win" if delta < 0 else ("lir_loss" if delta > 0 else "tie")
            r["holm_direction"] = r["direction"] if r["holm_significant"] else "not_significant"
            r.pop("_exact_p_fraction", None)

    cell_both_mean_wins = 0
    for ds in DATASETS:
        for rho in RHOS:
            crh = next(r for r in comparisons if r["dataset"] == ds and r["rho"] == rho and r["comparator"] == "crh")
            pptd = next(r for r in comparisons if r["dataset"] == ds and r["rho"] == rho and r["comparator"] == "pptd_liang")
            if crh["direction"] == "lir_win" and pptd["direction"] == "lir_win":
                cell_both_mean_wins += 1

    per_comp = {}
    for comparator in ("crh", "pptd_liang", "tqpp_liu"):
        family = [r for r in comparisons if r["comparator"] == comparator]
        per_comp[comparator] = {
            "mean_wins": sum(r["direction"] == "lir_win" for r in family),
            "mean_ties": sum(r["direction"] == "tie" for r in family),
            "mean_losses": sum(r["direction"] == "lir_loss" for r in family),
            "holm_significant_wins": sum(r["holm_direction"] == "lir_win" for r in family),
            "holm_significant_losses": sum(r["holm_direction"] == "lir_loss" for r in family),
        }
    core_pass = (
        cell_both_mean_wins >= 18
        and per_comp["crh"]["holm_significant_losses"] == 0
        and per_comp["pptd_liang"]["holm_significant_losses"] == 0
    )
    summary = {
        "b1_cell_count": 20,
        "cell_mean_wins_against_both_crh_and_pptd": cell_both_mean_wins,
        "per_comparator": per_comp,
        "core_robustness_gate": "PASS" if core_pass else "REVIEW_REQUIRED",
        "core_gate_rule": "at least 18/20 cell mean wins against both CRH and PPTD, with zero Holm-significant losses against either; TQPP is reported separately and is not a gate because it retains a different trusted-information model",
    }
    return comparisons, means, summary


def validate(root: Path) -> tuple[dict[str, RealDataset], dict[str, Any], str, list[Stream]]:
    required = [
        root / P0_2_MANIFEST_REL,
        root / DATASET_HASHES_REL,
        root / "src/lir_pptd/core/hidden_calibration.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise P01CHiddenB1Error("MISSING:" + ",".join(missing))
    manifest_path = root / P0_2_MANIFEST_REL
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise P01CHiddenB1Error(f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}")
    manifest = _load_json(manifest_path)
    total = _validate_manifest_shape(manifest)
    datasets = _load_datasets(root)
    streams = _plan_streams(manifest, datasets)
    print("P0_1C_HIDDEN_B1_VALIDATE=PASS")
    print(f"P0_2_MANIFEST_SHA256={manifest_sha}")
    print(f"STREAMS={total}")
    print("STANDARD_CELLS=18x20")
    print("WEATHER_EXTREME_CELLS=2x9")
    print(f"METHODS={len(METHODS)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print("NOMINAL_CALIBRATION_PERIOD=20")
    print("NOMINAL_CALIBRATION_RATE=0.05")
    print("SELECTOR=HMAC_SHA256_TASK_ID_BERNOULLI")
    print("SELECTOR_SEED_SCOPE=ONE_PRIVATE_SEED_PER_DATASET")
    print("SAME_CALIBRATION_MASK_ACROSS_METHODS_RHOS_REPLICATES=YES")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("ORDINARY_PERSISTENT_REPUTATION_UPDATE=DISABLED")
    print("CALIBRATION_PERSISTENT_REPUTATION_UPDATE=ENABLED")
    print("TQPP_RECEIVES_CALIBRATION_REFERENCE=NO")
    print("FORMAL_B1_MIGRATION=YES")
    return datasets, manifest, manifest_sha, streams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    datasets, manifest, manifest_sha, streams = validate(root)
    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    selector_dir = root / SELECTOR_DIR_REL
    private_seed_path = selector_dir / PRIVATE_SEEDS_NAME
    seeds = _load_or_create_selector_seeds(private_seed_path, resume=args.resume)
    selectors = _build_selectors(datasets, seeds, manifest_sha)
    masks, selector_audit, commitments = _selector_masks(datasets, selectors)
    _atomic_json(selector_dir / PUBLIC_COMMITMENTS_NAME, commitments)

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / RAW_NAME
    summary_path = outdir / SUMMARY_NAME
    if raw_path.exists() and not args.resume:
        raise P01CHiddenB1Error(f"FORMAL_OUTPUT_EXISTS_USE_RESUME:{raw_path}")
    existing = _read_existing(raw_path)

    # Cache fixed per-dataset mask hashes for all streams/methods.
    mask_hashes = {ds: _sha_obj(sorted(masks[ds])) for ds in DATASETS}
    success_new = 0; failure_new = 0
    for idx, stream in enumerate(streams, start=1):
        dataset = datasets[stream.dataset_id]
        # Reports are corrupted before the hidden mode mask is supplied to any
        # method. The attack function has no calibration argument.
        attacked = _attacked_tasks(dataset, stream)
        cal_ids = masks[stream.dataset_id]
        mask_hash = mask_hashes[stream.dataset_id]
        for method in METHODS:
            rid = _run_id(stream, method, manifest_sha, mask_hash)
            if rid in existing and existing[rid].get("status") == "success":
                continue
            t0 = time.perf_counter()
            try:
                result = _run_method(dataset, attacked, cal_ids, stream, method)
                if int(result["calibration_task_count"]) != len(cal_ids):
                    raise P01CHiddenB1Error(f"CALIBRATION_COUNT_MISMATCH:{stream.dataset_id}:{method}")
                if method == "lir_pptd":
                    if int(result["ordinary_persistent_commit_count"]) != 0:
                        raise P01CHiddenB1Error("ORDINARY_PERSISTENCE_VIOLATION")
                    if int(result["calibration_persistent_commit_count"]) != len(cal_ids):
                        raise P01CHiddenB1Error("CALIBRATION_COMMIT_COUNT_MISMATCH")
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
                    "method": method,
                    "status": "success",
                    "p0_2_manifest_sha256": manifest_sha,
                    "calibration_mask_sha256": mask_hash,
                    "seed_commitment_sha256": selectors[stream.dataset_id].seed_commitment_sha256,
                    "pre_freeze_mode_visible_to_attack": False,
                    "attack_generation_before_mode_designation": True,
                    "same_mask_all_methods": True,
                    **result,
                    "elapsed_seconds": time.perf_counter() - t0,
                }
                success_new += 1
            except Exception as exc:
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
                    "method": method,
                    "status": "failure",
                    "p0_2_manifest_sha256": manifest_sha,
                    "calibration_mask_sha256": mask_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - t0,
                }
                failure_new += 1
            _append(raw_path, row)
            existing[rid] = row
        if idx % 10 == 0 or idx == len(streams):
            completed = sum(r.get("status") == "success" for r in existing.values())
            print(f"PROGRESS={idx}/{len(streams)} SUCCESS_METHOD_RUNS={completed}/{len(streams)*len(METHODS)}", flush=True)

    final_rows = list(existing.values())
    _rewrite_canonical(raw_path, final_rows)
    final_rows = list(_read_existing(raw_path).values())
    success = sum(r.get("status") == "success" for r in final_rows)
    failure = sum(r.get("status") != "success" for r in final_rows)

    comparisons: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    analysis_summary: dict[str, Any] | None = None
    if success == len(streams) * len(METHODS) and failure == 0:
        comparisons, means, analysis_summary = _analyze(final_rows, manifest_sha)
        _write_csv(outdir / COMPARISONS_NAME, comparisons)
        _write_csv(outdir / MEANS_NAME, means)
    _atomic_json(outdir / SELECTOR_AUDIT_NAME, {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "note": "Seeds remain private for reuse by later formal real-data migrations; only commitments and mask hashes are published at this stage.",
        "selectors": selector_audit,
    })

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256": manifest_sha,
        "stream_count": len(streams),
        "method_count": len(METHODS),
        "planned_method_runs": len(streams) * len(METHODS),
        "completed_unique_method_runs": len(final_rows),
        "success": success,
        "failure": failure,
        "methods": list(METHODS),
        "selector": {
            "nominal_period": PERIOD,
            "nominal_rate": 1 / PERIOD,
            "one_private_seed_per_dataset": True,
            "same_mask_all_methods_rhos_replicates": True,
            "pre_freeze_mode_visible_to_attack": False,
            "seed_revealed": False,
        },
        "semantics": {
            "attack_generation_before_mode_designation": True,
            "ordinary_persistent_reputation_update": False,
            "calibration_persistent_reputation_update": True,
            "tqpp_receives_calibration_reference": False,
        },
        "analysis": analysis_summary,
        "scope": {
            "formal_b1_hidden_selector_migration": True,
            "production_vrf": False,
            "production_mpc": False,
        },
    }
    _atomic_json(summary_path, summary)
    print("P0_1C_HIDDEN_B1_FORMAL_RUN=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")
    for item in selector_audit:
        print(f"CALIBRATION={item['dataset']}:{item['calibration_task_count']}/{item['task_count']}:{item['realized_calibration_rate']:.8f}")
    if analysis_summary is not None:
        print(f"B1_BOTH_CRH_PPTD_MEAN_WINS={analysis_summary['cell_mean_wins_against_both_crh_and_pptd']}/20")
        print(f"B1_CORE_ROBUSTNESS_GATE={analysis_summary['core_robustness_gate']}")
        for comp in ("crh", "pptd_liang", "tqpp_liu"):
            x = analysis_summary["per_comparator"][comp]
            print(f"{comp.upper()}_MEAN_WINS_TIES_LOSSES={x['mean_wins']}/{x['mean_ties']}/{x['mean_losses']}")
            print(f"{comp.upper()}_HOLM_SIG_WINS_LOSSES={x['holm_significant_wins']}/{x['holm_significant_losses']}")
    print(f"SUMMARY={summary_path.relative_to(root)}")
    print(f"FORMAL_RUNNER_MIGRATION_COMPLETE={'YES' if success == len(streams)*len(METHODS) and failure == 0 else 'NO'}")
    return 0 if success == len(streams)*len(METHODS) and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
