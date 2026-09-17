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

STUDY_ID = "cowa_dynamic_formal_stress_v1"
P0_1C_SELECTOR_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

DATASETS = ("product", "duck", "dog", "weather")
SCENARIOS = ("benign_to_malicious", "malicious_to_benign")
RHO = "7/10"

# Probe used replicates 1-5. Formal inference is intentionally held out.
FORMAL_REPLICATES = tuple(range(6, 21))
METHODS = ("lir_pptd", "cowa", "no_cw", "no_hr")
PERIOD = 20
EARLY_POST_FRACTION = Fraction(1, 4)

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
OUT_REL = "results/cowa_dynamic_formal_stress_v1"

CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}

PRIMARY_CONTRASTS = (
    ("lir_pptd", "cowa", "full_vs_cowa"),
    ("no_cw", "cowa", "no_cw_vs_cowa"),
    ("lir_pptd", "no_cw", "full_vs_no_cw"),
    ("lir_pptd", "no_hr", "full_vs_no_hr"),
)

FORMAL_GATE = {
    "full_vs_cowa_min_mean_win_cells": 6,
    "full_vs_cowa_min_holm_win_cells": 4,
    "full_vs_cowa_max_holm_loss_cells": 1,
}


class DynamicFormalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stream:
    dataset_id: str
    scenario: str
    replicate: int
    subset_seed: int
    switched_workers: tuple[str, ...]

    @property
    def pairing_id(self) -> str:
        digest = hashlib.sha256("\n".join(self.switched_workers).encode("utf-8")).hexdigest()
        return (
            f"dataset={self.dataset_id}|rho={RHO}|scenario={self.scenario}|"
            f"rep={self.replicate}|subset={digest}"
        )


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
        n = self.counts.get(w, 0)
        return self.c0 if n <= 0 else self.sums[w] / n

    def observe(self, worker_id: str, q_cal: Any) -> None:
        w = str(worker_id)
        self.sums.setdefault(w, Fraction(0, 1))
        self.counts.setdefault(w, 0)
        self.sums[w] += _fraction(q_cal)
        self.counts[w] += 1


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    text = str(value)
    if "/" in text:
        return Fraction(text)
    return Fraction(Decimal(text))


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _rewrite_canonical(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ds_ord = {d: i for i, d in enumerate(DATASETS)}
    sc_ord = {s: i for i, s in enumerate(SCENARIOS)}
    m_ord = {m: i for i, m in enumerate(METHODS)}
    ordered = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            ds_ord[str(r["dataset"])],
            sc_ord[str(r["scenario"])],
            int(r["replicate"]),
            m_ord[str(r["method"])],
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


def _selector_config_hash(dataset: RealDataset, manifest_sha: str) -> str:
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


def _load_selectors(
    root: Path,
    datasets: Mapping[str, RealDataset],
    manifest_sha: str,
) -> dict[str, HiddenCalibrationSchedule]:
    seeds_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seeds_obj.get("seeds", {}).items()}
    if set(seeds) != set(DATASETS):
        raise DynamicFormalError("P0_1C_SELECTOR_SEED_DATASET_SET_MISMATCH")
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
    audit: Mapping[str, Any],
) -> tuple[dict[str, frozenset[str]], dict[str, str], dict[str, int]]:
    expected = {str(x["dataset"]): dict(x) for x in audit["selectors"]}
    masks: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for ds in DATASETS:
        task_ids = [t.task_id for t in datasets[ds].tasks]
        selected = frozenset(selectors[ds].selected_task_ids(task_ids))
        mask_hash = _sha_obj(sorted(selected))
        exp = expected[ds]
        if mask_hash != str(exp["calibration_mask_sha256"]):
            raise DynamicFormalError(f"MASK_HASH_MISMATCH:{ds}")
        if len(selected) != int(exp["calibration_task_count"]):
            raise DynamicFormalError(f"MASK_COUNT_MISMATCH:{ds}")
        if selectors[ds].seed_commitment_sha256 != str(exp["seed_commitment_sha256"]):
            raise DynamicFormalError(f"SEED_COMMITMENT_MISMATCH:{ds}")
        masks[ds] = selected
        hashes[ds] = mask_hash
        counts[ds] = len(selected)
    return masks, hashes, counts


def _plan_streams(manifest: Mapping[str, Any], datasets: Mapping[str, RealDataset]) -> list[Stream]:
    out: list[Stream] = []
    for ds in DATASETS:
        rows = manifest["datasets"][ds][RHO]["subsets"]
        if len(rows) < max(FORMAL_REPLICATES):
            raise DynamicFormalError(f"INSUFFICIENT_SUBSETS:{ds}:{len(rows)}")
        valid_workers = set(datasets[ds].worker_ids)
        for scenario in SCENARIOS:
            for replicate in FORMAL_REPLICATES:
                row = rows[replicate - 1]
                workers = tuple(sorted(map(str, row["workers"])))
                if not set(workers) <= valid_workers:
                    raise DynamicFormalError(f"UNKNOWN_WORKER:{ds}:{replicate}")
                out.append(
                    Stream(
                        dataset_id=ds,
                        scenario=scenario,
                        replicate=replicate,
                        subset_seed=int(row["seed"]),
                        switched_workers=workers,
                    )
                )
    expected = len(DATASETS) * len(SCENARIOS) * len(FORMAL_REPLICATES)
    if len(out) != expected:
        raise DynamicFormalError("STREAM_COUNT_MISMATCH")
    return out


def active_for_scenario(scenario: str, task_index_1based: int, switch_after_index: int) -> bool:
    if scenario == "benign_to_malicious":
        return task_index_1based > switch_after_index
    if scenario == "malicious_to_benign":
        return task_index_1based <= switch_after_index
    raise DynamicFormalError(f"UNKNOWN_SCENARIO:{scenario}")


def _dynamic_tasks(dataset: RealDataset, stream: Stream) -> tuple[tuple[RealTask, ...], int]:
    # No calibration information is passed here.  The behavior switch is based
    # only on public chronology and the predeclared midpoint.
    switch_after = len(dataset.tasks) // 2
    tasks: list[RealTask] = []
    for index, task in enumerate(dataset.tasks, start=1):
        if active_for_scenario(stream.scenario, index, switch_after):
            reports = apply_response_corruption(
                task.reports,
                stream.switched_workers,
                modality=task.modality,
                active=True,
            )
        else:
            reports = {w: tuple(v) for w, v in task.reports.items()}
        tasks.append(replace(task, reports=reports))
    return tuple(tasks), switch_after


def _class_from_vector(values: Sequence[float]) -> int:
    maximum = max(values)
    return min(index for index, value in enumerate(values) if value == maximum)


def _cowa_predict(task: RealTask, history: COWAHistory) -> tuple[tuple[float, ...], int | None]:
    dimension = len(next(iter(task.reports.values())))
    numerator = [Fraction(0, 1) for _ in range(dimension)]
    denominator = Fraction(0, 1)
    for worker in task.participant_ids:
        weight = history.weight(worker)
        denominator += weight
        report = task.reports[worker]
        for h in range(dimension):
            numerator[h] += weight * _fraction(report[h])
    if denominator <= 0:
        raise DynamicFormalError("NONPOSITIVE_COWA_DENOMINATOR")
    vector = tuple(float(x / denominator) for x in numerator)
    cls = _class_from_vector(vector) if task.modality == "categorical" else None
    return vector, cls


def _group_mean(weights: Mapping[str, Any], group: Sequence[str]) -> float:
    values = [float(_fraction(weights[w])) for w in group if w in weights]
    if not values:
        raise DynamicFormalError("EMPTY_GROUP_WEIGHT")
    return statistics.fmean(values)


def _state_weights(state: Any, workers: Sequence[str]) -> dict[str, Any]:
    reps = dict(state.reputations or {})
    return {str(w): reps[str(w)] for w in workers}


def _cowa_weights(history: COWAHistory, workers: Sequence[str]) -> dict[str, Any]:
    return {str(w): history.weight(str(w)) for w in workers}


def alignment_score(scenario: str, honest_mean: float, switched_mean: float) -> float:
    gap = honest_mean - switched_mean
    if scenario == "benign_to_malicious":
        # Higher is better: newly malicious workers should be pushed below stable honest workers.
        return gap
    if scenario == "malicious_to_benign":
        # Higher is better: a rehabilitated cohort should converge back toward stable honest workers.
        return -abs(gap)
    raise DynamicFormalError(f"UNKNOWN_SCENARIO:{scenario}")


def _metric_from_records(
    dataset: RealDataset,
    predicted_classes: list[int],
    truth_classes: list[int],
    predicted_values: list[float],
    truth_values: list[float],
) -> dict[str, Any]:
    losses = dataset_losses(
        dataset.dataset_id,
        predicted_classes=predicted_classes if dataset.modality == "categorical" else None,
        truth_classes=truth_classes if dataset.modality == "categorical" else None,
        predicted_values=predicted_values if dataset.modality == "numerical" else None,
        truth_values=truth_values if dataset.modality == "numerical" else None,
    )
    primary_loss = float(losses["primary_loss"])
    if dataset.dataset_id == "weather":
        return {"primary_loss": primary_loss, "paper_metric_name": "MAE", "paper_metric": primary_loss}
    metric_name = "Accuracy" if dataset.dataset_id == "duck" else "MacroF1"
    return {
        "primary_loss": primary_loss,
        "paper_metric_name": metric_name,
        "paper_metric": 1.0 - primary_loss,
    }


def _point_loss(dataset: RealDataset, task: RealTask, vector: Sequence[float], cls: int | None) -> float:
    if dataset.modality == "categorical":
        if cls is None or task.truth_class is None:
            raise DynamicFormalError("MISSING_CLASS")
        return 0.0 if int(cls) == int(task.truth_class) else 1.0
    return abs(float(vector[0]) - float(task.truth_vector[0]))


def _ordinary_predict(
    method: str,
    task: RealTask,
    full_state: Any,
    no_cw_state: Any,
    no_hr_state: Any,
    cowa_history: COWAHistory,
) -> tuple[tuple[float, ...], int | None]:
    if method == "cowa":
        return _cowa_predict(task, cowa_history)
    if method == "lir_pptd":
        prediction = run_lir_task(task, CONFIG, full_state, ablation="full", dps=80)
        return prediction.prediction_vector, prediction.prediction_class
    if method == "no_cw":
        # "ho" is the source implementation of historical-only ordinary aggregation.
        prediction = run_lir_task(task, CONFIG, no_cw_state, ablation="ho", dps=80)
        return prediction.prediction_vector, prediction.prediction_class
    if method == "no_hr":
        # "nr" uses neutral c0 and does not carry historical state across tasks.
        prediction = run_lir_task(task, CONFIG, no_hr_state, ablation="nr", dps=80)
        return prediction.prediction_vector, prediction.prediction_class
    raise DynamicFormalError(f"UNKNOWN_METHOD:{method}")


def _run_method(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    calibration_ids: frozenset[str],
    stream: Stream,
    switch_after: int,
    method: str,
) -> dict[str, Any]:
    switched = tuple(stream.switched_workers)
    switched_set = set(switched)
    stable_honest = tuple(w for w in dataset.worker_ids if w not in switched_set)
    if not stable_honest:
        raise DynamicFormalError("EMPTY_STABLE_HONEST_GROUP")

    base_state = initial_global_state(dataset.worker_ids, CONFIG["c0"])
    full_state = base_state
    no_cw_state = base_state
    no_hr_state = base_state
    cowa_history = COWAHistory.initialize(dataset.worker_ids, CONFIG["c0"])
    cowa_q_state = base_state

    post_pred_classes: list[int] = []
    post_truth_classes: list[int] = []
    post_pred_values: list[float] = []
    post_truth_values: list[float] = []
    post_point_losses: list[float] = []
    early_point_losses: list[float] = []
    post_alignment_scores: list[float] = []
    post_calibration_events = 0

    post_eval_total = sum(
        1
        for index, task in enumerate(tasks, start=1)
        if index > switch_after and task.task_id not in calibration_ids
    )
    early_limit = max(1, math.ceil(post_eval_total * float(EARLY_POST_FRACTION)))

    for index, task in enumerate(tasks, start=1):
        if task.task_id in calibration_ids:
            if method == "lir_pptd":
                _, full_state = run_calibration_task(task, CONFIG, full_state, dps=80)
                weights = _state_weights(full_state, dataset.worker_ids)
            elif method == "no_cw":
                _, no_cw_state = run_calibration_task(task, CONFIG, no_cw_state, dps=80)
                weights = _state_weights(no_cw_state, dataset.worker_ids)
            elif method == "no_hr":
                # Final-paper No-HR: calibration does not modify persistent state.
                weights = {str(w): _fraction(CONFIG["c0"]) for w in dataset.worker_ids}
            elif method == "cowa":
                result, _ = run_calibration_task(task, CONFIG, cowa_q_state, dps=80)
                for evidence in result.evidence:
                    cowa_history.observe(str(evidence.worker_id), evidence.evidence)
                weights = _cowa_weights(cowa_history, dataset.worker_ids)
            else:
                raise DynamicFormalError(f"UNKNOWN_METHOD:{method}")

            if index > switch_after:
                post_calibration_events += 1
                score = alignment_score(
                    stream.scenario,
                    _group_mean(weights, stable_honest),
                    _group_mean(weights, switched),
                )
                post_alignment_scores.append(score)
            continue

        vector, cls = _ordinary_predict(
            method, task, full_state, no_cw_state, no_hr_state, cowa_history
        )

        if index <= switch_after:
            continue

        if dataset.modality == "categorical":
            if cls is None or task.truth_class is None:
                raise DynamicFormalError("MISSING_POST_CLASS")
            post_pred_classes.append(int(cls))
            post_truth_classes.append(int(task.truth_class))
        else:
            post_pred_values.append(float(vector[0]))
            post_truth_values.append(float(task.truth_vector[0]))

        loss = _point_loss(dataset, task, vector, cls)
        post_point_losses.append(loss)
        if len(early_point_losses) < early_limit:
            early_point_losses.append(loss)

    metrics = _metric_from_records(
        dataset,
        post_pred_classes,
        post_truth_classes,
        post_pred_values,
        post_truth_values,
    )

    if method == "lir_pptd":
        final_weights = _state_weights(full_state, dataset.worker_ids)
    elif method == "no_cw":
        final_weights = _state_weights(no_cw_state, dataset.worker_ids)
    elif method == "no_hr":
        final_weights = {str(w): _fraction(CONFIG["c0"]) for w in dataset.worker_ids}
    else:
        final_weights = _cowa_weights(cowa_history, dataset.worker_ids)

    final_alignment = alignment_score(
        stream.scenario,
        _group_mean(final_weights, stable_honest),
        _group_mean(final_weights, switched),
    )

    return {
        **metrics,
        "post_switch_evaluation_task_count": len(post_point_losses),
        "post_switch_point_loss_mean": statistics.fmean(post_point_losses),
        "early_post_fraction": "1/4",
        "early_post_task_count": len(early_point_losses),
        "early_post_point_loss_mean": statistics.fmean(early_point_losses),
        "post_switch_calibration_events": post_calibration_events,
        "final_behavior_alignment_score_higher_is_better": final_alignment,
        "mean_post_calibration_alignment_score_higher_is_better": (
            statistics.fmean(post_alignment_scores) if post_alignment_scores else None
        ),
    }


def _run_id(stream: Stream, method: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{STUDY_ID}|{mask_hash}|{stream.pairing_id}|{method}".encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[Fraction]) -> float:
    return float(sum(values, Fraction(0, 1)) / len(values))


def _comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successes = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["scenario"]), int(r["replicate"]), str(r["method"])): r
        for r in successes
    }
    output: list[dict[str, Any]] = []

    endpoints = (
        ("primary_loss", "post_switch_primary_loss", "lower"),
        ("early_post_point_loss_mean", "early_post_point_loss", "lower"),
        (
            "mean_post_calibration_alignment_score_higher_is_better",
            "post_calibration_alignment",
            "higher",
        ),
    )

    for full_method, comparator, contrast_name in PRIMARY_CONTRASTS:
        for endpoint_field, endpoint_name, favorable in endpoints:
            hypotheses: dict[str, Fraction] = {}
            provisional: list[dict[str, Any]] = []
            for ds in DATASETS:
                for scenario in SCENARIOS:
                    deltas: list[Fraction] = []
                    for rep in FORMAL_REPLICATES:
                        a = by[(ds, scenario, rep, full_method)]
                        b = by[(ds, scenario, rep, comparator)]
                        av = a.get(endpoint_field)
                        bv = b.get(endpoint_field)
                        if av is None or bv is None:
                            continue
                        # Always define delta as first method minus comparator.
                        deltas.append(_fraction(av) - _fraction(bv))
                    hypothesis_id = f"{contrast_name}|{endpoint_name}|{ds}|{scenario}"

                    # Alignment is structurally undefined if the fixed hidden
                    # selector has no calibration event after the switch.
                    # This is not a failed run and must not trigger selector
                    # reselection or imputation.
                    if len(deltas) == 0 and endpoint_name == "post_calibration_alignment":
                        provisional.append({
                            "hypothesis_id": hypothesis_id,
                            "contrast": contrast_name,
                            "first_method": full_method,
                            "comparator": comparator,
                            "endpoint": endpoint_name,
                            "favorable_direction_for_first_method": favorable,
                            "dataset": ds,
                            "scenario": scenario,
                            "endpoint_status": "unavailable_no_post_switch_calibration",
                            "n": 0,
                            "first_method_pair_wins": 0,
                            "ties": 0,
                            "comparator_pair_wins": 0,
                            "mean_delta_first_minus_comparator": None,
                            "bootstrap_95_ci_low": None,
                            "bootstrap_95_ci_high": None,
                            "bootstrap_seed": None,
                            "exact_two_sided_sign_flip_p": None,
                            "paired_rank_biserial_first_minus_comparator": None,
                            "mean_direction": "not_tested",
                        })
                        continue

                    if len(deltas) != len(FORMAL_REPLICATES):
                        raise DynamicFormalError(
                            f"PAIR_COUNT:{contrast_name}:{endpoint_name}:{ds}:{scenario}:{len(deltas)}"
                        )

                    p = exact_two_sided_sign_flip(deltas)
                    hypotheses[hypothesis_id] = p
                    ci_low, ci_high, bootstrap_seed = bootstrap_mean_ci(
                        deltas,
                        seed_text=f"{STUDY_ID}|{hypothesis_id}",
                    )
                    mean_delta = _mean(deltas)
                    if favorable == "lower":
                        first_wins = sum(d < 0 for d in deltas)
                        comparator_wins = sum(d > 0 for d in deltas)
                        mean_direction = "first_win" if mean_delta < 0 else "comparator_win" if mean_delta > 0 else "tie"
                    else:
                        first_wins = sum(d > 0 for d in deltas)
                        comparator_wins = sum(d < 0 for d in deltas)
                        mean_direction = "first_win" if mean_delta > 0 else "comparator_win" if mean_delta < 0 else "tie"
                    provisional.append({
                        "hypothesis_id": hypothesis_id,
                        "contrast": contrast_name,
                        "first_method": full_method,
                        "comparator": comparator,
                        "endpoint": endpoint_name,
                        "favorable_direction_for_first_method": favorable,
                        "dataset": ds,
                        "scenario": scenario,
                        "endpoint_status": "available",
                        "n": len(deltas),
                        "first_method_pair_wins": first_wins,
                        "ties": sum(d == 0 for d in deltas),
                        "comparator_pair_wins": comparator_wins,
                        "mean_delta_first_minus_comparator": mean_delta,
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "bootstrap_seed": bootstrap_seed,
                        "exact_two_sided_sign_flip_p": float(p),
                        "paired_rank_biserial_first_minus_comparator": float(paired_rank_biserial(deltas)),
                        "mean_direction": mean_direction,
                    })
            adjusted = holm_adjust(hypotheses)
            for row in provisional:
                if row.get("endpoint_status") != "available":
                    row["holm_adjusted_p"] = None
                    row["holm_direction"] = "not_tested"
                    row["holm_family_size"] = len(hypotheses)
                    output.append(row)
                    continue
                adj = adjusted[row["hypothesis_id"]]
                row["holm_adjusted_p"] = float(adj)
                row["holm_family_size"] = len(hypotheses)
                if adj < Fraction(1, 20):
                    row["holm_direction"] = row["mean_direction"]
                else:
                    row["holm_direction"] = "nonsignificant"
                output.append(row)
    return output


def _cell_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successes = [dict(r) for r in rows if r.get("status") == "success"]
    out: list[dict[str, Any]] = []
    for ds in DATASETS:
        for scenario in SCENARIOS:
            cell = {"dataset": ds, "scenario": scenario, "n": len(FORMAL_REPLICATES)}
            for method in METHODS:
                subset = [
                    r for r in successes
                    if r["dataset"] == ds and r["scenario"] == scenario and r["method"] == method
                ]
                if len(subset) != len(FORMAL_REPLICATES):
                    raise DynamicFormalError(f"CELL_COUNT:{ds}:{scenario}:{method}:{len(subset)}")
                cell[f"{method}_primary_loss_mean"] = statistics.fmean(float(r["primary_loss"]) for r in subset)
                cell[f"{method}_paper_metric_mean"] = statistics.fmean(float(r["paper_metric"]) for r in subset)
                cell[f"{method}_early_post_point_loss_mean"] = statistics.fmean(
                    float(r["early_post_point_loss_mean"]) for r in subset
                )
                align = [r["mean_post_calibration_alignment_score_higher_is_better"] for r in subset]
                cell[f"{method}_post_cal_alignment_mean"] = (
                    statistics.fmean(float(v) for v in align if v is not None)
                    if any(v is not None for v in align) else None
                )
            out.append(cell)
    return out


def _analysis_summary(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [r for r in comparisons if r["endpoint"] == "post_switch_primary_loss"]
    by_contrast: dict[str, Any] = {}
    for _, _, contrast in PRIMARY_CONTRASTS:
        rows = [r for r in primary if r["contrast"] == contrast]
        by_contrast[contrast] = {
            "cell_count": len(rows),
            "mean_first_method_wins": sum(r["mean_direction"] == "first_win" for r in rows),
            "mean_ties": sum(r["mean_direction"] == "tie" for r in rows),
            "mean_comparator_wins": sum(r["mean_direction"] == "comparator_win" for r in rows),
            "holm_significant_first_method_wins": sum(r["holm_direction"] == "first_win" for r in rows),
            "holm_significant_comparator_wins": sum(r["holm_direction"] == "comparator_win" for r in rows),
        }

    core = by_contrast["full_vs_cowa"]
    gate_pass = (
        core["mean_first_method_wins"] >= FORMAL_GATE["full_vs_cowa_min_mean_win_cells"]
        and core["holm_significant_first_method_wins"] >= FORMAL_GATE["full_vs_cowa_min_holm_win_cells"]
        and core["holm_significant_comparator_wins"] <= FORMAL_GATE["full_vs_cowa_max_holm_loss_cells"]
    )

    categorical_rows = [
        r for r in primary
        if r["contrast"] == "full_vs_cowa" and r["dataset"] in {"product", "duck", "dog"}
    ]

    return {
        "formal_holdout_replicates": list(FORMAL_REPLICATES),
        "probe_replicates_excluded_from_inference": [1, 2, 3, 4, 5],
        "primary_endpoint": "post_switch_primary_loss",
        "by_contrast": by_contrast,
        "categorical_full_vs_cowa": {
            "cell_count": len(categorical_rows),
            "mean_lir_wins": sum(r["mean_direction"] == "first_win" for r in categorical_rows),
            "holm_significant_lir_wins": sum(r["holm_direction"] == "first_win" for r in categorical_rows),
            "holm_significant_cowa_wins": sum(r["holm_direction"] == "comparator_win" for r in categorical_rows),
        },
        "secondary_endpoint_availability": {
            "post_switch_primary_loss_cells": 8,
            "early_post_switch_loss_cells": 8,
            "post_calibration_alignment_cells": sum(
                1
                for r in comparisons
                if r["contrast"] == "full_vs_cowa"
                and r["endpoint"] == "post_calibration_alignment"
                and r.get("endpoint_status") == "available"
            ),
            "post_calibration_alignment_unavailable_cells": [
                f"{r['dataset']}|{r['scenario']}"
                for r in comparisons
                if r["contrast"] == "full_vs_cowa"
                and r["endpoint"] == "post_calibration_alignment"
                and r.get("endpoint_status") != "available"
            ],
            "reason": "The frozen hidden selector may yield zero post-switch calibration events on a short dataset; such cells are N/A and excluded from that endpoint's Holm family.",
        },
        "formal_dynamic_necessity_gate_rule": FORMAL_GATE,
        "formal_dynamic_necessity_gate": "PASS" if gate_pass else "REVIEW_REQUIRED",
        "retuning_allowed": False,
    }


def validate(root: Path):
    required = [
        root / P0_2_MANIFEST_REL,
        root / P0_1C_SEEDS_REL,
        root / P0_1C_AUDIT_REL,
        root / DATASET_HASHES_REL,
        root / "src/lir_pptd/core/hidden_calibration.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise DynamicFormalError("MISSING:" + ",".join(missing))

    manifest_path = root / P0_2_MANIFEST_REL
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise DynamicFormalError(
            f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}"
        )

    manifest = _load_json(manifest_path)
    datasets = _load_datasets(root)
    selectors = _load_selectors(root, datasets, manifest_sha)
    masks, mask_hashes, calibration_counts = _verify_masks(
        datasets, selectors, _load_json(root / P0_1C_AUDIT_REL)
    )
    streams = _plan_streams(manifest, datasets)

    print("COWA_DYNAMIC_FORMAL_VALIDATE=PASS")
    print("DATASETS=product,duck,dog,weather")
    print("RHO=7/10")
    print("SCENARIOS=benign_to_malicious,malicious_to_benign")
    print("PROBE_REPLICATES_EXCLUDED=1-5")
    print("FORMAL_HOLDOUT_REPLICATES=6-20")
    print(f"FORMAL_REPLICATES_PER_CELL={len(FORMAL_REPLICATES)}")
    print(f"STREAMS={len(streams)}")
    print(f"METHODS={len(METHODS)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print("METHOD_SET=lir_pptd,cowa,no_cw,no_hr")
    print("P0_1C_HIDDEN_SELECTOR_MASK_REUSE=EXACT")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("SWITCH_POINT=HALF_OF_CHRONOLOGICAL_TASK_SEQUENCE")
    print("COWA_SAME_Q_CAL_REFERENCE_AS_LIR=YES")
    print("NO_CW_RETAINS_CALIBRATION_REPUTATION=YES")
    print("NO_CW_CURRENT_TASK_CONSISTENCY=NO")
    print("NO_HR_PERSISTENT_REPUTATION=NO")
    print("PRIMARY_ENDPOINT=POST_SWITCH_PRIMARY_LOSS")
    print("BOOTSTRAP_RESAMPLES=10000")
    print("EXACT_SIGN_FLIP=YES")
    print("HOLM_FAMILY=8_CELLS_PER_CONTRAST_PER_ENDPOINT")
    print("ALGORITHM_RETUNING=NO")
    for ds in DATASETS:
        print(f"CALIBRATION_COUNT_{ds.upper()}={calibration_counts[ds]}")
    return datasets, selectors, masks, mask_hashes, streams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    datasets, selectors, masks, mask_hashes, streams = validate(root)
    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"
    if raw_path.exists() and not args.resume:
        raise DynamicFormalError(f"OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)
    for index, stream in enumerate(streams, start=1):
        dataset = datasets[stream.dataset_id]
        tasks, switch_after = _dynamic_tasks(dataset, stream)
        for method in METHODS:
            run_id = _run_id(stream, method, mask_hashes[stream.dataset_id])
            if run_id in existing and existing[run_id].get("status") == "success":
                continue
            started = time.perf_counter()
            try:
                result = _run_method(
                    dataset,
                    tasks,
                    masks[stream.dataset_id],
                    stream,
                    switch_after,
                    method,
                )
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": RHO,
                    "scenario": stream.scenario,
                    "replicate": stream.replicate,
                    "subset_seed": stream.subset_seed,
                    "switched_workers": list(stream.switched_workers),
                    "method": method,
                    "status": "success",
                    "switch_after_task_index": switch_after,
                    "calibration_mask_sha256": mask_hashes[stream.dataset_id],
                    "seed_commitment_sha256": selectors[stream.dataset_id].seed_commitment_sha256,
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
                    "rho": RHO,
                    "scenario": stream.scenario,
                    "replicate": stream.replicate,
                    "method": method,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            _append(raw_path, row)
            existing[run_id] = row

        if index % 5 == 0 or index == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"PROGRESS={index}/{len(streams)} "
                f"METHOD_RUNS={len(existing)}/{len(streams)*len(METHODS)} "
                f"SUCCESS={success}"
            )

    final = list(existing.values())
    _rewrite_canonical(raw_path, final)
    final = list(_read_existing(raw_path).values())
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)

    comparisons: list[dict[str, Any]] = []
    cell_means: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}
    planned = len(streams) * len(METHODS)

    if failure == 0 and success == planned:
        comparisons = _comparison_rows(final)
        cell_means = _cell_means(final)
        analysis = _analysis_summary(comparisons)
        _write_csv(outdir / "paired_comparisons.csv", comparisons)
        _write_csv(outdir / "dynamic_cell_means.csv", cell_means)

    audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "p0_2_manifest_sha256": P0_2_MANIFEST_SHA256,
        "probe_replicates_excluded": [1, 2, 3, 4, 5],
        "formal_holdout_replicates": list(FORMAL_REPLICATES),
        "same_p0_1c_hidden_selector_masks": True,
        "attack_generation_before_mode_designation": True,
        "switch_point": "half chronological task sequence",
        "methods": list(METHODS),
        "contrasts": [
            {
                "first_method": a,
                "comparator": b,
                "contrast": c,
            }
            for a, b, c in PRIMARY_CONTRASTS
        ],
        "retuning_allowed": False,
    }
    _atomic_json(outdir / "formal_audit.json", audit)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "streams": len(streams),
        "planned_method_runs": planned,
        "completed_unique_method_runs": len(final),
        "success": success,
        "failure": failure,
        "datasets": list(DATASETS),
        "rho": RHO,
        "scenarios": list(SCENARIOS),
        "formal_holdout_replicates": list(FORMAL_REPLICATES),
        "methods": list(METHODS),
        "analysis": analysis,
        "scope": {
            "formal_dynamic_cowa_stress": True,
            "probe_replicates_used_for_inference": False,
            "algorithm_retuning": False,
            "selector_reselection": False,
        },
    }
    _atomic_json(outdir / "formal_summary.json", summary)

    print("COWA_DYNAMIC_FORMAL_STRESS=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"PLANNED_METHOD_RUNS={planned}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")
    if analysis:
        core = analysis["by_contrast"]["full_vs_cowa"]
        print(
            "FULL_VS_COWA_MEAN_WINS_TIES_LOSSES="
            f"{core['mean_first_method_wins']}/"
            f"{core['mean_ties']}/"
            f"{core['mean_comparator_wins']}"
        )
        print(
            "FULL_VS_COWA_HOLM_SIG_WINS_LOSSES="
            f"{core['holm_significant_first_method_wins']}/"
            f"{core['holm_significant_comparator_wins']}"
        )
        nocw = analysis["by_contrast"]["no_cw_vs_cowa"]
        print(
            "NO_CW_VS_COWA_MEAN_WINS_TIES_LOSSES="
            f"{nocw['mean_first_method_wins']}/"
            f"{nocw['mean_ties']}/"
            f"{nocw['mean_comparator_wins']}"
        )
        full_nocw = analysis["by_contrast"]["full_vs_no_cw"]
        print(
            "FULL_VS_NO_CW_MEAN_WINS_TIES_LOSSES="
            f"{full_nocw['mean_first_method_wins']}/"
            f"{full_nocw['mean_ties']}/"
            f"{full_nocw['mean_comparator_wins']}"
        )
        full_nohr = analysis["by_contrast"]["full_vs_no_hr"]
        print(
            "FULL_VS_NO_HR_MEAN_WINS_TIES_LOSSES="
            f"{full_nohr['mean_first_method_wins']}/"
            f"{full_nohr['mean_ties']}/"
            f"{full_nohr['mean_comparator_wins']}"
        )
        cat = analysis["categorical_full_vs_cowa"]
        print(
            "CATEGORICAL_FULL_VS_COWA_MEAN_WINS="
            f"{cat['mean_lir_wins']}/{cat['cell_count']}"
        )
        print(
            "FORMAL_DYNAMIC_NECESSITY_GATE="
            f"{analysis['formal_dynamic_necessity_gate']}"
        )
    print(f"SUMMARY={outdir / 'formal_summary.json'}")
    return 0 if success == planned and failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
