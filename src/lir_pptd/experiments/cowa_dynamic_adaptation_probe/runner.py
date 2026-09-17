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

STUDY_ID = "cowa_dynamic_adaptation_probe_v1"
P0_1C_SELECTOR_STUDY_ID = "p0_1c_hidden_b1_formal_migration_v1"

DATASETS = ("product", "duck", "dog", "weather")
SCENARIOS = ("benign_to_malicious", "malicious_to_benign")
RHO = "7/10"
REPLICATES = 5
METHODS = ("lir_pptd", "cowa")
PERIOD = 20
EARLY_POST_FRACTION = Fraction(1, 4)

P0_2_MANIFEST_REL = "configs/p0_2_b1_sample_count/final_b1_subset_manifest.json"
P0_2_MANIFEST_SHA256 = "288f24398ba3a5dca4f86388b818400ae5129ef4f02822c2fea79a956a47d06b"
P0_1C_SEEDS_REL = "configs/p0_1c_hidden_b1/private_selector_seeds.json"
P0_1C_AUDIT_REL = "results/p0_1c_hidden_b1_v1/selector_audit.json"
DATASET_HASHES_REL = "configs/gold20_discrepancy/dataset_hashes.json"
OUT_REL = "results/cowa_dynamic_adaptation_probe_v1"

CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "mu": "1/5",
    "eta": "1/10",
}


class DynamicProbeError(RuntimeError):
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
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
    seed_obj = _load_json(root / P0_1C_SEEDS_REL)
    seeds = {str(k): str(v) for k, v in seed_obj.get("seeds", {}).items()}
    if set(seeds) != set(DATASETS):
        raise DynamicProbeError("P0_1C_SELECTOR_SEED_DATASET_SET_MISMATCH")
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
) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    expected = {str(x["dataset"]): dict(x) for x in audit["selectors"]}
    masks: dict[str, frozenset[str]] = {}
    hashes: dict[str, str] = {}
    for ds in DATASETS:
        ids = [t.task_id for t in datasets[ds].tasks]
        cal = frozenset(selectors[ds].selected_task_ids(ids))
        mask_hash = _sha_obj(sorted(cal))
        e = expected[ds]
        if mask_hash != str(e["calibration_mask_sha256"]):
            raise DynamicProbeError(f"MASK_HASH_MISMATCH:{ds}")
        if len(cal) != int(e["calibration_task_count"]):
            raise DynamicProbeError(f"MASK_COUNT_MISMATCH:{ds}")
        if selectors[ds].seed_commitment_sha256 != str(e["seed_commitment_sha256"]):
            raise DynamicProbeError(f"SEED_COMMITMENT_MISMATCH:{ds}")
        masks[ds] = cal
        hashes[ds] = mask_hash
    return masks, hashes


def _plan_streams(manifest: Mapping[str, Any], datasets: Mapping[str, RealDataset]) -> list[Stream]:
    out: list[Stream] = []
    for ds in DATASETS:
        rows = manifest["datasets"][ds][RHO]["subsets"]
        if len(rows) < REPLICATES:
            raise DynamicProbeError(f"INSUFFICIENT_SUBSETS:{ds}:{len(rows)}")
        valid = set(datasets[ds].worker_ids)
        for scenario in SCENARIOS:
            for rep, row in enumerate(rows[:REPLICATES], start=1):
                workers = tuple(sorted(map(str, row["workers"])))
                if not set(workers) <= valid:
                    raise DynamicProbeError(f"UNKNOWN_WORKER:{ds}:{rep}")
                out.append(Stream(ds, scenario, rep, int(row["seed"]), workers))
    if len(out) != len(DATASETS) * len(SCENARIOS) * REPLICATES:
        raise DynamicProbeError("STREAM_COUNT_MISMATCH")
    return out


def active_for_scenario(scenario: str, task_index_1based: int, switch_after_index: int) -> bool:
    if scenario == "benign_to_malicious":
        return task_index_1based > switch_after_index
    if scenario == "malicious_to_benign":
        return task_index_1based <= switch_after_index
    raise DynamicProbeError(f"UNKNOWN_SCENARIO:{scenario}")


def _dynamic_tasks(dataset: RealDataset, stream: Stream) -> tuple[tuple[RealTask, ...], int]:
    # Crucial: this transformation knows only public chronology and the
    # preregistered behavior switch. It has no calibration-mask argument and
    # therefore cannot condition the attack on hidden mode designation.
    switch_after = len(dataset.tasks) // 2
    out: list[RealTask] = []
    for idx, task in enumerate(dataset.tasks, start=1):
        if active_for_scenario(stream.scenario, idx, switch_after):
            reports = apply_response_corruption(
                task.reports,
                stream.switched_workers,
                modality=task.modality,
                active=True,
            )
        else:
            reports = {w: tuple(v) for w, v in task.reports.items()}
        out.append(replace(task, reports=reports))
    return tuple(out), switch_after


def _class_from_vector(values: Sequence[float]) -> int:
    maximum = max(values)
    return min(i for i, v in enumerate(values) if v == maximum)


def _cowa_predict(task: RealTask, history: COWAHistory) -> tuple[tuple[float, ...], int | None]:
    dimension = len(next(iter(task.reports.values())))
    num = [Fraction(0, 1) for _ in range(dimension)]
    den = Fraction(0, 1)
    for worker in task.participant_ids:
        w = history.weight(worker)
        den += w
        report = task.reports[worker]
        for h in range(dimension):
            num[h] += w * _fraction(report[h])
    if den <= 0:
        raise DynamicProbeError("NONPOSITIVE_COWA_DENOMINATOR")
    vector = tuple(float(x / den) for x in num)
    cls = _class_from_vector(vector) if task.modality == "categorical" else None
    return vector, cls


def _group_mean(weights: Mapping[str, Any], group: Sequence[str]) -> float:
    vals = [float(_fraction(weights[w])) for w in group if w in weights]
    if not vals:
        raise DynamicProbeError("EMPTY_GROUP_WEIGHT")
    return statistics.fmean(vals)


def _state_weights(state: Any, workers: Sequence[str]) -> dict[str, Any]:
    reps = dict(state.reputations or {})
    return {str(w): reps[str(w)] for w in workers}


def _cowa_weights(history: COWAHistory, workers: Sequence[str]) -> dict[str, Any]:
    return {str(w): history.weight(str(w)) for w in workers}


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
    name = "Accuracy" if dataset.dataset_id == "duck" else "MacroF1"
    return {"primary_loss": primary_loss, "paper_metric_name": name, "paper_metric": 1.0 - primary_loss}


def _point_loss(dataset: RealDataset, task: RealTask, vector: Sequence[float], cls: int | None) -> float:
    if dataset.modality == "categorical":
        if cls is None or task.truth_class is None:
            raise DynamicProbeError("MISSING_CLASS")
        return 0.0 if int(cls) == int(task.truth_class) else 1.0
    return abs(float(vector[0]) - float(task.truth_vector[0]))


def _run_method(
    dataset: RealDataset,
    tasks: Sequence[RealTask],
    cal_ids: frozenset[str],
    stream: Stream,
    switch_after: int,
    method: str,
) -> dict[str, Any]:
    switched = tuple(stream.switched_workers)
    honest = tuple(w for w in dataset.worker_ids if w not in set(switched))
    if not honest:
        raise DynamicProbeError("EMPTY_STABLE_HONEST_GROUP")

    if method == "lir_pptd":
        state = initial_global_state(dataset.worker_ids, CONFIG["c0"])
        history = None
    elif method == "cowa":
        state = initial_global_state(dataset.worker_ids, CONFIG["c0"])  # q_cal extraction only
        history = COWAHistory.initialize(dataset.worker_ids, CONFIG["c0"])
    else:
        raise DynamicProbeError(f"UNKNOWN_METHOD:{method}")

    post_pc: list[int] = []
    post_tc: list[int] = []
    post_pv: list[float] = []
    post_tv: list[float] = []
    post_point_losses: list[float] = []
    early_point_losses: list[float] = []
    gap_series: list[float] = []
    post_cal_events = 0
    post_eval_total = sum(
        1 for idx, task in enumerate(tasks, start=1)
        if idx > switch_after and task.task_id not in cal_ids
    )
    early_limit = max(1, math.ceil(post_eval_total * float(EARLY_POST_FRACTION)))

    for idx, task in enumerate(tasks, start=1):
        if task.task_id in cal_ids:
            result, next_state = run_calibration_task(task, CONFIG, state, dps=80)
            if method == "lir_pptd":
                state = next_state
                weights = _state_weights(state, dataset.worker_ids)
            else:
                assert history is not None
                for item in result.evidence:
                    history.observe(str(item.worker_id), item.evidence)
                weights = _cowa_weights(history, dataset.worker_ids)
                # Keep extraction state unchanged: COWA uses q_cal but not F(c,q).
            if idx > switch_after:
                post_cal_events += 1
                gap_series.append(_group_mean(weights, honest) - _group_mean(weights, switched))
            continue

        if method == "lir_pptd":
            pred = run_lir_task(task, CONFIG, state, ablation="full", dps=80)
            vector = pred.prediction_vector
            cls = pred.prediction_class
            # Evidence-gated invariant: ordinary pred.next_state is NOT committed.
        else:
            assert history is not None
            vector, cls = _cowa_predict(task, history)

        if idx <= switch_after:
            continue

        if dataset.modality == "categorical":
            if cls is None or task.truth_class is None:
                raise DynamicProbeError("MISSING_POST_CLASS")
            post_pc.append(int(cls))
            post_tc.append(int(task.truth_class))
        else:
            post_pv.append(float(vector[0]))
            post_tv.append(float(task.truth_vector[0]))

        loss = _point_loss(dataset, task, vector, cls)
        post_point_losses.append(loss)
        if len(early_point_losses) < early_limit:
            early_point_losses.append(loss)

    if method == "lir_pptd":
        final_weights = _state_weights(state, dataset.worker_ids)
    else:
        assert history is not None
        final_weights = _cowa_weights(history, dataset.worker_ids)

    metrics = _metric_from_records(dataset, post_pc, post_tc, post_pv, post_tv)
    final_gap = _group_mean(final_weights, honest) - _group_mean(final_weights, switched)

    return {
        **metrics,
        "post_switch_evaluation_task_count": len(post_point_losses),
        "post_switch_point_loss_mean": statistics.fmean(post_point_losses),
        "early_post_fraction": "1/4",
        "early_post_task_count": len(early_point_losses),
        "early_post_point_loss_mean": statistics.fmean(early_point_losses),
        "post_switch_calibration_events": post_cal_events,
        "final_honest_minus_switched_weight_gap": final_gap,
        "mean_post_calibration_honest_minus_switched_gap": (
            statistics.fmean(gap_series) if gap_series else None
        ),
    }


def adaptation_advantage(
    scenario: str,
    lir_final_gap: float,
    cowa_final_gap: float,
) -> float:
    # Positive always favors faster/more appropriate LIR adaptation.
    if scenario == "benign_to_malicious":
        # Newly malicious cohort should be separated downward: larger H-M gap.
        return lir_final_gap - cowa_final_gap
    if scenario == "malicious_to_benign":
        # Rehabilitated cohort should return toward honest weights: smaller |gap|.
        return abs(cowa_final_gap) - abs(lir_final_gap)
    raise DynamicProbeError(f"UNKNOWN_SCENARIO:{scenario}")


def _run_id(stream: Stream, method: str, mask_hash: str) -> str:
    return hashlib.sha256(
        f"{STUDY_ID}|{mask_hash}|{stream.pairing_id}|{method}".encode("utf-8")
    ).hexdigest()


def _analyze(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success = [dict(r) for r in rows if r.get("status") == "success"]
    by = {
        (str(r["dataset"]), str(r["scenario"]), int(r["replicate"]), str(r["method"])): r
        for r in success
    }
    cells: list[dict[str, Any]] = []
    for ds in DATASETS:
        for scenario in SCENARIOS:
            truth_deltas: list[float] = []
            early_deltas: list[float] = []
            adaptation: list[float] = []
            for rep in range(1, REPLICATES + 1):
                lir = by[(ds, scenario, rep, "lir_pptd")]
                cowa = by[(ds, scenario, rep, "cowa")]
                truth_deltas.append(float(lir["primary_loss"]) - float(cowa["primary_loss"]))
                early_deltas.append(
                    float(lir["early_post_point_loss_mean"]) - float(cowa["early_post_point_loss_mean"])
                )
                adaptation.append(adaptation_advantage(
                    scenario,
                    float(lir["final_honest_minus_switched_weight_gap"]),
                    float(cowa["final_honest_minus_switched_weight_gap"]),
                ))
            cells.append({
                "dataset": ds,
                "scenario": scenario,
                "n": REPLICATES,
                "lir_post_switch_truth_wins": sum(x < 0 for x in truth_deltas),
                "truth_ties": sum(x == 0 for x in truth_deltas),
                "cowa_post_switch_truth_wins": sum(x > 0 for x in truth_deltas),
                "mean_post_switch_loss_delta_lir_minus_cowa": statistics.fmean(truth_deltas),
                "lir_early_post_wins": sum(x < 0 for x in early_deltas),
                "mean_early_post_point_loss_delta_lir_minus_cowa": statistics.fmean(early_deltas),
                "lir_adaptation_wins": sum(x > 0 for x in adaptation),
                "mean_adaptation_advantage_positive_favors_lir": statistics.fmean(adaptation),
            })

    categorical = [r for r in cells if r["dataset"] in {"product", "duck", "dog"}]
    summary = {
        "cell_count": len(cells),
        "probe_n_per_cell": REPLICATES,
        "inferential_significance_claimed": False,
        "truth_mean_lir_win_cells": sum(r["mean_post_switch_loss_delta_lir_minus_cowa"] < 0 for r in cells),
        "truth_mean_cowa_win_cells": sum(r["mean_post_switch_loss_delta_lir_minus_cowa"] > 0 for r in cells),
        "adaptation_mean_lir_win_cells": sum(
            r["mean_adaptation_advantage_positive_favors_lir"] > 0 for r in cells
        ),
        "categorical_truth_mean_lir_win_cells": sum(
            r["mean_post_switch_loss_delta_lir_minus_cowa"] < 0 for r in categorical
        ),
        "categorical_cell_count": len(categorical),
        "decision": "REVIEW_AFTER_PROBE",
        "retuning_allowed": False,
    }
    return cells, summary


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
        raise DynamicProbeError("MISSING:" + ",".join(missing))

    manifest_path = root / P0_2_MANIFEST_REL
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != P0_2_MANIFEST_SHA256:
        raise DynamicProbeError(f"P0_2_MANIFEST_SHA256:{manifest_sha}:{P0_2_MANIFEST_SHA256}")

    manifest = _load_json(manifest_path)
    datasets = _load_datasets(root)
    selectors = _load_selectors(root, datasets, manifest_sha)
    masks, mask_hashes = _verify_masks(
        datasets, selectors, _load_json(root / P0_1C_AUDIT_REL)
    )
    streams = _plan_streams(manifest, datasets)

    print("COWA_DYNAMIC_ADAPTATION_PROBE_VALIDATE=PASS")
    print("DATASETS=product,duck,dog,weather")
    print("RHO=7/10")
    print("SCENARIOS=benign_to_malicious,malicious_to_benign")
    print("REPLICATES_PER_CELL=5")
    print(f"STREAMS={len(streams)}")
    print(f"METHODS={len(METHODS)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print("P0_1C_HIDDEN_SELECTOR_MASK_REUSE=EXACT")
    print("ATTACK_GENERATION_BEFORE_MODE_DESIGNATION=YES")
    print("SWITCH_POINT=HALF_OF_CHRONOLOGICAL_TASK_SEQUENCE")
    print("COWA_SAME_Q_CAL_REFERENCE_AS_LIR=YES")
    print("LIR_ORDINARY_PERSISTENT_REPUTATION_UPDATE=DISABLED")
    print("PROBE_ONLY=YES")
    print("INFERENTIAL_SIGNIFICANCE_CLAIMED=NO")
    print("ALGORITHM_RETUNING=NO")
    return datasets, selectors, masks, mask_hashes, streams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    datasets, selectors, masks, mask_hashes, streams = validate(root)
    if args.validate_only:
        print("PROBE_RUNS_STARTED=NO")
        return 0

    outdir = root / OUT_REL
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "formal_runs.jsonl"
    if raw_path.exists() and not args.resume:
        raise DynamicProbeError(f"OUTPUT_EXISTS_USE_RESUME:{raw_path}")

    existing = _read_existing(raw_path)
    for idx, stream in enumerate(streams, start=1):
        dataset = datasets[stream.dataset_id]
        tasks, switch_after = _dynamic_tasks(dataset, stream)
        for method in METHODS:
            rid = _run_id(stream, method, mask_hashes[stream.dataset_id])
            if rid in existing and existing[rid].get("status") == "success":
                continue
            t0 = time.perf_counter()
            try:
                result = _run_method(
                    dataset, tasks, masks[stream.dataset_id], stream, switch_after, method
                )
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": rid,
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
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "study_id": STUDY_ID,
                    "run_id": rid,
                    "pairing_id": stream.pairing_id,
                    "dataset": stream.dataset_id,
                    "rho": RHO,
                    "scenario": stream.scenario,
                    "replicate": stream.replicate,
                    "method": method,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            _append(raw_path, row)
            existing[rid] = row

        if idx % 5 == 0 or idx == len(streams):
            success = sum(r.get("status") == "success" for r in existing.values())
            print(
                f"PROGRESS={idx}/{len(streams)} "
                f"METHOD_RUNS={len(existing)}/{len(streams)*len(METHODS)} "
                f"SUCCESS={success}"
            )

    final = list(existing.values())
    _rewrite_canonical(raw_path, final)
    final = list(_read_existing(raw_path).values())
    success = sum(r.get("status") == "success" for r in final)
    failure = sum(r.get("status") != "success" for r in final)

    cells: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}
    if failure == 0 and success == len(streams) * len(METHODS):
        cells, analysis = _analyze(final)
        _write_csv(outdir / "dynamic_cell_summary.csv", cells)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "streams": len(streams),
        "planned_method_runs": len(streams) * len(METHODS),
        "completed_unique_method_runs": len(final),
        "success": success,
        "failure": failure,
        "datasets": list(DATASETS),
        "rho": RHO,
        "scenarios": list(SCENARIOS),
        "replicates_per_cell": REPLICATES,
        "analysis": analysis,
        "scope": {
            "probe_only": True,
            "inferential_significance_claimed": False,
            "algorithm_retuning": False,
            "formal_dynamic_stress": False,
        },
    }
    _atomic_json(outdir / "formal_summary.json", summary)

    print("COWA_DYNAMIC_ADAPTATION_PROBE=COMPLETE")
    print(f"STREAMS={len(streams)}")
    print(f"PLANNED_METHOD_RUNS={len(streams)*len(METHODS)}")
    print(f"SUCCESS={success}")
    print(f"FAILURE={failure}")
    if analysis:
        print(
            "DYNAMIC_TRUTH_MEAN_LIR_COWA_WIN_CELLS="
            f"{analysis['truth_mean_lir_win_cells']}/"
            f"{analysis['truth_mean_cowa_win_cells']}"
        )
        print(
            "DYNAMIC_ADAPTATION_MEAN_LIR_WIN_CELLS="
            f"{analysis['adaptation_mean_lir_win_cells']}/{analysis['cell_count']}"
        )
        print(
            "CATEGORICAL_DYNAMIC_TRUTH_LIR_WIN_CELLS="
            f"{analysis['categorical_truth_mean_lir_win_cells']}/"
            f"{analysis['categorical_cell_count']}"
        )
        print("FORMAL_DYNAMIC_STRESS_DECISION=REVIEW_AFTER_PROBE")
    print(f"SUMMARY={outdir / 'formal_summary.json'}")
    return 0 if failure == 0 and success == len(streams) * len(METHODS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
