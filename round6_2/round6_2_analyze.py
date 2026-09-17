from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

STUDY_ID = "round6_2_multi_selector_seed_robustness_v1"
PRIMARY_COMPARATORS = ("crh", "pptd_liang")
SECONDARY_COMPARATORS = ("tqpp_liu",)
LIR = "lir_pptd"
BOOTSTRAP_RESAMPLES = 10000


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSONL_PARSE_ERROR:{lineno}:{e}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _seed_for(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _bootstrap_ci(values: list[float], *, key: str, n: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(_seed_for(STUDY_ID, "outer-bootstrap", key))
    m = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(m)] for _ in range(m)) / m)
    means.sort()
    lo = means[int(0.025 * (n - 1))]
    hi = means[int(0.975 * (n - 1))]
    return lo, hi


def _exact_sign_flip(values: list[float]) -> float:
    vals = [float(v) for v in values if abs(float(v)) > 1e-15]
    n = len(vals)
    if n == 0:
        return 1.0
    if n > 22:
        raise SystemExit(f"EXACT_SIGN_FLIP_N_TOO_LARGE:{n}")
    obs = abs(sum(vals))
    sums = [0.0]
    for x in vals:
        sums = [s + x for s in sums] + [s - x for s in sums]
    threshold = obs - 1e-12
    extreme = sum(abs(s) >= threshold for s in sums)
    return extreme / len(sums)


def _holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda x: x[1])
    m = len(ordered)
    raw_adj = []
    for i, (key, p) in enumerate(ordered):
        raw_adj.append((key, min(1.0, (m - i) * p)))
    out = {}
    running = 0.0
    for key, adj in raw_adj:
        running = max(running, adj)
        out[key] = min(1.0, running)
    return out


def _sample_variance(xs: list[float]) -> float:
    return statistics.variance(xs) if len(xs) >= 2 else 0.0


def _hierarchical_ci(seed_to_deltas: dict[int, list[float]], *, key: str, n: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    seed_ids = sorted(seed_to_deltas)
    if not seed_ids:
        return math.nan, math.nan
    rng = random.Random(_seed_for(STUDY_ID, "hier-bootstrap", key))
    boot = []
    for _ in range(n):
        selected = [seed_ids[rng.randrange(len(seed_ids))] for _ in seed_ids]
        seed_means = []
        for sid in selected:
            vals = seed_to_deltas[sid]
            sampled = [vals[rng.randrange(len(vals))] for _ in vals]
            seed_means.append(sum(sampled) / len(sampled))
        boot.append(sum(seed_means) / len(seed_means))
    boot.sort()
    return boot[int(0.025 * (n - 1))], boot[int(0.975 * (n - 1))]


def _validate(rows: list[dict[str, Any]]) -> None:
    required = {"dataset", "rho", "selector_seed_index", "pairing_id", "method", "primary_loss", "paper_metric", "calibration_count", "calibration_mask_sha256"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"ROW_SCHEMA_MISSING:{i}:{sorted(missing)}")
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["rho"], int(row["selector_seed_index"]), row["pairing_id"])
        method = row["method"]
        if method in grouped[key]:
            raise SystemExit(f"DUPLICATE_METHOD_ROW:{key}:{method}")
        grouped[key][method] = row
    for key, methods in grouped.items():
        if LIR not in methods:
            raise SystemExit(f"LIR_ROW_MISSING:{key}")
        for comp in PRIMARY_COMPARATORS:
            if comp not in methods:
                raise SystemExit(f"PRIMARY_COMPARATOR_ROW_MISSING:{key}:{comp}")
        masks = {m["calibration_mask_sha256"] for m in methods.values()}
        counts = {int(m["calibration_count"]) for m in methods.values()}
        if len(masks) != 1 or len(counts) != 1:
            raise SystemExit(f"PAIRED_MASK_MISMATCH:{key}")


def analyze(rows: list[dict[str, Any]], out_dir: Path) -> None:
    _validate(rows)
    by = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["rho"], int(row["selector_seed_index"]), row["pairing_id"])
        by[key][row["method"]] = row

    comparators = [c for c in PRIMARY_COMPARATORS + SECONDARY_COMPARATORS if any(c in methods for methods in by.values())]
    seed_delta_rows = []
    cell_seed_map: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_map = {}
    count_map = defaultdict(set)
    mask_map = defaultdict(set)

    for (dataset, rho, sid, pairing), methods in by.items():
        lir = methods[LIR]
        metric_map[(dataset, rho)] = lir["paper_metric"]
        count_map[(dataset, sid)].add(int(lir["calibration_count"]))
        mask_map[(dataset, sid)].add(lir["calibration_mask_sha256"])
        for comp in comparators:
            if comp not in methods:
                continue
            delta = float(lir["primary_loss"]) - float(methods[comp]["primary_loss"])
            cell_seed_map[(dataset, rho, comp)][sid].append(delta)

    for (dataset, rho, comp), seed_map in sorted(cell_seed_map.items()):
        for sid, deltas in sorted(seed_map.items()):
            seed_delta_rows.append({
                "dataset": dataset,
                "rho": rho,
                "comparator": comp,
                "selector_seed_index": sid,
                "n_malicious_subsets": len(deltas),
                "mean_loss_delta_lir_minus_comparator": sum(deltas) / len(deltas),
                "median_loss_delta_lir_minus_comparator": statistics.median(deltas),
                "within_seed_subset_sd": statistics.stdev(deltas) if len(deltas) >= 2 else 0.0,
                "paper_metric": metric_map[(dataset, rho)],
            })

    _write_csv(out_dir / "seed_level_deltas.csv", seed_delta_rows, list(seed_delta_rows[0].keys()))

    cell_rows = []
    variance_rows = []
    hier_rows = []
    p_by_comp = defaultdict(list)
    raw_cell = {}

    for (dataset, rho, comp), seed_map in sorted(cell_seed_map.items()):
        seed_means = [sum(seed_map[s]) / len(seed_map[s]) for s in sorted(seed_map)]
        key = f"{dataset}|{rho}|{comp}"
        ci_lo, ci_hi = _bootstrap_ci(seed_means, key=key)
        p = _exact_sign_flip(seed_means)
        h_lo, h_hi = _hierarchical_ci(seed_map, key=key)
        between_var = _sample_variance(seed_means)
        within_vars = [_sample_variance(vals) for vals in seed_map.values()]
        mean_within_var = sum(within_vars) / len(within_vars) if within_vars else 0.0
        share = between_var / (between_var + mean_within_var) if between_var + mean_within_var > 0 else 0.0
        raw_cell[key] = {
            "dataset": dataset, "rho": rho, "comparator": comp,
            "selector_seed_n": len(seed_means),
            "mean_loss_delta_lir_minus_comparator": sum(seed_means) / len(seed_means),
            "median_seed_mean_delta": statistics.median(seed_means),
            "between_seed_sd": statistics.stdev(seed_means) if len(seed_means) >= 2 else 0.0,
            "bootstrap95_low_seed_unit": ci_lo,
            "bootstrap95_high_seed_unit": ci_hi,
            "exact_seed_level_sign_flip_p": p,
            "seed_mean_wins": sum(x < 0 for x in seed_means),
            "seed_mean_ties": sum(abs(x) <= 1e-15 for x in seed_means),
            "seed_mean_losses": sum(x > 0 for x in seed_means),
            "paper_metric": metric_map[(dataset, rho)],
        }
        p_by_comp[comp].append((key, p))
        variance_rows.append({
            "dataset": dataset, "rho": rho, "comparator": comp,
            "selector_seed_n": len(seed_means),
            "between_selector_seed_variance_of_seed_means": between_var,
            "mean_within_selector_seed_subset_variance": mean_within_var,
            "descriptive_between_seed_variance_share": share,
        })
        hier_rows.append({
            "dataset": dataset, "rho": rho, "comparator": comp,
            "hierarchical_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "hierarchical_bootstrap95_low": h_lo,
            "hierarchical_bootstrap95_high": h_hi,
        })

    holm = {comp: _holm(vals) for comp, vals in p_by_comp.items()}
    for key, row in raw_cell.items():
        row["holm_adjusted_seed_level_p"] = holm[row["comparator"]][key]
        mean = row["mean_loss_delta_lir_minus_comparator"]
        row["mean_direction"] = "lir_win" if mean < 0 else "lir_loss" if mean > 0 else "tie"
        row["holm_direction"] = row["mean_direction"] if row["holm_adjusted_seed_level_p"] < 0.05 else "not_significant"
        cell_rows.append(row)

    fields = [
        "dataset", "rho", "comparator", "selector_seed_n", "mean_loss_delta_lir_minus_comparator",
        "median_seed_mean_delta", "between_seed_sd", "bootstrap95_low_seed_unit", "bootstrap95_high_seed_unit",
        "exact_seed_level_sign_flip_p", "holm_adjusted_seed_level_p", "seed_mean_wins", "seed_mean_ties",
        "seed_mean_losses", "mean_direction", "holm_direction", "paper_metric"
    ]
    _write_csv(out_dir / "multi_selector_cell_summary.csv", cell_rows, fields)
    _write_csv(out_dir / "variance_decomposition.csv", variance_rows, list(variance_rows[0].keys()))
    _write_csv(out_dir / "hierarchical_bootstrap_summary.csv", hier_rows, list(hier_rows[0].keys()))

    selector_rows = []
    for (dataset, sid), counts in sorted(count_map.items()):
        masks = mask_map[(dataset, sid)]
        if len(counts) != 1 or len(masks) != 1:
            raise SystemExit(f"SELECTOR_MASK_NOT_STABLE_WITHIN_DATASET_SEED:{dataset}:{sid}")
        selector_rows.append({
            "dataset": dataset,
            "selector_seed_index": sid,
            "calibration_count": next(iter(counts)),
            "calibration_mask_sha256": next(iter(masks)),
        })
    _write_csv(out_dir / "selector_count_summary.csv", selector_rows, list(selector_rows[0].keys()))

    aggregate = {}
    for comp in comparators:
        subset = [r for r in cell_rows if r["comparator"] == comp]
        aggregate[comp] = {
            "cell_count": len(subset),
            "mean_lir_wins": sum(r["mean_direction"] == "lir_win" for r in subset),
            "mean_ties": sum(r["mean_direction"] == "tie" for r in subset),
            "mean_lir_losses": sum(r["mean_direction"] == "lir_loss" for r in subset),
            "holm_significant_lir_wins": sum(r["holm_direction"] == "lir_win" for r in subset),
            "holm_significant_lir_losses": sum(r["holm_direction"] == "lir_loss" for r in subset),
        }
    duck_counts = [r["calibration_count"] for r in selector_rows if r["dataset"] == "duck"]
    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "ANALYSIS_COMPLETE",
        "rows": len(rows),
        "selector_seed_count_per_dataset": {d: len({r["selector_seed_index"] for r in selector_rows if r["dataset"] == d}) for d in {r["dataset"] for r in selector_rows}},
        "aggregate_by_comparator": aggregate,
        "duck_calibration_count_distribution": {
            "n": len(duck_counts),
            "min": min(duck_counts),
            "max": max(duck_counts),
            "mean": sum(duck_counts) / len(duck_counts),
            "median": statistics.median(duck_counts),
            "counts": duck_counts,
        },
        "claim_policy": "No selector seed may be removed or replaced after analysis. Manuscript wording must follow the observed selector-seed distribution, including unfavorable seeds.",
    }
    _atomic_json(out_dir / "multi_selector_formal_summary.json", summary)
    print("ROUND6_2_ANALYSIS=PASS")
    print(json.dumps(aggregate, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-runs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = _load_jsonl(args.formal_runs)
    analyze(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
