from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

BOOTSTRAP_RESAMPLES = 10_000
ALPHA = Fraction(1, 20)


class PhaseR1StatisticsError(RuntimeError):
    pass


def _fraction(value: Any) -> Fraction:
    return Fraction(Decimal(str(value)))


def _fraction_json(value: Fraction) -> dict[str, Any]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator), "decimal": float(value)}


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def _scaled_integer_deltas(deltas: Sequence[Fraction]) -> list[int]:
    if not deltas:
        raise PhaseR1StatisticsError("SIGN_FLIP_EMPTY_SAMPLE")
    scale = reduce(_lcm, (v.denominator for v in deltas), 1)
    return [v.numerator * (scale // v.denominator) for v in deltas]


def _subset_sums(values: Sequence[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums += [current + value for current in sums]
    return sums


def exact_two_sided_sign_flip(deltas: Sequence[Fraction]) -> Fraction:
    scaled = _scaled_integer_deltas(deltas)
    magnitudes = [abs(v) for v in scaled]
    observed = sum(scaled)
    split = len(magnitudes) // 2
    left_weights = magnitudes[:split]
    right_weights = magnitudes[split:]
    left_total = sum(left_weights)
    right_total = sum(right_weights)
    left = [2 * s - left_total for s in _subset_sums(left_weights)]
    right = [2 * s - right_total for s in _subset_sums(right_weights)]
    right.sort()
    total = 1 << len(magnitudes)
    abs_obs = abs(observed)
    if abs_obs == 0:
        return Fraction(1, 1)
    count = 0
    for lv in left:
        low_threshold = -abs_obs - lv
        high_threshold = abs_obs - lv
        low = bisect.bisect_right(right, low_threshold)
        high = len(right) - bisect.bisect_left(right, high_threshold)
        count += low + high
    return min(Fraction(1, 1), Fraction(count, total))


def paired_rank_biserial(deltas: Sequence[Fraction]) -> Fraction:
    nonzero = [(abs(v), 1 if v > 0 else -1) for v in deltas if v != 0]
    if not nonzero:
        return Fraction(0, 1)
    nonzero.sort(key=lambda x: x[0])
    w_plus = Fraction(0, 1)
    w_minus = Fraction(0, 1)
    i = 0
    while i < len(nonzero):
        j = i + 1
        while j < len(nonzero) and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = Fraction((i + 1) + j, 2)
        for _, sign in nonzero[i:j]:
            if sign > 0:
                w_plus += avg_rank
            else:
                w_minus += avg_rank
        i = j
    den = w_plus + w_minus
    return Fraction(0, 1) if den == 0 else (w_plus - w_minus) / den


def holm_adjust(p_values: Mapping[str, Fraction]) -> dict[str, Fraction]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    out: list[tuple[str, Fraction]] = []
    running = Fraction(0, 1)
    for i, (hypothesis_id, p) in enumerate(ordered):
        candidate = min(Fraction(1, 1), p * (m - i))
        running = max(running, candidate)
        out.append((hypothesis_id, running))
    return dict(out)


def bootstrap_mean_ci(deltas: Sequence[Fraction], *, seed_text: str) -> tuple[float, float, int]:
    if not deltas:
        raise PhaseR1StatisticsError("BOOTSTRAP_EMPTY_SAMPLE")
    seed = int(hashlib.sha256(("phase-r1-bootstrap-v1|" + seed_text).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    vals = [float(v) for v in deltas]
    n = len(vals)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    def q(p: float) -> float:
        pos = (len(means) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return means[lo]
        f = pos - lo
        return means[lo] * (1 - f) + means[hi] * f

    return q(0.025), q(0.975), seed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PhaseR1StatisticsError(f"RAW_MISSING:{path}")
    out = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            run_id = obj.get("run_id")
            if not run_id or run_id in seen:
                raise PhaseR1StatisticsError(f"RAW_DUPLICATE_RUN_ID:line={line_no}")
            seen.add(run_id)
            out.append(obj)
    return out


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        raise PhaseR1StatisticsError("MEDIAN_EMPTY")
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def _planned_pair_count(stage: str, dataset: str, rho: str) -> int:
    if stage == "E1":
        if rho == "0":
            return 1
        if dataset == "weather" and rho in {"1/10", "9/10"}:
            return 9
        return 30
    if stage in {"E2", "E3"}:
        return 10
    if stage == "E4":
        return 5
    raise PhaseR1StatisticsError(f"UNKNOWN_STAGE:{stage}")


def analyze(root: Path) -> dict[str, Any]:
    summary_path = root / "results/summary/phase_r1_formal_summary.json"
    if not summary_path.is_file():
        raise PhaseR1StatisticsError("FORMAL_SUMMARY_MISSING")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    binding_sha = str(summary["start_binding_sha256"])
    raw_path = root / summary["raw_path"]
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != summary["raw_sha256"]:
        raise PhaseR1StatisticsError("RAW_HASH_MISMATCH")
    records = _read_jsonl(raw_path)
    if any(r.get("start_binding_sha256") != binding_sha for r in records):
        raise PhaseR1StatisticsError("RAW_BINDING_MISMATCH")

    by_key = {(r["pairing_id"], r["method"]): r for r in records}
    pairings = sorted({r["pairing_id"] for r in records})
    contrasts: list[dict[str, Any]] = []

    comparison_specs = [
        ("E1", "lir_pptd_full", "crh", "external_primary"),
        ("E1", "lir_pptd_full", "mean_vote", "external_secondary"),
        ("E2", "lir_pptd_full", "crh", "external_secondary_on_off"),
        ("E2", "lir_pptd_full", "mean_vote", "external_secondary_on_off"),
        ("E3", "lir_pptd_full", "lir_pptd_nr", "ablation_secondary"),
        ("E3", "lir_pptd_full", "lir_pptd_ho", "ablation_secondary"),
        ("E4", "lir_pptd_full", "lir_pptd_nr", "order_sensitivity_secondary"),
        ("E4", "lir_pptd_full", "lir_pptd_ho", "order_sensitivity_secondary"),
    ]

    for stage, full_method, comparator, analysis_role in comparison_specs:
        relevant_pairings = [p for p in pairings if f"stage={stage}|" in p]
        groups: dict[tuple[str, str, str, str], list[str]] = {}
        for pairing in relevant_pairings:
            full = by_key.get((pairing, full_method))
            comp = by_key.get((pairing, comparator))
            sample = full or comp
            if not sample:
                continue
            key = (sample["dataset_id"], sample["attack_condition"], sample["rho"], str(sample.get("task_order_seed")))
            groups.setdefault(key, []).append(pairing)

        for (dataset, attack_condition, rho, order_seed_text), members in sorted(groups.items()):
            # E4 pools five task-order seeds into one contrast rather than one contrast per seed.
            if stage == "E4":
                if order_seed_text != "6101":
                    # only process once by gathering all same dataset/attack/rho pairings below
                    continue
                members = [p for p in relevant_pairings if f"dataset={dataset}|" in p and f"|attack={attack_condition}|rho={rho}|" in p]
            for endpoint_field, endpoint_label in (("primary_loss", "primary"), ("secondary_loss", "secondary")):
                deltas: list[Fraction] = []
                full_losses: list[Fraction] = []
                comp_losses: list[Fraction] = []
                failed_or_missing = 0
                for pairing in sorted(set(members)):
                    full = by_key.get((pairing, full_method))
                    comp = by_key.get((pairing, comparator))
                    if not full or not comp or full.get("status") != "success" or comp.get("status") != "success":
                        failed_or_missing += 1
                        continue
                    f = _fraction(full[endpoint_field])
                    c = _fraction(comp[endpoint_field])
                    full_losses.append(f)
                    comp_losses.append(c)
                    deltas.append(f - c)
                planned = _planned_pair_count(stage, dataset, rho)
                # E4 planned n=5 irrespective of internal grouping.
                if stage == "E4":
                    planned = 5
                if len(deltas) + failed_or_missing > planned:
                    raise PhaseR1StatisticsError("PAIR_COUNT_EXCEEDS_PLAN")
                hypothesis_id = (
                    f"stage={stage}|dataset={dataset}|attack={attack_condition}|endpoint={endpoint_label}"
                    f"|comparator={comparator}|rho={rho}"
                )
                if not deltas:
                    contrast = {
                        "hypothesis_id": hypothesis_id,
                        "analysis_role": analysis_role,
                        "stage": stage,
                        "dataset_id": dataset,
                        "attack_condition": attack_condition,
                        "rho": rho,
                        "endpoint": endpoint_label,
                        "comparator": comparator,
                        "planned_n": planned,
                        "observed_n": 0,
                        "failure_or_missing_n": planned,
                        "inference_status": "no_complete_pairs",
                        "holm": None,
                    }
                    contrasts.append(contrast)
                    continue
                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                med_delta = _median_fraction(deltas)
                ci_low, ci_high, boot_seed = bootstrap_mean_ci(deltas, seed_text=binding_sha + "|" + hypothesis_id)
                p_two = exact_two_sided_sign_flip(deltas) if len(deltas) >= 2 else Fraction(1, 1)
                rb = paired_rank_biserial(deltas)
                contrast = {
                    "hypothesis_id": hypothesis_id,
                    "analysis_role": analysis_role,
                    "stage": stage,
                    "dataset_id": dataset,
                    "attack_condition": attack_condition,
                    "rho": rho,
                    "endpoint": endpoint_label,
                    "comparator": comparator,
                    "planned_n": planned,
                    "observed_n": len(deltas),
                    "failure_or_missing_n": planned - len(deltas),
                    "inference_status": "complete" if len(deltas) == planned else "degraded_by_failures_or_missing_pairs",
                    "mean_full_loss": _fraction_json(sum(full_losses, Fraction(0, 1)) / len(full_losses)),
                    "mean_comparator_loss": _fraction_json(sum(comp_losses, Fraction(0, 1)) / len(comp_losses)),
                    "mean_difference": _fraction_json(mean_delta),
                    "median_difference": _fraction_json(med_delta),
                    "bootstrap_95_ci": {"low": ci_low, "high": ci_high, "resamples": BOOTSTRAP_RESAMPLES, "seed": boot_seed},
                    "two_sided_exact_sign_flip_p": _fraction_json(p_two),
                    "paired_rank_biserial": _fraction_json(rb),
                    "holm": None,
                }
                contrasts.append(contrast)

    # Holm within stage/dataset/attack/endpoint/comparator across the rho members actually planned.
    families: dict[str, list[dict[str, Any]]] = {}
    for c in contrasts:
        if c["rho"] == "0" or c.get("observed_n", 0) == 0 or c["stage"] == "E4":
            continue
        family_id = (
            f"stage={c['stage']}|dataset={c['dataset_id']}|attack={c['attack_condition']}"
            f"|endpoint={c['endpoint']}|comparator={c['comparator']}"
        )
        families.setdefault(family_id, []).append(c)
    for family_id, items in families.items():
        raw = {c["hypothesis_id"]: Fraction(int(c["two_sided_exact_sign_flip_p"]["numerator"]), int(c["two_sided_exact_sign_flip_p"]["denominator"])) for c in items}
        adj = holm_adjust(raw)
        for c in items:
            c["holm"] = {
                "family_id": family_id,
                "family_size": len(items),
                "adjusted_p": _fraction_json(adj[c["hypothesis_id"]]),
                "alpha": float(ALPHA),
                "reject": adj[c["hypothesis_id"]] <= ALPHA,
            }

    return {
        "schema_version": "1.0",
        "phase": "Phase-R1",
        "analysis_status": "post_phase11_extension",
        "start_binding_sha256": binding_sha,
        "formal_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "raw_sha256": summary["raw_sha256"],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "test_direction": "two_sided",
        "delta_definition": "loss(lir_pptd_full)-loss(comparator)",
        "contrasts": contrasts,
        "holm_family_count": len(families),
        "reporting_rule": "failures/missing pairs retained in counts; non-supporting outcomes are not automatically baseline wins",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-R1 frozen statistical analyzer")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="results/summary/phase_r1_statistical_analysis.json")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    result = analyze(root)
    out = root / args.output
    if out.exists():
        raise PhaseR1StatisticsError(f"REFUSING_TO_OVERWRITE:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("R1_STATISTICAL_ANALYSIS=PASS")
    print(f"CONTRASTS={len(result['contrasts'])}")
    print(f"HOLM_FAMILIES={result['holm_family_count']}")
    print(f"OUTPUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
