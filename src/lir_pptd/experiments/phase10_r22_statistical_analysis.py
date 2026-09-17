from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

REVISION = "r2.2"

STATISTICAL_PLAN_RELATIVE_PATH = Path("docs/PHASE10_R22_STATISTICAL_ANALYSIS_PLAN.json")
STATISTICAL_PLAN_SHA256 = "826215145b1026e26c168d059a176a12945d456c58235168c45492f54936d88d"

FORMAL_RESULT_FREEZE_RELATIVE_PATH = Path("docs/PHASE10_R22_FORMAL_RESULT_FREEZE.json")
FORMAL_RESULT_FREEZE_SHA256 = "c7f3a27134ce3e3a871fa35abe179d0afba80198a4069ba2d5668c57b240b2a8"

SUMMARY_RELATIVE_PATH = Path("results/summary/phase10_r22_corrective.json")
SUMMARY_SHA256 = "ef178e8e61c968264958f2ed4b73bcc22ab1ea3cfd613faa1bd67b673c7a7a1d"

RAW_RELATIVE_PATH = Path("results/raw/phase10_r22_corrective")
RAW_MANIFEST_SHA256 = "d4954d310ce0a82df11dd47bc4f8c4f77c3db79dd5099069d861febe7945eaff"

OUTPUT_RELATIVE_PATH = Path("results/summary/phase10_r22_statistical_analysis.json")

PHASE9_PREREG_SHA256 = "3201f372b527620503fe9cdc9ef8ae29bcd822d618e7f74773d83054d7ad0645"

FINAL_SEEDS = tuple(range(3001, 3031))
FAMILIES = (
    "synthetic_num_small",
    "synthetic_num_shifted",
    "synthetic_num_longitudinal",
    "synthetic_cat_binary",
    "synthetic_cat_tie",
)
STARTS = ("cold", "warm")
RHO_GRID = ("0", "1/10", "3/10", "1/2", "7/10", "9/10")
PRIMARY_RHOS = ("1/10", "3/10", "1/2", "7/10", "9/10")
ATTACK_CONDITIONS = ("fixed_target", "on_off_5_1", "on_off_5_5", "on_off_1_5")
PRIMARY_COMPARATOR = "mean_vote"
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = Fraction(1, 20)

ENDPOINTS = {
    "synthetic_num_small": "mae",
    "synthetic_num_shifted": "mae",
    "synthetic_num_longitudinal": "mae",
    "synthetic_cat_binary": "1-Accuracy",
    "synthetic_cat_tie": "1-MacroF1",
}


class StatisticalAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedContrast:
    seed: int
    full_loss: Fraction
    mean_vote_loss: Fraction
    delta: Fraction


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise StatisticalAnalysisError(f"BOUND_FILE_MISSING:{path.as_posix()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction_from_decimal_text(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _fraction_json(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal": float(value),
    }


def _attack_condition_id(attack_class: str, schedule: Sequence[int] | None) -> str:
    if attack_class == "fixed_target":
        if schedule is not None:
            raise StatisticalAnalysisError("FIXED_TARGET_SCHEDULE_INVALID")
        return "fixed_target"
    if attack_class == "on_off":
        if schedule is None or len(schedule) != 2:
            raise StatisticalAnalysisError("ON_OFF_SCHEDULE_INVALID")
        first, second = int(schedule[0]), int(schedule[1])
        condition_id = f"on_off_{first}_{second}"
        if condition_id not in ATTACK_CONDITIONS:
            raise StatisticalAnalysisError("UNKNOWN_ATTACK_CONDITION")
        return condition_id
    raise StatisticalAnalysisError("UNKNOWN_ATTACK_CLASS")


def _one_minus_accuracy_fraction(predictions: Sequence[int], truths: Sequence[int]) -> Fraction:
    if len(predictions) != len(truths) or not truths:
        raise StatisticalAnalysisError("CATEGORICAL_EVIDENCE_INVALID")
    errors = sum(int(predicted != truth) for predicted, truth in zip(predictions, truths))
    return Fraction(errors, len(truths))


def _one_minus_macro_f1_fraction(predictions: Sequence[int], truths: Sequence[int]) -> Fraction:
    if len(predictions) != len(truths) or not truths:
        raise StatisticalAnalysisError("CATEGORICAL_EVIDENCE_INVALID")
    supported = sorted(set(truths))
    f1_values: list[Fraction] = []
    for class_index in supported:
        tp = sum(
            predicted == class_index and truth == class_index
            for predicted, truth in zip(predictions, truths)
        )
        fp = sum(
            predicted == class_index and truth != class_index
            for predicted, truth in zip(predictions, truths)
        )
        fn = sum(
            predicted != class_index and truth == class_index
            for predicted, truth in zip(predictions, truths)
        )
        denominator = 2 * tp + fp + fn
        if denominator <= 0:
            raise StatisticalAnalysisError("POSITIVE_SUPPORT_F1_DENOMINATOR_INVALID")
        f1_values.append(Fraction(2 * tp, denominator))
    macro_f1 = sum(f1_values, start=Fraction(0, 1)) / len(f1_values)
    return Fraction(1, 1) - macro_f1


def recompute_condition_losses(condition: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    evidence = condition.get("task_evidence")
    family = str(condition.get("family"))

    if not isinstance(evidence, list) or len(evidence) != 100:
        raise StatisticalAnalysisError("TASK_EVIDENCE_COUNT_MISMATCH")

    if family.startswith("synthetic_num"):
        full_numerator = Fraction(0, 1)
        mean_numerator = Fraction(0, 1)
        denominator = 0
        for item in evidence:
            dimension = int(item["dimension"])
            if dimension <= 0:
                raise StatisticalAnalysisError("NUMERICAL_DIMENSION_INVALID")
            full_numerator += _fraction_from_decimal_text(str(item["full_abs_error_sum"]))
            mean_numerator += _fraction_from_decimal_text(str(item["mean_vote_abs_error_sum"]))
            denominator += dimension
        if denominator <= 0:
            raise StatisticalAnalysisError("NUMERICAL_DENOMINATOR_INVALID")
        return full_numerator / denominator, mean_numerator / denominator

    truths = [int(item["truth_class"]) for item in evidence]
    full_predictions = [int(item["full_prediction"]) for item in evidence]
    mean_predictions = [int(item["mean_vote_prediction"]) for item in evidence]

    if family == "synthetic_cat_binary":
        return (
            _one_minus_accuracy_fraction(full_predictions, truths),
            _one_minus_accuracy_fraction(mean_predictions, truths),
        )
    if family == "synthetic_cat_tie":
        return (
            _one_minus_macro_f1_fraction(full_predictions, truths),
            _one_minus_macro_f1_fraction(mean_predictions, truths),
        )
    raise StatisticalAnalysisError("UNKNOWN_FAMILY")


def _assert_stored_loss_matches(
    recomputed: Fraction,
    stored: Any,
    label: str,
    *,
    abs_tol: float = 1e-12,
) -> None:
    if stored is None:
        raise StatisticalAnalysisError(f"{label}_MISSING")
    if not math.isclose(float(recomputed), float(stored), rel_tol=0.0, abs_tol=abs_tol):
        raise StatisticalAnalysisError(f"{label}_RECOMPUTE_MISMATCH")


def canonical_hypothesis_id(
    *,
    family: str,
    attack_condition: str,
    start: str,
    endpoint: str,
    comparator: str,
    rho: str,
) -> str:
    return (
        f"dataset={family}"
        f"|attack_condition={attack_condition}"
        f"|start={start}"
        f"|endpoint={endpoint}"
        f"|comparator={comparator}"
        f"|rho={rho}"
    )


def bootstrap_seed(hypothesis_id: str) -> int:
    payload = (
        "phase10-r22-bootstrap-v1|"
        + PHASE9_PREREG_SHA256
        + "|"
        + hypothesis_id
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def paired_bootstrap_mean_ci(
    deltas: Sequence[Fraction],
    *,
    hypothesis_id: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, int]:
    if not deltas:
        raise StatisticalAnalysisError("BOOTSTRAP_EMPTY_SAMPLE")
    values = tuple(float(value) for value in deltas)
    n = len(values)
    seed = bootstrap_seed(hypothesis_id)
    rng = random.Random(seed)
    means: list[float] = []
    append = means.append
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        append(total / n)
    means.sort()

    # Deterministic linear interpolation matching the usual percentile
    # quantile definition on the sorted bootstrap sample.
    def quantile(q: float) -> float:
        position = (len(means) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return means[lower]
        fraction = position - lower
        return means[lower] * (1.0 - fraction) + means[upper] * fraction

    return quantile(0.025), quantile(0.975), seed


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def _scaled_integer_deltas(deltas: Sequence[Fraction]) -> tuple[list[int], int]:
    if not deltas:
        raise StatisticalAnalysisError("SIGN_FLIP_EMPTY_SAMPLE")
    scale = reduce(_lcm, (value.denominator for value in deltas), 1)
    scaled = [
        value.numerator * (scale // value.denominator)
        for value in deltas
    ]
    return scaled, scale


def _subset_sums(values: Sequence[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums += [current + value for current in sums]
    return sums


def exact_paired_sign_flip(
    deltas: Sequence[Fraction],
) -> tuple[Fraction, Fraction]:
    """Exact paired sign-flip test using meet-in-the-middle counting.

    Primary one-sided p-value is P(T <= T_obs) for alternative mean(delta) < 0.
    Secondary two-sided p-value is P(|T| >= |T_obs|).
    """
    scaled, _ = _scaled_integer_deltas(deltas)
    magnitudes = [abs(value) for value in scaled]
    observed = sum(scaled)

    split = len(magnitudes) // 2
    left_weights = magnitudes[:split]
    right_weights = magnitudes[split:]

    left_total = sum(left_weights)
    right_total = sum(right_weights)

    # A sign assignment sum is 2*subset_sum - total.
    left = [2 * value - left_total for value in _subset_sums(left_weights)]
    right = [2 * value - right_total for value in _subset_sums(right_weights)]
    right.sort()

    total_assignments = 1 << len(magnitudes)

    less_count = 0
    for left_value in left:
        threshold = observed - left_value
        less_count += bisect.bisect_right(right, threshold)

    p_less = Fraction(less_count, total_assignments)

    abs_observed = abs(observed)
    if abs_observed == 0:
        p_two = Fraction(1, 1)
    else:
        two_count = 0
        for left_value in left:
            low_threshold = -abs_observed - left_value
            high_threshold = abs_observed - left_value
            low_count = bisect.bisect_right(right, low_threshold)
            high_count = len(right) - bisect.bisect_left(right, high_threshold)
            two_count += low_count + high_count
        p_two = Fraction(two_count, total_assignments)

    return p_less, min(Fraction(1, 1), p_two)


def paired_rank_biserial(deltas: Sequence[Fraction]) -> Fraction:
    nonzero = [(abs(value), 1 if value > 0 else -1) for value in deltas if value != 0]
    if not nonzero:
        return Fraction(0, 1)

    nonzero.sort(key=lambda item: item[0])
    w_plus = Fraction(0, 1)
    w_minus = Fraction(0, 1)

    i = 0
    while i < len(nonzero):
        j = i + 1
        while j < len(nonzero) and nonzero[j][0] == nonzero[i][0]:
            j += 1
        # ranks are one-based positions i+1 .. j
        average_rank = Fraction((i + 1) + j, 2)
        for _, sign in nonzero[i:j]:
            if sign > 0:
                w_plus += average_rank
            else:
                w_minus += average_rank
        i = j

    denominator = w_plus + w_minus
    if denominator == 0:
        return Fraction(0, 1)
    return (w_plus - w_minus) / denominator


def holm_adjust(p_values: Mapping[str, Fraction]) -> dict[str, Fraction]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted_sorted: list[tuple[str, Fraction]] = []
    running = Fraction(0, 1)
    for index, (hypothesis_id, p_value) in enumerate(ordered):
        factor = m - index
        candidate = min(Fraction(1, 1), p_value * factor)
        running = max(running, candidate)
        adjusted_sorted.append((hypothesis_id, running))
    return dict(adjusted_sorted)


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise StatisticalAnalysisError("MEDIAN_EMPTY_SAMPLE")
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_frozen_inputs(root: Path) -> dict[str, Any]:
    plan_path = root / STATISTICAL_PLAN_RELATIVE_PATH
    freeze_path = root / FORMAL_RESULT_FREEZE_RELATIVE_PATH
    summary_path = root / SUMMARY_RELATIVE_PATH

    if _sha256(plan_path) != STATISTICAL_PLAN_SHA256:
        raise StatisticalAnalysisError("STATISTICAL_PLAN_SHA_MISMATCH")
    if _sha256(freeze_path) != FORMAL_RESULT_FREEZE_SHA256:
        raise StatisticalAnalysisError("FORMAL_RESULT_FREEZE_SHA_MISMATCH")
    if _sha256(summary_path) != SUMMARY_SHA256:
        raise StatisticalAnalysisError("R22_SUMMARY_SHA_MISMATCH")

    plan = _load_json(plan_path)
    freeze = _load_json(freeze_path)
    summary = _load_json(summary_path)

    if plan.get("statistics_started") is not False:
        raise StatisticalAnalysisError("STATISTICAL_PLAN_STATE_INVALID")
    if plan.get("scope", {}).get("pairing_unit") != "seed":
        raise StatisticalAnalysisError("PAIRING_UNIT_MISMATCH")
    if int(plan.get("scope", {}).get("paired_seed_count", 0)) != 30:
        raise StatisticalAnalysisError("PAIRED_SEED_COUNT_MISMATCH")
    if plan.get("scope", {}).get("primary_comparator") != PRIMARY_COMPARATOR:
        raise StatisticalAnalysisError("PRIMARY_COMPARATOR_MISMATCH")
    if tuple(plan.get("rho_policy", {}).get("primary_superiority_ratios", ())) != PRIMARY_RHOS:
        raise StatisticalAnalysisError("PRIMARY_RHO_PLAN_MISMATCH")

    if freeze.get("status") != "frozen":
        raise StatisticalAnalysisError("FORMAL_RESULT_FREEZE_STATE_INVALID")
    if freeze.get("raw", {}).get("manifest_sha256") != RAW_MANIFEST_SHA256:
        raise StatisticalAnalysisError("RAW_MANIFEST_BINDING_MISMATCH")
    if freeze.get("summary", {}).get("sha256") != SUMMARY_SHA256:
        raise StatisticalAnalysisError("SUMMARY_FREEZE_BINDING_MISMATCH")

    if summary.get("revision") != REVISION:
        raise StatisticalAnalysisError("SUMMARY_REVISION_MISMATCH")
    if summary.get("status") != "completed":
        raise StatisticalAnalysisError("SUMMARY_NOT_COMPLETED")
    if int(summary.get("denominator", 0)) != 7200:
        raise StatisticalAnalysisError("SUMMARY_DENOMINATOR_MISMATCH")
    if summary.get("status_counts") != {"success": 7200, "failure": 0, "timeout": 0}:
        raise StatisticalAnalysisError("SUMMARY_STATUS_COUNTS_MISMATCH")

    checkpoints = freeze.get("raw", {}).get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 30:
        raise StatisticalAnalysisError("CHECKPOINT_MANIFEST_INVALID")

    manifest_lines: list[str] = []
    for item in checkpoints:
        path = root / Path(str(item["path"]))
        actual_sha = _sha256(path)
        expected_sha = str(item["sha256"])
        if actual_sha != expected_sha:
            raise StatisticalAnalysisError(f"RAW_CHECKPOINT_SHA_MISMATCH:{item['seed']}")
        size = path.stat().st_size
        if size != int(item["bytes"]):
            raise StatisticalAnalysisError(f"RAW_CHECKPOINT_SIZE_MISMATCH:{item['seed']}")
        manifest_lines.append(
            f"{int(item['seed'])}|{path.relative_to(root).as_posix()}|{size}|{actual_sha}"
        )

    manifest_sha = hashlib.sha256(
        ("\n".join(manifest_lines) + "\n").encode("utf-8")
    ).hexdigest()
    if manifest_sha != RAW_MANIFEST_SHA256:
        raise StatisticalAnalysisError("RAW_MANIFEST_RECOMPUTE_MISMATCH")

    return {"plan": plan, "freeze": freeze, "summary": summary}


def load_seed_contrasts(root: Path, freeze: Mapping[str, Any]) -> dict[tuple[str, str, str, str], list[SeedContrast]]:
    groups: dict[tuple[str, str, str, str], list[SeedContrast]] = {}

    for checkpoint_meta in freeze["raw"]["checkpoints"]:
        seed = int(checkpoint_meta["seed"])
        if seed not in FINAL_SEEDS:
            raise StatisticalAnalysisError("UNEXPECTED_FINAL_SEED")
        payload = _load_json(root / Path(str(checkpoint_meta["path"])))
        conditions = payload.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 240:
            raise StatisticalAnalysisError(f"CHECKPOINT_CONDITION_COUNT_MISMATCH:{seed}")

        for condition in conditions:
            if condition.get("status") != "success":
                raise StatisticalAnalysisError("NON_SUCCESS_CONDITION_IN_FROZEN_COMPLETED_RESULT")
            family = str(condition["family"])
            start = str(condition["start"])
            rho = str(condition["rho"])
            attack_condition = _attack_condition_id(
                str(condition["attack_class"]),
                condition.get("schedule"),
            )
            if family not in FAMILIES or start not in STARTS or rho not in RHO_GRID:
                raise StatisticalAnalysisError("CONDITION_DIMENSION_OUT_OF_SCOPE")

            full_loss, mean_loss = recompute_condition_losses(condition)
            _assert_stored_loss_matches(full_loss, condition.get("full_primary_loss"), "FULL_PRIMARY_LOSS")
            _assert_stored_loss_matches(mean_loss, condition.get("mean_vote_primary_loss"), "MEAN_VOTE_PRIMARY_LOSS")

            stored_delta = condition.get("delta")
            if stored_delta is None or not math.isclose(
                float(full_loss - mean_loss),
                float(stored_delta),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise StatisticalAnalysisError("DELTA_RECOMPUTE_MISMATCH")

            key = (family, attack_condition, start, rho)
            groups.setdefault(key, []).append(
                SeedContrast(
                    seed=seed,
                    full_loss=full_loss,
                    mean_vote_loss=mean_loss,
                    delta=full_loss - mean_loss,
                )
            )

    expected_group_count = len(FAMILIES) * len(ATTACK_CONDITIONS) * len(STARTS) * len(RHO_GRID)
    if len(groups) != expected_group_count:
        raise StatisticalAnalysisError("CONTRAST_GROUP_COUNT_MISMATCH")

    for key, items in groups.items():
        items.sort(key=lambda item: item.seed)
        if tuple(item.seed for item in items) != FINAL_SEEDS:
            raise StatisticalAnalysisError(f"PAIRING_SEED_SET_MISMATCH:{key}")

    return groups


def analyze_contrast(
    *,
    family: str,
    attack_condition: str,
    start: str,
    rho: str,
    seed_contrasts: Sequence[SeedContrast],
) -> dict[str, Any]:
    endpoint = ENDPOINTS[family]
    hypothesis_id = canonical_hypothesis_id(
        family=family,
        attack_condition=attack_condition,
        start=start,
        endpoint=endpoint,
        comparator=PRIMARY_COMPARATOR,
        rho=rho,
    )
    deltas = [item.delta for item in seed_contrasts]
    full_losses = [item.full_loss for item in seed_contrasts]
    mean_losses = [item.mean_vote_loss for item in seed_contrasts]

    mean_delta = sum(deltas, start=Fraction(0, 1)) / len(deltas)
    mean_full = sum(full_losses, start=Fraction(0, 1)) / len(full_losses)
    mean_mean_vote = sum(mean_losses, start=Fraction(0, 1)) / len(mean_losses)
    median_delta = _median_fraction(deltas)

    ci_low, ci_high, rng_seed = paired_bootstrap_mean_ci(
        deltas,
        hypothesis_id=hypothesis_id,
    )
    p_less, p_two = exact_paired_sign_flip(deltas)
    rank_biserial = paired_rank_biserial(deltas)

    direction = (
        "lir_pptd_full_better"
        if mean_delta < 0
        else "mean_vote_better"
        if mean_delta > 0
        else "tie"
    )

    return {
        "hypothesis_id": hypothesis_id,
        "family": family,
        "endpoint": endpoint,
        "attack_condition": attack_condition,
        "start": start,
        "rho": rho,
        "primary": rho in PRIMARY_RHOS,
        "comparator": PRIMARY_COMPARATOR,
        "n": len(seed_contrasts),
        "seed_order": [item.seed for item in seed_contrasts],
        "mean_full_loss": _fraction_json(mean_full),
        "mean_mean_vote_loss": _fraction_json(mean_mean_vote),
        "mean_difference": _fraction_json(mean_delta),
        "median_difference": _fraction_json(median_delta),
        "direction": direction,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": str(rng_seed),
            "percentile_95_ci": [ci_low, ci_high],
        },
        "randomization": {
            "one_sided_less_exact_p": _fraction_json(p_less),
            "two_sided_absolute_exact_p": _fraction_json(p_two),
        },
        "paired_rank_biserial": _fraction_json(rank_biserial),
        "seed_level": [
            {
                "seed": item.seed,
                "full_loss": _fraction_json(item.full_loss),
                "mean_vote_loss": _fraction_json(item.mean_vote_loss),
                "delta": _fraction_json(item.delta),
            }
            for item in seed_contrasts
        ],
    }


def apply_holm_families(contrasts: list[dict[str, Any]]) -> None:
    primary = [item for item in contrasts if item["primary"]]
    families: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for item in primary:
        key = (
            item["family"],
            item["attack_condition"],
            item["start"],
            item["endpoint"],
            item["comparator"],
        )
        families.setdefault(key, []).append(item)

    if len(families) != 40:
        raise StatisticalAnalysisError("HOLM_FAMILY_COUNT_MISMATCH")

    for key, items in families.items():
        if len(items) != 5 or {item["rho"] for item in items} != set(PRIMARY_RHOS):
            raise StatisticalAnalysisError(f"HOLM_FAMILY_MEMBERSHIP_MISMATCH:{key}")

        raw = {
            item["hypothesis_id"]: Fraction(
                int(item["randomization"]["one_sided_less_exact_p"]["numerator"]),
                int(item["randomization"]["one_sided_less_exact_p"]["denominator"]),
            )
            for item in items
        }
        adjusted = holm_adjust(raw)
        for item in items:
            p_adj = adjusted[item["hypothesis_id"]]
            item["holm"] = {
                "family_id": (
                    f"dataset={item['family']}"
                    f"|attack_condition={item['attack_condition']}"
                    f"|start={item['start']}"
                    f"|endpoint={item['endpoint']}"
                    f"|comparator={item['comparator']}"
                ),
                "family_size": 5,
                "adjusted_one_sided_p": _fraction_json(p_adj),
                "alpha": float(ALPHA),
                "reject": p_adj <= ALPHA,
            }
            mean_delta = Fraction(
                int(item["mean_difference"]["numerator"]),
                int(item["mean_difference"]["denominator"]),
            )
            if p_adj <= ALPHA and mean_delta < 0:
                category = "supporting"
                superiority_supported = True
            elif mean_delta < 0:
                category = "unresolved_or_underpowered"
                superiority_supported = False
            else:
                category = "baseline_win_or_no_difference"
                superiority_supported = False
            item["interpretation"] = {
                "category": category,
                "primary_superiority_supported": superiority_supported,
            }

    for item in contrasts:
        if not item["primary"]:
            item["holm"] = None
            item["interpretation"] = {
                "category": "control_descriptive",
                "primary_superiority_supported": None,
            }


def analyze(root: Path) -> dict[str, Any]:
    if (root / OUTPUT_RELATIVE_PATH).exists():
        raise StatisticalAnalysisError("STATISTICAL_ANALYSIS_OUTPUT_ALREADY_EXISTS")

    frozen = verify_frozen_inputs(root)
    groups = load_seed_contrasts(root, frozen["freeze"])

    contrasts: list[dict[str, Any]] = []
    for family in FAMILIES:
        for attack_condition in ATTACK_CONDITIONS:
            for start in STARTS:
                for rho in RHO_GRID:
                    key = (family, attack_condition, start, rho)
                    contrasts.append(
                        analyze_contrast(
                            family=family,
                            attack_condition=attack_condition,
                            start=start,
                            rho=rho,
                            seed_contrasts=groups[key],
                        )
                    )

    if len(contrasts) != 240:
        raise StatisticalAnalysisError("TOTAL_CONTRAST_COUNT_MISMATCH")
    if sum(item["primary"] for item in contrasts) != 200:
        raise StatisticalAnalysisError("PRIMARY_CONTRAST_COUNT_MISMATCH")

    apply_holm_families(contrasts)

    primary = [item for item in contrasts if item["primary"]]
    category_counts: dict[str, int] = {}
    for item in primary:
        category = item["interpretation"]["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    output = {
        "schema_version": "phase10-r22-statistical-analysis-v1",
        "phase": "Phase10",
        "revision": REVISION,
        "status": "completed",
        "analysis_type": "post-unblinding_corrective_statistical_analysis",
        "frozen_bindings": {
            "statistical_plan": {
                "path": STATISTICAL_PLAN_RELATIVE_PATH.as_posix(),
                "sha256": STATISTICAL_PLAN_SHA256,
            },
            "formal_result_freeze": {
                "path": FORMAL_RESULT_FREEZE_RELATIVE_PATH.as_posix(),
                "sha256": FORMAL_RESULT_FREEZE_SHA256,
            },
            "summary": {
                "path": SUMMARY_RELATIVE_PATH.as_posix(),
                "sha256": SUMMARY_SHA256,
            },
            "raw_manifest_sha256": RAW_MANIFEST_SHA256,
            "phase9_preregistration_sha256": PHASE9_PREREG_SHA256,
        },
        "contract": {
            "pairing_unit": "seed",
            "paired_seed_count": 30,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "primary_test": "exact_paired_sign_flip_less",
            "secondary_test": "exact_paired_sign_flip_absolute_two_sided",
            "holm_family_count": 40,
            "hypotheses_per_holm_family": 5,
            "primary_contrast_count": 200,
            "rho_zero_primary": False,
            "alpha": float(ALPHA),
        },
        "counts": {
            "total_contrasts": len(contrasts),
            "primary_contrasts": len(primary),
            "control_contrasts": len(contrasts) - len(primary),
            "primary_interpretation_categories": category_counts,
            "primary_superiority_supported": sum(
                item["interpretation"]["primary_superiority_supported"] is True
                for item in primary
            ),
        },
        "contrasts": contrasts,
    }

    output_path = root / OUTPUT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    tmp.replace(output_path)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase10 r2.2 frozen-result statistical analysis"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.analyze:
        raise StatisticalAnalysisError("EXPLICIT_ANALYZE_FLAG_REQUIRED")
    output = analyze(args.root)
    print(f"STATUS={output['status']}")
    print(f"TOTAL_CONTRASTS={output['counts']['total_contrasts']}")
    print(f"PRIMARY_CONTRASTS={output['counts']['primary_contrasts']}")
    print(f"CONTROL_CONTRASTS={output['counts']['control_contrasts']}")
    print(
        "PRIMARY_SUPERIORITY_SUPPORTED="
        f"{output['counts']['primary_superiority_supported']}"
    )
    print(f"OUTPUT={OUTPUT_RELATIVE_PATH.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
