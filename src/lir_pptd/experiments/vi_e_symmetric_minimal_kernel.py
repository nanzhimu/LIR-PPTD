"""VI-E.1 symmetric minimal-kernel benchmark gate.

Primary lightweight evidence remains the source-normalized analytical comparison.
This gate supplies optional *secondary* timing evidence only after two independent
functional conformance checks pass:

* minimal LIR kernel == frozen deterministic fixed-point LIR functionality;
* minimal Fog kernel == admitted Fog-PPTD semantic adapter functionality.

Both timed methods use the exact same ``MinimalShamirKernel`` implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from lir_pptd.baselines.fog_pptd_secure_adapter import (
    SOURCE_A1,
    SOURCE_A2,
    FogAdapterProfile,
    FogPPTDSemanticAdapter,
)
from lir_pptd.benchmarks.minimal_shamir_kernel import (
    COMMON_FIELD_P,
    KernelCounters,
    MinimalShamirKernel,
    ShareBundle,
)
from lir_pptd.core import ExactTaskInput
from lir_pptd.fixedpoint import default_profile, run_fixed
from lir_pptd.fixedpoint.types import FixedTaskFailure

N = 10
T = 4
K = 10
D = 1
F = 24
DELTA = 1 << F
THETA_ROUND = 1_000_000
WORKER_GRID = (20, 50, 100, 200)
WARMUP_PAIRS = 5
MEASURED_PAIRS = 30
BOOTSTRAP_RESAMPLES = 10_000

# Frozen admission rule for timing as secondary paper evidence.
MIN_PAIR_WINS_PER_STRATUM = 20  # of 30
MIN_CI_POSITIVE_STRATA = 3      # of 4


class SymmetricKernelGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedLIR:
    reports: tuple[ShareBundle, ...]
    reputations: tuple[ShareBundle, ...]
    epsilon: ShareBundle
    tau: ShareBundle
    beta: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedFog:
    reports: tuple[ShareBundle, ...]
    m_constant: ShareBundle
    a1_constant: ShareBundle
    a2_constant: ShareBundle


def conceptual_reports_million(m: int) -> tuple[int, ...]:
    if m <= 1:
        raise ValueError("m must exceed one")
    vals = tuple(200_000 + ((37 * i + 13) % 601) * 1_000 for i in range(m))
    if min(vals) < 0 or max(vals) > THETA_ROUND or len(set(vals)) < 2:
        raise AssertionError("invalid workload")
    return vals


def _scaled_fraction(x: Fraction) -> int:
    return (x.numerator * DELTA) // x.denominator


def conceptual_to_lir_fixed(v: int) -> int:
    return (int(v) * DELTA) // THETA_ROUND


def build_lir_reference_task(m: int, *, task_id: str) -> ExactTaskInput:
    vals = conceptual_reports_million(m)
    ids = tuple(f"w{i:04d}" for i in range(m))
    return ExactTaskInput(
        task_id=task_id,
        task_kind="numerical",
        K=K,
        tau=Fraction(1, 5),
        epsilon_c=Fraction(1, 1024),
        kappa=2,
        eta=Fraction(1, 10),
        reports={w: (Fraction(v, THETA_ROUND),) for w, v in zip(ids, vals)},
        reputations={w: Fraction(1, 2) for w in ids},
        epochs={w: 0 for w in ids},
        participant_ids=ids,
        report_lower=0,
        report_upper=1,
    )


def prepare_lir(kernel: MinimalShamirKernel, m: int) -> PreparedLIR:
    reports_i = tuple(conceptual_to_lir_fixed(v) for v in conceptual_reports_million(m))
    rep_i = DELTA // 2
    eps_i = _scaled_fraction(Fraction(1, 1024))
    tau_i = _scaled_fraction(Fraction(1, 5))
    beta = (
        _scaled_fraction(Fraction(4, 5)),
        _scaled_fraction(Fraction(1, 10)),
        _scaled_fraction(Fraction(1, 10)),
    )
    return PreparedLIR(
        reports=tuple(kernel.share_secret(x) for x in reports_i),
        reputations=tuple(kernel.share_secret(rep_i) for _ in range(m)),
        epsilon=kernel.share_secret(eps_i),
        tau=kernel.share_secret(tau_i),
        beta=beta,
    )


def run_lir_minimal(
    kernel: MinimalShamirKernel,
    prepared: PreparedLIR,
    *,
    capture_trace: bool = False,
) -> dict[str, Any]:
    reports = prepared.reports
    reps = prepared.reputations
    m = len(reports)
    effective = tuple(kernel.local_add(c, prepared.epsilon) for c in reps)
    den = effective[0]
    for x in effective[1:]:
        den = kernel.local_add(den, x)
    num = kernel.secure_mul(effective[0], reports[0], scale=DELTA)
    for i in range(1, m):
        num = kernel.local_add(num, kernel.secure_mul(effective[i], reports[i], scale=DELTA))
    truth = kernel.secure_div(num, den, scale=DELTA)
    trajectory: list[int] | None = [kernel.reconstruct_signed(truth)] if capture_trace else None

    for _ in range(K):
        influences: list[ShareBundle] = []
        for i in range(m):
            diff = kernel.local_sub(reports[i], truth)
            distance = kernel.secure_sqr(diff, scale=DELTA)
            denom_q = kernel.local_add(prepared.tau, distance)
            q = kernel.secure_div(prepared.tau, denom_q, scale=DELTA)
            influences.append(kernel.secure_mul(effective[i], q, scale=DELTA))
        aden = influences[0]
        for x in influences[1:]:
            aden = kernel.local_add(aden, x)
        num = kernel.secure_mul(influences[0], reports[0], scale=DELTA)
        for i in range(1, m):
            num = kernel.local_add(num, kernel.secure_mul(influences[i], reports[i], scale=DELTA))
        truth = kernel.secure_div(num, aden, scale=DELTA)
        if trajectory is not None:
            trajectory.append(kernel.reconstruct_signed(truth))

    candidates: list[int] | None = [] if capture_trace else None
    checksum = 0
    b0, b1, b2 = prepared.beta
    for i in range(m):
        diff = kernel.local_sub(reports[i], truth)
        distance = kernel.secure_sqr(diff, scale=DELTA)
        denom_q = kernel.local_add(prepared.tau, distance)
        q = kernel.secure_div(prepared.tau, denom_q, scale=DELTA)
        cq = kernel.secure_mul(reps[i], q, scale=DELTA)
        t0 = kernel.public_mul(reps[i], b0, scale=DELTA)
        t1 = kernel.public_mul(q, b1, scale=DELTA)
        t2 = kernel.public_mul(cq, b2, scale=DELTA)
        candidate = kernel.local_add(kernel.local_add(t0, t1), t2)
        if candidates is not None:
            candidates.append(kernel.reconstruct_signed(candidate))
        else:
            # One inexpensive plaintext checksum prevents accidental dead-code
            # removal if this module is translated by a future runner.
            checksum ^= candidate[0]

    final_truth = kernel.reconstruct_signed(truth)
    return {
        "final_truth_integer": final_truth,
        "truth_trajectory": tuple(trajectory) if trajectory is not None else None,
        "candidate_reputations": tuple(candidates) if candidates is not None else None,
        "checksum": checksum,
    }


def prepare_fog(kernel: MinimalShamirKernel, m: int) -> PreparedFog:
    reports = conceptual_reports_million(m)
    return PreparedFog(
        reports=tuple(kernel.share_secret(x) for x in reports),
        m_constant=kernel.share_secret(m),
        a1_constant=kernel.share_secret(SOURCE_A1),
        a2_constant=kernel.share_secret(SOURCE_A2),
    )


def run_fog_minimal(
    kernel: MinimalShamirKernel,
    prepared: PreparedFog,
    *,
    capture_trace: bool = False,
) -> dict[str, Any]:
    reports = prepared.reports
    m = len(reports)
    total = reports[0]
    for x in reports[1:]:
        total = kernel.local_add(total, x)
    truth = kernel.secure_div(total, prepared.m_constant, scale=1)
    trajectory: list[int] | None = [kernel.reconstruct_signed(truth)] if capture_trace else None

    for _ in range(K):
        distances: list[ShareBundle] = []
        total_distance: ShareBundle | None = None
        for report in reports:
            diff = kernel.local_sub(report, truth)
            distance = kernel.secure_sqr(diff, scale=1)
            distances.append(distance)
            total_distance = distance if total_distance is None else kernel.local_add(total_distance, distance)
        assert total_distance is not None
        if kernel.reconstruct_signed(total_distance) <= 0:
            raise SymmetricKernelGateError("FOG_DEGENERATE_DISTANCE_SUM")

        weights: list[ShareBundle] = []
        for distance in distances:
            numerator = kernel.public_mul(distance, SOURCE_A2, scale=1)
            ratio = kernel.secure_div(numerator, total_distance, scale=1)
            diff = kernel.local_sub(ratio, prepared.a2_constant)
            sq = kernel.secure_sqr(diff, scale=1)
            weight = kernel.secure_div(sq, prepared.a1_constant, scale=1)
            weights.append(weight)

        numerator = kernel.secure_mul(weights[0], reports[0], scale=1)
        denominator = weights[0]
        for i in range(1, m):
            numerator = kernel.local_add(numerator, kernel.secure_mul(weights[i], reports[i], scale=1))
            denominator = kernel.local_add(denominator, weights[i])
        truth = kernel.secure_div(numerator, denominator, scale=1)
        if trajectory is not None:
            trajectory.append(kernel.reconstruct_signed(truth))

    return {
        "final_truth_integer": kernel.reconstruct_signed(truth),
        "truth_trajectory": tuple(trajectory) if trajectory is not None else None,
    }


def expected_lir_counts(m: int) -> dict[str, int]:
    return {
        "secure_sqr": 11 * m,
        "secure_mul": 22 * m,
        "secure_div": 11 * m + 11,
        "public_mul": 3 * m,
        "public_mul_local": 0,
        "public_mul_rescaled": 3 * m,
        "nonlinear_or_rescaled_calls": 47 * m + 11,
        "high_level_calls": 47 * m + 11,
    }


def expected_fog_counts(m: int) -> dict[str, int]:
    # Admitted adapter semantics under the common kernel:
    # init: 1 div; each iteration: 2m squares + m raw multiplications
    # + (2m+1) divisions + m public multiplications = 6m+1.
    return {
        "secure_sqr": 20 * m,
        "secure_mul": 10 * m,
        "secure_div": 20 * m + 11,
        "public_mul": 10 * m,
        "public_mul_local": 10 * m,
        "public_mul_rescaled": 0,
        "nonlinear_or_rescaled_calls": 50 * m + 11,
        "high_level_calls": 60 * m + 11,
    }


def _candidate_map_from_fixed(result: Any) -> tuple[int, ...]:
    return tuple(int(x.candidate_integer) for x in result.reputation_transitions)


def validate_lir_conformance(m: int, *, seed: int) -> dict[str, Any]:
    task = build_lir_reference_task(m, task_id=f"vi-e1-symmetric-lir-ref-m{m}")
    fixed = run_fixed(task, default_profile())
    if isinstance(fixed, FixedTaskFailure):
        raise SymmetricKernelGateError(f"LIR_FIXED_REFERENCE_FAILED:{fixed.reason_code}")
    kernel = MinimalShamirKernel(seed=seed)
    prepared = prepare_lir(kernel, m)
    kernel.reset_counters()
    minimal = run_lir_minimal(kernel, prepared, capture_trace=True)
    expected_traj = tuple(int(row[0]) for row in fixed.truth_integer_trajectory)
    expected_rep = _candidate_map_from_fixed(fixed)
    ok = (
        minimal["final_truth_integer"] == int(fixed.final_output_integer[0])
        and minimal["truth_trajectory"] == expected_traj
        and minimal["candidate_reputations"] == expected_rep
    )
    if not ok:
        raise SymmetricKernelGateError(f"LIR_MINIMAL_CONFORMANCE_MISMATCH:m={m}")
    counts = kernel.counters.as_dict()
    exp = expected_lir_counts(m)
    for key, val in exp.items():
        if counts[key] != val:
            raise SymmetricKernelGateError(f"LIR_COUNT_MISMATCH:m={m}:{key}:{counts[key]}!={val}")
    return {
        "m": m,
        "status": "PASS",
        "final_truth_integer": minimal["final_truth_integer"],
        "trajectory_equal": True,
        "reputation_candidates_equal": True,
        "counts": counts,
    }


def validate_fog_conformance(m: int, *, seed: int) -> dict[str, Any]:
    vals = conceptual_reports_million(m)
    adapter = FogPPTDSemanticAdapter(
        profile=FogAdapterProfile(
            committee_size_n=N,
            threshold_t=T,
            prime_p=COMMON_FIELD_P,
            local_proxy_field_bits=COMMON_FIELD_P.bit_length(),
            field_policy="VI-E1 symmetric common field",
        ),
        master_seed=seed,
    )
    reference = adapter.run_d1(vals, iterations=K, theta_round=THETA_ROUND)
    kernel = MinimalShamirKernel(seed=seed)
    prepared = prepare_fog(kernel, m)
    kernel.reset_counters()
    minimal = run_fog_minimal(kernel, prepared, capture_trace=True)
    if minimal["final_truth_integer"] != int(reference.truth_integer):
        raise SymmetricKernelGateError(f"FOG_MINIMAL_CONFORMANCE_MISMATCH:m={m}")
    counts = kernel.counters.as_dict()
    exp = expected_fog_counts(m)
    for key, val in exp.items():
        if counts[key] != val:
            raise SymmetricKernelGateError(f"FOG_COUNT_MISMATCH:m={m}:{key}:{counts[key]}!={val}")
    return {
        "m": m,
        "status": "PASS",
        "final_truth_integer": minimal["final_truth_integer"],
        "adapter_final_truth_integer": int(reference.truth_integer),
        "counts": counts,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_repo(root: Path) -> dict[str, Any]:
    required = {
        "fixed_algorithm": root / "src/lir_pptd/fixedpoint/algorithm.py",
        "fixed_arithmetic": root / "src/lir_pptd/fixedpoint/arithmetic.py",
        "fog_adapter": root / "src/lir_pptd/baselines/fog_pptd_secure_adapter.py",
        "minimal_kernel": root / "src/lir_pptd/benchmarks/minimal_shamir_kernel.py",
        "benchmark_module": root / "src/lir_pptd/experiments/vi_e_symmetric_minimal_kernel.py",
    }
    missing = [str(p.relative_to(root)) for p in required.values() if not p.exists()]
    if missing:
        raise SymmetricKernelGateError("MISSING_REQUIRED_FILES:" + ",".join(missing))
    profile = default_profile()
    if profile.fractional_bits_f != F:
        raise SymmetricKernelGateError("UNEXPECTED_FRACTIONAL_BITS")
    kernel = MinimalShamirKernel(seed=991001)
    kernel.assert_all_threshold_subsets_reconstruct(123456789)
    try:
        kernel.reconstruct_subset(kernel.share_secret(5), (0, 1, 2))
        raise SymmetricKernelGateError("T_MINUS_1_WAS_ACCEPTED")
    except ValueError:
        pass
    # Micro semantic checks through the shared primitives themselves.
    a = kernel.share_secret(7 * DELTA)
    b = kernel.share_secret(3 * DELTA)
    if kernel.reconstruct_signed(kernel.secure_mul(a, b, scale=DELTA)) != 21 * DELTA:
        raise SymmetricKernelGateError("COMMON_MUL_SEMANTICS_FAILED")
    if kernel.reconstruct_signed(kernel.secure_div(a, b, scale=DELTA)) != (7 * DELTA * DELTA) // (3 * DELTA):
        raise SymmetricKernelGateError("COMMON_DIV_SEMANTICS_FAILED")

    lir_cases = [validate_lir_conformance(m, seed=992000 + m) for m in (2, 20, 200)]
    fog_cases = [validate_fog_conformance(m, seed=993000 + m) for m in (20, 200)]
    return {
        "status": "PASS",
        "gate_id": "VI-E1-SYMMETRIC-MINIMAL-KERNEL-BENCHMARK-GATE-v1",
        "common_kernel": {
            "prime_bits": COMMON_FIELD_P.bit_length(),
            "N": N,
            "T": T,
            "implementation": "single MinimalShamirKernel class used by both methods",
            "nonlinear_rule": "private reconstruct -> exact integer functionality -> re-share",
            "transcript": False,
            "pydantic_or_result_scaffolding_in_timed_region": False,
            "network_or_production_mpc_claim": False,
        },
        "timing_boundary": {
            "metric_label": "symmetric minimal-kernel server-side arithmetic time",
            "included": [
                "all server-side local share additions/subtractions after prepared inputs",
                "all common-kernel secure multiplication/square/division operations",
                "all common-kernel exact public-scalar operations",
                "all nonlinear result re-sharing performed by the common kernel",
                "K=10 truth iterations",
                "LIR final output-evidence and reputation update",
                "one final truth reconstruction",
            ],
            "excluded": [
                "workload construction",
                "initial worker report sharing",
                "initial persistent reputation sharing",
                "sharing of public/frozen constants",
                "Python import/startup",
                "validation/conformance checks",
                "transcript/audit logging",
                "Pydantic/dataclass result construction beyond minimal method return",
                "file I/O and serialization",
                "network transport and production preprocessing",
            ],
        },
        "lir_conformance": lir_cases,
        "fog_conformance": fog_cases,
        "bound_hashes": {name: _sha256(path) for name, path in required.items()},
        "frozen_acceptance": {
            "mean_lir_faster_all_4_strata": True,
            "median_lir_faster_all_4_strata": True,
            "min_paired_lir_wins_each_stratum": MIN_PAIR_WINS_PER_STRATUM,
            "bootstrap_ci_strictly_favors_lir_min_strata": MIN_CI_POSITIVE_STRATA,
            "holm_significant_fog_wins_max": 0,
            "rule_frozen_before_formal_timing": True,
        },
    }


def _timed(fn, *args, **kwargs) -> tuple[int, Any]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter_ns()
    finally:
        if was_enabled:
            gc.enable()
    return t1 - t0, out


def paired_order(m_index: int, replicate: int) -> tuple[str, str]:
    return ("lir_pptd", "fog_pptd") if (m_index + replicate) % 2 == 0 else ("fog_pptd", "lir_pptd")


def _prepare_method(method: str, m: int, seed: int) -> tuple[MinimalShamirKernel, PreparedLIR | PreparedFog]:
    kernel = MinimalShamirKernel(seed=seed)
    prepared: PreparedLIR | PreparedFog
    if method == "lir_pptd":
        prepared = prepare_lir(kernel, m)
    elif method == "fog_pptd":
        prepared = prepare_fog(kernel, m)
    else:
        raise ValueError(method)
    kernel.reset_counters()
    return kernel, prepared


def _execute_prepared(method: str, kernel: MinimalShamirKernel, prepared: PreparedLIR | PreparedFog) -> dict[str, Any]:
    if method == "lir_pptd":
        assert isinstance(prepared, PreparedLIR)
        return run_lir_minimal(kernel, prepared, capture_trace=False)
    assert isinstance(prepared, PreparedFog)
    return run_fog_minimal(kernel, prepared, capture_trace=False)


def run_pair(m: int, *, replicate: int, m_index: int, measured: bool) -> list[dict[str, Any]]:
    base_seed = 880_000 + 10_000 * m_index + (replicate + 100) * 20
    prepared_by_method = {
        method: _prepare_method(method, m, base_seed + offset)
        for offset, method in enumerate(("lir_pptd", "fog_pptd"), start=1)
    }
    rows: list[dict[str, Any]] = []
    for order_index, method in enumerate(paired_order(m_index, replicate)):
        kernel, prepared = prepared_by_method[method]
        gc.collect()
        elapsed_ns, out = _timed(_execute_prepared, method, kernel, prepared)
        rows.append({
            "m": m,
            "replicate": replicate,
            "measured": measured,
            "order_index": order_index,
            "method": method,
            "elapsed_ns": elapsed_ns,
            "elapsed_seconds": elapsed_ns / 1e9,
            "final_truth_integer": int(out["final_truth_integer"]),
            "kernel_counts": kernel.counters.as_dict(),
        })
    return rows


def _det_index(namespace: str, upper: int) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big") % upper


def bootstrap_mean_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        raise ValueError("empty values")
    means = []
    for b in range(resamples):
        sample = [values[_det_index(f"vi-e1-sym-bootstrap|{b}|{j}", n)] for j in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return (
        means[int(0.025 * (resamples - 1))],
        means[int(0.975 * (resamples - 1))],
    )


def exact_two_sided_sign_flip_p(deltas: Sequence[float]) -> float:
    nonzero = [x for x in deltas if x != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(x > 0 for x in nonzero)
    k = min(positives, n - positives)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def holm_adjust(pvals: dict[int, float]) -> dict[int, float]:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        candidate = min(1.0, (m - rank) * p)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[int, dict[str, Any]] = {}
    pvals: dict[int, float] = {}
    for m in WORKER_GRID:
        rr = [r for r in rows if r["m"] == m and r["measured"]]
        by_rep: dict[int, dict[str, float]] = {}
        for r in rr:
            by_rep.setdefault(int(r["replicate"]), {})[str(r["method"])] = float(r["elapsed_seconds"])
        if len(by_rep) != MEASURED_PAIRS or any(set(v) != {"lir_pptd", "fog_pptd"} for v in by_rep.values()):
            raise SymmetricKernelGateError(f"PAIRING_INCOMPLETE:m={m}")
        reps = sorted(by_rep)
        lir = [by_rep[r]["lir_pptd"] for r in reps]
        fog = [by_rep[r]["fog_pptd"] for r in reps]
        delta = [f - l for l, f in zip(lir, fog)]  # positive => LIR faster
        ci = bootstrap_mean_ci(delta)
        p = exact_two_sided_sign_flip_p(delta)
        pvals[m] = p
        lmean = statistics.fmean(lir)
        fmean = statistics.fmean(fog)
        strata[m] = {
            "m": m,
            "n_pairs": len(delta),
            "lir_mean_s": lmean,
            "fog_mean_s": fmean,
            "fog_minus_lir_mean_s": statistics.fmean(delta),
            "fog_minus_lir_median_s": statistics.median(delta),
            "fog_minus_lir_bootstrap95ci_s": list(ci),
            "paired_lir_wins": sum(x > 0 for x in delta),
            "paired_fog_wins": sum(x < 0 for x in delta),
            "ties": sum(x == 0 for x in delta),
            "lir_speedup_factor_vs_fog": fmean / lmean,
            "lir_runtime_reduction_percent_vs_fog": (fmean - lmean) / fmean * 100.0,
            "raw_sign_flip_p": p,
        }
    adjusted = holm_adjust(pvals)
    for m, p in adjusted.items():
        strata[m]["holm_adjusted_p"] = p
        strata[m]["significant_alpha_0_05"] = p < 0.05
        direction = strata[m]["fog_minus_lir_mean_s"]
        strata[m]["significant_direction"] = "LIR" if p < 0.05 and direction > 0 else ("FOG" if p < 0.05 and direction < 0 else "NS")

    ss = [strata[m] for m in WORKER_GRID]
    criteria = {
        "mean_lir_faster_all_4_strata": all(s["fog_minus_lir_mean_s"] > 0 for s in ss),
        "median_lir_faster_all_4_strata": all(s["fog_minus_lir_median_s"] > 0 for s in ss),
        "paired_lir_wins_at_least_20_each_stratum": all(s["paired_lir_wins"] >= MIN_PAIR_WINS_PER_STRATUM for s in ss),
        "ci_strictly_favors_lir_strata": sum(s["fog_minus_lir_bootstrap95ci_s"][0] > 0 for s in ss),
        "holm_significant_fog_wins": sum(s["significant_direction"] == "FOG" for s in ss),
    }
    admitted = (
        criteria["mean_lir_faster_all_4_strata"]
        and criteria["median_lir_faster_all_4_strata"]
        and criteria["paired_lir_wins_at_least_20_each_stratum"]
        and criteria["ci_strictly_favors_lir_strata"] >= MIN_CI_POSITIVE_STRATA
        and criteria["holm_significant_fog_wins"] == 0
    )
    return {
        "strata": ss,
        "admission_criteria_observed": criteria,
        "paper_runtime_admissible": admitted,
        "runtime_evidence_status": "PASS_SUPPORTIVE" if admitted else "PASS_NON_SUPPORTIVE",
    }


def execute_formal(root: Path) -> dict[str, Any]:
    validation = validate_repo(root)
    raw = root / "results/raw/vi_e_external/symmetric_minimal_kernel_raw.jsonl"
    summary_path = root / "results/summary/vi_e_external/symmetric_minimal_kernel_summary.json"
    csv_path = root / "results/summary/vi_e_external/symmetric_minimal_kernel_paper.csv"
    counts_path = root / "results/summary/vi_e_external/symmetric_minimal_kernel_counts.csv"
    for path in (raw, summary_path, csv_path, counts_path):
        if path.exists():
            raise SymmetricKernelGateError(f"FORMAL_OUTPUT_ALREADY_EXISTS:{path.relative_to(root)}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mi, m in enumerate(WORKER_GRID):
        for rep in range(WARMUP_PAIRS):
            run_pair(m, replicate=-(rep + 1), m_index=mi, measured=False)
        for rep in range(MEASURED_PAIRS):
            pair = run_pair(m, replicate=rep, m_index=mi, measured=True)
            rows.extend(pair)
            with raw.open("a", encoding="utf-8") as fh:
                for row in pair:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()

    stats = summarize_rows(rows)
    summary = {
        "gate_id": "VI-E1-SYMMETRIC-MINIMAL-KERNEL-BENCHMARK-GATE-v1",
        "status": stats["runtime_evidence_status"],
        "formal_method_runs": len(rows),
        "measured_pairs": MEASURED_PAIRS * len(WORKER_GRID),
        "validation": validation,
        "design": {
            "workers": list(WORKER_GRID),
            "warmup_pairs_per_m": WARMUP_PAIRS,
            "measured_pairs_per_m": MEASURED_PAIRS,
            "paired_order_balanced": True,
            "common_field_bits": COMMON_FIELD_P.bit_length(),
            "N": N,
            "T": T,
            "K": K,
            "D": D,
            "f": F,
            "input_sharing_outside_timed_region": True,
        },
        "statistics": stats,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "paper_claim_policy": {
            "primary_lightweight_evidence": "source-normalized Shamir arithmetic comparison from VI-E.1 Lightweight Evidence Redesign",
            "secondary_timing_admitted": stats["paper_runtime_admissible"],
            "allowed_if_admitted": [
                "symmetric minimal-kernel server-side arithmetic timing under the stated local simulation boundary",
                "paired same-machine runtime reduction/speedup reported only as secondary evidence",
            ],
            "prohibited": [
                "production MPC latency",
                "network communication latency or bytes",
                "claim that benchmark timing reproduces Fog-PPTD author code",
                "TQPP wall-clock speedup",
                "using the earlier asymmetric common-field timing as algorithm-efficiency evidence",
            ],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "m", "lir_mean_s", "fog_mean_s", "fog_minus_lir_mean_s", "ci_low_s", "ci_high_s",
            "paired_lir_wins", "paired_fog_wins", "lir_speedup_factor_vs_fog",
            "lir_runtime_reduction_percent_vs_fog", "holm_adjusted_p", "significant_direction",
        ])
        for s in stats["strata"]:
            lo, hi = s["fog_minus_lir_bootstrap95ci_s"]
            writer.writerow([
                s["m"], f'{s["lir_mean_s"]:.9f}', f'{s["fog_mean_s"]:.9f}', f'{s["fog_minus_lir_mean_s"]:.9f}',
                f"{lo:.9f}", f"{hi:.9f}", s["paired_lir_wins"], s["paired_fog_wins"],
                f'{s["lir_speedup_factor_vs_fog"]:.6f}', f'{s["lir_runtime_reduction_percent_vs_fog"]:.6f}',
                f'{s["holm_adjusted_p"]:.12g}', s["significant_direction"],
            ])

    with counts_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["m", "method", "secure_sqr", "secure_mul", "secure_div", "public_mul", "public_mul_local", "public_mul_rescaled", "nonlinear_or_rescaled_calls", "high_level_calls"])
        for m in WORKER_GRID:
            for method, counts in (("LIR-PPTD", expected_lir_counts(m)), ("Fog-PPTD", expected_fog_counts(m))):
                writer.writerow([m, method, counts["secure_sqr"], counts["secure_mul"], counts["secure_div"], counts["public_mul"], counts["public_mul_local"], counts["public_mul_rescaled"], counts["nonlinear_or_rescaled_calls"], counts["high_level_calls"]])
    return summary
