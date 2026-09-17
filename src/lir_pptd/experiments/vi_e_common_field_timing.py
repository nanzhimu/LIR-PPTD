from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from lir_pptd.baselines.fog_pptd_secure_adapter import (
    FOG_PROXY_PRIME,
    FogAdapterProfile,
    FogPPTDSemanticAdapter,
)
from lir_pptd.core import ExactTaskInput
from lir_pptd.fixedpoint import default_profile
from lir_pptd.mpc import SimulatedShamirBackend, run_secure
from lir_pptd.mpc.simulated_shamir import SimulatedBackendProfile

COMMON_FIELD_P = FOG_PROXY_PRIME
COMMON_FIELD_BITS = COMMON_FIELD_P.bit_length()
N = 10
T = 4
K = 10
D = 1
F = 24
THETA_ROUND = 1_000_000
WORKER_GRID = (20, 50, 100, 200)
WARMUP_PAIRS = 5
MEASURED_PAIRS = 30
BOOTSTRAP_RESAMPLES = 10_000


class CommonFieldTimingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommonFieldTimingProfile:
    field_prime_p: int = COMMON_FIELD_P
    threshold_t: int = T
    committee_size_n: int = N
    server_ids: tuple[str, ...] = tuple(f"s{i}" for i in range(1, N + 1))
    x_coordinates: tuple[int, ...] = tuple(range(1, N + 1))
    backend_kind: str = "simulated_shamir"
    profile_id: str = "vi-e1-common-field-521bit-timing-v1"
    production_ready: bool = False
    cryptographic_security_claim: bool = False

    def digest(self) -> str:
        payload = {
            "field_prime_p": str(self.field_prime_p),
            "threshold_t": self.threshold_t,
            "committee_size_n": self.committee_size_n,
            "server_ids": self.server_ids,
            "x_coordinates": self.x_coordinates,
            "backend_kind": self.backend_kind,
            "profile_id": self.profile_id,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CommonFieldSimulatedShamirBackend(SimulatedShamirBackend):
    """Timing-only LIR-PPTD backend using the same 521-bit field as Fog-PPTD.

    The frozen production/conformance profile is not modified.  This subclass is
    isolated to VI-E.1 timing.  It keeps f=24 and all fixed-point rules unchanged,
    while replacing only the modulus and committee with the common timing values.
    """

    def __init__(self, *, master_seed: int = 0):
        # Initialize all private transcript/seed/counter state using the frozen backend.
        super().__init__(master_seed=master_seed)
        base = default_profile()
        if base.fractional_bits_f != F:
            raise CommonFieldTimingError(f"UNEXPECTED_FIXED_PRECISION:{base.fractional_bits_f}")
        # Pydantic v2 model_copy deliberately creates an isolated timing profile.
        self.fixed = base.model_copy(
            update={
                "prime_p": COMMON_FIELD_P,
                "profile_id": f"{base.profile_id}-vi-e1-commonfield521",
            }
        )
        self.profile = CommonFieldTimingProfile()
        max_bound = max(int(v) for v in self.fixed.no_wrap_bounds.values())
        if max_bound >= COMMON_FIELD_P // 4:
            raise CommonFieldTimingError("COMMON_FIELD_NO_WRAP_MARGIN_FAILED")


def native_lir_profile_n10_t4() -> SimulatedBackendProfile:
    fixed = default_profile()
    return SimulatedBackendProfile(
        field_prime_p=fixed.prime_p,
        threshold_t=T,
        committee_size_n=N,
        server_ids=tuple(f"s{i}" for i in range(1, N + 1)),
        x_coordinates=tuple(range(1, N + 1)),
        fixed_backend_profile_id=fixed.profile_id,
        fixed_backend_profile_hash=str(fixed.digest()),
    )


def conceptual_reports_million(m: int) -> tuple[int, ...]:
    """Deterministic non-degenerate D=1 workload shared by both methods.

    Values are in [0.2, 0.8] on a 1e6 public conceptual scale.  The permutation
    avoids an all-equal CRH singularity while remaining independent of either method.
    """
    if m <= 1:
        raise ValueError("m must exceed one")
    vals = tuple(200_000 + ((37 * i + 13) % 601) * 1_000 for i in range(m))
    if min(vals) < 0 or max(vals) > THETA_ROUND or len(set(vals)) < 2:
        raise AssertionError("invalid timing workload")
    return vals


def build_lir_task(m: int, *, task_id: str) -> ExactTaskInput:
    vals = conceptual_reports_million(m)
    ids = tuple(f"w{i:04d}" for i in range(m))
    reports = {w: (Fraction(v, THETA_ROUND),) for w, v in zip(ids, vals)}
    reputations = {w: Fraction(1, 2) for w in ids}
    epochs = {w: 0 for w in ids}
    return ExactTaskInput(
        task_id=task_id,
        task_kind="numerical",
        K=K,
        tau=Fraction(1, 5),
        epsilon_c=Fraction(1, 1024),
        kappa=2,
        eta=Fraction(1, 10),
        reports=reports,
        reputations=reputations,
        epochs=epochs,
        participant_ids=ids,
        report_lower=0,
        report_upper=1,
    )


def run_lir_common(m: int, *, seed: int) -> dict[str, Any]:
    task = build_lir_task(m, task_id=f"vi-e1-lir-m{m}-s{seed}")
    backend = CommonFieldSimulatedShamirBackend(master_seed=seed)
    result = run_secure(task, backend)
    final = tuple(result.l3_result["final_output_integer"])
    return {
        "status": result.status,
        "final_output_integer": final,
        "operation_count": len(result.operation_trace),
        "backend_profile_hash": str(backend.profile.digest()),
        "fixed_profile_hash": str(backend.fixed.digest()),
    }


def run_lir_native(m: int, *, seed: int) -> dict[str, Any]:
    task = build_lir_task(m, task_id=f"vi-e1-lir-native-m{m}-s{seed}")
    backend = SimulatedShamirBackend(native_lir_profile_n10_t4(), master_seed=seed)
    result = run_secure(task, backend)
    return {
        "status": result.status,
        "final_output_integer": tuple(result.l3_result["final_output_integer"]),
        "truth_integer_trajectory": tuple(result.l3_result["truth_integer_trajectory"]),
        "reputation_transitions": tuple(result.l3_result["reputation_transitions"]),
    }


def run_lir_common_conformance(m: int, *, seed: int) -> dict[str, Any]:
    task = build_lir_task(m, task_id=f"vi-e1-lir-common-conf-m{m}-s{seed}")
    backend = CommonFieldSimulatedShamirBackend(master_seed=seed)
    result = run_secure(task, backend)
    return {
        "status": result.status,
        "final_output_integer": tuple(result.l3_result["final_output_integer"]),
        "truth_integer_trajectory": tuple(result.l3_result["truth_integer_trajectory"]),
        "reputation_transitions": tuple(result.l3_result["reputation_transitions"]),
    }


def assert_lir_field_semantics_preserved() -> dict[str, Any]:
    rows = []
    for m in (20, 200):
        native = run_lir_native(m, seed=91000 + m)
        common = run_lir_common_conformance(m, seed=91000 + m)
        equal = native == common
        rows.append({"m": m, "exact_l3_equal": equal})
        if not equal:
            raise CommonFieldTimingError(f"LIR_FIELD_SEMANTICS_MISMATCH:m={m}")
    return {"status": "PASS", "cases": rows}


def run_fog_common(m: int, *, seed: int) -> dict[str, Any]:
    vals = conceptual_reports_million(m)
    profile = FogAdapterProfile(
        committee_size_n=N,
        threshold_t=T,
        prime_p=COMMON_FIELD_P,
        source_nominal_field_bits=512,
        local_proxy_field_bits=COMMON_FIELD_BITS,
        field_policy="common VI-E1 521-bit field; same proxy used by admitted Fog adapter",
    )
    adapter = FogPPTDSemanticAdapter(profile=profile, master_seed=seed)
    result = adapter.run_d1(vals, iterations=K, theta_round=THETA_ROUND)
    if not (0 <= result.truth_integer <= THETA_ROUND):
        raise CommonFieldTimingError("FOG_OUTPUT_OUT_OF_CONCEPTUAL_RANGE")
    return {
        "truth_integer": result.truth_integer,
        "counters": result.counters,
        "profile": profile.as_dict(),
    }


def _timed_call(fn, *args, **kwargs) -> tuple[int, Any]:
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return t1 - t0, out


def paired_order(m_index: int, replicate: int) -> tuple[str, str]:
    return ("lir_pptd", "fog_pptd") if (m_index + replicate) % 2 == 0 else ("fog_pptd", "lir_pptd")


def run_pair(m: int, *, replicate: int, m_index: int, measured: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_seed = 730_000 + 10_000 * m_index + replicate * 10
    vals = conceptual_reports_million(m)
    # Prepare method objects outside the timed region so construction/harness work
    # does not contaminate the matched arithmetic timing boundary.
    lir_task = build_lir_task(m, task_id=f"vi-e1-formal-lir-m{m}-r{replicate}")
    lir_backend = CommonFieldSimulatedShamirBackend(master_seed=base_seed)
    fog_profile = FogAdapterProfile(
        committee_size_n=N,
        threshold_t=T,
        prime_p=COMMON_FIELD_P,
        source_nominal_field_bits=512,
        local_proxy_field_bits=COMMON_FIELD_BITS,
        field_policy="common VI-E1 521-bit field; same proxy used by admitted Fog adapter",
    )
    fog_adapter = FogPPTDSemanticAdapter(profile=fog_profile, master_seed=base_seed + 1)

    for order_index, method in enumerate(paired_order(m_index, replicate)):
        if method == "lir_pptd":
            seed = base_seed
            ns, result = _timed_call(run_secure, lir_task, lir_backend)
            payload = {
                "status": result.status,
                "final_output_integer": tuple(result.l3_result["final_output_integer"]),
                "operation_count": len(result.operation_trace),
                "backend_profile_hash": str(lir_backend.profile.digest()),
                "fixed_profile_hash": str(lir_backend.fixed.digest()),
            }
        else:
            seed = base_seed + 1
            ns, result = _timed_call(fog_adapter.run_d1, vals, iterations=K, theta_round=THETA_ROUND)
            if not (0 <= result.truth_integer <= THETA_ROUND):
                raise CommonFieldTimingError("FOG_OUTPUT_OUT_OF_CONCEPTUAL_RANGE")
            payload = {
                "truth_integer": result.truth_integer,
                "counters": result.counters,
                "profile": fog_profile.as_dict(),
            }
        rows.append(
            {
                "m": m,
                "replicate": replicate,
                "measured": measured,
                "order_index": order_index,
                "method": method,
                "seed": seed,
                "elapsed_ns": ns,
                "elapsed_seconds": ns / 1e9,
                "payload": payload,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_repo(root: Path) -> dict[str, Any]:
    required = [
        root / "src/lir_pptd/mpc/secure_algorithm.py",
        root / "src/lir_pptd/mpc/simulated_shamir.py",
        root / "src/lir_pptd/fixedpoint/__init__.py",
        root / "src/lir_pptd/baselines/fog_pptd_secure_adapter.py",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise CommonFieldTimingError("MISSING_REQUIRED_FILES:" + ",".join(missing))
    base = default_profile()
    if base.prime_p.bit_length() != 127 or base.fractional_bits_f != F:
        raise CommonFieldTimingError("UNEXPECTED_NATIVE_LIR_PROFILE")
    native = native_lir_profile_n10_t4()
    if (native.committee_size_n, native.threshold_t) != (N, T):
        raise CommonFieldTimingError("NATIVE_NT_PROFILE_CONSTRUCTION_FAILED")
    if COMMON_FIELD_BITS != 521:
        raise CommonFieldTimingError("COMMON_FIELD_BIT_LENGTH_CHANGED")
    semantics = assert_lir_field_semantics_preserved()
    fog_smoke = run_fog_common(20, seed=777001)
    lir_smoke = run_lir_common(20, seed=777002)
    return {
        "status": "PASS",
        "common_field_bits": COMMON_FIELD_BITS,
        "common_field_prime": str(COMMON_FIELD_P),
        "N": N,
        "T": T,
        "K": K,
        "D": D,
        "f": F,
        "timing_boundary": {
            "included": [
                "method-specific input share generation",
                "all local simulated secure arithmetic",
                "all modeled internal openings and resharings",
                "K=10 truth-discovery iterations",
                "LIR-PPTD final output-evidence/reputation maintenance",
                "final local reconstruction returned by each method runner",
            ],
            "excluded": [
                "Python import/startup",
                "workload construction",
                "backend object construction before timer",
                "result serialization and file I/O",
                "network transport and production preprocessing",
            ],
            "metric_label": "matched local simulated end-to-end arithmetic time",
        },
        "lir_native_to_common_field_semantics": semantics,
        "fog_smoke_truth_integer": fog_smoke["truth_integer"],
        "lir_smoke_final_output_integer": lir_smoke["final_output_integer"],
        "bound_hashes": {str(p.relative_to(root)).replace('\\','/'): _sha256(p) for p in required},
        "claim_boundary": {
            "production_mpc": False,
            "network_latency": False,
            "communication_bytes": False,
            "allowed_speedup_scope": "same-machine same-field same-(N,T) local simulated execution only",
        },
    }


def _det_index(namespace: str, upper: int) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big") % upper


def bootstrap_mean_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if not values:
        raise ValueError("empty bootstrap values")
    n = len(values)
    means = []
    for b in range(resamples):
        sample = [values[_det_index(f"vi-e1-bootstrap|{b}|{j}", n)] for j in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return lo, hi


def exact_two_sided_sign_flip_p(deltas: Sequence[float]) -> float:
    nonzero = [x for x in deltas if x != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(x > 0 for x in nonzero)
    k = min(positives, n - positives)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**n))


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
        if len(by_rep) != MEASURED_PAIRS or any(set(x) != {"lir_pptd", "fog_pptd"} for x in by_rep.values()):
            raise CommonFieldTimingError(f"PAIRING_INCOMPLETE:m={m}")
        lir = [by_rep[i]["lir_pptd"] for i in sorted(by_rep)]
        fog = [by_rep[i]["fog_pptd"] for i in sorted(by_rep)]
        # Positive delta means Fog is slower; speedup >1 means LIR is faster.
        delta = [b - a for a, b in zip(lir, fog)]
        ratio = [b / a for a, b in zip(lir, fog)]
        ci = bootstrap_mean_ci(delta)
        p = exact_two_sided_sign_flip_p(delta)
        pvals[m] = p
        strata[m] = {
            "m": m,
            "n_pairs": len(delta),
            "lir_mean_s": statistics.fmean(lir),
            "fog_mean_s": statistics.fmean(fog),
            "fog_minus_lir_mean_s": statistics.fmean(delta),
            "fog_minus_lir_median_s": statistics.median(delta),
            "fog_minus_lir_bootstrap95ci_s": list(ci),
            "fog_over_lir_mean_paired_ratio": statistics.fmean(ratio),
            "lir_speedup_factor_vs_fog_from_mean_times": statistics.fmean(fog) / statistics.fmean(lir),
            "lir_runtime_reduction_percent_vs_fog": (statistics.fmean(fog) - statistics.fmean(lir)) / statistics.fmean(fog) * 100.0,
            "raw_sign_flip_p": p,
            "delta_positive_count": sum(x > 0 for x in delta),
            "delta_negative_count": sum(x < 0 for x in delta),
            "delta_zero_count": sum(x == 0 for x in delta),
        }
    adj = holm_adjust(pvals)
    for m, p in adj.items():
        strata[m]["holm_adjusted_p"] = p
        strata[m]["significant_alpha_0_05"] = p < 0.05
    return {
        "strata": [strata[m] for m in WORKER_GRID],
        "all_four_ci_same_direction": all(
            (s["fog_minus_lir_bootstrap95ci_s"][0] > 0 and s["fog_minus_lir_bootstrap95ci_s"][1] > 0)
            or (s["fog_minus_lir_bootstrap95ci_s"][0] < 0 and s["fog_minus_lir_bootstrap95ci_s"][1] < 0)
            for s in (strata[m] for m in WORKER_GRID)
        ),
    }


def execute_formal(root: Path) -> dict[str, Any]:
    validation = validate_repo(root)
    raw = root / "results/raw/vi_e_external/common_field_timing_raw.jsonl"
    summary_path = root / "results/summary/vi_e_external/common_field_timing_summary.json"
    csv_path = root / "results/summary/vi_e_external/common_field_timing_paper.csv"
    for p in (raw, summary_path, csv_path):
        if p.exists():
            raise CommonFieldTimingError(f"FORMAL_OUTPUT_ALREADY_EXISTS:{p.relative_to(root)}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mi, m in enumerate(WORKER_GRID):
        for rep in range(WARMUP_PAIRS):
            run_pair(m, replicate=-(rep + 1), m_index=mi, measured=False)
        for rep in range(MEASURED_PAIRS):
            pair_rows = run_pair(m, replicate=rep, m_index=mi, measured=True)
            rows.extend(pair_rows)
            with raw.open("a", encoding="utf-8") as fh:
                for row in pair_rows:
                    fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    fh.flush()

    stats = summarize_rows(rows)
    summary = {
        "gate_id": "VI-E1-COMMON-FIELD-COMMON-BOUNDARY-TIMING-GATE-v1",
        "status": "PASS",
        "formal_experiments_run": len(rows),
        "measured_pairs": MEASURED_PAIRS * len(WORKER_GRID),
        "validation": validation,
        "design": {
            "workers": WORKER_GRID,
            "warmup_pairs_per_m": WARMUP_PAIRS,
            "measured_pairs_per_m": MEASURED_PAIRS,
            "paired_order_balanced": True,
            "common_field_bits": COMMON_FIELD_BITS,
            "N": N,
            "T": T,
            "K": K,
            "D": D,
            "f": F,
        },
        "statistics": stats,
        "paper_claim_policy": {
            "allowed": [
                "matched local simulated timing comparison of LIR-PPTD and the admitted Fog-PPTD semantic adapter under the same 521-bit field and (N,T)=(10,4)",
                "same-machine speedup percentage explicitly qualified as simulated/local and implementation-specific",
            ],
            "prohibited": [
                "production MPC latency",
                "network communication latency or bytes",
                "claim that the Fog adapter is author-code-identical",
                "TQPP wall-clock speedup",
                "unqualified universal lightweightness",
            ],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("m,lir_mean_s,fog_mean_s,fog_minus_lir_mean_s,ci_low_s,ci_high_s,lir_runtime_reduction_percent_vs_fog,holm_adjusted_p\n")
        for s in stats["strata"]:
            lo, hi = s["fog_minus_lir_bootstrap95ci_s"]
            fh.write(
                f'{s["m"]},{s["lir_mean_s"]:.9f},{s["fog_mean_s"]:.9f},{s["fog_minus_lir_mean_s"]:.9f},'
                f'{lo:.9f},{hi:.9f},{s["lir_runtime_reduction_percent_vs_fog"]:.6f},{s["holm_adjusted_p"]:.12g}\n'
            )
    return summary
