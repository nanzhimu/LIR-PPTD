from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from lir_pptd.core import ExactTaskInput
from lir_pptd.fixedpoint import default_profile
from lir_pptd.fixedpoint.encoding import trunc_scaled
from lir_pptd.fixedpoint.range_analysis import PreflightStatus, preflight
from lir_pptd.mpc import SimulatedShamirBackend, run_secure
from lir_pptd.mpc.secure_algorithm import _ok, _value
from lir_pptd.mpc.simulated_shamir import SimulatedBackendProfile
from lir_pptd.mpc.types import SecureTaskResult

DESIGN_REL = Path("configs/phase_vi_e/vi_e_formal_design.json")
BINDING_REL = Path("results/summary/vi_e_start_binding.json")
APPROVAL_REL = Path("approvals/phase_vi_e/VI_E_START_APPROVAL.json")
RAW_REL = Path("results/raw/vi_e/matched_efficiency_pairs.jsonl")
CHECKPOINT_REL = Path("results/summary/vi_e_checkpoint.json")
SUMMARY_REL = Path("results/summary/vi_e_formal_summary.json")

EXPECTED_HASHES = {
    "src/lir_pptd/mpc/secure_algorithm.py": "facad0c673969dfdc32dfc67e18e922e1446e7819c96bed83852b36c4cde956d",
    "src/lir_pptd/mpc/simulated_shamir.py": "d0d2b123a872aa2f88aeaec1270ea28dce9ff16adf529ebb151418995967e940",
    "src/lir_pptd/mpc/types.py": "7fd70decd0aea8d4971c76d726abb6824564cb058378ca4fa97e5bfa6ed2bd30",
    "src/lir_pptd/mpc/interfaces.py": "7cbb0f61dc32c60fbc0cae6987bf320cada9bb732992f4acc6226f7697f55b1d",
    "src/lir_pptd/fixedpoint/profile.py": "b9b504c5221d5ef5c67311076db3102cf9cec60fa3a56743fb93cb1e952dbedd",
    "configs/backend_profiles/simulated_shamir.yaml": "8953d5a6c4587f90c23a6b3a31dc3bece7863eb31037769a071587f12d817ab9",
    "configs/backend_profiles/deterministic_fixed_plaintext.yaml": "9521a1ad4d5ec4ad0334ca49db83c04f1d6fb35e114e0b80e97efb3a576e4d1a",
    "docs/COMPARISON_FAIRNESS_SPEC.md": "2c540e3ed3d9dd82459339fdf3098c6fd31b37678022b41524e848caff473332",
    "configs/frozen/comparison_fairness_manifest.json": "b2f4e95556b77044c708c1907cc79b08801b276c1a163fd78f69ed92da186a59",
}

WORKER_COUNTS = (20, 50, 100, 200)
WARMUPS = 5
MEASURED = 30
BOOTSTRAP = 10_000
ALPHA = 0.05

class VIEError(RuntimeError):
    pass

def sha(path: Path) -> str:
    if not path.is_file():
        raise VIEError(f"MISSING_FILE:{path.as_posix()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise VIEError(f"INVALID_JSONL:{path}:{i}") from exc
    return out

def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        raise VIEError(f"GIT_HEAD_FAILED:{exc}") from exc

def benchmark_profile() -> SimulatedBackendProfile:
    fixed = default_profile()
    ids = tuple(f"s{i}" for i in range(1, 11))
    xs = tuple(range(1, 11))
    return SimulatedBackendProfile(
        profile_id="vi-e-simulated-shamir-n10-t4-v1",
        field_prime_p=fixed.prime_p,
        threshold_t=4,
        committee_size_n=10,
        server_ids=ids,
        x_coordinates=xs,
        fixed_backend_profile_id=fixed.profile_id,
        fixed_backend_profile_hash=str(fixed.digest()),
    )

def build_task(worker_count: int, replicate: int, *, K: int = 10) -> ExactTaskInput:
    ids = tuple(f"w{i:04d}" for i in range(worker_count))
    # Exact deterministic workload.  Values vary across replicate and worker but
    # remain in [0,1], with no Python float entering the algorithm.
    reports = {}
    for i, wid in enumerate(ids):
        numerator = (97 * (i + 1) + 53 * replicate + 17 * worker_count) % 1001
        reports[wid] = (Fraction(numerator, 1000),)
    reps = {wid: Fraction(1, 2) for wid in ids}
    epochs = {wid: 0 for wid in ids}
    return ExactTaskInput(
        task_id=f"vi-e-m{worker_count}-r{replicate:02d}",
        task_kind="numerical",
        K=K,
        tau=Fraction(1, 5),
        epsilon_c=Fraction(1, 1024),
        kappa=Fraction(2, 1),
        eta=Fraction(1, 10),
        reports=reports,
        reputations=reps,
        epochs=epochs,
        participant_ids=ids,
        report_lower=Fraction(0, 1),
        report_upper=Fraction(1, 1),
    )

def run_matched_pptd_core(task: ExactTaskInput, backend: SimulatedShamirBackend) -> dict[str, Any]:
    """
    Exact matched control for VI-E.

    It is intentionally identical to the bound run_secure truth-discovery path
    through the final K-th truth, then stops before q_out recomputation and the
    longitudinal reputation transition.  Therefore it is an internal matched
    PPTD-core control, NOT an implementation of Fog-PPTD/FPTD/TQPP.
    """
    p = default_profile()
    ids = tuple(sorted(task.participant_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("invalid participant set")
    if task.task_kind not in {"numerical", "categorical"}:
        raise ValueError("unknown task kind")
    dims = {len(task.reports[w]) for w in ids}
    if len(dims) != 1 or next(iter(dims)) < 1:
        raise ValueError("dimension mismatch")
    D = next(iter(dims))
    cert = preflight(
        p,
        participants=len(ids),
        dimension=D,
        K=task.K,
        epsilon_c=task.epsilon_c,
        tau=task.tau,
        eta=task.eta,
        kappa=task.kappa,
    )
    if cert.preflight_status != PreflightStatus.PASSED:
        raise ValueError(cert.reason_code or "PREFLIGHT_REJECTED")

    delta = p.scale_delta
    reports = {
        w: tuple(trunc_scaled(x, delta) for x in task.reports[w])
        for w in ids
    }
    reps = {w: trunc_scaled(task.reputations[w], delta) for w in ids}
    if task.task_kind == "categorical" and any(
        any(x not in (0, delta) for x in row) or sum(row) != delta
        for row in reports.values()
    ):
        raise ValueError("INVALID_ONE_HOT")

    tau = trunc_scaled(task.tau, delta)
    eps = trunc_scaled(task.epsilon_c, delta)

    def share(secret, value, namespace="algorithm", iteration=None, worker=None, coordinate=None):
        backend.set_context(
            task_id=task.task_id,
            iteration=iteration,
            worker_id=worker,
            coordinate=coordinate,
        )
        return backend.share_secret(secret, value, namespace=namespace)

    report_shares = {
        w: tuple(
            share(f"report:{w}:{h}", x, "input", worker=w, coordinate=h)
            for h, x in enumerate(row)
        )
        for w, row in reports.items()
    }
    rep_shares = {
        w: share(f"reputation:{w}", x, "input", worker=w)
        for w, x in reps.items()
    }
    eps_s = share("epsilon", eps, "public")
    tau_s = share("tau", tau, "public")

    effective = {w: _ok(backend.local_add(rep_shares[w], eps_s)) for w in ids}
    den = effective[ids[0]]
    for w in ids[1:]:
        den = _ok(backend.local_add(den, effective[w]))

    truth = []
    for h in range(D):
        terms = [
            _ok(backend.SecMulPositive(effective[w], report_shares[w][h]))
            for w in ids
        ]
        num = terms[0]
        for term in terms[1:]:
            num = _ok(backend.local_add(num, term))
        truth.append(
            _ok(backend.SecDivPositive(num, den, 1, p.no_wrap_bounds["state"]))
        )

    for k in range(task.K):
        influences = {}
        for w in ids:
            squares = []
            for h in range(D):
                backend.set_context(
                    task_id=task.task_id, iteration=k, worker_id=w, coordinate=h
                )
                diff = _ok(backend.local_sub(report_shares[w][h], truth[h]))
                squares.append(_ok(backend.SecSqr(diff)))
            distance = squares[0]
            for sq in squares[1:]:
                distance = _ok(backend.local_add(distance, sq))
            tau_distance = _ok(backend.local_add(tau_s, distance))
            q = _ok(
                backend.SecDivPositive(
                    tau_s, tau_distance, tau, tau + D * delta
                )
            )
            influences[w] = _ok(backend.SecMulPositive(effective[w], q))

        aden = influences[ids[0]]
        for w in ids[1:]:
            aden = _ok(backend.local_add(aden, influences[w]))

        nums = []
        for h in range(D):
            products = [
                _ok(backend.SecMulPositive(influences[w], report_shares[w][h]))
                for w in ids
            ]
            num = products[0]
            for product in products[1:]:
                num = _ok(backend.local_add(num, product))
            nums.append(num)
        truth = [
            _ok(backend.SecDivPositive(num, aden, 1, p.no_wrap_bounds["state"]))
            for num in nums
        ]

    final = tuple(_value(backend, x) for x in truth)
    backend.prepare_output(task.task_id, truth[0], False)
    return {
        "task_id": task.task_id,
        "final_output_integer": final,
        "operation_trace": backend.export_transcript(public=False).entries,
        "reputation_updates_per_participant": 0,
    }

def operation_counts(trace) -> dict[str, int]:
    c = collections.Counter(x.operation_name for x in trace)
    return dict(sorted(c.items()))

def high_level_counts(trace) -> dict[str, int]:
    names = ("SecSqr", "SecMulPositive", "SecDivPositive", "PubMulPositive")
    c = operation_counts(trace)
    return {name: int(c.get(name, 0)) for name in names}

def final_output_full(result: SecureTaskResult) -> tuple[int, ...]:
    return tuple(int(x) for x in result.l3_result["final_output_integer"])

def timed_execute(method: str, task: ExactTaskInput, *, master_seed: int) -> dict[str, Any]:
    backend = SimulatedShamirBackend(benchmark_profile(), master_seed=master_seed)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        wall0 = time.perf_counter_ns()
        cpu0 = time.process_time_ns()
        if method == "full":
            result = run_secure(task, backend)
            final = final_output_full(result)
            trace = result.operation_trace
            rep_updates = int(result.reputation_updates_per_participant)
        elif method == "core":
            result = run_matched_pptd_core(task, backend)
            final = tuple(int(x) for x in result["final_output_integer"])
            trace = result["operation_trace"]
            rep_updates = int(result["reputation_updates_per_participant"])
        else:
            raise VIEError(f"UNKNOWN_METHOD:{method}")
        cpu1 = time.process_time_ns()
        wall1 = time.perf_counter_ns()
    finally:
        if was_enabled:
            gc.enable()

    return {
        "wall_seconds": (wall1 - wall0) / 1e9,
        "cpu_seconds": (cpu1 - cpu0) / 1e9,
        "final_output_integer": final,
        "operation_counts": operation_counts(trace),
        "high_level_counts": high_level_counts(trace),
        "transcript_entry_count": len(trace),
        "reputation_updates_per_participant": rep_updates,
    }

def count_delta(full: Mapping[str, int], core: Mapping[str, int]) -> dict[str, int]:
    keys = sorted(set(full) | set(core))
    return {k: int(full.get(k, 0)) - int(core.get(k, 0)) for k in keys}

def assert_matched_pair(worker_count: int, full: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    if tuple(full["final_output_integer"]) != tuple(core["final_output_integer"]):
        raise VIEError(f"MATCHED_OUTPUT_MISMATCH:m={worker_count}")
    if full["reputation_updates_per_participant"] != 1:
        raise VIEError("FULL_REPUTATION_UPDATE_COUNT_MISMATCH")
    if core["reputation_updates_per_participant"] != 0:
        raise VIEError("CORE_REPUTATION_UPDATE_COUNT_MISMATCH")

    delta = count_delta(full["high_level_counts"], core["high_level_counts"])
    expected = {
        "SecSqr": worker_count,
        "SecMulPositive": worker_count,
        "SecDivPositive": worker_count,
        "PubMulPositive": 3 * worker_count,
    }
    if delta != expected:
        raise VIEError(
            "PRIMITIVE_DELTA_MISMATCH:"
            + json.dumps({"actual": delta, "expected": expected}, sort_keys=True)
        )

def static_preflight(root: Path) -> dict[str, Any]:
    actual = {}
    for rel, expected in EXPECTED_HASHES.items():
        got = sha(root / rel)
        actual[rel] = got
        if got != expected:
            raise VIEError(f"BOUND_SOURCE_SHA_MISMATCH:{rel}:{got}:{expected}")

    fairness = read_json(root / "configs/frozen/comparison_fairness_manifest.json")
    status = {m["method_id"]: m["comparison_status"] for m in fairness["methods"]}
    if status.get("fptd") != "BLOCKED" or status.get("tqpp") != "BLOCKED":
        raise VIEError("EXTERNAL_BASELINE_STATUS_CHANGED_REVIEW_REQUIRED")

    p = default_profile()
    profile = benchmark_profile()
    if p.fractional_bits_f != 24:
        raise VIEError("FIXED_PRECISION_DRIFT")
    if profile.committee_size_n != 10 or profile.threshold_t != 4:
        raise VIEError("VI_E_THRESHOLD_PROFILE_MISMATCH")

    for m in WORKER_COUNTS:
        cert = preflight(
            p,
            participants=m,
            dimension=1,
            K=10,
            epsilon_c=Fraction(1, 1024),
            tau=Fraction(1, 5),
            eta=Fraction(1, 10),
            kappa=Fraction(2, 1),
        )
        if cert.preflight_status != PreflightStatus.PASSED:
            raise VIEError(f"WORKLOAD_PREFLIGHT_REJECTED:m={m}:{cert.reason_code}")

    # One tiny diagnostic pair proves that the matched control preserves the
    # truth path and that the isolated high-level primitive delta is exact.
    task = build_task(4, 999, K=2)
    core = timed_execute("core", task, master_seed=999_001)
    full = timed_execute("full", task, master_seed=999_001)
    assert_matched_pair(4, full, core)

    return {
        "bound_source_hashes": actual,
        "fixed_profile_id": p.profile_id,
        "fixed_fractional_bits_f": p.fractional_bits_f,
        "simulated_profile": {
            "profile_id": profile.profile_id,
            "committee_size_N": profile.committee_size_n,
            "threshold_T": profile.threshold_t,
            "field_prime_p": profile.field_prime_p,
            "fixed_profile_hash": profile.fixed_backend_profile_hash,
            "production_ready": profile.production_ready,
            "cryptographic_security_claim": profile.cryptographic_security_claim,
        },
        "external_baselines": {
            "fptd": status.get("fptd"),
            "tqpp": status.get("tqpp"),
            "fog_pptd_runnable": False,
        },
        "diagnostic_pair": {
            "worker_count": 4,
            "K": 2,
            "output_equal": True,
            "high_level_delta": count_delta(
                full["high_level_counts"], core["high_level_counts"]
            ),
        },
    }

def environment_binding() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

def binding_payload(root: Path) -> dict[str, Any]:
    checks = static_preflight(root)
    return {
        "schema_version": "1.0",
        "experiment_namespace": "VI-E",
        "status": "START_BINDING_READY_NOT_APPROVED",
        "design": read_json(root / DESIGN_REL),
        "design_sha256": sha(root / DESIGN_REL),
        "runner_sha256": sha(root / "src/lir_pptd/experiments/phase_vi_e/runner.py"),
        "repo_head": git_head(root),
        "environment": environment_binding(),
        "checks": checks,
        "planned_measured_pairs": len(WORKER_COUNTS) * MEASURED,
        "warmup_pairs": len(WORKER_COUNTS) * WARMUPS,
        "formal_experiments_run_before_approval": 0,
    }

def binding_sha(payload: Mapping[str, Any]) -> str:
    p = dict(payload)
    p.pop("start_binding_sha256", None)
    return hashlib.sha256(canonical_bytes(p)).hexdigest()

def preflight_and_bind(root: Path) -> None:
    payload = binding_payload(root)
    payload["start_binding_sha256"] = binding_sha(payload)
    path = root / BINDING_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise VIEError("EXISTING_VI_E_BINDING_DIFFERS")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("VI_E_PREFLIGHT=PASS")
    print("FORMAL_EXPERIMENTS_RUN=0")
    print("MATCHED_CORE_CONFORMANCE=PASS")
    print("FIXED_POINT_F=24")
    print("SIMULATED_THRESHOLD_N=10")
    print("SIMULATED_THRESHOLD_T=4")
    print("PRODUCTION_MPC_USED=NO")
    print("NETWORK_BYTES_MEASURED=NO")
    print("ONLINE_ROUNDS_MEASURED=NO")
    print("OFFLINE_PREPROCESSING_MEASURED=NO")
    print("PLANNED_MEASURED_PAIRS=120")
    print("WARMUP_PAIRS=20")
    print("VI_E_START_BINDING_SHA256=" + payload["start_binding_sha256"])
    print("BINDING_PATH=" + BINDING_REL.as_posix())

def verify_binding(root: Path) -> dict[str, Any]:
    path = root / BINDING_REL
    if not path.is_file():
        raise VIEError("VI_E_START_BINDING_MISSING")
    existing = read_json(path)
    current = binding_payload(root)
    current["start_binding_sha256"] = binding_sha(current)
    if current != existing:
        raise VIEError("VI_E_START_BINDING_DRIFT")
    return existing

def record_approval(root: Path, binding: str, text: str) -> None:
    b = verify_binding(root)
    if b["start_binding_sha256"] != binding:
        raise VIEError("VI_E_APPROVAL_BINDING_MISMATCH")
    expected = f"APPROVE VI-E {binding}"
    if text.strip() != expected:
        raise VIEError("VI_E_HUMAN_APPROVAL_TEXT_MISMATCH")
    path = root / APPROVAL_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "experiment_namespace": "VI-E",
        "status": "PASS",
        "approval_type": "formal_start_approval",
        "start_binding_sha256": binding,
        "human_approval_source": "explicit_terminal_entry_by_user",
        "planned_measured_pairs": 120,
        "runtime_gate_enabled": False,
    }
    if path.exists():
        if read_json(path) != payload:
            raise VIEError("EXISTING_VI_E_APPROVAL_DIFFERS")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("VI_E_START_APPROVAL=PASS")
    print("APPROVAL_SHA256=" + sha(path))

def verify_approval(root: Path, binding: str) -> None:
    path = root / APPROVAL_REL
    if not path.is_file():
        raise VIEError("VI_E_START_APPROVAL_MISSING")
    a = read_json(path)
    if a.get("status") != "PASS" or a.get("start_binding_sha256") != binding:
        raise VIEError("VI_E_START_APPROVAL_INVALID")

def checkpoint(root: Path, binding: str, ids: set[str], failures: int, stage: str) -> None:
    p = root / CHECKPOINT_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema_version": "1.0",
        "experiment_namespace": "VI-E",
        "start_binding_sha256": binding,
        "stage": stage,
        "committed_measured_pairs": len(ids),
        "planned_measured_pairs": 120,
        "retained_failure_pairs": failures,
        "runtime_gate_enabled": False,
        "updated_unix": time.time(),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)

def run_pair(worker_count: int, replicate: int, binding: str, *, warmup: bool) -> dict[str, Any]:
    task = build_task(worker_count, replicate)
    master_seed = 7_100_000 + worker_count * 1000 + replicate
    order = ("core", "full") if replicate % 2 else ("full", "core")
    results = {}
    for method in order:
        results[method] = timed_execute(method, task, master_seed=master_seed)
    assert_matched_pair(worker_count, results["full"], results["core"])

    if warmup:
        return {"status": "warmup_pass"}

    full = results["full"]
    core = results["core"]
    delta_wall = full["wall_seconds"] - core["wall_seconds"]
    delta_cpu = full["cpu_seconds"] - core["cpu_seconds"]
    relative = (
        100.0 * delta_wall / core["wall_seconds"]
        if core["wall_seconds"] > 0
        else None
    )
    return {
        "schema_version": "1.0",
        "run_id": f"VI-E|m={worker_count}|rep={replicate}",
        "stage": "VI-E",
        "status": "success",
        "start_binding_sha256": binding,
        "worker_count": worker_count,
        "dimension_D": 1,
        "K": 10,
        "replicate_index": replicate,
        "method_order": list(order),
        "simulated_committee_N": 10,
        "simulated_threshold_T": 4,
        "fixed_fractional_bits_f": 24,
        "full": full,
        "matched_core": core,
        "paired_delta_wall_seconds": delta_wall,
        "paired_delta_cpu_seconds": delta_cpu,
        "relative_wall_overhead_percent": relative,
        "high_level_primitive_delta": count_delta(
            full["high_level_counts"], core["high_level_counts"]
        ),
        "transcript_entry_delta": (
            full["transcript_entry_count"] - core["transcript_entry_count"]
        ),
        "production_mpc_used": False,
        "cryptographic_security_claim": False,
        "network_communication_bytes_measured": False,
        "online_rounds_measured": False,
        "offline_preprocessing_measured": False,
    }

def bootstrap_mean_ci(values: list[float], *, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(BOOTSTRAP):
        samples.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    lo = samples[int(0.025 * BOOTSTRAP)]
    hi = samples[min(BOOTSTRAP - 1, int(0.975 * BOOTSTRAP))]
    return lo, hi

def exact_two_sided_sign_flip(values: list[float]) -> float:
    n = len(values)
    observed = abs(statistics.mean(values))
    if n <= 20:
        extreme = 0
        total = 1 << n
        for bits in range(total):
            s = 0.0
            for i, v in enumerate(values):
                s += v if (bits >> i) & 1 else -v
            if abs(s / n) >= observed - 1e-15:
                extreme += 1
        return extreme / total

    # Meet-in-the-middle exact enumeration for n=30.
    half = n // 2
    a, b = values[:half], values[half:]
    sums_a = []
    sums_b = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(a)):
        sums_a.append(sum(s * v for s, v in zip(signs, a)))
    for signs in itertools.product((-1.0, 1.0), repeat=len(b)):
        sums_b.append(sum(s * v for s, v in zip(signs, b)))
    sums_b.sort()
    import bisect
    threshold = observed * n - 1e-15
    extreme = 0
    for x in sums_a:
        left = bisect.bisect_right(sums_b, -threshold - x)
        right = len(sums_b) - bisect.bisect_left(sums_b, threshold - x)
        extreme += left + right
    return extreme / (2 ** n)

def holm(p_by_key: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_by_key.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[key] = running
    return adjusted

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_m = {}
    raw_p = {}
    for m in WORKER_COUNTS:
        rs = [r for r in rows if r["worker_count"] == m and r["status"] == "success"]
        if len(rs) != MEASURED:
            raise VIEError(f"AGGREGATE_INCOMPLETE:m={m}:n={len(rs)}")
        deltas = [float(r["paired_delta_wall_seconds"]) for r in rs]
        core = [float(r["matched_core"]["wall_seconds"]) for r in rs]
        full = [float(r["full"]["wall_seconds"]) for r in rs]
        cpu_delta = [float(r["paired_delta_cpu_seconds"]) for r in rs]
        rel = [float(r["relative_wall_overhead_percent"]) for r in rs]
        ci = bootstrap_mean_ci(deltas, seed=91_000 + m)
        p = exact_two_sided_sign_flip(deltas)
        raw_p[str(m)] = p

        primitive_delta_sets = {}
        for name in ("SecSqr", "SecMulPositive", "SecDivPositive", "PubMulPositive"):
            primitive_delta_sets[name] = sorted({
                int(r["high_level_primitive_delta"][name]) for r in rs
            })

        by_m[str(m)] = {
            "n_pairs": len(rs),
            "matched_core_wall_mean_seconds": statistics.mean(core),
            "matched_core_wall_median_seconds": statistics.median(core),
            "full_wall_mean_seconds": statistics.mean(full),
            "full_wall_median_seconds": statistics.median(full),
            "paired_delta_wall_mean_seconds": statistics.mean(deltas),
            "paired_delta_wall_median_seconds": statistics.median(deltas),
            "paired_delta_wall_bootstrap95ci": list(ci),
            "paired_delta_cpu_mean_seconds": statistics.mean(cpu_delta),
            "relative_wall_overhead_mean_percent": statistics.mean(rel),
            "relative_wall_overhead_median_percent": statistics.median(rel),
            "raw_two_sided_sign_flip_p": p,
            "primitive_delta_unique_values": primitive_delta_sets,
            "transcript_entry_delta_unique_values": sorted({
                int(r["transcript_entry_delta"]) for r in rs
            }),
        }

    adjusted = holm(raw_p)
    for m, p in adjusted.items():
        by_m[m]["holm_adjusted_p"] = p
        by_m[m]["significant_alpha_0_05"] = p < ALPHA
    return by_m

def formal_run(root: Path) -> None:
    b = verify_binding(root)
    binding = b["start_binding_sha256"]
    verify_approval(root, binding)

    summary_path = root / SUMMARY_REL
    if summary_path.exists():
        s = read_json(summary_path)
        if s.get("status") == "complete":
            print("VI_E_FORMAL_ALREADY_COMPLETE=YES")
            print("SUMMARY=" + SUMMARY_REL.as_posix())
            return
        raise VIEError("INCOMPLETE_VI_E_SUMMARY_ALREADY_EXISTS")

    existing = read_jsonl(root / RAW_REL)
    ids = {r["run_id"] for r in existing}
    if len(ids) != len(existing):
        raise VIEError("VI_E_RAW_DUPLICATE_RUN_ID")
    for r in existing:
        if r.get("start_binding_sha256") != binding:
            raise VIEError("VI_E_RAW_BINDING_MISMATCH")

    failures = sum(r.get("status") != "success" for r in existing)
    started = time.time()
    wall0 = time.perf_counter()

    print("=== VI-E FORMAL MATCHED EFFICIENCY RUN ===")
    print("VI_E_FORMAL_START_GATE=PASS")
    print("START_BINDING_SHA256=" + binding)
    print("SIMULATED_THRESHOLD_N=10")
    print("SIMULATED_THRESHOLD_T=4")
    print("FIXED_POINT_F=24")
    print("PRODUCTION_MPC_USED=NO")
    print("PLANNED_MEASURED_PAIRS=120")
    print("RUNTIME_GATE_ENABLED=NO")
    print("RESUME_EXISTING_PAIRS=" + str(len(ids)))

    for m in WORKER_COUNTS:
        remaining = [
            rep for rep in range(1, MEASURED + 1)
            if f"VI-E|m={m}|rep={rep}" not in ids
        ]
        if not remaining:
            continue

        print(f"WARMUP m={m} pairs={WARMUPS}")
        for w in range(1, WARMUPS + 1):
            run_pair(m, 10_000 + w, binding, warmup=True)

        for rep in remaining:
            run_id = f"VI-E|m={m}|rep={rep}"
            try:
                row = run_pair(m, rep, binding, warmup=False)
            except Exception as exc:
                row = {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "stage": "VI-E",
                    "status": "failure",
                    "start_binding_sha256": binding,
                    "worker_count": m,
                    "dimension_D": 1,
                    "K": 10,
                    "replicate_index": rep,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "production_mpc_used": False,
                }
                failures += 1
            append_jsonl(root / RAW_REL, row)
            ids.add(run_id)
            checkpoint(root, binding, ids, failures, f"m={m}")
            print(
                f"PROGRESS VI-E {len(ids)}/120 "
                f"m={m} rep={rep}/30 failures={failures}"
            )

    if len(ids) != 120:
        raise VIEError(f"VI_E_COMPLETENESS_FAILURE:{len(ids)}/120")

    rows = read_jsonl(root / RAW_REL)
    success = [r for r in rows if r.get("status") == "success"]
    if len(success) != 120:
        # Failures are retained and summary is still written, but no favorable
        # aggregate silently drops them.
        aggregate_result = None
    else:
        aggregate_result = aggregate(success)

    summary = {
        "schema_version": "1.0",
        "experiment_namespace": "VI-E",
        "formal_experiment": True,
        "status": "complete",
        "start_binding_sha256": binding,
        "planned_measured_pairs": 120,
        "committed_measured_pairs": 120,
        "success_pairs": len(success),
        "failure_pairs": 120 - len(success),
        "deferred_pairs": 0,
        "runtime_gate_enabled": False,
        "raw_path": RAW_REL.as_posix(),
        "raw_sha256": sha(root / RAW_REL),
        "simulated_profile": {
            "N": 10,
            "T": 4,
            "fixed_fractional_bits_f": 24,
            "backend": "simulated_shamir",
            "production_mpc_used": False,
            "cryptographic_security_claim": False,
        },
        "baseline": {
            "id": "matched_pptd_core_no_longitudinal_update",
            "public_label": "Matched PPTD-core",
            "external_baseline": False,
            "fog_pptd": False,
            "fptd": False,
            "tqpp": False,
        },
        "measurement_boundaries": {
            "network_communication_bytes_measured": False,
            "online_rounds_measured": False,
            "offline_preprocessing_measured": False,
            "reason": "existing simulated backend does not model those quantities",
        },
        "aggregate_by_worker_count": aggregate_result,
        "started_unix_this_invocation": started,
        "wall_elapsed_seconds_this_invocation": time.perf_counter() - wall0,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    checkpoint(root, binding, ids, 120 - len(success), "COMPLETE")

    print("VI_E_FORMAL_STATUS=complete")
    print("COMMITTED_MEASURED_PAIRS=120")
    print("SUCCESS_PAIRS=" + str(len(success)))
    print("FAILURE_PAIRS=" + str(120 - len(success)))
    print("DEFERRED_PAIRS=0")
    print("RUNTIME_GATE_ENABLED=NO")
    print("RAW_SHA256=" + summary["raw_sha256"])
    print("SUMMARY=" + SUMMARY_REL.as_posix())
    print("VI_E_FORMAL_INVOCATION=PASS")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--preflight-and-bind", action="store_true")
    ap.add_argument("--record-approval", action="store_true")
    ap.add_argument("--binding")
    ap.add_argument("--approval-text")
    ap.add_argument("--formal-run", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    try:
        if args.preflight_and_bind:
            preflight_and_bind(root)
        elif args.record_approval:
            if not args.binding or args.approval_text is None:
                raise VIEError("VI_E_APPROVAL_ARGUMENTS_REQUIRED")
            record_approval(root, args.binding, args.approval_text)
        elif args.formal_run:
            formal_run(root)
        else:
            raise VIEError("NO_ACTION")
    except Exception as exc:
        print(f"VI_E_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
