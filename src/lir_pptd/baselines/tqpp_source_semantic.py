from __future__ import annotations

"""Source-semantic reproduction of the TQPP truth/reputation mechanism.

Recovered directly from the supplied manuscript:
  Eq. (17): top-h reputation-weighted truth initialization.
  Algorithm 2 / 4: CRH-style report weights and top-h weighted truth update.
  Eq. (20): increment alpha if w_i >= average weight, otherwise beta.
  Eq. (21): Beta/Thompson reputation sample from two uniform random values.

The original experiments use a small initially trusted worker set and recruit
roughly 50/289 workers in Scene 1.  The current four benchmark datasets do not
provide TQPP-native recruitment metadata, so ``TQPPAdaptationProfile`` exposes
this mapping explicitly.  The default uses the manuscript's Scene-1 ratios.
"""

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Mapping, Sequence

_EPS = 1e-15


@dataclass(frozen=True)
class TQPPAdaptationProfile:
    # Source Scene 1: 10 initially trusted workers / 289-worker pool;
    # 50 workers recruited per round.
    trusted_fraction: float = 10.0 / 289.0
    top_h_fraction: float = 50.0 / 289.0
    reputation_mode: str = "source_eq21"  # source_eq21 | posterior_mean

    def trusted_count(self, worker_count: int) -> int:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        return max(1, int(round(self.trusted_fraction * worker_count)))

    def top_h_count(self, worker_count: int) -> int:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        return max(1, int(round(self.top_h_fraction * worker_count)))


@dataclass(frozen=True)
class TQPPState:
    alpha: dict[str, int]
    beta: dict[str, int]
    reputation: dict[str, float]
    trusted_workers: frozenset[str]


@dataclass(frozen=True)
class TQPPTaskResult:
    truth_vector: tuple[float, ...]
    prediction_class: int | None
    initial_truth_vector: tuple[float, ...]
    weights: dict[str, float]
    next_state: TQPPState
    top_h_workers: tuple[str, ...]


def choose_initial_trusted_workers(
    worker_ids: Sequence[str],
    honest_workers: Sequence[str],
    coverage: Mapping[str, int],
    profile: TQPPAdaptationProfile | None = None,
) -> tuple[str, ...]:
    profile = profile or TQPPAdaptationProfile()
    honest = set(map(str, honest_workers))
    available = [str(w) for w in worker_ids if str(w) in honest]
    if not available:
        raise ValueError("TQPP requires at least one known honest worker")
    count = min(profile.trusted_count(len(worker_ids)), len(available))
    # Coverage is an availability property, not a report/truth value.  Prefer
    # known trusted workers that can actually participate in sparse datasets.
    ranked = sorted(available, key=lambda w: (-int(coverage.get(w, 0)), w))
    return tuple(ranked[:count])


def initial_state(worker_ids: Sequence[str], trusted_workers: Sequence[str]) -> TQPPState:
    ids = tuple(map(str, worker_ids))
    trusted = frozenset(map(str, trusted_workers))
    if not trusted or not trusted.issubset(set(ids)):
        raise ValueError("invalid trusted-worker set")
    alpha = {w: 1 for w in ids}
    beta = {w: 1 for w in ids}
    rep = {w: (1.0 if w in trusted else 0.5) for w in ids}
    return TQPPState(alpha=alpha, beta=beta, reputation=rep, trusted_workers=trusted)


def _seeded_uniforms(seed_namespace: str, worker_id: str, alpha: int, beta: int) -> tuple[float, float]:
    raw = hashlib.sha256(f"TQPP-v1|{seed_namespace}|{worker_id}|{alpha}|{beta}".encode()).digest()
    seed = int.from_bytes(raw[:8], "big")
    rng = random.Random(seed)
    # Avoid exact 0 which is outside the intended open interval.
    u = max(rng.random(), 2.0**-53)
    v = max(rng.random(), 2.0**-53)
    return u, v


def _source_eq21(alpha: int, beta: int, *, seed_namespace: str, worker_id: str) -> float:
    u, v = _seeded_uniforms(seed_namespace, worker_id, alpha, beta)
    x = u ** (1.0 / alpha)
    y = v ** (1.0 / beta)
    return x / (x + y)


def _posterior_mean(alpha: int, beta: int) -> float:
    return alpha / (alpha + beta)


def _class_from_vector(values: Sequence[float]) -> int:
    maximum = max(values)
    return min(i for i, value in enumerate(values) if value == maximum)


def run_task(
    reports: Mapping[str, Sequence[float | int | str]],
    modality: str,
    state: TQPPState,
    *,
    top_h_global: int,
    seed_namespace: str,
    profile: TQPPAdaptationProfile | None = None,
) -> TQPPTaskResult:
    profile = profile or TQPPAdaptationProfile()
    participants = tuple(sorted(map(str, reports)))
    if not participants:
        raise ValueError("empty reports")
    vectors = {w: tuple(float(x) for x in reports[w]) for w in participants}
    width = len(vectors[participants[0]])
    if width == 0 or any(len(vectors[w]) != width for w in participants):
        raise ValueError("report dimensions differ")

    # Source Algorithm 2: top-h workers are selected by current reputation.
    h = min(max(1, int(top_h_global)), len(participants))
    ranked = sorted(participants, key=lambda w: (-float(state.reputation[w]), w))
    top_h = tuple(ranked[:h])

    csum = sum(float(state.reputation[w]) for w in top_h)
    if csum <= 0:
        raise ValueError("nonpositive TQPP reputation sum")
    initial_truth = tuple(
        sum(vectors[w][j] * float(state.reputation[w]) for w in top_h) / csum
        for j in range(width)
    )

    distances = {
        w: sum((vectors[w][j] - initial_truth[j]) ** 2 for j in range(width))
        for w in participants
    }
    clipped = {w: max(float(d), _EPS) for w, d in distances.items()}
    total_d = sum(clipped.values())
    weights = {w: math.log(total_d) - math.log(clipped[w]) for w in participants}

    wsum = sum(weights[w] for w in top_h)
    if not math.isfinite(wsum) or wsum <= 0:
        # Degenerate all-equal reports: equal weights preserve the common value.
        weights = {w: 1.0 for w in participants}
        wsum = float(len(top_h))
    truth = tuple(
        sum(vectors[w][j] * weights[w] for w in top_h) / wsum
        for j in range(width)
    )

    # Manuscript prose for Algorithm 5 explicitly says average weight.
    average_weight = sum(weights.values()) / len(weights)
    alpha = dict(state.alpha)
    beta = dict(state.beta)
    rep = dict(state.reputation)
    for w in participants:
        if w in state.trusted_workers:
            # Definition 4: inherently trustworthy workers are preset to 1.
            rep[w] = 1.0
            continue
        if weights[w] >= average_weight:
            alpha[w] += 1
        else:
            beta[w] += 1
        if profile.reputation_mode == "source_eq21":
            rep[w] = _source_eq21(alpha[w], beta[w], seed_namespace=seed_namespace, worker_id=w)
        elif profile.reputation_mode == "posterior_mean":
            rep[w] = _posterior_mean(alpha[w], beta[w])
        else:
            raise ValueError(f"unsupported reputation mode: {profile.reputation_mode}")

    vector = tuple(float(x) for x in truth)
    if modality == "categorical":
        cls = _class_from_vector(vector)
        released = tuple(1.0 if i == cls else 0.0 for i in range(width))
        prediction_class = cls
    elif modality == "numerical":
        released = vector
        prediction_class = None
    else:
        raise ValueError(f"unsupported modality: {modality}")

    next_state = TQPPState(alpha=alpha, beta=beta, reputation=rep, trusted_workers=state.trusted_workers)
    return TQPPTaskResult(
        truth_vector=released,
        prediction_class=prediction_class,
        initial_truth_vector=initial_truth,
        weights=weights,
        next_state=next_state,
        top_h_workers=top_h,
    )
