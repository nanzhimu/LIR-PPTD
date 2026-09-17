"""Source-semantic secure-primitive adapter for Liang et al. Fog-PPTD.

This adapter is intentionally a *protocol-semantics simulation*, not a production
network implementation and not a claim to reproduce the authors' exact source
code.  It models the source paper's distinctive operation classes:

* T-out-of-N Shamir sharing with N=10 and T=4 by default;
* ``privMul`` via local product followed by all-dealer resharing/degree reduction;
* public-denominator ``privDiv`` with an admissible floor(x/d) representative of
  the source paper's stated floor(x/d) +/- 1 output contract;
* the quadratic logarithm approximation L(x)=floor((x-a2)^2/a1);
* mean initialization, CRH weight calculation, and weighted truth update.

The exact prime used by the paper is not published in the available source.  The
adapter therefore uses the public Mersenne prime 2^521-1 as a deterministic
source-size-class proxy.  This is deliberately different from the current
LIR-PPTD 127-bit simulated backend and therefore does *not* authorize a direct
wall-clock speedup claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from typing import Iterable, Mapping, Sequence

# Public, well-known 521-bit Mersenne prime.  It is a size-class proxy only.
FOG_PROXY_PRIME = (1 << 521) - 1
SOURCE_A2 = 10**25
SOURCE_A1 = 10 * SOURCE_A2
DEFAULT_THETA_ROUND = 10**6


@dataclass(frozen=True)
class FogAdapterProfile:
    committee_size_n: int = 10
    threshold_t: int = 4
    prime_p: int = FOG_PROXY_PRIME
    source_nominal_field_bits: int = 512
    local_proxy_field_bits: int = FOG_PROXY_PRIME.bit_length()
    field_policy: str = "source-size-class proxy; exact source prime unavailable"
    privdiv_policy: str = "deterministic floor representative within source floor(x/d)+/-1 contract"
    production_ready: bool = False
    cryptographic_security_claim: bool = False

    def __post_init__(self) -> None:
        if self.threshold_t < 2 or self.threshold_t > self.committee_size_n:
            raise ValueError("invalid threshold")
        if 2 * (self.threshold_t - 1) > self.committee_size_n - 1:
            raise ValueError("source privMul degree-reduction condition violated")
        if self.prime_p <= self.committee_size_n or self.prime_p % 2 == 0:
            raise ValueError("invalid field prime")

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["prime_p"] = str(self.prime_p)
        return d


@dataclass(frozen=True)
class FogShareBundle:
    secret_id: str
    shares: tuple[int, ...]
    epoch: int = 0


@dataclass
class FogOperationCounters:
    share_secret: int = 0
    reconstruct: int = 0
    local_add: int = 0
    local_sub: int = 0
    public_mul: int = 0
    priv_mul: int = 0
    priv_mul_interaction_phases: int = 0
    priv_div_public: int = 0
    priv_div_modeled_public_openings: int = 0
    priv_div_modeled_reshares: int = 0
    explicit_source_public_openings: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FogRunResult:
    truth_integer: int
    truth_scaled: float
    iterations: int
    theta_round: int
    weights_integer: tuple[int, ...]
    counters: dict[str, int]


class FogPPTDSemanticAdapter:
    def __init__(self, profile: FogAdapterProfile | None = None, *, master_seed: int = 0):
        self.profile = profile or FogAdapterProfile()
        self.master_seed = int(master_seed)
        self.counters = FogOperationCounters()
        self._op_index = 0
        self._xs = tuple(range(1, self.profile.committee_size_n + 1))
        self._beta = self._lagrange_weights_at_zero(self._xs)

    @property
    def p(self) -> int:
        return self.profile.prime_p

    def _next_op(self, label: str) -> str:
        self._op_index += 1
        return f"{label}:{self._op_index}"

    def _derive(self, namespace: str) -> int:
        msg = f"Fog-PPTD-adapter-v1|{self.master_seed}|{namespace}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(msg).digest(), "big")

    def _coeff(self, namespace: str, index: int) -> int:
        return self._derive(f"{namespace}|coef|{index}") % self.p

    def _lagrange_weights_at_zero(self, xs: Sequence[int]) -> tuple[int, ...]:
        p = self.p
        out: list[int] = []
        for i, xi in enumerate(xs):
            num = 1
            den = 1
            for j, xj in enumerate(xs):
                if i == j:
                    continue
                num = (num * (-xj)) % p
                den = (den * (xi - xj)) % p
            out.append((num * pow(den, -1, p)) % p)
        return tuple(out)

    def share_secret(self, value: int, *, secret_id: str, namespace: str = "share", epoch: int = 0) -> FogShareBundle:
        if not isinstance(value, int):
            raise TypeError("Fog adapter accepts integer field values only")
        coeffs = [value % self.p]
        coeffs.extend(
            self._coeff(f"{namespace}|{secret_id}|{epoch}", i)
            for i in range(1, self.profile.threshold_t)
        )
        shares = tuple(
            sum(c * pow(x, i, self.p) for i, c in enumerate(coeffs)) % self.p
            for x in self._xs
        )
        self.counters.share_secret += 1
        return FogShareBundle(secret_id=secret_id, shares=shares, epoch=epoch)

    def reconstruct_subset(self, bundle: FogShareBundle, indices: Sequence[int], *, count: bool = True) -> int:
        if len(bundle.shares) != self.profile.committee_size_n:
            raise ValueError("bundle has wrong committee size")
        if len(indices) < self.profile.threshold_t:
            raise ValueError("insufficient shares")
        if len(set(indices)) != len(indices) or any(i < 0 or i >= self.profile.committee_size_n for i in indices):
            raise ValueError("invalid share indices")
        if count:
            self.counters.reconstruct += 1
        chosen = tuple(indices[: self.profile.threshold_t])
        xs = tuple(self._xs[i] for i in chosen)
        ys = tuple(bundle.shares[i] for i in chosen)
        weights = self._lagrange_weights_at_zero(xs)
        return sum(y * w for y, w in zip(ys, weights)) % self.p

    def reconstruct(self, bundle: FogShareBundle, *, count: bool = True) -> int:
        return self.reconstruct_subset(bundle, tuple(range(self.profile.threshold_t)), count=count)

    def local_add(self, a: FogShareBundle, b: FogShareBundle, *, secret_id: str | None = None) -> FogShareBundle:
        self._compatible(a, b)
        self.counters.local_add += 1
        return FogShareBundle(
            secret_id=secret_id or self._next_op("add"),
            shares=tuple((x + y) % self.p for x, y in zip(a.shares, b.shares)),
            epoch=a.epoch,
        )

    def local_sub(self, a: FogShareBundle, b: FogShareBundle, *, secret_id: str | None = None) -> FogShareBundle:
        self._compatible(a, b)
        self.counters.local_sub += 1
        return FogShareBundle(
            secret_id=secret_id or self._next_op("sub"),
            shares=tuple((x - y) % self.p for x, y in zip(a.shares, b.shares)),
            epoch=a.epoch,
        )

    def public_mul(self, a: FogShareBundle, c: int, *, secret_id: str | None = None) -> FogShareBundle:
        if c < 0:
            raise ValueError("public multiplier must be nonnegative in this gate")
        self.counters.public_mul += 1
        return FogShareBundle(
            secret_id=secret_id or self._next_op("pubmul"),
            shares=tuple((x * c) % self.p for x in a.shares),
            epoch=a.epoch,
        )

    def _compatible(self, a: FogShareBundle, b: FogShareBundle) -> None:
        if a.epoch != b.epoch or len(a.shares) != len(b.shares):
            raise ValueError("incompatible bundles")

    def priv_mul(self, a: FogShareBundle, b: FogShareBundle, *, secret_id: str | None = None) -> FogShareBundle:
        """Model Algorithm 1: local product, beta_j weighting, all-dealer resharing."""
        self._compatible(a, b)
        p = self.p
        op = self._next_op("privMul")
        dealer_constants = [
            (a.shares[j] * b.shares[j] * self._beta[j]) % p
            for j in range(self.profile.committee_size_n)
        ]
        reshared_by_dealer: list[tuple[int, ...]] = []
        for j, z_j in enumerate(dealer_constants):
            # Each server re-shares its local beta-weighted product with degree T-1.
            coeffs = [z_j]
            coeffs.extend(
                self._coeff(f"{op}|dealer={j}", degree)
                for degree in range(1, self.profile.threshold_t)
            )
            dealer_shares = tuple(
                sum(c * pow(x, degree, p) for degree, c in enumerate(coeffs)) % p
                for x in self._xs
            )
            reshared_by_dealer.append(dealer_shares)
        output_shares = tuple(
            sum(reshared_by_dealer[dealer][receiver] for dealer in range(self.profile.committee_size_n)) % p
            for receiver in range(self.profile.committee_size_n)
        )
        self.counters.priv_mul += 1
        self.counters.priv_mul_interaction_phases += 2
        return FogShareBundle(secret_id=secret_id or op, shares=output_shares, epoch=a.epoch)

    def priv_div_public(self, a: FogShareBundle, d: int, *, secret_id: str | None = None) -> FogShareBundle:
        """Model the source public-denominator division functionality.

        The paper states that omitting Compare() yields a share of floor(x/d) with
        an error bounded by +/-1.  For deterministic conformance we select the exact
        floor as one admissible representative.  We still count the source-shaped
        public opening and output resharing phases; this is not a reproduction of
        the paper's exact masking distribution.
        """
        if not isinstance(d, int) or d <= 0:
            raise ValueError("public divisor must be a positive integer")
        x = self.reconstruct(a, count=False)
        q = x // d
        op = self._next_op("privDivPublic")
        out = self.share_secret(q, secret_id=secret_id or op, namespace=f"{op}|dealer-reshare", epoch=a.epoch)
        self.counters.priv_div_public += 1
        self.counters.priv_div_modeled_public_openings += 1
        self.counters.priv_div_modeled_reshares += 1
        return out

    def reveal_source_public(self, a: FogShareBundle) -> int:
        self.counters.explicit_source_public_openings += 1
        return self.reconstruct(a, count=False)

    def mean_initialize(self, reports: Sequence[FogShareBundle]) -> FogShareBundle:
        if not reports:
            raise ValueError("empty reports")
        total = reports[0]
        for r in reports[1:]:
            total = self.local_add(total, r)
        return self.priv_div_public(total, len(reports), secret_id="truth:init")

    def quadratic_log_weight(self, distance: FogShareBundle, distance_sum_public: int, *, worker_index: int) -> FogShareBundle:
        if distance_sum_public <= 0:
            raise ValueError("distance sum must be positive")
        # b_scaled = floor(distance * a2 / sum_distance), so 0 <= b_scaled <= a2.
        numerator = self.public_mul(distance, SOURCE_A2, secret_id=f"ratio-num:{worker_index}")
        b_scaled = self.priv_div_public(numerator, distance_sum_public, secret_id=f"ratio:{worker_index}")
        a2_share = self.share_secret(SOURCE_A2, secret_id=f"a2:{worker_index}", namespace="public-constant")
        diff = self.local_sub(b_scaled, a2_share, secret_id=f"log-diff:{worker_index}")
        squared = self.priv_mul(diff, diff, secret_id=f"log-square:{worker_index}")
        return self.priv_div_public(squared, SOURCE_A1, secret_id=f"weight:{worker_index}")

    def calc_weights(self, reports: Sequence[FogShareBundle], truth: FogShareBundle) -> tuple[FogShareBundle, ...]:
        distances: list[FogShareBundle] = []
        for i, report in enumerate(reports):
            diff = self.local_sub(report, truth, secret_id=f"distance-diff:{i}")
            distances.append(self.priv_mul(diff, diff, secret_id=f"distance:{i}"))
        total = distances[0]
        for d in distances[1:]:
            total = self.local_add(total, d)
        s1 = self.reveal_source_public(total)
        if s1 == 0:
            # Degenerate all-equal case. CRH weights are undefined; equal positive
            # weights preserve the common truth and make the adapter total.
            return tuple(
                self.share_secret(1, secret_id=f"weight:{i}", namespace="degenerate-weight")
                for i in range(len(reports))
            )
        return tuple(
            self.quadratic_log_weight(d, s1, worker_index=i)
            for i, d in enumerate(distances)
        )

    def calc_truth(self, reports: Sequence[FogShareBundle], weights: Sequence[FogShareBundle]) -> FogShareBundle:
        if len(reports) != len(weights) or not reports:
            raise ValueError("report/weight cardinality mismatch")
        products = [
            self.priv_mul(w, x, secret_id=f"weighted-report:{i}")
            for i, (w, x) in enumerate(zip(weights, reports))
        ]
        numerator = products[0]
        denominator = weights[0]
        for prod in products[1:]:
            numerator = self.local_add(numerator, prod)
        for w in weights[1:]:
            denominator = self.local_add(denominator, w)
        weight_sum = self.reveal_source_public(denominator)
        if weight_sum <= 0:
            raise ValueError("nonpositive weight sum")
        return self.priv_div_public(numerator, weight_sum, secret_id="truth:update")

    def run_d1(self, reports_integer: Sequence[int], *, iterations: int = 10, theta_round: int = DEFAULT_THETA_ROUND) -> FogRunResult:
        if not reports_integer:
            raise ValueError("empty reports")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if any((not isinstance(x, int)) or x < 0 for x in reports_integer):
            raise ValueError("reports must be nonnegative integers")
        bundles = tuple(
            self.share_secret(x, secret_id=f"report:{i}", namespace="worker-upload")
            for i, x in enumerate(reports_integer)
        )
        truth = self.mean_initialize(bundles)
        weights: tuple[FogShareBundle, ...] = tuple()
        for _ in range(iterations):
            weights = self.calc_weights(bundles, truth)
            truth = self.calc_truth(bundles, weights)
        truth_i = self.reconstruct(truth)
        return FogRunResult(
            truth_integer=truth_i,
            truth_scaled=truth_i / theta_round,
            iterations=iterations,
            theta_round=theta_round,
            weights_integer=tuple(self.reconstruct(w, count=False) for w in weights),
            counters=self.counters.as_dict(),
        )


def quadratic_log_integer(x: int, *, a1: int = SOURCE_A1, a2: int = SOURCE_A2) -> int:
    if x < 0 or x > a2:
        raise ValueError("x outside source approximation domain")
    return ((x - a2) ** 2) // a1


def plaintext_integer_reference_d1(
    reports_integer: Sequence[int],
    *,
    iterations: int = 10,
    a1: int = SOURCE_A1,
    a2: int = SOURCE_A2,
) -> int:
    """Integer reference matching the adapter's deterministic admissible privDiv representative."""
    if not reports_integer:
        raise ValueError("empty reports")
    truth = sum(reports_integer) // len(reports_integer)
    for _ in range(iterations):
        distances = [(x - truth) ** 2 for x in reports_integer]
        s1 = sum(distances)
        if s1 == 0:
            return truth
        weights = []
        for d in distances:
            b_scaled = (d * a2) // s1
            weights.append(quadratic_log_integer(b_scaled, a1=a1, a2=a2))
        wsum = sum(weights)
        if wsum <= 0:
            raise ValueError("nonpositive reference weight sum")
        truth = sum(w * x for w, x in zip(weights, reports_integer)) // wsum
    return truth
