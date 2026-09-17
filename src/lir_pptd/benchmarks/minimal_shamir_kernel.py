"""Symmetric minimal Shamir semantic kernel for VI-E.1.

The purpose of this module is narrow: remove research-simulator scaffolding from
both LIR-PPTD and Fog-PPTD and realize *both* algorithm kernels with exactly the
same share/reconstruct/multiply/divide/public-scalar code paths.

This is a local semantic benchmark, not a network protocol and not production
MPC.  Nonlinear primitives privately reconstruct their operands, evaluate the
integer functionality, and deterministically re-share the result.  That is the
same minimal realization for both compared algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from itertools import combinations
from typing import Sequence

COMMON_FIELD_P = (1 << 521) - 1
DEFAULT_N = 10
DEFAULT_T = 4

ShareBundle = tuple[int, ...]


@dataclass
class KernelCounters:
    share_secret: int = 0
    reconstruct: int = 0
    local_add: int = 0
    local_sub: int = 0
    secure_mul: int = 0
    secure_sqr: int = 0
    secure_div: int = 0
    public_mul: int = 0
    public_mul_local: int = 0
    public_mul_rescaled: int = 0

    @property
    def nonlinear_or_rescaled_calls(self) -> int:
        return self.secure_mul + self.secure_sqr + self.secure_div + self.public_mul_rescaled

    @property
    def high_level_calls(self) -> int:
        return self.secure_mul + self.secure_sqr + self.secure_div + self.public_mul

    def as_dict(self) -> dict[str, int]:
        out = asdict(self)
        out["nonlinear_or_rescaled_calls"] = self.nonlinear_or_rescaled_calls
        out["high_level_calls"] = self.high_level_calls
        return out


class MinimalShamirKernel:
    """One instrumentation-minimal kernel shared by both benchmark methods."""

    def __init__(
        self,
        *,
        prime_p: int = COMMON_FIELD_P,
        committee_size_n: int = DEFAULT_N,
        threshold_t: int = DEFAULT_T,
        seed: int = 0,
    ) -> None:
        if threshold_t < 2 or threshold_t > committee_size_n:
            raise ValueError("invalid threshold")
        if prime_p <= committee_size_n or prime_p % 2 == 0:
            raise ValueError("invalid field")
        self.p = int(prime_p)
        self.n = int(committee_size_n)
        self.t = int(threshold_t)
        self.xs = tuple(range(1, self.n + 1))
        self._weights_first_t = self._lagrange_weights_at_zero(self.xs[: self.t])
        self._rng = random.Random(int(seed))
        self.counters = KernelCounters()

    def reset_counters(self) -> None:
        self.counters = KernelCounters()

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

    def _balanced(self, value: int) -> int:
        value %= self.p
        return value if value <= self.p // 2 else value - self.p

    @staticmethod
    def _trunc_toward_zero(raw: int, divisor: int) -> int:
        if divisor <= 0:
            raise ValueError("nonpositive scale")
        if raw >= 0:
            return raw // divisor
        return -((-raw) // divisor)

    def share_secret(self, value: int) -> ShareBundle:
        """Degree-(T-1) Shamir share using the common benchmark RNG."""
        coeffs = [int(value) % self.p]
        coeffs.extend(self._rng.getrandbits(self.p.bit_length()) % self.p for _ in range(1, self.t))
        shares = []
        for x in self.xs:
            acc = 0
            for c in reversed(coeffs):
                acc = (acc * x + c) % self.p
            shares.append(acc)
        self.counters.share_secret += 1
        return tuple(shares)

    def reconstruct_subset(self, bundle: ShareBundle, indices: Sequence[int]) -> int:
        if len(bundle) != self.n:
            raise ValueError("wrong committee size")
        if len(indices) < self.t:
            raise ValueError("insufficient shares")
        chosen = tuple(indices[: self.t])
        if len(set(chosen)) != len(chosen) or any(i < 0 or i >= self.n for i in chosen):
            raise ValueError("invalid indices")
        xs = tuple(self.xs[i] for i in chosen)
        weights = self._lagrange_weights_at_zero(xs)
        return sum(bundle[i] * w for i, w in zip(chosen, weights)) % self.p

    def reconstruct_raw(self, bundle: ShareBundle) -> int:
        self.counters.reconstruct += 1
        return sum(bundle[i] * self._weights_first_t[i] for i in range(self.t)) % self.p

    def reconstruct_signed(self, bundle: ShareBundle) -> int:
        return self._balanced(self.reconstruct_raw(bundle))

    def local_add(self, a: ShareBundle, b: ShareBundle) -> ShareBundle:
        self.counters.local_add += 1
        p = self.p
        return tuple((x + y) % p for x, y in zip(a, b))

    def local_sub(self, a: ShareBundle, b: ShareBundle) -> ShareBundle:
        self.counters.local_sub += 1
        p = self.p
        return tuple((x - y) % p for x, y in zip(a, b))

    def secure_mul(self, a: ShareBundle, b: ShareBundle, *, scale: int = 1) -> ShareBundle:
        """Common reconstruct/evaluate/re-share multiplication kernel."""
        x = self.reconstruct_signed(a)
        y = self.reconstruct_signed(b)
        z = self._trunc_toward_zero(x * y, scale)
        self.counters.secure_mul += 1
        return self.share_secret(z)

    def secure_sqr(self, a: ShareBundle, *, scale: int = 1) -> ShareBundle:
        x = self.reconstruct_signed(a)
        z = (x * x) // scale
        self.counters.secure_sqr += 1
        return self.share_secret(z)

    def secure_div(self, numerator: ShareBundle, denominator: ShareBundle, *, scale: int = 1) -> ShareBundle:
        """Common positive quotient kernel used by *both* algorithms."""
        u = self.reconstruct_signed(numerator)
        v = self.reconstruct_signed(denominator)
        if u < 0 or v <= 0:
            raise ValueError("positive division precondition failed")
        q = (u * scale) // v
        self.counters.secure_div += 1
        return self.share_secret(q)

    def public_mul(self, a: ShareBundle, scalar: int, *, scale: int = 1) -> ShareBundle:
        """Common public-scalar kernel.

        When no fixed-point rescaling is required (``scale == 1``), public
        multiplication is performed locally on every share, as in Shamir MPC.
        When exact integer rescaling/truncation is part of the algorithm
        functionality, the same method uses the common reconstruct/evaluate/
        re-share path.  The branch depends only on the requested arithmetic
        semantics, not on the method identity.
        """
        self.counters.public_mul += 1
        if scale == 1:
            self.counters.public_mul_local += 1
            p = self.p
            c = int(scalar) % p
            return tuple((x * c) % p for x in a)
        x = self.reconstruct_signed(a)
        z = self._trunc_toward_zero(x * int(scalar), scale)
        self.counters.public_mul_rescaled += 1
        return self.share_secret(z)

    def assert_all_threshold_subsets_reconstruct(self, secret: int) -> None:
        bundle = self.share_secret(secret)
        target = secret % self.p
        for combo in combinations(range(self.n), self.t):
            if self.reconstruct_subset(bundle, combo) != target:
                raise AssertionError(f"threshold reconstruction failed: {combo}")
