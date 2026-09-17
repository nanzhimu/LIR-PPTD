from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

ConsistencyScaleMode = Literal["legacy_D_scaling", "geometry_diameter_scaling"]
DomainType = Literal["numerical_box", "categorical_one_hot"]

LEGACY_D_SCALING: ConsistencyScaleMode = "legacy_D_scaling"
GEOMETRY_DIAMETER_SCALING: ConsistencyScaleMode = "geometry_diameter_scaling"
DEFAULT_CONSISTENCY_SCALE_MODE: ConsistencyScaleMode = GEOMETRY_DIAMETER_SCALING


class ConsistencyScaleError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedConsistencyScale:
    consistency_scale_mode: ConsistencyScaleMode
    domain_type: DomainType
    D: int
    squared_diameter: Fraction
    lambda_tau: Fraction
    resolved_tau: Fraction

    def metadata(self) -> dict[str, str | int]:
        def f(x: Fraction) -> str:
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
        return {
            "consistency_scale_mode": self.consistency_scale_mode,
            "domain_type": self.domain_type,
            "D": self.D,
            "squared_diameter": f(self.squared_diameter),
            "lambda_tau": f(self.lambda_tau),
            "resolved_tau": f(self.resolved_tau),
        }


def _frac(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def domain_type_from_modality(modality: str) -> DomainType:
    if modality == "numerical":
        return "numerical_box"
    if modality == "categorical":
        return "categorical_one_hot"
    raise ConsistencyScaleError(f"unsupported modality: {modality}")


def squared_diameter(domain_spec: DomainType, D: int) -> Fraction:
    if D < 1:
        raise ConsistencyScaleError("D must be positive")
    if domain_spec == "numerical_box":
        # Frozen GEO-TAU scope: normalized numerical box [0,1]^D.
        return Fraction(D, 1)
    if domain_spec == "categorical_one_hot":
        if D < 2:
            raise ConsistencyScaleError("one-hot categorical domain requires D>=2")
        return Fraction(2, 1)
    raise ConsistencyScaleError(f"unsupported domain_spec: {domain_spec}")


def resolve_consistency_scale(
    domain_spec: DomainType,
    D: int,
    lambda_tau: Any,
    mode: ConsistencyScaleMode = DEFAULT_CONSISTENCY_SCALE_MODE,
) -> ResolvedConsistencyScale:
    lam = _frac(lambda_tau)
    if lam <= 0:
        raise ConsistencyScaleError("lambda_tau must be positive")
    diam2 = squared_diameter(domain_spec, D)
    if mode == LEGACY_D_SCALING:
        tau = lam * D
    elif mode == GEOMETRY_DIAMETER_SCALING:
        tau = lam * diam2
    else:
        raise ConsistencyScaleError(f"unsupported consistency_scale_mode: {mode}")
    return ResolvedConsistencyScale(mode, domain_spec, D, diam2, lam, tau)


def resolve_from_modality(
    modality: str,
    D: int,
    lambda_tau: Any,
    mode: ConsistencyScaleMode = DEFAULT_CONSISTENCY_SCALE_MODE,
) -> ResolvedConsistencyScale:
    return resolve_consistency_scale(domain_type_from_modality(modality), D, lambda_tau, mode)


def squared_distance_bound_from_modality(modality: str, D: int) -> Fraction:
    """Public squared-distance bound for the frozen normalized report domain.

    This is the same geometric quantity used to resolve the final consistency
    scale: D on [0,1]^D and 2 on a D-class one-hot simplex.
    """
    return squared_diameter(domain_type_from_modality(modality), D)
