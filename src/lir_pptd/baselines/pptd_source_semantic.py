from __future__ import annotations

"""Source-semantic plaintext reproduction of Liang et al. Fog-PPTD accuracy logic.

The paper explicitly builds PPTD on CRH.  Its secure realization replaces the
CRH logarithm with the quadratic approximation L(x)=floor((x-a2)^2/a1).
For accuracy experiments we reproduce the same vector truth-discovery semantics
in plaintext.  The global constant factor in L(.) cancels in the weighted truth
update, so the source approximation is represented by (1-d_i/sum_d)^2.

This module is for truth-accuracy comparison only.  It is not a claim to
reproduce the authors' network runtime or exact finite-field rounding.
"""

from dataclasses import dataclass
from math import isfinite, log
from typing import Mapping, Sequence

_EPS = 1e-15


@dataclass(frozen=True)
class PPTDResult:
    truth_vector: tuple[float, ...]
    prediction_class: int | None
    iterations: int
    converged: bool


def _ordered_reports(reports: Mapping[str, Sequence[float | int | str]]) -> tuple[tuple[float, ...], ...]:
    if not reports:
        raise ValueError("empty reports")
    ordered = tuple(tuple(float(x) for x in reports[w]) for w in sorted(reports))
    width = len(ordered[0])
    if width == 0 or any(len(x) != width for x in ordered):
        raise ValueError("report dimensions differ")
    return ordered


def _class_from_vector(values: Sequence[float]) -> int:
    maximum = max(values)
    return min(i for i, value in enumerate(values) if value == maximum)


def predict(
    reports: Mapping[str, Sequence[float | int | str]],
    modality: str,
    *,
    weight_mode: str = "source_quadratic",
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> PPTDResult:
    """Run the PPTD truth-functionality reproduction.

    ``source_quadratic`` follows the source paper's privacy-preserving logarithm
    approximation.  ``crh_exact`` is provided only as a conformance/reference
    mode and should not be used to invent a separate baseline.
    """
    ordered = _ordered_reports(reports)
    n_workers = len(ordered)
    width = len(ordered[0])
    truth = [sum(row[j] for row in ordered) / n_workers for j in range(width)]
    converged = False
    used = 0

    for iteration in range(1, max_iterations + 1):
        used = iteration
        distances = [sum((row[j] - truth[j]) ** 2 for j in range(width)) for row in ordered]
        total = sum(distances)
        if total <= _EPS:
            converged = True
            break

        if weight_mode == "source_quadratic":
            # Source Eq. (3): L(b) = floor((b_scaled-a2)^2/a1), with
            # b=d_i/sum_d.  The positive constant scale cancels in Eq. (2).
            weights = [(1.0 - (d / total)) ** 2 for d in distances]
        elif weight_mode == "crh_exact":
            clipped = [max(d, _EPS) for d in distances]
            clipped_total = sum(clipped)
            weights = [-log(d / clipped_total) for d in clipped]
        else:
            raise ValueError(f"unsupported PPTD weight mode: {weight_mode}")

        wsum = sum(weights)
        if not isfinite(wsum) or wsum <= 0:
            raise ValueError("invalid PPTD weight sum")
        updated = [
            sum(w * row[j] for w, row in zip(weights, ordered)) / wsum
            for j in range(width)
        ]
        delta = sum((a - b) ** 2 for a, b in zip(updated, truth))
        truth = updated
        if delta < tolerance:
            converged = True
            break

    vector = tuple(float(x) for x in truth)
    if modality == "categorical":
        cls = _class_from_vector(vector)
        # Match the paper's released categorical decision semantics.
        released = tuple(1.0 if i == cls else 0.0 for i in range(len(vector)))
        return PPTDResult(released, cls, used, converged)
    if modality != "numerical":
        raise ValueError(f"unsupported modality: {modality}")
    return PPTDResult(vector, None, used, converged)
