from __future__ import annotations

import math
from typing import Sequence

BETA_MIN = 1e-12
TOLERANCE = 1e-10
MAX_ITERATIONS = 50


def predict(reports: dict[str, Sequence[float | int]], modality: str) -> list[float]:
    if not reports:
        raise ValueError("empty reports")
    ordered = [tuple(float(value) for value in reports[key]) for key in sorted(reports)]
    width = len(ordered[0])
    if any(len(report) != width for report in ordered):
        raise ValueError("report dimensions differ")
    truth = [sum(report[index] for report in ordered) / len(ordered) for index in range(width)]
    for _ in range(MAX_ITERATIONS):
        distances = [sum((report[index] - truth[index]) ** 2 for index in range(width)) for report in ordered]
        clipped = [max(distance, BETA_MIN) for distance in distances]
        total = sum(clipped)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("invalid CRH denominator")
        weights = [-math.log(value / total) for value in clipped]
        weight_total = sum(weights)
        if not math.isfinite(weight_total) or weight_total <= 0:
            raise ValueError("invalid CRH reliability weights")
        updated = [sum(weight * report[index] for weight, report in zip(weights, ordered)) / weight_total for index in range(width)]
        if max(abs(a - b) for a, b in zip(updated, truth)) <= TOLERANCE:
            truth = updated
            break
        truth = updated
    if modality == "categorical":
        maximum = max(truth)
        winner = min(index for index, value in enumerate(truth) if value == maximum)
        return [1.0 if index == winner else 0.0 for index in range(width)]
    if modality != "numerical":
        raise ValueError(f"unsupported modality: {modality}")
    return truth
