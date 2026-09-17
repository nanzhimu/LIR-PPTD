from __future__ import annotations

from typing import Sequence


def predict(reports: dict[str, Sequence[float | int]], modality: str) -> list[float | int]:
    if not reports:
        raise ValueError("empty reports")
    ordered = [reports[key] for key in sorted(reports)]
    width = len(ordered[0])
    if any(len(report) != width for report in ordered):
        raise ValueError("report dimensions differ")
    if modality == "numerical":
        return [sum(float(report[index]) for report in ordered) / len(ordered) for index in range(width)]
    if modality == "categorical":
        counts = [sum(int(report[index]) for report in ordered) for index in range(width)]
        return [1 if index == min(i for i, count in enumerate(counts) if count == max(counts)) else 0 for index in range(width)]
    raise ValueError(f"unsupported modality: {modality}")
