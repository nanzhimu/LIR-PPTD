from __future__ import annotations

import math
from typing import Sequence

from ..phase10_r22_metrics import one_minus_accuracy, one_minus_macro_f1


class RealMetricError(ValueError):
    pass


def mae(predictions: Sequence[float], truths: Sequence[float]) -> float:
    if len(predictions) != len(truths) or not truths:
        raise RealMetricError("prediction and truth counts must match and be non-empty")
    return sum(abs(float(p) - float(t)) for p, t in zip(predictions, truths)) / len(truths)


def rmse(predictions: Sequence[float], truths: Sequence[float]) -> float:
    if len(predictions) != len(truths) or not truths:
        raise RealMetricError("prediction and truth counts must match and be non-empty")
    return math.sqrt(sum((float(p) - float(t)) ** 2 for p, t in zip(predictions, truths)) / len(truths))


def dataset_losses(
    dataset_id: str,
    *,
    predicted_classes: Sequence[int] | None = None,
    truth_classes: Sequence[int] | None = None,
    predicted_values: Sequence[float] | None = None,
    truth_values: Sequence[float] | None = None,
) -> dict[str, float]:
    if dataset_id == "product":
        if predicted_classes is None or truth_classes is None:
            raise RealMetricError("categorical predictions required")
        return {
            "primary_loss": one_minus_macro_f1(predicted_classes, truth_classes),
            "secondary_loss": one_minus_accuracy(predicted_classes, truth_classes),
            "primary_endpoint": "one_minus_macro_f1",
            "secondary_endpoint": "one_minus_accuracy",
        }
    if dataset_id == "duck":
        if predicted_classes is None or truth_classes is None:
            raise RealMetricError("categorical predictions required")
        return {
            "primary_loss": one_minus_accuracy(predicted_classes, truth_classes),
            "secondary_loss": one_minus_macro_f1(predicted_classes, truth_classes),
            "primary_endpoint": "one_minus_accuracy",
            "secondary_endpoint": "one_minus_macro_f1",
        }
    if dataset_id == "dog":
        if predicted_classes is None or truth_classes is None:
            raise RealMetricError("categorical predictions required")
        return {
            "primary_loss": one_minus_macro_f1(predicted_classes, truth_classes),
            "secondary_loss": one_minus_accuracy(predicted_classes, truth_classes),
            "primary_endpoint": "one_minus_macro_f1",
            "secondary_endpoint": "one_minus_accuracy",
        }
    if dataset_id == "weather":
        if predicted_values is None or truth_values is None:
            raise RealMetricError("numerical predictions required")
        return {
            "primary_loss": mae(predicted_values, truth_values),
            "secondary_loss": rmse(predicted_values, truth_values),
            "primary_endpoint": "mae",
            "secondary_endpoint": "rmse",
        }
    raise RealMetricError(f"unsupported dataset: {dataset_id}")
