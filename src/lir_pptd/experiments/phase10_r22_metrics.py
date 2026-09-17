from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import TypeAlias

NumericValue: TypeAlias = Decimal | float | int | str


def _decimal(value: NumericValue) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc


def argmax_zero_based(values: Sequence[NumericValue]) -> int:
    """Return the smallest zero-based index attaining the exact maximum.

    Decimal conversion is intentional: Phase10 exact outputs are high-precision
    decimal strings, and converting them through binary float can manufacture a
    false tie (especially in synthetic categorical tie conditions).
    """
    if not values:
        raise ValueError("class score vector must not be empty")
    numeric = [_decimal(value) for value in values]
    maximum = max(numeric)
    return next(index for index, value in enumerate(numeric) if value == maximum)


def numerical_abs_error_sum(
    prediction: Sequence[NumericValue],
    truth: Sequence[NumericValue],
) -> Decimal:
    if len(prediction) != len(truth) or not truth:
        raise ValueError("prediction and truth dimensions must match and be non-empty")
    return sum(
        (abs(_decimal(predicted) - _decimal(expected)) for predicted, expected in zip(prediction, truth)),
        start=Decimal(0),
    )


def numerical_mae_from_abs_error_sums(
    abs_error_sums: Sequence[NumericValue],
    dimensions: Sequence[int],
) -> Decimal:
    if len(abs_error_sums) != len(dimensions) or not dimensions:
        raise ValueError("error sums and dimensions must match and be non-empty")
    denominator = sum(dimensions)
    if denominator <= 0:
        raise ValueError("total numerical dimension must be positive")
    numerator = sum((_decimal(value) for value in abs_error_sums), start=Decimal(0))
    return numerator / Decimal(denominator)


def numerical_mae(
    predictions: Sequence[Sequence[NumericValue]],
    truths: Sequence[Sequence[NumericValue]],
) -> Decimal:
    if len(predictions) != len(truths) or not truths:
        raise ValueError("prediction and truth task counts must match and be non-empty")
    sums = [numerical_abs_error_sum(prediction, truth) for prediction, truth in zip(predictions, truths)]
    return numerical_mae_from_abs_error_sums(sums, [len(truth) for truth in truths])


def one_minus_accuracy(predictions: Sequence[int], truths: Sequence[int]) -> float:
    if len(predictions) != len(truths) or not truths:
        raise ValueError("prediction and truth class counts must match and be non-empty")
    correct = sum(predicted == truth for predicted, truth in zip(predictions, truths))
    return 1.0 - correct / len(truths)


def one_minus_macro_f1(predictions: Sequence[int], truths: Sequence[int]) -> float:
    """Return 1-MacroF1 over classes with positive true support only."""
    if len(predictions) != len(truths) or not truths:
        raise ValueError("prediction and truth class counts must match and be non-empty")
    supported_classes = sorted(set(truths))
    f1_values: list[float] = []
    for class_index in supported_classes:
        true_positive = sum(
            predicted == class_index and truth == class_index
            for predicted, truth in zip(predictions, truths)
        )
        false_positive = sum(
            predicted == class_index and truth != class_index
            for predicted, truth in zip(predictions, truths)
        )
        false_negative = sum(
            predicted != class_index and truth == class_index
            for predicted, truth in zip(predictions, truths)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        # Because class_index comes from set(truths), true support is positive,
        # so denominator cannot be zero. Keep the guard fail-closed anyway.
        if denominator <= 0:
            raise ValueError("positive-support class has undefined F1 denominator")
        f1_values.append(2 * true_positive / denominator)
    return 1.0 - sum(f1_values) / len(f1_values)


def categorical_primary_loss(family: str, predictions: Sequence[int], truths: Sequence[int]) -> float:
    if family == "synthetic_cat_binary":
        return one_minus_accuracy(predictions, truths)
    if family == "synthetic_cat_tie":
        return one_minus_macro_f1(predictions, truths)
    raise ValueError(f"unsupported categorical family: {family}")
