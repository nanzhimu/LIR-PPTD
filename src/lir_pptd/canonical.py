from __future__ import annotations

import json
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

_DECIMAL_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_CANONICAL_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)$")
_MAX_ABS_EXPONENT = 10_000
_MAX_POWER_OF_TWO_EXPONENT = 16_384


class CanonicalRational(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    numerator: str
    denominator: str

    @field_validator("numerator", "denominator")
    @classmethod
    def decimal_integer_string(cls, value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
            raise ValueError("canonical integer must be a decimal string without leading zeros")
        return value

    @field_validator("denominator")
    @classmethod
    def positive_denominator(cls, value: str) -> str:
        if int(value) <= 0:
            raise ValueError("denominator must be positive")
        return value

    def as_fraction(self) -> Fraction:
        return Fraction(int(self.numerator), int(self.denominator))


def _canonical_fraction(value: Fraction) -> CanonicalRational:
    if value == 0:
        return CanonicalRational(numerator="0", denominator="1")
    return CanonicalRational(numerator=str(value.numerator), denominator=str(value.denominator))


def parse_rational(value: object) -> CanonicalRational:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("algorithm parameters must not use YAML/JSON floating-point values")
    if isinstance(value, CanonicalRational):
        return _canonical_fraction(value.as_fraction())
    if isinstance(value, int):
        return _canonical_fraction(Fraction(value, 1))
    if isinstance(value, dict):
        if set(value) == {"power_of_two_exponent"}:
            exponent_literal = value["power_of_two_exponent"]
            if type(exponent_literal) is not str or not _CANONICAL_INTEGER_PATTERN.fullmatch(exponent_literal):
                raise ValueError("power-of-two exponent must be a canonical decimal integer string")
            exponent = int(exponent_literal)
            if abs(exponent) > _MAX_POWER_OF_TWO_EXPONENT:
                raise ValueError("power-of-two exponent is out of bounds")
            return _canonical_fraction(
                Fraction(2**exponent, 1) if exponent >= 0 else Fraction(1, 2 ** (-exponent))
            )
        if "power_of_two_exponent" in value:
            raise ValueError("power-of-two form cannot be combined with numerator or denominator")
        if set(value) != {"numerator", "denominator"}:
            raise ValueError("rational object requires only numerator and denominator")
        if any(type(value[key]) not in (str, int) or isinstance(value[key], bool) for key in value):
            raise ValueError("rational numerator and denominator must be integers")
        try:
            numerator = int(value["numerator"])
            denominator = int(value["denominator"])
        except (TypeError, ValueError) as exc:
            raise ValueError("rational numerator and denominator must be integers") from exc
        if denominator == 0:
            raise ValueError("denominator must not be zero")
        return _canonical_fraction(Fraction(numerator, denominator))
    if not isinstance(value, str) or not _DECIMAL_PATTERN.fullmatch(value):
        raise ValueError("parameter must be a finite decimal string, integer, or rational object")
    exponent_match = re.search(r"[eE]([+-]?\d+)$", value)
    if exponent_match and abs(int(exponent_match.group(1))) > _MAX_ABS_EXPONENT:
        raise ValueError("decimal exponent is out of bounds")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal value") from exc
    if not decimal_value.is_finite():
        raise ValueError("non-finite values are forbidden")
    exponent = decimal_value.as_tuple().exponent
    if abs(exponent) > _MAX_ABS_EXPONENT:
        raise ValueError("normalized decimal exponent is out of bounds")
    return _canonical_fraction(Fraction(decimal_value))


def canonicalize(value: Any) -> Any:
    if isinstance(value, CanonicalRational):
        return value.model_dump(mode="json")
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", str(key)): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floating-point values are forbidden")
        raise TypeError("floating-point values are forbidden in canonical semantic objects")
    if value is None or isinstance(value, (bool, int)):
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_round_trip(value: Any) -> Any:
    encoded = canonical_json_bytes(value)
    decoded = json.loads(encoded.decode("utf-8"))
    if canonical_json_bytes(decoded) != encoded:
        raise ValueError("canonical round-trip failed")
    return decoded
