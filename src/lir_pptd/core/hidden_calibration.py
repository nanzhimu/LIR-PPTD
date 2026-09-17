from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Iterable, Sequence

from ..canonical import canonical_json_bytes

_SELECTOR_SCHEMA = "lir-pptd-hidden-calibration-v1"
_COMMIT_DOMAIN = "LIR-PPTD/CALIBRATION-SEED-COMMIT/V1"
_PRF_DOMAIN = "LIR-PPTD/CALIBRATION-DESIGNATION/V1"


@dataclass(frozen=True)
class HiddenCalibrationCommitment:
    """Public commitment to one hidden calibration-selector key."""

    schema_version: str
    session_id: str
    selector_epoch: int
    config_hash: str
    probability_numerator: int
    probability_denominator: int
    seed_commitment_sha256: str


@dataclass(frozen=True)
class CalibrationDesignation:
    """Mode credential payload released only after participant freezing."""

    schema_version: str
    session_id: str
    selector_epoch: int
    config_hash: str
    task_id: str
    participant_hash: str
    is_calibration: bool
    seed_commitment_sha256: str


class HiddenCalibrationSchedule:
    """Secret-keyed, report-independent calibration designation.

    Selection is a PRF-style Bernoulli decision over the immutable task ID.
    Using the task ID rather than its current sequence position ensures that a
    task-order permutation does not silently change which underlying tasks are
    calibration tasks.  The selector key is committed before the active epoch
    and remains hidden from workers until the epoch has retired.

    `period` is a nominal interval: the calibration probability is 1 / period.
    """

    def __init__(
        self,
        *,
        seed: bytes,
        session_id: str,
        selector_epoch: int,
        config_hash: str,
        period: int = 20,
    ) -> None:
        if not isinstance(seed, (bytes, bytearray)) or len(seed) < 32:
            raise ValueError("hidden calibration seed must contain at least 256 bits")
        if not session_id:
            raise ValueError("session_id must be nonempty")
        if selector_epoch < 0:
            raise ValueError("selector_epoch must be nonnegative")
        if not config_hash:
            raise ValueError("config_hash must be nonempty")
        if period <= 0:
            raise ValueError("period must be positive")
        self._seed = bytes(seed)
        self.session_id = str(session_id)
        self.selector_epoch = int(selector_epoch)
        self.config_hash = str(config_hash)
        self.period = int(period)

    @property
    def probability_numerator(self) -> int:
        return 1

    @property
    def probability_denominator(self) -> int:
        return self.period

    def _commitment_preimage(self) -> bytes:
        return canonical_json_bytes(
            {
                "domain": _COMMIT_DOMAIN,
                "schema_version": _SELECTOR_SCHEMA,
                "session_id": self.session_id,
                "selector_epoch": self.selector_epoch,
                "config_hash": self.config_hash,
                "probability_numerator": 1,
                "probability_denominator": self.period,
                "seed_hex": self._seed.hex(),
            }
        )

    @property
    def seed_commitment_sha256(self) -> str:
        return hashlib.sha256(self._commitment_preimage()).hexdigest()

    def public_commitment(self) -> HiddenCalibrationCommitment:
        return HiddenCalibrationCommitment(
            schema_version=_SELECTOR_SCHEMA,
            session_id=self.session_id,
            selector_epoch=self.selector_epoch,
            config_hash=self.config_hash,
            probability_numerator=1,
            probability_denominator=self.period,
            seed_commitment_sha256=self.seed_commitment_sha256,
        )

    def _prf_digest(self, task_id: str) -> bytes:
        if not task_id:
            raise ValueError("task_id must be nonempty")
        message = canonical_json_bytes(
            {
                "domain": _PRF_DOMAIN,
                "schema_version": _SELECTOR_SCHEMA,
                "session_id": self.session_id,
                "selector_epoch": self.selector_epoch,
                "config_hash": self.config_hash,
                "task_id": str(task_id),
            }
        )
        return hmac.new(self._seed, message, hashlib.sha256).digest()

    def is_calibration_task(self, task_id: str) -> bool:
        digest_value = int.from_bytes(self._prf_digest(task_id), "big")
        return digest_value * self.period < (1 << 256)

    def designation_after_freeze(
        self,
        *,
        task_id: str,
        participant_hash: str,
    ) -> CalibrationDesignation:
        if not participant_hash:
            raise ValueError("participant_hash must be nonempty")
        return CalibrationDesignation(
            schema_version=_SELECTOR_SCHEMA,
            session_id=self.session_id,
            selector_epoch=self.selector_epoch,
            config_hash=self.config_hash,
            task_id=str(task_id),
            participant_hash=str(participant_hash),
            is_calibration=self.is_calibration_task(task_id),
            seed_commitment_sha256=self.seed_commitment_sha256,
        )

    def reveal_retired_seed_hex(self) -> str:
        """Return the selector seed only after this selector epoch is retired."""

        return self._seed.hex()

    @classmethod
    def from_revealed_seed(
        cls,
        *,
        seed_hex: str,
        commitment: HiddenCalibrationCommitment,
    ) -> "HiddenCalibrationSchedule":
        selector = cls(
            seed=bytes.fromhex(seed_hex),
            session_id=commitment.session_id,
            selector_epoch=commitment.selector_epoch,
            config_hash=commitment.config_hash,
            period=commitment.probability_denominator,
        )
        if commitment.probability_numerator != 1:
            raise ValueError("unsupported calibration probability numerator")
        if selector.seed_commitment_sha256 != commitment.seed_commitment_sha256:
            raise ValueError("calibration seed does not match public commitment")
        return selector

    def selected_task_ids(self, task_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(str(task_id) for task_id in task_ids if self.is_calibration_task(str(task_id)))

    def mask(self, task_ids: Sequence[str]) -> tuple[bool, ...]:
        return tuple(self.is_calibration_task(str(task_id)) for task_id in task_ids)
