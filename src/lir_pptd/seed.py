from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedManager:
    master_seed: bytes

    @classmethod
    def from_int(cls, master_seed: int) -> "SeedManager":
        if isinstance(master_seed, bool) or master_seed < 0:
            raise ValueError("master seed must be a nonnegative integer")
        width = max(1, (master_seed.bit_length() + 7) // 8)
        return cls(master_seed.to_bytes(width, "big"))

    def derive(self, namespace: str, *, bits: int = 64) -> int:
        if not namespace or bits <= 0 or bits > 256:
            raise ValueError("namespace must be nonempty and bits must be in 1..256")
        digest = hmac.new(
            self.master_seed,
            b"LIR-PPTD/seed-v1\x00" + namespace.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return int.from_bytes(digest, "big") >> (256 - bits)
