from __future__ import annotations
from math import fsum
from ...core.consistency_scale import resolve_consistency_scale

LAMBDA_TAU = 0.2
DOMAIN_TYPE = "numerical_box"
D = 1
_SCALE = resolve_consistency_scale(DOMAIN_TYPE, D, LAMBDA_TAU, "geometry_diameter_scaling")
TAU = float(_SCALE.resolved_tau)
EPS = 1.0 / 1024.0
ETA = 0.1
KAPPA = 2.0
K = 10

def deterministic_workload(m: int, replicate: int = 0) -> dict:
    if m < 2:
        raise ValueError("m must be >= 2")

    reports = [
        (101 + ((97 * (i + 1) + 53 * replicate + 17 * m) % 799)) / 1000.0
        for i in range(m)
    ]
    reputations = [
        (300 + ((61 * (i + 3) + 29 * replicate) % 601)) / 1000.0
        for i in range(m)
    ]
    reference = (211 + (37 * replicate + 11 * m) % 579) / 1000.0

    effective = [c + EPS for c in reputations]
    truth = fsum(a * x for a, x in zip(effective, reports)) / fsum(effective)
    for _ in range(K):
        q = [TAU / (TAU + (x - truth) ** 2) for x in reports]
        a = [e * qi for e, qi in zip(effective, q)]
        truth = fsum(ai * x for ai, x in zip(a, reports)) / fsum(a)

    qcal = [TAU / (TAU + (x - reference) ** 2) for x in reports]
    nxt = [
        (1.0 - ETA * KAPPA) * c
        + ETA * q
        + ETA * (KAPPA - 1.0) * c * q
        for c, q in zip(reputations, qcal)
    ]

    return {
        "schema_version": "1.0",
        "study_id": "p1d_mpspdz_real_e2e_v1",
        "m": m,
        "N": 10,
        "paper_reconstruction_threshold": 4,
        "mpspdz_max_corrupted_T": 3,
        "replicate": replicate,
        "consistency_scale_mode": "geometry_diameter_scaling",
        "domain_type": DOMAIN_TYPE,
        "D": D,
        "squared_diameter": float(_SCALE.squared_diameter),
        "lambda_tau": LAMBDA_TAU,
        "resolved_tau": TAU,
        "reports": reports,
        "reputations": reputations,
        "calibration_reference": reference,
        "expected_truth": truth,
        "expected_calibration_mean_reputation": fsum(nxt) / m,
    }
