#!/usr/bin/env python3
"""ExternalIO benchmark client for P1-D2.

Run from the MP-SPDZ root. It reads one deterministic workload JSON,
integer-encodes the fixed-point inputs, privately sends them to the MPC parties,
and privately receives one validation/output value.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from client import Client  # MP-SPDZ ExternalIO/client.py

F = 24
SCALE = 1 << F
PORT_BASE = 14000

def encode(x: float) -> int:
    return int(round(float(x) * SCALE))

def domain_int(x) -> int:
    if hasattr(x, "v"):
        return int(x.v)
    if hasattr(x, "value"):
        return int(x.value)
    return int(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--hosts", required=True,
                    help="comma-separated host/IP list, one per logical party")
    ap.add_argument("--mode", choices=["ordinary", "calibration"], required=True)
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    hosts = [x.strip() for x in args.hosts.split(",") if x.strip()]
    if len(hosts) != int(row["N"]):
        raise SystemExit(f"host count {len(hosts)} != N={row['N']}")

    if row.get("consistency_scale_mode") != "geometry_diameter_scaling":
        raise SystemExit("workload must bind final geometry_diameter_scaling")
    tau_payload = [encode(row["resolved_tau"])]
    report_payload = [encode(x) for x in row["reports"]]
    reputation_payload = [encode(x) for x in row["reputations"]]
    reference_payload = (
        [encode(row["calibration_reference"])]
        if args.mode == "calibration"
        else []
    )

    t0 = time.perf_counter()
    c = Client(hosts, PORT_BASE, args.client_id)
    t_connected = time.perf_counter()

    # ExternalIO private-input calls are message-framed. These sends must
    # match the MPC program's receive_from_client() calls exactly.
    c.send_private_inputs(tau_payload)
    c.send_private_inputs(report_payload)
    c.send_private_inputs(reputation_payload)
    if reference_payload:
        c.send_private_inputs(reference_payload)

    t_sent = time.perf_counter()
    output = c.receive_outputs(1)
    t_done = time.perf_counter()

    raw = domain_int(output[0])
    decoded = raw / SCALE
    expected = (row["expected_truth"] if args.mode == "ordinary"
                else row["expected_calibration_mean_reputation"])
    result = {
        "schema_version": "1.0",
        "study_id": "p1d_mpspdz_real_e2e_v1",
        "mode": args.mode,
        "m": row["m"],
        "N": row["N"],
        "client_id": args.client_id,
        "private_input_batches": [1, len(report_payload), len(reputation_payload)] + ([1] if reference_payload else []),
        "consistency_scale_mode": row["consistency_scale_mode"],
        "domain_type": row["domain_type"],
        "D": row["D"],
        "squared_diameter": row["squared_diameter"],
        "lambda_tau": row["lambda_tau"],
        "resolved_tau": row["resolved_tau"],
        "connect_seconds": t_connected - t0,
        "send_seconds": t_sent - t_connected,
        "wait_output_seconds": t_done - t_sent,
        "client_wall_seconds": t_done - t0,
        "output_raw": raw,
        "output_decoded": decoded,
        "expected": expected,
        "abs_error": abs(decoded - expected),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
