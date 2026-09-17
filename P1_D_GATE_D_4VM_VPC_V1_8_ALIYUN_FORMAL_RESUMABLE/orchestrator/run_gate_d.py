#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, shlex, statistics, subprocess, time
from pathlib import Path

PARTY_PLACEMENT = {
    0: "A", 1: "A", 2: "A", 3: "A",
    4: "B", 5: "B", 6: "B",
    7: "C", 8: "C", 9: "C",
}
MODES = {"ordinary": 0, "calibration": 1}

def load_env(path: Path):
    vals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k,v=line.split("=",1)
        vals[k.strip()] = os.path.expandvars(os.path.expanduser(v.strip()))
    return vals

def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture)

def ssh_cmd(env, public_ip, command, capture=False, check=True, timeout=None):
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-i", env["SSH_KEY"],
        f'{env["SSH_USER"]}@{public_ip}',
        command
    ]
    return subprocess.run(
        cmd, check=check, text=True, capture_output=capture, timeout=timeout
    )

def scp_from(env, public_ip, remote, local):
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    return run([
        "scp", "-o", "StrictHostKeyChecking=accept-new",
        "-i", env["SSH_KEY"],
        f'{env["SSH_USER"]}@{public_ip}:{remote}',
        str(local)
    ])

def host_public(env, label):
    return env[f"COMPUTE_{label}_PUBLIC"]

def logical_private_hosts(env):
    return (
        [env["COMPUTE_A_PRIVATE"]] * 4
        + [env["COMPUTE_B_PRIVATE"]] * 3
        + [env["COMPUTE_C_PRIVATE"]] * 3
    )

def launch_party(env, party, sched, run_id, verbose=False):
    label = PARTY_PLACEMENT[party]
    host = host_public(env, label)
    log = f"$HOME/p1d_gate_d_logs/{run_id}_P{party}.log"
    cmd = (
        'cd "$HOME/MP-SPDZ"; '
        f'nohup ./shamir-party.x '
        f'-p {party} -N 10 -T 3 '
        f'-ip Player-Data/p1d_hosts10.txt '
        f'-b {env.get("P1D_BATCH_SIZE","101")} '
        f'{"-v " if verbose else ""}'
        f'{shlex.quote(sched)} '
        f'> {log} 2>&1 < /dev/null & echo $!'
    )
    r = ssh_cmd(env, host, cmd, capture=True)
    pid = int(r.stdout.strip().splitlines()[-1])
    return label, host, pid, log

def wait_remote_pid(env, host, pid, timeout_s=900):
    deadline=time.time()+timeout_s
    while time.time() < deadline:
        r=ssh_cmd(env, host, f"kill -0 {pid} 2>/dev/null", capture=True, check=False)
        if r.returncode != 0:
            return
        time.sleep(0.5)
    ssh_cmd(env, host, f"kill {pid} 2>/dev/null || true")
    raise RuntimeError(f"remote party pid {pid} timed out on {host}")

def parse_party0(text):
    def one(pattern, cast=float, flags=0):
        m=re.search(pattern,text,flags)
        return cast(m.group(1)) if m else None

    actual_triples = one(
        r"Actual preprocessing cost of program:\s*"
        r"Type int\s*([0-9]+)\s+Triples",
        int, re.S
    )
    actual_bits = one(
        r"Actual preprocessing cost of program:\s*"
        r"Type int.*?([0-9]+)\s+Bits",
        int, re.S
    )
    phase = re.search(
        r"Spent ([0-9.eE+-]+) seconds .*? on the online phase and "
        r"([0-9.eE+-]+) seconds .*? on the preprocessing/offline phase",
        text, re.S
    )

    return {
        "mpc_wall_seconds": one(r"\bTime = ([0-9.eE+-]+) seconds"),
        "cpu_seconds": one(r"\bCPU time = ([0-9.eE+-]+)"),
        "party0_data_sent_mb": one(r"\bData sent = ([0-9.eE+-]+) MB"),
        "global_data_sent_mb": one(r"\bGlobal data sent = ([0-9.eE+-]+) MB"),
        "approx_rounds": one(
            r"\bData sent = [0-9.eE+-]+ MB in ~([0-9]+) rounds", int),
        "actual_preprocessing_triples": actual_triples,
        "actual_preprocessing_bits": actual_bits,
        "triples_left": one(r"([0-9]+) triples of Shamir gfp left", int),
        "bits_left": one(r"([0-9]+) bits of Shamir gfp left", int),
        "online_phase_seconds": float(phase.group(1)) if phase else None,
        "offline_phase_seconds": float(phase.group(2)) if phase else None,
        "unused_triples_warning": "unused triples" in text.lower(),
    }

def run_cell(env, out_dir, m, mode, rep, smoke=False, verbose=False, client_timeout_s=1800):
    mode_num=MODES[mode]
    sched=f"lir_pptd_e2e-{m}-{mode_num}"
    run_id=f"{mode}_m{m}_r{rep:02d}"
    workload=f"$HOME/p1d_gate_d_workloads/workload_m{m}_r{rep:02d}.json"
    client_remote=f"$HOME/p1d_gate_d_client_results/{run_id}.json"

    parties=[]
    try:
        for p in range(10):
            parties.append(launch_party(env,p,sched,run_id,verbose=verbose))
        time.sleep(2.0)

        hosts=",".join(logical_private_hosts(env))
        client_cmd=(
            'cd "$HOME/MP-SPDZ"; '
            'python3 ExternalIO/p1d_lir_client.py '
            f'--workload {workload} '
            f'--hosts {shlex.quote(hosts)} '
            f'--mode {mode} '
            f'--out {client_remote}'
        )
        try:
            cr=ssh_cmd(
                env, env["CLIENT_PUBLIC"], client_cmd,
                capture=True, check=False, timeout=client_timeout_s
            )
        except subprocess.TimeoutExpired:
            ssh_cmd(
                env, env["CLIENT_PUBLIC"],
                "pkill -f p1d_lir_client.py || true",
                capture=True, check=False
            )
            raise RuntimeError(
                f"client timeout after {client_timeout_s}s for {run_id}"
            )
        if cr.returncode != 0:
            raise RuntimeError(f"client failed: {cr.stderr}")

        for _,host,pid,_ in parties:
            wait_remote_pid(env,host,pid)

        local_client=out_dir/f"{run_id}_client.json"
        scp_from(env, env["CLIENT_PUBLIC"],
                 f"~/p1d_gate_d_client_results/{run_id}.json", local_client)
        client=json.loads(local_client.read_text(encoding="utf-8"))

        # Collect all party logs; party 0 is parsed for canonical framework metrics.
        p0_text=None
        for p,(label,host,pid,remote_log) in enumerate(parties):
            local_log=out_dir/f"{run_id}_P{p}.log"
            remote_expanded=remote_log.replace("$HOME","~")
            scp_from(env,host,remote_expanded,local_log)
            if p==0:
                p0_text=local_log.read_text(encoding="utf-8",errors="replace")

        metrics=parse_party0(p0_text or "")
        row={
            "run_id":run_id,"m":m,"mode":mode,"replicate":rep,
            "warmup": rep < 5 and not smoke,
            "batch_size": int(env.get("P1D_BATCH_SIZE", "101")),
            "logical_parties": 10,
            "party_placement": "4+3+3",
            "paper_reconstruction_threshold": 4,
            "mpspdz_max_corrupted_T": 3,
            "mpspdz_tag": "v0.4.3",
            "mpspdz_commit": "26a605368e40fed3a7e9cee78c9a3f4390b85eb5",
            "mpspdz_source_sha256": "82c93103852a4c4c1053060f117c20fe0f0d6ad0b9ba4808b226b2dc89d93ab3",
            **client, **metrics,
        }
        tol=2e-4 if mode=="ordinary" else 5e-4
        row["correctness_pass"]=client["abs_error"] <= tol

        (out_dir/f"{run_id}_summary.json").write_text(
            json.dumps(row,indent=2,sort_keys=True),encoding="utf-8")
        return row
    finally:
        # Defensive cleanup.
        for _,host,pid,_ in parties:
            ssh_cmd(env,host,f"kill {pid} 2>/dev/null || true",capture=True,check=False)

def write_csv(rows,path):
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


FORMAL_M = (20, 50, 100, 200)
FORMAL_MODES = ("ordinary", "calibration")
FORMAL_REPS = tuple(range(25))
EXPECTED_COMMIT = "26a605368e40fed3a7e9cee78c9a3f4390b85eb5"
EXPECTED_SOURCE_SHA256 = "82c93103852a4c4c1053060f117c20fe0f0d6ad0b9ba4808b226b2dc89d93ab3"

def formal_run_id(m, mode, rep):
    return f"{mode}_m{m}_r{rep:02d}"

def expected_formal_keys():
    return [
        (m, mode, rep)
        for m in FORMAL_M
        for mode in FORMAL_MODES
        for rep in FORMAL_REPS
    ]

def validate_hygiene_proof(path: Path):
    if not path.is_file():
        raise SystemExit(f"P1D_HYGIENE_PROOF_MISSING path={path}")
    proof=json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "batch_size": 101,
        "cells": 8,
        "correctness_failures": 0,
        "unused_triples_warning_runs": 0,
        "mpspdz_tag": "v0.4.3",
        "mpspdz_commit": EXPECTED_COMMIT,
    }
    for key,value in required.items():
        if proof.get(key) != value:
            raise SystemExit(
                f"P1D_HYGIENE_PROOF_INVALID key={key} "
                f"actual={proof.get(key)!r} expected={value!r}"
            )
    return proof

def validate_completed_row(row, m, mode, rep):
    checks = {
        "run_id": formal_run_id(m,mode,rep),
        "m": m,
        "mode": mode,
        "replicate": rep,
        "batch_size": 101,
        "logical_parties": 10,
        "party_placement": "4+3+3",
        "paper_reconstruction_threshold": 4,
        "mpspdz_max_corrupted_T": 3,
        "mpspdz_tag": "v0.4.3",
        "mpspdz_commit": EXPECTED_COMMIT,
        "mpspdz_source_sha256": EXPECTED_SOURCE_SHA256,
        "correctness_pass": True,
        "unused_triples_warning": False,
    }
    for key,value in checks.items():
        if row.get(key) != value:
            return False, f"{key}: actual={row.get(key)!r} expected={value!r}"
    expected_warmup = rep < 5
    if row.get("warmup") != expected_warmup:
        return False, (
            f"warmup: actual={row.get('warmup')!r} "
            f"expected={expected_warmup!r}"
        )
    return True, ""

def load_valid_completed_rows(out: Path):
    completed={}
    invalid=[]
    for m,mode,rep in expected_formal_keys():
        p=out/f"{formal_run_id(m,mode,rep)}_summary.json"
        if not p.exists():
            continue
        try:
            row=json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append((str(p),f"json_error={exc}"))
            continue
        ok,reason=validate_completed_row(row,m,mode,rep)
        if ok:
            completed[(m,mode,rep)]=row
        else:
            invalid.append((str(p),reason))
    return completed,invalid

def write_formal_manifest(out: Path, hygiene_proof_path: Path):
    proof_sha=hashlib.sha256(hygiene_proof_path.read_bytes()).hexdigest()
    manifest={
        "schema_version":"1.0",
        "study_id":"p1d_mpspdz_real_e2e_v1",
        "formal_configuration_id":"aliyun_batch101_final",
        "batch_size":101,
        "logical_parties":10,
        "party_placement":"4+3+3",
        "paper_reconstruction_threshold":4,
        "mpspdz_max_corrupted_T":3,
        "mpspdz_tag":"v0.4.3",
        "mpspdz_commit":EXPECTED_COMMIT,
        "mpspdz_source_sha256":EXPECTED_SOURCE_SHA256,
        "m_values":list(FORMAL_M),
        "paths":list(FORMAL_MODES),
        "warmups_per_cell":5,
        "measured_per_cell":20,
        "total_runs":200,
        "measured_runs":160,
        "hygiene_proof_sha256":proof_sha,
    }
    path=out/"formal_manifest.json"
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise SystemExit("P1D_FORMAL_MANIFEST_MISMATCH")
    else:
        path.write_text(
            json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8"
        )
    return manifest

def build_formal_summary(rows, env):
    measured=[r for r in rows if not r["warmup"]]
    bad=[r for r in rows if not r["correctness_pass"]]
    warnings=[r for r in rows if r["unused_triples_warning"]]
    summary={
        "schema_version":"1.1",
        "study_id":"p1d_mpspdz_real_e2e_v1",
        "formal_configuration_id":"aliyun_batch101_final",
        "deployment":"4vm_vpc_3compute_1requester",
        "logical_parties":10,
        "physical_or_virtual_hosts_for_mpc":3,
        "party_placement":"4+3+3",
        "requester_hosts":1,
        "batch_size":int(env["P1D_BATCH_SIZE"]),
        "formal_runs":len(rows),
        "measured_runs":len(measured),
        "correctness_failures":len(bad),
        "unused_triples_warning_runs":len(warnings),
        "cells":{},
    }
    for m in FORMAL_M:
        for mode in FORMAL_MODES:
            cell=[r for r in measured if r["m"]==m and r["mode"]==mode]
            xs=[r["client_wall_seconds"] for r in cell]
            if len(cell) != 20:
                raise SystemExit(
                    f"P1D_FORMAL_CELL_COUNT_INVALID mode={mode} m={m} n={len(cell)}"
                )
            summary["cells"][f"{mode}_m{m}"]={
                "n":len(cell),
                "client_wall_mean":statistics.mean(xs),
                "client_wall_median":statistics.median(xs),
                "client_wall_p95":sorted(xs)[18],
                "global_data_sent_mb_median":statistics.median(
                    [r["global_data_sent_mb"] for r in cell
                     if r["global_data_sent_mb"] is not None]),
                "approx_rounds_median":statistics.median(
                    [r["approx_rounds"] for r in cell
                     if r["approx_rounds"] is not None]),
                "max_abs_error":max(r["abs_error"] for r in cell),
            }
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--env",default="config/gate_d_hosts.env")
    ap.add_argument("--out",default="results/p1d_gate_d_4vm_v1")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--diagnose-preprocessing",action="store_true")
    ap.add_argument("--inventory-preprocessing",action="store_true")
    ap.add_argument("--batch101-hygiene",action="store_true")
    ap.add_argument("--diag-m",type=int,choices=[20,50,100,200],default=50)
    ap.add_argument("--diag-mode",choices=["ordinary","calibration"],default="calibration")
    ap.add_argument("--formal",action="store_true")
    ap.add_argument("--resume-formal",action="store_true")
    ap.add_argument(
        "--hygiene-proof",
        default="results/p1d_gate_d_bs101_hygiene/batch101_hygiene_pass.json"
    )
    args=ap.parse_args()

    env=load_env(Path(args.env))
    configured_batch = int(env.get("P1D_BATCH_SIZE", "0"))
    if configured_batch != 101:
        raise SystemExit(
            f"P1D_BATCH_SIZE_MISMATCH configured={configured_batch} expected=101"
        )
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    selected = (
        int(args.smoke)
        + int(args.diagnose_preprocessing)
        + int(args.inventory_preprocessing)
        + int(args.batch101_hygiene)
        + int(args.formal)
    )
    if selected != 1:
        raise SystemExit(
            "choose exactly one of --smoke, --diagnose-preprocessing, "
            "--inventory-preprocessing, --batch101-hygiene, or --formal"
        )

    if args.resume_formal and not args.formal:
        raise SystemExit("--resume-formal requires --formal")

    hygiene_proof_path=Path(args.hygiene_proof)
    if args.formal:
        validate_hygiene_proof(hygiene_proof_path)
        write_formal_manifest(out,hygiene_proof_path)

    rows=[]
    if args.batch101_hygiene:
        for m in (20,50,100,200):
            for mode in ("ordinary","calibration"):
                row=run_cell(
                    env,out,m,mode,0,
                    smoke=True,verbose=True,client_timeout_s=1800
                )
                rows.append(row)
                write_csv(rows,out/"batch101_hygiene_runs.csv")
                print(
                    f"P1D_BATCH101_HYGIENE progress={len(rows)}/8 "
                    f"mode={mode} m={m} "
                    f"triples={row.get('actual_preprocessing_triples')} "
                    f"left={row.get('triples_left')} "
                    f"correct={row['correctness_pass']} "
                    f"unused={row['unused_triples_warning']} "
                    f"client={row['client_wall_seconds']:.6f}s"
                )
                if not row["correctness_pass"]:
                    print("P1D_BATCH101_HYGIENE=CORRECTNESS_FAILURE")
                    raise SystemExit(8)
                if row["unused_triples_warning"]:
                    print("P1D_BATCH101_HYGIENE=UNUSED_TRIPLES_WARNING")
                    raise SystemExit(9)

        result={
            "status":"PASS",
            "batch_size":101,
            "cells":8,
            "correctness_failures":0,
            "unused_triples_warning_runs":0,
            "mpspdz_tag":"v0.4.3",
            "mpspdz_commit":"26a605368e40fed3a7e9cee78c9a3f4390b85eb5",
            "derivation":{
                "ordinary_triples":"2m",
                "calibration_triples":"2m+1"
            }
        }
        (out/"batch101_hygiene_pass.json").write_text(
            json.dumps(result,indent=2,sort_keys=True),encoding="utf-8"
        )
        print("P1D_BATCH101_HYGIENE_RUNS=8")
        print("P1D_BATCH101_HYGIENE_BAD=0")
        print("P1D_BATCH101_HYGIENE=PASS")
        return

    if args.inventory_preprocessing:
        for m in (20,50,100,200):
            for mode in ("ordinary","calibration"):
                row=run_cell(
                    env,out,m,mode,0,
                    smoke=True,verbose=True,client_timeout_s=1800
                )
                rows.append(row)
                write_csv(rows,out/"preprocessing_inventory.csv")
                print(
                    f"P1D_PREPROCESSING_INVENTORY progress={len(rows)}/8 "
                    f"mode={mode} m={m} "
                    f"triples={row.get('actual_preprocessing_triples')} "
                    f"left={row.get('triples_left')} "
                    f"bits={row.get('actual_preprocessing_bits')} "
                    f"unused={row['unused_triples_warning']} "
                    f"client={row['client_wall_seconds']:.6f}s"
                )

        summary = {
            "status":"COMPLETE",
            "batch_size":101,
            "cells":8,
            "mpspdz_tag":"v0.4.3",
            "mpspdz_commit":"26a605368e40fed3a7e9cee78c9a3f4390b85eb5",
            "rows":[
                {
                    "m":r["m"],
                    "mode":r["mode"],
                    "actual_preprocessing_triples":r.get("actual_preprocessing_triples"),
                    "triples_left":r.get("triples_left"),
                    "actual_preprocessing_bits":r.get("actual_preprocessing_bits"),
                    "bits_left":r.get("bits_left"),
                    "unused_triples_warning":r["unused_triples_warning"],
                    "client_wall_seconds":r["client_wall_seconds"],
                    "mpc_wall_seconds":r.get("mpc_wall_seconds"),
                    "online_phase_seconds":r.get("online_phase_seconds"),
                    "offline_phase_seconds":r.get("offline_phase_seconds"),
                    "correctness_pass":r["correctness_pass"],
                }
                for r in rows
            ]
        }
        (out/"preprocessing_inventory_summary.json").write_text(
            json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"
        )
        print("P1D_PREPROCESSING_INVENTORY_RUNS=8")
        print("P1D_PREPROCESSING_INVENTORY=COMPLETE")
        return

    if args.diagnose_preprocessing:
        row=run_cell(
            env,out,args.diag_m,args.diag_mode,0,
            smoke=True,verbose=True,client_timeout_s=1800
        )
        rows.append(row)
        write_csv(rows,out/"preprocessing_diagnostic.csv")
        print(
            f"P1D_PREPROCESSING_DIAG=COMPLETE "
            f"mode={args.diag_mode} m={args.diag_m} "
            f"correct={row['correctness_pass']} "
            f"unused={row['unused_triples_warning']}"
        )
        print("P1D_PREPROCESSING_DIAG_LOG="
              f"{args.diag_mode}_m{args.diag_m}_r00_P0.log")
        return

    if args.smoke:
        # One warm-up-like run for m=20 on each path to validate multi-host
        # networking and batch-size hygiene before formal measurements.
        for mode in ("ordinary","calibration"):
            rows.append(run_cell(env,out,20,mode,0,smoke=True))
        write_csv(rows,out/"smoke_runs.csv")
        bad=[r for r in rows if (not r["correctness_pass"])
             or r["unused_triples_warning"]]
        print(f"P1D_GATE_D_SMOKE_RUNS={len(rows)}")
        print(f"P1D_GATE_D_SMOKE_BAD={len(bad)}")
        if bad:
            print("P1D_GATE_D_SMOKE=REVIEW_REQUIRED")
            raise SystemExit(3)
        print("P1D_GATE_D_SMOKE=PASS")
        return

    completed,invalid=load_valid_completed_rows(out)
    if invalid:
        print("P1D_FORMAL_INVALID_EXISTING_ROWS")
        for path,reason in invalid:
            print(f"INVALID {path}: {reason}")
        raise SystemExit(10)

    if completed and not args.resume_formal:
        raise SystemExit(
            f"P1D_FORMAL_OUTPUT_NOT_EMPTY valid_completed={len(completed)} "
            "Use --resume-formal only if continuing the same frozen batch-101 study."
        )

    if args.resume_formal:
        print(f"P1D_FORMAL_RESUME_VALID_COMPLETED={len(completed)}")

    ordered_rows=[]
    progress_completed=len(completed)

    for m,mode,rep in expected_formal_keys():
        key=(m,mode,rep)
        if key in completed:
            ordered_rows.append(completed[key])
            print(
                f"P1D_FORMAL skip_completed={len(ordered_rows)}/200 "
                f"mode={mode} m={m} rep={rep}"
            )
            continue

        row=run_cell(
            env,out,m,mode,rep,
            client_timeout_s=1800
        )
        if not row["correctness_pass"]:
            print("P1D_FORMAL_ABORT=CORRECTNESS_FAILURE")
            raise SystemExit(4)
        if row["unused_triples_warning"]:
            print("P1D_FORMAL_ABORT=UNUSED_TRIPLES_WARNING")
            raise SystemExit(5)
        ok,reason=validate_completed_row(row,m,mode,rep)
        if not ok:
            print(f"P1D_FORMAL_ABORT=ROW_VALIDATION_FAILURE reason={reason}")
            raise SystemExit(11)

        completed[key]=row
        ordered_rows.append(row)

        # Rebuild partial CSV in canonical order using every valid completed row.
        canonical_partial=[
            completed[k] for k in expected_formal_keys() if k in completed
        ]
        write_csv(canonical_partial,out/"formal_runs_partial.csv")

        print(
            f"P1D_FORMAL progress={len(completed)}/200 "
            f"mode={mode} m={m} rep={rep} "
            f"client={row['client_wall_seconds']:.6f}s "
            f"error={row['abs_error']:.3e}"
        )

    # Final canonical reconstruction prevents resume order from affecting output.
    rows=[completed[k] for k in expected_formal_keys()]
    if len(rows) != 200:
        raise SystemExit(f"P1D_FORMAL_COUNT_INVALID n={len(rows)}")

    write_csv(rows,out/"formal_runs.csv")
    summary=build_formal_summary(rows,env)
    (out/"formal_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"
    )

    print("P1D_GATE_D_FORMAL=COMPLETE")
    print("FORMAL_RUNS=200")
    print("MEASURED_RUNS=160")
    print("CORRECTNESS_FAILURES=0")
    print("UNUSED_TRIPLES_WARNING_RUNS=0")
    print("BATCH_SIZE=101")

if __name__=="__main__":
    main()
