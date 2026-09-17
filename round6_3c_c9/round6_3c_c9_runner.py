#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, hmac, json, math, os, sys, time
from pathlib import Path
import numpy as np

STUDY_ID = "round6_3c_c9_tau_dimension_v1"
D_VALUES = (2,4,8,16,32)
TAU_MODES = ("dimension","diameter")
SCENARIOS = ("binary_support","full_support")
RHOS = (0.3,0.5,0.7)
SEEDS = [int.from_bytes(hashlib.sha256(f"LIR-PPTD-C9-SEED-{i:02d}".encode()).digest()[:8],"big") & ((1<<63)-1) for i in range(1,21)]

K=10
C0=0.5
EPS=2**-10
KAPPA=2.0
ETA=0.1
LAMBDA_TAU=0.2
PI_CAL=0.05
TASKS=200
WORKERS=40
P_HONEST_CORRECT=0.85
P_MAL_CORRECT=0.10

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def selector_mask(master_seed: int) -> np.ndarray:
    key=hashlib.sha256(f"C9-selector-key-{master_seed}".encode()).digest()
    threshold=int(PI_CAL*(1<<256))
    out=[]
    for t in range(TASKS):
        tid=f"c9-task-{t:06d}".encode()
        v=int.from_bytes(hmac.new(key,tid,hashlib.sha256).digest(),"big")
        out.append(v < threshold)
    return np.asarray(out,dtype=bool)

def tau_value(D: int, mode: str) -> float:
    if mode=="dimension":
        return LAMBDA_TAU*D
    if mode=="diameter":
        return LAMBDA_TAU*2.0
    raise ValueError(mode)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, D: int) -> float:
    vals=[]
    for c in range(D):
        tp=np.sum((y_true==c)&(y_pred==c))
        fp=np.sum((y_true!=c)&(y_pred==c))
        fn=np.sum((y_true==c)&(y_pred!=c))
        p=tp/(tp+fp) if (tp+fp)>0 else 0.0
        r=tp/(tp+fn) if (tp+fn)>0 else 0.0
        vals.append(2*p*r/(p+r) if (p+r)>0 else 0.0)
    return float(np.mean(vals))

def make_reports(seed: int, D: int, rho: float, scenario: str):
    rng=np.random.default_rng(seed)
    # Common random objects; malicious cohorts are nested by this frozen ranking.
    worker_rank=rng.permutation(WORKERS)
    mal_n=int(round(rho*WORKERS))
    is_mal=np.zeros(WORKERS,dtype=bool)
    is_mal[worker_rank[:mal_n]]=True

    truth_u=rng.random(TASKS)
    correctness_u=rng.random((TASKS,WORKERS))
    wrong_u=rng.random((TASKS,WORKERS))

    truth=np.floor(truth_u*D).astype(int)
    truth=np.minimum(truth,D-1)
    labels=np.empty((TASKS,WORKERS),dtype=np.int16)

    for t in range(TASKS):
        y=int(truth[t])
        adjacent=(y+1)%D
        for w in range(WORKERS):
            if is_mal[w]:
                correct=correctness_u[t,w] < P_MAL_CORRECT
                labels[t,w] = y if correct else adjacent
            else:
                correct=correctness_u[t,w] < P_HONEST_CORRECT
                if correct:
                    labels[t,w]=y
                else:
                    if scenario=="binary_support":
                        labels[t,w]=adjacent
                    elif scenario=="full_support":
                        j=min(int(wrong_u[t,w]*(D-1)),D-2)
                        labels[t,w]=j if j<y else j+1
                    else:
                        raise ValueError(scenario)
    return truth, labels, is_mal

def run_stream(seed: int, D: int, rho: float, scenario: str, tau_mode: str):
    truth, labels, is_mal = make_reports(seed,D,rho,scenario)
    cal=selector_mask(seed)
    tau=tau_value(D,tau_mode)
    c=np.full(WORKERS,C0,dtype=float)

    y_true=[]
    y_pred=[]
    brier=[]
    true_mass=[]
    honest_qcal=[]
    malicious_qcal=[]

    eye=np.eye(D,dtype=float)

    for t in range(TASKS):
        X=eye[labels[t]]
        y=int(truth[t])
        if cal[t]:
            ref=eye[y]
            d=np.sum((X-ref)**2,axis=1)
            q=tau/(tau+d)
            honest_qcal.extend(q[~is_mal].tolist())
            malicious_qcal.extend(q[is_mal].tolist())
            c = c + ETA*((1.0-c)*q - KAPPA*c*(1.0-q))
            continue

        weights=c+EPS
        xhat=np.sum(weights[:,None]*X,axis=0)/np.sum(weights)
        for _ in range(K):
            d=np.sum((X-xhat)**2,axis=1)
            q=tau/(tau+d)
            a=(c+EPS)*q
            xhat=np.sum(a[:,None]*X,axis=0)/np.sum(a)

        pred=int(np.argmax(xhat)) # numpy argmax returns lowest index on ties
        target=eye[y]
        y_true.append(y)
        y_pred.append(pred)
        brier.append(float(np.sum((xhat-target)**2)))
        true_mass.append(float(xhat[y]))

    y_true=np.asarray(y_true,dtype=int)
    y_pred=np.asarray(y_pred,dtype=int)
    acc=float(np.mean(y_true==y_pred))
    hmean=float(np.mean(c[~is_mal]))
    mmean=float(np.mean(c[is_mal]))
    return {
        "study_id":STUDY_ID,
        "seed":int(seed),
        "D":int(D),
        "rho":float(rho),
        "scenario":scenario,
        "tau_mode":tau_mode,
        "tau":float(tau),
        "q_wrong_vertex":float(tau/(tau+2.0)),
        "calibration_count":int(np.sum(cal)),
        "ordinary_count":int(np.sum(~cal)),
        "accuracy":acc,
        "classification_loss":1.0-acc,
        "macro_f1":macro_f1(y_true,y_pred,D),
        "brier_loss":float(np.mean(brier)),
        "mean_true_mass":float(np.mean(true_mass)),
        "final_honest_reputation":hmean,
        "final_malicious_reputation":mmean,
        "final_reputation_gap":hmean-mmean,
        "mean_honest_qcal":float(np.mean(honest_qcal)) if honest_qcal else None,
        "mean_malicious_qcal":float(np.mean(malicious_qcal)) if malicious_qcal else None,
        "status":"success",
    }

def load_done(path: Path):
    done=set()
    if not path.is_file(): return done
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            done.add((r["seed"],r["D"],r["rho"],r["scenario"],r["tau_mode"]))
    return done

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",required=True)
    ap.add_argument("--out-dir",default="results/round6_3c_c9_tau_dimension_v1")
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    root=Path(args.repo_root).resolve()
    out=Path(args.out_dir)
    if not out.is_absolute(): out=root/out
    out.mkdir(parents=True,exist_ok=True)
    raw=out/"formal_runs.jsonl"

    dims=(2,4) if args.smoke else D_VALUES
    rhos=(0.3,) if args.smoke else RHOS
    seeds=SEEDS[:2] if args.smoke else SEEDS
    scenarios=SCENARIOS
    modes=TAU_MODES
    planned=len(dims)*len(rhos)*len(seeds)*len(scenarios)*len(modes)

    done=load_done(raw)
    success=0
    t0=time.time()
    for seed in seeds:
        for rho in rhos:
            for scenario in scenarios:
                for D in dims:
                    # Generate paired tau-mode results from identical reports.
                    for mode in modes:
                        key=(seed,D,rho,scenario,mode)
                        if key in done:
                            success+=1
                            continue
                        row=run_stream(seed,D,rho,scenario,mode)
                        with raw.open("a",encoding="utf-8") as f:
                            f.write(json.dumps(row,sort_keys=True)+"\n")
                        success+=1
                        if success%50==0 or success==planned:
                            print(f"PROGRESS={success}/{planned}",flush=True)

    summary={
        "study_id":STUDY_ID,
        "status":"SMOKE_COMPLETE" if args.smoke else "FORMAL_COMPLETE",
        "planned_runs":planned,
        "success":success,
        "failure":0,
        "elapsed_seconds":time.time()-t0,
        "result_file":"formal_runs.jsonl",
        "result_sha256":sha256_file(raw),
        "dimensions":list(dims),
        "rhos":list(rhos),
        "scenarios":list(scenarios),
        "tau_modes":list(modes),
        "seed_count":len(seeds),
    }
    (out/"execution_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("ROUND6_3C_C9_EXECUTION="+summary["status"])
    print("RUNS="+str(success))
    print("RAW_SHA256="+summary["result_sha256"])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
