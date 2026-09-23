from __future__ import annotations
from pathlib import Path
import csv, json, subprocess, sys, tempfile, shutil

ROOT=Path(__file__).resolve().parent
P0=ROOT/"results/final_scientific_closure_p0"
P1=ROOT/"results/final_scientific_closure_p1/deployment_boundary"
errors=[]

def req(p):
    if not p.is_file(): errors.append(f"MISSING {p.relative_to(ROOT)}")
for p in [P0/"mechanism/equal_tuning_summary.json",P0/"mechanism/selected_hyperparameters.json",P0/"mechanism/equal_tuning_cell_comparisons.csv",P0/"fastpath/fastpath_summary.json",P0/"fastpath/kernel_timing_summary.csv",P0/"fastpath/stream_results.jsonl",P1/"deployment_boundary_summary.json",P1/"worker_calibration_coverage_by_capability.csv",P1/"task_cold_start_by_capability.csv"]: req(p)
if errors:
    print("\n".join(errors)); sys.exit(1)

s=json.loads((P0/"mechanism/equal_tuning_summary.json").read_text())
expected={"A1-persistent-ratio":("19/0/1","15/0"),"A2-attack-rho07":("9/1/2","8/0"),"A3-behavior-switch":("7/0/1","5/1")}
for fam,(wtl,holm) in expected.items():
    z=s[fam]["Tuned-LIR-vs-Tuned-EMA"]
    if (z["mean_W_T_L_tuned_lir"],z["holm_W_L_tuned_lir"])!=(wtl,holm): errors.append(f"P0_EQUAL_BUDGET {fam}")
hp=json.loads((P0/"mechanism/selected_hyperparameters.json").read_text())
if (hp.get("lir_eta"),hp.get("ema_alpha"),hp.get("window_w")) != ("2/5","2/5",3): errors.append("P0_HYPERPARAMETERS")
fp=json.loads((P0/"fastpath/fastpath_summary.json").read_text())
ov=fp["overall"]
if not fp.get("gate_pass") or (ov.get("streams"),ov.get("ordinary_tasks"),ov.get("mismatches"),ov.get("init_exact_ties"),ov.get("full_exact_ties"))!=(570,1662310,0,384,384): errors.append("P0_FASTPATH")
ks=list(csv.DictReader((P0/"fastpath/kernel_timing_summary.csv").open()))
reds=[float(r["runtime_reduction_fraction"]) for r in ks]
if not (0.959 <= min(reds) <= max(reds) <= 0.966): errors.append("P0_FASTPATH_TIMING")
p1=json.loads((P1/"deployment_boundary_summary.json").read_text())
wc=p1["worker_capability_coverage"]; tc=p1["task_history_coverage"]
if wc.get("worker_capability_trajectories")!=3574 or abs(wc.get("ge1_fraction")-0.827923894795747)>1e-12 or tc.get("scored_tasks")!=949647 or abs(tc.get("history_complete_fraction")-0.9512724201729695)>1e-12: errors.append("P1_COVERAGE")

# Re-run deterministic P0 cell reanalysis.
r=subprocess.run([sys.executable,str(ROOT/"analysis/final_scientific_closure_p0/reanalyze_equal_budget.py")],cwd=ROOT,capture_output=True,text=True)
if r.returncode: errors.append("P0_REANALYSIS_RUN")

# Re-run P1 postprocess and compare the primary JSON semantically.
r=subprocess.run([sys.executable,str(ROOT/"analysis/final_scientific_closure_p1/deployment_boundary_postprocess.py")],cwd=ROOT,capture_output=True,text=True)
if r.returncode: errors.append("P1_REANALYSIS_RUN")
else:
    rerun=json.loads((ROOT/"results/final_scientific_closure_p1/deployment_boundary_rerun/deployment_boundary_summary.json").read_text())
    if rerun != p1: errors.append("P1_REANALYSIS_MISMATCH")
    shutil.rmtree(ROOT/"results/final_scientific_closure_p1/deployment_boundary_rerun",ignore_errors=True)

print(f"P0_P1_EVIDENCE_ERRORS={len(errors)}")
for e in errors: print(e)
sys.exit(1 if errors else 0)
