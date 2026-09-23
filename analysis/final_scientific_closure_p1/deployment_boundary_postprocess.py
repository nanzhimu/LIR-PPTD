from __future__ import annotations
from pathlib import Path
import csv, gzip, json, math, statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/p1c_netease_longitudinal_geometry_v1"
OUT = ROOT / "results/final_scientific_closure_p1/deployment_boundary_rerun"
OUT.mkdir(parents=True, exist_ok=True)
CAPS = [50, 52, 53, 56, 69, 126]

def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def qlinear(vals, q):
    x = sorted(float(v) for v in vals)
    if not x: return float("nan")
    pos = (len(x)-1)*q; lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi: return x[lo]
    return x[lo] + (pos-lo)*(x[hi]-x[lo])

def fmean(vals): return sum(vals)/len(vals) if vals else float("nan")

def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

w = read_csv(BASE / "worker_lifecycle.csv")
bycap={}
for r in w: bycap.setdefault(int(r["capability"]), []).append(r)
cap_rows=[]
for cap in sorted(bycap):
    g=bycap[cap]; e=[int(float(r["hidden_calibration_exposures"])) for r in g]; t=[int(float(r["task_count"])) for r in g]
    cap_rows.append({
        "capability":cap,"worker_capability_trajectories":len(g),"zero_exposure_count":sum(v==0 for v in e),
        "zero_exposure_fraction":sum(v==0 for v in e)/len(e),"ge1_fraction":sum(v>=1 for v in e)/len(e),
        "ge2_fraction":sum(v>=2 for v in e)/len(e),"ge5_fraction":sum(v>=5 for v in e)/len(e),
        "ge10_fraction":sum(v>=10 for v in e)/len(e),"exposure_median":statistics.median(e),
        "exposure_q1":qlinear(e,.25),"exposure_q3":qlinear(e,.75),"exposure_p90":qlinear(e,.90),
        "exposure_max":max(e),"participant_annotations":sum(t),"calibration_exposures":sum(e),
        "participant_exposure_rate":sum(e)/sum(t),})
write_csv(OUT/"worker_calibration_coverage_by_capability.csv", cap_rows)

e=[int(float(r["hidden_calibration_exposures"])) for r in w]; t=[int(float(r["task_count"])) for r in w]
overall_pair={"worker_capability_trajectories":len(w),"zero_exposure_count":sum(v==0 for v in e),"zero_exposure_fraction":sum(v==0 for v in e)/len(e),"ge1_fraction":sum(v>=1 for v in e)/len(e),"ge2_fraction":sum(v>=2 for v in e)/len(e),"ge5_fraction":sum(v>=5 for v in e)/len(e),"ge10_fraction":sum(v>=10 for v in e)/len(e),"exposure_median":statistics.median(e),"exposure_q1":qlinear(e,.25),"exposure_q3":qlinear(e,.75),"exposure_p90":qlinear(e,.90),"participant_annotations":sum(t),"calibration_exposures":sum(e),"participant_exposure_rate":sum(e)/sum(t)}

workers={}
for r in w:
    wid=r["worker_id"]; workers.setdefault(wid,0); workers[wid]+=int(float(r["hidden_calibration_exposures"]))
ue=list(workers.values())
unique_worker={"unique_worker_ids":len(ue),"zero_exposure_count_any_capability":sum(v==0 for v in ue),"zero_exposure_fraction_any_capability":sum(v==0 for v in ue)/len(ue),"ge1_fraction_any_capability":sum(v>=1 for v in ue)/len(ue),"exposure_median_across_capabilities":statistics.median(ue),"exposure_q1_across_capabilities":qlinear(ue,.25),"exposure_q3_across_capabilities":qlinear(ue,.75)}

task_rows=[]; allp=[]
for cap in CAPS:
    p=read_csv(BASE/f"predictions_cap_{cap}.csv.gz"); allp.extend(p)
    cold=[float(r["cold_fraction"]) for r in p]; bins=[r["cold_bin"] for r in p]
    hc=[v==0 for v in cold]; first=next((i for i,v in enumerate(hc) if v), None)
    first_end=int(float(p[first]["task_end"])) if first is not None else None; first_task=int(float(p[0]["task_end"]))
    task_rows.append({"capability":cap,"scored_tasks":len(p),"mean_cold_fraction":fmean(cold),"history_complete_count":sum(hc),"history_complete_fraction":sum(hc)/len(hc),"minority_cold_count":sum(x=="minority_cold" for x in bins),"majority_cold_count":sum(x=="majority_cold" for x in bins),"all_cold_count":sum(x=="all_cold" for x in bins),"first_history_complete_scored_task_index":None if first is None else first+1,"days_from_first_scored_to_first_history_complete":None if first_end is None else (first_end-first_task)/86400000})
write_csv(OUT/"task_cold_start_by_capability.csv", task_rows)
allcold=[float(r["cold_fraction"]) for r in allp]; allbins=[r["cold_bin"] for r in allp]
overall_task={"scored_tasks":len(allp),"mean_cold_fraction":fmean(allcold),"history_complete_count":sum(v==0 for v in allcold),"history_complete_fraction":sum(v==0 for v in allcold)/len(allcold),"minority_cold_count":sum(x=="minority_cold" for x in allbins),"majority_cold_count":sum(x=="majority_cold" for x in allbins),"all_cold_count":sum(x=="all_cold" for x in allbins)}
cap69=next(r for r in cap_rows if r["capability"]==69); task69=next(r for r in task_rows if r["capability"]==69)
summary={"schema_version":"1.0","study":"Final Scientific Closure — P1 Deployment Boundary","source_study":"p1c_netease_longitudinal_geometry_v1","status":"PASS","worker_capability_coverage":overall_pair,"unique_worker_descriptive":unique_worker,"task_history_coverage":overall_task,"capability_69_boundary":{"worker_zero_exposure_fraction":cap69["zero_exposure_fraction"],"history_complete_fraction":task69["history_complete_fraction"],"median_exposures":cap69["exposure_median"]},"scope_notes":["Persistent reputation is capability-specific; worker-capability trajectories are the primary coverage unit.","The lean artifact does not retain participant-level first-calibration timestamps, so exact worker-level waiting time is not claimed.","Reference-provider acquisition latency is external to the MPC path and was not measured.","Selector hiddenness does not imply that task content hides application-level verifiability cues.","SQLite-WAL experiments validate persistence/replay semantics at component level, not distributed failover.","The 4+3+3 VM layout is a performance topology, not ten independent administrative trust domains."]}
(OUT/"deployment_boundary_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
print("P1_DEPLOYMENT_BOUNDARY_REANALYSIS=PASS")
print(json.dumps(summary, indent=2))
