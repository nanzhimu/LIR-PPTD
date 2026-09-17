from __future__ import annotations

import argparse,csv,hashlib,json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import mean,median
from typing import Any,Mapping,Sequence

import run_vi_g_final_semantics as design

RESULT_RELATIVE_DIR=Path("results/vi_g_final_semantics_v1")
class VIGAnalysisError(RuntimeError): pass

def _frac(v:Any)->Fraction:
    if isinstance(v,Fraction): return v
    return Fraction(Decimal(str(v)))

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _read_jsonl(path:Path)->list[dict[str,Any]]:
    rows=[]; seen=set()
    with path.open("r",encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line); rid=str(row.get("run_id",""))
            if not rid or rid in seen: raise VIGAnalysisError(f"DUPLICATE_OR_INVALID_RUN_ID:line={n}")
            seen.add(rid); rows.append(row)
    return rows

def _write_csv(path:Path,rows:Sequence[Mapping[str,Any]],fields:Sequence[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(fields)); w.writeheader(); w.writerows(rows)

def _stats_api():
    from lir_pptd.experiments.phase10_r22_statistical_analysis import exact_paired_sign_flip,holm_adjust,paired_rank_biserial
    from lir_pptd.experiments.phase_r1.statistics import ALPHA,BOOTSTRAP_RESAMPLES,bootstrap_mean_ci
    return exact_paired_sign_flip,holm_adjust,paired_rank_biserial,ALPHA,BOOTSTRAP_RESAMPLES,bootstrap_mean_ci

def hypothesis_id(family:str,attack:str,rho:str)->str:
    endpoint="MAE" if family.startswith("synthetic_num") else "1-Accuracy" if family=="synthetic_cat_binary" else "1-MacroF1"
    return f"family={family}|attack={attack}|endpoint={endpoint}|comparator=mean_vote|rho={rho}"

def analyze(repo_root:Path)->dict[str,Any]:
    outdir=repo_root/RESULT_RELATIVE_DIR; raw=outdir/"formal_runs.jsonl"; summary_path=outdir/"formal_summary.json"
    if not raw.is_file() or not summary_path.is_file(): raise VIGAnalysisError("FORMAL_RESULTS_MISSING")
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status")!="complete": raise VIGAnalysisError("FORMAL_SUMMARY_NOT_COMPLETE")
    if summary.get("design_sha256")!=design.design_sha256(): raise VIGAnalysisError("DESIGN_SHA_MISMATCH")
    if summary.get("raw_sha256")!=_sha256(raw): raise VIGAnalysisError("RAW_SHA_MISMATCH")
    rows=_read_jsonl(raw)
    if len(rows)!=design.PLANNED_STREAMS or any(x.get("status")!="success" for x in rows): raise VIGAnalysisError("RAW_CARDINALITY_OR_STATUS")

    cells={}
    for row in rows: cells.setdefault((row["family"],row["attack"],row["rho"]),[]).append(row)
    expected=len(design.FAMILIES)*len(design.ATTACKS)*len(design.RHO_GRID)
    if len(cells)!=expected: raise VIGAnalysisError(f"CELL_COUNT:{len(cells)}:{expected}")
    for key,members in cells.items():
        if len(members)!=len(design.SEEDS) or {int(x["seed"]) for x in members}!=set(design.SEEDS): raise VIGAnalysisError(f"PAIRING_MISMATCH:{key}")

    exact_flip,holm_adjust,rank_biserial,ALPHA,BOOTSTRAP_RESAMPLES,bootstrap_mean_ci=_stats_api()

    seed_rows=[{"seed":x["seed"],"family":x["family"],"attack":x["attack"],"rho":x["rho"],"endpoint":x["endpoint"],
      "lir_pptd_primary_loss":x["lir_pptd_primary_loss"],"mean_vote_primary_loss":x["mean_vote_primary_loss"],
      "delta_lir_minus_mean":x["paired_loss_difference_lir_minus_mean"],"final_reputation_gap":x["final_honest_minus_malicious_reputation_gap"],
      "effective_attacked_report_fraction":x["effective_attacked_report_fraction"],"calibration_attacked_count":x["calibration_attacked_count"],
      "on_off_starting_phase":x["on_off_starting_phase"]} for x in rows]
    _write_csv(outdir/"vi_g_seed_results.csv",seed_rows,list(seed_rows[0]))

    means_rows=[]; comps=[]; p_families={}
    for family in design.FAMILIES:
      for attack in design.ATTACKS:
       for rho in design.RHO_GRID:
        members=sorted(cells[(family,attack,rho)],key=lambda x:int(x["seed"]))
        lir=[_frac(x["lir_pptd_primary_loss"]) for x in members]; base=[_frac(x["mean_vote_primary_loss"]) for x in members]
        deltas=[a-b for a,b in zip(lir,base)]; endpoint=members[0]["endpoint"]
        ml=sum(lir,Fraction())/len(lir); mb=sum(base,Fraction())/len(base); md=sum(deltas,Fraction())/len(deltas)
        means_rows.append({"family":family,"attack":attack,"rho":rho,"endpoint":endpoint,"n":len(members),
          "lir_pptd_mean_loss":float(ml),"mean_vote_mean_loss":float(mb),"mean_delta_lir_minus_mean":float(md),
          "mean_final_reputation_gap":mean(float(x["final_honest_minus_malicious_reputation_gap"]) for x in members) if rho!="0" else "",
          "mean_effective_attacked_report_fraction":mean(float(x["effective_attacked_report_fraction"]) for x in members)})
        hid=hypothesis_id(family,attack,rho); p_less,p_two=exact_flip(deltas); p_mean,_=exact_flip([-d for d in deltas])
        rb=rank_biserial(deltas); lo,hi,bootseed=bootstrap_mean_ci(deltas,seed_text=design.design_sha256()+"|"+hid); primary=rho in design.PRIMARY_RHOS
        if primary: p_families.setdefault((family,attack),{})[hid]=p_less
        comps.append({"hypothesis_id":hid,"family":family,"attack":attack,"rho":rho,"endpoint":endpoint,"primary":primary,"n":len(members),
          "mean_lir_loss":float(ml),"mean_mean_vote_loss":float(mb),"mean_delta_lir_minus_mean":float(md),"median_delta_lir_minus_mean":float(median(deltas)),
          "seed_wins_lir":sum(d<0 for d in deltas),"seed_ties":sum(d==0 for d in deltas),"seed_losses_lir":sum(d>0 for d in deltas),
          "bootstrap95_low":lo,"bootstrap95_high":hi,"bootstrap_resamples":BOOTSTRAP_RESAMPLES,"bootstrap_seed":bootseed,
          "p_one_sided_lir_less":float(p_less),"p_one_sided_mean_less":float(p_mean),"p_two_sided":float(p_two),"paired_rank_biserial":float(rb),
          "p_holm_lir_less":"","holm_reject_lir_less":"","direction":"lir_pptd_better" if md<0 else "mean_vote_better" if md>0 else "tie",
          "evidence_class":"control" if not primary else ""})

    if len(p_families)!=design.HOLM_FAMILIES: raise VIGAnalysisError("HOLM_FAMILY_COUNT_MISMATCH")
    adjusted={}
    for key,pmap in p_families.items():
        if len(pmap)!=5: raise VIGAnalysisError(f"HOLM_FAMILY_SIZE:{key}:{len(pmap)}")
        adjusted.update(holm_adjust(pmap))
    for row in comps:
        if not row["primary"]: continue
        adj=adjusted[row["hypothesis_id"]]; reject=adj<=ALPHA; row["p_holm_lir_less"]=float(adj); row["holm_reject_lir_less"]=bool(reject)
        d=float(row["mean_delta_lir_minus_mean"])
        row["evidence_class"]="supporting" if d<0 and reject else "unresolved_lir_mean_better" if d<0 else "comparator_lower_loss" if d>0 else "exact_tie"

    _write_csv(outdir/"vi_g_stress_means.csv",means_rows,list(means_rows[0]))
    _write_csv(outdir/"vi_g_pairwise_comparisons.csv",comps,list(comps[0]))

    gap_rows=[]
    for family in design.FAMILIES:
      for attack in design.ATTACKS:
       for rho in design.PRIMARY_RHOS:
        members=cells[(family,attack,rho)]; gaps=[float(x["final_honest_minus_malicious_reputation_gap"]) for x in members]
        gap_rows.append({"family":family,"attack":attack,"rho":rho,"n":len(gaps),"mean_final_honest_minus_malicious_gap":mean(gaps),
          "min_gap":min(gaps),"max_gap":max(gaps),"positive_gap_seeds":sum(g>0 for g in gaps),"zero_gap_seeds":sum(g==0 for g in gaps),"negative_gap_seeds":sum(g<0 for g in gaps)})
    _write_csv(outdir/"vi_g_reputation_gap_summary.csv",gap_rows,list(gap_rows[0]))

    heat=[{"family":r["family"],"attack":r["attack"],"rho":r["rho"],"mean_delta_lir_minus_mean":r["mean_delta_lir_minus_mean"],
      "holm_reject_lir_less":r["holm_reject_lir_less"],"evidence_class":r["evidence_class"]} for r in comps if r["primary"]]
    _write_csv(outdir/"vi_g_heatmap_matrix.csv",heat,list(heat[0]))

    labels=("supporting","unresolved_lir_mean_better","comparator_lower_loss","exact_tie")
    counts={lab:sum(r["evidence_class"]==lab for r in comps if r["primary"]) for lab in labels}
    attack_summary={}
    for attack in design.ATTACKS:
        sub=[r for r in comps if r["primary"] and r["attack"]==attack]
        attack_summary[attack]={"primary_cells":len(sub),**{lab:sum(r["evidence_class"]==lab for r in sub) for lab in labels}}

    changes=[]
    order=list(design.PRIMARY_RHOS)
    for family in design.FAMILIES:
      for attack in design.ATTACKS:
        vals={r["rho"]:float(r["mean_delta_lir_minus_mean"]) for r in comps if r["primary"] and r["family"]==family and r["attack"]==attack}
        for left,right in zip(order[:-1],order[1:]):
            a,b=vals[left],vals[right]
            if a==0 or b==0 or (a<0<b) or (a>0>b):
                changes.append({"family":family,"attack":attack,"rho_left":left,"rho_right":right,"delta_left":a,"delta_right":b,
                  "interpretation":"empirical sign-change interval only; not a theoretical critical threshold"})

    result={"schema_version":"1.0","study_id":design.STUDY_ID,"status":"complete","design_sha256":design.design_sha256(),"raw_sha256":_sha256(raw),
      "primary_cell_count":design.PRIMARY_CONTRASTS,"control_cell_count":len(design.FAMILIES)*len(design.ATTACKS),"holm_family_count":design.HOLM_FAMILIES,
      "paired_seed_count":len(design.SEEDS),"alpha":float(ALPHA),"classification_counts":counts,"attack_summary":attack_summary,
      "empirical_sign_change_intervals":changes,"scope":"Synthetic evidence-gated operating-boundary study. Crossovers are empirical and not theoretical malicious-ratio thresholds.",
      "comparisons":comps}
    path=outdir/"vi_g_statistical_analysis.json"; path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    md=["# VI-G Final Synthetic Stress Analysis","",f"- Study: `{design.STUDY_ID}`",f"- Design SHA-256: `{design.design_sha256()}`",f"- Raw SHA-256: `{result['raw_sha256']}`",
      f"- Successful stream-level paired evaluations: **{len(rows)}/{design.PLANNED_STREAMS}**",f"- Primary nonzero-rho contrasts: **{design.PRIMARY_CONTRASTS}**",
      f"- Holm families: **{design.HOLM_FAMILIES}**, five hypotheses each","",
      "## Primary evidence classification","",f"- supporting: **{counts['supporting']}**",f"- unresolved LIR mean advantage: **{counts['unresolved_lir_mean_better']}**",
      f"- comparator lower mean loss: **{counts['comparator_lower_loss']}**",f"- exact tie: **{counts['exact_tie']}**","",
      "## Scope","","Any sign reversal is an empirical operating-boundary observation, not a theoretical critical threshold. Authenticated calibration-reference integrity is assumed."]
    (outdir/"VI_G_ANALYSIS_SUMMARY.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("VI_G_FINAL_ANALYSIS=COMPLETE"); print(f"PRIMARY_CONTRASTS={design.PRIMARY_CONTRASTS}"); print(f"SUPPORTING={counts['supporting']}")
    print(f"UNRESOLVED_LIR_MEAN_BETTER={counts['unresolved_lir_mean_better']}"); print(f"COMPARATOR_LOWER_LOSS={counts['comparator_lower_loss']}"); print(f"EXACT_TIE={counts['exact_tie']}")
    print(f"STATISTICAL_ANALYSIS={path}")
    return result

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",default=".",type=Path); args=p.parse_args(); analyze(args.repo_root.resolve()); return 0
if __name__=="__main__": raise SystemExit(main())
