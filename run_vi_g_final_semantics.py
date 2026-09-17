from __future__ import annotations
from lir_pptd.core.consistency_scale import DEFAULT_CONSISTENCY_SCALE_MODE, resolve_from_modality

import argparse, hashlib, inspect, json, os, time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

STUDY_ID="vi_g_final_semantics_v1"
CONFIG_ID="lir_cfg_05_lambda020"
FAMILIES=("synthetic_num_small","synthetic_num_shifted","synthetic_num_longitudinal","synthetic_cat_binary","synthetic_cat_tie")
ATTACKS=("persistent","random","on_off")
RHO_GRID=("0","1/10","3/10","1/2","7/10","9/10")
PRIMARY_RHOS=("1/10","3/10","1/2","7/10","9/10")
SEEDS=tuple(range(4001,4021))
TASKS_PER_STREAM=100
CALIBRATION_PERIOD=20
CALIBRATION_INDICES=tuple(range(CALIBRATION_PERIOD,TASKS_PER_STREAM+1,CALIBRATION_PERIOD))
CALIBRATION_TASKS_PER_STREAM=len(CALIBRATION_INDICES)
SCORED_ORDINARY_TASKS_PER_STREAM=TASKS_PER_STREAM-CALIBRATION_TASKS_PER_STREAM
PLANNED_STREAMS=len(FAMILIES)*len(ATTACKS)*len(RHO_GRID)*len(SEEDS)
PRIMARY_CONTRASTS=len(FAMILIES)*len(ATTACKS)*len(PRIMARY_RHOS)
HOLM_FAMILIES=len(FAMILIES)*len(ATTACKS)
HYPOTHESES_PER_FAMILY=len(PRIMARY_RHOS)
RESULT_RELATIVE_DIR=Path("results/vi_g_final_semantics_v1")
RAW_FILENAME="formal_runs.jsonl"
SUMMARY_FILENAME="formal_summary.json"
EXPECTED_CONFIG={"K":"10","lambda_tau":"1/5","epsilon_c":"1/1024","kappa":"2","eta":"1/10","c0":"1/2"}

class VIGError(RuntimeError): pass

@dataclass(frozen=True)
class RepoAPI:
    mean_vote_predict: Any
    canonical_json_bytes: Any
    run_ablation_once: Any
    candidate_config: Any
    LongitudinalState: Any
    build_scored_task: Any
    malicious_ids: Any
    worker_ids: tuple[str,...]
    argmax_zero_based: Any
    categorical_primary_loss: Any
    numerical_abs_error_sum: Any
    numerical_mae_from_abs_error_sums: Any

def _repo_api()->RepoAPI:
    from lir_pptd.baselines.mean_vote import predict as mean_vote_predict
    from lir_pptd.canonical import canonical_json_bytes
    from lir_pptd.experiments.phase6_ablation_runner import run_ablation_once
    from lir_pptd.experiments.phase6_candidate_runner import _candidate_config
    from lir_pptd.experiments.phase6_exact_adapter import LongitudinalState
    from lir_pptd.experiments.phase6_generator import WORKER_IDS,_build_scored_task,malicious_ids
    from lir_pptd.experiments.phase10_r22_metrics import argmax_zero_based,categorical_primary_loss,numerical_abs_error_sum,numerical_mae_from_abs_error_sums
    return RepoAPI(mean_vote_predict,canonical_json_bytes,run_ablation_once,_candidate_config,LongitudinalState,_build_scored_task,malicious_ids,tuple(WORKER_IDS),argmax_zero_based,categorical_primary_loss,numerical_abs_error_sum,numerical_mae_from_abs_error_sums)

def _fraction(v:Any)->Fraction:
    if isinstance(v,Fraction): return v
    if isinstance(v,int): return Fraction(v,1)
    return Fraction(str(v))

def _fraction_text(v:Fraction)->str: return f"{v.numerator}/{v.denominator}"
def _number(v:float)->str: return format(v,".17g")

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _atomic_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def _append_jsonl(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="") as f:
        f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
        f.flush(); os.fsync(f.fileno())

def _read_jsonl(path:Path)->list[dict[str,Any]]:
    if not path.exists(): return []
    rows=[]; seen=set()
    with path.open("r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line); rid=str(row.get("run_id",""))
            if not rid or rid in seen: raise VIGError(f"RAW_DUPLICATE_OR_INVALID_RUN_ID:line={line_no}")
            seen.add(rid); rows.append(row)
    return rows

def design_manifest()->dict[str,Any]:
    return {
      "schema_version":"1.0","study_id":STUDY_ID,
      "purpose":"Synthetic stress and empirical operating-boundary evaluation under final evidence-gated LIR-PPTD semantics.",
      "candidate":{"config_id":CONFIG_ID,
        "ordinary_semantics":"use persistent reputation during K truth iterations; do not commit ordinary q_out reputation candidate",
        "calibration_semantics":"authenticated ground truth; excluded from scoring; exactly one persistent reputation transition",
        "calibration_period":CALIBRATION_PERIOD,
        "nominal_calibration_fraction":CALIBRATION_TASKS_PER_STREAM/TASKS_PER_STREAM},
      "families":list(FAMILIES),
      "attacks":{"persistent":"truth-independent response corruption: numerical x->1-x; categorical cyclic class shift",
                 "random":"deterministic hash-derived random report independent of truth and calibration status",
                 "on_off":"5 benign + 5 response-corrupted tasks; starting phase balanced across seeds"},
      "rho_grid":list(RHO_GRID),"primary_rhos":list(PRIMARY_RHOS),"seeds":list(SEEDS),
      "tasks_per_stream":TASKS_PER_STREAM,"calibration_indices_one_based":list(CALIBRATION_INDICES),
      "scored_ordinary_tasks_per_stream":SCORED_ORDINARY_TASKS_PER_STREAM,"warm_start":"removed",
      "primary_comparator":"mean_vote",
      "endpoints":{"synthetic_num_small":"MAE","synthetic_num_shifted":"MAE","synthetic_num_longitudinal":"MAE","synthetic_cat_binary":"1-Accuracy","synthetic_cat_tie":"1-MacroF1"},
      "statistics":{"paired_seed_count":len(SEEDS),"bootstrap_resamples":10000,
        "primary_randomization_test":"exact one-sided paired sign-flip, H1: LIR-PPTD loss < Mean/MV loss",
        "supplementary_randomization_test":"exact two-sided paired sign-flip",
        "multiplicity":"Holm within each family x attack over the five nonzero rho hypotheses",
        "holm_family_count":HOLM_FAMILIES,"hypotheses_per_family":HYPOTHESES_PER_FAMILY},
      "scope":["synthetic stress test only","operating-boundary observations are empirical, not theoretical critical thresholds","calibration reference integrity is assumed","attack generation never receives the calibration flag or trusted reference"]
    }

def design_sha256()->str:
    p=json.dumps(design_manifest(),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(p).hexdigest()

def validate_config(config:Mapping[str,Any])->None:
    for k,e in EXPECTED_CONFIG.items():
        if k not in config: raise VIGError(f"CONFIG_FIELD_MISSING:{k}")
        if _fraction(config[k])!=_fraction(e): raise VIGError(f"CONFIG_MISMATCH:{k}:{config[k]}:{e}")

def is_calibration(task_index:int)->bool: return task_index in CALIBRATION_INDICES

def on_off_attack_first(seed:int)->bool:
    if seed not in SEEDS: raise ValueError("seed outside VI-G final namespace")
    return seed>=SEEDS[len(SEEDS)//2]

def on_off_active(seed:int,task_index:int)->bool:
    pos=(task_index-1)%10
    first_half=pos<5
    return first_half if on_off_attack_first(seed) else not first_half

def _u01(seed:int,family:str,task_index:int,worker:str,coordinate:int)->float:
    p=f"{STUDY_ID}|random-attack|seed={seed}|family={family}|task={task_index}|worker={worker}|coord={coordinate}".encode()
    x=int.from_bytes(hashlib.sha256(p).digest()[:8],"big")>>11
    return x/float(1<<53)

def _response_corrupt(report:Sequence[Any],modality:str)->list[Any]:
    if modality=="numerical": return [_number(1.0-float(v)) for v in report]
    if modality=="categorical":
        nums=[float(v) for v in report]
        if not nums: raise VIGError("EMPTY_CATEGORICAL_REPORT")
        label=next(i for i,v in enumerate(nums) if v==max(nums)); w=len(nums); shifted=(label+1)%w
        return [1 if i==shifted else 0 for i in range(w)]
    raise VIGError(f"UNKNOWN_MODALITY:{modality}")

def _random_report(report:Sequence[Any],modality:str,*,seed:int,family:str,task_index:int,worker:str)->list[Any]:
    if modality=="numerical": return [_number(_u01(seed,family,task_index,worker,h)) for h in range(len(report))]
    if modality=="categorical":
        w=len(report)
        if w<2: raise VIGError("CATEGORICAL_WIDTH_TOO_SMALL")
        label=min(w-1,int(_u01(seed,family,task_index,worker,0)*w))
        return [1 if i==label else 0 for i in range(w)]
    raise VIGError(f"UNKNOWN_MODALITY:{modality}")

def attack_task(task:Mapping[str,Any],*,attack:str,seed:int,family:str,rho:str,malicious_workers:Sequence[str])->tuple[dict[str,Any],bool,int]:
    """Attack reports before calibration designation; no calibration flag or trusted truth is accepted."""
    if attack not in ATTACKS: raise VIGError(f"UNKNOWN_ATTACK:{attack}")
    idx=int(task["task_index"]); active=rho!="0" and (attack!="on_off" or on_off_active(seed,idx))
    bad=set(malicious_workers); reports={}
    for worker,report in task["reports"].items():
        original=list(report)
        if worker not in bad or not active: reports[worker]=original
        elif attack in {"persistent","on_off"}: reports[worker]=_response_corrupt(original,str(task["modality"]))
        else: reports[worker]=_random_report(original,str(task["modality"]),seed=seed,family=family,task_index=idx,worker=worker)
    out=dict(task); out["reports"]=reports; out["malicious_ids"]=sorted(bad); out["attack_id"]=attack
    out["attack_state"]={"active":active,"phase":"attack" if active else "benign","phase_balanced":attack=="on_off",
      "starting_phase":("attack-first" if attack=="on_off" and on_off_attack_first(seed) else "benign-first" if attack=="on_off" else "not_applicable")}
    return out,active,sum(1 for w in reports if w in bad and active)

def _state_reputation(state:Any,worker:str,config:Mapping[str,Any])->Fraction:
    reputations = state.reputations or {}
    return _fraction(reputations.get(worker,config["c0"]))
def _state_epoch(state:Any,worker:str)->int:
    epochs = state.epochs or {}
    return int(epochs.get(worker,0))

def _calibration_update(task:Mapping[str,Any],state:Any,config:Mapping[str,Any],state_type:Any)->Any:
    truth=tuple(_fraction(v) for v in task["ground_truth"]); D=len(truth)
    if D<1: raise VIGError("EMPTY_CALIBRATION_TRUTH")
    tau=resolve_from_modality(
        str(task.get("modality", "numerical")), D, config["lambda_tau"],
        str(config.get("consistency_scale_mode", DEFAULT_CONSISTENCY_SCALE_MODE)),
    ).resolved_tau; eta=_fraction(config["eta"]); kappa=_fraction(config["kappa"])
    reps=dict(state.reputations or {}); epochs=dict(state.epochs or {})
    for worker,report in task["reports"].items():
        x=tuple(_fraction(v) for v in report)
        if len(x)!=D: raise VIGError("CALIBRATION_DIMENSION_MISMATCH")
        d=sum((a-b)**2 for a,b in zip(x,truth)); q=tau/(tau+d); c=_state_reputation(state,worker,config)
        u=c+eta*((1-c)*q-kappa*c*(1-q))
        if not 0<=u<=1: raise VIGError("CALIBRATION_REPUTATION_OUT_OF_RANGE")
        reps[worker]=_fraction_text(u); epochs[worker]=_state_epoch(state,worker)+1
    return state_type(reputations=reps,epochs=epochs)

def _stream_hash(tasks:Sequence[Mapping[str,Any]],canonical_json_bytes:Any)->str:
    h=hashlib.sha256()
    for task in tasks:
        p=canonical_json_bytes(task); h.update(len(p).to_bytes(8,"big")); h.update(p)
    return h.hexdigest()

def _score_stream(*,family:str,truths:Sequence[Sequence[Any]],lir_outputs:Sequence[Sequence[Any]],mean_outputs:Sequence[Sequence[Any]],api:RepoAPI)->tuple[float,float,str]:
    if len(truths)!=SCORED_ORDINARY_TASKS_PER_STREAM or not(len(lir_outputs)==len(mean_outputs)==len(truths)): raise VIGError("SCORE_DENOMINATOR_MISMATCH")
    if family.startswith("synthetic_num"):
        le=[]; me=[]; dims=[]
        for t,l,b in zip(truths,lir_outputs,mean_outputs):
            le.append(api.numerical_abs_error_sum(l,t)); me.append(api.numerical_abs_error_sum(b,t)); dims.append(len(t))
        return api.numerical_mae_from_abs_error_sums(le,dims),api.numerical_mae_from_abs_error_sums(me,dims),"MAE"
    tc=[api.argmax_zero_based(x) for x in truths]; lp=[api.argmax_zero_based(x) for x in lir_outputs]; mp=[api.argmax_zero_based(x) for x in mean_outputs]
    endpoint="1-Accuracy" if family=="synthetic_cat_binary" else "1-MacroF1"
    return api.categorical_primary_loss(family,lp,tc),api.categorical_primary_loss(family,mp,tc),endpoint

def _final_gap(state:Any,bad_workers:Sequence[str],workers:Sequence[str],config:Mapping[str,Any])->float|None:
    bad=set(bad_workers)
    if not bad: return None
    good=[w for w in workers if w not in bad]; evil=[w for w in workers if w in bad]
    if not good or not evil: return None
    return mean(float(_state_reputation(state,w,config)) for w in good)-mean(float(_state_reputation(state,w,config)) for w in evil)

def _run_id(family:str,attack:str,rho:str,seed:int)->str:
    return hashlib.sha256(f"{design_sha256()}|family={family}|attack={attack}|rho={rho}|seed={seed}".encode()).hexdigest()

def run_stream(*,repo_root:Path,family:str,attack:str,rho:str,seed:int,api:RepoAPI|None=None)->dict[str,Any]:
    api=api or _repo_api(); config=api.candidate_config(repo_root,CONFIG_ID); validate_config(config)
    bad=tuple(api.malicious_ids(seed,rho))
    expected={"0":0,"1/10":2,"3/10":6,"1/2":10,"7/10":14,"9/10":18}[rho]
    if len(bad)!=expected: raise VIGError("MALICIOUS_COUNT_MISMATCH")
    state=api.LongitudinalState(reputations={}, epochs={}); attacked_tasks=[]; truths=[]; lir=[]; baseline=[]
    ordinary=cal=sem_ord=cal_inst=attacked_instances=total_instances=cal_attacked=0
    started=time.perf_counter()
    for idx in range(1,TASKS_PER_STREAM+1):
        base=api.build_scored_task(seed,family,idx,"0","no_attack",fixed_target_selection=None)
        task,active,n_att=attack_task(base,attack=attack,seed=seed,family=family,rho=rho,malicious_workers=bad)
        attacked_tasks.append(task); attacked_instances+=n_att; total_instances+=len(task["reports"])
        if is_calibration(idx):
            cal+=1; cal_attacked+=int(active and bool(bad))
            state=_calibration_update(task,state,config,api.LongitudinalState); cal_inst+=len(task["reports"]); continue
        ordinary+=1
        result=api.run_ablation_once(task,config,state,ablation="full",dps=80)
        # Final H1: ordinary candidate reputation transition is intentionally not committed.
        lir.append(tuple(result.final_output)); baseline.append(tuple(api.mean_vote_predict(task["reports"],task["modality"]))); truths.append(tuple(task["ground_truth"]))
        sem_ord+=0
    elapsed=time.perf_counter()-started
    if ordinary!=95 or cal!=5 or sem_ord!=0: raise VIGError("SEMANTIC_COUNT_VIOLATION")
    epochs={_state_epoch(state,w) for w in api.worker_ids}
    if epochs!={5}: raise VIGError(f"FINAL_CALIBRATION_EPOCH_MISMATCH:{sorted(epochs)}")
    ll,ml,endpoint=_score_stream(family=family,truths=truths,lir_outputs=lir,mean_outputs=baseline,api=api)
    return {"schema_version":"1.0","study_id":STUDY_ID,"design_sha256":design_sha256(),"run_id":_run_id(family,attack,rho,seed),"status":"success",
      "seed":seed,"family":family,"attack":attack,"rho":rho,"endpoint":endpoint,"config_id":CONFIG_ID,"task_count":100,
      "calibration_event_count":cal,"scored_ordinary_task_count":ordinary,"ordinary_persistent_transition_count":sem_ord,
      "calibration_transition_install_count":cal_inst,"final_calibration_epoch":5,"malicious_worker_count":len(bad),"malicious_workers":list(bad),
      "on_off_starting_phase":("attack-first" if attack=="on_off" and on_off_attack_first(seed) else "benign-first" if attack=="on_off" else "not_applicable"),
      "calibration_attacked_count":cal_attacked,"effective_attacked_report_fraction":0.0 if total_instances==0 else attacked_instances/total_instances,
      "lir_pptd_primary_loss":_number(ll),"mean_vote_primary_loss":_number(ml),"paired_loss_difference_lir_minus_mean":_number(ll-ml),
      "final_honest_minus_malicious_reputation_gap":_final_gap(state,bad,api.worker_ids,config),"attacked_stream_sha256":_stream_hash(attacked_tasks,api.canonical_json_bytes),
      "elapsed_seconds":elapsed}

def _implementation_hashes(api:RepoAPI)->dict[str,str]:
    objects={"mean_vote_predict":api.mean_vote_predict,"run_ablation_once":api.run_ablation_once,"candidate_config_loader":api.candidate_config,"build_scored_task":api.build_scored_task}
    return {k:f"{Path(inspect.getfile(v)).as_posix()}@{_sha256(Path(inspect.getfile(v)))}" for k,v in objects.items()}

def validate_repo(repo_root:Path)->dict[str,Any]:
    api=_repo_api(); config=api.candidate_config(repo_root,CONFIG_ID); validate_config(config)
    if len(api.worker_ids)!=20: raise VIGError("WORKER_COUNT_MISMATCH")
    counts={"0":0,"1/10":2,"3/10":6,"1/2":10,"7/10":14,"9/10":18}
    for rho,n in counts.items():
        if len(api.malicious_ids(SEEDS[0],rho))!=n: raise VIGError(f"MALICIOUS_COUNT_MISMATCH:{rho}")
    probe=api.build_scored_task(SEEDS[0],FAMILIES[0],1,"0","no_attack",fixed_target_selection=None)
    if len(probe.get("reports",{}))!=20 or "ground_truth" not in probe: raise VIGError("PROBE_TASK_CONTRACT_MISMATCH")
    empty_state=api.LongitudinalState(reputations={},epochs={})
    if dict(empty_state.reputations or {})!={} or dict(empty_state.epochs or {})!={}: raise VIGError("EMPTY_STATE_NORMALIZATION_MISMATCH")
    attacked_cals=sum(is_calibration(t) and on_off_active(s,t) for s in SEEDS for t in range(1,101))
    if attacked_cals!=50: raise VIGError(f"ON_OFF_CALIBRATION_PHASE_BALANCE_MISMATCH:{attacked_cals}:50")
    return {"status":"PASS","design_sha256":design_sha256(),"implementation_bindings":_implementation_hashes(api)}

def _validation_lines(r:Mapping[str,Any])->str:
    return "\n".join(["VI_G_FINAL_SEMANTICS_VALIDATE=PASS",f"DESIGN_SHA256={r['design_sha256']}",f"FAMILIES={len(FAMILIES)}",f"ATTACKS={','.join(ATTACKS)}",
      f"RHO_GRID={','.join(RHO_GRID)}",f"PRIMARY_RHOS={','.join(PRIMARY_RHOS)}",f"SEEDS={SEEDS[0]}-{SEEDS[-1]}",f"PAIRED_SEEDS={len(SEEDS)}",
      "TASKS_PER_STREAM=100","CALIBRATION_PERIOD=20","CALIBRATION_TASKS_PER_STREAM=5","SCORED_ORDINARY_TASKS_PER_STREAM=95",
      f"PLANNED_STREAMS={PLANNED_STREAMS}",f"PRIMARY_CONTRASTS={PRIMARY_CONTRASTS}",f"HOLM_FAMILIES={HOLM_FAMILIES}",f"HYPOTHESES_PER_FAMILY={HYPOTHESES_PER_FAMILY}",
      "WARM_START=REMOVED","ORDINARY_PERSISTENCE=DISABLED","CALIBRATION_PERSISTENCE=ENABLED"])

def execute(repo_root:Path)->dict[str,Any]:
    validation=validate_repo(repo_root); api=_repo_api(); outdir=repo_root/RESULT_RELATIVE_DIR; raw=outdir/RAW_FILENAME; summary_path=outdir/SUMMARY_FILENAME
    if summary_path.exists():
        s=json.loads(summary_path.read_text(encoding="utf-8"))
        if s.get("status")=="complete":
            if not raw.is_file() or _sha256(raw)!=s.get("raw_sha256"): raise VIGError("EXISTING_SUMMARY_RAW_BINDING_MISMATCH")
            print("VI_G_FINAL_SEMANTICS_ALREADY_COMPLETE=PASS"); return s
        raise VIGError("EXISTING_NONFINAL_SUMMARY_PRESENT")
    existing=_read_jsonl(raw); by={x["run_id"]:x for x in existing}
    if any(x.get("design_sha256")!=design_sha256() or x.get("status")!="success" for x in existing): raise VIGError("RAW_BINDING_OR_STATUS_INVALID")
    plan=[(f,a,r,s) for f in FAMILIES for a in ATTACKS for r in RHO_GRID for s in SEEDS]
    if len(plan)!=PLANNED_STREAMS: raise VIGError("PLAN_CARDINALITY_MISMATCH")
    started=time.perf_counter(); done=len(by); print(f"VI_G_FINAL_EXECUTE_START existing={done} total={PLANNED_STREAMS}",flush=True)
    for f,a,r,s in plan:
        rid=_run_id(f,a,r,s)
        if rid in by: continue
        row=run_stream(repo_root=repo_root,family=f,attack=a,rho=r,seed=s,api=api); _append_jsonl(raw,row); by[rid]=row; done+=1
        if done%10==0 or done==PLANNED_STREAMS:
            print(f"PROGRESS={done}/{PLANNED_STREAMS} CURRENT={f}/{a}/rho={r}/seed={s} SESSION_ELAPSED_S={time.perf_counter()-started:.1f}",flush=True)
    rows=_read_jsonl(raw)
    if len(rows)!=PLANNED_STREAMS or len({x["run_id"] for x in rows})!=PLANNED_STREAMS: raise VIGError("FINAL_CARDINALITY_OR_DUPLICATION")
    if any(x["status"]!="success" or int(x["ordinary_persistent_transition_count"])!=0 or int(x["calibration_event_count"])!=5 or int(x["final_calibration_epoch"])!=5 or int(x["scored_ordinary_task_count"])!=95 for x in rows): raise VIGError("FINAL_SEMANTIC_INVARIANT_VIOLATION")
    summary={"schema_version":"1.0","study_id":STUDY_ID,"status":"complete","design_sha256":design_sha256(),"planned_streams":PLANNED_STREAMS,
      "successful_streams":len(rows),"failure_streams":0,"families":list(FAMILIES),"attacks":list(ATTACKS),"rho_grid":list(RHO_GRID),"seeds":[SEEDS[0],SEEDS[-1]],
      "paired_seed_count":len(SEEDS),"tasks_per_stream":100,"calibration_tasks_per_stream":5,"scored_ordinary_tasks_per_stream":95,"ordinary_persistence_violations":0,
      "implementation_bindings":validation["implementation_bindings"],"raw_path":str(RESULT_RELATIVE_DIR/RAW_FILENAME).replace("\\","/"),"raw_sha256":_sha256(raw),
      "total_recorded_elapsed_seconds":sum(float(x["elapsed_seconds"]) for x in rows)}
    _atomic_json(summary_path,summary); print("VI_G_FINAL_SEMANTICS_EXECUTION=COMPLETE"); print(f"SUCCESSFUL_STREAMS={len(rows)}"); print("FAILURE_STREAMS=0"); print(f"RAW_SHA256={summary['raw_sha256']}")
    return summary

def smoke(repo_root:Path)->dict[str,Any]:
    validate_repo(repo_root); row=run_stream(repo_root=repo_root,family="synthetic_num_small",attack="persistent",rho="1/10",seed=SEEDS[0])
    print("VI_G_FINAL_SEMANTICS_SMOKE=PASS"); print(f"LIR_LOSS={row['lir_pptd_primary_loss']}"); print(f"MEAN_LOSS={row['mean_vote_primary_loss']}")
    print(f"CALIBRATION_EVENTS={row['calibration_event_count']}"); print(f"SCORED_ORDINARY_TASKS={row['scored_ordinary_task_count']}"); print(f"FINAL_CALIBRATION_EPOCH={row['final_calibration_epoch']}")
    return row

def main()->int:
    p=argparse.ArgumentParser(description="VI-G final evidence-gated synthetic stress runner"); p.add_argument("--repo-root",default=".",type=Path)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument("--validate-only",action="store_true"); g.add_argument("--smoke",action="store_true"); g.add_argument("--execute",action="store_true")
    args=p.parse_args(); root=args.repo_root.resolve()
    if args.validate_only: print(_validation_lines(validate_repo(root)))
    elif args.smoke: smoke(root)
    else: execute(root)
    return 0
if __name__=="__main__": raise SystemExit(main())
