#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, json, random, math
from pathlib import Path

CAPS=(50,52,53,56,69,126)
METHODS=("lir_pptd","cowa","no_hr")
LABELS=(0,1,2); IDX={v:i for i,v in enumerate(LABELS)}
N_BOOT=10000; SEED=20260902

def blank_cm(): return [[0,0,0] for _ in range(3)]
def add_cm(dst,src,mult=1):
    for i in range(3):
        for j in range(3): dst[i][j]+=src[i][j]*mult

def metrics(cm):
    total=sum(sum(r) for r in cm)
    acc=sum(cm[i][i] for i in range(3))/total if total else 0.0
    f1=[]
    for k in range(3):
        tp=cm[k][k]; pred=sum(cm[i][k] for i in range(3)); actual=sum(cm[k][j] for j in range(3)); den=pred+actual
        f1.append(2*tp/den if den else 0.0)
    return acc,sum(f1)/3

def quantile(xs,q):
    ys=sorted(xs); pos=(len(ys)-1)*q; lo=int(pos); hi=min(lo+1,len(ys)-1); frac=pos-lo
    return ys[lo]*(1-frac)+ys[hi]*frac

def load_clusters(path):
    clusters={}; n=0
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            cid=row['taskset_id']; clusters.setdefault(cid,{m:blank_cm() for m in METHODS})
            ti=IDX[int(row['truth'])]
            for m in METHODS: clusters[cid][m][ti][IDX[int(row[m])]] += 1
            n+=1
    return clusters,n

def full_metrics(clusters, exclude=None):
    full={m:blank_cm() for m in METHODS}
    for cid,d in clusters.items():
        if cid==exclude: continue
        for m in METHODS: add_cm(full[m],d[m])
    return {m:metrics(full[m]) for m in METHODS}

def delta(metric_map,a,b,idx): return metric_map[a][idx]-metric_map[b][idx]

def jackknife_summary(vals, full):
    n=len(vals); mean=sum(vals)/n
    se=math.sqrt((n-1)/n * sum((v-mean)**2 for v in vals)) if n>1 else 0.0
    return {"full":full,"loo_min":min(vals),"loo_max":max(vals),"loo_mean":mean,"loo_all_positive":all(v>0 for v in vals),"loo_all_negative":all(v<0 for v in vals),"jackknife_se":se,"normal_ci95_low":full-1.96*se,"normal_ci95_high":full+1.96*se}

def main(root,out_csv,out_json,out_loo_csv,out_loo_json):
    rng=random.Random(SEED)
    rows=[]; report={"schema_version":"2.0","analysis":"native-taskset cluster bootstrap over frozen NetEaseCrowd predictions","comparisons":["lir_pptd-cowa","lir_pptd-no_hr"],"cluster_key":"taskset_id","bootstrap_replicates":N_BOOT,"seed":SEED,"scope":"post-hoc uncertainty only; no algorithm rerun, selector change, or retuning","capabilities":{}}
    cap126_clusters=None
    for cap in CAPS:
        clusters,n_tasks=load_clusters(root/f'predictions_cap_{cap}.csv.gz'); ids=sorted(clusters); ncl=len(ids)
        if cap==126: cap126_clusters=clusters
        point=full_metrics(clusters)
        arrays={k:[] for k in ('lc_acc','lc_f1','ln_acc','ln_f1')}
        for _ in range(N_BOOT):
            agg={m:blank_cm() for m in METHODS}
            for _draw in range(ncl):
                cid=ids[rng.randrange(ncl)]
                for m in METHODS: add_cm(agg[m],clusters[cid][m])
            bm={m:metrics(agg[m]) for m in METHODS}
            arrays['lc_acc'].append(delta(bm,'lir_pptd','cowa',0)); arrays['lc_f1'].append(delta(bm,'lir_pptd','cowa',1))
            arrays['ln_acc'].append(delta(bm,'lir_pptd','no_hr',0)); arrays['ln_f1'].append(delta(bm,'lir_pptd','no_hr',1))
        row={
            'capability':cap,'scored_tasks':n_tasks,'taskset_clusters':ncl,
            'delta_lir_cowa_accuracy':delta(point,'lir_pptd','cowa',0),'delta_lir_cowa_accuracy_ci95_low':quantile(arrays['lc_acc'],.025),'delta_lir_cowa_accuracy_ci95_high':quantile(arrays['lc_acc'],.975),
            'delta_lir_cowa_macro_f1':delta(point,'lir_pptd','cowa',1),'delta_lir_cowa_macro_f1_ci95_low':quantile(arrays['lc_f1'],.025),'delta_lir_cowa_macro_f1_ci95_high':quantile(arrays['lc_f1'],.975),
            'delta_lir_no_hr_accuracy':delta(point,'lir_pptd','no_hr',0),'delta_lir_no_hr_accuracy_ci95_low':quantile(arrays['ln_acc'],.025),'delta_lir_no_hr_accuracy_ci95_high':quantile(arrays['ln_acc'],.975),
            'delta_lir_no_hr_macro_f1':delta(point,'lir_pptd','no_hr',1),'delta_lir_no_hr_macro_f1_ci95_low':quantile(arrays['ln_f1'],.025),'delta_lir_no_hr_macro_f1_ci95_high':quantile(arrays['ln_f1'],.975),
        }
        rows.append(row)
        report['capabilities'][str(cap)] = row | {
            'bootstrap_probability_lir_cowa_accuracy_gt_0':sum(x>0 for x in arrays['lc_acc'])/N_BOOT,
            'bootstrap_probability_lir_cowa_macro_f1_gt_0':sum(x>0 for x in arrays['lc_f1'])/N_BOOT,
            'bootstrap_probability_lir_no_hr_accuracy_gt_0':sum(x>0 for x in arrays['ln_acc'])/N_BOOT,
            'bootstrap_probability_lir_no_hr_macro_f1_gt_0':sum(x>0 for x in arrays['ln_f1'])/N_BOOT,
        }
    with open(out_csv,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out_json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    # capability 126 leave-one-taskset-block-out sensitivity
    clusters=cap126_clusters; ids=sorted(clusters); full=full_metrics(clusters)
    loo_rows=[]; vals={k:[] for k in ('lc_acc','lc_f1','ln_acc','ln_f1')}
    for cid in ids:
        mm=full_metrics(clusters,exclude=cid)
        r={'excluded_taskset_id':cid,
           'delta_lir_cowa_accuracy':delta(mm,'lir_pptd','cowa',0),'delta_lir_cowa_macro_f1':delta(mm,'lir_pptd','cowa',1),
           'delta_lir_no_hr_accuracy':delta(mm,'lir_pptd','no_hr',0),'delta_lir_no_hr_macro_f1':delta(mm,'lir_pptd','no_hr',1)}
        loo_rows.append(r); vals['lc_acc'].append(r['delta_lir_cowa_accuracy']); vals['lc_f1'].append(r['delta_lir_cowa_macro_f1']); vals['ln_acc'].append(r['delta_lir_no_hr_accuracy']); vals['ln_f1'].append(r['delta_lir_no_hr_macro_f1'])
    with open(out_loo_csv,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(loo_rows[0].keys())); w.writeheader(); w.writerows(loo_rows)
    loo_report={'schema_version':'1.0','analysis':'capability 126 leave-one-taskset-block-out sensitivity','capability':126,'taskset_clusters':len(ids),'scope':'frozen predictions; each native taskset_id cluster omitted once','comparisons':{
        'lir_pptd-cowa':{'accuracy':jackknife_summary(vals['lc_acc'],delta(full,'lir_pptd','cowa',0)),'macro_f1':jackknife_summary(vals['lc_f1'],delta(full,'lir_pptd','cowa',1))},
        'lir_pptd-no_hr':{'accuracy':jackknife_summary(vals['ln_acc'],delta(full,'lir_pptd','no_hr',0)),'macro_f1':jackknife_summary(vals['ln_f1'],delta(full,'lir_pptd','no_hr',1))}}}
    out_loo_json.write_text(json.dumps(loo_report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out-csv',type=Path,required=True); p.add_argument('--out-json',type=Path,required=True); p.add_argument('--out-loo-csv',type=Path,required=True); p.add_argument('--out-loo-json',type=Path,required=True); a=p.parse_args(); main(a.root,a.out_csv,a.out_json,a.out_loo_csv,a.out_loo_json)
