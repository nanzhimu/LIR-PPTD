from __future__ import annotations
import sys,time,statistics,csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/final_scientific_closure_p0/fastpath_rerun'; sys.path.insert(0,str(ROOT/'src'))
from lir_pptd.benchmarks.minimal_shamir_kernel import MinimalShamirKernel
DELTA=1<<24; K=10; WARM=5; REPS=20

def prepare(k,m,D):
    # Deterministic nonuniform classes and reputation states in [0.2,0.9].
    reports=[]; reps=[]
    for i in range(m):
        c=(7*i+3)%D; reports.append(tuple(k.share_secret(DELTA if h==c else 0) for h in range(D)))
        rv=int((0.2+0.7*((37*i+11)%101)/100.0)*DELTA); reps.append(k.share_secret(rv))
    eps=k.share_secret(DELTA//1024); tau=k.share_secret((2*DELTA)//5) # categorical lambda=.2 * diam^2=2 => .4
    return reports,reps,eps,tau

def init(k,prepared,D):
    reports,reps,eps,tau=prepared;m=len(reports);eff=[k.local_add(c,eps) for c in reps];den=eff[0]
    for x in eff[1:]: den=k.local_add(den,x)
    truth=[]
    for h in range(D):
        num=k.secure_mul(eff[0],reports[0][h],scale=DELTA)
        for i in range(1,m):num=k.local_add(num,k.secure_mul(eff[i],reports[i][h],scale=DELTA))
        truth.append(k.secure_div(num,den,scale=DELTA))
    return eff,truth

def run_fast(k,prepared,D):
    eff,truth=init(k,prepared,D);return tuple(k.reconstruct_signed(x) for x in truth)

def run_full(k,prepared,D):
    reports,reps,eps,tau=prepared;m=len(reports);eff,truth=init(k,prepared,D)
    for _ in range(K):
        inf=[]
        for i in range(m):
            d=None
            for h in range(D):
                diff=k.local_sub(reports[i][h],truth[h]);sq=k.secure_sqr(diff,scale=DELTA);d=sq if d is None else k.local_add(d,sq)
            q=k.secure_div(tau,k.local_add(tau,d),scale=DELTA);inf.append(k.secure_mul(eff[i],q,scale=DELTA))
        aden=inf[0]
        for x in inf[1:]:aden=k.local_add(aden,x)
        nxt=[]
        for h in range(D):
            num=k.secure_mul(inf[0],reports[0][h],scale=DELTA)
            for i in range(1,m):num=k.local_add(num,k.secure_mul(inf[i],reports[i][h],scale=DELTA))
            nxt.append(k.secure_div(num,aden,scale=DELTA))
        truth=nxt
    # Current ordinary circuit re-evaluates final nonpersistent q_out.
    for i in range(m):
        d=None
        for h in range(D):
            diff=k.local_sub(reports[i][h],truth[h]);sq=k.secure_sqr(diff,scale=DELTA);d=sq if d is None else k.local_add(d,sq)
        _=k.secure_div(tau,k.local_add(tau,d),scale=DELTA)
    return tuple(k.reconstruct_signed(x) for x in truth)

def one(method,m,D,rep):
    seed=990000+D*10000+m*100+rep
    k=MinimalShamirKernel(seed=seed);prep=prepare(k,m,D);k.reset_counters();st=time.perf_counter();out=run_full(k,prep,D) if method=='full' else run_fast(k,prep,D);dt=time.perf_counter()-st;return dt,k.counters.as_dict(),out

def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=[];summary=[]
    for D in (2,4):
      for m in (20,50,100,200):
        for r in range(WARM): one('full',m,D,-100-r); one('fast',m,D,-100-r)
        for r in range(REPS):
          for method in ('full','fast'):
            dt,c,out=one(method,m,D,r);raw.append({'D':D,'m':m,'rep':r,'method':method,'seconds':dt,**c})
        f=[x['seconds'] for x in raw if x['D']==D and x['m']==m and x['method']=='full'];q=[x['seconds'] for x in raw if x['D']==D and x['m']==m and x['method']=='fast'];red=1-statistics.fmean(q)/statistics.fmean(f)
        summary.append({'D':D,'m':m,'full_mean_ms':1000*statistics.fmean(f),'fast_mean_ms':1000*statistics.fmean(q),'full_median_ms':1000*statistics.median(f),'fast_median_ms':1000*statistics.median(q),'runtime_reduction_fraction':red,'full_nonlinear_calls':next(x['nonlinear_or_rescaled_calls'] for x in raw if x['D']==D and x['m']==m and x['method']=='full'),'fast_nonlinear_calls':next(x['nonlinear_or_rescaled_calls'] for x in raw if x['D']==D and x['m']==m and x['method']=='fast')});print(summary[-1],flush=True)
    for path,rows in [(OUT/'kernel_timing_runs.csv',raw),(OUT/'kernel_timing_summary.csv',summary)]:
      with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (OUT/'kernel_benchmark_summary.json').write_text(json.dumps({'warmups':WARM,'measured_pairs':REPS,'kernel':'MinimalShamirKernel/common 521-bit field/N=10/T=4','rows':summary},indent=2)+'\n')
if __name__=='__main__':main()
