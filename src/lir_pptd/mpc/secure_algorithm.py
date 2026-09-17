from __future__ import annotations
from ..core import ExactTaskInput
from ..fixedpoint import default_profile
from ..fixedpoint.encoding import decimal,rational,trunc_scaled
from ..core.consistency_scale import squared_distance_bound_from_modality
from ..fixedpoint.range_analysis import PreflightStatus,preflight
from .simulated_shamir import SimulatedBackendError,SimulatedShamirBackend
from .types import SecureTaskResult

def _ok(result):
    if result.status!='completed' or result.output is None:raise SimulatedBackendError(result.reason_code or 'OPERATION_REJECTED','backend operation rejected')
    return result.output
def _value(backend,bundle):
    result=backend.reconstruct(bundle)
    if result.status!='reconstructed' or result.value is None:raise SimulatedBackendError(result.reason_code or 'RECONSTRUCTION_FAILED','reconstruction failed')
    return result.value if result.value<=backend.profile.field_prime_p//2 else result.value-backend.profile.field_prime_p
def run_secure(task:ExactTaskInput,backend:SimulatedShamirBackend|None=None)->SecureTaskResult:
    backend=backend or SimulatedShamirBackend();p=default_profile();ids=tuple(sorted(task.participant_ids))
    if not ids or len(set(ids))!=len(ids):raise ValueError('invalid participant set')
    if task.task_kind not in {'numerical','categorical'}:raise ValueError('unknown task kind')
    dims={len(task.reports[w]) for w in ids}
    if len(dims)!=1 or next(iter(dims))<1:raise ValueError('dimension mismatch')
    D=next(iter(dims)); distance_bound=int(squared_distance_bound_from_modality(task.task_kind,D));cert=preflight(p,participants=len(ids),dimension=D,K=task.K,epsilon_c=task.epsilon_c,tau=task.tau,eta=task.eta,kappa=task.kappa,task_kind=task.task_kind)
    if cert.preflight_status!=PreflightStatus.PASSED:raise ValueError(cert.reason_code or 'PREFLIGHT_REJECTED')
    delta=p.scale_delta;reports={w:tuple(trunc_scaled(x,delta) for x in task.reports[w]) for w in ids};reps={w:trunc_scaled(task.reputations[w],delta) for w in ids}
    if task.task_kind=='categorical' and any(any(x not in (0,delta) for x in row) or sum(row)!=delta for row in reports.values()):raise ValueError('INVALID_ONE_HOT')
    tau=trunc_scaled(task.tau,delta);eps=trunc_scaled(task.epsilon_c,delta)
    def share(secret,value,namespace='algorithm',iteration=None,worker=None,coordinate=None):backend.set_context(task_id=task.task_id,iteration=iteration,worker_id=worker,coordinate=coordinate);return backend.share_secret(secret,value,namespace=namespace)
    report_shares={w:tuple(share(f'report:{w}:{h}',x,'input',worker=w,coordinate=h) for h,x in enumerate(row)) for w,row in reports.items()};rep_shares={w:share(f'reputation:{w}',x,'input',worker=w) for w,x in reps.items()};eps_s=share('epsilon',eps,'public');tau_s=share('tau',tau,'public')
    effective={w:_ok(backend.local_add(rep_shares[w],eps_s)) for w in ids};den=effective[ids[0]]
    for w in ids[1:]:den=_ok(backend.local_add(den,effective[w]))
    truth=[]
    for h in range(D):
        terms=[_ok(backend.SecMulPositive(effective[w],report_shares[w][h])) for w in ids];num=terms[0]
        for term in terms[1:]:num=_ok(backend.local_add(num,term))
        truth.append(_ok(backend.SecDivPositive(num,den,1,p.no_wrap_bounds['state'])))
    initial=tuple(_value(backend,x) for x in truth);trajectory=[initial];iterations=[]
    for k in range(task.K):
        distances={};qs={};influences={}
        for w in ids:
            squares=[]
            for h in range(D):
                backend.set_context(task_id=task.task_id,iteration=k,worker_id=w,coordinate=h);diff=_ok(backend.local_sub(report_shares[w][h],truth[h]));squares.append(_ok(backend.SecSqr(diff)))
            distance=squares[0]
            for sq in squares[1:]:distance=_ok(backend.local_add(distance,sq))
            tau_distance=_ok(backend.local_add(tau_s,distance));q=_ok(backend.SecDivPositive(tau_s,tau_distance,tau,tau+distance_bound*delta));influence=_ok(backend.SecMulPositive(effective[w],q));distances[w]=distance;qs[w]=q;influences[w]=influence
        aden=influences[ids[0]]
        for w in ids[1:]:aden=_ok(backend.local_add(aden,influences[w]))
        nums=[]
        for h in range(D):
            products=[_ok(backend.SecMulPositive(influences[w],report_shares[w][h])) for w in ids];num=products[0]
            for product in products[1:]:num=_ok(backend.local_add(num,product))
            nums.append(num)
        truth=[_ok(backend.SecDivPositive(num,aden,1,p.no_wrap_bounds['state'])) for num in nums]
        truth_values=tuple(_value(backend,x) for x in truth);trajectory.append(truth_values);iterations.append({'iteration':k,'truth_integer':truth_values,'distance_integer':{w:_value(backend,x) for w,x in distances.items()},'q_integer':{w:_value(backend,x) for w,x in qs.items()},'influence_integer':{w:_value(backend,x) for w,x in influences.items()},'aggregate_numerator':tuple(_value(backend,x) for x in nums),'aggregate_denominator':_value(backend,aden)})
    evidence=[];transitions=[]
    et,kappa=rational(task.eta),rational(task.kappa);beta=tuple(trunc_scaled(x,delta) for x in (1-et*kappa,et,et*(kappa-1)))
    for w in ids:
        squares=[_ok(backend.SecSqr(_ok(backend.local_sub(report_shares[w][h],truth[h])))) for h in range(D)];distance=squares[0]
        for sq in squares[1:]:distance=_ok(backend.local_add(distance,sq))
        q=_ok(backend.SecDivPositive(tau_s,_ok(backend.local_add(tau_s,distance)),tau,tau+distance_bound*delta));cq=_ok(backend.SecMulPositive(rep_shares[w],q));terms=[_ok(backend.PubMulPositive(rep_shares[w],beta[0])),_ok(backend.PubMulPositive(q,beta[1])),_ok(backend.PubMulPositive(cq,beta[2]))];candidate=_ok(backend.local_add(_ok(backend.local_add(terms[0],terms[1])),terms[2]));prepared=backend.prepare_reputation(w,task.epochs.get(w,0),task.epochs.get(w,0)+1);evidence.append({'worker_id':w,'distance_out_integer':_value(backend,distance),'q_out_integer':_value(backend,q)});transitions.append({'worker_id':w,'previous_integer':reps[w],'candidate_integer':_value(backend,candidate),'update_count':prepared.update_count,'previous_epoch':prepared.previous_epoch,'next_epoch':prepared.next_epoch})
    backend.prepare_output(task.task_id,truth[0],True);non=tuple({'worker_id':w,'reputation_integer':trunc_scaled(task.reputations[w],delta),'epoch':task.epochs.get(w,0)} for w in sorted(set(task.reputations)-set(ids)));final=tuple(_value(backend,x) for x in truth);released=min(i for i,x in enumerate(final,1) if x==max(final)) if task.task_kind=='categorical' else None
    l3={'input_report_integers':reports,'initial_reputation_integers':reps,'truth_integer_trajectory':tuple(trajectory),'iterations':tuple(iterations),'final_output_integer':final,'output_evidence':tuple(evidence),'reputation_transitions':tuple(transitions),'nonparticipant_carry_forward':non,'released_class_index':released}
    return SecureTaskResult(task_id=task.task_id,task_kind=task.task_kind,K=task.K,iterations_executed=len(iterations),q_out_recomputed=True,reputation_updates_per_participant=1,nonparticipant_carry_forward=bool(non),l3_result=l3,operation_trace=backend.export_transcript(public=False).entries)
