from __future__ import annotations
from dataclasses import asdict
from .arithmetic import *
from .encoding import FixedPointError,decimal,rational,trunc_scaled
from .profile import FixedBackendProfile
from .range_analysis import PreflightStatus,preflight
from .reputation import update_reputation
from .types import *
from ..core.consistency_scale import squared_distance_bound_from_modality
from ..core.exact import ExactTaskInput

def _pt(r:PrimitiveResult)->FixedPrimitiveTrace: return FixedPrimitiveTrace(primitive=r.primitive,output_integer=r.output,raw_integer=r.raw,truncation_remainder=r.remainder,source_reference=r.source_reference)
def run_fixed(task:ExactTaskInput,p:FixedBackendProfile)->FixedTaskResult|FixedTaskFailure:
 try:
  if task.task_kind not in {'numerical','categorical'}: return FixedTaskFailure(reason_code='UNKNOWN_TASK_KIND',message='unknown task kind')
  ids=tuple(sorted(task.participant_ids));
  if not ids:return FixedTaskFailure(reason_code='EMPTY_PARTICIPANTS',message='empty participants')
  dims={len(task.reports[w]) for w in ids};
  if len(dims)!=1 or not dims or next(iter(dims))<1:return FixedTaskFailure(reason_code='DIMENSION_MISMATCH',message='dimensions differ')
  D=next(iter(dims)); distance_bound=int(squared_distance_bound_from_modality(task.task_kind,D)); cert=preflight(p,participants=len(ids),dimension=D,K=task.K,epsilon_c=task.epsilon_c,tau=task.tau,eta=task.eta,kappa=task.kappa,task_kind=task.task_kind)
  if cert.preflight_status!=PreflightStatus.PASSED:return FixedTaskFailure(reason_code=cert.reason_code or 'PREFLIGHT_UNRESOLVED',message='static preflight did not pass')
  delta=p.scale_delta; reports={w:tuple(trunc_scaled(x,delta) for x in task.reports[w]) for w in ids}
  if any(x<0 or x>delta for v in reports.values() for x in v):return FixedTaskFailure(reason_code='REPORT_OUT_OF_DOMAIN',message='report outside [0,1]')
  if task.task_kind=='categorical' and any(any(x not in (0,delta) for x in v) or sum(v)!=delta for v in reports.values()):return FixedTaskFailure(reason_code='INVALID_ONE_HOT',message='one-hot required')
  reps={w:trunc_scaled(task.reputations[w],delta) for w in ids}
  if any(x<0 or x>delta for x in reps.values()):return FixedTaskFailure(reason_code='REPUTATION_OUT_OF_RANGE',message='reputation outside [0,1]')
  eps=trunc_scaled(task.epsilon_c,delta); tau=trunc_scaled(task.tau,delta)
  e={w:local_add(reps[w],eps,p).output for w in ids}; den=sum(e.values()); truth=[]
  for h in range(D): truth.append(div_positive(sum(mul_positive(e[w],reports[w][h],p).output for w in ids),den,p,1,p.no_wrap_bounds['state']).output)
  traj=[tuple(truth)]; traces=[]
  for k in range(task.K):
   prim=[]; dist={}; q={}; inf={}
   for w in ids:
    squares=[]
    for h in range(D):
     r=local_sub(reports[w][h],truth[h],p); s=square(r.output,p); prim.extend((_pt(r),_pt(s))); squares.append(s.output)
    dist[w]=sum(squares); dv=local_add(tau,dist[w],p); qr=div_positive(tau,dv.output,p,tau,tau+distance_bound*delta); ir=mul_positive(e[w],qr.output,p); prim.extend((_pt(dv),_pt(qr),_pt(ir)));q[w]=qr.output;inf[w]=ir.output
   aden=sum(inf.values()); nums=tuple(sum(mul_positive(inf[w],reports[w][h],p).output for w in ids) for h in range(D)); truth=tuple(div_positive(n,aden,p,1,p.no_wrap_bounds['state']).output for n in nums); traj.append(truth)
   traces.append(FixedIterationTrace(iteration=k,truth_integer=truth,truth_decoded=tuple(decimal(x,delta) for x in truth),distance_integer=dist,q_integer=q,influence_integer=inf,aggregate_numerator=nums,aggregate_denominator=aden,primitives=tuple(prim),score_sum_residual=decimal(abs(sum(truth)-delta),delta) if task.task_kind=='categorical' else None))
  evid=[]; qout={}
  for w in ids:
   d=sum(square(reports[w][h]-truth[h],p).output for h in range(D)); q=div_positive(tau,tau+d,p,tau,tau+distance_bound*delta).output;qout[w]=q;evid.append(FixedWorkerEvidence(worker_id=w,distance_out_integer=d,q_out_integer=q,distance_out_decoded=decimal(d,delta),q_out_decoded=decimal(q,delta)))
  transitions=tuple(update_reputation(w,reps[w],qout[w],task.eta,task.kappa,task.epochs.get(w,0),p) for w in ids)
  non=tuple(sorted(set(task.reputations)-set(ids))); carry=tuple(FixedNonparticipantCarryForward(worker_id=w,reputation_integer=trunc_scaled(task.reputations[w],delta),epoch=task.epochs.get(w,0)) for w in non)
  released=min(i for i,x in enumerate(truth,1) if x==max(truth)) if task.task_kind=='categorical' else None
  diags=(FixedDiagnosticResult(diagnostic='fixed_k',eligibility='eligible',passed=len(traces)==task.K,note='public fixed loop'),FixedDiagnosticResult(diagnostic='fixed_mm',eligibility='ineligible',passed=False,note='requires task-specific certified objective margin'),FixedDiagnosticResult(diagnostic='fixed_direction',eligibility='unresolved',passed=False,note='requires change exceeding 7/Delta'))
  return FixedTaskResult(task_id=task.task_id,task_kind=task.task_kind,profile_id=p.profile_id,backend_profile_hash=str(p.digest()),K=task.K,iterations_executed=len(traces),participant_ids=ids,nonparticipant_ids=non,input_report_integers=reports,initial_reputation_integers=reps,truth_integer_trajectory=tuple(traj),truth_decoded_trajectory=tuple(tuple(decimal(x,delta) for x in row) for row in traj),iterations=tuple(traces),final_output_integer=tuple(truth),final_output_decoded=tuple(decimal(x,delta) for x in truth),released_class_index=released,output_evidence=tuple(evid),reputation_transitions=transitions,nonparticipant_carry_forward=carry,preflight_certificate=asdict(cert),diagnostics=diags,source_references=('eq:secure-regularized-reputation--eq:fixed-output-evidence','eq:secure-reputation-candidate'))
 except (FixedPointError,ValueError,KeyError,TypeError) as e:return FixedTaskFailure(reason_code=getattr(e,'reason_code','INVALID_FIXED_INPUT'),message=str(e))
