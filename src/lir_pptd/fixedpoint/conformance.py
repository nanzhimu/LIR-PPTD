from __future__ import annotations
from collections import Counter
from fractions import Fraction
from typing import Literal
from pydantic import BaseModel,ConfigDict
from ..canonical import CanonicalRational
from ..core import ExactTaskInput,run_with_precision_doubling
from .algorithm import run_fixed
from .encoding import rational
from .types import FixedTaskFailure,FixedConformanceResult
class M(BaseModel):model_config=ConfigDict(extra='forbid',frozen=True)
class ConformanceValue(M):exact_value:str;fixed_encoded_integer:int;fixed_decoded_value:str;canonical_actual_error:CanonicalRational;canonical_certified_bound:CanonicalRational;decimal_actual_error:str;decimal_certified_bound:str
class ConformanceRecord(M):record_id:str;task_id:str;task_kind:str;component:str;iteration:int|None;worker_id:str|None;coordinate:int|None;value:ConformanceValue;error_source:str;bound_derivation_id:str;eligibility:Literal['eligible','ineligible','unresolved'];comparison_status:Literal['passed','failed'];theorem_status:Literal['supported','not_applicable','not_established'];source_reference:str;note:str=''
class ConformanceRecordSet(M):schema_version:str='1.0';task_id:str;task_kind:str;profile_id:str;backend_profile_hash:str;K:int;participant_ids:tuple[str,...];nonparticipant_ids:tuple[str,...];dimension:int;category_count:int|None;records:tuple[ConformanceRecord,...];component_counts:dict[str,int];expected_component_counts:dict[str,int];missing_components:tuple[str,...];duplicate_record_ids:tuple[str,...];failed_record_ids:tuple[str,...];unresolved_mandatory_record_ids:tuple[str,...];record_set_status:Literal['complete','failed']
class ConformanceSummary(M):total_tasks:int;complete_record_sets:int;incomplete_record_sets:int;total_records:int;passed_records:int;failed_records:int;eligible_records:int;ineligible_records:int;unresolved_diagnostic_records:int;missing_records:int;duplicate_records:int;status:Literal['passed','failed']
BOUND_REGISTRY={'grid':{'formula':'64/Delta cumulative operation contract','dependencies':['Delta','operation sequence'],'source_reference':'eq:cumulative-error-theorem'}}
def F(s):
 try:return rational(s)
 except:return Fraction(s)
def rec(t,c,a,z,d,it=None,w=None,h=None,b=64,elig='eligible',th='supported'):
 e=abs(F(a)-Fraction(z,d));bd=Fraction(b,d);cv=ConformanceValue(exact_value=str(a),fixed_encoded_integer=z,fixed_decoded_value=f'{z}/{d}',canonical_actual_error=CanonicalRational(numerator=str(e.numerator),denominator=str(e.denominator)),canonical_certified_bound=CanonicalRational(numerator=str(bd.numerator),denominator=str(bd.denominator)),decimal_actual_error=str(e.numerator/e.denominator),decimal_certified_bound=str(bd.numerator/bd.denominator));return ConformanceRecord(record_id='|'.join(map(str,(t.task_id,c,it,w,h))),task_id=t.task_id,task_kind=t.task_kind,component=c,iteration=it,worker_id=w,coordinate=h,value=cv,error_source='encoding/truncation/division accumulation',bound_derivation_id='grid',eligibility=elig,comparison_status='passed' if e<=bd else 'failed',theorem_status=th,source_reference='eq:cumulative-error-theorem')
def expected_counts(k,K,N,D):
 b={'distance':K*N,'q':K*N,'influence':K*N,'aggregation_numerator':K*D,'aggregation_denominator':K,'d_out':N,'q_out':N,'next_reputation':N}
 b.update({'initial_truth':D,'iteration_truth':K*D,'final_output':D} if k=='numerical' else {'initial_category_score':D,'iteration_category_score':K*D,'final_category_score':D,'score_sum_residual':K+1,'published_label':1,'margin_eligibility':1});return b
def generate_record_set(t:ExactTaskInput,p):
 x=run_with_precision_doubling(t);f=run_fixed(t,p)
 if isinstance(f,FixedTaskFailure):raise ValueError(f.reason_code)
 d=p.scale_delta;R=[];cat=t.task_kind=='categorical';pre='category_score' if cat else 'truth'
 for h,(a,z) in enumerate(zip(x.truth_trajectory[0],f.truth_integer_trajectory[0])):R.append(rec(t,'initial_'+pre,a,z,d,0,h=h))
 for k,(a,z) in enumerate(zip(x.iterations,f.iterations)):
  for h,(v,q) in enumerate(zip(a.truth,z.truth_integer)):R.append(rec(t,'iteration_'+pre,v,q,d,k+1,h=h))
  for w in f.participant_ids:
   for c,v,q in [('distance',a.distance_by_worker[w],z.distance_integer[w]),('q',a.evidence_by_worker[w],z.q_integer[w]),('influence',a.influence_by_worker[w],z.influence_integer[w])]:R.append(rec(t,c,v,q,d,k,w))
  for h,q in enumerate(z.aggregate_numerator):R.append(rec(t,'aggregation_numerator',str(F(a.denominator)*F(a.truth[h])),q,d,k,h=h))
  R.append(rec(t,'aggregation_denominator',a.denominator,z.aggregate_denominator,d,k))
 final='final_category_score' if cat else 'final_output'
 for h,(a,z) in enumerate(zip(x.final_output,f.final_output_integer)):R.append(rec(t,final,a,z,d,h=h))
 for a,z in zip(x.output_evidence,f.output_evidence):R.extend([rec(t,'d_out',a.distance_out,z.distance_out_integer,d,w=z.worker_id),rec(t,'q_out',a.evidence_out,z.q_out_integer,d,w=z.worker_id)])
 for a,z in zip(x.reputation_transitions,f.reputation_transitions):R.append(rec(t,'next_reputation',a.next,z.candidate_integer,d,w=z.worker_id,b=7))
 if cat:
  for k,row in enumerate(f.truth_integer_trajectory):R.append(rec(t,'score_sum_residual','0',abs(sum(row)-d),d,k))
  R.append(rec(t,'published_label',str(x.released_class_index),f.released_class_index or 0,1,b=0));v=sorted(map(F,x.final_output),reverse=True);tie=v[0]==v[1];margin=v[0]-v[1];R.append(rec(t,'margin_eligibility',str(margin),int(margin*d),d,b=0,elig='ineligible' if tie else 'eligible',th='not_applicable' if tie else 'supported'))
 counts=dict(Counter(q.component for q in R));exp=expected_counts(t.task_kind,t.K,len(t.participant_ids),len(next(iter(t.reports.values()))));missing=tuple(k for k,v in exp.items() if counts.get(k)!=v);ids=[q.record_id for q in R];dups=tuple(k for k,v in Counter(ids).items() if v>1);failed=tuple(q.record_id for q in R if q.comparison_status=='failed');status='complete' if not missing and not dups and not failed else 'failed';return ConformanceRecordSet(task_id=t.task_id,task_kind=t.task_kind,profile_id=p.profile_id,backend_profile_hash=str(p.digest()),K=t.K,participant_ids=f.participant_ids,nonparticipant_ids=f.nonparticipant_ids,dimension=len(next(iter(t.reports.values()))),category_count=len(next(iter(t.reports.values()))) if cat else None,records=tuple(R),component_counts=counts,expected_component_counts=exp,missing_components=missing,duplicate_record_ids=dups,failed_record_ids=failed,unresolved_mandatory_record_ids=(),record_set_status=status)
def summarize(S):
 R=[r for s in S for r in s.records];n=sum(s.record_set_status=='complete' for s in S);return ConformanceSummary(total_tasks=len(S),complete_record_sets=n,incomplete_record_sets=len(S)-n,total_records=len(R),passed_records=sum(r.comparison_status=='passed' for r in R),failed_records=sum(r.comparison_status=='failed' for r in R),eligible_records=sum(r.eligibility=='eligible' for r in R),ineligible_records=sum(r.eligibility=='ineligible' for r in R),unresolved_diagnostic_records=sum(r.eligibility=='unresolved' for r in R),missing_records=sum(len(s.missing_components) for s in S),duplicate_records=sum(len(s.duplicate_record_ids) for s in S),status='passed' if n==len(S) else 'failed')
def compare_integer_decoded(component,exact,fixed,bound,source='quantization/truncation'):
 from decimal import Decimal
 e=abs(Decimal(exact)-Decimal(fixed));b=Decimal(bound);return FixedConformanceResult(component=component,actual_error=str(e),certified_bound=bound,error_sources=(source,),eligibility='eligible',passed=e<=b)
