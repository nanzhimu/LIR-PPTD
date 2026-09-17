from __future__ import annotations
from typing import Any
from ..core import ExactTaskInput
from ..fixedpoint import default_profile,run_fixed
from ..fixedpoint.types import FixedTaskFailure
from .secure_algorithm import run_secure
from .simulated_shamir import SimulatedShamirBackend
from .types import SecureConformanceResult,SecureTaskResult

FIELDS=('input_report_integers','initial_reputation_integers','truth_integer_trajectory','iterations','final_output_integer','output_evidence','reputation_transitions','nonparticipant_carry_forward','released_class_index')
def _project_l2(raw):
    return {'input_report_integers':raw['input_report_integers'],'initial_reputation_integers':raw['initial_reputation_integers'],'truth_integer_trajectory':raw['truth_integer_trajectory'],'iterations':tuple({'iteration':x['iteration'],'truth_integer':x['truth_integer'],'distance_integer':x['distance_integer'],'q_integer':x['q_integer'],'influence_integer':x['influence_integer'],'aggregate_numerator':x['aggregate_numerator'],'aggregate_denominator':x['aggregate_denominator']} for x in raw['iterations']),'final_output_integer':raw['final_output_integer'],'output_evidence':tuple({'worker_id':x['worker_id'],'distance_out_integer':x['distance_out_integer'],'q_out_integer':x['q_out_integer']} for x in raw['output_evidence']),'reputation_transitions':tuple({'worker_id':x['worker_id'],'previous_integer':x['previous_integer'],'candidate_integer':x['candidate_integer'],'update_count':x['update_count'],'previous_epoch':x['previous_epoch'],'next_epoch':x['next_epoch']} for x in raw['reputation_transitions']),'nonparticipant_carry_forward':raw['nonparticipant_carry_forward'],'released_class_index':raw['released_class_index']}
def _leaves(value:Any,path=''):
    if isinstance(value,dict):
        for key in sorted(value):yield from _leaves(value[key],f'{path}.{key}')
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):yield from _leaves(item,f'{path}[{i}]')
    else:yield path,value
def compare_results(l2:dict,l3:dict):
    a=dict(_leaves(l2));b=dict(_leaves(l3));keys=set(a)|set(b);mismatches=tuple(sorted(k for k in keys if a.get(k)!=b.get(k)));return len(keys),mismatches
def validate_task(task:ExactTaskInput,backend:SimulatedShamirBackend|None=None)->SecureConformanceResult:
    secure=run_secure(task,backend);fixed=run_fixed(task,default_profile())
    if isinstance(fixed,FixedTaskFailure):raise ValueError(fixed.reason_code)
    l2=_project_l2(fixed.model_dump(mode='json'));count,mismatches=compare_results(l2,secure.l3_result);return SecureConformanceResult(task_id=task.task_id,status='passed' if not mismatches else 'failed',compared_fields=count,mismatch_count=len(mismatches),V_sec=len(mismatches))
def transcript_has_public_leakage(backend:SimulatedShamirBackend)->bool:
    deny=('y_value','reconstructed_secret','secret_value','report_value','reputation_value','share_value','polynomial_coefficients','internal_opened_value')
    return any(any(token in entry.model_dump_json() for token in deny) for entry in backend.export_transcript(public=True).entries)
