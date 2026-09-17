from __future__ import annotations
from typing import Any,Literal
from pydantic import BaseModel,ConfigDict
class M(BaseModel): model_config=ConfigDict(extra='forbid',frozen=True)
class FixedPrimitiveTrace(M): primitive:str; output_integer:int; raw_integer:int; truncation_remainder:int; source_reference:str
class FixedIterationTrace(M): iteration:int; truth_integer:tuple[int,...]; truth_decoded:tuple[str,...]; distance_integer:dict[str,int]; q_integer:dict[str,int]; influence_integer:dict[str,int]; aggregate_numerator:tuple[int,...]; aggregate_denominator:int; primitives:tuple[FixedPrimitiveTrace,...]; score_sum_residual:str|None=None
class FixedWorkerEvidence(M): worker_id:str; distance_out_integer:int; q_out_integer:int; distance_out_decoded:str; q_out_decoded:str
class FixedReputationTransition(M): worker_id:str; previous_integer:int; beta_integers:tuple[int,int,int]; cq_integer:int; term_integers:tuple[int,int,int]; candidate_integer:int; candidate_decoded:str; update_count:Literal[1]=1; previous_epoch:int; next_epoch:int
class FixedNonparticipantCarryForward(M): worker_id:str; reputation_integer:int; epoch:int
class FixedDiagnosticResult(M): diagnostic:str; eligibility:Literal['eligible','ineligible','unresolved']; passed:bool; note:str
class FixedConformanceResult(M): component:str; actual_error:str; certified_bound:str; error_sources:tuple[str,...]; eligibility:Literal['eligible','ineligible','unresolved']; passed:bool
class FixedTaskFailure(M): success:Literal[False]=False; reason_code:str; message:str; reputation_updated:Literal[False]=False; raw_success_written:Literal[False]=False
class FixedTaskResult(M):
 success:Literal[True]=True; backend_kind:Literal['deterministic_fixed_plaintext']='deterministic_fixed_plaintext'; result_scope:Literal['algorithmic_conformance']='algorithmic_conformance'; production_mpc_used:Literal[False]=False; real_data_used:Literal[False]=False; distributed_e8b_used:Literal[False]=False; cryptographic_security_claim:Literal[False]=False
 task_id:str; task_kind:Literal['numerical','categorical']; profile_id:str; backend_profile_hash:str; preflight_status:Literal['passed']='passed'; algorithm_executed:Literal[True]=True; K:int; iterations_executed:int; q_out_recomputed:Literal[True]=True; reputation_updates_per_participant:Literal[1]=1
 participant_ids:tuple[str,...]; nonparticipant_ids:tuple[str,...]; input_report_integers:dict[str,tuple[int,...]]; initial_reputation_integers:dict[str,int]; truth_integer_trajectory:tuple[tuple[int,...],...]; truth_decoded_trajectory:tuple[tuple[str,...],...]; iterations:tuple[FixedIterationTrace,...]; final_output_integer:tuple[int,...]; final_output_decoded:tuple[str,...]; released_class_index:int|None=None; output_evidence:tuple[FixedWorkerEvidence,...]; reputation_transitions:tuple[FixedReputationTransition,...]; nonparticipant_carry_forward:tuple[FixedNonparticipantCarryForward,...]; preflight_certificate:dict[str,Any]; diagnostics:tuple[FixedDiagnosticResult,...]; conformance:tuple[FixedConformanceResult,...]=(); source_references:tuple[str,...]
