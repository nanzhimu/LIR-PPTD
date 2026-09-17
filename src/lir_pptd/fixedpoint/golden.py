from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from pydantic import BaseModel,ConfigDict
from .arithmetic import div_positive,fixed_truncate,local_add,local_sub,mul_positive,pub_mul_positive,square
from .encoding import FixedPointError,balanced,decode,encode_signed
from .profile import default_profile
from .reputation import update_reputation

class GoldenVector(BaseModel):
 model_config=ConfigDict(extra='forbid')
 vector_id:str;category:str;primitive:str;profile_id:str;source_reference:str;semantic_inputs:dict[str,Any];encoded_inputs:dict[str,int];intermediate_integers:list[int];truncation_points:list[str];expected_output_integer:int|None;expected_decoded_output:str|None;certified_error_bound:str;expected_status:str;expected_reason_code:str|None
class GoldenMismatch(AssertionError):pass

def _execute(v:GoldenVector):
 p=default_profile();e=v.encoded_inputs;prim=v.primitive
 if prim=='encoding': out=encode_signed(v.semantic_inputs['value'],p,v.semantic_inputs.get('bound')); raw=[balanced(out,p)]; points=['signed truncation toward zero']
 elif prim=='decoding': out=balanced(e['representative'],p);raw=[out];points=[]
 elif prim=='local_add': r=local_add(e['a'],e['b'],p);out=r.output;raw=[r.raw];points=[]
 elif prim=='local_sub': r=local_sub(e['a'],e['b'],p);out=r.output;raw=[r.raw];points=[]
 elif prim=='fixed_truncate': r=fixed_truncate(e['raw'],p,e.get('nonnegative',1)==1);out=r.output;raw=[r.raw,r.remainder];points=['divide by Delta using declared direction']
 elif prim=='SecMulPositive': r=mul_positive(e['a'],e['b'],p);out=r.output;raw=[r.raw,r.remainder];points=['floor raw product / Delta']
 elif prim=='SecSqr': r=square(e['a'],p);out=r.output;raw=[r.raw,r.remainder];points=['floor square / Delta']
 elif prim=='PubMulPositive': r=pub_mul_positive(e['a'],e['gamma'],p,bool(e.get('strict_positive')));out=r.output;raw=[r.raw,r.remainder];points=['floor public product / Delta']
 elif prim=='DivPositive': r=div_positive(e['u'],e['v'],p,e['lower'],e['upper']);out=r.output;raw=[r.raw,r.remainder];points=['floor u*Delta/v']
 elif prim=='reputation': r=update_reputation('w',e['c'],e['q'],v.semantic_inputs['eta'],v.semantic_inputs['kappa'],0,p);out=r.candidate_integer;raw=[r.cq_integer,*r.term_integers,out];points=['c*q','three public products']
 else: raise FixedPointError('UNKNOWN_PRIMITIVE',prim)
 return out,f'{out}/{p.scale_delta}',raw,points

def execute_vector(v:GoldenVector)->dict[str,Any]:
 try:
  out,dec,raw,points=_execute(v);status='passed';reason=None
 except (FixedPointError,ValueError) as exc:
  out=dec=None;raw=[];points=[];status='rejected';reason=getattr(exc,'reason_code',str(exc))
 actual={'actual_status':status,'actual_reason_code':reason,'actual_output_integer':out,'actual_decoded_output':dec,'actual_intermediate_integers':raw,'actual_truncation_points':points}
 expected=(v.expected_status,v.expected_reason_code,v.expected_output_integer,v.expected_decoded_output,v.intermediate_integers,v.truncation_points)
 got=(status,reason,out,dec,raw,points)
 if got!=expected: raise GoldenMismatch(f'{v.vector_id}: expected={expected!r}, actual={got!r}')
 return actual

def validate_golden_file(path:Path)->dict[str,Any]:
 raw=json.loads(path.read_text(encoding='utf-8'));items=raw.get('vectors',[]);ids=[x.get('vector_id') for x in items];duplicates=sorted(k for k,n in Counter(ids).items() if n>1)
 if duplicates: raise GoldenMismatch(f'duplicate vector ids: {duplicates}')
 vectors=[GoldenVector.model_validate(x) for x in items];failures=[]
 for v in vectors:
  try:execute_vector(v)
  except Exception as exc:failures.append({'vector_id':v.vector_id,'error':str(exc)})
 return {'status':'validated' if not failures else 'failed','profile_id':default_profile().profile_id,'backend_profile_hash':str(default_profile().digest()),'vector_file':str(path),'total_vectors':len(vectors),'passed_vectors':len(vectors)-len(failures),'failed_vectors':len(failures),'rejected_vectors_expected':sum(v.expected_status=='rejected' for v in vectors),'rejected_vectors_matched':sum(v.expected_status=='rejected' for v in vectors)-sum(v.expected_status=='rejected' and any(f['vector_id']==v.vector_id for f in failures) for v in vectors),'category_counts':dict(Counter(v.category for v in vectors)),'duplicate_vector_ids':duplicates,'schema_errors':[],'execution_failures':failures,'all_vectors_executed':True}
