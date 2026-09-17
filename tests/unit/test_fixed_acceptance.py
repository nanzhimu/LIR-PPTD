from __future__ import annotations
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from lir_pptd.core import ExactTaskInput,run_with_precision_doubling
from lir_pptd.fixedpoint import *
from lir_pptd.fixedpoint.arithmetic import *
from lir_pptd.fixedpoint.conformance import compare_integer_decoded
from lir_pptd.fixedpoint.encoding import *
from lir_pptd.fixedpoint.types import FixedTaskFailure
from lir_pptd.fixedpoint.golden import validate_golden_file
ROOT=Path(__file__).parents[2]
def T(kind='numerical',K=2,reports=None):
 if reports is None: reports={'w1':('0',),'w2':('1',)} if kind=='numerical' else {'w1':('1','0'),'w2':('0','1')}
 return ExactTaskInput(task_id='audit',task_kind=kind,K=K,tau='1',epsilon_c='1',kappa='1',eta='1/2',reports=reports,reputations={'w1':'1/2','w2':'1/2','n':'3/4'},epochs={'w1':1,'w2':2,'n':8},participant_ids=('w2','w1'))
@pytest.mark.parametrize('field,value', [('fractional_bits_f',0),('prime_p',2),('backend_kind','secure_mpc'),('production_ready',True),('cryptographic_security_claim',True)])
def test_profile_invalid_contracts_rejected(field,value):
 d=default_profile().model_dump();d[field]=value
 with pytest.raises(ValidationError): FixedBackendProfile.model_validate(d)
def test_profile_hash_semantics_and_literals():
 p=default_profile(); assert p.digest()==default_profile().digest(); assert p.model_copy(update={'original_literals':{'x':'01'}}).digest()==p.digest(); assert p.model_copy(update={'profile_id':'changed'}).digest()!=p.digest()
@pytest.mark.parametrize('value,expected',[('0',0),('1/16777216',1),('1/2',8388608),('-1/2',-8388608),('1',16777216),('-1',-16777216)])
def test_encoding_grid_and_sign(value,expected):
 p=default_profile(); assert balanced(encode_signed(value,p,'1'),p)==expected
@pytest.mark.parametrize('value,trunc,floor',[('-1/3',-5592405,-5592406),('1/3',5592405,5592405)])
def test_truncation_differs_from_floor(value,trunc,floor): assert trunc_scaled(value,2**24)==trunc and floor_scaled(value,2**24)==floor
def test_encoding_rejects_float_noncanonical_and_range():
 p=default_profile()
 for f in (lambda:encode_signed(.1,p),lambda:balanced(-1,p),lambda:encode_signed('2',p,'1')):
  with pytest.raises(FixedPointError): f()
@pytest.mark.parametrize('a,b,expected',[(0,0,0),(2**24,0,0),(2**24,2**24,2**24),(2**23,2**23,2**22),(1,1,0)])
def test_multiplication_vectors(a,b,expected): assert mul_positive(a,b,default_profile()).output==expected
@pytest.mark.parametrize('a,expected',[(0,0),(1,0),(-2**24,2**24),(2**24,2**24)])
def test_square_vectors(a,expected): assert square(a,default_profile()).output==expected
@pytest.mark.parametrize('u,v,expected',[(0,2**24,0),(2**23,2**24,2**23),(2**24,2**24,2**24)])
def test_division_vectors(u,v,expected): assert div_positive(u,v,default_profile(),1,2**25).output==expected
def test_primitive_negative_bounds_retention_wrap():
 p=default_profile();
 with pytest.raises(FixedPointError): div_positive(1,0,p,1,10)
 with pytest.raises(FixedPointError): div_positive(1,1,p,2,10)
 with pytest.raises(FixedPointError): div_positive(1,11,p,1,10)
 with pytest.raises(FixedPointError): pub_mul_positive(1,0,p,True)
 with pytest.raises(FixedPointError): local_add(p.no_wrap_bounds['state'],1,p)
def test_algorithm_semantics_and_qout_difference():
 t=ExactTaskInput(task_id='asym',task_kind='numerical',K=1,tau='1',epsilon_c='1',kappa='1',eta='1/2',reports={'w1':('0',),'w2':('0',),'w3':('1/4',)},reputations={'w1':'1/2','w2':'0','w3':'0','n':'3/4'},epochs={'w1':1,'w2':2,'w3':3,'n':8},participant_ids=('w1','w2','w3'))
 p=default_profile();r=run_fixed(t,p);assert not isinstance(r,FixedTaskFailure);assert len(r.iterations)==1 and len(r.truth_integer_trajectory)==2; assert any(e.q_out_integer!=r.iterations[-1].q_integer[e.worker_id] for e in r.output_evidence); assert all(x.update_count==1 for x in r.reputation_transitions);assert r.nonparticipant_carry_forward[0].epoch==8
def test_algorithm_kind_validation_determinism_and_k():
 p=default_profile();a=run_fixed(T(),p);b=run_fixed(T(),p);c=run_fixed(T(K=3),p); assert a==b and a.iterations_executed==2 and c.iterations_executed==3
 bad=run_fixed(T('categorical',reports={'w1':('1/2','1/2'),'w2':('0','1')}),p);assert isinstance(bad,FixedTaskFailure) and bad.reason_code=='INVALID_ONE_HOT'
def test_diagnostics_ineligible_unresolved_not_passed():
 r=run_fixed(T(),default_profile());d={x.diagnostic:x for x in r.diagnostics};assert not d['fixed_mm'].passed and d['fixed_mm'].eligibility=='ineligible';assert not d['fixed_direction'].passed and d['fixed_direction'].eligibility=='unresolved'
def test_conformance_failure_when_bound_exceeded(): assert not compare_integer_decoded('x','1','0','0.5').passed
@pytest.mark.parametrize('case_id',['numerical_equal_reports','categorical_unanimous','categorical_tie','reputation_boundaries'])
def test_all_golden_l1_l2_final_conformance(case_id):
 case=next(x for x in json.loads((ROOT/'tests/golden/spec_examples.json').read_text())['cases'] if x['id']==case_id);ids=tuple(f'w{i}' for i in range(len(case['reports'])));t=ExactTaskInput(task_id=case_id,task_kind=case['kind'],K=case['parameters']['K'],tau=case['parameters']['tau'],epsilon_c=case['parameters']['epsilon_c'],kappa=case['parameters']['kappa'],eta=case['parameters']['eta'],reports=dict(zip(ids,map(tuple,case['reports']))),reputations=dict(zip(ids,case['reputations'])),epochs={w:0 for w in ids},participant_ids=ids);l1=run_with_precision_doubling(t);l2=run_fixed(t,default_profile());assert not isinstance(l2,FixedTaskFailure);assert len(l1.truth_trajectory)==len(l2.truth_integer_trajectory);assert all(abs(float(a)-z/default_profile().scale_delta)<1e-4 for a,z in zip(l1.final_output,l2.final_output_integer));assert all(x.update_count==1 for x in l2.reputation_transitions)
def test_golden_vector_schema_and_categories():
 v=json.loads((ROOT/'tests/golden/fixed_primitive_vectors.json').read_text())['vectors'];required={'vector_id','primitive','profile_id','source_reference','semantic_inputs','encoded_inputs','intermediate_integers','truncation_points','expected_output_integer','expected_decoded_output','certified_error_bound','expected_status','expected_reason_code'};assert all(required<=x.keys() for x in v);assert {'encoding','local_add','local_sub','SecMulPositive','SecSqr','PubMulPositive','DivPositive','reputation'}<={x['primitive'] for x in v}
def test_public_golden_vectors_execute_cleanly():
 r=validate_golden_file(ROOT/'tests/golden/fixed_primitive_vectors.json'); assert r['status']=='validated' and r['failed_vectors']==0 and r['passed_vectors']==r['total_vectors']
