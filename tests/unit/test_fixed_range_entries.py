import pytest
from lir_pptd.fixedpoint import default_profile,preflight
from lir_pptd.fixedpoint.range_analysis import PreflightStatus
REQUIRED=('report_encoding','reputation_encoding','epsilon_c','tau','eta','kappa','beta0','beta1','beta2','report_difference','coordinate_square','dimension_distance_sum','tau_plus_distance','q_denominator','q_output','influence','minimum_positive_influence','influence_sum_denominator','aggregate_numerator','aggregate_denominator','aggregate_truth','d_out','q_out','c_times_q','reputation_beta0_term','reputation_beta1_term','reputation_beta2_term','reputation_candidate_sum','canonical_modulo_representation','balanced_representation','full_flow_maximum')
def cert(**kw):return preflight(default_profile(),participants=kw.pop('N',2),dimension=kw.pop('D',2),K=2,epsilon_c='1',tau=kw.pop('tau','1'),eta='1/2',kappa='1',**kw)
@pytest.mark.parametrize('entry_id',REQUIRED)
def test_independently_derived_range_entry(entry_id):
 e=next(x for x in cert().range_entries if x.entry_id==entry_id);assert e.derivation_formula and e.dependencies and e.assumptions and e.source_reference;assert e.encoded_lower_bound<=e.encoded_upper_bound and e.maximum_absolute_integer==max(abs(e.encoded_lower_bound),abs(e.encoded_upper_bound))
def test_dimension_sensitivity_is_local():
 a,b=cert(D=1),cert(D=4);A={x.entry_id:x for x in a.range_entries};B={x.entry_id:x for x in b.range_entries};assert B['dimension_distance_sum'].maximum_absolute_integer>A['dimension_distance_sum'].maximum_absolute_integer;assert B['coordinate_square'].maximum_absolute_integer==A['coordinate_square'].maximum_absolute_integer;assert B['report_encoding'].maximum_absolute_integer==A['report_encoding'].maximum_absolute_integer
def test_participant_sensitivity_is_local():
 A={x.entry_id:x for x in cert(N=2).range_entries};B={x.entry_id:x for x in cert(N=5).range_entries};assert B['influence_sum_denominator'].maximum_absolute_integer>A['influence_sum_denominator'].maximum_absolute_integer;assert B['aggregate_numerator'].maximum_absolute_integer>A['aggregate_numerator'].maximum_absolute_integer;assert B['coordinate_square'].maximum_absolute_integer==A['coordinate_square'].maximum_absolute_integer
def test_domain_tau_reputation_sensitivity():
 A={x.entry_id:x for x in cert().range_entries};B={x.entry_id:x for x in cert(report_upper='2',reputation_upper='2',tau='2').range_entries};assert B['report_encoding'].maximum_absolute_integer>A['report_encoding'].maximum_absolute_integer;assert B['coordinate_square'].maximum_absolute_integer>A['coordinate_square'].maximum_absolute_integer;assert B['reputation_encoding'].maximum_absolute_integer>A['reputation_encoding'].maximum_absolute_integer;assert B['tau'].maximum_absolute_integer>A['tau'].maximum_absolute_integer
def test_coefficient_and_denominator_rejections():
 assert preflight(default_profile(),participants=2,dimension=1,K=1,epsilon_c='1',tau='0',eta='1/2',kappa='1').reason_code=='DENOMINATOR_BELOW_BOUND';assert preflight(default_profile(),participants=2,dimension=1,K=1,epsilon_c='1',tau='1',eta='1/33554432',kappa='1').reason_code=='COEFFICIENT_QUANTIZED_ZERO'

def test_geometry_aware_categorical_distance_certificate_is_dimension_invariant():
 a=preflight(default_profile(),participants=2,dimension=4,K=2,epsilon_c='1/1024',tau='2/5',eta='1/10',kappa='2',task_kind='categorical')
 b=preflight(default_profile(),participants=2,dimension=32,K=2,epsilon_c='1/1024',tau='2/5',eta='1/10',kappa='2',task_kind='categorical')
 assert a.preflight_status==b.preflight_status==PreflightStatus.PASSED
 A={x.entry_id:x for x in a.range_entries};B={x.entry_id:x for x in b.range_entries}
 assert A['dimension_distance_sum'].semantic_upper_bound=='2'
 assert B['dimension_distance_sum'].semantic_upper_bound=='2'
 assert A['tau_plus_distance'].semantic_upper_bound==B['tau_plus_distance'].semantic_upper_bound=='12/5'
 assert a.public_bounds['squared_distance_bound']=='2'==b.public_bounds['squared_distance_bound']

def test_geometry_aware_numerical_certificate_preserves_D_bound():
 c=preflight(default_profile(),participants=2,dimension=5,K=2,epsilon_c='1/1024',tau='1',eta='1/10',kappa='2',task_kind='numerical')
 C={x.entry_id:x for x in c.range_entries}
 assert C['dimension_distance_sum'].semantic_upper_bound=='5'
 assert C['tau_plus_distance'].semantic_upper_bound=='6'
 assert c.public_bounds['squared_distance_bound']=='5'
