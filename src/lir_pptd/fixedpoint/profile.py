from __future__ import annotations
from typing import Any, Literal
from pydantic import Field, field_validator, model_validator
from ..canonical import CanonicalRational, parse_rational
from ..config import CanonicalModel
from ..hashing import backend_profile_hash

class FixedBackendProfile(CanonicalModel):
 profile_id:str; backend_kind:Literal['deterministic_fixed_plaintext']; implemented:Literal[True]
 prime_p:int=Field(gt=2); fractional_bits_f:int=Field(ge=1,le=64); scale_delta:int
 signed_encoding:Literal['balanced_mod_p']; canonical_representative_rule:Literal['least_nonnegative_then_balanced']
 quantization_rule:Literal['signed_truncation_toward_zero']; multiplication_truncation_rule:Literal['nonnegative_floor_after_product']
 square_rule:Literal['nonnegative_floor_after_square']; public_positive_multiplication_rule:Literal['nonnegative_floor_after_product']
 positive_division_rule:Literal['nonnegative_floor_ratio_to_grid']
 denominator_public_lower_bound:CanonicalRational; denominator_public_upper_bound:CanonicalRational
 input_range_bounds:dict[str,CanonicalRational]; intermediate_range_bounds:dict[str,CanonicalRational]
 no_wrap_bounds:dict[str,int]; primitive_error_bounds:dict[str,CanonicalRational]
 cumulative_error_policy:str; golden_vector_version:str; production_ready:Literal[False]; cryptographic_security_claim:Literal[False]
 @field_validator('denominator_public_lower_bound','denominator_public_upper_bound',mode='before')
 @classmethod
 def rat(cls,v:Any)->CanonicalRational: return parse_rational(v)
 @field_validator('input_range_bounds','intermediate_range_bounds','primitive_error_bounds',mode='before')
 @classmethod
 def ratmap(cls,v:Any)->dict[str,CanonicalRational]:
  if not isinstance(v,dict) or not v: raise ValueError('contract map required')
  return {str(k):parse_rational(x) for k,x in v.items()}
 @model_validator(mode='after')
 def contract(self)->'FixedBackendProfile':
  if self.prime_p%2==0: raise ValueError('prime_p must be odd')
  if self.scale_delta!=1<<self.fractional_bits_f: raise ValueError('Delta must equal 2^f')
  if self.denominator_public_lower_bound.as_fraction()<=0 or self.denominator_public_upper_bound.as_fraction()<self.denominator_public_lower_bound.as_fraction(): raise ValueError('invalid denominator bounds')
  if not {'raw_product','state'}<=self.no_wrap_bounds.keys(): raise ValueError('no-wrap contracts missing')
  if not {'multiplication','division'}<=self.primitive_error_bounds.keys(): raise ValueError('error contracts missing')
  return self
 def digest(self): return backend_profile_hash(self.schema_version,self.semantic_object())

def default_profile()->FixedBackendProfile:
 d={'power_of_two_exponent':'-24'}
 return FixedBackendProfile(profile_id='l2-plaintext-f24-v1',backend_kind='deterministic_fixed_plaintext',implemented=True,prime_p=170141183460469231731687303715884105727,fractional_bits_f=24,scale_delta=1<<24,signed_encoding='balanced_mod_p',canonical_representative_rule='least_nonnegative_then_balanced',quantization_rule='signed_truncation_toward_zero',multiplication_truncation_rule='nonnegative_floor_after_product',square_rule='nonnegative_floor_after_square',public_positive_multiplication_rule='nonnegative_floor_after_product',positive_division_rule='nonnegative_floor_ratio_to_grid',denominator_public_lower_bound=d,denominator_public_upper_bound='1024',input_range_bounds={'report':'1','reputation':'1'},intermediate_range_bounds={'distance':'64','state':'1024'},no_wrap_bounds={'raw_product':1<<80,'state':1<<48},primitive_error_bounds={'multiplication':d,'division':{'numerator':'1','denominator':1<<23}},cumulative_error_policy='sum certified per-operation residual bounds without cancellation',golden_vector_version='1.0',production_ready=False,cryptographic_security_claim=False)
