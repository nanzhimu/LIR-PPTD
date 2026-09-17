from __future__ import annotations
from dataclasses import dataclass
from .encoding import FixedPointError
from .profile import FixedBackendProfile

@dataclass(frozen=True)
class PrimitiveResult:
 primitive:str; output:int; raw:int; remainder:int; source_reference:str

def _safe(value:int,p:FixedBackendProfile,raw:bool=False)->None:
 bound=p.no_wrap_bounds['raw_product' if raw else 'state']
 if abs(value)>bound or 2*abs(value)>=p.prime_p: raise FixedPointError('NO_WRAP_VIOLATION','certified integer bound exceeded')
def local_add(a:int,b:int,p:FixedBackendProfile)->PrimitiveResult:
 z=a+b; _safe(z,p); return PrimitiveResult('local_add',z,z,0,'eq:fixed-grid')
def local_sub(a:int,b:int,p:FixedBackendProfile)->PrimitiveResult:
 z=a-b; _safe(z,p); return PrimitiveResult('local_sub',z,z,0,'eq:balanced-representative')
def fixed_truncate(raw:int,p:FixedBackendProfile,nonnegative:bool=True)->PrimitiveResult:
 _safe(raw,p,True)
 if nonnegative and raw<0: raise FixedPointError('NEGATIVE_POSITIVE_OPERAND','positive primitive received negative input')
 q=raw//p.scale_delta if raw>=0 else -((-raw)//p.scale_delta)
 return PrimitiveResult('fixed_truncate',q,raw,abs(raw)%p.scale_delta,'eq:nonnegative-quantizer' if nonnegative else 'eq:signed-quantizer')
def mul_positive(a:int,b:int,p:FixedBackendProfile)->PrimitiveResult:
 if a<0 or b<0: raise FixedPointError('NEGATIVE_POSITIVE_OPERAND','positive multiplication operands required')
 r=fixed_truncate(a*b,p); return PrimitiveResult('SecMulPositive',r.output,r.raw,r.remainder,'eq:secure-multiplication')
def square(a:int,p:FixedBackendProfile)->PrimitiveResult:
 r=fixed_truncate(a*a,p); return PrimitiveResult('SecSqr',r.output,r.raw,r.remainder,'eq:secure-square')
def pub_mul_positive(a:int,gamma:int,p:FixedBackendProfile,strict_positive:bool=False)->PrimitiveResult:
 if strict_positive and gamma==0: raise FixedPointError('COEFFICIENT_QUANTIZED_ZERO','positive coefficient was removed')
 r=mul_positive(a,gamma,p); return PrimitiveResult('PubMulPositive',r.output,r.raw,r.remainder,'eq:public-mul')
def div_positive(u:int,v:int,p:FixedBackendProfile,lower:int,upper:int)->PrimitiveResult:
 if u<0 or v<=0: raise FixedPointError('INVALID_POSITIVE_DIVISION','positive division precondition failed')
 if v<lower: raise FixedPointError('DENOMINATOR_BELOW_BOUND','denominator below public bound')
 if v>upper: raise FixedPointError('DENOMINATOR_ABOVE_BOUND','denominator above public bound')
 if u>v: raise FixedPointError('NUMERATOR_EXCEEDS_DENOMINATOR','requires u<=v')
 raw=u*p.scale_delta; _safe(raw,p,True); q,rem=divmod(raw,v)
 return PrimitiveResult('DivPositive',q,raw,rem,'eq:division-function-range--eq:division-error')
