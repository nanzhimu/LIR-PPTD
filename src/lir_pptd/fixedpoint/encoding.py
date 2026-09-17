from __future__ import annotations
from fractions import Fraction
from ..canonical import CanonicalRational, parse_rational
from .profile import FixedBackendProfile

class FixedPointError(ValueError):
 def __init__(self,reason_code:str,message:str): self.reason_code=reason_code; super().__init__(message)

def rational(value:object)->Fraction:
 if isinstance(value,Fraction): return value
 if isinstance(value,float): raise FixedPointError('FLOAT_FORBIDDEN','Python float forbidden')
 if isinstance(value,str) and '/' in value:
  a,b=value.split('/'); return Fraction(int(a),int(b))
 return parse_rational(value).as_fraction()
def trunc_scaled(value:object,delta:int)->int:
 x=rational(value)*delta
 return x.numerator//x.denominator if x>=0 else -((-x.numerator)//x.denominator)
def floor_scaled(value:object,delta:int)->int:
 x=rational(value)*delta; return x.numerator//x.denominator
def encode_signed(value:object,profile:FixedBackendProfile,bound:object|None=None)->int:
 z=trunc_scaled(value,profile.scale_delta)
 if bound is not None and abs(rational(value))>rational(bound): raise FixedPointError('ENCODING_OUT_OF_RANGE','semantic input exceeds bound')
 if abs(z)>(profile.prime_p-1)//2: raise FixedPointError('ENCODING_OUT_OF_RANGE','balanced range exceeded')
 return z%profile.prime_p
def balanced(representative:int,profile:FixedBackendProfile)->int:
 if type(representative) is not int or not 0<=representative<profile.prime_p: raise FixedPointError('NONCANONICAL_REPRESENTATIVE','representative must be in [0,p)')
 return representative if representative<=(profile.prime_p-1)//2 else representative-profile.prime_p
def decode(representative:int,profile:FixedBackendProfile)->CanonicalRational:
 return CanonicalRational(numerator=str(balanced(representative,profile)),denominator=str(profile.scale_delta))
def decimal(integer:int,delta:int)->str:
 return f'{integer}/{delta}'
