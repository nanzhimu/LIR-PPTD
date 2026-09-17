from __future__ import annotations
from itertools import count
from typing import Literal
from pydantic import Field, field_validator, model_validator
from ..config import CanonicalModel
from ..hashing import backend_profile_hash
from ..seed import SeedManager
from ..fixedpoint import default_profile
from ..fixedpoint.arithmetic import div_positive, local_add, local_sub, mul_positive, pub_mul_positive, square
from .types import *

class SimulatedBackendProfile(CanonicalModel):
    profile_id: str = "l3-simulated-shamir-v1"
    backend_kind: Literal["simulated_shamir"] = "simulated_shamir"
    implemented: Literal[True] = True
    production_ready: Literal[False] = False
    cryptographic_security_claim: Literal[False] = False
    field_prime_p: int = Field(gt=2)
    threshold_t: int = Field(ge=2)
    committee_size_n: int = Field(ge=2)
    server_ids: tuple[str, ...]
    x_coordinates: tuple[int, ...]
    fixed_backend_profile_id: str
    fixed_backend_profile_hash: str
    share_generation_rule: Literal["seeded_polynomial_v1"] = "seeded_polynomial_v1"
    reconstruction_rule: Literal["lagrange_at_zero_v1"] = "lagrange_at_zero_v1"
    refresh_rule: Literal["zero_constant_seeded_polynomial_v1"] = "zero_constant_seeded_polynomial_v1"
    operation_simulation_rule: Literal["private_reconstruct_l2_reshare_v1"] = "private_reconstruct_l2_reshare_v1"
    transcript_schema_version: str = "1.0"
    internal_opening_policy: Literal["private_simulated_only"] = "private_simulated_only"
    public_transcript_policy: Literal["metadata_only"] = "metadata_only"
    private_artifact_policy: Literal["never_public_export"] = "never_public_export"
    @field_validator("field_prime_p", "threshold_t", "committee_size_n", "x_coordinates", mode="before")
    @classmethod
    def no_float(cls, value):
        if isinstance(value, float) or (isinstance(value, (tuple,list)) and any(isinstance(x,float) for x in value)): raise TypeError("float forbidden")
        return value
    @model_validator(mode="after")
    def valid(self):
        if self.threshold_t > self.committee_size_n: raise ValueError("threshold exceeds committee")
        if len(self.server_ids) != self.committee_size_n or len(set(self.server_ids)) != len(self.server_ids): raise ValueError("duplicate or missing server ID")
        if len(self.x_coordinates) != self.committee_size_n or any(x == 0 for x in self.x_coordinates) or len(set(self.x_coordinates)) != len(self.x_coordinates): raise ValueError("invalid x coordinates")
        if self.committee_size_n >= self.field_prime_p: raise ValueError("committee must be smaller than field")
        fixed=default_profile()
        if self.field_prime_p != fixed.prime_p or self.fixed_backend_profile_hash != str(fixed.digest()): raise ValueError("fixed profile mismatch")
        return self
    def digest(self): return backend_profile_hash(self.schema_version, self.semantic_object())

def default_simulated_profile() -> SimulatedBackendProfile:
    fixed=default_profile()
    return SimulatedBackendProfile(field_prime_p=fixed.prime_p,threshold_t=2,committee_size_n=3,server_ids=("s1","s2","s3"),x_coordinates=(1,2,3),fixed_backend_profile_id=fixed.profile_id,fixed_backend_profile_hash=str(fixed.digest()))

class SimulatedShamirBackend:
    backend_kind="simulated_shamir"
    def __init__(self, profile: SimulatedBackendProfile|None=None, *, master_seed:int=0):
        self.profile=profile or default_simulated_profile(); self.fixed=default_profile(); self.seed=SeedManager.from_int(master_seed); self._private=[]; self._public=[]; self._counter=count(1);self._context={}
    def set_context(self, *, task_id="phase4", iteration=None, worker_id=None, coordinate=None):
        self._context={"task_id":task_id,"iteration":iteration,"worker_id":worker_id,"coordinate":coordinate}
    def _entry(self,name, inputs=(), outputs=(), *, private=False, status="completed", reason=None, epoch=0):
        context=getattr(self,"_context",{})
        x=TranscriptEntry(operation_id=f"op-{next(self._counter):06d}",operation_name=name,backend_kind="simulated_shamir",input_secret_ids=tuple(inputs),output_secret_ids=tuple(outputs),threshold=self.profile.threshold_t,committee_size=self.profile.committee_size_n,share_epoch=epoch,backend_profile_hash=str(self.profile.digest()),fixed_profile_hash=str(self.fixed.digest()),simulated_internal_opening=private,status=status,reason_code=reason,privacy_class="private" if private else "public",**context)
        (self._private if private else self._public).append(x)
    def _validate(self,b:ShareBundle):
        if b.field_prime!=self.profile.field_prime_p: raise SimulatedBackendError("MIXED_FIELD_PRIME","field prime mismatch")
        if b.threshold!=self.profile.threshold_t: raise SimulatedBackendError("MIXED_THRESHOLD","threshold mismatch")
        if b.committee_size!=self.profile.committee_size_n: raise SimulatedBackendError("MIXED_COMMITTEE","committee mismatch")
        if b.backend_profile_hash!=str(self.profile.digest()): raise SimulatedBackendError("MIXED_PROFILE_HASH","profile mismatch")
        if len(b.shares) > self.profile.committee_size_n or len({s.server_id for s in b.shares}) != len(b.shares): raise SimulatedBackendError("DUPLICATE_SHARE","invalid share set")
        for s in b.shares:
            if s.secret_id!=b.secret_id or s.field_prime!=b.field_prime or s.threshold!=b.threshold or s.committee_size!=b.committee_size or s.backend_profile_hash!=b.backend_profile_hash or s.share_epoch!=b.share_epoch: raise SimulatedBackendError("INCONSISTENT_BUNDLE","share bundle mismatch")
    def share_secret(self,secret_id:str,value:int,*,namespace:str="share",epoch:int=0)->ShareBundle:
        if isinstance(value,float): raise SimulatedBackendError("FLOAT_FORBIDDEN","float secret")
        p=self.profile.field_prime_p; coeff=[value%p]+[self.seed.derive(f"{namespace}:{secret_id}:{epoch}:{i}",bits=256)%p for i in range(1,self.profile.threshold_t)]
        shares=tuple(SharePoint(server_id=s,x_coordinate=x,y_value=sum(c*pow(x,i,p) for i,c in enumerate(coeff))%p,field_prime=p,threshold=self.profile.threshold_t,committee_size=self.profile.committee_size_n,secret_id=secret_id,backend_profile_hash=str(self.profile.digest()),share_epoch=epoch) for s,x in zip(self.profile.server_ids,self.profile.x_coordinates))
        self._entry("share_secret",outputs=(secret_id,),private=True,epoch=epoch);return ShareBundle(shares=shares,secret_id=secret_id,field_prime=p,threshold=self.profile.threshold_t,committee_size=self.profile.committee_size_n,backend_profile_hash=str(self.profile.digest()),share_epoch=epoch)
    def reconstruct(self,bundle:ShareBundle)->ReconstructionResult:
        try:self._validate(bundle)
        except SimulatedBackendError as e:self._entry("reconstruct",(bundle.secret_id,),private=True,status="rejected",reason=e.reason_code,epoch=bundle.share_epoch);return ReconstructionResult(status="rejected",reason_code=e.reason_code)
        pts=bundle.shares
        if len(pts)<bundle.threshold:return ReconstructionResult(status="rejected",reason_code="INSUFFICIENT_SHARES")
        p=bundle.field_prime; selected=pts[:bundle.threshold]; value=0
        for point in selected:
            weight=1
            for other in selected:
                if other.server_id != point.server_id: weight=(weight*(-other.x_coordinate)*pow(point.x_coordinate-other.x_coordinate,-1,p))%p
            value=(value+point.y_value*weight)%p
        self._entry("reconstruct",(bundle.secret_id,),private=True,epoch=bundle.share_epoch);return ReconstructionResult(status="reconstructed",value=value)
    def _open(self,b):
        r=self.reconstruct(b)
        if r.status!="reconstructed":raise SimulatedBackendError(r.reason_code or "REJECTED","cannot open")
        return r.value if r.value <= self.profile.field_prime_p//2 else r.value-self.profile.field_prime_p
    def _compatible(self,a,b):
        self._validate(a);self._validate(b)
        if a.secret_id==b.secret_id: return
        if a.share_epoch!=b.share_epoch:raise SimulatedBackendError("MIXED_EPOCH","epoch mismatch")
    def _operation(self,name,inputs,fn):
        try:
            values=[self._open(x) for x in inputs]; r=fn(*values); out=self.share_secret(f"{name}:{next(self._counter)}",r,namespace=name,epoch=inputs[0].share_epoch);self._entry(name,[x.secret_id for x in inputs],[out.secret_id],private=True,epoch=out.share_epoch);return BackendOperationResult(status="completed",output=out)
        except (SimulatedBackendError,Exception) as e:
            code=getattr(e,"reason_code","L2_PRIMITIVE_REJECTED");self._entry(name,[x.secret_id for x in inputs],private=True,status="rejected",reason=code,epoch=inputs[0].share_epoch);return BackendOperationResult(status="rejected",reason_code=code)
    def local_add(self,a,b):self._compatible(a,b);return self._operation("local_add",(a,b),lambda x,y:local_add(x,y,self.fixed).output)
    def local_sub(self,a,b):self._compatible(a,b);return self._operation("local_sub",(a,b),lambda x,y:local_sub(x,y,self.fixed).output)
    def SecSqr(self,a):return self._operation("SecSqr",(a,),lambda x:square(x,self.fixed).output)
    def SecMulPositive(self,a,b):self._compatible(a,b);return self._operation("SecMulPositive",(a,b),lambda x,y:mul_positive(x,y,self.fixed).output)
    def PubMulPositive(self,a,c):return self._operation("PubMulPositive",(a,),lambda x:pub_mul_positive(x,c,self.fixed).output)
    def SecDivPositive(self,a,b,lower,upper):self._compatible(a,b);return self._operation("SecDivPositive",(a,b),lambda x,y:div_positive(x,y,self.fixed,lower,upper).output)
    def refresh(self,bundle,*,namespace):
        value=self._open(bundle);out=self.share_secret(bundle.secret_id,value,namespace=namespace,epoch=bundle.share_epoch+1);self._entry("refresh",(bundle.secret_id,),(out.secret_id,),private=True,epoch=out.share_epoch);return out
    def prepare_output(self,task_id,output,q_out_recomputed):return PreparedOutput(task_id=task_id,output_secret_id=output.secret_id,q_out_recomputed=q_out_recomputed)
    def prepare_reputation(self,worker_id,previous_epoch,next_epoch,update_count=1):
        if update_count!=1 or next_epoch!=previous_epoch+1:raise SimulatedBackendError("INVALID_REPUTATION_PREPARATION","must have exactly one update")
        return PreparedReputation(worker_id=worker_id,previous_epoch=previous_epoch,next_epoch=next_epoch,update_count=update_count)
    def export_transcript(self,*,public=True):return TranscriptBundle(entries=tuple(self._public if public else self._private))
    def conformance_hook(self):return {"simulated_only":True,"cryptographic_security_claim":False}
