from __future__ import annotations
from typing import Any
from fractions import Fraction
from pydantic import field_validator
from ..canonical import canonical_json_bytes,canonicalize,parse_rational
from ..config import CanonicalModel
from ..hashing import run_id,semantic_hash
from ..identifiers import BackendProfileHash,CodeHash,ConfigHash,DataHash,RunID
class FixedRunIdentity(CanonicalModel):
 run_id:RunID;run_namespace:str;code_hash:CodeHash;config_hash:ConfigHash;data_hash:DataHash;backend_profile_hash:BackendProfileHash;master_seed:int;canonical_identity_payload:dict[str,Any];identity_hash_algorithm:str='sha256/run-id-v2';source_reference:str='src/lir_pptd/hashing.py:run_id'
 @field_validator('master_seed')
 @classmethod
 def seed(cls,v):
  if isinstance(v,bool) or not isinstance(v,int) or v<0:raise TypeError('master_seed must be a nonnegative integer')
  return v
 @field_validator('run_namespace')
 @classmethod
 def namespace(cls,v):
  if not v:raise ValueError('run_namespace must be nonempty')
  return v
 @classmethod
 def from_hashes(cls,*,schema_version:str,code_hash:str,config_hash:str,data_hash:str,backend_profile_hash:str,master_seed:int,run_namespace:str):
  payload={'schema_version':schema_version,'code_hash':CodeHash(code_hash),'config_hash':ConfigHash(config_hash),'data_hash':DataHash(data_hash),'backend_profile_hash':BackendProfileHash(backend_profile_hash),'master_seed':master_seed,'run_namespace':run_namespace};canonical_json_bytes(payload);rid=run_id(schema_version=schema_version,config_digest=payload['config_hash'],code_digest=payload['code_hash'],data_digest=payload['data_hash'],backend_profile_digest=payload['backend_profile_hash'],master_seed=master_seed,run_namespace=run_namespace)
  return cls(schema_version=schema_version,run_id=rid,run_namespace=run_namespace,code_hash=payload['code_hash'],config_hash=payload['config_hash'],data_hash=payload['data_hash'],backend_profile_hash=payload['backend_profile_hash'],master_seed=master_seed,canonical_identity_payload=payload)
def _exact(v):
 if isinstance(v,float):raise TypeError('floating-point semantic fields are forbidden')
 if isinstance(v,str) and '/' in v:
  f=Fraction(v);v={'numerator':str(f.numerator),'denominator':str(f.denominator)}
 r=parse_rational(v);return r.model_dump(mode='json')
def build_identity(*,task:Any,profile:Any,code_digest:str,master_seed:int,run_namespace:str,schema_version:str='1.0',artifact_schema_version:str='1.0',synthetic_only:bool=True)->FixedRunIdentity:
 if synthetic_only is not True:raise ValueError('synthetic_only must be true')
 participants=tuple(task.participant_ids)
 if len(set(participants))!=len(participants):raise ValueError('duplicate participant ID')
 all_workers=set(task.reputations);nonparticipants=tuple(sorted(all_workers-set(participants)))
 if set(participants)&set(nonparticipants):raise ValueError('participant/nonparticipant overlap')
 if set(task.reports)!=set(participants):raise ValueError('report worker IDs must equal participant IDs')
 config={'task_kind':task.task_kind,'K':task.K,'participant_ids':list(participants),'nonparticipant_ids':list(nonparticipants),'epsilon_c':_exact(task.epsilon_c),'tau':_exact(task.tau),'eta':_exact(task.eta),'kappa':_exact(task.kappa),'profile_id':profile.profile_id,'artifact_schema_version':artifact_schema_version}
 data={'synthetic_task_id':task.task_id,'reports':canonicalize(task.reports),'initial_reputations':{k:_exact(v) for k,v in task.reputations.items()},'epochs':canonicalize(task.epochs),'synthetic_only':True}
 ch=semantic_hash('phase3-fixed-config-v1',config);dh=semantic_hash('phase3-synthetic-data-v1',data);return FixedRunIdentity.from_hashes(schema_version=schema_version,code_hash=code_digest,config_hash=ch,data_hash=dh,backend_profile_hash=str(profile.digest()),master_seed=master_seed,run_namespace=run_namespace)
