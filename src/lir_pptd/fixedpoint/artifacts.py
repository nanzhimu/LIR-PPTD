from __future__ import annotations
import hashlib,json
from pathlib import Path,PurePosixPath
from pydantic import BaseModel,ConfigDict,field_validator
from ..storage import AppendOnlyError,RawRunDirectory
from .types import FixedTaskFailure,FixedTaskResult
from .run_identity import FixedRunIdentity
PUBLIC='public';PRIVATE='private';UNKNOWN='unknown';RESTRICTED='restricted'
class InventoryEntry(BaseModel):
 model_config=ConfigDict(extra='forbid',frozen=True)
 artifact_id:str;relative_path:str;media_type:str='application/json';privacy_class:str;publishable:bool;content_hash:str;size_bytes:int;schema_version:str='1.0';description:str
 @field_validator('relative_path')
 @classmethod
 def relative(cls,v):
  p=PurePosixPath(v)
  if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe artifact path')
  return v
 @field_validator('publishable')
 @classmethod
 def publication(cls,v,info):
  privacy=info.data.get('privacy_class')
  if privacy is not None and v!=(privacy==PUBLIC):raise ValueError('publishable conflicts with privacy class')
  return v
 @field_validator('privacy_class')
 @classmethod
 def privacy(cls,v):
  if v not in {PUBLIC,PRIVATE,UNKNOWN,RESTRICTED}:raise ValueError('invalid privacy class')
  return v
def assert_publishable(classification:str)->None:
 if classification!=PUBLIC:raise PermissionError(f'{classification} artifact is not publishable')
def _bytes(body):return json.dumps(body,sort_keys=True,separators=(',',':'),default=str).encode()
def persist_fixed_run(root:Path,identity:FixedRunIdentity|str,result:FixedTaskResult|FixedTaskFailure,profile:dict,conformance:dict|None=None)->Path|None:
 if isinstance(result,FixedTaskFailure):return None
 if isinstance(identity,str):rid=identity;identity_data={'run_namespace':'legacy','code_hash':'0'*64,'config_hash':'0'*64,'data_hash':'0'*64,'master_seed':0}
 else:rid=str(identity.run_id);identity_data=identity.model_dump(mode='json')
 run=RawRunDirectory(root,'phase3_fixed',rid);run.initialize()
 cert=result.preflight_certificate
 artifacts={
 'backend_profile.json':(profile,PUBLIC,'public backend profile'),
 'preflight_certificate.json':(cert,PUBLIC,'public static certificate'),
 'no_wrap_certificate.json':({'status':cert.get('no_wrap_status') if isinstance(cert,dict) else None},PUBLIC,'no-wrap summary'),
 'fixed_task_result.json':(result.model_dump(mode='json'),PRIVATE,'individual fixed result'),
 'integer_trajectory.json':(result.truth_integer_trajectory,PRIVATE,'integer trajectory'),
 'decoded_trajectory.json':(result.truth_decoded_trajectory,PRIVATE,'decoded trajectory'),
 'primitive_traces.json':([x.model_dump(mode='json') for it in result.iterations for x in it.primitives],PRIVATE,'primitive traces'),
 'conformance_summary.json':(conformance or {'status':'not_requested'},PUBLIC,'aggregate conformance summary'),
 'diagnostics.json':([x.model_dump(mode='json') for x in result.diagnostics],PRIVATE,'diagnostics'),
 }
 inventory=[]
 for i,(name,(body,privacy,desc)) in enumerate(artifacts.items()):
  content=_bytes(body);run.write_once(Path(name),content);inventory.append(InventoryEntry(artifact_id=f'A{i+1:02}',relative_path=name,privacy_class=privacy,publishable=privacy==PUBLIC,content_hash=hashlib.sha256(content).hexdigest(),size_bytes=len(content),description=desc))
 inv_body=[x.model_dump(mode='json') for x in inventory];inv_content=_bytes(inv_body);run.write_once(Path('artifact_inventory.json'),inv_content);inventory.append(InventoryEntry(artifact_id='A10',relative_path='artifact_inventory.json',privacy_class=PUBLIC,publishable=True,content_hash=hashlib.sha256(inv_content).hexdigest(),size_bytes=len(inv_content),description='artifact inventory'))
 schema_content=_bytes({'schema_version':'1.0','inventory_entry_fields':list(InventoryEntry.model_fields)});run.write_once(Path('artifact_schema.json'),schema_content);inventory.append(InventoryEntry(artifact_id='A11',relative_path='artifact_schema.json',privacy_class=PUBLIC,publishable=True,content_hash=hashlib.sha256(schema_content).hexdigest(),size_bytes=len(schema_content),description='public artifact schema'))
 manifest={'schema_version':'1.0','run_id':rid,'run_namespace':identity_data['run_namespace'],'lifecycle_state':'complete','task_id':result.task_id,'task_kind':result.task_kind,'backend_kind':'deterministic_fixed_plaintext','profile_id':profile['profile_id'],'backend_profile_hash':result.backend_profile_hash,'code_hash':identity_data['code_hash'],'config_hash':identity_data['config_hash'],'data_hash':identity_data['data_hash'],'master_seed':identity_data['master_seed'],'K':result.K,'iterations_executed':result.iterations_executed,'q_out_recomputed':result.q_out_recomputed,'result_scope':'algorithmic_conformance','target_RESULT_SCOPE_SET':['algorithmic_conformance'],'achieved_RESULT_SCOPE_SET':[],'production_mpc_used':False,'real_data_used':False,'distributed_e8b_used':False,'cryptographic_security_claim':False,'synthetic_only':True,'artifact_inventory':[x.model_dump(mode='json') for x in inventory],'source_hash_summary':{'code_hash':identity_data['code_hash']}}
 return run.complete(manifest)
__all__=['persist_fixed_run','assert_publishable','AppendOnlyError','InventoryEntry','PUBLIC','PRIVATE','UNKNOWN','RESTRICTED']
