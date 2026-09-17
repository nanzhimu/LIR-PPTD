from __future__ import annotations
import hashlib,json
from pathlib import Path
from ..fixedpoint.artifacts import InventoryEntry,PRIVATE,PUBLIC,assert_publishable
from ..storage import AppendOnlyError,RawRunDirectory
from .conformance import transcript_has_public_leakage

class Phase4ArtifactError(ValueError):
    def __init__(self,reason_code,message):self.reason_code=reason_code;super().__init__(message)
def validate_public_transcript(backend):
    if transcript_has_public_leakage(backend):raise Phase4ArtifactError('PUBLIC_TRANSCRIPT_SECRET_LEAK','public transcript contains secret field')

def build_failed_result(reason_code):
    return {'status':'failed','reason_code':reason_code,'successful_result_created':False,'complete_directory_created':False,'reputation_committed':False}
def export_public(classification):assert_publishable(classification);return True

def _bytes(value):return json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()
def persist_secure_run(root:Path,run_id:str,result,backend)->Path:
    validate_public_transcript(backend);run=RawRunDirectory(root,"phase4_simulated",run_id)
    try:run.initialize()
    except AppendOnlyError as exc:raise Phase4ArtifactError('DUPLICATE_RUN_ID','run ID already complete') from exc
    bodies={"simulated_backend_profile.json":(backend.profile.model_dump(mode="json"),PUBLIC,"simulated backend profile"),"fixed_backend_profile_reference.json":({"profile_id":backend.fixed.profile_id,"hash":str(backend.fixed.digest())},PUBLIC,"L2 fixed reference"),"secure_task_result.json":(result.model_dump(mode="json"),PRIVATE,"secure simulated task result"),"share_metadata.json":({"threshold":backend.profile.threshold_t,"committee_size":backend.profile.committee_size_n,"server_ids":backend.profile.server_ids},PRIVATE,"share metadata"),"public_transcript.json":(backend.export_transcript(public=True).model_dump(mode="json"),PUBLIC,"metadata-only transcript"),"private_transcript.json":(backend.export_transcript(public=False).model_dump(mode="json"),PRIVATE,"internal opening transcript"),"l2_l3_conformance.json":({"V_sec":result.V_sec,"compared_fields":result.conformance_fields},PUBLIC,"L2/L3 exact integer conformance"),"diagnostics.json":({"q_out_recomputed":result.q_out_recomputed,"fixed_K":result.iterations_executed==result.K},PUBLIC,"secure diagnostics")}
    entries=[]
    for n,(body,privacy,description) in bodies.items():
        b=_bytes(body);run.write_once(Path(n),b);entries.append(InventoryEntry(artifact_id=f"P4-{len(entries)+1:02}",relative_path=n,privacy_class=privacy,publishable=privacy==PUBLIC,content_hash=hashlib.sha256(b).hexdigest(),size_bytes=len(b),description=description))
    raw=_bytes([x.model_dump(mode="json") for x in entries]);run.write_once(Path("artifact_inventory.json"),raw);entries.append(InventoryEntry(artifact_id="P4-09",relative_path="artifact_inventory.json",privacy_class=PUBLIC,publishable=True,content_hash=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),description="artifact inventory"))
    manifest={"schema_version":"1.0","run_id":run_id,"lifecycle_state":"complete","backend_kind":"simulated_shamir","simulated_only":True,"production_mpc_used":False,"real_data_used":False,"distributed_e8b_used":False,"cryptographic_security_claim":False,"artifact_inventory":[x.model_dump(mode="json") for x in entries]}
    return run.complete(manifest)
