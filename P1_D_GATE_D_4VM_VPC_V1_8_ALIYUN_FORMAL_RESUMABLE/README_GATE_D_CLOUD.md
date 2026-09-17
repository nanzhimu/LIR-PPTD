# P1-D2 Gate D — 4-VM VPC Deployment

## Topology

This package is for the user's current constraint: only one Windows 11 laptop.

Do **not** treat multiple local WSL/VM instances on that laptop as multi-host
evidence. Instead use four short-lived Ubuntu cloud VMs in one region/VPC:

- Compute A: logical parties P0-P3
- Compute B: logical parties P4-P6
- Compute C: logical parties P7-P9
- Requester: ExternalIO client only

The Windows/WSL laptop only performs SSH/SCP orchestration. It is outside the
measured requester-to-MPC path.

The paper must describe this as **ten logical MPC parties distributed over
three distinct cloud compute VMs (4+3+3), plus one requester VM**. Do not call
them ten physical fog servers or ten independent physical trust domains.

## Recommended VM sizing

For cleaner timing, prefer three identical dedicated/CPU-optimized compute VMs
with at least 2 vCPU/4 GiB; 4 vCPU/8 GiB is preferred because Compute A hosts
four party processes. The requester can be a small 1 GiB Ubuntu VM.

Create all four in:
- the same cloud region;
- the same VPC/private network;
- the same Ubuntu release.

Record exact provider, region, VM type, vCPU, RAM, OS image, and private IPs.

## Firewall

Measured MPC traffic must use private IPs.

Allow **only inside the VPC/subnet**:
- TCP 5000-5009 for MPC party networking;
- TCP 14000-14009 for ExternalIO;
- TCP 22 for administration as needed.

Do not expose the MPC/ExternalIO ports to the public internet.

## Step 1 — Fill host config

Copy:

```bash
cp config/gate_d_hosts.example.env config/gate_d_hosts.env
```

Fill four public IPs and four private VPC IPs.

Public IPs are used only for SSH/SCP orchestration.
Private IPs are used inside all measured runs.

## Step 2 — Bootstrap software

From WSL:

```bash
cd /mnt/d/LIR_PPTD
bash P1_D_GATE_D_4VM_VPC_V1/scripts/setup_gate_d_from_wsl.sh /mnt/d/LIR_PPTD
```

This:
- installs MP-SPDZ v0.4.3 on the three compute VMs;
- builds `shamir-party.x`;
- compiles all eight LIR-PPTD schedules on every compute host;
- installs the ExternalIO client on the requester;
- generates/copies the deterministic 25-replicate workloads;
- creates the 10-line private-IP party map.

## Step 3 — Formal SSL key distribution

From WSL:

```bash
bash P1_D_GATE_D_4VM_VPC_V1/scripts/distribute_formal_ssl.sh
```

Private keys are intentionally isolated:
- Compute A gets P0-P3 keys only;
- Compute B gets P4-P6 keys only;
- Compute C gets P7-P9 keys only;
- Requester gets C0 key only.

All hosts get the public certificates required for verification.

## Step 4 — Gate D smoke / benchmark-hygiene check

```bash
python3 P1_D_GATE_D_4VM_VPC_V1/orchestrator/run_gate_d.py \
  --env P1_D_GATE_D_4VM_VPC_V1/config/gate_d_hosts.env \
  --out results/p1d_gate_d_4vm_v1 \
  --smoke
```

The smoke uses:
- real private VPC networking;
- 10 logical Shamir parties;
- fixed `--batch-size 1000`;
- m=20 ordinary + calibration.

Required:
```text
P1D_GATE_D_SMOKE_BAD=0
P1D_GATE_D_SMOKE=PASS
```

The smoke deliberately fails if the MP-SPDZ log still contains the
`unused triples ... distorting the benchmark` warning. If that happens, stop
and return the smoke directory for adjudication. Do **not** auto-tune the batch
size.

## Step 5 — Formal run

Only after smoke PASS:

```bash
python3 P1_D_GATE_D_4VM_VPC_V1/orchestrator/run_gate_d.py \
  --env P1_D_GATE_D_4VM_VPC_V1/config/gate_d_hosts.env \
  --out results/p1d_gate_d_4vm_v1 \
  --formal
```

Frozen matrix:
- N=10 logical parties;
- paper reconstruction threshold=4;
- MP-SPDZ max-corrupted `-T 3`;
- placement 4+3+3 over three compute VMs;
- m=20,50,100,200;
- ordinary + calibration;
- 5 warmups + 20 measured repetitions/cell;
- 200 total executions, 160 measured;
- batch size 1000;
- no LIR-PPTD retuning.

## Return artifacts

After smoke:
- `results/p1d_gate_d_4vm_v1/smoke_runs.csv`
- both smoke client JSON files
- all 20 party logs

After formal:
- `results/p1d_gate_d_4vm_v1/formal_runs.csv`
- `results/p1d_gate_d_4vm_v1/formal_summary.json`
- all `*_summary.json`
- party-0 logs are mandatory; retain all party logs for the final archive.

Do not delete the VMs until the formal audit passes.
