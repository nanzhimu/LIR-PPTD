# P1-D2 — Real End-to-End MPC Evaluation (Frozen Design)

## Why a new runtime is necessary

The repository's existing `SimulatedShamirBackend` is intentionally
non-production and simulates nonlinear secure operations by reconstructing and
re-sharing values. It remains valid for semantic/conformance and matched local
kernel studies, but **cannot support a claim of real networked MPC execution**.

P1-D2 therefore uses MP-SPDZ v0.4.3 as an external real MPC runtime.

## Security-model mapping

Paper default:
- committee size \(N=10\);
- Shamir reconstruction threshold \(T_{\rm paper}=4\);
- static semi-honest fog adversary with at most \(3\) colluding servers.

MP-SPDZ Shamir runtime:
- `shamir-party.x`;
- `-N 10`;
- `-T 3`, because MP-SPDZ uses `T` for the **maximum number of corrupted
  parties**, not the reconstruction threshold.

Do not rewrite the paper's \((N,T)=(10,4)\) as \((10,3)\).

## Arithmetic mapping

Primary implementation:
- prime-field Shamir;
- compile field-size requirement: 192 bits;
- `sfix.set_precision(24, 56)`;
- `sfix.round_nearest = True`.

The paper's bespoke deterministic fixed-point layer uses 24 fractional bits
with its own truncation rules. MP-SPDZ `sfix` is therefore subjected to a
**numerical/functionality correctness gate**, not a bit-exact transcript gate.

## E2E boundary

Included:
1. external-client connection establishment;
2. secure private input of worker reports and persistent reputation states;
3. SSL-protected MPC-party networking;
4. full \(K=10\) ordinary numerical truth-discovery circuit;
5. final task-local \(q^{out}\) recomputation;
6. private truth output to the requester/client;
7. calibration \(q^{cal}\) and one bounded reputation transition;
8. protocol work performed by MP-SPDZ during the measured execution;
9. framework-reported communication and round counts.

Excluded:
- sensing/task assignment;
- WAL/storage durability (already evaluated separately);
- external signature/credential-verification cost;
- application serialization outside the ExternalIO benchmark client;
- malicious/Byzantine-secure MPC.

## Workloads

Numerical D=1:
- \(m\in\{20,50,100,200\}\);
- \(K=10\);
- \(\tau=0.2\);
- \(\epsilon_c=2^{-10}\);
- \(\eta=0.1\);
- \(\kappa=2\).

Two paths:
- `ordinary`: full task-local LIR-PPTD inference, zero persistent update;
- `calibration`: \(q^{cal}\) plus one bounded persistent transition.

One ExternalIO benchmark client injects deterministic workload values. This
exercises real private client-to-MPC I/O but is **not** a claim that all mobile
workers are one physical client.

## Gates

### Gate A — package contract
Focused tests only.

### Gate B — MP-SPDZ v0.4.3 build/compile
- `shamir-party.x` builds;
- all programs compile with `-F 192`.

### Gate C — localhost real-MPC probe
- 10 actual Shamir party processes;
- `-T 3`;
- SSL and ExternalIO;
- m=20,50;
- one warm-up + three measured trials;
- truth error <= \(2\times10^{-4}\);
- calibration validation checksum error <= \(5\times10^{-4}\);
- clean process exit;
- nonzero framework communication.

A Gate-C pass supports only **real local multi-process MPC**, not multi-host.

### Gate D — real multi-host formal study
Preferred: 10 logical parties on 10 separate LAN hosts/VMs.
If infrastructure is limited, >=3 distinct LAN hosts may host 10 logical
parties, but the paper must disclose the exact placement and must not imply ten
independent physical trust domains.

Primary:
- paper \((N,T)=(10,4)\), runtime `-N 10 -T 3`;
- m=20,50,100,200;
- ordinary and calibration measured separately;
- 5 warmups + 20 measured runs per cell.

Optional only after the primary run:
- committee scaling `(3,2), (5,3), (7,4), (10,4)`.

No WAN emulation is required for P1-D closure.

## Metrics

Retain per run:
- client wall time;
- MP-SPDZ `Time`;
- CPU time;
- party-0 data sent;
- global data sent;
- approximate rounds;
- client communication bytes/time when emitted;
- peak RSS if available;
- correctness delta;
- topology/host-placement metadata.

Use mean, median, p95 and 95% bootstrap CI for wall time. Do not manufacture
cross-paper p-values against published SOTA numbers.

## Claim boundary

Safe after Gate D:

> We deploy the frozen numerical LIR-PPTD arithmetic path on MP-SPDZ's
> semi-honest Shamir protocol and measure secure client input, networked
> multi-party computation, and private output over a real LAN deployment.

Unsafe:
- production-ready MPC;
- Byzantine fault tolerance;
- faster than FPTD unless FPTD author code is executed on the same platform;
- ten independent fog trust domains if logical parties are colocated;
- bit-exact equivalence to the bespoke fixed-point backend.
