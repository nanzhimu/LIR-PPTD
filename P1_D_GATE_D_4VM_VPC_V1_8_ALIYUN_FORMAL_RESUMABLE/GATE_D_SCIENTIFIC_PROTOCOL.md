# Gate D Scientific Protocol

Frozen deployment:
- MP-SPDZ v0.4.3;
- semi-honest Shamir;
- paper \((N,T_{\rm reconstruction})=(10,4)\);
- runtime `-N 10 -T 3`;
- 10 logical parties across 3 distinct cloud VMs, placement 4+3+3;
- one separate requester VM;
- one VPC/private network;
- SSL/TLS enabled by the honest-majority runtime;
- private client input and output;
- `--batch-size 101`;
- numerical D=1 workload;
- m=20,50,100,200;
- K=10, tau=0.2, epsilon=2^-10, eta=0.1, kappa=2;
- 5 warmups + 20 measured runs per path/worker cell.

Interpretation boundary:
- This is real networked multi-host MPC.
- It is not ten physical independent fog servers.
- Co-located logical parties on one VM share that VM's failure/trust domain.
- It does not add malicious/Byzantine security.
- It does not establish production readiness.
- It is an absolute LIR-PPTD system measurement, not an author-code runtime
  comparison against FPTD/ETBP-TD/etc.

Primary E2E request latency:
- requester-side `client_wall_seconds`.

Secondary framework metrics:
- MP-SPDZ Time;
- CPU time;
- party-0 data sent;
- global data sent;
- approximate rounds.

Formal publication summary:
- mean, median, p95;
- 95% bootstrap CI added during final adjudication;
- correctness max error;
- exact topology/hardware metadata.

Benchmark hygiene:
- Gate D smoke must pass with batch size 101 and no MP-SPDZ
  `unused triples ... distorting the benchmark` warning.
- If the warning remains, stop. No automatic batch-size search is allowed.


## V1.7 preprocessing-hygiene refinement

The complete verbose preprocessing inventory establishes exact triple costs
`2m` for ordinary tasks and `2m+1` for calibration tasks. Batch size 101 is
therefore the analytically derived candidate for the final runtime. It must
pass the eight-schedule hygiene matrix before the 200-run formal study.
