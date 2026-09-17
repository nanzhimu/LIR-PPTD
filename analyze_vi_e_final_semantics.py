from __future__ import annotations
import argparse, csv, json
from pathlib import Path

RESULT_DIR=Path("results/vi_e_final_semantics_v1")

def read_csv(p: Path):
    with p.open(encoding="utf-8") as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",default="."); args=ap.parse_args(); root=Path(args.repo_root).resolve()
    summary=json.loads((root/RESULT_DIR/"formal_summary.json").read_text(encoding="utf-8"))
    external=read_csv(root/RESULT_DIR/"vi_e_external_runtime.csv")
    cal=read_csv(root/RESULT_DIR/"vi_e_calibration_overhead.csv")
    counts=read_csv(root/RESULT_DIR/"vi_e_arithmetic_counts.csv")
    if summary["failure"]!=0 or summary["success"]!=360: raise RuntimeError("incomplete formal run")
    print("VI_E_FINAL_SEMANTICS_ANALYSIS=PASS")
    print("EXTERNAL_FINAL_ORDINARY_VS_FOG")
    for r in external:
        print(f"m={r['m']} LIR_MS={float(r['lir_mean_ms']):.4f} FOG_MS={float(r['fog_mean_ms']):.4f} REDUCTION={float(r['runtime_reduction_percent']):.2f}% WINS={r['lir_wins']}/30 HOLM_P={float(r['holm_p']):.3g}")
    print("CALIBRATION_TAIL_DEFAULT_P20")
    for r in cal:
        print(f"m={r['m']} EVENT_MS={float(r['calibration_tail_mean_ms']):.4f} AMORTIZED_MS_TASK={float(r['amortized_time_ms_per_task_P20']):.4f} AMORTIZED_CALLS_TASK={float(r['amortized_calls_per_task_P20']):.2f}")
    print("SOURCE_NORMALIZED_FINAL_ORDINARY")
    for r in counts:
        print(f"m={r['m']} TMUL_REDUCTION={float(r['tmul_reduction_percent']):.2f}% TADD_REDUCTION={float(r['tadd_reduction_percent']):.2f}%")
    print("PAPER_SCOPE=local arithmetic benchmark + analytical arithmetic counts; not production-network latency")
    return 0
if __name__=="__main__": raise SystemExit(main())
