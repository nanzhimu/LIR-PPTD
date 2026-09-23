from __future__ import annotations
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/final_scientific_closure_p0/mechanism"

FAMILIES = ("A1-persistent-ratio", "A2-attack-rho07", "A3-behavior-switch")
CONTRASTS = ("Tuned-LIR-vs-Tuned-EMA", "Tuned-LIR-vs-Window-COWA")

def summarize(rows):
    out = {}
    for fam in FAMILIES:
        out[fam] = {}
        for contrast in CONTRASTS:
            rr = [r for r in rows if r["family"] == fam and r["contrast"] == contrast]
            win = sum(r["direction"] == "candidate_win" for r in rr)
            loss = sum(r["direction"] == "comparator_win" for r in rr)
            tie = len(rr) - win - loss
            hw = sum(str(r["holm_significant"]).lower() == "true" and r["direction"] == "candidate_win" for r in rr)
            hl = sum(str(r["holm_significant"]).lower() == "true" and r["direction"] == "comparator_win" for r in rr)
            out[fam][contrast] = {
                "mean_W_T_L_tuned_lir": f"{win}/{tie}/{loss}",
                "holm_W_L_tuned_lir": f"{hw}/{hl}",
            }
    return out

if __name__ == "__main__":
    rows = list(csv.DictReader((BASE / "equal_tuning_cell_comparisons.csv").open(encoding="utf-8-sig")))
    derived = summarize(rows)
    released = json.loads((BASE / "equal_tuning_summary.json").read_text(encoding="utf-8"))
    # The released file additionally includes Tuned-LIR-vs-COWA; check the two equal-budget comparators here.
    for fam in FAMILIES:
        for c in CONTRASTS:
            assert derived[fam][c] == released[fam][c], (fam, c, derived[fam][c], released[fam][c])
    hp = json.loads((BASE / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    assert hp["ema_alpha"] == "2/5" and hp["lir_eta"] == "2/5" and hp["window_w"] == 3
    print("P0_EQUAL_BUDGET_REANALYSIS=PASS")
    print(json.dumps(derived, indent=2))
