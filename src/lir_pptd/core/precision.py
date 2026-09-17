from __future__ import annotations

from dataclasses import replace
from typing import Any

from mpmath import mp

from ..canonical import CanonicalRational, parse_rational
from .exact import ExactTaskInput, run_exact_once
from .types import ExactTaskFailure, ExactTaskResult, PrecisionDoublingResult, PrecisionRunSummary, StabilityStatus


def run_with_precision_doubling(task: ExactTaskInput, *, initial_dps: int = 80, max_dps: int = 320,
                                epsilon_hp: object = {"power_of_two_exponent":"-200"}) -> ExactTaskResult | ExactTaskFailure:
    epsilon = parse_rational(epsilon_hp)
    if initial_dps < 80 or max_dps < 2 * initial_dps:
        return ExactTaskFailure(reason_code="INVALID_PRECISION_CONFIG",message="initial_dps>=80 and max_dps>=2*initial_dps required")
    runs=[]; previous=None; dps=initial_dps; final=None; status=StabilityStatus.UNRESOLVED
    while dps <= max_dps:
        current=run_exact_once(task,dps=dps)
        if isinstance(current,ExactTaskFailure): return current
        runs.append(PrecisionRunSummary(dps=dps,final_output=current.final_output,
            q_out={e.worker_id:e.evidence_out for e in current.output_evidence},next_reputation=current.next_reputation))
        if previous is not None and _close(previous,current,epsilon,dps):
            final=current; status=StabilityStatus.STABLE; break
        previous=current; final=current; dps*=2
    assert final is not None
    precision=PrecisionDoublingResult(initial_dps=initial_dps,final_dps=runs[-1].dps,
        precision_doubling_steps=len(runs)-1,epsilon_hp=epsilon,stability_status=status,runs=tuple(runs))
    return final.model_copy(update={"precision":precision})


def _close(first: ExactTaskResult, second: ExactTaskResult, epsilon: CanonicalRational, dps: int) -> bool:
    ctx=mp.clone(); ctx.dps=dps
    tolerance=ctx.mpf(epsilon.numerator)/ctx.mpf(epsilon.denominator)
    pairs=list(zip(first.final_output,second.final_output))
    pairs += [(a.evidence_out,b.evidence_out) for a,b in zip(first.output_evidence,second.output_evidence)]
    pairs += [(first.next_reputation[w],second.next_reputation[w]) for w in first.participant_ids]
    return all(ctx.fabs(ctx.mpf(a)-ctx.mpf(b)) <= tolerance for a,b in pairs)
