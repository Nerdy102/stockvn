from __future__ import annotations

from pydantic import BaseModel


class CSCVReport(BaseModel):
    s_segments: int
    n_trials: int
    n_combinations: int
    logits_p05: float
    logits_p50: float
    logits_p95: float
    pbo_phi: float
    perf_decay_beta: float
    prob_loss_oos: float
    notes_vi: str


class RcSpaReport(BaseModel):
    benchmark_name_vi: str
    rc_stat: float
    rc_pvalue: float | None
    spa_stat: float
    spa_pvalue: float | None
    bootstrap_b: int
    q_param: float
    notes_vi: str
