# EVAL LAB QA Notes

## Mandatory Local Inventory Snapshot

- `git rev-parse HEAD`: `9152cc1f1bf5d14b2b1464b1ed44ed38198f993a`
- `git status --porcelain`: clean at inventory time
- `git diff --stat`: no diff at inventory time

## Formula map (exact files/functions)

### 1) Turnover / traded notional / costs / drags
- File: `scripts/run_eval_lab.py`
- Functions:
  - `_simulate(...)`
  - `_metrics_from_eq(...)`

Formulas:
- Per-day turnover (L1):
  - `turnover_l1_t = Σ_i |w_t(i) - w_{t-1}(i)|`
- Aggregates:
  - `turnover_l1_total = Σ_t turnover_l1_t`
  - `turnover_l1_daily_avg = turnover_l1_total / days`
  - `turnover_l1_annualized = turnover_l1_daily_avg * 252`
- Traded notional:
  - `traded_notional_total = Σ_t (turnover_l1_t * equity_net_t)`
  - `traded_notional_over_nav = traded_notional_total / avg_nav`
- Cost components:
  - `commission_t = (buy_turn_t + sell_turn_t) * commission_bps / 10000`
  - `sell_tax_t = sell_turn_t * sell_tax_bps / 10000`
  - `slippage_t = turnover_l1_t * slippage_bps / 10000`
  - `total_cost_total = Σ_t (commission_t + sell_tax_t + slippage_t)`
- Cost drags:
  - `cost_drag_vs_gross = total_cost_total / gross_end`
  - `cost_drag_vs_traded = total_cost_total / traded_notional_total`

### 2) Gross/Net equity definition and identity checks
- File: `scripts/run_eval_lab.py`
- Function: `_simulate(...)`
  - `gross_eq_t = gross_eq_{t-1} * (1 + port_ret_t)`
  - `net_eq_t = net_eq_{t-1} * (1 + port_ret_t - cost_t)`
  - `cum_cost_t = gross_eq_t - net_eq_t`
- File: `packages/core/core/eval_lab/consistency.py`
- Functions:
  - `check_equity_identity(equity_gross, equity_net, cum_cost)` checks `max| (gross-net)-cum_cost | < 1e-6`
  - `check_end_identity(gross_end, net_end, total_cost_total)` checks `|(gross_end-net_end)-total_cost_total| < 1e-6`
  - `check_cost_nonnegativity(...)` checks all cost components `>= 0`

### 3) RC/SPA p-value computation and bootstrap B
- Files:
  - `packages/core/core/eval_lab/multiple_testing.py`
  - `scripts/run_eval_lab.py`
- Functions:
  - `reality_check(diff, block_size, b, seed)`
  - `spa(diff, block_size, b, seed)`
- In eval run:
  - `B = cfg['bootstrap']['samples']`
  - display uses floor `max(p, 1/B)` via `format_pvalue(...)`
  - bootstrap `B` is recorded into `summary.json`.

### 4) PSR / DSR / MinTRL / PBO / N_trials / N_eff
- Files:
  - `packages/core/core/eval_lab/multiple_testing.py`
  - `scripts/run_eval_lab.py`
- Functions:
  - `psr(sr_hat, sr0, n_obs)`
  - `dsr(sr_hat, n_obs, n_trials)`
  - `min_trl(sr_hat, sr0)`
  - `pbo_cscv(ret_matrix, s, seed)`
- Eval usage:
  - `n_trials = len(strategy_registry)`
  - `n_eff` from return-correlation effective independent tests approximation:
    - `n_eff = round((tr(C)^2) / sum(C^2))`, clipped to >= 1.

