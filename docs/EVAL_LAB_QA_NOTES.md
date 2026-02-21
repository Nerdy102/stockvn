# EVAL LAB QA Notes

## Metric source map and formulas

- **Turnover metrics** are computed in `scripts/run_eval_lab.py` inside `_simulate`.
  - `turnover_l1_total = sum_t sum_i |w_t(i)-w_{t-1}(i)|`
  - `turnover_l1_daily_avg = turnover_l1_total / n_days`
  - `turnover_l1_annualized = turnover_l1_daily_avg * 252`
  - `traded_notional_total = sum_t (turnover_l1_t * NAV_t)`
  - `traded_notional_over_nav = traded_notional_total / avg_nav`

- **Cost attribution** is computed in `scripts/run_eval_lab.py` inside `_simulate`.
  - `commission_cost_t = (buy_turn_t + sell_turn_t) * commission_bps / 10000`
  - `sell_tax_cost_t = sell_turn_t * sell_tax_bps / 10000`
  - `slippage_cost_t = turnover_l1_t * slippage_bps / 10000`
  - totals are sums over time.
  - `cost_drag_vs_gross = total_cost_total / gross_equity_end`
  - `cost_drag_vs_traded = total_cost_total / traded_notional_total`

- **RC/SPA p-values** are computed in `packages/core/core/eval_lab/multiple_testing.py`.
  - `reality_check(diff, block_size, b, seed)` and `spa(...)`.
  - bootstrap p-value is lower-bounded by `1/B` and bounded in `[0,1]`.
  - display formatting uses `format_pvalue(p, B)` where small values print as `p < 1/B`.

- **PSR/DSR/MinTRL** are computed in `packages/core/core/eval_lab/multiple_testing.py` and used in `scripts/run_eval_lab.py`.
  - `psr(sr_hat, sr0, n_obs) = Phi((sr_hat - sr0)*sqrt(n_obs-1))`
  - `dsr(sr_hat, n_obs, n_trials)` applies trial-penalty `sqrt(2*log(n_trials))/sqrt(n_obs)` before PSR.
  - `min_trl` uses the one-sided normal approximation at alpha=0.05.

## Consistency hard gate
- Added in `packages/core/core/eval_lab/consistency.py` and enforced in `scripts/run_eval_lab.py`.
- Checks:
  - equity identity: `gross - net == cum_cost` (max abs error < 1e-6)
  - return identity: reported total return equals implied final equity return
  - non-negative costs
- Any failure sets reliability reason `consistency_failed`.
