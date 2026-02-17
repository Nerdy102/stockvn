# ALPHA v2 Spec

This document restates the locked ALPHA v2 specification:
- Label: `y_excess` rank percentile per date, transformed to `y_rank_z`.
- Models: `ridge_rank_v2`, `hgbr_rank_v2`, and quantile `hgbr_q10_v2`, `hgbr_q50_v2`, `hgbr_q90_v2`.
- Final score: `score_final = mu - 0.35 * uncert`, where `mu=q50`, `uncert=q90-q10`.
- Portfolio: monthly rebalance, long-only, IVP + uncertainty penalty, ordered constraints, no-trade band.
- Costs: net-of-costs with fees/taxes/slippage/fill penalties/tick rounding through market rules.
- Regime overlay: trend_up=1.0, sideways=0.8, risk_off=0.5 with min cash 20% risk-off.
- Feature additions: regime flags, foreign-flow rollups, orderbook imbalance, intraday realized volatility.
- Diagnostics: RankIC, IC decay, decile spread, turnover/cost attribution, capacity proxies, regime breakdown, block bootstrap CI.
- Tests: offline-only and no network calls.

Disclaimers: past performance does not guarantee future results; overfit and liquidity/limit risks remain.
