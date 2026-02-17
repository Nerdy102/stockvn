# MASTER PROMPT v5 — VN INVEST TOOLKIT: ALPHA v2 (RANK + UNCERTAINTY + COST-AWARE + FLOW/MICROSTRUCTURE)

## 0) GUARDRAILS
- Không khuyến nghị mua/bán cụ thể. Không cam kết lợi nhuận.
- Không hướng dẫn hành vi trái pháp luật.
- Backtest/report phải NET of costs + disclaimers.
- Tests offline, không gọi network.

## 1) ALPHA v2 — DECISIONS LOCKED
### 1.1 Target/Label
- Horizon H = 21 trading days.
- `y = close_{t+H}/close_t - 1`.
- `y_vnindex` cùng horizon.
- `y_excess = y - y_vnindex`.
- Per-date cross-section: percentile rank `r` của `y_excess`.
- Training label: `y_rank_z = zscore(r)` per date.
- Model predicts `y_rank_z` (không predict raw return).

### 1.2 Models
- Giữ nguyên toàn bộ v1.
- Add v2:
  - `ridge_rank_v2`: Ridge(alpha=10.0) predict `y_rank_z`
  - `hgbr_rank_v2`: HistGradientBoostingRegressor (same hparams v1)
  - `hgbr_q10_v2`, `hgbr_q50_v2`, `hgbr_q90_v2`: HGBR quantile 0.1/0.5/0.9
- Production score v2:
  - `mu = pred_q50`
  - `uncert = pred_q90 - pred_q10`
  - `score_final = mu - 0.35 * uncert`
- `/ml/predict` export:
  - `{symbol, date, score_final, mu, uncert, score_rank_z}`

### 1.3 Portfolio formation
- Rebalance mỗi 21 trading days.
- Universe:
  - exchange in HOSE/HNX/UPCOM
  - instrument_type = stock
  - adv20_value >= 1e9 VND
- Select topK = 30 theo `score_final` giảm dần.
- Weighting:
  - base IVP: `w_i ∝ 1/(vol_60d + eps)`
  - uncertainty penalty: `w_i = w_i / (1 + uncert_i / median_uncert_selected)`
  - normalize tổng trọng số = 1.0 trước constraints.
- Constraints theo thứ tự:
  1) liquidity: `position_value <= ADTV*0.05*3` và `<= NAV*0.10`
  2) max_single = 10%
  3) max_sector = 25%
  4) min_cash = 10% (risk_off => 20%)
- No-trade band:
  - `|target_w-current_w| < 0.0025` -> skip
  - qty (after board-lot rounding) `< 100` -> skip
  - order_notional `< 5,000,000 VND` -> skip
- Execution costs:
  - dùng fee/tax + slippage + fill penalty + tick rounding hiện có từ market_rules.

### 1.4 Regime overlay
- Dùng VNINDEX regime đã có: trend_up/sideways/risk_off.
- exposure multiplier:
  - trend_up=1.0
  - sideways=0.8
  - risk_off=0.5 và min_cash=0.20
- Apply exposure sau constraints: risky weights scale theo multiplier, phần còn lại vào cash.

## 2) FEATURE SET v2 — ADDITIONS LOCKED
### 2.1 Regime flags
- `regime_trend_up`, `regime_sideways`, `regime_risk_off`

### 2.2 Foreign flow features
- `net_foreign_val_5d` (rolling sum)
- `net_foreign_val_20d` (rolling sum)
- `foreign_flow_intensity = net_foreign_val_20d / adv20_value`
- `foreign_room_util = 1 - current_room/total_room`

### 2.3 Orderbook imbalance (quotes_l2)
- `imb_1_day = avg((bidVol1-askVol1)/(bidVol1+askVol1+eps))`
- `imb_3_day = avg((sum bidVol1..3 - sum askVol1..3)/(sum bidVol1..3 + sum askVol1..3 + eps))`
- `spread_day = avg((askPrice1-bidPrice1)/mid)`
- Nếu thiếu quotes -> NaN.

### 2.4 Intraday realized volatility (1m bars)
- `rv_day = sqrt(sum(r_1m^2))`
- `vol_first_hour_ratio = vol(09:15-10:15)/vol_total_day`
- Nếu thiếu intraday -> NaN.

### 2.5 Imputation
- Numeric NaN: median cross-section tại date t.
- Categorical: giữ nguyên.

## 3) DIAGNOSTICS SUITE
- RankIC (Spearman) per-date và average.
- IC decay cho k in [1,5,21,63].
- Decile returns (top-bottom spread) NET costs.
- Turnover + cost attribution.
- Capacity proxy: avg(order_notional/ADTV) + %days liquidity binds.
- Regime breakdown metrics.
- Bootstrap CI block bootstrap (block=20d, 1000 resamples) cho Sharpe_net và CAGR_net.
- Store:
  - `diagnostics_runs(run_id, model_id, config_hash, created_at)`
  - `diagnostics_metrics(run_id, metric_name, metric_value, metric_json)`

## 4) IMPLEMENTATION TASKS — FILE LIST LOCKED
- `packages/core/core/ml/targets.py`
- `packages/core/core/ml/models_v2.py`
- `packages/core/core/ml/portfolio_v2.py`
- `packages/core/core/ml/features_v2.py`
- `packages/core/core/ml/diagnostics.py`
- `services/worker_scheduler/worker_scheduler/jobs.py`
- `services/api_fastapi/api_fastapi/routers/ml.py`
- `apps/dashboard_streamlit/pages/7_Alpha_v2_Lab.py`
- `docs/alpha_v2_spec.md`
- `README.md`

## 5) TESTS — MUST ADD
- `test_rank_z_label_no_leakage.py`
- `test_quantile_models_monotonic.py`
- `test_score_final_penalizes_uncertainty.py`
- `test_no_trade_band_skips_small_trades.py`
- `test_orderbook_imbalance_daily.py`
- `test_foreign_flow_rollups.py`
- `test_block_bootstrap_ci_shapes.py`
- `test_alpha_v2_smoke_walkforward.py`

Tất cả tests chạy offline.
