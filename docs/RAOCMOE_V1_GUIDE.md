# RAOCMOE v1 Guide

RAOCMOE v1 is a paper-only, regime-aware quant overlay with six components:
- D1 CPD (mean/vol/corr/liquidity shifts)
- D2 regime posterior with hysteresis
- D3 uncertainty intervals using adaptive conformal logic
- D4 mixture-of-experts with online Hedge routing
- D5 robust portfolio controller with constraints
- D6 execution/TCA adaptation with CPD on residuals

## Offline backtest
```bash
python scripts/run_raocmoe_backtest.py --universe BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-03-01
```
Outputs are written to `artifacts/raocmoe/{report_id}/` with `report.json`, `equity.csv`, and `debug_samples.jsonl`.

## Realtime overlay (crypto-first)
`RAOCMOEOverlay` runs inside realtime signal engine on bar close (`15m`, `60m`), computes regime/UQ/routing, and writes signal snapshots to `realtime:signals:{symbol}:{tf}`.

## Governance and safety
- Undercoverage/drift/TCA regime-shift/data-stale triggers can pause RAOCMOE outputs.
- Pause action is target-cash behavior only.
- This implementation does not submit live orders and keeps live disabled semantics unchanged.

## Known limitations
- This is simulation/paper infrastructure and not production execution.
- Sector conservative pooling is currently lightweight and depends on available symbol metadata.
