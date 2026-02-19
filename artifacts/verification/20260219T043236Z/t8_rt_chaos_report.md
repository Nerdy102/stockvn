# RT Chaos Report

- seed: 42
- symbols: 500
- days: 2
- events_after_chaos: 9454

## Fault injections
- 5% duplicates burst
- 2% out-of-order delayed 30s
- redis disconnect/reconnect once
- backlog pause 60s then resume

## Invariants
- no_duplicate_bars: True
- bars_hash_deterministic: True
- signals_idempotent: True
- no_negative_qty: True
- no_negative_cash: True
- lag_recovered: True

## Perf budgets
- signal_update_500_symbols_one_bar_s: 1.75 (budget < 3s)
- peak_memory_mb: 512.0 (budget < 1536MB)
