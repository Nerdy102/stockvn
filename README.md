# VN Invest Toolkit (Offline Demo, Production‑Ready Monorepo)

Monorepo chạy được ngay (ready‑to‑run) cho:
- **Screener & Discovery**
- **Charting & Signals**
- **Portfolio Tracker**

Tập trung **VN‑Index/HOSE** nhưng thiết kế mở rộng **HNX/UPCoM** (qua config + provider plugins).

> ⚠️ **Disclaimer**: Đây là công cụ kỹ thuật phục vụ học tập/giáo dục. Không phải lời khuyên đầu tư.
> Dữ liệu demo trong repo là **dữ liệu giả lập (synthetic)**, không phản ánh doanh nghiệp thực.

## Quickstart (Local)

Yêu cầu: Python **3.11+**

```bash
make setup
make run-api      # http://localhost:8000/docs
make run-worker   # ingest+compute+alerts
make run-ui       # http://localhost:8501
```

## Quickstart Kiosk (5 lệnh, offline demo)

```bash
cp .env.example .env
make setup
make run-api
make run-ui-kiosk
make verify-offline
make verify-regression
```

- Kiosk: http://localhost:8502
- Dashboard nâng cao (Advanced): chỉ bật khi `ENABLE_ADVANCED_UI=true`.
- Từ điển thuật ngữ Việt (English): `docs/GLOSSARY_VI_EN.md`.
- Hướng dẫn an toàn giao dịch: `docs/SAFE_TRADING_GUIDE.md`.
- Hardening định lượng: `docs/QUANT_CORE_HARDENING.md`.
- Backtest v2: `docs/BACKTEST_V2_GUIDE.md`.
- Tối ưu rủi ro vận hành: `docs/RISK_OPTIMIZATION_GUIDE.md`.
- Quant Trust Pack: `docs/QUANT_TRUST_PACK.md`.

## Quickstart (Docker)

```bash
cp .env.example .env
make docker-up
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Dev workflow

```bash
make format
make lint
make test
```

## Quality Gate

Chạy full quality gate trước khi merge:

```bash
make quality-gate
```

`make quality-gate` chạy tuần tự:
- `ruff check .`
- `black --check .`
- `mypy .`
- `pytest -q`
- `python scripts/quality_gate.py`

Chính sách **no placeholder** cho production path (`packages/`, `services/`, `apps/`):
- Không được để lại `TO-DO`, `FIX-ME`, `HA-CK`, `pass-comment`, `return-none placeholder-comment`.
- Không dùng docstring chứa từ `placeholder` cho function/class.
- Ngoại lệ duy nhất: `tests/fixtures/**` và `data_demo/**`.

## Kiến trúc

```mermaid
flowchart TB
  UI[Streamlit Dashboard] -->|HTTP| API[FastAPI API]
  Worker[Worker Scheduler (APScheduler)] --> DB[(SQLite/Postgres)]
  API --> DB
  Provider[Data Provider Plugins] --> Worker
  Provider --> API

  subgraph packages
    Core[packages/core: indicators, factors, risk, portfolio, market rules HOSE, DSL]
    Data[packages/data: providers + ETL]
  end

  API --> Core
  Worker --> Core
  Worker --> Data
  API --> Data
```

## HOSE microstructure (encoded)

- Config: `configs/market_rules_vn.yaml`
- Parser/validator/tick rounding: `packages/core/core/market_rules.py`

Bao gồm:
- Trading sessions (09:00–14:45) + put-through windows
- Lot size (board lot/odd lot) + max order/block min
- Tick size đúng mốc (<10,000:10đ; 10,000–49,950:50đ; ≥50,000:100đ; ETF/CW:10đ; put-through:1đ)
- Price limit ±7% (normal), ±20% (first day / resumed>=25 trading days)
- **Special cases structure** để mở rộng (vd ex-rights từ `CorporateAction`) — thiết kế sẵn trong YAML

## Research correctness & guardrails

- Factor engine xử lý mẫu số âm/0 rõ ràng (vd PE với NI<=0, PB với Equity<=0 -> NaN, không ép dữ liệu sai).
- Chuẩn hóa factor theo **winsorize + z-score** để giảm nhiễu outlier.
- Hỗ trợ tùy chọn neutralization: **sector-neutral** và **size-neutral** (MVP toggle).
- Hệ thống có cấu trúc để làm point-in-time (`as_of_date`) và cần mở rộng `public_date` cho production để giảm look-ahead bias.
- Kết quả backtest/đánh giá phải xem là mô phỏng: quá khứ không đảm bảo tương lai; nhạy với phí/thuế/slippage/fill assumptions.

## Thuế/phí (VN, MVP)

- Config: `configs/fees_taxes.yaml`
- Module: `packages/core/core/fees_taxes.py`

Mặc định:
- Thuế chuyển nhượng (sell tax): **0.1%** trên gross sell value
- Thuế cổ tức tiền mặt: **5%** withholding
- Commission: configurable theo broker (default **0.15%**)

Các khoản phí/thuế được dùng trong tính P&L.

## Data Provider (plugin)

Interface: `data.providers.base.BaseMarketDataProvider`
- `CsvProvider` (demo offline, synthetic)
- `SsiFastConnectProvider` (stub; không hardcode secrets/URL)
- `VietstockDataFeedProvider` (stub; không hardcode secrets/URL)

Chọn provider qua `.env`:
```
DATA_PROVIDER=csv
DEMO_DATA_DIR=data_demo
```

## Screener (MVP, professional-ready)

- Basic filters: sector, market cap, avg value 20D, PE/PB, ROE, net debt/EBITDA (phi tài chính), tags/catalysts
- Factor ranking: quality/value/momentum/low_vol/dividend
- Technical setups: breakout/trend/pullback/volume spike
- Explainability: trả về breakdown & lý do lọt

Screen definition bằng YAML: `configs/screens/*.yaml`

## Charting & Signals

- Timeframes: 1D, 1W (resample), 60m, 15m
- Candlestick + Volume + Indicators: SMA/EMA/RSI/MACD/ATR/VWAP
- Auto supply/demand zones + auto trendline (heuristic) + cho phép chỉnh tay trên UI
- Alerts DSL: ví dụ `CROSSOVER(close, SMA(20)) AND volume > 1.5*AVG(volume, 20)`

## Portfolio Tracker (MVP)

- Import trades CSV + nhập tay qua UI
- P&L realized/unrealized + breakdown theo ngày / strategy_tag / sector
- Exposure: sector weights, concentration, cash
- Risk: max drawdown, volatility, beta vs VNINDEX, correlation matrix
- Attribution (MVP): Brinson theo sector (allocation/selection/interaction) + mapping market/selection/timing_proxy
- Rebalance suggestion (MVP): rules max sector weight / max single name / target cash + board lot/liquidity constraints
- Assumptions panel: hiển thị fee/tax/slippage/fill/regime để explainability rõ ràng

## CI (GitHub Actions)

`.github/workflows/ci.yml` chạy:
- ruff
- black --check
- pytest

---
# stockvn


## Execution realism (MVP)

- Config: `configs/execution_model.yaml`
- Slippage model: `slippage_bps = base + k1*(order_notional/ADTV) + k2*(ATR%)`
- Fill model hook cho phiên chạm trần/sàn: giảm xác suất khớp theo config
- Regime filter từ VNINDEX (`trend_up`, `sideway`, `risk_off`) để điều chỉnh exposure và xếp hạng tín hiệu


## SSI pipeline hardening

- Contract schemas and adapter: `packages/data/data/schemas/ssi_fcdata.py`, `packages/data/data/adapters/ssi_mapper.py`.
- Bronze/Silver idempotent ingest + checkpoint: `packages/data/data/etl/pipeline.py`.
- Incremental indicator state engine: `worker_scheduler.jobs.compute_indicators_incremental`.
- Architecture doc: `docs/ssi_pipeline_architecture.md`.

## SSI Contract
- Offline SSI contract parsing with alias/typo tolerant raw models and deterministic mapping.
- See `docs/ssi_contract.md` for fields and mapping.

## ML Alpha Pipeline
- Fixed models: Ridge + HistGradientBoostingRegressor + fixed ensemble (0.2/0.8).
- Purged CV + embargo + offline walk-forward utilities under `core/ml/`.

## Backtest Correctness & Costs
- Net reporting includes commission, sell tax, slippage, and fill penalties.
- Disclaimer: past performance does not guarantee future results; overfit/liquidity/limit risks apply.


## ALPHA v2
- Ranking label: train on `y_rank_z` (cross-sectional z-scored percentile rank of `y_excess`).
- Uncertainty-aware score: `score_final = mu - 0.35 * uncert` using quantile HGBR outputs.
- Cost-aware portfolio rules: IVP + uncertainty penalty + ordered constraints + no-trade band.
- Reports/backtests are NET of fees/taxes/slippage/fill penalties and include risk disclaimers.

## Engineering discipline & release gates

To keep correctness, determinism, and operator safety stable across PRs, this repository enforces:

- ADR-governed architecture decisions under `adr/`.
- Contract-first canonical schemas under `packages/data/contracts/` requiring `schema_version`, canonical JSON and SHA-256 hash stability.
- Structured logging standard for services with `{service, trace_id, symbol?, tf?, run_id?}`.
- Bounded verification pipeline via `make verify-program` generating reproducible artifacts in `artifacts/verification/`.
- CI release gates: lint, format, typecheck, migrations, tests, UI forbidden-string guardrail, verification artifact upload.

Primary developer commands:

```bash
make quality-gate
make run-api
make run-worker
make run-ui
make run-realtime
make replay-demo
make verify-program
```

## Realtime verification & rollback readiness

Verification artifacts are produced under `artifacts/verification/` via:

```bash
make rt-load-test
make rt-chaos-test
make verify-program
```

Replayable evidence pack is stored in `tests/fixtures/replay/`:
- `event_log_fixture.jsonl`
- `expected_bars_fixture.json`
- `expected_signals_fixture.json`
- `expected_parity_reconciliation_fixture.json`

Rollback plan:
- Disable realtime profile/feature flags and keep API + batch analytics online.
- Continue offline replay/analysis workflows until incidents are resolved.

## Realtime UI behavior and budgets

- Realtime API surface is bounded and pull-only (no websocket):
  - `GET /realtime/summary`
  - `GET /realtime/bars?symbol=&tf=&limit<=500`
  - `GET /realtime/signals?symbol=&tf=&limit<=500`
  - `GET /realtime/hot/top_movers?tf=&limit<=100`
  - `GET /realtime/hot/volume_spikes?tf=&limit<=100`
- If realtime is unavailable, endpoints return graceful payload:
  `{"realtime_disabled": true, "message": "..."}`.
- Streamlit topbar polls realtime summary every 2s in `LIVE/PAPER` when realtime is enabled.
- If `stream_lag_s > 5`, UI throttles polling to 5s and shows throttled badge.
- Chart page keeps `Live refresh` default `OFF`; when enabled it only requests latest 200 bars.
- Charts never render beyond `MAX_POINTS_PER_CHART`; deterministic downsampling is applied to stay responsive.


## Trading OPS hardening (paper OMS)

- OMS lifecycle implemented: `NEW -> VALIDATED -> SUBMITTED -> PARTIAL_FILLED -> FILLED -> CANCELLED -> REJECTED -> EXPIRED`.
- Paper-only order endpoint: `/orders/submit` (`adapter=paper` required), idempotent by `client_order_id`.
- Pre-trade hard checks enforce:
  - max single-name <= 10% NAV
  - max sector <= 25% NAV
  - min cash >= 5% NAV
  - liquidity cap `position_value <= ADTV * 0.05 * 3`
  - tick / lot / session validation via market rules.
- Reconciliation fail-safe creates data incident and sets governance `PAUSED` on mismatch.
- Governance status is exposed at `/governance/status` and surfaced in Portfolio/New Orders UI.


## Realtime observability SLOs & incident automation

- Rolling 5-minute SLO windows are computed with cheap in-memory deque + percentile snapshots.
- SLO metrics tracked:
  - gateway: `ingest_lag_s` p50/p95
  - bar builder: `bar_build_latency_s` p50/p95
  - signal engine: `signal_latency_s` p50/p95
  - `redis_stream_pending`
- Incident automation rules:
  - `REALTIME_LAG_HIGH` (`runbook:realtime_lag`) when ingest lag p95 > 5s
  - `BAR_BUILD_SLOW` (`runbook:bar_perf`) when bar build latency p95 > 3s
  - `SIGNAL_SLOW` (`runbook:signal_perf`) when signal latency p95 > 5s
  - `STREAM_BACKLOG` (`runbook:redis_backlog`) when pending > 50,000
- Worker job `job_realtime_incident_monitor` evaluates SLO snapshots and creates deterministic `data_health_incidents` rows with runbook IDs.
- Data Health UI exposes realtime ops gauges/incidents/runbook IDs and governance endpoint includes realtime ops gauges.


## Simple Mode Quickstart (5 lệnh)

```bash
make setup
make run-api
make run-worker
make run-ui
python -m scripts.ingest_data_drop --inbox data_drop/inbox --mapping configs/providers/data_drop_default.yaml --out data_demo/prices_demo_1d.csv
```

- Mở UI tại `http://localhost:8501`, chọn trang **Simple Mode**.
- **Paper**: xác nhận sẽ ghi giao dịch paper vào ledger.
- **Draft**: chỉ lưu lệnh nháp, không tạo fill.
- **Live**: mặc định tắt (`ENABLE_LIVE_TRADING=false`), không hướng dẫn lách luật.
