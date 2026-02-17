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
- Rebalance suggestion (MVP): rules max sector weight / max single name / target cash

## CI (GitHub Actions)

`.github/workflows/ci.yml` chạy:
- ruff
- black --check
- pytest

---
# stockvn
