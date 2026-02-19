from __future__ import annotations

import datetime as dt
from typing import Any

from core.db.models import Fill, Portfolio, Trade
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.settings import get_settings
from core.simple_mode.backtest import quick_backtest
from core.simple_mode.models import MODEL_PROFILES, run_signal
from core.simple_mode.orchestrator import build_client_order_id, generate_order_draft
from core.simple_mode.safety import ensure_disclaimers
from core.simple_mode.schemas import ConfirmExecutePayload
from data.providers.factory import get_provider
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/simple", tags=["simple_mode"])

MAX_POINTS_PER_CHART = 300


class RunSignalIn(BaseModel):
    symbol: str
    timeframe: str = "1D"
    model_id: str = "model_1"
    mode: str = "paper"


class RunCompareIn(BaseModel):
    symbols: list[str] = Field(default_factory=list, min_length=1, max_length=20)
    timeframe: str = "1D"
    lookback_days: int = Field(default=252, ge=60, le=756)


@router.get("/models")
def get_models() -> dict[str, Any]:
    settings = get_settings()
    return {
        "models": [m.__dict__ for m in MODEL_PROFILES],
        "live_enabled": str(__import__("os").getenv("ENABLE_LIVE_TRADING", "false")).lower()
        == "true",
        "max_points_per_chart": MAX_POINTS_PER_CHART,
        "warning": "Đây là tín hiệu nghiên cứu, không phải lời khuyên đầu tư.",
        "provider": settings.DATA_PROVIDER,
    }


@router.post("/run_signal")
def run_signal_api(payload: RunSignalIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    settings = get_settings()
    provider = get_provider(settings)
    df = provider.get_ohlcv(payload.symbol, payload.timeframe)
    if len(df) > MAX_POINTS_PER_CHART:
        step = max(1, len(df) // MAX_POINTS_PER_CHART)
        df = df.iloc[::step].copy()

    signal = run_signal(payload.model_id, payload.symbol, payload.timeframe, df)
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    draft = generate_order_draft(signal=signal, market_rules=mr, fees_taxes=fees, mode=payload.mode)
    data_status = {
        "rows": int(len(df)),
        "last_update": str(df.iloc[-1].get("date") if not df.empty else ""),
    }
    return {
        "signal": signal.model_dump(),
        "draft": None if draft is None else draft.model_dump(),
        "data_status": data_status,
    }


@router.post("/run_compare")
def run_compare_api(payload: RunCompareIn) -> dict[str, Any]:
    settings = get_settings()
    provider = get_provider(settings)
    end = dt.date.today()
    start = end - dt.timedelta(days=payload.lookback_days)

    rows: list[dict[str, Any]] = []
    for model in ["model_1", "model_2", "model_3"]:
        reports = []
        for symbol in payload.symbols:
            df = provider.get_ohlcv(symbol, payload.timeframe)
            reports.append(quick_backtest(model, symbol, df, start, end))
        avg = {
            "model_id": model,
            "cagr": sum(r.cagr for r in reports) / len(reports),
            "mdd": sum(r.mdd for r in reports) / len(reports),
            "sharpe": sum(r.sharpe for r in reports) / len(reports),
            "sortino": sum(r.sortino for r in reports) / len(reports),
            "turnover": sum(r.turnover for r in reports) / len(reports),
            "net_return": sum(r.net_return for r in reports) / len(reports),
            "config_hash": reports[0].config_hash,
            "dataset_hash": reports[0].dataset_hash,
            "code_hash": reports[0].code_hash,
        }
        rows.append(avg)
    rows.sort(key=lambda x: x["sharpe"], reverse=True)
    return {
        "leaderboard": rows,
        "warning": "Quá khứ không đảm bảo tương lai. Có rủi ro overfit và chi phí thực tế khác mô phỏng.",
    }


class CreateDraftIn(RunSignalIn):
    pass


@router.post("/create_order_draft")
def create_order_draft(payload: CreateDraftIn) -> dict[str, Any]:
    settings = get_settings()
    provider = get_provider(settings)
    df = provider.get_ohlcv(payload.symbol, payload.timeframe)
    signal = run_signal(payload.model_id, payload.symbol, payload.timeframe, df)
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    draft = generate_order_draft(signal=signal, market_rules=mr, fees_taxes=fees, mode=payload.mode)
    return {"signal": signal.model_dump(), "draft": None if draft is None else draft.model_dump()}


@router.post("/confirm_execute")
def confirm_execute(
    payload: ConfirmExecutePayload, db: Session = Depends(get_db)
) -> dict[str, Any]:
    ensure_disclaimers(
        acknowledged_educational=payload.acknowledged_educational,
        acknowledged_loss=payload.acknowledged_loss,
        mode=payload.mode,
        acknowledged_live_eligibility=payload.acknowledged_live_eligibility,
        age=payload.age,
    )

    if payload.mode == "live":
        raise HTTPException(
            status_code=422, detail="LiveBrokerStub: mặc định không hoạt động để bảo vệ an toàn"
        )

    portfolio = db.exec(select(Portfolio).where(Portfolio.id == payload.portfolio_id)).first()
    if portfolio is None:
        portfolio = Portfolio(id=payload.portfolio_id, name="Simple Mode Portfolio")
        db.add(portfolio)
        db.flush()

    client_order_id = build_client_order_id(payload.draft.symbol)

    if payload.mode == "paper":
        trade = Trade(
            portfolio_id=payload.portfolio_id,
            trade_date=dt.date.today(),
            symbol=payload.draft.symbol,
            side=payload.draft.side,
            quantity=float(payload.draft.qty),
            price=float(payload.draft.price),
            strategy_tag=f"simple:{payload.draft.mode}",
            notes=client_order_id,
            commission=payload.draft.fee_tax.commission,
            taxes=payload.draft.fee_tax.sell_tax,
            external_id=client_order_id,
        )
        db.add(trade)
        db.flush()
        fill = Fill(
            order_id=int(trade.id or 0),
            execution_id=client_order_id,
            quantity=float(payload.draft.qty),
            price=float(payload.draft.price),
            commission=payload.draft.fee_tax.commission,
            taxes=payload.draft.fee_tax.sell_tax,
        )
        db.add(fill)
        db.commit()
        return {
            "status": "paper_filled",
            "trade_id": trade.id,
            "fill_id": fill.id,
            "off_session": payload.draft.off_session,
        }

    db.commit()
    return {
        "status": "draft_saved",
        "draft": payload.draft.model_dump(),
        "off_session": payload.draft.off_session,
    }


@router.get("/sync_status")
def sync_status(symbol: str = Query(...), timeframe: str = Query(default="1D")) -> dict[str, Any]:
    settings = get_settings()
    provider = get_provider(settings)
    df = provider.get_ohlcv(symbol, timeframe)
    if df.empty:
        return {
            "rows": 0,
            "last_update": None,
            "missing": "Thiếu dữ liệu giá; hãy dùng demo hoặc nạp data_drop",
        }
    return {
        "rows": len(df),
        "last_update": str(df.iloc[-1].get("date") or df.iloc[-1].get("timestamp")),
        "missing": "",
    }
