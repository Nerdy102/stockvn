from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any

import pandas as pd
from core.db.models import (
    FactorScore,
    PriceOHLCV,
    SavedScreen,
    ScreenRun,
    ScreenRunItem,
    Signal,
    Ticker,
    WatchlistItem,
)
from core.universe.manager import UniverseManager
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlmodel import Session, select

from api_fastapi.deps import get_db, parse_cursor

router = APIRouter(prefix="/screeners", tags=["screeners"])

REQUIRED_EXPLAIN_KEYS = ["filters", "factor", "technicals", "story", "final"]


class UniverseSpec(BaseModel):
    preset: str = "ALL"


class NeutralizationSpec(BaseModel):
    enabled: bool = False


class FilterSpec(BaseModel):
    min_adv20_value: float = 1_000_000_000.0
    sector_in: list[str] = Field(default_factory=list)
    exchange_in: list[str] = Field(default_factory=list)
    tags_any: list[str] = Field(default_factory=list)
    tags_all: list[str] = Field(default_factory=list)
    neutralization: NeutralizationSpec = Field(default_factory=NeutralizationSpec)


class TechnicalSpec(BaseModel):
    breakout: bool = False
    trend: bool = False
    pullback: bool = False
    volume_spike: bool = False


class ScreenSchemaV4(BaseModel):
    name: str = "screen-v4"
    as_of_date: dt.date
    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    factor_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "value": 0.2,
            "quality": 0.2,
            "momentum": 0.2,
            "lowvol": 0.2,
            "dividend": 0.2,
        }
    )
    technical_setups: TechnicalSpec = Field(default_factory=TechnicalSpec)

    @model_validator(mode="after")
    def _check_weights(self) -> ScreenSchemaV4:
        if not self.factor_weights:
            raise ValueError("factor_weights is required")
        if any(float(v) < 0 for v in self.factor_weights.values()):
            raise ValueError("factor_weights must be non-negative")
        return self


class ValidateRequest(BaseModel):
    screen: dict[str, Any]


class SaveScreenRequest(BaseModel):
    workspace_id: str
    name: str
    screen: dict[str, Any]
    saved_screen_id: str | None = None


class RunScreenRequest(BaseModel):
    screen: dict[str, Any]
    saved_screen_id: str | None = None


class CanonicalResult(BaseModel):
    normalized: dict[str, Any]
    errors: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    run_id: str
    reused: bool
    summary: dict[str, Any]
    diff: dict[str, Any]
    results: list[dict[str, Any]]


def _canonicalize_and_validate(screen: dict[str, Any]) -> CanonicalResult:
    warnings: list[str] = []
    try:
        parsed = ScreenSchemaV4.model_validate(screen)
    except Exception as exc:
        return CanonicalResult(normalized={}, errors=[str(exc)])

    norm = parsed.model_dump(mode="json")
    norm["filters"]["sector_in"] = sorted(set(norm["filters"].get("sector_in", [])))
    norm["filters"]["exchange_in"] = sorted(set(norm["filters"].get("exchange_in", [])))
    norm["filters"]["tags_any"] = sorted(
        set([str(x).lower() for x in norm["filters"].get("tags_any", [])])
    )
    norm["filters"]["tags_all"] = sorted(
        set([str(x).lower() for x in norm["filters"].get("tags_all", [])])
    )

    fweights = {str(k): float(v) for k, v in norm.get("factor_weights", {}).items()}
    fweights = dict(sorted(fweights.items(), key=lambda kv: kv[0]))
    total = sum(fweights.values())
    if total <= 1e-12:
        raise HTTPException(status_code=400, detail="factor_weights sum must be > 0")
    if abs(total - 1.0) > 1e-9:
        fweights = {k: v / total for k, v in fweights.items()}
        warnings.append("factor_weights normalized to sum=1")
    norm["factor_weights"] = fweights
    if warnings:
        norm["_warnings"] = warnings

    return CanonicalResult(normalized=norm, errors=[])


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_screen(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_universe(symbols: list[str]) -> str:
    canonical = json.dumps(
        sorted(set([s.upper() for s in symbols])), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pull_features_df(db: Session, as_of_date: dt.date, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol"])

    px_rows = db.exec(
        select(PriceOHLCV)
        .where(PriceOHLCV.timeframe == "1D")
        .where(PriceOHLCV.symbol.in_(symbols))
        .where(PriceOHLCV.timestamp <= dt.datetime.combine(as_of_date, dt.time.max))
        .order_by(PriceOHLCV.symbol, PriceOHLCV.timestamp)
    ).all()
    if not px_rows:
        return pd.DataFrame(columns=["symbol"])

    px = pd.DataFrame([r.model_dump() for r in px_rows])
    px["date"] = pd.to_datetime(px["timestamp"]).dt.date
    px = px.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    px["adv20_value"] = px.groupby("symbol")["value_vnd"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )
    px_last = px.groupby("symbol", as_index=False).last()[
        ["symbol", "adv20_value", "close", "volume", "value_vnd"]
    ]

    factor_rows = db.exec(
        select(FactorScore)
        .where(FactorScore.as_of_date == as_of_date)
        .where(FactorScore.symbol.in_(symbols))
    ).all()
    if factor_rows:
        fac = pd.DataFrame([r.model_dump() for r in factor_rows])
        fac = fac.pivot_table(
            index="symbol", columns="factor", values="score", aggfunc="last"
        ).reset_index()
    else:
        fac = pd.DataFrame({"symbol": symbols})

    sig_rows = db.exec(
        select(Signal)
        .where(Signal.symbol.in_(symbols))
        .where(Signal.timestamp <= dt.datetime.combine(as_of_date, dt.time.max))
        .where(Signal.signal_type.in_(["trend", "breakout", "pullback", "volume_spike"]))
        .order_by(Signal.symbol, Signal.timestamp)
    ).all()
    if sig_rows:
        sig = pd.DataFrame([r.model_dump() for r in sig_rows])
        sig["date"] = pd.to_datetime(sig["timestamp"]).dt.date
        sig = sig[sig["date"] <= as_of_date]
        sig = (
            sig.sort_values(["symbol", "date"])
            .groupby(["symbol", "signal_type"], as_index=False)
            .last()
        )
        sig["flag"] = (sig["strength"].fillna(0.0).astype(float) > 0).astype(int)
        sig = sig.pivot_table(
            index="symbol", columns="signal_type", values="flag", aggfunc="max", fill_value=0
        ).reset_index()
    else:
        sig = pd.DataFrame({"symbol": symbols})

    wl_rows = db.exec(
        select(WatchlistItem.symbol, WatchlistItem.tags_json).where(
            WatchlistItem.symbol.in_(symbols)
        )
    ).all()
    if wl_rows:
        tags = pd.DataFrame(wl_rows, columns=["symbol", "tags_json"])
        tags["tags_list"] = tags["tags_json"].map(lambda x: json.loads(x or "[]"))
        tags = tags.explode("tags_list")
        tags["tags_list"] = tags["tags_list"].fillna("").astype(str).str.lower()
        tags = tags[tags["tags_list"] != ""]
        tags = tags.groupby("symbol", as_index=False)["tags_list"].agg(
            lambda s: sorted(set(s.tolist()))
        )
    else:
        tags = pd.DataFrame({"symbol": symbols, "tags_list": [[] for _ in symbols]})

    t_rows = db.exec(
        select(Ticker.symbol, Ticker.exchange, Ticker.sector).where(Ticker.symbol.in_(symbols))
    ).all()
    tdf = (
        pd.DataFrame(t_rows, columns=["symbol", "exchange", "sector"])
        if t_rows
        else pd.DataFrame({"symbol": symbols})
    )

    out = pd.DataFrame({"symbol": symbols})
    out = out.merge(tdf, on="symbol", how="left")
    out = out.merge(px_last, on="symbol", how="left")
    out = out.merge(fac, on="symbol", how="left")
    out = out.merge(sig, on="symbol", how="left")
    out = out.merge(tags[["symbol", "tags_list"]], on="symbol", how="left")

    for col in ["trend", "breakout", "pullback", "volume_spike"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

    out["tags_list"] = out["tags_list"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def _apply_filters_and_score(df: pd.DataFrame, screen: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    f = screen["filters"]
    weights = screen["factor_weights"]

    work = df.copy()
    work["filter_adv20_pass"] = work["adv20_value"].fillna(0.0) >= float(f["min_adv20_value"])

    for fac in ["value", "quality", "momentum", "lowvol", "dividend"]:
        if fac not in work.columns:
            work[fac] = pd.NA
        work[fac] = pd.to_numeric(work[fac], errors="coerce")

    factor_cols = [k for k in weights.keys() if k in work.columns]
    if factor_cols:
        fac_mat = work[factor_cols].fillna(0.0)
        w = pd.Series(weights)
        work["factor_composite"] = fac_mat.mul(w.reindex(factor_cols), axis=1).sum(axis=1)
    else:
        work["factor_composite"] = 0.0

    tech_score = (
        0.10 * work["trend"].astype(float)
        + 0.10 * work["breakout"].astype(float)
        + 0.10 * work["pullback"].astype(float)
        + 0.10 * work["volume_spike"].astype(float)
    )
    work["technical_score"] = tech_score.clip(upper=0.30)

    tags_any = set(f.get("tags_any", []))
    tags_all = set(f.get("tags_all", []))
    if tags_any:
        work["story_any_match"] = work["tags_list"].map(lambda tags: len(set(tags) & tags_any) > 0)
    else:
        work["story_any_match"] = False
    if tags_all:
        work["story_all_match"] = work["tags_list"].map(lambda tags: tags_all.issubset(set(tags)))
    else:
        work["story_all_match"] = False
    work["story_score"] = 0.05 * work["story_any_match"].astype(float) + 0.10 * work[
        "story_all_match"
    ].astype(float)

    work["penalty_liquidity"] = (
        work["adv20_value"].fillna(0.0) < 2.0 * float(f["min_adv20_value"])
    ) * -0.20
    critical_missing = (
        work[
            [c for c in ["value", "quality", "momentum", "lowvol", "dividend"] if c in work.columns]
        ]
        .isna()
        .any(axis=1)
    )
    missing_pen = -0.50 if bool(f.get("neutralization", {}).get("enabled", False)) else 0.0
    work["penalty_missing"] = critical_missing.astype(float) * missing_pen

    work["final_score"] = (
        work["factor_composite"]
        + work["technical_score"]
        + work["story_score"]
        + work["penalty_liquidity"]
        + work["penalty_missing"]
    )

    mask_sector = (
        pd.Series(True, index=work.index)
        if not f.get("sector_in")
        else work.get("sector", pd.Series([None] * len(work), index=work.index)).isin(
            f["sector_in"]
        )
    )
    mask_exchange = (
        pd.Series(True, index=work.index)
        if not f.get("exchange_in")
        else work.get("exchange", pd.Series([None] * len(work), index=work.index)).isin(
            f["exchange_in"]
        )
    )
    work = work[mask_sector & mask_exchange]

    work = work.sort_values(["final_score", "adv20_value"], ascending=[False, False]).reset_index(
        drop=True
    )
    work["rank"] = work.index + 1

    work["explain_json"] = work.apply(
        lambda r: {
            "filters": {
                "min_adv20_value": float(f["min_adv20_value"]),
                "adv20_value": float(r.get("adv20_value") or 0.0),
                "pass": bool(r.get("filter_adv20_pass")),
            },
            "factor": {
                "z_scores": {
                    k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                    for k in ["value", "quality", "momentum", "lowvol", "dividend"]
                },
                "weights": dict(weights),
                "composite": float(r.get("factor_composite") or 0.0),
            },
            "technicals": {
                "trend": int(r.get("trend", 0)),
                "breakout": int(r.get("breakout", 0)),
                "pullback": int(r.get("pullback", 0)),
                "volume_spike": int(r.get("volume_spike", 0)),
                "score": float(r.get("technical_score") or 0.0),
            },
            "story": {
                "tags": list(r.get("tags_list") or []),
                "tags_any": sorted(list(tags_any)),
                "tags_all": sorted(list(tags_all)),
                "any_match": bool(r.get("story_any_match")),
                "all_match": bool(r.get("story_all_match")),
                "score": float(r.get("story_score") or 0.0),
            },
            "final": {
                "liquidity_penalty": float(r.get("penalty_liquidity") or 0.0),
                "missing_penalty": float(r.get("penalty_missing") or 0.0),
                "score": float(r.get("final_score") or 0.0),
            },
        },
        axis=1,
    )
    return work


def _compute_diff(prev_items: pd.DataFrame, cur_items: pd.DataFrame) -> dict[str, Any]:
    if prev_items.empty:
        return {"entrants": cur_items["symbol"].tolist(), "dropped": [], "rank_delta": []}

    prev_rank = prev_items.set_index("symbol")["rank"]
    cur_rank = cur_items.set_index("symbol")["rank"]
    prev_set = set(prev_rank.index)
    cur_set = set(cur_rank.index)

    entrants = sorted(list(cur_set - prev_set))
    dropped = sorted(list(prev_set - cur_set))
    common = sorted(list(cur_set & prev_set))
    rank_delta = [
        {
            "symbol": s,
            "prev_rank": int(prev_rank[s]),
            "cur_rank": int(cur_rank[s]),
            "delta": int(prev_rank[s] - cur_rank[s]),
        }
        for s in common
        if int(prev_rank[s]) != int(cur_rank[s])
    ]

    return {"entrants": entrants, "dropped": dropped, "rank_delta": rank_delta}


@router.post("/validate", response_model=CanonicalResult)
def validate_screen(payload: ValidateRequest) -> CanonicalResult:
    return _canonicalize_and_validate(payload.screen)


@router.post("/save")
def save_screen(payload: SaveScreenRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    val = _canonicalize_and_validate(payload.screen)
    if val.errors:
        raise HTTPException(status_code=400, detail=val.errors)

    now = dt.datetime.utcnow()
    if payload.saved_screen_id:
        row = db.exec(select(SavedScreen).where(SavedScreen.id == payload.saved_screen_id)).first()
        if row is None:
            raise HTTPException(status_code=404, detail="saved screen not found")
        row.name = payload.name
        row.screen_json = val.normalized
        row.updated_at = now
    else:
        row = SavedScreen(
            id=str(uuid.uuid4()),
            workspace_id=payload.workspace_id,
            name=payload.name,
            screen_json=val.normalized,
            created_at=now,
            updated_at=now,
        )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "name": row.name, "screen": row.screen_json}


@router.get("")
def list_saved_screens(
    workspace_id: str | None = Query(default=None), db: Session = Depends(get_db)
) -> dict[str, Any]:
    q = select(SavedScreen)
    if workspace_id:
        q = q.where(SavedScreen.workspace_id == workspace_id)
    rows = list(db.exec(q.order_by(SavedScreen.updated_at.desc())).all())
    return {"screens": [r.model_dump() for r in rows]}


@router.post("/run", response_model=RunResponse)
def run_screen_endpoint(payload: RunScreenRequest, db: Session = Depends(get_db)) -> RunResponse:
    val = _canonicalize_and_validate(payload.screen)
    if val.errors:
        raise HTTPException(status_code=400, detail=val.errors)
    screen = val.normalized

    as_of_date = dt.date.fromisoformat(str(screen["as_of_date"]))
    universe_name = str(screen.get("universe", {}).get("preset", "ALL")).upper()
    symbols, _ = UniverseManager(db).universe(date=as_of_date, name=universe_name)
    universe_hash = _hash_universe(symbols)
    screen_hash = _hash_screen(screen)

    if payload.saved_screen_id:
        existing = db.exec(
            select(ScreenRun)
            .where(ScreenRun.saved_screen_id == payload.saved_screen_id)
            .where(ScreenRun.as_of_date == as_of_date)
            .where(ScreenRun.screen_hash == screen_hash)
            .where(ScreenRun.universe_hash == universe_hash)
        ).first()
        if existing is not None:
            items = list(
                db.exec(
                    select(ScreenRunItem)
                    .where(ScreenRunItem.run_id == existing.id)
                    .order_by(ScreenRunItem.rank)
                ).all()
            )
            return RunResponse(
                run_id=existing.id,
                reused=True,
                summary=existing.summary_json,
                diff=existing.diff_json,
                results=[
                    {
                        "symbol": it.symbol,
                        "rank": it.rank,
                        "score": it.score,
                        "explain": it.explain_json,
                    }
                    for it in items
                ],
            )

    feat = _pull_features_df(db, as_of_date, symbols)
    scored = _apply_filters_and_score(feat, screen)

    summary = {
        "as_of_date": as_of_date.isoformat(),
        "screen_hash": screen_hash,
        "universe_hash": universe_hash,
        "warnings": screen.get("_warnings", []),
        "count": int(len(scored)),
    }

    run_id = str(uuid.uuid4())
    diff_json: dict[str, Any] = {"entrants": [], "dropped": [], "rank_delta": []}

    if payload.saved_screen_id:
        prev = db.exec(
            select(ScreenRun)
            .where(ScreenRun.saved_screen_id == payload.saved_screen_id)
            .order_by(ScreenRun.run_at.desc())
            .limit(1)
        ).first()

        if prev is not None:
            prev_items = pd.DataFrame(
                [
                    {"symbol": r.symbol, "rank": r.rank, "score": r.score}
                    for r in db.exec(
                        select(ScreenRunItem).where(ScreenRunItem.run_id == prev.id)
                    ).all()
                ]
            )
        else:
            prev_items = pd.DataFrame(columns=["symbol", "rank", "score"])
        if scored.empty or not {"symbol", "rank", "final_score"}.issubset(set(scored.columns)):
            cur_items = pd.DataFrame(columns=["symbol", "rank", "score"])
        else:
            cur_items = scored[["symbol", "rank", "final_score"]].rename(
                columns={"final_score": "score"}
            )
        diff_json = _compute_diff(prev_items, cur_items)

        run = ScreenRun(
            id=run_id,
            saved_screen_id=payload.saved_screen_id,
            as_of_date=as_of_date,
            run_at=dt.datetime.utcnow(),
            screen_hash=screen_hash,
            universe_hash=universe_hash,
            summary_json=summary,
            diff_json=diff_json,
        )
        db.add(run)
        if not scored.empty:
            items = [
                ScreenRunItem(
                    run_id=run_id,
                    symbol=str(r.symbol),
                    rank=int(r.rank),
                    score=float(r.final_score),
                    explain_json=r.explain_json,
                )
                for r in scored.itertuples(index=False)
            ]
            for it in items:
                db.add(it)
        db.commit()

    results = []
    if not scored.empty:
        for r in scored.itertuples(index=False):
            ex = r.explain_json
            if sorted(ex.keys()) != sorted(REQUIRED_EXPLAIN_KEYS):
                raise HTTPException(status_code=500, detail="invalid explain schema")
            results.append(
                {
                    "symbol": str(r.symbol),
                    "rank": int(r.rank),
                    "score": float(r.final_score),
                    "adv20_value": float(r.adv20_value) if pd.notna(r.adv20_value) else None,
                    "trend": int(r.trend),
                    "breakout": int(r.breakout),
                    "pullback": int(r.pullback),
                    "volume_spike": int(r.volume_spike),
                    "tags": list(r.tags_list),
                    "explain": ex,
                }
            )

    return RunResponse(
        run_id=run_id, reused=False, summary=summary, diff=diff_json, results=results
    )


@router.get("/runs")
def list_runs(
    saved_screen_id: str,
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    offset = parse_cursor(cursor) if cursor is not None else 0
    rows = list(
        db.exec(
            select(ScreenRun)
            .where(ScreenRun.saved_screen_id == saved_screen_id)
            .order_by(ScreenRun.run_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    next_cursor = str(offset + len(rows)) if len(rows) == limit else None
    return {"runs": [r.model_dump() for r in rows], "next_cursor": next_cursor}


@router.get("/runs/{run_id}/items")
def list_run_items(
    run_id: str,
    cursor: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    offset = parse_cursor(cursor) if cursor is not None else 0
    rows = list(
        db.exec(
            select(ScreenRunItem)
            .where(ScreenRunItem.run_id == run_id)
            .order_by(ScreenRunItem.rank)
            .offset(offset)
            .limit(limit)
        ).all()
    )
    next_cursor = str(offset + len(rows)) if len(rows) == limit else None
    return {"items": [r.model_dump() for r in rows], "next_cursor": next_cursor}
