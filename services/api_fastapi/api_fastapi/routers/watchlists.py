from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import re
import uuid
from email.parser import BytesParser
from email.policy import default
from typing import Any

import pandas as pd
from core.db.models import TagDictionary, Ticker, Watchlist, WatchlistItem, Workspace
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["watchlists"])
log = logging.getLogger(__name__)

TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$")
MAX_TAGS_PER_ITEM = 20

BUILTIN_TAGS = [
    "kqkd",
    "policy",
    "cycle",
    "commodity",
    "dividend",
    "turnaround",
    "defensive",
    "growth",
    "value",
    "momentum",
]


class WorkspaceCreate(BaseModel):
    user_id: str | None = None
    name: str = Field(min_length=1, max_length=200)


class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class WatchlistItemIn(BaseModel):
    symbol: str
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    pinned: bool = False


class WatchlistItemPatch(BaseModel):
    tags: list[str] | None = None
    note: str | None = None
    pinned: bool | None = None


class WatchlistItemOut(BaseModel):
    id: str
    watchlist_id: str
    symbol: str
    tags: list[str]
    note: str
    pinned: bool
    created_at: dt.datetime
    updated_at: dt.datetime


class UpsertItemResponse(BaseModel):
    status: str
    item: WatchlistItemOut


class ImportResponse(BaseModel):
    inserted: int
    updated: int
    invalid_symbols: list[str]
    invalid_rows: int


def normalize_tags(raw_tags: list[str]) -> list[str]:
    values = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    dedup = sorted(set(values))
    if len(dedup) > MAX_TAGS_PER_ITEM:
        raise HTTPException(status_code=400, detail=f"max {MAX_TAGS_PER_ITEM} tags per item")
    bad = [t for t in dedup if not TAG_PATTERN.fullmatch(t)]
    if bad:
        log.warning("invalid tag attempt", extra={"tags": bad})
        raise HTTPException(status_code=400, detail=f"invalid tags: {', '.join(bad)}")
    return dedup


def parse_tags_value(value: str | float | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    chunks = [c.strip() for c in re.split(r"[,\s]+", text) if c.strip()]
    return normalize_tags(chunks)


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _item_out(row: WatchlistItem) -> WatchlistItemOut:
    tags = json.loads(row.tags_json) if row.tags_json else []
    return WatchlistItemOut(
        id=row.id,
        watchlist_id=row.watchlist_id,
        symbol=row.symbol,
        tags=tags,
        note=row.note_text,
        pinned=row.pinned,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _ensure_symbol_exists(db: Session, symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise HTTPException(status_code=400, detail="symbol is required")
    exists = db.exec(select(Ticker.symbol).where(Ticker.symbol == normalized)).first()
    if not exists:
        raise HTTPException(status_code=400, detail=f"invalid symbol: {normalized}")
    return normalized


def _upsert_item(
    db: Session, watchlist_id: str, payload: WatchlistItemIn
) -> tuple[str, WatchlistItem]:
    symbol = _ensure_symbol_exists(db, payload.symbol)
    tags = normalize_tags(payload.tags)
    note = (payload.note or "")[:2000]
    pinned = bool(payload.pinned)

    existing = db.exec(
        select(WatchlistItem)
        .where(WatchlistItem.watchlist_id == watchlist_id)
        .where(WatchlistItem.symbol == symbol)
    ).first()

    now = dt.datetime.utcnow()
    if existing is None:
        item = WatchlistItem(
            id=str(uuid.uuid4()),
            watchlist_id=watchlist_id,
            symbol=symbol,
            tags_json=json.dumps(tags, ensure_ascii=False),
            note_text=note,
            pinned=pinned,
            created_at=now,
            updated_at=now,
        )
        db.add(item)
        status = "inserted"
    else:
        existing.tags_json = json.dumps(tags, ensure_ascii=False)
        existing.note_text = note
        existing.pinned = pinned
        existing.updated_at = now
        db.add(existing)
        item = existing
        status = "updated"

    db.commit()
    db.refresh(item)
    return status, item


def seed_tag_dictionary(db: Session) -> int:
    inserted = 0
    for tag in BUILTIN_TAGS:
        exists = db.exec(select(TagDictionary.tag).where(TagDictionary.tag == tag)).first()
        if exists:
            continue
        db.add(
            TagDictionary(
                tag=tag,
                description="",
                category="builtin",
                created_at=dt.datetime.utcnow(),
            )
        )
        inserted += 1
    if inserted > 0:
        db.commit()
    return inserted


@router.get("/workspaces", response_model=list[Workspace])
def list_workspaces(
    user_id: str | None = Query(default=None), db: Session = Depends(get_db)
) -> list[Workspace]:
    q = select(Workspace)
    if user_id is None:
        q = q.where(Workspace.user_id.is_(None))
    else:
        q = q.where(Workspace.user_id == user_id)
    return list(db.exec(q.order_by(Workspace.created_at)).all())


@router.post("/workspaces", response_model=Workspace)
def create_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db)) -> Workspace:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="workspace name is required")

    user_is_null = payload.user_id is None or payload.user_id.strip() == ""
    user_value = None if user_is_null else payload.user_id.strip()

    dup = db.exec(
        select(Workspace)
        .where(Workspace.name == name)
        .where(
            Workspace.user_id.is_(None) if user_value is None else Workspace.user_id == user_value
        )
    ).first()
    if dup:
        raise HTTPException(status_code=409, detail="workspace already exists")

    row = Workspace(
        id=str(uuid.uuid4()),
        user_id=user_value,
        name=name,
        created_at=dt.datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/workspaces/{workspace_id}/watchlists", response_model=list[Watchlist])
def list_watchlists(workspace_id: str, db: Session = Depends(get_db)) -> list[Watchlist]:
    return list(
        db.exec(
            select(Watchlist)
            .where(Watchlist.workspace_id == workspace_id)
            .order_by(Watchlist.created_at)
        ).all()
    )


@router.post("/workspaces/{workspace_id}/watchlists", response_model=Watchlist)
def create_watchlist(
    workspace_id: str, payload: WatchlistCreate, db: Session = Depends(get_db)
) -> Watchlist:
    ws = db.exec(select(Workspace).where(Workspace.id == workspace_id)).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")

    name = payload.name.strip()
    dup = db.exec(
        select(Watchlist)
        .where(Watchlist.workspace_id == workspace_id)
        .where(Watchlist.name == name)
    ).first()
    if dup:
        raise HTTPException(status_code=409, detail="watchlist already exists")

    row = Watchlist(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        name=name,
        created_at=dt.datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/watchlists/{watchlist_id}/items", response_model=list[WatchlistItemOut])
def list_watchlist_items(
    watchlist_id: str, db: Session = Depends(get_db)
) -> list[WatchlistItemOut]:
    rows = list(
        db.exec(
            select(WatchlistItem)
            .where(WatchlistItem.watchlist_id == watchlist_id)
            .order_by(WatchlistItem.pinned.desc(), WatchlistItem.symbol.asc())
        ).all()
    )
    return [_item_out(r) for r in rows]


@router.post("/watchlists/{watchlist_id}/items", response_model=UpsertItemResponse)
def upsert_watchlist_item(
    watchlist_id: str, payload: WatchlistItemIn, db: Session = Depends(get_db)
) -> UpsertItemResponse:
    status, item = _upsert_item(db, watchlist_id, payload)
    return UpsertItemResponse(status=status, item=_item_out(item))


@router.patch("/watchlists/{watchlist_id}/items/{item_id}", response_model=WatchlistItemOut)
def patch_watchlist_item(
    watchlist_id: str,
    item_id: str,
    payload: WatchlistItemPatch,
    db: Session = Depends(get_db),
) -> WatchlistItemOut:
    row = db.exec(
        select(WatchlistItem)
        .where(WatchlistItem.id == item_id)
        .where(WatchlistItem.watchlist_id == watchlist_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="watchlist item not found")

    if payload.tags is not None:
        row.tags_json = json.dumps(normalize_tags(payload.tags), ensure_ascii=False)
    if payload.note is not None:
        row.note_text = payload.note[:2000]
    if payload.pinned is not None:
        row.pinned = payload.pinned
    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.commit()
    db.refresh(row)
    return _item_out(row)


def _extract_csv_bytes_from_request(content_type: str, body: bytes) -> bytes:
    if content_type.startswith("text/csv"):
        return body
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="expected multipart/form-data or text/csv")

    msg = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not msg.is_multipart():
        raise HTTPException(status_code=400, detail="invalid multipart payload")

    for part in msg.iter_parts():
        disp = part.get("Content-Disposition", "")
        if 'name="file"' in disp:
            payload = part.get_payload(decode=True) or b""
            if payload:
                return payload
    raise HTTPException(status_code=400, detail="missing file field")


@router.post("/watchlists/{watchlist_id}/import", response_model=ImportResponse)
async def import_watchlist_items(
    watchlist_id: str, request: Request, db: Session = Depends(get_db)
) -> ImportResponse:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    raw = _extract_csv_bytes_from_request(content_type, body)
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    text = raw.decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    if "symbol" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV requires column 'symbol'")

    inserted = 0
    updated = 0
    invalid_symbols: list[str] = []
    invalid_rows = 0

    for _, row in df.iterrows():
        symbol_value = row.get("symbol", "")
        if pd.isna(symbol_value):
            invalid_rows += 1
            continue
        symbol_raw = str(symbol_value).strip().upper()
        if not symbol_raw:
            invalid_rows += 1
            continue

        sym_exists = db.exec(select(Ticker.symbol).where(Ticker.symbol == symbol_raw)).first()
        if not sym_exists:
            invalid_symbols.append(symbol_raw)
            continue

        try:
            tags = parse_tags_value(row.get("tags"))
        except HTTPException:
            invalid_rows += 1
            continue

        note = str(row.get("note", "") if not pd.isna(row.get("note", "")) else "")[:2000]
        pinned = parse_bool_value(row.get("pinned", False))

        status, _ = _upsert_item(
            db,
            watchlist_id,
            WatchlistItemIn(symbol=symbol_raw, tags=tags, note=note, pinned=pinned),
        )
        if status == "inserted":
            inserted += 1
        else:
            updated += 1

    invalid_symbols = sorted(set(invalid_symbols))
    log.info(
        "watchlist import summary",
        extra={
            "watchlist_id": watchlist_id,
            "inserted": inserted,
            "updated": updated,
            "invalid_symbols": len(invalid_symbols),
            "invalid_rows": invalid_rows,
        },
    )
    return ImportResponse(
        inserted=inserted,
        updated=updated,
        invalid_symbols=invalid_symbols,
        invalid_rows=invalid_rows,
    )


@router.get("/watchlists/{watchlist_id}/export")
def export_watchlist_items(watchlist_id: str, db: Session = Depends(get_db)) -> Response:
    rows = db.exec(
        select(
            WatchlistItem.symbol,
            Ticker.exchange,
            Ticker.sector,
            WatchlistItem.tags_json,
            WatchlistItem.note_text,
            WatchlistItem.pinned,
        )
        .select_from(WatchlistItem)
        .join(Ticker, Ticker.symbol == WatchlistItem.symbol, isouter=True)
        .where(WatchlistItem.watchlist_id == watchlist_id)
        .order_by(WatchlistItem.pinned.desc(), WatchlistItem.symbol.asc())
    ).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["symbol", "exchange", "sector", "tags", "note", "pinned"])
    for symbol, exchange, sector, tags_json, note_text, pinned in rows:
        tags = ",".join(json.loads(tags_json or "[]"))
        writer.writerow([symbol, exchange or "", sector or "", tags, note_text or "", bool(pinned)])

    content = buf.getvalue().encode("utf-8")
    return Response(content=content, media_type="text/csv")


@router.get("/tag_dictionary")
def list_tag_dictionary(db: Session = Depends(get_db)) -> list[TagDictionary]:
    seed_tag_dictionary(db)
    return list(db.exec(select(TagDictionary).order_by(TagDictionary.tag)).all())
