from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

MIN_ROWS_1D = 200
MIN_ROWS_60M = 600
MAX_MISSING_RATIO = 0.02


@dataclass
class GateResult:
    ok: bool
    cleaned: pd.DataFrame
    failures: list[str]
    warnings: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    outlier_jump_flagged: bool = False


def _ts_col(df: pd.DataFrame) -> str:
    return "timestamp" if "timestamp" in df.columns else "date"


def check_monotonic_ts(df: pd.DataFrame) -> bool:
    c = _ts_col(df)
    return pd.to_datetime(df[c], errors="coerce").is_monotonic_increasing


def check_unique_ts(df: pd.DataFrame) -> bool:
    c = _ts_col(df)
    return not pd.to_datetime(df[c], errors="coerce").duplicated().any()


def dedup_keep_last(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    c = _ts_col(df)
    out = df.copy()
    out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    had_dup = bool(out[c].duplicated().any())
    out = out.drop_duplicates(subset=[c], keep="last").sort_values(c).reset_index(drop=True)
    return out, had_dup


def check_positive_prices(df: pd.DataFrame) -> bool:
    cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    return bool((df[cols] > 0).all().all()) if cols else False


def check_high_low_consistency(df: pd.DataFrame) -> bool:
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return False
    high_ok = (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    low_ok = (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    return bool(high_ok and low_ok)


def check_volume_nonnegative(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns:
        return False
    return bool((df["volume"] >= 0).all())


def check_min_rows(df: pd.DataFrame, min_rows: int) -> bool:
    return len(df) >= min_rows


def check_missing_ratio(df: pd.DataFrame, max_missing_ratio: float = MAX_MISSING_RATIO) -> bool:
    required = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not required:
        return False
    ratio = float(df[required].isna().mean().mean())
    return ratio <= max_missing_ratio


def check_outlier_jump(df: pd.DataFrame, market: str) -> bool:
    if "close" not in df.columns or len(df) < 2:
        return False
    rets = df["close"].pct_change().abs()
    threshold = 0.4 if market == "vn" else 0.8
    return bool((rets > threshold).any())


def dataset_hash(*, market: str, symbol: str, timeframe: str, start: str, end: str, df: pd.DataFrame) -> str:
    c = _ts_col(df)
    ordered = df.sort_values(c)
    payload = "".join(
        f"{row[c]}|{row['open']}|{row['high']}|{row['low']}|{row['close']}|{row['volume']}\n"
        for _, row in ordered.iterrows()
    )
    checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    root = f"{market}|{symbol}|{timeframe}|{start}|{end}|{len(ordered)}|{checksum}"
    return hashlib.sha256(root.encode("utf-8")).hexdigest()


def run_quality_gates(
    df: pd.DataFrame,
    timeframe: str,
    *,
    market: str = "vn",
    min_rows_1d: int = MIN_ROWS_1D,
    min_rows_60m: int = MIN_ROWS_60M,
    max_missing_ratio: float = MAX_MISSING_RATIO,
) -> GateResult:
    cleaned, had_dup = dedup_keep_last(df)
    failures: list[str] = []
    warnings: list[str] = []
    risk_tags: list[str] = []
    if had_dup:
        warnings.append("DUPLICATE_TIMESTAMP_DROPPED_KEEP_LAST")
    if not check_monotonic_ts(cleaned):
        failures.append("TIMESTAMP_NOT_MONOTONIC")
    if not check_unique_ts(cleaned):
        failures.append("TIMESTAMP_NOT_UNIQUE")
    if not check_positive_prices(cleaned):
        failures.append("PRICE_NOT_POSITIVE")
    if not check_high_low_consistency(cleaned):
        failures.append("HIGH_LOW_INCONSISTENT")
    if not check_volume_nonnegative(cleaned):
        failures.append("VOLUME_NEGATIVE")
    if not check_missing_ratio(cleaned, max_missing_ratio):
        failures.append("MISSING_RATIO_TOO_HIGH")
    min_rows = min_rows_60m if timeframe == "60m" else min_rows_1d
    if not check_min_rows(cleaned, min_rows):
        failures.append(f"MIN_ROWS_FAIL_{min_rows}")
    outlier = check_outlier_jump(cleaned, market)
    if outlier:
        risk_tags.append("Dữ liệu bất thường")

    return GateResult(
        ok=len(failures) == 0,
        cleaned=cleaned,
        failures=failures,
        warnings=warnings,
        risk_tags=risk_tags,
        outlier_jump_flagged=outlier,
    )


# Tương thích ngược
check_monotonic_timestamp = check_monotonic_ts
check_no_duplicate_timestamp = check_unique_ts
