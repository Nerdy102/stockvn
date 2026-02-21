from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.user_model import UserModel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_code_hash() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if commit:
            return commit
    except Exception:
        pass
    h = hashlib.sha256()
    for p in [Path(__file__), Path("models/user_model.py")]:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["symbol", "date", "open", "high", "low", "close", "volume", "value_vnd"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df[needed].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume", "value_vnd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["symbol", "date", "close"])
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    return df


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    df["daily_return"] = g["close"].pct_change()
    df["mom5"] = g["close"].transform(lambda s: s / s.shift(5) - 1.0)
    df["mom20"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    df["vol20"] = g["daily_return"].transform(lambda s: s.rolling(20).std())
    df["sma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    df["sma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["next_day_return"] = g["close"].shift(-1) / df["close"] - 1.0
    df["exec_price_next"] = g["open"].shift(-1)
    fallback_close_next = g["close"].shift(-1)
    df["exec_price_next"] = df["exec_price_next"].fillna(fallback_close_next)
    df["next_exec_price"] = g["open"].shift(-2).fillna(g["close"].shift(-2))
    return df


def compute_split_dates(
    dates: list[pd.Timestamp], train_end: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(dates) < 40:
        raise ValueError("Need at least 40 trading days for train/test split")
    start = dates[0]
    if train_end:
        train_end_ts = pd.Timestamp(train_end)
    else:
        idx = int(len(dates) * 0.7)
        train_end_ts = dates[max(1, idx)]
    if train_end_ts >= dates[-2]:
        raise ValueError("train_end too late; not enough test period")
    return start, train_end_ts


def run_walk_forward_backtest(
    features: pd.DataFrame,
    train_end: str | None,
    top_k: int,
    min_score: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fee = fee_bps / 10000.0
    slippage = slippage_bps / 10000.0
    total_cost_rate = fee + slippage

    dates = sorted(features["date"].dropna().unique().tolist())
    _, train_end_ts = compute_split_dates(dates, train_end)

    model = UserModel()
    prev_weights: dict[str, float] = {}
    equity = 1.0
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    turnover_sum = 0.0
    trades_count = 0

    test_signal_dates = [d for d in dates if d > train_end_ts and d <= dates[-3]]
    feature_cols = ["daily_return", "mom5", "mom20", "vol20", "sma5", "sma20"]

    for signal_date in test_signal_dates:
        train = features[
            (features["date"] <= train_end_ts) | (features["date"] < signal_date)
        ].copy()
        train = train.dropna(subset=feature_cols + ["next_day_return"])
        if train.empty:
            continue
        X_train = train[feature_cols]
        y_train = train["next_day_return"]
        model.fit(X_train, y_train)

        day_rows = features[features["date"] == signal_date].copy()
        day_rows = day_rows.dropna(subset=feature_cols)
        if day_rows.empty:
            continue
        scores = model.predict(day_rows[feature_cols]).astype(float)
        day_rows = day_rows.assign(score=scores.values)

        selected = (
            day_rows[day_rows["score"] > min_score]
            .sort_values("score", ascending=False)
            .head(top_k)
        )
        target_weights = (
            {s: 1.0 / len(selected) for s in selected["symbol"].tolist()} if len(selected) else {}
        )

        idx_signal = dates.index(signal_date)
        exec_date = dates[idx_signal + 1]
        next_exec_date = dates[idx_signal + 2]
        open_rows = features[features["date"] == signal_date][
            ["symbol", "exec_price_next", "next_exec_price"]
        ].copy()

        gross_turnover = 0.0
        symbols = sorted(set(prev_weights) | set(target_weights))
        for symbol in symbols:
            old_w = prev_weights.get(symbol, 0.0)
            new_w = target_weights.get(symbol, 0.0)
            delta = abs(new_w - old_w)
            if delta > 0:
                trades_count += 1
                trade_rows.append(
                    {
                        "date": exec_date.date().isoformat(),
                        "signal_date": signal_date.date().isoformat(),
                        "symbol": symbol,
                        "target_weight": new_w,
                        "trade_value": delta * equity,
                        "cost": delta * equity * total_cost_rate,
                    }
                )
            gross_turnover += delta

        open_map = open_rows.set_index("symbol")
        period_return = 0.0
        for symbol, w in target_weights.items():
            if symbol not in open_map.index:
                continue
            px0 = float(open_map.loc[symbol, "exec_price_next"])
            px1 = float(open_map.loc[symbol, "next_exec_price"])
            if not np.isfinite(px0) or not np.isfinite(px1) or px0 <= 0:
                continue
            period_return += w * (px1 / px0 - 1.0)

        period_cost = gross_turnover * total_cost_rate
        turnover_sum += gross_turnover
        equity *= 1.0 + period_return - period_cost
        equity_rows.append(
            {
                "date": next_exec_date.date().isoformat(),
                "equity": equity,
                "signal_date": signal_date.date().isoformat(),
                "exec_date": exec_date.date().isoformat(),
            }
        )
        prev_weights = target_weights

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    meta = {
        "train_end": str(train_end_ts.date()),
        "test_start": str((train_end_ts + pd.Timedelta(days=1)).date()),
        "total_turnover": turnover_sum,
        "trades_count": trades_count,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
    }
    return equity_curve, trades, meta


def compute_metrics(
    equity_curve: pd.DataFrame, turnover: float, trades_count: int
) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "turnover": turnover,
            "trades": float(trades_count),
            "n_days": 0.0,
            "years": 0.0,
        }
    e = equity_curve.copy()
    e["ret"] = e["equity"].pct_change().fillna(0.0)
    n_days = float(len(e))
    years = max(n_days / 252.0, 1e-9)
    total_return = float(e["equity"].iloc[-1] - 1.0)
    cagr = float((e["equity"].iloc[-1] ** (1.0 / years)) - 1.0)
    vol = float(e["ret"].std(ddof=0) * np.sqrt(252.0))
    sharpe = float((e["ret"].mean() * 252.0) / vol) if vol > 0 else 0.0
    running_max = e["equity"].cummax()
    drawdown = e["equity"] / running_max - 1.0
    mdd = float(drawdown.min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "annual_vol": vol,
        "sharpe": sharpe,
        "turnover": float(turnover),
        "trades": float(trades_count),
        "n_days": n_days,
        "years": years,
    }


def _write_plotly(equity_curve: pd.DataFrame, out_equity_html: Path, out_dd_html: Path) -> None:
    out_equity_html.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(
        data=[go.Scatter(x=equity_curve["date"], y=equity_curve["equity"], mode="lines")]
    )
    fig.update_layout(title="VN10 Equity Curve", xaxis_title="Date", yaxis_title="Equity")
    fig.write_html(out_equity_html, include_plotlyjs="cdn")

    running_max = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / running_max - 1.0
    fig_dd = go.Figure(data=[go.Scatter(x=equity_curve["date"], y=drawdown, mode="lines")])
    fig_dd.update_layout(title="VN10 Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    fig_dd.write_html(out_dd_html, include_plotlyjs="cdn")


def format_report(
    metrics: dict[str, float],
    hashes: dict[str, str],
    config: dict[str, Any],
    equity_curve: pd.DataFrame,
) -> str:
    start_date = equity_curve["date"].iloc[0] if not equity_curve.empty else "-"
    end_date = equity_curve["date"].iloc[-1] if not equity_curve.empty else "-"
    return "\n".join(
        [
            "# VN10 Backtest Report",
            "",
            "## Reproducibility",
            f"- dataset_hash: `{hashes['dataset_hash']}`",
            f"- config_hash: `{hashes['config_hash']}`",
            f"- code_hash: `{hashes['code_hash']}`",
            "",
            "## Period",
            f"- start_date: {start_date}",
            f"- end_date: {end_date}",
            f"- trading_days: {int(metrics['n_days'])}",
            f"- years_equivalent: {metrics['years']:.2f}",
            "",
            "## Performance",
            f"- total_return: {metrics['total_return'] * 100:.2f}%",
            f"- CAGR: {metrics['cagr'] * 100:.2f}%",
            f"- max_drawdown: {metrics['max_drawdown'] * 100:.2f}%",
            f"- annual_vol: {metrics['annual_vol'] * 100:.2f}%",
            f"- Sharpe (rf=0): {metrics['sharpe']:.2f}",
            f"- turnover: {metrics['turnover']:.2f}",
            f"- trades: {int(metrics['trades'])}",
            "",
            "## Config",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
            "```",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate user model on VN10 EOD data (walk-forward, T+1)"
    )
    parser.add_argument("--prices", default="data_demo/prices_demo_1d.csv")
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices_path = Path(args.prices)
    prices = load_prices(prices_path)
    features = build_features(prices)

    config = {
        "prices": str(prices_path),
        "train_end": args.train_end,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "generated_at_utc": dt.datetime.utcnow().isoformat(),
    }
    config_payload = json.dumps(config, sort_keys=True)
    hashes = {
        "dataset_hash": sha256_file(prices_path),
        "config_hash": hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
        "code_hash": get_code_hash(),
    }

    equity_curve, trades, meta = run_walk_forward_backtest(
        features,
        train_end=args.train_end,
        top_k=args.top_k,
        min_score=args.min_score,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    metrics = compute_metrics(equity_curve, meta["total_turnover"], int(meta["trades_count"]))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    equity_path = reports_dir / "vn10_equity_curve.csv"
    trades_path = reports_dir / "vn10_trades.csv"
    cfg_path = reports_dir / "vn10_config.json"
    md_path = reports_dir / "vn10_backtest_report.md"

    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)
    cfg_path.write_text(
        json.dumps({"config": config, "hashes": hashes, "meta": meta}, indent=2), encoding="utf-8"
    )
    report_text = format_report(metrics, hashes, config, equity_curve)
    md_path.write_text(report_text, encoding="utf-8")

    if not equity_curve.empty:
        _write_plotly(
            equity_curve, reports_dir / "vn10_equity_curve.html", reports_dir / "vn10_drawdown.html"
        )

    print("=== VN10 Backtest Summary ===")
    print(f"start_date={equity_curve['date'].iloc[0] if not equity_curve.empty else '-'}")
    print(f"end_date={equity_curve['date'].iloc[-1] if not equity_curve.empty else '-'}")
    print(f"trading_days={int(metrics['n_days'])} years={metrics['years']:.2f}")
    print(f"total_return={metrics['total_return'] * 100:.2f}%")
    print(f"CAGR={metrics['cagr'] * 100:.2f}%")
    print(f"max_drawdown={metrics['max_drawdown'] * 100:.2f}%")
    print(f"annual_vol={metrics['annual_vol'] * 100:.2f}%")
    print(f"sharpe={metrics['sharpe']:.2f}")
    print(f"turnover={metrics['turnover']:.2f} trades={int(metrics['trades'])}")


if __name__ == "__main__":
    main()
