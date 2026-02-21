from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.quant_stats.bootstrap import block_bootstrap_ci
from core.quant_stats.moments import sample_kurtosis_gamma4, sample_skewness_gamma3
from core.quant_stats.psr_dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from core.raocmoe import (
    ExecutionTCA,
    ExpertSet,
    GovernanceEngine,
    HedgeRouter,
    PortfolioController,
    RegimeEngine,
    UncertaintyEngine,
    build_default_detectors,
    load_config,
    update_all_detectors,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_dataset(universe: list[str], start: str, end: str, seed: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for symbol in universe:
        ret = rng.normal(0.0005, 0.02, len(dates))
        px = 100.0
        for i, d in enumerate(dates):
            px = px * (1.0 + float(ret[i]))
            rows.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "close": px,
                    "ret_1": float(ret[i]),
                    "realized_vol": abs(float(ret[i])),
                    "corr_index": 0.2,
                    "liq_stress": 0.1,
                    "sector": "OTHER",
                }
            )
    return pd.DataFrame(rows)


def run_raocmoe_backtest(params: dict, outdir: Path) -> dict:
    universe = [str(x) for x in params.get("universe", ["BTCUSDT", "ETHUSDT"])]
    start = str(params.get("start", "2023-01-01"))
    end = str(params.get("end", "2023-12-31"))
    cfg = load_config()
    seed = int(cfg["random_seed"])
    data = _load_dataset(universe, start, end, seed)

    detectors = build_default_detectors(cfg)
    regime = RegimeEngine(cfg)
    uq = UncertaintyEngine(cfg)
    experts = ExpertSet()
    router = HedgeRouter(
        regimes=list(cfg["regime"]["states"]),
        experts=list(cfg["moe"]["experts"]),
        eta=float(cfg["moe"]["routing"]["eta"]),
        loss_clip=float(cfg["moe"]["routing"]["loss_clip"]),
    )
    portfolio = PortfolioController(cfg)
    execution = ExecutionTCA(cfg)
    governance = GovernanceEngine(cfg)

    prev_weights = {s: 0.0 for s in universe}
    nav = 1.0
    equity_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    cp_count = {k: 0 for k in detectors}
    regime_count = {k: 0 for k in cfg["regime"]["states"]}
    entropy_hist: list[float] = []

    dates = sorted(data["date"].unique())
    hist_close: dict[str, list[float]] = {s: [] for s in universe}

    for t, date in enumerate(dates, start=1):
        day = data[data["date"] == date]
        day_mu: dict[str, float] = {}
        day_uncert: dict[str, float] = {}
        day_vol: dict[str, float] = {}
        day_adv: dict[str, float] = {}
        sectors: dict[str, str] = {}

        for _, row in day.iterrows():
            symbol = str(row["symbol"])
            close = float(row["close"])
            ret_1 = float(row["ret_1"])
            hist_close[symbol].append(close)
            ema20 = float(np.mean(hist_close[symbol][-20:]))
            ema50 = float(np.mean(hist_close[symbol][-50:]))
            vwap = float(np.mean(hist_close[symbol][-20:]))

            cp_events, cp_vec = update_all_detectors(
                detectors,
                {
                    "mean_shift": ret_1,
                    "vol_shift": float(row["realized_vol"]),
                    "corr_shift": float(row["corr_index"]),
                    "liq_shift": float(row["liq_stress"]),
                },
                t,
            )
            for ev in cp_events:
                cp_count[ev.detector_name] += 1

            post = regime.update(
                t=t,
                features={
                    "trend_strength": ret_1 * 10.0,
                    "realized_vol": float(row["realized_vol"]),
                    "corr_index": float(row["corr_index"]),
                    "liq_stress": float(row["liq_stress"]),
                    "cpd_score": max(cp_vec.values()) if cp_vec else 0.0,
                },
                cp_events=cp_events,
            )
            regime_count[post.hard_regime] += 1

            f_mu, _ = experts.factor_ml(
                {"ret_1": ret_1, "realized_vol": float(row["realized_vol"])}
            )
            t_mu, _ = experts.trend_tech(close, ema20, ema50)
            m_mu, _ = experts.meanrev_vwap(close, vwap, float(row["liq_stress"]))
            d_mu, _ = experts.defensive_lowvol(post.hard_regime)
            expert_mus = {
                "FACTOR_ML": f_mu,
                "TREND_TECH": t_mu,
                "MEANREV_VWAP": m_mu,
                "DEFENSIVE_LOWVOL": d_mu,
            }
            moe = router.combine(post.hard_regime, expert_mus)
            entropy_hist.append(moe.entropy)

            intervals, uncert = uq.get_intervals(
                t=t,
                regime=post.hard_regime,
                symbol=symbol,
                sector=str(row["sector"]),
                yhat=moe.mu_hat_combined,
                realized_vol=max(1e-6, float(row["realized_vol"])),
            )
            uq.update_with_label(
                t=t,
                horizon="h1",
                y_true=ret_1,
                psi=0.0,
                cp_score=max(cp_vec.values()) if cp_vec else 0.0,
                sector=str(row["sector"]),
            )
            router.update(
                post.hard_regime,
                expert_mus,
                realized_r=ret_1,
                realized_vol=max(1e-6, float(row["realized_vol"])),
            )

            day_mu[symbol] = moe.mu_hat_combined
            day_uncert[symbol] = uncert
            day_vol[symbol] = max(0.001, float(row["realized_vol"]))
            day_adv[symbol] = 1e12
            sectors[symbol] = str(row["sector"])

        paused_uc, uc_details = uq.governance_undercoverage()
        gov = governance.evaluate(
            uq_undercoverage=paused_uc,
            psi=0.0,
            tca_shift=False,
            data_stale=False,
            extra={"uc": uc_details},
        )
        target = portfolio.build_target(
            universe=universe,
            mu=day_mu,
            uncertainty=day_uncert,
            prev=prev_weights,
            vols=day_vol,
            adv20=day_adv,
            nav=nav,
            regime=regime.hard_regime,
            sectors=sectors,
            paused=gov.paused,
        )

        pnl = 0.0
        for symbol in universe:
            row = day[day["symbol"] == symbol].iloc[0]
            ret_1 = float(row["ret_1"])
            pnl += target.weights.get(symbol, 0.0) * ret_1
            ex = execution.estimate(
                symbol=symbol,
                side=(
                    "BUY"
                    if target.weights.get(symbol, 0.0) >= prev_weights.get(symbol, 0.0)
                    else "SELL"
                ),
                delta_weight=target.weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0),
                nav=nav,
                price=float(row["close"]),
                adtv=1e9,
                atr14=max(1e-6, float(row["realized_vol"]) * float(row["close"])),
                instrument_type="crypto",
                regime_id=cfg["regime"]["states"].index(regime.hard_regime),
            )
            _ = ex

        nav *= 1.0 + pnl
        equity_rows.append({"date": pd.Timestamp(date).date().isoformat(), "equity": float(nav)})
        debug_rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "regime": regime.hard_regime,
                "paused": gov.paused,
                "turnover": target.turnover,
            }
        )
        prev_weights = dict(target.weights)

    equity_df = pd.DataFrame(equity_rows)
    rets = equity_df["equity"].pct_change().fillna(0.0)
    ann_vol = float(rets.std() * np.sqrt(252.0))
    sharpe = float((rets.mean() / max(1e-8, rets.std())) * np.sqrt(252.0))
    total_return = float(equity_df["equity"].iloc[-1] - 1.0)
    cagr = float((equity_df["equity"].iloc[-1]) ** (252.0 / max(1, len(equity_df))) - 1.0)
    dd = (equity_df["equity"] / equity_df["equity"].cummax() - 1.0).min()
    skew = sample_skewness_gamma3(rets.values)
    kurt = sample_kurtosis_gamma4(rets.values)
    psr = float(probabilistic_sharpe_ratio(sharpe, 0.0, len(rets), skew, kurt))
    dsr = float(deflated_sharpe_ratio(sharpe, len(rets), skew, kurt, [sharpe], 2)[0])
    ci_sharpe = block_bootstrap_ci(rets.tolist(), "sharpe_non_annualized", block_size=5, n_iter=100)

    dataset_hash = _sha(
        json.dumps(
            {"universe": universe, "start": start, "end": end, "rows": int(len(data))},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    config_hash = _sha(Path("configs/raocmoe.yaml").read_text(encoding="utf-8"))
    code_hash = _sha(Path(".git/HEAD").read_text(encoding="utf-8"))
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_id = _sha(dataset_hash + config_hash + code_hash + ts)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(out / "equity.csv", index=False)
    (out / "debug_samples.jsonl").write_text(
        "\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in debug_rows[:200]),
        encoding="utf-8",
    )

    report = {
        "report_id": report_id,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "metrics": {
            "total_return": total_return,
            "cagr": cagr,
            "mdd": float(dd),
            "vol": ann_vol,
            "sharpe": sharpe,
            "psr": psr,
            "dsr": dsr,
            "bootstrap_ci_sharpe": ci_sharpe,
        },
        "cpd_metrics": cp_count,
        "regime_occupancy": regime_count,
        "routing": {"entropy_mean": float(np.mean(entropy_hist)) if entropy_hist else 0.0},
        "uq": {"coverage": uq.coverage_status()},
        "safety": {"live_trading": False},
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    trades = pd.DataFrame(debug_rows)
    trades.to_csv(out / "trades.csv", index=False)
    pd.DataFrame(debug_rows).to_json(out / "diagnostics.json", orient="records")
    return {"report_id": report_id, "dataset_hash": dataset_hash, "config_hash": config_hash, "code_hash": code_hash, "metrics": report["metrics"]}



def run(universe: list[str], start: str, end: str) -> Path:
    out = Path("reports/raocmoe/manual")
    run_raocmoe_backtest({"universe": universe, "start": start, "end": end}, out)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default="artifacts/raocmoe/manual")
    args = parser.parse_args()
    summary = run_raocmoe_backtest(
        {"universe": args.universe.split(","), "start": args.start, "end": args.end},
        Path(args.outdir),
    )
    print(json.dumps(summary, sort_keys=True))
