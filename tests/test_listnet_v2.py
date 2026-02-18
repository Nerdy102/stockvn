from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import SQLModel, Session, create_engine, select

from core.alpha_v3.backtest import BacktestV3Config, run_backtest_v3
from core.db.models import AlphaPrediction, MlFeature, MlLabel
from core.market_rules import load_market_rules
from core.ml.listnet import (
    ListNetConfig,
    cross_validate_listnet,
    listnet_loss_and_grad,
    percentile_gain,
    prepare_listnet_training_frame,
    train_listnet_linear,
)
from worker_scheduler.jobs import job_train_alpha_listnet_v2


def _toy_frame() -> pd.DataFrame:
    rows = []
    start = dt.date(2024, 1, 1)
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for i in range(80):
        d = start + dt.timedelta(days=i)
        for j, sym in enumerate(symbols):
            y = float(0.003 * j + 0.0002 * i)
            rows.append(
                {
                    "symbol": sym,
                    "as_of_date": d,
                    "y_excess": y,
                    "ret_1d": y,
                    "ret_5d": y * 2,
                    "vol_20d": 1.0 / (j + 1),
                }
            )
    return pd.DataFrame(rows)


def test_listnet_gradient_oracle_matches_golden() -> None:
    golden = json.loads(Path("tests/golden/listnet_grad_oracle.json").read_text(encoding="utf-8"))
    x = np.array(golden["X"], dtype=float)
    y_excess = np.array(golden["y_excess"], dtype=float)
    y_gain = percentile_gain(y_excess)
    w0 = np.array(golden["w0"], dtype=float)

    loss, grad, aux = listnet_loss_and_grad(x, y_gain, w0, tau=1.0, lambda_l2=1e-3)

    assert np.allclose(y_gain, np.array(golden["y_gain"], dtype=float), atol=1e-12)
    assert np.allclose(aux["p_true"], np.array(golden["p_true"], dtype=float), atol=1e-6)
    assert np.allclose(aux["p_pred"], np.array(golden["p_pred"], dtype=float), atol=1e-12)
    assert np.allclose(grad, np.array(golden["grad"], dtype=float), atol=1e-6)
    assert abs(loss - float(golden["initial_loss"])) <= 1e-12


def test_listnet_loss_decreases_after_one_adam_step() -> None:
    golden = json.loads(Path("tests/golden/listnet_grad_oracle.json").read_text(encoding="utf-8"))
    x = np.array(golden["X"], dtype=float)
    y_gain = np.array(golden["y_gain"], dtype=float)
    w = np.zeros(2, dtype=float)

    loss0, grad, _ = listnet_loss_and_grad(x, y_gain, w, tau=1.0, lambda_l2=1e-3)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad * grad)
    m_hat = m / (1 - 0.9)
    v_hat = v / (1 - 0.999)
    w1 = w - 0.01 * (m_hat / (np.sqrt(v_hat) + 1e-8))
    loss1, _, _ = listnet_loss_and_grad(x, y_gain, w1, tau=1.0, lambda_l2=1e-3)

    assert loss1 < (loss0 - 1e-6)
    assert abs(loss1 - float(golden["loss_after_one_adam_step"])) <= 1e-10


def test_listnet_training_deterministic_weights() -> None:
    raw = _toy_frame()
    prep, cols = prepare_listnet_training_frame(raw)
    cfg = ListNetConfig(seed=42, epochs=20)
    m1 = train_listnet_linear(prep, feature_columns=cols, config=cfg)
    m2 = train_listnet_linear(prep, feature_columns=cols, config=cfg)
    assert np.allclose(m1.w, m2.w, atol=1e-12)


def test_listnet_integration_smoke_predictions_and_backtest_metrics() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        start = dt.date(2024, 1, 1)
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        for i in range(120):
            d = start + dt.timedelta(days=i)
            for j, sym in enumerate(symbols):
                ret = float((j + 1) * 0.01 + 0.0001 * i)
                session.add(
                    MlFeature(
                        symbol=sym,
                        as_of_date=d,
                        feature_version="v3",
                        ret_1d=ret,
                        ret_5d=ret * 2,
                        vol_20d=1.0 / (j + 1),
                        rsi14=40.0 + j,
                        ema50_slope=ret,
                    )
                )
                session.add(
                    MlLabel(
                        symbol=sym,
                        date=d,
                        y_excess=ret,
                        y_rank_z=float(j - 1.5),
                        label_version="v3",
                    )
                )
        session.commit()

        res = job_train_alpha_listnet_v2(session)
        assert res["trained"] == 1
        assert res["predictions"] > 0

        rows = session.exec(select(AlphaPrediction).where(AlphaPrediction.model_id == "alpha_listnet_v2")).all()
        assert len(rows) > 0
        assert all(np.isfinite(r.score) for r in rows)

        # Build a simple single-name signal and run backtest sanity check
        aaa = sorted([r for r in rows if r.symbol == "AAA"], key=lambda x: x.as_of_date)
        assert aaa
        bars = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([r.as_of_date for r in aaa]),
                "open": [10000 + i * 10 for i in range(len(aaa))],
                "high": [10050 + i * 10 for i in range(len(aaa))],
                "low": [9950 + i * 10 for i in range(len(aaa))],
                "close": [10020 + i * 10 for i in range(len(aaa))],
                "value_vnd": [7e9] * len(aaa),
                "atr14": [120] * len(aaa),
                "ceiling_price": [12000] * len(aaa),
                "floor_price": [8000] * len(aaa),
            }
        )
        signal = pd.Series([1 if r.score > 0 else 0 for r in aaa], index=pd.to_datetime([r.as_of_date for r in aaa]))
        out = run_backtest_v3(
            bars,
            signal,
            load_market_rules("configs/market_rules_vn.yaml"),
            BacktestV3Config(initial_cash=300_000_000.0),
        )
        vals = np.array(list(out["metrics"].values()), dtype=float)
        assert np.isfinite(vals).all()


def test_listnet_cv_metrics_finite() -> None:
    raw = _toy_frame()
    prep, cols = prepare_listnet_training_frame(raw)
    cv = cross_validate_listnet(prep, cols, config=ListNetConfig(epochs=8))
    assert not cv.empty
    assert np.isfinite(cv["ndcg_at_30"].to_numpy(dtype=float)).all()
    assert np.isfinite(cv["rank_ic"].to_numpy(dtype=float)).all()
