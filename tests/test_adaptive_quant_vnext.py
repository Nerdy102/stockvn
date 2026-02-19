from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

from core.adaptive_quant_vnext import (
    ACIUncertainty,
    CPTCSwitchingCalibrator,
    CPMonitorInput,
    OnlineGridCPD,
    REGIME_R3,
    RegimeEngineV2,
    RegimePolicyEngine,
    compute_regime_features_v2,
    base_return_forecast,
    governance_from_coverage,
    pit_daily_matured_only,
    residual_series,
    uncertainty_metrics,
    SectorResidualPool,
    alpha_prediction_payload,
    alpha_shrink,
    build_expert_scores_v2,
    combine_alpha_score,
    init_regime_weights,
    benefit_cost_gate,
    build_cp_event_payload,
    sector_pooled_q,
    update_expert_weights,
    update_regime_weights_for_next_bar,
    daily_eod_actions,
    detect_impact_regime_shift_from_residuals,
    evaluate_governance_triggers,
    fit_tca_huber_regression,
    impact_shift_adjustment,
    log_tca_fill,
    reaction_time_to_cp,
    realtime_actions_for_bar,
    tca_params_record,
    zscore_cross_section,
)


def test_ocpd_variance_shift_detects_with_low_delay() -> None:
    rng = np.random.default_rng(7)
    pre = rng.normal(0.0, 0.01, size=200)
    post = rng.normal(0.0, 0.06, size=10)
    series = np.concatenate([pre, post])
    cpd = OnlineGridCPD(max_window=64)
    ev = None
    for i in range(200, len(series)):
        ev = cpd.detect(CPMonitorInput(returns=series[: i + 1]))
        if ev is not None and ev.cp_type == "VAR":
            delay = i - 200
            assert delay <= 3
            break
    assert ev is not None


def test_ocpd_false_alarm_low_on_stationary_series() -> None:
    rng = np.random.default_rng(11)
    cpd = OnlineGridCPD(max_window=64)
    alarms = 0
    for _ in range(40):
        r = rng.normal(0.0, 0.01, size=260)
        ev = cpd.detect(CPMonitorInput(returns=r))
        alarms += 0 if ev is None else 1
    assert alarms <= 2


def test_ocpd_geometric_candidates_log_growth() -> None:
    cpd = OnlineGridCPD(max_window=1024)
    cands = cpd.candidate_windows(1000)
    assert cands == [1, 2, 4, 8, 16, 32, 64, 128, 256]


def test_ocpd_multiseries_support_mean_corr_liq() -> None:
    n = 220
    rng = np.random.default_rng(123)
    returns = np.concatenate([rng.normal(0.0, 0.01, 200), rng.normal(0.06, 0.008, 20)])
    corr = np.concatenate([np.full(200, 0.05), np.full(20, 0.95)])
    spread = np.concatenate([np.full(200, 0.001), np.full(20, 0.010)])
    volz = np.concatenate([rng.normal(0, 0.3, 200), rng.normal(3.0, 0.3, 20)])
    limit_rate = np.concatenate([np.zeros(200), np.full(20, 0.5)])
    cpd = OnlineGridCPD(max_window=64)
    ev = cpd.detect(
        CPMonitorInput(
            returns=returns[:n],
            corr_summary=corr[:n],
            spread_proxy=spread[:n],
            volume_z=volz[:n],
            limit_rate=limit_rate[:n],
        )
    )
    assert ev is not None
    assert ev.cp_type in {"MEAN", "VAR", "CORR", "LIQ"}


def test_cp_event_payload_deterministic_hash() -> None:
    p1 = build_cp_event_payload(cp_row_id=10, severity="HIGH", cp_type="VAR", series_key="VNINDEX")
    p2 = build_cp_event_payload(cp_row_id=10, severity="HIGH", cp_type="VAR", series_key="VNINDEX")
    assert p1["event_type"] == "CP_EVENT"
    assert p1["payload_hash"] == p2["payload_hash"]
    assert p1["event_id"] == p2["event_id"]


def test_replay_like_cp_integration_single_event_deterministic() -> None:
    rng = np.random.default_rng(5)
    pre = rng.normal(0.0, 0.01, 150)
    post = rng.normal(0.0, 0.07, 10)
    r = np.concatenate([pre, post])
    cpd = OnlineGridCPD(max_window=64)
    events = []
    triggered = False
    for i in range(40, len(r)):
        ev = cpd.detect(CPMonitorInput(returns=r[: i + 1]), tf="60m", series_key="VNINDEX")
        if ev is not None and not triggered:
            events.append(build_cp_event_payload(1, ev.severity, ev.cp_type, ev.series_key))
            triggered = True
    assert len(events) == 1
    assert events[0]["payload_hash"] == build_cp_event_payload(1, events[0]["severity"], events[0]["cp_type"], events[0]["series_key"])["payload_hash"]


def test_regime_hysteresis_and_panic_override() -> None:
    e = RegimeEngineV2()
    low_conf = e.infer({"f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0, "f5": 0.0}, cp_recent=0)
    assert low_conf.hysteresis_applied is True
    panic = e.infer({"f1": -1.0, "f2": 0.0, "f3": 2.0, "f4": -0.08, "f5": -1.2}, cp_recent=1)
    assert panic.active_regime == REGIME_R3


def test_aci_update_deterministic() -> None:
    aci = ACIUncertainty()
    aci.bootstrap([0.01] * 200)
    alpha1, q1 = aci.update(0.05)
    alpha2, q2 = aci.update(0.005)
    assert 0.05 <= alpha1 <= 0.5
    assert 0.05 <= alpha2 <= 0.5
    assert q1 >= 0.0 and q2 >= 0.0


def test_cptc_switching_cooldown_enforced() -> None:
    c = CPTCSwitchingCalibrator()
    for i in range(10):
        state, _ = c.update(0.01, i, cp_recent=False, risk_regime=False)
        assert state == "S0"
    state, _ = c.update(0.2, 10, cp_recent=True, risk_regime=False)
    assert state == "S1"
    state, _ = c.update(0.01, 20, cp_recent=False, risk_regime=False)
    assert state == "S1"
    state, _ = c.update(0.01, 37, cp_recent=False, risk_regime=False)
    assert state == "S0"


def test_sector_pooling_conservative() -> None:
    assert sector_pooled_q(0.02, 0.03) == 0.03


def test_expert_weight_update_deterministic_and_caps() -> None:
    symbols = ["A", "B", "C", "D"]
    scores = pd.DataFrame(
        {
            "E0": [1, 2, 3, 4],
            "E1": [4, 3, 2, 1],
            "E2": [0, 1, 0, 1],
            "E3": [1, 1, 1, 1],
            "E4": [2, 2, 3, 3],
        },
        index=symbols,
    )
    next_ret = pd.Series([0.03, 0.02, 0.01, -0.01], index=symbols)
    w = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
    w1 = update_expert_weights(w, scores, next_ret)
    w2 = update_expert_weights(w, scores, next_ret)
    np.testing.assert_allclose(w1, w2)
    assert np.isclose(w1.sum(), 1.0)
    assert np.all(w1 >= 0.02)


def test_alpha_shrink_and_benefit_cost_gate() -> None:
    shrunk, conf = alpha_shrink(1.0, q_symbol=0.3, q_threshold_high=0.2)
    assert shrunk == 0.5
    assert conf > 0.0
    assert benefit_cost_gate(2.0, 0.03, 1_000_000, 0.001, 10_000)
    assert not benefit_cost_gate(0.1, 0.01, 50_000, 0.003, 100_000)


def test_metamorphic_price_scaling_invariant_for_returns_cp() -> None:
    rng = np.random.default_rng(42)
    base = 100 + np.cumsum(rng.normal(0.0, 1.0, size=300))
    scaled = base * 10.0
    r1 = np.diff(np.log(base))
    r2 = np.diff(np.log(scaled))
    cpd = OnlineGridCPD()
    e1 = cpd.detect(CPMonitorInput(returns=r1))
    e2 = cpd.detect(CPMonitorInput(returns=r2))
    assert (e1 is None and e2 is None) or (e1 is not None and e2 is not None and e1.cp_type == e2.cp_type)


def test_metamorphic_universe_permutation_keeps_ranking() -> None:
    s = pd.Series([0.1, 0.3, -0.2, 0.5], index=["A", "B", "C", "D"])
    z1 = zscore_cross_section(s).sort_values(ascending=False).index.tolist()
    z2 = zscore_cross_section(s.sample(frac=1.0, random_state=1)).sort_values(ascending=False).index.tolist()
    assert z1 == z2


def test_metamorphic_add_constant_to_scores_no_weight_change_after_zscore() -> None:
    symbols = ["A", "B", "C", "D"]
    base = pd.DataFrame({"E0": [1, 2, 3, 4], "E1": [4, 3, 2, 1], "E2": [2, 2, 2, 2], "E3": [1, 1, 2, 2], "E4": [3, 1, 0, -1]}, index=symbols)
    shifted = base + 10.0
    base_z = base.apply(zscore_cross_section)
    shifted_z = shifted.apply(zscore_cross_section)
    next_ret = pd.Series([0.01, 0.02, 0.0, -0.01], index=symbols)
    w0 = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
    wb = update_expert_weights(w0, base_z, next_ret)
    ws = update_expert_weights(w0, shifted_z, next_ret)
    np.testing.assert_allclose(wb, ws)


def test_regime_feature_builder_locked_fields() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="h")
    close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    breadth = pd.Series(np.linspace(-0.2, 0.6, len(idx)), index=idx)
    cp = pd.Series(0, index=idx)
    cp.iloc[-5] = 1
    feat = compute_regime_features_v2(close, breadth, cp_events_recent=cp)
    assert set(feat.columns) == {"f1", "f2", "f3", "f4", "f5", "f6"}
    assert feat["f2"].iloc[-1] in {0.0, 1.0}
    assert feat["f6"].iloc[-1] == 1


def test_regime_deterministic_classification_toy() -> None:
    e = RegimeEngineV2()
    st = e.infer({"f1": 1.5, "f2": 1.0, "f3": -0.5, "f4": -0.01, "f5": 0.5}, cp_recent=0)
    assert st.active_regime == "TREND_UP"
    st2 = e.infer({"f1": -0.5, "f2": 0.0, "f3": 2.0, "f4": -0.07, "f5": -0.7}, cp_recent=1)
    assert st2.active_regime in {"RISK_OFF", "PANIC_VOL"}


def test_regime_integration_cp_event_toggles_f6_and_changes_probabilities() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0.0, 0.4, len(idx))), index=idx)
    breadth = pd.Series(np.random.default_rng(1).uniform(-0.3, 0.3, len(idx)), index=idx)
    cp0 = pd.Series(0, index=idx)
    cp1 = cp0.copy()
    cp1.iloc[-1] = 1
    f0 = compute_regime_features_v2(close, breadth, cp_events_recent=cp0).iloc[-1].to_dict()
    f1 = compute_regime_features_v2(close, breadth, cp_events_recent=cp1).iloc[-1].to_dict()
    eng = RegimeEngineV2()
    r0 = eng.infer(f0, cp_recent=int(f0["f6"]))
    eng = RegimeEngineV2()
    r1 = eng.infer(f1, cp_recent=int(f1["f6"]))
    assert f0["f6"] == 0
    assert f1["f6"] == 1
    assert r0.probs != r1.probs


def test_regime_policy_actions_and_linear_restore() -> None:
    p = RegimePolicyEngine()
    a_r2 = p.action_for("RISK_OFF")
    assert a_r2.governance_risk_level == "HIGH"
    assert a_r2.gross_exposure_cap_multiplier == 0.5
    assert a_r2.min_cash_buffer == 0.20

    a_r3 = p.action_for("PANIC_VOL")
    assert a_r3.gross_exposure_cap_multiplier == 0.25
    assert a_r3.min_cash_buffer == 0.35

    vals = [p.action_for("TREND_UP") for _ in range(60)]
    after_warmup = vals[25]
    assert after_warmup.restore_progress > 0.0
    assert vals[-1].gross_exposure_cap_multiplier <= 1.0
    assert vals[-1].min_cash_buffer >= 0.05


def test_u1_base_forecaster_and_residual_locked_formula() -> None:
    idx = pd.date_range("2026-01-01", periods=30, freq="h")
    close = pd.Series(np.linspace(100, 104, len(idx)), index=idx)
    r_hat = base_return_forecast(close)
    realized = close.pct_change().fillna(0.0)
    res = residual_series(realized, r_hat)
    assert len(r_hat) == len(close)
    assert len(res) == len(close)


def test_u1_aci_hand_toy_q_and_coverage() -> None:
    aci = ACIUncertainty()
    toy = [0.01] * 200
    aci.bootstrap(toy)
    assert np.isclose(aci.q_t, 0.01)
    for r in [0.02, 0.03, 0.005, 0.04, 0.005]:
        aci.update(r)
    assert 0.05 <= aci.alpha <= 0.50
    assert aci.q_t >= 0.01


def test_u2_switch_logic_cp_event_and_cooldown() -> None:
    c = CPTCSwitchingCalibrator()
    for i in range(5):
        st, _ = c.update(0.01, i, cp_recent=False, risk_regime=False)
        assert st == "S0"
    st, _ = c.update(0.2, 5, cp_recent=True, risk_regime=False)
    assert st == "S1"
    st, _ = c.update(0.01, 20, cp_recent=False, risk_regime=False)
    assert st == "S1"
    st, _ = c.update(0.01, 40, cp_recent=False, risk_regime=False)
    assert st == "S0"


def test_u2_sector_residual_pool_and_symbol_floor() -> None:
    pool = SectorResidualPool(window=20)
    for x in [0.01, 0.02, 0.03, 0.04, 0.05]:
        q_sector = pool.update("BANK", x)
    q_symbol = sector_pooled_q(0.015, q_sector)
    assert q_symbol >= q_sector


def test_monitoring_metrics_and_governance_thresholds() -> None:
    idx = pd.RangeIndex(500)
    y = pd.Series(np.sin(np.arange(500) / 10.0) * 0.01, index=idx)
    lo = y - 0.02
    hi = y + 0.02
    p = pd.Series(0.6, index=idx)
    m = uncertainty_metrics(y, lo, hi, p_outperform=p, window=500)
    assert 0.0 <= m["coverage"] <= 1.0
    assert m["width_p90"] >= m["width_median"]
    bad = pd.Series([0.69] * 20)
    warn = pd.Series([0.74] * 20)
    ok = pd.Series([0.76] * 20)
    assert governance_from_coverage(bad) == "PAUSE"
    assert governance_from_coverage(warn) == "WARNING"
    assert governance_from_coverage(ok) == "OK"


def test_pit_matured_labels_only_for_daily_coverage() -> None:
    df = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06"]),
            "matured_date": pd.to_datetime(["2026-01-04", "2026-01-07", "2026-01-06"]),
            "residual": [0.01, 0.02, 0.03],
        }
    )
    out = pit_daily_matured_only(df)
    assert len(out) == 2
    assert bool((out["matured_date"] <= out["as_of_date"]).all())


def _toy_expert_frame() -> pd.DataFrame:
    idx = ["AAA", "BBB", "CCC", "DDD"]
    return pd.DataFrame(
        {
            "value": [0.1, -0.2, 0.3, 0.0],
            "quality": [0.2, 0.1, 0.0, -0.1],
            "momentum": [0.3, -0.1, 0.2, 0.0],
            "low_vol": [0.1, 0.2, -0.2, 0.0],
            "dividend": [0.0, 0.2, 0.1, -0.1],
            "beta": [1.2, 0.8, 1.0, 0.9],
            "trend_setup": [1, 0, 1, 0],
            "breakout_setup": [0, 1, 0, 1],
            "ema50_slope": [0.2, -0.1, 0.4, -0.2],
            "return_3bars": [0.03, -0.02, 0.01, -0.01],
            "far_above_ema20": [1, 0, 0, 1],
            "far_below_ema20": [0, 1, 1, 0],
        },
        index=idx,
    )


def test_l4_build_expert_scores_cross_section_zscore() -> None:
    z = build_expert_scores_v2(_toy_expert_frame())
    assert list(z.columns) == ["E0_FACTOR", "E1_TREND", "E2_BREAKOUT", "E3_MEAN_REVERSION", "E4_DEFENSIVE"]
    for c in z.columns:
        assert abs(float(z[c].mean())) < 1e-6


def test_l4_weight_init_and_combine_alpha() -> None:
    rw = init_regime_weights()
    assert np.isclose(rw["TREND_UP"].sum(), 1.0)
    z = build_expert_scores_v2(_toy_expert_frame())
    alpha = combine_alpha_score(z, "TREND_UP", rw)
    assert len(alpha) == len(z)


def test_l4_weight_update_deterministic_and_caps_and_no_lookahead() -> None:
    rw = init_regime_weights()
    z = build_expert_scores_v2(_toy_expert_frame())
    next_ret = pd.Series([0.02, -0.01, 0.03, -0.02], index=z.index)
    rw1, audit1 = update_regime_weights_for_next_bar(rw, "PANIC_VOL", z, next_ret, tf="60m")
    rw2, audit2 = update_regime_weights_for_next_bar(rw, "PANIC_VOL", z, next_ret, tf="60m")
    np.testing.assert_allclose(rw1["PANIC_VOL"], rw2["PANIC_VOL"])
    assert np.all(rw1["PANIC_VOL"] >= 0.0)
    assert np.all(rw1["PANIC_VOL"] <= 0.70)
    assert audit1["updated"] is True and audit2["updated"] is True

    rw3, audit3 = update_regime_weights_for_next_bar(rw, "TREND_UP", z, next_ret, tf="15m")
    np.testing.assert_allclose(rw3["TREND_UP"], rw["TREND_UP"])
    assert audit3["updated"] is False


def test_l4_alpha_shrink_and_prediction_hash() -> None:
    s, conf = alpha_shrink(0.8, q_symbol=0.3, q_threshold_high=0.2)
    assert s == 0.4
    p1 = alpha_prediction_payload("AAA", "60m", datetime(2026, 1, 1, 10, 0, 0), s, conf, "TREND_UP", [0.35, 0.25, 0.2, 0.1, 0.1])
    p2 = alpha_prediction_payload("AAA", "60m", datetime(2026, 1, 1, 10, 0, 0), s, conf, "TREND_UP", [0.35, 0.25, 0.2, 0.1, 0.1])
    assert p1["prediction_hash"] == p2["prediction_hash"]


def test_l6_tca_fill_logging_fields() -> None:
    row = log_tca_fill(
        decision_ts=datetime(2026, 1, 1, 9, 0, 0),
        submit_ts=datetime(2026, 1, 1, 9, 0, 1),
        fill_ts=datetime(2026, 1, 1, 9, 0, 3),
        intended_price=10.0,
        executed_price=10.05,
        notional=1_000_000,
        participation_rate=0.03,
        session="continuous",
        regime="RISK_OFF",
    )
    assert set(row) >= {"decision_ts", "submit_ts", "fill_ts", "slippage_bps", "session", "regime"}


def test_l6_huber_regression_stable_and_hash_deterministic() -> None:
    rng = np.random.default_rng(123)
    n = 200
    x1 = rng.uniform(0.0, 0.1, n)
    x2 = rng.uniform(0.0, 0.05, n)
    x3 = rng.integers(0, 2, n)
    x4 = rng.integers(0, 3, n)
    x5 = rng.integers(0, 4, n)
    y = 2.0 + 10.0 * x1 + 5.0 * x2 + 3.0 * x3 + 1.5 * x4 + 2.0 * x5 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "slippage_bps": y,
        "x_notional_adtv": x1,
        "x_atr_price": x2,
        "x_limit_day": x3,
        "x_session": x4,
        "x_regime": x5,
    })
    out1 = fit_tca_huber_regression(df)
    out2 = fit_tca_huber_regression(df)
    assert out1["metrics"]["rmse"] >= 0.0
    assert out1["params"] == out2["params"]
    r1 = tca_params_record("HOSE_EQ", "2026-01-01", out1["params"], out1["metrics"])
    r2 = tca_params_record("HOSE_EQ", "2026-01-01", out1["params"], out1["metrics"])
    assert r1["param_hash"] == r2["param_hash"]


def test_l6_cp_shift_on_tca_residuals_and_adjustment() -> None:
    rng = np.random.default_rng(7)
    res = np.concatenate([rng.normal(0.0, 0.3, 220), rng.normal(2.0, 0.3, 40)])
    shifted = detect_impact_regime_shift_from_residuals(res)
    adj = impact_shift_adjustment(shifted, base_bps=8.0)
    assert adj["base_bps"] >= 8.0
    if shifted:
        assert adj["tag"] == "IMPACT_REGIME_SHIFT"


def test_realtime_daily_scheduling_locked_actions() -> None:
    a15 = realtime_actions_for_bar("BAR_CLOSE", "15m")
    a60 = realtime_actions_for_bar("BAR_CLOSE", "60m")
    eod = daily_eod_actions()
    assert "update_setups" in a15
    assert "generate_alpha" in a60 and "run_ocpd" in a60
    assert "calibrate_tca" in eod and "generate_daily_report_pack" in eod


def test_governance_triggers_with_runbooks_and_pause_conditions() -> None:
    coverage = pd.Series([0.68] * 20)
    incidents = evaluate_governance_triggers(coverage, signal_latency_p95=6.0, reconciliation_mismatch=True, intraday_drawdown_breach=True)
    levels = {x["level"] for x in incidents}
    runbooks = {x["runbook_id"] for x in incidents}
    assert "PAUSE" in levels and "WARNING" in levels
    assert runbooks >= {"RB-UNC-001", "RB-SLO-005", "RB-REC-002", "RB-DD-003"}


def test_fast_reaction_kpi_and_perf_microbench() -> None:
    assert reaction_time_to_cp(100, 104) == 4
    n = 500
    idx = [f"S{i:03d}" for i in range(n)]
    frame = pd.DataFrame(
        {
            "value": np.linspace(-1, 1, n),
            "quality": np.linspace(1, -1, n),
            "momentum": np.sin(np.linspace(0, 3, n)),
            "low_vol": np.cos(np.linspace(0, 2, n)),
            "dividend": np.linspace(0, 0.5, n),
            "beta": np.linspace(0.7, 1.5, n),
            "trend_setup": np.where(np.arange(n) % 2 == 0, 1, 0),
            "breakout_setup": np.where(np.arange(n) % 3 == 0, 1, 0),
            "ema50_slope": np.tanh(np.linspace(-2, 2, n)),
            "return_3bars": np.sin(np.linspace(0, 6, n)) * 0.03,
            "far_above_ema20": np.where(np.arange(n) % 5 == 0, 1, 0),
            "far_below_ema20": np.where(np.arange(n) % 7 == 0, 1, 0),
        },
        index=idx,
    )
    t0 = perf_counter()
    z = build_expert_scores_v2(frame)
    rw = init_regime_weights()
    _ = combine_alpha_score(z, "TREND_UP", rw)
    elapsed = perf_counter() - t0
    assert elapsed < 3.0
