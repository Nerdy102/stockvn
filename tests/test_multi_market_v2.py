from __future__ import annotations

from core.market_rules import load_market_rules
from data.adapters.market_mapping_v2 import map_provider_security_to_market


def test_tick_boundaries_per_market_and_instrument() -> None:
    rules = load_market_rules("configs/market_rules_vn.yaml")

    # HOSE stocks/funds tiers
    assert rules.get_tick_size(9_990, instrument="stock", exchange="HOSE") == 10
    assert rules.get_tick_size(10_000, instrument="stock", exchange="HOSE") == 50
    assert rules.get_tick_size(50_000, instrument="stock", exchange="HOSE") == 100

    # HNX/UPCOM config-driven fixed ticks
    assert rules.get_tick_size(12_345, instrument="stock", exchange="HNX") == 100
    assert rules.get_tick_size(12_345, instrument="stock", exchange="UPCOM") == 100

    # ETF/CW per-market
    assert rules.get_tick_size(20_000, instrument="etf", exchange="HOSE") == 10
    assert rules.get_tick_size(20_000, instrument="cw", exchange="HOSE") == 10
    assert rules.get_tick_size(20_000, instrument="etf", exchange="HNX") == 100


def test_provider_mapping_correctness() -> None:
    row_hose = {"Exchange": "HSX", "SecType": "ETF"}
    row_hnx = {"Market": "HNX", "SecType": "EQUITY"}
    row_upcom = {"market": "UPCoM", "instrument": "CW"}

    assert map_provider_security_to_market(row_hose) == {"exchange": "HOSE", "instrument": "etf"}
    assert map_provider_security_to_market(row_hnx) == {"exchange": "HNX", "instrument": "stock"}
    assert map_provider_security_to_market(row_upcom) == {"exchange": "UPCOM", "instrument": "cw"}
