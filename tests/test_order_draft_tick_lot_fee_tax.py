from __future__ import annotations

from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.settings import get_settings
from core.simple_mode.models import run_signal
from core.simple_mode.orchestrator import generate_order_draft
from data.providers.factory import get_provider


def test_order_draft_tick_lot_fee_tax() -> None:
    settings = get_settings()
    provider = get_provider(settings)
    df = provider.get_ohlcv("FPT", "1D")
    signal = run_signal("model_1", "FPT", "1D", df)
    signal.proposed_side = "BUY"
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    draft = generate_order_draft(signal=signal, market_rules=mr, fees_taxes=fees)
    assert draft is not None
    assert draft.qty % 100 == 0
    assert mr.validate_tick(draft.price)
    assert draft.fee_tax.commission >= 0
    assert draft.fee_tax.sell_tax == 0
