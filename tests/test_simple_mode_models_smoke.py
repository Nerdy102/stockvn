from __future__ import annotations

from core.settings import get_settings
from core.simple_mode.models import run_signal
from data.providers.factory import get_provider


def test_model_1_2_3_signal_smoke() -> None:
    provider = get_provider(get_settings())
    df = provider.get_ohlcv("FPT", "1D")
    for model in ["model_1", "model_2", "model_3"]:
        out = run_signal(model, "FPT", "1D", df)
        assert out.symbol == "FPT"
        assert out.signal in {"TANG", "GIAM", "TRUNG_TINH", "UU_TIEN_QUAN_SAT"}
