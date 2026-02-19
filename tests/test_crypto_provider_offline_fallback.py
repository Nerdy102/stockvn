from __future__ import annotations

from data.providers.crypto_public_ohlcv import CryptoPublicOHLCVProvider


def test_crypto_provider_offline_fallback() -> None:
    provider = CryptoPublicOHLCVProvider(exchange="binance_public")
    provider.api_base = "http://127.0.0.1:9/unreachable"
    df = provider.get_ohlcv("BTC", "1D")
    assert not df.empty
    assert "close" in df.columns
