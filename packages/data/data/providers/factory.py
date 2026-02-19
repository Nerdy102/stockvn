from __future__ import annotations

from core.settings import Settings

from data.providers.base import BaseMarketDataProvider
from data.providers.crypto_public_ohlcv import CryptoPublicOHLCVProvider
from data.providers.csv_provider import CsvProvider
from data.providers.ssi_fastconnect import SsiFastConnectProvider
from data.providers.vietstock_datafeed import VietstockDataFeedProvider


def get_provider(settings: Settings) -> BaseMarketDataProvider:
    p = settings.DATA_PROVIDER.lower().strip()
    if p == "csv":
        return CsvProvider(settings.DEMO_DATA_DIR)
    if p == "ssi_fastconnect":
        return SsiFastConnectProvider()
    if p == "crypto_public":
        return CryptoPublicOHLCVProvider(exchange=settings.CRYPTO_DEFAULT_EXCHANGE)
    if p == "vietstock_datafeed":
        return VietstockDataFeedProvider()
    raise ValueError(f"Unknown DATA_PROVIDER: {settings.DATA_PROVIDER}")
