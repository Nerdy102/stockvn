import datetime as dt

from data.providers.csv_provider import CsvProvider


def test_intraday_generated_until_last_daily_date() -> None:
    p = CsvProvider("data_demo")
    symbol = p.get_tickers().iloc[0]["symbol"]
    daily = p.get_ohlcv(symbol, "1D")
    intraday = p.get_intraday(symbol, "15m")
    assert not intraday.empty
    assert intraday["timestamp"].max().date() == daily["date"].max()
    assert intraday["timestamp"].max().date() != dt.date.today()
