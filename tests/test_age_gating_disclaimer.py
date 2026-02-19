from __future__ import annotations

from fastapi import HTTPException
import pytest

from core.simple_mode.safety import ensure_disclaimers, live_trading_enabled


def test_live_trading_default_disabled() -> None:
    assert live_trading_enabled() is False


def test_disclaimer_required() -> None:
    with pytest.raises(HTTPException):
        ensure_disclaimers(
            acknowledged_educational=False,
            acknowledged_loss=True,
            mode="paper",
            acknowledged_live_eligibility=False,
            age=30,
        )
