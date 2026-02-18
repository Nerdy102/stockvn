from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class MarketProviderAdapter(ABC):
    @abstractmethod
    def iter_raw_events(self) -> Iterable[dict[str, Any]]:
        raise NotImplementedError
