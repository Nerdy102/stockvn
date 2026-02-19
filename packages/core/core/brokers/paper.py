from __future__ import annotations

from core.brokers.sandbox import SandboxBroker


class PaperBroker(SandboxBroker):
    """Paper broker dùng simulator nội bộ, hoàn toàn offline."""
