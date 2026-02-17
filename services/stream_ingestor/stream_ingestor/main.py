from __future__ import annotations

import asyncio
import os
from pathlib import Path

from core.logging import setup_logging
from core.settings import get_settings
from data.providers.ssi_fastconnect.provider_stream import SsiStreamIngestor
from data.providers.ssi_fastconnect.token import SsiTokenManager


class _MockAsyncRedis:
    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}

    async def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True) -> str:
        del approximate
        rows = self._streams.setdefault(stream, [])
        msg_id = f"{len(rows)+1}-0"
        rows.append((msg_id, {k: str(v) for k, v in fields.items()}))
        if maxlen is not None and len(rows) > maxlen:
            self._streams[stream] = rows[-maxlen:]
        return msg_id


async def _run_mock(ingestor: SsiStreamIngestor, fixture_path: str) -> None:
    p = Path(fixture_path)
    if p.is_file():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        lines = []
        for f in sorted(p.glob("msg_*.json")):
            lines.append(f.read_text(encoding="utf-8").strip())
    for raw in lines:
        await ingestor._handle_ws_message(raw)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    token_manager = SsiTokenManager(base_url=settings.SSI_FCDATA_BASE_URL)
    ingestor = SsiStreamIngestor(
        stream_url=settings.SSI_STREAM_URL,
        redis_url=settings.REDIS_URL,
        token_manager=token_manager,
        subscribe_universe=os.getenv("SSI_STREAM_SUBSCRIBE_UNIVERSE", "VN30,VNINDEX"),
    )

    mock_path = os.getenv("SSI_STREAM_MOCK_MESSAGES_PATH", "").strip()
    if mock_path:
        if ingestor.redis_client is None:
            ingestor.redis_client = _MockAsyncRedis()
        asyncio.run(_run_mock(ingestor, mock_path))
        return

    asyncio.run(ingestor.run_forever())


if __name__ == "__main__":
    main()
