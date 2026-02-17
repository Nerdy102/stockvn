from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _GroupState:
    last_seq: int = 0


@dataclass
class _StreamState:
    seq: int = 0
    entries: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    groups: dict[str, _GroupState] = field(default_factory=dict)


class FakeRedisCompat:
    def __init__(self, decode_responses: bool = True) -> None:
        self.decode_responses = decode_responses
        self._streams: dict[str, _StreamState] = {}

    def _stream(self, key: str) -> _StreamState:
        return self._streams.setdefault(key, _StreamState())

    def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):
        del approximate
        st = self._stream(stream)
        st.seq += 1
        msg_id = f"{st.seq}-0"
        st.entries.append((msg_id, {k: str(v) for k, v in fields.items()}))
        if maxlen is not None and len(st.entries) > maxlen:
            st.entries = st.entries[-maxlen:]
        return msg_id

    def xrange(self, stream: str):
        return list(self._stream(stream).entries)

    def xgroup_create(self, stream: str, groupname: str, id: str = "0", mkstream: bool = False):
        if mkstream:
            self._stream(stream)
        st = self._stream(stream)
        if groupname in st.groups:
            raise RuntimeError("BUSYGROUP Consumer Group name already exists")
        st.groups[groupname] = _GroupState(last_seq=int(id.split("-", 1)[0]) if id and id != "$" else 0)
        return True

    def xreadgroup(self, *, groupname: str, consumername: str, streams: dict[str, str], count: int, block: int | None = None):
        del consumername, block
        out = []
        remaining = count
        for stream, cursor in streams.items():
            if cursor != ">":
                continue
            st = self._stream(stream)
            grp = st.groups.setdefault(groupname, _GroupState())
            msgs = []
            for msg_id, fields in st.entries:
                seq = int(msg_id.split("-", 1)[0])
                if seq > grp.last_seq:
                    msgs.append((msg_id, fields))
                if len(msgs) >= remaining:
                    break
            if msgs:
                grp.last_seq = int(msgs[-1][0].split("-", 1)[0])
                out.append((stream, msgs))
                remaining -= len(msgs)
                if remaining <= 0:
                    break
        return out

    def xack(self, stream: str, groupname: str, msg_id: str):
        del stream, groupname, msg_id
        return 1


class FakeAsyncRedisCompat(FakeRedisCompat):
    async def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):
        return super().xadd(stream, fields, maxlen=maxlen, approximate=approximate)

    async def xrange(self, stream: str):
        return super().xrange(stream)
