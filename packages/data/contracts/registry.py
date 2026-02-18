from __future__ import annotations

from dataclasses import dataclass

from .models import CanonicalBarV1, MarketEventV1, ProviderSnapshot


@dataclass(frozen=True)
class SchemaRef:
    name: str
    version: str
    cls: type[object]


class SchemaRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], type[object]] = {}

    def register(self, name: str, version: str, cls: type[object]) -> None:
        key = (name, version)
        if key in self._entries and self._entries[key] is not cls:
            raise ValueError(f"schema already registered with different class: {name}@{version}")
        self._entries[key] = cls

    def resolve(self, name: str, version: str) -> type[object]:
        key = (name, version)
        if key not in self._entries:
            raise KeyError(f"schema not found: {name}@{version}")
        return self._entries[key]

    def versions(self, name: str) -> list[str]:
        versions = [version for (schema_name, version) in self._entries if schema_name == name]
        return sorted(versions)


def build_default_registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register("MarketEvent", "v1", MarketEventV1)
    registry.register("CanonicalBar", "v1", CanonicalBarV1)
    registry.register("ProviderSnapshot", "v1", ProviderSnapshot)
    return registry
