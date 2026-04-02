from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SemanticUnitEmbeddingDocument:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopicSliceEmbeddingDocument:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


class EmbeddingStore(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    async def upsert_semantic_unit(self, doc: SemanticUnitEmbeddingDocument) -> bool:
        ...

    async def upsert_topic_slice(self, doc: TopicSliceEmbeddingDocument) -> bool:
        ...

    async def query_semantic_units(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        topic_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        ...

    async def query_topic_slices(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        topic_id: str | None = None,
        query_vector: list[float] | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        recent_days: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        ...
