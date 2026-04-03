from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SemanticUnitEmbeddingDocument:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopicHeadEmbeddingDocument:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopicSliceEmbeddingDocument(TopicHeadEmbeddingDocument):
    """兼容旧命名：TopicSliceEmbeddingDocument 等同于 TopicHeadEmbeddingDocument。"""


class EmbeddingStore(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    async def upsert_semantic_unit(self, doc: SemanticUnitEmbeddingDocument) -> bool:
        ...

    async def upsert_topic_head(self, doc: TopicHeadEmbeddingDocument) -> bool:
        ...

    async def query_semantic_units(
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

    async def query_topic_heads(
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

    # 兼容旧接口：历史上叫 topic_slice。
    async def upsert_topic_slice(self, doc: TopicHeadEmbeddingDocument) -> bool:
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
