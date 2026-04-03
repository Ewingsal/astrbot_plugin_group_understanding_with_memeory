from __future__ import annotations

from typing import Any

from .base import EmbeddingStore, SemanticUnitEmbeddingDocument, TopicHeadEmbeddingDocument


class NoopEmbeddingStore(EmbeddingStore):
    @property
    def enabled(self) -> bool:
        return False

    async def upsert_semantic_unit(self, doc: SemanticUnitEmbeddingDocument) -> bool:
        _ = doc
        return False

    async def upsert_topic_head(self, doc: TopicHeadEmbeddingDocument) -> bool:
        _ = doc
        return False

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
        _ = (group_id, date_label, topic_id, query_vector, start_ts, end_ts, recent_days, limit)
        return []

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
        _ = (group_id, date_label, topic_id, query_vector, start_ts, end_ts, recent_days, limit)
        return []

    async def upsert_topic_slice(self, doc: TopicHeadEmbeddingDocument) -> bool:
        _ = doc
        return False

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
        return await self.query_topic_heads(
            group_id=group_id,
            date_label=date_label,
            topic_id=topic_id,
            query_vector=query_vector,
            start_ts=start_ts,
            end_ts=end_ts,
            recent_days=recent_days,
            limit=limit,
        )
