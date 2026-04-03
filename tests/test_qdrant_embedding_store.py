from __future__ import annotations

import asyncio

from astrbot_plugin_group_digest.services.embedding_store.base import (
    SemanticUnitEmbeddingDocument,
    TopicSliceEmbeddingDocument,
)
from astrbot_plugin_group_digest.services.embedding_store.qdrant_store import QdrantEmbeddingStore


def _run(coro):
    return asyncio.run(coro)


def test_qdrant_store_degrades_when_url_missing() -> None:
    store = QdrantEmbeddingStore(
        enabled=True,
        qdrant_url="",
    )
    ok = _run(
        store.upsert_semantic_unit(
            SemanticUnitEmbeddingDocument(
                point_id="su_test",
                vector=[0.1, 0.2],
                payload={"object_type": "semantic_unit"},
            )
        )
    )
    assert ok is False
    assert store.enabled is False


def test_qdrant_store_upsert_and_payload_shape() -> None:
    store = QdrantEmbeddingStore(
        enabled=True,
        qdrant_url="http://qdrant.local:6333",
        semantic_unit_collection="su_col",
        topic_slice_collection="ts_col",
        vector_size=2,
    )
    calls: list[dict] = []

    async def _fake_request_json(*, method, path, body, allowed_statuses):
        calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "allowed_statuses": allowed_statuses,
            }
        )
        if method == "GET" and path.startswith("/collections/"):
            return 404, {}
        return 200, {"status": "ok"}

    store._request_json = _fake_request_json  # type: ignore[method-assign]

    ok_semantic = _run(
        store.upsert_semantic_unit(
            SemanticUnitEmbeddingDocument(
                point_id="su_point_1",
                vector=[0.3, 0.7],
                payload={
                    "object_type": "semantic_unit",
                    "group_id": "group_1001",
                    "date_label": "2026-03-22",
                    "topic_id": "20260322_0001",
                    "semantic_unit_id": "unit_123",
                    "embedding_model": "test-model",
                    "embedding_version": "v1",
                },
            )
        )
    )
    ok_slice = _run(
        store.upsert_topic_slice(
            TopicSliceEmbeddingDocument(
                point_id="ts_point_1",
                vector=[0.4, 0.6],
                payload={
                    "object_type": "topic_slice",
                    "group_id": "group_1001",
                    "date_label": "2026-03-22",
                    "topic_id": "20260322_0001",
                    "embedding_model": "test-model",
                    "embedding_version": "v1",
                },
            )
        )
    )

    assert ok_semantic is True
    assert ok_slice is True

    upsert_calls = [row for row in calls if row["path"].endswith("/points?wait=true")]
    assert len(upsert_calls) == 2
    semantic_body = upsert_calls[0]["body"]
    assert semantic_body["points"][0]["payload"]["object_type"] == "semantic_unit"
    assert semantic_body["points"][0]["payload"]["group_id"] == "group_1001"


def test_qdrant_store_minimal_query_hook() -> None:
    store = QdrantEmbeddingStore(
        enabled=True,
        qdrant_url="http://qdrant.local:6333",
        semantic_unit_collection="su_col",
        topic_slice_collection="ts_col",
        vector_size=2,
    )

    calls: list[dict] = []

    async def _fake_request_json(*, method, path, body, allowed_statuses):
        calls.append({"method": method, "path": path, "body": body, "allowed_statuses": allowed_statuses})
        if method == "GET" and path.startswith("/collections/"):
            return 200, {"status": "ok"}
        if path.endswith("/points/scroll"):
            return 200, {"result": {"points": [{"payload": {"group_id": "group_1001", "object_type": "topic_slice"}}]}}
        if path.endswith("/points/search"):
            return 200, {"result": [{"payload": {"group_id": "group_1001", "object_type": "topic_slice"}}]}
        return 200, {"status": "ok"}

    store._request_json = _fake_request_json  # type: ignore[method-assign]

    rows_semantic = _run(
        store.query_semantic_units(
            group_id="group_1001",
            date_label="2026-03-22",
            topic_id=None,
            limit=5,
        )
    )
    assert rows_semantic == [{"group_id": "group_1001", "object_type": "topic_slice"}]

    rows_slice = _run(
        store.query_topic_slices(
            group_id="group_1001",
            query_vector=[0.1, 0.2],
            recent_days=2,
            limit=3,
        )
    )
    assert rows_slice == [{"group_id": "group_1001", "object_type": "topic_slice"}]

    search_call = next(item for item in calls if item["path"].endswith("/points/search"))
    assert search_call["body"]["filter"]["must"][0]["key"] == "object_type"
    assert search_call["body"]["filter"]["must"][0]["match"]["value"] == "topic_slice"
    assert search_call["body"]["filter"]["must"][1]["key"] == "group_id"


def test_qdrant_store_topic_head_query_uses_topic_head_object_type() -> None:
    store = QdrantEmbeddingStore(
        enabled=True,
        qdrant_url="http://qdrant.local:6333",
        semantic_unit_collection="su_col",
        topic_head_collection="th_col",
        vector_size=2,
    )

    calls: list[dict] = []

    async def _fake_request_json(*, method, path, body, allowed_statuses):
        calls.append({"method": method, "path": path, "body": body, "allowed_statuses": allowed_statuses})
        if method == "GET" and path.startswith("/collections/"):
            return 200, {"status": "ok"}
        if path.endswith("/points/search"):
            return 200, {"result": [{"payload": {"group_id": "group_1001", "object_type": "topic_head"}}]}
        return 200, {"status": "ok"}

    store._request_json = _fake_request_json  # type: ignore[method-assign]

    rows = _run(
        store.query_topic_heads(
            group_id="group_1001",
            query_vector=[0.1, 0.2],
            recent_days=2,
            limit=3,
        )
    )

    assert rows == [{"group_id": "group_1001", "object_type": "topic_head"}]
    search_call = next(item for item in calls if item["path"].endswith("/points/search"))
    assert search_call["body"]["filter"]["must"][0]["key"] == "object_type"
    assert search_call["body"]["filter"]["must"][0]["match"]["value"] == "topic_head"
