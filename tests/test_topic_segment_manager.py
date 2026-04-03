from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from astrbot_plugin_group_digest.services.group_topic_segment_manager import (
    TOPIC_STATUS_ACTIVE,
    TOPIC_STATUS_CLOSED,
    GroupTopicSegmentManager,
)
from astrbot_plugin_group_digest.services.embedding_store.base import (
    SemanticUnitEmbeddingDocument,
    TopicSliceEmbeddingDocument,
)
from astrbot_plugin_group_digest.services.models import MessageRecord
from astrbot_plugin_group_digest.services.topic_slice_store import TopicSliceStore


class _KeywordEmbeddingBackend:
    async def embed_text(self, text: str) -> list[float] | None:
        lowered = str(text or "").lower()
        if "部署" in lowered or "上线" in lowered:
            return [0.0, 1.0]
        return [1.0, 0.0]


class _FailingEmbeddingBackend:
    async def embed_text(self, text: str) -> list[float] | None:
        _ = text
        raise RuntimeError("embedding backend unavailable")


class _AlternatingEmbeddingBackend:
    def __init__(self):
        self._call_index = 0

    async def embed_text(self, text: str) -> list[float] | None:
        _ = text
        self._call_index += 1
        if self._call_index % 2 == 1:
            return [1.0, 0.0]
        return [0.0, 1.0]


class _RecordingEmbeddingStore:
    def __init__(self):
        self.semantic_docs: list[SemanticUnitEmbeddingDocument] = []
        self.topic_slice_docs: list[TopicSliceEmbeddingDocument] = []

    @property
    def enabled(self) -> bool:
        return True

    async def upsert_semantic_unit(self, doc: SemanticUnitEmbeddingDocument) -> bool:
        self.semantic_docs.append(doc)
        return True

    async def upsert_topic_slice(self, doc: TopicSliceEmbeddingDocument) -> bool:
        self.topic_slice_docs.append(doc)
        return True

    async def query_semantic_units(self, **kwargs):
        _ = kwargs
        return []

    async def query_topic_slices(self, **kwargs):
        _ = kwargs
        return []


def _msg(
    *,
    group_id: str = "group_1001",
    sender_id: str = "u1",
    sender_name: str = "Alice",
    content: str,
    timestamp: int,
    message_id: str,
) -> MessageRecord:
    return MessageRecord(
        group_id=group_id,
        sender_id=sender_id,
        sender_name=sender_name,
        content=content,
        timestamp=timestamp,
        message_id=message_id,
    )


def _run(coro):
    return asyncio.run(coro)


def _new_manager(
    tmp_path: Path,
    *,
    enable_topic_embedding: bool = False,
    embedding_backend=None,
    embedding_store=None,
    new_topic_gap_seconds: int = 1800,
    topic_close_gap_seconds: int = 1200,
    single_message_topic_timeout_seconds: int = 900,
    transfer_similarity_threshold: float = 0.75,
    transfer_buffer_size: int = 3,
    closed_topic_prune_seconds: int = 3600,
) -> tuple[GroupTopicSegmentManager, TopicSliceStore]:
    store = TopicSliceStore(tmp_path / "topic_slices")
    manager = GroupTopicSegmentManager(
        topic_slice_store=store,
        enable_topic_embedding=enable_topic_embedding,
        embedding_backend=embedding_backend,
        embedding_store=embedding_store,
        embedding_model="test-embedding-model",
        embedding_version="v1",
        new_topic_gap_seconds=new_topic_gap_seconds,
        topic_close_gap_seconds=topic_close_gap_seconds,
        single_message_topic_timeout_seconds=single_message_topic_timeout_seconds,
        transfer_similarity_threshold=transfer_similarity_threshold,
        transfer_buffer_size=transfer_buffer_size,
        closed_topic_prune_seconds=closed_topic_prune_seconds,
    )
    return manager, store


def test_time_gap_starts_new_topic_without_multi_topic_scoring(tmp_path: Path) -> None:
    manager, _store = _new_manager(tmp_path, new_topic_gap_seconds=300, topic_close_gap_seconds=300)
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))
    _run(manager.ingest_message(_msg(content="晚上部署发布流程", timestamp=base_ts + 400, message_id="m3")))
    _run(manager.ingest_message(_msg(content="再确认上线时间", timestamp=base_ts + 420, message_id="m4")))

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    assert len(rows) == 2
    assert rows[0]["status"] == TOPIC_STATUS_CLOSED
    assert rows[1]["status"] == TOPIC_STATUS_ACTIVE


def test_single_message_topic_timeout_creates_closed_topic(tmp_path: Path) -> None:
    manager, store = _new_manager(tmp_path, single_message_topic_timeout_seconds=120)
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天晚上有空吗", timestamp=base_ts, message_id="m1")))
    summary = _run(
        manager.sweep_topics(
            now_ts=base_ts + 180,
            group_id="group_1001",
            date_label="2026-03-22",
        )
    )

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    slices = store.load_slices(group_id="group_1001", date_label="2026-03-22")
    assert summary["created_topics"] == 1
    assert len(rows) == 1
    assert rows[0]["status"] == TOPIC_STATUS_CLOSED
    assert len(slices) == 1
    assert slices[0].core_text


def test_topic_close_by_effective_message_gap(tmp_path: Path) -> None:
    manager, store = _new_manager(tmp_path, topic_close_gap_seconds=180)
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))

    summary = _run(
        manager.sweep_topics(
            now_ts=base_ts + 240,
            group_id="group_1001",
            date_label="2026-03-22",
        )
    )

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    slices = store.load_slices(group_id="group_1001", date_label="2026-03-22")
    assert summary["closed_transitions"] == 1
    assert summary["persisted_slices"] == 1
    assert rows[0]["status"] == TOPIC_STATUS_CLOSED
    assert len(slices) == 1


def test_transfer_buffer_promotes_new_topic_and_keeps_old_core_stable(tmp_path: Path) -> None:
    manager, _store = _new_manager(
        tmp_path,
        enable_topic_embedding=True,
        embedding_backend=_KeywordEmbeddingBackend(),
        transfer_similarity_threshold=0.8,
        transfer_buffer_size=2,
        topic_close_gap_seconds=3600,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    # 首个 semantic unit -> 初始 topic core
    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))
    # 第一个偏离 unit -> buffer(1)
    _run(manager.ingest_message(_msg(content="部署发布流水线", timestamp=base_ts + 40, message_id="m3")))
    _run(manager.ingest_message(_msg(content="晚上上线窗口", timestamp=base_ts + 60, message_id="m4")))
    # 第二个偏离 unit -> buffer(2) 触发 transfer
    _run(manager.ingest_message(_msg(content="继续部署灰度", timestamp=base_ts + 80, message_id="m5")))
    _run(manager.ingest_message(_msg(content="确认上线检查项", timestamp=base_ts + 100, message_id="m6")))

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    assert len(rows) == 2
    old_topic = rows[0]
    new_topic = rows[1]
    assert old_topic["status"] == TOPIC_STATUS_CLOSED
    assert new_topic["status"] == TOPIC_STATUS_ACTIVE
    assert "篮球" in old_topic["core_text"]
    assert "部署" in new_topic["core_text"] or "上线" in new_topic["core_text"]


def test_embedding_failure_degrades_to_time_only_without_crash(tmp_path: Path) -> None:
    manager, _store = _new_manager(
        tmp_path,
        enable_topic_embedding=True,
        embedding_backend=_FailingEmbeddingBackend(),
        topic_close_gap_seconds=3600,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    # 即使语义明显变化，也不应因 embedding 失败导致崩溃或异常分裂。
    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))
    _run(manager.ingest_message(_msg(content="部署发布流水线", timestamp=base_ts + 40, message_id="m3")))
    _run(manager.ingest_message(_msg(content="晚上上线窗口", timestamp=base_ts + 60, message_id="m4")))

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    assert len(rows) == 1
    assert rows[0]["status"] == TOPIC_STATUS_ACTIVE


def test_collect_slice_contexts_reads_new_closed_slices_after_sweep(tmp_path: Path) -> None:
    manager, _store = _new_manager(tmp_path, topic_close_gap_seconds=180)
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))
    _run(
        manager.sweep_topics(
            now_ts=base_ts + 260,
            group_id="group_1001",
            date_label="2026-03-22",
        )
    )

    contexts = manager.collect_slice_contexts(
        group_id="group_1001",
        date_label="2026-03-22",
        time_window="2026-03-22 00:00 - 2026-03-22 23:59",
        mode="today",
        limit=5,
    )
    assert len(contexts) == 1
    assert "topic_id=" in contexts[0]
    assert "core_text=" in contexts[0]


def test_semantic_unit_embedding_is_upserted_with_metadata(tmp_path: Path) -> None:
    embedding_store = _RecordingEmbeddingStore()
    manager, _store = _new_manager(
        tmp_path,
        enable_topic_embedding=True,
        embedding_backend=_KeywordEmbeddingBackend(),
        embedding_store=embedding_store,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))

    assert len(embedding_store.semantic_docs) == 1
    doc = embedding_store.semantic_docs[0]
    assert doc.point_id.startswith("su_")
    assert doc.payload["object_type"] == "semantic_unit"
    assert doc.payload["group_id"] == "group_1001"
    assert doc.payload["date_label"] == "2026-03-22"
    assert doc.payload["topic_id"]
    assert doc.payload["semantic_unit_id"].startswith("unit_")
    assert doc.payload["embedding_model"] == "test-embedding-model"
    assert doc.payload["embedding_version"] == "v1"


def test_topic_close_upserts_topic_slice_embedding_with_metadata(tmp_path: Path) -> None:
    embedding_store = _RecordingEmbeddingStore()
    manager, _store = _new_manager(
        tmp_path,
        enable_topic_embedding=True,
        embedding_backend=_KeywordEmbeddingBackend(),
        embedding_store=embedding_store,
        topic_close_gap_seconds=120,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))
    _run(manager.sweep_topics(now_ts=base_ts + 200, group_id="group_1001", date_label="2026-03-22"))

    assert len(embedding_store.topic_slice_docs) == 1
    doc = embedding_store.topic_slice_docs[0]
    assert doc.point_id.startswith("ts_")
    assert doc.payload["object_type"] == "topic_head"
    assert doc.payload["group_id"] == "group_1001"
    assert doc.payload["date_label"] == "2026-03-22"
    assert doc.payload["topic_id"]
    assert doc.payload["embedding_model"] == "test-embedding-model"
    assert doc.payload["embedding_version"] == "v1"


def test_topic_close_builds_head_embedding_from_semantic_unit_mean(tmp_path: Path) -> None:
    manager, store = _new_manager(
        tmp_path,
        enable_topic_embedding=True,
        embedding_backend=_AlternatingEmbeddingBackend(),
        transfer_similarity_threshold=0.0,
        topic_close_gap_seconds=120,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="第一段主题开场", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="第一段主题补充", timestamp=base_ts + 10, message_id="m2")))
    _run(manager.ingest_message(_msg(content="第二段仍属同主题", timestamp=base_ts + 20, message_id="m3")))
    _run(manager.ingest_message(_msg(content="第二段继续补充", timestamp=base_ts + 30, message_id="m4")))
    _run(manager.sweep_topics(now_ts=base_ts + 300, group_id="group_1001", date_label="2026-03-22"))

    heads = store.load_heads(group_id="group_1001", date_label="2026-03-22")
    assert len(heads) == 1
    head = heads[0]
    assert head.semantic_unit_ids
    assert len(head.semantic_unit_ids) == 2
    assert len(head.head_embedding) == 2
    # mean([1,0], [0,1]) -> [0.5, 0.5], 归一化后约 [0.7071, 0.7071]
    assert abs(head.head_embedding[0] - 0.7071) < 0.01
    assert abs(head.head_embedding[1] - 0.7071) < 0.01


def test_embedding_disabled_skips_embedding_store_without_crash(tmp_path: Path) -> None:
    embedding_store = _RecordingEmbeddingStore()
    manager, _store = _new_manager(
        tmp_path,
        enable_topic_embedding=False,
        embedding_backend=_KeywordEmbeddingBackend(),
        embedding_store=embedding_store,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())

    _run(manager.ingest_message(_msg(content="今天聊篮球战术", timestamp=base_ts, message_id="m1")))
    _run(manager.ingest_message(_msg(content="先看防守轮转", timestamp=base_ts + 20, message_id="m2")))

    rows = manager.get_day_topics_snapshot(group_id="group_1001", date_label="2026-03-22")
    assert len(rows) == 1
    assert rows[0]["status"] == TOPIC_STATUS_ACTIVE
    assert embedding_store.semantic_docs == []
