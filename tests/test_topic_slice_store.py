from __future__ import annotations

from datetime import datetime
from pathlib import Path

from astrbot_plugin_group_digest.services.models import TopicSliceRecord
from astrbot_plugin_group_digest.services.topic_slice_store import TopicSliceStore


def test_topic_slice_store_append_and_load(tmp_path: Path) -> None:
    store = TopicSliceStore(tmp_path / "topic_slices")
    row = TopicSliceRecord(
        group_id="group_1001",
        date_label="2026-03-22",
        topic_id="20260322_0001",
        start_ts=int(datetime(2026, 3, 22, 9, 0, 0).timestamp()),
        end_ts=int(datetime(2026, 3, 22, 9, 20, 0).timestamp()),
        message_count=6,
        participants=["Alice(u1)", "Bob(u2)"],
        recent_keywords=["训练计划", "复盘"],
        first_message_id="m_1",
        last_message_id="m_6",
    )
    store.append_slice(row)

    rows = store.load_slices(group_id="group_1001", date_label="2026-03-22")
    assert len(rows) == 1
    loaded = rows[0]
    assert loaded.topic_id == "20260322_0001"
    assert loaded.message_count == 6
    assert loaded.participants == ["Alice(u1)", "Bob(u2)"]
    assert loaded.recent_keywords == ["训练计划", "复盘"]


def test_topic_slice_store_load_with_limit_returns_latest(tmp_path: Path) -> None:
    store = TopicSliceStore(tmp_path / "topic_slices")
    store.append_slice(
        TopicSliceRecord(
            group_id="group_1001",
            date_label="2026-03-22",
            topic_id="20260322_0001",
            start_ts=int(datetime(2026, 3, 22, 8, 0, 0).timestamp()),
            end_ts=int(datetime(2026, 3, 22, 8, 20, 0).timestamp()),
            message_count=3,
        )
    )
    store.append_slice(
        TopicSliceRecord(
            group_id="group_1001",
            date_label="2026-03-22",
            topic_id="20260322_0002",
            start_ts=int(datetime(2026, 3, 22, 9, 0, 0).timestamp()),
            end_ts=int(datetime(2026, 3, 22, 9, 20, 0).timestamp()),
            message_count=5,
        )
    )

    rows = store.load_slices(group_id="group_1001", date_label="2026-03-22", limit=1)
    assert len(rows) == 1
    assert rows[0].topic_id == "20260322_0002"
