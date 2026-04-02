from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from astrbot_plugin_group_digest.services.models import MessageRecord
from astrbot_plugin_group_digest.services.storage import JsonMessageStorage


def _run(coro):
    return asyncio.run(coro)


def _append(storage: JsonMessageStorage, record: MessageRecord) -> None:
    _run(storage.append_message(record))


def test_append_writes_jsonl_by_group_and_day(tmp_path: Path) -> None:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    record = MessageRecord(
        group_id="group_1001",
        sender_id="u1",
        sender_name="Alice",
        content="今天先复盘昨天的训练结论",
        timestamp=int(datetime(2026, 3, 22, 10, 0, 0).timestamp()),
        message_id="m1",
    )
    _append(storage, record)

    target = tmp_path / "messages" / "group_1001" / "2026-03-22.jsonl"
    assert target.exists()

    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["group_id"] == "group_1001"
    assert payload["content"] == "今天先复盘昨天的训练结论"
    assert payload["message_id"] == "m1"


def test_load_messages_supports_group_and_window(tmp_path: Path) -> None:
    storage = JsonMessageStorage(tmp_path / "messages.json")

    _append(
        storage,
        MessageRecord(
            group_id="group_a",
            sender_id="u1",
            sender_name="Alice",
            content="A群 3月21日",
            timestamp=int(datetime(2026, 3, 21, 23, 50, 0).timestamp()),
        ),
    )
    _append(
        storage,
        MessageRecord(
            group_id="group_a",
            sender_id="u2",
            sender_name="Bob",
            content="A群 3月22日 09:00",
            timestamp=int(datetime(2026, 3, 22, 9, 0, 0).timestamp()),
        ),
    )
    _append(
        storage,
        MessageRecord(
            group_id="group_a",
            sender_id="u3",
            sender_name="Carol",
            content="A群 3月22日 12:00",
            timestamp=int(datetime(2026, 3, 22, 12, 0, 0).timestamp()),
        ),
    )
    _append(
        storage,
        MessageRecord(
            group_id="group_b",
            sender_id="u9",
            sender_name="Dave",
            content="B群 3月22日 09:00",
            timestamp=int(datetime(2026, 3, 22, 9, 0, 0).timestamp()),
        ),
    )

    start_ts = int(datetime(2026, 3, 22, 0, 0, 0).timestamp())
    end_ts = int(datetime(2026, 3, 22, 11, 0, 0).timestamp())
    rows = storage.load_messages(group_id="group_a", start_ts=start_ts, end_ts=end_ts)

    assert [row.content for row in rows] == ["A群 3月22日 09:00"]


def test_legacy_messages_json_is_read_only_fallback(tmp_path: Path) -> None:
    legacy_file = tmp_path / "messages.json"
    legacy_payload = [
        {
            "group_id": "group_1001",
            "sender_id": "legacy_u1",
            "sender_name": "Legacy",
            "content": "旧版历史消息",
            "timestamp": int(datetime(2026, 3, 21, 10, 0, 0).timestamp()),
            "message_id": "legacy_1",
        }
    ]
    legacy_file.write_text(json.dumps(legacy_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    storage = JsonMessageStorage(legacy_file)
    initial = storage.load_messages(group_id="group_1001")
    assert len(initial) == 1
    assert initial[0].content == "旧版历史消息"

    _append(
        storage,
        MessageRecord(
            group_id="group_1001",
            sender_id="u2",
            sender_name="Alice",
            content="新写入走 JSONL",
            timestamp=int(datetime(2026, 3, 22, 9, 0, 0).timestamp()),
            message_id="new_1",
        ),
    )

    legacy_after = json.loads(legacy_file.read_text(encoding="utf-8"))
    assert len(legacy_after) == 1
    assert legacy_after[0]["message_id"] == "legacy_1"

    merged = storage.load_messages(group_id="group_1001")
    assert len(merged) == 2
    assert {row.message_id for row in merged} == {"legacy_1", "new_1"}

