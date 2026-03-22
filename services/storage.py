from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger

from .models import MessageRecord


class JsonMessageStorage:
    """本地 JSON 文件存储（MVP 可用版）。

    TODO: 下一阶段升级为按天分片或轻量数据库，降低全量读写成本。
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock: asyncio.Lock | None = None
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    async def append_message(self, record: MessageRecord) -> None:
        async with self._get_lock():
            records = self._read_records()
            records.append(record)
            self._write_records(records)

    def load_messages(
        self,
        group_id: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[MessageRecord]:
        rows = self._read_records()
        filtered: list[MessageRecord] = []

        for row in rows:
            if group_id is not None and row.group_id != group_id:
                continue
            if start_ts is not None and row.timestamp < start_ts:
                continue
            if end_ts is not None and row.timestamp >= end_ts:
                continue
            filtered.append(row)

        return filtered

    def get_message_stats(
        self,
        *,
        group_id: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> tuple[int, int]:
        """返回指定窗口内的 (消息数, 最后一条消息时间戳)。"""
        rows = self._read_records()
        count = 0
        last_ts = 0

        for row in rows:
            if group_id is not None and row.group_id != group_id:
                continue
            if start_ts is not None and row.timestamp < start_ts:
                continue
            if end_ts is not None and row.timestamp >= end_ts:
                continue

            count += 1
            if row.timestamp > last_ts:
                last_ts = row.timestamp

        return count, last_ts

    def load_yesterday_messages(self, group_id: str, now: datetime) -> list[MessageRecord]:
        start_ts, end_ts = self._yesterday_window(now=now)
        return self.load_messages(group_id=group_id, start_ts=start_ts, end_ts=end_ts)

    def load_today_messages(self, group_id: str, now: datetime) -> list[MessageRecord]:
        start_ts, end_ts = self._today_window(now=now)
        return self.load_messages(group_id=group_id, start_ts=start_ts, end_ts=end_ts)

    def _yesterday_window(self, now: datetime) -> tuple[int, int]:
        tzinfo = now.tzinfo
        today_start = datetime(now.year, now.month, now.day, tzinfo=tzinfo)
        yesterday_start = today_start - timedelta(days=1)
        return int(yesterday_start.timestamp()), int(today_start.timestamp())

    def _today_window(self, now: datetime) -> tuple[int, int]:
        tzinfo = now.tzinfo
        today_start = datetime(now.year, now.month, now.day, tzinfo=tzinfo)
        tomorrow_start = today_start + timedelta(days=1)
        return int(today_start.timestamp()), int(tomorrow_start.timestamp())

    def _read_records(self) -> list[MessageRecord]:
        try:
            payload = self.file_path.read_text(encoding="utf-8")
            data = json.loads(payload)
            if not isinstance(data, list):
                logger.warning("[group_digest.storage] invalid_payload_type expected=list got=%s", type(data).__name__)
                return []
            records: list[MessageRecord] = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    logger.warning(
                        "[group_digest.storage] skip_non_dict_record index=%d type=%s",
                        idx,
                        type(item).__name__,
                    )
                    continue
                row = MessageRecord.from_dict(item)
                if row is None:
                    logger.warning("[group_digest.storage] skip_invalid_record index=%d", idx)
                    continue
                records.append(row)
            return records
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            logger.warning("[group_digest.storage] json_decode_error file=%s error=%s", self.file_path, exc)
            return []
        except Exception as exc:
            logger.warning("[group_digest.storage] read_failed file=%s error=%s", self.file_path, exc)
            return []

    def _write_records(self, rows: list[MessageRecord]) -> None:
        payload = [row.to_dict() for row in rows]
        self.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
