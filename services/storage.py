from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from astrbot.api import logger

from .models import MessageRecord


class JsonMessageStorage:
    """消息事实存储（阶段一）：按群/按天 append-only JSONL。

    新写入路径：
    - {base_dir}/messages/{group_id}/{YYYY-MM-DD}.jsonl

    兼容策略：
    - 旧版 ``messages.json`` 仅作为历史只读回退来源。
    - 新消息不再回写旧文件，避免持续全量重写。
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.legacy_file_path = file_path
        self.messages_root_dir = self.file_path.parent / "messages"

        self._lock: asyncio.Lock | None = None
        self._file_lock = threading.RLock()

        self.messages_root_dir.mkdir(parents=True, exist_ok=True)
        if self.legacy_file_path.exists():
            logger.info(
                "[group_digest.storage] legacy_messages_json_detected file=%s",
                self.legacy_file_path,
            )

    async def append_message(self, record: MessageRecord) -> None:
        daily_file = self._resolve_daily_file_path(
            group_id=record.group_id,
            timestamp=record.timestamp,
        )
        payload = json.dumps(record.to_dict(), ensure_ascii=False)

        async with self._get_lock():
            with self._file_lock:
                daily_file.parent.mkdir(parents=True, exist_ok=True)
                with daily_file.open("a", encoding="utf-8") as fp:
                    fp.write(payload)
                    fp.write("\n")
                    fp.flush()
                    os.fsync(fp.fileno())

    def load_messages(
        self,
        group_id: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[MessageRecord]:
        with self._file_lock:
            rows = list(
                self._iter_jsonl_records(
                    group_id=group_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )
            legacy_rows = self._read_legacy_records(
                group_id=group_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            if legacy_rows:
                rows.extend(legacy_rows)
                rows = self._dedupe_records(rows)
            return rows

    def get_message_stats(
        self,
        *,
        group_id: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> tuple[int, int]:
        """返回指定窗口内的 (消息数, 最后一条消息时间戳)。"""
        rows = self.load_messages(group_id=group_id, start_ts=start_ts, end_ts=end_ts)
        count = len(rows)
        last_ts = max((row.timestamp for row in rows), default=0)
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

    def _iter_jsonl_records(
        self,
        *,
        group_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ):
        for file_path in self._iter_candidate_jsonl_files(
            group_id=group_id,
            start_ts=start_ts,
            end_ts=end_ts,
        ):
            yield from self._iter_jsonl_records_from_file(
                file_path=file_path,
                group_id=group_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )

    def _iter_candidate_jsonl_files(
        self,
        *,
        group_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ):
        date_labels = self._resolve_candidate_date_labels(start_ts=start_ts, end_ts=end_ts)

        if group_id is not None:
            group_dirs = [self.messages_root_dir / self._group_dir_name(group_id)]
        else:
            if not self.messages_root_dir.exists():
                return
            group_dirs = sorted(
                (path for path in self.messages_root_dir.iterdir() if path.is_dir()),
                key=lambda p: p.name,
            )

        for group_dir in group_dirs:
            if not group_dir.exists():
                continue
            if date_labels is None:
                for file_path in sorted(group_dir.glob("*.jsonl")):
                    if file_path.is_file():
                        yield file_path
                continue

            for date_label in date_labels:
                file_path = group_dir / f"{date_label}.jsonl"
                if file_path.is_file():
                    yield file_path

    def _iter_jsonl_records_from_file(
        self,
        *,
        file_path: Path,
        group_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ):
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                for line_no, line in enumerate(fp, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "[group_digest.storage] jsonl_decode_error file=%s line=%d error=%s",
                            file_path,
                            line_no,
                            exc,
                        )
                        continue
                    if not isinstance(payload, dict):
                        logger.warning(
                            "[group_digest.storage] jsonl_invalid_payload file=%s line=%d type=%s",
                            file_path,
                            line_no,
                            type(payload).__name__,
                        )
                        continue

                    row = MessageRecord.from_dict(payload)
                    if row is None:
                        continue
                    if group_id is not None and row.group_id != group_id:
                        continue
                    if start_ts is not None and row.timestamp < start_ts:
                        continue
                    if end_ts is not None and row.timestamp >= end_ts:
                        continue
                    yield row
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("[group_digest.storage] jsonl_read_failed file=%s error=%s", file_path, exc)

    def _read_legacy_records(
        self,
        *,
        group_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[MessageRecord]:
        legacy = self.legacy_file_path
        if not legacy.exists():
            return []
        if not legacy.is_file():
            return []
        if legacy.suffix.lower() != ".json":
            return []

        try:
            payload = legacy.read_text(encoding="utf-8")
            data = json.loads(payload)
            if not isinstance(data, list):
                logger.warning(
                    "[group_digest.storage] legacy_invalid_payload_type expected=list got=%s file=%s",
                    type(data).__name__,
                    legacy,
                )
                return []

            records: list[MessageRecord] = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    logger.warning(
                        "[group_digest.storage] legacy_skip_non_dict_record index=%d type=%s",
                        idx,
                        type(item).__name__,
                    )
                    continue
                row = MessageRecord.from_dict(item)
                if row is None:
                    continue
                if group_id is not None and row.group_id != group_id:
                    continue
                if start_ts is not None and row.timestamp < start_ts:
                    continue
                if end_ts is not None and row.timestamp >= end_ts:
                    continue
                records.append(row)
            return records
        except json.JSONDecodeError as exc:
            logger.warning(
                "[group_digest.storage] legacy_json_decode_error file=%s error=%s",
                legacy,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning("[group_digest.storage] legacy_read_failed file=%s error=%s", legacy, exc)
            return []

    def _resolve_candidate_date_labels(
        self,
        *,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[str] | None:
        if start_ts is None or end_ts is None:
            return None
        if end_ts <= start_ts:
            return []

        start_day = datetime.fromtimestamp(start_ts).date()
        end_day = datetime.fromtimestamp(end_ts - 1).date()
        labels: list[str] = []
        current: date = start_day
        while current <= end_day:
            labels.append(current.strftime("%Y-%m-%d"))
            current = current + timedelta(days=1)
        return labels

    def _resolve_daily_file_path(self, *, group_id: str, timestamp: int) -> Path:
        day_label = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        group_dir = self.messages_root_dir / self._group_dir_name(group_id)
        return group_dir / f"{day_label}.jsonl"

    def _group_dir_name(self, group_id: str) -> str:
        group = str(group_id or "").strip()
        if not group:
            return "unknown_group"

        safe = group.replace("/", "_")
        if os.sep != "/":
            safe = safe.replace(os.sep, "_")
        if os.altsep:
            safe = safe.replace(os.altsep, "_")
        return safe

    def _dedupe_records(self, rows: list[MessageRecord]) -> list[MessageRecord]:
        seen: set[tuple[object, ...]] = set()
        result: list[MessageRecord] = []
        for row in rows:
            message_id = str(row.message_id or "").strip()
            if message_id:
                key: tuple[object, ...] = ("id", row.group_id, message_id)
            else:
                key = (
                    "fallback",
                    row.group_id,
                    row.sender_id,
                    row.timestamp,
                    row.content,
                )
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

