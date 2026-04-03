from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from astrbot.api import logger

from .models import TopicHeadRecord, TopicSliceRecord


class TopicSliceStore:
    """按群/按天的 topic head 轻量存储（append-only JSONL）。"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._file_lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def append_head(self, record: TopicHeadRecord) -> None:
        file_path = self._resolve_daily_file_path(
            group_id=record.group_id,
            date_label=record.date_label,
        )
        payload = json.dumps(record.to_dict(), ensure_ascii=False)

        with self._file_lock:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", encoding="utf-8") as fp:
                fp.write(payload)
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())

    def append_slice(self, record: TopicSliceRecord) -> None:
        """兼容旧调用：append_slice 与 append_head 等价。"""
        self.append_head(record)

    def load_heads(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        limit: int | None = None,
    ) -> list[TopicHeadRecord]:
        with self._file_lock:
            rows = list(
                self._iter_head_records(
                    group_id=group_id,
                    date_label=date_label,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )
            rows.sort(key=lambda row: (row.end_ts, row.topic_id))
            if limit is not None and limit > 0:
                return rows[-limit:]
            return rows

    def load_slices(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        limit: int | None = None,
    ) -> list[TopicSliceRecord]:
        """兼容旧调用：load_slices 与 load_heads 等价。"""
        heads = self.load_heads(
            group_id=group_id,
            date_label=date_label,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
        )
        rows: list[TopicSliceRecord] = []
        for head in heads:
            item = TopicSliceRecord.from_dict(head.to_dict())
            if item is not None:
                rows.append(item)
        return rows

    def _iter_head_records(
        self,
        *,
        group_id: str,
        date_label: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ):
        for file_path in self._iter_candidate_files(
            group_id=group_id,
            date_label=date_label,
            start_ts=start_ts,
            end_ts=end_ts,
        ):
            yield from self._iter_head_records_from_file(
                file_path=file_path,
                group_id=group_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )

    def _iter_candidate_files(
        self,
        *,
        group_id: str,
        date_label: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ):
        group_dir = self.root_dir / self._group_dir_name(group_id)
        if not group_dir.exists():
            return

        if date_label:
            file_path = group_dir / f"{date_label}.jsonl"
            if file_path.is_file():
                yield file_path
            return

        date_labels = self._resolve_candidate_date_labels(start_ts=start_ts, end_ts=end_ts)
        if date_labels is None:
            for file_path in sorted(group_dir.glob("*.jsonl")):
                if file_path.is_file():
                    yield file_path
            return

        for label in date_labels:
            file_path = group_dir / f"{label}.jsonl"
            if file_path.is_file():
                yield file_path

    def _iter_head_records_from_file(
        self,
        *,
        file_path: Path,
        group_id: str,
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
                            "[group_digest.topic_slice_store] jsonl_decode_error file=%s line=%d error=%s",
                            file_path,
                            line_no,
                            exc,
                        )
                        continue

                    row = TopicHeadRecord.from_dict(payload)
                    if row is None:
                        continue
                    if row.group_id != group_id:
                        continue
                    if start_ts is not None and row.end_ts < start_ts:
                        continue
                    if end_ts is not None and row.start_ts >= end_ts:
                        continue
                    yield row
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning(
                "[group_digest.topic_slice_store] jsonl_read_failed file=%s error=%s",
                file_path,
                exc,
            )

    def _resolve_daily_file_path(self, *, group_id: str, date_label: str) -> Path:
        label = str(date_label or "").strip()
        if not label:
            label = datetime.now().strftime("%Y-%m-%d")
        group_dir = self.root_dir / self._group_dir_name(group_id)
        return group_dir / f"{label}.jsonl"

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
