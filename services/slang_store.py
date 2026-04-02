from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from astrbot.api import logger

from .models import SlangExplanationRecord


class SlangStore:
    """轻量黑话解释存储（按群 JSONL，append-only）。"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._file_lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def upsert(self, record: SlangExplanationRecord) -> None:
        group_id = str(record.group_id or "").strip()
        term = str(record.slang_term or "").strip()
        if not group_id or not term:
            return

        file_path = self._resolve_group_file_path(group_id=group_id)
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._file_lock:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", encoding="utf-8") as fp:
                fp.write(payload)
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())

    def get(self, *, group_id: str, slang_term: str) -> SlangExplanationRecord | None:
        term = str(slang_term or "").strip()
        if not term:
            return None
        records = self.list_group_records(group_id=group_id)
        return records.get(term)

    def list_group_records(
        self,
        *,
        group_id: str,
        limit: int | None = None,
    ) -> dict[str, SlangExplanationRecord]:
        file_path = self._resolve_group_file_path(group_id=group_id)
        if not file_path.exists():
            return {}

        with self._file_lock:
            rows = self._read_group_records_unlocked(file_path=file_path)

        latest: dict[str, SlangExplanationRecord] = {}
        for row in rows:
            term = str(row.slang_term or "").strip()
            if not term:
                continue
            previous = latest.get(term)
            if previous is None:
                latest[term] = row
                continue
            if int(row.updated_at or 0) >= int(previous.updated_at or 0):
                latest[term] = row

        if limit is not None and limit > 0:
            sorted_rows = sorted(
                latest.values(),
                key=lambda item: (int(item.updated_at or 0), item.slang_term),
                reverse=True,
            )[:limit]
            return {item.slang_term: item for item in sorted_rows}
        return latest

    def find_relevant(
        self,
        *,
        group_id: str,
        text: str,
        limit: int = 5,
    ) -> list[SlangExplanationRecord]:
        normalized = str(text or "").strip()
        if not normalized:
            return []

        mapping = self.list_group_records(group_id=group_id)
        if not mapping:
            return []

        hits: list[SlangExplanationRecord] = []
        for term, row in mapping.items():
            if term and term in normalized:
                hits.append(row)
        hits.sort(key=lambda item: (float(item.confidence), int(item.updated_at or 0)), reverse=True)
        return hits[: max(1, int(limit))]

    def _read_group_records_unlocked(self, *, file_path: Path) -> list[SlangExplanationRecord]:
        records: list[SlangExplanationRecord] = []
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
                            "[group_digest.slang_store] jsonl_decode_error file=%s line=%d error=%s",
                            file_path,
                            line_no,
                            exc,
                        )
                        continue
                    row = SlangExplanationRecord.from_dict(payload)
                    if row is None:
                        continue
                    records.append(row)
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.warning(
                "[group_digest.slang_store] jsonl_read_failed file=%s error=%s",
                file_path,
                exc,
            )
            return []
        return records

    def _resolve_group_file_path(self, *, group_id: str) -> Path:
        safe_group = self._group_dir_name(group_id)
        return self.root_dir / safe_group / "slang.jsonl"

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

