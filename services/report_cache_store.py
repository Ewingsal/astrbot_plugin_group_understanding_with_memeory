from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger


@dataclass
class ReportCacheRecord:
    group_id: str
    date: str
    mode: str
    window_start: int
    window_end: int
    generated_at: str
    last_message_timestamp: int
    message_count: int
    provider_id: str
    analysis_provider_notice: str
    max_messages_for_analysis: int
    prompt_signature: str
    cache_version: int
    source: str
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportCacheRecord":
        report_payload = data.get("report", {})
        if not isinstance(report_payload, dict):
            logger.warning(
                "[group_digest.cache_store] invalid_report_payload_type expected=dict got=%s",
                type(report_payload).__name__,
            )
            report_payload = {}

        return cls(
            group_id=str(data.get("group_id", "")),
            date=str(data.get("date", "")),
            mode=str(data.get("mode", "")),
            window_start=cls._safe_int(data.get("window_start", 0), field="window_start"),
            window_end=cls._safe_int(data.get("window_end", 0), field="window_end"),
            generated_at=str(data.get("generated_at", "")),
            last_message_timestamp=cls._safe_int(
                data.get("last_message_timestamp", 0),
                field="last_message_timestamp",
            ),
            message_count=cls._safe_int(data.get("message_count", 0), field="message_count"),
            provider_id=str(data.get("provider_id", "")),
            analysis_provider_notice=str(data.get("analysis_provider_notice", "")),
            max_messages_for_analysis=cls._safe_int(
                data.get("max_messages_for_analysis", 0),
                field="max_messages_for_analysis",
            ),
            prompt_signature=str(data.get("prompt_signature", "")),
            cache_version=cls._safe_int(data.get("cache_version", 0), field="cache_version"),
            source=str(data.get("source", "")),
            report=report_payload,
        )

    @staticmethod
    def _safe_int(value: object, *, field: str, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.cache_store] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default


class ReportCacheStore:
    def __init__(self, file_path: Path, cache_version: int = 1):
        self.file_path = file_path
        self.cache_version = int(cache_version)
        self._lock: asyncio.Lock | None = None
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_raw({"cache_version": self.cache_version, "entries": {}})

    def get_record(self, *, group_id: str, date: str, mode: str) -> ReportCacheRecord | None:
        payload = self._read_raw()
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            return None

        key = self._build_key(group_id=group_id, date=date, mode=mode)
        raw = entries.get(key)
        if not isinstance(raw, dict):
            return None

        try:
            record = ReportCacheRecord.from_dict(raw)
        except Exception as exc:
            logger.warning(
                "[group_digest.cache_store] invalid_cache_record key=%s error=%s",
                key,
                exc,
            )
            return None
        if record.cache_version != self.cache_version:
            return None
        return record

    async def upsert_record(self, record: ReportCacheRecord) -> None:
        async with self._get_lock():
            payload = self._read_raw()
            entries = payload.setdefault("entries", {})
            if not isinstance(entries, dict):
                entries = {}
                payload["entries"] = entries

            key = self._build_key(group_id=record.group_id, date=record.date, mode=record.mode)
            entries[key] = record.to_dict()
            payload["cache_version"] = self.cache_version
            self._write_raw(payload)

    def _build_key(self, *, group_id: str, date: str, mode: str) -> str:
        return f"{group_id}::{date}::{mode}"

    def _read_raw(self) -> dict[str, Any]:
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning(
                    "[group_digest.cache_store] invalid_payload_type expected=dict got=%s",
                    type(data).__name__,
                )
                return {"cache_version": self.cache_version, "entries": {}}
            return data
        except FileNotFoundError:
            return {"cache_version": self.cache_version, "entries": {}}
        except json.JSONDecodeError as exc:
            logger.warning("[group_digest.cache_store] json_decode_error file=%s error=%s", self.file_path, exc)
            return {"cache_version": self.cache_version, "entries": {}}
        except Exception as exc:
            logger.warning("[group_digest.cache_store] read_failed file=%s error=%s", self.file_path, exc)
            return {"cache_version": self.cache_version, "entries": {}}

    def _write_raw(self, payload: dict[str, Any]) -> None:
        self.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
