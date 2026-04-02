from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
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
    effective_message_count: int = 0
    effective_last_message_ts: int = 0
    effective_last_message_fingerprint: str = ""
    stats_state: dict[str, Any] = field(default_factory=dict)
    semantic_state: dict[str, Any] = field(default_factory=dict)
    incremental_round: int = 0
    semantic_input_source: str = ""
    topic_slice_signature: str = ""
    topic_slice_count: int = 0
    topic_slice_total_chars: int = 0
    topic_slice_selected_chars: int = 0
    topic_slice_truncated: bool = False

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

        stats_state = data.get("stats_state", {})
        if not isinstance(stats_state, dict):
            logger.warning(
                "[group_digest.cache_store] invalid_stats_state_type expected=dict got=%s",
                type(stats_state).__name__,
            )
            stats_state = {}

        semantic_state = data.get("semantic_state", {})
        if not isinstance(semantic_state, dict):
            logger.warning(
                "[group_digest.cache_store] invalid_semantic_state_type expected=dict got=%s",
                type(semantic_state).__name__,
            )
            semantic_state = {}

        legacy_last_ts = cls._safe_int(
            data.get("last_message_timestamp", 0),
            field="last_message_timestamp",
        )
        legacy_count = cls._safe_int(data.get("message_count", 0), field="message_count")

        return cls(
            group_id=str(data.get("group_id", "")),
            date=str(data.get("date", "")),
            mode=str(data.get("mode", "")),
            window_start=cls._safe_int(data.get("window_start", 0), field="window_start"),
            window_end=cls._safe_int(data.get("window_end", 0), field="window_end"),
            generated_at=str(data.get("generated_at", "")),
            last_message_timestamp=legacy_last_ts,
            message_count=legacy_count,
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
            effective_message_count=cls._safe_int(
                data.get("effective_message_count", legacy_count),
                field="effective_message_count",
            ),
            effective_last_message_ts=cls._safe_int(
                data.get("effective_last_message_ts", legacy_last_ts),
                field="effective_last_message_ts",
            ),
            effective_last_message_fingerprint=str(data.get("effective_last_message_fingerprint", "")),
            stats_state=stats_state,
            semantic_state=semantic_state,
            incremental_round=cls._safe_int(data.get("incremental_round", 0), field="incremental_round"),
            semantic_input_source=str(data.get("semantic_input_source", "")).strip(),
            topic_slice_signature=str(data.get("topic_slice_signature", "")).strip(),
            topic_slice_count=max(0, cls._safe_int(data.get("topic_slice_count", 0), field="topic_slice_count")),
            topic_slice_total_chars=max(
                0,
                cls._safe_int(
                    data.get("topic_slice_total_chars", 0),
                    field="topic_slice_total_chars",
                ),
            ),
            topic_slice_selected_chars=max(
                0,
                cls._safe_int(
                    data.get("topic_slice_selected_chars", 0),
                    field="topic_slice_selected_chars",
                ),
            ),
            topic_slice_truncated=cls._safe_bool(
                data.get("topic_slice_truncated", False),
                field="topic_slice_truncated",
            ),
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

    @staticmethod
    def _safe_bool(value: object, *, field: str, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        logger.warning(
            "[group_digest.cache_store] invalid_bool field=%s value=%r fallback=%s",
            field,
            value,
            "true" if default else "false",
        )
        return default


class ReportCacheStore:
    def __init__(self, file_path: Path, cache_version: int = 1):
        self.file_path = file_path
        self.cache_version = int(cache_version)
        self._lock: asyncio.Lock | None = None
        self._file_lock = threading.RLock()
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
            with self._file_lock:
                payload = self._read_raw_unlocked()
                entries = payload.setdefault("entries", {})
                if not isinstance(entries, dict):
                    entries = {}
                    payload["entries"] = entries

                key = self._build_key(group_id=record.group_id, date=record.date, mode=record.mode)
                entries[key] = record.to_dict()
                payload["cache_version"] = self.cache_version
                self._write_raw_unlocked(payload)

    def _build_key(self, *, group_id: str, date: str, mode: str) -> str:
        return f"{group_id}::{date}::{mode}"

    def _read_raw(self) -> dict[str, Any]:
        with self._file_lock:
            return self._read_raw_unlocked()

    def _read_raw_unlocked(self) -> dict[str, Any]:
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
        with self._file_lock:
            self._write_raw_unlocked(payload)

    def _write_raw_unlocked(self, payload: dict[str, Any]) -> None:
        self._atomic_write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _atomic_write_text(self, text: str) -> None:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.file_path.parent),
                prefix=f".{self.file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_file.write(text)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = Path(tmp_file.name)
            os.replace(tmp_path, self.file_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
