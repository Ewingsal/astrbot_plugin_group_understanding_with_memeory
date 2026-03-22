from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from astrbot.api import logger


@dataclass
class GroupOriginRecord:
    group_id: str
    unified_msg_origin: str
    last_active_at: int
    updated_at: str


class GroupOriginStore:
    """持久化群聊会话标识（group_id -> unified_msg_origin）。"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock: asyncio.Lock | None = None
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_raw({"groups": {}})

    async def upsert_group_origin(
        self,
        *,
        group_id: str,
        unified_msg_origin: str,
        last_active_at: int,
    ) -> None:
        if not group_id or not unified_msg_origin:
            return

        async with self._get_lock():
            payload = self._read_raw()
            groups = payload.setdefault("groups", {})
            groups[str(group_id)] = {
                "unified_msg_origin": str(unified_msg_origin),
                "last_active_at": self._safe_int(last_active_at, default=0, field="last_active_at"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._write_raw(payload)

    def list_group_records(self) -> list[GroupOriginRecord]:
        payload = self._read_raw()
        groups = payload.get("groups", {})
        if not isinstance(groups, dict):
            return []

        records: list[GroupOriginRecord] = []
        for gid, data in groups.items():
            if not isinstance(data, dict):
                logger.warning(
                    "[group_digest.group_origin] skip_non_dict_group_record group_id=%s type=%s",
                    gid,
                    type(data).__name__,
                )
                continue
            records.append(
                GroupOriginRecord(
                    group_id=str(gid),
                    unified_msg_origin=str(data.get("unified_msg_origin", "")),
                    last_active_at=self._safe_int(
                        data.get("last_active_at", 0),
                        default=0,
                        field=f"last_active_at[{gid}]",
                    ),
                    updated_at=str(data.get("updated_at", "")),
                )
            )

        records.sort(key=lambda item: item.group_id)
        return records

    def _read_raw(self) -> dict:
        try:
            text = self.file_path.read_text(encoding="utf-8")
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            logger.warning(
                "[group_digest.group_origin] invalid_payload_type expected=dict got=%s",
                type(obj).__name__,
            )
            return {"groups": {}}
        except FileNotFoundError:
            return {"groups": {}}
        except json.JSONDecodeError as exc:
            logger.warning("[group_digest.group_origin] json_decode_error file=%s error=%s", self.file_path, exc)
            return {"groups": {}}
        except Exception as exc:
            logger.warning("[group_digest.group_origin] read_failed file=%s error=%s", self.file_path, exc)
            return {"groups": {}}

    def _write_raw(self, payload: dict) -> None:
        self.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _safe_int(self, value: object, *, default: int, field: str) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.group_origin] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
