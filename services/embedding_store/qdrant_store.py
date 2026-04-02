from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any

from astrbot.api import logger

from .base import EmbeddingStore, SemanticUnitEmbeddingDocument, TopicSliceEmbeddingDocument


DEFAULT_QDRANT_TIMEOUT_SECONDS = 5
DEFAULT_QDRANT_DISTANCE = "Cosine"


class QdrantEmbeddingStore(EmbeddingStore):
    """Qdrant embedding 存储（第一版，HTTP API）。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        semantic_unit_collection: str = "group_digest_semantic_units",
        topic_slice_collection: str = "group_digest_topic_slices",
        vector_size: int = 1536,
        distance_metric: str = "cosine",
        prefer_grpc: bool = False,
        timeout_seconds: int = DEFAULT_QDRANT_TIMEOUT_SECONDS,
    ) -> None:
        self._enabled = bool(enabled)
        self.qdrant_url = str(qdrant_url or "").strip().rstrip("/")
        self.qdrant_api_key = str(qdrant_api_key or "").strip()
        self.semantic_unit_collection = str(semantic_unit_collection or "").strip() or "group_digest_semantic_units"
        self.topic_slice_collection = str(topic_slice_collection or "").strip() or "group_digest_topic_slices"
        self.vector_size = max(1, int(vector_size))
        self.distance_metric = self._normalize_distance_metric(distance_metric)
        self.prefer_grpc = bool(prefer_grpc)
        self.timeout_seconds = max(1, int(timeout_seconds))

        self._ready = False
        self._disabled_reason = ""
        self._init_lock: asyncio.Lock | None = None

        if not self._enabled:
            self._disabled_reason = "disabled_by_config"
        elif not self.qdrant_url:
            self._enabled = False
            self._disabled_reason = "qdrant_url_missing"
            logger.warning(
                "[group_digest.embedding_store] qdrant_disabled reason=%s",
                self._disabled_reason,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def upsert_semantic_unit(self, doc: SemanticUnitEmbeddingDocument) -> bool:
        if not doc.vector:
            return False
        if not await self._ensure_ready():
            return False
        return await self._upsert_point(
            collection=self.semantic_unit_collection,
            point_id=doc.point_id,
            vector=doc.vector,
            payload=doc.payload,
        )

    async def upsert_topic_slice(self, doc: TopicSliceEmbeddingDocument) -> bool:
        if not doc.vector:
            return False
        if not await self._ensure_ready():
            return False
        return await self._upsert_point(
            collection=self.topic_slice_collection,
            point_id=doc.point_id,
            vector=doc.vector,
            payload=doc.payload,
        )

    async def query_semantic_units(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        topic_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not await self._ensure_ready():
            return []
        return await self._scroll_points(
            collection=self.semantic_unit_collection,
            group_id=group_id,
            date_label=date_label,
            topic_id=topic_id,
            limit=limit,
        )

    async def query_topic_slices(
        self,
        *,
        group_id: str,
        date_label: str | None = None,
        topic_id: str | None = None,
        query_vector: list[float] | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        recent_days: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not await self._ensure_ready():
            return []
        if query_vector:
            return await self._search_topic_slice_points(
                collection=self.topic_slice_collection,
                group_id=group_id,
                date_label=date_label,
                topic_id=topic_id,
                query_vector=query_vector,
                start_ts=start_ts,
                end_ts=end_ts,
                recent_days=recent_days,
                limit=limit,
            )
        return await self._scroll_topic_slice_points(
            collection=self.topic_slice_collection,
            group_id=group_id,
            date_label=date_label,
            topic_id=topic_id,
            start_ts=start_ts,
            end_ts=end_ts,
            recent_days=recent_days,
            limit=limit,
        )

    async def _ensure_ready(self) -> bool:
        if not self._enabled:
            return False
        if self._ready:
            return True

        async with self._get_init_lock():
            if not self._enabled:
                return False
            if self._ready:
                return True

            try:
                await self._ensure_collection(self.semantic_unit_collection)
                await self._ensure_collection(self.topic_slice_collection)
            except Exception as exc:
                self._enabled = False
                self._disabled_reason = f"qdrant_init_failed:{exc}"
                logger.warning(
                    "[group_digest.embedding_store] qdrant_init_failed url=%s error=%s",
                    self.qdrant_url,
                    exc,
                )
                return False

            self._ready = True
            logger.info(
                "[group_digest.embedding_store] qdrant_ready url=%s semantic_collection=%s topic_collection=%s vector_size=%d distance=%s prefer_grpc=%s",
                self.qdrant_url,
                self.semantic_unit_collection,
                self.topic_slice_collection,
                self.vector_size,
                self.distance_metric,
                "true" if self.prefer_grpc else "false",
            )
            return True

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    async def _ensure_collection(self, collection: str) -> None:
        status, _ = await self._request_json(
            method="GET",
            path=f"/collections/{collection}",
            body=None,
            allowed_statuses={200, 404},
        )
        if status == 200:
            return
        payload = {
            "vectors": {
                "size": self.vector_size,
                "distance": self.distance_metric,
            }
        }
        await self._request_json(
            method="PUT",
            path=f"/collections/{collection}",
            body=payload,
            allowed_statuses={200},
        )

    async def _upsert_point(
        self,
        *,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> bool:
        body = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }
            ]
        }
        try:
            await self._request_json(
                method="PUT",
                path=f"/collections/{collection}/points?wait=true",
                body=body,
                allowed_statuses={200},
            )
            return True
        except Exception as exc:
            logger.warning(
                "[group_digest.embedding_store] qdrant_upsert_failed collection=%s point_id=%s error=%s",
                collection,
                point_id,
                exc,
            )
            return False

    async def _scroll_points(
        self,
        *,
        collection: str,
        group_id: str,
        date_label: str | None,
        topic_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [
            {
                "key": "group_id",
                "match": {"value": group_id},
            }
        ]
        if date_label:
            filters.append(
                {
                    "key": "date_label",
                    "match": {"value": date_label},
                }
            )
        if topic_id:
            filters.append(
                {
                    "key": "topic_id",
                    "match": {"value": topic_id},
                }
            )

        body = {
            "limit": max(1, int(limit)),
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": filters},
        }
        try:
            _status, data = await self._request_json(
                method="POST",
                path=f"/collections/{collection}/points/scroll",
                body=body,
                allowed_statuses={200},
            )
        except Exception as exc:
            logger.warning(
                "[group_digest.embedding_store] qdrant_scroll_failed collection=%s error=%s",
                collection,
                exc,
            )
            return []

        if not isinstance(data, dict):
            return []
        result = data.get("result", {})
        if not isinstance(result, dict):
            return []
        points = result.get("points", [])
        if not isinstance(points, list):
            return []

        payloads: list[dict[str, Any]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            row = point.get("payload")
            if isinstance(row, dict):
                payloads.append(row)
        return payloads

    async def _search_topic_slice_points(
        self,
        *,
        collection: str,
        group_id: str,
        date_label: str | None,
        topic_id: str | None,
        query_vector: list[float],
        start_ts: int | None,
        end_ts: int | None,
        recent_days: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        resolved_start_ts, resolved_end_ts = self._resolve_time_range(
            start_ts=start_ts,
            end_ts=end_ts,
            recent_days=recent_days,
        )
        body = {
            "vector": list(query_vector),
            "limit": max(1, int(limit)),
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": self._build_topic_slice_must_filters(
                    group_id=group_id,
                    date_label=date_label,
                    topic_id=topic_id,
                    start_ts=resolved_start_ts,
                    end_ts=resolved_end_ts,
                )
            },
        }
        try:
            _status, data = await self._request_json(
                method="POST",
                path=f"/collections/{collection}/points/search",
                body=body,
                allowed_statuses={200},
            )
        except Exception as exc:
            logger.warning(
                "[group_digest.embedding_store] qdrant_search_failed collection=%s error=%s",
                collection,
                exc,
            )
            return []
        return self._extract_payload_rows(data)

    async def _scroll_topic_slice_points(
        self,
        *,
        collection: str,
        group_id: str,
        date_label: str | None,
        topic_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
        recent_days: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        resolved_start_ts, resolved_end_ts = self._resolve_time_range(
            start_ts=start_ts,
            end_ts=end_ts,
            recent_days=recent_days,
        )
        body = {
            "limit": max(1, int(limit)),
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": self._build_topic_slice_must_filters(
                    group_id=group_id,
                    date_label=date_label,
                    topic_id=topic_id,
                    start_ts=resolved_start_ts,
                    end_ts=resolved_end_ts,
                )
            },
        }
        try:
            _status, data = await self._request_json(
                method="POST",
                path=f"/collections/{collection}/points/scroll",
                body=body,
                allowed_statuses={200},
            )
        except Exception as exc:
            logger.warning(
                "[group_digest.embedding_store] qdrant_scroll_failed collection=%s error=%s",
                collection,
                exc,
            )
            return []
        return self._extract_payload_rows(data)

    def _build_topic_slice_must_filters(
        self,
        *,
        group_id: str,
        date_label: str | None,
        topic_id: str | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[dict[str, Any]]:
        must_filters: list[dict[str, Any]] = [
            {
                "key": "object_type",
                "match": {"value": "topic_slice"},
            },
            {
                "key": "group_id",
                "match": {"value": group_id},
            },
        ]
        if date_label:
            must_filters.append(
                {
                    "key": "date_label",
                    "match": {"value": date_label},
                }
            )
        if topic_id:
            must_filters.append(
                {
                    "key": "topic_id",
                    "match": {"value": topic_id},
                }
            )
        if start_ts is not None or end_ts is not None:
            range_payload: dict[str, int] = {}
            if start_ts is not None:
                range_payload["gte"] = int(start_ts)
            if end_ts is not None:
                range_payload["lt"] = int(end_ts)
            must_filters.append(
                {
                    "key": "end_ts",
                    "range": range_payload,
                }
            )
        return must_filters

    def _resolve_time_range(
        self,
        *,
        start_ts: int | None,
        end_ts: int | None,
        recent_days: int | None,
    ) -> tuple[int | None, int | None]:
        if start_ts is not None or end_ts is not None:
            return start_ts, end_ts
        if recent_days is None:
            return None, None
        safe_days = max(1, int(recent_days))
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=safe_days)
        return int(start_dt.timestamp()), int(end_dt.timestamp())

    def _extract_payload_rows(
        self,
        data: dict[str, Any] | list[Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        result = data.get("result")
        points: list[Any] = []
        if isinstance(result, dict):
            maybe_points = result.get("points")
            if isinstance(maybe_points, list):
                points = maybe_points
        elif isinstance(result, list):
            points = result

        payloads: list[dict[str, Any]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            row = point.get("payload")
            if isinstance(row, dict):
                payloads.append(row)
        return payloads

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        allowed_statuses: set[int],
    ) -> tuple[int, dict[str, Any] | list[Any] | None]:
        return await asyncio.to_thread(
            self._request_json_sync,
            method=method,
            path=path,
            body=body,
            allowed_statuses=allowed_statuses,
        )

    def _request_json_sync(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        allowed_statuses: set[int],
    ) -> tuple[int, dict[str, Any] | list[Any] | None]:
        url = f"{self.qdrant_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            text = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"qdrant_request_error url={url} error={exc}") from exc

        if status not in allowed_statuses:
            raise RuntimeError(f"qdrant_unexpected_status url={url} status={status} body={text[:500]}")

        if not text:
            return status, None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return status, parsed
            return status, None
        except json.JSONDecodeError:
            return status, None

    def _normalize_distance_metric(self, value: str) -> str:
        lowered = str(value or "").strip().lower()
        if lowered in {"cosine", ""}:
            return "Cosine"
        if lowered in {"dot", "dotproduct", "dot_product"}:
            return "Dot"
        if lowered in {"euclid", "euclidean", "l2"}:
            return "Euclid"
        if lowered in {"manhattan", "l1"}:
            return "Manhattan"
        return DEFAULT_QDRANT_DISTANCE
