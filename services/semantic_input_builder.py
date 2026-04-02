from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from astrbot.api import logger

from .embedding.base import EmbeddingBackend
from .embedding.noop_backend import NoopEmbeddingBackend
from .embedding_store.base import EmbeddingStore
from .embedding_store.noop_store import NoopEmbeddingStore
from .group_topic_segment_manager import GroupTopicSegmentManager
from .models import MessageRecord, SlangExplanationRecord
from .slang_candidate_miner import SlangCandidateMiner
from .slang_interpretation_service import SlangInterpretationService
from .slang_store import SlangStore
from .topic_slice_store import TopicSliceStore


@dataclass(frozen=True)
class SemanticInputMaterial:
    """日报语义层输入材料。"""

    messages: list[MessageRecord]
    topic_slice_contexts: list[str]
    source: str
    total_effective_messages: int
    selected_message_count: int
    truncated: bool
    topic_slice_total_count: int = 0
    topic_slice_selected_count: int = 0
    topic_slice_total_chars: int = 0
    topic_slice_selected_chars: int = 0
    topic_slice_truncated: bool = False
    topic_slice_signature: str = ""
    retrieved_topic_slice_count: int = 0
    current_day_topic_slice_count: int = 0
    retrieval_enabled: bool = False
    retrieval_degraded: bool = False
    retrieval_query_chars: int = 0
    retrieval_query_tail_count: int = 0
    retrieval_query_topic_hint_count: int = 0
    retrieval_query_mode: str = ""
    slang_context_count: int = 0
    slang_context_chars: int = 0
    slang_candidate_count: int = 0
    slang_inferred_count: int = 0
    slang_reused_count: int = 0
    slang_degraded: bool = False
    slang_signature: str = ""


class SemanticInputBuilder:
    """统一构建“本次 LLM 语义分析输入材料”的中间层。"""

    DEFAULT_MAX_TOPIC_SLICE_CONTEXT_CHARS = 6000
    DEFAULT_TOPIC_SLICE_RETRIEVAL_RECENT_DAYS = 3
    DEFAULT_TOPIC_SLICE_RETRIEVAL_LIMIT = 5
    DEFAULT_TOPIC_SLICE_RETRIEVAL_QUERY_MESSAGE_COUNT = 8
    DEFAULT_MAX_SLANG_CONTEXT_CHARS = 1200
    DEFAULT_SLANG_INJECTION_LIMIT = 5
    DEFAULT_SLANG_RECENT_DAYS = 7

    def __init__(
        self,
        topic_segment_manager: GroupTopicSegmentManager | None = None,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        embedding_store: EmbeddingStore | None = None,
        enable_topic_slice_contexts: bool = True,
        max_topic_slice_context_chars: int = DEFAULT_MAX_TOPIC_SLICE_CONTEXT_CHARS,
        enable_topic_slice_retrieval: bool = True,
        topic_slice_retrieval_recent_days: int = DEFAULT_TOPIC_SLICE_RETRIEVAL_RECENT_DAYS,
        topic_slice_retrieval_limit: int = DEFAULT_TOPIC_SLICE_RETRIEVAL_LIMIT,
        topic_slice_retrieval_query_message_count: int = DEFAULT_TOPIC_SLICE_RETRIEVAL_QUERY_MESSAGE_COUNT,
        topic_slice_store: TopicSliceStore | None = None,
        slang_store: SlangStore | None = None,
        slang_candidate_miner: SlangCandidateMiner | None = None,
        slang_interpretation_service: SlangInterpretationService | None = None,
        enable_slang_contexts: bool = False,
        max_slang_context_chars: int = DEFAULT_MAX_SLANG_CONTEXT_CHARS,
        slang_injection_limit: int = DEFAULT_SLANG_INJECTION_LIMIT,
        slang_recent_days: int = DEFAULT_SLANG_RECENT_DAYS,
    ):
        self.topic_segment_manager = topic_segment_manager
        self.embedding_backend = embedding_backend or NoopEmbeddingBackend()
        self.embedding_store = embedding_store or NoopEmbeddingStore()
        self.enable_topic_slice_contexts = bool(enable_topic_slice_contexts)
        self.max_topic_slice_context_chars = max(1, int(max_topic_slice_context_chars))
        self.enable_topic_slice_retrieval = bool(enable_topic_slice_retrieval)
        self.topic_slice_retrieval_recent_days = max(1, int(topic_slice_retrieval_recent_days))
        self.topic_slice_retrieval_limit = max(1, int(topic_slice_retrieval_limit))
        self.topic_slice_retrieval_query_message_count = max(1, int(topic_slice_retrieval_query_message_count))
        self.topic_slice_store = topic_slice_store
        self.slang_store = slang_store
        self.slang_candidate_miner = slang_candidate_miner
        self.slang_interpretation_service = slang_interpretation_service
        self.enable_slang_contexts = bool(enable_slang_contexts)
        self.max_slang_context_chars = max(1, int(max_slang_context_chars))
        self.slang_injection_limit = max(1, int(slang_injection_limit))
        self.slang_recent_days = max(1, int(slang_recent_days))

    async def build_for_full_window(
        self,
        *,
        group_id: str,
        date_label: str,
        time_window: str,
        mode: str,
        effective_messages: list[MessageRecord],
        max_messages_for_analysis: int,
        context: Any | None = None,
        event: Any | None = None,
        analysis_provider_id: str = "",
    ) -> SemanticInputMaterial:
        selected_messages = self._select_tail_messages(
            messages=effective_messages,
            max_count=max_messages_for_analysis,
        )
        raw_current_day_slice_contexts = self._collect_topic_slice_contexts(
            group_id=group_id,
            date_label=date_label,
            time_window=time_window,
            mode=mode,
        )
        current_day_slice_contexts, current_day_guard = self._guard_topic_slice_contexts(raw_current_day_slice_contexts)

        retrieved_slice_contexts, retrieval_meta = await self._collect_retrieved_slice_contexts(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            messages=selected_messages,
        )
        topic_slice_contexts = self._merge_slice_contexts(
            retrieved_contexts=retrieved_slice_contexts,
            current_day_contexts=current_day_slice_contexts,
        )
        slang_contexts, slang_meta = await self._collect_slang_contexts(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            selected_messages=selected_messages,
            topic_slice_contexts=topic_slice_contexts,
            context=context,
            event=event,
            analysis_provider_id=analysis_provider_id,
        )
        all_contexts = self._merge_all_contexts(
            retrieved_contexts=retrieved_slice_contexts,
            current_day_contexts=current_day_slice_contexts,
            slang_contexts=slang_contexts,
        )

        source = self._resolve_source_label(
            retrieved_count=len(retrieved_slice_contexts),
            current_day_count=len(current_day_slice_contexts),
            slang_count=len(slang_contexts),
            fallback_source="raw_effective_messages_tail",
            with_delta=False,
        )
        material = SemanticInputMaterial(
            messages=selected_messages,
            topic_slice_contexts=all_contexts,
            source=source,
            total_effective_messages=len(effective_messages),
            selected_message_count=len(selected_messages),
            truncated=len(selected_messages) < len(effective_messages),
            topic_slice_total_count=current_day_guard["total_count"],
            topic_slice_selected_count=current_day_guard["selected_count"],
            topic_slice_total_chars=current_day_guard["total_chars"],
            topic_slice_selected_chars=current_day_guard["selected_chars"],
            topic_slice_truncated=current_day_guard["truncated"],
            topic_slice_signature=self._build_topic_slice_signature(all_contexts),
            retrieved_topic_slice_count=len(retrieved_slice_contexts),
            current_day_topic_slice_count=len(current_day_slice_contexts),
            retrieval_enabled=retrieval_meta["enabled"],
            retrieval_degraded=retrieval_meta["degraded"],
            retrieval_query_chars=retrieval_meta["query_chars"],
            retrieval_query_tail_count=retrieval_meta["tail_count"],
            retrieval_query_topic_hint_count=retrieval_meta["topic_hint_count"],
            retrieval_query_mode=str(retrieval_meta["query_mode"]),
            slang_context_count=len(slang_contexts),
            slang_context_chars=sum(len(item) for item in slang_contexts),
            slang_candidate_count=int(slang_meta.get("candidate_count", 0)),
            slang_inferred_count=int(slang_meta.get("inferred_count", 0)),
            slang_reused_count=int(slang_meta.get("reused_count", 0)),
            slang_degraded=bool(slang_meta.get("degraded", False)),
            slang_signature=self._build_topic_slice_signature(slang_contexts),
        )
        self._log_material(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            material=material,
        )
        return material

    async def build_for_incremental(
        self,
        *,
        group_id: str,
        date_label: str,
        time_window: str,
        mode: str,
        delta_messages: list[MessageRecord],
        max_messages_for_analysis: int,
        context: Any | None = None,
        event: Any | None = None,
        analysis_provider_id: str = "",
    ) -> SemanticInputMaterial:
        selected_messages = self._select_tail_messages(
            messages=delta_messages,
            max_count=max_messages_for_analysis,
        )
        raw_current_day_slice_contexts = self._collect_topic_slice_contexts(
            group_id=group_id,
            date_label=date_label,
            time_window=time_window,
            mode=mode,
        )
        current_day_slice_contexts, current_day_guard = self._guard_topic_slice_contexts(raw_current_day_slice_contexts)

        retrieved_slice_contexts, retrieval_meta = await self._collect_retrieved_slice_contexts(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            messages=selected_messages,
        )
        topic_slice_contexts = self._merge_slice_contexts(
            retrieved_contexts=retrieved_slice_contexts,
            current_day_contexts=current_day_slice_contexts,
        )
        slang_contexts, slang_meta = await self._collect_slang_contexts(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            selected_messages=selected_messages,
            topic_slice_contexts=topic_slice_contexts,
            context=context,
            event=event,
            analysis_provider_id=analysis_provider_id,
        )
        all_contexts = self._merge_all_contexts(
            retrieved_contexts=retrieved_slice_contexts,
            current_day_contexts=current_day_slice_contexts,
            slang_contexts=slang_contexts,
        )

        source = self._resolve_source_label(
            retrieved_count=len(retrieved_slice_contexts),
            current_day_count=len(current_day_slice_contexts),
            slang_count=len(slang_contexts),
            fallback_source="delta_tail_messages",
            with_delta=True,
        )
        material = SemanticInputMaterial(
            messages=selected_messages,
            topic_slice_contexts=all_contexts,
            source=source,
            total_effective_messages=len(delta_messages),
            selected_message_count=len(selected_messages),
            truncated=len(selected_messages) < len(delta_messages),
            topic_slice_total_count=current_day_guard["total_count"],
            topic_slice_selected_count=current_day_guard["selected_count"],
            topic_slice_total_chars=current_day_guard["total_chars"],
            topic_slice_selected_chars=current_day_guard["selected_chars"],
            topic_slice_truncated=current_day_guard["truncated"],
            topic_slice_signature=self._build_topic_slice_signature(all_contexts),
            retrieved_topic_slice_count=len(retrieved_slice_contexts),
            current_day_topic_slice_count=len(current_day_slice_contexts),
            retrieval_enabled=retrieval_meta["enabled"],
            retrieval_degraded=retrieval_meta["degraded"],
            retrieval_query_chars=retrieval_meta["query_chars"],
            retrieval_query_tail_count=retrieval_meta["tail_count"],
            retrieval_query_topic_hint_count=retrieval_meta["topic_hint_count"],
            retrieval_query_mode=str(retrieval_meta["query_mode"]),
            slang_context_count=len(slang_contexts),
            slang_context_chars=sum(len(item) for item in slang_contexts),
            slang_candidate_count=int(slang_meta.get("candidate_count", 0)),
            slang_inferred_count=int(slang_meta.get("inferred_count", 0)),
            slang_reused_count=int(slang_meta.get("reused_count", 0)),
            slang_degraded=bool(slang_meta.get("degraded", False)),
            slang_signature=self._build_topic_slice_signature(slang_contexts),
        )
        self._log_material(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            material=material,
        )
        return material

    async def _collect_retrieved_slice_contexts(
        self,
        *,
        group_id: str,
        date_label: str,
        mode: str,
        messages: list[MessageRecord],
    ) -> tuple[list[str], dict[str, Any]]:
        if not self.enable_topic_slice_retrieval:
            return [], {
                "enabled": False,
                "degraded": False,
                "query_chars": 0,
                "tail_count": 0,
                "topic_hint_count": 0,
                "query_mode": "",
            }

        query_payload = self._build_retrieval_query_payload(
            group_id=group_id,
            date_label=date_label,
            mode=mode,
            messages=messages,
        )
        query_text = query_payload["text"]
        query_chars = int(query_payload["query_chars"])
        tail_count = int(query_payload["tail_count"])
        topic_hint_count = int(query_payload["topic_hint_count"])
        query_mode = str(query_payload["query_mode"])
        query_preview = str(query_payload["query_preview"])
        if not query_text:
            return [], {
                "enabled": True,
                "degraded": False,
                "query_chars": 0,
                "tail_count": tail_count,
                "topic_hint_count": topic_hint_count,
                "query_mode": query_mode,
            }

        logger.info(
            "[group_digest.semantic_input] topic_slice_retrieval_query_built group_id=%s mode=%s query_mode=%s query_chars=%d tail_count=%d topic_hint_count=%d query_preview=%s",
            group_id,
            mode,
            query_mode,
            query_chars,
            tail_count,
            topic_hint_count,
            query_preview,
        )

        if not self.embedding_store.enabled:
            logger.info(
                "[group_digest.semantic_input] topic_slice_retrieval_noop group_id=%s mode=%s reason=embedding_store_disabled",
                group_id,
                mode,
            )
            return [], {
                "enabled": True,
                "degraded": True,
                "query_chars": query_chars,
                "tail_count": tail_count,
                "topic_hint_count": topic_hint_count,
                "query_mode": query_mode,
            }

        try:
            vector = await self.embedding_backend.embed_text(query_text)
        except Exception as exc:
            logger.warning(
                "[group_digest.semantic_input] topic_slice_retrieval_embed_failed group_id=%s mode=%s error=%s",
                group_id,
                mode,
                exc,
            )
            return [], {
                "enabled": True,
                "degraded": True,
                "query_chars": query_chars,
                "tail_count": tail_count,
                "topic_hint_count": topic_hint_count,
                "query_mode": query_mode,
            }

        if not vector:
            logger.info(
                "[group_digest.semantic_input] topic_slice_retrieval_noop group_id=%s mode=%s reason=empty_query_embedding",
                group_id,
                mode,
            )
            return [], {
                "enabled": True,
                "degraded": True,
                "query_chars": query_chars,
                "tail_count": tail_count,
                "topic_hint_count": topic_hint_count,
                "query_mode": query_mode,
            }

        day_start_ts = self._resolve_day_start_ts(date_label=date_label)
        start_ts = day_start_ts - self.topic_slice_retrieval_recent_days * 24 * 60 * 60
        end_ts = day_start_ts
        try:
            rows = await self.embedding_store.query_topic_slices(
                group_id=group_id,
                query_vector=[float(item) for item in vector],
                start_ts=start_ts,
                end_ts=end_ts,
                recent_days=self.topic_slice_retrieval_recent_days,
                limit=self.topic_slice_retrieval_limit,
            )
        except Exception as exc:
            logger.warning(
                "[group_digest.semantic_input] topic_slice_retrieval_query_failed group_id=%s mode=%s error=%s",
                group_id,
                mode,
                exc,
            )
            return [], {
                "enabled": True,
                "degraded": True,
                "query_chars": query_chars,
                "tail_count": tail_count,
                "topic_hint_count": topic_hint_count,
                "query_mode": query_mode,
            }

        contexts: list[str] = []
        for row in rows:
            context = self._format_retrieved_slice_context(row)
            if context:
                contexts.append(context)

        logger.info(
            "[group_digest.semantic_input] topic_slice_retrieval group_id=%s mode=%s query_mode=%s query_chars=%d tail_count=%d topic_hint_count=%d results=%d recent_days=%d limit=%d",
            group_id,
            mode,
            query_mode,
            query_chars,
            tail_count,
            topic_hint_count,
            len(contexts),
            self.topic_slice_retrieval_recent_days,
            self.topic_slice_retrieval_limit,
        )
        return contexts, {
            "enabled": True,
            "degraded": False,
            "query_chars": query_chars,
            "tail_count": tail_count,
            "topic_hint_count": topic_hint_count,
            "query_mode": query_mode,
        }

    def _build_retrieval_query_payload(
        self,
        *,
        group_id: str,
        date_label: str,
        mode: str,
        messages: list[MessageRecord],
    ) -> dict[str, Any]:
        if not messages:
            return {
                "text": "",
                "query_chars": 0,
                "tail_count": 0,
                "topic_hint_count": 0,
                "query_mode": "empty",
                "query_preview": "",
            }
        ordered = sorted(messages, key=lambda item: item.timestamp)
        tail = ordered[-self.topic_slice_retrieval_query_message_count :]
        tail_lines: list[str] = []
        for row in tail:
            sender = str(row.sender_name or row.sender_id or "unknown")
            content = str(row.content or "").strip()
            if not content:
                continue
            tail_lines.append(f"{sender}: {content}")

        mode_label = str(mode or "").strip().lower()
        topic_hint_lines: list[str] = []
        query_mode = "tail_only"
        if mode_label == "scheduled":
            topic_hint_lines = self._collect_scheduled_topic_hint_lines(
                group_id=group_id,
                date_label=date_label,
            )
            if topic_hint_lines:
                query_mode = "scheduled_topic_plus_tail"
            else:
                query_mode = "scheduled_tail_only"

        rows: list[str] = []
        if topic_hint_lines:
            rows.append("scheduled_runtime_topics:")
            rows.extend(topic_hint_lines)
        if tail_lines:
            rows.append("tail_effective_messages:")
            rows.extend(tail_lines)

        text = "\n".join(rows).strip()
        if not text and tail_lines:
            text = "\n".join(tail_lines).strip()

        return {
            "text": text,
            "query_chars": len(text),
            "tail_count": len(tail_lines),
            "topic_hint_count": len(topic_hint_lines),
            "query_mode": query_mode,
            "query_preview": self._normalize_preview(text),
        }

    def _collect_scheduled_topic_hint_lines(
        self,
        *,
        group_id: str,
        date_label: str,
    ) -> list[str]:
        manager = self.topic_segment_manager
        if manager is None:
            return []

        getter = getattr(manager, "get_day_topics_snapshot", None)
        if not callable(getter):
            return []

        try:
            rows = getter(group_id=group_id, date_label=date_label)
        except Exception as exc:
            logger.warning(
                "[group_digest.semantic_input] retrieval_topic_snapshot_failed group_id=%s date=%s error=%s",
                group_id,
                date_label,
                exc,
            )
            return []
        if not isinstance(rows, list):
            return []

        active_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("status", "")).strip() in {"active", "created"}
        ]
        if not active_rows:
            return []

        active_rows.sort(
            key=lambda item: self._safe_int(item.get("last_active_at", 0)),
            reverse=True,
        )
        hints: list[str] = []
        for row in active_rows[:2]:
            hint = self._format_topic_hint_line(row=row)
            if hint:
                hints.append(hint)
        return hints

    def _format_topic_hint_line(self, *, row: dict[str, Any]) -> str:
        topic_id = str(row.get("topic_id", "")).strip()
        if not topic_id:
            return ""
        core_text = str(row.get("core_text", "")).strip()
        if len(core_text) > 120:
            core_text = f"{core_text[:120]}..."

        participant_text = "无"
        participants = row.get("participants", [])
        if isinstance(participants, list):
            names = [str(item).strip() for item in participants if str(item).strip()]
            if names:
                participant_text = "、".join(names[:5])

        message_count = max(0, self._safe_int(row.get("message_count", 0)))
        return (
            f"topic_id={topic_id}; message_count={message_count}; "
            f"participants={participant_text}; core_text={core_text or '无'}"
        )

    def _normalize_preview(self, text: str, max_chars: int = 120) -> str:
        normalized = " ".join(str(text).split()).strip()
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars]}..."

    def _format_retrieved_slice_context(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        if str(payload.get("object_type", "")).strip() not in {"", "topic_slice"}:
            return ""
        topic_id = str(payload.get("topic_id", "")).strip()
        if not topic_id:
            return ""
        date_label = str(payload.get("date_label", "")).strip()
        start_ts = self._safe_int(payload.get("start_ts", 0))
        end_ts = self._safe_int(payload.get("end_ts", 0))
        message_count = max(0, self._safe_int(payload.get("message_count", 0)))
        participants = payload.get("participants", [])
        participant_text = "无"
        if isinstance(participants, list):
            rows = [str(item).strip() for item in participants if str(item).strip()]
            if rows:
                participant_text = "、".join(rows[:5])
        core_text = str(payload.get("core_text", "")).strip() or "无"
        if len(core_text) > 120:
            core_text = f"{core_text[:120]}..."
        start_text = datetime.fromtimestamp(start_ts).strftime("%H:%M") if start_ts > 0 else "--:--"
        end_text = datetime.fromtimestamp(end_ts).strftime("%H:%M") if end_ts > 0 else "--:--"
        return (
            f"retrieved_topic_id={topic_id}; date={date_label}; time={start_text}-{end_text}; "
            f"message_count={message_count}; participants={participant_text}; core_text={core_text}"
        )

    def _merge_slice_contexts(
        self,
        *,
        retrieved_contexts: list[str],
        current_day_contexts: list[str],
    ) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for text in retrieved_contexts + current_day_contexts:
            normalized = str(text).strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            rows.append(normalized)
        return rows

    def _merge_all_contexts(
        self,
        *,
        retrieved_contexts: list[str],
        current_day_contexts: list[str],
        slang_contexts: list[str],
    ) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for text in retrieved_contexts + current_day_contexts + slang_contexts:
            normalized = str(text).strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            rows.append(normalized)
        return rows

    def _resolve_source_label(
        self,
        *,
        retrieved_count: int,
        current_day_count: int,
        slang_count: int,
        fallback_source: str,
        with_delta: bool,
    ) -> str:
        base = fallback_source
        if retrieved_count > 0 and current_day_count > 0:
            if with_delta:
                base = "retrieved_topic_slices_plus_current_day_slices_plus_delta_tail_messages"
            else:
                base = "retrieved_topic_slices_plus_current_day_slices_plus_tail_raw_messages"
        elif retrieved_count > 0:
            base = (
                "retrieved_topic_slices_plus_delta_tail_messages"
                if with_delta
                else "retrieved_topic_slices_plus_tail_raw_messages"
            )
        elif current_day_count > 0:
            base = (
                "topic_slices_plus_delta_tail_messages"
                if with_delta
                else "topic_slices_plus_tail_raw_messages"
            )

        if slang_count > 0:
            return f"{base}_plus_slang_contexts"
        return base

    async def _collect_slang_contexts(
        self,
        *,
        group_id: str,
        date_label: str,
        mode: str,
        selected_messages: list[MessageRecord],
        topic_slice_contexts: list[str],
        context: Any | None,
        event: Any | None,
        analysis_provider_id: str,
    ) -> tuple[list[str], dict[str, Any]]:
        if not self.enable_slang_contexts:
            return [], {
                "enabled": False,
                "degraded": False,
                "candidate_count": 0,
                "inferred_count": 0,
                "reused_count": 0,
            }
        if self.slang_store is None:
            return [], {
                "enabled": True,
                "degraded": True,
                "candidate_count": 0,
                "inferred_count": 0,
                "reused_count": 0,
            }

        reference_text = self._build_slang_reference_text(
            selected_messages=selected_messages,
            topic_slice_contexts=topic_slice_contexts,
        )
        if not reference_text:
            return [], {
                "enabled": True,
                "degraded": False,
                "candidate_count": 0,
                "inferred_count": 0,
                "reused_count": 0,
            }

        relevant_records = self.slang_store.find_relevant(
            group_id=group_id,
            text=reference_text,
            limit=self.slang_injection_limit,
        )
        relevant_by_term: dict[str, SlangExplanationRecord] = {
            str(row.slang_term).strip(): row
            for row in relevant_records
            if str(row.slang_term).strip()
        }

        infer_meta: dict[str, Any] = {
            "enabled": True,
            "degraded": False,
            "candidate_count": 0,
            "inferred_count": 0,
            "reused_count": len(relevant_by_term),
        }

        if (
            self.slang_candidate_miner is None
            or self.slang_interpretation_service is None
            or self.topic_slice_store is None
        ):
            return self._format_guarded_slang_contexts(
                records=list(relevant_by_term.values()),
                meta=infer_meta,
            )

        if context is None or event is None:
            # 未提供 LLM 执行上下文时仅复用已知黑话，避免影响主链路。
            infer_meta["degraded"] = True
            return self._format_guarded_slang_contexts(
                records=list(relevant_by_term.values()),
                meta=infer_meta,
            )

        current_day_slices = self.topic_slice_store.load_slices(
            group_id=group_id,
            date_label=date_label,
        )
        day_start_ts = self._resolve_day_start_ts(date_label=date_label)
        day_end_ts = day_start_ts + 24 * 60 * 60
        recent_start_ts = day_start_ts - self.slang_recent_days * 24 * 60 * 60
        recent_slices = self.topic_slice_store.load_slices(
            group_id=group_id,
            start_ts=recent_start_ts,
            end_ts=day_end_ts,
        )

        candidates = self.slang_candidate_miner.mine_candidates(
            current_day_slices=current_day_slices,
            recent_slices=recent_slices,
            exclude_terms=set(),
        )
        infer_meta["candidate_count"] = len(candidates)

        if candidates:
            resolved_records, resolved_meta = await self.slang_interpretation_service.resolve_candidates(
                context=context,
                event=event,
                analysis_provider_id=analysis_provider_id,
                group_id=group_id,
                date_label=date_label,
                candidates=candidates,
            )
            infer_meta["degraded"] = bool(resolved_meta.get("degraded", False))
            infer_meta["inferred_count"] = int(resolved_meta.get("inferred_count", 0))
            infer_meta["reused_count"] = max(
                int(resolved_meta.get("reused_count", 0)),
                infer_meta["reused_count"],
            )
            for row in resolved_records:
                term = str(row.slang_term or "").strip()
                if not term:
                    continue
                if term not in reference_text:
                    continue
                relevant_by_term[term] = row

        logger.info(
            "[group_digest.slang] semantic_builder group_id=%s mode=%s candidate_count=%d relevant_records=%d inferred=%d reused=%d degraded=%s",
            group_id,
            mode,
            int(infer_meta["candidate_count"]),
            len(relevant_by_term),
            int(infer_meta["inferred_count"]),
            int(infer_meta["reused_count"]),
            "true" if bool(infer_meta["degraded"]) else "false",
        )

        return self._format_guarded_slang_contexts(
            records=list(relevant_by_term.values()),
            meta=infer_meta,
        )

    def _build_slang_reference_text(
        self,
        *,
        selected_messages: list[MessageRecord],
        topic_slice_contexts: list[str],
    ) -> str:
        rows: list[str] = []
        ordered_messages = sorted(selected_messages, key=lambda item: item.timestamp)
        tail_messages = ordered_messages[-self.topic_slice_retrieval_query_message_count :]
        for row in tail_messages:
            content = str(row.content or "").strip()
            if content:
                rows.append(content)
        for context in topic_slice_contexts[:3]:
            normalized = str(context or "").strip()
            if normalized:
                rows.append(normalized)
        return "\n".join(rows).strip()

    def _format_guarded_slang_contexts(
        self,
        *,
        records: list[SlangExplanationRecord],
        meta: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        if not records:
            return [], meta
        ordered = sorted(
            records,
            key=lambda item: (float(item.confidence), int(item.updated_at or 0)),
            reverse=True,
        )[: self.slang_injection_limit]

        contexts = [self._format_slang_context(record=item) for item in ordered]
        contexts = [item for item in contexts if item]
        guarded, guard_meta = self._guard_topic_slice_contexts(
            contexts,
            max_chars=self.max_slang_context_chars,
        )
        if bool(guard_meta["truncated"]):
            logger.info(
                "[group_digest.slang] context_truncated selected=%d total=%d selected_chars=%d total_chars=%d",
                int(guard_meta["selected_count"]),
                int(guard_meta["total_count"]),
                int(guard_meta["selected_chars"]),
                int(guard_meta["total_chars"]),
            )
        return guarded, meta

    def _format_slang_context(self, *, record: SlangExplanationRecord) -> str:
        term = str(record.slang_term or "").strip()
        explanation = str(record.explanation or "").strip()
        if not term or not explanation:
            return ""
        usage_context = str(record.usage_context or "").strip() or "无"
        if len(explanation) > 120:
            explanation = f"{explanation[:120]}..."
        if len(usage_context) > 80:
            usage_context = f"{usage_context[:80]}..."
        confidence = float(record.confidence or 0.0)
        return (
            f"slang_term={term}; explanation={explanation}; usage_context={usage_context}; "
            f"confidence={confidence:.2f}; evidence_count={max(0, int(record.evidence_count or 0))}"
        )

    def _collect_topic_slice_contexts(
        self,
        *,
        group_id: str,
        date_label: str,
        time_window: str,
        mode: str,
    ) -> list[str]:
        if not self.enable_topic_slice_contexts:
            return []

        manager = self.topic_segment_manager
        if manager is None:
            return []

        try:
            contexts = manager.collect_slice_contexts(
                group_id=group_id,
                date_label=date_label,
                time_window=time_window,
                mode=mode,
            )
            return contexts if isinstance(contexts, list) else []
        except Exception as exc:
            logger.warning(
                "[group_digest.semantic_input] topic_slice_context_failed group_id=%s mode=%s error=%s",
                group_id,
                mode,
                exc,
            )
            return []

    def _select_tail_messages(
        self,
        *,
        messages: list[MessageRecord],
        max_count: int,
    ) -> list[MessageRecord]:
        ordered = sorted(messages, key=lambda item: item.timestamp)
        if max_count <= 0:
            return ordered
        return ordered[-max_count:]

    def _guard_topic_slice_contexts(
        self,
        contexts: list[str],
        *,
        max_chars: int | None = None,
    ) -> tuple[list[str], dict[str, int | bool]]:
        rows = [str(item).strip() for item in contexts if str(item).strip()]
        total_count = len(rows)
        total_chars = sum(len(item) for item in rows)
        resolved_max_chars = self.max_topic_slice_context_chars if max_chars is None else max(1, int(max_chars))

        if total_chars <= resolved_max_chars:
            return rows, {
                "total_count": total_count,
                "selected_count": total_count,
                "total_chars": total_chars,
                "selected_chars": total_chars,
                "truncated": False,
            }

        selected: list[str] = []
        remaining = resolved_max_chars
        for text in rows:
            if remaining <= 0:
                break
            if len(text) <= remaining:
                selected.append(text)
                remaining -= len(text)
                continue
            selected.append(text[:remaining])
            remaining = 0
            break

        selected_chars = sum(len(item) for item in selected)
        return selected, {
            "total_count": total_count,
            "selected_count": len(selected),
            "total_chars": total_chars,
            "selected_chars": selected_chars,
            "truncated": True,
        }

    def _build_topic_slice_signature(self, contexts: list[str]) -> str:
        if not contexts:
            return ""
        raw = json.dumps(contexts, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _resolve_day_start_ts(self, *, date_label: str) -> int:
        try:
            day = datetime.strptime(date_label, "%Y-%m-%d")
            return int(day.timestamp())
        except Exception:
            now = datetime.now()
            day = datetime(now.year, now.month, now.day)
            return int(day.timestamp())

    def _safe_int(self, value: object, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def _log_material(
        self,
        *,
        group_id: str,
        date_label: str,
        mode: str,
        material: SemanticInputMaterial,
    ) -> None:
        logger.info(
            "[group_digest.semantic_input] group_id=%s date=%s mode=%s source=%s total_effective=%d tail_selected=%d tail_truncated=%s retrieved_slice_count=%d current_day_slice_count=%d topic_slice_total=%d topic_slice_selected=%d topic_slice_total_chars=%d topic_slice_selected_chars=%d topic_slice_truncated=%s retrieval_enabled=%s retrieval_degraded=%s retrieval_query_chars=%d retrieval_query_tail_count=%d retrieval_query_topic_hint_count=%d retrieval_query_mode=%s slang_context_count=%d slang_context_chars=%d slang_candidate_count=%d slang_inferred_count=%d slang_reused_count=%d slang_degraded=%s topic_slice_hit=%s",
            group_id,
            date_label,
            mode,
            material.source,
            material.total_effective_messages,
            material.selected_message_count,
            "true" if material.truncated else "false",
            material.retrieved_topic_slice_count,
            material.current_day_topic_slice_count,
            material.topic_slice_total_count,
            material.topic_slice_selected_count,
            material.topic_slice_total_chars,
            material.topic_slice_selected_chars,
            "true" if material.topic_slice_truncated else "false",
            "true" if material.retrieval_enabled else "false",
            "true" if material.retrieval_degraded else "false",
            material.retrieval_query_chars,
            material.retrieval_query_tail_count,
            material.retrieval_query_topic_hint_count,
            material.retrieval_query_mode or "-",
            material.slang_context_count,
            material.slang_context_chars,
            material.slang_candidate_count,
            material.slang_inferred_count,
            material.slang_reused_count,
            "true" if material.slang_degraded else "false",
            "true" if len(material.topic_slice_contexts) > 0 else "false",
        )

    def describe_extension_point(self) -> dict[str, Any]:
        """用于文档与调试的扩展点说明。"""
        return {
            "current_source": "retrieved_topic_slices_plus_current_day_slices_plus_tail_raw_messages_plus_slang_contexts",
            "future_source": "retrieved_topic_slices_plus_current_day_slices_plus_tail_raw_messages_plus_slang_contexts",
            "future_manager": "GroupTopicSegmentManager",
            "topic_slice_contexts_enabled": self.enable_topic_slice_contexts,
            "topic_slice_context_char_guard": self.max_topic_slice_context_chars,
            "topic_slice_retrieval_enabled": self.enable_topic_slice_retrieval,
            "topic_slice_retrieval_recent_days": self.topic_slice_retrieval_recent_days,
            "topic_slice_retrieval_limit": self.topic_slice_retrieval_limit,
            "topic_slice_retrieval_query_message_count": self.topic_slice_retrieval_query_message_count,
            "scheduled_retrieval_query_strategy": "runtime_active_topics_plus_tail_messages",
            "slang_contexts_enabled": self.enable_slang_contexts,
            "slang_context_char_guard": self.max_slang_context_chars,
            "slang_injection_limit": self.slang_injection_limit,
            "slang_recent_days": self.slang_recent_days,
        }
