from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from astrbot.api import logger

from .embedding.base import EmbeddingBackend
from .embedding.noop_backend import NoopEmbeddingBackend
from .embedding_store.base import (
    EmbeddingStore,
    SemanticUnitEmbeddingDocument,
    TopicHeadEmbeddingDocument,
)
from .embedding_store.noop_store import NoopEmbeddingStore
from .models import (
    GroupDayTopicRuntimeState,
    MessageRecord,
    RuntimeTopic,
    SemanticUnitRecord,
    TopicHeadRecord,
)
from .topic_message_filter import classify_topic_message
from .topic_slice_store import TopicSliceStore


TOPIC_STATUS_CREATED = "created"
TOPIC_STATUS_ACTIVE = "active"
TOPIC_STATUS_CLOSED = "closed"

# 兼容旧常量引用：第一版重构后不再使用 cooling。
TOPIC_STATUS_COOLING = "cooling"

DEFAULT_NEW_TOPIC_GAP_SECONDS = 30 * 60
DEFAULT_TOPIC_CLOSE_GAP_SECONDS = 20 * 60
DEFAULT_SINGLE_MESSAGE_TOPIC_TIMEOUT_SECONDS = 15 * 60
DEFAULT_TRANSFER_SIMILARITY_THRESHOLD = 0.75
DEFAULT_TRANSFER_BUFFER_SIZE = 3
DEFAULT_CLOSED_TOPIC_PRUNE_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class SweepSummary:
    scanned_states: int = 0
    scanned_topics: int = 0
    created_topics: int = 0
    closed_transitions: int = 0
    persisted_slices: int = 0
    pruned_topics: int = 0
    pruned_states: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned_states": self.scanned_states,
            "scanned_topics": self.scanned_topics,
            "created_topics": self.created_topics,
            "cooling_transitions": 0,
            "closed_transitions": self.closed_transitions,
            "persisted_slices": self.persisted_slices,
            "pruned_topics": self.pruned_topics,
            "pruned_states": self.pruned_states,
        }


class GroupTopicSegmentManager:
    """群 topic 生命周期中间层（第一版重构）。

    核心机制：
    1. 先过滤低信息量消息，仅有效消息进入 topic 状态机。
    2. topic 生命周期由“有效消息时间间隔 + transfer buffer”驱动。
    3. topic core 一旦创建，默认保持稳定，不做滚动重写。
    4. 不做 reopen / merge / split / 多 topic 竞争打分。
    """

    def __init__(
        self,
        topic_slice_store: TopicSliceStore,
        *,
        enable_topic_embedding: bool = False,
        embedding_backend: EmbeddingBackend | None = None,
        embedding_store: EmbeddingStore | None = None,
        embedding_model: str = "",
        embedding_version: str = "v1",
        new_topic_gap_seconds: int = DEFAULT_NEW_TOPIC_GAP_SECONDS,
        topic_close_gap_seconds: int = DEFAULT_TOPIC_CLOSE_GAP_SECONDS,
        single_message_topic_timeout_seconds: int = DEFAULT_SINGLE_MESSAGE_TOPIC_TIMEOUT_SECONDS,
        transfer_similarity_threshold: float = DEFAULT_TRANSFER_SIMILARITY_THRESHOLD,
        transfer_buffer_size: int = DEFAULT_TRANSFER_BUFFER_SIZE,
        closed_topic_prune_seconds: int = DEFAULT_CLOSED_TOPIC_PRUNE_SECONDS,
    ) -> None:
        self.topic_slice_store = topic_slice_store
        self.enable_topic_embedding = bool(enable_topic_embedding)
        self.embedding_backend: EmbeddingBackend = embedding_backend or NoopEmbeddingBackend()
        self.embedding_store: EmbeddingStore = embedding_store or NoopEmbeddingStore()
        self.embedding_model = str(embedding_model or "").strip()
        self.embedding_version = str(embedding_version or "v1").strip() or "v1"

        self.new_topic_gap_seconds = max(60, int(new_topic_gap_seconds))
        self.topic_close_gap_seconds = max(60, int(topic_close_gap_seconds))
        self.single_message_topic_timeout_seconds = max(60, int(single_message_topic_timeout_seconds))
        self.transfer_similarity_threshold = float(transfer_similarity_threshold)
        self.transfer_buffer_size = max(1, int(transfer_buffer_size))
        self.closed_topic_prune_seconds = max(60, int(closed_topic_prune_seconds))

        self._state_by_group_day: dict[tuple[str, str], GroupDayTopicRuntimeState] = {}
        self._pending_topic_head_embedding_docs: list[TopicHeadEmbeddingDocument] = []
        self._lock = threading.RLock()

    async def ingest_message(self, record: MessageRecord) -> None:
        """仅有效消息参与 topic 生命周期。"""
        group_id = str(record.group_id or "").strip()
        if not group_id:
            return

        filter_result = classify_topic_message(record.content)
        if not filter_result.is_effective:
            logger.debug(
                "[group_digest.topic_segment] skip_low_information group_id=%s message_id=%s reason=%s",
                group_id,
                self._resolve_message_id(record),
                filter_result.reason,
            )
            return

        date_label = self._date_label_from_ts(record.timestamp)
        semantic_pair: tuple[MessageRecord, MessageRecord] | None = None

        with self._lock:
            state = self._get_or_create_state(group_id=group_id, date_label=date_label)
            close_delta = self._close_current_topic_if_gap_unlocked(
                state=state,
                now_ts=record.timestamp,
                gap_seconds=self.topic_close_gap_seconds,
                reason="topic_close_gap",
            )
            if close_delta["closed"] > 0:
                state.transfer_buffer.clear()

            if state.last_effective_message_ts > 0:
                gap = max(0, int(record.timestamp) - int(state.last_effective_message_ts))
                if gap >= self.new_topic_gap_seconds:
                    force_close_delta = self._close_current_topic_if_gap_unlocked(
                        state=state,
                        now_ts=record.timestamp,
                        gap_seconds=0,
                        reason="new_topic_gap",
                    )
                    if force_close_delta["closed"] > 0:
                        state.transfer_buffer.clear()
                    state.pending_effective_messages.clear()
                    logger.info(
                        "[group_digest.topic_segment] new_topic_gap group_id=%s date=%s gap=%d",
                        group_id,
                        date_label,
                        gap,
                    )

            state.last_effective_message_ts = int(record.timestamp)
            state.pending_effective_messages.append(record)
            if len(state.pending_effective_messages) >= 2:
                first = state.pending_effective_messages.pop(0)
                second = state.pending_effective_messages.pop(0)
                semantic_pair = (first, second)

        if semantic_pair is None:
            await self._flush_pending_topic_head_embedding_docs()
            return

        unit = await self.build_semantic_unit_from_messages(*semantic_pair)
        await self._route_semantic_unit(
            group_id=group_id,
            date_label=date_label,
            unit=unit,
        )
        await self._flush_pending_topic_head_embedding_docs()

    async def build_semantic_unit_from_messages(
        self,
        msg_a: MessageRecord,
        msg_b: MessageRecord,
    ) -> SemanticUnitRecord:
        text = (
            f"{msg_a.sender_name}: {msg_a.content}\n"
            f"{msg_b.sender_name}: {msg_b.content}"
        )
        unit_id_seed = (
            f"{msg_a.group_id}|{msg_a.timestamp}|{self._resolve_message_id(msg_a)}|"
            f"{msg_b.timestamp}|{self._resolve_message_id(msg_b)}"
        )
        unit_id = f"unit_{hashlib.sha1(unit_id_seed.encode('utf-8')).hexdigest()[:16]}"
        embedding = await self._embed_text_safe(text)
        participants = self._dedupe_strings(
            [
                self._participant_label(msg_a),
                self._participant_label(msg_b),
            ]
        )
        return SemanticUnitRecord(
            unit_id=unit_id,
            group_id=str(msg_a.group_id),
            date_label=self._date_label_from_ts(min(msg_a.timestamp, msg_b.timestamp)),
            message_ids=[self._resolve_message_id(msg_a), self._resolve_message_id(msg_b)],
            text=text,
            start_ts=min(msg_a.timestamp, msg_b.timestamp),
            end_ts=max(msg_a.timestamp, msg_b.timestamp),
            topic_id="",
            embedding=embedding or [],
            participants=participants,
            embedding_model=self.embedding_model if embedding else "",
            embedding_version=self.embedding_version if embedding else "",
        )

    async def sweep_topics(
        self,
        *,
        now_ts: int | None = None,
        group_id: str | None = None,
        date_label: str | None = None,
        enable_prune: bool = True,
    ) -> dict[str, int]:
        """周期 sweep：在无新消息时推进 close，并清理 runtime 状态。"""
        current_ts = int(now_ts) if now_ts is not None else int(datetime.now().timestamp())
        single_message_plans: list[tuple[str, str, MessageRecord]] = []

        scanned_states = 0
        scanned_topics = 0
        created_topics = 0
        closed_transitions = 0
        persisted_slices = 0
        pruned_topics = 0
        pruned_states = 0

        with self._lock:
            keys = self._collect_state_keys_for_sweep_unlocked(group_id=group_id, date_label=date_label)
            scanned_states = len(keys)
            for key in keys:
                state = self._state_by_group_day.get(key)
                if state is None:
                    continue

                scanned_topics += len(state.topics)
                close_delta = self._close_current_topic_if_gap_unlocked(
                    state=state,
                    now_ts=current_ts,
                    gap_seconds=self.topic_close_gap_seconds,
                    reason="sweep_topic_close_gap",
                )
                closed_transitions += close_delta["closed"]
                persisted_slices += close_delta["persisted_slices"]
                if close_delta["closed"] > 0:
                    state.transfer_buffer.clear()

                if (
                    not state.current_topic_id
                    and len(state.pending_effective_messages) == 1
                ):
                    pending = state.pending_effective_messages[0]
                    wait_seconds = max(0, current_ts - int(pending.timestamp))
                    if wait_seconds >= self.single_message_topic_timeout_seconds:
                        pending_msg = state.pending_effective_messages.pop(0)
                        single_message_plans.append((state.group_id, state.date_label, pending_msg))

                if enable_prune:
                    pruned_topics += self._prune_state_unlocked(state=state, now_ts=current_ts)
                if not state.topics and not state.pending_effective_messages:
                    self._state_by_group_day.pop(key, None)
                    pruned_states += 1

        for plan_group_id, plan_date_label, message in single_message_plans:
            core_text = f"{message.sender_name}: {message.content}"
            core_embedding = await self._embed_text_safe(core_text)
            with self._lock:
                state = self._get_or_create_state(group_id=plan_group_id, date_label=plan_date_label)
                topic = self._create_topic_from_single_message_unlocked(
                    state=state,
                    message=message,
                    core_text=core_text,
                    core_embedding=core_embedding or [],
                )
                close_delta = self._close_topic_unlocked(
                    state=state,
                    topic=topic,
                    close_ts=message.timestamp,
                    reason="single_message_timeout",
                )
                created_topics += 1
                closed_transitions += close_delta["closed"]
                persisted_slices += close_delta["persisted_slices"]

        await self._flush_pending_topic_head_embedding_docs()

        summary = SweepSummary(
            scanned_states=scanned_states,
            scanned_topics=scanned_topics,
            created_topics=created_topics,
            closed_transitions=closed_transitions,
            persisted_slices=persisted_slices,
            pruned_topics=pruned_topics,
            pruned_states=pruned_states,
        ).to_dict()

        if (
            summary["created_topics"] > 0
            or summary["closed_transitions"] > 0
            or summary["persisted_slices"] > 0
            or summary["pruned_topics"] > 0
            or summary["pruned_states"] > 0
        ):
            logger.info(
                "[group_digest.topic_segment] sweep_triggered now_ts=%d scanned_states=%d scanned_topics=%d created_topics=%d closed_transitions=%d persisted_slices=%d pruned_topics=%d pruned_states=%d",
                current_ts,
                summary["scanned_states"],
                summary["scanned_topics"],
                summary["created_topics"],
                summary["closed_transitions"],
                summary["persisted_slices"],
                summary["pruned_topics"],
                summary["pruned_states"],
            )
        else:
            logger.debug(
                "[group_digest.topic_segment] sweep_noop now_ts=%d scanned_states=%d scanned_topics=%d",
                current_ts,
                summary["scanned_states"],
                summary["scanned_topics"],
            )
        return summary

    def collect_slice_contexts(
        self,
        *,
        group_id: str,
        date_label: str,
        time_window: str,
        mode: str,
        limit: int | None = None,
    ) -> list[str]:
        _ = (time_window, mode)
        safe_limit = max(1, int(limit)) if limit is not None else None
        heads = self.topic_slice_store.load_heads(
            group_id=group_id,
            date_label=date_label,
            limit=safe_limit,
        )
        if not heads:
            return []

        contexts: list[str] = []
        for row in heads:
            start_text = datetime.fromtimestamp(row.start_ts).strftime("%H:%M")
            end_text = datetime.fromtimestamp(row.end_ts).strftime("%H:%M")
            participants = "、".join(row.participants[:5]) if row.participants else "无"
            head_text = row.head_text or "无"
            core_text = head_text
            if len(core_text) > 120:
                core_text = f"{core_text[:120]}..."
            contexts.append(
                (
                    f"topic_id={row.topic_id}; time={start_text}-{end_text}; "
                    f"effective_message_count={row.effective_message_count or row.message_count}; "
                    f"participants={participants}; core_text={core_text}; "
                    f"first_message_id={row.first_message_id}; last_message_id={row.last_message_id}"
                )
            )
        return contexts

    def get_day_topics_snapshot(self, *, group_id: str, date_label: str) -> list[dict[str, Any]]:
        with self._lock:
            state = self._state_by_group_day.get((group_id, date_label))
            if state is None:
                return []
            rows = [topic.to_summary_dict() for topic in state.topics.values()]
            rows.sort(key=lambda item: (int(item.get("created_at", 0)), str(item.get("topic_id", ""))))
            return rows

    def describe_extension_point(self) -> dict[str, Any]:
        return {
            "status": "phase1_topic_core_transfer_buffer",
            "states": [TOPIC_STATUS_CREATED, TOPIC_STATUS_ACTIVE, TOPIC_STATUS_CLOSED],
            "routing": "effective_message_sequence_state_machine",
            "semantic_unit": "pairwise_two_effective_messages",
            "transfer_buffer_size": self.transfer_buffer_size,
            "transfer_similarity_threshold": self.transfer_similarity_threshold,
            "topic_close_gap_seconds": self.topic_close_gap_seconds,
            "new_topic_gap_seconds": self.new_topic_gap_seconds,
            "single_message_topic_timeout_seconds": self.single_message_topic_timeout_seconds,
            "runtime_prune_seconds": self.closed_topic_prune_seconds,
            "topic_embedding_enabled": self.enable_topic_embedding,
            "embedding_store_enabled": self.embedding_store.enabled,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
        }

    async def _route_semantic_unit(
        self,
        *,
        group_id: str,
        date_label: str,
        unit: SemanticUnitRecord,
    ) -> None:
        transfer_units: list[SemanticUnitRecord] | None = None
        previous_topic_id = ""
        semantic_unit_docs: list[SemanticUnitEmbeddingDocument] = []
        immediate_return = False

        with self._lock:
            state = self._get_or_create_state(group_id=group_id, date_label=date_label)
            current_topic = self._get_current_topic_unlocked(state)
            if current_topic is None:
                created = self._create_topic_from_units_unlocked(
                    state=state,
                    units=[unit],
                    core_text=unit.text,
                    core_embedding=list(unit.embedding),
                    create_reason="bootstrap_from_first_semantic_unit",
                )
                unit.topic_id = created.topic_id
                doc = self._build_semantic_unit_embedding_doc(unit=unit, topic_id=created.topic_id)
                if doc is not None:
                    semantic_unit_docs.append(doc)
                logger.info(
                    "[group_digest.topic_segment] topic_created group_id=%s date=%s topic_id=%s reason=%s",
                    group_id,
                    date_label,
                    created.topic_id,
                    "bootstrap_from_first_semantic_unit",
                )
                immediate_return = True
            else:
                should_buffer, similarity = self._should_buffer_for_transfer(
                    topic=current_topic,
                    unit=unit,
                )
                if should_buffer:
                    state.transfer_buffer.units.append(unit)
                    logger.info(
                        "[group_digest.topic_segment] transfer_buffer_append group_id=%s date=%s topic_id=%s buffer_size=%d similarity=%.4f threshold=%.4f",
                        group_id,
                        date_label,
                        current_topic.topic_id,
                        len(state.transfer_buffer.units),
                        similarity,
                        self.transfer_similarity_threshold,
                    )
                    if len(state.transfer_buffer.units) < self.transfer_buffer_size:
                        immediate_return = True
                    else:
                        transfer_units = list(state.transfer_buffer.units)
                        state.transfer_buffer.clear()
                        previous_topic_id = current_topic.topic_id
                        close_delta = self._close_topic_unlocked(
                            state=state,
                            topic=current_topic,
                            close_ts=unit.end_ts,
                            reason="transfer_buffer_triggered",
                        )
                        if close_delta["closed"] > 0:
                            logger.info(
                                "[group_digest.topic_segment] topic_closed group_id=%s date=%s topic_id=%s reason=%s",
                                group_id,
                                date_label,
                                current_topic.topic_id,
                                "transfer_buffer_triggered",
                            )
                else:
                    if state.transfer_buffer.units:
                        state.transfer_buffer.clear()
                        logger.info(
                            "[group_digest.topic_segment] transfer_buffer_cleared group_id=%s date=%s topic_id=%s reason=semantic_match",
                            group_id,
                            date_label,
                            current_topic.topic_id,
                        )
                    self._append_semantic_unit_to_topic_unlocked(
                        state=state,
                        topic=current_topic,
                        unit=unit,
                    )
                    unit.topic_id = current_topic.topic_id
                    doc = self._build_semantic_unit_embedding_doc(
                        unit=unit,
                        topic_id=current_topic.topic_id,
                    )
                    if doc is not None:
                        semantic_unit_docs.append(doc)
                    immediate_return = True

        if immediate_return:
            await self._upsert_semantic_unit_docs(semantic_unit_docs)
            return

        if transfer_units is None:
            await self._upsert_semantic_unit_docs(semantic_unit_docs)
            return

        new_core_text = self._join_unit_texts(transfer_units)
        new_core_embedding = await self._embed_text_safe(new_core_text)
        with self._lock:
            state = self._get_or_create_state(group_id=group_id, date_label=date_label)
            new_topic = self._create_topic_from_units_unlocked(
                state=state,
                units=transfer_units,
                core_text=new_core_text,
                core_embedding=new_core_embedding or [],
                create_reason="transfer_buffer_promoted",
            )
            for transfer_unit in transfer_units:
                transfer_unit.topic_id = new_topic.topic_id
                doc = self._build_semantic_unit_embedding_doc(
                    unit=transfer_unit,
                    topic_id=new_topic.topic_id,
                )
                if doc is not None:
                    semantic_unit_docs.append(doc)
            logger.info(
                "[group_digest.topic_segment] topic_transfer_created group_id=%s date=%s previous_topic_id=%s new_topic_id=%s promoted_units=%d",
                group_id,
                date_label,
                previous_topic_id,
                new_topic.topic_id,
                len(transfer_units),
            )
        await self._upsert_semantic_unit_docs(semantic_unit_docs)

    def _should_buffer_for_transfer(
        self,
        *,
        topic: RuntimeTopic,
        unit: SemanticUnitRecord,
    ) -> tuple[bool, float]:
        if not self.enable_topic_embedding:
            return False, 1.0
        if not topic.core_embedding:
            return False, 1.0
        if not unit.embedding:
            return False, 1.0
        similarity = self._cosine_similarity(topic.core_embedding, unit.embedding)
        return similarity < self.transfer_similarity_threshold, similarity

    def _get_or_create_state(self, *, group_id: str, date_label: str) -> GroupDayTopicRuntimeState:
        key = (group_id, date_label)
        state = self._state_by_group_day.get(key)
        if state is None:
            state = GroupDayTopicRuntimeState(
                group_id=group_id,
                date_label=date_label,
            )
            self._state_by_group_day[key] = state
        return state

    def _get_current_topic_unlocked(self, state: GroupDayTopicRuntimeState) -> RuntimeTopic | None:
        topic_id = str(state.current_topic_id or "").strip()
        if not topic_id:
            return None
        topic = state.topics.get(topic_id)
        if topic is None:
            state.current_topic_id = ""
            return None
        if topic.status == TOPIC_STATUS_CLOSED:
            state.current_topic_id = ""
            return None
        return topic

    def _create_topic_from_units_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        units: list[SemanticUnitRecord],
        core_text: str,
        core_embedding: list[float],
        create_reason: str,
    ) -> RuntimeTopic:
        topic_id = f"{state.date_label.replace('-', '')}_{state.next_topic_index:04d}"
        state.next_topic_index += 1

        created_at = min(unit.start_ts for unit in units)
        last_active_at = max(unit.end_ts for unit in units)
        topic = RuntimeTopic(
            topic_id=topic_id,
            group_id=state.group_id,
            date_label=state.date_label,
            status=TOPIC_STATUS_CREATED,
            created_at=created_at,
            last_active_at=last_active_at,
            core_text=core_text,
            core_embedding=list(core_embedding or []),
            core_embedding_model=self.embedding_model if core_embedding else "",
            core_embedding_version=self.embedding_version if core_embedding else "",
        )
        for unit in units:
            self._append_semantic_unit_to_topic_unlocked(
                state=state,
                topic=topic,
                unit=unit,
            )
        topic.status = TOPIC_STATUS_ACTIVE
        state.topics[topic.topic_id] = topic
        state.current_topic_id = topic.topic_id
        logger.info(
            "[group_digest.topic_segment] topic_created group_id=%s date=%s topic_id=%s reason=%s message_count=%d",
            state.group_id,
            state.date_label,
            topic.topic_id,
            create_reason,
            topic.message_count,
        )
        return topic

    def _create_topic_from_single_message_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        message: MessageRecord,
        core_text: str,
        core_embedding: list[float],
    ) -> RuntimeTopic:
        topic_id = f"{state.date_label.replace('-', '')}_{state.next_topic_index:04d}"
        state.next_topic_index += 1
        message_id = self._resolve_message_id(message)
        participant = self._participant_label(message)
        topic = RuntimeTopic(
            topic_id=topic_id,
            group_id=state.group_id,
            date_label=state.date_label,
            status=TOPIC_STATUS_CREATED,
            created_at=message.timestamp,
            last_active_at=message.timestamp,
            first_message_id=message_id,
            last_message_id=message_id,
            message_ids=[message_id],
            message_count=1,
            effective_message_count=1,
            participants=[participant] if participant else [],
            core_text=core_text,
            core_embedding=list(core_embedding or []),
            core_embedding_model=self.embedding_model if core_embedding else "",
            core_embedding_version=self.embedding_version if core_embedding else "",
            core_message_ids=[message_id],
        )
        state.topics[topic.topic_id] = topic
        state.current_topic_id = ""
        logger.info(
            "[group_digest.topic_segment] topic_created group_id=%s date=%s topic_id=%s reason=%s message_count=1",
            state.group_id,
            state.date_label,
            topic.topic_id,
            "single_message_timeout",
        )
        return topic

    def _append_semantic_unit_to_topic_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        topic: RuntimeTopic,
        unit: SemanticUnitRecord,
    ) -> None:
        unit.topic_id = topic.topic_id
        state.semantic_units[unit.unit_id] = unit
        topic.last_active_at = max(int(topic.last_active_at), int(unit.end_ts))
        topic.semantic_unit_ids = self._append_unique(topic.semantic_unit_ids, unit.unit_id)
        topic.participants = self._dedupe_strings(topic.participants + list(unit.participants))

        for message_id in unit.message_ids:
            topic.message_ids = self._append_unique(topic.message_ids, message_id)
            if not topic.first_message_id:
                topic.first_message_id = message_id
            topic.last_message_id = message_id

        topic.message_count = len(topic.message_ids)
        topic.effective_message_count = topic.message_count
        if not topic.core_message_ids:
            topic.core_message_ids = list(unit.message_ids)

    def _close_current_topic_if_gap_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        now_ts: int,
        gap_seconds: int,
        reason: str,
    ) -> dict[str, int]:
        topic = self._get_current_topic_unlocked(state)
        if topic is None:
            return {"closed": 0, "persisted_slices": 0}
        idle_seconds = max(0, int(now_ts) - int(topic.last_active_at))
        if idle_seconds < max(0, int(gap_seconds)):
            return {"closed": 0, "persisted_slices": 0}
        return self._close_topic_unlocked(
            state=state,
            topic=topic,
            close_ts=topic.last_active_at,
            reason=reason,
        )

    def _close_topic_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        topic: RuntimeTopic,
        close_ts: int,
        reason: str,
    ) -> dict[str, int]:
        if topic.status == TOPIC_STATUS_CLOSED:
            return {"closed": 0, "persisted_slices": 0}

        topic.status = TOPIC_STATUS_CLOSED
        topic.closed_at = int(close_ts)
        if state.current_topic_id == topic.topic_id:
            state.current_topic_id = ""

        persisted = 1 if self._persist_closed_topic_unlocked(state=state, topic=topic) else 0
        logger.info(
            "[group_digest.topic_segment] topic_closed group_id=%s date=%s topic_id=%s reason=%s message_count=%d",
            topic.group_id,
            topic.date_label,
            topic.topic_id,
            reason,
            topic.message_count,
        )
        return {"closed": 1, "persisted_slices": persisted}

    def _persist_closed_topic_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        topic: RuntimeTopic,
    ) -> bool:
        if topic.slice_persisted:
            return False
        if topic.message_count <= 0:
            topic.slice_persisted = True
            return False

        topic_units = self._collect_topic_units_unlocked(state=state, topic=topic)
        head_embedding = self._build_head_embedding(topic_units)
        if not head_embedding and topic.core_embedding:
            head_embedding = list(topic.core_embedding)
        head_text = self._build_head_text(topic_units=topic_units, fallback_text=topic.core_text)
        row = TopicHeadRecord(
            group_id=topic.group_id,
            date_label=topic.date_label,
            topic_id=topic.topic_id,
            start_ts=topic.created_at,
            end_ts=topic.last_active_at,
            message_count=topic.message_count,
            effective_message_count=topic.effective_message_count,
            participants=list(topic.participants),
            recent_keywords=[],
            message_ids=list(topic.message_ids),
            semantic_unit_ids=list(topic.semantic_unit_ids),
            first_message_id=topic.first_message_id,
            last_message_id=topic.last_message_id,
            head_text=head_text,
            head_embedding=list(head_embedding),
            semantic_unit_count=len(topic.semantic_unit_ids),
            head_embedding_model=self.embedding_model if head_embedding else "",
            head_embedding_version=self.embedding_version if head_embedding else "",
        )
        self.topic_slice_store.append_head(row)
        topic_head_doc = self._build_topic_head_embedding_doc(row=row)
        if topic_head_doc is not None:
            self._pending_topic_head_embedding_docs.append(topic_head_doc)
        topic.slice_persisted = True
        logger.info(
            "[group_digest.topic_segment] topic_head_persisted group_id=%s date=%s topic_id=%s message_count=%d semantic_unit_count=%d",
            topic.group_id,
            topic.date_label,
            topic.topic_id,
            topic.message_count,
            len(topic.semantic_unit_ids),
        )
        return True

    def _prune_state_unlocked(self, *, state: GroupDayTopicRuntimeState, now_ts: int) -> int:
        to_remove: list[str] = []
        for topic_id, topic in state.topics.items():
            if topic.status != TOPIC_STATUS_CLOSED:
                continue
            if not topic.slice_persisted:
                continue
            anchor = int(topic.closed_at or topic.last_active_at)
            idle_seconds = max(0, int(now_ts) - anchor)
            if idle_seconds < self.closed_topic_prune_seconds:
                continue
            to_remove.append(topic_id)

        for topic_id in to_remove:
            topic = state.topics.pop(topic_id, None)
            if topic is not None:
                for unit_id in topic.semantic_unit_ids:
                    state.semantic_units.pop(unit_id, None)
            logger.info(
                "[group_digest.topic_segment] topic_pruned group_id=%s date=%s topic_id=%s",
                state.group_id,
                state.date_label,
                topic_id,
            )
        return len(to_remove)

    def _collect_state_keys_for_sweep_unlocked(
        self,
        *,
        group_id: str | None,
        date_label: str | None,
    ) -> list[tuple[str, str]]:
        if group_id is None and date_label is None:
            return sorted(self._state_by_group_day.keys())
        result: list[tuple[str, str]] = []
        for key in sorted(self._state_by_group_day.keys()):
            key_group, key_date = key
            if group_id is not None and key_group != group_id:
                continue
            if date_label is not None and key_date != date_label:
                continue
            result.append(key)
        return result

    def _collect_topic_units_unlocked(
        self,
        *,
        state: GroupDayTopicRuntimeState,
        topic: RuntimeTopic,
    ) -> list[SemanticUnitRecord]:
        rows: list[SemanticUnitRecord] = []
        for unit_id in topic.semantic_unit_ids:
            unit = state.semantic_units.get(unit_id)
            if unit is None:
                continue
            rows.append(unit)
        rows.sort(key=lambda item: (int(item.start_ts), str(item.unit_id)))
        return rows

    def _build_head_text(
        self,
        *,
        topic_units: list[SemanticUnitRecord],
        fallback_text: str,
    ) -> str:
        if not topic_units:
            return str(fallback_text or "").strip()
        rows: list[str] = []
        for unit in topic_units[:4]:
            text = str(unit.text or "").strip()
            if not text:
                continue
            rows.append(text)
        joined = "\n\n".join(rows).strip()
        if joined:
            return joined
        return str(fallback_text or "").strip()

    def _build_head_embedding(self, topic_units: list[SemanticUnitRecord]) -> list[float]:
        vectors = [list(unit.embedding) for unit in topic_units if unit.embedding]
        if not vectors:
            return []
        dim = len(vectors[0])
        if dim <= 0:
            return []
        filtered: list[list[float]] = [vec for vec in vectors if len(vec) == dim]
        if not filtered:
            return []

        total = [0.0] * dim
        for vec in filtered:
            for idx, value in enumerate(vec):
                total[idx] += float(value)
        count = float(len(filtered))
        mean_vector = [value / count for value in total]
        return self._normalize_vector(mean_vector)

    def _normalize_vector(self, values: list[float]) -> list[float]:
        if not values:
            return []
        norm = math.sqrt(sum(float(item) * float(item) for item in values))
        if norm <= 0:
            return []
        return [float(item) / norm for item in values]

    async def _upsert_semantic_unit_docs(
        self,
        docs: list[SemanticUnitEmbeddingDocument],
    ) -> None:
        if not docs:
            return
        if not self.embedding_store.enabled:
            return
        for doc in docs:
            try:
                await self.embedding_store.upsert_semantic_unit(doc)
            except Exception as exc:
                logger.warning(
                    "[group_digest.embedding_store] semantic_unit_upsert_failed point_id=%s error=%s",
                    doc.point_id,
                    exc,
                )

    async def _flush_pending_topic_head_embedding_docs(self) -> None:
        if not self.embedding_store.enabled:
            with self._lock:
                if self._pending_topic_head_embedding_docs:
                    self._pending_topic_head_embedding_docs.clear()
            return

        with self._lock:
            docs = list(self._pending_topic_head_embedding_docs)
            self._pending_topic_head_embedding_docs.clear()
        if not docs:
            return

        for doc in docs:
            try:
                upsert_topic_head = getattr(self.embedding_store, "upsert_topic_head", None)
                if callable(upsert_topic_head):
                    await upsert_topic_head(doc)
                else:
                    await self.embedding_store.upsert_topic_slice(doc)
            except Exception as exc:
                logger.warning(
                    "[group_digest.embedding_store] topic_head_upsert_failed point_id=%s error=%s",
                    doc.point_id,
                    exc,
                )

    def _build_semantic_unit_embedding_doc(
        self,
        *,
        unit: SemanticUnitRecord,
        topic_id: str,
    ) -> SemanticUnitEmbeddingDocument | None:
        if not unit.embedding:
            return None
        resolved_topic_id = str(topic_id or "").strip()
        payload = {
            "object_type": "semantic_unit",
            "group_id": unit.group_id,
            "date_label": unit.date_label,
            "topic_id": resolved_topic_id,
            "semantic_unit_id": unit.unit_id,
            "start_ts": int(unit.start_ts),
            "end_ts": int(unit.end_ts),
            "message_ids": list(unit.message_ids),
            "text": unit.text,
            "unit_text": unit.text,
            "embedding_model": unit.embedding_model or self.embedding_model,
            "embedding_version": unit.embedding_version or self.embedding_version,
        }
        point_id = self._semantic_unit_point_id(unit=unit)
        return SemanticUnitEmbeddingDocument(
            point_id=point_id,
            vector=list(unit.embedding),
            payload=payload,
        )

    def _build_topic_head_embedding_doc(
        self,
        *,
        row: TopicHeadRecord,
    ) -> TopicHeadEmbeddingDocument | None:
        if not row.head_embedding:
            return None
        payload = {
            "object_type": "topic_head",
            "group_id": row.group_id,
            "date_label": row.date_label,
            "topic_id": row.topic_id,
            "start_ts": int(row.start_ts),
            "end_ts": int(row.end_ts),
            "first_message_id": row.first_message_id,
            "last_message_id": row.last_message_id,
            "message_count": int(row.message_count),
            "effective_message_count": int(row.effective_message_count),
            "participants": list(row.participants),
            "recent_keywords": list(row.recent_keywords),
            "head_text": row.head_text,
            "message_ids": list(row.message_ids),
            "semantic_unit_ids": list(row.semantic_unit_ids),
            "embedding_model": row.head_embedding_model or self.embedding_model,
            "embedding_version": row.head_embedding_version or self.embedding_version,
        }
        point_id = self._topic_head_point_id(
            group_id=row.group_id,
            date_label=row.date_label,
            topic_id=row.topic_id,
        )
        return TopicHeadEmbeddingDocument(
            point_id=point_id,
            vector=list(row.head_embedding),
            payload=payload,
        )

    def _semantic_unit_point_id(self, *, unit: SemanticUnitRecord) -> str:
        message_ids = "|".join(sorted(str(item).strip() for item in unit.message_ids if str(item).strip()))
        seed = (
            f"semantic_unit|{unit.group_id}|{unit.date_label}|{message_ids}|"
            f"{int(unit.start_ts)}|{int(unit.end_ts)}"
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return f"su_{digest}"

    def _topic_head_point_id(self, *, group_id: str, date_label: str, topic_id: str) -> str:
        seed = f"topic_head|{group_id}|{date_label}|{topic_id}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return f"ts_{digest}"

    async def _embed_text_safe(self, text: str) -> list[float] | None:
        if not self.enable_topic_embedding:
            return None
        try:
            vector = await self.embedding_backend.embed_text(text)
            if not vector:
                return None
            return [float(item) for item in vector]
        except Exception as exc:
            logger.warning(
                "[group_digest.topic_segment] embedding_failed error=%s",
                exc,
            )
            return None

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if norm_left <= 0 or norm_right <= 0:
            return 0.0
        return dot / (norm_left * norm_right)

    def _join_unit_texts(self, units: list[SemanticUnitRecord]) -> str:
        return "\n\n".join(unit.text for unit in units if str(unit.text).strip())

    def _append_unique(self, rows: list[str], value: str) -> list[str]:
        text = str(value or "").strip()
        if not text:
            return list(rows)
        if text in rows:
            return list(rows)
        result = list(rows)
        result.append(text)
        return result

    def _dedupe_strings(self, rows: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in rows:
            text = str(item or "").strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _participant_label(self, message: MessageRecord) -> str:
        sender_name = str(message.sender_name or "").strip()
        sender_id = str(message.sender_id or "").strip()
        if sender_name and sender_id:
            return f"{sender_name}({sender_id})"
        if sender_name:
            return sender_name
        if sender_id:
            return sender_id
        return "unknown_sender"

    def _resolve_message_id(self, record: MessageRecord) -> str:
        message_id = str(record.message_id or "").strip()
        if message_id:
            return message_id
        seed = f"{record.group_id}|{record.sender_id}|{record.timestamp}|{record.content}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"fallback_{digest}"

    def _date_label_from_ts(self, timestamp: int) -> str:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
