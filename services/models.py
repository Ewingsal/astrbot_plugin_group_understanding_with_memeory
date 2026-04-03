from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from astrbot.api import logger


@dataclass
class MessageRecord:
    group_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: int
    message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageRecord | None":
        if not isinstance(data, dict):
            logger.warning(
                "[group_digest.model] invalid_message_record_type expected=dict got=%s",
                type(data).__name__,
            )
            return None

        timestamp = cls._safe_int(data.get("timestamp", 0), field="message.timestamp")
        return cls(
            group_id=str(data.get("group_id", "")),
            sender_id=str(data.get("sender_id", "unknown_sender")),
            sender_name=str(data.get("sender_name", "未知成员")),
            content=str(data.get("content", "")),
            timestamp=timestamp,
            message_id=str(data.get("message_id", "")),
        )

    @staticmethod
    def _safe_int(value: object, *, field: str, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.model] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default


@dataclass
class TopicHeadRecord:
    group_id: str
    date_label: str
    topic_id: str
    start_ts: int
    end_ts: int
    message_count: int
    effective_message_count: int = 0
    participants: list[str] = field(default_factory=list)
    recent_keywords: list[str] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    semantic_unit_ids: list[str] = field(default_factory=list)
    first_message_id: str = ""
    last_message_id: str = ""
    head_text: str = ""
    core_text: str = ""
    head_embedding: list[float] = field(default_factory=list)
    semantic_unit_count: int = 0
    head_embedding_model: str = ""
    head_embedding_version: str = ""
    core_embedding_model: str = ""
    core_embedding_version: str = ""

    def __post_init__(self) -> None:
        if not self.head_text and self.core_text:
            self.head_text = str(self.core_text)
        elif not self.core_text and self.head_text:
            self.core_text = str(self.head_text)

        if not self.head_embedding_model and self.core_embedding_model:
            self.head_embedding_model = str(self.core_embedding_model)
        elif not self.core_embedding_model and self.head_embedding_model:
            self.core_embedding_model = str(self.head_embedding_model)

        if not self.head_embedding_version and self.core_embedding_version:
            self.head_embedding_version = str(self.core_embedding_version)
        elif not self.core_embedding_version and self.head_embedding_version:
            self.core_embedding_version = str(self.head_embedding_version)

        if self.semantic_unit_count <= 0 and self.semantic_unit_ids:
            self.semantic_unit_count = len(self.semantic_unit_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # 兼容旧字段，便于平滑迁移。
        payload["head_text"] = self.head_text or self.core_text
        payload["core_text"] = self.core_text or self.head_text
        payload["head_embedding_model"] = self.head_embedding_model or self.core_embedding_model
        payload["head_embedding_version"] = self.head_embedding_version or self.core_embedding_version
        payload["core_embedding_model"] = self.core_embedding_model or self.head_embedding_model
        payload["core_embedding_version"] = self.core_embedding_version or self.head_embedding_version
        payload["semantic_unit_count"] = (
            int(self.semantic_unit_count)
            if int(self.semantic_unit_count) > 0
            else len(self.semantic_unit_ids)
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicHeadRecord | None":
        if not isinstance(data, dict):
            logger.warning(
                "[group_digest.model] invalid_topic_head_type expected=dict got=%s",
                type(data).__name__,
            )
            return None

        participants = cls._normalize_string_list(data.get("participants", []))
        recent_keywords = cls._normalize_string_list(data.get("recent_keywords", []))
        message_ids = cls._normalize_string_list(data.get("message_ids", []))
        semantic_unit_ids = cls._normalize_string_list(data.get("semantic_unit_ids", []))
        message_count = max(
            0,
            cls._safe_int(data.get("message_count", 0), field="topic_head.message_count"),
        )
        effective_message_count = max(
            0,
            cls._safe_int(
                data.get("effective_message_count", message_count),
                field="topic_head.effective_message_count",
            ),
        )
        head_text = str(data.get("head_text", "")).strip() or str(data.get("core_text", "")).strip()
        head_embedding = cls._normalize_float_list(data.get("head_embedding", []))
        semantic_unit_count = max(
            0,
            cls._safe_int(
                data.get("semantic_unit_count", len(semantic_unit_ids)),
                field="topic_head.semantic_unit_count",
            ),
        )
        return cls(
            group_id=str(data.get("group_id", "")).strip(),
            date_label=str(data.get("date_label", "")).strip(),
            topic_id=str(data.get("topic_id", "")).strip(),
            start_ts=cls._safe_int(data.get("start_ts", 0), field="topic_head.start_ts"),
            end_ts=cls._safe_int(data.get("end_ts", 0), field="topic_head.end_ts"),
            message_count=message_count,
            effective_message_count=effective_message_count,
            participants=participants,
            recent_keywords=recent_keywords,
            message_ids=message_ids,
            semantic_unit_ids=semantic_unit_ids,
            first_message_id=str(data.get("first_message_id", "")).strip(),
            last_message_id=str(data.get("last_message_id", "")).strip(),
            head_text=head_text,
            core_text=head_text,
            head_embedding=head_embedding,
            semantic_unit_count=semantic_unit_count,
            head_embedding_model=str(data.get("head_embedding_model", "")).strip()
            or str(data.get("core_embedding_model", "")).strip(),
            head_embedding_version=str(data.get("head_embedding_version", "")).strip()
            or str(data.get("core_embedding_version", "")).strip(),
            core_embedding_model=str(data.get("core_embedding_model", "")).strip()
            or str(data.get("head_embedding_model", "")).strip(),
            core_embedding_version=str(data.get("core_embedding_version", "")).strip()
            or str(data.get("head_embedding_version", "")).strip(),
        )

    @staticmethod
    def _safe_int(value: object, *, field: str, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.model] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default

    @staticmethod
    def _normalize_float_list(value: object) -> list[float]:
        if not isinstance(value, list):
            return []
        result: list[float] = []
        for item in value:
            try:
                result.append(float(item))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _normalize_string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

@dataclass
class TopicSliceRecord(TopicHeadRecord):
    """兼容旧命名：TopicSliceRecord 等同于 TopicHeadRecord。"""


@dataclass
class SlangExplanationRecord:
    group_id: str
    slang_term: str
    explanation: str
    usage_context: str
    confidence: float
    evidence_count: int
    source_slice_ids: list[str] = field(default_factory=list)
    source_semantic_unit_ids: list[str] = field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlangExplanationRecord | None":
        if not isinstance(data, dict):
            logger.warning(
                "[group_digest.model] invalid_slang_record_type expected=dict got=%s",
                type(data).__name__,
            )
            return None

        confidence = cls._safe_float(
            data.get("confidence", 0.0),
            field="slang.confidence",
        )
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        return cls(
            group_id=str(data.get("group_id", "")).strip(),
            slang_term=str(data.get("slang_term", "")).strip(),
            explanation=str(data.get("explanation", "")).strip(),
            usage_context=str(data.get("usage_context", "")).strip(),
            confidence=confidence,
            evidence_count=max(
                0,
                cls._safe_int(
                    data.get("evidence_count", 0),
                    field="slang.evidence_count",
                ),
            ),
            source_slice_ids=cls._normalize_string_list(data.get("source_slice_ids", [])),
            source_semantic_unit_ids=cls._normalize_string_list(
                data.get("source_semantic_unit_ids", [])
            ),
            created_at=max(
                0,
                cls._safe_int(data.get("created_at", 0), field="slang.created_at"),
            ),
            updated_at=max(
                0,
                cls._safe_int(data.get("updated_at", 0), field="slang.updated_at"),
            ),
        )

    @staticmethod
    def _safe_int(value: object, *, field: str, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.model] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default

    @staticmethod
    def _safe_float(value: object, *, field: str, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.model] invalid_float field=%s value=%r fallback=%.2f",
                field,
                value,
                default,
            )
            return default

    @staticmethod
    def _normalize_string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result


@dataclass
class SemanticUnitRecord:
    unit_id: str
    group_id: str
    date_label: str
    message_ids: list[str]
    text: str
    start_ts: int
    end_ts: int
    topic_id: str = ""
    embedding: list[float] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    embedding_model: str = ""
    embedding_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["semantic_unit_id"] = self.unit_id
        payload["unit_text"] = self.text
        return payload

    @property
    def semantic_unit_id(self) -> str:
        return self.unit_id

    @semantic_unit_id.setter
    def semantic_unit_id(self, value: str) -> None:
        self.unit_id = str(value or "").strip()

    @property
    def unit_text(self) -> str:
        return self.text

    @unit_text.setter
    def unit_text(self, value: str) -> None:
        self.text = str(value or "")


@dataclass
class TransferBufferState:
    units: list[SemanticUnitRecord] = field(default_factory=list)

    def clear(self) -> None:
        self.units.clear()


@dataclass
class RuntimeTopic:
    topic_id: str
    group_id: str
    date_label: str
    status: str
    created_at: int
    last_active_at: int
    first_message_id: str = ""
    last_message_id: str = ""
    message_ids: list[str] = field(default_factory=list)
    message_count: int = 0
    effective_message_count: int = 0
    participants: list[str] = field(default_factory=list)
    core_text: str = ""
    core_embedding: list[float] = field(default_factory=list)
    core_embedding_model: str = ""
    core_embedding_version: str = ""
    core_message_ids: list[str] = field(default_factory=list)
    semantic_unit_ids: list[str] = field(default_factory=list)
    slice_persisted: bool = False
    closed_at: int = 0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "group_id": self.group_id,
            "date_label": self.date_label,
            "status": self.status,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "first_message_id": self.first_message_id,
            "last_message_id": self.last_message_id,
            "message_ids": list(self.message_ids),
            "message_count": self.message_count,
            "effective_message_count": self.effective_message_count,
            "participants": list(self.participants),
            "core_text": self.core_text,
            "core_embedding_model": self.core_embedding_model,
            "core_embedding_version": self.core_embedding_version,
            "core_message_ids": list(self.core_message_ids),
            "semantic_unit_ids": list(self.semantic_unit_ids),
            "slice_persisted": self.slice_persisted,
            "closed_at": self.closed_at,
        }


@dataclass
class GroupDayTopicRuntimeState:
    group_id: str
    date_label: str
    next_topic_index: int = 1
    topics: dict[str, RuntimeTopic] = field(default_factory=dict)
    semantic_units: dict[str, SemanticUnitRecord] = field(default_factory=dict)
    current_topic_id: str = ""
    pending_effective_messages: list[MessageRecord] = field(default_factory=list)
    transfer_buffer: TransferBufferState = field(default_factory=TransferBufferState)
    last_effective_message_ts: int = 0


@dataclass
class MemberDigest:
    sender_id: str
    sender_name: str
    message_count: int


@dataclass
class LLMSemanticResult:
    group_topics: list[str]
    member_interests: dict[str, str]
    overall_summary: str
    suggested_bot_reply: str


@dataclass
class LLMAnalysisConfig:
    use_llm_topic_analysis: bool = True
    analysis_provider_id: str = ""
    analysis_prompt_template: str = ""
    interaction_prompt_template: str = ""
    max_messages_for_analysis: int = 80
    fallback_to_stats_only: bool = True


@dataclass
class SchedulerConfig:
    enable_scheduled_proactive_message: bool = False
    scheduled_send_hour: int = 18
    scheduled_send_minute: int = 0
    scheduled_mode: str = "today_until_scheduled_time"
    store_group_origin: bool = True
    scheduled_group_whitelist_enabled: bool = False
    scheduled_group_whitelist: list[str] = field(default_factory=list)
    scheduled_send_timezone: str = "Asia/Shanghai"


@dataclass
class DigestReport:
    period: str
    date_label: str
    time_window: str
    group_id: str
    total_messages: int
    participant_count: int
    active_members: list[MemberDigest]
    llm_semantic: LLMSemanticResult | None = None
    stats_only: bool = False
    analysis_notice: str = ""
    analysis_provider_id: str = ""
