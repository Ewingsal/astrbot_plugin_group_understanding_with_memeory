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
