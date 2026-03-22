from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from services.digest_service import GroupDigestService
from services.interaction_service import InteractionService
from services.llm_analysis_service import LLMAnalysisService
from services.models import LLMAnalysisConfig, MessageRecord
from services.storage import JsonMessageStorage


class _Resp:
    def __init__(self, text: str):
        self.completion_text = text


class _StubContext:
    def __init__(
        self,
        *,
        current_provider: str = "session_provider",
        responses: list[str] | None = None,
        provider_error: Exception | None = None,
        llm_error: Exception | None = None,
    ):
        self.current_provider = current_provider
        self.responses = list(responses or [])
        self.provider_error = provider_error
        self.llm_error = llm_error
        self.provider_calls = 0
        self.llm_calls: list[tuple[str, str]] = []

    async def get_current_chat_provider_id(self, umo=None):
        self.provider_calls += 1
        if self.provider_error:
            raise self.provider_error
        return self.current_provider

    async def llm_generate(self, chat_provider_id: str, prompt: str):
        self.llm_calls.append((chat_provider_id, prompt))
        if self.llm_error:
            raise self.llm_error
        if not self.responses:
            raise RuntimeError("no mock llm response")
        return _Resp(self.responses.pop(0))


class _ContextWithoutProvider:
    async def llm_generate(self, chat_provider_id: str, prompt: str):
        raise RuntimeError("should not call llm_generate when provider missing")


def _event() -> SimpleNamespace:
    return SimpleNamespace(unified_msg_origin="platform:group:1001")


def _build_service(tmp_path: Path) -> tuple[GroupDigestService, JsonMessageStorage]:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    template_path = Path(__file__).resolve().parents[1] / "templates" / "daily_digest.md.j2"
    service = GroupDigestService(
        storage=storage,
        llm_analysis_service=LLMAnalysisService(),
        interaction_service=InteractionService(),
        template_path=template_path,
    )
    return service, storage


def _valid_analysis_json() -> str:
    return (
        '{"group_topics":["篮球战术","训练安排"],'
        '"member_interests":{"Alice":"关注训练细节","Bob":"关注复盘结论"},'
        '"overall_summary":"群聊集中讨论了训练与复盘安排。",'
        '"suggested_bot_reply":"今天要不要先定一个30分钟训练复盘清单，大家各补一条？"}'
    )


def _base_config(**kwargs) -> LLMAnalysisConfig:
    config = LLMAnalysisConfig(
        use_llm_topic_analysis=True,
        analysis_provider_id="",
        analysis_prompt_template="",
        interaction_prompt_template="",
        max_messages_for_analysis=80,
        fallback_to_stats_only=True,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _run(coro):
    return asyncio.run(coro)


def _append(storage: JsonMessageStorage, record: MessageRecord) -> None:
    _run(storage.append_message(record))


def test_configured_analysis_provider_has_priority(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论篮球战术", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(
        current_provider="session_provider",
        responses=[_valid_analysis_json()],
    )

    _run(
        service.generate_digest_text_for_period(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(analysis_provider_id="custom_provider"),
            source="command_group_digest_today",
        )
    )

    assert context.provider_calls == 0
    assert len(context.llm_calls) == 1
    assert context.llm_calls[0][0] == "custom_provider"


def test_use_session_provider_when_not_configured(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论训练计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_analysis_json()])

    text = _run(
        service.generate_digest_text_for_period(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(),
            source="command_group_digest_today",
        )
    )

    assert context.provider_calls == 1
    assert context.llm_calls[0][0] == "session_provider"
    assert "【热门话题】" in text
    assert "【建议 Bot 主动发言】" in text


def test_provider_unavailable_fallback_to_stats_only(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论训练计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    text = _run(
        service.generate_digest_text_for_period(
            context=_ContextWithoutProvider(),
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(fallback_to_stats_only=True),
            source="command_group_digest_today",
        )
    )

    assert "【统计总览】" in text
    assert "【语义分析状态】" in text
    assert "已降级为仅统计" in text
    assert "【热门话题】" not in text


def test_model_output_parse_failure_fallback(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论训练计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(responses=["not-json-response"])
    text = _run(
        service.generate_digest_text_for_period(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(fallback_to_stats_only=True),
            source="command_group_digest_today",
        )
    )

    assert "已降级为仅统计" in text
    assert "解析失败" in text or "不是合法 JSON" in text


def test_group_digest_and_group_digest_today_share_same_core_pipeline(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)

    # 昨天
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "AI 训练 复盘", int(datetime(2026, 3, 21, 9, 0, 0).timestamp()))
    )
    _append(storage, 
        MessageRecord("group_1001", "u2", "Bob", "AI 战术", int(datetime(2026, 3, 21, 10, 0, 0).timestamp()))
    )
    # 今天
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "AI 训练 复盘", int(datetime(2026, 3, 22, 9, 0, 0).timestamp()))
    )
    _append(storage, 
        MessageRecord("group_1001", "u2", "Bob", "AI 战术", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(
        responses=[
            _valid_analysis_json(),
            _valid_analysis_json(),
        ]
    )
    now = datetime(2026, 3, 22, 20, 0, 0)

    yesterday_report = _run(
        service.build_report_for_period(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=now,
            period="yesterday",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(),
        )
    )
    today_report = _run(
        service.build_report_for_period(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=now,
            period="today",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(),
        )
    )

    assert yesterday_report is not None
    assert today_report is not None
    assert yesterday_report.total_messages == today_report.total_messages
    assert yesterday_report.participant_count == today_report.participant_count
    assert yesterday_report.llm_semantic is not None
    assert today_report.llm_semantic is not None
    assert len(context.llm_calls) == 2

def test_provider_unavailable_without_fallback_returns_clear_error(tmp_path: Path) -> None:
    service, storage = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论训练计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    text = _run(
        service.generate_digest_text_for_period(
            context=_ContextWithoutProvider(),
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
            analysis_config=_base_config(fallback_to_stats_only=False),
            source="command_group_digest_today",
        )
    )

    assert "语义分析不可用，且未启用统计降级" in text
    assert "请检查模型配置后重试" in text
