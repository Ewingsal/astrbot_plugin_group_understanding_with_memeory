from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from services.digest_service import GroupDigestService
from services.interaction_service import InteractionService
from services.llm_analysis_service import LLMAnalysisService
from services.models import LLMAnalysisConfig, MessageRecord
from services.report_cache_store import ReportCacheStore
from services.storage import JsonMessageStorage


class _Resp:
    def __init__(self, text: str):
        self.completion_text = text


class _StubContext:
    def __init__(self, *, provider_id: str = "session_provider", responses: list[str] | None = None):
        self.provider_id = provider_id
        self.responses = list(responses or [])
        self.llm_calls = 0
        self.prompts: list[str] = []

    async def get_current_chat_provider_id(self, umo=None):
        return self.provider_id

    async def llm_generate(self, chat_provider_id: str, prompt: str):
        self.llm_calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no mock llm response")
        return _Resp(self.responses.pop(0))


def _valid_unified_json(reply: str = "大家要不要把今天的重点结论整理成三条？") -> str:
    return (
        '{"group_topics":["训练计划","复盘安排"],'
        '"member_interests":{"Alice":"关注训练细节"},'
        '"overall_summary":"大家围绕训练和复盘展开讨论。",'
        f'"suggested_bot_reply":"{reply}"'
        "}"
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


def _build_service(tmp_path: Path) -> tuple[GroupDigestService, JsonMessageStorage, ReportCacheStore]:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    cache_store = ReportCacheStore(tmp_path / "report_cache.json", cache_version=1)
    template_path = Path(__file__).resolve().parents[1] / "templates" / "daily_digest.md.j2"
    service = GroupDigestService(
        storage=storage,
        llm_analysis_service=LLMAnalysisService(),
        interaction_service=InteractionService(),
        template_path=template_path,
        report_cache_store=cache_store,
        cache_version=1,
    )
    return service, storage, cache_store


def _event() -> SimpleNamespace:
    return SimpleNamespace(unified_msg_origin="platform:group:1001")


def _run(coro):
    return asyncio.run(coro)


def _append(storage: JsonMessageStorage, record: MessageRecord) -> None:
    _run(storage.append_message(record))


def test_cache_hit_without_new_messages(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "今天讨论训练计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context_first = _StubContext(responses=[_valid_unified_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context_first,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )
    assert context_first.llm_calls == 1

    # 第二次命令消息与上一条日报输出（历史脏数据）不应进入有效消息集合，也不应导致缓存失效。
    _append(storage, 
        MessageRecord(
            "group_1001",
            "u1",
            "Alice",
            "/group_digest_today",
            int(datetime(2026, 3, 22, 12, 59, 0).timestamp()),
        )
    )
    _append(storage, 
        MessageRecord(
            "group_1001",
            "bot_1",
            "Bot",
            "群聊兴趣日报（2026-03-22）\n统计日期：2026-03-22\n统计范围：2026-03-22 00:00 - 2026-03-22 12:00\n群组：group_1001",
            int(datetime(2026, 3, 22, 12, 59, 30).timestamp()),
        )
    )

    context_second = _StubContext(responses=[])
    report, _metrics = _run(
        service.build_report_for_period_with_metrics(
            context=context_second,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 13, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    assert report is not None
    assert context_second.llm_calls == 0
    assert report.total_messages == 1


def test_new_messages_trigger_full_rebuild(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "第一条消息", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json(), _valid_unified_json("第二次重算文案")])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    _append(storage, 
        MessageRecord("group_1001", "u2", "Bob", "第二条消息", int(datetime(2026, 3, 22, 12, 30, 0).timestamp()))
    )

    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 13, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    assert context.llm_calls == 2


def test_mode_isolation_no_cross_cache(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "同一天消息", int(datetime(2026, 3, 22, 9, 0, 0).timestamp()))
    )
    context = _StubContext(responses=[_valid_unified_json(), _valid_unified_json("scheduled文案")])

    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 18, 0, 0),
            period="today",
            mode="scheduled",
            source="scheduler",
            analysis_config=_base_config(),
        )
    )

    assert context.llm_calls == 2


def test_group_digest_command_message_does_not_invalidate_today_cache(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "有效聊天", int(datetime(2026, 3, 22, 9, 0, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    _append(storage, 
        MessageRecord(
            "group_1001",
            "u2",
            "Bob",
            "/group_digest",
            int(datetime(2026, 3, 22, 12, 5, 0).timestamp()),
        )
    )

    context2 = _StubContext(responses=[])
    report, _metrics = _run(
        service.build_report_for_period_with_metrics(
            context=context2,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 6, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    assert report is not None
    assert report.total_messages == 1
    assert context2.llm_calls == 0


def test_group_isolation_no_cross_cache(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_a", "u1", "Alice", "A 群消息", int(datetime(2026, 3, 22, 9, 0, 0).timestamp()))
    )
    _append(storage, 
        MessageRecord("group_a", "u1", "Alice", "/group_digest_today", int(datetime(2026, 3, 22, 9, 1, 0).timestamp()))
    )
    _append(storage, 
        MessageRecord("group_b", "u2", "Bob", "B 群消息", int(datetime(2026, 3, 22, 9, 5, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json(), _valid_unified_json("B群文案")])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_a",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_b",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    assert context.llm_calls == 2


def test_group_digest_debug_today_not_in_llm_input(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "先聊一个正常话题", int(datetime(2026, 3, 22, 9, 0, 0).timestamp()))
    )
    _append(storage, 
        MessageRecord(
            "group_1001",
            "u2",
            "Bob",
            "/group_digest_debug_today",
            int(datetime(2026, 3, 22, 9, 1, 0).timestamp()),
        )
    )

    context = _StubContext(responses=[_valid_unified_json()])
    report, _metrics = _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 10, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    assert report is not None
    assert report.total_messages == 1
    assert context.llm_calls == 1
    assert context.prompts
    assert "/group_digest_debug_today" not in context.prompts[0]


def test_provider_or_max_messages_change_invalidates_cache(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "固定消息", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json(), _valid_unified_json("provider changed")])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(analysis_provider_id="provider_a"),
        )
    )
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 13, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(analysis_provider_id="provider_b"),
        )
    )

    assert context.llm_calls == 2


def test_scheduler_source_writes_cache(tmp_path: Path) -> None:
    service, storage, cache_store = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "调度消息", int(datetime(2026, 3, 22, 17, 50, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 18, 0, 0),
            period="today",
            mode="scheduled",
            source="scheduler",
            analysis_config=_base_config(),
        )
    )

    record = cache_store.get_record(group_id="group_1001", date="2026-03-22", mode="scheduled")
    assert record is not None
    assert record.source == "scheduler"


def test_scheduler_reuses_cache_without_real_new_messages(tmp_path: Path) -> None:
    service, storage, _cache = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "调度前有效消息", int(datetime(2026, 3, 22, 17, 50, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 18, 0, 0),
            period="today",
            mode="scheduled",
            source="scheduler",
            analysis_config=_base_config(),
        )
    )

    _append(storage, 
        MessageRecord(
            "group_1001",
            "bot_1",
            "Bot",
            "群聊兴趣日报（2026-03-22）\n统计日期：2026-03-22\n统计范围：2026-03-22 00:00 - 2026-03-22 18:00\n群组：group_1001",
            int(datetime(2026, 3, 22, 18, 1, 0).timestamp()),
        )
    )

    context2 = _StubContext(responses=[])
    _run(
        service.build_report_for_period_with_metrics(
            context=context2,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 18, 2, 0),
            period="today",
            mode="scheduled",
            source="scheduler",
            analysis_config=_base_config(),
        )
    )

    assert context.llm_calls == 1
    assert context2.llm_calls == 0


def test_failed_generation_does_not_overwrite_existing_cache(tmp_path: Path) -> None:
    service, storage, cache_store = _build_service(tmp_path)
    _append(storage, 
        MessageRecord("group_1001", "u1", "Alice", "第一条", int(datetime(2026, 3, 22, 10, 0, 0).timestamp()))
    )

    context = _StubContext(responses=[_valid_unified_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(),
        )
    )

    old_record = cache_store.get_record(group_id="group_1001", date="2026-03-22", mode="today")
    assert old_record is not None
    old_message_count = old_record.message_count

    _append(storage, 
        MessageRecord("group_1001", "u2", "Bob", "第二条", int(datetime(2026, 3, 22, 12, 10, 0).timestamp()))
    )

    original_load_messages = service.storage.load_messages

    def _boom(*args, **kwargs):
        raise RuntimeError("rebuild failed")

    service.storage.load_messages = _boom  # type: ignore[method-assign]
    try:
        try:
            _run(
                service.build_report_for_period_with_metrics(
                    context=context,
                    event=_event(),
                    group_id="group_1001",
                    now=datetime(2026, 3, 22, 13, 0, 0),
                    period="today",
                    mode="today",
                    source="command_group_digest_today",
                    analysis_config=_base_config(),
                )
            )
            assert False, "expected rebuild failure"
        except RuntimeError:
            pass
    finally:
        service.storage.load_messages = original_load_messages  # type: ignore[method-assign]

    new_record = cache_store.get_record(group_id="group_1001", date="2026-03-22", mode="today")
    assert new_record is not None
    assert new_record.message_count == old_message_count
