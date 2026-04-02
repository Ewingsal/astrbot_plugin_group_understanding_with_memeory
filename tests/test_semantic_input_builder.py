from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from astrbot_plugin_group_digest.services.digest_service import GroupDigestService
from astrbot_plugin_group_digest.services.interaction_service import InteractionService
from astrbot_plugin_group_digest.services.llm_analysis_service import LLMAnalysisService
from astrbot_plugin_group_digest.services.models import LLMAnalysisConfig, MessageRecord
from astrbot_plugin_group_digest.services.semantic_input_builder import (
    SemanticInputBuilder,
    SemanticInputMaterial,
)
from astrbot_plugin_group_digest.services.storage import JsonMessageStorage


class _Resp:
    def __init__(self, text: str):
        self.completion_text = text


class _StubContext:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def get_current_chat_provider_id(self, umo=None):
        return "session_provider"

    async def llm_generate(self, chat_provider_id: str, prompt: str):
        self.prompts.append(prompt)
        return _Resp(self.responses.pop(0))


class _StubSemanticInputBuilder:
    def __init__(self):
        self.full_calls: list[dict] = []

    async def build_for_full_window(
        self,
        *,
        group_id: str,
        date_label: str,
        time_window: str,
        mode: str,
        effective_messages: list[MessageRecord],
        max_messages_for_analysis: int,
        context=None,
        event=None,
        analysis_provider_id: str = "",
    ) -> SemanticInputMaterial:
        _ = (context, event, analysis_provider_id)
        self.full_calls.append(
            {
                "group_id": group_id,
                "date_label": date_label,
                "mode": mode,
                "max_messages_for_analysis": max_messages_for_analysis,
                "effective_count": len(effective_messages),
            }
        )
        selected = sorted(effective_messages, key=lambda x: x.timestamp)[:1]
        return SemanticInputMaterial(
            messages=selected,
            topic_slice_contexts=[],
            source="stub_builder",
            total_effective_messages=len(effective_messages),
            selected_message_count=len(selected),
            truncated=len(selected) < len(effective_messages),
        )

    async def build_for_incremental(self, **kwargs):
        raise AssertionError("this test should not call incremental builder")

    def describe_extension_point(self):
        return {
            "topic_slice_contexts_enabled": False,
            "topic_slice_context_char_guard": 0,
        }


class _StubTopicManager:
    def collect_slice_contexts(self, **kwargs):
        _ = kwargs
        return ["slice_context_1"]

    def get_day_topics_snapshot(self, **kwargs):
        _ = kwargs
        return []


class _EmptyTopicManager:
    def collect_slice_contexts(self, **kwargs):
        _ = kwargs
        return []


class _LongSliceTopicManager:
    def collect_slice_contexts(self, **kwargs):
        _ = kwargs
        return [
            "A" * 30,
            "B" * 30,
        ]

    def get_day_topics_snapshot(self, **kwargs):
        _ = kwargs
        return []


class _ScheduledTopicManager:
    def collect_slice_contexts(self, **kwargs):
        _ = kwargs
        return ["slice_context_1"]

    def get_day_topics_snapshot(self, **kwargs):
        _ = kwargs
        return [
            {
                "topic_id": "20260322_0002",
                "status": "active",
                "last_active_at": int(datetime(2026, 3, 22, 11, 58, 0).timestamp()),
                "message_count": 7,
                "participants": ["Alice(u1)", "Bob(u2)"],
                "core_text": "当前正在讨论今晚部署窗口和回滚预案",
            }
        ]


class _StubEmbeddingBackend:
    async def embed_text(self, text: str):
        _ = text
        return [0.1, 0.2]


class _CaptureEmbeddingBackend:
    def __init__(self):
        self.calls: list[str] = []

    async def embed_text(self, text: str):
        self.calls.append(text)
        return [0.1, 0.2]


class _FailingEmbeddingBackend:
    async def embed_text(self, text: str):
        _ = text
        raise RuntimeError("embed failed")


class _StubEmbeddingStore:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = list(rows or [])
        self.calls: list[dict] = []

    @property
    def enabled(self) -> bool:
        return True

    async def upsert_semantic_unit(self, doc):
        _ = doc
        return True

    async def upsert_topic_slice(self, doc):
        _ = doc
        return True

    async def query_semantic_units(self, **kwargs):
        _ = kwargs
        return []

    async def query_topic_slices(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.rows)


class _FailingEmbeddingStore:
    @property
    def enabled(self) -> bool:
        return True

    async def upsert_semantic_unit(self, doc):
        _ = doc
        return True

    async def upsert_topic_slice(self, doc):
        _ = doc
        return True

    async def query_semantic_units(self, **kwargs):
        _ = kwargs
        return []

    async def query_topic_slices(self, **kwargs):
        _ = kwargs
        raise RuntimeError("store unavailable")


def _run(coro):
    return asyncio.run(coro)


def _append(storage: JsonMessageStorage, record: MessageRecord) -> None:
    _run(storage.append_message(record))


def _valid_analysis_json() -> str:
    return (
        '{"group_topics":["训练计划"],'
        '"member_interests":{"Alice":"关注训练"},'
        '"overall_summary":"今天主要聊训练安排。",'
        '"suggested_bot_reply":"大家先各报一个可执行的训练目标？"}'
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


def _event() -> SimpleNamespace:
    return SimpleNamespace(unified_msg_origin="platform:group:1001")


def test_digest_service_uses_semantic_input_builder_output(tmp_path: Path) -> None:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    builder = _StubSemanticInputBuilder()
    service = GroupDigestService(
        storage=storage,
        llm_analysis_service=LLMAnalysisService(),
        interaction_service=InteractionService(),
        template_path=Path(__file__).resolve().parents[1] / "templates" / "daily_digest.md.j2",
        semantic_input_builder=builder,  # type: ignore[arg-type]
    )

    _append(
        storage,
        MessageRecord("group_1001", "u1", "Alice", "旧消息（应被 builder 选中）", int(datetime(2026, 3, 22, 9, 0, 0).timestamp())),
    )
    _append(
        storage,
        MessageRecord("group_1001", "u2", "Bob", "新消息（不应进入 LLM 输入）", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
    )

    context = _StubContext(responses=[_valid_analysis_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(max_messages_for_analysis=2),
        )
    )

    assert len(builder.full_calls) == 1
    assert builder.full_calls[0]["effective_count"] == 2
    assert context.prompts
    assert "旧消息（应被 builder 选中）" in context.prompts[0]
    assert "新消息（不应进入 LLM 输入）" not in context.prompts[0]


def test_default_semantic_input_builder_keeps_tail_behavior(tmp_path: Path) -> None:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    service = GroupDigestService(
        storage=storage,
        llm_analysis_service=LLMAnalysisService(),
        interaction_service=InteractionService(),
        template_path=Path(__file__).resolve().parents[1] / "templates" / "daily_digest.md.j2",
        semantic_input_builder=SemanticInputBuilder(),
    )

    _append(
        storage,
        MessageRecord("group_1001", "u1", "Alice", "M1", int(datetime(2026, 3, 22, 9, 0, 0).timestamp())),
    )
    _append(
        storage,
        MessageRecord("group_1001", "u2", "Bob", "M2", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
    )
    _append(
        storage,
        MessageRecord("group_1001", "u3", "Carol", "M3", int(datetime(2026, 3, 22, 11, 0, 0).timestamp())),
    )

    context = _StubContext(responses=[_valid_analysis_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(max_messages_for_analysis=2),
        )
    )

    assert context.prompts
    assert "M1" not in context.prompts[0]
    assert "M2" in context.prompts[0]
    assert "M3" in context.prompts[0]


def test_semantic_input_builder_enables_slice_context_by_default() -> None:
    builder = SemanticInputBuilder(topic_segment_manager=_StubTopicManager())  # type: ignore[arg-type]
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="today",
            effective_messages=[],
            max_messages_for_analysis=80,
        )
    )

    assert material.topic_slice_contexts == ["slice_context_1"]
    assert material.source == "topic_slices_plus_tail_raw_messages"
    assert material.topic_slice_signature


def test_semantic_input_builder_falls_back_to_tail_when_no_slice() -> None:
    builder = SemanticInputBuilder(topic_segment_manager=_EmptyTopicManager())  # type: ignore[arg-type]
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="today",
            effective_messages=[
                MessageRecord("group_1001", "u1", "Alice", "M1", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
                MessageRecord("group_1001", "u2", "Bob", "M2", int(datetime(2026, 3, 22, 10, 1, 0).timestamp())),
            ],
            max_messages_for_analysis=1,
        )
    )

    assert material.topic_slice_contexts == []
    assert material.source == "raw_effective_messages_tail"
    assert material.selected_message_count == 1
    assert material.truncated is True


def test_semantic_input_builder_topic_slice_length_guard() -> None:
    builder = SemanticInputBuilder(
        topic_segment_manager=_LongSliceTopicManager(),  # type: ignore[arg-type]
        max_topic_slice_context_chars=40,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="today",
            effective_messages=[],
            max_messages_for_analysis=80,
        )
    )

    assert material.topic_slice_total_count == 2
    assert material.topic_slice_truncated is True
    assert material.topic_slice_total_chars == 60
    assert material.topic_slice_selected_chars == 40
    assert sum(len(item) for item in material.topic_slice_contexts) == 40


def test_digest_service_with_default_builder_includes_slice_contexts_when_available(tmp_path: Path) -> None:
    storage = JsonMessageStorage(tmp_path / "messages.json")
    builder = SemanticInputBuilder(topic_segment_manager=_StubTopicManager())  # type: ignore[arg-type]
    service = GroupDigestService(
        storage=storage,
        llm_analysis_service=LLMAnalysisService(),
        interaction_service=InteractionService(),
        template_path=Path(__file__).resolve().parents[1] / "templates" / "daily_digest.md.j2",
        semantic_input_builder=builder,
    )

    _append(
        storage,
        MessageRecord("group_1001", "u1", "Alice", "M1", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
    )

    context = _StubContext(responses=[_valid_analysis_json()])
    _run(
        service.build_report_for_period_with_metrics(
            context=context,
            event=_event(),
            group_id="group_1001",
            now=datetime(2026, 3, 22, 12, 0, 0),
            period="today",
            mode="today",
            source="command_group_digest_today",
            analysis_config=_base_config(max_messages_for_analysis=2),
        )
    )

    assert context.prompts
    assert "补充话题切片上下文" in context.prompts[0]
    assert "slice_context_1" in context.prompts[0]


def test_semantic_input_builder_includes_retrieved_topic_slices(tmp_path: Path) -> None:
    _ = tmp_path
    embedding_store = _StubEmbeddingStore(
        rows=[
            {
                "object_type": "topic_slice",
                "group_id": "group_1001",
                "date_label": "2026-03-21",
                "topic_id": "20260321_0001",
                "start_ts": int(datetime(2026, 3, 21, 10, 0, 0).timestamp()),
                "end_ts": int(datetime(2026, 3, 21, 10, 30, 0).timestamp()),
                "message_count": 6,
                "participants": ["Alice(u1)"],
                "core_text": "昨天讨论了部署回滚策略",
            }
        ]
    )
    builder = SemanticInputBuilder(
        topic_segment_manager=_StubTopicManager(),  # type: ignore[arg-type]
        embedding_backend=_StubEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=embedding_store,  # type: ignore[arg-type]
        enable_topic_slice_retrieval=True,
        topic_slice_retrieval_recent_days=3,
        topic_slice_retrieval_limit=5,
        topic_slice_retrieval_query_message_count=2,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="today",
            effective_messages=[
                MessageRecord("group_1001", "u1", "Alice", "今天讨论上线计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
                MessageRecord("group_1001", "u2", "Bob", "关注回滚演练", int(datetime(2026, 3, 22, 10, 5, 0).timestamp())),
            ],
            max_messages_for_analysis=2,
        )
    )

    assert embedding_store.calls
    call = embedding_store.calls[0]
    assert call["group_id"] == "group_1001"
    assert isinstance(call["query_vector"], list)
    assert call["start_ts"] < call["end_ts"]
    assert call["limit"] == 5
    assert material.retrieved_topic_slice_count == 1
    assert material.current_day_topic_slice_count == 1
    assert len(material.topic_slice_contexts) >= 2
    assert material.retrieval_enabled is True
    assert material.retrieval_degraded is False
    assert material.source == "retrieved_topic_slices_plus_current_day_slices_plus_tail_raw_messages"
    assert any("retrieved_topic_id=" in item for item in material.topic_slice_contexts)


def test_semantic_input_builder_retrieval_failure_degrades_safely() -> None:
    builder = SemanticInputBuilder(
        topic_segment_manager=_StubTopicManager(),  # type: ignore[arg-type]
        embedding_backend=_FailingEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=_FailingEmbeddingStore(),  # type: ignore[arg-type]
        enable_topic_slice_retrieval=True,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="today",
            effective_messages=[
                MessageRecord("group_1001", "u1", "Alice", "今天讨论上线计划", int(datetime(2026, 3, 22, 10, 0, 0).timestamp())),
            ],
            max_messages_for_analysis=2,
        )
    )

    assert material.retrieval_enabled is True
    assert material.retrieval_degraded is True
    assert material.current_day_topic_slice_count == 1
    assert material.retrieved_topic_slice_count == 0
    assert material.source == "topic_slices_plus_tail_raw_messages"


def test_semantic_input_builder_scheduled_query_includes_runtime_topic_context() -> None:
    embedding_backend = _CaptureEmbeddingBackend()
    embedding_store = _StubEmbeddingStore(
        rows=[
            {
                "object_type": "topic_slice",
                "group_id": "group_1001",
                "date_label": "2026-03-21",
                "topic_id": "20260321_0001",
                "start_ts": int(datetime(2026, 3, 21, 20, 0, 0).timestamp()),
                "end_ts": int(datetime(2026, 3, 21, 20, 30, 0).timestamp()),
                "message_count": 4,
                "participants": ["Alice(u1)"],
                "core_text": "历史相似话题：部署和回滚",
            }
        ]
    )
    builder = SemanticInputBuilder(
        topic_segment_manager=_ScheduledTopicManager(),  # type: ignore[arg-type]
        embedding_backend=embedding_backend,  # type: ignore[arg-type]
        embedding_store=embedding_store,  # type: ignore[arg-type]
        enable_topic_slice_retrieval=True,
        topic_slice_retrieval_query_message_count=2,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="scheduled",
            effective_messages=[
                MessageRecord("group_1001", "u1", "Alice", "今晚部署窗口在 22:00", int(datetime(2026, 3, 22, 11, 55, 0).timestamp())),
                MessageRecord("group_1001", "u2", "Bob", "回滚预案再确认一下", int(datetime(2026, 3, 22, 11, 56, 0).timestamp())),
            ],
            max_messages_for_analysis=5,
        )
    )

    assert embedding_backend.calls
    query_text = embedding_backend.calls[0]
    assert "scheduled_runtime_topics:" in query_text
    assert "当前正在讨论今晚部署窗口和回滚预案" in query_text
    assert "tail_effective_messages:" in query_text
    assert material.retrieval_query_mode == "scheduled_topic_plus_tail"
    assert material.retrieval_query_topic_hint_count == 1
    assert material.retrieved_topic_slice_count == 1
    assert material.source == "retrieved_topic_slices_plus_current_day_slices_plus_tail_raw_messages"


def test_semantic_input_builder_scheduled_mode_falls_back_when_no_retrieval_hits() -> None:
    builder = SemanticInputBuilder(
        topic_segment_manager=_ScheduledTopicManager(),  # type: ignore[arg-type]
        embedding_backend=_StubEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=_StubEmbeddingStore(rows=[]),  # type: ignore[arg-type]
        enable_topic_slice_retrieval=True,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 12:00",
            mode="scheduled",
            effective_messages=[
                MessageRecord("group_1001", "u1", "Alice", "今晚部署窗口在 22:00", int(datetime(2026, 3, 22, 11, 55, 0).timestamp())),
            ],
            max_messages_for_analysis=5,
        )
    )

    assert material.retrieval_enabled is True
    assert material.retrieval_degraded is False
    assert material.retrieved_topic_slice_count == 0
    assert material.current_day_topic_slice_count == 1
    assert material.retrieval_query_mode in {"scheduled_topic_plus_tail", "scheduled_tail_only"}
    assert material.source == "topic_slices_plus_tail_raw_messages"
